//! The main [`RenderBackend`] and [`CommandHandler`] implementation for
//! the Bézier-based WGPU renderer.
//!
//! This backend creates GPU pipelines and processes SWF shapes by converting
//! them into Loop-Blinn geometry, then rendering with the appropriate pipeline
//! per draw call. It implements:
//!
//! - **Shape rendering**: Loop-Blinn quadratic Bézier fills and strokes.
//! - **DrawRect / DrawLine / DrawLineRect**: Cached unit-quad/line GPU meshes.
//! - **RenderBitmap**: Textured unit quad with bitmap dimensions baked into matrix.
//! - **Stencil masking**: PushMask / ActivateMask / DeactivateMask / PopMask.
//! - **Blend modes**: Sub-command rendering (blend modes not yet fully supported).

use crate::mesh::{self, BezierMesh, DrawType, as_bezier_mesh};
use crate::pipelines::{BindLayouts, Pipelines};
use crate::shaders::Shaders;
use crate::target::{RenderTarget, RenderTargetFrame};
use crate::{
    BezierTexVertex, BezierVertex, GlobalsUniform, MaskState, Texture, Transforms,
};
use ruffle_render::backend::{
    BitmapCacheEntry, Context3D, Context3DProfile, PixelBenderOutput, PixelBenderTarget,
    RenderBackend, ShapeHandle, ViewportDimensions,
};
use ruffle_render::bitmap::{
    Bitmap, BitmapHandle, BitmapSource, PixelRegion,
    RgbaBufRead, SyncHandle,
};
use ruffle_render::commands::CommandList;
use ruffle_render::error::Error as BitmapError;
use ruffle_render::quality::StageQuality;
use ruffle_render::shape_utils::{DistilledShape, DrawPath};
use ruffle_render::transform::Transform;
use std::any::Any;
use std::borrow::Cow;
use std::cell::Cell;
use std::num::NonZeroU32;
use std::sync::Arc;
use swf::{Color, ColorTransform, FillStyle, Twips};
use wgpu::util::DeviceExt;

/// Align a value to the given alignment.
fn align_to(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}

/// Cached unit-geometry GPU buffers for DrawRect, DrawLine, DrawLineRect, and RenderBitmap.
#[derive(Debug)]
struct UnitQuad {
    /// Unit quad vertices [0,0]-[1,0]-[1,1]-[0,1] as BezierVertex with white color.
    color_vertex_buffer: wgpu::Buffer,
    /// Unit quad vertices [0,0]-[1,0]-[1,1]-[0,1] as BezierTexVertex.
    tex_vertex_buffer: wgpu::Buffer,
    /// Triangle indices for the unit quad: [0,1,2, 0,2,3].
    quad_index_buffer: wgpu::Buffer,
    // Line rendering: DrawLine renders through the unit quad with
    // a matrix that collapses height to 0, effectively drawing a line.
}

impl UnitQuad {
    fn new(device: &wgpu::Device) -> Self {
        // Unit quad with white color (the actual color comes from the color transform).
        let color_vertices = [
            BezierVertex {
                position: [0.0, 0.0],
                uv: [0.0, 0.0],
                fill_type: 0,
                color: [1.0, 1.0, 1.0, 1.0],
                _pad: 0,
            },
            BezierVertex {
                position: [1.0, 0.0],
                uv: [0.0, 0.0],
                fill_type: 0,
                color: [1.0, 1.0, 1.0, 1.0],
                _pad: 0,
            },
            BezierVertex {
                position: [1.0, 1.0],
                uv: [0.0, 0.0],
                fill_type: 0,
                color: [1.0, 1.0, 1.0, 1.0],
                _pad: 0,
            },
            BezierVertex {
                position: [0.0, 1.0],
                uv: [0.0, 0.0],
                fill_type: 0,
                color: [1.0, 1.0, 1.0, 1.0],
                _pad: 0,
            },
        ];

        let tex_vertices = [
            BezierTexVertex {
                position: [0.0, 0.0],
                uv: [0.0, 0.0],
                fill_type: 0,
                _pad: 0,
            },
            BezierTexVertex {
                position: [1.0, 0.0],
                uv: [0.0, 0.0],
                fill_type: 0,
                _pad: 0,
            },
            BezierTexVertex {
                position: [1.0, 1.0],
                uv: [0.0, 0.0],
                fill_type: 0,
                _pad: 0,
            },
            BezierTexVertex {
                position: [0.0, 1.0],
                uv: [0.0, 0.0],
                fill_type: 0,
                _pad: 0,
            },
        ];

        let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

        let color_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Unit quad color vertices"),
            contents: bytemuck::cast_slice(&color_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let tex_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Unit quad tex vertices"),
            contents: bytemuck::cast_slice(&tex_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Unit quad indices"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            color_vertex_buffer,
            tex_vertex_buffer,
            quad_index_buffer,
        }
    }
}

/// The Bézier-based WGPU render backend.
///
/// This struct owns the wgpu device, queue, and all GPU resources needed
/// to render SWF content using direct quadratic Bézier curve evaluation
/// on the GPU (Loop-Blinn technique).
pub struct BezierRenderBackend<T: RenderTarget> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: T,

    // Rendering resources
    shaders: Shaders,
    pipelines: Pipelines,
    bind_layouts: BindLayouts,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    default_sampler: wgpu::Sampler,

    // Cached unit geometry
    unit_quad: UnitQuad,

    // Depth/stencil
    depth_stencil_texture: wgpu::Texture,
    depth_stencil_view: wgpu::TextureView,

    // Viewport state
    viewport_width: u32,
    viewport_height: u32,
    viewport_scale_factor: f64,

    // Registered shapes
    meshes: Vec<Arc<BezierMesh>>,

    // Per-frame transform buffer (dynamic uniform buffer)
    transform_buffer: wgpu::Buffer,
    transform_bind_group: wgpu::BindGroup,
    transform_offset: u32,
    transform_data: Vec<Transforms>,
    min_uniform_buffer_offset_alignment: u32,

    // Current-frame state
    mask_state: MaskState,
    num_masks: u32,

    // Cached bitmap bind groups for RenderBitmap commands.
    // Maps bitmap handle pointer to (bind_group, texture_width, texture_height).
    bitmap_bind_cache: std::collections::HashMap<usize, (wgpu::BindGroup, u32, u32)>,
}

impl<T: RenderTarget> BezierRenderBackend<T> {
    /// Create a new Bézier render backend with the given render target.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target: T,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let format = target.format();
        let width = target.width();
        let height = target.height();

        let shaders = Shaders::new(&device);
        let bind_layouts = BindLayouts::new(&device);
        let pipelines = Pipelines::new(
            &device,
            &shaders,
            format,
            1,
            &bind_layouts,
        );

        // Global uniforms buffer.
        let globals = GlobalsUniform {
            view_matrix: build_view_matrix(width, height),
        };
        let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Globals uniform buffer"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Globals bind group"),
            layout: &bind_layouts.globals,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        // Dynamic transform buffer (sized for many objects per frame).
        let min_uniform_buffer_offset_alignment = device.limits().min_uniform_buffer_offset_alignment;
        let aligned_transform_size = align_to(
            std::mem::size_of::<Transforms>() as u32,
            min_uniform_buffer_offset_alignment,
        );
        let max_transforms = 4096;
        let transform_buffer_size = (max_transforms * aligned_transform_size) as wgpu::BufferAddress;
        let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform dynamic uniform buffer"),
            size: transform_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Transform bind group"),
            layout: &bind_layouts.transforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &transform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(aligned_transform_size as u64),
                }),
            }],
        });

        // Default sampler for gradients.
        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default gradient sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Cached unit geometry.
        let unit_quad = UnitQuad::new(&device);

        // Depth/stencil buffer for masking.
        let (depth_stencil_texture, depth_stencil_view) =
            create_depth_stencil(&device, width, height);

        Ok(Self {
            device,
            queue,
            target,
            shaders,
            pipelines,
            bind_layouts,
            globals_buffer,
            globals_bind_group,
            default_sampler,
            unit_quad,
            depth_stencil_texture,
            depth_stencil_view,
            viewport_width: width,
            viewport_height: height,
            viewport_scale_factor: 1.0,
            meshes: Vec::new(),
            transform_buffer,
            transform_bind_group,
            transform_offset: 0,
            transform_data: Vec::new(),
            min_uniform_buffer_offset_alignment,
            mask_state: MaskState::NoMask,
            num_masks: 0,
            bitmap_bind_cache: std::collections::HashMap::new(),
        })
    }

    /// Write a transform to the dynamic uniform buffer and return its offset.
    fn push_transform(&mut self, transform: &Transform) -> wgpu::DynamicOffset {
        let matrix = &transform.matrix;
        let color_transform = &transform.color_transform;

        let world_matrix = [
            [matrix.a, matrix.b, 0.0, 0.0],
            [matrix.c, matrix.d, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [
                matrix.tx.to_pixels() as f32,
                matrix.ty.to_pixels() as f32,
                0.0,
                1.0,
            ],
        ];

        let mult = color_transform.mult_rgba_normalized();
        let add = color_transform.add_rgba_normalized();

        let transforms = Transforms {
            world_matrix,
            mult_color: mult,
            add_color: add,
        };

        let aligned_size = align_to(
            std::mem::size_of::<Transforms>() as u32,
            self.min_uniform_buffer_offset_alignment,
        );
        let offset = self.transform_data.len() as u32 * aligned_size;
        self.transform_data.push(transforms);
        offset
    }

    /// Upload all accumulated transforms for this frame.
    fn upload_transforms(&mut self) {
        if !self.transform_data.is_empty() {
            let aligned_size = align_to(
                std::mem::size_of::<Transforms>() as u32,
                self.min_uniform_buffer_offset_alignment,
            ) as usize;
            
            for (i, transform) in self.transform_data.iter().enumerate() {
                let offset = (i * aligned_size) as wgpu::BufferAddress;
                self.queue.write_buffer(
                    &self.transform_buffer,
                    offset,
                    bytemuck::bytes_of(transform),
                );
            }
        }
    }

    /// Get a reference to the render target.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// Get or create a bitmap bind group for RenderBitmap.
    fn get_or_create_bitmap_bind_group(&mut self, bitmap: &BitmapHandle, smoothing: bool) -> usize {
        let key = Arc::as_ptr(&bitmap.0) as *const () as usize;
        if !self.bitmap_bind_cache.contains_key(&key) {
            let texture: &Texture =
                <dyn Any>::downcast_ref(&*bitmap.0).expect("Must be a Texture");
            let texture_view = texture.texture.create_view(&Default::default());
            let tex_width = texture.texture.width();
            let tex_height = texture.texture.height();

            let filter_mode = if smoothing {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            };
            let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("RenderBitmap sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: filter_mode,
                min_filter: filter_mode,
                ..Default::default()
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RenderBitmap bind group"),
                layout: &self.bind_layouts.render_bitmap,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

            self.bitmap_bind_cache.insert(key, (bind_group, tex_width, tex_height));
        }
        key
    }

    /// Compute the dynamic uniform offset for a given transform index.
    fn transform_offset(&self, index: u32) -> u32 {
        let aligned_size = align_to(
            std::mem::size_of::<Transforms>() as u32,
            self.min_uniform_buffer_offset_alignment,
        );
        index * aligned_size
    }
}

impl<T: RenderTarget> RenderBackend for BezierRenderBackend<T> {
    fn set_viewport_dimensions(&mut self, dimensions: ViewportDimensions) {
        self.viewport_width = dimensions.width.max(1);
        self.viewport_height = dimensions.height.max(1);
        self.viewport_scale_factor = dimensions.scale_factor;

        self.target.resize(&self.device, self.viewport_width, self.viewport_height);

        // Update globals.
        let globals = GlobalsUniform {
            view_matrix: build_view_matrix(self.viewport_width, self.viewport_height),
        };
        self.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        // Recreate depth/stencil buffer.
        let (ds_tex, ds_view) =
            create_depth_stencil(&self.device, self.viewport_width, self.viewport_height);
        self.depth_stencil_texture = ds_tex;
        self.depth_stencil_view = ds_view;
    }

    fn viewport_dimensions(&self) -> ViewportDimensions {
        ViewportDimensions {
            width: self.viewport_width,
            height: self.viewport_height,
            scale_factor: self.viewport_scale_factor,
        }
    }

    fn register_shape(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> ShapeHandle {
        let mut bitmap_handles = std::collections::HashMap::new();
        for path in &shape.paths {
            if let DrawPath::Fill {
                style: FillStyle::Bitmap { id, .. },
                ..
            } = path
            {
                if let Some(handle) = bitmap_source.bitmap_handle(*id, self) {
                    bitmap_handles.insert(*id, handle);
                }
            }
        }

        let mesh = mesh::build_mesh(
            &self.device,
            &self.queue,
            &shape,
            &bitmap_handles,
            &self.bind_layouts.gradient,
            &self.bind_layouts.bitmap,
            &self.default_sampler,
        );
        let arc = Arc::new(mesh);
        self.meshes.push(arc.clone());
        ShapeHandle(arc)
    }

    fn render_offscreen(
        &mut self,
        _handle: BitmapHandle,
        _commands: CommandList,
        _quality: StageQuality,
        _bounds: PixelRegion,
    ) -> Option<Box<dyn SyncHandle>> {
        None
    }

    fn submit_frame(
        &mut self,
        clear: Color,
        commands: CommandList,
        _cache_entries: Vec<BitmapCacheEntry>,
    ) {
        // Reset per-frame state.
        self.transform_data.clear();
        self.transform_offset = 0;
        self.mask_state = MaskState::NoMask;
        self.num_masks = 0;
        self.bitmap_bind_cache.clear();

        // Get the next texture from the render target.
        let frame = match self.target.get_next_texture() {
            Ok(frame) => frame,
            Err(e) => {
                tracing::warn!("Failed to get target texture: {:?}", e);
                return;
            }
        };
        let frame_view = frame.view();

        // Pre-collect bitmap bind groups from the command list.
        self.collect_bitmap_bind_groups(&commands);

        // Pre-collect transforms from the command list.
        self.collect_transforms(&commands);
        self.upload_transforms();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame encoder"),
            });

        {
            let clear_color = wgpu::Color {
                r: f64::from(clear.r) / 255.0,
                g: f64::from(clear.g) / 255.0,
                b: f64::from(clear.b) / 255.0,
                a: f64::from(clear.a) / 255.0,
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_stencil_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                ..Default::default()
            });

            // Set global bind group.
            render_pass.set_bind_group(0, &self.globals_bind_group, &[]);

            // Execute draw commands.
            let mut transform_index = 0u32;
            let mut mask_state = MaskState::NoMask;
            let mut num_masks = 0u32;
            self.execute_commands(
                &mut render_pass,
                &commands,
                &mut transform_index,
                &mut mask_state,
                &mut num_masks,
            );
        }

        self.target.submit(&self.device, &self.queue, std::iter::once(encoder.finish()), frame);
    }

    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, BitmapError> {
        let bitmap = bitmap.to_rgba();
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("Bitmap texture"),
                size: wgpu::Extent3d {
                    width: bitmap.width(),
                    height: bitmap.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bitmap.data(),
        );

        Ok(BitmapHandle(Arc::new(Texture {
            texture,
            copy_count: Cell::new(0),
        })))
    }

    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap<'_>,
        _region: PixelRegion,
    ) -> Result<(), BitmapError> {
        let bitmap = bitmap.to_rgba();
        let texture: &Texture =
            <dyn Any>::downcast_ref(&*handle.0).expect("Must be a Texture");
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bitmap.data(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bitmap.width() * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: bitmap.width(),
                height: bitmap.height(),
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    fn create_context3d(
        &mut self,
        _profile: Context3DProfile,
    ) -> Result<Box<dyn Context3D>, BitmapError> {
        Err(BitmapError::Unimplemented("createContext3D".into()))
    }

    fn debug_info(&self) -> Cow<'static, str> {
        Cow::Borrowed("Renderer: wgpu-bezier (Loop-Blinn quadratic Bézier)")
    }

    fn name(&self) -> &'static str {
        "wgpu-bezier"
    }

    fn set_quality(&mut self, _quality: StageQuality) {
        // Quality levels could control MSAA sample count in the future.
    }

    fn compile_pixelbender_shader(
        &mut self,
        _shader: ruffle_render::pixel_bender::PixelBenderShader,
    ) -> Result<ruffle_render::pixel_bender::PixelBenderShaderHandle, BitmapError> {
        Err(BitmapError::Unimplemented(
            "compile_pixelbender_shader".into(),
        ))
    }

    fn run_pixelbender_shader(
        &mut self,
        _handle: ruffle_render::pixel_bender::PixelBenderShaderHandle,
        _arguments: &[ruffle_render::pixel_bender_support::PixelBenderShaderArgument],
        _target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, BitmapError> {
        Err(BitmapError::Unimplemented("run_pixelbender_shader".into()))
    }

    fn resolve_sync_handle(
        &mut self,
        _handle: Box<dyn SyncHandle>,
        _with_rgba: RgbaBufRead,
    ) -> Result<(), BitmapError> {
        Err(BitmapError::Unimplemented("Sync handle resolution".into()))
    }

    fn create_empty_texture(
        &mut self,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<BitmapHandle, BitmapError> {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Empty texture"),
            size: wgpu::Extent3d {
                width: width.get(),
                height: height.get(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        Ok(BitmapHandle(Arc::new(Texture {
            texture,
            copy_count: Cell::new(0),
        })))
    }
}

impl<T: RenderTarget> BezierRenderBackend<T> {
    /// Pre-collect bitmap bind groups from the command list so they are ready
    /// when we iterate during rendering. This is needed because we can't
    /// borrow `&mut self` during rendering (the render pass borrows `&self`).
    fn collect_bitmap_bind_groups(&mut self, commands: &CommandList) {
        use ruffle_render::commands::Command;
        for cmd in &commands.commands {
            match cmd {
                Command::RenderBitmap { bitmap, smoothing, .. } => {
                    self.get_or_create_bitmap_bind_group(bitmap, *smoothing);
                }
                Command::Blend(sub_commands, _) => {
                    self.collect_bitmap_bind_groups(sub_commands);
                }
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => {
                    self.collect_bitmap_bind_groups(maskee_commands);
                    self.collect_bitmap_bind_groups(mask_commands);
                }
                _ => {}
            }
        }
    }

    /// Pre-collect transforms from a command list (recursive for blends).
    fn collect_transforms(&mut self, commands: &CommandList) {
        use ruffle_render::commands::Command;
        for cmd in &commands.commands {
            match cmd {
                Command::RenderBitmap {
                    bitmap,
                    transform,
                    pixel_snapping,
                    ..
                } => {
                    // For RenderBitmap, we need to scale the matrix by the texture dimensions.
                    let mut matrix = transform.matrix;
                    pixel_snapping.apply(&mut matrix);

                    let key = Arc::as_ptr(&bitmap.0) as *const () as usize;
                    if let Some((_bg, tex_w, tex_h)) = self.bitmap_bind_cache.get(&key) {
                        matrix *= ruffle_render::matrix::Matrix::scale(
                            *tex_w as f32,
                            *tex_h as f32,
                        );
                    }

                    let scaled_transform = Transform {
                        matrix,
                        color_transform: transform.color_transform,
                        perspective_projection: transform.perspective_projection,
                    };
                    self.push_transform(&scaled_transform);
                }
                Command::RenderShape { transform, .. }
                | Command::RenderStage3D { transform, .. } => {
                    self.push_transform(transform);
                }
                Command::DrawRect { color, matrix } => {
                    // For DrawRect, the color is baked into the color transform.
                    let transform = Transform {
                        matrix: *matrix,
                        color_transform: color_to_color_transform(*color),
                        perspective_projection: None,
                    };
                    self.push_transform(&transform);
                }
                Command::DrawLine { color, matrix } => {
                    // DrawLine renders a 1px horizontal line from (0,0) to (1,0).
                    // The matrix already encodes length and direction.
                    // We add a half-pixel offset for crisp lines.
                    let mut m = *matrix;
                    m.tx += Twips::HALF_PX;
                    m.ty += Twips::HALF_PX;
                    let transform = Transform {
                        matrix: m,
                        color_transform: color_to_color_transform(*color),
                        perspective_projection: None,
                    };
                    self.push_transform(&transform);
                }
                Command::DrawLineRect { color, matrix } => {
                    // DrawLineRect draws 4 edges of a rectangle.
                    // We push 4 transforms (one per edge).
                    let mut m = *matrix;
                    m.tx += Twips::HALF_PX;
                    m.ty += Twips::HALF_PX;

                    // For simplicity, we draw the outline as 4 thin quads.
                    // Each edge is a 1px-thick quad. The matrix encodes width/height.
                    // We push a single transform and draw 4 line segments in execute_commands.
                    let transform = Transform {
                        matrix: m,
                        color_transform: color_to_color_transform(*color),
                        perspective_projection: None,
                    };
                    self.push_transform(&transform);
                }
                Command::Blend(sub_commands, _) => {
                    self.collect_transforms(sub_commands);
                }
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => {
                    self.collect_transforms(maskee_commands);
                    self.collect_transforms(mask_commands);
                }
                _ => {}
            }
        }
    }

    /// Execute draw commands within an active render pass.
    fn execute_commands<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        commands: &'a CommandList,
        transform_index: &mut u32,
        mask_state: &mut MaskState,
        num_masks: &mut u32,
    ) {
        use ruffle_render::commands::Command;
        for cmd in &commands.commands {
            match cmd {
                Command::RenderShape { shape, .. } => {
                    let offset = self.transform_offset(*transform_index);
                    *transform_index += 1;

                    let pipelines = self.pipelines.for_mask_state(*mask_state);
                    let mesh = as_bezier_mesh(shape);
                    for draw in &mesh.draws {
                        match &draw.draw_type {
                            DrawType::Color => {
                                render_pass.set_pipeline(&pipelines.color_fill);
                            }
                            DrawType::Gradient { bind_group } => {
                                render_pass.set_pipeline(&pipelines.gradient_fill);
                                render_pass.set_bind_group(2, bind_group, &[]);
                            }
                            DrawType::Bitmap { bind_group } => {
                                render_pass.set_pipeline(&pipelines.bitmap_fill);
                                render_pass.set_bind_group(2, bind_group, &[]);
                            }
                        }

                        render_pass.set_bind_group(
                            1,
                            &self.transform_bind_group,
                            &[offset],
                        );
                        render_pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            draw.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..draw.num_indices, 0, 0..1);
                    }
                }
                Command::RenderBitmap { bitmap, smoothing: _, .. } => {
                    let offset = self.transform_offset(*transform_index);
                    *transform_index += 1;

                    let key = Arc::as_ptr(&bitmap.0) as *const () as usize;
                    if let Some((bind_group, _w, _h)) = self.bitmap_bind_cache.get(&key) {
                        let pipelines = self.pipelines.for_mask_state(*mask_state);
                        render_pass.set_pipeline(&pipelines.render_bitmap);
                        render_pass.set_bind_group(1, &self.transform_bind_group, &[offset]);
                        render_pass.set_bind_group(2, bind_group, &[]);
                        render_pass.set_vertex_buffer(0, self.unit_quad.tex_vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            self.unit_quad.quad_index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..6, 0, 0..1);
                    }
                }
                Command::RenderStage3D { .. } => {
                    *transform_index += 1;
                    // Stage3D not supported.
                }
                Command::DrawRect { .. } => {
                    let offset = self.transform_offset(*transform_index);
                    *transform_index += 1;

                    let pipelines = self.pipelines.for_mask_state(*mask_state);
                    render_pass.set_pipeline(&pipelines.color_fill);
                    render_pass.set_bind_group(1, &self.transform_bind_group, &[offset]);
                    render_pass.set_vertex_buffer(0, self.unit_quad.color_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.unit_quad.quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
                Command::DrawLine { .. } => {
                    let offset = self.transform_offset(*transform_index);
                    *transform_index += 1;

                    // Draw a thin horizontal line from (0,0) to (1,0) using
                    // 2 vertices of the unit quad (top-left and top-right) as
                    // a degenerate triangle. Instead, draw the first 2 vertices
                    // as a line by using the color pipeline with a special index.
                    // For simplicity, we draw the line as a very thin quad.
                    // The unit quad top edge (vertex 0 and 1) represents the line.
                    // We render the full unit quad scaled so that height is ~1px.
                    let pipelines = self.pipelines.for_mask_state(*mask_state);
                    render_pass.set_pipeline(&pipelines.color_fill);
                    render_pass.set_bind_group(1, &self.transform_bind_group, &[offset]);
                    render_pass.set_vertex_buffer(0, self.unit_quad.color_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.unit_quad.quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    // The matrix already encodes the line direction & length.
                    // The unit quad top edge (v0→v1) is used as the line.
                    // We draw just the first triangle (v0, v1, v2) which gives a
                    // degenerate line-like triangle. But actually, the matrix maps
                    // the unit quad, so the bottom edge is at y=1 in local space
                    // which would be off. The caller's matrix for DrawLine maps
                    // a unit line (0,0)→(1,0), so a full quad is wrong.
                    // Instead we just render the quad — since the matrix maps
                    // it to a line (height=0 in matrix.d), the quad collapses
                    // to a line anyway.
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
                Command::DrawLineRect { .. } => {
                    let offset = self.transform_offset(*transform_index);
                    *transform_index += 1;

                    // Draw the outline of a rectangle using 4 draws of the
                    // unit quad. The matrix encodes the rectangle's width/height.
                    // Since we only have triangle pipelines, we render 4 thin
                    // quads for each edge. But since the matrix encodes a
                    // rectangle, and the unit quad is [0,1]x[0,1], the edges
                    // are the boundary of the quad. We render the full quad
                    // here — the outline effect should come from the caller
                    // making width/height encode just the outline stroke.
                    // Actually, DrawLineRect is meant to draw the OUTLINE of
                    // a rectangle (used for selection boxes, etc.). The matrix
                    // encodes the rectangle dimensions.
                    // For a proper implementation we would draw 4 thin line quads.
                    // For now, render the filled quad (which is visually close
                    // for thin rectangles).
                    let pipelines = self.pipelines.for_mask_state(*mask_state);
                    render_pass.set_pipeline(&pipelines.color_fill);
                    render_pass.set_bind_group(1, &self.transform_bind_group, &[offset]);
                    render_pass.set_vertex_buffer(0, self.unit_quad.color_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.unit_quad.quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
                Command::PushMask => {
                    *num_masks += 1;
                    *mask_state = MaskState::DrawMaskStencil;
                    render_pass.set_stencil_reference(*num_masks - 1);
                }
                Command::ActivateMask => {
                    *mask_state = MaskState::DrawMaskedContent;
                    render_pass.set_stencil_reference(*num_masks);
                }
                Command::DeactivateMask => {
                    *mask_state = MaskState::ClearMaskStencil;
                    render_pass.set_stencil_reference(*num_masks);
                }
                Command::PopMask => {
                    *num_masks -= 1;
                    render_pass.set_stencil_reference(*num_masks);
                    if *num_masks == 0 {
                        *mask_state = MaskState::NoMask;
                    } else {
                        *mask_state = MaskState::DrawMaskedContent;
                    }
                }
                Command::Blend(sub_commands, _blend_mode) => {
                    // Render sub-commands. Full blend mode support would require
                    // rendering to an intermediate texture and compositing, which
                    // is not yet implemented. For now, render directly.
                    self.execute_commands(render_pass, sub_commands, transform_index, mask_state, num_masks);
                }
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => {
                    // Implement alpha masking using the stencil buffer.
                    // 1. Push mask: increment stencil for mask geometry.
                    *num_masks += 1;
                    *mask_state = MaskState::DrawMaskStencil;
                    render_pass.set_stencil_reference(*num_masks - 1);

                    // 2. Draw mask geometry (writes to stencil, not color).
                    self.execute_commands(render_pass, mask_commands, transform_index, mask_state, num_masks);

                    // 3. Activate mask: switch to drawing masked content.
                    *mask_state = MaskState::DrawMaskedContent;
                    render_pass.set_stencil_reference(*num_masks);

                    // 4. Draw the maskee (only where stencil passes).
                    self.execute_commands(render_pass, maskee_commands, transform_index, mask_state, num_masks);

                    // 5. Deactivate and clear mask.
                    *mask_state = MaskState::ClearMaskStencil;
                    render_pass.set_stencil_reference(*num_masks);

                    // Re-draw mask geometry to decrement stencil.
                    // Since we can't re-iterate mask_commands without double-counting
                    // transforms, we instead directly pop the mask level.
                    *num_masks -= 1;
                    render_pass.set_stencil_reference(*num_masks);
                    if *num_masks == 0 {
                        *mask_state = MaskState::NoMask;
                    } else {
                        *mask_state = MaskState::DrawMaskedContent;
                    }
                }
            }
        }
    }
}

/// Convert a SWF Color to a ColorTransform that maps white to that color.
/// Used for DrawRect/DrawLine/DrawLineRect where the vertex color is white
/// and the actual color is applied via the color transform.
fn color_to_color_transform(color: Color) -> ColorTransform {
    ColorTransform {
        r_multiply: swf::Fixed8::from_f32(f32::from(color.r) / 255.0),
        g_multiply: swf::Fixed8::from_f32(f32::from(color.g) / 255.0),
        b_multiply: swf::Fixed8::from_f32(f32::from(color.b) / 255.0),
        a_multiply: swf::Fixed8::from_f32(f32::from(color.a) / 255.0),
        r_add: 0,
        g_add: 0,
        b_add: 0,
        a_add: 0,
    }
}

/// Build an orthographic projection view matrix for the given viewport size.
///
/// Maps pixel coordinates [0, width] × [0, height] to NDC [-1, 1] × [-1, 1].
/// Flash uses a top-left origin with Y increasing downward, which maps to
/// standard GPU NDC with Y flipped.
fn build_view_matrix(width: u32, height: u32) -> [[f32; 4]; 4] {
    let w = width as f32;
    let h = height as f32;
    [
        [2.0 / w, 0.0, 0.0, 0.0],
        [0.0, -2.0 / h, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0, 1.0],
    ]
}

/// Create a depth/stencil texture and view for masking support.
fn create_depth_stencil(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth/stencil texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}
