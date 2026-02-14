//! The main [`RenderBackend`] and [`CommandHandler`] implementation for
//! the Bézier-based WGPU renderer.
//!
//! This backend creates GPU pipelines and processes SWF shapes by converting
//! them into Loop-Blinn geometry, then rendering with the appropriate pipeline
//! per draw call.

use crate::mesh::{self, BezierMesh, DrawType, as_bezier_mesh};
use crate::pipelines::{BindLayouts, Pipelines};
use crate::shaders::Shaders;
use crate::target::{RenderTarget, RenderTargetFrame};
use crate::{
    GlobalsUniform, MaskState, Texture, Transforms,
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
use swf::{Color, ColorTransform, FillStyle};
use wgpu::util::DeviceExt;

/// Align a value to the given alignment.
fn align_to(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
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
}

impl<T: RenderTarget> BezierRenderBackend<T> {
    /// Create a new Bézier render backend with the given render target.
    ///
    /// This initializes the wgpu device, creates all shader modules and
    /// render pipelines, and sets up the GPU resources.
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
            1, // sample_count
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
            
            // Write each transform at its aligned offset
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
        // Pre-resolve all bitmap handles before building the mesh.
        // This avoids needing `&mut self` (as RenderBackend) inside build_mesh.
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
        // TODO: Implement offscreen rendering.
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

        // Get the next texture from the render target.
        let frame = match self.target.get_next_texture() {
            Ok(frame) => frame,
            Err(e) => {
                tracing::warn!("Failed to get target texture: {:?}", e);
                return;
            }
        };
        let frame_view = frame.view();

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
            self.execute_commands(
                &mut render_pass,
                &commands,
                &mut transform_index,
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
        // TODO: Implement MSAA quality levels.
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
    /// Pre-collect transforms from a command list (recursive for blends).
    fn collect_transforms(&mut self, commands: &CommandList) {
        use ruffle_render::commands::Command;
        for cmd in &commands.commands {
            match cmd {
                Command::RenderBitmap { transform, .. }
                | Command::RenderShape { transform, .. }
                | Command::RenderStage3D { transform, .. } => {
                    self.push_transform(transform);
                }
                Command::DrawRect { matrix, .. }
                | Command::DrawLine { matrix, .. }
                | Command::DrawLineRect { matrix, .. } => {
                    // Create a simple identity color transform for rect/line draws.
                    let transform = Transform {
                        matrix: *matrix,
                        color_transform: ColorTransform::IDENTITY,
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
    ) {
        use ruffle_render::commands::Command;
        for cmd in &commands.commands {
            match cmd {
                Command::RenderShape { shape, .. } => {
                    let aligned_size = align_to(
                        std::mem::size_of::<Transforms>() as u32,
                        self.min_uniform_buffer_offset_alignment,
                    );
                    let offset = *transform_index * aligned_size;
                    *transform_index += 1;

                    let mesh = as_bezier_mesh(shape);
                    for draw in &mesh.draws {
                        match &draw.draw_type {
                            DrawType::Color => {
                                render_pass.set_pipeline(&self.pipelines.color_fill);
                            }
                            DrawType::Gradient { bind_group } => {
                                render_pass.set_pipeline(&self.pipelines.gradient_fill);
                                render_pass.set_bind_group(2, bind_group, &[]);
                            }
                            DrawType::Bitmap { bind_group } => {
                                render_pass.set_pipeline(&self.pipelines.bitmap_fill);
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
                Command::RenderBitmap { .. } => {
                    let aligned_size = align_to(
                        std::mem::size_of::<Transforms>() as u32,
                        self.min_uniform_buffer_offset_alignment,
                    );
                    let _offset = *transform_index * aligned_size;
                    *transform_index += 1;
                    // TODO: Implement bitmap rendering with a quad mesh.
                }
                Command::RenderStage3D { .. } => {
                    *transform_index += 1;
                    // Stage3D not supported.
                }
                Command::DrawRect { .. } => {
                    let aligned_size = align_to(
                        std::mem::size_of::<Transforms>() as u32,
                        self.min_uniform_buffer_offset_alignment,
                    );
                    let _offset = *transform_index * aligned_size;
                    *transform_index += 1;
                    // TODO: Create a cached unit quad mesh for this.
                }
                Command::DrawLine { .. } => {
                    let _offset = *transform_index * std::mem::size_of::<Transforms>() as u32;
                    *transform_index += 1;
                    // TODO: Implement line drawing.
                }
                Command::DrawLineRect { .. } => {
                    let _offset = *transform_index * std::mem::size_of::<Transforms>() as u32;
                    *transform_index += 1;
                    // TODO: Implement line rect drawing.
                }
                Command::PushMask => {
                    // TODO: stencil masking
                }
                Command::ActivateMask => {}
                Command::DeactivateMask => {}
                Command::PopMask => {}
                Command::Blend(sub_commands, _blend_mode) => {
                    // For now, just render sub-commands without blend mode changes.
                    // TODO: Implement proper blend mode support with intermediate textures.
                    self.execute_commands(render_pass, sub_commands, transform_index);
                }
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => {
                    // For now, just render the maskee without masking.
                    self.execute_commands(render_pass, maskee_commands, transform_index);
                    // Skip mask commands from transform counting perspective.
                    let mut discard = *transform_index;
                    self.execute_commands(render_pass, mask_commands, &mut discard);
                    *transform_index = discard;
                }
            }
        }
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
