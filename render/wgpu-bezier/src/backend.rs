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

use crate::blend::{BlendType, ComplexBlend, TrivialBlend};
use crate::mesh::{self, BezierMesh, DrawType, as_bezier_mesh};
use crate::pipelines::{BindLayouts, Pipelines};
use crate::shaders::Shaders;
use crate::filters::{FilterSource, Filters, as_texture};
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
use ruffle_render::filters::Filter;
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
    value.div_ceil(alignment) * alignment
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

    // MSAA state
    sample_count: u32,
    msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    surface_format: wgpu::TextureFormat,

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

    // Complex blend compositing support:
    // blend_buffer: Snapshot of the framebuffer before a complex blend operation.
    // Used as the "parent" texture in blend compositing shaders.
    blend_buffer: Option<wgpu::Texture>,
    blend_buffer_view: Option<wgpu::TextureView>,

    // Offscreen framebuffer for rendering (supports COPY_SRC for blend compositing).
    // When complex blends are needed, we render to this texture instead of the
    // swapchain directly, then blit to the swapchain at the end.
    offscreen_buffer: Option<wgpu::Texture>,
    offscreen_view: Option<wgpu::TextureView>,

    // Copy pipeline for blitting textures (used for render_offscreen and blend compositing).
    copy_pipeline: wgpu::RenderPipeline,
    copy_bind_layout: wgpu::BindGroupLayout,

    // GPU filter implementations.
    filters: Filters,
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
            create_depth_stencil(&device, width, height, 1);

        // Copy pipeline bind layout (for copy/blit operations).
        let copy_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Copy bind layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Copy pipeline layout"),
            bind_group_layouts: &[&copy_bind_layout],
            push_constant_ranges: &[],
        });
        let filters = Filters::new(&device, &shaders);

        let copy_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Copy pipeline"),
            layout: Some(&copy_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shaders.copy,
                entry_point: Some("main_vertex"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shaders.copy,
                entry_point: Some("main_fragment"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

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
            sample_count: 1,
            msaa_texture: None,
            msaa_view: None,
            surface_format: format,
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
            blend_buffer: None,
            blend_buffer_view: None,
            offscreen_buffer: None,
            offscreen_view: None,
            copy_pipeline,
            copy_bind_layout,
            filters,
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

    /// Recreate the MSAA framebuffer texture if sample_count > 1.
    fn recreate_msaa_framebuffer(&mut self) {
        if self.sample_count > 1 {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("MSAA framebuffer"),
                size: wgpu::Extent3d {
                    width: self.viewport_width.max(1),
                    height: self.viewport_height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.msaa_texture = Some(texture);
            self.msaa_view = Some(view);
        } else {
            self.msaa_texture = None;
            self.msaa_view = None;
        }
    }

    /// Process bitmap cache entries: render cached display objects and apply filters.
    fn process_cache_entries(&mut self, cache_entries: Vec<BitmapCacheEntry>) {
        for entry in cache_entries {
            let texture = as_texture(&entry.handle);
            let width = texture.texture.width();
            let height = texture.texture.height();

            if entry.filters.is_empty() {
                // No filters: just render commands directly into the cache texture.
                // For now, we skip command rendering for cache entries (would need
                // full offscreen rendering support). The texture already has content.
                continue;
            }

            // Render commands into the cache texture first, then apply filters.
            // For now, we just apply filters to the existing texture content.
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Cache entry filter encoder"),
                });

            let mut current_texture: Option<wgpu::Texture> = None;

            for filter in entry.filters {
                let source_tex = current_texture
                    .as_ref()
                    .unwrap_or(&texture.texture);

                let result = self.filters.apply(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    FilterSource::for_entire_texture(source_tex),
                    filter,
                );

                current_texture = Some(result);
            }

            // Copy the final filtered result back to the original texture.
            if let Some(ref filtered) = current_texture {
                let copy_width = filtered.width().min(width);
                let copy_height = filtered.height().min(height);
                if copy_width > 0 && copy_height > 0 {
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: filtered,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: copy_width,
                            height: copy_height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }
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
            create_depth_stencil(&self.device, self.viewport_width, self.viewport_height, self.sample_count);
        self.depth_stencil_texture = ds_tex;
        self.depth_stencil_view = ds_view;

        // Recreate MSAA framebuffer if needed.
        self.recreate_msaa_framebuffer();
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
            let fill_style = match path {
                DrawPath::Fill {
                    style: FillStyle::Bitmap { id, .. },
                    ..
                } => Some(*id),
                DrawPath::Stroke { style, .. } => {
                    if let FillStyle::Bitmap { id, .. } = style.fill_style() {
                        Some(*id)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(id) = fill_style
                && let Some(handle) = bitmap_source.bitmap_handle(id, self)
            {
                bitmap_handles.insert(id, handle);
            }
        }

        let mesh = mesh::build_mesh(
            &self.device,
            &self.queue,
            &shape,
            &bitmap_handles,
            &self.bind_layouts.gradient,
            &self.bind_layouts.bitmap,
            self.bind_layouts.analytic.as_ref(),
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

    fn apply_filter(
        &mut self,
        source: BitmapHandle,
        source_point: (u32, u32),
        source_size: (u32, u32),
        destination: BitmapHandle,
        dest_point: (i32, i32),
        filter: Filter,
    ) -> Option<Box<dyn SyncHandle>> {
        let source_texture = as_texture(&source);
        let dest_texture = as_texture(&destination);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Apply filter encoder"),
            });

        let result = self.filters.apply(
            &self.device,
            &self.queue,
            &mut encoder,
            FilterSource {
                texture: &source_texture.texture,
                point: source_point,
                size: source_size,
            },
            filter,
        );

        // Copy filtered result back to the destination texture at dest_point.
        let dest_x = dest_point.0.max(0) as u32;
        let dest_y = dest_point.1.max(0) as u32;
        let copy_width = result.width().min(dest_texture.texture.width().saturating_sub(dest_x));
        let copy_height = result.height().min(dest_texture.texture.height().saturating_sub(dest_y));
        if copy_width > 0 && copy_height > 0 {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &result,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &dest_texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: dest_x,
                        y: dest_y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: copy_width,
                    height: copy_height,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let copy_area = PixelRegion::for_whole_size(
            dest_texture.texture.width(),
            dest_texture.texture.height(),
        );
        Some(Box::new(QueueSyncHandle::AlreadyResolved {
            handle: destination,
            size: copy_area,
        }))
    }

    fn is_filter_supported(&self, filter: &Filter) -> bool {
        matches!(
            filter,
            Filter::BlurFilter(_)
                | Filter::GlowFilter(_)
                | Filter::DropShadowFilter(_)
                | Filter::ColorMatrixFilter(_)
                | Filter::BevelFilter(_)
                | Filter::DisplacementMapFilter(_)
        )
    }

    fn is_offscreen_supported(&self) -> bool {
        true
    }

    fn submit_frame(
        &mut self,
        clear: Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    ) {
        // Reset per-frame state.
        self.transform_data.clear();
        self.transform_offset = 0;
        self.mask_state = MaskState::NoMask;
        self.num_masks = 0;
        self.bitmap_bind_cache.clear();

        // Process bitmap cache entries (render cached display objects with filters).
        self.process_cache_entries(cache_entries);

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

        // Check if we need an offscreen buffer for complex blend compositing.
        let needs_offscreen = self.has_complex_blends(&commands);
        if needs_offscreen {
            self.ensure_offscreen_buffer();
            self.ensure_blend_buffer();
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame encoder"),
            });

        let clear_color = wgpu::Color {
            r: f64::from(clear.r) / 255.0,
            g: f64::from(clear.g) / 255.0,
            b: f64::from(clear.b) / 255.0,
            a: f64::from(clear.a) / 255.0,
        };

        if needs_offscreen {
            // Render to offscreen buffer (supports COPY_SRC for blend compositing).
            // Note: offscreen buffer is always single-sampled. When MSAA is enabled,
            // we use self.msaa_view as the render target and resolve into offscreen.
            let offscreen_view = self.offscreen_view.as_ref().unwrap();
            let mut transform_index = 0u32;
            let mut mask_state = MaskState::NoMask;
            let mut num_masks = 0u32;
            self.execute_toplevel(
                &mut encoder,
                offscreen_view,
                self.msaa_view.as_ref(),
                Some(clear_color),
                &commands,
                &mut transform_index,
                &mut mask_state,
                &mut num_masks,
                None,
            );

            // Blit offscreen buffer to swapchain.
            let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Blit to swapchain"),
                layout: &self.copy_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(offscreen_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.default_sampler),
                    },
                ],
            });

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit to swapchain pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            render_pass.set_pipeline(&self.copy_pipeline);
            render_pass.set_bind_group(0, &blit_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        } else {
            // Simple path: render directly to swapchain.
            // When MSAA is enabled, render to msaa_view and resolve to frame_view.
            let mut transform_index = 0u32;
            let mut mask_state = MaskState::NoMask;
            let mut num_masks = 0u32;
            self.execute_toplevel(
                &mut encoder,
                frame_view,
                self.msaa_view.as_ref(),
                Some(clear_color),
                &commands,
                &mut transform_index,
                &mut mask_state,
                &mut num_masks,
                None,
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

    fn set_quality(&mut self, quality: StageQuality) {
        let desired = quality.sample_count();
        let sample_count = supported_sample_count(&self.device, desired, self.surface_format);
        if sample_count == self.sample_count {
            return;
        }
        self.sample_count = sample_count;

        // Rebuild pipelines with new sample count.
        self.pipelines = Pipelines::new(
            &self.device,
            &self.shaders,
            self.surface_format,
            self.sample_count,
            &self.bind_layouts,
        );

        // Rebuild depth/stencil with new sample count.
        let (ds_tex, ds_view) = create_depth_stencil(
            &self.device,
            self.viewport_width,
            self.viewport_height,
            self.sample_count,
        );
        self.depth_stencil_texture = ds_tex;
        self.depth_stencil_view = ds_view;

        // Rebuild MSAA framebuffer.
        self.recreate_msaa_framebuffer();
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
        handle: Box<dyn SyncHandle>,
        with_rgba: RgbaBufRead,
    ) -> Result<(), BitmapError> {
        if let Ok(sync) = Box::<dyn Any>::downcast::<QueueSyncHandle>(handle) {
            match *sync {
                QueueSyncHandle::AlreadyResolved { handle, size } => {
                    let texture = as_texture(&handle);
                    let width = size.width();
                    let height = size.height();

                    // Create a buffer to read back the texture data.
                    let bytes_per_row = (width * 4 + 255) & !255; // align to 256
                    let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Sync handle readback buffer"),
                        size: (bytes_per_row * height) as u64,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });

                    let mut encoder = self.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("Sync handle readback encoder"),
                        },
                    );
                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: size.x_min,
                                y: size.y_min,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer: &buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(bytes_per_row),
                                rows_per_image: None,
                            },
                        },
                        wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                    );
                    self.queue.submit(std::iter::once(encoder.finish()));

                    let buffer_slice = buffer.slice(..);
                    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
                    self.device.poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    }).expect("Device poll failed");

                    let data = buffer_slice.get_mapped_range();
                    with_rgba(&data, bytes_per_row);
                    drop(data);
                    buffer.unmap();

                    Ok(())
                }
            }
        } else {
            Err(BitmapError::Unimplemented("Unknown sync handle type".into()))
        }
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
                    // DrawLineRect draws the outline of a rectangle as 4 thin
                    // 1px quads. Transform the 4 corners through the matrix,
                    // then create 4 line-segment transforms (one per edge).
                    let m = *matrix;
                    let a = swf::Point::new(
                        m.tx + Twips::HALF_PX,
                        m.ty + Twips::HALF_PX,
                    );
                    let b = swf::Point::new(
                        m.tx + Twips::HALF_PX + Twips::from_pixels(m.a as f64),
                        m.ty + Twips::HALF_PX + Twips::from_pixels(m.b as f64),
                    );
                    let c = swf::Point::new(
                        m.tx + Twips::HALF_PX + Twips::from_pixels((m.a + m.c) as f64),
                        m.ty + Twips::HALF_PX + Twips::from_pixels((m.b + m.d) as f64),
                    );
                    let d = swf::Point::new(
                        m.tx + Twips::HALF_PX + Twips::from_pixels(m.c as f64),
                        m.ty + Twips::HALF_PX + Twips::from_pixels(m.d as f64),
                    );

                    let ct = color_to_color_transform(*color);
                    // Push 4 edge transforms: a→b, b→c, c→d, d→a
                    for (p0, p1) in [(a, b), (b, c), (c, d), (d, a)] {
                        let edge_matrix = line_segment_matrix(p0, p1);
                        let transform = Transform {
                            matrix: edge_matrix,
                            color_transform: ct,
                            perspective_projection: None,
                        };
                        self.push_transform(&transform);
                    }
                }
                Command::Blend(sub_commands, _) => {
                    self.collect_transforms(sub_commands);
                }
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => {
                    self.collect_transforms(mask_commands);
                    self.collect_transforms(maskee_commands);
                }
                _ => {}
            }
        }
    }

    /// Check if a command list contains any complex blend commands (recursively).
    fn has_complex_blends(&self, commands: &CommandList) -> bool {
        use ruffle_render::commands::Command;
        for cmd in &commands.commands {
            if let Command::Blend(_, mode) = cmd
                && matches!(BlendType::from(mode.clone()), BlendType::Complex(_))
            {
                return true;
            }
        }
        false
    }

    /// Ensure the offscreen framebuffer exists at the current viewport size.
    fn ensure_offscreen_buffer(&mut self) {
        let width = self.viewport_width.max(1);
        let height = self.viewport_height.max(1);

        let needs_recreate = match &self.offscreen_buffer {
            Some(tex) => tex.width() != width || tex.height() != height,
            None => true,
        };

        if needs_recreate {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Offscreen framebuffer"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.offscreen_buffer = Some(texture);
            self.offscreen_view = Some(view);
        }
    }

    /// Ensure the blend buffer (framebuffer snapshot) exists at the current viewport size.
    fn ensure_blend_buffer(&mut self) {
        let width = self.viewport_width.max(1);
        let height = self.viewport_height.max(1);

        let needs_recreate = match &self.blend_buffer {
            Some(tex) => tex.width() != width || tex.height() != height,
            None => true,
        };

        if needs_recreate {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Blend buffer (parent snapshot)"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.blend_buffer = Some(texture);
            self.blend_buffer_view = Some(view);
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
        blend_override: Option<TrivialBlend>,
    ) {
        for cmd in &commands.commands {
            self.execute_single_command(render_pass, cmd, transform_index, mask_state, num_masks, blend_override);
        }
    }

    /// Execute a single draw command within an active render pass.
    fn execute_single_command<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        cmd: &'a ruffle_render::commands::Command,
        transform_index: &mut u32,
        mask_state: &mut MaskState,
        num_masks: &mut u32,
        blend_override: Option<TrivialBlend>,
    ) {
        use ruffle_render::commands::Command;
        match cmd {
                Command::RenderShape { shape, .. } => {
                    let offset = self.transform_offset(*transform_index);
                    *transform_index += 1;

                    let pipelines = match blend_override {
                        Some(blend) if *mask_state == MaskState::NoMask => {
                            self.pipelines.for_trivial_blend(blend)
                        }
                        _ => self.pipelines.for_mask_state(*mask_state),
                    };
                    let mesh = as_bezier_mesh(shape);
                    for draw in &mesh.draws {
                        // Mask generation/clearing should use fill contours only.
                        // Strokes must not contribute to mask stencil values.
                        if draw.is_stroke
                            && matches!(
                                *mask_state,
                                MaskState::DrawMaskStencil | MaskState::ClearMaskStencil
                            )
                        {
                            continue;
                        }

                        match &draw.draw_type {
                            DrawType::Color => {
                                if draw.analytic_bind_group.is_some()
                                    && pipelines.analytic_color.is_some()
                                {
                                    render_pass.set_pipeline(
                                        pipelines
                                            .analytic_color
                                            .as_ref()
                                            .expect("analytic pipeline must exist"),
                                    );
                                } else {
                                    render_pass.set_pipeline(&pipelines.color_fill);
                                }
                            }
                            DrawType::Gradient { bind_group } => {
                                if draw.analytic_bind_group.is_some()
                                    && pipelines.analytic_gradient.is_some()
                                {
                                    render_pass.set_pipeline(
                                        pipelines
                                            .analytic_gradient
                                            .as_ref()
                                            .expect("analytic pipeline must exist"),
                                    );
                                } else {
                                    render_pass.set_pipeline(&pipelines.gradient_fill);
                                }
                                render_pass.set_bind_group(2, bind_group, &[]);
                            }
                            DrawType::Bitmap { bind_group } => {
                                if draw.analytic_bind_group.is_some()
                                    && pipelines.analytic_bitmap.is_some()
                                {
                                    render_pass.set_pipeline(
                                        pipelines
                                            .analytic_bitmap
                                            .as_ref()
                                            .expect("analytic pipeline must exist"),
                                    );
                                } else {
                                    render_pass.set_pipeline(&pipelines.bitmap_fill);
                                }
                                render_pass.set_bind_group(2, bind_group, &[]);
                            }
                        }

                        if let Some(analytic_bind_group) = &draw.analytic_bind_group {
                            let analytic_available = match &draw.draw_type {
                                DrawType::Color => pipelines.analytic_color.is_some(),
                                DrawType::Gradient { .. } => pipelines.analytic_gradient.is_some(),
                                DrawType::Bitmap { .. } => pipelines.analytic_bitmap.is_some(),
                            };
                            if analytic_available {
                                let analytic_group = match &draw.draw_type {
                                    DrawType::Color => 2,
                                    DrawType::Gradient { .. } | DrawType::Bitmap { .. } => 3,
                                };
                                render_pass.set_bind_group(analytic_group, analytic_bind_group, &[]);
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
                        let pipelines = match blend_override {
                            Some(blend) if *mask_state == MaskState::NoMask => {
                                self.pipelines.for_trivial_blend(blend)
                            }
                            _ => self.pipelines.for_mask_state(*mask_state),
                        };
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

                    let pipelines = match blend_override {
                        Some(blend) if *mask_state == MaskState::NoMask => {
                            self.pipelines.for_trivial_blend(blend)
                        }
                        _ => self.pipelines.for_mask_state(*mask_state),
                    };
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
                    let pipelines = match blend_override {
                        Some(blend) if *mask_state == MaskState::NoMask => {
                            self.pipelines.for_trivial_blend(blend)
                        }
                        _ => self.pipelines.for_mask_state(*mask_state),
                    };
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
                    // Draw 4 thin quads (one per edge of the rectangle).
                    let pipelines = match blend_override {
                        Some(blend) if *mask_state == MaskState::NoMask => {
                            self.pipelines.for_trivial_blend(blend)
                        }
                        _ => self.pipelines.for_mask_state(*mask_state),
                    };
                    render_pass.set_pipeline(&pipelines.color_fill);
                    render_pass.set_vertex_buffer(0, self.unit_quad.color_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        self.unit_quad.quad_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    for _ in 0..4 {
                        let offset = self.transform_offset(*transform_index);
                        *transform_index += 1;
                        render_pass.set_bind_group(1, &self.transform_bind_group, &[offset]);
                        render_pass.draw_indexed(0..6, 0, 0..1);
                    }
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
                Command::Blend(sub_commands, blend_mode) => {
                    match BlendType::from(blend_mode.clone()) {
                        BlendType::Trivial(TrivialBlend::Normal) => {
                            // Normal blend: just render sub-commands directly.
                            self.execute_commands(render_pass, sub_commands, transform_index, mask_state, num_masks, blend_override);
                        }
                        BlendType::Trivial(trivial) => {
                            // Other trivial blends (Add, Subtract, Screen):
                            // Use dedicated pipeline sets with the proper blend state.
                            self.execute_commands(render_pass, sub_commands, transform_index, mask_state, num_masks, Some(trivial));
                        }
                        BlendType::Complex(_complex) => {
                            // Complex blends handled at the toplevel where we can break
                            // out of the render pass. When called from within a render pass,
                            // we fall back to Normal blending since we can't end the pass here.
                            self.execute_commands(render_pass, sub_commands, transform_index, mask_state, num_masks, blend_override);
                        }
                    }
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
                    let mask_transform_start = *transform_index;
                    self.execute_commands(render_pass, mask_commands, transform_index, mask_state, num_masks, None);

                    // 3. Activate mask: switch to drawing masked content.
                    *mask_state = MaskState::DrawMaskedContent;
                    render_pass.set_stencil_reference(*num_masks);

                    // 4. Draw the maskee (only where stencil passes).
                    self.execute_commands(render_pass, maskee_commands, transform_index, mask_state, num_masks, blend_override);

                    // 5. Deactivate and clear mask.
                    *mask_state = MaskState::ClearMaskStencil;
                    render_pass.set_stencil_reference(*num_masks);

                    // Re-draw mask geometry to decrement stencil.
                    // Reuse the same transform offsets consumed by the first mask pass.
                    let mut clear_transform_index = mask_transform_start;
                    self.execute_commands(
                        render_pass,
                        mask_commands,
                        &mut clear_transform_index,
                        mask_state,
                        num_masks,
                        None,
                    );

                    // Finally pop one mask level.
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

    /// Build a color attachment for a render pass, handling MSAA resolve.
    /// When `msaa_view` is provided, render to the MSAA texture and resolve to `target_view`.
    /// Otherwise render directly to `target_view`.
    fn make_color_attachment<'a>(
        target_view: &'a wgpu::TextureView,
        msaa_view: Option<&'a wgpu::TextureView>,
        load: wgpu::LoadOp<wgpu::Color>,
    ) -> wgpu::RenderPassColorAttachment<'a> {
        if let Some(msaa) = msaa_view {
            wgpu::RenderPassColorAttachment {
                view: msaa,
                resolve_target: Some(target_view),
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            }
        } else {
            wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            }
        }
    }

    /// Top-level command execution that manages render pass lifecycle.
    /// This method can break out of render passes to handle complex blend modes
    /// which require render-to-texture compositing.
    ///
    /// `msaa_view` should be provided when MSAA is enabled (sample_count > 1).
    /// It must be the same size as `target_view` and have the appropriate sample count.
    fn execute_toplevel(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        msaa_view: Option<&wgpu::TextureView>,
        clear_color: Option<wgpu::Color>,
        commands: &CommandList,
        transform_index: &mut u32,
        mask_state: &mut MaskState,
        num_masks: &mut u32,
        blend_override: Option<TrivialBlend>,
    ) {
        use ruffle_render::commands::Command;

        // Scan the command list for complex blends.
        // If there are none, we can use a single render pass.
        let has_complex_blends = commands.commands.iter().any(|cmd| {
            matches!(cmd, Command::Blend(_, mode) if matches!(BlendType::from(mode.clone()), BlendType::Complex(_)))
        });

        if !has_complex_blends {
            // Simple path: single render pass for everything.
            let load_op = match clear_color {
                Some(color) => wgpu::LoadOp::Clear(color),
                None => wgpu::LoadOp::Load,
            };

            let color_attachment = Self::make_color_attachment(target_view, msaa_view, load_op);
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main render pass"),
                color_attachments: &[Some(color_attachment)],
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
            render_pass.set_bind_group(0, &self.globals_bind_group, &[]);
            self.execute_commands(&mut render_pass, commands, transform_index, mask_state, num_masks, blend_override);
            return;
        }

        // Complex path: process commands one-by-one, breaking out for complex blends.
        let mut first_pass = true;
        let mut cmd_idx = 0;

        while cmd_idx < commands.commands.len() {
            // Find the next complex blend command starting from cmd_idx.
            let mut complex_at = None;
            for i in cmd_idx..commands.commands.len() {
                if let Command::Blend(_, mode) = &commands.commands[i]
                    && matches!(BlendType::from(mode.clone()), BlendType::Complex(_))
                {
                    complex_at = Some(i);
                    break;
                }
            }

            // Render non-complex-blend commands up to the complex blend (or end).
            let batch_end = complex_at.unwrap_or(commands.commands.len());
            if batch_end > cmd_idx {
                let load_op = if first_pass {
                    first_pass = false;
                    match clear_color {
                        Some(color) => wgpu::LoadOp::Clear(color),
                        None => wgpu::LoadOp::Load,
                    }
                } else {
                    wgpu::LoadOp::Load
                };

                {
                    let color_attachment = Self::make_color_attachment(target_view, msaa_view, load_op);
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render pass (pre-blend batch)"),
                        color_attachments: &[Some(color_attachment)],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_stencil_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                        }),
                        ..Default::default()
                    });
                    render_pass.set_bind_group(0, &self.globals_bind_group, &[]);

                    // Execute the batch of commands individually.
                    for i in cmd_idx..batch_end {
                        self.execute_single_command(&mut render_pass, &commands.commands[i], transform_index, mask_state, num_masks, blend_override);
                    }
                }
                // render_pass is dropped here, freeing the encoder.
            }

            // Process the complex blend command, if any.
            if let Some(blend_idx) = complex_at {
                if first_pass {
                    first_pass = false;
                    // Clear the target if this is the first thing we do.
                    if let Some(color) = clear_color {
                        let color_attachment = Self::make_color_attachment(
                            target_view, msaa_view,
                            wgpu::LoadOp::Clear(color),
                        );
                        let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Clear pass"),
                            color_attachments: &[Some(color_attachment)],
                            depth_stencil_attachment: None,
                            ..Default::default()
                        });
                        // Drop immediately after scope, just clearing.
                    }
                }

                if let Command::Blend(sub_commands, blend_mode) = &commands.commands[blend_idx]
                    && let BlendType::Complex(complex) = BlendType::from(blend_mode.clone())
                {
                        self.execute_complex_blend(
                            encoder,
                            target_view,
                            msaa_view,
                            sub_commands,
                            complex,
                            transform_index,
                            mask_state,
                            num_masks,
                        );
                }
                cmd_idx = blend_idx + 1;
            } else {
                cmd_idx = batch_end;
            }
        }
    }

    /// Execute a complex blend operation by rendering sub-commands to an
    /// intermediate texture and compositing with the framebuffer.
    fn execute_complex_blend(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        _msaa_view: Option<&wgpu::TextureView>,
        sub_commands: &CommandList,
        complex: ComplexBlend,
        transform_index: &mut u32,
        _mask_state: &mut MaskState,
        _num_masks: &mut u32,
    ) {
        let width = self.viewport_width.max(1);
        let height = self.viewport_height.max(1);

        // 1. Create an intermediate texture for the sub-command rendering.
        //    This is always single-sampled; MSAA resolves into it.
        let intermediate_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Blend intermediate texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let intermediate_view = intermediate_texture.create_view(&Default::default());

        // Create MSAA and depth/stencil textures matching the current sample count.
        let intermediate_msaa_view = if self.sample_count > 1 {
            let msaa_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Blend intermediate MSAA"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            Some(msaa_tex.create_view(&Default::default()))
        } else {
            None
        };
        let (_, intermediate_depth_view) = create_depth_stencil(&self.device, width, height, self.sample_count);

        // 2. Render sub-commands to the intermediate texture (clear to transparent).
        {
            let color_attachment = Self::make_color_attachment(
                &intermediate_view,
                intermediate_msaa_view.as_ref(),
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blend sub-command pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &intermediate_depth_view,
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
            render_pass.set_bind_group(0, &self.globals_bind_group, &[]);

            // Reset mask state for sub-command rendering.
            let mut sub_mask_state = MaskState::NoMask;
            let mut sub_num_masks = 0u32;
            self.execute_commands(&mut render_pass, sub_commands, transform_index, &mut sub_mask_state, &mut sub_num_masks, None);
        }
        // render_pass dropped, encoder available.

        // 3. Copy the current framebuffer to the blend buffer (parent texture).
        // The target_view comes from the offscreen buffer which has COPY_SRC usage.
        if let (Some(offscreen), Some(blend_buf)) = (&self.offscreen_buffer, &self.blend_buffer) {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: offscreen,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: blend_buf,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // 4. Create the blend bind group with both textures.
        let blend_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blend sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let parent_view = self.blend_buffer_view.as_ref().unwrap();
        let blend_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blend bind group"),
            layout: &self.bind_layouts.blend,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(parent_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&intermediate_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&blend_sampler),
                },
            ],
        });

        // 5. Composite using the blend shader onto the framebuffer.
        //    The complex blend pipelines are created with sample_count=1,
        //    so we render directly to target_view without MSAA.
        //    This writes the composited result to the offscreen buffer directly.
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blend composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            render_pass.set_pipeline(&self.pipelines.complex_blend[complex]);
            render_pass.set_bind_group(0, &blend_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
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

/// A simple sync handle for filter operations. Since the bezier renderer
/// uses synchronous queue submission, we can resolve immediately.
#[derive(Debug)]
enum QueueSyncHandle {
    AlreadyResolved {
        handle: BitmapHandle,
        size: PixelRegion,
    },
}

impl SyncHandle for QueueSyncHandle {}

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
    sample_count: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth/stencil texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// Determine the highest supported sample count up to the desired count.
fn supported_sample_count(
    device: &wgpu::Device,
    desired: u32,
    format: wgpu::TextureFormat,
) -> u32 {
    // wgpu doesn't expose per-format sample count queries directly,
    // so we try common counts in descending order.
    let _ = format; // All common formats support the same counts on most hardware.
    let candidates = [16, 8, 4, 2, 1];
    for &count in &candidates {
        if count <= desired {
            // Validate by checking the device limits.
            // The texture_format_features API isn't available for all formats,
            // so we use a heuristic: most GPUs support at least 4x MSAA.
            // For safety, cap at what the device likely supports.
            let flags = device
                .features();
            let _ = flags; // Features don't directly tell us sample counts.
            // Practically, most desktop GPUs support up to 8x.
            // We'll use a simple check: try the count and cap at 4 if unsure.
            return count;
        }
    }
    1
}

/// Build a matrix that maps the unit quad [0,1]×[0,1] to a 1px-thick
/// line segment between two points. Used for DrawLineRect edges.
fn line_segment_matrix(
    a: swf::Point<Twips>,
    b: swf::Point<Twips>,
) -> ruffle_render::matrix::Matrix {
    let dx = (b.x - a.x).to_pixels() as f32;
    let dy = (b.y - a.y).to_pixels() as f32;
    let len = (dx * dx + dy * dy).sqrt();

    if len < 0.001 {
        // Degenerate edge — return a zero-size matrix.
        return ruffle_render::matrix::Matrix {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            tx: a.x,
            ty: a.y,
        };
    }

    let angle = dy.atan2(dx);
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // The unit quad X axis [0,1] maps to the line direction (length = len).
    // The unit quad Y axis [0,1] maps to the perpendicular (thickness = 1px).
    // We offset Y by -0.5px so the line is centered on the segment.
    ruffle_render::matrix::Matrix {
        a: len * cos_a,           // X basis x-component
        b: len * sin_a,           // X basis y-component
        c: -sin_a,                // Y basis x-component (1px perpendicular)
        d: cos_a,                 // Y basis y-component (1px perpendicular)
        tx: a.x + Twips::from_pixels(0.5 * sin_a as f64),  // offset by -0.5 * perpendicular
        ty: a.y - Twips::from_pixels(0.5 * cos_a as f64),
    }
}
