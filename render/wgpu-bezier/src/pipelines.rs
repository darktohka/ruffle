//! GPU render pipeline definitions for the Bézier renderer.
//!
//! Three main pipelines correspond to the three fill types:
//! - **Color pipeline**: Uses `BezierVertex` (position + uv + fill_type + color).
//! - **Gradient pipeline**: Uses `BezierTexVertex` (position + uv + fill_type).
//! - **Bitmap pipeline**: Uses `BezierTexVertex` (position + uv + fill_type).
//!
//! Each pipeline also has stencil variants for masking support:
//! - **NoMask / DrawMaskedContent**: Normal stencil states.
//! - **DrawMaskStencil**: Writes to stencil only (no color output).
//! - **ClearMaskStencil**: Decrements stencil (no color output).

use crate::blend::ComplexBlend;
use crate::shaders::Shaders;
use crate::{BezierTexVertex, BezierVertex, MaskState};
use enum_map::{EnumMap, enum_map};

/// Bind group layouts used by the Bézier renderer.
#[derive(Debug)]
pub struct BindLayouts {
    /// Group 0: Global uniforms (view matrix).
    pub globals: wgpu::BindGroupLayout,
    /// Group 1: Per-object transforms (world matrix + color transform). Dynamic offset.
    pub transforms: wgpu::BindGroupLayout,
    /// Group 2: Gradient fill resources (texture transforms + gradient uniform + texture + sampler).
    pub gradient: wgpu::BindGroupLayout,
    /// Group 2: Bitmap fill resources (texture transforms + texture + sampler).
    pub bitmap: wgpu::BindGroupLayout,
    /// Layout for RenderBitmap (texture + sampler only).
    pub render_bitmap: wgpu::BindGroupLayout,
    /// Layout for complex blend compositing (parent_texture + current_texture + sampler).
    pub blend: wgpu::BindGroupLayout,
}

impl BindLayouts {
    pub fn new(device: &wgpu::Device) -> Self {
        let globals = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bezier globals layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<crate::GlobalsUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let transforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bezier transforms layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<crate::Transforms>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let gradient = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bezier gradient layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<crate::TextureTransforms>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<crate::GradientUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bitmap = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bezier bitmap layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<crate::TextureTransforms>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let render_bitmap = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bezier render_bitmap layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
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

        let blend = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bezier blend layout"),
            entries: &[
                // binding 0: parent (background) texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // binding 1: current (foreground) texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // binding 2: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        Self {
            globals,
            transforms,
            gradient,
            bitmap,
            render_bitmap,
            blend,
        }
    }
}

/// A set of pipelines for one stencil/mask state.
#[derive(Debug)]
pub struct PipelineSet {
    /// Solid color fill pipeline (per-vertex color + Loop-Blinn).
    pub color_fill: wgpu::RenderPipeline,
    /// Gradient fill pipeline (gradient texture + Loop-Blinn).
    pub gradient_fill: wgpu::RenderPipeline,
    /// Bitmap fill pipeline (bitmap texture + Loop-Blinn).
    pub bitmap_fill: wgpu::RenderPipeline,
    /// Direct bitmap render pipeline (for Command::RenderBitmap).
    pub render_bitmap: wgpu::RenderPipeline,
}

/// All render pipelines for the Bézier renderer, organized by mask state.
#[derive(Debug)]
pub struct Pipelines {
    /// Pipelines for normal rendering (no mask).
    pub no_mask: PipelineSet,
    /// Pipelines for writing to the stencil buffer (mask geometry).
    pub draw_mask_stencil: PipelineSet,
    /// Pipelines for rendering content clipped by the stencil mask.
    pub draw_masked_content: PipelineSet,
    /// Pipelines for clearing the stencil buffer (decrementing).
    pub clear_mask_stencil: PipelineSet,
    /// Complex blend mode compositing pipelines (one per ComplexBlend variant).
    pub complex_blend: EnumMap<ComplexBlend, wgpu::RenderPipeline>,
}

impl Pipelines {
    pub fn new(
        device: &wgpu::Device,
        shaders: &Shaders,
        format: wgpu::TextureFormat,
        sample_count: u32,
        layouts: &BindLayouts,
    ) -> Self {
        let complex_blend = create_complex_blend_pipelines(
            device,
            shaders,
            format,
            &layouts.blend,
        );

        Self {
            no_mask: create_pipeline_set(
                device, shaders, format, sample_count, layouts,
                MaskState::NoMask,
            ),
            draw_mask_stencil: create_pipeline_set(
                device, shaders, format, sample_count, layouts,
                MaskState::DrawMaskStencil,
            ),
            draw_masked_content: create_pipeline_set(
                device, shaders, format, sample_count, layouts,
                MaskState::DrawMaskedContent,
            ),
            clear_mask_stencil: create_pipeline_set(
                device, shaders, format, sample_count, layouts,
                MaskState::ClearMaskStencil,
            ),
            complex_blend,
        }
    }

    /// Get the pipeline set for the current mask state.
    pub fn for_mask_state(&self, state: MaskState) -> &PipelineSet {
        match state {
            MaskState::NoMask => &self.no_mask,
            MaskState::DrawMaskStencil => &self.draw_mask_stencil,
            MaskState::DrawMaskedContent => &self.draw_masked_content,
            MaskState::ClearMaskStencil => &self.clear_mask_stencil,
        }
    }
}

/// Build the stencil and depth/stencil state for a given mask state.
fn stencil_config(mask_state: MaskState) -> (wgpu::DepthStencilState, wgpu::ColorWrites) {
    let (stencil_face, color_writes) = match mask_state {
        MaskState::NoMask => (
            wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Always,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Keep,
            },
            wgpu::ColorWrites::ALL,
        ),
        MaskState::DrawMaskStencil => (
            wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::IncrementClamp,
            },
            wgpu::ColorWrites::empty(),
        ),
        MaskState::DrawMaskedContent => (
            wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Keep,
            },
            wgpu::ColorWrites::ALL,
        ),
        MaskState::ClearMaskStencil => (
            wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::DecrementClamp,
            },
            wgpu::ColorWrites::empty(),
        ),
    };

    let depth_stencil = wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: wgpu::StencilState {
            front: stencil_face,
            back: stencil_face,
            read_mask: 0xff,
            write_mask: 0xff,
        },
        bias: wgpu::DepthBiasState::default(),
    };

    (depth_stencil, color_writes)
}

/// Create a full set of pipelines for a given mask state.
fn create_pipeline_set(
    device: &wgpu::Device,
    shaders: &Shaders,
    format: wgpu::TextureFormat,
    sample_count: u32,
    layouts: &BindLayouts,
    mask_state: MaskState,
) -> PipelineSet {
    let (depth_stencil, color_writes) = stencil_config(mask_state);
    let blend_state = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;

    let suffix = match mask_state {
        MaskState::NoMask => "",
        MaskState::DrawMaskStencil => " [mask-write]",
        MaskState::DrawMaskedContent => " [masked]",
        MaskState::ClearMaskStencil => " [mask-clear]",
    };

    let color_vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<BezierVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Sint32,
                offset: 16,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 20,
                shader_location: 3,
            },
        ],
    };

    let tex_vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<BezierTexVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Sint32,
                offset: 16,
                shader_location: 2,
            },
        ],
    };

    let color_target = Some(wgpu::ColorTargetState {
        format,
        blend: Some(blend_state),
        write_mask: color_writes,
    });

    let multisample = wgpu::MultisampleState {
        count: sample_count,
        mask: !0,
        alpha_to_coverage_enabled: false,
    };

    let primitive = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        ..Default::default()
    };

    // ---- Color fill pipeline ----
    let color_fill_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Bezier color fill layout{suffix}")),
            bind_group_layouts: &[&layouts.globals, &layouts.transforms],
            push_constant_ranges: &[],
        });

    let color_fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("Bezier color fill pipeline{suffix}")),
        layout: Some(&color_fill_layout),
        vertex: wgpu::VertexState {
            module: &shaders.bezier_fill,
            entry_point: Some("main_vertex"),
            buffers: &[color_vertex_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shaders.bezier_fill,
            entry_point: Some("main_fragment"),
            targets: &[color_target.clone()],
            compilation_options: Default::default(),
        }),
        primitive,
        depth_stencil: Some(depth_stencil.clone()),
        multisample,
        multiview: None,
        cache: None,
    });

    // ---- Gradient fill pipeline ----
    let gradient_fill_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Bezier gradient fill layout{suffix}")),
            bind_group_layouts: &[&layouts.globals, &layouts.transforms, &layouts.gradient],
            push_constant_ranges: &[],
        });

    let gradient_fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("Bezier gradient fill pipeline{suffix}")),
        layout: Some(&gradient_fill_layout),
        vertex: wgpu::VertexState {
            module: &shaders.bezier_gradient,
            entry_point: Some("main_vertex"),
            buffers: &[tex_vertex_layout.clone()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shaders.bezier_gradient,
            entry_point: Some("main_fragment"),
            targets: &[color_target.clone()],
            compilation_options: Default::default(),
        }),
        primitive,
        depth_stencil: Some(depth_stencil.clone()),
        multisample,
        multiview: None,
        cache: None,
    });

    // ---- Bitmap fill pipeline ----
    let bitmap_fill_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Bezier bitmap fill layout{suffix}")),
            bind_group_layouts: &[&layouts.globals, &layouts.transforms, &layouts.bitmap],
            push_constant_ranges: &[],
        });

    let bitmap_fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("Bezier bitmap fill pipeline{suffix}")),
        layout: Some(&bitmap_fill_layout),
        vertex: wgpu::VertexState {
            module: &shaders.bezier_bitmap,
            entry_point: Some("main_vertex"),
            buffers: &[tex_vertex_layout.clone()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shaders.bezier_bitmap,
            entry_point: Some("main_fragment"),
            targets: &[color_target.clone()],
            compilation_options: Default::default(),
        }),
        primitive,
        depth_stencil: Some(depth_stencil.clone()),
        multisample,
        multiview: None,
        cache: None,
    });

    // ---- RenderBitmap pipeline ----
    let render_bitmap_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Bezier render_bitmap layout{suffix}")),
            bind_group_layouts: &[&layouts.globals, &layouts.transforms, &layouts.render_bitmap],
            push_constant_ranges: &[],
        });

    let render_bitmap = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("Bezier render_bitmap pipeline{suffix}")),
        layout: Some(&render_bitmap_layout),
        vertex: wgpu::VertexState {
            module: &shaders.render_bitmap,
            entry_point: Some("main_vertex"),
            buffers: &[tex_vertex_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shaders.render_bitmap,
            entry_point: Some("main_fragment"),
            targets: &[color_target],
            compilation_options: Default::default(),
        }),
        primitive,
        depth_stencil: Some(depth_stencil),
        multisample,
        multiview: None,
        cache: None,
    });

    PipelineSet {
        color_fill,
        gradient_fill,
        bitmap_fill,
        render_bitmap,
    }
}

/// Create complex blend mode compositing pipelines.
/// These use full-screen triangles generated from vertex_index (no vertex buffers).
/// They composite a foreground texture onto a background texture using the blend shader.
fn create_complex_blend_pipelines(
    device: &wgpu::Device,
    shaders: &Shaders,
    format: wgpu::TextureFormat,
    blend_layout: &wgpu::BindGroupLayout,
) -> EnumMap<ComplexBlend, wgpu::RenderPipeline> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Bezier complex blend layout"),
        bind_group_layouts: &[blend_layout],
        push_constant_ranges: &[],
    });

    enum_map! {
        blend => {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("Bezier complex blend: {blend:?}")),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shaders.blend_shaders[blend],
                    entry_point: Some("main_vertex"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shaders.blend_shaders[blend],
                    entry_point: Some("main_fragment"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
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
            })
        }
    }
}
