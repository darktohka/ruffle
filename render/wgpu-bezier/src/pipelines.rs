//! GPU render pipeline definitions for the Bézier renderer.
//!
//! Three main pipelines correspond to the three fill types:
//! - **Color pipeline**: Uses `BezierVertex` (position + uv + fill_type + color).
//! - **Gradient pipeline**: Uses `BezierTexVertex` (position + uv + fill_type).
//! - **Bitmap pipeline**: Uses `BezierTexVertex` (position + uv + fill_type).

use crate::shaders::Shaders;
use crate::{BezierTexVertex, BezierVertex};

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
                // Texture transforms
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
                // Gradient uniforms
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
                // Gradient LUT texture
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
                // Sampler
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
                // Texture transforms
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
                // Bitmap texture
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
                // Sampler
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
        }
    }
}

/// All render pipelines for the Bézier renderer.
#[derive(Debug)]
pub struct Pipelines {
    /// Solid color fill pipeline (per-vertex color + Loop-Blinn).
    pub color_fill: wgpu::RenderPipeline,
    /// Gradient fill pipeline (gradient texture + Loop-Blinn).
    pub gradient_fill: wgpu::RenderPipeline,
    /// Bitmap fill pipeline (bitmap texture + Loop-Blinn).
    pub bitmap_fill: wgpu::RenderPipeline,
}

impl Pipelines {
    pub fn new(
        device: &wgpu::Device,
        shaders: &Shaders,
        format: wgpu::TextureFormat,
        sample_count: u32,
        layouts: &BindLayouts,
    ) -> Self {
        // Vertex buffer layout for BezierVertex (color fills).
        let color_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BezierVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: vec2<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                // uv: vec2<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
                // fill_type: i32
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Sint32,
                    offset: 16,
                    shader_location: 2,
                },
                // color: vec4<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 20,
                    shader_location: 3,
                },
            ],
        };

        // Vertex buffer layout for BezierTexVertex (gradient/bitmap fills).
        let tex_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BezierTexVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: vec2<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                // uv: vec2<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
                // fill_type: i32
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Sint32,
                    offset: 16,
                    shader_location: 2,
                },
            ],
        };

        let blend_state = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;

        let stencil_state = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Keep,
        };

        let depth_stencil = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState {
                front: stencil_state,
                back: stencil_state,
                read_mask: 0xff,
                write_mask: 0xff,
            },
            bias: wgpu::DepthBiasState::default(),
        });

        // ---- Color fill pipeline ----
        let color_fill_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bezier color fill layout"),
                bind_group_layouts: &[&layouts.globals, &layouts.transforms],
                push_constant_ranges: &[],
            });

        let color_fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bezier color fill pipeline"),
            layout: Some(&color_fill_layout),
            vertex: wgpu::VertexState {
                module: &shaders.bezier_fill,
                entry_point: Some("main_vertex"),
                buffers: &[color_vertex_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shaders.bezier_fill,
                entry_point: Some("main_fragment"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Don't cull — both sides of triangles may be visible.
                ..Default::default()
            },
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // ---- Gradient fill pipeline ----
        let gradient_fill_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bezier gradient fill layout"),
                bind_group_layouts: &[&layouts.globals, &layouts.transforms, &layouts.gradient],
                push_constant_ranges: &[],
            });

        let gradient_fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bezier gradient fill pipeline"),
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
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend_state),
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
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // ---- Bitmap fill pipeline ----
        let bitmap_fill_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bezier bitmap fill layout"),
                bind_group_layouts: &[&layouts.globals, &layouts.transforms, &layouts.bitmap],
                push_constant_ranges: &[],
            });

        let bitmap_fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bezier bitmap fill pipeline"),
            layout: Some(&bitmap_fill_layout),
            vertex: wgpu::VertexState {
                module: &shaders.bezier_bitmap,
                entry_point: Some("main_vertex"),
                buffers: &[tex_vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shaders.bezier_bitmap,
                entry_point: Some("main_fragment"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend_state),
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
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            color_fill,
            gradient_fill,
            bitmap_fill,
        }
    }
}
