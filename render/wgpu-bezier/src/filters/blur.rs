//! Blur filter implementation.

use crate::filters::{
    FilterQuad, FilterSource, FilterVertex, VERTEX_BUFFERS_DESCRIPTION_FILTERS,
    create_filter_texture,
};
use crate::shaders::Shaders;
use bytemuck::{Pod, Zeroable};
use swf::BlurFilter as BlurFilterArgs;

/// Matches `struct Filter` in `blur.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
struct BlurUniform {
    direction: [f32; 2],
    full_size: f32,
    m: f32,
    m2: f32,
    first_weight: f32,
    last_offset: f32,
    last_weight: f32,
}

pub struct BlurFilter {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    vertices_size: u64,
    uniform_size: u64,
}

impl BlurFilter {
    pub fn new(device: &wgpu::Device, shaders: &Shaders) -> Self {
        let uniform_size = std::mem::size_of::<BlurUniform>() as u64;
        let vertices_size = std::mem::size_of::<[FilterVertex; 4]>() as u64;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(uniform_size),
                    },
                    count: None,
                },
            ],
            label: Some("Blur filter bind group layout"),
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blur filter vertices"),
            size: vertices_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blur filter uniform"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur filter pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur filter pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shaders.blur_filter,
                entry_point: Some("main_vertex"),
                buffers: &VERTEX_BUFFERS_DESCRIPTION_FILTERS,
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shaders.blur_filter,
                entry_point: Some("main_fragment"),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            vertex_buffer,
            uniform_buffer,
            vertices_size,
            uniform_size,
        }
    }

    pub fn apply(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        filter_quad: &FilterQuad,
        linear_sampler: &wgpu::Sampler,
        source: &FilterSource,
        filter: &BlurFilterArgs,
    ) -> Option<wgpu::Texture> {
        let mut flip = create_filter_texture(device, source.size.0, source.size.1);
        let mut flop = create_filter_texture(device, source.size.0, source.size.1);

        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&[source.vertices()]),
        );

        let source_view = source.texture.create_view(&Default::default());
        let mut first = true;

        for _ in 0..(filter.num_passes() as usize) {
            for i in 0..2 {
                let horizontal = i % 2 == 0;
                let strength = if horizontal {
                    filter.blur_x.to_f32()
                } else {
                    filter.blur_y.to_f32()
                };
                let full_size = strength.min(255.0);
                if full_size <= 1.0 {
                    continue;
                }

                let (previous_view, previous_vertices, previous_width, previous_height) = if first {
                    first = false;
                    (
                        &source_view,
                        self.vertex_buffer.slice(..),
                        source.texture.width() as f32,
                        source.texture.height() as f32,
                    )
                } else {
                    (
                        &flip.create_view(&Default::default()),
                        filter_quad.filter_vertices.slice(..),
                        flip.width() as f32,
                        flip.height() as f32,
                    )
                };

                let radius = (full_size - 1.0) / 2.0;
                let m = radius.ceil() - 1.0;
                let alpha = ((radius - m) * 255.0).floor() / 255.0;
                let last_offset = 1.0 / ((1.0 / alpha) + 1.0);
                let last_weight = alpha + 1.0;

                let uniform = BlurUniform {
                    direction: if horizontal {
                        [1.0 / previous_width, 0.0]
                    } else {
                        [0.0, 1.0 / previous_height]
                    },
                    full_size,
                    m,
                    m2: m * 2.0,
                    first_weight: alpha,
                    last_offset,
                    last_weight,
                };
                queue.write_buffer(
                    &self.uniform_buffer,
                    0,
                    bytemuck::cast_slice(&[uniform]),
                );

                // We need to create the view before moving into the render pass.
                // For non-first passes, create a new view from the flip texture.
                let previous_view_owned: wgpu::TextureView;
                let actual_previous_view = if first {
                    // Already handled above, won't reach here
                    unreachable!()
                } else {
                    // Reborrow the view we already have
                    previous_view
                };
                let _ = previous_view_owned; // suppress unused warning

                let flop_view = flop.create_view(&Default::default());
                let filter_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Blur filter bind group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(actual_previous_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                    ],
                });

                {
                    let mut render_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Blur filter pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &flop_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            ..Default::default()
                        });
                    render_pass.set_pipeline(&self.pipeline);
                    render_pass.set_bind_group(0, &filter_group, &[]);
                    render_pass.set_vertex_buffer(0, previous_vertices);
                    render_pass.set_index_buffer(
                        filter_quad.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }

                std::mem::swap(&mut flip, &mut flop);
            }
        }

        if first {
            None
        } else {
            Some(flip)
        }
    }
}
