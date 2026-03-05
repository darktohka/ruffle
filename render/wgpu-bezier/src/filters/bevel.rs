//! Bevel filter implementation.

use crate::filters::blur::BlurFilter;
use crate::filters::{
    FilterQuad, FilterSource, FilterVertexWithDoubleBlur,
    VERTEX_BUFFERS_DESCRIPTION_FILTERS_WITH_DOUBLE_BLUR, create_filter_texture,
};
use crate::shaders::Shaders;
use bytemuck::{Pod, Zeroable};
use swf::BevelFilter as BevelFilterArgs;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
struct BevelUniform {
    highlight_color: [f32; 4],
    shadow_color: [f32; 4],
    strength: f32,
    bevel_type: u32,       // 0 outer, 1 inner, 2 full
    knockout: u32,         // bool
    composite_source: u32, // bool
}

pub struct BevelFilter {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    vertices_size: u64,
    uniform_size: u64,
}

impl BevelFilter {
    pub fn new(device: &wgpu::Device, shaders: &Shaders) -> Self {
        let uniform_size = std::mem::size_of::<BevelUniform>() as u64;
        let vertices_size = std::mem::size_of::<[FilterVertexWithDoubleBlur; 4]>() as u64;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(uniform_size),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
            label: Some("Bevel filter bind group layout"),
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bevel filter vertices"),
            size: vertices_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bevel filter uniform"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bevel filter pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bevel filter pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shaders.bevel_filter,
                entry_point: Some("main_vertex"),
                buffers: &VERTEX_BUFFERS_DESCRIPTION_FILTERS_WITH_DOUBLE_BLUR,
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shaders.bevel_filter,
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

    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        filter_quad: &FilterQuad,
        nearest_sampler: &wgpu::Sampler,
        linear_sampler: &wgpu::Sampler,
        source: &FilterSource,
        filter: &BevelFilterArgs,
        blur_filter: &BlurFilter,
    ) -> wgpu::Texture {
        let blurred = blur_filter.apply(
            device,
            queue,
            encoder,
            filter_quad,
            linear_sampler,
            source,
            &filter.inner_blur_filter(),
        );
        let blurred_texture = blurred.as_ref().unwrap_or(source.texture);
        let source_view = source.texture.create_view(&Default::default());
        let blurred_view = blurred_texture.create_view(&Default::default());

        let distance = filter.distance.to_f32();
        let angle = filter.angle.to_f32();
        let blur_offset = (angle.cos() * distance, angle.sin() * distance);

        let target = create_filter_texture(device, source.size.0, source.size.1);
        let target_view = target.create_view(&Default::default());

        let mut highlight_color = [
            f32::from(filter.highlight_color.r) / 255.0,
            f32::from(filter.highlight_color.g) / 255.0,
            f32::from(filter.highlight_color.b) / 255.0,
            f32::from(filter.highlight_color.a) / 255.0,
        ];
        highlight_color[0] *= highlight_color[3];
        highlight_color[1] *= highlight_color[3];
        highlight_color[2] *= highlight_color[3];
        let mut shadow_color = [
            f32::from(filter.shadow_color.r) / 255.0,
            f32::from(filter.shadow_color.g) / 255.0,
            f32::from(filter.shadow_color.b) / 255.0,
            f32::from(filter.shadow_color.a) / 255.0,
        ];
        shadow_color[0] *= shadow_color[3];
        shadow_color[1] *= shadow_color[3];
        shadow_color[2] *= shadow_color[3];

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[BevelUniform {
                highlight_color,
                shadow_color,
                strength: filter.strength.to_f32(),
                bevel_type: if filter.is_on_top() {
                    2
                } else if filter.is_inner() {
                    1
                } else {
                    0
                },
                knockout: if filter.is_knockout() { 1 } else { 0 },
                composite_source: 1,
            }]),
        );
        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&[source.vertices_with_highlight_and_shadow(blur_offset)]),
        );

        let filter_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bevel filter bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&blurred_view),
                },
            ],
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bevel filter pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
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
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                filter_quad.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..6, 0, 0..1);
        }

        target
    }
}
