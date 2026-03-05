//! Displacement map filter implementation.

use crate::filters::{
    FilterQuad, FilterSource, FilterVertex, VERTEX_BUFFERS_DESCRIPTION_FILTERS,
    as_texture, create_filter_texture,
};
use crate::shaders::Shaders;
use bytemuck::{Pod, Zeroable};
use ruffle_render::filters::{
    DisplacementMapFilter as DisplacementMapFilterArgs, DisplacementMapFilterMode,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
struct DisplacementMapUniform {
    color: [f32; 4],
    components: u32, // 00000000 00000000 XXXXXXXX YYYYYYYY
    mode: u32,       // 0 wrap, 1 clamp, 2 ignore, 3 color
    scale_x: f32,
    scale_y: f32,
    source_width: f32,
    source_height: f32,
    map_width: f32,
    map_height: f32,
    offset_x: f32,
    offset_y: f32,
    viewscale_x: f32,
    viewscale_y: f32,
}

pub struct DisplacementMapFilter {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    vertices_size: u64,
    uniform_size: u64,
}

impl DisplacementMapFilter {
    pub fn new(device: &wgpu::Device, shaders: &Shaders) -> Self {
        let uniform_size = std::mem::size_of::<DisplacementMapUniform>() as u64;
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(uniform_size),
                    },
                    count: None,
                },
            ],
            label: Some("Displacement map filter bind group layout"),
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Displacement map filter vertices"),
            size: vertices_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Displacement map filter uniform"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Displacement map filter pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Displacement map filter pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shaders.displacement_map_filter,
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
                module: &shaders.displacement_map_filter,
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
        repeat_linear_sampler: &wgpu::Sampler,
        nearest_sampler: &wgpu::Sampler,
        source: &FilterSource,
        filter: &DisplacementMapFilterArgs,
    ) -> Option<wgpu::Texture> {
        let map_handle = filter.map_bitmap.clone()?;
        let map_texture = as_texture(&map_handle);

        let source_view = source.texture.create_view(&Default::default());
        let map_view = map_texture.texture.create_view(&Default::default());

        let target = create_filter_texture(device, source.size.0, source.size.1);
        let target_view = target.create_view(&Default::default());

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[DisplacementMapUniform {
                color: [
                    f32::from(filter.color.r) / 255.0,
                    f32::from(filter.color.g) / 255.0,
                    f32::from(filter.color.b) / 255.0,
                    f32::from(filter.color.a) / 255.0,
                ],
                components: ((filter.component_x as u32) << 8) | (filter.component_y as u32),
                mode: match filter.mode {
                    DisplacementMapFilterMode::Wrap => 0,
                    DisplacementMapFilterMode::Clamp => 1,
                    DisplacementMapFilterMode::Ignore => 2,
                    DisplacementMapFilterMode::Color => 3,
                },
                scale_x: filter.scale_x,
                scale_y: filter.scale_y,
                source_width: source.texture.width() as f32,
                source_height: source.texture.height() as f32,
                map_width: map_texture.texture.width() as f32,
                map_height: map_texture.texture.height() as f32,
                offset_x: filter.map_point.0 as f32,
                offset_y: filter.map_point.1 as f32,
                viewscale_x: filter.viewscale_x,
                viewscale_y: filter.viewscale_y,
            }]),
        );
        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&[source.vertices()]),
        );

        let filter_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Displacement map filter bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(repeat_linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Displacement map filter pass"),
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

        Some(target)
    }
}
