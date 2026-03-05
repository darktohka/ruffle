//! GPU-accelerated SWF filter implementations for the Bézier renderer.
//!
//! Filters are image-space effects (blur, glow, drop shadow, bevel, color matrix,
//! displacement map) applied to rendered display objects. Each filter reads from
//! a source texture and writes to a new target texture.
//!
//! This module is adapted from the original WGPU renderer's filter system,
//! simplified to work with the Bézier renderer's direct device/queue/encoder
//! pattern (no Descriptors, TexturePool, StagingBelt, or CommandTarget).

mod bevel;
mod blur;
mod color_matrix;
mod displacement_map;
mod drop_shadow;
mod glow;

use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use crate::Texture;
use crate::filters::bevel::BevelFilter;
use crate::filters::blur::BlurFilter;
use crate::filters::color_matrix::ColorMatrixFilter;
use crate::filters::displacement_map::DisplacementMapFilter;
use crate::filters::drop_shadow::DropShadowFilter;
use crate::filters::glow::GlowFilter;
use crate::shaders::Shaders;
use bytemuck::{Pod, Zeroable};
use ruffle_render::bitmap::BitmapHandle;
use ruffle_render::filters::Filter;
use std::any::Any;
use wgpu::vertex_attr_array;

/// Downcast a `BitmapHandle` to our concrete `Texture` type.
pub fn as_texture(handle: &BitmapHandle) -> &Texture {
    <dyn Any>::downcast_ref(&*handle.0).unwrap()
}

/// Describes a region of a source texture to use as filter input.
#[derive(Debug)]
pub struct FilterSource<'a> {
    pub texture: &'a wgpu::Texture,
    pub point: (u32, u32),
    pub size: (u32, u32),
}

impl<'a> FilterSource<'a> {
    pub fn for_entire_texture(texture: &'a wgpu::Texture) -> Self {
        Self {
            texture,
            point: (0, 0),
            size: (texture.width(), texture.height()),
        }
    }

    pub fn vertices(&self) -> [FilterVertex; 4] {
        let source_width = self.texture.width() as f32;
        let source_height = self.texture.height() as f32;
        let left = self.point.0;
        let top = self.point.1;
        let right = left + self.size.0;
        let bottom = top + self.size.1;
        [
            FilterVertex {
                position: [0.0, 0.0],
                uv: [left as f32 / source_width, top as f32 / source_height],
            },
            FilterVertex {
                position: [1.0, 0.0],
                uv: [right as f32 / source_width, top as f32 / source_height],
            },
            FilterVertex {
                position: [1.0, 1.0],
                uv: [right as f32 / source_width, bottom as f32 / source_height],
            },
            FilterVertex {
                position: [0.0, 1.0],
                uv: [left as f32 / source_width, bottom as f32 / source_height],
            },
        ]
    }

    pub fn vertices_with_blur_offset(&self, blur_offset: (f32, f32)) -> [FilterVertexWithBlur; 4] {
        let source_width = self.texture.width() as f32;
        let source_height = self.texture.height() as f32;
        let source_left = self.point.0;
        let source_top = self.point.1;
        let source_right = source_left + self.size.0;
        let source_bottom = source_top + self.size.1;
        [
            FilterVertexWithBlur {
                position: [0.0, 0.0],
                source_uv: [
                    source_left as f32 / source_width,
                    source_top as f32 / source_height,
                ],
                blur_uv: [
                    (source_left as f32 + blur_offset.0) / source_width,
                    (source_top as f32 + blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithBlur {
                position: [1.0, 0.0],
                source_uv: [
                    source_right as f32 / source_width,
                    source_top as f32 / source_height,
                ],
                blur_uv: [
                    (source_right as f32 + blur_offset.0) / source_width,
                    (source_top as f32 + blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithBlur {
                position: [1.0, 1.0],
                source_uv: [
                    source_right as f32 / source_width,
                    source_bottom as f32 / source_height,
                ],
                blur_uv: [
                    (source_right as f32 + blur_offset.0) / source_width,
                    (source_bottom as f32 + blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithBlur {
                position: [0.0, 1.0],
                source_uv: [
                    source_left as f32 / source_width,
                    source_bottom as f32 / source_height,
                ],
                blur_uv: [
                    (source_left as f32 + blur_offset.0) / source_width,
                    (source_bottom as f32 + blur_offset.1) / source_height,
                ],
            },
        ]
    }

    pub fn vertices_with_highlight_and_shadow(
        &self,
        blur_offset: (f32, f32),
    ) -> [FilterVertexWithDoubleBlur; 4] {
        let source_width = self.texture.width() as f32;
        let source_height = self.texture.height() as f32;
        let source_left = self.point.0 as f32;
        let source_top = self.point.1 as f32;
        let source_right = (self.point.0 + self.size.0) as f32;
        let source_bottom = (self.point.1 + self.size.1) as f32;
        [
            FilterVertexWithDoubleBlur {
                position: [0.0, 0.0],
                source_uv: [source_left / source_width, source_top / source_height],
                blur_uv_left: [
                    (source_left + blur_offset.0) / source_width,
                    (source_top + blur_offset.1) / source_height,
                ],
                blur_uv_right: [
                    (source_left - blur_offset.0) / source_width,
                    (source_top - blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithDoubleBlur {
                position: [1.0, 0.0],
                source_uv: [source_right / source_width, source_top / source_height],
                blur_uv_left: [
                    (source_right + blur_offset.0) / source_width,
                    (source_top + blur_offset.1) / source_height,
                ],
                blur_uv_right: [
                    (source_right - blur_offset.0) / source_width,
                    (source_top - blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithDoubleBlur {
                position: [1.0, 1.0],
                source_uv: [source_right / source_width, source_bottom / source_height],
                blur_uv_left: [
                    (source_right + blur_offset.0) / source_width,
                    (source_bottom + blur_offset.1) / source_height,
                ],
                blur_uv_right: [
                    (source_right - blur_offset.0) / source_width,
                    (source_bottom - blur_offset.1) / source_height,
                ],
            },
            FilterVertexWithDoubleBlur {
                position: [0.0, 1.0],
                source_uv: [source_left / source_width, source_bottom / source_height],
                blur_uv_left: [
                    (source_left + blur_offset.0) / source_width,
                    (source_bottom + blur_offset.1) / source_height,
                ],
                blur_uv_right: [
                    (source_left - blur_offset.0) / source_width,
                    (source_bottom - blur_offset.1) / source_height,
                ],
            },
        ]
    }
}

/// Quad index buffer for filter rendering: two triangles covering the quad.
const FILTER_QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// Cached GPU resources shared by all filters.
pub struct FilterQuad {
    pub index_buffer: wgpu::Buffer,
    /// Full-screen filter vertices: position [0,0]-[1,1] with matching UVs.
    pub filter_vertices: wgpu::Buffer,
}

impl FilterQuad {
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::util::DeviceExt;
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Filter quad indices"),
            contents: bytemuck::cast_slice(&FILTER_QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let vertices = [
            FilterVertex {
                position: [0.0, 0.0],
                uv: [0.0, 0.0],
            },
            FilterVertex {
                position: [1.0, 0.0],
                uv: [1.0, 0.0],
            },
            FilterVertex {
                position: [1.0, 1.0],
                uv: [1.0, 1.0],
            },
            FilterVertex {
                position: [0.0, 1.0],
                uv: [0.0, 1.0],
            },
        ];
        let filter_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Filter quad vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            index_buffer,
            filter_vertices,
        }
    }
}

/// Container for all filter implementations.
pub struct Filters {
    pub blur: BlurFilter,
    pub color_matrix: ColorMatrixFilter,
    pub glow: GlowFilter,
    pub bevel: BevelFilter,
    pub displacement_map: DisplacementMapFilter,
    pub quad: FilterQuad,
    pub linear_sampler: wgpu::Sampler,
    pub nearest_sampler: wgpu::Sampler,
    pub repeat_linear_sampler: wgpu::Sampler,
}

impl Filters {
    pub fn new(device: &wgpu::Device, shaders: &Shaders) -> Self {
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Filter linear sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Filter nearest sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let repeat_linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Filter repeat linear sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            blur: BlurFilter::new(device, shaders),
            color_matrix: ColorMatrixFilter::new(device, shaders),
            glow: GlowFilter::new(device, shaders),
            bevel: BevelFilter::new(device, shaders),
            displacement_map: DisplacementMapFilter::new(device, shaders),
            quad: FilterQuad::new(device),
            linear_sampler,
            nearest_sampler,
            repeat_linear_sampler,
        }
    }

    /// Apply a filter to a source texture and return the resulting texture.
    /// The returned texture is freshly created (not pooled).
    pub fn apply(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        source: FilterSource,
        filter: Filter,
    ) -> wgpu::Texture {
        let result = match filter {
            Filter::ColorMatrixFilter(filter) => Some(self.color_matrix.apply(
                device,
                queue,
                encoder,
                &self.quad,
                &self.nearest_sampler,
                &source,
                &filter,
            )),
            Filter::BlurFilter(filter) => self.blur.apply(
                device,
                queue,
                encoder,
                &self.quad,
                &self.linear_sampler,
                &source,
                &filter,
            ),
            Filter::GlowFilter(filter) => Some(self.glow.apply(
                device,
                queue,
                encoder,
                &self.quad,
                &self.nearest_sampler,
                &self.linear_sampler,
                &source,
                &filter,
                &self.blur,
                (0.0, 0.0),
            )),
            Filter::DropShadowFilter(filter) => Some(DropShadowFilter::apply(
                device,
                queue,
                encoder,
                &self.quad,
                &self.nearest_sampler,
                &self.linear_sampler,
                &source,
                &filter,
                &self.blur,
                &self.glow,
            )),
            Filter::BevelFilter(filter) => Some(self.bevel.apply(
                device,
                queue,
                encoder,
                &self.quad,
                &self.nearest_sampler,
                &self.linear_sampler,
                &source,
                &filter,
                &self.blur,
            )),
            Filter::DisplacementMapFilter(filter) => self.displacement_map.apply(
                device,
                queue,
                encoder,
                &self.quad,
                &self.repeat_linear_sampler,
                &self.nearest_sampler,
                &source,
                &filter,
            ),
            filter => {
                static WARNED_FILTERS: LazyLock<Mutex<HashSet<&'static str>>> =
                    LazyLock::new(Default::default);

                let name = match filter {
                    Filter::GradientGlowFilter(_) => "GradientGlowFilter",
                    Filter::GradientBevelFilter(_) => "GradientBevelFilter",
                    Filter::ConvolutionFilter(_) => "ConvolutionFilter",
                    Filter::ShaderFilter(_) => "ShaderFilter",
                    Filter::ColorMatrixFilter(_)
                    | Filter::BlurFilter(_)
                    | Filter::GlowFilter(_)
                    | Filter::DropShadowFilter(_)
                    | Filter::BevelFilter(_)
                    | Filter::DisplacementMapFilter(_) => unreachable!(),
                };
                if WARNED_FILTERS.lock().unwrap().insert(name) {
                    tracing::warn!("Unsupported filter {filter:?}");
                }
                None
            }
        };

        result.unwrap_or_else(|| {
            // Fallback: apply an identity color matrix (essentially a blit).
            self.color_matrix.apply(
                device,
                queue,
                encoder,
                &self.quad,
                &self.nearest_sampler,
                &source,
                &Default::default(),
            )
        })
    }
}

/// Create a fresh Rgba8Unorm texture for filter output.
pub fn create_filter_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Filter output"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FilterVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

pub const VERTEX_BUFFERS_DESCRIPTION_FILTERS: [wgpu::VertexBufferLayout; 1] =
    [wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<FilterVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attr_array![
            0 => Float32x2,
            1 => Float32x2,
        ],
    }];

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FilterVertexWithBlur {
    pub position: [f32; 2],
    pub source_uv: [f32; 2],
    pub blur_uv: [f32; 2],
}

pub const VERTEX_BUFFERS_DESCRIPTION_FILTERS_WITH_BLUR: [wgpu::VertexBufferLayout; 1] =
    [wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<FilterVertexWithBlur>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attr_array![
            0 => Float32x2,
            1 => Float32x2,
            2 => Float32x2,
        ],
    }];

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FilterVertexWithDoubleBlur {
    pub position: [f32; 2],
    pub source_uv: [f32; 2],
    pub blur_uv_left: [f32; 2],
    pub blur_uv_right: [f32; 2],
}

pub const VERTEX_BUFFERS_DESCRIPTION_FILTERS_WITH_DOUBLE_BLUR: [wgpu::VertexBufferLayout; 1] =
    [wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<FilterVertexWithDoubleBlur>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attr_array![
            0 => Float32x2,
            1 => Float32x2,
            2 => Float32x2,
            3 => Float32x2,
        ],
    }];
