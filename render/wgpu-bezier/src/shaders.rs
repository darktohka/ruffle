//! Shader compilation for the Bézier renderer.
//!
//! All shaders have `common.wgsl` prepended before compilation.

use crate::blend::ComplexBlend;
use enum_map::{EnumMap, enum_map};

/// Compiled shader modules for the Bézier renderer.
#[derive(Debug)]
pub struct Shaders {
    /// Loop-Blinn quadratic Bézier fill with per-vertex color.
    pub bezier_fill: wgpu::ShaderModule,
    /// Loop-Blinn fill with gradient texture lookup.
    pub bezier_gradient: wgpu::ShaderModule,
    /// Loop-Blinn fill with bitmap texture lookup.
    pub bezier_bitmap: wgpu::ShaderModule,
    /// Bitmap rendering for Command::RenderBitmap (simpler bind group).
    pub render_bitmap: wgpu::ShaderModule,
    /// Simple copy/blit shader for final presentation.
    pub copy: wgpu::ShaderModule,
    /// Complex blend mode shaders (one per ComplexBlend variant).
    pub blend_shaders: EnumMap<ComplexBlend, wgpu::ShaderModule>,
}

impl Shaders {
    pub fn new(device: &wgpu::Device) -> Self {
        let common = include_str!("../shaders/common.wgsl");

        let blend_shaders = enum_map! {
            ComplexBlend::Multiply => make_blend_shader(device, "blend/multiply.wgsl", include_str!("../shaders/blend/multiply.wgsl")),
            ComplexBlend::Lighten => make_blend_shader(device, "blend/lighten.wgsl", include_str!("../shaders/blend/lighten.wgsl")),
            ComplexBlend::Darken => make_blend_shader(device, "blend/darken.wgsl", include_str!("../shaders/blend/darken.wgsl")),
            ComplexBlend::Difference => make_blend_shader(device, "blend/difference.wgsl", include_str!("../shaders/blend/difference.wgsl")),
            ComplexBlend::Invert => make_blend_shader(device, "blend/invert.wgsl", include_str!("../shaders/blend/invert.wgsl")),
            ComplexBlend::Alpha => make_blend_shader(device, "blend/alpha.wgsl", include_str!("../shaders/blend/alpha.wgsl")),
            ComplexBlend::Erase => make_blend_shader(device, "blend/erase.wgsl", include_str!("../shaders/blend/erase.wgsl")),
            ComplexBlend::Overlay => make_blend_shader(device, "blend/overlay.wgsl", include_str!("../shaders/blend/overlay.wgsl")),
            ComplexBlend::HardLight => make_blend_shader(device, "blend/hardlight.wgsl", include_str!("../shaders/blend/hardlight.wgsl")),
        };

        Self {
            bezier_fill: make_shader(device, "bezier_fill.wgsl", common, include_str!("../shaders/bezier_fill.wgsl")),
            bezier_gradient: make_shader(device, "bezier_gradient.wgsl", common, include_str!("../shaders/bezier_gradient.wgsl")),
            bezier_bitmap: make_shader(device, "bezier_bitmap.wgsl", common, include_str!("../shaders/bezier_bitmap.wgsl")),
            render_bitmap: make_shader(device, "render_bitmap.wgsl", common, include_str!("../shaders/render_bitmap.wgsl")),
            copy: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("copy.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/copy.wgsl").into()),
            }),
            blend_shaders,
        }
    }
}

fn make_shader(
    device: &wgpu::Device,
    name: &str,
    common: &str,
    source: &str,
) -> wgpu::ShaderModule {
    let full_source = [common, source].concat();
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(full_source.into()),
    })
}

/// Blend shaders are standalone (they don't use common.wgsl since they
/// use a full-screen triangle generated from vertex_index, not BezierVertex).
fn make_blend_shader(
    device: &wgpu::Device,
    name: &str,
    source: &str,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
