//! Shader compilation for the Bézier renderer.
//!
//! All shaders have `common.wgsl` prepended before compilation.

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
}

impl Shaders {
    pub fn new(device: &wgpu::Device) -> Self {
        let common = include_str!("../shaders/common.wgsl");

        Self {
            bezier_fill: make_shader(device, "bezier_fill.wgsl", common, include_str!("../shaders/bezier_fill.wgsl")),
            bezier_gradient: make_shader(device, "bezier_gradient.wgsl", common, include_str!("../shaders/bezier_gradient.wgsl")),
            bezier_bitmap: make_shader(device, "bezier_bitmap.wgsl", common, include_str!("../shaders/bezier_bitmap.wgsl")),
            render_bitmap: make_shader(device, "render_bitmap.wgsl", common, include_str!("../shaders/render_bitmap.wgsl")),
            copy: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("copy.wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/copy.wgsl").into()),
            }),
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
