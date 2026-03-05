//! Drop shadow filter implementation.
//! Drop shadow is just Glow with an offset.

use crate::filters::blur::BlurFilter;
use crate::filters::glow::GlowFilter;
use crate::filters::FilterQuad;
use crate::filters::FilterSource;
use swf::DropShadowFilter as DropShadowFilterArgs;

pub struct DropShadowFilter;

impl DropShadowFilter {
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        filter_quad: &FilterQuad,
        nearest_sampler: &wgpu::Sampler,
        linear_sampler: &wgpu::Sampler,
        source: &FilterSource,
        filter: &DropShadowFilterArgs,
        blur_filter: &BlurFilter,
        glow_filter: &GlowFilter,
    ) -> wgpu::Texture {
        let distance = filter.distance.to_f32();
        let angle = filter.angle.to_f32();
        let x = angle.cos() * distance;
        let y = angle.sin() * distance;
        glow_filter.apply(
            device,
            queue,
            encoder,
            filter_quad,
            nearest_sampler,
            linear_sampler,
            source,
            &filter.inner_glow_filter(),
            blur_filter,
            (-x, -y),
        )
    }
}
