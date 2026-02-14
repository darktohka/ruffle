/// Complex blend mode: HardLight
/// Prepended with common.wgsl at compile time.

struct BlendVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var parent_texture: texture_2d<f32>;
@group(0) @binding(1) var current_texture: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

@vertex
fn main_vertex(@builtin(vertex_index) vertex_index: u32) -> BlendVertexOutput {
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    let pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return BlendVertexOutput(pos, vec2<f32>(uv.x, 1.0 - uv.y));
}

fn blend_func(src: vec3<f32>, dst: vec3<f32>) -> vec3<f32> {
    var out = src;
    if (src.r <= 0.5) { out.r = (2.0 * src.r * dst.r); } else { out.r = (1.0 - 2.0 * (1.0 - dst.r) * (1.0 - src.r)); }
    if (src.g <= 0.5) { out.g = (2.0 * src.g * dst.g); } else { out.g = (1.0 - 2.0 * (1.0 - dst.g) * (1.0 - src.g)); }
    if (src.b <= 0.5) { out.b = (2.0 * src.b * dst.b); } else { out.b = (1.0 - 2.0 * (1.0 - dst.b) * (1.0 - src.b)); }
    return out;
}

@fragment
fn main_fragment(in: BlendVertexOutput) -> @location(0) vec4<f32> {
    var dst: vec4<f32> = textureSample(parent_texture, texture_sampler, in.uv);
    var src: vec4<f32> = textureSample(current_texture, texture_sampler, in.uv);

    if (src.a > 0.0) {
        return vec4<f32>(src.rgb * (1.0 - dst.a) + dst.rgb * (1.0 - src.a) + src.a * dst.a * blend_func(src.rgb / src.a, dst.rgb / dst.a), src.a + dst.a * (1.0 - src.a));
    } else {
        if (true) {
            discard;
        }
        return dst;
    }
}
