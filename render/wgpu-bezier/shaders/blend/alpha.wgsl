/// Complex blend mode: Alpha
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

@fragment
fn main_fragment(in: BlendVertexOutput) -> @location(0) vec4<f32> {
    var dst: vec4<f32> = textureSample(parent_texture, texture_sampler, in.uv);
    var src: vec4<f32> = textureSample(current_texture, texture_sampler, in.uv);

    if (src.a > 0.0) {
        return vec4<f32>(dst.rgb * src.a, src.a * dst.a);
    } else {
        if (true) {
            discard;
        }
        return dst;
    }
}
