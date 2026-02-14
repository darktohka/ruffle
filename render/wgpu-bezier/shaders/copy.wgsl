/// Simple copy/blit shader for final presentation.
/// Copies a texture to the screen, optionally with sRGB conversion.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

// Full-screen triangle vertices generated from vertex ID.
@vertex
fn main_vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generates a full-screen triangle from 3 vertices.
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    let pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    // Flip Y for texture coordinates.
    return VertexOutput(pos, vec2<f32>(uv.x, 1.0 - uv.y));
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source_texture, source_sampler, in.uv);
}
