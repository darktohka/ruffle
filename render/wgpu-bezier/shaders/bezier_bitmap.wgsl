/// Quadratic Bézier bitmap fill shader.
///
/// Combines the Loop-Blinn curve evaluation with bitmap texture sampling.
/// Interior triangles are always filled; curve-edge triangles use the
/// u² - v test to determine inside/outside.
///
/// Bitmap UVs are computed from the object-space position via a
/// texture transform matrix.

// NOTE: The `common.wgsl` source is prepended to this before compilation.

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) fill_type: i32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) fill_type: i32,
    @location(2) tex_uv: vec2<f32>,
};

@group(1) @binding(0) var<uniform> transforms: common__Transforms;
@group(2) @binding(0) var<uniform> textureTransforms: common__TextureTransforms;
@group(2) @binding(1) var bitmap_texture: texture_2d<f32>;
@group(2) @binding(2) var bitmap_sampler: sampler;

@vertex
fn main_vertex(in: VertexInput) -> VertexOutput {
    let pos = common__globals.view_matrix * transforms.world_matrix * vec4<f32>(in.position, 0.0, 1.0);
    // Compute bitmap UVs from object-space position.
    let matrix_ = textureTransforms.texture_matrix;
    let tex_uv = (mat3x3<f32>(matrix_[0].xyz, matrix_[1].xyz, matrix_[2].xyz) * vec3<f32>(in.position, 1.0)).xy;
    return VertexOutput(pos, in.uv, in.fill_type, tex_uv);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Loop-Blinn curve test for edge triangles.
    if (in.fill_type == 1) {
        let f = in.uv.x * in.uv.x - in.uv.y;
        if (f >= 0.0) {
            discard;
        }
    }

    var color: vec4<f32> = textureSample(bitmap_texture, bitmap_sampler, in.tex_uv);
    // Texture is premultiplied by alpha.
    // Unmultiply alpha, apply color transform, remultiply alpha.
    if (color.a > 0.0) {
        color = vec4<f32>(color.rgb / color.a, color.a);
        color = saturate(color * transforms.mult_color + transforms.add_color);
        color = vec4<f32>(color.rgb * color.a, color.a);
    }
    return color;
}
