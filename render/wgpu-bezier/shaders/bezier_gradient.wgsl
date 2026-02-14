/// Quadratic Bézier gradient fill shader.
///
/// Combines the Loop-Blinn curve evaluation with gradient color lookup.
/// Interior triangles are always filled; curve-edge triangles use the
/// u² - v test to determine inside/outside.
///
/// Gradient UVs are computed from the object-space position via a
/// texture transform matrix, then used to look up the gradient color
/// from a 1D gradient LUT texture.

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
    @location(2) grad_uv: vec2<f32>,
};

@group(1) @binding(0) var<uniform> transforms: common__Transforms;
@group(2) @binding(0) var<uniform> textureTransforms: common__TextureTransforms;

struct Gradient {
    focal_point: f32,
    interpolation: i32,
    shape: i32,
    repeat: i32,
};

@group(2) @binding(1) var<uniform> gradient: Gradient;
@group(2) @binding(2) var gradient_texture: texture_2d<f32>;
@group(2) @binding(3) var gradient_sampler: sampler;

@vertex
fn main_vertex(in: VertexInput) -> VertexOutput {
    let pos = common__globals.view_matrix * transforms.world_matrix * vec4<f32>(in.position, 0.0, 1.0);
    // Compute gradient UVs from object-space position.
    let matrix_ = textureTransforms.texture_matrix;
    let grad_uv = (mat3x3<f32>(matrix_[0].xyz, matrix_[1].xyz, matrix_[2].xyz) * vec3<f32>(in.position, 1.0)).xy;
    return VertexOutput(pos, in.uv, in.fill_type, grad_uv);
}

fn find_t(uv: vec2<f32>) -> f32 {
    if (gradient.shape == 1) {
        // Linear gradient.
        return uv.x;
    } else if (gradient.shape == 2) {
        // Radial gradient.
        return length(uv * 2.0 - 1.0);
    } else {
        // Focal gradient.
        let centered = uv * 2.0 - 1.0;
        var d: vec2<f32> = vec2<f32>(gradient.focal_point, 0.0) - centered;
        let l = length(d);
        d = d / max(l, 1e-6);
        return l / (sqrt(max(1.0 - gradient.focal_point * gradient.focal_point * d.y * d.y, 1e-6)) + gradient.focal_point * d.x);
    }
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

    // Compute gradient parameter t.
    var t: f32 = find_t(in.grad_uv);

    if (gradient.repeat == 1) {
        // Pad
        t = saturate(t);
    } else if (gradient.repeat == 2) {
        // Reflect
        var t_abs = abs(t);
        // Guard integer conversion range for very large values.
        t_abs = min(t_abs, 2147483000.0);
        let whole = floor(t_abs);
        let frac_t = fract(t_abs);
        if ((i32(whole) & 1) == 0) {
            t = frac_t;
        } else {
            t = 1.0 - frac_t;
        }
    } else if (gradient.repeat == 3) {
        // Repeat
        t = fract(t);
    }

    var color = textureSample(gradient_texture, gradient_sampler, vec2<f32>(t, 0.0));
    if (gradient.interpolation != 0) {
        color = common__linear_to_srgb(color);
    }
    let out = saturate(color * transforms.mult_color + transforms.add_color);
    let alpha = saturate(out.a);
    return vec4<f32>(out.rgb * alpha, alpha);
}
