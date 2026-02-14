struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) _uv: vec2<f32>,
    @location(2) _fill_type: i32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) object_pos: vec2<f32>,
    @location(1) grad_uv: vec2<f32>,
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

struct AnalyticSegment {
    start: vec2<f32>,
    control: vec2<f32>,
    end: vec2<f32>,
    _pad0: vec2<f32>,
};

struct AnalyticParams {
    bounds: vec4<f32>,
    num_segments: u32,
    mode: u32,
    fill_rule: u32,
    half_width: f32,
    cap_join_flags: u32,
    _pad0: vec2<u32>,
};

@group(3) @binding(0) var<storage, read> analytic_segments: array<AnalyticSegment>;
@group(3) @binding(1) var<uniform> analytic_params: AnalyticParams;

@vertex
fn main_vertex(in: VertexInput) -> VertexOutput {
    let pos = common__globals.view_matrix * transforms.world_matrix * vec4<f32>(in.position, 0.0, 1.0);
    let matrix_ = textureTransforms.texture_matrix;
    let grad_uv = (mat3x3<f32>(matrix_[0].xyz, matrix_[1].xyz, matrix_[2].xyz) * vec3<f32>(in.position, 1.0)).xy;
    return VertexOutput(pos, in.position, grad_uv);
}

fn segment_point(seg: AnalyticSegment, t: f32) -> vec2<f32> {
    let mt = 1.0 - t;
    return mt * mt * seg.start + 2.0 * mt * t * seg.control + t * t * seg.end;
}

fn segment_d1(seg: AnalyticSegment, t: f32) -> vec2<f32> {
    return 2.0 * ((1.0 - t) * (seg.control - seg.start) + t * (seg.end - seg.control));
}

fn segment_d2(seg: AnalyticSegment) -> vec2<f32> {
    return 2.0 * (seg.end - 2.0 * seg.control + seg.start);
}

fn solve_quadratic(a: f32, b: f32, c: f32) -> vec2<f32> {
    if abs(a) < 1e-7 {
        if abs(b) < 1e-7 {
            return vec2<f32>(1e30, 1e30);
        }
        return vec2<f32>(-c / b, 1e30);
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return vec2<f32>(1e30, 1e30);
    }
    let s = sqrt(disc);
    return vec2<f32>((-b - s) / (2.0 * a), (-b + s) / (2.0 * a));
}

fn winding_contribution(seg: AnalyticSegment, p: vec2<f32>) -> i32 {
    let y0 = seg.start.y - p.y;
    let y1 = seg.control.y - p.y;
    let y2 = seg.end.y - p.y;

    let a = y0 - 2.0 * y1 + y2;
    let b = 2.0 * (y1 - y0);
    let c = y0;

    let roots = solve_quadratic(a, b, c);
    var wind: i32 = 0;

    for (var ri: i32 = 0; ri < 2; ri = ri + 1) {
        let t = select(roots.x, roots.y, ri == 1);
        if t < 0.0 || t >= 1.0 || abs(t) > 1e20 {
            continue;
        }

        let q = segment_point(seg, t);
        if q.x <= p.x {
            continue;
        }

        let dy = segment_d1(seg, t).y;
        if dy > 0.0 {
            wind = wind + 1;
        } else if dy < 0.0 {
            wind = wind - 1;
        }
    }

    return wind;
}

fn min_dist2_to_segment(seg: AnalyticSegment, p: vec2<f32>) -> f32 {
    var best = 1e30;
    for (var i: i32 = 0; i <= 4; i = i + 1) {
        var t = f32(i) * 0.25;
        for (var it: i32 = 0; it < 5; it = it + 1) {
            let q = segment_point(seg, t);
            let d1 = segment_d1(seg, t);
            let d2 = segment_d2(seg);
            let r = q - p;
            let f = dot(r, d1);
            let fp = dot(d1, d1) + dot(r, d2);
            if abs(fp) > 1e-6 {
                t = clamp(t - f / fp, 0.0, 1.0);
            }
        }
        let q = segment_point(seg, t);
        let d = q - p;
        best = min(best, dot(d, d));
    }
    return best;
}

fn analytic_coverage(object_pos: vec2<f32>) -> f32 {
    if object_pos.x < analytic_params.bounds.x
        || object_pos.y < analytic_params.bounds.y
        || object_pos.x > analytic_params.bounds.z
        || object_pos.y > analytic_params.bounds.w {
        return 0.0;
    }

    if analytic_params.mode == 0u {
        var winding: i32 = 0;
        for (var i: u32 = 0u; i < analytic_params.num_segments; i = i + 1u) {
            winding = winding + winding_contribution(analytic_segments[i], object_pos);
        }
        if analytic_params.fill_rule == 0u {
            return select(0.0, 1.0, (winding & 1) != 0);
        }
        return select(0.0, 1.0, winding != 0);
    }

    var best = 1e30;
    for (var i: u32 = 0u; i < analytic_params.num_segments; i = i + 1u) {
        best = min(best, min_dist2_to_segment(analytic_segments[i], object_pos));
    }
    let w2 = analytic_params.half_width * analytic_params.half_width;
    return select(0.0, 1.0, best <= w2);
}

fn find_t(uv: vec2<f32>) -> f32 {
    if (gradient.shape == 1) {
        return uv.x;
    } else if (gradient.shape == 2) {
        return length(uv * 2.0 - 1.0);
    } else {
        let centered = uv * 2.0 - 1.0;
        var d: vec2<f32> = vec2<f32>(gradient.focal_point, 0.0) - centered;
        let l = length(d);
        d = d / max(l, 1e-6);
        return l / (sqrt(max(1.0 - gradient.focal_point * gradient.focal_point * d.y * d.y, 1e-6)) + gradient.focal_point * d.x);
    }
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let cov = analytic_coverage(in.object_pos);
    if cov <= 0.0 {
        discard;
    }

    var t: f32 = find_t(in.grad_uv);
    if (gradient.repeat == 1) {
        t = saturate(t);
    } else if (gradient.repeat == 2) {
        if (t < 0.0) { t = -t; }
        if ((i32(t) & 1) == 0) {
            t = fract(t);
        } else {
            t = 1.0 - fract(t);
        }
    } else if (gradient.repeat == 3) {
        t = fract(t);
    }

    var color = textureSample(gradient_texture, gradient_sampler, vec2<f32>(t, 0.0));
    if (gradient.interpolation != 0) {
        color = common__linear_to_srgb(color);
    }
    let out = saturate(color * transforms.mult_color + transforms.add_color);
    let alpha = saturate(out.a) * cov;
    return vec4<f32>(out.rgb * alpha, alpha);
}
