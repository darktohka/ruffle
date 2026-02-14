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
    scale_mode: u32,
    _pad0: u32,
    _pad1: u32,
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

fn solve_quadratic_ordered(a: f32, b: f32, c: f32) -> vec2<f32> {
    if abs(a) <= 1e-7 {
        if abs(b) <= 1e-7 {
            return vec2<f32>(1e30, 1e30);
        }
        let t = -c / b;
        // Linear: y'(t) = b. Place root in .x when b < 0 (downward),
        // in .y when b >= 0 (upward).
        if b >= 0.0 {
            return vec2<f32>(1e30, t);
        }
        return vec2<f32>(t, 1e30);
    }

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return vec2<f32>(1e30, 1e30);
    }

    // Roots ordered so that y'(.x) = -sqrt(disc) < 0 (downward crossing)
    // and y'(.y) = +sqrt(disc) > 0 (upward crossing).
    let s = sqrt(disc);
    if b >= 0.0 {
        return vec2<f32>((-b - s) / (2.0 * a), (-b + s) / (2.0 * a));
    }
    let r0 = (-b + s) / (2.0 * a);
    let r1 = (-b - s) / (2.0 * a);
    return vec2<f32>(r1, r0);
}

fn in_half_open_range(x: f32, start: f32, end: f32) -> bool {
    return x >= start && x < end;
}

fn winding_contribution(seg: AnalyticSegment, p: vec2<f32>) -> i32 {
    let y0 = seg.start.y - p.y;
    let y1 = seg.control.y - p.y;
    let y2 = seg.end.y - p.y;

    let a = y0 - 2.0 * y1 + y2;
    let b = 2.0 * (y1 - y0);
    let c = y0;

    let roots = solve_quadratic_ordered(a, b, c);
    let t0 = roots.x;
    let t1 = roots.y;
    let is_t0_valid = abs(t0) < 1e20;
    let is_t1_valid = abs(t1) < 1e20;
    if !is_t0_valid && !is_t1_valid {
        return 0;
    }

    let x0 = seg.start.x - p.x;
    let x1 = seg.control.x - p.x;
    let x2 = seg.end.x - p.x;

    if (y0 < 0.0 && y1 < 0.0 && y2 < 0.0)
        || (y0 > 0.0 && y1 > 0.0 && y2 > 0.0)
        || (x0 <= 0.0 && x1 <= 0.0 && x2 <= 0.0) {
        return 0;
    }

    var wind: i32 = 0;

    let ax = x0 - 2.0 * x1 + x2;
    let bx = 2.0 * (x1 - x0);
    let is_monotonic = abs(a) <= 1e-7;
    let t_extrema = select(-0.5 * b / a, 0.0, is_monotonic);

    if a >= 0.0 {
        let y_min = select(a * t_extrema * t_extrema + b * t_extrema + c, min(y0, y2), is_monotonic);

        // First subcurve: downward (start → extremum), include extremum.
        if is_t0_valid && in_half_open_range(0.0, y_min, y0) {
            let x = x0 + bx * t0 + ax * t0 * t0;
            if x > 0.0 {
                wind = wind - 1;
            }
        }
        // Second subcurve: upward (extremum → end), include extremum.
        if is_t1_valid && in_half_open_range(0.0, y_min, y2) {
            let x = x0 + bx * t1 + ax * t1 * t1;
            if x > 0.0 {
                wind = wind + 1;
            }
        }
    } else {
        let y_max = select(a * t_extrema * t_extrema + b * t_extrema + c, max(y0, y2), is_monotonic);

        // First subcurve: upward (start → extremum), include start.
        if is_t1_valid && in_half_open_range(0.0, y0, y_max) {
            let x = x0 + bx * t1 + ax * t1 * t1;
            if x > 0.0 {
                wind = wind + 1;
            }
        }
        // Second subcurve: downward (extremum → end), include end.
        if is_t0_valid && in_half_open_range(0.0, y2, y_max) {
            let x = x0 + bx * t0 + ax * t0 * t0;
            if x > 0.0 {
                wind = wind - 1;
            }
        }
    }

    return wind;
}

fn to_world(p: vec2<f32>) -> vec2<f32> {
    let a = transforms.world_matrix[0][0];
    let b = transforms.world_matrix[0][1];
    let c = transforms.world_matrix[1][0];
    let d = transforms.world_matrix[1][1];
    return vec2<f32>(a * p.x + c * p.y, b * p.x + d * p.y);
}

fn segment_to_world(seg: AnalyticSegment) -> AnalyticSegment {
    var ws: AnalyticSegment;
    ws.start = to_world(seg.start);
    ws.control = to_world(seg.control);
    ws.end = to_world(seg.end);
    ws._pad0 = vec2<f32>(0.0, 0.0);
    return ws;
}

fn compute_mode_scale() -> f32 {
    let a = transforms.world_matrix[0][0];
    let b = transforms.world_matrix[0][1];
    let c = transforms.world_matrix[1][0];
    let d = transforms.world_matrix[1][1];
    let line_scale_x = abs(a + c);
    let line_scale_y = abs(b + d);
    switch analytic_params.scale_mode {
        case 0u: { return 1.0; }
        case 1u: { return line_scale_x; }
        case 2u: { return line_scale_y; }
        default: { return sqrt((line_scale_x * line_scale_x + line_scale_y * line_scale_y) / 2.0); }
    }
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
    let d0 = seg.start - p;
    let d1 = seg.end - p;
    best = min(best, dot(d0, d0));
    best = min(best, dot(d1, d1));
    return best;
}

/// Compute signed distance from point to the stroke with cap handling.
/// Cap encoding in cap_join_flags: bits 0-1 = start cap, bits 2-3 = end cap.
/// 0=Butt, 1=Round, 2=Square.
fn stroke_signed_distance(world_pos: vec2<f32>, effective_hw: f32) -> f32 {
    let start_cap = analytic_params.cap_join_flags & 3u;
    let end_cap = (analytic_params.cap_join_flags >> 2u) & 3u;
    let n = analytic_params.num_segments;

    if n == 0u {
        return 1e30;
    }

    var best = 1e30;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let ws = segment_to_world(analytic_segments[i]);
        best = min(best, min_dist2_to_segment(ws, world_pos));
    }
    var dist = sqrt(best);

    // Start cap
    let first_seg = segment_to_world(analytic_segments[0u]);
    var start_tangent = segment_d1(first_seg, 0.0);
    if dot(start_tangent, start_tangent) < 1e-10 {
        start_tangent = first_seg.end - first_seg.start;
    }
    let d_along_start = -dot(world_pos - first_seg.start, normalize(start_tangent));

    if start_cap == 0u {
        if d_along_start > 0.0 {
            dist = max(dist, d_along_start);
        }
    } else if start_cap == 2u {
        let beyond = d_along_start - effective_hw;
        if beyond > 0.0 {
            dist = max(dist, beyond);
        }
    }

    // End cap
    let last_seg = segment_to_world(analytic_segments[n - 1u]);
    var end_tangent = segment_d1(last_seg, 1.0);
    if dot(end_tangent, end_tangent) < 1e-10 {
        end_tangent = last_seg.end - last_seg.start;
    }
    let d_along_end = dot(world_pos - last_seg.end, normalize(end_tangent));

    if end_cap == 0u {
        if d_along_end > 0.0 {
            dist = max(dist, d_along_end);
        }
    } else if end_cap == 2u {
        let beyond = d_along_end - effective_hw;
        if beyond > 0.0 {
            dist = max(dist, beyond);
        }
    }

    return dist - effective_hw;
}

fn analytic_coverage(object_pos: vec2<f32>, aa_object: f32) -> f32 {
    let bounds_pad = select(0.0, aa_object, analytic_params.mode == 1u);
    if object_pos.x < analytic_params.bounds.x
        - bounds_pad
        || object_pos.y < analytic_params.bounds.y - bounds_pad
        || object_pos.x > analytic_params.bounds.z + bounds_pad
        || object_pos.y > analytic_params.bounds.w + bounds_pad {
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

    let world_pos = to_world(object_pos);
    let mode_scale = compute_mode_scale();
    let desired_hw = analytic_params.half_width * mode_scale;
    let effective_hw = max(desired_hw, 0.5);

    let signed_distance = stroke_signed_distance(world_pos, effective_hw);
    let aa_world = max(length(dpdx(world_pos)), length(dpdy(world_pos)));
    return saturate(0.5 - signed_distance / max(aa_world, 1e-5));
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
    let aa_object = max(length(dpdx(in.object_pos)), length(dpdy(in.object_pos)));
    let cov = analytic_coverage(in.object_pos, aa_object);
    if cov <= 0.0 {
        discard;
    }

    var t: f32 = find_t(in.grad_uv);
    if (gradient.repeat == 1) {
        t = saturate(t);
    } else if (gradient.repeat == 2) {
        var t_abs = abs(t);
        // Guard integer conversion range for very large values.
        t_abs = min(t_abs, 2147483000.0);
        let whole = floor(t_abs);
        let frac = fract(t_abs);
        if ((i32(whole) & 1) == 0) {
            t = frac;
        } else {
            t = 1.0 - frac;
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
