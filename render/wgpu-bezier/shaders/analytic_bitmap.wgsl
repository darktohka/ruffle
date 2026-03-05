struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) _uv: vec2<f32>,
    @location(2) _fill_type: i32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) object_pos: vec2<f32>,
    @location(1) tex_uv: vec2<f32>,
};

@group(1) @binding(0) var<uniform> transforms: common__Transforms;
@group(2) @binding(0) var<uniform> textureTransforms: common__TextureTransforms;
@group(2) @binding(1) var bitmap_texture: texture_2d<f32>;
@group(2) @binding(2) var bitmap_sampler: sampler;

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
    join_style: u32,
    miter_limit: f32,
};

@group(3) @binding(0) var<storage, read> analytic_segments: array<AnalyticSegment>;
@group(3) @binding(1) var<uniform> analytic_params: AnalyticParams;

@vertex
fn main_vertex(in: VertexInput) -> VertexOutput {
    let pos = common__globals.view_matrix * transforms.world_matrix * vec4<f32>(in.position, 0.0, 1.0);
    let matrix_ = textureTransforms.texture_matrix;
    let tex_uv = (mat3x3<f32>(matrix_[0].xyz, matrix_[1].xyz, matrix_[2].xyz) * vec3<f32>(in.position, 1.0)).xy;
    return VertexOutput(pos, in.position, tex_uv);
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

/// Compute signed distance from point to the stroke with cap and join handling.
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

    // ---- Join handling ----
    if n > 1u {
        for (var i: u32 = 0u; i < n - 1u; i = i + 1u) {
            let seg_a = segment_to_world(analytic_segments[i]);
            let seg_b = segment_to_world(analytic_segments[i + 1u]);
            let gap = seg_b.start - seg_a.end;
            if dot(gap, gap) < 1e-4 {
                let join_pt = seg_a.end;
                var tan_a = segment_d1(seg_a, 1.0);
                if dot(tan_a, tan_a) < 1e-10 { tan_a = seg_a.end - seg_a.start; }
                var tan_b = segment_d1(seg_b, 0.0);
                if dot(tan_b, tan_b) < 1e-10 { tan_b = seg_b.end - seg_b.start; }
                let len_a = length(tan_a);
                let len_b = length(tan_b);
                if len_a < 1e-6 || len_b < 1e-6 { continue; }
                let dir_a = tan_a / len_a;
                let dir_b = tan_b / len_b;
                let cross_val = dir_a.x * dir_b.y - dir_a.y * dir_b.x;
                if abs(cross_val) < 1e-6 { continue; }
                dist = apply_join(world_pos, join_pt, dir_a, dir_b, cross_val, effective_hw, dist);
            }
        }
    }

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

fn apply_join(
    p: vec2<f32>,
    join_pt: vec2<f32>,
    dir_a: vec2<f32>,
    dir_b: vec2<f32>,
    cross_val: f32,
    effective_hw: f32,
    current_dist: f32,
) -> f32 {
    var dist = current_dist;
    let join_type = analytic_params.join_style;
    let n_a = vec2<f32>(-dir_a.y, dir_a.x) * effective_hw;
    let n_b = vec2<f32>(-dir_b.y, dir_b.x) * effective_hw;
    var omx: f32; var omy: f32; var mx: f32; var my: f32;
    if cross_val > 0.0 {
        omx = n_a.x; omy = n_a.y; mx = n_b.x; my = n_b.y;
    } else {
        omx = -n_a.x; omy = -n_a.y; mx = -n_b.x; my = -n_b.y;
    }

    if join_type == 0u { return dist; } // Round

    if join_type == 1u {
        // Bevel
        let p0_outer = join_pt + vec2<f32>(omx, omy);
        let p1_outer = join_pt + vec2<f32>(mx, my);
        let bevel_mid = (p0_outer + p1_outer) * 0.5;
        let bevel_dir = p1_outer - p0_outer;
        let bevel_normal = normalize(vec2<f32>(-bevel_dir.y, bevel_dir.x));
        let to_mid = bevel_mid - join_pt;
        let bevel_n = select(-bevel_normal, bevel_normal, dot(bevel_normal, to_mid) > 0.0);
        let d_bevel = dot(p - bevel_mid, bevel_n);
        if d_bevel > 0.0 { dist = max(dist, d_bevel); }
        return dist;
    }

    // Miter
    let x10 = dir_a.x; let y10 = dir_a.y;
    let x10p = -dir_b.x; let y10p = -dir_b.y;
    let den = x10 * y10p - x10p * y10;
    if abs(den) < 1e-10 {
        let p0_outer = join_pt + vec2<f32>(omx, omy);
        let p1_outer = join_pt + vec2<f32>(mx, my);
        let bevel_mid = (p0_outer + p1_outer) * 0.5;
        let bevel_dir = p1_outer - p0_outer;
        let bevel_normal = normalize(vec2<f32>(-bevel_dir.y, bevel_dir.x));
        let to_mid = bevel_mid - join_pt;
        let bevel_n = select(-bevel_normal, bevel_normal, dot(bevel_normal, to_mid) > 0.0);
        let d_bevel = dot(p - bevel_mid, bevel_n);
        if d_bevel > 0.0 { dist = max(dist, d_bevel); }
        return dist;
    }
    let p_a = join_pt + vec2<f32>(omx, omy);
    let p_b = join_pt + vec2<f32>(mx, my);
    let t_val = (x10p * (p_a.y - p_b.y) - y10p * (p_a.x - p_b.x)) / den;
    let miter_pt = p_a + t_val * dir_a;
    let miter_vec = miter_pt - join_pt;
    let miter_len = length(miter_vec);
    let miter_lim = analytic_params.miter_limit;
    if miter_len <= miter_lim && miter_len > 1e-6 {
        let d_to_miter = length(p - miter_pt);
        dist = min(dist, d_to_miter);
    } else {
        let p0_outer = join_pt + vec2<f32>(omx, omy);
        let p1_outer = join_pt + vec2<f32>(mx, my);
        let bevel_mid = (p0_outer + p1_outer) * 0.5;
        let bevel_dir = p1_outer - p0_outer;
        let bevel_normal = normalize(vec2<f32>(-bevel_dir.y, bevel_dir.x));
        let to_mid = bevel_mid - join_pt;
        let bevel_n = select(-bevel_normal, bevel_normal, dot(bevel_normal, to_mid) > 0.0);
        let d_bevel = dot(p - bevel_mid, bevel_n);
        if d_bevel > 0.0 { dist = max(dist, d_bevel); }
    }
    return dist;
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

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let aa_object = max(length(dpdx(in.object_pos)), length(dpdy(in.object_pos)));
    let cov = analytic_coverage(in.object_pos, aa_object);
    if cov <= 0.0 {
        discard;
    }

    var color: vec4<f32> = textureSample(bitmap_texture, bitmap_sampler, in.tex_uv);
    if (color.a > 0.0) {
        color = vec4<f32>(color.rgb / color.a, color.a);
        color = saturate(color * transforms.mult_color + transforms.add_color);
        color = vec4<f32>(color.rgb * color.a, color.a);
    }
    return vec4<f32>(color.rgb * cov, color.a * cov);
}
