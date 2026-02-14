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

@group(2) @binding(0) var<storage, read> analytic_segments: array<AnalyticSegment>;
@group(2) @binding(1) var<uniform> analytic_params: AnalyticParams;

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

const ROOT_EPS: f32 = 1e-5;
const TANGENT_EPS: f32 = 1e-5;

fn winding_contribution(seg: AnalyticSegment, p: vec2<f32>) -> i32 {
    let y0 = seg.start.y - p.y;
    let y1 = seg.control.y - p.y;
    let y2 = seg.end.y - p.y;

    let a = y0 - 2.0 * y1 + y2;
    let b = 2.0 * (y1 - y0);
    let c = y0;

    let roots = solve_quadratic(a, b, c);
    var t0 = roots.x;
    var t1 = roots.y;
    if (t1 < t0) {
        let tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    var wind: i32 = 0;

    for (var ri: i32 = 0; ri < 2; ri = ri + 1) {
        var t = select(t0, t1, ri == 1);
        if !(t == t) || abs(t) > 1e30 {
            continue;
        }

        // Deduplicate near-equal roots (tangent/double-root cases).
        if (ri == 1 && abs(t1 - t0) <= ROOT_EPS) {
            continue;
        }

        if (t < -ROOT_EPS || t > 1.0 + ROOT_EPS) {
            continue;
        }
        t = clamp(t, 0.0, 1.0);

        // Half-open interval: include t=0, exclude t=1 to avoid endpoint double-counting.
        if (t >= 1.0 - ROOT_EPS) {
            continue;
        }

        let q = segment_point(seg, t);
        if q.x <= p.x + ROOT_EPS {
            continue;
        }

        let dy = segment_d1(seg, t).y;
        // Ignore tangential contacts with the scanline.
        if abs(dy) <= TANGENT_EPS {
            continue;
        }

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

    // Multi-start Newton solve for d/dt ||Q(t)-p||^2 = 0.
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

    // Endpoints.
    let d0 = seg.start - p;
    let d1 = seg.end - p;
    best = min(best, dot(d0, d0));
    best = min(best, dot(d1, d1));

    return best;
}

/// Compute signed distance from point to the stroke with cap handling.
/// Cap encoding in cap_join_flags: bits 0-1 = start cap, bits 2-3 = end cap.
/// 0=Butt, 1=Round, 2=Square.
fn stroke_signed_distance(pos: vec2<f32>, effective_hw: f32) -> f32 {
    let start_cap = analytic_params.cap_join_flags & 3u;
    let end_cap = (analytic_params.cap_join_flags >> 2u) & 3u;
    let n = analytic_params.num_segments;

    if n == 0u {
        return 1e30;
    }

    var best = 1e30;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        best = min(best, min_dist2_to_segment(analytic_segments[i], pos));
    }
    var dist = sqrt(best);

    // Start cap
    let first_seg = analytic_segments[0u];
    var start_tangent = segment_d1(first_seg, 0.0);
    if dot(start_tangent, start_tangent) < 1e-10 {
        start_tangent = first_seg.end - first_seg.start;
    }
    let d_along_start = -dot(pos - first_seg.start, normalize(start_tangent));

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
    let last_seg = analytic_segments[n - 1u];
    var end_tangent = segment_d1(last_seg, 1.0);
    if dot(end_tangent, end_tangent) < 1e-10 {
        end_tangent = last_seg.end - last_seg.start;
    }
    let d_along_end = dot(pos - last_seg.end, normalize(end_tangent));

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
    // Quick reject against draw bounds for both fill/stroke modes.
    if object_pos.x < analytic_params.bounds.x
        - bounds_pad
        || object_pos.y < analytic_params.bounds.y - bounds_pad
        || object_pos.x > analytic_params.bounds.z + bounds_pad
        || object_pos.y > analytic_params.bounds.w + bounds_pad {
        return 0.0;
    }

    if analytic_params.mode == 0u {
        // Fill mode.
        var winding: i32 = 0;
        for (var i: u32 = 0u; i < analytic_params.num_segments; i = i + 1u) {
            winding = winding + winding_contribution(analytic_segments[i], object_pos);
        }

        if analytic_params.fill_rule == 0u {
            return select(0.0, 1.0, (winding & 1) != 0);
        }
        return select(0.0, 1.0, winding != 0);
    }

    // Stroke mode.
    let min_half_width = 0.5 * aa_object;
    let effective_half_width = max(analytic_params.half_width, min_half_width);

    let signed_distance = stroke_signed_distance(object_pos, effective_half_width);
    return saturate(0.5 - signed_distance / max(aa_object, 1e-5));
}
