/// Quadratic Bézier fill shader using the Loop-Blinn technique.
///
/// This shader renders quadratic Bézier curves directly on the GPU without
/// tessellation. Each quadratic Bézier segment is rendered as a single triangle
/// whose three vertices correspond to the curve's three control points.
///
/// ## Loop-Blinn Technique
///
/// Each vertex carries a 2D texture coordinate (u, v) that parameterizes the
/// curve. The three control points of a quadratic Bézier receive:
///   P0 (start):   (u, v) = (0, 0)
///   P1 (control): (u, v) = (0.5, 0)
///   P2 (end):     (u, v) = (1, 1)
///
/// In the fragment shader, the implicit curve equation is:
///   f(u, v) = u² - v
///
/// - If f < 0, the fragment is **inside** the curve (filled).
/// - If f ≥ 0, the fragment is **outside** the curve (discarded).
///
/// The `fill_type` flag distinguishes:
///   0 = interior triangle (always filled, no curve test)
///   1 = curve triangle (apply Loop-Blinn test)
///
/// ## Vertex Format
///
/// Each vertex has:
///   - position: vec2<f32>  — object-space position
///   - uv: vec2<f32>        — Loop-Blinn parameter (u, v)
///   - fill_type: i32       — 0 = interior, 1 = curve edge
///   - color: vec4<f32>     — per-vertex color (premultiplied alpha)

// NOTE: The `common.wgsl` source is prepended to this before compilation.

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) fill_type: i32,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) fill_type: i32,
    @location(2) color: vec4<f32>,
};

@group(1) @binding(0) var<uniform> transforms: common__Transforms;

@vertex
fn main_vertex(in: VertexInput) -> VertexOutput {
    let pos = common__globals.view_matrix * transforms.world_matrix * vec4<f32>(in.position, 0.0, 1.0);
    // Apply SWF color transform to per-vertex color.
    let color = saturate(in.color * transforms.mult_color + transforms.add_color);
    // Premultiply alpha.
    let out_color = vec4<f32>(color.rgb * color.a, color.a);
    return VertexOutput(pos, in.uv, in.fill_type, out_color);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.fill_type == 1) {
        // Quadratic Bézier curve test: f(u,v) = u² - v
        // Pixels with f >= 0 are outside the curve.
        let f = in.uv.x * in.uv.x - in.uv.y;
        if (f >= 0.0) {
            discard;
        }
    }
    // fill_type == 0: interior triangle, always filled.
    return in.color;
}
