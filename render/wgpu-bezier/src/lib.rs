//! # WGPU Bézier Render Backend for Ruffle
//!
//! This crate implements Ruffle's [`RenderBackend`] trait using **direct
//! quadratic Bézier curve rendering** on the GPU via the Loop-Blinn technique,
//! completely avoiding CPU-side tessellation (lyon).
//!
//! ## Architecture Overview
//!
//! ### The Loop-Blinn Technique
//!
//! Every quadratic Bézier curve can be rendered with a single triangle.
//! The three vertices of the triangle correspond to the curve's three
//! control points (P0, P1, P2). Each vertex is assigned a 2D texture
//! coordinate:
//!
//! ```text
//!   P0 (start):   (u, v) = (0, 0)
//!   P1 (control): (u, v) = (0.5, 0)
//!   P2 (end):     (u, v) = (1, 1)
//! ```
//!
//! In the fragment shader, the implicit curve equation `f(u,v) = u² - v`
//! classifies each pixel:
//! - `f < 0`: inside the curve (filled)
//! - `f ≥ 0`: outside the curve (discarded)
//!
//! This provides mathematically exact, resolution-independent rendering
//! of quadratic Bézier curves with zero tessellation artifacts.
//!
//! ### Shape Processing Pipeline
//!
//! When a shape is registered (`register_shape`):
//!
//! 1. **Path decomposition**: Each [`DrawPath`] (fill or stroke) is processed.
//!
//! 2. **Fill paths**: Converted to GPU geometry via:
//!    - **Fan triangulation**: A reference point (first vertex) fans out to
//!      create interior triangles from consecutive line segments. These
//!      triangles have `fill_type = 0` (always filled).
//!    - **Curve triangles**: Each quadratic Bézier creates one triangle from
//!      its three control points, with `fill_type = 1` and the Loop-Blinn
//!      UV coordinates. The fragment shader evaluates the curve equation.
//!
//! 3. **Stroke paths**: Expanded to screen-space quads on the CPU:
//!    - Each line segment becomes a rectangle (2 triangles) with the
//!      appropriate stroke width.
//!    - Each Bézier stroke segment is approximated by subdividing into
//!      small line segments and expanding each.
//!
//! 4. **GPU upload**: All vertices and indices are uploaded to GPU buffers.
//!    Each draw call records which pipeline to use (color, gradient, bitmap)
//!    and any associated bind groups.
//!
//! ### Rendering Pipeline
//!
//! Three shader pipelines handle different fill types:
//!
//! | Pipeline | Shader | Purpose |
//! |----------|--------|---------|
//! | Color | `bezier_fill.wgsl` | Solid color fills with per-vertex color |
//! | Gradient | `bezier_gradient.wgsl` | Linear/radial/focal gradient fills |
//! | Bitmap | `bezier_bitmap.wgsl` | Bitmap texture fills |
//!
//! All three use the same Loop-Blinn curve evaluation in their fragment
//! shaders; they differ only in how they determine the output color.
//!
//! ### Advantages Over Tessellation
//!
//! - **No tessellation artifacts**: Curves are mathematically exact at any zoom.
//! - **Simpler CPU processing**: No lyon dependency; just fan triangulation.
//! - **GPU-native**: All curve evaluation happens in the fragment shader.
//! - **Resolution independent**: No need to re-tessellate when zooming.

// Remove this when we decide on how to handle multithreaded rendering
#![allow(clippy::arc_with_non_send_sync)]
#![allow(clippy::needless_pass_by_ref_mut)]
// The backend module will be used once the crate is wired into a binary.
#![allow(dead_code)]

pub mod backend;
pub mod blend;
mod mesh;
mod pipelines;
mod shaders;
pub mod target;

use bytemuck::{Pod, Zeroable};
use ruffle_render::bitmap::BitmapHandleImpl;
use std::cell::Cell;
pub use wgpu;

pub use mesh::{BezierMesh, DrawType};
pub use target::{RenderTarget, RenderTargetFrame, SwapChainTarget, TextureTarget};

/// Mask states for stencil-based masking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskState {
    NoMask,
    DrawMaskStencil,
    DrawMaskedContent,
    ClearMaskStencil,
}

/// Per-object transform uniforms uploaded to the GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Transforms {
    pub world_matrix: [[f32; 4]; 4],
    pub mult_color: [f32; 4],
    pub add_color: [f32; 4],
}

/// Texture coordinate transform for gradient/bitmap fills.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TextureTransforms {
    pub u_matrix: [[f32; 4]; 4],
}

/// Vertex format for the Bézier fill pipeline.
///
/// Carries position, Loop-Blinn UV parameters, a fill type flag,
/// and per-vertex color.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BezierVertex {
    /// Object-space position.
    pub position: [f32; 2],
    /// Loop-Blinn texture coordinates:
    /// - For interior triangles: unused (can be zero).
    /// - For curve triangles: (u, v) as described in the Loop-Blinn technique.
    pub uv: [f32; 2],
    /// Fill type flag:
    /// - 0 = interior triangle (always filled)
    /// - 1 = curve edge triangle (apply u² - v test)
    pub fill_type: i32,
    /// Per-vertex RGBA color (in 0.0–1.0 range, NOT premultiplied).
    pub color: [f32; 4],
    /// Padding for alignment.
    pub _pad: i32,
}

/// Vertex format for gradient/bitmap Bézier fills.
/// Same as BezierVertex but without per-vertex color (color comes from texture).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BezierTexVertex {
    /// Object-space position.
    pub position: [f32; 2],
    /// Loop-Blinn texture coordinates.
    pub uv: [f32; 2],
    /// Fill type flag: 0 = interior, 1 = curve edge.
    pub fill_type: i32,
    /// Padding for alignment.
    pub _pad: i32,
}

/// Gradient uniform data uploaded per gradient draw.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GradientUniforms {
    pub focal_point: f32,
    pub interpolation: i32,
    pub shape: i32,
    pub repeat: i32,
}

/// Global view matrix uniform.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GlobalsUniform {
    pub view_matrix: [[f32; 4]; 4],
}

/// Per-draw analytic path parameters for winding-correct fills and direct
/// quadratic stroke rasterization.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct AnalyticParams {
    /// [min_x, min_y, max_x, max_y] in object-space pixels.
    pub bounds: [f32; 4],
    /// Segment count in the storage buffer.
    pub num_segments: u32,
    /// 0 = fill, 1 = stroke.
    pub mode: u32,
    /// 0 = even-odd, 1 = non-zero.
    pub fill_rule: u32,
    /// Stroke half-width in object-space pixels.
    pub half_width: f32,
    /// Packed cap/join flags for future expansion.
    pub cap_join_flags: u32,
    /// Reserved. Kept at 3 words so the struct size is 48 bytes, matching WGSL
    /// uniform layout rounding to 16-byte alignment.
    pub _pad0: [u32; 3],
}

/// A GPU texture owned by this backend.
#[derive(Debug)]
pub struct Texture {
    pub texture: wgpu::Texture,
    pub copy_count: Cell<u8>,
}

impl BitmapHandleImpl for Texture {}
