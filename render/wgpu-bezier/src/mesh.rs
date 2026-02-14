//! Mesh generation for the Bézier renderer.
//!
//! This module converts SWF [`DistilledShape`] data into GPU-ready geometry
//! using the Loop-Blinn technique for quadratic Bézier curves.
//!
//! ## Geometry Generation Strategy
//!
//! ### Fills
//!
//! Each fill path is decomposed into two kinds of triangles:
//!
//! 1. **Interior triangles** (`fill_type = 0`): Created by fan triangulation
//!    from a reference point (the first MoveTo) to each consecutive pair of
//!    on-curve points. These are always filled.
//!
//! 2. **Curve triangles** (`fill_type = 1`): One triangle per quadratic Bézier,
//!    formed by the curve's three control points. The fragment shader evaluates
//!    `u² - v` to add or subtract the curved area.
//!
//! The combination of interior fan triangles and curve correction triangles
//! produces the exact filled shape with even-odd winding.
//!
//! ### Strokes
//!
//! Strokes are expanded into screen-space quads on the CPU:
//! - Line segments → rectangles (2 triangles each).
//! - Quadratic Bézier segments → subdivided into line segments, then expanded.
//!
//! The stroke width is applied as a perpendicular offset from each edge.

use crate::{BezierTexVertex, BezierVertex, GradientUniforms, TextureTransforms};
use ruffle_render::backend::{ShapeHandle, ShapeHandleImpl};
use ruffle_render::bitmap::BitmapHandle;
use ruffle_render::shape_utils::{
    DistilledShape, DrawCommand, DrawPath, FillRule, GradientType,
};
use std::any::Any;
use std::collections::HashMap;
use swf::{FillStyle, GradientRecord, LineCapStyle, LineJoinStyle, LineStyle, Twips};
use wgpu::util::DeviceExt;

/// How big to make gradient lookup textures.
const GRADIENT_SIZE: usize = 256;

/// Number of subdivisions when flattening a Bézier stroke to line segments.
const STROKE_BEZIER_SUBDIVISIONS: usize = 16;

/// A complete mesh for one registered shape, ready to be drawn.
#[derive(Debug)]
pub struct BezierMesh {
    pub draws: Vec<Draw>,
}

impl ShapeHandleImpl for BezierMesh {}

pub fn as_bezier_mesh(handle: &ShapeHandle) -> &BezierMesh {
    <dyn Any>::downcast_ref(&*handle.0).expect("Shape handle must be a BezierMesh")
}

/// One draw call within a mesh.
#[derive(Debug)]
pub struct Draw {
    pub draw_type: DrawType,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    /// True when this draw comes from `DrawPath::Stroke`.
    /// Mask rendering should ignore stroke draws and only use fill geometry.
    pub is_stroke: bool,
    /// Fill rule for this draw when `is_stroke == false`.
    pub fill_rule: Option<FillRule>,
    /// Analytic path payload for future winding-correct GPU fill evaluation.
    ///
    /// Phase 1 only stores/uploads this data; later passes will consume it in
    /// dedicated analytic fill/stroke pipelines.
    pub analytic_fill: Option<AnalyticFillData>,
    /// Analytic stroke payload for future direct GPU stroke rendering.
    pub analytic_stroke: Option<AnalyticStrokeData>,
    /// Bind group containing analytic segment buffer and params uniform.
    pub analytic_bind_group: Option<wgpu::BindGroup>,
}

/// One quadratic segment for analytic fill/stroke evaluation on GPU.
///
/// For line segments, `control` is set to the midpoint of `start` and `end`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuQuadraticSegment {
    pub start: [f32; 2],
    pub control: [f32; 2],
    pub end: [f32; 2],
    pub _pad0: [f32; 2],
}

/// Preprocessed analytic fill data for one `DrawPath::Fill` path.
#[derive(Clone, Debug)]
pub struct AnalyticFillData {
    pub segment_buffer: wgpu::Buffer,
    pub num_segments: u32,
    /// Path bounds in object-space pixels: [min_x, min_y, max_x, max_y].
    pub bounds: [f32; 4],
}

/// Preprocessed analytic stroke data for one `DrawPath::Stroke` path.
#[derive(Clone, Debug)]
pub struct AnalyticStrokeData {
    pub segment_buffer: wgpu::Buffer,
    pub num_segments: u32,
    pub bounds: [f32; 4],
    pub half_width: f32,
    pub is_closed: bool,
    pub start_cap: LineCapStyle,
    pub end_cap: LineCapStyle,
    pub join_style: LineJoinStyle,
}

/// What kind of fill this draw uses.
#[derive(Debug)]
pub enum DrawType {
    /// Solid color fill — vertices carry per-vertex color.
    Color,
    /// Gradient fill — needs gradient bind group.
    Gradient {
        bind_group: wgpu::BindGroup,
    },
    /// Bitmap fill — needs bitmap bind group.
    Bitmap {
        bind_group: wgpu::BindGroup,
    },
}

/// Build GPU mesh data from a distilled SWF shape.
///
/// This is called once per shape during `register_shape`. It produces
/// a [`BezierMesh`] containing one [`Draw`] per fill/stroke path.
pub fn build_mesh(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shape: &DistilledShape,
    bitmap_handles: &HashMap<u16, BitmapHandle>,
    gradient_layout: &wgpu::BindGroupLayout,
    bitmap_layout: &wgpu::BindGroupLayout,
    analytic_layout: Option<&wgpu::BindGroupLayout>,
    default_sampler: &wgpu::Sampler,
) -> BezierMesh {
    let mut draws = Vec::new();

    for path in &shape.paths {
        match path {
            DrawPath::Fill {
                style,
                commands,
                winding_rule,
            } => {
                let draw = build_fill_draw(
                    device,
                    queue,
                    commands,
                    *winding_rule,
                    style,
                    shape.id,
                    bitmap_handles,
                    gradient_layout,
                    bitmap_layout,
                    analytic_layout,
                    default_sampler,
                );
                if let Some(draw) = draw {
                    draws.push(draw);
                }
            }
            DrawPath::Stroke {
                style,
                commands,
                is_closed,
            } => {
                let draw = build_stroke_draw(
                    device,
                    queue,
                    commands,
                    style,
                    *is_closed,
                    shape.id,
                    bitmap_handles,
                    gradient_layout,
                    bitmap_layout,
                    analytic_layout,
                    default_sampler,
                );
                if let Some(draw) = draw {
                    draws.push(draw);
                }
            }
        }
    }

    BezierMesh { draws }
}

/// Build geometry for a single fill path.
///
/// Uses fan triangulation for interior regions and Loop-Blinn triangles
/// for quadratic Bézier curve edges.
fn build_fill_draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    commands: &[DrawCommand],
    winding_rule: FillRule,
    style: &FillStyle,
    _shape_id: swf::CharacterId,
    bitmap_handles: &HashMap<u16, BitmapHandle>,
    gradient_layout: &wgpu::BindGroupLayout,
    bitmap_layout: &wgpu::BindGroupLayout,
    analytic_layout: Option<&wgpu::BindGroupLayout>,
    default_sampler: &wgpu::Sampler,
) -> Option<Draw> {
    if commands.is_empty() {
        return None;
    }

    let (_draw_type, use_color_vertices) = match style {
        FillStyle::Color(_) => (FillKind::Color, true),
        FillStyle::LinearGradient(_)
        | FillStyle::RadialGradient(_)
        | FillStyle::FocalGradient { .. } => (FillKind::Gradient, false),
        FillStyle::Bitmap { .. } => (FillKind::Bitmap, false),
    };

    let maybe_analytic = if let Some(analytic_layout) = analytic_layout {
        let analytic_fill = build_analytic_fill_data(device, commands)?;
        let analytic_bind_group = create_analytic_bind_group(
            device,
            analytic_layout,
            &analytic_fill.segment_buffer,
            analytic_fill.num_segments,
            analytic_fill.bounds,
            0,
            winding_rule,
            0.0,
        );
        Some((analytic_fill, analytic_bind_group))
    } else {
        None
    };

    if use_color_vertices {
        let color = match style {
            FillStyle::Color(c) => *c,
            _ => unreachable!(),
        };
        let color_f = [
            f32::from(color.r) / 255.0,
            f32::from(color.g) / 255.0,
            f32::from(color.b) / 255.0,
            f32::from(color.a) / 255.0,
        ];

        let (vertices, indices) = if let Some((analytic_fill, _)) = &maybe_analytic {
            build_analytic_quad_color(analytic_fill.bounds, color_f)
        } else {
            build_fill_geometry_color(commands, color_f)
        };

        if indices.is_empty() {
            return None;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bezier fill vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bezier fill indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Some(Draw {
            draw_type: DrawType::Color,
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            is_stroke: false,
            fill_rule: Some(winding_rule),
            analytic_fill: maybe_analytic.as_ref().map(|(fill, _)| fill.clone()),
            analytic_stroke: None,
            analytic_bind_group: maybe_analytic.map(|(_, bg)| bg),
        })
    } else {
        let (vertices, indices) = if let Some((analytic_fill, _)) = &maybe_analytic {
            build_analytic_quad_tex(analytic_fill.bounds)
        } else {
            build_fill_geometry_tex(commands)
        };

        if indices.is_empty() {
            return None;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bezier tex fill vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bezier tex fill indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let draw_type = match style {
            FillStyle::LinearGradient(gradient) => {
                let bind_group = create_gradient_bind_group(
                    device,
                    queue,
                    gradient_layout,
                    default_sampler,
                    GradientType::Linear,
                    gradient,
                    swf::Fixed8::ZERO,
                );
                DrawType::Gradient { bind_group }
            }
            FillStyle::RadialGradient(gradient) => {
                let bind_group = create_gradient_bind_group(
                    device,
                    queue,
                    gradient_layout,
                    default_sampler,
                    GradientType::Radial,
                    gradient,
                    swf::Fixed8::ZERO,
                );
                DrawType::Gradient { bind_group }
            }
            FillStyle::FocalGradient {
                gradient,
                focal_point,
            } => {
                let bind_group = create_gradient_bind_group(
                    device,
                    queue,
                    gradient_layout,
                    default_sampler,
                    GradientType::Focal,
                    gradient,
                    *focal_point,
                );
                DrawType::Gradient { bind_group }
            }
            FillStyle::Bitmap {
                id,
                matrix,
                is_smoothed,
                is_repeating,
            } => {
                if let Some(handle) = bitmap_handles.get(id) {
                    let texture: &crate::Texture =
                        <dyn Any>::downcast_ref(&*handle.0).expect("Must be a Texture");
                    let texture_view = texture.texture.create_view(&Default::default());

                    let tex_matrix = swf_bitmap_to_gl_matrix(
                        (*matrix).into(),
                        texture.texture.width(),
                        texture.texture.height(),
                    );
                    let tex_transforms_data = TextureTransforms {
                        u_matrix: matrix_3x3_to_4x4(&tex_matrix),
                    };
                    let tex_transforms_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Bitmap tex transforms"),
                            contents: bytemuck::bytes_of(&tex_transforms_data),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                    // Create sampler based on smoothing/repeating flags.
                    let address_mode = if *is_repeating {
                        wgpu::AddressMode::Repeat
                    } else {
                        wgpu::AddressMode::ClampToEdge
                    };
                    let filter_mode = if *is_smoothed {
                        wgpu::FilterMode::Linear
                    } else {
                        wgpu::FilterMode::Nearest
                    };
                    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                        label: Some("Bitmap sampler"),
                        address_mode_u: address_mode,
                        address_mode_v: address_mode,
                        mag_filter: filter_mode,
                        min_filter: filter_mode,
                        ..Default::default()
                    });

                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Bitmap fill bind group"),
                        layout: bitmap_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: tex_transforms_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&texture_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                        ],
                    });
                    DrawType::Bitmap { bind_group }
                } else {
                    return None;
                }
            }
            _ => unreachable!(),
        };

        Some(Draw {
            draw_type,
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            is_stroke: false,
            fill_rule: Some(winding_rule),
            analytic_fill: maybe_analytic.as_ref().map(|(fill, _)| fill.clone()),
            analytic_stroke: None,
            analytic_bind_group: maybe_analytic.map(|(_, bg)| bg),
        })
    }
}

/// Classification of fill type for dispatch.
enum FillKind {
    Color,
    Gradient,
    Bitmap,
}

/// Build fill geometry with per-vertex color (for solid color fills).
///
/// Returns (vertices, indices) using fan triangulation for interior and
/// Loop-Blinn triangles for curve edges.
fn build_fill_geometry_color(
    commands: &[DrawCommand],
    color: [f32; 4],
) -> (Vec<BezierVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // We process each sub-path (delimited by MoveTo commands).
    let mut fan_origin: Option<u32> = None;
    let mut prev_index: Option<u32> = None;

    for cmd in commands {
        match cmd {
            DrawCommand::MoveTo(pt) => {
                // Start a new sub-path. The first point becomes the fan origin.
                let idx = vertices.len() as u32;
                vertices.push(BezierVertex {
                    position: [pt.x.to_pixels() as f32, pt.y.to_pixels() as f32],
                    uv: [0.0, 0.0],
                    fill_type: 0,
                    color,
                    _pad: 0,
                });
                fan_origin = Some(idx);
                prev_index = Some(idx);
            }
            DrawCommand::LineTo(pt) => {
                let idx = vertices.len() as u32;
                vertices.push(BezierVertex {
                    position: [pt.x.to_pixels() as f32, pt.y.to_pixels() as f32],
                    uv: [0.0, 0.0],
                    fill_type: 0,
                    color,
                    _pad: 0,
                });

                // Fan triangle: fan_origin → prev → current.
                if let (Some(origin), Some(prev)) = (fan_origin, prev_index) {
                    if origin != prev && prev != idx {
                        indices.push(origin);
                        indices.push(prev);
                        indices.push(idx);
                    }
                }
                prev_index = Some(idx);
            }
            DrawCommand::QuadraticCurveTo { control, anchor } => {
                // The anchor (end) point gets added to the vertex list for the
                // main path so fan triangulation can continue.
                let anchor_idx = vertices.len() as u32;
                vertices.push(BezierVertex {
                    position: [
                        anchor.x.to_pixels() as f32,
                        anchor.y.to_pixels() as f32,
                    ],
                    uv: [0.0, 0.0],
                    fill_type: 0,
                    color,
                    _pad: 0,
                });

                // Fan triangle from origin → prev on-curve point → anchor.
                if let (Some(origin), Some(prev)) = (fan_origin, prev_index) {
                    if origin != prev && prev != anchor_idx {
                        indices.push(origin);
                        indices.push(prev);
                        indices.push(anchor_idx);
                    }
                }

                // Now create the Loop-Blinn curve correction triangle.
                // This triangle's three vertices are: prev_on_curve, control, anchor.
                // They receive Loop-Blinn UVs: (0,0), (0.5,0), (1,1).
                let prev_pt = if let Some(prev) = prev_index {
                    vertices[prev as usize].position
                } else {
                    continue;
                };

                let v0 = vertices.len() as u32;
                vertices.push(BezierVertex {
                    position: prev_pt,
                    uv: [0.0, 0.0],
                    fill_type: 1,
                    color,
                    _pad: 0,
                });
                let v1 = vertices.len() as u32;
                vertices.push(BezierVertex {
                    position: [
                        control.x.to_pixels() as f32,
                        control.y.to_pixels() as f32,
                    ],
                    uv: [0.5, 0.0],
                    fill_type: 1,
                    color,
                    _pad: 0,
                });
                let v2 = vertices.len() as u32;
                vertices.push(BezierVertex {
                    position: [
                        anchor.x.to_pixels() as f32,
                        anchor.y.to_pixels() as f32,
                    ],
                    uv: [1.0, 1.0],
                    fill_type: 1,
                    color,
                    _pad: 0,
                });

                indices.push(v0);
                indices.push(v1);
                indices.push(v2);

                prev_index = Some(anchor_idx);
            }
            DrawCommand::CubicCurveTo {
                control_a,
                control_b,
                anchor,
            } => {
                // Approximate cubic as a series of quadratics.
                // For simplicity, use the midpoint subdivision approach.
                // First, flatten to line segments as a fallback.
                let prev_pt = if let Some(prev) = prev_index {
                    let p = vertices[prev as usize].position;
                    swf::Point::new(
                        Twips::from_pixels(p[0] as f64),
                        Twips::from_pixels(p[1] as f64),
                    )
                } else {
                    continue;
                };

                // Subdivide cubic into line segments for the fill.
                let steps = 16;
                let _last_pt = prev_pt;
                for i in 1..=steps {
                    let t = i as f64 / steps as f64;
                    let t2 = t * t;
                    let t3 = t2 * t;
                    let mt = 1.0 - t;
                    let mt2 = mt * mt;
                    let mt3 = mt2 * mt;

                    let x = mt3 * prev_pt.x.to_pixels()
                        + 3.0 * mt2 * t * control_a.x.to_pixels()
                        + 3.0 * mt * t2 * control_b.x.to_pixels()
                        + t3 * anchor.x.to_pixels();
                    let y = mt3 * prev_pt.y.to_pixels()
                        + 3.0 * mt2 * t * control_a.y.to_pixels()
                        + 3.0 * mt * t2 * control_b.y.to_pixels()
                        + t3 * anchor.y.to_pixels();

                    let idx = vertices.len() as u32;
                    vertices.push(BezierVertex {
                        position: [x as f32, y as f32],
                        uv: [0.0, 0.0],
                        fill_type: 0,
                        color,
                        _pad: 0,
                    });

                    if let (Some(origin), Some(prev)) = (fan_origin, prev_index) {
                        if origin != prev && prev != idx {
                            indices.push(origin);
                            indices.push(prev);
                            indices.push(idx);
                        }
                    }
                    prev_index = Some(idx);
                }
            }
        }
    }

    (vertices, indices)
}

/// Build fill geometry for textured fills (gradient/bitmap).
/// Same structure as color fills but uses `BezierTexVertex` (no per-vertex color).
fn build_fill_geometry_tex(
    commands: &[DrawCommand],
) -> (Vec<BezierTexVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let mut fan_origin: Option<u32> = None;
    let mut prev_index: Option<u32> = None;

    for cmd in commands {
        match cmd {
            DrawCommand::MoveTo(pt) => {
                let idx = vertices.len() as u32;
                vertices.push(BezierTexVertex {
                    position: [pt.x.to_pixels() as f32, pt.y.to_pixels() as f32],
                    uv: [0.0, 0.0],
                    fill_type: 0,
                    _pad: 0,
                });
                fan_origin = Some(idx);
                prev_index = Some(idx);
            }
            DrawCommand::LineTo(pt) => {
                let idx = vertices.len() as u32;
                vertices.push(BezierTexVertex {
                    position: [pt.x.to_pixels() as f32, pt.y.to_pixels() as f32],
                    uv: [0.0, 0.0],
                    fill_type: 0,
                    _pad: 0,
                });

                if let (Some(origin), Some(prev)) = (fan_origin, prev_index) {
                    if origin != prev && prev != idx {
                        indices.push(origin);
                        indices.push(prev);
                        indices.push(idx);
                    }
                }
                prev_index = Some(idx);
            }
            DrawCommand::QuadraticCurveTo { control, anchor } => {
                let anchor_idx = vertices.len() as u32;
                vertices.push(BezierTexVertex {
                    position: [
                        anchor.x.to_pixels() as f32,
                        anchor.y.to_pixels() as f32,
                    ],
                    uv: [0.0, 0.0],
                    fill_type: 0,
                    _pad: 0,
                });

                if let (Some(origin), Some(prev)) = (fan_origin, prev_index) {
                    if origin != prev && prev != anchor_idx {
                        indices.push(origin);
                        indices.push(prev);
                        indices.push(anchor_idx);
                    }
                }

                // Loop-Blinn curve correction triangle.
                let prev_pt = if let Some(prev) = prev_index {
                    vertices[prev as usize].position
                } else {
                    continue;
                };

                let v0 = vertices.len() as u32;
                vertices.push(BezierTexVertex {
                    position: prev_pt,
                    uv: [0.0, 0.0],
                    fill_type: 1,
                    _pad: 0,
                });
                let v1 = vertices.len() as u32;
                vertices.push(BezierTexVertex {
                    position: [
                        control.x.to_pixels() as f32,
                        control.y.to_pixels() as f32,
                    ],
                    uv: [0.5, 0.0],
                    fill_type: 1,
                    _pad: 0,
                });
                let v2 = vertices.len() as u32;
                vertices.push(BezierTexVertex {
                    position: [
                        anchor.x.to_pixels() as f32,
                        anchor.y.to_pixels() as f32,
                    ],
                    uv: [1.0, 1.0],
                    fill_type: 1,
                    _pad: 0,
                });

                indices.push(v0);
                indices.push(v1);
                indices.push(v2);

                prev_index = Some(anchor_idx);
            }
            DrawCommand::CubicCurveTo {
                control_a,
                control_b,
                anchor,
            } => {
                // Approximate cubic with line segments.
                let prev_pt = if let Some(prev) = prev_index {
                    vertices[prev as usize].position
                } else {
                    continue;
                };

                let steps = 16;
                for i in 1..=steps {
                    let t = i as f64 / steps as f64;
                    let t2 = t * t;
                    let t3 = t2 * t;
                    let mt = 1.0 - t;
                    let mt2 = mt * mt;
                    let mt3 = mt2 * mt;

                    let p0x = prev_pt[0] as f64;
                    let p0y = prev_pt[1] as f64;

                    let x = mt3 * p0x
                        + 3.0 * mt2 * t * control_a.x.to_pixels()
                        + 3.0 * mt * t2 * control_b.x.to_pixels()
                        + t3 * anchor.x.to_pixels();
                    let y = mt3 * p0y
                        + 3.0 * mt2 * t * control_a.y.to_pixels()
                        + 3.0 * mt * t2 * control_b.y.to_pixels()
                        + t3 * anchor.y.to_pixels();

                    let idx = vertices.len() as u32;
                    vertices.push(BezierTexVertex {
                        position: [x as f32, y as f32],
                        uv: [0.0, 0.0],
                        fill_type: 0,
                        _pad: 0,
                    });

                    if let (Some(origin), Some(prev_i)) = (fan_origin, prev_index) {
                        if origin != prev_i && prev_i != idx {
                            indices.push(origin);
                            indices.push(prev_i);
                            indices.push(idx);
                        }
                    }
                    prev_index = Some(idx);
                }
            }
        }
    }

    (vertices, indices)
}

/// Build geometry for a stroke path.
///
/// Supports solid color, gradient, and bitmap stroke fills.
/// Each line segment is expanded into a quad (2 triangles) using the stroke width.
/// Bézier segments are subdivided into line segments first.
/// Proper line caps (butt/round/square) and joins (miter/round/bevel) are applied.
fn build_stroke_draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    commands: &[DrawCommand],
    style: &LineStyle,
    is_closed: bool,
    _shape_id: swf::CharacterId,
    bitmap_handles: &HashMap<u16, BitmapHandle>,
    gradient_layout: &wgpu::BindGroupLayout,
    bitmap_layout: &wgpu::BindGroupLayout,
    analytic_layout: Option<&wgpu::BindGroupLayout>,
    default_sampler: &wgpu::Sampler,
) -> Option<Draw> {
    let width = style.width();
    // Flash draws hairline strokes at 1px minimum.
    let half_width = (width.to_pixels() as f32 / 2.0).max(0.5);

    let fill_style = style.fill_style();
    let start_cap = style.start_cap();
    let end_cap = style.end_cap();
    let join_style = style.join_style();

    // The current analytic stroke shader models stroke coverage as distance to
    // the curve set, which naturally matches round caps/joins. For other cap/join
    // styles, keep classic mesh expansion for better parity.
    let supports_analytic_stroke_style = matches!(start_cap, LineCapStyle::Round)
        && matches!(end_cap, LineCapStyle::Round)
        && matches!(join_style, LineJoinStyle::Round);

    let maybe_analytic = if let Some(analytic_layout) = analytic_layout
        && supports_analytic_stroke_style
    {
        let analytic_stroke = build_analytic_stroke_data(
            device,
            commands,
            half_width,
            is_closed,
            start_cap,
            end_cap,
            join_style,
        )?;

        let analytic_bind_group = create_analytic_bind_group(
            device,
            analytic_layout,
            &analytic_stroke.segment_buffer,
            analytic_stroke.num_segments,
            analytic_stroke.bounds,
            1,
            FillRule::EvenOdd,
            half_width,
        );
        Some((analytic_stroke, analytic_bind_group))
    } else {
        None
    };

    let use_color_vertices = matches!(fill_style, FillStyle::Color(_));

    if use_color_vertices {
        let color = match fill_style {
            FillStyle::Color(c) => [
                f32::from(c.r) / 255.0,
                f32::from(c.g) / 255.0,
                f32::from(c.b) / 255.0,
                f32::from(c.a) / 255.0,
            ],
            _ => unreachable!(),
        };

        let (vertices, indices) = if let Some((analytic_stroke, _)) = &maybe_analytic {
            build_analytic_quad_color(analytic_stroke.bounds, color)
        } else {
            let mut vertices: Vec<BezierVertex> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let sub_paths = flatten_commands_to_polylines(commands);
            for path in &sub_paths {
                expand_stroke_path_color(
                    path,
                    half_width,
                    color,
                    is_closed,
                    start_cap,
                    end_cap,
                    join_style,
                    &mut vertices,
                    &mut indices,
                );
            }
            (vertices, indices)
        };

        if indices.is_empty() {
            return None;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stroke color vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stroke color indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Some(Draw {
            draw_type: DrawType::Color,
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            is_stroke: true,
            fill_rule: None,
            analytic_fill: None,
            analytic_stroke: maybe_analytic.as_ref().map(|(stroke, _)| stroke.clone()),
            analytic_bind_group: maybe_analytic.map(|(_, bg)| bg),
        })
    } else {
        // Gradient or bitmap stroke: use BezierTexVertex.
        let (vertices, indices) = if let Some((analytic_stroke, _)) = &maybe_analytic {
            build_analytic_quad_tex(analytic_stroke.bounds)
        } else {
            let mut vertices: Vec<BezierTexVertex> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let sub_paths = flatten_commands_to_polylines(commands);
            for path in &sub_paths {
                expand_stroke_path_tex(
                    path,
                    half_width,
                    is_closed,
                    start_cap,
                    end_cap,
                    join_style,
                    &mut vertices,
                    &mut indices,
                );
            }
            (vertices, indices)
        };

        if indices.is_empty() {
            return None;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stroke tex vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stroke tex indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let draw_type = match fill_style {
            FillStyle::LinearGradient(gradient) => {
                let bind_group = create_gradient_bind_group(
                    device, queue, gradient_layout, default_sampler,
                    GradientType::Linear, gradient, swf::Fixed8::ZERO,
                );
                DrawType::Gradient { bind_group }
            }
            FillStyle::RadialGradient(gradient) => {
                let bind_group = create_gradient_bind_group(
                    device, queue, gradient_layout, default_sampler,
                    GradientType::Radial, gradient, swf::Fixed8::ZERO,
                );
                DrawType::Gradient { bind_group }
            }
            FillStyle::FocalGradient { gradient, focal_point } => {
                let bind_group = create_gradient_bind_group(
                    device, queue, gradient_layout, default_sampler,
                    GradientType::Focal, gradient, *focal_point,
                );
                DrawType::Gradient { bind_group }
            }
            FillStyle::Bitmap { id, matrix, is_smoothed, is_repeating } => {
                if let Some(handle) = bitmap_handles.get(id) {
                    let texture: &crate::Texture =
                        <dyn Any>::downcast_ref(&*handle.0).expect("Must be a Texture");
                    let texture_view = texture.texture.create_view(&Default::default());

                    let tex_matrix = swf_bitmap_to_gl_matrix(
                        (*matrix).into(),
                        texture.texture.width(),
                        texture.texture.height(),
                    );
                    let tex_transforms_data = TextureTransforms {
                        u_matrix: matrix_3x3_to_4x4(&tex_matrix),
                    };
                    let tex_transforms_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Stroke bitmap tex transforms"),
                            contents: bytemuck::bytes_of(&tex_transforms_data),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                    let address_mode = if *is_repeating {
                        wgpu::AddressMode::Repeat
                    } else {
                        wgpu::AddressMode::ClampToEdge
                    };
                    let filter_mode = if *is_smoothed {
                        wgpu::FilterMode::Linear
                    } else {
                        wgpu::FilterMode::Nearest
                    };
                    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                        label: Some("Stroke bitmap sampler"),
                        address_mode_u: address_mode,
                        address_mode_v: address_mode,
                        mag_filter: filter_mode,
                        min_filter: filter_mode,
                        ..Default::default()
                    });

                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Stroke bitmap bind group"),
                        layout: bitmap_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: tex_transforms_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&texture_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                        ],
                    });
                    DrawType::Bitmap { bind_group }
                } else {
                    return None;
                }
            }
            _ => unreachable!(),
        };

        Some(Draw {
            draw_type,
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            is_stroke: true,
            fill_rule: None,
            analytic_fill: None,
            analytic_stroke: maybe_analytic.as_ref().map(|(stroke, _)| stroke.clone()),
            analytic_bind_group: maybe_analytic.map(|(_, bg)| bg),
        })
    }
}

fn build_analytic_quad_color(bounds: [f32; 4], color: [f32; 4]) -> (Vec<BezierVertex>, Vec<u32>) {
    let [min_x, min_y, max_x, max_y] = bounds;
    let vertices = vec![
        BezierVertex { position: [min_x, min_y], uv: [0.0, 0.0], fill_type: 0, color, _pad: 0 },
        BezierVertex { position: [max_x, min_y], uv: [0.0, 0.0], fill_type: 0, color, _pad: 0 },
        BezierVertex { position: [max_x, max_y], uv: [0.0, 0.0], fill_type: 0, color, _pad: 0 },
        BezierVertex { position: [min_x, max_y], uv: [0.0, 0.0], fill_type: 0, color, _pad: 0 },
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    (vertices, indices)
}

fn build_analytic_quad_tex(bounds: [f32; 4]) -> (Vec<BezierTexVertex>, Vec<u32>) {
    let [min_x, min_y, max_x, max_y] = bounds;
    let vertices = vec![
        BezierTexVertex { position: [min_x, min_y], uv: [0.0, 0.0], fill_type: 0, _pad: 0 },
        BezierTexVertex { position: [max_x, min_y], uv: [0.0, 0.0], fill_type: 0, _pad: 0 },
        BezierTexVertex { position: [max_x, max_y], uv: [0.0, 0.0], fill_type: 0, _pad: 0 },
        BezierTexVertex { position: [min_x, max_y], uv: [0.0, 0.0], fill_type: 0, _pad: 0 },
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    (vertices, indices)
}

fn create_analytic_bind_group(
    device: &wgpu::Device,
    analytic_layout: &wgpu::BindGroupLayout,
    segment_buffer: &wgpu::Buffer,
    num_segments: u32,
    bounds: [f32; 4],
    mode: u32,
    fill_rule: FillRule,
    half_width: f32,
) -> wgpu::BindGroup {
    let params = crate::AnalyticParams {
        bounds,
        num_segments,
        mode,
        fill_rule: match fill_rule {
            FillRule::EvenOdd => 0,
            FillRule::NonZero => 1,
        },
        half_width,
        cap_join_flags: 0,
        _pad0: [0, 0, 0],
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Analytic params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Analytic bind group"),
        layout: analytic_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: segment_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}

fn build_analytic_stroke_data(
    device: &wgpu::Device,
    commands: &[DrawCommand],
    half_width: f32,
    is_closed: bool,
    start_cap: LineCapStyle,
    end_cap: LineCapStyle,
    join_style: LineJoinStyle,
) -> Option<AnalyticStrokeData> {
    let (segments, bounds) = collect_quadratic_segments(commands);
    if segments.is_empty() {
        return None;
    }

    let expanded_bounds = [
        bounds[0] - half_width,
        bounds[1] - half_width,
        bounds[2] + half_width,
        bounds[3] + half_width,
    ];

    let segment_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Analytic stroke segments"),
        contents: bytemuck::cast_slice(&segments),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    Some(AnalyticStrokeData {
        segment_buffer,
        num_segments: segments.len() as u32,
        bounds: expanded_bounds,
        half_width,
        is_closed,
        start_cap,
        end_cap,
        join_style,
    })
}

/// Build/upload analytic quadratic segments and path bounds for a fill path.
fn build_analytic_fill_data(
    device: &wgpu::Device,
    commands: &[DrawCommand],
) -> Option<AnalyticFillData> {
    let (segments, bounds) = collect_quadratic_segments(commands);
    if segments.is_empty() {
        return None;
    }

    let segment_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Analytic fill segments"),
        contents: bytemuck::cast_slice(&segments),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    Some(AnalyticFillData {
        segment_buffer,
        num_segments: segments.len() as u32,
        bounds,
    })
}

/// Convert draw commands into a list of oriented quadratic segments and bounds.
fn collect_quadratic_segments(commands: &[DrawCommand]) -> (Vec<GpuQuadraticSegment>, [f32; 4]) {
    let mut segments = Vec::new();

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    let mut cursor: Option<[f32; 2]> = None;
    let mut subpath_start: Option<[f32; 2]> = None;

    let mut include = |p: [f32; 2]| {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    };

    let push_line_as_quad = |from: [f32; 2], to: [f32; 2], out: &mut Vec<GpuQuadraticSegment>| {
        let control = [(from[0] + to[0]) * 0.5, (from[1] + to[1]) * 0.5];
        out.push(GpuQuadraticSegment {
            start: from,
            control,
            end: to,
            _pad0: [0.0, 0.0],
        });
    };

    for cmd in commands {
        match cmd {
            DrawCommand::MoveTo(pt) => {
                let p = [pt.x.to_pixels() as f32, pt.y.to_pixels() as f32];

                // Close previous subpath if needed.
                if let (Some(start), Some(cur)) = (subpath_start, cursor) {
                    if start != cur {
                        push_line_as_quad(cur, start, &mut segments);
                        include(start);
                    }
                }

                cursor = Some(p);
                subpath_start = Some(p);
                include(p);
            }
            DrawCommand::LineTo(pt) => {
                let to = [pt.x.to_pixels() as f32, pt.y.to_pixels() as f32];
                if let Some(from) = cursor {
                    push_line_as_quad(from, to, &mut segments);
                    include(from);
                    include(to);
                    cursor = Some(to);
                }
            }
            DrawCommand::QuadraticCurveTo { control, anchor } => {
                let c = [control.x.to_pixels() as f32, control.y.to_pixels() as f32];
                let a = [anchor.x.to_pixels() as f32, anchor.y.to_pixels() as f32];
                if let Some(from) = cursor {
                    segments.push(GpuQuadraticSegment {
                        start: from,
                        control: c,
                        end: a,
                        _pad0: [0.0, 0.0],
                    });
                    include(from);
                    include(c);
                    include(a);
                    cursor = Some(a);
                }
            }
            DrawCommand::CubicCurveTo {
                control_a,
                control_b,
                anchor,
            } => {
                // Phase 1 fallback: flatten cubic to line-as-quadratic segments.
                let Some(from) = cursor else { continue };

                let steps = 16;
                let mut prev = from;
                for i in 1..=steps {
                    let t = i as f32 / steps as f32;
                    let mt = 1.0 - t;
                    let mt2 = mt * mt;
                    let mt3 = mt2 * mt;
                    let t2 = t * t;
                    let t3 = t2 * t;

                    let p = [
                        mt3 * from[0]
                            + 3.0 * mt2 * t * control_a.x.to_pixels() as f32
                            + 3.0 * mt * t2 * control_b.x.to_pixels() as f32
                            + t3 * anchor.x.to_pixels() as f32,
                        mt3 * from[1]
                            + 3.0 * mt2 * t * control_a.y.to_pixels() as f32
                            + 3.0 * mt * t2 * control_b.y.to_pixels() as f32
                            + t3 * anchor.y.to_pixels() as f32,
                    ];

                    push_line_as_quad(prev, p, &mut segments);
                    include(prev);
                    include(p);
                    prev = p;
                }
                cursor = Some(prev);
            }
        }
    }

    // Close final subpath if needed.
    if let (Some(start), Some(cur)) = (subpath_start, cursor) {
        if start != cur {
            let control = [(cur[0] + start[0]) * 0.5, (cur[1] + start[1]) * 0.5];
            segments.push(GpuQuadraticSegment {
                start: cur,
                control,
                end: start,
                _pad0: [0.0, 0.0],
            });
            include(start);
        }
    }

    let bounds = if min_x.is_finite() {
        [min_x, min_y, max_x, max_y]
    } else {
        [0.0, 0.0, 0.0, 0.0]
    };

    (segments, bounds)
}

/// Flatten DrawCommands into polyline sub-paths.
fn flatten_commands_to_polylines(commands: &[DrawCommand]) -> Vec<Vec<[f32; 2]>> {
    let mut sub_paths: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut points: Vec<[f32; 2]> = Vec::new();

    for cmd in commands {
        match cmd {
            DrawCommand::MoveTo(pt) => {
                if points.len() >= 2 {
                    sub_paths.push(std::mem::take(&mut points));
                } else {
                    points.clear();
                }
                points.push([pt.x.to_pixels() as f32, pt.y.to_pixels() as f32]);
            }
            DrawCommand::LineTo(pt) => {
                points.push([pt.x.to_pixels() as f32, pt.y.to_pixels() as f32]);
            }
            DrawCommand::QuadraticCurveTo { control, anchor } => {
                let start = *points.last().unwrap_or(&[0.0, 0.0]);
                for i in 1..=STROKE_BEZIER_SUBDIVISIONS {
                    let t = i as f32 / STROKE_BEZIER_SUBDIVISIONS as f32;
                    let mt = 1.0 - t;
                    let x = mt * mt * start[0]
                        + 2.0 * mt * t * control.x.to_pixels() as f32
                        + t * t * anchor.x.to_pixels() as f32;
                    let y = mt * mt * start[1]
                        + 2.0 * mt * t * control.y.to_pixels() as f32
                        + t * t * anchor.y.to_pixels() as f32;
                    points.push([x, y]);
                }
            }
            DrawCommand::CubicCurveTo { control_a, control_b, anchor } => {
                let start = *points.last().unwrap_or(&[0.0, 0.0]);
                for i in 1..=STROKE_BEZIER_SUBDIVISIONS {
                    let t = i as f32 / STROKE_BEZIER_SUBDIVISIONS as f32;
                    let mt = 1.0 - t;
                    let mt2 = mt * mt;
                    let mt3 = mt2 * mt;
                    let t2 = t * t;
                    let t3 = t2 * t;
                    let x = mt3 * start[0]
                        + 3.0 * mt2 * t * control_a.x.to_pixels() as f32
                        + 3.0 * mt * t2 * control_b.x.to_pixels() as f32
                        + t3 * anchor.x.to_pixels() as f32;
                    let y = mt3 * start[1]
                        + 3.0 * mt2 * t * control_a.y.to_pixels() as f32
                        + 3.0 * mt * t2 * control_b.y.to_pixels() as f32
                        + t3 * anchor.y.to_pixels() as f32;
                    points.push([x, y]);
                }
            }
        }
    }
    if points.len() >= 2 {
        sub_paths.push(points);
    }
    sub_paths
}

/// Compute the perpendicular normal of a line segment, scaled by half_width.
fn segment_normal(p0: [f32; 2], p1: [f32; 2], half_width: f32) -> Option<[f32; 2]> {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-6 {
        return None;
    }
    Some([-dy / len * half_width, dx / len * half_width])
}

/// Number of segments used to approximate a round cap or round join.
const ROUND_CAP_SEGMENTS: usize = 8;

/// Add a line cap to the stroke geometry (color vertices).
fn add_cap_color(
    point: [f32; 2],
    normal: [f32; 2],
    direction: [f32; 2],  // unit direction vector outward from the stroke endpoint
    half_width: f32,
    cap_style: LineCapStyle,
    color: [f32; 4],
    vertices: &mut Vec<BezierVertex>,
    indices: &mut Vec<u32>,
) {
    match cap_style {
        LineCapStyle::None => {
            // Butt cap: no extension. Already handled by the segment quads.
        }
        LineCapStyle::Square => {
            // Extend the stroke by half_width in the direction.
            let ext = [direction[0] * half_width, direction[1] * half_width];
            let base = vertices.len() as u32;
            // The 4 corners of the square cap.
            vertices.push(BezierVertex {
                position: [point[0] + normal[0], point[1] + normal[1]],
                uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
            });
            vertices.push(BezierVertex {
                position: [point[0] - normal[0], point[1] - normal[1]],
                uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
            });
            vertices.push(BezierVertex {
                position: [point[0] + normal[0] + ext[0], point[1] + normal[1] + ext[1]],
                uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
            });
            vertices.push(BezierVertex {
                position: [point[0] - normal[0] + ext[0], point[1] - normal[1] + ext[1]],
                uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
            });
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
        }
        LineCapStyle::Round => {
            // Semicircle fan at the endpoint.
            let center_idx = vertices.len() as u32;
            vertices.push(BezierVertex {
                position: point,
                uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
            });
            // Compute the start angle from the normal.
            let start_angle = normal[1].atan2(normal[0]);
            for i in 0..=ROUND_CAP_SEGMENTS {
                let angle = start_angle + std::f32::consts::PI * (i as f32 / ROUND_CAP_SEGMENTS as f32);
                let vx = point[0] + angle.cos() * half_width;
                let vy = point[1] + angle.sin() * half_width;
                let idx = vertices.len() as u32;
                vertices.push(BezierVertex {
                    position: [vx, vy],
                    uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
                });
                if i > 0 {
                    indices.extend_from_slice(&[center_idx, idx - 1, idx]);
                }
            }
        }
    }
}

/// Add a line join at a vertex where two segments meet (color vertices).
fn add_join_color(
    point: [f32; 2],
    n0: [f32; 2],   // normal of incoming segment
    n1: [f32; 2],   // normal of outgoing segment
    half_width: f32,
    join_style: LineJoinStyle,
    color: [f32; 4],
    vertices: &mut Vec<BezierVertex>,
    indices: &mut Vec<u32>,
) {
    // Determine which side the join is on using the cross product.
    let cross = n0[0] * n1[1] - n0[1] * n1[0];
    if cross.abs() < 1e-6 {
        // Segments are nearly parallel, no join needed.
        return;
    }

    match join_style {
        LineJoinStyle::Miter(miter_limit) => {
            // Compute the miter point.
            let dot = n0[0] * n1[0] + n0[1] * n1[1];
            let hw_sq = half_width * half_width;
            let cos_half = ((1.0 + dot / hw_sq) / 2.0).sqrt().max(1e-6);
            let miter_length = half_width / cos_half;
            let limit = miter_limit.to_f32() * half_width;

            if miter_length <= limit {
                // Miter: extend to the intersection point.
                let avg_n = [
                    (n0[0] + n1[0]) / (2.0 * cos_half * cos_half),
                    (n0[1] + n1[1]) / (2.0 * cos_half * cos_half),
                ];
                // Fill the miter triangle on the outer side.
                let base = vertices.len() as u32;
                if cross > 0.0 {
                    // Join is on the +normal side.
                    vertices.push(BezierVertex {
                        position: point,
                        uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
                    });
                    vertices.push(BezierVertex {
                        position: [point[0] + n0[0], point[1] + n0[1]],
                        uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
                    });
                    vertices.push(BezierVertex {
                        position: [point[0] + avg_n[0], point[1] + avg_n[1]],
                        uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
                    });
                    vertices.push(BezierVertex {
                        position: [point[0] + n1[0], point[1] + n1[1]],
                        uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
                    });
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                } else {
                    vertices.push(BezierVertex {
                        position: point,
                        uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
                    });
                    vertices.push(BezierVertex {
                        position: [point[0] - n0[0], point[1] - n0[1]],
                        uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
                    });
                    vertices.push(BezierVertex {
                        position: [point[0] - avg_n[0], point[1] - avg_n[1]],
                        uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
                    });
                    vertices.push(BezierVertex {
                        position: [point[0] - n1[0], point[1] - n1[1]],
                        uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
                    });
                    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            } else {
                // Miter limit exceeded: fall back to bevel.
                add_bevel_color(point, n0, n1, cross, color, vertices, indices);
            }
        }
        LineJoinStyle::Bevel => {
            add_bevel_color(point, n0, n1, cross, color, vertices, indices);
        }
        LineJoinStyle::Round => {
            // Arc fan between the two normals.
            let angle0 = n0[1].atan2(n0[0]);
            let angle1 = n1[1].atan2(n1[0]);
            let mut sweep = angle1 - angle0;
            if cross > 0.0 {
                if sweep < 0.0 { sweep += 2.0 * std::f32::consts::PI; }
            } else {
                if sweep > 0.0 { sweep -= 2.0 * std::f32::consts::PI; }
            }

            let steps = ((sweep.abs() / std::f32::consts::PI * ROUND_CAP_SEGMENTS as f32).ceil() as usize).max(2);
            let center_idx = vertices.len() as u32;
            vertices.push(BezierVertex {
                position: point,
                uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
            });
            for i in 0..=steps {
                let angle = angle0 + sweep * (i as f32 / steps as f32);
                let vx = point[0] + angle.cos() * half_width;
                let vy = point[1] + angle.sin() * half_width;
                let idx = vertices.len() as u32;
                vertices.push(BezierVertex {
                    position: [vx, vy],
                    uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
                });
                if i > 0 {
                    indices.extend_from_slice(&[center_idx, idx - 1, idx]);
                }
            }
        }
    }
}

/// Add a bevel join triangle (color).
fn add_bevel_color(
    point: [f32; 2],
    n0: [f32; 2],
    n1: [f32; 2],
    cross: f32,
    color: [f32; 4],
    vertices: &mut Vec<BezierVertex>,
    indices: &mut Vec<u32>,
) {
    let base = vertices.len() as u32;
    vertices.push(BezierVertex {
        position: point,
        uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
    });
    if cross > 0.0 {
        vertices.push(BezierVertex {
            position: [point[0] + n0[0], point[1] + n0[1]],
            uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
        });
        vertices.push(BezierVertex {
            position: [point[0] + n1[0], point[1] + n1[1]],
            uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
        });
    } else {
        vertices.push(BezierVertex {
            position: [point[0] - n0[0], point[1] - n0[1]],
            uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
        });
        vertices.push(BezierVertex {
            position: [point[0] - n1[0], point[1] - n1[1]],
            uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2]);
}

/// Expand a polyline into a thick stroke with caps and joins (color vertices).
fn expand_stroke_path_color(
    points: &[[f32; 2]],
    half_width: f32,
    color: [f32; 4],
    is_closed: bool,
    start_cap: LineCapStyle,
    end_cap: LineCapStyle,
    join_style: LineJoinStyle,
    vertices: &mut Vec<BezierVertex>,
    indices: &mut Vec<u32>,
) {
    if points.len() < 2 {
        return;
    }

    // Collect segment normals.
    let mut normals: Vec<[f32; 2]> = Vec::new();
    let mut directions: Vec<[f32; 2]> = Vec::new();
    for i in 0..points.len() - 1 {
        let dx = points[i + 1][0] - points[i][0];
        let dy = points[i + 1][1] - points[i][1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            // Use previous normal or zero.
            normals.push(*normals.last().unwrap_or(&[0.0, half_width]));
            directions.push(*directions.last().unwrap_or(&[1.0, 0.0]));
        } else {
            normals.push([-dy / len * half_width, dx / len * half_width]);
            directions.push([dx / len, dy / len]);
        }
    }

    if normals.is_empty() {
        return;
    }

    // Emit segment quads.
    for i in 0..normals.len() {
        let p0 = points[i];
        let p1 = points[i + 1];
        let n = normals[i];

        let base = vertices.len() as u32;
        vertices.push(BezierVertex {
            position: [p0[0] + n[0], p0[1] + n[1]],
            uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
        });
        vertices.push(BezierVertex {
            position: [p0[0] - n[0], p0[1] - n[1]],
            uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
        });
        vertices.push(BezierVertex {
            position: [p1[0] + n[0], p1[1] + n[1]],
            uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
        });
        vertices.push(BezierVertex {
            position: [p1[0] - n[0], p1[1] - n[1]],
            uv: [0.0, 0.0], fill_type: 0, color, _pad: 0,
        });
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
    }

    // Emit joins at interior vertices.
    for i in 0..normals.len().saturating_sub(1) {
        let join_point = points[i + 1];
        add_join_color(
            join_point, normals[i], normals[i + 1],
            half_width, join_style, color, vertices, indices,
        );
    }

    // Handle caps or closing join.
    if is_closed && normals.len() >= 2 {
        // Closing join between last and first segment.
        add_join_color(
            points[points.len() - 1],
            *normals.last().unwrap(),
            normals[0],
            half_width, join_style, color, vertices, indices,
        );
    } else {
        // Start cap.
        let dir0 = directions[0];
        add_cap_color(
            points[0], normals[0],
            [-dir0[0], -dir0[1]],  // outward = opposite of first segment direction
            half_width, start_cap, color, vertices, indices,
        );
        // End cap.
        let last_dir = *directions.last().unwrap();
        let last_norm = *normals.last().unwrap();
        add_cap_color(
            *points.last().unwrap(), last_norm,
            last_dir,  // outward = same as last segment direction
            half_width, end_cap, color, vertices, indices,
        );
    }
}

// ---- Tex vertex versions for gradient/bitmap strokes ----

/// Add a line cap (tex vertices, for gradient/bitmap strokes).
fn add_cap_tex(
    point: [f32; 2],
    normal: [f32; 2],
    direction: [f32; 2],
    half_width: f32,
    cap_style: LineCapStyle,
    vertices: &mut Vec<BezierTexVertex>,
    indices: &mut Vec<u32>,
) {
    match cap_style {
        LineCapStyle::None => {}
        LineCapStyle::Square => {
            let ext = [direction[0] * half_width, direction[1] * half_width];
            let base = vertices.len() as u32;
            vertices.push(BezierTexVertex {
                position: [point[0] + normal[0], point[1] + normal[1]],
                uv: [0.0, 0.0], fill_type: 0, _pad: 0,
            });
            vertices.push(BezierTexVertex {
                position: [point[0] - normal[0], point[1] - normal[1]],
                uv: [0.0, 0.0], fill_type: 0, _pad: 0,
            });
            vertices.push(BezierTexVertex {
                position: [point[0] + normal[0] + ext[0], point[1] + normal[1] + ext[1]],
                uv: [0.0, 0.0], fill_type: 0, _pad: 0,
            });
            vertices.push(BezierTexVertex {
                position: [point[0] - normal[0] + ext[0], point[1] - normal[1] + ext[1]],
                uv: [0.0, 0.0], fill_type: 0, _pad: 0,
            });
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
        }
        LineCapStyle::Round => {
            let center_idx = vertices.len() as u32;
            vertices.push(BezierTexVertex {
                position: point,
                uv: [0.0, 0.0], fill_type: 0, _pad: 0,
            });
            let start_angle = normal[1].atan2(normal[0]);
            for i in 0..=ROUND_CAP_SEGMENTS {
                let angle = start_angle + std::f32::consts::PI * (i as f32 / ROUND_CAP_SEGMENTS as f32);
                let vx = point[0] + angle.cos() * half_width;
                let vy = point[1] + angle.sin() * half_width;
                let idx = vertices.len() as u32;
                vertices.push(BezierTexVertex {
                    position: [vx, vy],
                    uv: [0.0, 0.0], fill_type: 0, _pad: 0,
                });
                if i > 0 {
                    indices.extend_from_slice(&[center_idx, idx - 1, idx]);
                }
            }
        }
    }
}

/// Add a line join (tex vertices, for gradient/bitmap strokes).
fn add_join_tex(
    point: [f32; 2],
    n0: [f32; 2],
    n1: [f32; 2],
    half_width: f32,
    join_style: LineJoinStyle,
    vertices: &mut Vec<BezierTexVertex>,
    indices: &mut Vec<u32>,
) {
    let cross = n0[0] * n1[1] - n0[1] * n1[0];
    if cross.abs() < 1e-6 {
        return;
    }

    match join_style {
        LineJoinStyle::Miter(miter_limit) => {
            let dot = n0[0] * n1[0] + n0[1] * n1[1];
            let hw_sq = half_width * half_width;
            let cos_half = ((1.0 + dot / hw_sq) / 2.0).sqrt().max(1e-6);
            let miter_length = half_width / cos_half;
            let limit = miter_limit.to_f32() * half_width;

            if miter_length <= limit {
                let avg_n = [
                    (n0[0] + n1[0]) / (2.0 * cos_half * cos_half),
                    (n0[1] + n1[1]) / (2.0 * cos_half * cos_half),
                ];
                let base = vertices.len() as u32;
                if cross > 0.0 {
                    vertices.push(BezierTexVertex { position: point, uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
                    vertices.push(BezierTexVertex { position: [point[0] + n0[0], point[1] + n0[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
                    vertices.push(BezierTexVertex { position: [point[0] + avg_n[0], point[1] + avg_n[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
                    vertices.push(BezierTexVertex { position: [point[0] + n1[0], point[1] + n1[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
                } else {
                    vertices.push(BezierTexVertex { position: point, uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
                    vertices.push(BezierTexVertex { position: [point[0] - n0[0], point[1] - n0[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
                    vertices.push(BezierTexVertex { position: [point[0] - avg_n[0], point[1] - avg_n[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
                    vertices.push(BezierTexVertex { position: [point[0] - n1[0], point[1] - n1[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            } else {
                add_bevel_tex(point, n0, n1, cross, vertices, indices);
            }
        }
        LineJoinStyle::Bevel => {
            add_bevel_tex(point, n0, n1, cross, vertices, indices);
        }
        LineJoinStyle::Round => {
            let angle0 = n0[1].atan2(n0[0]);
            let angle1 = n1[1].atan2(n1[0]);
            let mut sweep = angle1 - angle0;
            if cross > 0.0 {
                if sweep < 0.0 { sweep += 2.0 * std::f32::consts::PI; }
            } else {
                if sweep > 0.0 { sweep -= 2.0 * std::f32::consts::PI; }
            }
            let steps = ((sweep.abs() / std::f32::consts::PI * ROUND_CAP_SEGMENTS as f32).ceil() as usize).max(2);
            let center_idx = vertices.len() as u32;
            vertices.push(BezierTexVertex { position: point, uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
            for i in 0..=steps {
                let angle = angle0 + sweep * (i as f32 / steps as f32);
                let vx = point[0] + angle.cos() * half_width;
                let vy = point[1] + angle.sin() * half_width;
                let idx = vertices.len() as u32;
                vertices.push(BezierTexVertex { position: [vx, vy], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
                if i > 0 {
                    indices.extend_from_slice(&[center_idx, idx - 1, idx]);
                }
            }
        }
    }
}

/// Add a bevel join triangle (tex).
fn add_bevel_tex(
    point: [f32; 2],
    n0: [f32; 2],
    n1: [f32; 2],
    cross: f32,
    vertices: &mut Vec<BezierTexVertex>,
    indices: &mut Vec<u32>,
) {
    let base = vertices.len() as u32;
    vertices.push(BezierTexVertex { position: point, uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
    if cross > 0.0 {
        vertices.push(BezierTexVertex { position: [point[0] + n0[0], point[1] + n0[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
        vertices.push(BezierTexVertex { position: [point[0] + n1[0], point[1] + n1[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
    } else {
        vertices.push(BezierTexVertex { position: [point[0] - n0[0], point[1] - n0[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
        vertices.push(BezierTexVertex { position: [point[0] - n1[0], point[1] - n1[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2]);
}

/// Expand a polyline into a thick stroke with caps and joins (tex vertices).
fn expand_stroke_path_tex(
    points: &[[f32; 2]],
    half_width: f32,
    is_closed: bool,
    start_cap: LineCapStyle,
    end_cap: LineCapStyle,
    join_style: LineJoinStyle,
    vertices: &mut Vec<BezierTexVertex>,
    indices: &mut Vec<u32>,
) {
    if points.len() < 2 {
        return;
    }

    let mut normals: Vec<[f32; 2]> = Vec::new();
    let mut directions: Vec<[f32; 2]> = Vec::new();
    for i in 0..points.len() - 1 {
        let dx = points[i + 1][0] - points[i][0];
        let dy = points[i + 1][1] - points[i][1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            normals.push(*normals.last().unwrap_or(&[0.0, half_width]));
            directions.push(*directions.last().unwrap_or(&[1.0, 0.0]));
        } else {
            normals.push([-dy / len * half_width, dx / len * half_width]);
            directions.push([dx / len, dy / len]);
        }
    }

    if normals.is_empty() {
        return;
    }

    // Emit segment quads.
    for i in 0..normals.len() {
        let p0 = points[i];
        let p1 = points[i + 1];
        let n = normals[i];

        let base = vertices.len() as u32;
        vertices.push(BezierTexVertex { position: [p0[0] + n[0], p0[1] + n[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
        vertices.push(BezierTexVertex { position: [p0[0] - n[0], p0[1] - n[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
        vertices.push(BezierTexVertex { position: [p1[0] + n[0], p1[1] + n[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
        vertices.push(BezierTexVertex { position: [p1[0] - n[0], p1[1] - n[1]], uv: [0.0, 0.0], fill_type: 0, _pad: 0 });
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
    }

    // Joins.
    for i in 0..normals.len().saturating_sub(1) {
        add_join_tex(
            points[i + 1], normals[i], normals[i + 1],
            half_width, join_style, vertices, indices,
        );
    }

    // Caps or closing join.
    if is_closed && normals.len() >= 2 {
        add_join_tex(
            points[points.len() - 1],
            *normals.last().unwrap(),
            normals[0],
            half_width, join_style, vertices, indices,
        );
    } else {
        let dir0 = directions[0];
        add_cap_tex(
            points[0], normals[0],
            [-dir0[0], -dir0[1]],
            half_width, start_cap, vertices, indices,
        );
        let last_dir = *directions.last().unwrap();
        let last_norm = *normals.last().unwrap();
        add_cap_tex(
            *points.last().unwrap(), last_norm,
            last_dir,
            half_width, end_cap, vertices, indices,
        );
    }
}

// ---- Gradient Helpers ----

fn create_gradient_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    default_sampler: &wgpu::Sampler,
    gradient_type: GradientType,
    gradient: &swf::Gradient,
    focal_point: swf::Fixed8,
) -> wgpu::BindGroup {
    // Build gradient LUT texture.
    let colors = build_gradient_lut(&gradient.records);
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("Gradient LUT"),
            size: wgpu::Extent3d {
                width: GRADIENT_SIZE as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &colors,
    );
    let texture_view = texture.create_view(&Default::default());

    // Build texture transform matrix.
    let tex_matrix = swf_to_gl_matrix(gradient.matrix.into());
    let tex_transforms_data = TextureTransforms {
        u_matrix: matrix_3x3_to_4x4(&tex_matrix),
    };
    let tex_transforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Gradient tex transforms"),
        contents: bytemuck::bytes_of(&tex_transforms_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Build gradient uniforms.
    let gradient_uniforms = GradientUniforms {
        focal_point: focal_point.to_f32().clamp(-0.98, 0.98),
        interpolation: (gradient.interpolation == swf::GradientInterpolation::LinearRgb) as i32,
        shape: match gradient_type {
            GradientType::Linear => 1,
            GradientType::Radial => 2,
            GradientType::Focal => 3,
        },
        repeat: match gradient.spread {
            swf::GradientSpread::Pad => 1,
            swf::GradientSpread::Reflect => 2,
            swf::GradientSpread::Repeat => 3,
        },
    };
    let gradient_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Gradient uniforms"),
        contents: bytemuck::bytes_of(&gradient_uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Gradient bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tex_transforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gradient_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(default_sampler),
            },
        ],
    })
}

fn build_gradient_lut(records: &[GradientRecord]) -> [u8; GRADIENT_SIZE * 4] {
    if records.is_empty() {
        return [0; GRADIENT_SIZE * 4];
    }

    let mut colors = [0u8; GRADIENT_SIZE * 4];
    for t in 0..GRADIENT_SIZE {
        let ratio = (t as f32 / (GRADIENT_SIZE - 1) as f32) * 255.0;

        // Find the two surrounding gradient stops.
        let mut last = 0;
        let mut next = 0;
        for (i, record) in records.iter().enumerate().rev() {
            if (record.ratio as f32) <= ratio {
                last = i;
                next = (i + 1).min(records.len() - 1);
                break;
            }
        }

        let last_record = &records[last];
        let next_record = &records[next];

        let a = if next == last {
            0.0
        } else {
            (ratio - last_record.ratio as f32) / (next_record.ratio as f32 - last_record.ratio as f32)
        };

        colors[t * 4] = lerp_u8(last_record.color.r, next_record.color.r, a);
        colors[t * 4 + 1] = lerp_u8(last_record.color.g, next_record.color.g, a);
        colors[t * 4 + 2] = lerp_u8(last_record.color.b, next_record.color.b, a);
        colors[t * 4 + 3] = lerp_u8(last_record.color.a, next_record.color.a, a);
    }
    colors
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t) as u8
}

/// Convert a SWF gradient matrix into a 3x3 GL-style texture transform matrix.
///
/// The SWF gradient coordinate space uses a 32768×32768 space
/// (-16384 to +16384 twips). This matrix converts from object-space
/// pixels into normalized 0..1 UV space.
///
/// The factor of 20 converts the inverse matrix from "per twip" to
/// "per pixel" (since vertex positions in the shader are in pixels but
/// tx/ty are in raw twips). The 32768 normalizes to the gradient box.
/// The +0.5 centers the gradient in [0,1] UV space.
#[expect(clippy::many_single_char_names)]
fn swf_to_gl_matrix(m: ruffle_render::matrix::Matrix) -> [[f32; 3]; 3] {
    let tx = m.tx.get() as f32;
    let ty = m.ty.get() as f32;
    let det = m.a * m.d - m.c * m.b;
    if det.abs() < 1e-10 {
        // Degenerate matrix; return identity.
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let mut a = m.d / det;
    let mut b = -m.c / det;
    let mut c = -(tx * m.d - m.c * ty) / det;
    let mut d = -m.b / det;
    let mut e = m.a / det;
    let mut f = (tx * m.b - m.a * ty) / det;

    // Scale 2x2 part: twips→pixels (20) and gradient-space normalization (32768).
    a *= 20.0 / 32768.0;
    b *= 20.0 / 32768.0;
    d *= 20.0 / 32768.0;
    e *= 20.0 / 32768.0;

    // Translation: normalize and center in [0,1].
    c /= 32768.0;
    f /= 32768.0;
    c += 0.5;
    f += 0.5;

    [[a, d, 0.0], [b, e, 0.0], [c, f, 1.0]]
}

/// Convert a SWF bitmap matrix into a 3x3 GL-style texture transform matrix.
///
/// Similar to `swf_to_gl_matrix` but normalizes by bitmap pixel dimensions
/// instead of the gradient box size. No +0.5 centering offset because the
/// bitmap origin is at the corner, not the center.
#[expect(clippy::many_single_char_names)]
fn swf_bitmap_to_gl_matrix(
    m: ruffle_render::matrix::Matrix,
    bitmap_width: u32,
    bitmap_height: u32,
) -> [[f32; 3]; 3] {
    let bitmap_width = bitmap_width as f32;
    let bitmap_height = bitmap_height as f32;

    let tx = m.tx.get() as f32;
    let ty = m.ty.get() as f32;
    let det = m.a * m.d - m.c * m.b;
    if det.abs() < 1e-10 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let mut a = m.d / det;
    let mut b = -m.c / det;
    let mut c = -(tx * m.d - m.c * ty) / det;
    let mut d = -m.b / det;
    let mut e = m.a / det;
    let mut f = (tx * m.b - m.a * ty) / det;

    // Scale 2x2 part: twips→pixels (20) and bitmap-dimension normalization.
    a *= 20.0 / bitmap_width;
    b *= 20.0 / bitmap_width;
    d *= 20.0 / bitmap_height;
    e *= 20.0 / bitmap_height;

    // Translation: normalize by bitmap dimensions (no centering offset).
    c /= bitmap_width;
    f /= bitmap_height;

    [[a, d, 0.0], [b, e, 0.0], [c, f, 1.0]]
}

fn matrix_3x3_to_4x4(m: &[[f32; 3]; 3]) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[0][1], m[0][2], 0.0],
        [m[1][0], m[1][1], m[1][2], 0.0],
        [m[2][0], m[2][1], m[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
