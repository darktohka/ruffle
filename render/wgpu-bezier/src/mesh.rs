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
use ruffle_render::shape_utils::{DistilledShape, DrawCommand, DrawPath, GradientType};
use std::any::Any;
use std::collections::HashMap;
use swf::{FillStyle, GradientRecord, Twips};
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
    default_sampler: &wgpu::Sampler,
) -> BezierMesh {
    let mut draws = Vec::new();

    for path in &shape.paths {
        match path {
            DrawPath::Fill {
                style, commands, ..
            } => {
                let draw = build_fill_draw(
                    device,
                    queue,
                    commands,
                    style,
                    shape.id,
                    bitmap_handles,
                    gradient_layout,
                    bitmap_layout,
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
                    commands,
                    style.fill_style(),
                    style.width(),
                    *is_closed,
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
    style: &FillStyle,
    _shape_id: swf::CharacterId,
    bitmap_handles: &HashMap<u16, BitmapHandle>,
    gradient_layout: &wgpu::BindGroupLayout,
    bitmap_layout: &wgpu::BindGroupLayout,
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

        let (vertices, indices) = build_fill_geometry_color(commands, color_f);

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
        })
    } else {
        let (vertices, indices) = build_fill_geometry_tex(commands);

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

                    let tex_matrix = swf_to_gl_matrix((*matrix).into());
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
/// Each line segment is expanded into a quad (2 triangles) using the stroke width.
/// Bézier segments are subdivided into line segments first.
fn build_stroke_draw(
    device: &wgpu::Device,
    commands: &[DrawCommand],
    fill_style: &FillStyle,
    width: Twips,
    _is_closed: bool,
) -> Option<Draw> {
    let half_width = width.to_pixels() as f32 / 2.0;
    if half_width <= 0.0 {
        return None;
    }

    let color = match fill_style {
        FillStyle::Color(c) => [
            f32::from(c.r) / 255.0,
            f32::from(c.g) / 255.0,
            f32::from(c.b) / 255.0,
            f32::from(c.a) / 255.0,
        ],
        // For non-color strokes, default to white and let the color transform handle it.
        _ => [1.0, 1.0, 1.0, 1.0],
    };

    // Flatten all commands into a polyline.
    let mut points: Vec<[f32; 2]> = Vec::new();
    let mut sub_paths: Vec<Vec<[f32; 2]>> = Vec::new();

    for cmd in commands {
        match cmd {
            DrawCommand::MoveTo(pt) => {
                if points.len() >= 2 {
                    sub_paths.push(std::mem::take(&mut points));
                }
                points.push([pt.x.to_pixels() as f32, pt.y.to_pixels() as f32]);
            }
            DrawCommand::LineTo(pt) => {
                points.push([pt.x.to_pixels() as f32, pt.y.to_pixels() as f32]);
            }
            DrawCommand::QuadraticCurveTo { control, anchor } => {
                // Subdivide quadratic Bézier into line segments.
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
            DrawCommand::CubicCurveTo {
                control_a,
                control_b,
                anchor,
            } => {
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

    // Expand each sub-path into stroke quads.
    let mut vertices: Vec<BezierVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for path in &sub_paths {
        expand_stroke_path(path, half_width, color, &mut vertices, &mut indices);
    }

    if indices.is_empty() {
        return None;
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Stroke vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Stroke indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    Some(Draw {
        draw_type: DrawType::Color,
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
    })
}

/// Expand a polyline into a thick stroke (quads = 2 triangles per segment).
fn expand_stroke_path(
    points: &[[f32; 2]],
    half_width: f32,
    color: [f32; 4],
    vertices: &mut Vec<BezierVertex>,
    indices: &mut Vec<u32>,
) {
    if points.len() < 2 {
        return;
    }

    for i in 0..points.len() - 1 {
        let p0 = points[i];
        let p1 = points[i + 1];

        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            continue;
        }

        // Perpendicular normal.
        let nx = -dy / len * half_width;
        let ny = dx / len * half_width;

        let base = vertices.len() as u32;

        // Four corners of the stroke quad.
        vertices.push(BezierVertex {
            position: [p0[0] + nx, p0[1] + ny],
            uv: [0.0, 0.0],
            fill_type: 0,
            color,
            _pad: 0,
        });
        vertices.push(BezierVertex {
            position: [p0[0] - nx, p0[1] - ny],
            uv: [0.0, 0.0],
            fill_type: 0,
            color,
            _pad: 0,
        });
        vertices.push(BezierVertex {
            position: [p1[0] + nx, p1[1] + ny],
            uv: [0.0, 0.0],
            fill_type: 0,
            color,
            _pad: 0,
        });
        vertices.push(BezierVertex {
            position: [p1[0] - nx, p1[1] - ny],
            uv: [0.0, 0.0],
            fill_type: 0,
            color,
            _pad: 0,
        });

        // Two triangles for the quad.
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);

        indices.push(base + 1);
        indices.push(base + 3);
        indices.push(base + 2);
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

/// Convert a SWF matrix into a 3x3 GL-style texture transform matrix.
///
/// The SWF gradient/bitmap coordinate space uses a 32768×32768 space
/// (-16384 to +16384 twips). This matrix converts from object-space
/// twips into that normalized 0..1 UV space.
fn swf_to_gl_matrix(m: ruffle_render::matrix::Matrix) -> [[f32; 3]; 3] {
    let tx = m.tx.get() as f32;
    let ty = m.ty.get() as f32;
    let det = m.a * m.d - m.b * m.c;
    if det.abs() < 1e-10 {
        // Degenerate matrix; return identity.
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let inv_det = 1.0 / det;
    let a = m.d * inv_det;
    let b = -m.b * inv_det;
    let c = -m.c * inv_det;
    let d = m.a * inv_det;
    let out_tx = -(a * tx + c * ty);
    let out_ty = -(b * tx + d * ty);

    [
        [a, b, 0.0],
        [c, d, 0.0],
        [out_tx, out_ty, 1.0],
    ]
}

fn matrix_3x3_to_4x4(m: &[[f32; 3]; 3]) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[0][1], m[0][2], 0.0],
        [m[1][0], m[1][1], m[1][2], 0.0],
        [m[2][0], m[2][1], m[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
