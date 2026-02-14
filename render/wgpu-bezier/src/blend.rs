//! Blend mode classification and GPU blend state definitions.
//!
//! Blend modes are split into two tiers:
//! - **Trivial**: Can be expressed with hardware blend states (Normal, Add, Subtract, Screen).
//! - **Complex**: Require a shader pass compositing two textures (Multiply, Lighten, Darken, etc.).

use enum_map::Enum;
use ruffle_render::commands::RenderBlendMode;
use swf::BlendMode;

/// Complex blend modes that require shader-based compositing.
#[derive(Enum, Debug, Copy, Clone)]
pub enum ComplexBlend {
    Multiply,
    Lighten,
    Darken,
    Difference,
    Invert,
    Alpha,
    Erase,
    Overlay,
    HardLight,
}

/// Classification of a blend mode into trivial or complex.
#[derive(Debug, Clone)]
pub enum BlendType {
    /// Can be expressed with just a hardware blend state.
    Trivial(TrivialBlend),
    /// Requires a shader to composite two textures.
    Complex(ComplexBlend),
}

impl BlendType {
    pub fn from(mode: RenderBlendMode) -> BlendType {
        match mode {
            RenderBlendMode::Builtin(BlendMode::Normal) => BlendType::Trivial(TrivialBlend::Normal),
            RenderBlendMode::Builtin(BlendMode::Layer) => BlendType::Trivial(TrivialBlend::Normal),
            RenderBlendMode::Builtin(BlendMode::Multiply) => {
                BlendType::Complex(ComplexBlend::Multiply)
            }
            RenderBlendMode::Builtin(BlendMode::Screen) => {
                BlendType::Trivial(TrivialBlend::Screen)
            }
            RenderBlendMode::Builtin(BlendMode::Lighten) => {
                BlendType::Complex(ComplexBlend::Lighten)
            }
            RenderBlendMode::Builtin(BlendMode::Darken) => {
                BlendType::Complex(ComplexBlend::Darken)
            }
            RenderBlendMode::Builtin(BlendMode::Difference) => {
                BlendType::Complex(ComplexBlend::Difference)
            }
            RenderBlendMode::Builtin(BlendMode::Add) => BlendType::Trivial(TrivialBlend::Add),
            RenderBlendMode::Builtin(BlendMode::Subtract) => {
                BlendType::Trivial(TrivialBlend::Subtract)
            }
            RenderBlendMode::Builtin(BlendMode::Invert) => {
                BlendType::Complex(ComplexBlend::Invert)
            }
            RenderBlendMode::Builtin(BlendMode::Alpha) => {
                BlendType::Complex(ComplexBlend::Alpha)
            }
            RenderBlendMode::Builtin(BlendMode::Erase) => {
                BlendType::Complex(ComplexBlend::Erase)
            }
            RenderBlendMode::Builtin(BlendMode::Overlay) => {
                BlendType::Complex(ComplexBlend::Overlay)
            }
            RenderBlendMode::Builtin(BlendMode::HardLight) => {
                BlendType::Complex(ComplexBlend::HardLight)
            }
            // PixelBender shader blends are not supported; fall back to Normal.
            RenderBlendMode::Shader(_) => BlendType::Trivial(TrivialBlend::Normal),
        }
    }
}

/// Trivial blend modes expressible via wgpu hardware blend state.
#[derive(Enum, Debug, Copy, Clone)]
pub enum TrivialBlend {
    Normal,
    Add,
    Subtract,
    Screen,
}

impl TrivialBlend {
    /// Return the wgpu `BlendState` for this trivial blend mode.
    pub fn blend_state(self) -> wgpu::BlendState {
        match self {
            TrivialBlend::Normal => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            TrivialBlend::Add => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
            TrivialBlend::Screen => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
            TrivialBlend::Subtract => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::ReverseSubtract,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
        }
    }
}
