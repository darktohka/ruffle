//! Render target abstraction for the Bézier WGPU backend.
//!
//! This module provides the `RenderTarget` trait and implementations for
//! both surface-based rendering (SwapChainTarget) and texture-based
//! rendering (TextureTarget).

use std::fmt::Debug;

/// A frame from a render target.
pub trait RenderTargetFrame: Debug {
    /// Get a reference to the texture view for this frame.
    fn view(&self) -> &wgpu::TextureView;

    /// Consume this frame and return the texture view.
    fn into_view(self) -> wgpu::TextureView;
}

/// A render target that can be rendered to.
///
/// This trait abstracts over both window surfaces (SwapChainTarget) and
/// off-screen textures (TextureTarget), allowing the BezierRenderBackend
/// to work with either.
pub trait RenderTarget: Debug + 'static {
    /// The frame type returned by this target.
    type Frame: RenderTargetFrame;

    /// Resize this render target to the given dimensions.
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32);

    /// Get the texture format of this target.
    fn format(&self) -> wgpu::TextureFormat;

    /// Get the width of this target.
    fn width(&self) -> u32;

    /// Get the height of this target.
    fn height(&self) -> u32;

    /// Get the next texture to render to.
    fn get_next_texture(&mut self) -> Result<Self::Frame, wgpu::SurfaceError>;

    /// Submit command buffers and present/finalize the frame.
    fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        command_buffers: I,
        frame: Self::Frame,
    ) -> wgpu::SubmissionIndex;
}

/// A render target that renders to a window surface.
#[derive(Debug)]
pub struct SwapChainTarget {
    window_surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
}

/// A frame from a SwapChainTarget.
#[derive(Debug)]
pub struct SwapChainTargetFrame {
    texture: wgpu::SurfaceTexture,
    view: wgpu::TextureView,
}

impl RenderTargetFrame for SwapChainTargetFrame {
    fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    fn into_view(self) -> wgpu::TextureView {
        self.view
    }
}

impl SwapChainTarget {
    /// Create a new SwapChainTarget for the given surface.
    pub fn new(
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        (width, height): (u32, u32),
        device: &wgpu::Device,
    ) -> Self {
        let capabilities = surface.get_capabilities(adapter);
        let format = capabilities
            .formats
            .iter()
            .find(|format| {
                matches!(
                    format,
                    wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
                )
            })
            .or_else(|| capabilities.formats.first())
            .copied()
            .unwrap_or(wgpu::TextureFormat::Rgba8Unorm);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![format],
        };
        surface.configure(device, &surface_config);
        Self {
            surface_config,
            window_surface: surface,
        }
    }
}

impl RenderTarget for SwapChainTarget {
    type Frame = SwapChainTargetFrame;

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.window_surface.configure(device, &self.surface_config);
    }

    fn format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    fn width(&self) -> u32 {
        self.surface_config.width
    }

    fn height(&self) -> u32 {
        self.surface_config.height
    }

    fn get_next_texture(&mut self) -> Result<Self::Frame, wgpu::SurfaceError> {
        let texture = self.window_surface.get_current_texture()?;
        let view = texture.texture.create_view(&Default::default());
        Ok(SwapChainTargetFrame { texture, view })
    }

    fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        command_buffers: I,
        frame: Self::Frame,
    ) -> wgpu::SubmissionIndex {
        let index = queue.submit(command_buffers);
        frame.texture.present();
        index
    }
}

/// A render target that renders to an off-screen texture.
#[derive(Debug)]
pub struct TextureTarget {
    pub size: wgpu::Extent3d,
    pub texture: wgpu::Texture,
    pub format: wgpu::TextureFormat,
}

/// A frame from a TextureTarget.
#[derive(Debug)]
pub struct TextureTargetFrame(wgpu::TextureView);

impl RenderTargetFrame for TextureTargetFrame {
    fn view(&self) -> &wgpu::TextureView {
        &self.0
    }

    fn into_view(self) -> wgpu::TextureView {
        self.0
    }
}

impl TextureTarget {
    /// Create a new TextureTarget with the given dimensions.
    pub fn new(device: &wgpu::Device, size: (u32, u32)) -> Result<Self, String> {
        if size.0 > device.limits().max_texture_dimension_2d
            || size.1 > device.limits().max_texture_dimension_2d
            || size.0 < 1
            || size.1 < 1
        {
            return Err(format!(
                "Texture target cannot be smaller than 1 or larger than {}px on either dimension (requested {} x {})",
                device.limits().max_texture_dimension_2d,
                size.0,
                size.1
            ));
        }
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let size = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render target texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            view_formats: &[format],
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        Ok(Self {
            size,
            texture,
            format,
        })
    }

    /// Get the underlying wgpu texture.
    pub fn get_texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

impl RenderTarget for TextureTarget {
    type Frame = TextureTargetFrame;

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        *self = TextureTarget::new(device, (width, height))
            .expect("Unable to resize texture target");
    }

    fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    fn width(&self) -> u32 {
        self.size.width
    }

    fn height(&self) -> u32 {
        self.size.height
    }

    fn get_next_texture(&mut self) -> Result<Self::Frame, wgpu::SurfaceError> {
        Ok(TextureTargetFrame(
            self.texture.create_view(&Default::default()),
        ))
    }

    fn submit<I: IntoIterator<Item = wgpu::CommandBuffer>>(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        command_buffers: I,
        _frame: Self::Frame,
    ) -> wgpu::SubmissionIndex {
        queue.submit(command_buffers)
    }
}
