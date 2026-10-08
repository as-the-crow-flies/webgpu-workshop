//! A small framework for the WebGPU workshop.
//!
//! It takes care of the window, the camera, the mouse, the UI and loading
//! data. Everything on the GPU (buffers, bind groups, pipelines, passes) you
//! write yourself with plain `wgpu`. See `examples/template/main.rs`.

// Lets the examples say `webgpu_workshop::...` even when they are compiled
// as part of this crate (which is what the web build does, see `web.rs`).
extern crate self as webgpu_workshop;

mod app;
mod camera;
mod npy;
#[cfg(target_arch = "wasm32")]
mod web;

pub use app::run;
pub use camera::Camera;
pub use npy::{load_npy, Npy};

// Re-exported so the examples use exactly the same versions.
pub use {bytemuck, egui, glam, wgpu};

use std::sync::{Arc, Mutex};

/// What every example implements. The framework calls these methods; you
/// fill them in with plain `wgpu` code.
// `async fn` in a trait is fine here: the framework only ever calls it directly.
#[allow(async_fn_in_trait)]
pub trait App: Sized + 'static {
    /// Load your data and create buffers, pipelines and bind groups.
    /// It is `async` so it can `.await` file loading, which is a download on the web.
    async fn new(ctx: &mut Context) -> Self;

    /// Add your own sliders and buttons to the side panel.
    fn ui(&mut self, _ctx: &mut Context, _ui: &mut egui::Ui) {}

    /// Record compute passes (simulation). Not called while paused.
    fn update(&mut self, _ctx: &mut Context, _encoder: &mut wgpu::CommandEncoder) {}

    /// Record render passes. `color` is the window, `depth` a depth texture of the same
    /// size (format `ctx.depth_format`). The first pass should clear `color`.
    fn render(
        &mut self,
        ctx: &Context,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    );
}

/// Everything the framework shares with your app.
pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Format of the window's color target. Use it in your render pipelines.
    pub surface_format: wgpu::TextureFormat,
    /// Format of the depth texture passed to `App::render`. Pipelines that draw into a
    /// pass with that depth attachment need a depth state with this format.
    pub depth_format: wgpu::TextureFormat,

    /// The `Globals` uniform buffer (camera, mouse, time; see `shaders/lib/globals.wgsl`).
    /// Shaders declare it at `@group(0) @binding(0)`; by convention every bind group
    /// layout has it at binding 0, and every bind group puts this buffer there.
    pub globals_buffer: wgpu::Buffer,
    /// The values that were written to the globals buffer this frame.
    pub globals: Globals,

    pub camera: Camera,
    pub mouse: Mouse,
    /// Window size in pixels.
    pub size: [u32; 2],
    /// Seconds since start (stops while paused, scaled by `time_scale`).
    pub time: f32,
    /// Seconds since the previous frame (0 while paused).
    pub dt: f32,
    pub frame: u32,
    pub paused: bool,
    pub time_scale: f32,

    pub(crate) errors: Arc<Mutex<Vec<String>>>,
}

impl Context {
    /// Load a shader by its path relative to the crate root, e.g. `"examples/boids/render.wgsl"`.
    /// The shader was checked and its `#import`s resolved when you ran `cargo build`.
    pub fn shader(&self, path: &str) -> wgpu::ShaderModule {
        let Some((_, source)) = SHADERS.iter().find(|(p, _)| *p == path) else {
            panic!("no shader {path:?} (paths are relative to the crate root, e.g. \"examples/boids/render.wgsl\")");
        };
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(path),
                source: wgpu::ShaderSource::Wgsl((*source).into()),
            })
    }

    /// Show an error message in the window (and the log).
    pub fn error(&self, message: impl Into<String>) {
        push_error(&self.errors, message.into());
    }
}

pub(crate) fn push_error(errors: &Mutex<Vec<String>>, message: String) {
    let mut errors = errors.lock().unwrap();
    if !errors.contains(&message) {
        log::error!("{message}");
        errors.push(message);
    }
}

/// The mouse, in pixels and in world space.
#[derive(Clone, Debug, Default)]
pub struct Mouse {
    /// Position in pixels, from the top-left corner.
    pub position: glam::Vec2,
    /// Position in normalized device coordinates: (-1, -1) bottom-left to (1, 1) top-right.
    pub ndc: glam::Vec2,
    /// A world-space ray from the camera through the mouse.
    pub ray_origin: glam::Vec3,
    pub ray_dir: glam::Vec3,
    /// Buttons held down (only presses that started outside the UI count).
    pub left: bool,
    pub right: bool,
    pub middle: bool,
    /// Left button went down / up this frame.
    pub left_pressed: bool,
    pub left_released: bool,
    /// The mouse is over the UI panel.
    pub over_ui: bool,
}

/// The `globals` uniform, available in every shader that does `#import "lib/globals.wgsl"`.
/// Must match `struct Globals` in `shaders/lib/globals.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    pub view: glam::Mat4,
    pub proj: glam::Mat4,
    pub view_proj: glam::Mat4,
    pub inv_view_proj: glam::Mat4,
    pub eye: glam::Vec3,
    pub time: f32,
    pub ray_origin: glam::Vec3,
    pub dt: f32,
    pub ray_dir: glam::Vec3,
    pub frame: u32,
    pub resolution: glam::Vec2,
    pub mouse: glam::Vec2,
    pub mouse_ndc: glam::Vec2,
    pub buttons: u32,
    pub orthographic: u32,
}

// `SHADERS`: every .wgsl file, with its imports resolved; written by build.rs.
include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
