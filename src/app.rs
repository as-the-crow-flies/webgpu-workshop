//! The window and the frame loop. You shouldn't need to change this file,
//! but it is meant to be readable: every frame runs `Runner::frame`.

use std::sync::{Arc, Mutex};

use glam::{Vec2, Vec4Swizzles};
use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::{push_error, App, Context, Globals, Mouse};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Start the app: open a window, call `App::new`, and run the frame loop.
pub fn run<A: App>() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
        pollster::block_on(start::<A>());
    }
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(start::<A>());
}

async fn start<A: App>() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("No GPU adapter found. On the web, use a browser with WebGPU support.");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            // Ask for everything this GPU can do, not just the WebGPU minimums.
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("Could not create a GPU device");

    let event_loop = EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut runner = Runner::<A> {
        setup: Some((instance, adapter, device, queue)),
        state: None,
        #[cfg(target_arch = "wasm32")]
        loading: Default::default(),
    };

    #[cfg(not(target_arch = "wasm32"))]
    event_loop.run_app(&mut runner).unwrap();
    #[cfg(target_arch = "wasm32")]
    winit::platform::web::EventLoopExtWebSys::spawn_app(event_loop, runner);
}

struct Runner<A: App> {
    /// The GPU, until the window exists.
    setup: Option<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)>,
    state: Option<State<A>>,
    /// Web only: `App::new` runs in the background (it may download files);
    /// the finished state lands here.
    #[cfg(target_arch = "wasm32")]
    loading: std::rc::Rc<std::cell::RefCell<Option<State<A>>>>,
}

struct State<A: App> {
    app: A,
    ctx: Context,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth: wgpu::TextureView,
    egui: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    initial_camera: crate::Camera,
    /// Mouse movement and scrolling since the last frame.
    drag: Vec2,
    scroll: f32,
    shift: bool,
    step: bool,
    last_frame: Instant,
    frame_time: f32,
}

impl<A: App> ApplicationHandler for Runner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        exit_on_panic(|| self.start(event_loop));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        exit_on_panic(|| self.event(event_loop, event));
    }
}

/// A panic (e.g. a `.npy` file that doesn't exist) has already printed
/// its message; stop here instead of unwinding into the event loop, which would
/// abort with a long, unhelpful backtrace.
fn exit_on_panic(f: impl FnOnce()) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err() {
        #[cfg(not(target_arch = "wasm32"))]
        std::process::exit(1);
    }
}

impl<A: App> Runner<A> {
    fn start(&mut self, event_loop: &ActiveEventLoop) {
        let Some((instance, adapter, device, queue)) = self.setup.take() else {
            return;
        };

        let title = std::any::type_name::<A>()
            .rsplit("::")
            .next()
            .unwrap_or("workshop");
        #[allow(unused_mut)]
        let mut attributes = Window::default_attributes().with_title(title);
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            attributes = attributes.with_canvas(Some(crate::web::canvas()));
        }
        let window = Arc::new(event_loop.create_window(attributes).unwrap());

        // The surface is what we draw into; it belongs to the window.
        let surface = instance.create_surface(window.clone()).unwrap();
        let capabilities = surface.get_capabilities(&adapter);
        // A non-sRGB format: the colors you return from a fragment shader are shown as-is.
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .unwrap_or(capabilities.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            color_space: Default::default(),
        };
        surface.configure(&device, &config);

        // GPU errors are shown in the window instead of crashing the app.
        let errors = Arc::new(Mutex::new(Vec::new()));
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            push_error(&sink, error.to_string())
        }));

        // The `Globals` uniform (camera, mouse, time), written every frame.
        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let egui_ctx = egui::Context::default();
        let egui = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());

        let depth = depth_view(&device, &config);

        let mut ctx = Context {
            device,
            queue,
            surface_format: format,
            depth_format: DEPTH_FORMAT,
            globals_buffer,
            globals: Globals::default(),
            camera: crate::Camera::orbit(glam::Vec3::ZERO, 3.0),
            mouse: Mouse::default(),
            size: [config.width, config.height],
            time: 0.0,
            dt: 0.0,
            frame: 0,
            paused: false,
            time_scale: 1.0,
            errors,
        };
        update_globals(&mut ctx);

        // `App::new` is async because loading files is a download on the web.
        let make_state = async move {
            let app = A::new(&mut ctx).await;
            window.request_redraw();
            State {
                initial_camera: ctx.camera.clone(),
                app,
                ctx,
                window,
                surface,
                config,
                depth,
                egui,
                egui_renderer,
                drag: Vec2::ZERO,
                scroll: 0.0,
                shift: false,
                step: false,
                last_frame: Instant::now(),
                frame_time: 1.0 / 60.0,
            }
        };

        // Native: just wait for it. Web: run it in the background; `event` picks it up.
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(make_state));
        }
        #[cfg(target_arch = "wasm32")]
        {
            let loading = self.loading.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let state = make_state.await;
                *loading.borrow_mut() = Some(state);
            });
        }
    }

    fn event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        #[cfg(target_arch = "wasm32")]
        if self.state.is_none() {
            self.state = self.loading.borrow_mut().take();
        }
        let Some(state) = &mut self.state else { return };
        let _ = state.egui.on_window_event(&state.window, &event);
        let egui_ctx = state.egui.egui_ctx();
        let over_ui = egui_ctx.egui_wants_pointer_input() || egui_ctx.is_pointer_over_egui();
        let mouse = &mut state.ctx.mouse;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => state.frame(),
            WindowEvent::Occluded(false) => state.window.request_redraw(),
            WindowEvent::ModifiersChanged(modifiers) => state.shift = modifiers.state().shift_key(),
            WindowEvent::CursorMoved { position, .. } => {
                let position = Vec2::new(position.x as f32, position.y as f32);
                state.drag += position - mouse.position;
                mouse.position = position;
            }
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                let pressed = button_state == ElementState::Pressed;
                // Presses on the UI are for the UI; releases always count.
                if pressed && over_ui {
                    return;
                }
                match button {
                    MouseButton::Left => {
                        mouse.left = pressed;
                        mouse.left_pressed |= pressed;
                        mouse.left_released |= !pressed;
                    }
                    MouseButton::Right => mouse.right = pressed,
                    MouseButton::Middle => mouse.middle = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !over_ui => {
                state.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.002,
                };
            }
            WindowEvent::PinchGesture { delta, .. } if !over_ui => state.scroll += delta as f32,
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && !egui_ctx.egui_wants_keyboard_input()
                    && event.physical_key == PhysicalKey::Code(KeyCode::Space) =>
            {
                state.ctx.paused = !state.ctx.paused;
            }
            _ => {}
        }
    }
}

impl<A: App> State<A> {
    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.ctx.device, &self.config);
        self.depth = depth_view(&self.ctx.device, &self.config);
        self.ctx.size = [self.config.width, self.config.height];
    }

    /// One frame: UI -> camera -> globals -> `update` (compute) -> `render` (your passes) -> UI on top.
    fn frame(&mut self) {
        // Get the texture to draw into first. If there is none, skip the whole
        // frame *before* writing any buffers: writes are only freed by a submit.
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                // The window changed: reconfigure and try again.
                self.surface.configure(&self.ctx.device, &self.config);
                self.window.request_redraw();
                return;
            }
            // Minimized or hidden: stop drawing until `WindowEvent::Occluded(false)`.
            wgpu::CurrentSurfaceTexture::Occluded => return,
            // Try again on the next redraw.
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Validation => {
                self.window.request_redraw();
                return;
            }
        };

        let now = Instant::now();
        let elapsed = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.frame_time = 0.95 * self.frame_time + 0.05 * elapsed;

        let ctx = &mut self.ctx;
        let running = !ctx.paused || self.step;
        ctx.dt = if running {
            elapsed * ctx.time_scale
        } else {
            0.0
        };
        ctx.time += ctx.dt;

        // UI: the built-in panel plus your `App::ui`.
        let egui_ctx = self.egui.egui_ctx().clone();
        let input = self.egui.take_egui_input(&self.window);
        let mut reset_camera = false;
        let mut output = egui_ctx.run_ui(input, |ui| {
            panel(
                ui,
                ctx,
                &mut self.app,
                self.frame_time,
                &mut self.step,
                &mut reset_camera,
            );
        });
        self.egui
            .handle_platform_output(&self.window, output.platform_output.clone());
        // Upload new UI textures (fonts, images) right away; free old ones after drawing.
        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas {
                self.egui_renderer
                    .update_texture(&ctx.device, &ctx.queue, *id, delta);
            }
        }
        let textures_to_free = std::mem::take(&mut output.textures_delta.free);
        output.textures_delta.clear();
        if reset_camera {
            ctx.camera = self.initial_camera.clone();
        }

        // Camera and mouse.
        ctx.mouse.over_ui = egui_ctx.egui_wants_pointer_input() || egui_ctx.is_pointer_over_egui();
        let camera_drag =
            ctx.mouse.right || ctx.mouse.middle || (ctx.mouse.left && ctx.camera.left_drag);
        if camera_drag {
            let pan = ctx.mouse.middle || self.shift;
            ctx.camera.drag(self.drag, pan, ctx.size[1] as f32);
        }
        ctx.camera.zoom(self.scroll);
        self.drag = Vec2::ZERO;
        self.scroll = 0.0;
        update_globals(ctx);
        ctx.queue
            .write_buffer(&ctx.globals_buffer, 0, bytemuck::bytes_of(&ctx.globals));

        let view = surface_texture.texture.create_view(&Default::default());
        let mut encoder = ctx.device.create_command_encoder(&Default::default());

        // Your compute passes.
        if running {
            self.app.update(ctx, &mut encoder);
            self.step = false;
        }

        // Your render passes, into the window (`view`) and the depth texture.
        self.app.render(ctx, &mut encoder, &view, &self.depth);

        // The UI, drawn on top.
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: output.pixels_per_point,
        };
        let triangles = egui_ctx.tessellate(output.shapes, output.pixels_per_point);
        let mut extra = self.egui_renderer.update_buffers(
            &ctx.device,
            &ctx.queue,
            &mut encoder,
            &triangles,
            &screen,
        );
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, &triangles, &screen);
        }
        for id in &textures_to_free {
            self.egui_renderer.free_texture(id);
        }

        extra.push(encoder.finish());
        ctx.queue.submit(extra);
        self.window.pre_present_notify();
        ctx.queue.present(surface_texture);

        ctx.frame += 1;
        ctx.mouse.left_pressed = false;
        ctx.mouse.left_released = false;
        self.window.request_redraw();
    }
}

/// Fill `ctx.globals` from the camera, mouse and time.
fn update_globals(ctx: &mut Context) {
    let [width, height] = ctx.size.map(|s| s as f32);
    let view = ctx.camera.view();
    let proj = ctx.camera.projection(width / height);
    let view_proj = proj * view;
    let inv_view_proj = view_proj.inverse();

    // The mouse ray: un-project the mouse at the near (z = 0) and far (z = 1) planes.
    let mouse = &mut ctx.mouse;
    mouse.ndc = Vec2::new(
        2.0 * mouse.position.x / width - 1.0,
        1.0 - 2.0 * mouse.position.y / height,
    );
    let near = inv_view_proj * mouse.ndc.extend(0.0).extend(1.0);
    let far = inv_view_proj * mouse.ndc.extend(1.0).extend(1.0);
    mouse.ray_origin = near.xyz() / near.w;
    mouse.ray_dir = (far.xyz() / far.w - mouse.ray_origin).normalize();

    ctx.globals = Globals {
        view,
        proj,
        view_proj,
        inv_view_proj,
        eye: ctx.camera.eye(),
        time: ctx.time,
        ray_origin: mouse.ray_origin,
        dt: ctx.dt,
        ray_dir: mouse.ray_dir,
        frame: ctx.frame,
        resolution: Vec2::new(width, height),
        mouse: mouse.position,
        mouse_ndc: mouse.ndc,
        buttons: mouse.left as u32 | (mouse.right as u32) << 1 | (mouse.middle as u32) << 2,
        orthographic: ctx.camera.orthographic as u32,
    };
}

fn depth_view(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// The side panel: frame rate, time controls, camera, your `App::ui`, and errors.
fn panel<A: App>(
    ui: &mut egui::Ui,
    ctx: &mut Context,
    app: &mut A,
    frame_time: f32,
    step: &mut bool,
    reset_camera: &mut bool,
) {
    let egui_ctx = ui.ctx().clone();

    egui::Window::new("Workshop")
        .default_pos([12.0, 12.0])
        .default_width(260.0)
        .show(&egui_ctx, |ui| {
            ui.label(format!(
                "{:.0} fps ({:.1} ms)",
                1.0 / frame_time,
                frame_time * 1000.0
            ));
            ui.horizontal(|ui| {
                let label = if ctx.paused { "▶ Play" } else { "⏸ Pause" };
                if ui.button(label).on_hover_text("Space").clicked() {
                    ctx.paused = !ctx.paused;
                }
                if ui
                    .add_enabled(ctx.paused, egui::Button::new("Step"))
                    .clicked()
                {
                    *step = true;
                }
                ui.label(format!("t = {:.1} s", ctx.time));
            });
            ui.add(egui::Slider::new(&mut ctx.time_scale, 0.0..=4.0).text("time scale"));
            ui.horizontal(|ui| {
                ui.checkbox(&mut ctx.camera.orthographic, "orthographic");
                if ui.button("Reset camera").clicked() {
                    *reset_camera = true;
                }
            });
            ui.separator();
            app.ui(ctx, ui);
        });

    let errors = ctx.errors.lock().unwrap().clone();
    if !errors.is_empty() {
        egui::Window::new("Errors")
            .anchor(egui::Align2::LEFT_BOTTOM, [12.0, -12.0])
            .default_width(700.0)
            .show(&egui_ctx, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(300.0)
                    .show(ui, |ui| {
                        for error in &errors {
                            ui.label(
                                egui::RichText::new(error)
                                    .monospace()
                                    .color(egui::Color32::LIGHT_RED),
                            );
                            ui.separator();
                        }
                    });
                if ui.button("Clear").clicked() {
                    ctx.errors.lock().unwrap().clear();
                }
            });
    }
}
