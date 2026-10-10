//! Conway's Game of Life on a 512 x 512 grid, generated on the GPU (no .npy file).
//!
//!   cargo run --release --example life
//!
//! Paint living cells with the left mouse button; drag with the right button to pan.
//!
//! Pipelines:
//!   init.wgsl      random cells into buffer A (on start and on "Reset")
//!   simulate.wgsl  one generation: reads one buffer, writes the other
//!   render.wgsl    one quad; the fragment shader looks up the cell under each pixel

use webgpu_workshop::{bytemuck, egui, glam, wgpu, App, Camera, Context};
use wgpu::util::DeviceExt;

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;

/// The sliders. Must have the same layout as `struct Params` in types.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    density: f32,
    seed: u32,
    brush: f32,
}

pub struct Life {
    /// The slider values on the CPU.
    params: Params,
    /// The same values on the GPU (`var<uniform> params`).
    params_buffer: wgpu::Buffer,
    /// Two buffers with cells: ping and pong.
    cells: [wgpu::Buffer; 2],
    /// Which of the two buffers holds the latest state (0 or 1).
    current: usize,
    /// Run init.wgsl in the next `update` (on start and on "Reset").
    needs_init: bool,
    /// Generations per second, and the time since the last one.
    rate: f32,
    /// Time since the last generation.
    timer: f32,

    /// One pipeline per shader, and its bind groups (two for ping-pong).
    init_pipeline: wgpu::ComputePipeline,
    init_bind_group: wgpu::BindGroup,
    simulate_pipeline: wgpu::ComputePipeline,
    simulate_bind_groups: [wgpu::BindGroup; 2],
    render_pipeline: wgpu::RenderPipeline,
    render_bind_groups: [wgpu::BindGroup; 2],
}

impl App for Life {
    async fn new(ctx: &mut Context) -> Self {
        // ---------------------------------------------------------------
        // Step 1: Create Buffers and Textures
        // ---------------------------------------------------------------

        // Nothing to upload: init.wgsl fills buffer A on the GPU.
        let cell_size = 8; // bytes in one `Cell` (see types.wgsl)
                           // Two empty buffers for the cells (ping and pong).
        let cells = ["cells A", "cells B"].map(|label| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (WIDTH * HEIGHT) as u64 * cell_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // The slider values, and a uniform buffer to send them to the GPU every frame.
        let params = Params {
            width: WIDTH,
            height: HEIGHT,
            density: 0.25,
            seed: 0,
            brush: 4.0,
        };
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // ---------------------------------------------------------------
        // Step 2: Write a Shader (init.wgsl, simulate.wgsl and render.wgsl)
        // ---------------------------------------------------------------

        // The shaders are separate .wgsl files in this folder. `cargo build` checks
        // them and points at the file and line of any error.
        let init_shader = ctx.shader("examples/life/init.wgsl");
        let simulate_shader = ctx.shader("examples/life/simulate.wgsl");
        let render_shader = ctx.shader("examples/life/render.wgsl");

        // ---------------------------------------------------------------
        // Step 3: Compile a Shader Pipeline
        // ---------------------------------------------------------------

        // Bind group layouts: what each shader sees at `@group(0) @binding(n)`.
        // Binding 0 is always `globals`.
        use wgpu::BufferBindingType::{Storage, Uniform};
        use wgpu::ShaderStages as S;
        // A bind group layout from a list of entries.
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries,
                })
        };
        let init_layout = layout(
            "init",
            &[
                buffer(0, S::COMPUTE, Uniform),
                buffer(1, S::COMPUTE, Uniform),
                buffer(2, S::COMPUTE, Storage { read_only: false }),
            ],
        );
        let simulate_layout = layout(
            "simulate",
            &[
                buffer(0, S::COMPUTE, Uniform),
                buffer(1, S::COMPUTE, Uniform),
                buffer(2, S::COMPUTE, Storage { read_only: true }),
                buffer(3, S::COMPUTE, Storage { read_only: false }),
            ],
        );
        let render_layout = layout(
            "render",
            &[
                buffer(0, S::VERTEX_FRAGMENT, Uniform),
                buffer(1, S::VERTEX_FRAGMENT, Uniform),
                buffer(2, S::VERTEX_FRAGMENT, Storage { read_only: true }),
            ],
        );

        // A pipeline layout with one bind group layout (`@group(0)`).
        let pipeline_layout = |layout: &wgpu::BindGroupLayout| {
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(layout)],
                    immediate_size: 0,
                })
        };
        // A compute pipeline: a shader, run with `main` as entry point.
        let compute = |label, module: &wgpu::ShaderModule, layout: &wgpu::BindGroupLayout| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout(layout)),
                    module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(), // default shader compiler settings
                    cache: None,
                })
        };
        // The two compute pipelines.
        let init_pipeline = compute("init", &init_shader, &init_layout);
        let simulate_pipeline = compute("simulate", &simulate_shader, &simulate_layout);

        // The render pipeline: render.wgsl plus every fixed setting for drawing with it.
        let render_pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                // Name shown in error messages.
                label: Some("render"),
                // Which bind groups the shader expects.
                layout: Some(&pipeline_layout(&render_layout)),
                // The vertex stage: runs once per vertex, places it on the screen.
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                // How vertices are assembled into triangles.
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                // No depth test: 2D, later draws end up on top.
                depth_stencil: None,
                // Multisampling: off (1 sample per pixel); edges are smoothed in the shader.
                multisample: Default::default(),
                // The fragment stage: runs once per covered pixel, returns its color.
                fragment: Some(wgpu::FragmentState {
                    module: &render_shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                // Render to several array layers at once: no.
                multiview_mask: None,
                // Reuse compiled pipelines between runs: no.
                cache: None,
            });

        // ---------------------------------------------------------------
        // Step 4: Create a Bind Group (for each pipeline; two for ping-pong)
        // ---------------------------------------------------------------

        // The framework's globals buffer, at binding 0 of every bind group.
        let globals = &ctx.globals_buffer;
        let init_bind_group = bind_group(
            &ctx.device,
            &init_layout,
            &[(0, globals), (1, &params_buffer), (2, &cells[0])],
        );
        let simulate_bind_groups = [0, 1].map(|i| {
            bind_group(
                &ctx.device,
                &simulate_layout,
                &[
                    (0, globals),
                    (1, &params_buffer),
                    (2, &cells[i]),
                    (3, &cells[1 - i]),
                ],
            )
        });
        let render_bind_groups = [0, 1].map(|i| {
            bind_group(
                &ctx.device,
                &render_layout,
                &[(0, globals), (1, &params_buffer), (2, &cells[i])],
            )
        });

        // A 2D view of the grid, which covers [-1, 1]^2. Left mouse button paints.
        ctx.camera = Camera::view_2d(glam::Vec2::ZERO, 2.1);
        ctx.camera.left_drag = false;

        Self {
            params,
            params_buffer,
            cells,
            current: 0,
            needs_init: true,
            rate: 30.0,
            timer: 0.0,
            init_pipeline,
            init_bind_group,
            simulate_pipeline,
            simulate_bind_groups,
            render_pipeline,
            render_bind_groups,
        }
    }

    // Keep the slider values when a shader is hot reloaded.
    fn reloaded(&mut self, old: &Self) {
        self.params = old.params;
    }

    // ---------------------------------------------------------------
    // Step 5: Dispatch the shader!
    // ---------------------------------------------------------------
    fn update(&mut self, ctx: &mut Context, encoder: &mut wgpu::CommandEncoder) {
        if self.needs_init {
            self.needs_init = false;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.init_pipeline);
            pass.set_bind_group(0, &self.init_bind_group, &[]);
            pass.dispatch_workgroups((WIDTH * HEIGHT).div_ceil(64), 1, 1);
            drop(pass);
            encoder.copy_buffer_to_buffer(&self.cells[0], 0, &self.cells[1], 0, None);
            self.current = 0;
        }

        // Run as many generations as `rate` asks for (at most 4 per frame).
        self.timer = (self.timer + ctx.dt).min(4.0 / self.rate);
        while self.timer >= 1.0 / self.rate {
            self.timer -= 1.0 / self.rate;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.simulate_pipeline);
            pass.set_bind_group(0, &self.simulate_bind_groups[self.current], &[]);
            // One thread per cell, in tiles of 8 x 8 (see @workgroup_size in simulate.wgsl).
            pass.dispatch_workgroups(WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1);
            self.current = 1 - self.current;
        }
    }

    fn render(
        &mut self,
        ctx: &Context,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        _depth: &wgpu::TextureView,
    ) {
        ctx.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

        // A 2D render pass: no depth attachment.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_bind_groups[self.current], &[]);
        pass.draw(0..6, 0..1); // one quad
    }

    fn ui(&mut self, _ctx: &mut Context, ui: &mut egui::Ui) {
        ui.label(format!("{WIDTH} x {HEIGHT} cells"));
        ui.add(egui::Slider::new(&mut self.rate, 1.0..=120.0).text("generations / s"));
        ui.add(egui::Slider::new(&mut self.params.brush, 1.0..=20.0).text("brush"));
        ui.add(egui::Slider::new(&mut self.params.density, 0.0..=1.0).text("density"));
        if ui.button("Reset").clicked() {
            self.params.seed += 1;
            self.needs_init = true;
        }
    }
}

/// A bind group layout entry for a buffer. (The long form is in examples/template/main.rs.)
fn buffer(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A bind group from `(binding, buffer)` pairs. (The long form is in examples/template/main.rs.)
fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    entries: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let entries: Vec<_> = entries
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

#[allow(dead_code)]
fn main() {
    webgpu_workshop::run::<Life>();
}
