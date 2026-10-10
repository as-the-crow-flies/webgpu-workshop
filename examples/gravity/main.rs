//! Gravity: the template plus a simulation (part 3 of the workshop).
//! 4096 bodies orbit a heavy center, and every body pulls on every other body.
//!
//!   cargo run --release --example gravity
//!
//! The simulation uses "ping-pong": two buffers with bodies. Every frame,
//! simulate.wgsl reads one (ping) and writes the other (pong); then they swap.

use webgpu_workshop::{bytemuck, egui, glam, load_npy, wgpu, App, Camera, Context};
use wgpu::util::DeviceExt;

/// The sliders. Must have the same layout as `struct Params` in types.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    point_scale: f32,
    gravity: f32,
    softening: f32,
    central_mass: f32,
}

pub struct Gravity {
    /// The slider values on the CPU.
    params: Params,
    /// The same values on the GPU (`var<uniform> params`).
    params_buffer: wgpu::Buffer,
    /// Number of bodies.
    count: u32,
    /// Which of the two body buffers holds the latest state (0 or 1).
    current: usize,

    /// The compiled simulate.wgsl.
    simulate_pipeline: wgpu::ComputePipeline,
    /// `simulate_bind_groups[i]` reads buffer i and writes buffer 1 - i.
    simulate_bind_groups: [wgpu::BindGroup; 2],

    /// The compiled render.wgsl plus all fixed drawing settings.
    render_pipeline: wgpu::RenderPipeline,
    /// `render_bind_groups[i]` draws buffer i.
    render_bind_groups: [wgpu::BindGroup; 2],
}

impl App for Gravity {
    async fn new(ctx: &mut Context) -> Self {
        // ---------------------------------------------------------------
        // Step 1: Create Buffers and Textures
        // ---------------------------------------------------------------

        // One row of the file must be exactly one `Body` (see types.wgsl and make_data.py).
        let data = load_npy("examples/gravity/bodies.npy").await;
        // Number of bodies (rows in the file).
        let count = data.shape[0] as u32;

        // Uploaded twice: once for ping, once for pong.
        let bodies = ["bodies A", "bodies B"].map(|label| {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &data.bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        });

        // The slider values, and a uniform buffer to send them to the GPU every frame.
        let params = Params {
            point_scale: 1.0,
            gravity: 1.0,
            softening: 0.05,
            central_mass: 1.0,
        };
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // ---------------------------------------------------------------
        // Step 2: Write a Shader (simulate.wgsl and render.wgsl)
        // ---------------------------------------------------------------

        // The shaders are separate .wgsl files in this folder. `cargo build` checks
        // them and points at the file and line of any error.
        let simulate_shader = ctx.shader("examples/gravity/simulate.wgsl");
        let render_shader = ctx.shader("examples/gravity/render.wgsl");

        // ---------------------------------------------------------------
        // Step 3: Compile a Shader Pipeline
        // ---------------------------------------------------------------

        // simulate.wgsl: @binding(0) globals, (1) params, (2) ping (read), (3) pong (write).
        use wgpu::BufferBindingType::{Storage, Uniform};
        // The bind group layout of simulate.wgsl.
        let simulate_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("simulate"),
                    entries: &[
                        buffer(0, wgpu::ShaderStages::COMPUTE, Uniform),
                        buffer(1, wgpu::ShaderStages::COMPUTE, Uniform),
                        buffer(2, wgpu::ShaderStages::COMPUTE, Storage { read_only: true }),
                        buffer(3, wgpu::ShaderStages::COMPUTE, Storage { read_only: false }),
                    ],
                });
        // The compute pipeline: simulate.wgsl, run with `main` as entry point.
        let simulate_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("simulate"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("simulate"),
                            bind_group_layouts: &[Some(&simulate_layout)],
                            immediate_size: 0,
                        },
                    )),
                    module: &simulate_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(), // default shader compiler settings
                    cache: None,
                });

        // render.wgsl: @binding(0) globals, (1) params, (2) bodies (read).
        let render_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render"),
                entries: &[
                    buffer(0, wgpu::ShaderStages::VERTEX_FRAGMENT, Uniform),
                    buffer(1, wgpu::ShaderStages::VERTEX_FRAGMENT, Uniform),
                    buffer(
                        2,
                        wgpu::ShaderStages::VERTEX_FRAGMENT,
                        Storage { read_only: true },
                    ),
                ],
            });
        // The render pipeline: render.wgsl plus every fixed setting for drawing with it.
        let render_pipeline =
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    // Name shown in error messages.
                    label: Some("render"),
                    // Which bind groups the shader expects.
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("render"),
                            bind_group_layouts: &[Some(&render_layout)],
                            immediate_size: 0,
                        },
                    )),
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
                    // The depth test: only keep a fragment if it is closer than what was drawn there.
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: ctx.depth_format,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    // Multisampling: off (1 sample per pixel).
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
        // Step 4: Create a Bind Group (two per pipeline: one per ping-pong direction)
        // ---------------------------------------------------------------

        // Two of each, one per ping-pong direction.
        let simulate_bind_groups = [0, 1].map(|i| {
            bind_group(
                &ctx.device,
                &simulate_layout,
                &[
                    (0, &ctx.globals_buffer),
                    (1, &params_buffer),
                    (2, &bodies[i]),
                    (3, &bodies[1 - i]),
                ],
            )
        });
        let render_bind_groups = [0, 1].map(|i| {
            bind_group(
                &ctx.device,
                &render_layout,
                &[
                    (0, &ctx.globals_buffer),
                    (1, &params_buffer),
                    (2, &bodies[i]),
                ],
            )
        });

        ctx.camera = Camera::orbit(glam::Vec3::ZERO, 4.0);

        Self {
            params,
            params_buffer,
            count,
            current: 0,
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
    // Step 5: Dispatch the simulation shader!
    // ---------------------------------------------------------------
    fn update(&mut self, _ctx: &mut Context, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.simulate_pipeline);
        pass.set_bind_group(0, &self.simulate_bind_groups[self.current], &[]);
        // One thread per body, 64 threads per workgroup (see @workgroup_size in simulate.wgsl).
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);

        // The buffer we just wrote is now the current one.
        self.current = 1 - self.current;
    }

    // ---------------------------------------------------------------
    // Step 5: Dispatch the render shader!
    // ---------------------------------------------------------------
    fn render(
        &mut self,
        ctx: &Context,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) {
        ctx.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

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
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0), // 1.0 = as far away as possible
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_bind_groups[self.current], &[]);
        // 6 vertices (one quad) for each body.
        pass.draw(0..6, 0..self.count);
    }

    fn ui(&mut self, _ctx: &mut Context, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.label(format!("{} bodies", self.count));
        ui.add(egui::Slider::new(&mut p.point_scale, 0.1..=5.0).text("point size"));
        ui.add(egui::Slider::new(&mut p.gravity, 0.0..=3.0).text("gravity"));
        ui.add(egui::Slider::new(&mut p.central_mass, 0.0..=5.0).text("central mass"));
        ui.add(
            egui::Slider::new(&mut p.softening, 0.005..=0.3)
                .logarithmic(true)
                .text("softening"),
        );
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
    webgpu_workshop::run::<Gravity>();
}
