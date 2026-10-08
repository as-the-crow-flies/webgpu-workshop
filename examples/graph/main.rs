//! A force-directed graph: nodes and edges from two .npy files.
//!
//!   cargo run --release --example graph
//!
//! Drag nodes with the left mouse button; orbit with the right button.
//!
//! Pipelines:
//!   pick.wgsl          finds the node under the mouse when the left button goes down
//!   simulate.wgsl      forces (ping-pong), and moves the picked node to the mouse
//!   render_edges.wgsl  one anti-aliased line per edge
//!   render_nodes.wgsl  one sphere impostor per node

use webgpu_workshop::{bytemuck, egui, glam, load_npy, wgpu, App, Camera, Context};
use wgpu::util::DeviceExt;

const NO_PICK: u32 = u32::MAX;

/// The sliders. Must have the same layout as `struct Params` in types.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    repulsion: f32,
    spring_length: f32,
    spring: f32,
    damping: f32,
    gravity: f32,
    line_width: f32,
    flat: u32,
}

pub struct Graph {
    /// The slider values on the CPU.
    params: Params,
    /// The same values on the GPU (`var<uniform> params`).
    params_buffer: wgpu::Buffer,
    /// Number of nodes and edges.
    node_count: u32,
    edge_count: u32,
    /// Holds `NO_PICK` or the node under the mouse (written by pick.wgsl).
    pick_buffer: wgpu::Buffer,
    /// Which of the two node buffers holds the latest positions (0 or 1).
    current: usize,

    /// One pipeline per shader, and its bind groups (two for ping-pong).
    pick_pipeline: wgpu::ComputePipeline,
    pick_bind_groups: [wgpu::BindGroup; 2],
    simulate_pipeline: wgpu::ComputePipeline,
    simulate_bind_groups: [wgpu::BindGroup; 2],
    edges_pipeline: wgpu::RenderPipeline,
    edges_bind_groups: [wgpu::BindGroup; 2],
    nodes_pipeline: wgpu::RenderPipeline,
    nodes_bind_groups: [wgpu::BindGroup; 2],
}

impl App for Graph {
    async fn new(ctx: &mut Context) -> Self {
        // ---------------------------------------------------------------
        // Step 1: Create Buffers and Textures
        // ---------------------------------------------------------------

        // Two files: one row per `Node` and per `Edge` (see types.wgsl and make_data.py).
        let node_data = load_npy("examples/graph/nodes.npy").await;
        let edge_data = load_npy("examples/graph/edges.npy").await;
        let node_count = node_data.shape[0] as u32; // pick.wgsl supports at most 65536 nodes
                                                    // Number of edges (rows in the file).
        let edge_count = edge_data.shape[0] as u32;

        // The nodes are uploaded twice (ping-pong); the edges don't change.
        let nodes = ["nodes A", "nodes B"].map(|label| {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &node_data.bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        });
        // The edges on the GPU: they never change, so one buffer is enough.
        let edges = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("edges"),
                contents: &edge_data.bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        // One u32 for the node under the mouse; COPY_DST so we can reset it.
        let pick_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pick"),
                contents: bytemuck::bytes_of(&NO_PICK),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // The slider values, and a uniform buffer to send them to the GPU every frame.
        let params = Params {
            repulsion: 0.002,
            spring_length: 0.15,
            spring: 2.0,
            damping: 0.9,
            gravity: 0.3,
            line_width: 1.5,
            flat: 0,
        };
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // ---------------------------------------------------------------
        // Step 2: Write a Shader (pick.wgsl, simulate.wgsl, render_edges.wgsl, render_nodes.wgsl)
        // ---------------------------------------------------------------

        // The shaders are separate .wgsl files in this folder. `cargo build` checks
        // them and points at the file and line of any error.
        let pick_shader = ctx.shader("examples/graph/pick.wgsl");
        let simulate_shader = ctx.shader("examples/graph/simulate.wgsl");
        let edges_shader = ctx.shader("examples/graph/render_edges.wgsl");
        let nodes_shader = ctx.shader("examples/graph/render_nodes.wgsl");

        // ---------------------------------------------------------------
        // Step 3: Compile a Shader Pipeline
        // ---------------------------------------------------------------

        // Bind group layouts: what each shader sees at `@group(0) @binding(n)`.
        // Binding 0 is always `globals`.
        use wgpu::BufferBindingType::{Storage, Uniform};
        use wgpu::ShaderStages as S;
        // Shorthands for the two kinds of storage buffer.
        let read = Storage { read_only: true };
        let write = Storage { read_only: false };
        // A bind group layout from a list of entries.
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries,
                })
        };
        // pick.wgsl: globals, nodes, pick (atomic)
        let pick_layout = layout(
            "pick",
            &[
                buffer(0, S::COMPUTE, Uniform),
                buffer(1, S::COMPUTE, read),
                buffer(2, S::COMPUTE, write),
            ],
        );
        // simulate.wgsl: globals, params, ping, pong, edges, pick
        let simulate_layout = layout(
            "simulate",
            &[
                buffer(0, S::COMPUTE, Uniform),
                buffer(1, S::COMPUTE, Uniform),
                buffer(2, S::COMPUTE, read),
                buffer(3, S::COMPUTE, write),
                buffer(4, S::COMPUTE, read),
                buffer(5, S::COMPUTE, read),
            ],
        );
        // render_edges.wgsl: globals, params, nodes, edges
        let edges_layout = layout(
            "edges",
            &[
                buffer(0, S::VERTEX_FRAGMENT, Uniform),
                buffer(1, S::VERTEX_FRAGMENT, Uniform),
                buffer(2, S::VERTEX, read),
                buffer(3, S::VERTEX, read),
            ],
        );
        // render_nodes.wgsl: globals, nodes, pick
        let nodes_layout = layout(
            "nodes",
            &[
                buffer(0, S::VERTEX_FRAGMENT, Uniform),
                buffer(1, S::VERTEX_FRAGMENT, read),
                buffer(2, S::VERTEX_FRAGMENT, read),
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
        // Both render pipelines test depth; only the nodes write it,
        // so edges are hidden behind nodes.
        let render = |label,
                      shader: &wgpu::ShaderModule,
                      layout: &wgpu::BindGroupLayout,
                      depth_write: bool| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    // Name shown in error messages.
                    label: Some(label),
                    // Which bind groups the shader expects.
                    layout: Some(&pipeline_layout(layout)),
                    // The vertex stage: runs once per vertex, places it on the screen.
                    vertex: wgpu::VertexState {
                        module: shader,
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
                        depth_write_enabled: Some(depth_write),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    // Multisampling: off (1 sample per pixel).
                    multisample: Default::default(),
                    // The fragment stage: runs once per covered pixel, returns its color.
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
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
                })
        };
        // The four pipelines.
        let pick_pipeline = compute("pick", &pick_shader, &pick_layout);
        let simulate_pipeline = compute("simulate", &simulate_shader, &simulate_layout);
        let edges_pipeline = render("edges", &edges_shader, &edges_layout, false);
        let nodes_pipeline = render("nodes", &nodes_shader, &nodes_layout, true);

        // ---------------------------------------------------------------
        // Step 4: Create a Bind Group (for each pipeline; two for ping-pong)
        // ---------------------------------------------------------------

        // Index i: nodes[i] holds the current positions.
        let globals = &ctx.globals_buffer;
        let pick_bind_groups = [0, 1].map(|i| {
            bind_group(
                &ctx.device,
                &pick_layout,
                &[(0, globals), (1, &nodes[i]), (2, &pick_buffer)],
            )
        });
        let simulate_bind_groups = [0, 1].map(|i| {
            bind_group(
                &ctx.device,
                &simulate_layout,
                &[
                    (0, globals),
                    (1, &params_buffer),
                    (2, &nodes[i]),
                    (3, &nodes[1 - i]),
                    (4, &edges),
                    (5, &pick_buffer),
                ],
            )
        });
        let edges_bind_groups = [0, 1].map(|i| {
            bind_group(
                &ctx.device,
                &edges_layout,
                &[
                    (0, globals),
                    (1, &params_buffer),
                    (2, &nodes[i]),
                    (3, &edges),
                ],
            )
        });
        let nodes_bind_groups = [0, 1].map(|i| {
            bind_group(
                &ctx.device,
                &nodes_layout,
                &[(0, globals), (1, &nodes[i]), (2, &pick_buffer)],
            )
        });

        // The left mouse button drags nodes, so the camera uses the right button.
        ctx.camera = Camera::orbit(glam::Vec3::ZERO, 3.0);
        ctx.camera.left_drag = false;

        Self {
            params,
            params_buffer,
            node_count,
            edge_count,
            pick_buffer,
            current: 0,
            pick_pipeline,
            pick_bind_groups,
            simulate_pipeline,
            simulate_bind_groups,
            edges_pipeline,
            edges_bind_groups,
            nodes_pipeline,
            nodes_bind_groups,
        }
    }

    // ---------------------------------------------------------------
    // Step 5: Dispatch the shader!
    // ---------------------------------------------------------------
    fn update(&mut self, ctx: &mut Context, encoder: &mut wgpu::CommandEncoder) {
        // One thread per node, 64 threads per workgroup.
        let workgroups = self.node_count.div_ceil(64);

        // Left button pressed: forget the old pick and find the node under the mouse.
        // (`write_buffer` happens before this frame's passes run.)
        if ctx.mouse.left_pressed || ctx.mouse.left_released {
            ctx.queue
                .write_buffer(&self.pick_buffer, 0, bytemuck::bytes_of(&NO_PICK));
        }
        if ctx.mouse.left_pressed {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pick_pipeline);
            pass.set_bind_group(0, &self.pick_bind_groups[self.current], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.simulate_pipeline);
        pass.set_bind_group(0, &self.simulate_bind_groups[self.current], &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        self.current = 1 - self.current;
    }

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
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        // Edges first, then the nodes on top.
        pass.set_pipeline(&self.edges_pipeline);
        pass.set_bind_group(0, &self.edges_bind_groups[self.current], &[]);
        pass.draw(0..6, 0..self.edge_count);

        pass.set_pipeline(&self.nodes_pipeline);
        pass.set_bind_group(0, &self.nodes_bind_groups[self.current], &[]);
        pass.draw(0..6, 0..self.node_count);
    }

    fn ui(&mut self, _ctx: &mut Context, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.label(format!(
            "{} nodes, {} edges",
            self.node_count, self.edge_count
        ));
        ui.add(egui::Slider::new(&mut p.repulsion, 0.0..=0.01).text("repulsion"));
        ui.add(egui::Slider::new(&mut p.spring_length, 0.0..=0.5).text("spring length"));
        ui.add(egui::Slider::new(&mut p.spring, 0.0..=10.0).text("spring"));
        ui.add(egui::Slider::new(&mut p.damping, 0.5..=1.0).text("damping"));
        ui.add(egui::Slider::new(&mut p.gravity, 0.0..=2.0).text("gravity"));
        ui.add(egui::Slider::new(&mut p.line_width, 0.5..=5.0).text("edge width (px)"));
        let mut flat = p.flat == 1;
        ui.checkbox(&mut flat, "2D layout");
        p.flat = flat as u32;
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
    webgpu_workshop::run::<Graph>();
}
