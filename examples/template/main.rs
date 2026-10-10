//! The template: copy this folder to start your own project.
//!
//!   cargo run --release --example template
//!
//! Every example follows the same five steps:
//!   1. Create Buffers and Textures   (`new`; the data layout is in types.wgsl)
//!   2. Write a Shader                (render.wgsl)
//!   3. Compile a Shader Pipeline     (`new`)
//!   4. Create a Bind Group           (`new`)
//!   5. Dispatch the shader!          (`render`; and `update` for compute shaders)

use webgpu_workshop::{bytemuck, egui, glam, load_npy, wgpu, App, Camera, Context};
use wgpu::util::DeviceExt;

/// The sliders. Must have the same layout as `struct Params` in types.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    point_scale: f32,
}

pub struct Template {
    /// The slider values on the CPU.
    params: Params,
    /// The same values on the GPU (`var<uniform> params`).
    params_buffer: wgpu::Buffer,
    /// Number of points, i.e. instances to draw.
    count: u32,
    /// The compiled render.wgsl plus all fixed drawing settings.
    pipeline: wgpu::RenderPipeline,
    /// The buffers the pipeline reads.
    bind_group: wgpu::BindGroup,
}

impl App for Template {
    async fn new(ctx: &mut Context) -> Self {
        // ---------------------------------------------------------------
        // Step 1: Create Buffers and Textures
        // ---------------------------------------------------------------

        // The data: one row of the file must be exactly one `Point` (see types.wgsl and make_data.py).
        let data = load_npy("examples/template/points.npy").await;
        // Number of points (rows in the file).
        let count = data.shape[0] as u32;

        // The points on the GPU: the bytes of the file, uploaded as they are.
        let points = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("points"),
                contents: &data.bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        // The slider values, and a uniform buffer to send them to the GPU every frame.
        let params = Params { point_scale: 1.0 };
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // ---------------------------------------------------------------
        // Step 2: Write a Shader (render.wgsl, which imports types.wgsl)
        // ---------------------------------------------------------------

        // The shaders are separate .wgsl files in this folder.
        let shader = ctx.shader("examples/template/render.wgsl");

        // ---------------------------------------------------------------
        // Step 3: Compile a Shader Pipeline
        // ---------------------------------------------------------------

        // The bind group layout: which kinds of resources render.wgsl sees at
        // each `@group(0) @binding(n)`. Binding 0 is always `globals`.
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("render"),
                    entries: &[
                        // @binding(0) var<uniform> globals (camera, mouse, time)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // @binding(1) var<uniform> params
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // @binding(2) var<storage, read> points
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // The pipeline layout: the bind group layouts, one per `@group(n)` (we only use group 0).
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0, // no immediate (push constant) data
            });

        // The render pipeline: the shader plus every fixed setting for drawing with it.
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                // Name shown in error messages.
                label: Some("render"),
                // Which bind groups the shader expects.
                layout: Some(&pipeline_layout),
                // The vertex stage: runs once per vertex, places it on the screen.
                vertex: wgpu::VertexState {
                    module: &shader,
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
                    module: &shader,
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
        // Step 4: Create a Bind Group (your inputs and outputs)
        // ---------------------------------------------------------------

        // The bind group: which buffer goes into which `@binding` of render.wgsl.
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0, // var<uniform> globals (camera, mouse, time)
                    resource: ctx.globals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1, // var<uniform> params
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2, // var<storage, read> points
                    resource: points.as_entire_binding(),
                },
            ],
        });

        // The camera starts looking at the origin from a distance of 3.
        ctx.camera = Camera::orbit(glam::Vec3::ZERO, 3.0);

        Self {
            params,
            params_buffer,
            count,
            pipeline,
            bind_group,
        }
    }

    // Keep the slider values when a shader is hot reloaded.
    fn reloaded(&mut self, old: &Self) {
        self.params = old.params;
    }

    // ---------------------------------------------------------------
    // Step 5: Dispatch the shader!
    // ---------------------------------------------------------------
    fn render(
        &mut self,
        ctx: &Context,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) {
        // Send the params to the GPU.
        ctx.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

        // A render pass: draw into the window (`color`), testing against `depth`.
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

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..self.count); // 6 vertices (one quad) for each of the `count` points.
    }

    fn ui(&mut self, _ctx: &mut Context, ui: &mut egui::Ui) {
        ui.label(format!("{} points", self.count));
        ui.add(egui::Slider::new(&mut self.params.point_scale, 0.1..=5.0).text("point size"));
    }
}

#[allow(dead_code)]
fn main() {
    webgpu_workshop::run::<Template>();
}
