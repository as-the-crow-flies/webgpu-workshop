//! Time series: a float32 array of shape (series, samples), drawn as anti-aliased lines.
//!
//!   cargo run --release --example timeseries
//!
//! Scroll to zoom, drag to pan. Hover a line to read its values.
//!
//! Pipelines:
//!   render.wgsl  one quad per line segment

use webgpu_workshop::{bytemuck, egui, glam, load_npy, wgpu, App, Camera, Context};
use wgpu::util::DeviceExt;

/// Must match `LANE` and `sample_position` in types.wgsl (used for the hover readout).
const LANE: f32 = 0.5;

/// The sliders. Must have the same layout as `struct Params` in types.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    series: u32,
    samples: u32,
    line_width: f32,
    amplitude: f32,
    hovered: i32,
}

pub struct Timeseries {
    /// The slider values on the CPU.
    params: Params,
    /// The same values on the GPU (`var<uniform> params`).
    params_buffer: wgpu::Buffer,
    /// A copy of the data on the CPU, for the hover readout.
    values: Vec<f32>,
    /// The text shown under the sliders: the value under the mouse.
    hover_text: String,
    /// The compiled render.wgsl plus all fixed drawing settings.
    pipeline: wgpu::RenderPipeline,
    /// The buffers the pipeline reads.
    bind_group: wgpu::BindGroup,
}

impl App for Timeseries {
    async fn new(ctx: &mut Context) -> Self {
        // ---------------------------------------------------------------
        // Step 1: Create Buffers and Textures
        // ---------------------------------------------------------------

        // A float32 array of shape (series, samples); every float is one `Sample`.
        let file = load_npy("examples/timeseries/timeseries.npy").await;
        // The two dimensions of the array.
        let (series, samples) = (file.shape[0] as u32, file.shape[1] as u32);

        // The samples on the GPU: the bytes of the file, uploaded as they are.
        let samples_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("samples"),
                contents: &file.bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        // The slider values, and a uniform buffer to send them to the GPU every frame.
        let params = Params {
            series,
            samples,
            line_width: 1.5,
            amplitude: 0.2,
            hovered: -1,
        };
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

        // The shaders are separate .wgsl files in this folder. `cargo build` checks
        // them and points at the file and line of any error.
        let shader = ctx.shader("examples/timeseries/render.wgsl");

        // ---------------------------------------------------------------
        // Step 3: Compile a Shader Pipeline
        // ---------------------------------------------------------------

        // render.wgsl: @binding(0) globals, (1) params, (2) samples (read).
        use wgpu::BufferBindingType::{Storage, Uniform};
        use wgpu::ShaderStages as S;
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("render"),
                    entries: &[
                        buffer(0, S::VERTEX_FRAGMENT, Uniform),
                        buffer(1, S::VERTEX_FRAGMENT, Uniform),
                        buffer(2, S::VERTEX, Storage { read_only: true }),
                    ],
                });
        // The pipeline layout: the bind group layouts, one per `@group(n)`.
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        // The render pipeline: render.wgsl plus every fixed setting for drawing with it.
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
                // No depth test: 2D, later draws end up on top.
                depth_stencil: None,
                // Multisampling: off (1 sample per pixel); edges are smoothed in the shader.
                multisample: Default::default(),
                // The fragment stage: runs once per covered pixel, returns its color.
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.surface_format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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
                    binding: 0, // globals
                    resource: ctx.globals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1, // params
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2, // samples
                    resource: samples_buffer.as_entire_binding(),
                },
            ],
        });

        ctx.camera = Camera::view_2d(glam::Vec2::ZERO, series as f32 * LANE + 0.4);

        let values: Vec<f32> = bytemuck::pod_collect_to_vec(&file.bytes);
        Self {
            params,
            params_buffer,
            values,
            hover_text: String::new(),
            pipeline,
            bind_group,
        }
    }

    // ---------------------------------------------------------------
    // Step 5: Dispatch the shader!
    // ---------------------------------------------------------------
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

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        // Line segments: one fewer than samples, per series.
        let segments = self.params.series * (self.params.samples - 1);
        pass.draw(0..6, 0..segments); // one quad per segment
    }

    fn ui(&mut self, ctx: &mut Context, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.label(format!("{} series x {} samples", p.series, p.samples));
        ui.add(egui::Slider::new(&mut p.line_width, 0.5..=8.0).text("line width (px)"));
        ui.add(egui::Slider::new(&mut p.amplitude, 0.05..=1.0).text("amplitude"));

        // The mouse in world space (the camera is orthographic and looks down -z,
        // so the ray origin's x and y are the world position under the mouse).
        let mouse = ctx.mouse.ray_origin;
        // Sample (t) and series (s) under the mouse; see `sample_position` in types.wgsl.
        let t = ((mouse.x + 2.0) / 4.0 * (p.samples - 1) as f32).round();
        let s = ((0.5 * (p.series - 1) as f32 - mouse.y / LANE).round()) as i32;
        p.hovered = -1;
        self.hover_text.clear();
        if !ctx.mouse.over_ui
            && (0.0..p.samples as f32).contains(&t)
            && (0..p.series as i32).contains(&s)
        {
            p.hovered = s;
            let value = self.values[s as usize * p.samples as usize + t as usize];
            self.hover_text = format!("series {s}, sample {t}: {value:.4}");
        }
        ui.label(&self.hover_text);
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

#[allow(dead_code)]
fn main() {
    webgpu_workshop::run::<Timeseries>();
}
