//! The Mandelbrot set, computed per pixel in a fragment shader (no .npy file).
//!
//!   cargo run --release --example mandelbrot
//!
//! Drag to pan, scroll to zoom.
//!
//! Pipelines:
//!   render.wgsl  one full-screen quad; the fragment shader iterates z = z^2 + c for each pixel

use webgpu_workshop::{bytemuck, egui, glam, wgpu, App, Camera, Context};
use wgpu::util::DeviceExt;

/// The sliders. Must have the same layout as `struct Params` in types.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    iterations: u32,
}

pub struct Mandelbrot {
    /// The slider values on the CPU.
    params: Params,
    /// The same values on the GPU (`var<uniform> params`).
    params_buffer: wgpu::Buffer,

    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
}

impl App for Mandelbrot {
    async fn new(ctx: &mut Context) -> Self {
        // ---------------------------------------------------------------
        // Step 1: Create Buffers and Textures
        // ---------------------------------------------------------------

        // No data: the shader computes every pixel. Only the slider values.
        let params = Params { iterations: 200 };
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // ---------------------------------------------------------------
        // Step 2: Write a Shader (render.wgsl)
        // ---------------------------------------------------------------

        let render_shader = ctx.shader("examples/mandelbrot/render.wgsl");

        // ---------------------------------------------------------------
        // Step 3: Compile a Shader Pipeline
        // ---------------------------------------------------------------

        // What the shader sees at `@group(0) @binding(n)`: globals and params.
        let render_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render"),
                entries: &[uniform(0), uniform(1)],
            });

        let render_pipeline =
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("render"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[Some(&render_layout)],
                            immediate_size: 0,
                        },
                    )),
                    vertex: wgpu::VertexState {
                        module: &render_shader,
                        entry_point: Some("vs"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    // Triangles, no culling. (All settings are explained in examples/life/main.rs.)
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &render_shader,
                        entry_point: Some("fs"),
                        targets: &[Some(ctx.surface_format.into())],
                        compilation_options: Default::default(),
                    }),
                    multiview_mask: None,
                    cache: None,
                });

        // ---------------------------------------------------------------
        // Step 4: Create a Bind Group
        // ---------------------------------------------------------------

        let render_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render"),
            layout: &render_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.globals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // A 2D view of the whole set.
        ctx.camera = Camera::view_2d(glam::vec2(-0.5, 0.0), 3.0);

        Self {
            params,
            params_buffer,
            render_pipeline,
            render_bind_group,
        }
    }

    // Keep the slider values when a shader is hot reloaded.
    fn reloaded(&mut self, old: &Self) {
        self.params = old.params;
    }

    // ---------------------------------------------------------------
    // Step 5: Draw!
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
        pass.set_bind_group(0, &self.render_bind_group, &[]);
        pass.draw(0..6, 0..1); // one full-screen quad
    }

    fn ui(&mut self, _ctx: &mut Context, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.params.iterations, 10..=2000).text("iterations"));
    }
}

/// A bind group layout entry for a uniform buffer, visible to both shader stages.
fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[allow(dead_code)]
fn main() {
    webgpu_workshop::run::<Mandelbrot>();
}
