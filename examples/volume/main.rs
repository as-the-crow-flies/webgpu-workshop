//! Particles -> volume: splat particles into a 3D grid and ray-march it.
//!
//!   cargo run --release --example volume
//!
//! Every frame:
//!   clear_buffer   sets the grid to zero
//!   splat.wgsl     adds each particle's weight to its voxel (atomicAdd, fixed point)
//!   resolve.wgsl   copies the grid into an rgba16float 3D texture
//!   render.wgsl    ray-marches the texture (full-screen triangle)
//!
//! Splatting every frame means you can add a simulate.wgsl that moves the
//! particles (see examples/gravity) and the volume follows.

use webgpu_workshop::{bytemuck, egui, glam, load_npy, wgpu, App, Camera, Context};
use wgpu::util::DeviceExt;

/// Voxels along each side of the grid.
const GRID: u32 = 128;
/// The 3D texture format: writable from a compute shader and filterable, on every WebGPU device.
const VOLUME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The sliders. Must have the same layout as `struct Params` in types.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    grid: u32,
    splat: f32,
    density: f32,
    brightness: f32,
}

pub struct Volume {
    /// The slider values on the CPU.
    params: Params,
    /// The same values on the GPU (`var<uniform> params`).
    params_buffer: wgpu::Buffer,
    /// Number of particles.
    particle_count: u32,
    /// The fixed-point grid that splat.wgsl adds to (cleared every frame).
    grid: wgpu::Buffer,

    /// One pipeline per shader, and its bind group.
    splat_pipeline: wgpu::ComputePipeline,
    splat_bind_group: wgpu::BindGroup,
    resolve_pipeline: wgpu::ComputePipeline,
    resolve_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
}

impl App for Volume {
    async fn new(ctx: &mut Context) -> Self {
        // ---------------------------------------------------------------
        // Step 1: Create Buffers and Textures
        // ---------------------------------------------------------------

        // One row per `Particle`, positions in [0, 1]^3 (see types.wgsl and make_data.py).
        let data = load_npy("examples/volume/particles.npy").await;
        // Number of particles (rows in the file).
        let particle_count = data.shape[0] as u32;

        // The particles on the GPU: the bytes of the file, uploaded as they are.
        let particles = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particles"),
                contents: &data.bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        // One u32 per voxel (fixed-point sum of weights). COPY_DST lets us clear it.
        let grid = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid"),
            size: (GRID * GRID * GRID) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Written by resolve.wgsl (STORAGE_BINDING), sampled by render.wgsl (TEXTURE_BINDING).
        let volume = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume"),
            size: wgpu::Extent3d {
                width: GRID,
                height: GRID,
                depth_or_array_layers: GRID,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: VOLUME_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // A view of the texture: how a shader sees it (here: all of it).
        let volume_view = volume.create_view(&Default::default());
        // A sampler: how the texture is read between voxels (linear = trilinear interpolation).
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // The slider values, and a uniform buffer to send them to the GPU every frame.
        let params = Params {
            grid: GRID,
            splat: 1.0,
            density: 50.0,
            brightness: 0.8,
        };
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // ---------------------------------------------------------------
        // Step 2: Write a Shader (splat.wgsl, resolve.wgsl and render.wgsl)
        // ---------------------------------------------------------------

        // The shaders are separate .wgsl files in this folder. `cargo build` checks
        // them and points at the file and line of any error.
        let splat_shader = ctx.shader("examples/volume/splat.wgsl");
        let resolve_shader = ctx.shader("examples/volume/resolve.wgsl");
        let render_shader = ctx.shader("examples/volume/render.wgsl");

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
        // splat.wgsl: globals, params, particles, grid (atomic)
        let splat_layout = layout(
            "splat",
            &[
                buffer(0, S::COMPUTE, Uniform),
                buffer(1, S::COMPUTE, Uniform),
                buffer(2, S::COMPUTE, Storage { read_only: true }),
                buffer(3, S::COMPUTE, Storage { read_only: false }),
            ],
        );
        // resolve.wgsl: globals, params, grid, volume (storage texture)
        let resolve_layout = layout(
            "resolve",
            &[
                buffer(0, S::COMPUTE, Uniform),
                buffer(1, S::COMPUTE, Uniform),
                buffer(2, S::COMPUTE, Storage { read_only: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: S::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: VOLUME_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        );
        // render.wgsl: globals, params, volume (sampled texture), sampler
        let render_layout = layout(
            "render",
            &[
                buffer(0, S::FRAGMENT, Uniform),
                buffer(1, S::FRAGMENT, Uniform),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: S::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: S::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
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
        let splat_pipeline = compute("splat", &splat_shader, &splat_layout);
        let resolve_pipeline = compute("resolve", &resolve_shader, &resolve_layout);

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
        // Step 4: Create a Bind Group (for each pipeline)
        // ---------------------------------------------------------------

        // A bind group from resources in binding order (0, 1, 2, ...).
        let bind_group = |layout: &wgpu::BindGroupLayout, resources: Vec<wgpu::BindingResource>| {
            let entries: Vec<_> = resources
                .into_iter()
                .enumerate()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource,
                })
                .collect();
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &entries,
            })
        };
        // Resources in binding order: 0, 1, 2, 3.
        let splat_bind_group = bind_group(
            &splat_layout,
            vec![
                ctx.globals_buffer.as_entire_binding(),
                params_buffer.as_entire_binding(),
                particles.as_entire_binding(),
                grid.as_entire_binding(),
            ],
        );
        let resolve_bind_group = bind_group(
            &resolve_layout,
            vec![
                ctx.globals_buffer.as_entire_binding(),
                params_buffer.as_entire_binding(),
                grid.as_entire_binding(),
                wgpu::BindingResource::TextureView(&volume_view),
            ],
        );
        let render_bind_group = bind_group(
            &render_layout,
            vec![
                ctx.globals_buffer.as_entire_binding(),
                params_buffer.as_entire_binding(),
                wgpu::BindingResource::TextureView(&volume_view),
                wgpu::BindingResource::Sampler(&sampler),
            ],
        );

        ctx.camera = Camera::orbit(glam::Vec3::splat(0.5), 2.2);

        Self {
            params,
            params_buffer,
            particle_count,
            grid,
            splat_pipeline,
            splat_bind_group,
            resolve_pipeline,
            resolve_bind_group,
            render_pipeline,
            render_bind_group,
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
        ctx.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

        encoder.clear_buffer(&self.grid, 0, None);

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.splat_pipeline);
        pass.set_bind_group(0, &self.splat_bind_group, &[]);
        pass.dispatch_workgroups(self.particle_count.div_ceil(64), 1, 1);

        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, &self.resolve_bind_group, &[]);
        // One thread per voxel, in blocks of 4 x 4 x 4.
        let blocks = GRID.div_ceil(4);
        pass.dispatch_workgroups(blocks, blocks, blocks);
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

        // Only a full-screen triangle: no depth attachment needed.
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
        pass.draw(0..3, 0..1); // one full-screen triangle
    }

    fn ui(&mut self, _ctx: &mut Context, ui: &mut egui::Ui) {
        let p = &mut self.params;
        ui.label(format!(
            "{} particles, {GRID}^3 voxels",
            self.particle_count
        ));
        ui.add(
            egui::Slider::new(&mut p.splat, 0.1..=10.0)
                .logarithmic(true)
                .text("particle weight"),
        );
        ui.add(
            egui::Slider::new(&mut p.density, 1.0..=500.0)
                .logarithmic(true)
                .text("density"),
        );
        ui.add(
            egui::Slider::new(&mut p.brightness, 0.01..=2.0)
                .logarithmic(true)
                .text("brightness"),
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

#[allow(dead_code)]
fn main() {
    webgpu_workshop::run::<Volume>();
}
