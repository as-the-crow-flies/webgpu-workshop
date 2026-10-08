# WebGPU Workshop

A small [Rust](https://rust-lang.org/) + [wgpu](https://wgpu.rs) boilerplate for visualizing your own data on the GPU, natively and in the browser.

The boilerplate features a basic [WebGPU](https://en.wikipedia.org/wiki/WebGPU) application template, provides interactivity (camera, mouse, ui), and handles `.npy` file loading.

## Setup

1. Install Rust: <https://rustup.rs>
2. Run an example (initial compilation takes a bit of time):
   ```bash
   cargo run --example <example name>
   ```

## Examples

The following examples can be found in webgpu-workshop/examples

| Name      | Data                         | Shows                                                         |
|--------------|------------------------------|---------------------------------------------------------------|
| `template`   | `points.npy`                 | The starting point: data → buffer → quads (parts 1 and 2)      |
| `gravity` | `bodies.npy`        | The same, plus a ping-pong compute simulation: gravity (part 3) |
| `timeseries` | `timeseries.npy`             | 2D view, anti-aliased lines, hover readout                     |
| `graph`      | `nodes.npy`, `edges.npy`     | Two buffers, force-directed layout, dragging nodes, sphere impostors |
| `boids`      | generated on the GPU         | Flocking simulation, mouse interaction                         |
| `life`       | generated on the GPU         | Game of Life on a grid, painting with the mouse                |
| `volume`     | `particles.npy`              | Splatting particles into a 3D texture, volume rendering        |

Every example folder looks like this:

| File            | What it is                                                                  |
|-----------------|-----------------------------------------------------------------------------|
| `make_data.py`  | (optional) Writes the example's `.npy` files; a reference for your own data |
| `main.rs`       | Buffers, pipelines, bind groups and sliders: plain `wgpu`                   |
| `types.wgsl`    | Your structs, defined once and imported by every shader of the example      |
| `init.wgsl`     | (optional) Compute pipeline that generates data on the GPU                  |
| `simulate.wgsl` | (optional) Compute pipeline that runs every frame: reads `ping`, writes `pong` |
| `render.wgsl`   | A render pipeline: `vs` (vertex) and `fs` (fragment). One file per pipeline |

## Create your own project

Copy an existing template and update shader/data paths.

E.g. copy `examples/template` to `examples/my_project`, rename
`Template` in `main.rs`, change shader/data paths (`"examples/my_project/render.wgsl"`, `"examples/my_project/points.npy"`), and run
`cargo run --release --example my_project`.
All paths are relative to the repository root.

# Workshop

## Part 1: Data → GPU buffer

1. Define a data struct in WGSL
    ```wgsl
    // types.wgsl
    struct Point {
        pos: vec3f,    // bytes 0..12
        size: f32,     // bytes 12..16
        color: vec4f,  // bytes 16..32
    }
    ```

2. Export your data using numpy to match WGSL
    ```python
    import numpy as np
    
    point = np.dtype([("pos", "<f4", 3), ("size", "<f4"), ("color", "<f4", 4)])
    points = np.zeros(1000, dtype=point)
    points["pos"] = np.random.randn(1000, 3)
    points["size"] = 0.01
    points["color"] = [1, 0.5, 0.2, 1]
    np.save("points.npy", points)
    ```

3. Load data in Rust
    ```rust
    // main.rs, in `async fn new`
    let data = load_npy("examples/template/points.npy").await;

    let bytes = data.bytes; // data is an array of bytes
    let count = data.shape[0] as u32; // number of 'Point'
    ```

4. Upload data to the GPU as a Buffer
    ```rust
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("points"), // Used for debugging only
        contents: &data.bytes, // Raw bytes!
        usage: wgpu::BufferUsages::STORAGE, // Buffer type
    });
    ```

Make sure your numpy and WGSL types are always in sync, the GPU will read garbage otherwise, often leading to disappointingly funny results.

### Layout cheat sheet

WGSL lays out structs with alignment rules. numpy doesn't, so you sometimes need padding.

| WGSL type        | Size | Alignment | numpy                       |
|------------------|------|-----------|-----------------------------|
| `f32`, `u32`, `i32` | 4 | 4         | `"<f4"`, `"<u4"`, `"<i4"`   |
| `vec2f`          | 8    | 8         | `("name", "<f4", 2)`        |
| `vec3f`          | 12   | **16** :( | `("name", "<f4", 3)`        |
| `vec4f`          | 16   | 16        | `("name", "<f4", 4)`        |
| `mat4x4f`        | 64   | 16        | `("name", "<f4", (4, 4))` (column-major) |

- A `vec3f` must start at a multiple of 16 bytes. Put a 4-byte field right after it
  (`pos: vec3f, radius: f32`), or use `vec4f`.
- A struct's size is rounded up to its largest alignment. `struct { pos: vec3f }` is 16 bytes,
  so add `("_pad", "<f4")` in numpy.
- numpy defaults to `float64` and `int64`, but WebGPU always takes 32 bit types! Use `.astype(np.float32)` / `np.uint32`.
- These rules hold for all data uploaded to the GPU. I.e., also the `Params` struct that you fill in Rust (`#[repr(C)]`).

## Part 2: Rendering

After defining our data layout and uploading data to the GPU we can start rendering this data by creating a GPU Render Pipeline and writing Shaders. Most examples in this workshop render each object you want to appear on the screen as a **quad**: a rectangle which contains whatever you want to draw. Then you can use the GPU to define what your object looks like.

To actually get the GPU to do this for you, you need quite a bit of setup:
1. **Create Buffers and Textures**: upload data and settings to the gpu
2. **Write Shaders**: to define what to render and/or simulate (or any computation, really!)
3. **Compile a Shader Pipeline**: declare inputs and outputs for your shader
4. **Create a Bind Group**: bind your actual data (i.e. buffers and textures) to your shader
5. **Dispatch the shader!**: Actually run your shader

### 1. Create Buffers and Textures

Besides the points from Part 1, the template has a small `Params` struct holding the slider values.
```rust
// define 
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    point_scale: f32,
}
```

It must match `struct Params` in `types.wgsl`.
```wgsl
struct Params {
    point_scale: f32,
}
```

It is uploaded as a **uniform** buffer (small, read-only
values shared by all threads), with `COPY_DST` so we can overwrite it every frame:
```rust
let params = Params { point_scale: 1.0 };
let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("params"),
    contents: bytemuck::bytes_of(&params),
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
});
```

### 2. Write a Shader

Shaders are programs that run on the GPU. You can write shaders to compute anything, but to render data to the screen we normally use a combination of a **vertex shader** and a **fragment shader**:

- The **vertex shader** (`vs`) runs once per **vertex**. It decides *where* that vertex ends up on screen.
- The **fragment shader** (`fs`) runs once per **fragment** and decides its *color*.

Because objects may overlap, many fragments could land on a single pixel.

Let's look at `examples/template/render.wgsl`

First, other shader files are imported:

```wgsl
#import "types.wgsl"          // struct Point & Params
#import "lib/globals.wgsl"    // struct Globals
```

Then it declares its bindings (i.e. the resources it reads)

```wgsl
@group(0) @binding(0) var<uniform> globals: Globals;           // camera, mouse, time
@group(0) @binding(1) var<uniform> params: Params;             // the sliders
@group(0) @binding(2) var<storage, read> points: array<Point>; // your data
```

The template renders one quad per `Point`: a camera-facing rectangle made from 2 triangles and 6 vertices. In the vertex shader you decide what should happen to each vertex and what data to output. When rendering 1000 points, this function gets executed in parallel 6000 times.

```wgsl
@vertex                                         // Tell GPU: This is a vertex shader
fn vs(                                          // Name of your shader (can be anything)
    @builtin(vertex_index) vertex: u32,         // Index of vertex [0..5]
    @builtin(instance_index) instance: u32)     // Index of instance (i.e. Point) [0..count]
    -> VertexOutput                             // Output of Vertex Shader (define yourself)
{
    let point = points[instance];               // one instance per point
    let uv = quad_corner(vertex);               // vertex 0..5 -> corner of the quad, (-1, -1)..(1, 1)

    // Helper function that transforms 2D quad coordinate to 3D camera facing billboard
    let position = sphere_billboard(point.pos, point.size * params.point_scale, corner);

    return VertexOutput(position, uv, point.color, ..);
}
```

After the vertex shader, the fragment shader is run for each pixel your vertex output lands.
The fragment shader has the vertex output as its input, interpolated for each pixel.
In the fragment shader you can decide the color of the pixel (or discard the pixel)

```wgsl
@fragment
fn fs(in: VertexOutput) -> FragmentOutput {
    // Throw away the corners of the billboard to get a disk
    if (length(in.uv) > 1.0) { discard; }

    // Write color (in.color) and depth (in.position.z) to current pixel
    // When rendering in 3D the object with the lowest depth will be visible.
    return FragmentOutput(in.color, in.position.z);
}
```

After writing the shader, you can load it in Rust using:

```rust
let shader = ctx.shader("examples/template/render.wgsl");
```

#### 3. Compile a Shader Pipeline

Before the shader can be compiled, we need to declare the type of resources that will be bound to the shader in Rust. This part is a bit tedious, so please bear with me.

First, create a **Bind Group Layout**, which corresponds with one `@group(...)` in the shader.

```rust
let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("render"),
    entries: &[
        // binding 0: globals
        // binding 1: params
        // ...
        // @binding(2) var<storage, read> points
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ],
});
```

You can then attach multiple Bind Group Layouts to a shader, using a **Pipeline Layout**. We only use one Bind Group Layout, so it becomes trivial.

```rust
let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("render"),
    bind_group_layouts: &[Some(&bind_group_layout)],
    immediate_size: 0,
});
```

Then finally, we can create the **Pipeline** itself, which represents a compiled and runnable shader.
For a **Render Pipeline** (i.e. vertex/fragment combo), this needs quite a bit of settings.

```rust
let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    label: Some("render"),
    layout: Some(&pipeline_layout),
    vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs"),
        buffers: &[], // no vertex buffers: vs reads the storage buffer instead
        compilation_options: Default::default(),
    },
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
    // ... plus primitive (triangles), depth_stencil (depth test), multisample, etc.
});
```

Unlike Rust, shaders get compiled on the users machine at runtime. To know whether your shader and pipeline configuration are valid, you'll have to run the code up to this point. wgpu will complain when any of its validation fails! :)

#### 4. Create a Bind Group

Only one final step before we can actually run the shader, binding actual data to the shader program. We have defined the data layout using a **Bind Group Layout**, but haven't actually passed the data itself to it. This is done using a **Bind Group**, which should correspond with a **Bind Group Layout**.

```rust
let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("render"),
    layout: &bind_group_layout,
    entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: ctx.globals_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: points.as_entire_binding() },
    ],
});
```

#### 5. Dispatch the shader!

All steps up to now only run once, when the application starts. At render time, we use the **RenderPipeline** (our compiled shader) and our **Bind Group** (a reference to the data we've uploaded) to render a new image every frame.

At render time, we first update our parameters, which might have changed by the user via the UI.

```rust
ctx.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
```

Then, we begin a render pass, which declares to which output texture(s) to write color and depth to.

```rust
let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
    color_attachments: &[/* `color`, cleared to black */],
    depth_stencil_attachment: /* `depth`, cleared to 1.0 (far away) */,
    ..Default::default()
});
```

We set the render pipeline to the object we just created:

```rust
pass.set_pipeline(&self.pipeline);
```

We attach our **Bind Group** to group 0

```rust
pass.set_bind_group(0, &self.bind_group, &[]);
```

And, finally, we run the shader!!

(This draws 6 vertices (= one quad) for every instance (= one per point))

```rust
pass.draw(0..6, 0..self.count); // vertices 0..6, instances 0..count
```

## Part 3: simulation (stretch goal)

### Compute shaders

So far the GPU only drew things. A **compute shader** is a GPU program that can compute whatever you want.
Compute shaders read and write arbitrary buffers (or textures), which makes it a good fit for simulations or any general (parallel) computation.
You decide how many threads run, and each thread uses its id to pick which element to work on:

```wgsl
@compute @workgroup_size(64)          // threads are launched in groups of 64
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;                     // this thread's index: one thread per body
    if (i >= arrayLength(&ping)) {
        return;                       // the last group may stick out past the end
    }
    var body = ping[i];
    // ... update body ...
    pong[i] = body;
}
```

In Rust you launch enough groups to cover all elements, from a compute pass in `update`:

```rust
let mut pass = encoder.begin_compute_pass(&Default::default());
pass.set_pipeline(&self.simulate_pipeline);
pass.set_bind_group(0, &self.simulate_bind_groups[self.current], &[]);
pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1); // 64 = @workgroup_size
```

All threads run at the same time, in no particular order. If a thread reads its neighbours from the same buffer it is writing to, it might see some neighbours that were already updated and some that were not. That's why the simulation reads from one buffer (`ping`) and writes to another (`pong`), then swaps them for the next frame.

### Ping-pong

Ping-pong uses two buffers, A and B, that take turns. Each frame, the simulation reads the current
state from one buffer and writes the next state into the other. Then the two buffers swap roles,
like a ball going back and forth:

```
frame 0:  A ──simulate──▶ B     draw B
frame 1:  B ──simulate──▶ A     draw A
frame 2:  A ──simulate──▶ B     draw B
...
```

In the shader the two buffers are simply called `ping` (read) and `pong` (write):

```wgsl
// simulate.wgsl
@group(0) @binding(2) var<storage, read> ping: array<Body>;        // the current state
@group(0) @binding(3) var<storage, read_write> pong: array<Body>;  // the next state
```

The shader never knows which buffer is A and which is B. Instead, `examples/gravity/main.rs` creates
**two bind groups**, one ping, one pong, and picks one each frame:

```rust
// Step 1: the same data uploaded twice
let bodies = ["bodies A", "bodies B"].map(|label| {
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &data.bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
});

// Step 4: bind group i reads buffer i and writes buffer 1 - i
let simulate_bind_groups = [0, 1].map(|i| {
    bind_group(&ctx.device, &simulate_layout, &[
        (0, &ctx.globals_buffer),
        (1, &params_buffer),
        (2, &bodies[i]),      // ping
        (3, &bodies[1 - i]),  // pong
    ])
});

// Step 5, in `update`: simulate, then swap buffers
pass.set_bind_group(0, &self.simulate_bind_groups[self.current], &[]);
pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
self.current = 1 - self.current; // the buffer we just wrote is now the current one
```

The render pipeline gets two bind groups in the same way (`render_bind_groups[i]` draws buffer `i`),
so `render` always draws `render_bind_groups[self.current]`: the state that was just computed.

Two things to remember:

- Write **every** element of `pong`, even the ones that don't change. Otherwise those elements keep
  their value from two frames ago.
- Both buffers must start with valid data. `gravity` uploads the `.npy` file twice. Examples that generate their data on the GPU (`boids`, `life`) let `init.wgsl` write buffer A, then copy it to B with `encoder.copy_buffer_to_buffer`.

## Shader library (`shaders/lib`)

Import with `#import "lib/<file>.wgsl"`.

| File             | Contents                                                                 |
|------------------|--------------------------------------------------------------------------|
| `globals.wgsl`   | `struct Globals` (camera matrices, time, dt, mouse, resolution), `mouse_down`, `mouse_on_plane` |
| `math.wgsl`      | `PI`, `TAU`, `remap`, `rotate2`, `quad_corner`, `fullscreen_triangle`     |
| `camera.wgsl`    | `billboard`, `sphere_billboard`, `Ray`, `camera_ray`, `mouse_ray`, `depth_of` |
| `random.wgsl`    | `pcg`, `rand`, `rand2`, `rand3`, `rand_direction`                         |
| `sdf2d.wgsl`     | `sd_circle`, `sd_box`, `sd_round_box`, `sd_segment`, `sd_triangle`, `sd_ring`, `smooth_min` |
| `sdf3d.wgsl`     | `sd_sphere`, `sd_box3`, `sd_round_box3`, `sd_capsule`, `sd_torus`, `sd_cylinder`, `sd_plane`, `smooth_min3` |
| `intersect.wgsl` | `ray_sphere`, `ray_box`, `ray_plane`, `ray_disk`, `ray_capsule`, `ray_cylinder`, `ray_ellipsoid`, `ray_triangle`, normals, `ray_point_distance` |
| | Billboards (a quad covering the shape on screen): `box_billboard`, `disk_billboard`, `capsule_billboard`, `cylinder_billboard`, `ellipsoid_billboard`, `triangle_billboard` |
| `aa.wgsl`        | `aa_fill`, `aa_stroke`, `aa_fill_px`, `over`                              |
| `colormap.wgsl`  | `viridis`, `magma`, `plasma`, `coolwarm`, `category_color`                |
| `volume.wgsl`    | `composite` (front-to-back), `step_alpha`                                 |
| `sh.wgsl`        | `sh_basis`: real symmetric spherical harmonics up to lmax 8, MRtrix3 convention (for FODs) |

The SDFs and intersectors follow Inigo Quilez's articles: <https://iquilezles.org/articles/>.

## Shader errors

Every `.wgsl` file is checked when you run `cargo build`. Errors point at the original file and line:

```
error: invalid field accessor `colour`
  --> examples/template/render.wgsl:37:20
   |
37 |     let color = in.colour.rgb;
   |                    ^^^^^^ invalid accessor
```

## Running in the browser

The web build contains all examples in one `.wasm` file. You need
[wasm-pack](https://rustwasm.github.io/wasm-pack/) and a browser with WebGPU (recent Chrome,
Edge or Safari).

```bash
wasm-pack build --target web --release --out-dir pkg
```

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/?example=graph>. Files passed to `load_npy` are downloaded from
the same server, so keep them inside the repository. The web build uses the shaders from build
time; rebuild to see shader changes.
