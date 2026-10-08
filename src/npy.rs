//! Loading `.npy` files (with the `npyz` crate).
//!
//! The idea: shape your data in numpy so that one row has exactly the bytes of
//! one WGSL struct. The file's bytes can then be uploaded to a buffer as-is.
//!
//! ```python
//! node = np.dtype([("pos", "<f4", 3), ("radius", "<f4")])   # struct Node { pos: vec3f, radius: f32 }
//! np.save("nodes.npy", np.zeros(100, dtype=node))
//! ```

/// A loaded `.npy` file: its shape and the raw bytes of the array.
pub struct Npy {
    /// The array shape, e.g. `[100]` for 100 structs, or `[8, 2048]`.
    pub shape: Vec<usize>,
    /// The raw bytes of the array, ready for `create_buffer_init`.
    pub bytes: Vec<u8>,
}

/// Load a `.npy` file. Paths are relative to the crate root, e.g.
/// `load_npy("examples/template/points.npy").await`.
///
/// Native: read from disk. Web: fetched from the web server, which serves the crate root.
pub async fn load_npy(path: &str) -> Npy {
    let file = read(path)
        .await
        .unwrap_or_else(|error| panic!("cannot load {path}: {error}"));
    let npy = npyz::NpyFile::new(&file[..])
        .unwrap_or_else(|error| panic!("{path} is not a valid .npy file: {error}"));
    let shape = npy.shape().iter().map(|&n| n as usize).collect();
    // What is left after the header is the data itself.
    let bytes = npy.into_inner().to_vec();
    Npy { shape, bytes }
}

#[cfg(not(target_arch = "wasm32"))]
async fn read(path: &str) -> Result<Vec<u8>, String> {
    // Relative paths are tried from the current directory first, then from the crate root.
    std::fs::read(path)
        .or_else(|_| std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path)))
        .map_err(|e| e.to_string())
}

#[cfg(target_arch = "wasm32")]
async fn read(path: &str) -> Result<Vec<u8>, String> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().unwrap();
    let response: web_sys::Response = JsFuture::from(window.fetch_with_str(path))
        .await
        .map_err(|e| format!("{e:?}"))?
        .dyn_into()
        .unwrap();
    if !response.ok() {
        return Err(format!("HTTP {}", response.status()));
    }
    let buffer = JsFuture::from(response.array_buffer().map_err(|e| format!("{e:?}"))?)
        .await
        .map_err(|e| format!("{e:?}"))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}
