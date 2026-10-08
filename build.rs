//! Runs before the crate is compiled:
//! - checks every `.wgsl` file in `examples/` (with its imports from `shaders/`);
//!   a broken shader fails the build with the file and line of the error;
//! - embeds the shaders, with their imports resolved (see `Context::shader`).

#[path = "build/shader.rs"]
mod shader;

use std::fmt::Write;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo::rerun-if-changed=examples");
    println!("cargo::rerun-if-changed=shaders");

    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut files = Vec::new();
    // The example shaders; the files in shaders/lib are checked as part of
    // every shader that imports them.
    collect(&root, Path::new("examples"), &mut files);
    files.sort();

    let mut shaders = String::new();
    let mut errors = Vec::new();

    for path in &files {
        match shader::compile(&root, path) {
            Ok(flat) => {
                let _ = writeln!(shaders, "    ({path:?}, {:?}),", flat.source);
            }
            Err(error) => errors.push(error),
        }
    }

    if !errors.is_empty() {
        eprintln!("\n{}", errors.join("\n"));
        eprintln!("{} shader(s) failed to compile", errors.len());
        std::process::exit(1);
    }

    let generated = format!("pub(crate) static SHADERS: &[(&str, &str)] = &[\n{shaders}];\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("shaders.rs");
    std::fs::write(out, generated).unwrap();
}

/// All `.wgsl` files below `dir`, relative to `root`.
fn collect(root: &Path, dir: &Path, files: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(root.join(dir)) else {
        return;
    };
    for entry in entries.flatten() {
        let path = dir.join(entry.file_name());
        if entry.path().is_dir() {
            collect(root, &path, files);
        } else if path.extension().is_some_and(|ext| ext == "wgsl") {
            files.push(path.to_string_lossy().replace('\\', "/"));
        }
    }
}
