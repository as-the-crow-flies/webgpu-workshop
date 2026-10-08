use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct NativeFile {
    path: PathBuf,
}

impl From<PathBuf> for NativeFile {
    fn from(path: PathBuf) -> Self {
        Self { path }
    }
}

impl egui::DroppedFile for NativeFile {
    fn path(&self) -> &Path {
        &self.path
    }

    // Patched for the workshop: `egui::DroppedFile` requires `bytes` on native and
    // `bytes_async` on wasm32, but egui-winit 0.36 only implements `bytes`, so it does
    // not compile for the web (https://github.com/emilk/egui/issues/7052). winit's web
    // backend never sends dropped files, so the wasm32 version is never called.
    #[cfg(not(target_arch = "wasm32"))]
    fn bytes(&self) -> Result<Vec<u8>, String> {
        std::fs::read(&self.path).map_err(|err| err.to_string())
    }

    #[cfg(target_arch = "wasm32")]
    fn bytes_async(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, String>> + '_>> {
        Box::pin(std::future::ready(Err("dropped files are not supported on the web".to_owned())))
    }
}
