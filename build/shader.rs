//! Shader preprocessing for `build.rs`, which checks every shader when you run
//! `cargo build`.
//!
//! Two things happen here:
//! 1. `#import "file.wgsl"` lines are replaced by the contents of that file.
//!    The path is looked up next to the importing file first, then in `shaders/`.
//!    Every file is included at most once.
//! 2. The result is parsed and validated with naga. Errors are reported with
//!    the *original* file and line, not the line in the flattened source.
//!

use std::collections::HashSet;
use std::fmt::Write;
use std::path::Path;

/// A shader with all imports inlined.
pub struct Flat {
    /// The flattened WGSL source.
    pub source: String,
    /// Every file that ended up in `source`, relative to the crate root.
    pub files: Vec<String>,
    /// For every line of `source`: (index into `files`, 1-based line number).
    lines: Vec<(usize, u32)>,
}

/// Inline all imports of `path` (relative to `root`, the crate root).
pub fn flatten(root: &Path, path: &str) -> Result<Flat, String> {
    let mut flat = Flat {
        source: String::new(),
        files: Vec::new(),
        lines: Vec::new(),
    };
    include(root, &normalize(path), &mut flat, &mut HashSet::new(), None)?;
    Ok(flat)
}

/// Flatten and validate in one go.
pub fn compile(root: &Path, path: &str) -> Result<Flat, String> {
    let flat = flatten(root, path)?;
    flat.validate()?;
    Ok(flat)
}

fn include(
    root: &Path,
    path: &str,
    flat: &mut Flat,
    seen: &mut HashSet<String>,
    imported_from: Option<(&str, u32)>,
) -> Result<(), String> {
    if !seen.insert(path.to_string()) {
        return Ok(());
    }

    let text = std::fs::read_to_string(root.join(path)).map_err(|error| match imported_from {
        Some((file, line)) => {
            format!("error: cannot import \"{path}\": {error}\n  --> {file}:{line}")
        }
        None => format!("error: cannot read {path}: {error}"),
    })?;

    let index = flat.files.len();
    flat.files.push(path.to_string());

    for (i, line) in text.lines().enumerate() {
        let number = i as u32 + 1;

        if let Some(rest) = line.trim_start().strip_prefix("#import") {
            let name = rest.trim().trim_matches('"');
            let resolved = resolve(root, path, name).ok_or_else(|| {
                format!(
                    "error: cannot find import \"{name}\"\n  --> {path}:{number}\n   = note: looked next to {path} and in shaders/"
                )
            })?;
            include(root, &resolved, flat, seen, Some((path, number)))?;
            // Keep the import line (as a blank line) so line numbers stay in sync.
            flat.source.push('\n');
        } else {
            flat.source.push_str(line);
            flat.source.push('\n');
        }
        flat.lines.push((index, number));
    }

    Ok(())
}

/// Find an import: next to the importing file first, then in `shaders/`.
fn resolve(root: &Path, from: &str, name: &str) -> Option<String> {
    let dir = Path::new(from).parent().unwrap_or(Path::new(""));
    [dir.join(name), Path::new("shaders").join(name)]
        .into_iter()
        .map(|candidate| normalize(&candidate.to_string_lossy()))
        .find(|candidate| root.join(candidate).is_file())
}

/// `examples/a/../b/./c.wgsl` -> `examples/b/c.wgsl`, always with `/`.
fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            part => parts.push(part),
        }
    }
    parts.join("/")
}

impl Flat {
    /// Parse and validate with naga.
    pub fn validate(&self) -> Result<(), String> {
        let module = naga::front::wgsl::parse_str(&self.source).map_err(|error| {
            self.report(
                error.message(),
                error
                    .labels()
                    .map(|(span, label)| (span, label.to_string())),
                error.notes().map(str::to_string),
            )
        })?;

        let flags = naga::valid::ValidationFlags::all();
        let capabilities = naga::valid::Capabilities::default();
        naga::valid::Validator::new(flags, capabilities)
            .validate(&module)
            .map_err(|error| {
                let mut notes = Vec::new();
                let mut source = std::error::Error::source(error.as_inner());
                while let Some(next) = source {
                    notes.push(next.to_string());
                    source = next.source();
                }
                self.report(
                    &error.as_inner().to_string(),
                    error.spans().cloned(),
                    notes.into_iter(),
                )
            })?;

        Ok(())
    }

    /// Format an error like rustc does, pointing at the original file and line.
    fn report(
        &self,
        message: &str,
        labels: impl Iterator<Item = (naga::Span, String)>,
        notes: impl Iterator<Item = String>,
    ) -> String {
        let mut out = format!("error: {message}\n");
        let mut any_label = false;

        for (span, label) in labels {
            let Some(range) = span.to_range() else {
                continue;
            };
            any_label = true;

            // Which flattened line is this, and where does it start?
            let line_index = self.source[..range.start].matches('\n').count();
            let line_start = self.source[..range.start].rfind('\n').map_or(0, |i| i + 1);
            let line_text = self.source[line_start..].lines().next().unwrap_or("");
            let column = range.start - line_start;
            let width =
                (range.end - range.start).clamp(1, line_text.len().saturating_sub(column).max(1));

            let (file, line) = self.lines.get(line_index).copied().unwrap_or((0, 0));
            let file = &self.files[file];
            let gutter = " ".repeat(line.to_string().len());

            let _ = writeln!(out, "{gutter}--> {file}:{line}:{}", column + 1);
            let _ = writeln!(out, "{gutter} |");
            let _ = writeln!(out, "{line} | {line_text}");
            let _ = writeln!(
                out,
                "{gutter} | {}{} {label}",
                " ".repeat(column),
                "^".repeat(width)
            );
        }

        if !any_label {
            let _ = writeln!(out, "  --> {}", self.files[0]);
        }
        for note in notes {
            let _ = writeln!(out, "   = note: {note}");
        }
        out
    }
}
