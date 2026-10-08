//! The web build. All examples are compiled into one `.wasm` file; the URL
//! picks one: `index.html?example=boids`.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

// The examples, included as modules of this crate.
#[allow(dead_code)]
#[path = "../examples/boids/main.rs"]
mod boids;
#[allow(dead_code)]
#[path = "../examples/graph/main.rs"]
mod graph;
#[allow(dead_code)]
#[path = "../examples/gravity/main.rs"]
mod gravity;
#[allow(dead_code)]
#[path = "../examples/life/main.rs"]
mod life;
#[allow(dead_code)]
#[path = "../examples/template/main.rs"]
mod template;
#[allow(dead_code)]
#[path = "../examples/timeseries/main.rs"]
mod timeseries;
#[allow(dead_code)]
#[path = "../examples/volume/main.rs"]
mod volume;

#[wasm_bindgen(start)]
pub fn start() {
    console_log::init_with_level(log::Level::Warn).ok();
    std::panic::set_hook(Box::new(|info| {
        console_error_panic_hook::hook(info);
        show_message(&info.to_string());
    }));

    match query("example").as_deref().unwrap_or("template") {
        "template" => crate::run::<template::Template>(),
        "gravity" => crate::run::<gravity::Gravity>(),
        "timeseries" => crate::run::<timeseries::Timeseries>(),
        "graph" => crate::run::<graph::Graph>(),
        "boids" => crate::run::<boids::Boids>(),
        "life" => crate::run::<life::Life>(),
        "volume" => crate::run::<volume::Volume>(),
        other => show_message(&format!("unknown example '{other}'")),
    }
}

/// A parameter from the page URL, e.g. `query("example")` for `?example=boids`.
pub fn query(key: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    web_sys::UrlSearchParams::new_with_str(&search)
        .ok()?
        .get(key)
}

/// The `<canvas id="canvas">` element from `index.html`.
pub fn canvas() -> web_sys::HtmlCanvasElement {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("canvas"))
        .and_then(|e| e.dyn_into().ok())
        .expect("index.html needs a <canvas id=\"canvas\">")
}

/// Show a message (e.g. a panic) on the page.
fn show_message(message: &str) {
    if let Some(element) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("message"))
    {
        element.set_text_content(Some(message));
        let _ = element.remove_attribute("hidden");
    }
}
