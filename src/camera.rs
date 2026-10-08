//! An orbit camera (3D) that can also act as a pan/zoom camera (2D).
//!
//! Mouse controls:
//! - 3D: drag to orbit, shift+drag or middle-drag to pan, scroll/pinch to zoom.
//! - 2D: drag to pan, scroll/pinch to zoom.
//!
//! Dragging uses the left *and* right button. If your app uses the left button
//! itself (e.g. to drag nodes), set `camera.left_drag = false` in `new`.

use glam::{Mat4, Vec2, Vec3};

#[derive(Clone, Debug)]
pub struct Camera {
    /// The point the camera looks at (and orbits around).
    pub target: Vec3,
    /// Distance from the camera to `target`. In orthographic mode this sets the zoom.
    pub distance: f32,
    /// Rotation around the vertical axis, in radians.
    pub yaw: f32,
    /// Rotation up/down, in radians.
    pub pitch: f32,
    /// Vertical field of view, in radians.
    pub fov_y: f32,
    /// Orthographic instead of perspective projection.
    pub orthographic: bool,
    /// 2D mode: dragging pans instead of orbits, and the camera looks down -z.
    pub pan_only: bool,
    /// Whether dragging with the left mouse button moves the camera.
    pub left_drag: bool,
}

impl Camera {
    /// A perspective camera orbiting `target`.
    pub fn orbit(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            yaw: 0.6,
            pitch: 0.4,
            fov_y: 45f32.to_radians(),
            orthographic: false,
            pan_only: false,
            left_drag: true,
        }
    }

    /// An orthographic camera looking at the xy-plane, showing `height` world units vertically.
    pub fn view_2d(center: Vec2, height: f32) -> Self {
        let fov_y = 45f32.to_radians();
        Self {
            target: center.extend(0.0),
            distance: 0.5 * height / (0.5 * fov_y).tan(),
            yaw: 0.0,
            pitch: 0.0,
            fov_y,
            orthographic: true,
            pan_only: true,
            left_drag: true,
        }
    }

    /// The camera position in world space.
    pub fn eye(&self) -> Vec3 {
        let direction = Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        );
        self.target + direction * self.distance
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        let far = self.distance * 100.0 + 100.0;
        if self.orthographic {
            let h = self.half_height();
            Mat4::orthographic_rh(-h * aspect, h * aspect, -h, h, 0.0, far)
        } else {
            Mat4::perspective_rh(self.fov_y, aspect, self.distance * 0.01, far)
        }
    }

    /// Half the visible height (in world units) at the target.
    fn half_height(&self) -> f32 {
        self.distance * (0.5 * self.fov_y).tan()
    }

    /// Orbit (3D) or pan (2D) by a mouse movement of `delta` pixels.
    pub fn drag(&mut self, delta: Vec2, pan: bool, viewport_height: f32) {
        if pan || self.pan_only {
            // Move the target so the point under the mouse stays under the mouse.
            let scale = 2.0 * self.half_height() / viewport_height;
            let view = self.view();
            let right = view.row(0).truncate();
            let up = view.row(1).truncate();
            self.target += (-right * delta.x + up * delta.y) * scale;
        } else {
            self.yaw -= delta.x * 0.008;
            self.pitch = (self.pitch + delta.y * 0.008).clamp(-1.55, 1.55);
        }
    }

    /// Zoom in (positive) or out (negative).
    pub fn zoom(&mut self, amount: f32) {
        self.distance *= (-amount).exp();
    }
}
