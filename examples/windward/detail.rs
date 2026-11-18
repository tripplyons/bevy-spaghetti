use bevy::{camera::MainPassResolutionOverride, prelude::*};

pub type CameraView = (
    &'static Camera,
    &'static Transform,
    &'static Projection,
    Option<&'static MainPassResolutionOverride>,
);

pub struct ScreenDetail {
    eye: Vec3,
    forward: Vec3,
    pixels_per_unit: f32,
}

impl ScreenDetail {
    pub fn new(
        (camera, transform, projection, resolution): (
            &Camera,
            &Transform,
            &Projection,
            Option<&MainPassResolutionOverride>,
        ),
    ) -> Option<Self> {
        let Projection::Perspective(projection) = projection else {
            return None;
        };
        let size = resolution
            .map(|resolution| resolution.0)
            .or_else(|| camera.physical_viewport_size())?;
        if size.x == 0 || size.y == 0 {
            return None;
        }
        Some(Self {
            eye: transform.translation,
            forward: transform.forward().as_vec3(),
            pixels_per_unit: size.y as f32 / (2.0 * (0.5 * projection.fov).tan()),
        })
    }

    pub fn span_pixels(&self, center: Vec3, length: f32, depth_radius: f32) -> f32 {
        let depth = (center - self.eye).dot(self.forward);
        if depth + depth_radius <= 0.0 {
            return 0.0;
        }
        length * self.pixels_per_unit / (depth - depth_radius).max(0.25)
    }
}
