use bevy::{
    asset::AssetPath,
    camera::MainPassResolutionOverride,
    mesh::VertexAttributeValues,
    prelude::*,
};
use bevy_spaghetti::SpaghettiAsset;
use std::collections::HashMap;

/// Recomputes area-weighted smooth vertex normals welded by quantized 3D position,
/// removing hard shading seams on meshes whose exporter split normals across UV islands.
pub fn smooth_mesh_normals(mesh: &mut Mesh) {
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return;
    };
    let positions: Vec<Vec3> = positions.iter().map(|&p| Vec3::from_array(p)).collect();
    let Some(indices) = mesh.indices() else {
        return;
    };
    let tris: Vec<usize> = indices.iter().collect();
    let key = |p: Vec3| -> [i32; 3] {
        [
            (p.x * 10_000.0).round() as i32,
            (p.y * 10_000.0).round() as i32,
            (p.z * 10_000.0).round() as i32,
        ]
    };
    let mut accum: HashMap<[i32; 3], Vec3> = HashMap::with_capacity(positions.len());
    for tri in tris.chunks_exact(3) {
        let a = positions[tri[0]];
        let b = positions[tri[1]];
        let c = positions[tri[2]];
        let face = (b - a).cross(c - a);
        for &idx in tri {
            *accum.entry(key(positions[idx])).or_insert(Vec3::ZERO) += face;
        }
    }
    let normals: Vec<[f32; 3]> = positions
        .iter()
        .map(|&p| {
            accum
                .get(&key(p))
                .copied()
                .unwrap_or(Vec3::Y)
                .normalize_or(Vec3::Y)
                .to_array()
        })
        .collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
}

pub fn load_lit_cloud(
    server: &AssetServer,
    path: impl Into<AssetPath<'static>>,
) -> Handle<SpaghettiAsset> {
    server
        .load_builder()
        .with_settings(|bake: &mut Option<([f32; 3], f32)>| {
            // Scan splats overlap to reconstruct one surface.
            *bake = Some((crate::landscape::SUN.to_array(), 0.16));
        })
        .load(path)
}

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
