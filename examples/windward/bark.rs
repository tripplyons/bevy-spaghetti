//! Textured wood for standing deadwood and living conifers.
use bevy::{
    asset::RenderAssetUsages,
    camera::primitives::{Aabb, MeshAabb},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

use crate::{
    conifer::Stem,
    detail::{CameraView, ScreenDetail},
    landscape::{hash01, load_tile},
    player::HikerCamera,
};

#[derive(Component, Clone)]
pub struct BarkMeshes {
    fine: Handle<Mesh>,
    coarse: Handle<Mesh>,
    pub bounds: Aabb,
    height: f32,
    detail_span: Vec3,
    reach: f32,
    fine_pixels: f32,
}

impl BarkMeshes {
    pub fn snag(
        meshes: &mut Assets<Mesh>,
        seed: u32,
        height: f32,
        top: f32,
        reach: f32,
        girth: f32,
    ) -> Self {
        let fine = snag_mesh(seed, height, top, reach, girth, true);
        let coarse = snag_mesh(seed, height, top, reach, girth, false);
        Self::new(meshes, [fine, coarse], top, Vec3::Y * top, reach, 118.0)
    }

    pub fn living(meshes: &mut Assets<Mesh>, seed: u32, stem: Stem) -> Self {
        let fine = stem_mesh(seed, stem, 16, (stem.end() / 0.45).ceil() as usize + 2);
        let coarse = stem_mesh(seed, stem, 8, 4);
        Self::new(
            meshes,
            [fine, coarse],
            stem.end(),
            Vec3::X * (2.0 * stem.radius),
            stem.radius,
            18.0,
        )
    }

    fn new(
        meshes: &mut Assets<Mesh>,
        [fine, coarse]: [Mesh; 2],
        height: f32,
        detail_span: Vec3,
        reach: f32,
        fine_pixels: f32,
    ) -> Self {
        let fine_bounds = fine.get_aabb().expect("bark has vertices");
        let coarse_bounds = coarse.get_aabb().expect("bark has vertices");
        Self {
            fine: meshes.add(fine),
            coarse: meshes.add(coarse),
            // Both levels may be moved out of the main world after upload.
            bounds: Aabb::from_min_max(
                fine_bounds.min().min(coarse_bounds.min()).into(),
                fine_bounds.max().max(coarse_bounds.max()).into(),
            ),
            height,
            detail_span,
            reach,
            fine_pixels,
        }
    }

    pub fn coarse_mesh(&self) -> Mesh3d {
        Mesh3d(self.coarse.clone())
    }
}

pub fn select_detail(
    camera: Single<CameraView, With<HikerCamera>>,
    mut trunks: Query<(&GlobalTransform, &BarkMeshes, &mut Mesh3d)>,
) {
    let Some(view) = ScreenDetail::new(*camera) else {
        return;
    };
    for (transform, model, mut mesh) in &mut trunks {
        let center = transform.transform_point(Vec3::Y * (model.height * 0.5));
        let affine = transform.affine();
        let pixels = view.span_pixels(
            center,
            affine.transform_vector3(model.detail_span).length(),
            model.reach * affine.matrix3.x_axis.length(),
        );
        let threshold = model.fine_pixels * if mesh.0 == model.fine { 0.76 } else { 1.0 };
        let selected = if pixels >= threshold {
            &model.fine
        } else {
            &model.coarse
        };
        if mesh.0 != *selected {
            mesh.0 = selected.clone();
        }
    }
}

pub fn bark_material(server: &AssetServer) -> StandardMaterial {
    StandardMaterial {
        // The tile comes from Poly Haven's CC0 Dead Tree Trunk scan.
        base_color_texture: Some(load_tile(server, "windward/snag-bark-color.dds", true)),
        normal_map_texture: Some(load_tile(server, "windward/snag-bark-normal.dds", false)),
        perceptual_roughness: 0.94,
        reflectance: 0.24,
        ..default()
    }
}

fn stem_mesh(seed: u32, stem: Stem, sides: usize, rows: usize) -> Mesh {
    let mut wood = Wood::default();
    // Both meshes share these base rings to hold the silhouette at ground level.
    let flare = 0.32_f32.min(0.3 * stem.end());
    let upper = (0..=rows).map(|row| {
        let t = (row as f32 / rows as f32).powf(1.45);
        flare + (stem.end() - flare) * t
    });
    let trunk: Vec<_> = [-0.18, 0.0]
        .into_iter()
        .chain(upper)
        .map(|y| {
            let damp = (-y.max(0.0) / 0.46).exp();
            let shade = crate::landscape::smoothstep(stem.bole, stem.bole + 1.2, y);
            Ring {
                center: Vec3::Y * y,
                radius: stem.radius_at(y),
                tint: Vec3::ONE.lerp(Vec3::new(0.58, 0.68, 0.47), damp) * (1.0 - 0.57 * shade),
            }
        })
        .collect();
    let split = if stem.top < stem.height { 0.10 } else { 0.0 };
    wood.tube(&trunk, sides, seed, split);
    wood.into_mesh()
}

fn snag_mesh(seed: u32, height: f32, top: f32, crown: f32, girth: f32, fine: bool) -> Mesh {
    let mut wood = Wood::default();
    let phase = std::f32::consts::TAU * hash01(seed);
    let base_radius = (0.019 * height + 0.032) * girth;
    let lean = Vec2::from_angle(phase) * (0.08 + 0.24 * hash01(seed + 1));
    let center = |y: f32| {
        let t = y / top;
        let bend = Vec2::from_angle(phase + 2.0) * (0.075 * (t * 5.4).sin() * t);
        (lean * t * t + bend).extend(y).xzy()
    };
    let radius = |y: f32| {
        let flare = 1.0 + 0.75 * (-y / 0.44).exp();
        base_radius * (1.0 - 0.97 * y / height).max(0.055) * flare
    };

    let spacing = if fine { 0.34 } else { 0.68 };
    let rows = (top / spacing).ceil() as usize;
    let trunk: Vec<_> = (0..=rows)
        .map(|row| {
            let y = top * row as f32 / rows as f32;
            let wet = (-y / 0.68).exp();
            Ring {
                center: center(y),
                radius: radius(y),
                tint: Vec3::ONE.lerp(Vec3::new(0.44, 0.53, 0.39), wet),
            }
        })
        .collect();
    let split = if top < height { 0.49 } else { 0.13 };
    wood.tube(&trunk, if fine { 20 } else { 10 }, seed, split);

    let mut y = 0.6 + 0.44 * hash01(seed + 2);
    let mut branch = seed + 29;
    while y < top - 0.22 {
        let azimuth = phase + branch as f32 * 2.39996 + 1.9 * hash01(branch + 1);
        let direction = Vec2::from_angle(azimuth).extend(0.0).xzy();
        let taper = (1.0 - y / height).max(0.12);
        let remaining = hash01(branch + 2);
        let reach = (0.16 + crown * taper * (0.18 + 0.97 * remaining.powi(2))).min(1.57);
        let thickness = radius(y) * (0.16 + 0.21 * hash01(branch + 3));
        let start = center(y) + direction * radius(y) * 0.68;
        let sag = -0.43 + 0.65 * hash01(branch + 4);
        let sweep = direction.cross(Vec3::Y) * (0.32 * (hash01(branch + 5) - 0.5));
        let point =
            |t: f32| start + reach * t * (direction + sweep * t + Vec3::Y * (0.22 + sag * t));
        let rings: [_; 5] = std::array::from_fn(|index| {
            let t = index as f32 * 0.25;
            Ring {
                center: point(t),
                radius: thickness * (1.0 - 0.82 * t).powi(2) + 0.006,
                tint: Vec3::splat(0.86 + 0.14 * t),
            }
        });
        if fine {
            wood.tube(&rings, 7, branch, 0.025);
        } else {
            wood.tube(&[rings[0], rings[2], rings[4]], 5, branch, 0.025);
        }
        // A surviving side shoot on some of the long limbs breaks up the outline.
        if fine && reach > 0.65 && hash01(branch + 6) < 0.55 {
            let fork = if hash01(branch + 7) < 0.5 { 1.0 } else { -1.0 };
            let root = point(0.46);
            let side = direction * 0.65 + direction.cross(Vec3::Y) * (0.52 * fork);
            let rings: [_; 3] = std::array::from_fn(|index| {
                let t = index as f32 * 0.5;
                Ring {
                    center: root + reach * t * (side * 0.52 + Vec3::Y * (0.14 - 0.20 * t)),
                    radius: thickness * (0.36 - 0.30 * t) + 0.003,
                    tint: Vec3::splat(0.83),
                }
            });
            wood.tube(&rings, 5, branch + 7, 0.008);
        }
        y += 0.19 + 0.50 * hash01(branch + 8).powf(1.3);
        branch += 17;
    }
    wood.into_mesh()
}

#[derive(Clone, Copy)]
struct Ring {
    center: Vec3,
    radius: f32,
    tint: Vec3,
}

#[derive(Default)]
struct Wood {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uv: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl Wood {
    fn tube(&mut self, rings: &[Ring], sides: usize, seed: u32, split: f32) {
        let first = self.positions.len();
        let stride = sides + 1;
        let last = rings.len() - 1;
        let axis = (rings[last].center - rings[0].center).normalize();
        let reference = if axis.y.abs() > 0.8 { Vec3::X } else { Vec3::Y };
        let phase = std::f32::consts::TAU * hash01(seed);
        let wraps = (rings[0].radius * 7.0).round().max(1.0);
        let mut distance = 0.0;
        for (index, ring) in rings.iter().enumerate() {
            let previous = rings[index.saturating_sub(1)].center;
            let next = rings[(index + 1).min(last)].center;
            let tangent = (next - previous).normalize();
            let across = reference.cross(tangent).normalize();
            let around = tangent.cross(across);
            distance += ring.center.distance(previous);
            for side in 0..=sides {
                let turn = (side % sides) as f32 / sides as f32 * std::f32::consts::TAU;
                let (sin, cos) = turn.sin_cos();
                let outward = across * cos + around * sin;
                let ridges = 1.0
                    + 0.10 * (3.0 * turn + phase + distance * 0.55).sin()
                    + 0.062 * (7.0 * turn - phase - distance * 0.37).sin();
                let chipped = hash01(seed + (32 * (side % sides) / sides) as u32);
                let fracture = if index == last {
                    split * (chipped.powi(4) - 0.25)
                } else {
                    0.0
                };
                self.positions
                    .push(ring.center + ring.radius * ridges * outward + tangent * fracture);
                self.uv
                    .push([side as f32 / sides as f32 * wraps, -distance * 0.9]);
                self.colors
                    .push((ring.tint * (0.82 + 0.23 * chipped)).extend(1.0).to_array());
                if index < last && side < sides {
                    let vertex = (first + index * stride + side) as u32;
                    let next_ring = vertex + stride as u32;
                    self.indices.extend([
                        vertex,
                        vertex + 1,
                        next_ring,
                        vertex + 1,
                        next_ring + 1,
                        next_ring,
                    ]);
                }
            }
        }
        for row in 0..=last {
            for side in 0..=sides {
                let s = side % sides;
                let at = |r, s| self.positions[first + r * stride + s];
                let across = at(row, (s + 1) % sides) - at(row, (s + sides - 1) % sides);
                let along = at((row + 1).min(last), s) - at(row.saturating_sub(1), s);
                self.normals.push(across.cross(along).normalize());
            }
        }
        let rim = first + last * stride;
        let tip = rings[last].center;
        // Flat triangles at the split catch light on newly exposed wood.
        for side in 0..sides {
            let a = self.positions[rim + side];
            let b = self.positions[rim + side + 1];
            let face = (a - tip).cross(b - tip).normalize();
            let start = self.positions.len() as u32;
            let cross = reference.cross(axis).normalize();
            let other = axis.cross(cross);
            for point in [tip, a, b] {
                let delta = point - tip;
                self.positions.push(point);
                self.normals.push(face);
                self.uv
                    .push([0.4 + delta.dot(cross), 0.4 + delta.dot(other)]);
                self.colors.push([1.49, 1.24, 0.92, 1.0]);
            }
            self.indices.extend([start, start + 1, start + 2]);
        }
    }

    fn into_mesh(self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uv)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
        .with_inserted_indices(Indices::U32(self.indices));
        mesh.generate_tangents()
            .expect("bark UVs and normals support tangent generation");
        mesh
    }
}
