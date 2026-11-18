//! Crown meshes for the Cascade slopes and the distant skyline.
mod atlas;

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

pub use atlas::image as crown_atlas;

use super::Relief;
use crate::landscape::{hash01, smoothstep};

/// One candidate tree per cell of this size, jittered inside the cell.
const CELL: f32 = 3.2;

/// Two broad skirts suffice for the few pixels a skyline crown covers.
const RIDGE_TIERS: &[(f32, f32, f32)] = &[(0.03, 0.62, 1.0), (0.4, 1.0, 0.6)];
const POINTS: u32 = 6;

/// Where one fir stands, how big it is, and its linear albedo.
struct Fir {
    base: Vec3,
    height: f32,
    radius: f32,
    color: Vec3,
    seed: u32,
}

#[derive(Default)]
pub struct CanopyMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl CanopyMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn into_mesh(self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
        .with_inserted_indices(Indices::U32(self.indices));
        if !self.uvs.is_empty() {
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs);
        }
        mesh
    }

    fn push_crown(&mut self, fir: Fir) {
        let to_trail = super::WALK_CENTER - fir.base.xz();
        let inward = to_trail.normalize_or(Vec2::X);
        // Face the first card toward the walkable valley. A second view of
        // the same crown fills in when the hiker approaches from the side.
        let turn = (hash01(fir.seed + 19) - 0.5) * 0.52;
        let axis = (-inward.perp()).rotate(Vec2::from_angle(turn));
        let planes = if to_trail.length_squared() > 155.0_f32.powi(2) {
            1
        } else {
            2
        };
        let lean = Vec2::new(hash01(fir.seed + 20) - 0.5, hash01(fir.seed + 21) - 0.5);
        let top = fir.base
            + Vec3::Y * (1.03 * fir.height)
            + (lean * (0.11 * fir.height)).extend(0.0).xzy();
        let variant = (hash01(fir.seed + 23) * atlas::VARIANTS as f32) as u32;
        for (plane, axis) in [axis, axis.perp()].into_iter().take(planes).enumerate() {
            let direction = axis.extend(0.0).xzy();
            let half_width = direction * (1.46 * fir.radius);
            let facing = 0.66 * inward.extend(0.0).xzy() + 0.46 * Vec3::Y;
            let left = (facing - 0.75 * direction).normalize();
            let right = (facing + 0.75 * direction).normalize();
            let first = self.positions.len() as u32;
            self.positions.extend([
                (fir.base - half_width).to_array(),
                (fir.base + half_width).to_array(),
                (top - half_width).to_array(),
                (top + half_width).to_array(),
            ]);
            self.normals.extend([
                left.to_array(),
                right.to_array(),
                left.to_array(),
                right.to_array(),
            ]);
            self.colors.extend([fir.color.extend(1.0).to_array(); 4]);
            self.uvs.extend([
                atlas::uv(variant, plane, Vec2::Y),
                atlas::uv(variant, plane, Vec2::ONE),
                atlas::uv(variant, plane, Vec2::ZERO),
                atlas::uv(variant, plane, Vec2::X),
            ]);
            self.indices
                .extend([first, first + 1, first + 2, first + 1, first + 3, first + 2]);
        }
    }

    /// Bakes lighting and aerial perspective into the skyline fir's vertices.
    fn push_ridge_fir(&mut self, fir: Fir) {
        let Fir {
            base,
            height,
            radius,
            color,
            seed,
        } = fir;
        let lean = Vec2::new(hash01(seed + 20) - 0.5, hash01(seed + 21) - 0.5) * 0.07;
        let axis = |fraction: f32| base + Vec3::new(lean.x, 1.0, lean.y) * fraction * height;
        for (tier, &(bottom, top, width)) in RIDGE_TIERS.iter().enumerate() {
            let seed = seed + 31 * tier as u32;
            let apex = axis(top);
            // Upper tiers catch more sky and sun; lower skirts sit in shade.
            let shade = 0.55 + 0.6 * tier as f32 / RIDGE_TIERS.len() as f32;
            let turn = hash01(seed) * std::f32::consts::TAU;
            let rim: [Vec3; POINTS as usize] = std::array::from_fn(|point| {
                let angle = turn + std::f32::consts::TAU * point as f32 / POINTS as f32;
                let long = point % 2 == 0;
                let reach =
                    if long { 1.0 } else { 0.55 } * (0.8 + 0.4 * hash01(seed + point as u32 + 1));
                let droop = if long { 0.04 } else { -0.02 };
                axis(bottom - droop)
                    + (Vec2::from_angle(angle) * radius * width * reach)
                        .extend(0.0)
                        .xzy()
            });
            // Average the neighboring faces at each rim vertex. Keeping
            // separate vertices per tier preserves the shade under each skirt.
            let anchor = axis(bottom);
            let mut rim_normals = [Vec3::ZERO; POINTS as usize];
            for point in 0..POINTS as usize {
                let next = (point + 1) % POINTS as usize;
                let a = rim[point];
                let b = rim[next];
                let normal = (apex - a).cross(b - a);
                rim_normals[point] += normal;
                rim_normals[next] += normal;
            }
            let first = self.positions.len() as u32;
            for (position, normal) in rim.into_iter().zip(rim_normals) {
                // The narrow skirt faces nearly sideways. Tilt the foliage
                // normals toward the sky for the light caught by the sprays.
                debug_assert!(normal.dot(position - anchor) > 0.0);
                let normal = normal.normalize();
                let normal = (normal + Vec3::Y * 0.35).normalize();
                self.positions.push(position.to_array());
                self.normals.push(normal.to_array());
                self.colors.push(
                    super::ridge_light(color * shade, position, normal)
                        .extend(1.0)
                        .to_array(),
                );
            }
            self.positions.push(apex.to_array());
            self.normals.push(Vec3::Y.to_array());
            self.colors.push(
                super::ridge_light(color * (shade + 0.16), apex, Vec3::Y)
                    .extend(1.0)
                    .to_array(),
            );
            for point in 0..POINTS {
                self.indices
                    .extend([first + point, first + POINTS, first + (point + 1) % POINTS]);
            }
        }
    }
}

/// Plants firs across the backdrop and returns one mesh per terrain tile.
pub fn plant(relief: &Relief, tiles: UVec2, tile_size: f32) -> Vec<CanopyMesh> {
    let mut chunks: Vec<CanopyMesh> = (0..tiles.x * tiles.y).map(|_| default()).collect();
    let min = super::ORIGIN;
    let max = super::ORIGIN + relief.extent();
    let columns = ((max.x - min.x) / CELL) as u32;
    let rows = ((max.y - min.y) / CELL) as u32;
    for row in 0..rows {
        for column in 0..columns {
            let seed = (row * columns + column).wrapping_mul(97) + 0x5EED;
            let jitter = Vec2::new(hash01(seed), hash01(seed + 1));
            let at = min + (Vec2::new(column as f32, row as f32) + 0.1 + 0.8 * jitter) * CELL;
            let r = at.length();
            if r > super::HEIGHTMAP_RADIUS - 3.0 {
                continue;
            }
            let site = relief.sample(at);
            let cover = super::canopy_cover(at, site.normal, site.elevation, site.hollow);
            if hash01(seed + 2) >= cover.trees {
                continue;
            }
            // Firs shrink toward the timberline, and slightly with distance so far
            // slopes read as larger mountains.
            let stunted = 1.0 - 0.64 * cover.alpine;
            let distance = 1.0 - 0.24 * smoothstep(120.0, 450.0, r);
            let emergent = if hash01(seed + 7) < 0.04 { 1.30 } else { 1.0 };
            let height =
                6.4 * (0.55 + 0.8 * hash01(seed + 3).powf(1.5)) * emergent * stunted * distance;
            let radius = height * (0.145 + 0.11 * hash01(seed + 4));
            let tone = 0.65 + 0.7 * hash01(seed + 5);
            let hue = hash01(seed + 6);
            let color = Vec3::new(0.022, 0.039, 0.027)
                .lerp(Vec3::new(0.020, 0.036, 0.036), smoothstep(0.6, 1.0, hue))
                .lerp(
                    Vec3::new(0.034, 0.042, 0.023),
                    1.0 - smoothstep(0.0, 0.25, hue),
                )
                * tone
                * (1.0 - 0.18 * site.hollow.max(0.0));
            let base = Vec3::new(at.x, site.height - 0.3, at.y);
            let tile = ((at - min) / tile_size).floor().as_uvec2().min(tiles - 1);
            chunks[(tile.y * tiles.x + tile.x) as usize].push_crown(Fir {
                base,
                height,
                radius,
                color,
                seed,
            });
        }
    }
    chunks
}

/// Plants firs on the far ridge ring, below its treeline, with light and
/// aerial perspective baked into their colors like the ring itself.
pub fn plant_ring(sectors: u32) -> Vec<CanopyMesh> {
    const RING_CELL: f32 = 4.6;
    let mut chunks: Vec<CanopyMesh> = (0..sectors).map(|_| default()).collect();
    let reach = super::RING_OUTER;
    let cells = (2.0 * reach / RING_CELL) as u32;
    for row in 0..cells {
        for column in 0..cells {
            let seed = (row * cells + column).wrapping_mul(131) + 0xF1A5;
            let jitter = Vec2::new(hash01(seed), hash01(seed + 1));
            let at = Vec2::splat(-reach)
                + (Vec2::new(column as f32, row as f32) + 0.1 + 0.8 * jitter) * RING_CELL;
            let r = at.length();
            if !(super::HEIGHTMAP_RADIUS + 6.0..reach - 8.0).contains(&r) {
                continue;
            }
            let height_at = super::ridge_height(at);
            let dx = super::ridge_height(at + Vec2::X) - super::ridge_height(at - Vec2::X);
            let dz = super::ridge_height(at + Vec2::Y) - super::ridge_height(at - Vec2::Y);
            let normal = Vec3::new(-dx, 2.0, -dz).normalize();
            let chance = super::ridge_forest(at, height_at, normal);
            if hash01(seed + 2) >= chance {
                continue;
            }
            let height = 5.6 * (0.6 + 0.7 * hash01(seed + 3));
            let radius = height * (0.22 + 0.08 * hash01(seed + 4));
            let color = Vec3::new(0.022, 0.039, 0.027) * (0.7 + 0.6 * hash01(seed + 5));
            let base = Vec3::new(at.x, height_at - 1.4, at.y);
            let angle = at.y.atan2(at.x).rem_euclid(std::f32::consts::TAU);
            let sector = ((angle / std::f32::consts::TAU * sectors as f32) as u32).min(sectors - 1);
            chunks[sector as usize].push_ridge_fir(Fir {
                base,
                height,
                radius,
                color,
                seed,
            });
        }
    }
    chunks
}
