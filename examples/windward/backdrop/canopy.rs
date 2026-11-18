//! Low-poly spire firs that cover the backdrop slopes below the timberline.
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

use super::Relief;
use crate::landscape::{hash01, smoothstep};

/// One candidate tree per cell of this size, jittered inside the cell.
const CELL: f32 = 3.2;

/// Base height, apex height, and radius of each tier, as fractions of the
/// tree's height and radius. Stacked skirts give the notched fir outline.
/// Distant firs use fewer tiers, since their notches are under a pixel.
const TIERS: [&[(f32, f32, f32)]; 3] = [
    &[
        (0.03, 0.42, 1.0),
        (0.24, 0.62, 0.8),
        (0.45, 0.82, 0.58),
        (0.66, 1.0, 0.36),
    ],
    &[(0.03, 0.5, 1.0), (0.3, 0.76, 0.72), (0.56, 1.0, 0.44)],
    &[(0.03, 0.62, 1.0), (0.4, 1.0, 0.6)],
];
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
    indices: Vec<u32>,
}

impl CanopyMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn into_mesh(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
        .with_inserted_indices(Indices::U32(self.indices))
    }

    /// Adds a fir. `bake` turns albedo, position, and normal into the vertex color.
    fn push_tree(
        &mut self,
        tiers: &[(f32, f32, f32)],
        fir: Fir,
        bake: &dyn Fn(Vec3, Vec3, Vec3) -> Vec3,
    ) {
        let Fir {
            base,
            height,
            radius,
            color,
            seed,
        } = fir;
        let lean = Vec2::new(hash01(seed + 20) - 0.5, hash01(seed + 21) - 0.5) * 0.07;
        let axis = |fraction: f32| base + Vec3::new(lean.x, 1.0, lean.y) * fraction * height;
        for (tier, &(bottom, top, width)) in tiers.iter().enumerate() {
            let seed = seed + 31 * tier as u32;
            let apex = axis(top);
            // Upper tiers catch more sky and sun; lower skirts sit in shade.
            let shade = 0.55 + 0.6 * tier as f32 / tiers.len() as f32;
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
            // Flat facets break each skirt into lit and shaded sprays.
            for point in 0..POINTS as usize {
                let a = rim[point];
                let b = rim[(point + 1) % POINTS as usize];
                let mut normal = (b - a).cross(apex - a).normalize();
                if normal.dot(a + b - 2.0 * axis(bottom)) < 0.0 {
                    normal = -normal;
                }
                let normal = (normal + Vec3::Y * 0.35).normalize();
                let first = self.positions.len() as u32;
                self.positions
                    .extend([a.to_array(), b.to_array(), apex.to_array()]);
                self.normals.extend([normal.to_array(); 3]);
                let low = color * shade;
                let high = color * (shade + 0.5);
                self.colors.extend([
                    bake(low, a, normal).extend(1.0).to_array(),
                    bake(low, b, normal).extend(1.0).to_array(),
                    bake(high, apex, normal).extend(1.0).to_array(),
                ]);
                self.indices.extend([first, first + 2, first + 1]);
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
            let stunted = 0.5 + 0.5 * (1.0 - cover.alpine);
            let distance = 1.0 - 0.18 * smoothstep(120.0, 450.0, r);
            let emergent = if hash01(seed + 7) < 0.06 { 1.45 } else { 1.0 };
            let height =
                6.4 * (0.55 + 0.8 * hash01(seed + 3).powf(1.5)) * emergent * stunted * distance;
            let radius = height * (0.19 + 0.08 * hash01(seed + 4));
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
            let detail = usize::from(r > 210.0) + usize::from(r > 330.0);
            chunks[(tile.y * tiles.x + tile.x) as usize].push_tree(
                TIERS[detail],
                Fir {
                    base,
                    height,
                    radius,
                    color,
                    seed,
                },
                &|c, _, _| c,
            );
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
            chunks[sector as usize].push_tree(
                TIERS[2],
                Fir {
                    base,
                    height,
                    radius,
                    color,
                    seed,
                },
                &super::ridge_light,
            );
        }
    }
    chunks
}
