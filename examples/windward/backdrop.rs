use bevy::{
    asset::RenderAssetUsages,
    light::NotShadowCaster,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

use crate::landscape::{SUN, ground, noise, reflected_layers, smoothstep};

const COLUMNS: u32 = 320;
const ROWS: u32 = 352;
const TILE: u32 = 64;
const SPACING: f32 = 3.0;
const ORIGIN: Vec2 = Vec2::new(-480.0, -528.0);
/// The heightmap stops at this radius; the ridge ring carries the view past it.
const HEIGHTMAP_RADIUS: f32 = 474.0;
const RING_INNER: f32 = 440.0;
const RING_OUTER: f32 = 650.0;
const RING_SEGMENTS: u32 = 448;
const RING_SECTORS: u32 = 8;
const RING_RADII: u32 = 30;
const ELEVATION: &[u8; ((COLUMNS + 1) * (ROWS + 1) * 2) as usize] =
    include_bytes!("../../assets/windward/cascades-height.u16le");

pub fn spawn_backdrop(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mountain = materials.add(StandardMaterial {
        reflectance: 0.28,
        perceptual_roughness: 0.95,
        ..default()
    });
    for north in (0..ROWS).step_by(TILE as usize) {
        for west in (0..COLUMNS).step_by(TILE as usize) {
            commands.spawn((
                Name::new("Cascade ridgeline"),
                Mesh3d(meshes.add(terrain_tile(west, north))),
                MeshMaterial3d(mountain.clone()),
                reflected_layers(),
                NotShadowCaster,
            ));
        }
    }

    // The far ring bakes its own light and aerial perspective. Distance fog is off
    // because its color is already in the haze below.
    let ridges = materials.add(StandardMaterial {
        unlit: true,
        fog_enabled: false,
        ..default()
    });
    for sector in 0..RING_SECTORS {
        commands.spawn((
            Name::new("Distant Cascade ridges"),
            Mesh3d(meshes.add(ridge_sector(sector))),
            MeshMaterial3d(ridges.clone()),
            reflected_layers(),
            NotShadowCaster,
        ));
    }
}

fn near_edge(p: Vec2) -> f32 {
    (p.x.abs() - 69.1).max(p.y - 43.1).max(-98.0 - p.y)
}

fn elevation(column: u32, row: u32) -> f32 {
    let offset = ((COLUMNS + 1) * row + column) as usize * 2;
    f32::from(u16::from_le_bytes([
        ELEVATION[offset],
        ELEVATION[offset + 1],
    ]))
}

fn terrain_point(column: u32, row: u32) -> Vec3 {
    let p = ORIGIN + SPACING * Vec2::new(column as f32, row as f32);
    let mountain = (elevation(column, row) - 1325.0) * 0.030 + 3.2;
    let edge_blend = smoothstep(-4.0, 46.0, near_edge(p));
    let height = (ground(p) - 0.22).lerp(mountain, edge_blend);
    Vec3::new(p.x, height, p.y)
}

fn terrain_tile(west: u32, north: u32) -> Mesh {
    let east = (west + TILE).min(COLUMNS);
    let south = (north + TILE).min(ROWS);
    let stride = east + 1 - west;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    for row in north..=south {
        for column in west..=east {
            let p = terrain_point(column, row);
            let dx = terrain_point((column + 1).min(COLUMNS), row)
                - terrain_point(column.saturating_sub(1), row);
            let dz = terrain_point(column, (row + 1).min(ROWS))
                - terrain_point(column, row.saturating_sub(1));
            let normal = dz.cross(dx).normalize();
            positions.push(p.to_array());
            normals.push(normal.to_array());
            colors.push(
                mountain_color(p, normal, elevation(column, row))
                    .extend(1.0)
                    .to_array(),
            );
            if column < east
                && row < south
                && near_edge(p.xz()) > -8.0
                && p.xz().length() < HEIGHTMAP_RADIUS
            {
                let index = (row - north) * stride + column - west;
                indices.extend([
                    index,
                    index + stride,
                    index + 1,
                    index + 1,
                    index + stride,
                    index + stride + 1,
                ]);
            }
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

fn mountain_color(point: Vec3, normal: Vec3, actual_height: f32) -> Vec3 {
    let p = point.xz();
    let macro_detail = noise(p * 0.049);
    let strata = noise(p * 0.23 + Vec2::splat(13.4));
    let timberline = 1800.0 + 180.0 * macro_detail + 120.0 * normal.x;
    let tundra = smoothstep(timberline - 130.0, timberline + 150.0, actual_height);
    let steep = 1.0 - smoothstep(0.51, 0.93, normal.y);
    let forest = Vec3::new(0.055, 0.082, 0.067) * (0.70 + 0.46 * macro_detail + 0.21 * strata);
    let stone = Vec3::new(0.16, 0.148, 0.136) * (0.54 + 1.05 * strata);
    let bare = forest.lerp(stone, (0.9 * tundra + 0.52 * steep).min(1.0));
    let snowline = 2170.0 + 200.0 * macro_detail + 126.0 * normal.z;
    let altitude = smoothstep(snowline - 30.0, snowline + 185.0, actual_height);
    let snow = Vec3::new(0.63, 0.72, 0.82) * (0.76 + 0.24 * strata);
    bare.lerp(snow, altitude * smoothstep(0.37, 0.76, normal.y))
}

/// Ridged noise in 0..1, sharp along crests, for the far peaks.
fn ridged(p: Vec2) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 0.56;
    let mut frequency = 1.0 / 150.0;
    let mut weight = 1.0;
    for octave in 0..4 {
        let n = noise(p * frequency + Vec2::splat(31.7 * octave as f32));
        let crest = (1.0 - (2.0 * n - 1.0).abs()).powi(2) * weight;
        sum += amplitude * crest;
        weight = crest.clamp(0.0, 1.0);
        amplitude *= 0.5;
        frequency *= 2.1;
    }
    sum
}

fn ridge_height(p: Vec2) -> f32 {
    let r = p.length();
    let massif = noise(p / 240.0 + Vec2::splat(7.3));
    let peaks = 24.0 + 125.0 * ridged(p) * (0.45 + 0.85 * massif);
    // Start under the heightmap edge, then climb to a crest that is the skyline.
    let rise = smoothstep(478.0, 585.0, r);
    let crest = 0.72 + 0.28 * smoothstep(560.0, 630.0, r);
    (-44.0_f32).lerp(peaks * crest, rise)
}

fn ridge_point(segment: u32, ring: u32) -> Vec3 {
    let angle = std::f32::consts::TAU * segment as f32 / RING_SEGMENTS as f32;
    let t = ring as f32 / RING_RADII as f32;
    // Space the rings closer near the crest, where the silhouette needs detail.
    let radius = RING_INNER + (RING_OUTER - RING_INNER) * t.powf(0.8);
    let p = radius * Vec2::new(angle.cos(), angle.sin());
    p.extend(ridge_height(p)).xzy()
}

fn ridge_sector(sector: u32) -> Mesh {
    let per_sector = RING_SEGMENTS / RING_SECTORS;
    let first = sector * per_sector;
    let stride = RING_RADII + 1;
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    for segment in first..=first + per_sector {
        for ring in 0..=RING_RADII {
            let p = ridge_point(segment, ring);
            let along =
                ridge_point(segment + 1, ring) - ridge_point(segment + RING_SEGMENTS - 1, ring);
            let across = ridge_point(segment, (ring + 1).min(RING_RADII))
                - ridge_point(segment, ring.saturating_sub(1));
            let mut normal = along.cross(across).normalize();
            if normal.y < 0.0 {
                normal = -normal;
            }
            positions.push(p.to_array());
            colors.push(ridge_color(p, normal).extend(1.0).to_array());
            if segment < first + per_sector && ring < RING_RADII {
                let index = (segment - first) * stride + ring;
                indices.extend([
                    index,
                    index + stride,
                    index + 1,
                    index + 1,
                    index + stride,
                    index + stride + 1,
                ]);
            }
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

/// Linear color with baked sun, sky light, and aerial perspective.
fn ridge_color(point: Vec3, normal: Vec3) -> Vec3 {
    let p = point.xz();
    let height = point.y;
    let detail = noise(p * 0.09);
    let grain = noise(p * 0.37 + Vec2::splat(5.1));
    let treeline = 58.0 + 24.0 * detail;
    let steep = 1.0 - smoothstep(0.55, 0.9, normal.y);
    let forest = Vec3::new(0.030, 0.046, 0.038) * (0.6 + 0.6 * grain + 0.3 * detail);
    let rock = Vec3::new(0.12, 0.115, 0.11) * (0.7 + 0.6 * grain);
    let bare = smoothstep(treeline - 10.0, treeline + 22.0, height) + 0.8 * steep;
    let mut albedo = forest.lerp(rock, bare.min(1.0));
    let snowline = 96.0 + 20.0 * detail - 14.0 * normal.z.max(0.0);
    let snow = smoothstep(snowline, snowline + 24.0, height) * smoothstep(0.42, 0.72, normal.y);
    albedo = albedo.lerp(Vec3::new(0.70, 0.75, 0.82), snow);

    let sun = normal.dot(SUN).max(0.0);
    let sky = 0.55 + 0.45 * normal.y;
    let lit = albedo * (Vec3::new(0.30, 0.36, 0.46) * sky + Vec3::new(1.0, 0.80, 0.62) * 1.6 * sun);

    // Aerial perspective: farther and higher ridges fade to a lighter blue, as the
    // camera's DistanceFog would at this range, with its glow toward the sun.
    let r = p.length();
    let view = (point - Vec3::Y * 10.0).normalize();
    let glow = view.dot(SUN).max(0.0).powi(12);
    let haze = Vec3::new(0.105, 0.145, 0.215).lerp(Vec3::new(0.30, 0.25, 0.22), 0.6 * glow);
    // Valley mist pools low, so ridge bases match the fogged heightmap in front.
    let mist = 1.0 - smoothstep(-10.0, 55.0, height);
    let aerial = 0.45 + 0.35 * smoothstep(470.0, 640.0, r) + 0.08 * smoothstep(40.0, 150.0, height);
    let low_haze = Vec3::new(0.21, 0.25, 0.31);
    let haze = haze.lerp(low_haze, 0.7 * mist);
    let aerial = aerial + 0.3 * mist;
    lit.lerp(haze, aerial.min(0.92))
}
