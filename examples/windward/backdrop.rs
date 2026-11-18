use bevy::{
    asset::RenderAssetUsages,
    light::NotShadowCaster,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

use crate::landscape::{ground, noise, reflected_layers, smoothstep};

const COLUMNS: u32 = 320;
const ROWS: u32 = 352;
const TILE: u32 = 64;
const SPACING: f32 = 3.0;
const ORIGIN: Vec2 = Vec2::new(-480.0, -528.0);
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
            if column < east && row < south && near_edge(p.xz()) > -8.0 {
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
