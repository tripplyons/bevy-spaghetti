//! The Cascade slopes past the trail valley: a heightmap ring of forested
//! hills with a canopy of spire firs, and a baked skyline of far ridges.
mod canopy;
mod detail_texture;

use bevy::{
    asset::RenderAssetUsages,
    light::NotShadowCaster,
    mesh::{Indices, PrimitiveTopology, UvChannel},
    prelude::*,
    render::render_resource::Face,
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
/// World size of one repeat of the albedo and normal detail maps.
const ALBEDO_TILE: f32 = 37.0;
const NORMAL_TILE: f32 = 14.0;
const ELEVATION: &[u8; ((COLUMNS + 1) * (ROWS + 1) * 2) as usize] =
    include_bytes!("../../assets/windward/cascades-height.u16le");

pub fn spawn_backdrop(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let (albedo, albedo_mean) = detail_texture::albedo_detail();
    let mountain = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(albedo)),
        normal_map_texture: Some(images.add(detail_texture::normal_detail())),
        normal_map_channel: UvChannel::Uv1,
        reflectance: 0.12,
        perceptual_roughness: 0.95,
        ..default()
    });
    let relief = Relief::new();
    for north in (0..ROWS).step_by(TILE as usize) {
        for west in (0..COLUMNS).step_by(TILE as usize) {
            commands.spawn((
                Name::new("Cascade ridgeline"),
                Mesh3d(meshes.add(terrain_tile(&relief, west, north, albedo_mean))),
                MeshMaterial3d(mountain.clone()),
                reflected_layers(),
                NotShadowCaster,
            ));
        }
    }

    let firs = materials.add(StandardMaterial {
        reflectance: 0.2,
        perceptual_roughness: 0.9,
        double_sided: true,
        cull_mode: None::<Face>,
        ..default()
    });
    let tiles = UVec2::new(COLUMNS.div_ceil(TILE), ROWS.div_ceil(TILE));
    for chunk in canopy::plant(&relief, tiles, TILE as f32 * SPACING) {
        if chunk.is_empty() {
            continue;
        }
        commands.spawn((
            Name::new("Cascade backdrop forest"),
            Mesh3d(meshes.add(chunk.into_mesh())),
            MeshMaterial3d(firs.clone()),
            reflected_layers(),
            NotShadowCaster,
        ));
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
    let ridge_firs = materials.add(StandardMaterial {
        unlit: true,
        fog_enabled: false,
        double_sided: true,
        cull_mode: None::<Face>,
        ..default()
    });
    for chunk in canopy::plant_ring(RING_SECTORS) {
        if chunk.is_empty() {
            continue;
        }
        commands.spawn((
            Name::new("Distant Cascade forest"),
            Mesh3d(meshes.add(chunk.into_mesh())),
            MeshMaterial3d(ridge_firs.clone()),
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

/// One heightmap vertex with what the colors and the canopy read from it.
#[derive(Clone, Copy)]
struct Site {
    point: Vec3,
    normal: Vec3,
    /// Real elevation in meters, before the vertical squash.
    elevation: f32,
    /// From -1 on a spur or crest through 0 on an even slope to 1 in a gully.
    hollow: f32,
}

pub struct Sample {
    height: f32,
    normal: Vec3,
    elevation: f32,
    hollow: f32,
}

/// The backdrop heightmap, evaluated once for every vertex.
pub struct Relief {
    sites: Vec<Site>,
}

impl Relief {
    fn new() -> Self {
        let width = COLUMNS + 1;
        let points: Vec<Vec3> = (0..=ROWS)
            .flat_map(|row| (0..=COLUMNS).map(move |column| terrain_point(column, row)))
            .collect();
        let point = |column: u32, row: u32| points[(row * width + column) as usize];
        let height = |column: i64, row: i64| {
            elevation(
                column.clamp(0, COLUMNS as i64) as u32,
                row.clamp(0, ROWS as i64) as u32,
            )
        };
        let mut sites = Vec::with_capacity(points.len());
        for row in 0..=ROWS {
            for column in 0..=COLUMNS {
                let dx =
                    point((column + 1).min(COLUMNS), row) - point(column.saturating_sub(1), row);
                let dz = point(column, (row + 1).min(ROWS)) - point(column, row.saturating_sub(1));
                let (c, r) = (column as i64, row as i64);
                // A positive Laplacian of the real elevation marks a hollow.
                let laplacian =
                    height(c - 2, r) + height(c + 2, r) + height(c, r - 2) + height(c, r + 2)
                        - 4.0 * height(c, r);
                sites.push(Site {
                    point: point(column, row),
                    normal: dz.cross(dx).normalize(),
                    elevation: height(c, r),
                    hollow: smoothstep(25.0, 170.0, laplacian)
                        - smoothstep(25.0, 170.0, -laplacian),
                });
            }
        }
        Self { sites }
    }

    fn site(&self, column: u32, row: u32) -> &Site {
        &self.sites[(row * (COLUMNS + 1) + column) as usize]
    }

    fn extent(&self) -> Vec2 {
        SPACING * Vec2::new(COLUMNS as f32, ROWS as f32)
    }

    /// Height on the mesh triangles, and smooth fields between vertices.
    fn sample(&self, at: Vec2) -> Sample {
        let grid = ((at - ORIGIN) / SPACING)
            .clamp(Vec2::ZERO, Vec2::new(COLUMNS as f32, ROWS as f32) - 0.001);
        let cell = grid.floor().as_uvec2();
        let f = grid - grid.floor();
        let s00 = self.site(cell.x, cell.y);
        let s10 = self.site(cell.x + 1, cell.y);
        let s01 = self.site(cell.x, cell.y + 1);
        let s11 = self.site(cell.x + 1, cell.y + 1);
        // Each cell splits along the diagonal from (1, 0) to (0, 1).
        let (h00, h10, h01, h11) = (s00.point.y, s10.point.y, s01.point.y, s11.point.y);
        let height = if f.x + f.y < 1.0 {
            h00 + f.x * (h10 - h00) + f.y * (h01 - h00)
        } else {
            h11 + (1.0 - f.x) * (h01 - h11) + (1.0 - f.y) * (h10 - h11)
        };
        let mix = |a: f32, b: f32, c: f32, d: f32| a.lerp(b, f.x).lerp(c.lerp(d, f.x), f.y);
        Sample {
            height,
            normal: s00
                .normal
                .lerp(s10.normal, f.x)
                .lerp(s01.normal.lerp(s11.normal, f.x), f.y)
                .normalize(),
            elevation: mix(s00.elevation, s10.elevation, s01.elevation, s11.elevation),
            hollow: mix(s00.hollow, s10.hollow, s01.hollow, s11.hollow),
        }
    }
}

/// How the land is covered, shared by the ground color and the fir canopy.
pub struct Cover {
    /// Chance of a fir at this spot.
    trees: f32,
    /// 0 in the forest, 1 above the timberline.
    alpine: f32,
    /// Bare rock on cliffs and strata bands.
    rock: f32,
    /// Snow on high, gentle faces and in high gullies.
    snow: f32,
}

/// Distance from the area the hiker can walk, which `player` clamps to.
fn walk_distance(p: Vec2) -> f32 {
    let center = Vec2::new(1.0, -19.0);
    let half = Vec2::new(38.0, 43.0);
    ((p - center).abs() - half).max(Vec2::ZERO).length()
}

fn canopy_cover(p: Vec2, normal: Vec3, elevation: f32, hollow: f32) -> Cover {
    let gully = hollow.max(0.0);
    let macro_detail = noise(p * 0.049);
    let timberline = 1800.0 + 180.0 * macro_detail + 120.0 * normal.x;
    let alpine = smoothstep(timberline - 130.0, timberline + 150.0, elevation);
    // Cliff bands follow the contours, broken up along their length.
    let strata = noise(Vec2::new(elevation / 48.0, (p.x - p.y) * 0.006));
    let band = smoothstep(0.6, 0.74, strata) * (1.0 - smoothstep(0.82, 0.94, normal.y));
    let cliff = 1.0 - smoothstep(0.62, 0.8, normal.y);
    let rock = (band + cliff).min(1.0);
    let clump =
        0.64 * noise(p * 0.021 + Vec2::splat(7.7)) + 0.36 * noise(p * 0.083 + Vec2::splat(3.1));
    let stands = smoothstep(0.28, 0.5, clump);
    // Avalanche chutes keep steep gullies open.
    let chute = gully * (1.0 - smoothstep(0.8, 0.92, normal.y));
    // Near the trail the forest module plants real firs; the low-poly ones
    // stay far enough out that their facets never show.
    let clear = smoothstep(32.0, 44.0, walk_distance(p)) * smoothstep(2.0, 10.0, near_edge(p));
    let trees =
        (1.25 * (1.0 - alpine) * (1.0 - rock) * stands * (1.0 - 0.85 * chute)).min(1.0) * clear;
    let snowline = 2170.0 + 200.0 * macro_detail + 126.0 * normal.z - 160.0 * gully;
    let snow = smoothstep(snowline - 30.0, snowline + 185.0, elevation)
        * smoothstep(0.37, 0.76, normal.y + 0.3 * gully);
    Cover {
        trees,
        alpine,
        rock,
        snow,
    }
}

fn terrain_tile(relief: &Relief, west: u32, north: u32, albedo_mean: f32) -> Mesh {
    let east = (west + TILE).min(COLUMNS);
    let south = (north + TILE).min(ROWS);
    let stride = east + 1 - west;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut tangents = Vec::new();
    let mut colors = Vec::new();
    let mut albedo_uvs = Vec::new();
    let mut normal_uvs = Vec::new();
    let mut indices = Vec::new();
    for row in north..=south {
        for column in west..=east {
            let site = relief.site(column, row);
            let p = site.point;
            positions.push(p.to_array());
            normals.push(site.normal.to_array());
            // UVs follow world x and z, so the tangent is +x on the surface and
            // the bitangent, +z, is -(normal × tangent).
            let tangent = (Vec3::X - site.normal * site.normal.x).normalize();
            tangents.push(tangent.extend(-1.0).to_array());
            albedo_uvs.push((p.xz() / ALBEDO_TILE).to_array());
            normal_uvs.push((p.xz() / NORMAL_TILE).to_array());
            colors.push((mountain_color(site) / albedo_mean).extend(1.0).to_array());
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
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, albedo_uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, normal_uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

/// Linear albedo of the ground: dark duff under the firs, olive glades,
/// gray rock bands, tundra above the timberline, and snow on the peaks.
fn mountain_color(site: &Site) -> Vec3 {
    let p = site.point.xz();
    let normal = site.normal;
    let cover = canopy_cover(p, normal, site.elevation, site.hollow);
    let broad = noise(p * 0.013 + Vec2::splat(2.6));
    let patches = noise(p * 0.071 + Vec2::splat(9.2));
    let strata = noise(p * 0.23 + Vec2::splat(13.4));
    let duff = Vec3::new(0.027, 0.044, 0.025) * (0.8 + 0.4 * patches);
    let glade = Vec3::new(0.072, 0.078, 0.034).lerp(Vec3::new(0.052, 0.068, 0.038), broad)
        * (0.78 + 0.44 * patches);
    let mut albedo = glade.lerp(duff, smoothstep(0.05, 0.5, cover.trees));
    let tundra = Vec3::new(0.105, 0.098, 0.076) * (0.7 + 0.5 * patches + 0.2 * strata);
    albedo = albedo.lerp(tundra, cover.alpine);
    let stone = Vec3::new(0.15, 0.143, 0.135) * (0.5 + 0.7 * strata + 0.3 * broad);
    albedo = albedo.lerp(
        stone,
        (0.85 * cover.rock + cover.alpine * (1.0 - normal.y) * 1.8).min(1.0),
    );
    // Gullies hold shade and wet, dark growth; spurs and crests catch light.
    albedo *= 1.0 - 0.38 * site.hollow.max(0.0) + 0.22 * (-site.hollow).max(0.0);
    let snow = Vec3::new(0.63, 0.72, 0.82) * (0.76 + 0.24 * strata);
    albedo.lerp(snow, cover.snow)
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

fn ridge_treeline(p: Vec2) -> f32 {
    58.0 + 24.0 * noise(p * 0.09)
}

/// Chance of a fir on the far ring: forest below the treeline, in stands,
/// and none on steep faces.
fn ridge_forest(p: Vec2, height: f32, normal: Vec3) -> f32 {
    let steep = 1.0 - smoothstep(0.55, 0.9, normal.y);
    let below = 1.0 - smoothstep(ridge_treeline(p) - 22.0, ridge_treeline(p) + 4.0, height);
    let stands = smoothstep(0.25, 0.5, noise(p * 0.027 + Vec2::splat(4.4)));
    // The low ring base hides under the heightmap edge and the valley mist.
    let visible = smoothstep(-6.0, 6.0, height);
    (1.3 * below * stands * (1.0 - steep)).min(1.0) * visible
}

/// Linear color with baked sun, sky light, and aerial perspective.
fn ridge_color(point: Vec3, normal: Vec3) -> Vec3 {
    let p = point.xz();
    let height = point.y;
    let detail = noise(p * 0.09);
    let grain = noise(p * 0.37 + Vec2::splat(5.1));
    let treeline = ridge_treeline(p);
    let steep = 1.0 - smoothstep(0.55, 0.9, normal.y);
    // Gullies and ribs run down the faces, which on the ring point outward.
    let across = p.y.atan2(p.x) * p.length();
    let ribs = noise(Vec2::new(across * 0.055, p.length() * 0.012));
    let forest = Vec3::new(0.030, 0.046, 0.038) * (0.6 + 0.6 * grain + 0.3 * detail);
    let rock = Vec3::new(0.12, 0.115, 0.11) * (0.55 + 0.5 * grain + 0.45 * ribs);
    let bare = smoothstep(treeline - 10.0, treeline + 22.0, height) + 0.8 * steep;
    let mut albedo = forest.lerp(rock, bare.min(1.0));
    let snowline = 96.0 + 20.0 * detail - 14.0 * normal.z.max(0.0) - 22.0 * (0.5 - ribs);
    let snow = smoothstep(snowline, snowline + 24.0, height)
        * smoothstep(0.42, 0.72, normal.y + 0.2 * (0.5 - ribs));
    albedo = albedo.lerp(Vec3::new(0.70, 0.75, 0.82), snow);
    ridge_light(albedo, point, normal)
}

/// Bakes sun, sky light, and the aerial perspective of the far ring.
fn ridge_light(albedo: Vec3, point: Vec3, normal: Vec3) -> Vec3 {
    let p = point.xz();
    let height = point.y;
    let sun = normal.dot(SUN).max(0.0);
    let sky = 0.55 + 0.45 * normal.y;
    let lit = albedo * (Vec3::new(0.30, 0.36, 0.46) * sky + Vec3::new(1.0, 0.80, 0.62) * 1.6 * sun);

    // Aerial perspective: farther and higher ridges fade to a lighter blue, as the
    // camera's DistanceFog would at this range, with its glow toward the sun.
    let r = p.length();
    let view = (point - Vec3::Y * 10.0).normalize();
    let glow = view.dot(SUN).max(0.0).powi(12);
    let haze = Vec3::new(0.115, 0.158, 0.232).lerp(Vec3::new(0.30, 0.25, 0.22), 0.6 * glow);
    // Valley mist pools low, so ridge bases match the fogged heightmap in front.
    let mist = 1.0 - smoothstep(-10.0, 55.0, height);
    let aerial = 0.5 + 0.33 * smoothstep(470.0, 640.0, r) + 0.08 * smoothstep(40.0, 150.0, height);
    let low_haze = Vec3::new(0.21, 0.25, 0.31);
    let haze = haze.lerp(low_haze, 0.7 * mist);
    let aerial = aerial + 0.3 * mist;
    lit.lerp(haze, aerial.min(0.92))
}
