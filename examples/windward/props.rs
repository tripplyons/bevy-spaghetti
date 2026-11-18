//! Scanned forest-floor props: old conifer stumps in the stands and loose
//! stones along the trail shoulders and the lakeshore.

use std::f32::consts::TAU;

use bevy::{camera::visibility::RenderLayers, prelude::*};
use bevy_spaghetti::{Spaghetti3d, SpaghettiAsset, SpaghettiLighting};

use crate::{
    debris::LogDetail,
    detail::{CameraView, ScreenDetail},
    forest::ForestDetail,
    lake::REFLECTED_LAYER,
    landscape::{
        LAKE, LAKE_RADII, Obstacles, ROCKS, SUN, WATER_LEVEL, ground, hash01, mark_rock_shadow,
        noise, shore_distance, trail_x,
    },
    player::HikerCamera,
};

/// Poly Haven "Tree Stump 01" (CC0), kept as the shadow caster.
const STUMP: &str = "windward/stump/tree_stump_01_1k.gltf";
/// The same scan converted with spaghetti-convert at 30k, 9k, and 2.5k samples.
const STUMP_CLOUDS: [&str; 3] = [
    "windward/stump.spag",
    "windward/stump-mid.spag",
    "windward/stump-far.spag",
];
/// The scan's litter mound reaches 0.19 m below its origin about 0.6 to 0.9 m
/// from the trunk axis.
const STUMP_SKIRT: f32 = 0.17;
const STUMP_REACH: f32 = 0.8;

/// Rough positions for the stumps. Each one moves to the nearest clear spot.
const STUMP_ANCHORS: [Vec2; 8] = [
    Vec2::new(-6.2, 3.4),
    Vec2::new(-5.6, -3.2),
    Vec2::new(-9.6, -17.6),
    Vec2::new(-12.4, -25.6),
    Vec2::new(-9.2, -32.6),
    Vec2::new(-0.2, -37.6),
    Vec2::new(23.5, -6.5),
    Vec2::new(16.8, 4.6),
];

/// Poly Haven "Namaqualand Stones 01" (CC0), split into one glTF per stone
/// and centered on each stone's base. A base color factor of (0.66, 0.76,
/// 0.86) cools the desert tan toward wet basalt gray, and the red stone "a" is
/// left out. Sizes are the scans' bounds in meters.
const STONES: [(&str, Vec3); 4] = [
    ("b", Vec3::new(0.151, 0.084, 0.081)),
    ("c", Vec3::new(0.104, 0.057, 0.098)),
    ("d", Vec3::new(0.115, 0.037, 0.082)),
    ("e", Vec3::new(0.111, 0.077, 0.112)),
];

const LIGHTING: SpaghettiLighting = SpaghettiLighting {
    direction_to_light: SUN,
    ambient: Vec3::new(0.20, 0.235, 0.27),
    directional: Vec3::new(0.52, 0.44, 0.34),
};

/// Swaps a prop's splat cloud by its size on screen.
#[derive(Component)]
pub struct PropDetail {
    clouds: [Handle<SpaghettiAsset>; 3],
    /// Minimum on-screen spans in pixels for the near and middle clouds.
    near: f32,
    middle: f32,
    level: usize,
}

/// Places of the things already on the ground that props must not overlap.
struct Clearance {
    trees: Vec<Vec2>,
    /// Log centerlines as segment end points.
    logs: Vec<(Vec2, Vec2)>,
    props: Vec<(Vec2, f32)>,
}

impl Clearance {
    fn is_clear(&self, at: Vec2, radius: f32, tree_gap: f32) -> bool {
        self.trees
            .iter()
            .all(|tree| tree.distance(at) >= radius + tree_gap)
            && ROCKS
                .iter()
                .all(|&(x, z, scale)| Vec2::new(x, z).distance(at) >= radius + 1.05 * scale + 0.15)
            && self
                .logs
                .iter()
                .all(|&(a, b)| segment_distance(at, a, b) >= radius + 0.35)
            && self
                .props
                .iter()
                .all(|&(center, size)| center.distance(at) >= radius + size + 0.12)
    }
}

pub fn spawn_props(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut obstacles: ResMut<Obstacles>,
    trees: Query<&Transform, With<ForestDetail>>,
    logs: Query<&Transform, With<LogDetail>>,
) {
    let mut clearance = Clearance {
        trees: trees.iter().map(|t| t.translation.xz()).collect(),
        logs: logs
            .iter()
            .map(|t| {
                // The deadfall scan is 3.05 m long along its local +X axis.
                let half = (t.rotation * Vec3::X * 1.525 * t.scale.x).xz();
                (t.translation.xz() - half, t.translation.xz() + half)
            })
            .collect(),
        props: Vec::new(),
    };
    spawn_stumps(&mut commands, &server, &mut obstacles, &mut clearance);
    spawn_stones(&mut commands, &server, &mut obstacles, &mut clearance);
}

fn spawn_stumps(
    commands: &mut Commands,
    server: &AssetServer,
    obstacles: &mut Obstacles,
    clearance: &mut Clearance,
) {
    let shadow = server.load(GltfAssetLabel::Scene(0).from_asset(STUMP));
    let clouds = STUMP_CLOUDS.map(|path| server.load::<SpaghettiAsset>(path));
    for (index, anchor) in STUMP_ANCHORS.into_iter().enumerate() {
        let seed = 4100 + index as u32 * 37;
        let scale = 1.1 + 0.5 * hash01(seed);
        let reach = STUMP_REACH * scale;
        // Search outward from the anchor on a golden-angle spiral.
        let Some(at) = (0..40).map(|step| {
            let angle = step as f32 * 2.39996 + hash01(seed + 1) * TAU;
            anchor + Vec2::from_angle(angle) * 0.45 * (step as f32).sqrt()
        })
        .find(|&at| {
            (at.x - trail_x(at.y)).abs() >= 2.3 + reach
                && shore_distance(at) >= 1.22
                && ground(at) >= WATER_LEVEL + 0.25
                && clearance.is_clear(at, reach, 1.0)
        }) else {
            continue;
        };
        let floor = footprint_floor(at, reach);
        let transform = Transform::from_xyz(at.x, floor + STUMP_SKIRT * scale, at.y)
            .with_rotation(Quat::from_rotation_y(hash01(seed + 2) * TAU))
            .with_scale(Vec3::splat(scale));
        spawn_prop(
            commands,
            "Old-growth stump",
            transform,
            &clouds,
            &shadow,
            (178.0, 54.0),
        );
        clearance.props.push((at, reach));
        // The cut trunk is about 0.32 m in radius and 0.55 m tall.
        obstacles.add_boulder(at, 0.36 * scale, 0.55 * scale);
    }
}

fn spawn_stones(
    commands: &mut Commands,
    server: &AssetServer,
    obstacles: &mut Obstacles,
    clearance: &mut Clearance,
) {
    let kinds = STONES.map(|(name, size)| {
        let clouds = [
            format!("windward/stone-{name}.spag"),
            format!("windward/stone-{name}-mid.spag"),
            format!("windward/stone-{name}-far.spag"),
        ]
        .map(|path| server.load::<SpaghettiAsset>(path));
        let shadow = server.load(
            GltfAssetLabel::Scene(0).from_asset(format!("windward/stones/stone_{name}.gltf")),
        );
        (clouds, shadow, size)
    });
    let mut placed = 0;
    for seed in 1..=900u32 {
        if placed >= 34 {
            break;
        }
        let id = 7700 + seed * 211;
        // Alternate between the trail shoulders and the lakeshore.
        let at = if seed % 5 < 3 {
            let z = -58.0 + 73.0 * hash01(id);
            let side = if hash01(id + 1) < 0.5 { -1.0 } else { 1.0 };
            Vec2::new(trail_x(z) + side * (1.4 + 1.3 * hash01(id + 2)), z)
        } else {
            let angle = hash01(id) * TAU;
            LAKE + LAKE_RADII * Vec2::from_angle(angle) * (0.98 + 0.2 * hash01(id + 2))
        };
        // Clump the stones where the ground is rough.
        if hash01(id + 3) > 0.15 + 0.85 * noise(at * 0.37 + Vec2::splat(3.1)) {
            continue;
        }
        let shore = shore_distance(at);
        if ground(at) < WATER_LEVEL + 0.03 || shore < 1.0 || (seed % 5 >= 3 && shore > 1.16) {
            continue;
        }
        let kind = (hash01(id + 4) * STONES.len() as f32) as usize % STONES.len();
        let (clouds, shadow, size) = &kinds[kind];
        let length = 0.2 + 0.38 * hash01(id + 5).powi(2);
        let scale = length / size.x;
        let reach = 0.5 * length;
        if (at.x - trail_x(at.y)).abs() < 1.2 + reach || !clearance.is_clear(at, reach, 0.55) {
            continue;
        }
        let turn = hash01(id + 6) * TAU;
        let tilt = Quat::from_rotation_x((hash01(id + 7) - 0.5) * 0.24)
            * Quat::from_rotation_z((hash01(id + 8) - 0.5) * 0.24);
        // Bury the underside so the stone rests in the soil on any slope.
        let floor = footprint_floor(at, reach);
        let transform = Transform::from_xyz(at.x, floor - 0.25 * size.y * scale, at.y)
            .with_rotation(Quat::from_rotation_y(turn) * tilt)
            .with_scale(Vec3::splat(scale));
        spawn_prop(commands, "Loose stone", transform, clouds, shadow, (80.0, 30.0));
        clearance.props.push((at, reach));
        if length > 0.36 {
            obstacles.add_boulder(at, 0.4 * length, size.y * scale * 0.75);
        }
        placed += 1;
    }
}

fn spawn_prop(
    commands: &mut Commands,
    name: &'static str,
    transform: Transform,
    clouds: &[Handle<SpaghettiAsset>; 3],
    shadow: &Handle<WorldAsset>,
    (near, middle): (f32, f32),
) {
    commands
        .spawn((
            Name::new(name),
            Spaghetti3d(clouds[1].clone()),
            PropDetail {
                clouds: clouds.clone(),
                near,
                middle,
                level: 1,
            },
            LIGHTING,
            transform,
        ))
        .with_children(|parent| {
            // The glTF only casts the shadow.
            parent
                .spawn(WorldAssetRoot(shadow.clone()))
                .observe(mark_rock_shadow);
            parent.spawn((
                Name::new("Reflected prop"),
                Spaghetti3d(clouds[2].clone()),
                LIGHTING,
                RenderLayers::layer(REFLECTED_LAYER),
            ));
        });
}

/// Returns the lowest ground under a round footprint, so no edge floats.
fn footprint_floor(at: Vec2, reach: f32) -> f32 {
    (0..8)
        .map(|i| at + Vec2::from_angle(i as f32 * TAU / 8.0) * reach)
        .chain([at])
        .map(ground)
        .fold(f32::INFINITY, f32::min)
}

fn segment_distance(point: Vec2, a: Vec2, b: Vec2) -> f32 {
    let along = b - a;
    let t = ((point - a).dot(along) / along.length_squared().max(1e-6)).clamp(0.0, 1.0);
    point.distance(a + along * t)
}

pub fn select_prop_detail(
    camera: Single<CameraView, With<HikerCamera>>,
    assets: Res<Assets<SpaghettiAsset>>,
    mut props: Query<(&Transform, &mut PropDetail, &mut Spaghetti3d)>,
) {
    let Some(view) = ScreenDetail::new(*camera) else {
        return;
    };
    for (transform, mut detail, mut cloud) in &mut props {
        let Some(asset) = assets.get(&detail.clouds[0]) else {
            continue;
        };
        let (low, high) = asset.bounds();
        let center = transform.transform_point((low + high) * 0.5);
        let size = ((high - low) * transform.scale).xz().length() * 0.66;
        let pixels = view.span_pixels(center, size, 0.4 * size);
        // Hysteresis keeps the density stable as the shoulder camera bobs.
        let near = detail.near * if detail.level == 0 { 0.87 } else { 1.0 };
        let middle = detail.middle * if detail.level == 2 { 1.2 } else { 1.0 };
        let level = if pixels >= near {
            0
        } else if pixels >= middle {
            1
        } else {
            2
        };
        if level != detail.level && assets.contains(&detail.clouds[level]) {
            cloud.0 = detail.clouds[level].clone();
            detail.level = level;
        }
    }
}
