use std::{
    collections::BTreeMap,
    f32::consts::{FRAC_PI_2, TAU},
};

use bevy::{
    camera::visibility::{RenderLayers, VisibilityRange},
    prelude::*,
};
use bevy_spaghetti::{
    GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiCastShadows, SpaghettiCutout,
    SpaghettiLighting, SpaghettiReceiveShadows,
};

use crate::{
    detail::{CameraView, ScreenDetail, load_lit_cloud},
    lake::REFLECTED_LAYER,
    landscape::{SUN, WATER_LEVEL, ground, hash01, noise, trail_x},
    props,
};

/// Poly Haven "Dead Tree Trunk" (CC0), converted with spaghetti-convert at 30k, 9k, and 2.5k samples.
const TRUNK_CLOUDS: [&str; 3] = [
    "windward/deadfall.spag",
    "windward/deadfall-mid.spag",
    "windward/deadfall-far.spag",
];

/// Near the ground, bounced light from the grass and soil is less blue
/// than the open sky. Keep shaded bark visible between direct sun patches.
const LOG_LIGHTING: SpaghettiLighting = SpaghettiLighting {
    direction_to_light: SUN,
    ambient: Vec3::new(0.43, 0.44, 0.42),
    directional: Vec3::new(0.62, 0.52, 0.40),
};

/// Gravel is darker than the rocks' lichen albedo, so these gains match the
/// brightness of the trail texture around it.
const LIGHTING: SpaghettiLighting = SpaghettiLighting {
    direction_to_light: SUN,
    ambient: Vec3::new(0.33, 0.37, 0.42),
    directional: Vec3::new(0.86, 0.73, 0.56),
};

#[derive(Component)]
pub struct LogDetail {
    clouds: [Handle<SpaghettiAsset>; 3],
    level: usize,
}

pub fn spawn_debris(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut splats: ResMut<Assets<SpaghettiAsset>>,
) {
    let mut fallen = vec![
        (Vec2::new(-3.65, 7.4), 0.23, 3.4, 0.22),
        (Vec2::new(4.65, 5.6), -0.67, 1.85, 0.13),
        (Vec2::new(-6.6, -9.1), -0.80, 3.0, 0.22),
        (Vec2::new(17.8, -30.1), 0.54, 3.8, 0.24),
        (Vec2::new(-8.7, -28.0), -1.27, 3.3, 0.20),
        (Vec2::new(8.3, -38.9), 0.87, 3.4, 0.29),
        (Vec2::new(-11.6, -44.7), 0.21, 2.8, 0.20),
    ];
    fallen.extend(stand_logs(&fallen));
    // The scan is 3.05 m long and about 0.28 m thick, lying along +X.
    let clouds = TRUNK_CLOUDS.map(|path| load_lit_cloud(&server, path));
    for (index, (at, angle, length, radius)) in fallen.into_iter().enumerate() {
        let girth = radius / 0.14;
        let roll = hash01(197 + index as u32 * 15) * TAU;
        let ends = [
            at - Vec2::from_angle(angle) * length * 0.5,
            at + Vec2::from_angle(angle) * length * 0.5,
        ];
        let lowest = ground(ends[0]).min(ground(ends[1])).min(ground(at));
        commands
            .spawn((
                Name::new("Weathered deadfall"),
                Spaghetti3d(clouds[1].clone()),
                SpaghettiCutout,
                LogDetail {
                    clouds: clouds.clone(),
                    level: 1,
                },
                LOG_LIGHTING,
                SpaghettiReceiveShadows,
                SpaghettiCastShadows,
                Transform::from_xyz(at.x, lowest + 0.02 * girth, at.y)
                    .with_rotation(
                        Quat::from_rotation_y(-angle) * Quat::from_rotation_x(roll * 0.15),
                    )
                    .with_scale(Vec3::new(length / 3.05, girth, girth)),
            ))
            .with_children(|parent| {
                parent.spawn((
                    Name::new("Reflected deadfall"),
                    Spaghetti3d(clouds[2].clone()),
                    // The rippled lake blurs the reflection, so it skips the sorted path.
                    SpaghettiCutout,
                    LOG_LIGHTING,
                    SpaghettiReceiveShadows,
                    RenderLayers::layer(REFLECTED_LAYER),
                ));
            });
    }

    let mut patches: BTreeMap<(i32, i32), Vec<GaussianSegment>> = BTreeMap::new();
    for seed in 1..=3200 {
        let id = seed * 593;
        let z = -62.0 + 83.0 * hash01(id);
        let x = trail_x(z) + (hash01(id + 1) - 0.5) * 6.0;
        let at = Vec2::new(x, z);
        if ground(at) < WATER_LEVEL + 0.055 || hash01(id + 2) > 0.27 + 0.59 * noise(at * 0.49) {
            continue;
        }
        let cell = (at / 9.0).floor().as_ivec2();
        pebble(patches.entry((cell.x, cell.y)).or_default(), at, id);
    }
    for patch in patches.into_values() {
        commands.spawn((
            Name::new("Path gravel"),
            Spaghetti3d(splats.add(
                SpaghettiAsset::new(patch)
                    .and_then(|asset| asset.bake_occlusion_with_density(SUN, 0.16))
                    .expect("valid path gravel"),
            )),
            SpaghettiCutout,
            LIGHTING,
            SpaghettiReceiveShadows,
            Transform::IDENTITY,
        ));
    }
    spawn_twigs(&mut commands, &mut splats);
}

fn spawn_twigs(commands: &mut Commands, splats: &mut Assets<SpaghettiAsset>) {
    const CELL: f32 = 8.0;
    let forest = crate::forest::forest();
    let mut patches: BTreeMap<(i32, i32), Vec<GaussianSegment>> = BTreeMap::new();
    for (index, tree) in forest
        .trees
        .iter()
        .enumerate()
        .filter(|(_, tree)| tree.is_stand())
    {
        for twig in 0..9 {
            let seed = 48_913 + 173 * index as u32 + 13 * twig;
            let across = Vec2::from_angle(TAU * hash01(seed));
            let at = tree.at + across * tree.crown().max(0.9) * (0.42 + 0.80 * hash01(seed + 1));
            if (at.x - trail_x(at.y)).abs() < 1.8 || ground(at) < WATER_LEVEL + 0.27 {
                continue;
            }
            let cell = (at / CELL).floor().as_ivec2();
            let origin = cell.as_vec2() * CELL;
            fallen_twig(
                patches.entry((cell.x, cell.y)).or_default(),
                at,
                origin,
                seed,
            );
        }
    }
    for ((x, z), segments) in patches {
        commands.spawn((
            Name::new("Fallen twigs"),
            Spaghetti3d(splats.add(SpaghettiAsset::new(segments).expect("valid twigs"))),
            SpaghettiCutout,
            LIGHTING,
            SpaghettiReceiveShadows,
            Transform::from_xyz(x as f32 * CELL, 0.0, z as f32 * CELL),
            VisibilityRange {
                start_margin: 0.0..0.0,
                end_margin: 13.0..18.0,
                use_aabb: true,
            },
        ));
    }
}

fn fallen_twig(patch: &mut Vec<GaussianSegment>, at: Vec2, origin: Vec2, seed: u32) {
    let along = Vec2::from_angle(TAU * hash01(seed + 2));
    let side = along.perp() * (2.0 * hash01(seed + 3) - 1.0).signum();
    let length = 0.13 + 0.52 * hash01(seed + 4).powi(2);
    let width = 0.0055 + 0.012 * length;
    let color = Vec3::new(0.16, 0.095, 0.054).lerp(Vec3::new(0.30, 0.22, 0.14), hash01(seed + 5));
    let point = |point: Vec2, relief: f32| {
        let local = point - origin;
        Vec3::new(local.x, ground(point) + relief * width, local.y)
    };
    let root = point(at - along * (0.46 * length), 0.59);
    let bend = point(at + side * (0.09 * length), 1.5);
    let tip = point(at + along * (0.54 * length) + side * (0.14 * length), 0.75);
    let mut stick = |a: Vec3, b: Vec3, sigma: f32, taper: f32, tint| {
        let along = (b - a).normalize();
        let side = Vec3::Y.cross(along).normalize();
        let rotation =
            Quat::from_mat3(&Mat3::from_cols(along, side, along.cross(side))).normalize();
        patch.push(
            GaussianSegment::new(
                a,
                b,
                Vec3::splat(sigma),
                rotation,
                5.4 * a.distance(b) * sigma,
                color * tint,
            )
            .with_cutout_taper(taper),
        );
    };
    stick(root, bend, width, 0.67, 0.88);
    stick(bend, tip, 0.67 * width, 0.24, 1.03);
    if length > 0.25 {
        let fork = point(at + (0.28 * along + 0.37 * side) * length, 0.60);
        stick(bend, fork, 0.53 * width, 0.19, 0.85);
    }
}

/// Finds a few clear spots for small windfall logs under the closed canopy,
/// away from the trail, the lake, the trunks, and the other logs.
fn stand_logs(fixed: &[(Vec2, f32, f32, f32)]) -> Vec<(Vec2, f32, f32, f32)> {
    let forest = crate::forest::forest();
    let mut logs: Vec<(Vec2, f32, f32, f32)> = Vec::new();
    for seed in 0..400u32 {
        if logs.len() >= 9 {
            break;
        }
        let id = 8_803 + seed * 61;
        let at = Vec2::new(-34.0 + 68.0 * hash01(id), -66.0 + 92.0 * hash01(id + 1));
        let angle = TAU * hash01(id + 2);
        let length = 1.9 + 1.4 * hash01(id + 3);
        let radius = 0.11 + 0.07 * hash01(id + 4);
        let half = Vec2::from_angle(angle) * length * 0.5;
        let (a, b) = (at - half, at + half);
        let samples = [a, at, b];
        let clear =
            |point: Vec2, gap: f32| samples.iter().all(|&sample| sample.distance(point) >= gap);
        let all = fixed.iter().chain(logs.iter());
        if samples.iter().any(|&p| {
            forest.shade(p) < 0.6
                || (p.x - trail_x(p.y)).abs() < 3.0
                || crate::landscape::shore_distance(p) < 1.3
                || ground(p) < WATER_LEVEL + 0.2
        }) || (ground(a) - ground(b)).abs() > 0.35
            || !forest
                .trees
                .iter()
                .all(|tree| props::segment_distance(tree.at, a, b) >= 0.55)
            || !crate::landscape::ROCKS
                .iter()
                .all(|&(x, z, scale)| clear(Vec2::new(x, z), 1.2 * scale + 0.4))
            || !all
                .into_iter()
                .all(|&(other, ..)| other.distance(at) >= 4.0)
        {
            continue;
        }
        logs.push((at, angle, length, radius));
    }
    logs
}

/// Builds one pebble from a flattened body kernel and two tilted facets, so
/// the sun shades its crown and flanks differently.
fn pebble(patch: &mut Vec<GaussianSegment>, at: Vec2, seed: u32) {
    let radius = 0.010 + 0.060 * hash01(seed + 3).powf(3.4);
    let height = radius * (0.25 + 0.58 * hash01(seed + 4));
    let turn = hash01(seed + 5) * TAU;
    let tone = hash01(seed + 6);
    let stone = Vec3::new(0.13, 0.12, 0.105).lerp(Vec3::new(0.34, 0.31, 0.27), tone);
    let base = Vec3::new(at.x, ground(at) - height * 0.21, at.y);
    let long = Vec3::new(turn.cos(), 0.0, -turn.sin());
    let wide = Vec3::Y.cross(long);
    // Local +Z of each kernel is its surface normal.
    let lay = |normal: Vec3| {
        let flat = Quat::from_rotation_y(turn) * Quat::from_rotation_x(-FRAC_PI_2);
        Quat::from_rotation_arc(Vec3::Y, normal.normalize()) * flat
    };
    let kernel = |center: Vec3, half: f32, sigma: Vec3, normal: Vec3, color: Vec3| {
        let mass = 3.2 * (TAU * sigma.x * sigma.y + TAU.sqrt() * 2.0 * half * sigma.y);
        GaussianSegment::new(
            center - long * half,
            center + long * half,
            sigma,
            lay(normal),
            mass,
            color,
        )
    };
    let width = 0.75 * (0.85 + 0.3 * hash01(seed + 9));
    patch.push(kernel(
        base + Vec3::Y * height * 0.45,
        radius * 0.30,
        Vec3::new(0.42 * radius, 0.42 * radius * width, 0.12 * height),
        Vec3::Y,
        stone * (0.83 + 0.2 * hash01(seed + 15)),
    ));
    for (side, facet) in [(1.0, 16), (-1.0, 17)] {
        let lean = 0.35 + 0.4 * hash01(seed + facet);
        let normal = Vec3::Y + wide * side * lean + long * (hash01(seed + facet + 2) - 0.5) * 0.6;
        patch.push(kernel(
            base + Vec3::Y * height * 0.62 + wide * side * radius * width * 0.28,
            radius * 0.18,
            Vec3::new(0.30 * radius, 0.22 * radius * width, 0.10 * height),
            normal,
            stone * (0.86 + 0.26 * hash01(seed + facet + 4)),
        ));
    }
}

pub fn select_log_detail(
    camera: Single<CameraView, With<crate::player::HikerCamera>>,
    assets: Res<Assets<SpaghettiAsset>>,
    mut logs: Query<(&Transform, &mut LogDetail, &mut Spaghetti3d)>,
) {
    let Some(view) = ScreenDetail::new(*camera) else {
        return;
    };
    for (transform, mut detail, mut cloud) in &mut logs {
        let Some(asset) = assets.get(&detail.clouds[0]) else {
            continue;
        };
        let (low, high) = asset.bounds();
        let center = transform.transform_point((low + high) * 0.5);
        let size = ((high - low) * transform.scale).max_element();
        let pixels = view.span_pixels(center, size, 0.5 * size);
        // Kernel width grows with sample spacing, so a coarse level blurs the
        // bark before it shows gaps. Hysteresis keeps the density stable as
        // the shoulder camera bobs.
        let near = if detail.level == 0 { 180.0 } else { 200.0 };
        let middle = if detail.level == 2 { 90.0 } else { 80.0 };
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
