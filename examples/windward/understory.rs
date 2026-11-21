//! Ferns and broad-leaf groundcover along the trail.
//!
//! The plants are procedural splat clouds from `fern.rs`. The ferns cast
//! their own shadows from the fronds.

use bevy::prelude::*;
use bevy_spaghetti::{
    Spaghetti3d, SpaghettiAsset, SpaghettiCastShadows, SpaghettiLighting, SpaghettiSpecular,
    SpaghettiTransmission,
};

use crate::{
    detail::{CameraView, ScreenDetail},
    fern,
    landscape::{SUN, WATER_LEVEL, ground, hash01, noise, shore_distance, trail_x},
    player::{Hiker, HikerCamera},
};

/// Reach and height of each fern variant. These match the scenes in the
/// `fern_02.glb` model that the splat fronds replaced.
const FERN_SHAPES: [(f32, f32); 4] = [(0.48, 0.40), (0.43, 0.32), (0.30, 0.26), (0.31, 0.20)];

/// Reach and height of each broad-leaf variant. These match the scale of the
/// `weed_plant_02.glb` clumps they replace.
const LEAF_SHAPES: [(f32, f32); 5] = [
    (0.15, 0.064),
    (0.12, 0.066),
    (0.085, 0.053),
    (0.08, 0.040),
    (0.07, 0.037),
];

#[derive(Component)]
pub struct Understory {
    rest: Quat,
    flex: f32,
    phase: f32,
}

#[derive(Component)]
pub struct UnderstoryDetail {
    clouds: [Handle<SpaghettiAsset>; fern::LEVEL_COUNT],
    /// Unscaled width of the clump.
    size: f32,
    level: usize,
}

struct Variant {
    clouds: [Handle<SpaghettiAsset>; fern::LEVEL_COUNT],
    size: f32,
    /// Whether the plant is tall enough to cast a shadow.
    casts_shadow: bool,
}

pub fn spawn_understory(
    mut commands: Commands,
    mut assets: ResMut<Assets<SpaghettiAsset>>,
) {
    let mut cloud =
        |segments| assets.add(SpaghettiAsset::new(segments).expect("procedural plants are valid"));
    let ferns: Vec<Variant> = FERN_SHAPES
        .iter()
        .enumerate()
        .map(|(index, &(radius, height))| Variant {
            clouds: std::array::from_fn(|level| {
                cloud(fern::fern(index as u32 * 37 + 5, radius, height, level))
            }),
            size: 2.0 * radius,
            casts_shadow: true,
        })
        .collect();
    // The groundcover is a few centimeters tall and never cast a shadow.
    let leaves: Vec<Variant> = LEAF_SHAPES
        .iter()
        .enumerate()
        .map(|(index, &(radius, height))| Variant {
            clouds: std::array::from_fn(|level| {
                cloud(fern::groundcover(
                    index as u32 * 53 + 11,
                    radius,
                    height,
                    level,
                ))
            }),
            size: 2.0 * radius,
            casts_shadow: false,
        })
        .collect();
    for (index, at) in [
        Vec2::new(-2.4, 10.6),
        Vec2::new(-2.8, 8.8),
        Vec2::new(3.0, 6.3),
        Vec2::new(3.7, 5.7),
        Vec2::new(4.9, 4.3),
        Vec2::new(2.4, -2.7),
        Vec2::new(0.2, -6.5),
        Vec2::new(-2.7, -7.4),
        Vec2::new(-6.8, -12.3),
        Vec2::new(-7.5, -12.1),
        Vec2::new(-1.7, -18.0),
        Vec2::new(0.9, -20.5),
    ]
    .into_iter()
    .enumerate()
    {
        add_plant(
            &mut commands,
            &ferns[index % ferns.len()],
            at,
            index as u32 + 24,
            true,
            1.0,
        );
    }
    for seed in 1..174u32 {
        let id = seed * 31;
        let z = -62.0 + 83.0 * hash01(id);
        let trail = trail_x(z);
        let distance = 1.36 + 11.7 * hash01(id + 1).powi(2);
        let side = if seed.is_multiple_of(2) { -1.0 } else { 1.0 };
        let at = Vec2::new(trail + side * distance, z);
        if ground(at) < WATER_LEVEL + 0.13 || noise(at * 0.27) < 0.44 {
            continue;
        }
        let shaded = crate::forest::canopy(at) > 0.5;
        let fern = (shore_distance(at) < 1.5 || shaded) && hash01(id + 2) < 0.52;
        let variants = if fern { &ferns } else { &leaves };
        let variant = &variants[seed as usize % variants.len()];
        add_plant(&mut commands, variant, at, id, fern, 1.0);
        if !fern && hash01(id + 3) < 0.46 {
            let near = at + Vec2::from_angle(hash01(id + 4) * std::f32::consts::TAU) * 0.45;
            if (near.x - trail_x(near.y)).abs() > 1.16 && ground(near) >= WATER_LEVEL + 0.13 {
                add_plant(&mut commands, variant, near, id + 8, false, 1.0);
            }
        }
    }
    // Sword ferns and low groundcover in the shade of the stands, around the
    // drip line of each crown, where the lowest limbs no longer hide them.
    for (index, tree) in crate::forest::forest().trees.iter().enumerate() {
        let id = 70_001 + index as u32 * 41;
        let trail = (tree.at.x - trail_x(tree.at.y)).abs();
        if !tree.is_stand() || trail > 26.0 || hash01(id) > 0.62 {
            continue;
        }
        for plant in 0..5 {
            let seed = id + plant * 7;
            let angle = std::f32::consts::TAU * hash01(seed + 1);
            let reach = tree.crown() * (0.75 + 0.55 * hash01(seed + 2));
            let at = tree.at + Vec2::from_angle(angle) * reach;
            if (at.x - trail_x(at.y)).abs() < 1.5
                || ground(at) < WATER_LEVEL + 0.13
                || crate::forest::canopy(at) < 0.45
            {
                continue;
            }
            // Sword ferns grow larger in the shade than beside the trail.
            let fern = plant < 2 || hash01(seed + 3) < 0.3;
            let variants = if fern { &ferns } else { &leaves };
            let variant = &variants[(seed / 7) as usize % variants.len()];
            let size = if fern {
                1.35 + 0.4 * hash01(seed + 4)
            } else {
                1.2
            };
            add_plant(&mut commands, variant, at, seed, fern, size);
        }
    }
}

fn add_plant(
    commands: &mut Commands,
    variant: &Variant,
    at: Vec2,
    seed: u32,
    fern: bool,
    size: f32,
) {
    let scale = size
        * if fern {
            0.72 + 0.56 * hash01(seed)
        } else {
            0.50 + 0.66 * hash01(seed)
        };
    let rotation = Quat::from_rotation_y(std::f32::consts::TAU * hash01(seed + 1));
    // Understory sits in the forest shade, so it gets more sky than sun.
    let lighting = SpaghettiLighting {
        direction_to_light: SUN,
        ambient: Vec3::new(0.25, 0.28, 0.30),
        directional: Vec3::new(0.58, 0.49, 0.37),
    };
    let mut plant = commands.spawn((
        Name::new(if fern { "Fern" } else { "Woodland groundcover" }),
        Spaghetti3d(variant.clouds[1].clone()),
        UnderstoryDetail {
            clouds: variant.clouds.clone(),
            size: variant.size,
            level: 1,
        },
        lighting,
        // Thin fronds glow when the low sun is behind them, like the grass.
        SpaghettiTransmission(0.5),
        // Fern pinnae and salal leaves are waxy and catch the sun flecks.
        SpaghettiSpecular {
            strength: 0.07,
            shininess: 32.0,
            strand: 0.3,
        },
        Transform::from_xyz(at.x, ground(at) - 0.02, at.y)
            .with_scale(Vec3::splat(scale))
            .with_rotation(rotation),
        Understory {
            rest: rotation,
            flex: if fern { 0.060 } else { 0.096 },
            phase: hash01(seed + 7),
        },
    ));
    if variant.casts_shadow {
        plant.insert(SpaghettiCastShadows);
    }
}

pub fn select_understory_detail(
    camera: Single<CameraView, With<HikerCamera>>,
    mut plants: Query<(&Transform, &mut UnderstoryDetail, &mut Spaghetti3d)>,
) {
    let Some(view) = ScreenDetail::new(*camera) else {
        return;
    };
    for (transform, mut detail, mut cloud) in &mut plants {
        let size = detail.size * transform.scale.x;
        let center = transform.translation + Vec3::Y * 0.15 * size;
        let pixels = view.span_pixels(center, size, 0.5 * size);
        // Hysteresis keeps the level stable as the shoulder camera bobs.
        let near = if detail.level == 0 { 150.0 } else { 175.0 };
        let middle = if detail.level == 2 { 55.0 } else { 45.0 };
        let level = if pixels >= near {
            0
        } else if pixels >= middle {
            1
        } else {
            2
        };
        if level != detail.level {
            cloud.0 = detail.clouds[level].clone();
            detail.level = level;
        }
    }
}

pub fn bend_understory(
    time: Res<Time>,
    hiker: Single<&Transform, (With<Hiker>, Without<Understory>)>,
    mut plants: Query<(&Understory, &mut Transform)>,
) {
    let time = time.elapsed_secs();
    for (plant, mut transform) in &mut plants {
        let origin = transform.translation.xz();
        let away = origin - hiker.translation.xz();
        if away.length_squared() > 23.0_f32.powi(2) {
            continue;
        }
        let front = (origin.dot(Vec2::new(-0.29, 0.19)) + time * 1.8).sin();
        let pulse = (time * 2.6 + plant.phase * 5.0).sin();
        let contact = away.normalize_or_zero() * (1.0 - away.length()).max(0.0).powi(2);
        let sway = Vec2::new(0.82, 0.47) * plant.flex * (0.38 + 0.64 * front + 0.11 * pulse)
            + 0.37 * contact;
        transform.rotation = Quat::from_euler(EulerRot::XYZ, sway.y, 0.0, -sway.x) * plant.rest;
    }
}
