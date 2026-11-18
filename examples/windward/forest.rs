use bevy::prelude::*;

use bevy::camera::visibility::RenderLayers;
use bevy_spaghetti::{Spaghetti3d, SpaghettiAsset, SpaghettiLighting};

use crate::{
    conifer,
    detail::{CameraView, ScreenDetail},
    lake::REFLECTED_LAYER,
    landscape::{Obstacles, SUN, ground, hash01, mark_rock_shadow, noise, shore_distance, trail_x},
    player::HikerCamera,
};

const HEIGHTS: [f32; 3] = [8.93, 7.94, 5.76];
/// Crown widths of the three fir.glb scenes, which still cast the shadows.
const ATLAS_WIDTHS: [f32; 3] = [6.90, 5.82, 5.64];

/// Splat conifers: each height class has a fir and a hemlock.
const SPLAT_VARIANTS: usize = 6;
/// Minimum on-screen tree height in pixels for each finer level.
const LOD_PIXELS: [f32; conifer::LEVEL_COUNT - 1] = [340.0, 95.0, 36.0];
/// Crown radius at the lowest whorl, as a fraction of the impostor width.
const CROWN: f32 = 0.42;

#[derive(Component)]
pub struct ForestDetail {
    clouds: [Handle<SpaghettiAsset>; conifer::LEVEL_COUNT],
    height: f32,
    level: usize,
}

pub fn spawn_trees(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut obstacles: ResMut<Obstacles>,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
) {
    let splats: Vec<[Handle<SpaghettiAsset>; conifer::LEVEL_COUNT]> = (0..SPLAT_VARIANTS)
        .map(|variant| {
            let class = variant % HEIGHTS.len();
            let radius = ATLAS_WIDTHS[class] * CROWN;
            // Half the variants are hemlocks, which mix with fir in wet
            // Cascade forests.
            let habit = if variant >= HEIGHTS.len() {
                &conifer::HEMLOCK
            } else {
                &conifer::FIR
            };
            std::array::from_fn(|level| {
                let segments = conifer::fir(
                    variant as u32 * 101 + 17,
                    habit,
                    HEIGHTS[class],
                    radius,
                    level,
                );
                clouds.add(SpaghettiAsset::new(segments).expect("procedural firs are valid"))
            })
        })
        .collect();
    let lighting = SpaghettiLighting {
        direction_to_light: SUN,
        ambient: Vec3::new(0.24, 0.27, 0.29),
        directional: Vec3::new(0.62, 0.52, 0.40),
    };
    let firs: [Handle<WorldAsset>; 3] =
        [0, 1, 2].map(|i| server.load(GltfAssetLabel::Scene(i).from_asset("windward/fir.glb")));
    let stands = [
        (Vec2::new(-12.0, 4.0), Vec2::new(10.0, 17.0), 10),
        (Vec2::new(-17.0, -27.0), Vec2::new(13.0, 22.0), 15),
        (Vec2::new(-31.0, -38.0), Vec2::new(16.0, 25.0), 14),
        (Vec2::new(28.0, -20.0), Vec2::new(10.0, 22.0), 14),
        (Vec2::new(21.0, -47.0), Vec2::new(16.0, 19.0), 15),
        (Vec2::new(-9.0, -64.0), Vec2::new(25.0, 15.0), 17),
        (Vec2::new(15.0, 26.0), Vec2::new(25.0, 11.0), 12),
    ];
    let mut planted: Vec<Vec2> = Vec::new();
    for (stand, (center, radius, count)) in stands.into_iter().enumerate() {
        for index in 0..count {
            let seed = 199 + (stand as u32 * 31 + index) * 13;
            let angle = std::f32::consts::TAU * hash01(seed);
            let offset = Vec2::from_angle(angle) * radius * hash01(seed + 1).sqrt();
            let at = center + offset;
            let pick = (index as usize + 2 * stand) % SPLAT_VARIANTS;
            let variant = pick % HEIGHTS.len();
            let scale = 0.73 + 0.52 * hash01(seed + 2);
            // Splat limbs that cross the camera blur the whole frame, so keep
            // the lower crown clear of the trail.
            let clearance = 1.5 + CROWN * ATLAS_WIDTHS[variant] * scale * 0.85;
            if (at.x - trail_x(at.y)).abs() < clearance.max(3.25)
                || shore_distance(at) < 1.15
                || planted.iter().any(|p| p.distance_squared(at) < 6.25)
            {
                continue;
            }
            let detail = &splats[pick];
            commands
                .spawn((
                    Name::new("Conifer"),
                    placement(at, scale, seed),
                    Spaghetti3d(detail[2].clone()),
                    ForestDetail {
                        clouds: detail.clone(),
                        height: HEIGHTS[variant] * scale,
                        level: 2,
                    },
                    lighting,
                ))
                .with_children(|children| {
                    // The glTF crown only casts the shadow.
                    children
                        .spawn(WorldAssetRoot(firs[variant].clone()))
                        .observe(mark_rock_shadow);
                    children.spawn((
                        Name::new("Reflected conifer"),
                        Spaghetti3d(detail[conifer::LEVEL_COUNT - 1].clone()),
                        lighting,
                        RenderLayers::layer(REFLECTED_LAYER),
                    ));
                });
            planted.push(at);
            obstacles.add_tree(at, scale, HEIGHTS[variant]);
        }
    }

    // Sparse rows are visible as artificial gaps. Fill the outer hills with
    // clumps of low-detail trees while leaving the lakeshore and valley open.
    for seed in 1..=490u32 {
        let seed = seed * 233 + 773;
        let x = (hash01(seed) - 0.5) * 132.0;
        let z = -93.0 + 131.0 * hash01(seed + 1);
        let at = Vec2::new(x, z);
        if (x.abs() < 31.0 && z > -72.0 && z < 23.0)
            || (x - trail_x(z)).abs() < 3.9
            || noise(at * 0.092) < 0.39
            || planted.iter().any(|p| p.distance_squared(at) < 6.5)
        {
            continue;
        }
        let scale = 0.56 + 0.73 * hash01(seed + 2).powi(2);
        let pick = seed as usize % SPLAT_VARIANTS;
        let variant = pick % HEIGHTS.len();
        let detail = &splats[pick];
        let far = conifer::LEVEL_COUNT - 1;
        commands
            .spawn((
                Name::new("Distant conifer"),
                Spaghetti3d(detail[far].clone()),
                ForestDetail {
                    clouds: detail.clone(),
                    height: HEIGHTS[variant] * scale,
                    level: far,
                },
                lighting,
                placement(at, scale, seed),
            ))
            .with_children(|children| {
                children.spawn((
                    Spaghetti3d(detail[far].clone()),
                    lighting,
                    RenderLayers::layer(REFLECTED_LAYER),
                ));
            });
        planted.push(at);
        if x.abs() <= 40.5 && (-63.5..=25.5).contains(&z) {
            obstacles.add_tree(at, scale, HEIGHTS[variant]);
        }
    }
}

fn placement(at: Vec2, scale: f32, seed: u32) -> Transform {
    Transform::from_xyz(at.x, ground(at) - 0.075, at.y)
        .with_scale(Vec3::new(scale * 0.96, scale, scale * 0.96))
        .with_rotation(Quat::from_rotation_y(
            hash01(seed + 3) * std::f32::consts::TAU,
        ))
}

pub fn select_tree_detail(
    camera: Single<CameraView, With<HikerCamera>>,
    mut trees: Query<(&Transform, &mut ForestDetail, &mut Spaghetti3d)>,
) {
    let Some(view) = ScreenDetail::new(*camera) else {
        return;
    };
    for (transform, mut tree, mut cloud) in &mut trees {
        let center = transform.translation + Vec3::Y * tree.height * 0.52;
        let span = view.span_pixels(center, tree.height, tree.height * 0.37);
        // Hysteresis keeps the level stable as the shoulder camera bobs.
        let level = (0..LOD_PIXELS.len())
            .find(|&level| {
                let keep = if level < tree.level { 1.12 } else { 0.9 };
                span >= LOD_PIXELS[level] * keep
            })
            .unwrap_or(LOD_PIXELS.len());
        if level != tree.level {
            cloud.0 = tree.clouds[level].clone();
            tree.level = level;
        }
    }
}
