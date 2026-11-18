use bevy::prelude::*;

use bevy::camera::visibility::RenderLayers;
use bevy_spaghetti::{
    DetailTransition, GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiCutout,
    SpaghettiLighting, SpaghettiNearFade, SpaghettiReceiveShadows,
};

use crate::{
    conifer,
    lake::REFLECTED_LAYER,
    landscape::{Obstacles, SUN, ground, hash01, mark_rock_shadow, noise, shore_distance, trail_x},
};

const HEIGHTS: [f32; 3] = [8.93, 7.94, 5.76];
/// Crown widths of the three fir.glb scenes, which still cast the shadows.
const ATLAS_WIDTHS: [f32; 3] = [6.90, 5.82, 5.64];

/// Minimum on-screen tree height in pixels for each finer level.
const LOD_PIXELS: [f32; conifer::LEVEL_COUNT - 1] = [340.0, 95.0, 36.0];
/// Crown radius at the lowest whorl, as a fraction of the impostor width.
const CROWN: f32 = 0.42;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Living,
    Damaged,
    Snag,
    Old,
    Sapling,
}

/// One procedural tree shape, built once at every level of detail.
struct Variant {
    kind: Kind,
    habit: &'static conifer::Habit,
    form: conifer::Form,
    height: f32,
    radius: f32,
    /// The fir.glb scene that casts the shadow and its crown radius.
    shadow: Option<(usize, f32)>,
}

impl Variant {
    const fn new(
        kind: Kind,
        habit: &'static conifer::Habit,
        form: conifer::Form,
        height: f32,
        radius: f32,
        shadow: Option<(usize, f32)>,
    ) -> Self {
        Self {
            kind,
            habit,
            form,
            height,
            radius,
            shadow,
        }
    }

    /// Height of the highest wood or needles.
    fn top(&self) -> f32 {
        self.height * self.form.broken_at.min(1.0)
    }
}

const fn form(thin: f32, dead_above: f32, broken_at: f32, flag: f32) -> conifer::Form {
    conifer::Form {
        thin,
        dead_above,
        broken_at,
        flag,
        ..conifer::HEALTHY
    }
}

const fn snag(broken_at: f32, girth: f32) -> conifer::Form {
    conifer::Form {
        broken_at,
        girth,
        snag: true,
        ..conifer::HEALTHY
    }
}

const fn old(bole: f32, girth: f32, thin: f32) -> conifer::Form {
    conifer::Form {
        bole,
        girth,
        thin,
        ..conifer::HEALTHY
    }
}

const R: [f32; 3] = [
    ATLAS_WIDTHS[0] * CROWN,
    ATLAS_WIDTHS[1] * CROWN,
    ATLAS_WIDTHS[2] * CROWN,
];
const OK: conifer::Form = conifer::HEALTHY;
use Kind::*;
use conifer::{FIR, HEMLOCK, SUBALPINE};

const VARIANTS: [Variant; 20] = [
    // Grand fir and western hemlock in three sizes, then subalpine fir.
    Variant::new(Living, &FIR, OK, HEIGHTS[0], R[0], Some((0, R[0]))),
    Variant::new(Living, &FIR, OK, HEIGHTS[1], R[1], Some((1, R[1]))),
    Variant::new(Living, &FIR, OK, HEIGHTS[2], R[2], Some((2, R[2]))),
    Variant::new(Living, &HEMLOCK, OK, HEIGHTS[0], R[0], Some((0, R[0]))),
    Variant::new(Living, &HEMLOCK, OK, HEIGHTS[1], R[1], Some((1, R[1]))),
    Variant::new(Living, &HEMLOCK, OK, HEIGHTS[2], R[2], Some((2, R[2]))),
    Variant::new(Living, &SUBALPINE, OK, 8.6, 1.35, Some((1, 1.35))),
    Variant::new(Living, &SUBALPINE, OK, 6.2, 1.1, Some((2, 1.1))),
    // A wind-flagged fir, a hemlock with a dead top, a fir whose top broke
    // off, and a dying subalpine fir.
    Variant::new(
        Damaged,
        &FIR,
        form(0.12, 2.0, 1.0, 0.95),
        7.9,
        2.3,
        Some((1, 1.6)),
    ),
    Variant::new(
        Damaged,
        &HEMLOCK,
        form(0.05, 0.74, 1.0, 0.0),
        8.2,
        2.5,
        Some((1, 2.3)),
    ),
    Variant::new(
        Damaged,
        &FIR,
        form(0.08, 2.0, 0.7, 0.0),
        9.0,
        2.8,
        Some((0, 2.7)),
    ),
    Variant::new(
        Damaged,
        &SUBALPINE,
        form(0.38, 0.0, 1.0, 0.0),
        6.6,
        1.15,
        Some((2, 0.9)),
    ),
    // Snags: a tall broken fir, an intact grey spire, and an old stub.
    Variant::new(Snag, &FIR, snag(0.68, 1.45), 9.6, 1.8, Some((1, 0.32))),
    Variant::new(Snag, &SUBALPINE, snag(1.0, 1.15), 7.2, 0.9, Some((1, 0.22))),
    Variant::new(Snag, &HEMLOCK, snag(0.48, 1.7), 8.0, 1.6, Some((2, 0.38))),
    // Old trees with tall clear boles and ragged crowns.
    Variant::new(Old, &FIR, old(0.3, 1.6, 0.14), 12.2, 3.3, Some((0, 3.3))),
    Variant::new(
        Old,
        &HEMLOCK,
        old(0.26, 1.5, 0.1),
        11.2,
        3.1,
        Some((0, 3.1)),
    ),
    // Saplings in the sunny gaps and meadow edge.
    Variant::new(Sapling, &FIR, OK, 2.5, 1.0, Some((2, 1.0))),
    Variant::new(Sapling, &SUBALPINE, OK, 3.0, 0.62, Some((2, 0.62))),
    Variant::new(Sapling, &HEMLOCK, OK, 1.6, 0.78, None),
];

#[derive(Component)]
pub struct ForestDetail;

struct TreeClouds {
    adaptive: Handle<SpaghettiAsset>,
    reflection: Handle<SpaghettiAsset>,
}

fn tree_clouds(
    clouds: &mut Assets<SpaghettiAsset>,
    height: f32,
    levels: [Vec<GaussianSegment>; conifer::LEVEL_COUNT],
) -> TreeClouds {
    let reflection = clouds.add(
        SpaghettiAsset::new(levels[conifer::LEVEL_COUNT - 1].clone())
            .and_then(|asset| asset.bake_occlusion_with_density(SUN, 0.30))
            .expect("procedural firs are valid"),
    );
    let mut finer_end = 0;
    let transitions: [_; LOD_PIXELS.len()] = std::array::from_fn(|level| {
        finer_end += levels[level].len() as u32;
        let pixels = LOD_PIXELS[level] / height;
        DetailTransition {
            finer_end,
            pixels_per_unit: [0.8 * pixels, 1.2 * pixels],
        }
    });
    let adaptive = clouds.add(
        SpaghettiAsset::with_detail(levels.into_iter().flatten().collect(), &transitions)
            .and_then(|asset| asset.bake_occlusion_with_density(SUN, 0.30))
            .expect("procedural firs have valid detail levels"),
    );
    TreeClouds {
        adaptive,
        reflection,
    }
}

/// Picks a variant of `kind`, and for living trees a species and size.
fn choose(kind: Kind, seed: u32) -> usize {
    let roll = hash01(seed + 8);
    if kind == Living {
        let size = (hash01(seed + 9) * 3.0) as usize % 3;
        return match roll {
            r if r < 0.36 => size,
            r if r < 0.64 => 3 + size,
            _ => 6 + size % 2,
        };
    }
    let options: Vec<usize> = (0..VARIANTS.len())
        .filter(|&v| VARIANTS[v].kind == kind)
        .collect();
    options[(roll * options.len() as f32) as usize % options.len()]
}

/// Per-tree color: some crowns are yellower from stress, some bluer.
fn tint(lighting: SpaghettiLighting, seed: u32, kind: Kind) -> SpaghettiLighting {
    let bright = 0.84 + 0.3 * hash01(seed + 11);
    let hue = if kind == Snag {
        Vec3::ONE
    } else if kind == Sapling {
        // Open-grown young trees are lighter and yellower.
        Vec3::new(1.3, 1.25, 1.0)
    } else {
        Vec3::new(1.07, 1.03, 0.84).lerp(Vec3::new(0.92, 0.99, 1.12), hash01(seed + 12))
    };
    SpaghettiLighting {
        ambient: lighting.ambient * hue * bright,
        directional: lighting.directional * hue * bright,
        ..lighting
    }
}

pub fn spawn_trees(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut obstacles: ResMut<Obstacles>,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
) {
    let splats: Vec<_> = VARIANTS
        .iter()
        .enumerate()
        .map(|(index, variant)| {
            let levels = std::array::from_fn(|level| {
                conifer::fir(
                    index as u32 * 101 + 17,
                    variant.habit,
                    &variant.form,
                    variant.height,
                    variant.radius,
                    level,
                )
            });
            tree_clouds(&mut clouds, variant.top(), levels)
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
    let spawn = |commands: &mut Commands,
                 planted: &mut Vec<Vec2>,
                 at: Vec2,
                 scale: f32,
                 seed: u32,
                 pick: usize,
                 level: usize| {
        let variant = &VARIANTS[pick];
        let detail = &splats[pick];
        let lighting = tint(lighting, seed, variant.kind);
        let mut transform = placement(at, scale, seed);
        if variant.form.flag > 0.0 {
            // Flagged crowns all point downwind, toward +X.
            transform.rotation = Quat::from_rotation_y((hash01(seed + 3) - 0.5) * 0.6);
        }
        let far = conifer::LEVEL_COUNT - 1;
        let near = level < far;
        commands
            .spawn((
                Name::new(if near { "Conifer" } else { "Distant conifer" }),
                transform,
                Spaghetti3d(detail.adaptive.clone()),
                ForestDetail,
                lighting,
                SpaghettiReceiveShadows,
                // Limbs between the follow camera and the hiker dissolve
                // instead of smearing across the frame.
                SpaghettiNearFade {
                    start: 1.0,
                    end: 3.4,
                },
                // Depth-writing cutouts skip the per-view sort and let the
                // depth test reject hidden needles.
                SpaghettiCutout,
            ))
            .with_children(|children| {
                if near && let Some((scene, radius)) = variant.shadow {
                    // The glTF crown only casts the shadow. Snags use a
                    // squeezed crown, which reads as a thin stem.
                    let across = radius / (ATLAS_WIDTHS[scene] * CROWN);
                    children
                        .spawn((
                            WorldAssetRoot(firs[scene].clone()),
                            Transform::from_scale(Vec3::new(
                                across,
                                variant.top() / HEIGHTS[scene],
                                across,
                            )),
                        ))
                        .observe(mark_rock_shadow);
                }
                children.spawn((
                    Name::new("Reflected conifer"),
                    Spaghetti3d(detail.reflection.clone()),
                    lighting,
                    SpaghettiReceiveShadows,
                    RenderLayers::layer(REFLECTED_LAYER),
                ));
            });
        planted.push(at);
        (variant.height, variant.radius)
    };

    for (stand, (center, radius, count)) in stands.into_iter().enumerate() {
        // Trees grow in clumps that shelter each other, with gaps between.
        let clumps = (count as usize).div_ceil(3);
        let clump_centers: Vec<Vec2> = (0..clumps as u32)
            .map(|clump| {
                let seed = 7333 + (stand as u32 * 17 + clump) * 7;
                let angle = std::f32::consts::TAU * hash01(seed);
                center + Vec2::from_angle(angle) * radius * 0.85 * hash01(seed + 1).sqrt()
            })
            .collect();
        for index in 0..count {
            let seed = 199 + (stand as u32 * 31 + index) * 13;
            let angle = std::f32::consts::TAU * hash01(seed);
            let at = if hash01(seed + 6) < 0.2 {
                center + Vec2::from_angle(angle) * radius * hash01(seed + 1).sqrt()
            } else {
                clump_centers[index as usize % clumps]
                    + Vec2::from_angle(angle) * 3.0 * hash01(seed + 1).sqrt()
            };
            let trail = (at.x - trail_x(at.y)).abs();
            let kind = match hash01(seed + 7) {
                r if r < 0.1 => Snag,
                r if r < 0.21 => Damaged,
                r if r > 0.955 && trail > 9.0 => Old,
                _ => Living,
            };
            let pick = choose(kind, seed);
            let variant = &VARIANTS[pick];
            let scale = if kind == Old {
                0.92 + 0.16 * hash01(seed + 2)
            } else {
                0.73 + 0.52 * hash01(seed + 2)
            };
            // Splat limbs that cross the camera blur the whole frame, so keep
            // the lower crown clear of the trail.
            let clearance = 1.5 + variant.radius * scale * 0.85;
            if trail < clearance.max(3.25)
                || shore_distance(at) < 1.15
                || planted.iter().any(|p| p.distance_squared(at) < 3.4)
            {
                continue;
            }
            let (height, _) = spawn(&mut commands, &mut planted, at, scale, seed, pick, 2);
            add_obstacle(&mut obstacles, at, scale, height);
        }
    }

    // Saplings fill the sunny stand edges and the meadow fringe.
    let stand_trees = planted.clone();
    for seed in 1..=150u32 {
        let seed = seed * 389 + 9157;
        let (center, radius, _) = stands[seed as usize % stands.len()];
        let angle = std::f32::consts::TAU * hash01(seed);
        let at = center + Vec2::from_angle(angle) * radius * (0.95 + 0.6 * hash01(seed + 1));
        let nearest = stand_trees
            .iter()
            .map(|p| p.distance(at))
            .fold(f32::MAX, f32::min);
        let pick = choose(Sapling, seed);
        let scale = 0.7 + 0.6 * hash01(seed + 2);
        let clearance = 2.6 + VARIANTS[pick].radius * scale;
        if !(2.2..10.0).contains(&nearest)
            || (at.x - trail_x(at.y)).abs() < clearance
            || shore_distance(at) < 1.0
            || planted.iter().any(|p| p.distance_squared(at) < 2.0)
            || hash01(seed + 4) < 0.45
        {
            continue;
        }
        let (height, _) = spawn(&mut commands, &mut planted, at, scale, seed, pick, 2);
        add_obstacle(&mut obstacles, at, scale, height);
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
        let kind = match hash01(seed + 7) {
            r if r < 0.07 => Snag,
            r if r < 0.14 => Damaged,
            _ => Living,
        };
        let pick = choose(kind, seed);
        let (height, _) = spawn(
            &mut commands,
            &mut planted,
            at,
            scale,
            seed,
            pick,
            conifer::LEVEL_COUNT - 1,
        );
        if x.abs() <= 40.5 && (-63.5..=25.5).contains(&z) {
            add_obstacle(&mut obstacles, at, scale, height);
        }
    }
}

/// Registers a trunk and crown, scaled so saplings block less than old trees.
fn add_obstacle(obstacles: &mut Obstacles, at: Vec2, scale: f32, height: f32) {
    let size = (height / HEIGHTS[1]).clamp(0.4, 1.3);
    obstacles.add_tree(at, scale * size, height / size);
}

fn placement(at: Vec2, scale: f32, seed: u32) -> Transform {
    Transform::from_xyz(at.x, ground(at) - 0.075, at.y)
        .with_scale(Vec3::new(scale * 0.96, scale, scale * 0.96))
        .with_rotation(Quat::from_rotation_y(
            hash01(seed + 3) * std::f32::consts::TAU,
        ))
}
