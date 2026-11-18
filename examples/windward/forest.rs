use bevy::prelude::*;

use bevy::{
    camera::visibility::{NoAutoAabb, RenderLayers},
    light::NotShadowCaster,
};
use bevy_spaghetti::{
    DetailTransition, GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiCutout,
    SpaghettiDetailScale, SpaghettiGroundAmbient, SpaghettiLighting, SpaghettiNearFade,
    SpaghettiReceiveShadows, SpaghettiSway, SpaghettiTransmission,
};

use crate::{
    bark, conifer,
    lake::REFLECTED_LAYER,
    landscape::{
        Obstacles, SHADOW_ONLY, SUN, ground, hash01, mark_rock_shadow, noise, reflected_layers,
        shore_distance, smoothstep, trail_x,
    },
};

const HEIGHTS: [f32; 3] = [8.93, 7.94, 5.76];
/// Crown widths of the three fir.glb scenes, which still cast the shadows.
const ATLAS_WIDTHS: [f32; 3] = [6.90, 5.82, 5.64];

/// Minimum on-screen tree height in pixels for each finer level.
const LOD_PIXELS: [f32; conifer::LEVEL_COUNT - 1] = [340.0, 95.0, 36.0];
/// Core trees stand behind the stand edges, so they switch to coarser
/// levels this many times sooner.
const CORE_DETAIL: f32 = 1.8;
/// Crown radius at the lowest whorl, as a fraction of the impostor width.
const CROWN: f32 = 0.42;
/// Thin needle sprays let a little of the low sun through from behind.
const NEEDLE_TRANSMISSION: SpaghettiTransmission = SpaghettiTransmission(0.28);

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
    Variant::new(Snag, &FIR, snag(0.68, 1.45), 9.6, 1.8, None),
    Variant::new(Snag, &SUBALPINE, snag(1.0, 1.15), 7.2, 0.9, None),
    Variant::new(Snag, &HEMLOCK, snag(0.48, 1.7), 8.0, 1.6, None),
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

#[derive(Component)]
pub struct CrownWind {
    reflection: Entity,
    velocity: Vec2,
    phase: f32,
    frequency: f32,
    exposure: f32,
}

pub fn bend_crowns(
    time: Res<Time<Fixed>>,
    mut crowns: Query<(&mut CrownWind, &mut SpaghettiSway)>,
    mut reflections: Query<&mut SpaghettiSway, Without<CrownWind>>,
) {
    let dt = time.delta_secs();
    let t = time.elapsed_secs();
    for (mut crown, mut sway) in &mut crowns {
        // Share the meadow's traveling wind front. The slower pulse and
        // each tree's spring keep neighboring crowns from moving in sync.
        let front = (crown.phase + 1.8 * t).sin();
        let pulse = (0.62 * crown.phase + 0.79 * t).sin();
        let crosswind = (1.32 * crown.phase + 2.2 * t).sin();
        let target = crown.exposure
            * (Vec2::new(0.82, 0.47) * (0.052 + 0.055 * front + 0.033 * pulse)
                + Vec2::new(-0.47, 0.82) * (0.031 * crosswind));
        let slope = sway.0.xz();
        let frequency = crown.frequency;
        let acceleration =
            (target - slope) * (frequency * frequency) - crown.velocity * (1.55 * frequency);
        crown.velocity += acceleration * dt;
        let slope = slope + crown.velocity * dt;
        sway.0 = Vec3::new(slope.x, 0.0, slope.y);
        if let Ok(mut reflected) = reflections.get_mut(crown.reflection) {
            *reflected = *sway;
        }
    }
}

struct TreeClouds {
    adaptive: Handle<SpaghettiAsset>,
    reflection: Handle<SpaghettiAsset>,
}

enum TreeModel {
    Crown(TreeClouds, bark::BarkMeshes),
    Snag(bark::BarkMeshes),
}

fn tree_clouds(
    clouds: &mut Assets<SpaghettiAsset>,
    height: f32,
    levels: [conifer::Fir; conifer::LEVEL_COUNT],
) -> TreeClouds {
    let reflection = clouds.add(
        SpaghettiAsset::new(levels[conifer::LEVEL_COUNT - 1].splats.clone())
            .and_then(|asset| asset.bake_occlusion_with_density(SUN, 0.30))
            .expect("procedural firs are valid"),
    );
    let crowns = levels
        .each_ref()
        .map(|level| &level.splats[level.trunk_end..]);
    let segments: Vec<GaussianSegment> = crowns.into_iter().flatten().copied().collect();
    let mut finer_end = 0;
    let transitions: [_; LOD_PIXELS.len()] = std::array::from_fn(|level| {
        finer_end += crowns[level].len() as u32;
        let pixels = LOD_PIXELS[level] / height;
        DetailTransition {
            finer_end,
            pixels_per_unit: [0.8 * pixels, 1.2 * pixels],
        }
    });
    let adaptive = clouds.add(
        SpaghettiAsset::with_detail(segments, &transitions)
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
fn crown_tint(seed: u32, kind: Kind) -> Vec3 {
    let bright = 0.84 + 0.3 * hash01(seed + 11);
    let hue = if kind == Sapling {
        // Open-grown young trees are lighter and yellower.
        Vec3::new(1.3, 1.25, 1.0)
    } else {
        Vec3::new(1.07, 1.03, 0.84).lerp(Vec3::new(0.92, 0.99, 1.12), hash01(seed + 12))
    };
    hue * bright
}

/// One planted tree, chosen before any entity exists so the ground, grass,
/// and understory can read the canopy without waiting for `spawn_trees`.
pub struct Tree {
    pub at: Vec2,
    pub scale: f32,
    seed: u32,
    pick: usize,
    /// Low-detail hill trees without a shadow caster.
    distant: bool,
    obstacle: bool,
    core: bool,
}

impl Tree {
    /// Radius of the needle-covered crown footprint on the ground.
    pub fn crown(&self) -> f32 {
        let variant = &VARIANTS[self.pick];
        let reach = match variant.kind {
            Snag => 0.45,
            Sapling => 0.6 * variant.radius,
            _ => 0.96 * variant.radius,
        };
        reach * self.scale
    }

    /// Stand trees that reach the review shots and the trail views.
    pub fn is_stand(&self) -> bool {
        !self.distant && VARIANTS[self.pick].kind != Sapling
    }
}

/// The planted trees and a coarse grid of their crowns.
pub struct Forest {
    pub trees: Vec<Tree>,
    cells: std::collections::HashMap<IVec2, Vec<(Vec2, f32)>>,
}

/// Wide enough that a crown's shade never reaches past the neighboring cell.
const CANOPY_CELL: f32 = 5.5;

impl Forest {
    /// 1 under the middle of a crown, 0 in the open. Overlapping crowns
    /// combine, so the floor of a closed stand stays dark between trunks.
    pub fn shade(&self, at: Vec2) -> f32 {
        let cell = (at / CANOPY_CELL).floor().as_ivec2();
        let mut open = 1.0;
        for y in -1..=1 {
            for x in -1..=1 {
                let Some(crowns) = self.cells.get(&(cell + IVec2::new(x, y))) else {
                    continue;
                };
                for &(center, radius) in crowns {
                    let distance = center.distance(at);
                    open *= smoothstep(0.55 * radius, 1.4 * radius, distance);
                }
            }
        }
        1.0 - open
    }
}

/// The forest layout, built once from the same pure placement rules.
pub fn forest() -> &'static Forest {
    static FOREST: std::sync::OnceLock<Forest> = std::sync::OnceLock::new();
    FOREST.get_or_init(|| {
        let trees = layout();
        let mut cells: std::collections::HashMap<IVec2, Vec<(Vec2, f32)>> = Default::default();
        for tree in &trees {
            let cell = (tree.at / CANOPY_CELL).floor().as_ivec2();
            cells.entry(cell).or_default().push((tree.at, tree.crown()));
        }
        Forest { trees, cells }
    })
}

/// Shorthand for `forest().shade(at)`.
pub fn canopy(at: Vec2) -> f32 {
    forest().shade(at)
}

const STANDS: [(Vec2, Vec2, u32); 7] = [
    (Vec2::new(-12.0, 4.0), Vec2::new(10.0, 17.0), 10),
    (Vec2::new(-17.0, -27.0), Vec2::new(13.0, 22.0), 15),
    (Vec2::new(-31.0, -38.0), Vec2::new(16.0, 25.0), 14),
    (Vec2::new(28.0, -20.0), Vec2::new(10.0, 22.0), 14),
    (Vec2::new(21.0, -47.0), Vec2::new(16.0, 19.0), 15),
    (Vec2::new(-9.0, -64.0), Vec2::new(25.0, 15.0), 17),
    (Vec2::new(15.0, 26.0), Vec2::new(25.0, 11.0), 12),
];

/// Hiker spots of the review shots, with the camera yaw. The added core
/// trees keep out of the hiker's spot and the camera's.
const REVIEW_SPOTS: [(Vec2, f32); 6] = [
    (Vec2::new(-0.6, -7.0), -0.78),
    (Vec2::new(-5.7, -29.0), 0.18),
    (Vec2::new(-9.5, -22.0), 0.9),
    (Vec2::new(-8.0, -31.0), 2.4),
    (Vec2::new(-9.5, -22.0), -0.15),
    (Vec2::new(28.0, -38.0), -0.15),
];

fn near_review_spot(at: Vec2, reach: f32) -> bool {
    REVIEW_SPOTS.iter().any(|&(spot, yaw)| {
        // The follow camera sits about 5 m behind the hiker.
        let (sin, cos) = yaw.sin_cos();
        let eye = spot + Vec2::new(0.83 * cos + 4.95 * sin, -0.83 * sin + 4.95 * cos);
        let between = (0..=4).any(|step| {
            let point = spot.lerp(eye, step as f32 / 4.0);
            point.distance(at) < reach + 1.4
        });
        between || spot.distance(at) < reach + 2.2
    })
}

fn layout() -> Vec<Tree> {
    let mut trees: Vec<Tree> = Vec::new();
    let mut planted: Vec<Vec2> = Vec::new();
    let plant = |trees: &mut Vec<Tree>, planted: &mut Vec<Vec2>, tree: Tree| {
        planted.push(tree.at);
        trees.push(tree);
    };

    for (stand, (center, radius, count)) in STANDS.into_iter().enumerate() {
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
            plant(
                &mut trees,
                &mut planted,
                Tree {
                    at,
                    scale,
                    seed,
                    pick,
                    distant: false,
                    obstacle: true,
                    core: false,
                },
            );
        }
    }

    // Closed cores: tight clumps of overlapping crowns near each stand's
    // center, thinning out toward its edge.
    for (stand, (center, radius, count)) in STANDS.into_iter().enumerate() {
        let extra = count * 3;
        let clumps = extra.div_ceil(5);
        let clump_centers: Vec<Vec2> = (0..clumps)
            .map(|clump| {
                let seed = 41_017 + (stand as u32 * 29 + clump) * 11;
                let angle = std::f32::consts::TAU * hash01(seed);
                center + Vec2::from_angle(angle) * radius * 0.62 * hash01(seed + 1).sqrt()
            })
            .collect();
        for index in 0..extra {
            let seed = 52_361 + (stand as u32 * 67 + index) * 17;
            let angle = std::f32::consts::TAU * hash01(seed);
            let at = clump_centers[index as usize % clumps as usize]
                + Vec2::from_angle(angle) * 2.9 * hash01(seed + 1).sqrt();
            let edge = ((at - center) / radius).length();
            if hash01(seed + 5) > 1.0 - smoothstep(0.35, 1.0, edge) {
                continue;
            }
            let trail = (at.x - trail_x(at.y)).abs();
            let kind = match hash01(seed + 7) {
                r if r < 0.08 => Snag,
                r if r < 0.17 => Damaged,
                r if r > 0.95 && trail > 9.0 => Old,
                _ => Living,
            };
            let pick = choose(kind, seed);
            let variant = &VARIANTS[pick];
            let scale = if kind == Old {
                0.92 + 0.16 * hash01(seed + 2)
            } else {
                0.8 + 0.45 * hash01(seed + 2)
            };
            let reach = variant.radius * scale * 0.85;
            if trail < (1.5 + reach).max(3.25)
                || shore_distance(at) < 1.15
                || near_review_spot(at, reach)
                || planted.iter().any(|p| p.distance_squared(at) < 2.9)
            {
                continue;
            }
            plant(
                &mut trees,
                &mut planted,
                Tree {
                    at,
                    scale,
                    seed,
                    pick,
                    distant: false,
                    obstacle: true,
                    core: true,
                },
            );
        }
    }

    // Saplings fill the sunny stand edges and the meadow fringe.
    let stand_trees = planted.clone();
    for seed in 1..=150u32 {
        let seed = seed * 389 + 9157;
        let (center, radius, _) = STANDS[seed as usize % STANDS.len()];
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
        plant(
            &mut trees,
            &mut planted,
            Tree {
                at,
                scale,
                seed,
                pick,
                distant: false,
                obstacle: true,
                core: false,
            },
        );
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
        let obstacle = x.abs() <= 40.5 && (-63.5..=25.5).contains(&z);
        plant(
            &mut trees,
            &mut planted,
            Tree {
                at,
                scale,
                seed,
                pick,
                distant: true,
                obstacle,
                core: false,
            },
        );
    }
    trees
}

pub fn spawn_trees(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut obstacles: ResMut<Obstacles>,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let whorls = meshes.add(whorl_cones());
    let shadow_material = materials.add(StandardMaterial {
        cull_mode: None,
        ..default()
    });
    let bark_material = bark::bark_material(&server);
    let living_bark = materials.add(StandardMaterial {
        base_color: Color::linear_rgb(0.52, 0.39, 0.29),
        ..bark_material.clone()
    });
    let snag_bark = materials.add(bark_material);
    let models: Vec<_> = VARIANTS
        .iter()
        .enumerate()
        .map(|(index, variant)| {
            let seed = index as u32 * 101 + 17;
            if variant.kind == Snag {
                return TreeModel::Snag(bark::BarkMeshes::snag(
                    &mut meshes,
                    seed,
                    variant.height,
                    variant.top(),
                    variant.radius,
                    variant.form.girth,
                ));
            }
            let levels = std::array::from_fn(|level| {
                conifer::fir(
                    seed,
                    variant.habit,
                    &variant.form,
                    variant.height,
                    variant.radius,
                    level,
                )
            });
            let stem = conifer::Stem::for_tree(seed, variant.height, &variant.form);
            TreeModel::Crown(
                tree_clouds(&mut clouds, variant.top(), levels),
                bark::BarkMeshes::living(&mut meshes, seed, stem),
            )
        })
        .collect();
    let firs: [Handle<WorldAsset>; 3] =
        [0, 1, 2].map(|i| server.load(GltfAssetLabel::Scene(i).from_asset("windward/fir.glb")));
    for tree in &forest().trees {
        let &Tree {
            at,
            scale,
            seed,
            pick,
            distant,
            obstacle,
            core,
        } = tree;
        let variant = &VARIANTS[pick];
        let mut transform = placement(at, scale, seed);
        if obstacle {
            add_obstacle(&mut obstacles, at, scale, variant.height);
        }
        let (detail, stem) = match &models[pick] {
            TreeModel::Crown(detail, stem) => (detail, stem),
            TreeModel::Snag(model) => {
                let mut tree = commands.spawn((
                    Name::new("Standing deadwood"),
                    transform,
                    model.coarse_mesh(),
                    model.clone(),
                    model.bounds,
                    NoAutoAabb,
                    MeshMaterial3d(snag_bark.clone()),
                    reflected_layers(),
                    ForestDetail,
                ));
                if distant {
                    tree.insert(NotShadowCaster);
                }
                continue;
            }
        };
        let tint = crown_tint(seed, variant.kind);
        let lighting = SpaghettiLighting {
            direction_to_light: SUN,
            ambient: Vec3::new(0.64, 0.73, 0.75) * tint,
            directional: Vec3::new(0.62, 0.52, 0.40) * tint,
        };
        // The ground reflects warmer light into the undersides of sprays.
        let ground_ambient = SpaghettiGroundAmbient(Vec3::new(0.42, 0.39, 0.32) * tint);
        if variant.form.flag > 0.0 {
            // Flagged crowns all point downwind, toward +X.
            transform.rotation = Quat::from_rotation_y((hash01(seed + 3) - 0.5) * 0.6);
        }
        let mut reflected = commands.spawn((
            Name::new("Reflected conifer"),
            Spaghetti3d(detail.reflection.clone()),
            lighting,
            ground_ambient,
            NEEDLE_TRANSMISSION,
            SpaghettiReceiveShadows,
            RenderLayers::layer(REFLECTED_LAYER),
        ));
        if !distant {
            reflected.insert(SpaghettiSway::default());
        }
        let reflection = reflected.id();
        let mut tree = commands.spawn((
            Name::new(if distant {
                "Distant conifer"
            } else {
                "Conifer"
            }),
            transform,
            Spaghetti3d(detail.adaptive.clone()),
            SpaghettiDetailScale(if core { CORE_DETAIL } else { 1.0 }),
            ForestDetail,
            lighting,
            ground_ambient,
            NEEDLE_TRANSMISSION,
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
        ));
        tree.add_child(reflection).with_children(|children| {
            children.spawn((
                Name::new("Living bark"),
                stem.coarse_mesh(),
                stem.clone(),
                stem.bounds,
                NoAutoAabb,
                MeshMaterial3d(living_bark.clone()),
                // The crown's proxy already supplies its shadow.
                NotShadowCaster,
            ));
            if core && let Some((_, radius)) = variant.shadow {
                // Core trees stand behind the stand edges, where a few
                // stacked cones cast a close enough shadow for a
                // fraction of the glTF crown's cost.
                children.spawn((
                    Mesh3d(whorls.clone()),
                    MeshMaterial3d(shadow_material.clone()),
                    Transform::from_scale(Vec3::new(radius, variant.top(), radius)),
                    RenderLayers::layer(SHADOW_ONLY),
                ));
            } else if !distant && let Some((scene, radius)) = variant.shadow {
                // The glTF crown only casts the shadow.
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
        });
        if !distant {
            tree.insert((
                SpaghettiSway::default(),
                CrownWind {
                    reflection,
                    velocity: Vec2::ZERO,
                    phase: at.dot(Vec2::new(-0.29, 0.19)),
                    frequency: 4.2 + 1.6 * hash01(seed + 13),
                    exposure: scale * if core { 0.68 } else { 1.0 },
                },
            ));
        }
    }
}

/// A unit-high, unit-radius crown of three stacked open cones, used only as
/// a shadow caster.
fn whorl_cones() -> Mesh {
    const SIDES: u32 = 7;
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    for (base, apex, radius) in [(0.1, 0.55, 1.0), (0.36, 0.8, 0.74), (0.6, 1.0, 0.46)] {
        let tip = positions.len() as u32;
        positions.push([0.0, apex, 0.0]);
        for side in 0..SIDES {
            let around = Vec2::from_angle(side as f32 * std::f32::consts::TAU / SIDES as f32);
            positions.push([around.x * radius, base, around.y * radius]);
            let next = (side + 1) % SIDES;
            indices.extend([tip, tip + 1 + next, tip + 1 + side]);
        }
    }
    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
    Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(bevy::mesh::Indices::U32(indices))
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
