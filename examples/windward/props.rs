//! Scanned forest-floor props: old conifer stumps in the stands and loose
//! stones along the trail shoulders and the lakeshore.

use std::f32::consts::TAU;

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::{RenderLayers, VisibilityRange},
    light::{NotShadowCaster, NotShadowReceiver},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    world_serialization::WorldInstanceReady,
};
use bevy_spaghetti::{Spaghetti3d, SpaghettiAsset, SpaghettiLighting, SpaghettiReceiveShadows};

use crate::{
    debris::LogDetail,
    detail::{CameraView, ScreenDetail, load_lit_cloud},
    forest::ForestDetail,
    lake::REFLECTED_LAYER,
    landscape::{
        LAKE, LAKE_RADII, Obstacles, ROCKS, SUN, WATER_LEVEL, ground, hash01, mark_rock_shadow,
        noise, shore_distance, trail_x,
    },
    player::HikerCamera,
};

/// Poly Haven "Tree Stump 01" (CC0).
const STUMP: &str = "windward/stump/tree_stump_01_1k.gltf";
/// The same scan converted with spaghetti-convert at 2.5k samples.
const STUMP_REFLECTION: &str = "windward/stump-far.spag";
/// The scan's litter mound reaches 0.19 m below its origin about 0.6 to 0.9 m
/// from the trunk axis.
const STUMP_SKIRT: f32 = 0.17;
const STUMP_REACH: f32 = 0.8;
/// The scan is a low, rotted stump. Old-growth stumps in the Cascades stand
/// a meter or more, so stretch it upward without widening the footprint.
const STUMP_STRETCH: f32 = 1.6;

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

/// The conifer gains, so props sit in the same light as the stands.
pub const LIGHTING: SpaghettiLighting = SpaghettiLighting {
    direction_to_light: SUN,
    ambient: Vec3::new(0.24, 0.27, 0.29),
    directional: Vec3::new(0.62, 0.52, 0.40),
};

/// The opaque core is the scan mesh shrunk about its base, so splats overhang
/// its silhouette by a few centimeters and add only the surface detail.
pub const CORE_SCALE: f32 = 0.95;
/// Beyond this range the splats are too small to show the ground through
/// them, so the core fades out with a dither.
const CORE_RANGE: VisibilityRange = VisibilityRange {
    start_margin: 0.0..0.0,
    end_margin: 24.0..30.0,
    use_aabb: false,
};
pub const CONTACT_RANGE: VisibilityRange = VisibilityRange {
    start_margin: 0.0..0.0,
    end_margin: 34.0..42.0,
    use_aabb: false,
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

/// Ground taken by props and logs, so the meadow can leave it bare.
#[derive(Resource, Default)]
pub struct PropFootprints {
    /// Prop centers and reach radii.
    pub circles: Vec<(Vec2, f32)>,
    /// Log centerlines as segment end points.
    pub logs: Vec<(Vec2, Vec2)>,
}

impl PropFootprints {
    /// 0 on a prop's footprint, rising to 1 a little outside it.
    pub fn keep(&self, at: Vec2) -> f32 {
        let circles = self.circles.iter().fold(1.0_f32, |keep, &(center, reach)| {
            keep.min(crate::landscape::smoothstep(
                0.6 * reach,
                1.05 * reach,
                center.distance(at),
            ))
        });
        self.logs.iter().fold(circles, |keep, &(a, b)| {
            keep.min(crate::landscape::smoothstep(
                0.18,
                0.42,
                segment_distance(at, a, b),
            ))
        })
    }
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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
    let stumps = spawn_stumps(&mut commands, &server, &mut obstacles, &mut clearance);
    let stones = spawn_stones(&mut commands, &server, &mut obstacles, &mut clearance);
    commands.insert_resource(PropFootprints {
        circles: clearance.props.clone(),
        logs: clearance.logs.clone(),
    });

    let mut contacts = Vec::new();
    for (x, z, scale) in ROCKS {
        let at = Vec2::new(x, z);
        let radius = 0.89 * scale;
        if ground(at) > WATER_LEVEL + 0.02 {
            contacts.push((at, 0.55 * radius, 1.3 * radius, 0.72));
        }
    }
    contacts.extend(
        stumps
            .iter()
            .map(|&(at, scale)| (at, 0.35 * scale, 1.0 * scale, 0.7)),
    );
    contacts.extend(
        stones
            .iter()
            .map(|&(at, reach)| (at, 0.6 * reach, 1.5 * reach, 0.75)),
    );

    // A dark soil tone over the ground, strongest where the prop meets it.
    let soil = materials.add(StandardMaterial {
        base_color: Color::linear_rgb(0.016, 0.014, 0.010),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    for (index, (at, inner, outer, darkness)) in contacts.into_iter().enumerate() {
        commands.spawn((
            Name::new("Prop contact shadow"),
            Mesh3d(meshes.add(contact_shadow(at, inner, outer, darkness, index as u32))),
            MeshMaterial3d(soil.clone()),
            Transform::from_xyz(at.x, ground(at), at.y),
            NotShadowCaster,
            NotShadowReceiver,
            CONTACT_RANGE,
        ));
    }
}

/// Builds a disc that follows the terrain and darkens it toward the center.
pub fn contact_shadow(at: Vec2, inner: f32, outer: f32, darkness: f32, seed: u32) -> Mesh {
    const SIDES: usize = 28;
    // Radii as fractions from inner to outer, and their opacity.
    const RINGS: [(f32, f32); 4] = [(0.0, 1.0), (0.35, 0.72), (0.7, 0.28), (1.0, 0.0)];
    let base = ground(at);
    let phase = hash01(seed * 31 + 5) * 40.0;
    let mut positions = vec![[0.0, 0.03, 0.0]];
    let mut colors = vec![[1.0, 1.0, 1.0, darkness]];
    for side in 0..SIDES {
        let direction = Vec2::from_angle(side as f32 * TAU / SIDES as f32);
        // Wobble the outline so the rings don't read as circles.
        let wobble = 0.82 + 0.36 * noise(direction * 1.7 + Vec2::splat(phase));
        for (fraction, opacity) in RINGS {
            let radius = inner + (outer - inner) * fraction * wobble;
            let offset = direction * radius;
            let lift = 0.03 + 0.01 * fraction;
            positions.push([offset.x, ground(at + offset) - base + lift, offset.y]);
            colors.push([1.0, 1.0, 1.0, darkness * opacity]);
        }
    }
    let ring = RINGS.len() as u32;
    let mut indices = Vec::new();
    for side in 0..SIDES as u32 {
        let a = 1 + side * ring;
        let b = 1 + (side + 1) % SIDES as u32 * ring;
        indices.extend([0, b, a]);
        for step in 0..ring - 1 {
            indices.extend([a + step, b + step, a + step + 1]);
            indices.extend([a + step + 1, b + step, b + step + 1]);
        }
    }
    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

/// Spawns a visible, shrunk copy of a scan under its splat cloud.
pub fn spawn_core(commands: &mut Commands, scan: Handle<WorldAsset>, scale: Vec3) -> Entity {
    commands
        .spawn((
            Name::new("Prop core"),
            WorldAssetRoot(scan),
            Transform::from_scale(scale),
        ))
        .observe(mark_prop_core)
        .id()
}

/// The splats carry the lighting, and the separate scan casts the shadow, so
/// the core neither casts nor receives shadows.
pub fn mark_prop_core(
    ready: On<WorldInstanceReady>,
    children: Query<&Children>,
    meshes: Query<(), With<Mesh3d>>,
    mut commands: Commands,
) {
    for child in children.iter_descendants(ready.entity) {
        if meshes.contains(child) {
            commands
                .entity(child)
                .insert((NotShadowCaster, NotShadowReceiver, CORE_RANGE));
        }
    }
}

fn spawn_stumps(
    commands: &mut Commands,
    server: &AssetServer,
    obstacles: &mut Obstacles,
    clearance: &mut Clearance,
) -> Vec<(Vec2, f32)> {
    let mut placed = Vec::new();
    let scan = server.load(GltfAssetLabel::Scene(0).from_asset(STUMP));
    let reflection = load_lit_cloud(server, STUMP_REFLECTION);
    for (index, anchor) in STUMP_ANCHORS.into_iter().enumerate() {
        let seed = 4100 + index as u32 * 37;
        let scale = 1.1 + 0.5 * hash01(seed);
        let reach = STUMP_REACH * scale;
        // Search outward from the anchor on a golden-angle spiral.
        let Some(at) = (0..40)
            .map(|step| {
                let angle = step as f32 * 2.39996 + hash01(seed + 1) * TAU;
                anchor + Vec2::from_angle(angle) * 0.45 * (step as f32).sqrt()
            })
            .find(|&at| {
                (at.x - trail_x(at.y)).abs() >= 2.3 + reach
                    && shore_distance(at) >= 1.22
                    && ground(at) >= WATER_LEVEL + 0.25
                    && clearance.is_clear(at, reach, 1.0)
            })
        else {
            continue;
        };
        let floor = footprint_floor(at, reach);
        let height = scale * STUMP_STRETCH;
        let transform = Transform::from_xyz(at.x, floor + STUMP_SKIRT * height, at.y)
            .with_rotation(Quat::from_rotation_y(hash01(seed + 2) * TAU))
            .with_scale(Vec3::new(scale, height, scale));
        // The opaque scan uses the bark's normal map and supplies the shadow.
        commands
            .spawn((
                Name::new("Old-growth stump"),
                WorldAssetRoot(scan.clone()),
                transform,
            ))
            .with_child((
                Name::new("Reflected stump"),
                Spaghetti3d(reflection.clone()),
                LIGHTING,
                SpaghettiReceiveShadows,
                RenderLayers::layer(REFLECTED_LAYER),
            ));
        clearance.props.push((at, reach));
        placed.push((at, scale));
        // The unstretched scan is about 0.32 m in radius and 0.55 m tall.
        obstacles.add_boulder(at, 0.36 * scale, 0.55 * height);
    }
    placed
}

/// The splat levels, the scan, and the scan's size for one kind of stone.
pub type StoneKind = ([Handle<SpaghettiAsset>; 3], Handle<WorldAsset>, Vec3);

pub fn stone_kinds(server: &AssetServer) -> [StoneKind; 4] {
    STONES.map(|(name, size)| {
        let clouds = [
            format!("windward/stone-{name}.spag"),
            format!("windward/stone-{name}-mid.spag"),
            format!("windward/stone-{name}-far.spag"),
        ]
        .map(|path| load_lit_cloud(server, path));
        let shadow = server.load(
            GltfAssetLabel::Scene(0).from_asset(format!("windward/stones/stone_{name}.gltf")),
        );
        (clouds, shadow, size)
    })
}

fn spawn_stones(
    commands: &mut Commands,
    server: &AssetServer,
    obstacles: &mut Obstacles,
    clearance: &mut Clearance,
) -> Vec<(Vec2, f32)> {
    let mut contacts = Vec::new();
    let kinds = stone_kinds(server);
    for seed in 1..=900u32 {
        if contacts.len() >= 34 {
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
        spawn_prop(
            commands,
            "Loose stone",
            transform,
            clouds,
            shadow,
            (80.0, 30.0),
        );
        clearance.props.push((at, reach));
        if length > 0.36 {
            obstacles.add_boulder(at, 0.4 * length, size.y * scale * 0.75);
        }
        contacts.push((at, reach));
    }
    contacts
}

pub fn spawn_prop(
    commands: &mut Commands,
    name: &'static str,
    transform: Transform,
    clouds: &[Handle<SpaghettiAsset>; 3],
    shadow: &Handle<WorldAsset>,
    (near, middle): (f32, f32),
) {
    let core = spawn_core(commands, shadow.clone(), Vec3::splat(CORE_SCALE));
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
            SpaghettiReceiveShadows,
            transform,
        ))
        .with_children(|parent| {
            // This copy of the glTF only casts the shadow.
            parent
                .spawn(WorldAssetRoot(shadow.clone()))
                .observe(mark_rock_shadow);
            parent.spawn((
                Name::new("Reflected prop"),
                Spaghetti3d(clouds[2].clone()),
                LIGHTING,
                SpaghettiReceiveShadows,
                RenderLayers::layer(REFLECTED_LAYER),
            ));
        })
        .add_child(core);
}

/// Returns the lowest ground under a round footprint, so no edge floats.
pub fn footprint_floor(at: Vec2, reach: f32) -> f32 {
    (0..8)
        .map(|i| at + Vec2::from_angle(i as f32 * TAU / 8.0) * reach)
        .chain([at])
        .map(ground)
        .fold(f32::INFINITY, f32::min)
}

pub fn segment_distance(point: Vec2, a: Vec2, b: Vec2) -> f32 {
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
