use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::RenderLayers,
    image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor},
    light::{CascadeShadowConfigBuilder, VolumetricLight},
    mesh::{Indices, PrimitiveTopology},
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};
use bevy_spaghetti::{
    GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiLighting, SpaghettiReceiveShadows,
};

use crate::{
    detail::{CameraView, ScreenDetail, load_lit_cloud},
    lake::REFLECTED_LAYER,
};

pub const WATER_LEVEL: f32 = 0.20;
pub const LAKE: Vec2 = Vec2::new(10.0, -14.0);
pub const LAKE_RADII: Vec2 = Vec2::new(10.4, 16.8);
pub const SUN: Vec3 = Vec3::new(-0.1618, 0.1435, -0.9763);
pub const SUN_COLOR: Color = Color::srgb(1.0, 0.86, 0.71);
pub const SUN_LUX: f32 = 19800.0;
pub const SHADOW_ONLY: usize = 1;
const TERRAIN_COLUMNS: u32 = 192;
const TERRAIN_ROWS: u32 = 196;
const TERRAIN_STEP: f32 = 0.72;
const TERRAIN_ORIGIN: Vec2 = Vec2::new(-0.5 * TERRAIN_COLUMNS as f32 * TERRAIN_STEP, -98.0);

#[derive(Resource, Default)]
pub struct Obstacles(Vec<Obstacle>);

struct Obstacle {
    center: Vec2,
    body_radius: f32,
    canopy_radius: f32,
    bottom: f32,
    top: f32,
}

impl Obstacles {
    pub fn add_tree(&mut self, at: Vec2, scale: f32, height: f32) {
        let floor = ground(at);
        self.0.push(Obstacle {
            center: at,
            body_radius: 0.27 * scale,
            canopy_radius: 1.55 * scale,
            bottom: floor + 0.82 * scale,
            top: floor + height * scale,
        });
    }

    /// Adds a low solid prop, such as a stump or stone, that blocks the hiker.
    pub fn add_boulder(&mut self, at: Vec2, radius: f32, height: f32) {
        let floor = ground(at);
        self.0.push(Obstacle {
            center: at,
            body_radius: radius,
            canopy_radius: radius,
            bottom: floor,
            top: floor + height,
        });
    }

    pub fn collide(&self, at: Vec2, radius: f32) -> Vec2 {
        let mut position = at;
        for obstacle in &self.0 {
            let delta = position - obstacle.center;
            let minimum = radius + obstacle.body_radius;
            let distance = delta.length();
            if distance < minimum {
                let normal = if distance > 1e-5 {
                    delta / distance
                } else {
                    Vec2::X
                };
                position = obstacle.center + normal * minimum;
            }
        }
        position
    }

    pub fn camera_eye(&self, pivot: Vec3, desired: Vec3) -> Vec3 {
        let mut safe = pivot;
        for step in 1..=12 {
            let fraction = step as f32 / 12.0;
            let sample = pivot.lerp(desired, fraction);
            let at = sample.xz();
            let obscured = self.0.iter().any(|obstacle| {
                sample.y >= obstacle.bottom
                    && sample.y <= obstacle.top
                    && at.distance_squared(obstacle.center) < obstacle.canopy_radius.powi(2)
            });
            if sample.y < ground(at) + 0.30 || obscured {
                break;
            }
            safe = sample;
        }
        // Keep a minimum shoulder distance when the hiker is under branches.
        if safe.distance_squared(pivot) < 1.15_f32.powi(2) {
            safe = pivot + (desired - pivot).normalize_or_zero() * 1.15;
        }
        safe.y = safe.y.max(ground(safe.xz()) + 0.34);
        safe
    }
}

pub type GroundMaterial = ExtendedMaterial<StandardMaterial, TrailSurface>;

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct TrailSurface {
    #[texture(100)]
    #[sampler(101)]
    albedo: Handle<Image>,
    #[texture(102)]
    normal: Handle<Image>,
    #[texture(103)]
    litter_color: Handle<Image>,
    #[texture(104)]
    litter_surface: Handle<Image>,
    #[uniform(105)]
    water_level: f32,
}

impl MaterialExtension for TrailSurface {
    fn fragment_shader() -> ShaderRef {
        "windward/ground.wesl".into()
    }
}

pub fn trail_x(z: f32) -> f32 {
    2.7 * (0.073 * z + 0.04).sin() + 1.05 * (0.22 * z + 0.5).sin() - 1.4
}

/// Noisy distance from the lake center, normalized by `LAKE_RADII`. The
/// terrain basin, tree spacing, and props all read the shoreline from it.
pub fn shore_distance(at: Vec2) -> f32 {
    let offset = (at - LAKE) / LAKE_RADII;
    let radius = offset.length();
    let banks = 0.29 * (noise(at * 0.108 + Vec2::new(6.2, -1.3)) - 0.5);
    let coves = 0.115 * (noise(at * 0.25) - 0.5);
    radius + banks + coves + shoreline_lobes(at, offset, radius)
}

/// Low-frequency angular lobes and warped bays that turn the basin into
/// points and coves. The phases keep the fixed stones in `ROCKS` on dry land.
fn shoreline_lobes(at: Vec2, offset: Vec2, radius: f32) -> f32 {
    let warp = Vec2::new(
        noise(at * 0.09 + Vec2::new(3.1, 8.7)),
        noise(at * 0.09 + Vec2::new(-4.4, 1.9)),
    ) - 0.5;
    let angle = offset.y.atan2(offset.x);
    let lobes = 0.13 * (3.0 * angle + 3.3 + 2.0 * warp.x).sin()
        + 0.075 * (5.0 * angle + 0.4 + 2.5 * warp.y).sin()
        + 0.035 * (9.0 * angle + 2.3 + 3.0 * warp.x).sin();
    let bays = 0.30 * (noise(at * 0.16 + Vec2::new(11.0, -7.0) + 4.0 * warp) - 0.5);
    // Fade out toward the center, where the angle is undefined.
    let shift = (lobes + bays) * smoothstep(0.35, 0.8, radius);
    // Water may only recede near the trail, so the path stays dry.
    if shift < 0.0 {
        shift * smoothstep(2.4, 5.5, (at.x - trail_x(at.y)).abs())
    } else {
        shift
    }
}

pub fn ground(at: Vec2) -> f32 {
    let cell = (at - TERRAIN_ORIGIN) / TERRAIN_STEP;
    let base = cell.floor();
    let frac = cell - base;
    let origin = TERRAIN_ORIGIN + base * TERRAIN_STEP;
    let x = height_at_vertex(origin + Vec2::X * TERRAIN_STEP);
    let z = height_at_vertex(origin + Vec2::Y * TERRAIN_STEP);
    // Match the diagonal used by the mesh so collision meets the visible ground.
    if frac.x + frac.y <= 1.0 {
        let corner = height_at_vertex(origin);
        corner + (x - corner) * frac.x + (z - corner) * frac.y
    } else {
        let corner = height_at_vertex(origin + Vec2::splat(TERRAIN_STEP));
        corner + (z - corner) * (1.0 - frac.x) + (x - corner) * (1.0 - frac.y)
    }
}

fn height_at_vertex(at: Vec2) -> f32 {
    let x = at.x;
    let z = at.y;
    let relief =
        0.56 + 0.64 * noise(at * 0.076) + 0.23 * noise(at * 0.28) + 0.12 * noise(at * 0.74);
    let trail = 1.0 - smoothstep(0.6, 2.2, (x - trail_x(z)).abs());
    let bank = relief - 0.16 * trail;
    let radial = shore_distance(at);
    let basin = -0.94 + 0.63 * radial.min(1.2).powi(2);
    let near_ground = basin.lerp(bank, smoothstep(0.72, 1.115, radial));
    near_ground
        + hill(at, Vec2::new(-28.0, -34.0), Vec2::new(11.0, 18.0), 5.6)
        + hill(at, Vec2::new(28.0, -38.0), Vec2::new(13.0, 21.0), 8.2)
        + hill(at, Vec2::new(3.0, -66.0), Vec2::new(28.0, 12.0), 3.2)
}

fn hill(at: Vec2, center: Vec2, size: Vec2, height: f32) -> f32 {
    height * (-((at - center) / size).length_squared()).exp()
}

pub fn smoothstep(start: f32, end: f32, x: f32) -> f32 {
    let t = ((x - start) / (end - start)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn noise(at: Vec2) -> f32 {
    let cell = at.floor().as_ivec2();
    let fraction = at - at.floor();
    let w = fraction * fraction * (Vec2::splat(3.0) - 2.0 * fraction);
    let value = |x: i32, y: i32| {
        let hash = (x as u32)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add((y as u32).wrapping_mul(0x85EB_CA6B));
        hash01(hash)
    };
    value(cell.x, cell.y)
        .lerp(value(cell.x + 1, cell.y), w.x)
        .lerp(
            value(cell.x, cell.y + 1).lerp(value(cell.x + 1, cell.y + 1), w.x),
            w.y,
        )
}

pub fn hash01(mut v: u32) -> f32 {
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb_352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846c_a68b);
    v ^= v >> 16;
    (v >> 8) as f32 / 16_777_216.0
}

pub fn spawn_landscape(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ground_materials: ResMut<Assets<GroundMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut splats: ResMut<Assets<SpaghettiAsset>>,
) {
    commands.spawn((
        Name::new("Sun"),
        DirectionalLight {
            color: SUN_COLOR,
            illuminance: SUN_LUX,
            shadow_maps_enabled: true,
            shadow_normal_bias: 0.70,
            ..default()
        },
        Transform::IDENTITY.looking_to(-SUN, Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades: 3,
            first_cascade_far_bound: 17.0,
            maximum_distance: 100.0,
            ..default()
        }
        .build(),
        VolumetricLight,
        RenderLayers::default().with(SHADOW_ONLY),
    ));
    commands.spawn((
        Name::new("Light from the open sky"),
        DirectionalLight {
            color: Color::srgb(0.86, 0.92, 1.0),
            illuminance: 4300.0,
            ..default()
        },
        Transform::IDENTITY.looking_to(Vec3::new(-0.32, -0.58, -0.75), Vec3::Y),
        RenderLayers::default()
            .with(REFLECTED_LAYER)
            .with(crate::lake::LAKE_SURFACE_LAYER),
    ));

    let terrain_material = ground_materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color_texture: Some(load_tile(&server, "windward/ground-color.dds", true)),
            normal_map_texture: Some(load_tile(&server, "windward/ground-normal.dds", false)),
            reflectance: 0.29,
            perceptual_roughness: 0.94,
            ..default()
        },
        extension: TrailSurface {
            albedo: load_tile(&server, "windward/trail-color.dds", true),
            normal: load_tile(&server, "windward/trail-normal.dds", false),
            litter_color: load_tile(&server, "windward/litter-color.dds", true),
            litter_surface: load_tile(&server, "windward/litter-surface.dds", false),
            water_level: WATER_LEVEL,
        },
    });
    commands.spawn((
        Name::new("Trail, banks, and lake bed"),
        Mesh3d(meshes.add(terrain())),
        MeshMaterial3d(terrain_material),
        reflected_layers(),
    ));

    spawn_mist(&mut commands, &mut splats);

    let mut obstacles = Obstacles::default();
    spawn_rocks(&mut commands, &server, &mut materials, &mut obstacles);
    commands.insert_resource(obstacles);
}

fn terrain() -> Mesh {
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    let mut uv = Vec::new();
    let mut masks = Vec::new();
    let mut indices = Vec::new();
    for row in 0..=TERRAIN_ROWS {
        for column in 0..=TERRAIN_COLUMNS {
            let at = TERRAIN_ORIGIN + Vec2::new(column as f32, row as f32) * TERRAIN_STEP;
            let Vec2 { x, y: z } = at;
            let height = height_at_vertex(at);
            let path = 1.0
                - smoothstep(
                    0.55,
                    1.45,
                    (x - trail_x(z)).abs() + (noise(at * 0.85) - 0.5) * 0.36,
                );
            let growth = 0.60 * noise(at * 0.096) + 0.40 * noise(at * 0.33);
            let moss = Vec3::new(0.40, 0.53, 0.31).lerp(Vec3::new(0.66, 0.63, 0.45), growth);
            let wet = 1.0 - smoothstep(0.17, 0.62, height);
            let tint = moss.lerp(Vec3::new(0.44, 0.42, 0.35), wet);
            // Under the crowns, brown needle litter and duff with patches of
            // dark moss, dimmed as if the closed canopy blocked the sky.
            let canopy = crate::forest::canopy(at);
            let duff =
                Vec3::new(0.40, 0.24, 0.14).lerp(Vec3::new(0.27, 0.16, 0.09), noise(at * 1.3));
            let moss = Vec3::new(0.20, 0.27, 0.12);
            let floor = duff.lerp(
                moss,
                smoothstep(0.55, 0.8, noise(at * 0.45 + Vec2::new(5.0, 1.0))),
            );
            let litter = canopy * (1.0 - wet);
            let tint = tint.lerp(floor, litter);
            // Shade the soil under the grass so gaps between blades read as depth.
            let tint = tint * (1.0 - 0.48 * crate::meadow::cover(at));
            let patches =
                0.52 + 0.48 * smoothstep(0.26, 0.74, noise(at * 0.55 + Vec2::new(3.0, 6.0)));
            positions.push([x, height, z]);
            // The terrain is opaque. Its vertex alpha carries litter coverage.
            colors.push(tint.extend(litter * patches).to_array());
            uv.push((at * 0.41).to_array());
            masks.push([path, wet]);
            if column < TERRAIN_COLUMNS && row < TERRAIN_ROWS {
                let stride = TERRAIN_COLUMNS + 1;
                let i = row * stride + column;
                indices.extend([i, i + stride, i + 1, i + 1, i + stride, i + stride + 1]);
            }
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uv)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, masks)
    .with_inserted_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    mesh.generate_tangents()
        .expect("indexed terrain has normals and UVs");
    mesh
}

pub fn load_tile(server: &AssetServer, path: &'static str, srgb: bool) -> Handle<Image> {
    server
        .load_builder()
        .with_settings(move |settings: &mut ImageLoaderSettings| {
            settings.is_srgb = srgb;
            settings.sampler = repeating_sampler();
        })
        .load(path)
}

fn repeating_sampler() -> ImageSampler {
    ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        anisotropy_clamp: 8,
        ..ImageSamplerDescriptor::linear()
    })
}

fn spawn_mist(commands: &mut Commands, splats: &mut Assets<SpaghettiAsset>) {
    let mut banks = Vec::new();
    for i in 0..22 {
        let seed = 610 + i * 9;
        let z = -22.0 + 18.0 * hash01(seed);
        let x = 7.5 + 5.6 * hash01(seed + 1);
        let y = WATER_LEVEL + 0.36 + 0.38 * hash01(seed + 2);
        let length = 1.8 + 3.6 * hash01(seed + 3);
        let height = 0.13 + 0.13 * hash01(seed + 4);
        let start = Vec3::new(x - 0.5 * length, y, z);
        let end = Vec3::new(x + 0.5 * length, y + 0.14 * hash01(seed + 5), z - 1.0);
        banks.push(GaussianSegment::new(
            start,
            end,
            Vec3::new(1.8, height, 0.76 + hash01(seed + 6)),
            Quat::IDENTITY,
            length * height * 0.11,
            Vec3::new(0.32, 0.40, 0.45),
        ));
    }
    commands.spawn((
        Name::new("Low lake mist"),
        Spaghetti3d(splats.add(SpaghettiAsset::new(banks).expect("valid lake mist"))),
    ));
}

/// Stone centers on the ground plane and their uniform scales.
pub const ROCKS: [(f32, f32, f32); 15] = [
    (-1.4, 7.0, 0.78),
    (4.1, 3.3, 0.53),
    (4.2, -6.0, 1.42),
    (6.3, -4.8, 0.58),
    (-7.1, -13.0, 0.91),
    (-8.0, -14.3, 0.68),
    (-1.6, -20.4, 0.54),
    (2.6, -25.1, 0.95),
    (21.0, -10.1, 0.92),
    (15.0, 2.8, 0.88),
    (19.0, -24.3, 0.68),
    (-5.9, -34.1, 1.31),
    (-6.0, -36.2, 0.44),
    (7.0, 11.0, 0.94),
    (3.6, 14.2, 0.44),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum RockLevel {
    Scan,
    Medium,
    Far,
}

#[derive(Component)]
pub struct RockDetail {
    clouds: [Handle<SpaghettiAsset>; 2],
    scan: Entity,
    level: RockLevel,
}

fn spawn_rocks(
    commands: &mut Commands,
    server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
    obstacles: &mut Obstacles,
) {
    let rocks = [
        [
            // ShareTextures "Rock 4" scan, converted with spaghetti-convert.
            "windward/basalt-mid.spag",
            "windward/basalt-far.spag",
        ],
        [
            // Poly Haven "Boulder 01" (CC0), converted with spaghetti-convert.
            "windward/granite-mid.spag",
            "windward/granite-far.spag",
        ],
    ]
    .map(|paths| paths.map(|path| load_lit_cloud(server, path)));
    let scans: [Handle<Mesh>; 2] = [
        "windward/basalt/rock4.gltf",
        "windward/granite/boulder_01_1k.gltf",
    ]
    .map(|path| {
        server.load(
            GltfAssetLabel::Primitive {
                mesh: 0,
                primitive: 0,
            }
            .from_asset(path),
        )
    });
    let granite_surface = load_tile(server, "windward/granite-surface.dds", false);
    let stone = [
        materials.add(StandardMaterial {
            base_color_texture: Some(load_tile(server, "windward/basalt-color.dds", true)),
            perceptual_roughness: 0.90,
            reflectance: 0.26,
            ..default()
        }),
        materials.add(StandardMaterial {
            base_color_texture: Some(load_tile(server, "windward/granite-color.dds", true)),
            normal_map_texture: Some(load_tile(server, "windward/granite-normal.dds", false)),
            metallic_roughness_texture: Some(granite_surface.clone()),
            occlusion_texture: Some(granite_surface),
            perceptual_roughness: 1.0,
            reflectance: 0.26,
            ..default()
        }),
    ];
    // Bake the original scene node transform back into the basalt primitive.
    let source_transform = [
        Transform::from_scale(Vec3::splat(500.0))
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        Transform::IDENTITY,
    ];
    for (index, &(x, z, scale)) in ROCKS.iter().enumerate() {
        let at = Vec2::new(x, z);
        let variant = usize::from(index % 3 != 0);
        let clouds = &rocks[variant];
        // The granite scan is smaller than the ledge and rests on its base,
        // so enlarge it and bury the uneven underside.
        let (size, sink) = if variant == 1 {
            (1.45, 0.42)
        } else {
            (0.85, 0.30)
        };
        let placement = Transform::from_xyz(x, ground(at) - sink * scale, z)
            .with_scale(Vec3::splat(scale * size))
            .with_rotation(Quat::from_rotation_y(hash01(index as u32 + 14) * 6.0));
        let lighting = SpaghettiLighting {
            direction_to_light: SUN,
            // `stone_grade` turns the scan's warm albedo grey, so these gains
            // stay close to neutral.
            ambient: Vec3::new(0.22, 0.245, 0.27),
            directional: Vec3::new(0.49, 0.46, 0.41),
        };
        // The same mesh stays in the sun's shadow map at every detail level.
        let scan = commands
            .spawn((
                Name::new("Textured boulder scan"),
                Mesh3d(scans[variant].clone()),
                MeshMaterial3d(stone[variant].clone()),
                source_transform[variant],
                RenderLayers::layer(SHADOW_ONLY),
            ))
            .id();
        commands
            .spawn((
                Name::new("Moss-covered stone"),
                Spaghetti3d(clouds[0].clone()),
                RockDetail {
                    clouds: clouds.clone(),
                    scan,
                    level: RockLevel::Medium,
                },
                lighting,
                SpaghettiReceiveShadows,
                placement,
                RenderLayers::default(),
            ))
            .add_child(scan)
            .with_child((
                Name::new("Reflected stone"),
                Spaghetti3d(clouds[1].clone()),
                lighting,
                SpaghettiReceiveShadows,
                RenderLayers::layer(REFLECTED_LAYER),
            ));
        obstacles.0.push(Obstacle {
            center: at,
            body_radius: scale * 0.89,
            canopy_radius: scale * 0.92,
            bottom: ground(at),
            top: ground(at) + 1.25 * scale,
        });
    }
}

pub fn select_rock_detail(
    camera: Single<CameraView, With<crate::player::HikerCamera>>,
    assets: Res<Assets<SpaghettiAsset>>,
    server: Res<AssetServer>,
    materials: Res<Assets<StandardMaterial>>,
    mut rocks: Query<(
        &Transform,
        &mut RockDetail,
        &mut Spaghetti3d,
        &mut RenderLayers,
    )>,
    mut scans: Query<
        (
            &Mesh3d,
            &MeshMaterial3d<StandardMaterial>,
            &mut RenderLayers,
        ),
        Without<RockDetail>,
    >,
) {
    let Some(view) = ScreenDetail::new(*camera) else {
        return;
    };
    for (transform, mut detail, mut cloud, mut layers) in &mut rocks {
        let Some(asset) = assets.get(&detail.clouds[0]) else {
            continue;
        };
        let (low, high) = asset.bounds();
        let center = transform.transform_point((low + high) * 0.5);
        let dimensions = (high - low) * transform.scale;
        let size = dimensions.xz().length() * 0.66;
        let pixels = view.span_pixels(center, size, 0.4 * size);
        // Use hysteresis to keep the density stable as the shoulder camera bobs.
        let near = if detail.level == RockLevel::Scan {
            74.0
        } else {
            92.0
        };
        let middle = if detail.level == RockLevel::Far {
            65.0
        } else {
            54.0
        };
        let level = if pixels >= near {
            RockLevel::Scan
        } else if pixels >= middle {
            RockLevel::Medium
        } else {
            RockLevel::Far
        };
        if level == detail.level {
            continue;
        }
        let Ok((mesh, material, mut scan_layers)) = scans.get_mut(detail.scan) else {
            continue;
        };
        if level == RockLevel::Scan {
            let Some(material) = materials.get(&material.0) else {
                continue;
            };
            let textures = [
                &material.base_color_texture,
                &material.normal_map_texture,
                &material.metallic_roughness_texture,
            ];
            if !server.is_loaded_with_dependencies(mesh.0.id())
                || !textures
                    .into_iter()
                    .flatten()
                    .all(|map| server.is_loaded_with_dependencies(map.id()))
            {
                continue;
            }
            *layers = RenderLayers::none();
            *scan_layers = RenderLayers::default().with(SHADOW_ONLY);
        } else {
            let handle = &detail.clouds[usize::from(level == RockLevel::Far)];
            if !assets.contains(handle) {
                continue;
            }
            cloud.0 = handle.clone();
            *layers = RenderLayers::default();
            *scan_layers = RenderLayers::layer(SHADOW_ONLY);
        }
        detail.level = level;
    }
}

pub fn mark_rock_shadow(
    ready: On<bevy::world_serialization::WorldInstanceReady>,
    children: Query<&Children>,
    meshes: Query<(), With<Mesh3d>>,
    mut commands: Commands,
) {
    for child in children.iter_descendants(ready.entity) {
        if meshes.contains(child) {
            commands
                .entity(child)
                .insert(RenderLayers::layer(SHADOW_ONLY));
        }
    }
}

pub fn reflected_layers() -> RenderLayers {
    RenderLayers::default().with(REFLECTED_LAYER)
}

pub fn mark_reflected_meshes(
    ready: On<bevy::world_serialization::WorldInstanceReady>,
    children: Query<&Children>,
    meshes: Query<(), With<Mesh3d>>,
    mut commands: Commands,
) {
    for child in children.iter_descendants(ready.entity) {
        if meshes.contains(child) {
            commands.entity(child).insert(reflected_layers());
        }
    }
}
