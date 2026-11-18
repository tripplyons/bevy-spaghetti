use bevy::{
    asset::RenderAssetUsages,
    camera::{
        CameraProjection, Exposure, Hdr, RenderTarget, primitives::Aabb, visibility::RenderLayers,
    },
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    image::ImageSampler,
    light::{GeneratedEnvironmentMapLight, NotShadowCaster, Skybox},
    math::Vec3A,
    pbr::{DistanceFog, ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::{AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat},
    shader::ShaderRef,
};

use bevy_spaghetti::{Spaghetti3d, SpaghettiAsset};

use crate::{
    detail::CameraView,
    landscape::{LAKE, SUN, SUN_COLOR, SUN_LUX, WATER_LEVEL, ground, load_tile, shore_distance},
    meadow::Meadow,
    player::HikerCamera,
    reeds::ReedPatch,
};

pub const LAKE_SURFACE_LAYER: usize = 2;
pub const REFLECTED_LAYER: usize = 3;
const REFLECTION_LIMIT: UVec2 = UVec2::new(640, 400);
const REFLECTION_MIN_LONG_EDGE: f32 = 320.0;
const MAX_REFLECTION_HZ: f32 = 36.0;
const PLANE_SIZE: Vec2 = Vec2::new(28.0, 42.0);
const LAKE_BOUNDS: Aabb = Aabb {
    center: Vec3A::new(LAKE.x, WATER_LEVEL, LAKE.y),
    half_extents: Vec3A::new(PLANE_SIZE.x * 0.5, 0.02, PLANE_SIZE.y * 0.5),
};

#[derive(Resource)]
pub struct Lake(Handle<LakeMaterial>);

#[derive(Component, Default)]
pub struct ReflectedView {
    last_update: Option<f32>,
    size: UVec2,
}

pub type LakeMaterial = ExtendedMaterial<StandardMaterial, LakeSurface>;

#[derive(Clone, Debug, ShaderType)]
struct LakeUniform {
    reflected_clip_from_world: Mat4,
    elapsed: f32,
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct LakeSurface {
    #[uniform(100)]
    uniform: LakeUniform,
    #[texture(101)]
    #[sampler(102)]
    reflection: Handle<Image>,
    #[texture(103)]
    #[sampler(104)]
    lakebed: Handle<Image>,
    #[texture(105)]
    #[sampler(106)]
    sediment: Handle<Image>,
    /// The bank albedo, so the shallowest water can match the soil around it.
    #[texture(107)]
    #[sampler(108)]
    bank: Handle<Image>,
    #[texture(109)]
    sediment_normal: Handle<Image>,
}

impl MaterialExtension for LakeSurface {
    fn fragment_shader() -> ShaderRef {
        "windward/water.wesl".into()
    }
}

pub fn spawn_lake(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LakeMaterial>>,
    main: Single<
        (
            &Transform,
            &Projection,
            &Exposure,
            &Skybox,
            &GeneratedEnvironmentMapLight,
            &DistanceFog,
        ),
        With<HikerCamera>,
    >,
) {
    let (transform, projection, exposure, sky, environment, fog) = *main;
    let (mirror, perspective) = reflect_view(transform, projection, UVec2::ONE);
    // Allocate the full target when the main camera has a viewport and sees water.
    let mut target = Image::new_target_texture(1, 1, TextureFormat::Rgba16Float, None);
    target.data = None;
    target.copy_on_resize = false;
    target.sampler = ImageSampler::linear();
    let reflection = images.add(target);
    commands.spawn((
        Name::new("Lake reflection camera"),
        ReflectedView::default(),
        Camera3d::default(),
        Camera {
            is_active: false,
            order: -1,
            invert_culling: true,
            ..default()
        },
        RenderTarget::Image(reflection.clone().into()),
        mirror,
        Projection::Perspective(perspective),
        Hdr,
        *exposure,
        (Tonemapping::None, DebandDither::Disabled, Msaa::Off),
        sky.clone(),
        environment.clone(),
        fog.clone(),
        RenderLayers::layer(REFLECTED_LAYER),
    ));
    // A separate light gives the reflected view the same solar direction
    // without a second set of shadow cascade draws.
    commands.spawn((
        Name::new("Sun in lake reflection"),
        DirectionalLight {
            color: SUN_COLOR,
            illuminance: SUN_LUX,
            ..default()
        },
        Transform::IDENTITY.looking_to(-SUN, Vec3::Y),
        RenderLayers::layer(REFLECTED_LAYER),
    ));

    let water = materials.add(ExtendedMaterial {
        base: StandardMaterial {
            reflectance: 0.36,
            perceptual_roughness: 0.12,
            ..default()
        },
        extension: LakeSurface {
            uniform: LakeUniform {
                reflected_clip_from_world: Mat4::IDENTITY,
                elapsed: 0.0,
            },
            reflection,
            lakebed: images.add(lakebed_texture()),
            sediment: load_tile(&server, "windward/trail-color.dds", true),
            bank: load_tile(&server, "windward/ground-color.dds", true),
            sediment_normal: load_tile(&server, "windward/trail-normal.dds", false),
        },
    });
    commands.spawn((
        Name::new("Wind-rippled lake"),
        Mesh3d(meshes.add(Plane3d::default().mesh().size(PLANE_SIZE.x, PLANE_SIZE.y))),
        MeshMaterial3d(water.clone()),
        Transform::from_xyz(LAKE.x, WATER_LEVEL, LAKE.y),
        RenderLayers::layer(LAKE_SURFACE_LAYER),
        NotShadowCaster,
    ));
    commands.insert_resource(Lake(water));
}

/// Signed water depth and the slope of the bed along world X and Z.
/// Depth crosses zero at the visible terrain so the shoreline stays in place.
fn lakebed_texture() -> Image {
    const SIZE: UVec2 = UVec2::new(224, 336);
    let spacing = PLANE_SIZE / SIZE.as_vec2();
    let origin = LAKE - 0.5 * PLANE_SIZE;
    let mut heights = Vec::with_capacity((SIZE.x * SIZE.y) as usize);
    for row in 0..SIZE.y {
        for column in 0..SIZE.x {
            let at = origin + (Vec2::new(column as f32, row as f32) + 0.5) * spacing;
            heights.push(ground(at));
        }
    }
    let height = |x, z| heights[(z * SIZE.x + x) as usize];
    let mut samples = Vec::with_capacity(heights.len() * 8);
    for row in 0..SIZE.y {
        for column in 0..SIZE.x {
            // Average over half a meter to avoid a normal seam on each
            // triangle edge. The plane boundaries use one-sided differences.
            let left = column.saturating_sub(2);
            let right = (column + 2).min(SIZE.x - 1);
            let front = row.saturating_sub(2);
            let back = (row + 2).min(SIZE.y - 1);
            let dx = (height(right, row) - height(left, row)) / (spacing.x * (right - left) as f32);
            let dz = (height(column, back) - height(column, front))
                / (spacing.y * (back - front) as f32);
            let depth = (WATER_LEVEL - height(column, row)).clamp(-0.5, 2.5);
            for channel in [depth, dx, dz, 0.0] {
                samples.extend(f16_bits(channel).to_le_bytes());
            }
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: SIZE.x,
            height: SIZE.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        samples,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::linear();
    image
}

/// Converts a value in the normal half-float range, flushing tiny values to zero.
fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return sign;
    }
    let mantissa = bits & 0x7f_ffff;
    // Round to nearest; a carry correctly bumps the exponent.
    let rounded = ((exponent as u32) << 10 | mantissa >> 13) + ((mantissa >> 12) & 1);
    sign | rounded.min(0x7bff) as u16
}

type ShorePlants<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Spaghetti3d,
        &'static Transform,
        Has<Meadow>,
    ),
    (
        Or<(With<ReedPatch>, With<Meadow>)>,
        Without<RenderLayers>,
        Without<ReflectedView>,
    ),
>;

/// Puts reed beds and meadow chunks along the shore into the reflected view.
/// Others get the default layer so each entity is checked only once.
fn add_shore_plants_to_reflection(
    commands: &mut Commands,
    plants: &ShorePlants<'_, '_>,
    clouds: &Assets<SpaghettiAsset>,
) {
    for (entity, cloud, transform, meadow) in plants {
        let Some(asset) = clouds.get(&cloud.0) else {
            continue;
        };
        let (low, high) = asset.bounds();
        let center = transform.transform_point(0.5 * (low + high)).xz();
        let shore = shore_distance(center);
        // Grass chunks are large, so test a wider band around the waterline.
        let near = if meadow {
            (0.7..1.75).contains(&shore)
        } else {
            shore < 1.5
        };
        let layers = if near {
            RenderLayers::default().with(REFLECTED_LAYER)
        } else {
            RenderLayers::default()
        };
        commands.entity(entity).insert(layers);
    }
}

pub fn reflect_lake(
    mut commands: Commands,
    plants: ShorePlants,
    clouds: Res<Assets<SpaghettiAsset>>,
    time: Res<Time>,
    lake: Res<Lake>,
    main: Single<CameraView, (With<HikerCamera>, Without<ReflectedView>)>,
    mut reflected: Single<(
        &mut Transform,
        &mut Projection,
        &mut Camera,
        &mut ReflectedView,
    )>,
    mut materials: ResMut<Assets<LakeMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    add_shore_plants_to_reflection(&mut commands, &plants, &clouds);
    let (main_camera, transform, projection, resolution) = *main;
    let (mirror, mirror_projection, camera, state) = &mut *reflected;
    camera.is_active = false;
    let Some(view_size) = resolution
        .map(|resolution| resolution.0)
        .or_else(|| main_camera.physical_viewport_size())
        .filter(|size| size.min_element() > 0)
        .filter(|_| lake_in_view(transform, projection))
    else {
        state.last_update = None;
        return;
    };
    let elapsed = time.elapsed_secs();
    let Some(mut water) = materials.get_mut(&lake.0) else {
        return;
    };
    water.extension.uniform.elapsed = elapsed;
    let size = reflection_size(view_size);
    if size != state.size {
        let Some(mut image) = images.get_mut(&water.extension.reflection) else {
            return;
        };
        image.resize(Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        });
        state.size = size;
    } else if state
        .last_update
        .is_some_and(|last_update| elapsed - last_update < reflection_period(view_size))
    {
        return;
    }
    let (reflection, perspective) = reflect_view(transform, projection, size);
    water.extension.uniform.reflected_clip_from_world =
        perspective.get_clip_from_view() * reflection.compute_affine().inverse();
    **mirror = reflection;
    **mirror_projection = Projection::Perspective(perspective);
    state.last_update = Some(elapsed);
    camera.is_active = true;
}

fn reflection_size(view_size: UVec2) -> UVec2 {
    let view_size = view_size.as_vec2();
    // At small window sizes, keep enough pixels for the reflected scene's
    // detail levels. A smaller target was slower on the Metal test device.
    let half_scale = (REFLECTION_MIN_LONG_EDGE / view_size.max_element()).clamp(0.5, 1.0);
    let scale = (REFLECTION_LIMIT.as_vec2() / view_size)
        .min_element()
        .min(half_scale);
    (view_size * scale).round().as_uvec2().max(UVec2::ONE)
}

fn reflection_period(view_size: UVec2) -> f32 {
    // The reflected mesh can update less often in a small viewport while
    // the ripple shading and the world-space reprojection run every frame.
    let scale = (view_size.max_element() as f32 / (REFLECTION_LIMIT.x * 2) as f32).sqrt();
    (MAX_REFLECTION_HZ * scale)
        .clamp(16.0, MAX_REFLECTION_HZ)
        .recip()
}

fn lake_in_view(camera: &Transform, projection: &Projection) -> bool {
    projection
        .compute_frustum(&GlobalTransform::from(*camera))
        .intersects_obb_identity(&LAKE_BOUNDS)
}

fn reflect_view(
    camera: &Transform,
    projection: &Projection,
    size: UVec2,
) -> (Transform, PerspectiveProjection) {
    let mirror = Mat4::from_scale_rotation_translation(
        Vec3::new(1.0, -1.0, 1.0),
        Quat::IDENTITY,
        Vec3::Y * (2.0 * WATER_LEVEL),
    );
    let camera = Transform::from_matrix(mirror * camera.to_matrix());
    let Projection::Perspective(main) = projection else {
        unreachable!("the hiker camera has a perspective projection")
    };
    let view_from_world = camera.compute_affine().inverse();
    let normal = view_from_world.transform_vector3(Vec3::Y).normalize();
    // Geometry under the lake would otherwise occlude the reflection.
    let clip = normal.extend(camera.translation.y - WATER_LEVEL - 0.015);
    let projection = PerspectiveProjection {
        near_clip_plane: clip,
        aspect_ratio: size.x as f32 / size.y as f32,
        ..*main
    };
    (camera, projection)
}
