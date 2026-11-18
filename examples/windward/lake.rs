use bevy::{
    asset::RenderAssetUsages,
    camera::{CameraProjection, Exposure, Hdr, RenderTarget, visibility::RenderLayers},
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    image::ImageSampler,
    light::{GeneratedEnvironmentMapLight, NotShadowCaster, Skybox},
    pbr::{DistanceFog, ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::{AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat},
    shader::ShaderRef,
};

use bevy_spaghetti::{Spaghetti3d, SpaghettiAsset};

use crate::{
    landscape::{LAKE, SUN, SUN_COLOR, SUN_LUX, WATER_LEVEL, ground, load_tile, shore_distance},
    meadow::Meadow,
    player::HikerCamera,
    reeds::ReedPatch,
};

pub const LAKE_SURFACE_LAYER: usize = 2;
pub const REFLECTED_LAYER: usize = 3;
/// Half the 1280 x 800 main view in each axis.
const REFLECTION_SIZE: UVec2 = UVec2::new(640, 400);
const REFLECTION_PERIOD: f32 = 1.0 / 36.0;
const PLANE_SIZE: Vec2 = Vec2::new(28.0, 42.0);

#[derive(Resource)]
pub struct Lake(Handle<LakeMaterial>);

#[derive(Component, Default)]
pub struct ReflectedView {
    last_update: Option<f32>,
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
    lakebed_depth: Handle<Image>,
    #[texture(105)]
    #[sampler(106)]
    sediment: Handle<Image>,
    /// The bank albedo, so the shallowest water can match the soil around it.
    #[texture(107)]
    #[sampler(108)]
    bank: Handle<Image>,
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
    let (mirror, perspective) = reflect_view(transform, projection);
    let mut target = Image::new_target_texture(
        REFLECTION_SIZE.x,
        REFLECTION_SIZE.y,
        TextureFormat::Rgba16Float,
        None,
    );
    target.sampler = ImageSampler::linear();
    let reflection = images.add(target);
    commands.spawn((
        Name::new("Lake reflection camera"),
        ReflectedView::default(),
        Camera3d::default(),
        Camera {
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
            lakebed_depth: images.add(depth_texture()),
            sediment: load_tile(&server, "windward/trail-color.dds", true),
            bank: load_tile(&server, "windward/ground-color.dds", true),
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

/// Water depth in meters, negative under the bank. The signed values keep
/// the interpolated zero crossing on the real waterline, so the shore fade
/// is smooth at any texel size.
fn depth_texture() -> Image {
    const SIZE: UVec2 = UVec2::new(224, 336);
    let mut depths = Vec::with_capacity((SIZE.x * SIZE.y * 2) as usize);
    for row in 0..SIZE.y {
        for column in 0..SIZE.x {
            let uv = (Vec2::new(column as f32, row as f32) + 0.5) / SIZE.as_vec2();
            let at = LAKE + (uv - 0.5) * PLANE_SIZE;
            let depth = (WATER_LEVEL - ground(at)).clamp(-0.5, 2.5);
            depths.extend(f16_bits(depth).to_le_bytes());
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: SIZE.x,
            height: SIZE.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        depths,
        TextureFormat::R16Float,
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
    main: Single<(&Transform, &Projection), (With<HikerCamera>, Without<ReflectedView>)>,
    mut reflected: Single<(
        &mut Transform,
        &mut Projection,
        &mut Camera,
        &mut ReflectedView,
    )>,
    mut materials: ResMut<Assets<LakeMaterial>>,
) {
    add_shore_plants_to_reflection(&mut commands, &plants, &clouds);
    let (transform, projection) = *main;
    let (mirror, mirror_projection, camera, state) = &mut *reflected;
    let elapsed = time.elapsed_secs();
    let Some(mut water) = materials.get_mut(&lake.0) else {
        return;
    };
    water.extension.uniform.elapsed = elapsed;
    let delta_to_lake = Vec3::new(LAKE.x, WATER_LEVEL, LAKE.y) - transform.translation;
    let in_front = delta_to_lake.dot(transform.forward().as_vec3()) > -20.5;
    let refresh = state
        .last_update
        .is_none_or(|last_update| elapsed - last_update >= REFLECTION_PERIOD);
    camera.is_active = in_front && refresh;
    if camera.is_active {
        let (reflection, perspective) = reflect_view(transform, projection);
        water.extension.uniform.reflected_clip_from_world =
            perspective.get_clip_from_view() * reflection.compute_affine().inverse();
        **mirror = reflection;
        **mirror_projection = Projection::Perspective(perspective);
        state.last_update = Some(elapsed);
    }
}

fn reflect_view(camera: &Transform, projection: &Projection) -> (Transform, PerspectiveProjection) {
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
        aspect_ratio: REFLECTION_SIZE.x as f32 / REFLECTION_SIZE.y as f32,
        ..*main
    };
    (camera, projection)
}
