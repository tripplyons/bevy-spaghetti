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

use crate::{
    landscape::{LAKE, SUN, SUN_COLOR, SUN_LUX, WATER_LEVEL, ground, load_tile},
    player::HikerCamera,
};

pub const LAKE_SURFACE_LAYER: usize = 2;
pub const REFLECTED_LAYER: usize = 3;
const REFLECTION_SIZE: UVec2 = UVec2::new(512, 320);
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

fn depth_texture() -> Image {
    const SIZE: UVec2 = UVec2::new(160, 240);
    let mut depths = Vec::with_capacity((SIZE.x * SIZE.y) as usize);
    for row in 0..SIZE.y {
        for column in 0..SIZE.x {
            let uv = (Vec2::new(column as f32, row as f32) + 0.5) / SIZE.as_vec2();
            let at = LAKE + (uv - 0.5) * PLANE_SIZE;
            let depth = (WATER_LEVEL - ground(at)).clamp(0.0, 2.0);
            depths.push((depth * 127.5).round() as u8);
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
        TextureFormat::R8Unorm,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::linear();
    image
}

pub fn reflect_lake(
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
