//! Run with `cargo test --test headless -- --ignored --nocapture`.
#![cfg(feature = "render")]

use std::time::{Duration, Instant};

use bevy::{
    app::PluginsState,
    camera::{RenderTarget, ScalingMode, visibility::RenderLayers},
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{TextureFormat, TextureUsages},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_spaghetti::{
    DetailTransition, GaussianSegment, Spaghetti3d, SpaghettiAdditive, SpaghettiAsset,
    SpaghettiCutout, SpaghettiDetailScale, SpaghettiFlow, SpaghettiGroundAmbient,
    SpaghettiLighting, SpaghettiPlugin, SpaghettiSpecular, SpaghettiSway,
};

const SIZE: usize = 128;

#[derive(Resource, Default)]
struct Samples {
    strand: Option<[[u8; 4]; 16]>,
    bust: Option<usize>,
    effects: Option<[[u8; 4]; 6]>,
}

#[derive(Component)]
struct EditableCloud;

#[test]
#[ignore = "requires a native GPU adapter; no display server required"]
fn renders_strands_and_bundled_asset_offscreen() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(AssetPlugin {
                file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets").to_owned(),
                ..default()
            })
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
        SpaghettiPlugin,
    ))
    .insert_resource(ClearColor(Color::BLACK))
    .init_resource::<Samples>()
    .add_systems(Startup, setup);
    let deadline = Instant::now() + Duration::from_secs(20);
    while app.plugins_state() == PluginsState::Adding {
        assert!(Instant::now() < deadline, "GPU initialization timed out");
        std::thread::yield_now();
    }
    app.finish();
    app.cleanup();

    while Instant::now() < deadline {
        app.update();
        let samples = app.world().resource::<Samples>();
        if let (Some(pixels), Some(bust), Some(effects)) =
            (samples.strand, samples.bust, samples.effects)
        {
            let [added, covered, glint, matte, flow_start, flow_middle] = effects;
            println!("effects: {effects:?}");
            assert!(
                added[0] > 200 && added[1] > 240,
                "additive splat did not keep the color behind it"
            );
            assert!(
                covered[0] > 200 && covered[1] < 60,
                "blended splat did not cover the color behind it"
            );
            assert!(
                glint[0] > 120 && matte[0] < 20,
                "highlight did not follow the segment normals"
            );
            assert!(
                flow_middle[0] > 200 && flow_start[0] < 20,
                "flow did not carry the splat half of its loop"
            );
            let [
                left,
                joint,
                blocked,
                gap,
                point,
                blended,
                tapered,
                lit,
                unlit,
                cutout_root_edge,
                cutout_tip,
                cutout_tip_edge,
                shared_fine,
                shared_coarse,
                sky_light,
                ground_light,
            ] = pixels;
            println!(
                "strand: {pixels:?}, bust coverage: {bust} / {}",
                SIZE * SIZE
            );
            assert!(left[0] > 200 && joint[0] > 200, "strand missing");
            assert!(
                left[0].abs_diff(joint[0]) <= 3,
                "opacity changed at the joint"
            );
            assert!(
                blocked[1] > 240 && blocked[0] < 5,
                "opaque depth did not block splat"
            );
            assert!(
                gap[0..3].iter().all(|&v| v < 5),
                "expected clear background"
            );
            assert!(
                point[1] > 200 && point[2] > 200,
                "point limit did not render"
            );
            assert!(
                blended[0] > blended[2] && blended[2] > 40,
                "wrong transparent depth order"
            );
            assert!(
                tapered[0] > 30 && tapered[0] < left[0],
                "missing Gaussian end taper"
            );
            assert!((1500..9000).contains(&bust), "unexpected bust coverage");
            assert!(
                lit[0] > 200 && unlit[0] > 35 && lit[0] > unlit[0] + 100,
                "oriented splats did not receive distinct lighting"
            );
            assert!(
                cutout_root_edge[1] > 200 && cutout_tip[1] > 200 && cutout_tip_edge[1] < 5,
                "cutout did not narrow toward its tip"
            );
            assert!(
                shared_fine[0] > 240
                    && shared_fine[1] < 5
                    && shared_coarse[0] < 5
                    && shared_coarse[1] > 240,
                "instances of the same cloud did not choose distinct detail levels"
            );
            assert!(
                sky_light[2] > 220
                    && sky_light[0] < 5
                    && ground_light[0] > 220
                    && ground_light[2] < 5,
                "ambient hemisphere did not follow the world-space normals"
            );
            return check_same_size_update(&mut app);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("no complete offscreen result before deadline");
}

fn check_same_size_update(app: &mut App) {
    let world = app.world_mut();
    let (entity, cloud) = world
        .query_filtered::<(Entity, &Spaghetti3d), With<EditableCloud>>()
        .single(world)
        .unwrap();
    let cloud = cloud.0.clone();
    let mut assets = world.resource_mut::<Assets<SpaghettiAsset>>();
    let mut segments = assets.get(&cloud).unwrap().segments().to_vec();
    // Flip the points' depths while preserving the source buffer size and
    // camera pose. The upload and per-view draw order must both refresh.
    for segment in &mut segments[2..4] {
        let position = segment.start() * Vec3::new(1.0, 1.0, -1.0);
        *segment = GaussianSegment::new(
            position,
            position,
            segment.sigma(),
            segment.rotation(),
            segment.mass(),
            segment.color(),
        );
    }
    segments[2] = segments[2].with_bend_heights(1.0, 1.0);
    assets
        .insert(cloud.id(), SpaghettiAsset::new(segments).unwrap())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        app.update();
        let samples = app.world().resource::<Samples>().strand.unwrap();
        if samples[5][2] > samples[5][0] {
            println!("same-size edit reversed depth order: {:?}", samples[5]);
            app.world_mut()
                .entity_mut(entity)
                .insert(SpaghettiSway(Vec3::Z * 0.55));
            return check_sway_order(app);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("same-size edit did not update overlapping point depth order");
}

fn check_sway_order(app: &mut App) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        app.update();
        let samples = app.world().resource::<Samples>().strand.unwrap();
        if samples[5][0] > samples[5][2] && samples[5][2] > 40 {
            return;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("sway did not move the red point ahead of the pinned blue point");
}

fn setup(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut splats: ResMut<Assets<SpaghettiAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let make_splat = |a: Vec3, b: Vec3, mass, color| {
        GaussianSegment::new(a, b, Vec3::splat(0.08), Quat::IDENTITY, mass, color)
    };
    // Asset order puts the nearer red Gaussian before the farther blue one.
    let near = Vec3::new(0.0, -0.9, 0.2);
    let far = Vec3::new(0.0, -0.9, -0.2);
    let point = Vec3::new(0.6, 0.9, 0.0);
    let segments = vec![
        make_splat(-Vec3::X, Vec3::ZERO, 0.5, Vec3::X),
        make_splat(Vec3::ZERO, Vec3::X, 0.5, Vec3::X),
        make_splat(near, near, 0.04, Vec3::X),
        make_splat(far, far, 0.04, Vec3::Z),
        make_splat(point, point, 0.12, Vec3::Y + Vec3::Z),
    ];
    commands.spawn((
        EditableCloud,
        Spaghetti3d(splats.add(SpaghettiAsset::new(segments).unwrap())),
    ));
    let cutout = GaussianSegment::new(
        Vec3::new(-1.4, 1.45, 0.0),
        Vec3::new(1.4, 1.45, 0.0),
        Vec3::splat(0.16),
        Quat::IDENTITY,
        2.2,
        Vec3::Y,
    )
    .with_cutout_taper(0.20);
    commands.spawn((
        Spaghetti3d(splats.add(SpaghettiAsset::new(vec![cutout]).unwrap())),
        SpaghettiCutout,
    ));
    let levels =
        [Vec3::X, Vec3::Y].map(|color| make_splat(-0.3 * Vec3::X, 0.3 * Vec3::X, 0.35, color));
    let shared = splats.add(
        SpaghettiAsset::with_detail(
            levels.to_vec(),
            &[DetailTransition {
                finer_end: 1,
                pixels_per_unit: [14.0, 24.0],
            }],
        )
        .unwrap(),
    );
    commands.spawn((
        Spaghetti3d(shared.clone()),
        SpaghettiCutout,
        Transform::from_xyz(-1.0, -1.48, 0.0),
    ));
    commands.spawn((
        Spaghetti3d(shared),
        SpaghettiCutout,
        SpaghettiDetailScale(4.0),
        Transform::from_xyz(1.0, -1.48, 0.0),
    ));
    let lit_points = [
        (
            Vec3::new(-1.14, 0.9, 0.0),
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        ),
        (Vec3::new(-0.58, 0.9, 0.0), Quat::IDENTITY),
    ]
    .into_iter()
    .map(|(pos, rotation)| {
        GaussianSegment::new(pos, pos, Vec3::splat(0.08), rotation, 0.12, Vec3::ONE)
    })
    .collect();
    commands.spawn((
        Spaghetti3d(splats.add(SpaghettiAsset::new(lit_points).unwrap())),
        SpaghettiLighting {
            direction_to_light: Vec3::X,
            ambient: Vec3::splat(0.045),
            directional: Vec3::splat(0.85),
        },
    ));
    let hemisphere_points = [
        (-0.9, Quat::IDENTITY),
        (0.85, Quat::from_rotation_y(std::f32::consts::PI)),
    ]
    .into_iter()
    .map(|(x, rotation)| {
        let pos = Vec3::X * x;
        GaussianSegment::new(pos, pos, Vec3::splat(0.08), rotation, 0.12, Vec3::ONE)
    })
    .collect();
    commands.spawn((
        Spaghetti3d(splats.add(SpaghettiAsset::new(hemisphere_points).unwrap())),
        // Turn the two local normals into world-space up and down.
        Transform::from_xyz(0.0, 0.43, 0.0)
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        SpaghettiLighting {
            direction_to_light: Vec3::X,
            ambient: Vec3::Z,
            directional: Vec3::ZERO,
        },
        SpaghettiGroundAmbient(Vec3::X),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.35, 0.55, 0.1))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.0, 1.0, 0.0),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.5, 0.0, 0.5),
    ));
    spawn_effects(
        &mut commands,
        &mut images,
        &mut splats,
        &mut meshes,
        &mut materials,
    );
    let strand_target = target(&mut images);
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(strand_target.clone().into()),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 4.0,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 0.0, 4.0),
        Tonemapping::Linear,
        Msaa::Off,
    ));
    commands.spawn(Readback::texture(strand_target)).observe(
        |readback: On<ReadbackComplete>, mut samples: ResMut<Samples>| {
            let pixel = |x, y| {
                let offset: usize = (y * SIZE + x) * 4;
                <[u8; 4]>::try_from(&readback.data[offset..offset + 4]).unwrap()
            };
            let pixels = [
                pixel(48, 64),
                pixel(64, 64),
                pixel(80, 64),
                pixel(64, 12),
                pixel(83, 35),
                pixel(64, 92),
                pixel(99, 64),
                pixel(27, 35),
                pixel(45, 35),
                pixel(29, 22),
                pixel(101, 17),
                pixel(101, 22),
                pixel(32, 111),
                pixel(96, 111),
                pixel(35, 50),
                pixel(91, 50),
            ];
            if pixels[2][1] > 240
                && pixels[0][0] > 200
                && pixels[7][0] > 5
                && pixels[10][1] > 5
                && pixels[12][0].max(pixels[12][1]) > 5
                && pixels[13][0].max(pixels[13][1]) > 5
                && pixels[14][0].max(pixels[14][2]) > 5
                && pixels[15][0].max(pixels[15][2]) > 5
            {
                samples.strand = Some(pixels);
            }
        },
    );

    let center = Vec3::new(0.013, 0.257, 0.005);
    commands.spawn((
        Spaghetti3d(server.load("marble_bust_01.spag")),
        RenderLayers::layer(1),
    ));
    let bust_target = target(&mut images);
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(bust_target.clone().into()),
        RenderLayers::layer(1),
        Transform::from_translation(center + Vec3::new(0.25, 0.10, 0.78))
            .looking_at(center, Vec3::Y),
        Tonemapping::Linear,
        Msaa::Off,
    ));
    commands.spawn(Readback::texture(bust_target)).observe(
        |readback: On<ReadbackComplete>, mut samples: ResMut<Samples>| {
            let colored = readback
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|pixel| pixel[..3].iter().all(|&channel| channel > 25))
                .count();
            if colored > 100 {
                samples.bust = Some(colored);
            }
        },
    );
}

/// Spawns the additive, highlight and flow cases on render layer 2.
/// The camera maps one unit to 32 pixels around pixel (64, 64).
fn spawn_effects(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    splats: &mut Assets<SpaghettiAsset>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let layer = RenderLayers::layer(2);
    let point = |at: Vec3, rotation, color| {
        GaussianSegment::new(at, at, Vec3::splat(0.08), rotation, 0.3, color)
    };
    let mut cloud = |segments| Spaghetti3d(splats.add(SpaghettiAsset::new(segments).unwrap()));
    let green = materials.add(StandardMaterial {
        base_color: Color::linear_rgb(0.0, 1.0, 0.0),
        unlit: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.7, 0.6, 0.1))),
        MeshMaterial3d(green),
        Transform::from_xyz(-0.5, 1.0, -0.5),
        layer.clone(),
    ));
    commands.spawn((
        cloud(vec![point(Vec3::new(-1.0, 1.0, 0.0), Quat::IDENTITY, Vec3::X)]),
        SpaghettiAdditive,
        layer.clone(),
    ));
    commands.spawn((
        cloud(vec![point(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, Vec3::X)]),
        layer.clone(),
    ));
    // Black splats show only the highlight. One faces the light and the eye,
    // and the other is edge-on to both.
    commands.spawn((
        cloud(vec![
            point(Vec3::new(1.0, 1.0, 0.0), Quat::IDENTITY, Vec3::ZERO),
            point(
                Vec3::new(1.6, 1.0, 0.0),
                Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
                Vec3::ZERO,
            ),
        ]),
        SpaghettiLighting {
            direction_to_light: Vec3::Z,
            ambient: Vec3::ZERO,
            directional: Vec3::X,
        },
        SpaghettiSpecular {
            strength: 1.0,
            shininess: 20.0,
            strand: 0.0,
        },
        layer.clone(),
    ));
    // The shader gives segment 0 this start time within the loop.
    let state = 2_891_336_453_u32;
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    let start = (((word >> 22) ^ word) >> 8) as f32 / 16_777_216.0;
    let period = 2.0;
    commands.spawn((
        cloud(vec![point(Vec3::new(-1.0, -1.0, 0.0), Quat::IDENTITY, Vec3::X)]),
        SpaghettiFlow {
            time: period * (1.5 - start),
            velocity: Vec3::X,
            period,
            fade: [0.0, 0.0],
            ..default()
        },
        layer.clone(),
    ));
    let effects_target = target(images);
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(effects_target.clone().into()),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 4.0,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 0.0, 4.0),
        Tonemapping::Linear,
        Msaa::Off,
        layer,
    ));
    commands.spawn(Readback::texture(effects_target)).observe(
        |readback: On<ReadbackComplete>, mut samples: ResMut<Samples>| {
            let pixel = |x, y| {
                let offset: usize = (y * SIZE + x) * 4;
                <[u8; 4]>::try_from(&readback.data[offset..offset + 4]).unwrap()
            };
            let pixels = [
                pixel(32, 32),
                pixel(64, 32),
                pixel(96, 32),
                pixel(115, 32),
                pixel(32, 96),
                pixel(64, 96),
            ];
            if pixels[0][1] > 200 && pixels[1][0] > 5 {
                samples.effects = Some(pixels);
            }
        },
    );
}

fn target(images: &mut Assets<Image>) -> Handle<Image> {
    // 128 × 4 bytes per row is already aligned for GPU texture copies.
    let mut image = Image::new_target_texture(
        SIZE as u32,
        SIZE as u32,
        TextureFormat::Rgba8UnormSrgb,
        None,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    images.add(image)
}
