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
    GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiLighting, SpaghettiPlugin,
};

const SIZE: usize = 128;

#[derive(Resource, Default)]
struct Samples {
    strand: Option<[[u8; 4]; 9]>,
    bust: Option<usize>,
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
        if let (Some(pixels), Some(bust)) = (samples.strand, samples.bust) {
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
            return check_same_size_update(&mut app);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("no complete offscreen result before deadline");
}

fn check_same_size_update(app: &mut App) {
    let world = app.world_mut();
    let cloud = world
        .query_filtered::<&Spaghetti3d, With<EditableCloud>>()
        .single(world)
        .unwrap()
        .0
        .clone();
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
    assets
        .insert(cloud.id(), SpaghettiAsset::new(segments).unwrap())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        app.update();
        let samples = app.world().resource::<Samples>().strand.unwrap();
        if samples[5][2] > samples[5][0] {
            println!("same-size edit reversed depth order: {:?}", samples[5]);
            return;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("same-size edit did not update overlapping point depth order");
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
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.35, 0.55, 0.1))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.0, 1.0, 0.0),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.5, 0.0, 0.5),
    ));
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
            ];
            if pixels[2][1] > 240 && pixels[0][0] > 200 && pixels[7][0] > 5 {
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
