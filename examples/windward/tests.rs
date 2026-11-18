use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use bevy::{
    animation::transition::AnimationTransitions,
    app::PluginsState,
    asset::RenderAssetUsages,
    camera::RenderTarget,
    light::{EnvironmentMapLight, Skybox},
    render::{
        RenderApp,
        gpu_readback::{Readback, ReadbackComplete},
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{
            CachedPipelineState, Extent3d, PipelineCache, PollType, TextureDimension,
            TextureFormat, TextureUsages,
        },
        renderer::RenderDevice,
    },
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_spaghetti::{Spaghetti3d, SpaghettiAsset};

use super::*;
use crate::{landscape::ground, player::Hiker, reeds::ReedPatch};

const WIDTH: usize = 320;
const HEIGHT: usize = 192;

#[derive(Clone, Copy, Debug)]
struct Frame {
    samples: u32,
    checksum: u64,
    brightness_range: (u8, u8),
    distinct_colors: usize,
}

#[derive(Resource)]
struct FrameCapture(Option<Frame>);

#[derive(Resource)]
struct ReviewFrames {
    destination: PathBuf,
    pixels: Vec<u8>,
}

#[test]
#[ignore = "requires a native GPU; runs without a window or any image files"]
fn full_scene_renders_in_background() {
    let mut app = headless_app();
    if let Some(destination) = std::env::var_os("WINDWARD_REVIEW_DIR") {
        let destination = PathBuf::from(destination);
        std::fs::create_dir_all(&destination).unwrap();
        app.insert_resource(ReviewFrames {
            destination,
            pixels: Vec::new(),
        });
    }
    app.insert_resource(FrameCapture(None))
        .add_systems(Startup, capture_offscreen.after(spawn_camera));
    warm_up(&mut app, |world| {
        world.resource::<FrameCapture>().0.is_some()
    });

    let samples = if app.world().contains_resource::<ReviewFrames>() {
        1280 * 800
    } else {
        WIDTH * HEIGHT
    };
    save_review_frame(app.world(), "trailhead.png");
    let before = app.world().resource::<FrameCapture>().0.unwrap();
    let start = hiker_position(app.world_mut());
    let beginning = reed_end(app.world_mut());
    let start_camera = camera_position(app.world_mut());

    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::KeyW);
    for _ in 0..34 {
        app.update();
    }
    let walked = hiker_position(app.world_mut());
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::ShiftLeft);
    for _ in 0..34 {
        app.update();
    }
    let ran = hiker_position(app.world_mut());
    app.world_mut().resource_mut::<player::Controls>().jump = true;
    for _ in 0..12 {
        app.update();
    }
    let jumped = hiker_position(app.world_mut());

    let after = app.world().resource::<FrameCapture>().0.unwrap();
    println!("Before: {before:?}, after running: {after:?}");
    assert_eq!(after.samples, samples as u32);
    assert!(
        after.brightness_range.1 > after.brightness_range.0 + 55,
        "flat exposure"
    );
    assert!(
        after.distinct_colors > 48,
        "terrain, sky, and hiker did not produce a varied frame"
    );
    assert_ne!(before.checksum, after.checksum);
    assert!(
        start.z - walked.z > 0.55,
        "W did not move the hiker along the trail"
    );
    assert!(
        walked.z - ran.z > 1.65,
        "Shift did not increase hiker speed"
    );
    assert!(
        jumped.y > ground(jumped.xz()) + 0.22,
        "jump never left the terrain"
    );
    assert!(
        camera_position(app.world_mut()).distance(start_camera) > 1.2,
        "camera did not follow"
    );
    assert!(
        reed_end(app.world_mut()).distance(beginning) > 0.002,
        "wind did not deform the reed asset"
    );

    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .reset_all();
    for _ in 0..45 {
        app.update();
    }
    let landed = hiker_position(app.world_mut());
    assert!(
        (landed.y - ground(landed.xz())).abs() < 1e-5,
        "hiker did not land"
    );
    if app.world().contains_resource::<ReviewFrames>() {
        pose_review_camera(&mut app, Vec2::new(-0.6, -7.0), -0.78, 0.10);
        save_review_frame(app.world(), "lakeshore.png");
        pose_review_camera(&mut app, Vec2::new(-5.7, -29.0), 0.18, 0.11);
        save_review_frame(app.world(), "pine-ridge.png");
        pose_review_camera(&mut app, Vec2::new(-9.5, -22.0), 0.9, 0.16);
        save_review_frame(app.world(), "fir-closeup.png");
        pose_review_camera(&mut app, Vec2::new(-8.0, -31.0), 2.4, -0.30);
        save_review_frame(app.world(), "meadow-low.png");
        pose_review_camera(&mut app, Vec2::new(3.2, 1.5), -0.2, -0.38);
        save_review_frame(app.world(), "grass-low.png");
        // Turn the hiker toward the camera to check the front in shade.
        pose_review_camera(&mut app, Vec2::new(3.2, 1.5), 2.6, -0.12);
        turn_hiker_to_camera(&mut app);
        save_review_frame(app.world(), "hiker-front.png");
    }
    assert_shaders_loaded(&app);
}

fn pose_review_camera(app: &mut App, at: Vec2, yaw: f32, pitch: f32) {
    let world = app.world_mut();
    let mut hiker = world.query_filtered::<&mut Transform, With<Hiker>>();
    let mut hiker = hiker.single_mut(world).unwrap();
    hiker.translation = Vec3::new(at.x, ground(at), at.y);
    hiker.rotation = Quat::from_rotation_y(yaw + std::f32::consts::PI);

    let mut rig = world.query::<&mut HikerCamera>();
    let mut rig = rig.single_mut(world).unwrap();
    rig.yaw = yaw;
    rig.pitch = pitch;
    for _ in 0..48 {
        app.update();
    }
}

fn turn_hiker_to_camera(app: &mut App) {
    let world = app.world_mut();
    let mut hiker = world.query_filtered::<&mut Transform, With<Hiker>>();
    let mut hiker = hiker.single_mut(world).unwrap();
    hiker.rotate_y(std::f32::consts::PI);
    for _ in 0..24 {
        app.update();
    }
}

fn save_review_frame(world: &World, name: &str) {
    let Some(review) = world.get_resource::<ReviewFrames>() else {
        return;
    };
    let image = Image::new(
        Extent3d {
            width: 1280,
            height: 800,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        review.pixels.clone(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD,
    );
    let path = review.destination.join(name);
    image
        .try_into_dynamic()
        .unwrap()
        .into_rgb8()
        .save(&path)
        .unwrap();
    println!("Offscreen render: {}", path.display());
}

#[test]
#[ignore = "release-mode throughput benchmark on a native GPU"]
fn benchmark_scene_offscreen() {
    let mut app = headless_app();
    app.add_systems(Startup, benchmark_target.after(spawn_camera));
    warm_up(&mut app, |_| true);
    let device = app.world().resource::<RenderDevice>().clone();
    device.poll(PollType::wait_indefinitely()).unwrap();
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::KeyW);

    const SAMPLES: usize = 10;
    const FRAMES_PER_SAMPLE: usize = 10;
    let mut frame_times = [0.0; SAMPLES];
    for sample in &mut frame_times {
        let start = Instant::now();
        for _ in 0..FRAMES_PER_SAMPLE {
            app.update();
        }
        device.poll(PollType::wait_indefinitely()).unwrap();
        *sample = start.elapsed().as_secs_f64() * 1000.0 / FRAMES_PER_SAMPLE as f64;
    }

    frame_times.sort_by(f64::total_cmp);
    let mean = frame_times.iter().sum::<f64>() / SAMPLES as f64;
    println!(
        "1280 × 800 offscreen walk, {} frames: {:.2} ms/frame mean ({:.1} FPS), {:.2} ms median block, {:.2} ms slowest block",
        SAMPLES * FRAMES_PER_SAMPLE,
        mean,
        1000.0 / mean,
        (frame_times[4] + frame_times[5]) * 0.5,
        frame_times[9]
    );
}

fn headless_app() -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(asset_plugin())
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    );
    add_scene(&mut app);
    app.insert_resource(TimeUpdateStrategy::FixedTimesteps(1));
    app
}

fn warm_up(app: &mut App, ready: impl Fn(&World) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while app.plugins_state() == PluginsState::Adding {
        assert!(Instant::now() < deadline, "GPU initialization timed out");
        std::thread::yield_now();
    }
    app.finish();
    app.cleanup();
    let mut stable_frames = 0;
    loop {
        assert!(
            Instant::now() < deadline,
            "the full scene never became ready offscreen"
        );
        app.update();
        if assets_ready(app.world_mut()) && ready(app.world()) {
            stable_frames += 1;
            if stable_frames >= 24 {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn benchmark_target(
    mut commands: Commands,
    camera: Single<Entity, With<HikerCamera>>,
    mut images: ResMut<Assets<Image>>,
) {
    let image = images.add(Image::new_target_texture(
        1280,
        800,
        TextureFormat::Rgba8UnormSrgb,
        None,
    ));
    commands
        .entity(*camera)
        .insert(RenderTarget::Image(image.into()));
}

fn assets_ready(world: &mut World) -> bool {
    let skinned = world
        .query::<(&AnimationPlayer, &AnimationTransitions)>()
        .iter(world)
        .count();
    let mesh_count = world.query::<&Mesh3d>().iter(world).count();
    let ibl = world
        .query::<&EnvironmentMapLight>()
        .iter(world)
        .any(|probe| {
            let images = world.resource::<Assets<Image>>();
            images.contains(&probe.diffuse_map) && images.contains(&probe.specular_map)
        });
    let sky = world.query::<&Skybox>().iter(world).any(|sky| {
        let images = world.resource::<Assets<Image>>();
        sky.image
            .as_ref()
            .and_then(|handle| images.get(handle))
            .is_some_and(|image| image.texture_descriptor.size.depth_or_array_layers == 6)
    });
    let assets = world.query::<&Spaghetti3d>().iter(world).all(|cloud| {
        world
            .resource::<Assets<SpaghettiAsset>>()
            .contains(&cloud.0)
    });
    let understory = world
        .query_filtered::<&WorldAssetRoot, With<understory::Understory>>()
        .iter(world)
        .all(|root| {
            world
                .resource::<AssetServer>()
                .is_loaded_with_dependencies(&root.0)
        });
    skinned > 0 && mesh_count >= 85 && ibl && sky && assets && understory
}

fn assert_shaders_loaded(app: &App) {
    let cache = app.sub_app(RenderApp).world().resource::<PipelineCache>();
    for pipeline in cache.pipelines() {
        if let CachedPipelineState::Err(error) = &pipeline.state {
            panic!("offscreen pipeline failed: {error}");
        }
    }
}

fn hiker_position(world: &mut World) -> Vec3 {
    world
        .query_filtered::<&Transform, With<Hiker>>()
        .single(world)
        .unwrap()
        .translation
}

fn camera_position(world: &mut World) -> Vec3 {
    world
        .query_filtered::<&Transform, With<HikerCamera>>()
        .single(world)
        .unwrap()
        .translation
}

fn reed_end(world: &mut World) -> Vec3 {
    let handles: Vec<_> = world
        .query_filtered::<&Spaghetti3d, With<ReedPatch>>()
        .iter(world)
        .map(|cloud| cloud.0.id())
        .collect();
    let assets = world.resource::<Assets<SpaghettiAsset>>();
    assets.get(handles[handles.len() / 2]).unwrap().segments()[2].end()
}

fn capture_offscreen(
    mut commands: Commands,
    camera: Single<Entity, With<HikerCamera>>,
    mut images: ResMut<Assets<Image>>,
    review: Option<Res<ReviewFrames>>,
) {
    // Both widths × 4 bytes are aligned for a GPU texture copy.
    let (width, height) = if review.is_some() {
        (1280, 800)
    } else {
        (WIDTH as u32, HEIGHT as u32)
    };
    let mut image = Image::new_target_texture(width, height, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let handle = images.add(image);
    commands
        .entity(*camera)
        .insert(RenderTarget::Image(handle.clone().into()));
    commands.spawn(Readback::texture(handle)).observe(
        |event: On<ReadbackComplete>,
         mut readback: ResMut<FrameCapture>,
         review: Option<ResMut<ReviewFrames>>| {
            let mut min = 255;
            let mut max = 0;
            let mut checksum = 0u64;
            let mut palette = [false; 4096];
            let pixels = event.data.as_chunks::<4>().0;
            for &[r, g, b, _] in pixels {
                let luminance = ((u16::from(r) * 2 + u16::from(g) * 5 + u16::from(b)) / 8) as u8;
                min = min.min(luminance);
                max = max.max(luminance);
                checksum = checksum.wrapping_mul(0x100000001b3)
                    ^ u64::from(u32::from_le_bytes([r, g, b, 0]));
                let quantize =
                    usize::from(r >> 4) * 256 + usize::from(g >> 4) * 16 + usize::from(b >> 4);
                palette[quantize] = true;
            }
            if max > 35 {
                if let Some(mut review) = review {
                    review.pixels.clone_from(&event.data);
                }
                readback.0 = Some(Frame {
                    samples: pixels.len() as u32,
                    brightness_range: (min, max),
                    checksum,
                    distinct_colors: palette.iter().filter(|&&present| present).count(),
                });
            }
        },
    );
}

