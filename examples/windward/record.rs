//! Renders a scripted walk through Windward to a 60 fps video file.
//!
//! Run with `cargo run --release --example windward -- --record clip.mp4`.
//! The recorder runs without a window and steps the simulation by exactly
//! one frame per update, so the clip plays back smoothly at 60 fps however
//! long each frame takes to render. `ffmpeg` must be on the `PATH`.
//!
//! Options:
//! - `--size WIDTHxHEIGHT`: output size, 1920x1080 by default.
//! - `--shutter N`: average N rendered subframes into each video frame, for
//!   motion blur. 1 turns blur off. The default of 2 blends two subframes
//!   1/120 s apart.
//! - `--crf N`: x264 quality, 15 by default. Higher numbers make smaller
//!   files; 20 roughly halves the size.
//! - `--map PATH.png`: also write a top-down map of the route and the
//!   obstacles it avoids.

use std::{
    f32::consts::{PI, TAU},
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

use bevy::{
    animation::transition::AnimationTransitions,
    app::PluginsState,
    asset::RenderAssetUsages,
    camera::RenderTarget,
    light::{EnvironmentMapLight, Skybox},
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{Extent3d, PollType, TextureDimension, TextureFormat, TextureUsages},
        renderer::RenderDevice,
    },
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_spaghetti::{Spaghetti3d, SpaghettiAsset};

use super::*;
use crate::{
    landscape::{Obstacles, WATER_LEVEL, ground, trail_x},
    player::{Controls, Hiker},
};

const FPS: u32 = 60;
/// Video frames rendered before the clip starts, so the follow camera settles.
const PREROLL: usize = 45;
/// Video frames held after the last beat, while the fade to black plays.
const TAIL: usize = 75;
const FADE_IN: f32 = 0.8;
const FADE_OUT: f32 = 1.1;

/// The hiker keeps this far from trunks, stumps, and stones on the route.
const CLEARANCE: f32 = 0.85;
/// Distance between points of the sampled route.
const SPACING: f32 = 0.2;

/// What the hiker and camera do for one part of the clip.
struct Beat {
    action: Action,
    /// Camera yaw relative to the hiker's heading, in radians. Positive
    /// values swing the camera to the hiker's right side, looking left.
    orbit: f32,
    /// Camera pitch. Positive values raise the camera to look down.
    pitch: f32,
}

enum Action {
    /// Follow a smooth curve through these ground points.
    Go {
        through: &'static [(f32, f32)],
        run: bool,
        /// Jump after covering these distances along the curve, in meters.
        jumps: &'static [f32],
    },
    /// Stand still while the camera moves.
    Hold { seconds: f32 },
}

const fn go(through: &'static [(f32, f32)], run: bool, jumps: &'static [f32]) -> Action {
    Action::Go {
        through,
        run,
        jumps,
    }
}

/// The route. The hiker starts at the trailhead, walks past the camp to the
/// lake, runs and jumps down the trail into the spruce stand, crosses the
/// meadow below the lake, and climbs the far hillside toward the low sun.
const SCRIPT: &[Beat] = &[
    // Walk south down the trail, past the camp.
    Beat {
        action: go(&[(-1.0, 4.5), (-0.4, -0.5), (0.2, -4.0)], false, &[]),
        orbit: -0.35,
        pitch: 0.16,
    },
    // Step off the trail onto the bank and stop to look over the lake.
    Beat {
        action: go(&[(-0.3, -6.0), (0.9, -6.8)], false, &[]),
        orbit: -0.3,
        pitch: 0.12,
    },
    Beat {
        action: Action::Hold { seconds: 3.4 },
        orbit: -1.6,
        pitch: 0.07,
    },
    Beat {
        action: Action::Hold { seconds: 1.4 },
        orbit: 0.0,
        pitch: 0.12,
    },
    // Run down the trail with two jumps.
    Beat {
        action: go(
            &[(-1.6, -11.0), (-3.6, -15.5), (-6.0, -19.0)],
            true,
            &[4.5, 11.5],
        ),
        orbit: 0.55,
        pitch: 0.05,
    },
    // Slow to a walk under the spruces.
    Beat {
        action: go(&[(-8.6, -22.2), (-8.4, -25.6), (-6.4, -29.0)], false, &[]),
        orbit: -0.45,
        pitch: 0.20,
    },
    // Out into the meadow below the lake, then a running jump.
    Beat {
        action: go(
            &[
                (-3.0, -34.5),
                (3.5, -37.0),
                (10.0, -35.2),
                (16.0, -33.6),
                (21.0, -33.4),
            ],
            true,
            &[9.5, 17.0],
        ),
        orbit: -0.6,
        pitch: 0.06,
    },
    // Climb the hillside and stop facing the valley and the sun.
    Beat {
        action: go(&[(25.0, -34.8), (27.4, -37.0), (28.0, -38.5)], false, &[]),
        orbit: -0.45,
        pitch: 0.12,
    },
    Beat {
        action: Action::Hold { seconds: 4.0 },
        orbit: -2.4,
        pitch: -0.05,
    },
];

struct Options {
    output: PathBuf,
    size: UVec2,
    shutter: u32,
    crf: u32,
    map: Option<PathBuf>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Options {
            output: PathBuf::new(),
            size: UVec2::new(1920, 1080),
            shutter: 2,
            crf: 15,
            map: None,
        };
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .cloned()
                    .ok_or_else(|| format!("{arg} needs a value"))
            };
            match arg.as_str() {
                "--record" => options.output = value()?.into(),
                "--size" => {
                    let size = value()?;
                    let (width, height) = size
                        .split_once('x')
                        .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                        .ok_or_else(|| format!("bad --size {size}, expected WIDTHxHEIGHT"))?;
                    options.size = UVec2::new(width, height);
                }
                "--shutter" => {
                    options.shutter = value()?
                        .parse()
                        .ok()
                        .filter(|&count| count >= 1)
                        .ok_or("--shutter needs a whole number of at least 1")?;
                }
                "--crf" => {
                    options.crf = value()?
                        .parse()
                        .ok()
                        .filter(|&crf| crf <= 51)
                        .ok_or("--crf needs a whole number from 0 to 51")?;
                }
                "--map" => options.map = Some(value()?.into()),
                other => return Err(format!("unknown option {other}")),
            }
        }
        if options.output.as_os_str().is_empty() {
            return Err("--record needs an output path".into());
        }
        if !options.size.x.is_multiple_of(2) || !options.size.y.is_multiple_of(2) {
            return Err("--size needs even dimensions for the video encoder".into());
        }
        Ok(options)
    }
}

/// Returns true if the arguments ask for a recording.
pub fn requested() -> bool {
    std::env::args().any(|arg| arg == "--record")
}

pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let options = Options::parse(&args).unwrap_or_else(|error| {
        eprintln!("windward: {error}");
        eprintln!(
            "usage: windward --record OUT.mp4 [--size 1920x1080] [--shutter 2] [--crf 15] [--map ROUTE.png]"
        );
        std::process::exit(2);
    });
    let started = Instant::now();
    let mut app = headless_app();
    app.insert_resource(TargetSize(options.size))
        .init_resource::<Captured>()
        .add_systems(Startup, capture_target.after(spawn_camera))
        .add_systems(PreUpdate, direct.after(player::read_controls));
    println!("Loading the scene");
    warm_up(&mut app, |_| true);

    let routes = plan_routes(app.world().resource::<Obstacles>());
    if let Some(map) = &options.map {
        draw_map(app.world().resource::<Obstacles>(), &routes, map);
    }

    // From here on, each update advances exactly one subframe.
    let subframe = Duration::from_secs_f64(1.0 / f64::from(FPS * options.shutter));
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .set_timestep(subframe);
    app.insert_resource(TimeUpdateStrategy::FixedTimesteps(1));
    let device = app.world().resource::<RenderDevice>().clone();
    let step = |app: &mut App| {
        app.update();
        // Wait for the GPU so each readback arrives on the next update.
        device.poll(PollType::wait_indefinitely()).unwrap();
        std::mem::take(&mut app.world_mut().resource_mut::<Captured>().0)
    };
    for _ in 0..PREROLL as u32 * options.shutter {
        step(&mut app);
    }
    let director = Director::new(routes, app.world_mut());
    app.insert_resource(director);

    let encoder = Encoder::start(&options);
    let mut blend = Blend::new(options.size, options.shutter);
    // Readbacks lag one update behind, so the first update returns a preroll
    // frame and one more update after the end collects the last subframe.
    let mut skip = 1;
    let mut frames = 0usize;
    let mut tail = 0;
    let mut last_report = Instant::now();
    loop {
        let done = app.world().resource::<Director>().finished;
        if done {
            tail += 1;
        }
        let captured = step(&mut app);
        assert_eq!(captured.len(), 1, "expected one readback per update");
        for pixels in captured {
            if skip > 0 {
                skip -= 1;
                continue;
            }
            if let Some(frame) = blend.add(&pixels, options.size) {
                encoder.send(frame);
                frames += 1;
            }
        }
        if last_report.elapsed() > Duration::from_secs(5) {
            last_report = Instant::now();
            let director = app.world().resource::<Director>();
            println!(
                "{frames} frames ({:.1} s of video), beat {} of {}",
                frames as f32 / FPS as f32,
                (director.beat + 1).min(SCRIPT.len()),
                SCRIPT.len(),
            );
        }
        if tail > TAIL * options.shutter as usize {
            break;
        }
    }
    let seconds = frames as f32 / FPS as f32;
    encoder.finish(seconds);
    println!(
        "Wrote {} ({frames} frames, {seconds:.1} s at {} × {}, {FPS} fps) in {:.0} s",
        options.output.display(),
        options.size.x,
        options.size.y,
        started.elapsed().as_secs_f32(),
    );
}

#[derive(Resource)]
struct TargetSize(UVec2);

/// Subframes read back from the GPU since the recorder last took them.
#[derive(Resource, Default)]
struct Captured(Vec<Vec<u8>>);

fn capture_target(
    mut commands: Commands,
    camera: Single<Entity, With<HikerCamera>>,
    mut images: ResMut<Assets<Image>>,
    size: Res<TargetSize>,
) {
    let mut image =
        Image::new_target_texture(size.0.x, size.0.y, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let handle = images.add(image);
    commands
        .entity(*camera)
        .insert(RenderTarget::Image(handle.clone().into()));
    commands.spawn(Readback::texture(handle)).observe(
        |event: On<ReadbackComplete>, mut captured: ResMut<Captured>| {
            captured.0.push(event.data.clone());
        },
    );
}

/// Averages subframes in linear light and returns each finished RGBA frame.
struct Blend {
    count: u32,
    shutter: u32,
    sum: Vec<f32>,
    to_linear: [f32; 256],
}

impl Blend {
    fn new(size: UVec2, shutter: u32) -> Self {
        let to_linear = std::array::from_fn(|value| {
            let value = value as f32 / 255.0;
            if value <= 0.04045 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        });
        Self {
            count: 0,
            shutter,
            sum: vec![0.0; (size.x * size.y * 3) as usize],
            to_linear,
        }
    }

    fn add(&mut self, padded: &[u8], size: UVec2) -> Option<Vec<u8>> {
        // GPU readback rows are padded to 256 bytes.
        let row = (size.x * 4) as usize;
        let stride = row.next_multiple_of(256);
        if self.shutter == 1 {
            let mut frame = Vec::with_capacity(row * size.y as usize);
            for y in 0..size.y as usize {
                frame.extend_from_slice(&padded[y * stride..y * stride + row]);
            }
            return Some(frame);
        }
        for y in 0..size.y as usize {
            let source = &padded[y * stride..y * stride + row];
            let sum = &mut self.sum[y * size.x as usize * 3..][..size.x as usize * 3];
            let pixels = source.as_chunks::<4>().0;
            for (pixel, total) in pixels.iter().zip(sum.as_chunks_mut::<3>().0) {
                for (total, &value) in total.iter_mut().zip(pixel) {
                    *total += self.to_linear[value as usize];
                }
            }
        }
        self.count += 1;
        if self.count < self.shutter {
            return None;
        }
        let scale = 1.0 / self.shutter as f32;
        let mut frame = Vec::with_capacity(row * size.y as usize);
        for total in self.sum.as_chunks_mut::<3>().0 {
            for channel in total.iter_mut() {
                let value = (*channel * scale).clamp(0.0, 1.0);
                let encoded = if value <= 0.0031308 {
                    value * 12.92
                } else {
                    1.055 * value.powf(1.0 / 2.4) - 0.055
                };
                frame.push((encoded * 255.0 + 0.5) as u8);
                *channel = 0.0;
            }
            frame.push(255);
        }
        self.count = 0;
        Some(frame)
    }
}

/// An `ffmpeg` process fed raw frames from a writer thread, so encoding
/// overlaps rendering.
struct Encoder {
    sender: Option<mpsc::SyncSender<Vec<u8>>>,
    writer: Option<std::thread::JoinHandle<std::process::Child>>,
    part: PathBuf,
    output: PathBuf,
    crf: u32,
}

impl Encoder {
    fn start(options: &Options) -> Self {
        // The fade-out needs the clip length, which is known only at the
        // end, so the first pass writes an intermediate file.
        let part = options.output.with_extension("part.mkv");
        let mut child = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "rawvideo", "-pix_fmt", "rgba"])
            .args(["-s", &format!("{}x{}", options.size.x, options.size.y)])
            .args(["-framerate", &FPS.to_string(), "-i", "-"])
            // Near-lossless and fast, for the second pass to finish.
            .args(["-c:v", "libx264", "-preset", "veryfast", "-qp", "4"])
            .args(["-pix_fmt", "yuv444p"])
            .arg(&part)
            .stdin(Stdio::piped())
            .spawn()
            .expect("failed to start ffmpeg; is it installed and on the PATH?");
        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(6);
        let mut stdin = child.stdin.take().unwrap();
        let writer = std::thread::spawn(move || {
            for frame in receiver {
                stdin
                    .write_all(&frame)
                    .expect("ffmpeg stopped reading frames");
            }
            drop(stdin);
            child
        });
        Self {
            sender: Some(sender),
            writer: Some(writer),
            part,
            output: options.output.clone(),
            crf: options.crf,
        }
    }

    fn send(&self, frame: Vec<u8>) {
        self.sender.as_ref().unwrap().send(frame).unwrap();
    }

    fn finish(mut self, seconds: f32) {
        drop(self.sender.take());
        let mut child = self.writer.take().unwrap().join().unwrap();
        assert!(child.wait().unwrap().success(), "ffmpeg failed");
        println!("Encoding {}", self.output.display());
        let filters = format!(
            "fade=t=in:st=0:d={FADE_IN},fade=t=out:st={:.3}:d={FADE_OUT},\
             scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            (seconds - FADE_OUT).max(0.0),
        );
        let status = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(&self.part)
            .args(["-vf", &filters])
            .args(["-c:v", "libx264", "-preset", "slow"])
            .args(["-crf", &self.crf.to_string()])
            .args(["-profile:v", "high", "-tune", "film"])
            .args([
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
            ])
            .args(["-movflags", "+faststart"])
            .arg(&self.output)
            .status()
            .expect("failed to run ffmpeg");
        assert!(status.success(), "ffmpeg failed to encode the final clip");
        std::fs::remove_file(&self.part).ok();
    }
}

/// A route through the beat's points, sampled at even spacing.
struct Route {
    points: Vec<Vec2>,
    /// Distance along the route at each point.
    along: Vec<f32>,
}

impl Route {
    fn length(&self) -> f32 {
        *self.along.last().unwrap()
    }

    /// The point a given distance along the route.
    fn at(&self, distance: f32) -> Vec2 {
        let index = self.along.partition_point(|&along| along < distance);
        self.points[index.min(self.points.len() - 1)]
    }
}

/// Builds smooth routes through the script's points that keep clear of
/// obstacles and stay on dry ground.
fn plan_routes(obstacles: &Obstacles) -> Vec<Option<Route>> {
    let mut previous = start_point();
    let mut before = previous + Vec2::Y;
    SCRIPT
        .iter()
        .map(|beat| {
            let Action::Go { through, .. } = beat.action else {
                return None;
            };
            let mut controls = vec![before, previous];
            controls.extend(through.iter().map(|&(x, z)| Vec2::new(x, z)));
            let last = controls[controls.len() - 1];
            let next = last + (last - controls[controls.len() - 2]);
            controls.push(next);
            // Centripetal Catmull-Rom through the points.
            let mut points = Vec::new();
            for window in controls.windows(4) {
                let length = window[1].distance(window[2]);
                let steps = (length / (SPACING * 0.5)).ceil().max(1.0) as usize;
                for step in 0..steps {
                    points.push(catmull_rom(window, step as f32 / steps as f32));
                }
            }
            points.push(last);
            // Push the curve away from obstacles, then smooth the kinks
            // that leaves.
            for _ in 0..6 {
                for point in &mut points {
                    *point = obstacles.collide(*point, CLEARANCE);
                }
                let copy = points.clone();
                for index in 1..points.len() - 1 {
                    points[index] = (copy[index - 1] + 2.0 * copy[index] + copy[index + 1]) / 4.0;
                }
            }
            for point in &points {
                assert!(
                    ground(*point) > WATER_LEVEL + 0.02,
                    "the route enters the lake at ({:.1}, {:.1}); move the script's points",
                    point.x,
                    point.y,
                );
            }
            let mut along = vec![0.0];
            for pair in points.windows(2) {
                along.push(along.last().unwrap() + pair[0].distance(pair[1]));
            }
            before = points[points.len().saturating_sub(4)];
            previous = last;
            Some(Route { points, along })
        })
        .collect()
}

fn catmull_rom(p: &[Vec2], t: f32) -> Vec2 {
    let knot = |a: Vec2, b: Vec2| a.distance(b).max(1e-3).sqrt();
    let t0 = 0.0;
    let t1 = t0 + knot(p[0], p[1]);
    let t2 = t1 + knot(p[1], p[2]);
    let t3 = t2 + knot(p[2], p[3]);
    let t = t1 + (t2 - t1) * t;
    let a1 = p[0] * ((t1 - t) / (t1 - t0)) + p[1] * ((t - t0) / (t1 - t0));
    let a2 = p[1] * ((t2 - t) / (t2 - t1)) + p[2] * ((t - t1) / (t2 - t1));
    let a3 = p[2] * ((t3 - t) / (t3 - t2)) + p[3] * ((t - t2) / (t3 - t2));
    let b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0));
    let b2 = a2 * ((t3 - t) / (t3 - t1)) + a3 * ((t - t1) / (t3 - t1));
    b1 * ((t2 - t) / (t2 - t1)) + b2 * ((t - t1) / (t2 - t1))
}

fn start_point() -> Vec2 {
    let z = 8.5;
    Vec2::new(trail_x(z), z)
}

/// A critically damped spring, for camera moves that ease in and out.
#[derive(Clone, Copy)]
struct Spring {
    value: f32,
    speed: f32,
}

impl Spring {
    fn new(value: f32) -> Self {
        Self { value, speed: 0.0 }
    }

    /// Moves toward `target` with a response rate `omega` in radians per second.
    fn step(&mut self, error: f32, omega: f32, dt: f32) {
        let acceleration = omega * omega * error - 2.0 * omega * self.speed;
        self.speed += acceleration * dt;
        self.value += self.speed * dt;
    }

    fn toward(&mut self, target: f32, omega: f32, dt: f32) {
        self.step(target - self.value, omega, dt);
    }

    fn toward_angle(&mut self, target: f32, omega: f32, dt: f32) {
        self.step(wrap(target - self.value), omega, dt);
    }
}

fn wrap(angle: f32) -> f32 {
    (angle + PI).rem_euclid(TAU) - PI
}

/// Drives the hiker's controls and the camera rig through the script.
#[derive(Resource)]
struct Director {
    routes: Vec<Option<Route>>,
    beat: usize,
    /// Seconds into the current beat.
    elapsed: f32,
    /// Seconds into the clip.
    clock: f32,
    /// Index of the route point nearest the hiker.
    progress: usize,
    jumps_done: usize,
    heading: Spring,
    orbit: Spring,
    pitch: Spring,
    fov: Spring,
    /// Where the hiker was a moment ago, to notice when it gets stuck.
    checkpoint: (f32, Vec2),
    finished: bool,
}

impl Director {
    fn new(routes: Vec<Option<Route>>, world: &mut World) -> Self {
        let rig = world.query::<&HikerCamera>().single(world).unwrap();
        let (yaw, pitch) = (rig.yaw, rig.pitch);
        Self {
            routes,
            beat: 0,
            elapsed: 0.0,
            clock: 0.0,
            progress: 0,
            jumps_done: 0,
            heading: Spring::new(yaw),
            orbit: Spring::new(0.0),
            pitch: Spring::new(pitch),
            fov: Spring::new(54.0),
            checkpoint: (0.0, start_point()),
            finished: false,
        }
    }

    fn next_beat(&mut self) {
        self.beat += 1;
        self.elapsed = 0.0;
        self.progress = 0;
        self.jumps_done = 0;
        self.finished = self.beat >= SCRIPT.len();
    }
}

/// Yaw that points the camera along a ground direction.
fn yaw_toward(direction: Vec2) -> f32 {
    (-direction.x).atan2(-direction.y)
}

fn direct(
    time: Res<Time>,
    director: Option<ResMut<Director>>,
    mut controls: ResMut<Controls>,
    hiker: Single<(&Transform, &Hiker)>,
    mut rig: Single<&mut HikerCamera>,
    mut lens: Single<&mut Projection, With<HikerCamera>>,
) {
    let Some(mut director) = director else {
        return;
    };
    let director = &mut *director;
    let dt = time.delta_secs();
    director.clock += dt;
    director.elapsed += dt;
    let (transform, body) = *hiker;
    let at = transform.translation.xz();
    controls.movement = Vec2::ZERO;
    controls.run = false;

    // Slow, uneven drift, like a hand on the mouse.
    let t = director.clock;
    let drift = 0.16 * (0.29 * t + 0.4).sin() + 0.07 * (0.71 * t + 2.1).sin();
    let bob = 0.035 * (0.37 * t + 1.3).sin();

    if let Some(beat) = SCRIPT.get(director.beat) {
        match (&beat.action, &director.routes[director.beat]) {
            (Action::Go { run, jumps, .. }, Some(route)) => {
                let lookahead = if *run { 2.4 } else { 1.4 };
                // Track the nearest point a short way ahead of the last one.
                let window = director.progress..(director.progress + 40).min(route.points.len());
                director.progress = window
                    .min_by(|&a, &b| {
                        route.points[a]
                            .distance_squared(at)
                            .total_cmp(&route.points[b].distance_squared(at))
                    })
                    .unwrap();
                let covered = route.along[director.progress];
                let target = route.at(covered + lookahead);
                let direction = (target - at).normalize_or_zero();
                director
                    .heading
                    .toward_angle(yaw_toward(direction), 2.6, dt);
                // Set off only once the camera is behind the hiker, so the
                // hiker never walks backward toward it.
                if director.orbit.value.abs() < 1.0 && direction != Vec2::ZERO {
                    let yaw = director.heading.value + director.orbit.value;
                    let local =
                        Quat::from_rotation_y(-yaw) * Vec3::new(direction.x, 0.0, direction.y);
                    controls.movement = Vec2::new(local.x, -local.z);
                    controls.run = *run;
                }
                if let Some(&jump) = jumps.get(director.jumps_done)
                    && covered >= jump
                {
                    controls.jump = true;
                    director.jumps_done += 1;
                    println!("Jump at {:.1} s", director.clock);
                }
                let remaining = route.length() - covered;
                if remaining < 0.3 || (remaining < lookahead && at.distance(target) < 0.5) {
                    director.next_beat();
                }
                // Fail loudly instead of recording a hiker stuck on a trunk.
                if director.clock - director.checkpoint.0 > 2.5 {
                    if at.distance(director.checkpoint.1) < 0.4 && director.orbit.value.abs() < 1.0
                    {
                        panic!(
                            "the hiker is stuck at ({:.1}, {:.1}) in beat {}",
                            at.x, at.y, director.beat
                        );
                    }
                    director.checkpoint = (director.clock, at);
                }
            }
            (Action::Hold { seconds }, _) => {
                let heading = director.heading.value;
                director.heading.toward(heading, 2.6, dt);
                director.checkpoint = (director.clock, at);
                if director.elapsed >= *seconds {
                    director.next_beat();
                }
            }
            (Action::Go { .. }, None) => unreachable!("every walk beat has a route"),
        }
        director.orbit.toward(beat.orbit + drift, 1.5, dt);
        director.pitch.toward(beat.pitch + bob, 1.4, dt);
    } else {
        director.orbit.toward(director.orbit.value, 1.0, dt);
        director.pitch.toward(director.pitch.value, 1.0, dt);
        director.checkpoint = (director.clock, at);
    }

    // Widen the lens a little at a run.
    let speed = body.velocity.length();
    let fov = 54.0 + 6.0 * ((speed - 2.0) / 2.6).clamp(0.0, 1.0);
    director.fov.toward(fov, 2.0, dt);
    if let Projection::Perspective(lens) = &mut **lens {
        lens.fov = director.fov.value.to_radians();
    }
    rig.yaw = wrap(director.heading.value + director.orbit.value);
    rig.pitch = director.pitch.value.clamp(-0.36, 0.77);
}

/// Draws the terrain, obstacles, and planned route seen from above.
fn draw_map(obstacles: &Obstacles, routes: &[Option<Route>], path: &std::path::Path) {
    const SCALE: f32 = 6.0;
    let low = Vec2::new(-40.0, -66.0);
    let high = Vec2::new(42.0, 28.0);
    let width = ((high.x - low.x) * SCALE) as u32;
    let height = ((high.y - low.y) * SCALE) as u32;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let to_world = |x: u32, y: u32| low + Vec2::new(x as f32 + 0.5, y as f32 + 0.5) / SCALE;
    for y in 0..height {
        for x in 0..width {
            let at = to_world(x, y);
            let floor = ground(at);
            let color = if floor < WATER_LEVEL {
                [40, 70, 130]
            } else {
                let shade = (90.0 + floor * 22.0).clamp(30.0, 230.0) as u8;
                let trail = ((at.x - trail_x(at.y)).abs() < 0.8) as u8 * 40;
                [shade / 2 + trail, shade, shade / 2]
            };
            let index = ((y * width + x) * 4) as usize;
            pixels[index..index + 4].copy_from_slice(&[color[0], color[1], color[2], 255]);
        }
    }
    let mut plot = |at: Vec2, color: [u8; 3]| {
        let pixel = ((at - low) * SCALE).floor();
        if pixel.x >= 0.0 && pixel.y >= 0.0 && (pixel.x as u32) < width && (pixel.y as u32) < height
        {
            let index = ((pixel.y as u32 * width + pixel.x as u32) * 4) as usize;
            pixels[index..index + 3].copy_from_slice(&color);
        }
    };
    // Trunks in red, and canopies that push the camera in dark red.
    for y in 0..height {
        for x in 0..width {
            let at = to_world(x, y);
            if obstacles.collide(at, 0.0) != at {
                plot(at, [200, 40, 30]);
            } else if obstacles.obscures(Vec3::new(at.x, ground(at) + 2.6, at.y)) {
                plot(at, [110, 60, 40]);
            }
        }
    }
    for (index, route) in routes.iter().enumerate() {
        let Some(route) = route else { continue };
        let color = match SCRIPT[index].action {
            Action::Go { run: true, .. } => [255, 150, 20],
            _ => [255, 240, 60],
        };
        for point in &route.points {
            plot(*point, color);
        }
        if let Action::Go { jumps, .. } = SCRIPT[index].action {
            for &jump in jumps {
                let center = route.at(jump);
                for step in 0..24 {
                    plot(
                        center + 0.6 * Vec2::from_angle(step as f32 * TAU / 24.0),
                        [255, 255, 255],
                    );
                }
            }
        }
    }
    let image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD,
    );
    image
        .try_into_dynamic()
        .unwrap()
        .into_rgb8()
        .save(path)
        .unwrap();
    println!("Route map: {} (-z is up, +x is right)", path.display());
}

pub(crate) fn headless_app() -> App {
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

pub(crate) fn warm_up(app: &mut App, ready: impl Fn(&World) -> bool) {
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
