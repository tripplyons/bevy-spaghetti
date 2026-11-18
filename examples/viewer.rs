use bevy::{
    asset::LoadState,
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollPixelsPerLine},
    prelude::*,
    window::{PresentMode, WindowResolution},
};
use bevy_spaghetti::{Spaghetti3d, SpaghettiAsset, SpaghettiPlugin};

const DEMO: &str = "marble_bust_01.spag";
const FOV: f32 = 0.68;
const INITIAL_YAW: f32 = 0.28;
const INITIAL_PITCH: f32 = 0.08;

fn main() {
    let asset = std::env::args().nth(1).unwrap_or_else(|| DEMO.to_owned());
    App::new()
        .insert_resource(ViewerAsset(asset))
        .insert_resource(ClearColor(Color::srgb_u8(23, 27, 34)))
        .add_plugins((
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets").to_owned(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Spaghetti Splat Viewer".into(),
                        resolution: WindowResolution::new(1100, 760),
                        present_mode: PresentMode::AutoVsync,
                        ..default()
                    }),
                    ..default()
                }),
            SpaghettiPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (frame_asset, orbit_camera).chain())
        .run();
}

#[derive(Resource)]
struct ViewerAsset(String);

#[derive(Component)]
struct Status;

#[derive(Component)]
struct Orbit {
    center: Vec3,
    radius: f32,
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
    turntable: bool,
    ready: bool,
}

impl Orbit {
    fn reset(&mut self) {
        self.target = self.center;
        self.distance = self.radius / (FOV * 0.5).sin() * 1.14;
        self.yaw = INITIAL_YAW;
        self.pitch = INITIAL_PITCH;
        self.turntable = false;
    }
}

fn setup(mut commands: Commands, server: Res<AssetServer>, asset: Res<ViewerAsset>) {
    commands.spawn(Spaghetti3d(server.load(asset.0.clone())));
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: FOV,
            ..default()
        }),
        Msaa::Off,
        Tonemapping::Linear,
        Orbit {
            center: Vec3::ZERO,
            radius: 1.0,
            target: Vec3::ZERO,
            distance: 3.0,
            yaw: INITIAL_YAW,
            pitch: INITIAL_PITCH,
            turntable: false,
            ready: false,
        },
    ));

    let demo = asset.0 == DEMO;
    let title = if demo {
        "Marble Bust 01".to_string()
    } else {
        std::path::Path::new(&asset.0)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    };
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            top: px(30),
            left: px(32),
            right: px(32),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            ..default()
        })
        .with_children(|header| {
            header.spawn((
                Text::new(title),
                TextFont::from_font_size(30.0),
                TextColor(Color::srgb_u8(244, 239, 226)),
            ));
            if demo {
                header.spawn((
                    Text::new("Rico Cilliers / Poly Haven"),
                    TextFont::from_font_size(14.0),
                    TextColor(Color::srgb_u8(175, 182, 193)),
                ));
            }
            header.spawn((
                Status,
                Text::new("Loading splats…"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb_u8(175, 182, 193)),
            ));
        });

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: px(24),
                left: px(32),
                right: px(32),
                padding: UiRect::vertical(px(12)),
                border: UiRect::top(px(1)),
                column_gap: px(24),
                row_gap: px(8),
                flex_wrap: FlexWrap::Wrap,
                ..default()
            },
            BorderColor::all(Color::srgb_u8(64, 71, 83)),
        ))
        .with_children(|footer| {
            for control in [
                "Drag  Orbit",
                "Right drag  Pan",
                "Scroll  Zoom",
                "Space  Turntable",
                "R  Reset",
            ] {
                footer.spawn((
                    Text::new(control),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb_u8(198, 203, 211)),
                ));
            }
        });
}

fn frame_asset(
    cloud: Single<&Spaghetti3d>,
    mut camera: Single<(&mut Orbit, &mut Projection)>,
    mut status: Single<&mut Text, With<Status>>,
    assets: Res<Assets<SpaghettiAsset>>,
    server: Res<AssetServer>,
) {
    let (orbit, projection) = &mut *camera;
    if orbit.ready {
        return;
    }
    let Some(asset) = assets.get(&cloud.0) else {
        if matches!(
            server.get_load_state(cloud.0.id()),
            Some(LoadState::Failed(_))
        ) {
            status.0 = "Could not load the .spag asset. See the console for details.".into();
        }
        return;
    };

    let (low, high) = asset.bounds();
    orbit.center = (low + high) * 0.5;
    orbit.radius = ((high - low).length() * 0.5).max(0.0001);
    orbit.reset();
    orbit.ready = true;
    if let Projection::Perspective(perspective) = &mut **projection {
        perspective.near = orbit.radius * 0.01;
        perspective.far = orbit.radius * 100.0;
    }
    status.0 = format!("{} Gaussian segments", asset.segments().len());
}

fn orbit_camera(
    mut camera: Single<(&mut Transform, &mut Orbit)>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    scroll_ratio: Res<MouseScrollPixelsPerLine>,
    time: Res<Time>,
) {
    let (transform, orbit) = &mut *camera;
    if keys.just_pressed(KeyCode::KeyR) {
        orbit.reset();
    }
    if keys.just_pressed(KeyCode::Space) {
        orbit.turntable = !orbit.turntable;
    }
    if mouse.pressed(MouseButton::Left) {
        orbit.turntable = false;
        orbit.yaw -= motion.delta.x * 0.007;
        orbit.pitch = (orbit.pitch + motion.delta.y * 0.007).clamp(-1.54, 1.54);
    } else if orbit.turntable {
        orbit.yaw += time.delta_secs() * 0.45;
    }

    let zoom = scroll.to_lines(&scroll_ratio).delta.y.clamp(-12.0, 12.0);
    orbit.distance =
        (orbit.distance * (-0.14 * zoom).exp()).clamp(orbit.radius * 0.08, orbit.radius * 16.0);
    let rotation = Quat::from_rotation_y(orbit.yaw) * Quat::from_rotation_x(-orbit.pitch);
    if mouse.pressed(MouseButton::Right) {
        let pan = Vec3::new(-motion.delta.x, motion.delta.y, 0.0);
        let scale = 0.0012 * orbit.distance;
        orbit.target += rotation * pan * scale;
    }
    transform.translation = orbit.target + rotation * Vec3::Z * orbit.distance;
    transform.rotation = rotation;
}
