//! Explore a windswept wetland. Run with `cargo run --example windward`.
mod atmosphere;
mod backdrop;
mod bark;
mod conifer;
mod debris;
mod detail;
mod fern;
mod forest;
mod hair;
mod lake;
mod landscape;
mod meadow;
mod player;
mod props;
mod reeds;
mod stone_grade;
mod understory;

#[cfg(test)]
mod tests;

use bevy::{
    anti_alias::smaa::{Smaa, SmaaPreset},
    camera::{Hdr, visibility::RenderLayers},
    light::{GeneratedEnvironmentMapLight, Skybox},
    prelude::*,
    window::{CursorGrabMode, CursorOptions, WindowResolution},
};
use bevy_spaghetti::SpaghettiPlugin;

use landscape::spawn_landscape;
use player::{HikerCamera, animate_hiker, follow_hiker, move_hiker, read_controls, spawn_hiker};
use reeds::{spawn_reeds, step_reeds};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(asset_plugin()).set(WindowPlugin {
        primary_window: Some(Window {
            title: "Windward | Wetland trail".into(),
            resolution: WindowResolution::new(1280, 800),
            ..default()
        }),
        // Start with the cursor captured for mouse look.
        primary_cursor_options: Some(CursorOptions {
            grab_mode: CursorGrabMode::Locked,
            visible: false,
            ..default()
        }),
        ..default()
    }));
    add_scene(&mut app);
    app.add_systems(Startup, spawn_hud).run();
}

fn asset_plugin() -> AssetPlugin {
    AssetPlugin {
        file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/assets").to_owned(),
        ..default()
    }
}

fn add_scene(app: &mut App) {
    app.add_plugins((
        SpaghettiPlugin,
        MaterialPlugin::<landscape::GroundMaterial>::default(),
        MaterialPlugin::<lake::LakeMaterial>::default(),
        MaterialPlugin::<backdrop::HighlandMaterial>::default(),
    ))
    .add_plugins(player::HikerPosePlugin)
    .init_resource::<player::Controls>()
    .insert_resource(ClearColor(Color::srgb(0.32, 0.38, 0.48)))
    .insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.69, 0.76, 0.94),
        brightness: 160.0,
        ..default()
    })
    .add_systems(
        Startup,
        (
            spawn_landscape,
            backdrop::spawn_backdrop,
            forest::spawn_trees,
            atmosphere::spawn_atmosphere,
            debris::spawn_debris,
            spawn_reeds,
            // Props first, so the meadow leaves their footprints bare.
            props::spawn_props,
            meadow::spawn_meadow,
            understory::spawn_understory,
            spawn_hiker,
            spawn_camera,
            atmosphere::unblock_sun_shafts,
            lake::spawn_lake,
        )
            .chain(),
    )
    .add_systems(
        PreUpdate,
        (player::capture_cursor, read_controls)
            .chain()
            .after(bevy::input::InputSystems),
    )
    .add_systems(
        FixedUpdate,
        ((move_hiker, step_reeds).chain(), forest::bend_crowns),
    )
    .add_systems(
        Update,
        (
            animate_hiker,
            follow_hiker,
            landscape::select_rock_detail.after(follow_hiker),
            debris::select_log_detail.after(follow_hiker),
            lake::reflect_lake.after(follow_hiker),
            understory::select_understory_detail.after(follow_hiker),
            understory::bend_understory,
            props::select_prop_detail.after(follow_hiker),
            bark::select_detail.after(follow_hiker),
            meadow::part_meadow,
            stone_grade::grade_stones,
        ),
    );
}

fn spawn_camera(
    mut commands: Commands,
    server: Res<AssetServer>,
    obstacles: Res<landscape::Obstacles>,
) {
    // The cubemap continues the sky haze below the photographed horizon,
    // where the scene's own terrain and treeline supply the silhouette.
    let sky: Handle<Image> = server
        .load_builder()
        .with_settings(|settings: &mut bevy::image::ImageLoaderSettings| settings.is_srgb = false)
        .load("windward/sky.dds");
    commands.spawn((
        Name::new("Follow camera"),
        HikerCamera::default(),
        (
            Camera3d::default(),
            IsDefaultUiCamera,
            atmosphere::camera_haze(),
        ),
        Projection::Perspective(PerspectiveProjection {
            fov: 54.0_f32.to_radians(),
            near: 0.08,
            far: 720.0,
            ..default()
        }),
        player::initial_view(&obstacles),
        RenderLayers::default().with(lake::LAKE_SURFACE_LAYER),
        Hdr,
        atmosphere::camera_grade(),
        Msaa::Off,
        Smaa {
            preset: SmaaPreset::Ultra,
        },
        Skybox {
            image: Some(sky.clone()),
            brightness: 3750.0,
            // The HDRI's sun sits toward +z; turn it to match `SUN`.
            rotation: SKY_ROTATION,
            ..default()
        },
        GeneratedEnvironmentMapLight {
            environment_map: sky,
            intensity: 1450.0,
            rotation: SKY_ROTATION,
            ..default()
        },
    ));
}

/// Turns the sky cubemap so its sun disc lines up with the directional light.
const SKY_ROTATION: Quat = Quat::from_xyzw(0.0, 1.0, 0.0, 0.0);

fn spawn_hud(mut commands: Commands) {
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: px(30),
            right: px(30),
            bottom: px(27),
            justify_content: JustifyContent::SpaceBetween,
            flex_wrap: FlexWrap::Wrap,
            row_gap: px(10),
            column_gap: px(24),
            ..default()
        })
        .with_children(|parent| {
            parent
                .spawn((
                    Node { padding: UiRect::axes(px(14), px(8)), ..default() },
                    BackgroundColor(Color::srgba(0.02, 0.025, 0.035, 0.68)),
                ))
                .with_child((
                    Text::new("Windward"),
                    TextFont::from_font_size(19.0),
                    TextColor(Color::srgb(1.0, 0.95, 0.83)),
                ));
            parent
                .spawn((
                    Node { padding: UiRect::axes(px(14), px(8)), ..default() },
                    BackgroundColor(Color::srgba(0.02, 0.025, 0.035, 0.68)),
                ))
                .with_child((
                    Text::new("WASD  Move    ·    Shift  Run    ·    Space  Jump    ·    Mouse  Look    ·    Esc  Free cursor    ·    R  Reset"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.94, 0.94, 0.93)),
                ));
        });
}
