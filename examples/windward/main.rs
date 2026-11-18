//! Explore a windswept wetland. Run with `cargo run --example windward`.
mod atmosphere;
mod backdrop;
mod conifer;
mod debris;
mod detail;
mod fern;
mod forest;
mod lake;
mod landscape;
mod meadow;
mod player;
mod props;
mod reeds;
mod understory;

#[cfg(test)]
mod tests;

use bevy::{
    anti_alias::smaa::{Smaa, SmaaPreset},
    camera::{Exposure, Hdr, visibility::RenderLayers},
    core_pipeline::tonemapping::Tonemapping,
    light::{GeneratedEnvironmentMapLight, Skybox},
    pbr::{DistanceFog, FogFalloff},
    post_process::bloom::Bloom,
    prelude::*,
    render::view::{ColorGrading, ColorGradingGlobal, ColorGradingSection},
    window::WindowResolution,
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
            meadow::spawn_meadow,
            understory::spawn_understory,
            props::spawn_props,
            spawn_hiker,
            spawn_camera,
            lake::spawn_lake,
        )
            .chain(),
    )
    .add_systems(PreUpdate, read_controls.after(bevy::input::InputSystems))
    .add_systems(FixedUpdate, (move_hiker, step_reeds).chain())
    .add_systems(
        Update,
        (
            animate_hiker,
            follow_hiker,
            landscape::select_rock_detail.after(follow_hiker),
            debris::select_log_detail.after(follow_hiker),
            forest::select_tree_detail.after(follow_hiker),
            lake::reflect_lake.after(follow_hiker),
            understory::select_understory_detail.after(follow_hiker),
            understory::bend_understory,
            props::select_prop_detail.after(follow_hiker),
            meadow::part_meadow,
        ),
    );
}

fn spawn_camera(
    mut commands: Commands,
    server: Res<AssetServer>,
    obstacles: Res<landscape::Obstacles>,
) {
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
            atmosphere::camera_fog(),
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
        Exposure { ev100: 11.5 },
        (
            Tonemapping::AgX,
            ColorGrading {
                global: ColorGradingGlobal {
                    temperature: 0.015,
                    post_saturation: 1.04,
                    ..default()
                },
                shadows: ColorGradingSection {
                    gamma: 0.96,
                    ..default()
                },
                midtones: ColorGradingSection {
                    contrast: 1.035,
                    ..default()
                },
                ..default()
            },
            Bloom {
                intensity: 0.045,
                ..Bloom::NATURAL
            },
        ),
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
        DistanceFog {
            color: Color::srgba(0.46, 0.53, 0.61, 0.59),
            directional_light_color: Color::srgba(0.98, 0.79, 0.63, 0.095),
            directional_light_exponent: 29.0,
            falloff: FogFalloff::ExponentialSquared { density: 0.0072 },
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
                    Text::new("WASD  Move    ·    Shift  Run    ·    Space  Jump    ·    Right drag  Look    ·    R  Reset"),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::srgb(0.94, 0.94, 0.93)),
                ));
        });
}
