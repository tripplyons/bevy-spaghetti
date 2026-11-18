//! The camera's look and the air it sees through: haze, sun shafts, and the
//! color grade.
use bevy::{
    asset::RenderAssetUsages,
    camera::{Exposure, visibility::RenderLayers},
    core_pipeline::tonemapping::Tonemapping,
    image::ImageSampler,
    light::{CascadeShadowConfigBuilder, FogVolume, VolumetricFog, VolumetricLight},
    pbr::{DistanceFog, FogFalloff},
    post_process::{bloom::Bloom, effect_stack::Vignette},
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        view::{ColorGrading, ColorGradingGlobal, ColorGradingSection},
    },
};

use crate::{
    landscape::{WATER_LEVEL, ground, noise, shore_distance, smoothstep},
    player::HikerCamera,
};

// The box reaches well above the hill crests, so its faces never cut the
// view as a hard line.
const CENTER: Vec3 = Vec3::new(1.0, 14.0, -25.0);
const SIZE: Vec3 = Vec3::new(124.0, 34.0, 116.0);

/// Volumetric fog for sun shafts, and distance fog for aerial perspective.
pub fn camera_haze() -> impl Bundle {
    (
        VolumetricFog {
            ambient_color: Color::linear_rgb(0.72, 0.82, 0.96),
            // Sky light keeps haze in terrain shadow blue-gray. Without it,
            // shaded fog only absorbs and the low-sun valley turns brown-black.
            ambient_intensity: 0.04,
            step_count: 14,
            jitter: 0.035,
        },
        DistanceFog {
            color: Color::srgba(0.46, 0.53, 0.61, 0.59),
            // A broad warm glow toward the low sun, as in backlit haze.
            directional_light_color: Color::srgba(0.98, 0.79, 0.63, 0.13),
            directional_light_exponent: 18.0,
            falloff: FogFalloff::ExponentialSquared { density: 0.0072 },
        },
    )
}

/// Exposure, tone mapping, and a restrained photographic grade.
pub fn camera_grade() -> impl Bundle {
    (
        Exposure { ev100: 11.5 },
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
                contrast: 1.07,
                ..default()
            },
            ..default()
        },
        Bloom {
            intensity: 0.045,
            ..Bloom::NATURAL
        },
        // A faint lens falloff toward the corners.
        Vignette {
            intensity: 0.2,
            radius: 0.9,
            smoothness: 2.5,
            ..default()
        },
    )
}

/// Lets the sun light the volumetric fog.
///
/// Bevy 0.20.1 sorts directional lights by `(volumetric, shadows, entity)` in
/// ascending order, so volumetric lights come last. The fog shader stops at
/// the first light that is not volumetric, so any plain directional light that
/// the camera sees hides the sun from the fog. This system marks the camera's
/// other directional lights as volumetric, with shadow maps on and a single
/// cascade 0.1 m long. All of them then sort with the sun, and the sun sorts
/// first because it was spawned first. Fog samples beyond a light's cascade
/// get no light from it, so these lights add no visible fog.
pub fn unblock_sun_shafts(
    mut commands: Commands,
    camera: Single<&RenderLayers, With<HikerCamera>>,
    mut lights: Query<
        (Entity, &mut DirectionalLight, Option<&RenderLayers>),
        Without<VolumetricLight>,
    >,
) {
    for (entity, mut light, layers) in &mut lights {
        if !layers.unwrap_or_default().intersects(&camera) {
            continue;
        }
        light.shadow_maps_enabled = true;
        commands.entity(entity).insert((
            VolumetricLight,
            CascadeShadowConfigBuilder {
                num_cascades: 1,
                minimum_distance: 0.05,
                first_cascade_far_bound: 0.1,
                maximum_distance: 0.1,
                ..default()
            }
            .build(),
        ));
    }
}

pub fn spawn_atmosphere(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        Name::new("Low valley atmosphere"),
        FogVolume {
            density_texture: Some(images.add(density_map())),
            density_factor: 0.095,
            absorption: 0.11,
            scattering: 0.31,
            scattering_asymmetry: 0.59,
            fog_color: Color::linear_rgb(0.75, 0.84, 0.95),
            // Tuned with the sun visible to the fog; see `unblock_sun_shafts`.
            light_intensity: 0.3,
            ..default()
        },
        Transform::from_translation(CENTER).with_scale(SIZE),
    ));
}

fn density_map() -> Image {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 28;
    const DEPTH: u32 = 64;
    let resolution = Vec3::new(WIDTH as f32, HEIGHT as f32, DEPTH as f32);
    let mut samples = Vec::with_capacity((WIDTH * HEIGHT * DEPTH) as usize);
    for z in 0..DEPTH {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let uvw = (Vec3::new(x as f32, y as f32, z as f32) + 0.5) / resolution;
                let offset = uvw * 2.0 - 1.0;
                let edge = 1.0 - smoothstep(0.64, 0.97, offset.abs().max_element());
                let p = CENTER + (uvw - 0.5) * SIZE;
                let at = p.xz();
                let height = (p.y - ground(at).max(WATER_LEVEL)).max(0.0);
                let low_air = (-0.26 * height).exp();
                let lake = 1.0 - smoothstep(0.72, 1.34, shore_distance(at));
                let wisp = noise(at * 0.087 + Vec2::new(p.y * 0.75, -p.y * 0.22));
                let bank = lake * (-1.42 * height).exp() * (0.28 + 0.70 * wisp);
                let density = edge * (0.14 * low_air + 0.61 * bank);
                samples.push((density * 255.0).round() as u8);
            }
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: DEPTH,
        },
        TextureDimension::D3,
        samples,
        TextureFormat::R8Unorm,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::linear();
    image
}
