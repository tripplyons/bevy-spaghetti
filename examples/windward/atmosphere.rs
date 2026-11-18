use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    light::{FogVolume, VolumetricFog},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use crate::landscape::{WATER_LEVEL, ground, noise, shore_distance, smoothstep};

const CENTER: Vec3 = Vec3::new(1.0, 3.5, -25.0);
const SIZE: Vec3 = Vec3::new(124.0, 12.0, 116.0);

pub fn camera_fog() -> VolumetricFog {
    VolumetricFog {
        ambient_color: Color::linear_rgb(0.72, 0.82, 0.96),
        ambient_intensity: 0.0,
        step_count: 20,
        jitter: 0.035,
    }
}

pub fn spawn_atmosphere(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        Name::new("Low valley atmosphere"),
        FogVolume {
            density_texture: Some(images.add(density_map())),
            density_factor: 0.11,
            absorption: 0.11,
            scattering: 0.31,
            scattering_asymmetry: 0.59,
            fog_color: Color::linear_rgb(0.75, 0.84, 0.95),
            light_intensity: 0.66,
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
