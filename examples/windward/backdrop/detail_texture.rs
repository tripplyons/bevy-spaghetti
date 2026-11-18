//! Tiling detail maps for the backdrop slopes, built at startup.
use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use crate::landscape::hash01;

const SIZE: u32 = 256;

/// Value noise that repeats every `period` lattice cells across the unit tile.
fn tiled_noise(uv: Vec2, period: u32, seed: u32) -> f32 {
    let at = uv * period as f32;
    let cell = at.floor().as_ivec2();
    let fraction = at - at.floor();
    let w = fraction * fraction * (Vec2::splat(3.0) - 2.0 * fraction);
    let p = period as i32;
    let value = |x: i32, y: i32| {
        let x = x.rem_euclid(p) as u32;
        let y = y.rem_euclid(p) as u32;
        hash01(
            x.wrapping_mul(0x9E37_79B9)
                ^ y.wrapping_mul(0x85EB_CA6B)
                ^ seed.wrapping_mul(0xC2B2_AE35),
        )
    };
    value(cell.x, cell.y)
        .lerp(value(cell.x + 1, cell.y), w.x)
        .lerp(
            value(cell.x, cell.y + 1).lerp(value(cell.x + 1, cell.y + 1), w.x),
            w.y,
        )
}

/// Rounded bumps on a jittered lattice, like shrubs or crowns packed together.
fn tiled_bumps(uv: Vec2, period: u32, seed: u32) -> f32 {
    let at = uv * period as f32;
    let cell = at.floor().as_ivec2();
    let p = period as i32;
    let mut height: f32 = 0.0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let c = cell + IVec2::new(dx, dy);
            let wrapped = UVec2::new(c.x.rem_euclid(p) as u32, c.y.rem_euclid(p) as u32);
            let key = wrapped.x * 977 + wrapped.y * 131 + seed * 7919;
            let center = c.as_vec2() + Vec2::new(hash01(key), hash01(key + 1));
            let radius = 0.55 + 0.35 * hash01(key + 2);
            let d = (at - center).length() / radius;
            height = height.max((1.0 - d * d).max(0.0).sqrt() * (0.6 + 0.4 * hash01(key + 3)));
        }
    }
    height
}

fn image(pixels: Vec<u8>, format: TextureFormat) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 8,
        ..default()
    });
    image
}

fn uv(x: u32, y: u32) -> Vec2 {
    (Vec2::new(x as f32, y as f32) + 0.5) / SIZE as f32
}

/// Gray albedo detail from about 9 m down to 0.6 m on a 37 m tile, and its
/// mean, which the vertex colors divide out.
pub fn albedo_detail() -> (Image, f32) {
    let mut values = Vec::with_capacity((SIZE * SIZE) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let at = uv(x, y);
            let broad = tiled_noise(at, 4, 1);
            let patches = tiled_noise(at, 9, 2);
            let crowns = tiled_bumps(at, 22, 3);
            let grain = tiled_noise(at, 48, 4);
            let speckle = tiled_noise(at, 97, 5);
            let light =
                0.36 * broad + 0.26 * patches + 0.16 * crowns + 0.13 * grain + 0.09 * speckle;
            values.push((0.18 + 1.25 * light).clamp(0.08, 1.0));
        }
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let pixels = values
        .iter()
        .flat_map(|&v| {
            let byte = (v * 255.0).round() as u8;
            [byte, byte, byte, 255]
        })
        .collect();
    (
        generate_mips(image(pixels, TextureFormat::Rgba8Unorm)),
        mean,
    )
}

/// A tangent-space normal map of hummocks, shrubs, and soil roughness on a
/// 14 m tile.
pub fn normal_detail() -> Image {
    let height = |at: Vec2| {
        0.3 * tiled_bumps(at, 9, 11)
            + 0.25 * tiled_noise(at, 5, 14)
            + 0.3 * tiled_noise(at, 16, 12)
            + 0.15 * tiled_noise(at, 41, 13)
    };
    let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    let step = 1.0 / SIZE as f32;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let at = uv(x, y);
            let du = height(at + Vec2::X * step) - height(at - Vec2::X * step);
            let dv = height(at + Vec2::Y * step) - height(at - Vec2::Y * step);
            let normal = Vec3::new(-du * 7.0, -dv * 7.0, 1.0).normalize();
            let encoded = normal * 0.5 + 0.5;
            pixels.extend([
                (encoded.x * 255.0).round() as u8,
                (encoded.y * 255.0).round() as u8,
                (encoded.z * 255.0).round() as u8,
                255,
            ]);
        }
    }
    generate_mips(image(pixels, TextureFormat::Rgba8Unorm))
}

/// Box-filters a full mip chain so distant slopes average the detail instead
/// of shimmering.
fn generate_mips(mut image: Image) -> Image {
    let mut levels = vec![image.data.clone().unwrap()];
    let mut size = SIZE;
    while size > 1 {
        let previous = levels.last().unwrap();
        let half = size / 2;
        let mut next = Vec::with_capacity((half * half * 4) as usize);
        for y in 0..half {
            for x in 0..half {
                for channel in 0..4 {
                    let at = |dx: u32, dy: u32| {
                        u32::from(
                            previous[(((2 * y + dy) * size + 2 * x + dx) * 4 + channel) as usize],
                        )
                    };
                    next.push(((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8);
                }
            }
        }
        levels.push(next);
        size = half;
    }
    image.texture_descriptor.mip_level_count = levels.len() as u32;
    image.data = Some(levels.concat());
    image
}
