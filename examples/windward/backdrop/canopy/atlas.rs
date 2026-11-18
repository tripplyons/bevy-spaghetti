//! Orthographic crown cutouts baked from the same splat trees as the trail.
use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use crate::{
    conifer::{self, FIR, Form, HEALTHY, HEMLOCK, SUBALPINE},
    landscape::smoothstep,
};
use bevy_spaghetti::GaussianSegment;

use super::super::detail_texture::generate_mips;

const CELL: usize = 128;
const WIDTH: usize = VARIANTS as usize * CELL;
const HEIGHT: usize = 2 * CELL;
const PAD: f32 = 8.0;
const TREE_HEIGHT: f32 = 8.0;
const HALF_WIDTH: f32 = 2.85;

pub const VARIANTS: u32 = 4;

pub fn uv(variant: u32, view: usize, corner: Vec2) -> [f32; 2] {
    let origin = Vec2::new(variant as f32, view as f32) * CELL as f32;
    let pixel = origin + PAD + corner * (CELL as f32 - 2.0 * PAD);
    (pixel / Vec2::new(WIDTH as f32, HEIGHT as f32)).to_array()
}

pub fn image() -> Image {
    let mut data = vec![0; WIDTH * HEIGHT * 4];
    let varieties = [
        (71, &FIR, 2.72, 0.11, HEALTHY),
        (
            175,
            &HEMLOCK,
            2.46,
            0.8,
            Form {
                thin: 0.14,
                ..HEALTHY
            },
        ),
        (
            321,
            &SUBALPINE,
            1.95,
            1.1,
            Form {
                thin: 0.09,
                flag: 0.24,
                ..HEALTHY
            },
        ),
        (
            609,
            &FIR,
            2.65,
            1.8,
            Form {
                bole: 0.23,
                thin: 0.18,
                ..HEALTHY
            },
        ),
    ];
    for (variant, (seed, habit, width, angle, form)) in varieties.into_iter().enumerate() {
        let cloud = conifer::fir(seed, habit, &form, TREE_HEIGHT, width, 0).splats;
        bake_view(&mut data, variant, 0, angle, &cloud);
        bake_view(
            &mut data,
            variant,
            1,
            angle + std::f32::consts::FRAC_PI_2,
            &cloud,
        );
    }
    let mut image = Image::new(
        Extent3d {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::linear();
    // Stop before a mip's filter footprint reaches the neighboring crown.
    generate_mips(image, 32)
}

fn bake_view(data: &mut [u8], col: usize, row: usize, angle: f32, segments: &[GaussianSegment]) {
    let view = Quat::from_rotation_y(angle);
    let mut depth = vec![f32::NEG_INFINITY; CELL * CELL];
    let mut shade = vec![118; CELL * CELL];
    let mut coverage = vec![0; CELL * CELL];
    let scale = Vec2::new(0.5 / HALF_WIDTH, 1.0 / (TREE_HEIGHT + 0.25)) * (CELL as f32 - 2.0 * PAD);
    // Keep an open, tapered leader in the small silhouette. Several whorls
    // otherwise merge into one broad tip once filtered to a few pixels.
    let crown_width = |y: f32| 1.0 - 0.57 * smoothstep(0.46, 0.96, y / TREE_HEIGHT);
    let project = |point: Vec3| {
        let at = Vec2::new(point.x * crown_width(point.y), -point.y);
        Vec2::new(CELL as f32 * 0.5, CELL as f32 - PAD) + at * scale
    };
    for segment in segments {
        let a = view * segment.start();
        let b = view * segment.end();
        let start = project(a);
        let end = project(b);
        let line = end - start;
        let across = line.perp().normalize_or(Vec2::X);
        // Project the kernel using the width at the segment midpoint.
        // The atlas flips screen Y.
        let height = (a.y + b.y) * 0.5;
        let axis = Vec3::new(
            across.x * scale.x * crown_width(height),
            -across.y * scale.y,
            0.0,
        );
        let local = (view * segment.rotation()).inverse() * axis;
        let sigma = segment.sigma();
        // Use the runtime cutout's unit optical depth contour.
        let area = 2.506628 * sigma.x * a.distance(b).max(2.506628 * sigma.y);
        let peak = segment.mass() / area;
        if peak <= 1.0 {
            continue;
        }
        let radius = (local * sigma).length() * (2.0 * peak.ln()).sqrt();
        let taper = 1.0 - segment.cutout_end_width();
        let offset = Vec2::splat(radius.max(0.36) + 0.6);
        let min = (start.min(end) - offset)
            .floor()
            .max(Vec2::splat(PAD))
            .as_uvec2();
        let max = (start.max(end) + offset)
            .ceil()
            .min(Vec2::splat(CELL as f32 - PAD))
            .as_uvec2();
        let needle = segment.color().dot(Vec3::new(0.24, 0.65, 0.11));
        let elevation = (height / TREE_HEIGHT).clamp(0.0, 1.0);
        let light = 0.22 + 0.13 * elevation + 3.5 * needle;
        let reach = HALF_WIDTH * (1.0 - elevation).max(0.2);
        for y in min.y..max.y {
            for x in min.x..max.x {
                let at = UVec2::new(x, y).as_vec2() + 0.5;
                let t = ((at - start).dot(line) / line.length_squared().max(0.001)).clamp(0.0, 1.0);
                let d = at.distance(start + line * t);
                let width = (radius * (1.0 - taper * t)).max(0.36);
                let alpha = ((width + 0.54 - d) * (255.0 / 1.08)).clamp(0.0, 255.0) as u8;
                let index = y as usize * CELL + x as usize;
                let clear = (255 - u16::from(coverage[index])) * (255 - u16::from(alpha));
                coverage[index] = 255 - ((clear + 127) / 255) as u8;
                let z = a.z.lerp(b.z, t);
                if alpha >= 72 && z > depth[index] {
                    depth[index] = z;
                    // Branch tips on the outside catch more sky than the
                    // interior. Vary occlusion with the local crown width.
                    let outside = smoothstep(-0.08, 0.7, z / reach);
                    let light = (light + 0.46 * outside).clamp(0.26, 1.0);
                    shade[index] = (light * 255.0).round() as u8;
                }
            }
        }
    }
    for y in 0..CELL {
        for x in 0..CELL {
            let source = y * CELL + x;
            let target = ((row * CELL + y) * WIDTH + col * CELL + x) * 4;
            data[target..target + 4].copy_from_slice(&[
                shade[source],
                shade[source],
                shade[source],
                coverage[source],
            ]);
        }
    }
}
