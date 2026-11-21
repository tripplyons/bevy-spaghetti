//! Procedural standing deadwood (snags) built from Gaussian segments.
use std::f32::consts::TAU;

use bevy::prelude::*;
use bevy_spaghetti::{DetailTransition, GaussianSegment, SpaghettiAsset};

use crate::landscape::{SUN, hash01};

/// Minimum on-screen snag height in pixels for the fine and middle levels.
const LOD_PIXELS: [f32; 2] = [220.0, 65.0];

struct SnagLevel {
    strips: u32,
    spacing: f32,
    limb_steps: usize,
    shoots: bool,
}

const LEVELS: [SnagLevel; 3] = [
    SnagLevel {
        strips: 16,
        spacing: 0.42,
        limb_steps: 4,
        shoots: true,
    },
    SnagLevel {
        strips: 8,
        spacing: 0.72,
        limb_steps: 2,
        shoots: false,
    },
    SnagLevel {
        strips: 4,
        spacing: 1.25,
        limb_steps: 1,
        shoots: false,
    },
];

/// Builds a multi-LOD snag cloud and a coarse reflection cloud.
pub fn snag_clouds(
    clouds: &mut Assets<SpaghettiAsset>,
    seed: u32,
    height: f32,
    top: f32,
    crown: f32,
    girth: f32,
) -> (Handle<SpaghettiAsset>, Handle<SpaghettiAsset>) {
    let levels = LEVELS.each_ref().map(|level| {
        snag_segments(seed, height, top, crown, girth, level)
    });
    let reflection = clouds.add(
        SpaghettiAsset::new(levels[levels.len() - 1].clone())
            .and_then(|asset| asset.bake_occlusion_with_density(SUN, 0.22))
            .expect("procedural snags are valid"),
    );
    let segments: Vec<GaussianSegment> = levels.iter().flatten().copied().collect();
    let mut finer_end = 0;
    let transitions: [_; LOD_PIXELS.len()] = std::array::from_fn(|level| {
        finer_end += levels[level].len() as u32;
        let pixels = LOD_PIXELS[level] / top;
        DetailTransition {
            finer_end,
            pixels_per_unit: [0.8 * pixels, 1.2 * pixels],
        }
    });
    let adaptive = clouds.add(
        SpaghettiAsset::with_detail(segments, &transitions)
            .and_then(|asset| asset.bake_occlusion_with_density(SUN, 0.22))
            .expect("procedural snags have valid detail levels"),
    );
    (adaptive, reflection)
}

fn snag_segments(
    seed: u32,
    height: f32,
    top: f32,
    crown: f32,
    girth: f32,
    level: &SnagLevel,
) -> Vec<GaussianSegment> {
    let mut out = Vec::new();
    let phase = TAU * hash01(seed);
    let base_radius = (0.019 * height + 0.032) * girth;
    let lean = Vec2::from_angle(phase) * (0.08 + 0.24 * hash01(seed + 1));
    let center = |y: f32| {
        let t = (y / top).clamp(0.0, 1.0);
        let bend = Vec2::from_angle(phase + 2.0) * (0.075 * (t * 5.4).sin() * t);
        (lean * t * t + bend).extend(y).xzy()
    };
    let radius = |y: f32| {
        let y = y.max(0.0);
        let flare = 1.0 + 0.75 * (-y / 0.44).exp();
        base_radius * (1.0 - 0.97 * y / height).max(0.055) * flare
    };

    // Weathered silver-grey bark plates staggered along the leaning trunk.
    for strip in 0..level.strips {
        let base_angle = strip as f32 * TAU / level.strips as f32;
        let mut y = -0.18 - hash01(seed.wrapping_add(strip * 53 + 7)) * level.spacing * 0.7;
        let mut row = 0u32;
        while y < top {
            let id = seed.wrapping_add(row * 37 + strip * 101);
            let step = (level.spacing * (0.72 + 0.62 * hash01(id + 3))).min(top - y + 0.08);
            let y0 = y.max(-0.18);
            let y1 = (y + step).min(top);
            if y1 <= y0 + 0.01 {
                y += step;
                row += 1;
                continue;
            }
            let c0 = center(y0);
            let c1 = center(y1);
            let tangent = (c1 - c0).normalize_or(Vec3::Y);
            let r0 = radius(y0);
            let r1 = radius(y1);
            let mid_y = 0.5 * (y0 + y1);
            let wet = (-mid_y.max(0.0) / 0.68).exp();
            let base_tint = Vec3::ONE.lerp(Vec3::new(0.44, 0.53, 0.39), wet);
            let angle = base_angle + (hash01(id) - 0.5) * 0.20;
            let outward = Vec3::new(angle.cos(), 0.0, angle.sin());
            let relief = 0.60 + 0.16 * hash01(id + 4);
            let start = c0 + outward * r0 * relief;
            let end = c1 + outward * r1 * relief;
            let width = r0 * std::f32::consts::PI / level.strips as f32;
            let fissure = hash01(id + 1);
            let tone = 0.74 + 0.44 * hash01(id + 2);
            let silver = Vec3::new(0.185, 0.172, 0.158);
            let groove = Vec3::new(0.068, 0.060, 0.054);
            let color = silver.lerp(groove, fissure) * tone * base_tint;
            let side = tangent.cross(outward).normalize_or(Vec3::X);
            let normal = side.cross(tangent).normalize_or(outward);
            let rotation = Quat::from_mat3(&Mat3::from_cols(side, tangent, normal));
            out.push(GaussianSegment::new(
                start,
                end,
                Vec3::new(width * 0.75, width * 0.48, r0 * 0.30),
                rotation,
                6.5 * (y1 - y0) * width,
                color,
            ));
            y += step * 0.90;
            row += 1;
        }
    }

    // Splintered pale wood at the snapped top.
    let split = if top < height { 0.49 } else { 0.13 };
    let top_center = center(top);
    let top_r = radius(top);
    let splinters = if level.strips >= 5 { 4 } else { 2 };
    for index in 0..splinters {
        let id = seed.wrapping_add(500 + index * 19);
        let angle = index as f32 * TAU / splinters as f32 + hash01(id) * 0.7;
        let outward = Vec3::new(angle.cos(), 0.0, angle.sin());
        let start = top_center - Vec3::Y * 0.08 + outward * top_r * 0.38;
        let spike = split * (0.45 + 0.85 * hash01(id + 1));
        let end = start + (Vec3::Y + outward * (0.08 + 0.18 * hash01(id + 2))).normalize() * spike;
        let width = (top_r * (0.34 + 0.22 * hash01(id + 3))).max(0.012);
        let color = Vec3::new(0.34, 0.28, 0.20) * (0.85 + 0.3 * hash01(id + 4));
        out.push(
            GaussianSegment::new(
                start,
                end,
                Vec3::new(width, width, width * 0.75),
                Quat::from_rotation_arc(Vec3::Z, outward),
                5.5 * width * start.distance(end),
                color,
            )
            .with_cutout_taper(0.18),
        );
    }

    // Bare dead limbs and surviving side shoots.
    let mut y = 0.6 + 0.44 * hash01(seed + 2);
    let mut branch = seed + 29;
    while y < top - 0.22 {
        let azimuth = phase + branch as f32 * 2.39996 + 1.9 * hash01(branch + 1);
        let direction = Vec2::from_angle(azimuth).extend(0.0).xzy();
        let taper = (1.0 - y / height).max(0.12);
        let remaining = hash01(branch + 2);
        let reach = (0.16 + crown * taper * (0.18 + 0.97 * remaining.powi(2))).min(1.57);
        let thickness = radius(y) * (0.16 + 0.21 * hash01(branch + 3));
        let start = center(y) + direction * radius(y) * 0.68;
        let sag = -0.43 + 0.65 * hash01(branch + 4);
        let sweep = direction.cross(Vec3::Y) * (0.32 * (hash01(branch + 5) - 0.5));
        let point =
            |t: f32| start + reach * t * (direction + sweep * t + Vec3::Y * (0.22 + sag * t));
        let limb_radius = |t: f32| thickness * (1.0 - 0.82 * t).powi(2) + 0.006;

        for step in 0..level.limb_steps {
            let t0 = step as f32 / level.limb_steps as f32;
            let t1 = (step + 1) as f32 / level.limb_steps as f32;
            let a = point(t0);
            let b = point(t1);
            let r0 = limb_radius(t0);
            let r1 = limb_radius(t1);
            let along = (b - a).normalize_or(direction);
            let normal = (Vec3::Y + along.cross(Vec3::Y).cross(along) * 0.3).normalize_or(Vec3::Y);
            let tone = 0.86 + 0.14 * (0.5 * (t0 + t1));
            out.push(
                GaussianSegment::new(
                    a,
                    b,
                    Vec3::splat(r0.max(0.008)),
                    Quat::from_rotation_arc(Vec3::Z, normal),
                    5.5 * a.distance(b) * r0.max(0.008),
                    Vec3::new(0.21, 0.195, 0.175) * tone,
                )
                .with_cutout_taper((r1 / r0.max(1e-4)).clamp(0.15, 1.0)),
            );
        }

        if level.shoots && reach > 0.65 && hash01(branch + 6) < 0.55 {
            let fork = if hash01(branch + 7) < 0.5 { 1.0 } else { -1.0 };
            let root = point(0.46);
            let side = direction * 0.65 + direction.cross(Vec3::Y) * (0.52 * fork);
            let tip = root + reach * (side * 0.52 - Vec3::Y * 0.06);
            let r = (thickness * 0.36 + 0.004).max(0.006);
            out.push(
                GaussianSegment::new(
                    root,
                    tip,
                    Vec3::splat(r),
                    Quat::from_rotation_arc(Vec3::Z, Vec3::Y),
                    5.0 * root.distance(tip) * r,
                    Vec3::new(0.18, 0.165, 0.15),
                )
                .with_cutout_taper(0.22),
            );
        }
        y += 0.19 + 0.50 * hash01(branch + 8).powf(1.3);
        branch += 17;
    }

    out
}
