//! Chunked Gaussian grass blades with GPU wind and two detail levels.
use bevy::{camera::visibility::VisibilityRange, prelude::*};
use bevy_spaghetti::{
    GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiCutout, SpaghettiLighting,
    SpaghettiTransmission, SpaghettiWind,
};

use crate::{
    landscape::{ROCKS, SUN, WATER_LEVEL, ground, noise, smoothstep, trail_x},
    player::Hiker,
    props::PropFootprints,
};

const CHUNK: f32 = 8.0;
const NEAR_REGION: (Vec2, Vec2) = (Vec2::new(-44.0, -70.0), Vec2::new(46.0, 32.0));
const FAR_REGION: (Vec2, Vec2) = (Vec2::new(-64.0, -92.0), Vec2::new(64.0, 40.0));
const NEAR_END: std::ops::Range<f32> = 10.0..15.0;
const FAR_END: std::ops::Range<f32> = 74.0..92.0;
/// Clumps per square meter before the density mask removes some of them.
const CLUMPS: f32 = 9.0;
/// Every `FAR_STRIDE`-th blade survives in the far cloud, widened to keep coverage.
const FAR_STRIDE: usize = 6;

#[derive(Component)]
pub(super) struct Meadow;

#[derive(Clone, Copy)]
struct Blade {
    root: Vec3,
    height: f32,
    width: f32,
    lean: f32,
    lean_dir: Vec2,
    twist: f32,
    color: Vec3,
    tip: Vec3,
    /// Petal color for the few stems that carry a flower head.
    flower: Option<Vec3>,
}

impl Blade {
    fn append(self, segments: &mut Vec<GaussianSegment>, origin: Vec3, levels: &[(f32, f32)]) {
        let lean = Vec3::new(self.lean_dir.x, 0.0, self.lean_dir.y);
        let side =
            Quat::from_rotation_y(self.twist) * Vec3::new(-self.lean_dir.y, 0.0, self.lean_dir.x);
        let root = self.root - origin;
        let arc = |t: f32| {
            let angle = self.lean * t;
            let (sin, cos) = angle.sin_cos();
            root + (lean * (1.0 - cos) + Vec3::Y * sin) * (self.height / self.lean)
        };

        for pair in levels.windows(2) {
            let (from, width) = pair[0];
            let (to, end_width) = pair[1];
            let a = arc(from);
            let b = arc(to);
            let tangent = (b - a).normalize();
            let mut normal = side.cross(tangent).normalize();
            if normal.y < 0.0 {
                normal = -normal;
            }
            let cross = tangent.cross(normal);
            let rotation = Quat::from_mat3(&Mat3::from_cols(cross, tangent, normal));
            let t = (from + to) * 0.5;
            // Dense grass shades its own lower stems.
            let shade = 0.44 + 0.56 * t.sqrt();
            let color = self.color.lerp(self.tip, t * t * 0.65) * shade;
            let sigma = width * self.width * 0.72;
            let mass = 8.0 * sigma * a.distance(b);
            segments.push(
                GaussianSegment::new(
                    a,
                    b,
                    Vec3::new(sigma, sigma * 0.44, sigma * 0.26),
                    rotation,
                    mass,
                    color,
                )
                .with_blade_profile(self.height, [from, to], end_width / width),
            );
        }

        if let Some(petals) = self.flower {
            self.append_flower(segments, arc(1.0), petals);
        }
    }

    fn append_flower(self, segments: &mut Vec<GaussianSegment>, tip: Vec3, petals: Vec3) {
        let radius = 0.007 + 0.006 * (self.twist.abs() * 3.0).fract();
        let rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        for petal in 0..4 {
            let angle = self.twist * 4.0 + petal as f32 * std::f32::consts::FRAC_PI_2;
            let out = Vec3::new(angle.cos(), 0.0, angle.sin());
            let center = tip + (out * 0.62 + Vec3::Y * 0.25) * radius;
            let sigma = radius * 0.33;
            segments.push(
                GaussianSegment::new(
                    center,
                    center,
                    Vec3::new(sigma, sigma, sigma * 0.55),
                    rotation,
                    sigma * sigma * 21.0,
                    petals,
                )
                .with_blade_profile(self.height, [1.0, 1.0], 1.0),
            );
        }
    }
}

const NEAR_LEVELS: [(f32, f32); 4] = [(0.0, 0.5), (0.42, 0.42), (0.76, 0.26), (1.0, 0.0)];
const FAR_LEVELS: [(f32, f32); 3] = [(0.0, 0.5), (0.55, 0.32), (1.0, 0.0)];

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        // xorshift32 is plenty for placement jitter.
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / 16_777_216.0
    }
}

/// Where grass may root: zero on the trail, in the lake, under stones, and
/// in the needle litter under the conifer crowns.
pub(super) fn ground_mask(at: Vec2) -> f32 {
    let height = ground(at);
    let shore = smoothstep(WATER_LEVEL + 0.03, WATER_LEVEL + 0.22, height);
    let path = (at.x - trail_x(at.y)).abs() + (noise(at * 0.85) - 0.5) * 0.36;
    let worn = smoothstep(0.78, 1.75, path);
    let stones = ROCKS.iter().fold(1.0_f32, |keep, &(x, z, scale)| {
        keep.min(smoothstep(
            0.62 * scale,
            1.05 * scale,
            at.distance(Vec2::new(x, z)),
        ))
    });
    let shade = crate::forest::canopy(at);
    shore * worn * stones * (1.0 - shade).powi(2)
}

/// Lush patches grow tall tufts; elsewhere the turf stays short.
pub(super) fn lushness(at: Vec2) -> f32 {
    let patches = 0.62 * noise(at * 0.19 + Vec2::new(3.1, 7.7)) + 0.38 * noise(at * 0.71);
    smoothstep(0.22, 0.62, patches)
}

/// Grass cover used to shade the soil under the meadow.
pub fn cover(at: Vec2) -> f32 {
    ground_mask(at) * (0.45 + 0.55 * lushness(at))
}

fn blade_color(at: Vec2, rng: &mut Rng) -> (Vec3, Vec3) {
    let lush = Vec3::new(0.050, 0.092, 0.022);
    let olive = Vec3::new(0.105, 0.118, 0.036);
    let straw = Vec3::new(0.215, 0.188, 0.098);
    let dead = Vec3::new(0.120, 0.090, 0.055);
    let moisture = noise(at * 0.13 + Vec2::new(-4.0, 2.0));
    // Late-summer subalpine meadows stay mostly green, with cured patches
    // on the drier knolls.
    let dryness = smoothstep(
        0.40,
        0.90,
        noise(at * 0.37) * 0.55 + (1.0 - moisture) * 0.45,
    );
    let mut color = lush.lerp(olive, rng.next() * 0.8);
    color = color.lerp(straw, dryness * (0.45 + 0.55 * rng.next()));
    if rng.next() < 0.07 {
        color = dead;
    }
    let tip = color.lerp(straw * 1.15, 0.25 + 0.35 * dryness);
    (color * (0.82 + 0.36 * rng.next()), tip)
}

/// Meadow wildflowers: buttercup yellow, yarrow white and lupine purple,
/// with each color favoring its own drifts.
fn petal_color(at: Vec2, rng: &mut Rng) -> Vec3 {
    let pick = noise(at * 0.09 + Vec2::new(12.0, -5.0)) * 0.7 + rng.next() * 0.3;
    let color = if pick < 0.38 {
        Vec3::new(0.62, 0.48, 0.05)
    } else if pick < 0.62 {
        Vec3::new(0.62, 0.60, 0.52)
    } else {
        Vec3::new(0.20, 0.12, 0.42)
    };
    color * (0.8 + 0.3 * rng.next())
}

fn chunk_blades(min: Vec2, seed: u32, props: &PropFootprints) -> Vec<Blade> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9) | 1);
    let mut blades = Vec::new();
    let clumps = (CHUNK * CHUNK * CLUMPS) as usize;
    for _ in 0..clumps {
        let center = min + Vec2::new(rng.next(), rng.next()) * CHUNK;
        let keep = ground_mask(center) * props.keep(center);
        if rng.next() >= keep {
            continue;
        }
        let lush = lushness(center);
        let clump_height = (0.09 + 0.50 * lush * lush) * (0.6 + 0.4 * keep);
        let count = 6 + (rng.next() * (6.0 + 15.0 * lush)) as usize;
        let spread = 0.05 + 0.13 * rng.next();
        let clump_lean = Vec2::from_angle(rng.next() * std::f32::consts::TAU);
        for _ in 0..count {
            let offset =
                Vec2::from_angle(rng.next() * std::f32::consts::TAU) * spread * rng.next().sqrt();
            let at = center + offset;
            let (color, tip) = blade_color(at, &mut rng);
            let outward = (offset.normalize_or(clump_lean) + clump_lean * 0.6).normalize();
            blades.push(Blade {
                root: Vec3::new(at.x, ground(at) - 0.02, at.y),
                height: clump_height * (0.55 + 0.75 * rng.next()),
                width: 0.008 + 0.012 * rng.next(),
                lean: 0.18 + 0.95 * rng.next().powf(1.6),
                lean_dir: outward,
                twist: (rng.next() - 0.5) * 1.2,
                color,
                tip,
                flower: (lush > 0.45 && rng.next() < 0.018).then(|| petal_color(at, &mut rng)),
            });
        }
    }
    blades
}

pub fn spawn_meadow(
    mut commands: Commands,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
    props: Option<Res<PropFootprints>>,
) {
    let empty = PropFootprints::default();
    let props = props.as_deref().unwrap_or(&empty);
    let lighting = SpaghettiLighting {
        direction_to_light: SUN,
        ambient: Vec3::new(0.40, 0.47, 0.49),
        directional: Vec3::new(0.71, 0.55, 0.39),
    };
    let wind = SpaghettiWind {
        push_radius: 0.62,
        ..default()
    };
    let (low, high) = FAR_REGION;
    let columns = ((high.x - low.x) / CHUNK).ceil() as i32;
    let rows = ((high.y - low.y) / CHUNK).ceil() as i32;
    for row in 0..rows {
        for column in 0..columns {
            let min = low + Vec2::new(column as f32, row as f32) * CHUNK;
            let center = min + Vec2::splat(CHUNK * 0.5);
            let near = center.cmpge(NEAR_REGION.0).all() && center.cmple(NEAR_REGION.1).all();
            let blades = chunk_blades(min, (row * 977 + column) as u32 + 41, props);
            if blades.is_empty() {
                continue;
            }
            let origin = Vec3::new(center.x, ground(center), center.y);
            for (dense, name, levels, stride, range) in [
                (
                    false,
                    "Meadow grass, far",
                    FAR_LEVELS.as_slice(),
                    FAR_STRIDE,
                    VisibilityRange {
                        start_margin: if near { NEAR_END } else { 0.0..0.0 },
                        end_margin: FAR_END,
                        use_aabb: false,
                    },
                ),
                (
                    true,
                    "Meadow grass",
                    NEAR_LEVELS.as_slice(),
                    1,
                    VisibilityRange {
                        start_margin: 0.0..0.0,
                        end_margin: NEAR_END,
                        use_aabb: false,
                    },
                ),
            ] {
                if dense && !near {
                    continue;
                }
                let mut segments =
                    Vec::with_capacity(blades.len().div_ceil(stride) * (levels.len() - 1));
                for blade in blades.iter().step_by(stride) {
                    let blade = if dense {
                        *blade
                    } else {
                        Blade {
                            width: blade.width * 2.8,
                            flower: None,
                            ..*blade
                        }
                    };
                    blade.append(&mut segments, origin, levels);
                }
                commands.spawn((
                    Name::new(name),
                    Meadow,
                    Spaghetti3d(
                        clouds.add(SpaghettiAsset::new(segments).expect("valid grass splats")),
                    ),
                    SpaghettiCutout,
                    lighting,
                    SpaghettiTransmission(0.45),
                    wind,
                    Transform::from_translation(origin),
                    range,
                ));
            }
        }
    }
}

pub fn part_meadow(
    hiker: Single<&Transform, With<Hiker>>,
    time: Res<Time>,
    mut meadow: Query<&mut SpaghettiWind, With<Meadow>>,
) {
    for mut wind in &mut meadow {
        wind.time = time.elapsed_secs_wrapped();
        wind.pusher = hiker.translation;
    }
}
