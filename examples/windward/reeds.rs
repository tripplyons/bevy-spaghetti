use std::collections::BTreeMap;

use bevy::prelude::*;
use bevy_spaghetti::{GaussianSegment, Spaghetti3d, SpaghettiAsset};

use crate::{
    detail::{CameraView, ScreenDetail},
    landscape::{LAKE, LAKE_RADII, WATER_LEVEL, ground, hash01, noise, shore_distance, trail_x},
    player::{Hiker, HikerCamera},
};

const PIECES: usize = 3;
const CELL_SIZE: f32 = 11.0;
const TOUCH_RADIUS: f32 = 1.08;

#[derive(Component)]
pub struct ReedPatch {
    cloud: Handle<SpaghettiAsset>,
    center: Vec3,
    longest_blade: f32,
    pieces: usize,
    blades: Vec<Blade>,
}

struct Blade {
    root: Vec3,
    length: f32,
    rest_slope: Vec2,
    bend: Vec2,
    velocity: Vec2,
    width: f32,
    color: Vec3,
    twist_rotation: Quat,
    stiffness: f32,
    cat_tail: bool,
    /// How far the tip has dried to straw, from 0 to 1.
    dry: f32,
}

impl Blade {
    fn step(&mut self, time: f32, dt: f32, hiker: Vec3) {
        let at = self.root.xz();
        let front = (at.dot(Vec2::new(-0.29, 0.19)) + time * 1.8).sin();
        let pocket = noise(at * 0.12 - Vec2::new(0.82, 0.32) * time);
        let wind = Vec2::new(0.82, 0.47) * (0.14 + 0.50 * pocket * pocket + 0.17 * front)
            + Vec2::new(-0.30, 0.70) * (0.10 * (time * 1.34 + at.x * 0.22).sin());
        let away = at - hiker.xz();
        let distance_squared = away.length_squared();
        let touch = if distance_squared <= TOUCH_RADIUS * TOUCH_RADIUS
            && hiker.y < self.root.y + self.length + 0.22
        {
            let distance = distance_squared.sqrt();
            let overlap = (1.0 - distance / TOUCH_RADIUS).max(0.0);
            away / distance.max(0.12) * (3.7 * overlap.powi(2))
        } else {
            Vec2::ZERO
        };

        // Damped spring on the tip slope. We pin the root and reconstruct
        // the centerline from fixed-length links using this slope.
        let acceleration = self.stiffness * (wind + touch - self.bend) - 5.6 * self.velocity;
        self.velocity += acceleration * dt;
        self.bend = (self.bend + self.velocity * dt).clamp_length_max(2.8);
    }

    fn append_splats(&self, links: &[LinkShape], out: &mut Vec<GaussianSegment>) {
        let mut start = self.root;
        let link_length = self.length / links.len() as f32;
        let mut axis = Vec3::Y;
        for &LinkShape { t, bend_weight } in links {
            let slope = self.rest_slope * t + self.bend * bend_weight;
            axis = Vec3::new(slope.x, 1.0, slope.y).normalize();
            let end = start + axis * link_length;
            let width = self.width * if self.cat_tail { 1.0 } else { 1.07 - 0.91 * t };
            let rotation = Quat::from_rotation_arc(Vec3::Y, axis) * self.twist_rotation;
            let mut tint = self.color * (0.52 + 0.72 * t) + Vec3::new(0.018, 0.022, 0.005) * t * t;
            if !self.cat_tail {
                // Uneven pigment along the blade, and sun-bleached tips.
                let mottle = (self.root.x * 913.7 + self.root.z * 377.1 + t * 23.0).sin();
                tint *= 1.0 + 0.11 * mottle;
                let tip = ((t - 0.68) / 0.32).clamp(0.0, 1.0) * self.dry;
                tint = tint.lerp(Vec3::new(0.30, 0.235, 0.115), tip * tip);
            }
            out.push(GaussianSegment::new(
                start,
                end,
                Vec3::new(
                    width,
                    width * 0.64,
                    width * if self.cat_tail { 1.0 } else { 0.52 },
                ),
                rotation,
                link_length * width * 5.2,
                tint,
            ));
            start = end;
        }
        if self.cat_tail {
            let width = self.width * 2.8;
            let tip = start + axis * (self.length * 0.15);
            out.push(GaussianSegment::new(
                start,
                tip,
                Vec3::new(width, width * 0.67, width),
                Quat::from_rotation_arc(Vec3::Y, axis),
                tip.distance(start) * width * 6.3,
                Vec3::new(0.045, 0.025, 0.014),
            ));
        }
    }
}

pub fn spawn_reeds(mut commands: Commands, mut clouds: ResMut<Assets<SpaghettiAsset>>) {
    let mut buckets: BTreeMap<(i32, i32), Vec<Blade>> = BTreeMap::new();
    for seed in 1..=5750u32 {
        let id = seed * 11;
        let shoreline = seed.is_multiple_of(3);
        let (at, tall) = if shoreline {
            let turn = hash01(id) * std::f32::consts::TAU;
            let (sin, cos) = turn.sin_cos();
            let reeds = 0.072 * (turn * 5.0 + 0.6).sin() + 0.041 * (turn * 9.0).sin();
            let radius = 0.98 + 0.34 * hash01(id + 1).powi(2) + reeds;
            // Walk along the ray until the noisy shoreline reaches this
            // distance, so reeds follow its coves and points.
            let toward = Vec2::new(cos, sin) * LAKE_RADII;
            let mut along = radius;
            for _ in 0..5 {
                along += radius - shore_distance(LAKE + toward * along);
            }
            let at = LAKE + toward * along.max(0.3);
            (at, radius < 1.113 && noise(at * 0.29) > 0.28)
        } else {
            let z = -52.0 + 71.0 * hash01(id);
            let side = if seed % 2 == 0 { 1.0 } else { -1.0 };
            let x = trail_x(z) + side * (1.05 + 16.8 * hash01(id + 1).powf(1.66));
            (Vec2::new(x, z), false)
        };
        let base_height = ground(at);
        if base_height < WATER_LEVEL - if tall { 0.21 } else { 0.035 }
            || (at.x - trail_x(at.y)).abs() < 1.16
            || (!shoreline && noise(at * 0.24) < 0.26)
        {
            continue;
        }
        // Clumps share soil, but each stem leans and responds independently.
        let count = if tall { 4 } else { 3 };
        let tuft_height = if tall {
            0.55 + 0.87 * hash01(id + 2)
        } else {
            let moisture = noise(at * 0.25);
            0.10 + (0.34 + 0.22 * moisture) * hash01(id + 2).powf(1.8)
        };
        let green = Vec3::new(0.030, 0.086, 0.023);
        let dry = Vec3::new(0.21, 0.153, 0.048);
        let tint = green.lerp(dry, hash01(id + 3).powf(2.4));
        let index = (at / CELL_SIZE).floor().as_ivec2();
        let bucket = buckets.entry((index.x, index.y)).or_default();
        for stem in 0..count {
            let s = id.wrapping_mul(233).wrapping_add(stem * 7);
            let direction = Vec2::from_angle(hash01(s) * std::f32::consts::TAU);
            let origin = at + direction * (0.21 * hash01(s + 1));
            if (origin.x - trail_x(origin.y)).abs() < 0.97 {
                continue;
            }
            let cat_tail = tall && stem == 0 && hash01(s + 5) > 0.32;
            let length = tuft_height * (0.68 + 0.49 * hash01(s + 2));
            let spread = if cat_tail {
                0.18
            } else {
                0.49 + 0.75 * hash01(s + 3)
            };
            let width = if cat_tail {
                0.0064
            } else {
                0.0033 + 0.0092 * hash01(s + 4)
            };
            let twist = hash01(s + 5) * std::f32::consts::TAU;
            bucket.push(Blade {
                root: Vec3::new(origin.x, ground(origin) - 0.015, origin.y),
                length,
                rest_slope: direction * spread,
                bend: Vec2::new(0.13, 0.07),
                velocity: Vec2::ZERO,
                width,
                color: tint * (0.78 + 0.32 * hash01(s + 6)),
                twist_rotation: Quat::from_rotation_y(twist),
                stiffness: blade_stiffness(cat_tail, twist),
                cat_tail,
                dry: (hash01(s + 7) * 1.6 - 0.6).max(0.0),
            });
        }
    }

    for ((x, z), blades) in buckets {
        if blades.is_empty() {
            continue;
        }
        let center = (Vec2::new(x as f32, z as f32) + 0.5) * CELL_SIZE;
        let longest_blade = blades
            .iter()
            .map(|blade| blade.length)
            .fold(0.0_f32, f32::max);
        let cloud = clouds.add(build_cloud(&blades, PIECES));
        commands.spawn((
            Name::new("Root-pinned reed bed"),
            Spaghetti3d(cloud.clone()),
            // The low sun lights blades that face it and shines through the
            // ones that face away, so the bed is no longer evenly flat.
            bevy_spaghetti::SpaghettiLighting {
                direction_to_light: crate::landscape::SUN,
                ambient: Vec3::new(0.50, 0.56, 0.60),
                directional: Vec3::new(1.00, 0.84, 0.64),
            },
            bevy_spaghetti::SpaghettiTransmission(0.6),
            ReedPatch {
                cloud,
                blades,
                center: Vec3::new(center.x, ground(center) + longest_blade * 0.5, center.y),
                longest_blade,
                pieces: PIECES,
            },
        ));
    }
}

fn blade_stiffness(cat_tail: bool, twist: f32) -> f32 {
    if cat_tail {
        26.0
    } else {
        37.0 + 17.0 * twist.cos().abs()
    }
}

struct LinkShape {
    t: f32,
    bend_weight: f32,
}

fn build_cloud(blades: &[Blade], pieces: usize) -> SpaghettiAsset {
    let links: [LinkShape; PIECES] = std::array::from_fn(|i| {
        let t = (i as f32 + 0.5) / pieces as f32;
        LinkShape {
            t,
            bend_weight: t.powf(1.7),
        }
    });
    let mut splats = Vec::with_capacity(blades.len() * pieces);
    for blade in blades {
        blade.append_splats(&links[..pieces], &mut splats);
    }
    SpaghettiAsset::new(splats).expect("finite, nonempty reed patch")
}

pub fn step_reeds(
    time: Res<Time<Fixed>>,
    hiker: Single<&Transform, With<Hiker>>,
    camera: Single<CameraView, With<HikerCamera>>,
    mut patches: Query<&mut ReedPatch>,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
    mut tick: Local<u64>,
) {
    *tick = tick.wrapping_add(1);
    let dt = time.delta_secs();
    let elapsed = time.elapsed_secs();
    let hiker = hiker.translation;
    let view = ScreenDetail::new(*camera);
    for mut patch in &mut patches {
        let pixels = view.as_ref().map_or(42.0, |view| {
            view.span_pixels(patch.center, patch.longest_blade, CELL_SIZE * 0.64)
        });
        let full = if patch.pieces == PIECES { 52.0 } else { 64.0 };
        let doubled = if patch.pieces == 1 { 24.0 } else { 19.0 };
        let (pieces, interval) = if pixels > full {
            (PIECES, 1)
        } else if pixels > doubled {
            (2, 2)
        } else {
            (1, 4)
        };
        for blade in &mut patch.blades {
            blade.step(elapsed, dt, hiker);
        }
        if patch.pieces != pieces || tick.is_multiple_of(interval) {
            clouds
                .insert(patch.cloud.id(), build_cloud(&patch.blades, pieces))
                .expect("the reed entity owns the asset handle");
            patch.pieces = pieces;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gusts_and_contact_bend_the_stem_without_moving_its_root_or_stretching_links() {
        let mut blade = Blade {
            root: Vec3::new(2.0, 0.0, -1.0),
            length: 1.1,
            rest_slope: Vec2::new(0.17, 0.04),
            bend: Vec2::ZERO,
            velocity: Vec2::ZERO,
            width: 0.011,
            color: Vec3::new(0.15, 0.22, 0.08),
            twist_rotation: Quat::from_rotation_y(1.4),
            stiffness: blade_stiffness(false, 1.4),
            cat_tail: false,
            dry: 0.0,
        };
        let original = build_cloud(std::slice::from_ref(&blade), PIECES);
        let dt = 1.0 / 64.0;
        for step in 0..64 {
            blade.step(step as f32 * dt, dt, blade.root - 0.22 * Vec3::X);
        }
        let brushed = build_cloud(std::slice::from_ref(&blade), PIECES);
        assert!(brushed.segments()[2].end().x > original.segments()[2].end().x + 0.10);
        for pair in brushed.segments().windows(2) {
            assert_eq!(pair[0].end(), pair[1].start());
        }
        assert_eq!(brushed.segments()[0].start(), blade.root);
        for link in brushed.segments() {
            assert!(
                (link.start().distance(link.end()) - blade.length / PIECES as f32).abs() < 1e-5
            );
        }
        for step in 64..180 {
            blade.step(step as f32 * dt, dt, Vec3::splat(50.0));
        }
        let recovered = build_cloud(std::slice::from_ref(&blade), PIECES);
        assert!(recovered.segments()[2].end().x < brushed.segments()[2].end().x - 0.09);
    }
}
