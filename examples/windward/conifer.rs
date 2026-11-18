//! Procedural fir crowns built from Gaussian segments.
//!
//! Each tree has a tapered trunk, whorls of drooping branches, and flat
//! needle sprays. The sprays are thin kernels whose local +Z is the spray
//! normal, so the splat shader lights them as small leaves instead of as a
//! flat card. Three detail levels share one layout and differ only in how
//! finely the sprays are sampled.

use bevy::prelude::*;
use bevy_spaghetti::GaussianSegment;

use crate::landscape::hash01;

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9);
        hash01(self.0)
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next()
    }
}

/// Sampling parameters for one level of detail.
struct Level {
    /// Distance between spray pairs along a branch.
    spacing: f32,
    /// Scale applied to spray width so coverage stays constant.
    grow: f32,
    twiglets: bool,
    /// Number of bark strips around the trunk.
    strips: u32,
    trunk_step: f32,
    wood: bool,
}

const LEVELS: [Level; 4] = [
    Level {
        spacing: 0.07,
        grow: 1.0,
        twiglets: true,
        strips: 6,
        trunk_step: 0.22,
        wood: true,
    },
    Level {
        spacing: 0.085,
        grow: 1.75,
        twiglets: false,
        strips: 4,
        trunk_step: 0.45,
        wood: true,
    },
    Level {
        spacing: 0.3,
        grow: 3.0,
        twiglets: false,
        strips: 3,
        trunk_step: 1.1,
        wood: false,
    },
    Level {
        spacing: 0.62,
        grow: 4.6,
        twiglets: false,
        strips: 2,
        trunk_step: 2.2,
        wood: false,
    },
];

pub const LEVEL_COUNT: usize = LEVELS.len();

/// Growth habit of one species.
#[derive(Clone, Copy)]
pub struct Habit {
    /// Limb climb near the leader and how much it drops toward the base.
    climb_top: f32,
    climb_drop: f32,
    /// Sag of a limb as a fraction of its length.
    sag: f32,
    /// Upturn (positive) or droop (negative) at the limb tip.
    tip: f32,
    old: Vec3,
    young: Vec3,
    /// Sideways bend of the leader in meters.
    nod: f32,
}

/// Grand fir: stiff upturned limbs and dark blue-green needles.
pub const FIR: Habit = Habit {
    climb_top: 0.55,
    climb_drop: 0.85,
    sag: 0.26,
    tip: 0.12,
    old: Vec3::new(0.050, 0.082, 0.056),
    young: Vec3::new(0.105, 0.145, 0.068),
    nod: 0.0,
};

/// Western hemlock: soft pendulous limbs, a nodding leader, and yellower
/// green foliage.
pub const HEMLOCK: Habit = Habit {
    climb_top: 0.30,
    climb_drop: 0.62,
    sag: 0.42,
    tip: -0.10,
    old: Vec3::new(0.056, 0.090, 0.046),
    young: Vec3::new(0.125, 0.165, 0.062),
    nod: 0.55,
};

impl Habit {
    /// Offset of the stem axis at height `y`; only the top bends.
    fn bend(&self, y: f32, height: f32, toward: Vec3) -> Vec3 {
        let t = ((y / height - 0.82) / 0.18).max(0.0);
        toward * (self.nod * t * t)
    }
}

/// Builds one fir rooted at the origin with its leader at `height`.
pub fn fir(
    seed: u32,
    habit: &Habit,
    height: f32,
    crown_radius: f32,
    level: usize,
) -> Vec<GaussianSegment> {
    let detail = &LEVELS[level];
    let mut rng = Rng(seed.wrapping_mul(7919));
    let mut out = Vec::new();
    let base = height * rng.range(0.07, 0.12);
    let trunk_radius = 0.018 * height + 0.03;

    trunk(&mut out, &mut rng, height, base, trunk_radius, detail);

    // Annual whorls get closer together near the leader, where growth is
    // recent and the shoots are short.
    let lean = rng.range(0.0, std::f32::consts::TAU);
    let toward = Vec3::new(lean.cos(), 0.0, lean.sin());
    let mut y = base;
    let mut whorl = 0u32;
    while y < height - 0.25 {
        let rel = ((y - base) / (height - base)).clamp(0.0, 1.0);
        let reach = crown_radius * (1.0 - rel).powf(0.92) * rng.range(0.86, 1.08) + 0.12;
        let count = 5 + (rng.next() * 4.0) as u32;
        let turn = whorl as f32 * 2.399 + rng.range(0.0, 0.6);
        // Crowding and old storm damage leave short or missing limbs, and
        // one side of the crown usually reaches further toward the light.
        let vigor = rng.range(0.85, 1.12);
        for branch in 0..count {
            if rng.next() < 0.1 {
                continue;
            }
            let azimuth = turn
                + branch as f32 * std::f32::consts::TAU / count as f32
                + rng.range(-0.35, 0.35);
            let toward_light = 1.0 + 0.12 * (azimuth - lean).cos();
            let stunted = if rng.next() < 0.12 { 0.55 } else { 1.0 };
            let length = reach * rng.range(0.72, 1.08) * vigor * toward_light * stunted;
            limb(
                &mut out,
                &mut rng,
                habit,
                habit.bend(y, height, toward),
                y,
                azimuth,
                length,
                rel,
                height,
                detail,
            );
        }
        // Shorter internodal shoots fill the gaps between whorls.
        let step = (0.42 - 0.18 * rel) * height / 9.0;
        for _ in 0..3 {
            let at = y + step * rng.range(0.3, 0.75);
            let length = reach * rng.range(0.38, 0.62);
            let azimuth = rng.range(0.0, std::f32::consts::TAU);
            limb(
                &mut out,
                &mut rng,
                habit,
                habit.bend(at, height, toward),
                at,
                azimuth,
                length,
                rel,
                height,
                detail,
            );
        }
        y += step * rng.range(0.85, 1.15);
        whorl += 1;
    }

    // The leader is a short upright spray above the last whorl.
    let leader_from = Vec3::Y * (height - 0.55) + habit.bend(height - 0.55, height, toward);
    let top =
        Vec3::Y * height + habit.bend(height, height, toward) * 1.4 - Vec3::Y * habit.nod * 0.4;
    let color = needle_color(&mut rng, habit, 1.0, 1.0);
    spray(
        &mut out,
        leader_from,
        top,
        toward.cross(Vec3::Y),
        0.05 * detail.grow.min(2.0),
        color,
    );
    if level >= 2 {
        // A few dead stubs below the crown would vanish at this size anyway.
        return out;
    }
    // Dead lower branches: bare grey twigs below the living crown.
    let mut y = 0.9;
    while y < base {
        let azimuth = rng.range(0.0, std::f32::consts::TAU);
        let direction = Vec3::new(azimuth.cos(), rng.range(-0.25, 0.05), azimuth.sin());
        let length = rng.range(0.25, 0.75);
        let start = Vec3::Y * y + direction.with_y(0.0).normalize() * trunk_radius * 0.8;
        let end = start + direction.normalize() * length;
        let width = 0.012;
        out.push(GaussianSegment::new(
            start,
            end,
            Vec3::splat(width),
            Quat::from_rotation_arc(
                Vec3::Z,
                Vec3::new(direction.x, 0.6, direction.z).normalize(),
            ),
            5.0 * length * width,
            Vec3::new(0.085, 0.075, 0.065),
        ));
        y += rng.range(0.12, 0.35);
    }
    out
}

fn trunk(
    out: &mut Vec<GaussianSegment>,
    rng: &mut Rng,
    height: f32,
    base: f32,
    radius: f32,
    detail: &Level,
) {
    let mut y = 0.0;
    while y < height * 0.92 {
        let step = detail.trunk_step;
        let taper = |y: f32| {
            let flare = 1.0 + 0.7 * (-y / 0.35).exp();
            radius * (1.0 - y / height).max(0.06) * flare
        };
        let r0 = taper(y);
        let r1 = taper(y + step);
        for strip in 0..detail.strips {
            let angle = strip as f32 * std::f32::consts::TAU / detail.strips as f32
                + rng.range(-0.15, 0.15);
            let outward = Vec3::new(angle.cos(), 0.0, angle.sin());
            let start = Vec3::Y * y + outward * r0 * 0.62;
            let end = Vec3::Y * (y + step) + outward * r1 * 0.62;
            let width = r0 * std::f32::consts::PI / detail.strips as f32;
            // Ridged bark: dark grey with warm fissures.
            let fissure = rng.next();
            // The crown shades the stem above the lowest live limbs.
            let shade = if y > base { 0.42 } else { 1.0 };
            let tone = (0.75 + 0.5 * rng.next()) * shade;
            let color =
                Vec3::new(0.115, 0.105, 0.095).lerp(Vec3::new(0.060, 0.050, 0.042), fissure) * tone;
            let rotation = Quat::from_mat3(&Mat3::from_cols(
                Vec3::Y.cross(outward).normalize(),
                Vec3::Y,
                outward,
            ));
            out.push(GaussianSegment::new(
                start,
                end,
                Vec3::new(width * 0.55, width * 0.4, r0 * 0.3),
                rotation,
                6.0 * step * width,
                color,
            ));
        }
        y += step;
    }
}

#[allow(clippy::too_many_arguments)]
fn limb(
    out: &mut Vec<GaussianSegment>,
    rng: &mut Rng,
    habit: &Habit,
    shift: Vec3,
    y: f32,
    azimuth: f32,
    length: f32,
    rel: f32,
    height: f32,
    detail: &Level,
) {
    let outward = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
    let side = Vec3::Y.cross(outward).normalize();
    // Upper shoots climb; lower limbs droop and lift again at the tip.
    let climb = habit.climb_top - habit.climb_drop * (1.0 - rel) + rng.range(-0.1, 0.1);
    let sag = habit.sag * (1.0 - 0.6 * rel) * length;
    let point = |s: f32| {
        let lift = climb * s - sag / length.max(0.1) * s * s + habit.tip * (1.0 - rel) * s.powi(4);
        shift + Vec3::Y * y + outward * (0.05 + length * s) + Vec3::Y * (lift * length)
    };
    let shade_scale = 1.0 - 0.35 * (1.0 - rel);

    if detail.wood {
        let links = 5;
        for link in 0..links {
            let s0 = link as f32 / links as f32;
            let s1 = (link + 1) as f32 / links as f32;
            let width = 0.006 + 0.02 * (1.0 - s0) * (length / 2.5).min(1.0);
            out.push(GaussianSegment::new(
                point(s0),
                point(s1),
                Vec3::splat(width),
                Quat::from_rotation_arc(Vec3::Z, Vec3::Y),
                5.0 * width * length / links as f32,
                Vec3::new(0.07, 0.05, 0.036),
            ));
        }
    }

    // Side shoots spread the limb into a flat, ragged plate that is widest
    // near the trunk. Needles thin out on the shaded inner third.
    let mut s = 0.05;
    let mut sign = if rng.next() > 0.5 { 1.0f32 } else { -1.0 };
    let plates = !detail.wood;
    while s < 1.0 {
        let here = point(s);
        let tangent = (point((s + 0.02).min(1.0)) - here).normalize_or(outward);
        let inner = (1.0 - s / 0.35).max(0.0);
        let step = detail.spacing / length * rng.range(0.8, 1.2);
        if rng.next() < inner * 0.65 {
            s += step;
            sign = -sign;
            continue;
        }
        // Light reaches the outer shell; the interior stays dark.
        let depth = 0.42 + 0.58 * s.powf(0.7);
        let brightness = depth * shade_scale + 0.12 * (y / height);
        let tip_growth = (s - 0.72).max(0.0) * 2.5;
        let normal_up = tangent.cross(side).normalize();
        let shoot = (0.10 + 0.42 * (1.0 - s).powf(0.8))
            * (length / 2.0).clamp(0.4, 1.0)
            * rng.range(0.8, 1.15);
        let sides: &[f32] = if plates { &[-1.0, 1.0] } else { &[1.0] };
        for &flip in sides {
            let sign = sign * flip;
            let direction =
                (tangent * 0.5 + side * sign * 0.86 - Vec3::Y * rng.range(0.05, 0.25)).normalize();
            let roll = Quat::from_axis_angle(tangent, sign * rng.range(-0.15, 0.45));
            let normal = (roll * normal_up + outward * 0.25).normalize();
            if detail.twiglets {
                let across = normal.cross(direction).normalize();
                let color = needle_color(rng, habit, brightness, tip_growth);
                frond(
                    out,
                    here,
                    here + direction * shoot,
                    normal,
                    0.016,
                    color * 0.8,
                );
                let mut t = 0.0;
                while t < 1.0 {
                    let at = here + direction * shoot * t;
                    let reach = 0.10 * (1.0 - 0.45 * t) * rng.range(0.8, 1.2);
                    for lean in [-1.0f32, 1.0] {
                        let twig = (direction * 0.62 + across * lean * 0.78).normalize();
                        let tint = needle_color(rng, habit, brightness, tip_growth + t * 0.3);
                        frond(out, at, at + twig * reach, normal, 0.017, tint);
                    }
                    t += 0.06 / shoot * rng.range(0.85, 1.15);
                }
            } else {
                let color = needle_color(rng, habit, brightness, tip_growth);
                let width = if plates { 0.12 } else { 0.05 };
                frond(out, here, here + direction * shoot, normal, width, color);
            }
        }
        sign = -sign;
        s += step;
    }
    // Needles also clothe the limb itself.
    if !plates {
        let links = (length / 0.25).ceil().max(2.0) as u32;
        for link in 0..links {
            let s0 = link as f32 / links as f32;
            let s1 = (link + 1) as f32 / links as f32;
            if s1 < 0.3 {
                continue;
            }
            let normal = (point(s0) - point(s1)).cross(side).normalize_or(Vec3::Y) * -1.0;
            let normal = (normal + outward * 0.25).normalize();
            let color = needle_color(
                rng,
                habit,
                (0.42 + 0.58 * s1) * shade_scale,
                (s1 - 0.72).max(0.0) * 2.5,
            );
            frond(
                out,
                point(s0),
                point(s1),
                normal,
                0.045 * detail.grow.sqrt(),
                color,
            );
        }
    }
    // A terminal bud spray continues the limb past the last pair.
    let end = point(1.0);
    let tangent = (end - point(0.95)).normalize_or(outward);
    // Keep it narrow and only slightly lighter, or the tips read as dots.
    let color = needle_color(rng, habit, shade_scale, 0.45);
    spray(
        out,
        end,
        end + tangent * 0.26 * detail.grow.sqrt(),
        side,
        0.028 * detail.grow,
        color,
    );
}

/// A flat spray of needles along `start` → `end`, facing `normal`.
fn frond(
    out: &mut Vec<GaussianSegment>,
    start: Vec3,
    end: Vec3,
    normal: Vec3,
    width: f32,
    color: Vec3,
) {
    let along = (end - start).normalize_or(Vec3::X);
    let across = normal
        .cross(along)
        .try_normalize()
        .unwrap_or_else(|| along.any_orthonormal_vector());
    let normal = along.cross(across);
    let rotation = Quat::from_mat3(&Mat3::from_cols(along, across, normal)).normalize();
    let length = start.distance(end);
    out.push(GaussianSegment::new(
        start,
        end,
        Vec3::new(width * 0.7, width, width * 0.22),
        rotation,
        3.4 * length * width,
        color,
    ));
}

fn spray(
    out: &mut Vec<GaussianSegment>,
    start: Vec3,
    end: Vec3,
    side: Vec3,
    width: f32,
    color: Vec3,
) {
    let along = (end - start).normalize_or(Vec3::Y);
    let normal = side.cross(along).normalize_or(Vec3::Z);
    frond(out, start, end, normal, width, color);
}

/// Dark blue-green needles. New growth at the tips is lighter and yellower.
fn needle_color(rng: &mut Rng, habit: &Habit, brightness: f32, tip_growth: f32) -> Vec3 {
    let base = habit.old.lerp(
        habit.young,
        (tip_growth * rng.range(0.4, 1.0)).clamp(0.0, 1.0),
    );
    let tone = rng.range(0.78, 1.22);
    let hue = Vec3::new(rng.range(0.92, 1.08), 1.0, rng.range(0.9, 1.1));
    base * hue * tone * brightness
}
