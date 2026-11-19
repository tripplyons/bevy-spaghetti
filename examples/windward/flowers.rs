//! Subalpine wildflowers built from Gaussian cutouts.
//!
//! Five Cascade meadow species grow in drifts: broadleaf lupine, scarlet
//! paintbrush, subalpine daisy, mountain arnica and Sitka valerian. Each
//! plant has stems, leaves and separate florets up close, and merges into a
//! few colored kernels with distance.
use std::f32::consts::{FRAC_PI_2, TAU};

use bevy::{camera::visibility::VisibilityRange, prelude::*};
use bevy_spaghetti::{
    DetailTransition, GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiCutout,
    SpaghettiLighting, SpaghettiReceiveShadows, SpaghettiTransmission, SpaghettiWind,
};

use crate::{
    landscape::{SUN, ground, noise, smoothstep},
    meadow::{Meadow, ground_mask, lushness},
    props::PropFootprints,
};

const CHUNK: f32 = 16.0;
const REGION: (Vec2, Vec2) = (Vec2::new(-44.0, -70.0), Vec2::new(46.0, 32.0));
const RANGE_END: std::ops::Range<f32> = 84.0..100.0;
/// Projected pixels per meter where a level hands over to the next one.
/// Florets are about a centimeter wide, so they merge near 9 m at 800 lines.
const LEVEL_PIXELS: [f32; 2] = [88.0, 27.0];
const LEVELS: usize = 3;

const GOLDEN_ANGLE: f32 = 2.399_963;
const STEM: Vec3 = Vec3::new(0.060, 0.105, 0.034);
const LEAF: Vec3 = Vec3::new(0.052, 0.112, 0.036);

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9) | 1)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / 16_777_216.0
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next()
    }

    /// Starts a separate stream. A part that draws from its own stream uses
    /// one value of this one at every detail level, so the parts after it
    /// keep their shape as the level changes.
    fn fork(&mut self) -> Rng {
        Rng::new((self.next() * 16_777_216.0) as u32)
    }
}

/// Collects one plant's kernels. Bend heights follow each endpoint's height
/// above the root, so leaves and florets ride the stem in the wind.
struct Plant<'a> {
    out: &'a mut Vec<GaussianSegment>,
    root: Vec3,
    height: f32,
}

impl Plant<'_> {
    fn fraction(&self, at: Vec3) -> f32 {
        ((at.y - self.root.y) / self.height).clamp(0.0, 1.0)
    }

    /// A flat strap from `a` to `b` that faces `face`. `width` is its
    /// visible width at `a`, and `taper` is the end width as a fraction of it.
    fn blade(&mut self, a: Vec3, b: Vec3, face: Vec3, width: f32, taper: f32, color: Vec3) {
        let length = a.distance(b);
        if length < 1e-5 {
            return;
        }
        let tangent = (b - a) / length;
        let normal = (face - tangent * face.dot(tangent))
            .try_normalize()
            .unwrap_or_else(|| tangent.any_orthonormal_vector());
        let cross = tangent.cross(normal);
        // The cutout edge sits about 1.5 standard deviations from the axis.
        let sigma = width / 3.05;
        let extent = Vec3::new(sigma, sigma * 0.44, sigma * 0.26);
        let mass = 8.0 * sigma * length.max(2.5066 * extent.y);
        let fractions = [self.fraction(a), self.fraction(b)];
        self.out.push(
            GaussianSegment::new(
                a,
                b,
                extent,
                Quat::from_mat3(&Mat3::from_cols(cross, tangent, normal)),
                mass,
                color,
            )
            .with_blade_profile(self.height, fractions, taper),
        );
    }

    /// A round disc of visible radius `radius` that faces `face`.
    fn dot(&mut self, at: Vec3, face: Vec3, radius: f32, color: Vec3) {
        let sigma = radius / 1.52;
        let fraction = self.fraction(at);
        self.out.push(
            GaussianSegment::new(
                at,
                at,
                Vec3::new(sigma, sigma, sigma * 0.5),
                Quat::from_rotation_arc(Vec3::Z, face.normalize_or(Vec3::Y)),
                sigma * sigma * 21.0,
                color,
            )
            .with_blade_profile(self.height, [fraction, fraction], 1.0),
        );
    }
}

/// A leaning stem. `at(t)` runs from the root to the tip.
#[derive(Clone, Copy)]
struct Stem {
    root: Vec3,
    height: f32,
    lean: Vec3,
}

impl Stem {
    fn at(&self, t: f32) -> Vec3 {
        self.root + Vec3::Y * (self.height * t) + self.lean * (self.height * t * t)
    }

    fn draw(&self, plant: &mut Plant, parts: usize, to: f32, width: f32, color: Vec3) {
        for part in 0..parts {
            let a = self.at(to * part as f32 / parts as f32);
            let b = self.at(to * (part + 1) as f32 / parts as f32);
            plant.blade(a, b, self.lean + Vec3::X * 0.3, width, 1.0, color);
        }
    }
}

fn around(angle: f32) -> Vec3 {
    Vec3::new(angle.cos(), 0.0, angle.sin())
}

#[derive(Clone, Copy, PartialEq)]
enum Species {
    Lupine,
    Paintbrush,
    Daisy,
    Arnica,
    Valerian,
}

const SPECIES: [Species; 5] = [
    Species::Lupine,
    Species::Paintbrush,
    Species::Daisy,
    Species::Arnica,
    Species::Valerian,
];

impl Species {
    /// Plants per square meter inside a drift.
    fn density(self) -> f32 {
        match self {
            Species::Lupine => 0.60,
            Species::Paintbrush => 0.55,
            Species::Daisy => 1.5,
            Species::Arnica => 0.9,
            Species::Valerian => 0.40,
        }
    }

    /// 0 outside this species' drifts and 1 well inside them.
    fn drift(self, at: Vec2) -> f32 {
        let lush = lushness(at);
        match self {
            Species::Lupine => {
                smoothstep(0.42, 0.58, noise(at * 0.055 + Vec2::new(31.0, 4.0)))
                    * (0.35 + 0.65 * lush)
            }
            // Paintbrush likes the drier, shorter turf.
            Species::Paintbrush => {
                smoothstep(0.46, 0.60, noise(at * 0.075 + Vec2::new(-8.0, 19.0)))
                    * (1.0 - 0.6 * lush)
            }
            Species::Daisy => smoothstep(0.44, 0.58, noise(at * 0.065 + Vec2::new(5.0, -27.0))),
            Species::Arnica => smoothstep(0.48, 0.62, noise(at * 0.085 + Vec2::new(-21.0, -9.0))),
            Species::Valerian => {
                smoothstep(0.46, 0.60, noise(at * 0.06 + Vec2::new(14.0, 23.0))) * lush
            }
        }
    }

    fn grow(self, out: &mut Vec<GaussianSegment>, root: Vec3, seed: u32, level: usize) {
        // Each level replays the same random streams, so the plant keeps
        // its size, lean and color as it changes detail.
        let mut rng = Rng::new(seed);
        match self {
            Species::Lupine => lupine(out, root, &mut rng, level),
            Species::Paintbrush => paintbrush(out, root, &mut rng, level),
            Species::Daisy => daisy(out, root, &mut rng, level, false),
            Species::Arnica => daisy(out, root, &mut rng, level, true),
            Species::Valerian => valerian(out, root, &mut rng, level),
        }
    }
}

/// Broadleaf lupine: palmate leaves and a tall raceme of blue-violet pea
/// flowers that opens from the bottom, with pale buds at the tip.
fn lupine(out: &mut Vec<GaussianSegment>, root: Vec3, rng: &mut Rng, level: usize) {
    let stems = 2 + (rng.next() * 3.0) as usize;
    let clump_turn = rng.next() * TAU;
    let hue = rng.next();
    for index in 0..stems {
        let mut rng = rng.fork();
        let height = rng.range(0.42, 0.82);
        let turn = clump_turn + index as f32 * GOLDEN_ANGLE;
        let stem = Stem {
            root: root + around(turn) * rng.range(0.0, 0.07),
            height,
            lean: around(turn) * rng.range(0.02, 0.16),
        };
        let mut plant = Plant {
            out: &mut *out,
            root,
            height,
        };
        let spike = rng.range(0.52, 0.64);
        let violet = Vec3::new(0.040, 0.045, 0.30).lerp(Vec3::new(0.13, 0.035, 0.27), hue);
        let leaf = LEAF * rng.range(0.8, 1.25);
        let leaves = 4 + (rng.next() * 3.0) as usize;
        if level == 2 {
            plant.blade(
                stem.at(0.08),
                stem.at(spike),
                around(turn),
                0.11,
                0.5,
                leaf * 0.85,
            );
            plant.blade(
                stem.at(spike),
                stem.at(1.0),
                around(turn + 1.0),
                0.036,
                0.45,
                violet * 1.05,
            );
            continue;
        }
        stem.draw(
            &mut plant,
            if level == 0 { 4 } else { 1 },
            spike,
            0.006,
            STEM,
        );
        for leaf_index in 0..leaves {
            let mut rng = rng.fork();
            let t = 0.14 + 0.40 * (leaf_index as f32 + rng.next()) / leaves as f32;
            let angle = turn + leaf_index as f32 * GOLDEN_ANGLE + rng.range(-0.3, 0.3);
            let reach = rng.range(0.06, 0.11);
            let hub = stem.at(t) + (around(angle) + Vec3::Y * 0.55).normalize() * reach;
            let size = rng.range(0.042, 0.062);
            let shade = leaf * (0.72 + 0.4 * t);
            if level == 1 {
                plant.dot(hub, Vec3::Y, size * 0.95, shade);
                continue;
            }
            plant.blade(stem.at(t), hub, Vec3::Y, 0.003, 1.0, STEM);
            let leaflets = 7 + (rng.next() * 3.0) as usize;
            for leaflet in 0..leaflets {
                let spoke = angle + TAU * leaflet as f32 / leaflets as f32;
                // The leaflets cup upward around the petiole tip.
                let tip = hub + (around(spoke) + Vec3::Y * 0.28).normalize() * size;
                plant.blade(
                    hub,
                    tip,
                    Vec3::Y,
                    size * 0.30,
                    0.35,
                    shade * rng.range(0.85, 1.15),
                );
            }
        }
        // Whorls of florets, open below and in bud at the tip.
        let length = height * (1.0 - spike);
        if level == 1 {
            for part in 0..4 {
                let a = stem.at(spike + (1.0 - spike) * part as f32 / 4.0);
                let b = stem.at(spike + (1.0 - spike) * (part + 1) as f32 / 4.0);
                let bud = smoothstep(0.55, 1.0, (part as f32 + 0.5) / 4.0);
                let color = violet.lerp(Vec3::new(0.16, 0.20, 0.20), 0.7 * bud);
                plant.blade(
                    a,
                    b,
                    around(turn + part as f32),
                    0.034 * (1.0 - 0.55 * bud),
                    0.9,
                    color,
                );
            }
            continue;
        }
        stem.draw(&mut plant, 1, 1.0, 0.004, STEM * 0.9);
        let whorls = (length / 0.013) as usize;
        for whorl in 0..whorls {
            let along = (whorl as f32 + 0.5) / whorls as f32;
            let center = stem.at(spike + (1.0 - spike) * along);
            let bud = smoothstep(0.62, 1.0, along);
            let florets = if bud > 0.6 { 4 } else { 5 };
            for floret in 0..florets {
                let angle = whorl as f32 * 0.9 + TAU * floret as f32 / florets as f32;
                let outward = around(angle);
                let reach = 0.016 * (1.0 - 0.6 * bud);
                let from = center + outward * 0.003;
                let to = center + outward * reach + Vec3::Y * (0.004 + 0.007 * bud);
                let mut color = violet * rng.range(0.75, 1.25);
                if rng.next() < 0.22 {
                    // The banner petal carries a pale patch.
                    color = color.lerp(Vec3::new(0.40, 0.36, 0.60), 0.6);
                }
                color = color.lerp(Vec3::new(0.15, 0.20, 0.16), 0.75 * bud);
                plant.blade(
                    from,
                    to,
                    Vec3::Y + outward * 0.4,
                    0.011 * (1.0 - 0.5 * bud),
                    0.65,
                    color,
                );
            }
        }
    }
}

/// Scarlet paintbrush: narrow leaves up a purplish stem, crowned by a
/// brush of red bracts.
fn paintbrush(out: &mut Vec<GaussianSegment>, root: Vec3, rng: &mut Rng, level: usize) {
    let stems = 2 + (rng.next() * 4.0) as usize;
    let clump_turn = rng.next() * TAU;
    // Most plants are scarlet, and some lean magenta or orange.
    let hue = rng.next();
    let scarlet = if hue < 0.25 {
        Vec3::new(0.40, 0.022, 0.12)
    } else if hue > 0.8 {
        Vec3::new(0.50, 0.085, 0.012)
    } else {
        Vec3::new(0.46, 0.026, 0.016)
    };
    for index in 0..stems {
        let mut rng = rng.fork();
        let height = rng.range(0.17, 0.36);
        let turn = clump_turn + index as f32 * GOLDEN_ANGLE;
        let stem = Stem {
            root: root + around(turn) * rng.range(0.0, 0.05),
            height,
            lean: around(turn) * rng.range(0.10, 0.42),
        };
        let mut plant = Plant {
            out: &mut *out,
            root,
            height,
        };
        let head = 1.0 - rng.range(0.045, 0.065) / height;
        let stem_color = STEM.lerp(Vec3::new(0.11, 0.045, 0.05), 0.5);
        if level == 2 {
            plant.blade(
                stem.at(0.1),
                stem.at(head),
                around(turn),
                0.03,
                0.6,
                LEAF * 0.9,
            );
            plant.dot(stem.at(0.96), Vec3::Y, 0.024, scarlet);
            continue;
        }
        stem.draw(
            &mut plant,
            if level == 0 { 2 } else { 1 },
            1.0,
            0.004,
            stem_color,
        );
        if level == 1 {
            plant.blade(
                stem.at(0.12),
                stem.at(head),
                around(turn),
                0.026,
                0.6,
                LEAF * 0.9,
            );
            for part in 0..2 {
                let angle = turn + part as f32 * 1.7;
                plant.blade(
                    stem.at(head),
                    stem.at(1.02) + around(angle) * 0.008,
                    around(angle + FRAC_PI_2),
                    0.030,
                    0.4,
                    scarlet * (0.9 + 0.2 * part as f32),
                );
            }
            continue;
        }
        let leaves = 7 + (rng.next() * 4.0) as usize;
        for leaf in 0..leaves {
            let t = 0.12 + (head - 0.16) * leaf as f32 / leaves as f32;
            let angle = turn + leaf as f32 * GOLDEN_ANGLE;
            let from = stem.at(t);
            let to = from + (around(angle) + Vec3::Y * 0.9).normalize() * rng.range(0.025, 0.042);
            plant.blade(
                from,
                to,
                Vec3::Y - around(angle) * 0.6,
                0.006,
                0.3,
                LEAF * rng.range(0.8, 1.2),
            );
        }
        let bracts = 13 + (rng.next() * 6.0) as usize;
        for bract in 0..bracts {
            let along = bract as f32 / bracts as f32;
            let t = head + (1.0 - head) * along;
            let angle = turn + bract as f32 * GOLDEN_ANGLE;
            let from = stem.at(t);
            // Lower bracts spread, and the top ones stand almost upright.
            let rise = 1.1 + 2.4 * along;
            let length = rng.range(0.017, 0.030) * (1.0 - 0.3 * along);
            let to = from + (around(angle) + Vec3::Y * rise).normalize() * length;
            // Each bract is green at its base and red toward the lobed tip.
            let middle = from.lerp(to, 0.4);
            let color = scarlet * rng.range(0.8, 1.25);
            let face = around(angle) - Vec3::Y * 0.3;
            plant.blade(from, middle, face, 0.007, 1.0, color.lerp(LEAF, 0.45));
            plant.blade(
                middle,
                to,
                face,
                0.007,
                0.35,
                color.lerp(Vec3::new(0.70, 0.20, 0.03), 0.2 * rng.next()),
            );
        }
    }
}

/// A composite flower head on a slender stem. The subalpine daisy has
/// lavender rays, and arnica has broader yellow ones.
fn daisy(out: &mut Vec<GaussianSegment>, root: Vec3, rng: &mut Rng, level: usize, arnica: bool) {
    let stems = 1 + (rng.next() * 2.6) as usize;
    let clump_turn = rng.next() * TAU;
    let rays_color = if arnica {
        Vec3::new(0.66, 0.47, 0.035)
    } else {
        Vec3::new(0.24, 0.17, 0.42).lerp(Vec3::new(0.40, 0.30, 0.52), rng.next())
    };
    let disc = if arnica {
        Vec3::new(0.50, 0.30, 0.02)
    } else {
        Vec3::new(0.62, 0.43, 0.03)
    };
    for index in 0..stems {
        let mut rng = rng.fork();
        let height = rng.range(0.16, 0.36);
        let turn = clump_turn + index as f32 * GOLDEN_ANGLE;
        let stem = Stem {
            root: root + around(turn) * rng.range(0.0, 0.04),
            height,
            lean: around(turn) * rng.range(0.08, 0.36),
        };
        let mut plant = Plant {
            out: &mut *out,
            root,
            height,
        };
        let tip = stem.at(1.0);
        // The head tilts a little away from the stem's lean and toward the sun.
        let face = (Vec3::Y + around(turn) * 0.25 + SUN * 0.35).normalize();
        let radius = if arnica {
            rng.range(0.019, 0.026)
        } else {
            rng.range(0.015, 0.021)
        };
        let color = rays_color * rng.range(0.85, 1.15);
        if level == 2 {
            plant.blade(
                stem.at(0.0),
                stem.at(0.9),
                around(turn),
                0.016,
                0.5,
                LEAF * 0.9,
            );
            plant.dot(tip, face, radius * 0.95, color.lerp(disc, 0.18));
            continue;
        }
        stem.draw(&mut plant, if level == 0 { 2 } else { 1 }, 1.0, 0.003, STEM);
        let leaves = 3 + (rng.next() * 2.0) as usize;
        for leaf in 0..leaves {
            let mut rng = rng.fork();
            if level == 1 && leaf > 0 {
                continue;
            }
            let t = 0.04 + 0.5 * leaf as f32 / leaves as f32;
            let angle = turn + 1.0 + leaf as f32 * GOLDEN_ANGLE;
            let from = stem.at(t);
            let length = rng.range(0.04, 0.07) * if arnica { 1.2 } else { 1.0 };
            let to = from + (around(angle) + Vec3::Y * 0.8).normalize() * length;
            let width = if arnica { 0.020 } else { 0.011 };
            let width = if level == 1 { width * 1.8 } else { width };
            plant.blade(
                from,
                to,
                Vec3::Y - around(angle) * 0.5,
                width,
                0.3,
                LEAF * rng.range(0.8, 1.2),
            );
        }
        if level == 1 {
            plant.dot(tip, face, radius, color);
            plant.dot(tip + face * 0.002, face, radius * 0.34, disc);
            continue;
        }
        let side = face.any_orthonormal_vector();
        let other = face.cross(side);
        let rays = if arnica {
            10 + (rng.next() * 4.0) as usize
        } else {
            17 + (rng.next() * 8.0) as usize
        };
        for ray in 0..rays {
            let angle = TAU * (ray as f32 + rng.range(-0.25, 0.25)) / rays as f32;
            let outward = side * angle.cos() + other * angle.sin();
            // Rays droop a little past the rim of the disc.
            let from = tip + outward * radius * 0.22;
            let to = tip + outward * radius * rng.range(0.9, 1.1) - face * radius * 0.12;
            plant.blade(
                from,
                to,
                face,
                if arnica { 0.0075 } else { 0.0034 },
                0.7,
                color * rng.range(0.85, 1.15),
            );
        }
        plant.dot(tip + face * 0.003, face, radius * 0.30, disc);
        plant.dot(tip + face * 0.004, face, radius * 0.14, disc * 0.6);
    }
}

/// Sitka valerian: opposite leaves and a domed cluster of small white
/// flowers that open from pink buds.
fn valerian(out: &mut Vec<GaussianSegment>, root: Vec3, rng: &mut Rng, level: usize) {
    let height = rng.range(0.34, 0.68);
    let turn = rng.next() * TAU;
    let stem = Stem {
        root,
        height,
        lean: around(turn) * rng.range(0.03, 0.2),
    };
    let mut plant = Plant { out, root, height };
    let tip = stem.at(1.0);
    let dome = rng.range(0.028, 0.042);
    let white = Vec3::new(0.56, 0.55, 0.52);
    let pink = Vec3::new(0.50, 0.27, 0.31);
    if level == 2 {
        plant.blade(
            stem.at(0.05),
            stem.at(0.9),
            around(turn),
            0.05,
            0.4,
            LEAF * 0.9,
        );
        plant.dot(tip, Vec3::Y, dome * 0.95, white * 0.95);
        return;
    }
    stem.draw(&mut plant, if level == 0 { 3 } else { 1 }, 1.0, 0.005, STEM);
    let pairs = 2 + (rng.next() * 2.0) as usize;
    for pair in 0..pairs {
        let t = 0.18 + 0.5 * pair as f32 / pairs as f32;
        for side in 0..2 {
            let mut rng = rng.fork();
            let angle = turn + pair as f32 * FRAC_PI_2 + side as f32 * std::f32::consts::PI;
            let from = stem.at(t);
            let length = rng.range(0.06, 0.10) * (1.0 - 0.3 * t);
            let hub = from + (around(angle) + Vec3::Y * 0.5).normalize() * length;
            let color = LEAF * rng.range(0.85, 1.2);
            if level == 1 {
                plant.blade(from, hub, Vec3::Y, 0.05, 0.4, color);
                continue;
            }
            // A pinnate leaf: a broad end leaflet and two side leaflets.
            plant.blade(from, hub, Vec3::Y, 0.004, 1.0, STEM);
            for (spoke, size) in [(0.0_f32, 0.055), (1.1, 0.036), (-1.1, 0.036)] {
                let to = hub + (around(angle + spoke) + Vec3::Y * 0.15).normalize() * size;
                plant.blade(hub, to, Vec3::Y, size * 0.42, 0.25, color);
            }
        }
    }
    if level == 1 {
        for part in 0..4 {
            let offset = around(turn + part as f32 * FRAC_PI_2) * dome * 0.45;
            plant.dot(tip + offset, Vec3::Y, dome * 0.6, white);
        }
        return;
    }
    let florets = 30 + (rng.next() * 16.0) as usize;
    for floret in 0..florets {
        // A sunflower spiral over a shallow dome.
        let r = ((floret as f32 + 0.5) / florets as f32).sqrt();
        let outward = around(floret as f32 * GOLDEN_ANGLE);
        let at = tip + outward * (dome * r) + Vec3::Y * dome * 0.45 * (1.0 - r * r);
        let face = (Vec3::Y + outward * r * 0.7).normalize();
        let bud = rng.next() < 0.18 + 0.2 * r;
        let color = if bud { pink } else { white } * rng.range(0.85, 1.12);
        plant.dot(at, face, if bud { 0.0032 } else { 0.0046 }, color);
        if floret % 6 == 0 {
            plant.blade(tip - Vec3::Y * dome * 0.6, at, outward, 0.0018, 1.0, STEM);
        }
    }
}

pub fn spawn_flowers(
    mut commands: Commands,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
    props: Option<Res<PropFootprints>>,
) {
    let empty = PropFootprints::default();
    let props = props.as_deref().unwrap_or(&empty);
    // The meadow's gains, so petals sit in the same light as the grass.
    let lighting = SpaghettiLighting {
        direction_to_light: SUN,
        ambient: Vec3::new(0.40, 0.47, 0.49),
        directional: Vec3::new(0.71, 0.55, 0.39),
    };
    let wind = SpaghettiWind {
        push_radius: 0.62,
        ..default()
    };
    let (low, high) = REGION;
    let columns = ((high.x - low.x) / CHUNK).ceil() as i32;
    let rows = ((high.y - low.y) / CHUNK).ceil() as i32;
    for row in 0..rows {
        for column in 0..columns {
            let min = low + Vec2::new(column as f32, row as f32) * CHUNK;
            let center = min + Vec2::splat(CHUNK * 0.5);
            let origin = Vec3::new(center.x, ground(center), center.y);
            let mut rng = Rng::new((row * 613 + column) as u32 + 977);
            let mut plants = Vec::new();
            for species in SPECIES {
                let candidates = (CHUNK * CHUNK * species.density()) as usize;
                for _ in 0..candidates {
                    let at = min + Vec2::new(rng.next(), rng.next()) * CHUNK;
                    let seed = (rng.next() * 16_777_216.0) as u32;
                    let chance = rng.next();
                    let drift = species.drift(at);
                    if drift <= 0.0 || chance >= drift * ground_mask(at) * props.keep(at) {
                        continue;
                    }
                    let root = Vec3::new(at.x, ground(at) - 0.01, at.y) - origin;
                    plants.push((species, root, seed));
                }
            }
            if plants.is_empty() {
                continue;
            }
            let mut segments = Vec::new();
            let mut ends = [0; LEVELS];
            for (level, end) in ends.iter_mut().enumerate() {
                for &(species, root, seed) in &plants {
                    species.grow(&mut segments, root, seed, level);
                }
                *end = segments.len() as u32;
            }
            let transitions: [_; LEVELS - 1] = std::array::from_fn(|level| DetailTransition {
                finer_end: ends[level],
                pixels_per_unit: [0.82 * LEVEL_PIXELS[level], 1.18 * LEVEL_PIXELS[level]],
            });
            commands.spawn((
                Name::new("Wildflowers"),
                Meadow,
                Spaghetti3d(
                    clouds.add(
                        SpaghettiAsset::with_detail(segments, &transitions)
                            .expect("valid wildflower splats"),
                    ),
                ),
                SpaghettiCutout,
                lighting,
                SpaghettiReceiveShadows,
                SpaghettiTransmission(0.55),
                wind,
                Transform::from_translation(origin),
                VisibilityRange {
                    start_margin: 0.0..0.0,
                    end_margin: RANGE_END,
                    use_aabb: false,
                },
            ));
        }
    }
}
