//! Procedural ferns and broad-leaf groundcover built from Gaussian segments.
//!
//! A fern is a clump of arching fronds. Each frond has a rachis of chained
//! thin kernels and paired pinnae made of flattened kernels. A broad-leaf
//! clump is a rosette of ovate blades on short petioles. Every leaf kernel
//! uses its local +Z as the leaf normal, so the splat shader lights it as a
//! small leaf. The three detail levels share one layout and differ only in
//! how finely the leaves are sampled.

use std::f32::consts::TAU;

use bevy::prelude::*;
use bevy_spaghetti::GaussianSegment;

use crate::landscape::hash01;

/// Mass per unit length and unit width that keeps a leaf kernel mostly opaque.
const LEAF_MASS: f32 = 4.4;

pub const LEVEL_COUNT: usize = 3;

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(0x2c1b_3c6d) ^ 0x5bd1_e995)
    }

    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9);
        hash01(self.0)
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next()
    }
}

/// Sampling parameters for one fern level of detail.
struct FernLevel {
    /// Pinna pairs on a full-length frond.
    pairs: u32,
    /// Lobe pairs on the longest pinna. Zero draws each pinna as one kernel.
    lobes: u32,
    /// Width scale that keeps coverage constant with fewer pinnae.
    grow: f32,
    rachis_links: u32,
}

const FERN_LEVELS: [FernLevel; LEVEL_COUNT] = [
    FernLevel {
        pairs: 20,
        lobes: 4,
        grow: 1.0,
        rachis_links: 10,
    },
    FernLevel {
        pairs: 12,
        lobes: 0,
        grow: 1.3,
        rachis_links: 5,
    },
    FernLevel {
        pairs: 6,
        lobes: 0,
        grow: 2.2,
        rachis_links: 2,
    },
];

/// Builds one fern clump rooted at the origin. Fronds reach about `radius`
/// from the crown, and the highest arch is about `height` above the ground.
pub fn fern(seed: u32, radius: f32, height: f32, level: usize) -> Vec<GaussianSegment> {
    let detail = &FERN_LEVELS[level];
    let mut layout = Rng::new(seed);
    let mut out = Vec::new();
    let count = 6 + (layout.next() * 4.0) as u32;
    let turn = layout.range(0.0, TAU);
    for index in 0..count {
        // Golden-angle spacing spreads the fronds evenly around the crown.
        let azimuth = turn + index as f32 * 2.399 + layout.range(-0.3, 0.3);
        // Old fronds are long, low, and droop. Young fronds stand upright.
        let age = layout.next();
        let frond = Frond {
            azimuth,
            length: radius * layout.range(1.0, 1.22) * (0.78 + 0.32 * age),
            rise: 1.22 - 0.42 * age + layout.range(-0.1, 0.1),
            droop: 1.45 + 1.0 * age + layout.range(-0.2, 0.2),
            lean: layout.range(-0.25, 0.25),
            age,
            peak: height * layout.range(0.78, 1.0),
            seed: layout.0,
        };
        frond.build(&mut out, detail);
    }
    out
}

struct Frond {
    azimuth: f32,
    length: f32,
    /// Elevation of the stipe where it leaves the crown, in radians.
    rise: f32,
    /// Total bend of the rachis from base to tip, in radians.
    droop: f32,
    /// Sideways curve of the rachis.
    lean: f32,
    age: f32,
    peak: f32,
    seed: u32,
}

impl Frond {
    fn build(&self, out: &mut Vec<GaussianSegment>, detail: &FernLevel) {
        const STEPS: usize = 32;
        let outward = Vec3::new(self.azimuth.cos(), 0.0, self.azimuth.sin());
        let side = Vec3::Y.cross(outward).normalize();
        // Integrate the arching rachis, then flatten it to the clump height.
        let mut spine = [Vec3::ZERO; STEPS + 1];
        let step = self.length / STEPS as f32;
        for index in 1..=STEPS {
            let s = (index as f32 - 0.5) / STEPS as f32;
            let angle = self.rise - self.droop * s.powf(1.35);
            let heading = (outward + side * self.lean * s).normalize();
            spine[index] =
                spine[index - 1] + (heading * angle.cos() + Vec3::Y * angle.sin()) * step;
        }
        let top = spine.iter().map(|p| p.y).fold(0.0, f32::max);
        let squash = (self.peak / top.max(1e-3)).min(1.0);
        for point in &mut spine {
            point.y *= squash;
        }
        let point = |t: f32| {
            let x = t.clamp(0.0, 1.0) * STEPS as f32;
            let index = (x as usize).min(STEPS - 1);
            spine[index].lerp(spine[index + 1], x - index as f32)
        };
        let tangent = |t: f32| (point(t + 0.02) - point(t - 0.02)).normalize_or(outward);

        let mut rng = Rng::new(self.seed);
        let tint = rng.range(0.0, 1.0);

        // Rachis: a tapered chain from the brown stipe to the green tip.
        let links = detail.rachis_links;
        for link in 0..links {
            let t0 = link as f32 / links as f32;
            let t1 = (link + 1) as f32 / links as f32;
            let start = point(t0);
            let end = point(t1);
            let along = (end - start).normalize_or(outward);
            let normal = along.cross(side).normalize_or(Vec3::Y);
            let width = self.length * (0.0055 - 0.0035 * t0) * detail.grow.sqrt();
            let stipe = Vec3::new(0.070, 0.055, 0.032);
            let green = Vec3::new(0.058, 0.095, 0.030);
            let color = stipe.lerp(green, (t0 * 2.5).min(1.0));
            out.push(blade(start, end, normal, width, 5.0 * width, color));
        }

        // Pinnae start above the bare stipe and crowd together toward the tip.
        let stipe = 0.2 + 0.06 * rng.next();
        let pairs = detail.pairs;
        for pair in 0..pairs {
            let u = (pair as f32 + 0.45) / pairs as f32;
            let t = stipe + (1.0 - stipe) * u.powf(0.85);
            let here = point(t);
            let along = tangent(t);
            let normal = along.cross(side).normalize_or(Vec3::Y);
            // A lanceolate frond: widest a third of the way up.
            let profile = (u + 0.08).powf(0.45) * (1.0 - u).powf(0.85) / 0.5;
            let reach = 0.19 * self.length * profile * detail.grow.powf(0.25);
            if reach < 0.004 {
                continue;
            }
            let droop = 0.12 + 0.28 * self.age;
            // Lower pinnae sit in the shade of the clump.
            let light = 0.62 + 0.38 * (here.y / self.peak.max(0.05)).min(1.0);
            for sign in [-1.0f32, 1.0] {
                let direction = (along * (0.42 + 0.25 * u) + side * sign * 0.9
                    - normal * droop * rng.range(0.6, 1.4))
                .normalize();
                // Pinnae tilt up from the rachis, so the frond has a shallow keel.
                let leaf_normal = (normal - direction * 0.18 + side * sign * 0.12).normalize();
                let start = here + side * sign * self.length * 0.004;
                let length = reach * rng.range(0.88, 1.1);
                let color = fern_color(&mut rng, light, tint, self.age, u);
                self.pinna(
                    out,
                    detail,
                    start,
                    direction,
                    leaf_normal,
                    length,
                    color,
                    &mut rng,
                );
            }
        }
        // A terminal segment continues the frond past the last pair.
        let tip = point(0.93);
        let along = tangent(0.96);
        let normal = along.cross(side).normalize_or(Vec3::Y);
        let width = 0.010 * self.length * detail.grow.sqrt();
        let color = fern_color(&mut rng, 1.0, tint, self.age, 1.0);
        out.push(blade(
            tip,
            point(1.0) + along * 0.02 * self.length,
            normal,
            width,
            LEAF_MASS * width,
            color,
        ));
    }

    #[allow(clippy::too_many_arguments)]
    fn pinna(
        &self,
        out: &mut Vec<GaussianSegment>,
        detail: &FernLevel,
        start: Vec3,
        direction: Vec3,
        normal: Vec3,
        length: f32,
        color: Vec3,
        rng: &mut Rng,
    ) {
        // Pinnae curve down toward their tips.
        let end = start + direction * length - normal * length * 0.12;
        let width = 0.06 * length * detail.grow;
        if detail.lobes == 0 {
            out.push(blade(start, end, normal, width, LEAF_MASS * width, color));
            return;
        }
        // Close up, each pinna is a midrib with pairs of rounded lobes, which
        // gives the frond its toothed outline.
        let along = (end - start).normalize();
        let across = normal.cross(along).normalize();
        let lobes =
            ((detail.lobes as f32 * (length / (0.25 * self.length)).sqrt()).round() as u32).max(2);
        out.push(blade(
            start,
            end,
            normal,
            width * 0.32,
            LEAF_MASS * width * 0.32,
            color * 0.82,
        ));
        for lobe in 0..lobes {
            let u = (lobe as f32 + 0.5) / lobes as f32;
            let base = start + (end - start) * (0.06 + 0.86 * u);
            let reach = length * 0.17 * (1.0 - 0.55 * u);
            let lobe_width = reach * 0.34;
            for sign in [-1.0f32, 1.0] {
                let out_dir = (along * 0.62 + across * sign * 0.78).normalize();
                let tone = color * rng.range(0.9, 1.1);
                let tilt = (normal + across * sign * 0.15).normalize();
                out.push(blade(
                    base - out_dir * reach * 0.15,
                    base + out_dir * reach,
                    tilt,
                    lobe_width,
                    LEAF_MASS * lobe_width,
                    tone,
                ));
            }
        }
        // The pinna tip is a narrow lobe on the midrib.
        let tip_start = start + (end - start) * 0.82;
        out.push(blade(
            tip_start,
            end,
            normal,
            width * 0.7,
            LEAF_MASS * width * 0.7,
            color,
        ));
    }
}

/// Linear albedo of a fern pinna. Young fronds are a brighter yellow-green.
/// Old fronds are darker and yellow toward their tips.
fn fern_color(rng: &mut Rng, light: f32, tint: f32, age: f32, u: f32) -> Vec3 {
    let young = Vec3::new(0.070, 0.135, 0.030);
    let old = Vec3::new(0.046, 0.090, 0.024);
    let yellowed = Vec3::new(0.105, 0.105, 0.032);
    let mut base = young.lerp(old, age);
    let fade = (u - 0.72).max(0.0) * 3.0 * age * tint;
    base = base.lerp(yellowed, fade.min(0.7));
    let hue = Vec3::new(rng.range(0.92, 1.08), 1.0, rng.range(0.88, 1.12));
    base * hue * rng.range(0.85, 1.15) * light
}

/// Builds one broad-leaf groundcover clump rooted at the origin. Leaves reach
/// about `radius` from the center and rise about `height` above the ground.
pub fn groundcover(seed: u32, radius: f32, height: f32, level: usize) -> Vec<GaussianSegment> {
    let mut layout = Rng::new(seed ^ 0x00a7_f00d);
    let mut out = Vec::new();
    let count = 6 + (layout.next() * 4.0) as u32;
    let turn = layout.range(0.0, TAU);
    for index in 0..count {
        let azimuth = turn + index as f32 * 2.399 + layout.range(-0.25, 0.25);
        let outward = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
        let side = Vec3::Y.cross(outward).normalize();
        // Inner leaves are young, small, and upright.
        let inner = layout.next();
        let petiole = radius * layout.range(0.32, 0.5) * (1.0 - 0.35 * inner);
        let elevation = layout.range(0.55, 0.95) + 0.35 * inner;
        let blade_length = radius * layout.range(0.58, 0.78) * (1.0 - 0.3 * inner);
        let base = outward * petiole * elevation.cos()
            + Vec3::Y * (petiole * elevation.sin()).min(height * 0.7);
        // The blade lies near horizontal and sags at the tip.
        let pitch = layout.range(-0.05, 0.3) + 0.35 * inner;
        let heading = (outward + side * layout.range(-0.2, 0.2)).normalize();
        let along = (heading * pitch.cos() + Vec3::Y * pitch.sin()).normalize();
        let tip = base + along * blade_length - Vec3::Y * blade_length * layout.range(0.08, 0.22);
        let roll = layout.range(-0.25, 0.25);
        let leaf_side = (Vec3::Y.cross(along).normalize_or(side) + Vec3::Y * roll).normalize();
        let normal = (tip - base)
            .normalize()
            .cross(leaf_side)
            .normalize_or(Vec3::Y);
        let normal = if normal.y < 0.0 { -normal } else { normal };
        let width = blade_length * layout.range(0.62, 0.78);
        let old = layout.next() < 0.18;
        let light = 0.68 + 0.32 * (1.0 - inner * 0.5) * (base.y / height.max(0.01)).clamp(0.4, 1.0);
        let mut rng = Rng::new(layout.0);
        let color = leaf_color(&mut rng, light, old);
        broad_leaf(&mut out, level, base, tip, normal, width, color, &mut rng);
        if level < 2 {
            let stem = Vec3::new(0.050, 0.040, 0.026) * rng.range(0.8, 1.2);
            let w = 0.012 * radius;
            out.push(blade(Vec3::Y * 0.004, base, normal, w, 5.0 * w, stem));
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn broad_leaf(
    out: &mut Vec<GaussianSegment>,
    level: usize,
    base: Vec3,
    tip: Vec3,
    normal: Vec3,
    width: f32,
    color: Vec3,
    rng: &mut Rng,
) {
    let span = tip - base;
    let along = span.normalize();
    let across = normal.cross(along).normalize();
    let at = |t: f32| base + span * t;
    match level {
        0 => {
            // Two halves folded along the midrib, each an ovate base lobe and
            // a narrower tip. The base lobes reach behind the petiole, so the
            // blade has a heart-shaped base.
            for sign in [-1.0f32, 1.0] {
                let fold = (normal + across * sign * 0.28).normalize();
                let offset = across * sign * width * 0.17;
                let half = color * rng.range(0.92, 1.08);
                let w = width * 0.2;
                out.push(blade(
                    at(-0.06) + offset,
                    at(0.55) + offset,
                    fold,
                    w,
                    LEAF_MASS * w,
                    half,
                ));
                let w = width * 0.13;
                out.push(blade(
                    at(0.45) + offset * 0.6,
                    at(0.92) + offset * 0.2,
                    fold,
                    w,
                    LEAF_MASS * w,
                    half,
                ));
            }
            // A darker midrib shows where the halves meet.
            let w = width * 0.035;
            out.push(blade(at(0.0), at(0.9), normal, w, 5.0 * w, color * 0.7));
        }
        1 => {
            let w = width * 0.28;
            out.push(blade(at(0.0), at(0.55), normal, w, LEAF_MASS * w, color));
            let w = width * 0.18;
            out.push(blade(at(0.45), at(0.92), normal, w, LEAF_MASS * w, color));
        }
        _ => {
            let w = width * 0.27;
            out.push(blade(at(0.05), at(0.82), normal, w, LEAF_MASS * w, color));
        }
    }
}

/// Linear albedo of a broad leaf: a dark, slightly blue green. A few old
/// leaves are yellower.
fn leaf_color(rng: &mut Rng, light: f32, old: bool) -> Vec3 {
    let base = if old {
        Vec3::new(0.085, 0.092, 0.030)
    } else {
        Vec3::new(0.044, 0.088, 0.032)
    };
    let hue = Vec3::new(rng.range(0.9, 1.1), 1.0, rng.range(0.85, 1.15));
    base * hue * rng.range(0.85, 1.15) * light
}

/// A thin flattened kernel along `start` → `end` whose local +Z is `normal`.
/// `width` is the standard deviation across the leaf.
fn blade(
    start: Vec3,
    end: Vec3,
    normal: Vec3,
    width: f32,
    mass_per_length: f32,
    color: Vec3,
) -> GaussianSegment {
    let along = (end - start).normalize_or(Vec3::X);
    let across = normal
        .cross(along)
        .try_normalize()
        .unwrap_or_else(|| along.any_orthonormal_vector());
    let normal = along.cross(across);
    let rotation = Quat::from_mat3(&Mat3::from_cols(along, across, normal));
    let length = start.distance(end);
    GaussianSegment::new(
        start,
        end,
        Vec3::new(width * 0.6, width, width * 0.2),
        rotation,
        mass_per_length * length,
        color,
    )
}
