//! Procedural fir crowns built from Gaussian segments.
//!
//! Each tree has a tapered trunk and uneven whorls of limbs. A limb droops
//! from the trunk and, in firs, sweeps up at the tip. Side branchlets
//! alternate along it and hang below it by species, and the finest level
//! breaks them into tufted twigs. Needle sprays are thin kernels whose
//! local +Z is the spray normal, so the splat shader lights them as small
//! leaves instead of as a flat card. All four detail levels share one
//! limb and branchlet layout and differ only in how finely they draw it.

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

    /// A child stream that advances this one by exactly one step.
    ///
    /// Detail levels draw different numbers of samples for bark, shoots and
    /// twigs. Giving each of those calls its own stream keeps the whorls and
    /// limbs in the same places at every level, so trees keep their shape
    /// when they switch level.
    fn fork(&mut self) -> Rng {
        self.next();
        Rng(self.0.wrapping_mul(0x2c1b_3c6d) ^ 0x297a_2d39)
    }
}

/// Sampling parameters for one level of detail.
///
/// Every level draws the same branchlets. Finer levels draw each one with
/// its twigs; coarser levels merge neighbors on one side of a limb into a
/// single wider spray.
struct Level {
    /// Neighboring branchlets on one side of a limb merged into one spray.
    group: usize,
    /// Width of one spray.
    width: f32,
    /// Straight pieces along each spray.
    bends: usize,
    /// Twigs and needle tufts along each branchlet.
    twiglets: bool,
    /// Needles along the limb axis itself.
    clothe: bool,
    /// Short interior shoots on the stem, which only matter up close.
    interior: bool,
    /// Scale applied to the leader and limb-tip sprays.
    grow: f32,
    /// Number of bark strips around the trunk.
    strips: u32,
    trunk_step: f32,
    wood: bool,
}

const LEVELS: [Level; 4] = [
    Level {
        group: 1,
        width: 0.014,
        bends: 2,
        twiglets: true,
        clothe: true,
        interior: true,
        grow: 1.0,
        strips: 6,
        trunk_step: 0.22,
        wood: true,
    },
    Level {
        group: 1,
        width: 0.05,
        bends: 1,
        twiglets: false,
        clothe: true,
        interior: true,
        grow: 1.75,
        strips: 4,
        trunk_step: 0.45,
        wood: true,
    },
    Level {
        group: 2,
        width: 0.11,
        bends: 1,
        twiglets: false,
        clothe: false,
        interior: true,
        grow: 3.0,
        strips: 3,
        trunk_step: 1.1,
        wood: false,
    },
    Level {
        group: 4,
        width: 0.16,
        bends: 1,
        twiglets: false,
        clothe: false,
        interior: false,
        grow: 4.6,
        strips: 2,
        trunk_step: 2.2,
        wood: false,
    },
];

pub const LEVEL_COUNT: usize = LEVELS.len();

/// Growth habit of one species.
#[derive(Clone, Copy)]
pub struct Habit {
    /// Limb angle above horizontal, in radians, near the leader, and how
    /// much it drops toward the base of the crown.
    climb_top: f32,
    climb_drop: f32,
    /// How far a limb bends down along its length, in radians.
    sag: f32,
    /// Angle of the limb tip relative to the limb base, in radians. Positive
    /// tips turn up; negative tips hang.
    tip: f32,
    /// How steeply the branchlets hang below their limb. Zero keeps them
    /// level; one drops them into a vertical curtain.
    hang: f32,
    old: Vec3,
    young: Vec3,
    /// Sideways bend of the leader in meters.
    nod: f32,
    /// Exponent of the crown profile. Lower values give a columnar spire.
    profile: f32,
}

/// Grand and Douglas-fir: limbs droop from the trunk and sweep up at the
/// tips, with curtains of branchlets on the older lower limbs.
pub const FIR: Habit = Habit {
    climb_top: 0.5,
    climb_drop: 0.95,
    sag: 0.62,
    tip: 0.78,
    hang: 0.5,
    old: Vec3::new(0.050, 0.082, 0.056),
    young: Vec3::new(0.105, 0.145, 0.068),
    nod: 0.0,
    profile: 0.92,
};

/// Western hemlock: soft limbs that arch down, hanging branchlet sprays,
/// a nodding leader, and yellower green foliage.
pub const HEMLOCK: Habit = Habit {
    climb_top: 0.42,
    climb_drop: 0.72,
    sag: 0.72,
    tip: -0.12,
    hang: 0.9,
    old: Vec3::new(0.056, 0.090, 0.046),
    young: Vec3::new(0.125, 0.165, 0.062),
    nod: 0.55,
    profile: 0.92,
};

/// Subalpine fir: a very narrow spire with short, stiff limbs, a drooping
/// lower skirt, and grey blue-green needles. Build it with a crown radius
/// about half that of a grand fir of the same height.
pub const SUBALPINE: Habit = Habit {
    climb_top: 0.55,
    climb_drop: 1.1,
    sag: 0.3,
    tip: 0.6,
    hang: 0.18,
    old: Vec3::new(0.046, 0.074, 0.068),
    young: Vec3::new(0.088, 0.132, 0.110),
    nod: 0.0,
    profile: 0.72,
};

/// Needles that died this season: rusty above, grey-brown when old.
const DEAD_OLD: Vec3 = Vec3::new(0.200, 0.105, 0.052);
const DEAD_YOUNG: Vec3 = Vec3::new(0.330, 0.150, 0.055);

/// Age and damage of one tree. [`HEALTHY`] is a vigorous tree.
#[derive(Clone, Copy)]
pub struct Form {
    /// Clear lower trunk as a fraction of height; zero picks a young bole.
    pub bole: f32,
    /// Extra fraction of limbs that are missing.
    pub thin: f32,
    /// Fraction of height above which the needles are dead.
    pub dead_above: f32,
    /// Fraction of height where the stem has snapped off.
    pub broken_at: f32,
    /// Strength of wind flagging: the upper crown keeps limbs only on its
    /// leeward side, local +X.
    pub flag: f32,
    /// Trunk radius multiplier.
    pub girth: f32,
}

pub const HEALTHY: Form = Form {
    bole: 0.0,
    thin: 0.0,
    dead_above: 2.0,
    broken_at: 1.0,
    flag: 0.0,
    girth: 1.0,
};

impl Habit {
    /// Offset of the stem axis at height `y`; only the top bends.
    fn bend(&self, y: f32, height: f32, toward: Vec3) -> Vec3 {
        let t = ((y / height - 0.82) / 0.18).max(0.0);
        toward * (self.nod(height) * t * t)
    }

    /// Small trees nod less, in proportion to their height.
    fn nod(&self, height: f32) -> f32 {
        self.nod * (height / 8.0).min(1.0)
    }
}

#[derive(Clone, Copy)]
pub struct Stem {
    pub height: f32,
    pub top: f32,
    pub bole: f32,
    pub radius: f32,
}

impl Stem {
    pub fn for_tree(seed: u32, height: f32, form: &Form) -> Self {
        Self::new(&mut Rng(seed.wrapping_mul(7919)), height, form)
    }

    fn new(rng: &mut Rng, height: f32, form: &Form) -> Self {
        let bole = if form.bole > 0.0 {
            height * form.bole * rng.range(0.85, 1.15)
        } else {
            height * rng.range(0.07, 0.12)
        };
        Self {
            height,
            top: height * form.broken_at.min(1.0),
            bole,
            radius: (0.018 * height + 0.03) * form.girth,
        }
    }

    pub fn end(self) -> f32 {
        if self.top < self.height {
            self.top
        } else {
            0.92 * self.height
        }
    }

    pub fn radius_at(self, y: f32) -> f32 {
        let y = y.max(0.0);
        let flare = 1.0 + 0.7 * (-y / 0.35).exp();
        self.radius * (1.0 - y / self.height).max(0.06) * flare
    }
}

pub struct Fir {
    pub splats: Vec<GaussianSegment>,
    /// The stem appears first, so a mesh can supply it in the main view.
    pub trunk_end: usize,
}

/// Builds one fir rooted at the origin with its leader at `height`.
pub fn fir(
    seed: u32,
    habit: &Habit,
    form: &Form,
    height: f32,
    crown_radius: f32,
    level: usize,
) -> Fir {
    let detail = &LEVELS[level];
    let mut rng = Rng(seed.wrapping_mul(7919));
    let mut out = Vec::new();
    let stem = Stem::new(&mut rng, height, form);
    let Stem {
        bole: base,
        radius: trunk_radius,
        top,
        ..
    } = stem;
    let dead = Habit {
        old: DEAD_OLD,
        young: DEAD_YOUNG,
        ..*habit
    };

    trunk(&mut out, &mut rng.fork(), stem, detail);
    if form.broken_at < 1.0 {
        splinters(&mut out, &mut rng.fork(), height, top, trunk_radius);
    }
    let trunk_end = out.len();

    // Annual whorls get closer together near the leader, where growth is
    // recent and the shoots are short.
    let lean = rng.range(0.0, std::f32::consts::TAU);
    let toward = Vec3::new(lean.cos(), 0.0, lean.sin());
    // Neighbors and old damage push the crown out of round: two broad
    // lobes make one side fuller than the other.
    let lobes = [
        rng.range(0.0, std::f32::consts::TAU),
        rng.range(0.0, std::f32::consts::TAU),
    ];
    let lobe = rng.range(0.06, 0.15);
    let mut y = base;
    let mut whorl = 0u32;
    while y < top - 0.25 {
        let rel = ((y - base) / (height - base)).clamp(0.0, 1.0);
        let reach = crown_radius * (1.0 - rel).powf(habit.profile) * rng.range(0.8, 1.12) + 0.12;
        let foliage = if y / height > form.dead_above {
            &dead
        } else {
            habit
        };
        let thin = form.thin
            + if y / height > form.dead_above {
                0.3
            } else {
                0.0
            };
        // Wind and rime kill the exposed upper limbs on the windward side.
        let exposure = ((rel - 0.25) / 0.75).clamp(0.0, 1.0) * form.flag;
        let count = 5 + (rng.next() * 4.0) as u32;
        let turn = whorl as f32 * 2.399 + rng.range(0.0, 0.6);
        // Crowding and old storm damage leave short or missing limbs, and
        // one side of the crown usually reaches further toward the light.
        let vigor = rng.range(0.8, 1.14);
        for branch in 0..count {
            if rng.next() < 0.13 + thin {
                continue;
            }
            let azimuth = turn
                + branch as f32 * std::f32::consts::TAU / count as f32
                + rng.range(-0.35, 0.35);
            if azimuth.cos() < -1.0 + 1.6 * exposure {
                continue;
            }
            let toward_light = 1.0
                + 0.12 * (azimuth - lean).cos()
                + lobe * (azimuth + lobes[0]).sin()
                + 0.6 * lobe * (2.0 * azimuth + lobes[1]).sin();
            let stunted = if rng.next() < 0.16 {
                rng.range(0.4, 0.7)
            } else {
                1.0
            };
            let length = reach * rng.range(0.7, 1.1) * vigor * toward_light * stunted;
            let length = length * (1.0 + 0.3 * exposure * azimuth.cos());
            limb(
                &mut out,
                &mut rng.fork(),
                foliage,
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
        // A poor growing year leaves a short internode; a good one, a long
        // gap between whorls.
        let step = step
            * if rng.next() < 0.14 {
                rng.range(1.4, 1.75)
            } else {
                rng.range(0.7, 1.2)
            };
        for _ in 0..3 {
            let at = y + step * rng.range(0.25, 0.8);
            let length = reach * rng.range(0.3, 0.62);
            let azimuth = rng.range(0.0, std::f32::consts::TAU);
            if at > top - 0.2 || rng.next() < thin || azimuth.cos() < -1.0 + 1.6 * exposure {
                continue;
            }
            limb(
                &mut out,
                &mut rng.fork(),
                foliage,
                habit.bend(at, height, toward),
                at,
                azimuth,
                length,
                rel,
                height,
                detail,
            );
        }
        // Short interior shoots on the stem fill the inner crown, so the
        // trunk hides behind foliage above the clear bole.
        for _ in 0..2 {
            let at = y + step * rng.range(0.0, 0.9);
            let length = (reach * rng.range(0.18, 0.3)).min(0.6) + 0.15;
            let azimuth = rng.range(0.0, std::f32::consts::TAU);
            let mut stream = rng.fork();
            if at > top - 0.3 || rng.next() < thin || !detail.interior {
                continue;
            }
            limb(
                &mut out,
                &mut stream,
                foliage,
                habit.bend(at, height, toward),
                at,
                azimuth,
                length,
                rel,
                height,
                detail,
            );
        }
        y += step;
        whorl += 1;
    }

    let crest = if form.dead_above < 1.0 { &dead } else { habit };
    if form.broken_at < 1.0 {
        // A side limb below the break turns up to become a new leader.
        let from = Vec3::Y * (top - 0.35) + toward * trunk_radius;
        let to = from + (toward * 0.35 + Vec3::Y).normalize() * 0.9;
        let color = needle_color(&mut rng, crest, 1.0, 1.0);
        out.push(
            spray(
                from,
                to,
                toward.cross(Vec3::Y),
                0.05 * detail.grow.min(2.0),
                color,
            )
            .with_cutout_taper(0.32)
            .with_bend_heights(0.0, 0.45),
        );
    } else {
        // The leader is a short upright spray above the last whorl.
        let leader_from = Vec3::Y * (height - 0.55) + habit.bend(height - 0.55, height, toward);
        let top = Vec3::Y * height + habit.bend(height, height, toward) * 1.4
            - Vec3::Y * habit.nod(height) * 0.4;
        let color = needle_color(&mut rng, crest, 1.0, 1.0);
        out.push(
            spray(
                leader_from,
                top,
                toward.cross(Vec3::Y),
                0.05 * detail.grow.min(2.0),
                color,
            )
            .with_cutout_taper(0.32)
            .with_bend_heights(0.0, 0.55),
        );
    }
    if level >= 2 {
        // A few dead stubs below the crown would vanish at this size anyway.
        return Fir {
            splats: out,
            trunk_end,
        };
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
        out.push(
            GaussianSegment::new(
                start,
                end,
                Vec3::splat(width),
                Quat::from_rotation_arc(
                    Vec3::Z,
                    Vec3::new(direction.x, 0.6, direction.z).normalize(),
                ),
                5.0 * length * width,
                Vec3::new(0.085, 0.075, 0.065),
            )
            .with_cutout_taper(0.32),
        );
        y += rng.range(0.12, 0.35);
    }
    Fir {
        splats: out,
        trunk_end,
    }
}

fn trunk(out: &mut Vec<GaussianSegment>, rng: &mut Rng, stem: Stem, detail: &Level) {
    let mut y = 0.0;
    let end = stem.end();
    while y < end {
        let step = detail.trunk_step.min(end - y + 0.05);
        let r0 = stem.radius_at(y);
        let r1 = stem.radius_at(y + step);
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
            let shade = if y > stem.bole { 0.32 } else { 1.0 };
            let tone = (0.75 + 0.5 * rng.next()) * shade;
            let ridge = Vec3::new(0.115, 0.105, 0.095);
            let groove = Vec3::new(0.060, 0.050, 0.042);
            let color = ridge.lerp(groove, fissure) * tone;
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

/// Jagged spikes of wood where the stem snapped.
fn splinters(out: &mut Vec<GaussianSegment>, rng: &mut Rng, height: f32, top: f32, radius: f32) {
    let r = radius * (1.0 - top / height).max(0.06);
    let color = Vec3::new(0.20, 0.16, 0.11);
    for _ in 0..3 {
        let angle = rng.range(0.0, std::f32::consts::TAU);
        let outward = Vec3::new(angle.cos(), 0.0, angle.sin());
        let start = Vec3::Y * (top - 0.1) + outward * r * 0.4;
        let end =
            start + (Vec3::Y + outward * rng.range(0.05, 0.3)).normalize() * rng.range(0.25, 0.9);
        let width = r * rng.range(0.35, 0.6);
        out.push(GaussianSegment::new(
            start,
            end,
            Vec3::new(width, width, width * 0.7),
            Quat::from_rotation_arc(Vec3::Z, outward),
            5.0 * width * start.distance(end),
            color * rng.range(0.8, 1.15),
        ));
    }
}

/// Number of straight pieces in a limb's center line.
const LIMB_NODES: usize = 10;

/// One side branch of a limb, bent down by its own weight.
struct Branchlet {
    base: Vec3,
    mid: Vec3,
    tip: Vec3,
    /// Faces the open side of the needle spray.
    normal: Vec3,
    /// Fraction of the way from trunk to limb tip.
    s: f32,
    sign: f32,
    hang: f32,
}

impl Branchlet {
    /// Point on the curve through base, mid and tip.
    fn at(&self, t: f32) -> Vec3 {
        let a = self.base.lerp(self.mid, t);
        let b = self.mid.lerp(self.tip, t);
        a.lerp(b, t)
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
    let first = out.len();
    let outward = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
    let side = Vec3::Y.cross(outward).normalize();
    // The limb leaves the trunk at `start` and bends down along its length
    // under its weight. Near the tip new growth turns up toward the light,
    // or, in hemlock, keeps arching down. Bending by angle keeps the limb's
    // length fixed however far it droops.
    let start = habit.climb_top - habit.climb_drop * (1.0 - rel) + rng.range(-0.12, 0.12);
    let sag = habit.sag
        * (1.0 - 0.55 * rel)
        * (0.6 + 0.4 * (length / 2.5).min(1.4))
        * rng.range(0.8, 1.2);
    // Young upper limbs already climb, so only older limbs sweep up.
    // Short limbs bend less either way.
    let tip = habit.tip
        * (1.0 - rel).powf(0.7)
        * (0.45 + 0.55 * (length / 2.0).min(1.0))
        * rng.range(0.75, 1.25);
    let mut nodes = [Vec3::ZERO; LIMB_NODES + 1];
    nodes[0] = shift + Vec3::Y * y + outward * 0.05;
    for node in 0..LIMB_NODES {
        let s = (node as f32 + 0.5) / LIMB_NODES as f32;
        let turn = ((s - 0.45) / 0.55).clamp(0.0, 1.0);
        let angle = start - sag * s + (sag + tip) * turn * turn * (3.0 - 2.0 * turn);
        let direction = outward * angle.cos() + Vec3::Y * angle.sin();
        nodes[node + 1] = nodes[node] + direction * (length * 1.06 / LIMB_NODES as f32);
    }
    let point = |s: f32| {
        let at = s.clamp(0.0, 1.0) * LIMB_NODES as f32;
        let node = (at as usize).min(LIMB_NODES - 1);
        nodes[node].lerp(nodes[node + 1], at - node as f32)
    };
    let tangent = |s: f32| {
        let node = ((s.clamp(0.0, 1.0) * LIMB_NODES as f32) as usize).min(LIMB_NODES - 1);
        (nodes[node + 1] - nodes[node]).normalize_or(outward)
    };
    let shade_scale = 1.0 - 0.35 * (1.0 - rel);

    // Branchlets alternate along the limb, longest near the trunk. They
    // fill the inner limb too, so the crown hides the trunk. This layout
    // draws the same samples at every level.
    let mut branchlets = Vec::new();
    let gap = 0.085 * rng.range(0.85, 1.15);
    let mut s = 0.05;
    let mut sign = if rng.next() > 0.5 { 1.0f32 } else { -1.0 };
    let reach = (length / 2.0).clamp(0.4, 1.0);
    while s < 0.96 {
        let here = point(s);
        let along = tangent(s);
        let skip = rng.next() < (1.0 - s / 0.3).max(0.0) * 0.3;
        let size = (0.1 + 0.46 * (1.0 - s).powf(0.8)) * reach * rng.range(0.75, 1.2);
        let angle = rng.range(0.85, 1.2);
        let hang = habit.hang
            * (0.35 + 0.65 * (1.0 - rel))
            * (0.4 + 0.6 * (s / 0.6).min(1.0))
            * rng.range(0.7, 1.3);
        let roll = rng.range(-0.35, 0.35);
        let flick = rng.range(0.0, 0.2) * habit.tip.max(0.0);
        if !skip {
            let lateral = (along * angle.cos() + side * sign * angle.sin()).normalize();
            let mid = here + (lateral - Vec3::Y * hang * 0.15).normalize() * size * 0.5;
            let bend = (lateral * 0.8 + along * 0.25 - Vec3::Y * hang * 1.35 + Vec3::Y * flick)
                .normalize();
            let tip = mid + bend * size * 0.5;
            let chord = (tip - here).normalize_or(lateral);
            let mut normal = along.cross(chord).normalize_or(Vec3::Y);
            if normal.y < 0.0 {
                normal = -normal;
            }
            let normal = (Quat::from_axis_angle(chord, roll) * normal + outward * 0.25).normalize();
            branchlets.push(Branchlet {
                base: here,
                mid,
                tip,
                normal,
                s,
                sign,
                hang,
            });
        }
        sign = -sign;
        s += gap / length.max(0.3) * rng.range(0.8, 1.2);
    }

    // Everything below varies by level, so it may draw any number of
    // samples from `rng`.
    if detail.wood {
        // The outer limb is a thin twig hidden in its own needles.
        let links = 4;
        for link in 0..links {
            let s0 = 0.8 * link as f32 / links as f32;
            let s1 = 0.8 * (link + 1) as f32 / links as f32;
            let width = 0.005 + 0.018 * (1.0 - s0) * (length / 2.5).min(1.0);
            out.push(GaussianSegment::new(
                point(s0),
                point(s1),
                Vec3::splat(width),
                Quat::from_rotation_arc(Vec3::Z, Vec3::Y),
                4.0 * width * length / links as f32,
                Vec3::new(0.055, 0.042, 0.032),
            ));
        }
    }

    let brightness = |b: &Branchlet| {
        // Light reaches the outer shell; the interior and the undersides of
        // hanging sprays stay dark.
        let depth = (0.42 + 0.58 * b.s.powf(0.7)) * (1.0 - 0.18 * b.hang.min(1.0));
        depth * shade_scale + 0.12 * (y / height)
    };
    let growth = |b: &Branchlet| (b.s - 0.72).max(0.0) * 2.5;

    if detail.twiglets {
        for b in &branchlets {
            let light = brightness(b);
            let size = b.base.distance(b.tip);
            let width = |t: f32| detail.width * (1.0 - 0.58 * t);
            for bend in 0..detail.bends {
                let t0 = bend as f32 / detail.bends as f32;
                let t1 = (bend + 1) as f32 / detail.bends as f32;
                let color = needle_color(rng, habit, light, growth(b)) * 0.8;
                out.push(
                    frond(b.at(t0), b.at(t1), b.normal, width(t0), color)
                        .with_cutout_taper(width(t1) / width(t0)),
                );
            }
            // Stagger the twig roots and roll each spray independently.
            let mut t = 0.08;
            while t < 1.0 {
                let at = b.at(t);
                let axis =
                    (b.at((t + 0.05).min(1.0)) - b.at((t - 0.05).max(0.0))).normalize_or(b.normal);
                let across = b.normal.cross(axis).normalize_or(side);
                let twig_reach = 0.1 * (1.0 - 0.45 * t) * reach.sqrt() * rng.range(0.75, 1.2);
                for lean in [-1.0f32, 1.0] {
                    let roll = rng.range(-0.6, 0.6);
                    let root = b.at((t + lean * 0.018).clamp(0.0, 1.0));
                    let twig = (axis * (0.72 + roll * 0.3) + across * lean * 0.72
                        - Vec3::Y * (b.hang * 0.25 - roll * 0.2))
                        .normalize();
                    let normal = Quat::from_axis_angle(twig, roll) * b.normal;
                    let tint = needle_color(rng, habit, light, growth(b) + t * 0.3);
                    let end = root + twig * twig_reach * (1.0 + roll * 0.32);
                    out.push(
                        frond(root, end, normal, 0.019 * (1.0 - 0.20 * t), tint)
                            .with_cutout_taper(0.28),
                    );
                }
                if rng.next() < 0.22 {
                    // A tuft of needles standing up off the twig.
                    let up = (b.normal * 0.8 + axis * 0.5).normalize();
                    let tint = needle_color(rng, habit, light * 1.05, growth(b) + t * 0.4);
                    out.push(
                        frond(at, at + up * twig_reach * 0.6, across, 0.015, tint)
                            .with_cutout_taper(0.4),
                    );
                }
                t += 0.06 / size.max(0.05) * rng.range(0.85, 1.15);
            }
        }
    } else {
        // Merge each run of neighbors on one side into one spray from the
        // first base out to their mean tip.
        for flip in [-1.0f32, 1.0] {
            let one_side: Vec<&Branchlet> = branchlets.iter().filter(|b| b.sign == flip).collect();
            for run in one_side.chunks(detail.group) {
                let n = run.len() as f32;
                let base = run[0].base;
                let tip = run.iter().map(|b| b.tip).sum::<Vec3>() / n;
                let normal = run
                    .iter()
                    .map(|b| b.normal)
                    .sum::<Vec3>()
                    .normalize_or(Vec3::Y);
                let lead = run[run.len() / 2];
                let light = brightness(lead);
                // Wider runs cover the gaps between the merged branchlets.
                let width = detail.width * (0.75 + 0.25 * n);
                let color = needle_color(rng, habit, light, growth(lead));
                let (width, taper) = if detail.group == 1 {
                    (width * 1.14, 0.52)
                } else {
                    (width, 1.0)
                };
                out.push(frond(base, tip, normal, width, color).with_cutout_taper(taper));
            }
        }
    }
    // Short sprays alternate off the top of the limb. The wood can show
    // through between them without drawing a continuous strip of foliage.
    if detail.clothe {
        let links = (length / 0.25).ceil().max(2.0) as u32;
        let width = |s: f32| 0.028 * detail.grow.sqrt() * (1.0 - 0.65 * s.powf(1.7));
        for link in 0..links {
            let s0 = link as f32 / links as f32;
            let s1 = (link + 1) as f32 / links as f32;
            if s1 < 0.15 {
                continue;
            }
            let root = point(s0);
            let axis = point(s1) - root;
            let normal = axis.cross(side).normalize_or(Vec3::Y);
            let normal = (normal + outward * 0.25).normalize();
            let color = needle_color(
                rng,
                habit,
                (0.42 + 0.58 * s1) * shade_scale,
                (s1 - 0.72).max(0.0) * 2.5,
            );
            let lean = if link % 2 == 0 { 0.35 } else { -0.35 };
            let reach = axis.length();
            let lift = normal * 0.02 * detail.grow.sqrt();
            out.push(
                frond(
                    root + lift,
                    root + axis * 0.81 + side * (reach * lean) + lift * 1.5,
                    normal,
                    width(s0),
                    color,
                )
                .with_cutout_taper(0.36),
            );
        }
    }
    // A terminal bud spray continues the limb past the last pair.
    let end = point(1.0);
    let along = (tangent(1.0) + outward * 0.6).normalize();
    // Keep it narrow and only slightly lighter, or the tips read as dots.
    let color = needle_color(rng, habit, shade_scale, 0.45);
    let tip = if detail.clothe {
        spray(
            end,
            end + along * 0.19 * detail.grow.sqrt(),
            side,
            0.023 * detail.grow,
            color,
        )
        .with_cutout_taper(0.23)
    } else {
        spray(
            end,
            end + along * 0.26 * detail.grow.sqrt(),
            side,
            0.028 * detail.grow,
            color,
        )
    };
    out.push(tip);

    // Cantilever deflection under a tip load, with the root pinned and
    // smaller shoots inheriting the motion of their attachment point.
    let tip_flex = length * (0.52 + 0.36 * rel);
    let flex = |position: Vec3| {
        let u = ((position - nodes[0]).dot(outward) / length).clamp(0.0, 1.0);
        tip_flex * (u * u * (1.5 - 0.5 * u))
    };
    for segment in &mut out[first..] {
        *segment = segment.with_bend_heights(flex(segment.start()), flex(segment.end()));
    }
}

/// A flat spray of needles along `start` → `end`, facing `normal`.
fn frond(start: Vec3, end: Vec3, normal: Vec3, width: f32, color: Vec3) -> GaussianSegment {
    let along = (end - start).normalize_or(Vec3::X);
    let across = normal
        .cross(along)
        .try_normalize()
        .unwrap_or_else(|| along.any_orthonormal_vector());
    let normal = along.cross(across);
    let rotation = Quat::from_mat3(&Mat3::from_cols(along, across, normal)).normalize();
    let length = start.distance(end);
    GaussianSegment::new(
        start,
        end,
        Vec3::new(width * 0.7, width, width * 0.22),
        rotation,
        3.4 * length * width,
        color,
    )
}

fn spray(start: Vec3, end: Vec3, side: Vec3, width: f32, color: Vec3) -> GaussianSegment {
    let along = (end - start).normalize_or(Vec3::Y);
    let normal = side.cross(along).normalize_or(Vec3::Z);
    frond(start, end, normal, width, color)
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
