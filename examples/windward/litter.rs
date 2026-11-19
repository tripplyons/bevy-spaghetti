//! Forest floor litter built from Gaussian cutouts.
//!
//! Fallen cones, dead boughs and flakes of bark lie under each stand
//! tree. Moss cushions and small groups of mushrooms grow near the trunks.
//! Each object has separate scales, needles, fronds or cap facets up close,
//! and merges into one or a few kernels with distance.

use std::{collections::BTreeMap, f32::consts::TAU};

use bevy::{camera::visibility::VisibilityRange, prelude::*};
use bevy_spaghetti::{
    DetailTransition, GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiCutout,
    SpaghettiLighting, SpaghettiReceiveShadows, SpaghettiSpecular, SpaghettiTransmission,
};

use crate::{
    cutout::{blob, disc, rod, strap, tapered},
    flowers::Rng,
    forest::forest,
    landscape::{ROCKS, SUN, WATER_LEVEL, ground, trail_x},
    props::PropFootprints,
};

const CELL: f32 = 10.0;
/// Projected pixels per meter between which the fine level hands over to
/// the coarse one. A cone scale is a centimeter wide, so the scales merge
/// near 13 m at 800 lines.
const HANDOVER: [f32; 2] = [50.0, 68.0];
const RANGE: VisibilityRange = VisibilityRange {
    start_margin: 0.0..0.0,
    end_margin: 44.0..54.0,
    use_aabb: true,
};
const GOLDEN_ANGLE: f32 = 2.399_963;

/// The gains of the path gravel, so the litter sits in the same light.
const LIGHTING: SpaghettiLighting = SpaghettiLighting {
    direction_to_light: SUN,
    ambient: Vec3::new(0.33, 0.37, 0.42),
    directional: Vec3::new(0.86, 0.73, 0.56),
};

/// Stands in for the shadow that a small object casts on the duff under it.
const CONTACT: Vec3 = Vec3::new(0.016, 0.012, 0.009);

/// A fine and a coarse version of the same objects.
#[derive(Default)]
struct Levels {
    fine: Vec<GaussianSegment>,
    coarse: Vec<GaussianSegment>,
}

impl Levels {
    fn into_asset(mut self) -> Option<SpaghettiAsset> {
        if self.fine.is_empty() || self.coarse.is_empty() {
            self.fine.append(&mut self.coarse);
            return (!self.fine.is_empty())
                .then(|| SpaghettiAsset::new(self.fine).expect("valid litter"));
        }
        let finer_end = self.fine.len() as u32;
        self.fine.append(&mut self.coarse);
        let transition = DetailTransition {
            finer_end,
            pixels_per_unit: HANDOVER,
        };
        Some(SpaghettiAsset::with_detail(self.fine, &[transition]).expect("valid litter"))
    }
}

/// One cell of the floor. Dry litter and living growth draw as separate
/// clouds, because only the growth passes light and has a moist sheen.
#[derive(Default)]
struct Patch {
    litter: Levels,
    growth: Levels,
}

struct Floor<'a> {
    patches: BTreeMap<(i32, i32), Patch>,
    props: &'a PropFootprints,
}

impl Floor<'_> {
    /// The patch that holds `at`, with the ground point and its normal in
    /// patch coordinates. `gap` is the room the object needs around it.
    fn site(&mut self, at: Vec2, gap: f32) -> Option<(&mut Patch, Vec3, Vec3)> {
        let clear = (at.x - trail_x(at.y)).abs() > 1.5 + gap
            && ground(at) > WATER_LEVEL + 0.25
            && self.props.keep(at) > 0.95
            && ROCKS
                .iter()
                .all(|&(x, z, scale)| Vec2::new(x, z).distance(at) > 1.15 * scale + gap)
            && forest()
                .trees
                .iter()
                .all(|tree| tree.at.distance(at) > 0.27 * tree.scale + gap);
        if !clear {
            return None;
        }
        let cell = (at / CELL).floor().as_ivec2();
        let origin = cell.as_vec2() * CELL;
        let step = 0.05;
        let slope = Vec2::new(
            ground(at + Vec2::X * step) - ground(at - Vec2::X * step),
            ground(at + Vec2::Y * step) - ground(at - Vec2::Y * step),
        );
        let normal = Vec3::new(-slope.x, 2.0 * step, -slope.y).normalize();
        let base = Vec3::new(at.x - origin.x, ground(at), at.y - origin.y);
        Some((
            self.patches.entry((cell.x, cell.y)).or_default(),
            base,
            normal,
        ))
    }
}

pub fn spawn_litter(
    mut commands: Commands,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
    props: Option<Res<PropFootprints>>,
) {
    let empty = PropFootprints::default();
    let mut floor = Floor {
        patches: BTreeMap::new(),
        props: props.as_deref().unwrap_or(&empty),
    };
    for (index, tree) in forest()
        .trees
        .iter()
        .enumerate()
        .filter(|(_, tree)| tree.is_stand())
    {
        let mut rng = Rng::new(71_003 + 257 * index as u32);
        let reach = tree.crown().max(0.9);
        let trunk = 0.27 * tree.scale;
        let spot = |rng: &mut Rng, near: f32, far: f32| {
            tree.at + Vec2::from_angle(rng.next() * TAU) * rng.range(near, far)
        };

        for _ in 0..(4.0 + 7.0 * rng.next()) as usize {
            let at = spot(&mut rng, trunk + 0.15, 1.15 * reach);
            let mut rng = rng.fork();
            if let Some((patch, base, up)) = floor.site(at, 0.05) {
                cone(&mut patch.litter, base, up, &mut rng);
            }
        }
        for _ in 0..(2.0 + 3.6 * rng.next()) as usize {
            let at = spot(&mut rng, trunk + 0.2, 1.25 * reach);
            let mut rng = rng.fork();
            if let Some((patch, base, up)) = floor.site(at, 0.25) {
                bough(&mut patch.litter, base, up, &mut rng);
            }
        }
        for _ in 0..32 {
            let at = spot(&mut rng, trunk + 0.05, 1.3 * reach);
            let mut rng = rng.fork();
            if let Some((patch, base, up)) = floor.site(at, 0.02) {
                flake(&mut patch.litter.fine, base, up, &mut rng);
            }
        }
        // Moss keeps to the damp ground at the foot of the trunk.
        let cushions = (rng.next() * 3.4) as usize;
        for _ in 0..cushions {
            let at = spot(&mut rng, trunk + 0.3, trunk + 0.3 + 0.55 * reach);
            let mut rng = rng.fork();
            if let Some((patch, base, up)) = floor.site(at, 0.28) {
                moss(&mut patch.growth, base, up, &mut rng);
            }
        }
        if rng.next() < 0.42 {
            let at = spot(&mut rng, trunk + 0.25, trunk + 1.1);
            let mut rng = rng.fork();
            let kind = KINDS[(rng.next().powf(1.6) * KINDS.len() as f32) as usize];
            for _ in 0..(1.0 + 4.6 * rng.next()) as usize {
                let at = at + Vec2::from_angle(rng.next() * TAU) * rng.range(0.03, 0.17);
                let mut rng = rng.fork();
                if let Some((patch, base, up)) = floor.site(at, 0.05) {
                    mushroom(&mut patch.growth, base, up, &kind, &mut rng);
                }
            }
        }
    }

    for ((x, z), patch) in floor.patches {
        let place = Transform::from_xyz(x as f32 * CELL, 0.0, z as f32 * CELL);
        if let Some(asset) = patch.litter.into_asset() {
            commands.spawn((
                Name::new("Forest litter"),
                Spaghetti3d(clouds.add(asset)),
                SpaghettiCutout,
                LIGHTING,
                SpaghettiReceiveShadows,
                place,
                RANGE,
            ));
        }
        if let Some(asset) = patch.growth.into_asset() {
            commands.spawn((
                Name::new("Moss and mushrooms"),
                Spaghetti3d(clouds.add(asset)),
                SpaghettiCutout,
                LIGHTING,
                SpaghettiReceiveShadows,
                // Moss fronds and thin caps glow when the low sun is behind them.
                SpaghettiTransmission(0.4),
                SpaghettiSpecular {
                    strength: 0.06,
                    shininess: 30.0,
                    strand: 0.0,
                },
                place,
                RANGE,
            ));
        }
    }
}

/// A fallen pine cone: a dark core under spiral rows of woody scales.
fn cone(out: &mut Levels, base: Vec3, up: Vec3, rng: &mut Rng) {
    let length = rng.range(0.10, 0.19);
    let radius = length * rng.range(0.19, 0.24);
    let heading = rng.next() * TAU;
    let flat = Vec3::new(heading.cos(), 0.0, heading.sin());
    let axis = (flat - up * flat.dot(up)).normalize();
    let side = up.cross(axis);
    let center = base + up * (0.8 * radius);
    // Fresh cones are brown. Old ones weather to pale gray.
    let brown = Vec3::new(0.17, 0.10, 0.05).lerp(Vec3::new(0.25, 0.21, 0.16), rng.next().powi(2));
    let (tail, tip) = (
        center - axis * (0.40 * length),
        center + axis * (0.40 * length),
    );

    let ground = up * (0.003 - 0.8 * radius);
    strap(
        &mut out.fine,
        tail + ground,
        tip + ground,
        up,
        2.8 * radius,
        CONTACT,
    );
    rod(&mut out.fine, tail, tip, up, 0.6 * radius, brown * 0.12);
    let scales = 46;
    for index in 0..scales {
        let t = (index as f32 + 0.5) / scales as f32;
        let turn = index as f32 * GOLDEN_ANGLE;
        let outward = side * turn.cos() + up * turn.sin();
        let tone = rng.range(0.6, 1.25);
        // The scales under the cone are out of sight.
        if outward.dot(up) < -0.4 {
            continue;
        }
        let swell = (4.0 * t * (1.0 - t)).powf(0.45);
        let at =
            center + axis * ((t - 0.5) * 0.92 * length) + outward * (radius * (0.25 + 0.6 * swell));
        disc(
            &mut out.fine,
            at,
            outward + axis * 0.5,
            radius * (0.30 + 0.20 * swell),
            brown * tone,
        );
    }
    rod(&mut out.coarse, tail, tip, up, 0.85 * radius, brown * 0.8);
}

/// A fallen dead bough: a forked twig that still holds its rust-colored
/// needles, like the dead limbs on the snags above.
fn bough(out: &mut Levels, base: Vec3, up: Vec3, rng: &mut Rng) {
    let length = rng.range(0.40, 0.90);
    let heading = rng.next() * TAU;
    let flat = Vec3::new(heading.cos(), 0.0, heading.sin());
    let axis = (flat - up * flat.dot(up)).normalize();
    let side = up.cross(axis);
    let rust = Vec3::new(0.62, 0.27, 0.07).lerp(Vec3::new(0.50, 0.36, 0.17), rng.next().powi(2));
    let root = base + up * 0.012 - axis * (0.5 * length);
    sprig(
        out,
        root,
        axis,
        side * rng.range(-0.14, 0.14),
        length,
        up,
        rust,
        rng,
    );
    for _ in 0..(1.0 + 2.6 * rng.next()) as usize {
        let from = rng.range(0.15, 0.6);
        let hand = if rng.next() < 0.5 { -1.0 } else { 1.0 };
        let turn = hand * rng.range(0.5, 0.9);
        let heading = axis * turn.cos() + side * turn.sin();
        sprig(
            out,
            root + axis * (from * length),
            heading,
            up.cross(heading) * rng.range(-0.1, 0.1),
            length * (1.0 - from) * rng.range(0.45, 0.7),
            up,
            rust * rng.range(0.85, 1.1),
            rng,
        );
    }
}

/// One twig of a bough with a row of needles on each side. `bow` is the
/// sideways sag at the middle as a fraction of the length.
#[allow(clippy::too_many_arguments)]
fn sprig(
    out: &mut Levels,
    root: Vec3,
    axis: Vec3,
    bow: Vec3,
    length: f32,
    up: Vec3,
    rust: Vec3,
    rng: &mut Rng,
) {
    let spine = |t: f32| root + axis * (t * length) + bow * (length * 4.0 * t * (1.0 - t));
    let parts = 3;
    for part in 0..parts {
        let (a, b) = (part as f32 / parts as f32, (part + 1) as f32 / parts as f32);
        strap(
            &mut out.fine,
            spine(a),
            spine(b),
            up,
            0.008,
            Vec3::new(0.15, 0.095, 0.055),
        );
    }
    let pairs = (length / 0.030) as usize;
    let mut widest = 0.0_f32;
    for pair in 0..pairs {
        let t = (pair as f32 + 0.6) / (pairs as f32 + 0.6);
        let at = spine(t);
        let forward = (spine(t + 0.02) - spine(t - 0.02)).normalize();
        // The needles shorten toward the tip of the twig.
        let reach = rng.range(0.035, 0.060) * (1.0 - 0.45 * t * t);
        widest = widest.max(reach);
        for hand in [-1.0, 1.0] {
            let outward = (forward * rng.range(0.45, 0.8) + up.cross(forward) * hand).normalize();
            let tip = at + outward * reach + up * (reach * rng.range(0.0, 0.55));
            // The needles twist on the twig, so a few of them catch the sun.
            let face = up + Vec3::new(rng.range(-0.9, 0.9), 0.0, rng.range(-0.9, 0.9));
            tapered(
                &mut out.fine,
                (at, tip),
                face,
                0.009,
                0.4,
                rust * rng.range(0.7, 1.2),
            );
        }
    }
    tapered(
        &mut out.coarse,
        (spine(0.0), spine(1.0)),
        up,
        1.5 * widest,
        0.35,
        rust * 0.8,
    );
}

/// A slab of bark, a loose cone scale or a wood chip, lying flat.
fn flake(out: &mut Vec<GaussianSegment>, base: Vec3, up: Vec3, rng: &mut Rng) {
    const TONES: [Vec3; 5] = [
        Vec3::new(0.27, 0.20, 0.12),
        Vec3::new(0.085, 0.055, 0.034),
        Vec3::new(0.22, 0.21, 0.18),
        Vec3::new(0.30, 0.14, 0.05),
        Vec3::new(0.30, 0.26, 0.19),
    ];
    let size = rng.range(0.010, 0.030);
    let face = (up + Vec3::new(rng.range(-0.3, 0.3), 0.0, rng.range(-0.3, 0.3))).normalize();
    let color = TONES[(rng.next() * TONES.len() as f32) as usize] * rng.range(0.7, 1.2);
    let at = base + up * 0.004;
    if rng.next() < 0.4 {
        disc(out, at, face, size, color);
    } else {
        let heading = rng.next() * TAU;
        let half = Vec3::new(heading.cos(), 0.0, heading.sin()) * (size * rng.range(0.9, 2.2));
        strap(out, at - half, at + half, face, size * 1.2, color);
    }
}

/// A moss cushion: a low dome of dark pads under pale green frond tips.
fn moss(out: &mut Levels, base: Vec3, up: Vec3, rng: &mut Rng) {
    let radius = rng.range(0.11, 0.34);
    let height = radius * rng.range(0.30, 0.50);
    let hue = rng.next();
    let deep = Vec3::new(0.030, 0.070, 0.016).lerp(Vec3::new(0.050, 0.085, 0.018), hue);
    let bright = Vec3::new(0.115, 0.200, 0.035).lerp(Vec3::new(0.200, 0.250, 0.045), hue);
    let (u, v) = up.any_orthonormal_pair();
    let phase = [rng.next() * TAU, rng.next() * TAU];
    // A point on the cushion and its normal. `rise` runs from zero at the
    // rim to one at the crown.
    let dome = |turn: f32, rise: f32| {
        let spread = (1.0 - rise * rise).sqrt();
        let lump = 1.0
            + 0.16 * spread * (3.0 * turn + phase[0]).sin()
            + 0.10 * (5.0 * turn + phase[1]).cos();
        let ring = u * turn.cos() + v * turn.sin();
        let point = base + ring * (radius * spread * lump) + up * (height * rise * lump);
        let normal = (ring * (spread / radius) + up * (rise / height)).normalize();
        (point, normal)
    };

    // A dark rim closes the gaps between the lowest pads.
    for index in 0..14 {
        let (point, _) = dome(index as f32 * TAU / 14.0, 0.0);
        let at = base.lerp(point, 0.84) + up * 0.003;
        disc(&mut out.fine, at, up, 0.16 * radius, CONTACT * 2.0);
    }
    let pads = 110;
    for index in 0..pads {
        // Equal steps in `rise` cover equal areas of the dome.
        let rise = 1.0 - (index as f32 + 0.5) / pads as f32;
        let (point, normal) = dome(index as f32 * GOLDEN_ANGLE, 0.96 * rise);
        let tone = rng.next();
        disc(
            &mut out.fine,
            point - normal * (0.05 * radius),
            normal,
            0.15 * radius,
            deep.lerp(bright, 0.4 * tone * tone) * rng.range(0.8, 1.15),
        );
    }
    let fronds = 80 + (5200.0 * radius * radius).min(520.0) as usize;
    for index in 0..fronds {
        let rise = 1.0 - (index as f32 + 0.5) / fronds as f32;
        let turn = index as f32 * GOLDEN_ANGLE + rng.range(-0.3, 0.3);
        let (point, normal) = dome(turn, rise);
        let lie = rng.next() * TAU;
        let (p, q) = normal.any_orthonormal_pair();
        let along = p * lie.cos() + q * lie.sin();
        let reach = rng.range(0.022, 0.045);
        let tip = point + along * reach + normal * (reach * rng.range(0.3, 0.8));
        let tone = rng.next();
        tapered(
            &mut out.fine,
            (point, tip),
            normal,
            rng.range(0.010, 0.016),
            0.25,
            deep.lerp(bright, 0.25 + 0.85 * tone * tone),
        );
    }

    let mean = deep.lerp(bright, 0.42);
    let (crown, normal) = dome(0.0, 1.0);
    disc(
        &mut out.coarse,
        crown - normal * (0.25 * height),
        normal,
        0.72 * radius,
        mean,
    );
    for index in 0..5 {
        let (point, normal) = dome(index as f32 * TAU / 5.0 + phase[0], 0.45);
        disc(
            &mut out.coarse,
            point - normal * (0.12 * radius),
            normal,
            0.55 * radius,
            mean * rng.range(0.85, 1.1),
        );
    }
}

/// Cap, stem and gill colors, the cap radius range, and whether the cap
/// carries pale warts.
#[derive(Clone, Copy)]
struct Kind {
    cap: Vec3,
    stem: Vec3,
    gills: Vec3,
    radius: [f32; 2],
    warts: bool,
}

/// From the most to the least common: a brown bolete, a golden chanterelle
/// and a fly agaric.
const KINDS: [Kind; 3] = [
    Kind {
        cap: Vec3::new(0.21, 0.105, 0.042),
        stem: Vec3::new(0.40, 0.34, 0.24),
        gills: Vec3::new(0.30, 0.25, 0.13),
        radius: [0.030, 0.070],
        warts: false,
    },
    Kind {
        cap: Vec3::new(0.52, 0.27, 0.035),
        stem: Vec3::new(0.50, 0.32, 0.08),
        gills: Vec3::new(0.46, 0.25, 0.04),
        radius: [0.020, 0.045],
        warts: false,
    },
    Kind {
        cap: Vec3::new(0.50, 0.032, 0.018),
        stem: Vec3::new(0.56, 0.54, 0.47),
        gills: Vec3::new(0.50, 0.48, 0.41),
        radius: [0.035, 0.075],
        warts: true,
    },
];

/// One mushroom. The cap is a crown disc and a ring of tilted facets over a
/// dark gill disc, so the sun shades it as a dome.
fn mushroom(out: &mut Levels, base: Vec3, up: Vec3, kind: &Kind, rng: &mut Rng) {
    let cap = rng.range(kind.radius[0], kind.radius[1]);
    let tall = cap * rng.range(1.0, 1.7);
    let lean = (up + Vec3::new(rng.range(-0.25, 0.25), 0.0, rng.range(-0.25, 0.25))).normalize();
    let top = base + lean * tall;
    let girth = cap * rng.range(0.22, 0.32);
    let color = kind.cap * rng.range(0.8, 1.15);
    // A kernel has one normal. A sunlit half and a shaded half make the
    // stem read as round from the side.
    let toward = Vec3::new(SUN.x, 0.0, SUN.z).normalize();
    for (shift, face) in [(0.3, toward), (-0.3, -toward)] {
        let offset = toward * (shift * girth);
        rod(
            &mut out.fine,
            base + offset,
            top + offset,
            face,
            0.8 * girth,
            kind.stem,
        );
    }
    disc(&mut out.fine, base + up * 0.003, up, 0.9 * cap, CONTACT);
    disc(
        &mut out.fine,
        top + lean * (0.02 * cap),
        -lean,
        0.93 * cap,
        kind.gills * 0.7,
    );
    let crown = top + lean * (0.42 * cap);
    disc(&mut out.fine, crown, lean, 0.66 * cap, color);
    let (u, v) = lean.any_orthonormal_pair();
    let twist = rng.next() * TAU;
    let facets = 8;
    for index in 0..facets {
        let turn = twist + TAU * index as f32 / facets as f32;
        let outward = u * turn.cos() + v * turn.sin();
        disc(
            &mut out.fine,
            top + lean * (0.20 * cap) + outward * (0.56 * cap),
            outward * 0.8 + lean * 0.6,
            0.5 * cap,
            color * rng.range(0.85, 1.1),
        );
    }
    if kind.warts {
        for index in 0..9 {
            let turn = twist + index as f32 * GOLDEN_ANGLE;
            let spread = 0.2 + 0.7 * (index as f32 / 9.0).sqrt();
            let outward = u * turn.cos() + v * turn.sin();
            let face = outward * spread + lean * (1.0 - spread * spread).sqrt();
            // The facets flatten the dome, so its height is less than its radius.
            let at = top
                + outward * (0.92 * spread * cap)
                + lean * (cap * (0.2 + 0.36 * (1.0 - spread * spread).sqrt()));
            disc(
                &mut out.fine,
                at,
                face,
                cap * rng.range(0.07, 0.11),
                Vec3::new(0.62, 0.60, 0.52),
            );
        }
    }

    rod(&mut out.coarse, base, top, toward, girth, kind.stem);
    blob(
        &mut out.coarse,
        top + lean * (0.2 * cap),
        lean,
        cap,
        0.6,
        color,
    );
}
