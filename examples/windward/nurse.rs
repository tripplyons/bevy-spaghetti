//! Growth on the fallen logs.
//!
//! Moss carpets the upper side of each log and lichen crusts dot the bare
//! bark. Bracket fungi step down the flanks, broken limbs stand out as
//! stubs, and a few conifer seedlings root in the moss. Each part has
//! separate fronds, lobes, zones or needles up close, and merges into a few
//! kernels with distance.

use std::{
    f32::consts::{PI, TAU},
    sync::LazyLock,
};

use bevy::{camera::visibility::VisibilityRange, prelude::*};
use bevy_spaghetti::{
    DetailTransition, GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiCutout,
    SpaghettiLighting, SpaghettiReceiveShadows, SpaghettiSpecular, SpaghettiTransmission,
};

use crate::{
    cutout::{disc, rod, tapered, tapered_rod},
    debris::LogDetail,
    flowers::Rng,
    landscape::{SUN, noise},
};

/// Projected pixels per meter between which the fine level hands over to
/// the coarse one. A moss frond is a centimeter wide, so the fronds merge
/// near 13 m at 800 lines.
const HANDOVER: [f32; 2] = [50.0, 68.0];
const RANGE: VisibilityRange = VisibilityRange {
    start_margin: 0.0..0.0,
    end_margin: 44.0..54.0,
    use_aabb: true,
};

/// The gains of the forest litter, so the growth sits in the same light.
const LIGHTING: SpaghettiLighting = SpaghettiLighting {
    direction_to_light: SUN,
    ambient: Vec3::new(0.33, 0.37, 0.42),
    directional: Vec3::new(0.86, 0.73, 0.56),
};

/// Length of the log scan along its axis, in meters.
const SCAN_LENGTH: f32 = 3.05;
/// Mean radius of the log scan, in meters.
const SCAN_RADIUS: f32 = 0.14;
const STATIONS: usize = 20;
const BINS: usize = 10;

/// The outer surface of the log scan, as radii around its centerline.
struct Profile {
    low: f32,
    high: f32,
    /// Height and side offset of the centerline at each station.
    centers: [Vec2; STATIONS],
    radii: [[f32; BINS]; STATIONS],
}

static PROFILE: LazyLock<Profile> = LazyLock::new(|| {
    let scan =
        SpaghettiAsset::from_bytes(include_bytes!("../../assets/windward/deadfall-far.spag"))
            .expect("valid log scan");
    let points: Vec<Vec3> = scan
        .segments()
        .iter()
        .map(|segment| segment.start().midpoint(segment.end()))
        .collect();
    let (low, high) = points
        .iter()
        .fold((f32::MAX, f32::MIN), |(low, high), point| {
            (low.min(point.x), high.max(point.x))
        });
    let station =
        |x: f32| (((x - low) / (high - low) * STATIONS as f32) as usize).min(STATIONS - 1);
    let mut sums = [(Vec2::ZERO, 0.0_f32); STATIONS];
    for point in &points {
        let sum = &mut sums[station(point.x)];
        sum.0 += point.yz();
        sum.1 += 1.0;
    }
    let centers = sums.map(|(sum, count)| sum / count.max(1.0));
    let mut samples = vec![Vec::new(); STATIONS * BINS];
    for point in &points {
        let index = station(point.x);
        let offset = point.yz() - centers[index];
        let turn = offset.y.atan2(offset.x).rem_euclid(TAU);
        let bin = ((turn / TAU * BINS as f32) as usize).min(BINS - 1);
        samples[index * BINS + bin].push(offset.length());
    }
    // A high percentile follows the crests of the bark, so the growth sits
    // on the log and not under its surface.
    let crest = |radii: &mut Vec<f32>| {
        radii.sort_by(f32::total_cmp);
        radii.get(radii.len() * 4 / 5).copied()
    };
    let mut radii = [[SCAN_RADIUS; BINS]; STATIONS];
    for (index, row) in radii.iter_mut().enumerate() {
        let mut whole: Vec<f32> = samples[index * BINS..(index + 1) * BINS].concat();
        let fallback = crest(&mut whole).unwrap_or(SCAN_RADIUS);
        for (bin, radius) in row.iter_mut().enumerate() {
            *radius = crest(&mut samples[index * BINS + bin]).unwrap_or(fallback);
        }
    }
    Profile {
        low,
        high,
        centers,
        radii,
    }
});

impl Profile {
    /// The surface point and outward normal in the scan's frame. `along`
    /// runs from zero to one down the log. `turn` is the angle around the
    /// axis in radians, from the top toward +Z.
    fn surface(&self, along: f32, turn: f32) -> (Vec3, Vec3) {
        let station = (along * STATIONS as f32 - 0.5).clamp(0.0, (STATIONS - 1) as f32);
        let (near, far) = (station as usize, (station as usize + 1).min(STATIONS - 1));
        let bin = (turn.rem_euclid(TAU) / TAU * BINS as f32 - 0.5).rem_euclid(BINS as f32);
        let (left, right) = (bin as usize % BINS, (bin as usize + 1) % BINS);
        let radius_at = |station: usize| {
            self.radii[station][left].lerp(self.radii[station][right], bin.fract())
        };
        let radius = radius_at(near).lerp(radius_at(far), station.fract());
        let center = self.centers[near].lerp(self.centers[far], station.fract());
        let outward = Vec3::new(0.0, turn.cos(), turn.sin());
        (
            Vec3::new(self.low.lerp(self.high, along), center.x, center.y) + outward * radius,
            outward,
        )
    }
}

/// A fine and a coarse version of the same growth.
#[derive(Default)]
struct Levels {
    fine: Vec<GaussianSegment>,
    coarse: Vec<GaussianSegment>,
}

/// One log, in the frame of its own cloud: world axes, with the origin at
/// the log's translation.
struct Log {
    rotation: Quat,
    scale: Vec3,
    /// Shifts the moss pattern, so no two logs carry the same patches.
    shift: f32,
}

impl Log {
    fn length(&self) -> f32 {
        SCAN_LENGTH * self.scale.x
    }

    fn radius(&self) -> f32 {
        SCAN_RADIUS * self.scale.y
    }

    fn surface(&self, along: f32, turn: f32) -> (Vec3, Vec3) {
        let (point, normal) = PROFILE.surface(along, turn);
        (
            self.rotation * (self.scale * point),
            (self.rotation * (normal / self.scale)).normalize(),
        )
    }

    /// How strongly moss holds at a place. Moss grows where this is above a
    /// half: in broad patches, mostly on the upper side.
    fn moss(&self, along: f32, turn: f32, normal: Vec3) -> f32 {
        if normal.y < 0.05 {
            return 0.0;
        }
        let at = Vec2::new(along * self.length(), turn * self.radius());
        let patch = noise(at * 2.3 + self.shift);
        let fine = noise(at * 11.0 - self.shift);
        0.78 * patch + 0.22 * fine + 0.36 * (normal.y - 0.78)
    }
}

pub fn spawn_nurse_logs(
    mut commands: Commands,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
    logs: Query<&Transform, With<LogDetail>>,
) {
    for place in &logs {
        // The place gives the seed, so a log keeps its growth whatever the
        // order of the query.
        let seed = (place.translation.x * 53.0 + place.translation.z * 97.0).to_bits();
        let mut rng = Rng::new(seed);
        let log = Log {
            rotation: place.rotation,
            scale: place.scale,
            shift: rng.range(0.0, 90.0),
        };
        let mut growth = Levels::default();
        moss(&mut growth, &log, &mut rng.fork());
        lichen(&mut growth.fine, &log, &mut rng.fork());
        stubs(&mut growth, &log, &mut rng.fork());
        brackets(&mut growth, &log, &mut rng.fork());
        seedlings(&mut growth, &log, &mut rng.fork());
        if growth.fine.is_empty() || growth.coarse.is_empty() {
            continue;
        }
        let finer_end = growth.fine.len() as u32;
        growth.fine.append(&mut growth.coarse);
        let transition = DetailTransition {
            finer_end,
            pixels_per_unit: HANDOVER,
        };
        commands.spawn((
            Name::new("Log growth"),
            Spaghetti3d(clouds.add(
                SpaghettiAsset::with_detail(growth.fine, &[transition]).expect("valid log growth"),
            )),
            SpaghettiCutout,
            LIGHTING,
            SpaghettiReceiveShadows,
            // Moss fronds and thin fungi glow when the low sun is behind them.
            SpaghettiTransmission(0.4),
            SpaghettiSpecular {
                strength: 0.06,
                shininess: 30.0,
                strand: 0.0,
            },
            Transform::from_translation(place.translation),
            RANGE,
        ));
    }
}

/// A carpet of dark pads under pale frond tips, in patches along the top.
fn moss(out: &mut Levels, log: &Log, rng: &mut Rng) {
    let hue = rng.next();
    let deep = Vec3::new(0.030, 0.070, 0.016).lerp(Vec3::new(0.050, 0.085, 0.018), hue);
    let bright = Vec3::new(0.115, 0.200, 0.035).lerp(Vec3::new(0.200, 0.250, 0.045), hue);
    let area = log.length() * TAU * log.radius();
    for _ in 0..(3600.0 * area) as usize {
        let along = rng.range(0.03, 0.97);
        let turn = rng.range(-PI, PI);
        let (point, normal) = log.surface(along, turn);
        let cover = log.moss(along, turn, normal);
        if cover < 0.5 {
            continue;
        }
        // The carpet is thickest in the middle of a patch.
        let root = point + normal * (0.022 * ((cover - 0.5) / 0.25).min(1.0));
        let tone = rng.next();
        if rng.next() < 0.28 {
            disc(
                &mut out.fine,
                root - normal * 0.005,
                normal,
                rng.range(0.016, 0.026),
                deep.lerp(bright, 0.4 * tone * tone) * rng.range(0.8, 1.15),
            );
            continue;
        }
        let lie = rng.next() * TAU;
        let (p, q) = normal.any_orthonormal_pair();
        let reach = rng.range(0.014, 0.026);
        let tip =
            root + (p * lie.cos() + q * lie.sin()) * reach + normal * (reach * rng.range(0.5, 1.1));
        tapered(
            &mut out.fine,
            (root, tip),
            normal,
            rng.range(0.005, 0.008),
            0.3,
            deep.lerp(bright, 0.10 + 0.80 * tone * tone),
        );
    }

    // The coarse carpet is one disc for each covered cell of a grid.
    let mean = deep.lerp(bright, 0.30);
    let rows = (log.length() / 0.13).ceil() as usize;
    let columns = (TAU * log.radius() / 0.13).ceil() as usize;
    for row in 0..rows {
        for column in 0..columns {
            let along = 0.03 + 0.94 * (row as f32 + 0.5) / rows as f32;
            let turn = -PI + TAU * (column as f32 + 0.5) / columns as f32;
            let (point, normal) = log.surface(along, turn);
            if log.moss(along, turn, normal) < 0.5 {
                continue;
            }
            disc(
                &mut out.coarse,
                point + normal * 0.006,
                normal,
                0.085,
                mean * rng.range(0.85, 1.1),
            );
        }
    }
}

/// Crusts of pale lichen on the bark that the moss has not covered.
fn lichen(out: &mut Vec<GaussianSegment>, log: &Log, rng: &mut Rng) {
    let area = log.length() * TAU * log.radius();
    for _ in 0..(120.0 * area) as usize {
        let along = rng.range(0.02, 0.98);
        let turn = rng.range(-PI, PI);
        let (point, normal) = log.surface(along, turn);
        if normal.y < -0.35 || log.moss(along, turn, normal) > 0.44 {
            continue;
        }
        // Most crusts are gray-green. A few are the orange of sunburst lichen.
        let color = if rng.next() < 0.07 {
            Vec3::new(0.15, 0.095, 0.02)
        } else {
            Vec3::new(0.050, 0.058, 0.044).lerp(Vec3::new(0.090, 0.098, 0.075), rng.next())
        };
        let radius = rng.range(0.010, 0.026);
        let (p, q) = normal.any_orthonormal_pair();
        // Overlapping lobes give each crust a ragged outline.
        for _ in 0..7 {
            let lie = rng.next() * TAU;
            let offset = (p * lie.cos() + q * lie.sin()) * (radius * rng.range(0.0, 0.9));
            disc(
                out,
                point + offset + normal * 0.004,
                normal,
                radius * rng.range(0.28, 0.55),
                color * rng.range(0.8, 1.2),
            );
        }
    }
}

/// The weathered stubs of broken limbs.
fn stubs(out: &mut Levels, log: &Log, rng: &mut Rng) {
    let axis = log.rotation * Vec3::X;
    for _ in 0..(2.0 + 3.6 * rng.next()) as usize {
        let along = rng.range(0.1, 0.9);
        let turn = rng.range(-1.9, 1.9);
        let (point, normal) = log.surface(along, turn);
        if normal.y < -0.1 {
            continue;
        }
        // Limbs lean toward the crown end of the trunk.
        let lean = (normal + axis * rng.range(0.15, 0.6)).normalize();
        let radius = rng.range(0.012, 0.028) * log.scale.y.sqrt();
        let length = radius * rng.range(4.0, 9.0);
        let root = point - lean * 0.03;
        let tip = point + lean * length;
        let color = Vec3::new(0.055, 0.047, 0.036).lerp(Vec3::new(0.105, 0.095, 0.08), rng.next());
        // A kernel has one normal. A sunlit half and a shaded half make the
        // stub read as round.
        let toward = (SUN - lean * SUN.dot(lean)).normalize();
        for (side, face) in [(0.3, toward), (-0.3, -toward)] {
            let offset = toward * (side * radius);
            tapered_rod(
                &mut out.fine,
                (root + offset, tip + offset),
                face,
                0.8 * radius,
                0.7,
                color,
            );
        }
        // Paler splintered wood shows at the break.
        disc(
            &mut out.fine,
            tip,
            lean,
            0.6 * radius,
            Vec3::new(0.15, 0.12, 0.08),
        );
        rod(&mut out.coarse, root, tip, toward + Vec3::Y, radius, color);
    }
}

/// Clusters of bracket fungi on the flanks. Each shelf has a pale growing
/// edge, a rust band and a dark old center over a cream underside.
fn brackets(out: &mut Levels, log: &Log, rng: &mut Rng) {
    for _ in 0..(1.0 + 2.6 * rng.next()) as usize {
        let along = rng.range(0.1, 0.9);
        let side = if rng.next() < 0.5 { -1.0 } else { 1.0 };
        let size = rng.range(0.032, 0.075);
        let mut turn = side * rng.range(0.9, 1.5);
        let mut shift = 0.0;
        for _ in 0..(2.0 + 4.6 * rng.next()) as usize {
            let (point, normal) = log.surface(along + shift / log.length(), turn);
            // The next shelf sits lower and a little to one side.
            turn += side * rng.range(0.15, 0.35);
            shift += rng.range(-0.8, 0.8) * size;
            let outward = Vec3::new(normal.x, 0.0, normal.z);
            if normal.y.abs() > 0.75 || outward.length() < 0.3 {
                continue;
            }
            let outward = outward.normalize();
            let radius = size * rng.range(0.55, 1.0);
            // A shelf grows level and droops a little at its edge.
            let face = (Vec3::Y + outward * rng.range(0.1, 0.35)).normalize();
            let center = point + outward * (0.3 * radius);
            let warm = rng.next();
            let rust = Vec3::new(0.21, 0.08, 0.025).lerp(Vec3::new(0.28, 0.15, 0.045), warm);
            let edge = Vec3::new(0.26, 0.23, 0.16);
            disc(
                &mut out.fine,
                center - face * (0.10 * radius),
                -face,
                0.97 * radius,
                Vec3::new(0.20, 0.18, 0.13),
            );
            disc(&mut out.fine, center, face, radius, edge);
            disc(
                &mut out.fine,
                center - outward * (0.15 * radius) + face * (0.07 * radius),
                face,
                0.83 * radius,
                rust,
            );
            disc(
                &mut out.fine,
                center - outward * (0.34 * radius) + face * (0.14 * radius),
                face,
                0.62 * radius,
                Vec3::new(0.06, 0.04, 0.025).lerp(rust, 0.25),
            );
            disc(
                &mut out.coarse,
                center + face * (0.05 * radius),
                face,
                radius,
                rust.lerp(edge, 0.3),
            );
        }
    }
}

/// Conifer seedlings rooted in the moss on top of the log.
fn seedlings(out: &mut Levels, log: &Log, rng: &mut Rng) {
    let wanted = (3.4 * rng.next()) as usize;
    let mut grown = 0;
    for _ in 0..40 {
        if grown == wanted {
            break;
        }
        let along = rng.range(0.08, 0.92);
        let turn = rng.range(-0.6, 0.6);
        let (point, normal) = log.surface(along, turn);
        if normal.y < 0.75 || log.moss(along, turn, normal) < 0.55 {
            continue;
        }
        grown += 1;
        let tall = rng.range(0.07, 0.17);
        let up =
            (Vec3::Y + Vec3::new(rng.range(-0.15, 0.15), 0.0, rng.range(-0.15, 0.15))).normalize();
        let top = point + up * tall;
        let green = Vec3::new(0.075, 0.19, 0.04).lerp(Vec3::new(0.15, 0.29, 0.055), rng.next());
        rod(
            &mut out.fine,
            point,
            top,
            SUN,
            0.0022,
            Vec3::new(0.20, 0.13, 0.07),
        );
        let (u, v) = up.any_orthonormal_pair();
        let whorls = 4 + (tall / 0.04) as usize;
        for whorl in 0..whorls {
            let height = (whorl as f32 + 0.6) / whorls as f32;
            let twist = rng.next() * TAU;
            // The top whorl stands up as the leading shoot.
            let rise = if whorl + 1 == whorls { 1.6 } else { 0.55 };
            let reach = tall * (0.34 - 0.14 * height) * rng.range(0.85, 1.15);
            for needle in 0..6 {
                let lie = twist + TAU * needle as f32 / 6.0;
                let outward = u * lie.cos() + v * lie.sin();
                let root = point + up * (tall * height);
                tapered(
                    &mut out.fine,
                    (root, root + (outward + up * rise).normalize() * reach),
                    up - outward * 0.4,
                    0.0036,
                    0.3,
                    green * rng.range(0.85, 1.15),
                );
            }
        }
        tapered(&mut out.coarse, (point, top), SUN, 0.4 * tall, 0.3, green);
    }
}
