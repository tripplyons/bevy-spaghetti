//! Orb webs strung with dew.
//!
//! Each web is a translucent cloud of silk threads far thinner than a pixel.
//! The renderer keeps their mass, so a web shows as a faint veil from a
//! distance and as separate threads up close. The strand highlight makes
//! bands of threads flash where they turn the low sun toward the eye.

use std::f32::consts::TAU;

use bevy::{camera::visibility::VisibilityRange, prelude::*};
use bevy_spaghetti::{
    GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiLighting, SpaghettiReceiveShadows,
    SpaghettiSpecular, SpaghettiTransmission,
};

use crate::{
    flowers::Rng,
    forest::forest,
    landscape::{SUN, WATER_LEVEL, ground, shore_distance, trail_x},
    meadow::ground_mask,
    props::PropFootprints,
};

const RANGE: VisibilityRange = VisibilityRange {
    start_margin: 0.0..0.0,
    end_margin: 26.0..32.0,
    use_aabb: true,
};

/// Silk scatters the sun strongly and the sky weakly.
const LIGHTING: SpaghettiLighting = SpaghettiLighting {
    direction_to_light: SUN,
    ambient: Vec3::new(0.22, 0.25, 0.29),
    directional: Vec3::new(0.95, 0.80, 0.62),
};

/// Silk is dim away from the highlight, so the glinting threads stand out.
const SILK: Vec3 = Vec3::new(0.30, 0.32, 0.35);
/// Standard deviation of a thread's cross section, in meters.
const THREAD: f32 = 0.0004;
/// Optical mass per meter of a thread in a web of radius `FULL`. It makes a
/// thread one pixel wide about a tenth opaque from six meters.
const WEIGHT: f32 = 0.0015;
/// A smaller web has closer threads. Its threads are lighter in proportion,
/// so every web is the same faint veil from a distance.
const FULL: f32 = 0.3;

pub fn spawn_webs(
    mut commands: Commands,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
    props: Option<Res<PropFootprints>>,
) {
    let empty = PropFootprints::default();
    let props = props.as_deref().unwrap_or(&empty);
    let mut rng = Rng::new(40_127);
    // A web that faces the sun is lit across its whole sheet.
    let toward = Vec2::new(SUN.x, SUN.z).normalize();
    let mut sites = Vec::new();

    // Low webs between the tall stalks of the damp meadow by the lake.
    let mut tries = 0;
    while sites.len() < 18 && tries < 4000 {
        tries += 1;
        let at = Vec2::new(rng.range(-12.0, 22.0), rng.range(-34.0, 8.0));
        let shore = shore_distance(at);
        if !(1.12..1.75).contains(&shore)
            || ground(at) < WATER_LEVEL + 0.15
            || ground_mask(at) < 0.6
            || props.keep(at) < 0.95
            || (at.x - trail_x(at.y)).abs() < 1.4
            || sites
                .iter()
                .any(|&(other, ..): &(Vec2, f32, f32, Vec2)| other.distance(at) < 2.2)
        {
            continue;
        }
        let facing = Vec2::from_angle(rng.range(-0.7, 0.7)).rotate(toward);
        sites.push((at, rng.range(0.30, 0.52), rng.range(0.12, 0.20), facing));
    }

    // Larger webs beside the trunks, strung out toward the low limbs.
    for (index, tree) in forest()
        .trees
        .iter()
        .enumerate()
        .filter(|(_, tree)| tree.is_stand())
    {
        let mut rng = Rng::new(52_361 + 97 * index as u32);
        if rng.next() > 0.3 {
            continue;
        }
        let radius = rng.range(0.20, 0.34);
        let facing = Vec2::from_angle(rng.range(-0.6, 0.6)).rotate(toward);
        let hand = if rng.next() < 0.5 { -1.0 } else { 1.0 };
        let at = tree.at + facing.perp() * (hand * (0.27 * tree.scale + radius + 0.12));
        if (at.x - trail_x(at.y)).abs() < 1.2
            || ground(at) < WATER_LEVEL + 0.2
            || props.keep(at) < 0.95
            || forest()
                .trees
                .iter()
                .any(|other| other.at.distance(at) < 0.27 * other.scale + radius)
        {
            continue;
        }
        sites.push((at, rng.range(0.9, 1.8), radius, facing));
    }

    for (index, (at, height, radius, facing)) in sites.into_iter().enumerate() {
        let mut rng = Rng::new(61_001 + 131 * index as u32);
        let origin = Vec3::new(at.x, ground(at) + height, at.y);
        let normal = Vec3::new(facing.x, rng.range(-0.12, 0.12), facing.y).normalize();
        commands.spawn((
            Name::new("Orb web"),
            Spaghetti3d(
                clouds.add(
                    SpaghettiAsset::new(orb_web(radius, normal, &mut rng)).expect("valid web"),
                ),
            ),
            LIGHTING,
            SpaghettiReceiveShadows,
            // Silk is clear, so the sun lights the sheet from either side.
            SpaghettiTransmission(1.0),
            SpaghettiSpecular {
                strength: 1.1,
                shininess: 70.0,
                strand: 1.0,
            },
            Transform::from_translation(origin),
            RANGE,
        ));
    }
}

/// Builds a web of the given radius around the origin, in the plane that
/// faces `normal`.
fn orb_web(radius: f32, normal: Vec3, rng: &mut Rng) -> Vec<GaussianSegment> {
    let side = Vec3::Y.cross(normal).normalize();
    let up = normal.cross(side);
    let mut web = Vec::new();
    let rotation = Quat::from_mat3(&Mat3::from_cols(side, up, normal));
    let fine = radius / FULL;
    let thread = |web: &mut Vec<GaussianSegment>, a: Vec3, b: Vec3, weight: f32, tone: f32| {
        web.extend(silk(a, b, normal, weight * fine, tone));
    };

    // The hub sits above the middle, as the spider waits head down.
    let spokes = 15 + (rng.next() * 8.0) as usize;
    let reach: Vec<(f32, f32)> = (0..spokes)
        .map(|spoke| {
            let turn = TAU * (spoke as f32 + rng.range(-0.22, 0.22)) / spokes as f32;
            let low = 0.5 - 0.5 * turn.sin();
            (turn, radius * (0.82 + 0.34 * low) * rng.range(0.93, 1.07))
        })
        .collect();
    let point =
        |(turn, _): (f32, f32), distance: f32| (side * turn.cos() + up * turn.sin()) * distance;
    for &spoke in &reach {
        thread(
            &mut web,
            point(spoke, 0.012),
            point(spoke, spoke.1 * 1.08),
            1.3,
            1.0,
        );
    }
    // The frame joins the spoke ends.
    for (index, &spoke) in reach.iter().enumerate() {
        let next = reach[(index + 1) % spokes];
        thread(
            &mut web,
            point(spoke, spoke.1 * 1.08),
            point(next, next.1 * 1.08),
            1.3,
            1.0,
        );
    }
    // A few long threads hold the frame to whatever stands near.
    let anchors = 4 + (rng.next() * 2.0) as usize;
    for anchor in 0..anchors {
        let spoke = reach[(anchor * spokes / anchors + (rng.next() * 2.0) as usize) % spokes];
        let out = point(spoke, 1.0);
        let end = point(spoke, spoke.1 * 1.08)
            + out * (radius * rng.range(0.7, 1.9))
            + normal * (radius * rng.range(-0.5, 0.5));
        thread(&mut web, point(spoke, spoke.1 * 1.08), end, 1.3, 1.0);
    }

    // The sticky spiral, as straight chords between neighboring spokes.
    let turns = 13 + (rng.next() * 7.0) as usize;
    let torn = (rng.next() * spokes as f32) as usize;
    let tear = rng.range(0.35, 0.9);
    for turn in 0..turns {
        let ring = 0.18 + 0.78 * (turn as f32 + 0.5) / turns as f32;
        for (index, &spoke) in reach.iter().enumerate() {
            let next = reach[(index + 1) % spokes];
            // Old webs have a torn sector and a few lost chords.
            if (index == torn && ring > tear) || rng.next() < 0.04 {
                continue;
            }
            let a = point(spoke, spoke.1 * ring * rng.range(0.985, 1.015));
            let b = point(next, next.1 * ring * rng.range(0.985, 1.015));
            thread(&mut web, a, b, 0.8, 0.92);
            // Dew gathers into beads along the sticky chords.
            for _ in 0..(rng.next() * 2.6) as usize {
                let at = a.lerp(b, rng.range(0.1, 0.9));
                let sigma = rng.range(0.0009, 0.0016) * fine.sqrt();
                // Random normals make the beads catch the sun one by one.
                let face = Quat::from_rotation_arc(
                    Vec3::Z,
                    Vec3::new(
                        rng.range(-1.0, 1.0),
                        rng.range(-1.0, 1.0),
                        rng.range(-1.0, 1.0),
                    )
                    .normalize_or(Vec3::Z),
                );
                web.push(GaussianSegment::new(
                    at,
                    at,
                    Vec3::splat(sigma),
                    rotation * face,
                    16.0 * sigma * sigma,
                    Vec3::new(0.62, 0.66, 0.72),
                ));
            }
        }
    }
    web
}

/// One thread from `a` to `b`. `weight` scales its optical mass.
fn silk(a: Vec3, b: Vec3, normal: Vec3, weight: f32, tone: f32) -> Option<GaussianSegment> {
    let length = a.distance(b);
    if length < 1e-4 {
        return None;
    }
    let along = (b - a) / length;
    // Keep the kernel's normal near the web's normal as the thread turns.
    let across = along.cross(normal).try_normalize()?;
    let rotation = Quat::from_mat3(&Mat3::from_cols(across, along, across.cross(along)));
    Some(GaussianSegment::new(
        a,
        b,
        Vec3::splat(THREAD),
        rotation,
        WEIGHT * weight * length,
        SILK * tone,
    ))
}
