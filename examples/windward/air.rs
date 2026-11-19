//! Things that hang and drift in the air: mist over the lake, swarms of
//! midges in the low sun, and seed fluff on the wind. Each is a cloud of
//! soft Gaussian kernels that `SpaghettiFlow` moves on the GPU.

use std::f32::consts::TAU;

use bevy::prelude::*;
use bevy_spaghetti::{
    GaussianSegment, Spaghetti3d, SpaghettiAdditive, SpaghettiAsset, SpaghettiFlow,
};

use crate::{
    flowers::Rng,
    landscape::{LAKE, LAKE_RADII, WATER_LEVEL, ground},
};

/// The gusts in `spaghetti.wgsl` blow this way.
const WIND: Vec3 = Vec3::new(0.82, 0.0, 0.47);
const MIST: Vec3 = Vec3::new(0.32, 0.40, 0.45);
/// Places where midges gather: over the shallows and the warm meadow edge.
const SWARMS: [Vec2; 9] = [
    Vec2::new(3.8, 3.2),
    Vec2::new(2.4, -6.4),
    Vec2::new(0.8, -13.2),
    Vec2::new(-4.6, -3.0),
    Vec2::new(-2.4, -10.5),
    Vec2::new(3.4, -22.6),
    Vec2::new(-6.6, -24.0),
    Vec2::new(6.4, 9.6),
    Vec2::new(-1.2, 4.6),
];

/// Runs every flowing cloud from the same clock.
pub fn advance_flows(time: Res<Time>, mut flows: Query<&mut SpaghettiFlow>) {
    for mut flow in &mut flows {
        flow.time = time.elapsed_secs_wrapped();
    }
}

pub fn spawn_air(mut commands: Commands, mut clouds: ResMut<Assets<SpaghettiAsset>>) {
    let mut rng = Rng::new(0xA1E5);

    // Long banks that slide across the water and reform upwind.
    let drift = WIND * 0.12;
    let period = 32.0;
    commands.spawn((
        Name::new("Lake mist banks"),
        Spaghetti3d(clouds.add(
            SpaghettiAsset::new(mist_banks(&mut rng, drift * period)).expect("valid lake mist"),
        )),
        SpaghettiFlow {
            velocity: drift,
            period,
            turbulence: 0.25,
            growth: 0.25,
            fade: [0.3, 0.4],
            ..default()
        },
    ));
    // Steam fog: thin curls that lift off the water and dissolve as they
    // rise into the warmer air and the sun.
    let rise = Vec3::Y * 0.15 + WIND * 0.3;
    commands.spawn((
        Name::new("Lake steam"),
        Spaghetti3d(clouds.add(SpaghettiAsset::new(steam(&mut rng)).expect("valid lake steam"))),
        SpaghettiFlow {
            velocity: rise,
            period: 9.0,
            turbulence: 0.22,
            growth: 1.7,
            fade: [0.25, 0.5],
            tint: Vec3::new(1.45, 1.28, 1.12),
            ..default()
        },
    ));
    // The insects only dance in place, so the flow has no travel.
    commands.spawn((
        Name::new("Midges"),
        Spaghetti3d(clouds.add(SpaghettiAsset::new(midges(&mut rng)).expect("valid midges"))),
        SpaghettiAdditive,
        SpaghettiFlow {
            period: 2.4,
            turbulence: 0.30,
            flicker: 0.7,
            fade: [0.0, 0.0],
            ..default()
        },
    ));
    let carry = WIND * 0.55 + Vec3::Y * 0.03;
    let period = 26.0;
    commands.spawn((
        Name::new("Seed fluff"),
        Spaghetti3d(
            clouds.add(
                SpaghettiAsset::new(seeds(&mut rng, carry * period)).expect("valid seed fluff"),
            ),
        ),
        SpaghettiAdditive,
        SpaghettiFlow {
            velocity: carry,
            period,
            turbulence: 0.5,
            flicker: 0.25,
            fade: [0.15, 0.25],
            ..default()
        },
    ));
}

fn mist_banks(rng: &mut Rng, travel: Vec3) -> Vec<GaussianSegment> {
    (0..34)
        .map(|_| {
            // Start upwind, so the middle of each loop lies over the lake.
            let x = 7.5 + rng.range(0.0, 5.6) - 0.5 * travel.x;
            let z = -22.0 + rng.range(0.0, 18.0) - 0.5 * travel.z;
            let y = WATER_LEVEL + rng.range(0.36, 0.74);
            let length = rng.range(1.8, 5.4);
            let height = rng.range(0.13, 0.26);
            let start = Vec3::new(x - 0.5 * length, y, z);
            let end = Vec3::new(x + 0.5 * length, y + rng.range(0.0, 0.14), z - 1.0);
            GaussianSegment::new(
                start,
                end,
                Vec3::new(1.8, height, rng.range(0.76, 1.76)),
                Quat::IDENTITY,
                length * height * 0.13,
                MIST,
            )
        })
        .collect()
}

fn steam(rng: &mut Rng) -> Vec<GaussianSegment> {
    let mut wisps = Vec::new();
    while wisps.len() < 150 {
        let offset = rng.range(0.0, 1.0).sqrt() * Vec2::from_angle(rng.range(0.0, TAU));
        let at = LAKE + LAKE_RADII * offset;
        let (wide, tall) = (rng.range(0.22, 0.5), rng.range(0.09, 0.2));
        if ground(at) > WATER_LEVEL - 0.05 {
            continue;
        }
        let foot = Vec3::new(at.x, WATER_LEVEL + rng.range(0.08, 0.3), at.y);
        wisps.push(GaussianSegment::new(
            foot,
            // A short curl, tipped the way the wind will carry it.
            foot + Vec3::new(
                rng.range(0.1, 0.5),
                rng.range(0.05, 0.3),
                rng.range(0.0, 0.3),
            ),
            Vec3::new(wide, tall, wide * rng.range(0.7, 1.2)),
            Quat::IDENTITY,
            0.5 * wide * tall,
            MIST * rng.range(0.95, 1.2),
        ));
    }
    wisps
}

fn midges(rng: &mut Rng) -> Vec<GaussianSegment> {
    let mut insects = Vec::new();
    for center in SWARMS {
        let floor = ground(center).max(WATER_LEVEL);
        let height = rng.range(1.3, 2.3);
        let size = rng.range(0.3, 0.55);
        for _ in 0..70 {
            // A loose ball, taller than it is wide.
            let offset = Vec3::new(
                rng.range(-1.0, 1.0) + rng.range(-1.0, 1.0),
                1.4 * (rng.range(-1.0, 1.0) + rng.range(-1.0, 1.0)),
                rng.range(-1.0, 1.0) + rng.range(-1.0, 1.0),
            ) * (0.5 * size);
            let at = Vec3::new(center.x, floor + height, center.y) + offset;
            let sigma = rng.range(0.003, 0.0045);
            insects.push(GaussianSegment::new(
                at,
                at,
                Vec3::splat(sigma),
                Quat::IDENTITY,
                2.5 * TAU * sigma * sigma,
                // Wings scatter the low sun as warm sparks.
                Vec3::new(1.2, 0.95, 0.55) * rng.range(0.4, 1.0),
            ));
        }
    }
    insects
}

fn seeds(rng: &mut Rng, travel: Vec3) -> Vec<GaussianSegment> {
    (0..280)
        .map(|_| {
            let start = Vec2::new(rng.range(-24.0, 8.0), rng.range(-48.0, 10.0));
            let end = start + travel.xz();
            // Stay clear of the ground along the whole path.
            let floor = ground(start)
                .max(ground(0.5 * (start + end)))
                .max(ground(end))
                .max(WATER_LEVEL);
            let at = Vec3::new(start.x, floor + rng.range(0.5, 3.6), start.y);
            let sigma = rng.range(0.004, 0.007);
            GaussianSegment::new(
                at,
                at,
                Vec3::splat(sigma),
                Quat::IDENTITY,
                0.9 * TAU * sigma * sigma,
                Vec3::new(0.50, 0.45, 0.36) * rng.range(0.5, 1.0),
            )
        })
        .collect()
}
