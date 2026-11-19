//! A small camp beside the trail: a ring of stones around a burning lay of
//! charred logs, with a coal bed, flames, sparks, drifting smoke, and a stack
//! of firewood. The ring stones are scans. Everything else is built here from
//! Gaussian segments.

use std::f32::consts::TAU;

use bevy::{
    camera::visibility::VisibilityRange,
    light::{NotShadowCaster, NotShadowReceiver},
    prelude::*,
};
use bevy_spaghetti::{
    GaussianSegment, Spaghetti3d, SpaghettiAdditive, SpaghettiAsset, SpaghettiCutout,
    SpaghettiFlow, SpaghettiLighting, SpaghettiReceiveShadows, SpaghettiSpecular,
};

use crate::{
    flowers::Rng,
    forest::ForestDetail,
    landscape::{Obstacles, ROCKS, SUN, ground, shore_distance, trail_x},
    props::{self, PropFootprints},
};

/// The camp moves from here to the nearest level, clear ground.
const ANCHOR: Vec2 = Vec2::new(-3.4, 1.2);
/// Distance from the fire to the centers of the ring stones.
const RING: f32 = 0.60;
/// Radius of the trampled ground that the meadow leaves bare.
const REACH: f32 = 1.3;
/// The gusts in `spaghetti.wgsl` blow this way, so smoke leans with the grass.
const WIND: Vec2 = Vec2::new(0.82, 0.47);
/// The camp stands in the open meadow, between the stand gains and full sun.
const LIGHTING: SpaghettiLighting = SpaghettiLighting {
    direction_to_light: SUN,
    ambient: Vec3::new(0.30, 0.34, 0.36),
    directional: Vec3::new(0.70, 0.56, 0.40),
};
const RANGE: VisibilityRange = VisibilityRange {
    start_margin: 0.0..0.0,
    end_margin: 70.0..84.0,
    use_aabb: false,
};

pub fn spawn_camp(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut obstacles: ResMut<Obstacles>,
    mut footprints: ResMut<PropFootprints>,
    trees: Query<&Transform, With<ForestDetail>>,
) {
    let trees: Vec<Vec2> = trees.iter().map(|tree| tree.translation.xz()).collect();
    let site = find_site(&trees, &footprints);
    // The stack sits on the side away from the trail.
    let away = Vec2::new((site.x - trail_x(site.y)).signum(), 0.35).normalize();
    let stack = site + away * (REACH + 0.25);
    footprints.circles.push((site, REACH));
    footprints.circles.push((stack, 0.62));
    obstacles.add_boulder(site, RING + 0.12, 0.45);
    obstacles.add_boulder(stack, 0.34, 0.3);

    let origin = Vec3::new(site.x, ground(site), site.y);
    let mut rng = Rng::new(0xCA3F);
    spawn_ring(&mut commands, &server, site, &mut rng);

    // Scorched soil under the ash, wider than the ring.
    commands.spawn((
        Name::new("Scorched ground"),
        Mesh3d(meshes.add(props::contact_shadow(site, 0.34, 1.0, 0.9, 77))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.010, 0.009, 0.008),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        })),
        Transform::from_translation(origin),
        NotShadowCaster,
        NotShadowReceiver,
        props::CONTACT_RANGE,
    ));

    let mut wood = Vec::new();
    let mut glow = Vec::new();
    ash_bed(&mut wood, &mut glow, site, &mut rng);
    fire_lay(&mut wood, &mut glow, site, &mut rng);
    firewood(&mut wood, site, stack, away, &mut rng);
    commands.spawn((
        Name::new("Fire lay, ash, and firewood"),
        Spaghetti3d(clouds.add(SpaghettiAsset::new(wood).expect("valid camp wood"))),
        SpaghettiCutout,
        LIGHTING,
        SpaghettiReceiveShadows,
        // Charcoal has a faint graphite sheen where the sun grazes it.
        SpaghettiSpecular {
            strength: 0.05,
            shininess: 24.0,
            strand: 0.0,
        },
        Transform::from_translation(origin),
        RANGE,
    ));
    // Unlit, so the coals keep their own light in the shade of the logs.
    commands.spawn((
        Name::new("Coals"),
        Spaghetti3d(clouds.add(SpaghettiAsset::new(glow).expect("valid coals"))),
        SpaghettiCutout,
        Transform::from_translation(origin),
        RANGE,
    ));

    let hearth = origin + Vec3::Y * 0.05;
    // The glow breathes in place: no travel, only a slow pulse.
    commands.spawn((
        Name::new("Coal glow"),
        Spaghetti3d(clouds.add(SpaghettiAsset::new(coal_glow(&mut rng)).expect("valid glow"))),
        SpaghettiAdditive,
        SpaghettiFlow {
            period: 2.6,
            flicker: 0.35,
            fade: [0.5, 0.5],
            ..default()
        },
        Transform::from_translation(hearth),
        RANGE,
    ));
    let lean = Vec3::new(WIND.x, 0.0, WIND.y);
    commands.spawn((
        Name::new("Flame bodies"),
        Spaghetti3d(
            clouds.add(
                SpaghettiAsset::new(flames(
                    &mut rng,
                    64,
                    0.17,
                    0.10..0.19,
                    0.028..0.046,
                    Vec3::new(1.55, 0.86, 0.24),
                ))
                .expect("valid flames"),
            ),
        ),
        SpaghettiAdditive,
        SpaghettiFlow {
            velocity: Vec3::Y * 1.0 + lean * 0.12,
            period: 0.55,
            turbulence: 0.025,
            // A flame narrows and thins toward its tip as it cools to red.
            spread: -0.75,
            growth: -0.55,
            flicker: 0.45,
            fade: [0.08, 0.45],
            tint: Vec3::new(0.75, 0.17, 0.02),
            ..default()
        },
        Transform::from_translation(hearth),
        RANGE,
    ));
    commands.spawn((
        Name::new("Flame tongues"),
        Spaghetti3d(
            clouds.add(
                SpaghettiAsset::new(flames(
                    &mut rng,
                    28,
                    0.12,
                    0.16..0.30,
                    0.018..0.030,
                    Vec3::new(1.25, 0.50, 0.09),
                ))
                .expect("valid flames"),
            ),
        ),
        SpaghettiAdditive,
        SpaghettiFlow {
            velocity: Vec3::Y * 1.05 + lean * 0.2,
            period: 0.85,
            turbulence: 0.05,
            spread: -0.85,
            growth: -0.6,
            flicker: 0.6,
            fade: [0.1, 0.5],
            tint: Vec3::new(0.6, 0.08, 0.01),
            ..default()
        },
        Transform::from_translation(hearth),
        RANGE,
    ));
    commands.spawn((
        Name::new("Sparks"),
        Spaghetti3d(clouds.add(SpaghettiAsset::new(sparks(&mut rng)).expect("valid sparks"))),
        SpaghettiAdditive,
        SpaghettiFlow {
            velocity: Vec3::Y * 1.3 + lean * 0.3,
            period: 1.7,
            turbulence: 0.10,
            spread: 1.2,
            growth: -0.5,
            flicker: 0.8,
            fade: [0.02, 0.5],
            tint: Vec3::new(1.0, 0.25, 0.05),
            ..default()
        },
        Transform::from_translation(hearth),
        RANGE,
    ));
    // Smoke rises fast over the heat, then the wind takes it. Each puff
    // swells to several times its size and thins out as it goes.
    commands.spawn((
        Name::new("Smoke"),
        Spaghetti3d(clouds.add(SpaghettiAsset::new(smoke(&mut rng)).expect("valid smoke"))),
        SpaghettiFlow {
            velocity: Vec3::Y * 0.80 + lean * 0.52,
            period: 10.0,
            turbulence: 0.26,
            spread: 2.6,
            growth: 4.6,
            fade: [0.06, 0.55],
            tint: Vec3::new(1.5, 1.5, 1.55),
            ..default()
        },
        Transform::from_translation(hearth),
    ));
}

/// Walks a spiral out from the anchor to the first level, clear spot.
fn find_site(trees: &[Vec2], footprints: &PropFootprints) -> Vec2 {
    let clear = |at: Vec2| {
        let level = (0..8).all(|side| {
            let rim = at + REACH * Vec2::from_angle(side as f32 * TAU / 8.0);
            (ground(rim) - ground(at)).abs() < 0.11
        });
        level
            && (at.x - trail_x(at.y)).abs() > 2.5
            && shore_distance(at) > 1.2
            && trees.iter().all(|tree| tree.distance(at) > 3.4)
            && ROCKS
                .iter()
                .all(|&(x, z, scale)| Vec2::new(x, z).distance(at) > REACH + 1.1 * scale)
            && footprints
                .circles
                .iter()
                .all(|&(center, reach)| center.distance(at) > REACH + reach + 0.9)
            && footprints
                .logs
                .iter()
                .all(|&(a, b)| props::segment_distance(at, a, b) > REACH + 0.9)
    };
    (0..600)
        .map(|step| {
            let turn = 2.399_963 * step as f32;
            ANCHOR + 0.22 * (step as f32).sqrt() * Vec2::from_angle(turn)
        })
        .find(|&at| clear(at))
        .unwrap_or(ANCHOR)
}

/// Sets scanned stones end to end around the fire.
fn spawn_ring(commands: &mut Commands, server: &AssetServer, site: Vec2, rng: &mut Rng) {
    let kinds = props::stone_kinds(server);
    let mut angle = rng.range(0.0, TAU);
    let end = angle + TAU;
    while angle < end - 0.3 {
        // Stone "d" is a flat, near-black scan that reads as a hole here.
        let (clouds, shadow, size) = &kinds[[0, 1, 3][(rng.next() * 3.0) as usize % 3]];
        let length = rng.range(0.27, 0.40).min((end - angle) * RING);
        // The stone spans this much of the ring, with a small gap.
        let arc = (length + rng.range(0.01, 0.05)) / RING;
        let middle = angle + 0.5 * arc;
        angle += arc;
        let at = site + (RING + rng.range(-0.03, 0.03)) * Vec2::from_angle(middle);
        let scale = length / size.x;
        // The scan's long axis is +X. Lay it along the ring.
        let turn = -(middle + 0.25 * TAU) + rng.range(-0.22, 0.22);
        let tilt = Quat::from_rotation_x(rng.range(-0.14, 0.14))
            * Quat::from_rotation_z(rng.range(-0.14, 0.14));
        let floor = props::footprint_floor(at, 0.5 * length);
        props::spawn_prop(
            commands,
            "Fire ring stone",
            Transform::from_xyz(at.x, floor - 0.22 * size.y * scale, at.y)
                .with_rotation(Quat::from_rotation_y(turn) * tilt)
                .with_scale(Vec3::splat(scale)),
            clouds,
            shadow,
            (80.0, 30.0),
        );
    }
}

/// A flat strap from `a` to `b` that faces `face` and is `width` wide.
fn strap(out: &mut Vec<GaussianSegment>, a: Vec3, b: Vec3, face: Vec3, width: f32, color: Vec3) {
    let length = a.distance(b);
    if length < 1e-5 {
        return;
    }
    let tangent = (b - a) / length;
    let normal = (face - tangent * face.dot(tangent))
        .try_normalize()
        .unwrap_or_else(|| tangent.any_orthonormal_vector());
    // The cutout edge sits about 1.5 standard deviations from the axis.
    let sigma = width / 3.05;
    let extent = Vec3::new(sigma, sigma * 0.5, sigma * 0.3);
    out.push(GaussianSegment::new(
        a,
        b,
        extent,
        Quat::from_mat3(&Mat3::from_cols(tangent.cross(normal), tangent, normal)),
        8.0 * sigma * length.max(2.5066 * extent.y),
        color,
    ));
}

/// A round disc of visible radius `radius` that faces `face`.
fn disc(out: &mut Vec<GaussianSegment>, at: Vec3, face: Vec3, radius: f32, color: Vec3) {
    let sigma = radius / 1.52;
    out.push(GaussianSegment::new(
        at,
        at,
        Vec3::new(sigma, sigma, sigma * 0.4),
        Quat::from_rotation_arc(Vec3::Z, face.normalize_or(Vec3::Y)),
        sigma * sigma * 21.0,
        color,
    ));
}

const CHAR: Vec3 = Vec3::new(0.014, 0.013, 0.013);
const ASH: Vec3 = Vec3::new(0.34, 0.33, 0.31);
const BARK: Vec3 = Vec3::new(0.082, 0.056, 0.037);
const SAPWOOD: Vec3 = Vec3::new(0.50, 0.37, 0.21);
const HEARTWOOD: Vec3 = Vec3::new(0.33, 0.20, 0.11);
const EMBER: Vec3 = Vec3::new(3.4, 0.74, 0.07);

/// Covers a round log with lengthwise straps, like plates of bark, and
/// closes its ends. `surface` gives the color at a fraction along the log.
fn log(
    out: &mut Vec<GaussianSegment>,
    rng: &mut Rng,
    (a, b): (Vec3, Vec3),
    radius: f32,
    mut surface: impl FnMut(f32, &mut Rng) -> Vec3,
    ends: [Vec3; 2],
) {
    let length = a.distance(b);
    let axis = (b - a) / length;
    let u = axis.any_orthonormal_vector();
    let v = axis.cross(u);
    let spacing = 0.011;
    let strips = (TAU * radius / spacing).ceil() as usize;
    for strip in 0..strips {
        let angle = TAU * (strip as f32 + rng.range(-0.2, 0.2)) / strips as f32;
        let out_from_axis = u * angle.cos() + v * angle.sin();
        // Logs lie on the ground or on each other, so little sky reaches
        // their undersides. A baked pass cannot do this: the straps of one
        // log overlap and would shadow each other.
        let sky = 0.42 + 0.58 * crate::landscape::smoothstep(-0.75, 0.35, out_from_axis.y);
        let mut from = 0.0;
        while from < 1.0 {
            // Plates of uneven length, a little proud of or below the round.
            let to = (from + rng.range(0.05, 0.11) / length).min(1.0);
            let rise = radius * rng.range(0.95, 1.05);
            strap(
                out,
                a + axis * (from * length) + out_from_axis * rise,
                b - axis * ((1.0 - to) * length) + out_from_axis * rise,
                out_from_axis,
                spacing * rng.range(1.9, 2.5),
                surface(0.5 * (from + to), rng) * sky,
            );
            from = to;
        }
    }
    for (end, facing, color) in [(a, -axis, ends[0]), (b, axis, ends[1])] {
        disc(out, end, facing, radius * 1.04, color);
        // Growth rings: darker heartwood, then a pale band, then the pith.
        disc(
            out,
            end + facing * 0.001,
            facing,
            radius * 0.72,
            color * 0.82,
        );
        disc(
            out,
            end + facing * 0.002,
            facing,
            radius * 0.46,
            color * 0.96,
        );
        disc(
            out,
            end + facing * 0.003,
            facing,
            radius * 0.20,
            color * 0.66,
        );
    }
}

/// Spreads ash, charcoal lumps, and live coals over the ground in the ring.
fn ash_bed(
    wood: &mut Vec<GaussianSegment>,
    glow: &mut Vec<GaussianSegment>,
    site: Vec2,
    rng: &mut Rng,
) {
    let base = ground(site);
    let place = |offset: Vec2, lift: f32| {
        Vec3::new(offset.x, ground(site + offset) - base + lift, offset.y)
    };
    for _ in 0..1100 {
        let offset = rng.range(0.0, 1.0).sqrt() * 0.52 * Vec2::from_angle(rng.range(0.0, TAU));
        let center = 1.0 - offset.length() / 0.52;
        // Pale ash in the middle, sooty soil toward the stones.
        let shade = rng.range(0.45, 1.0) * (0.2 + 0.8 * center.sqrt());
        let color = CHAR.lerp(ASH, shade) * rng.range(0.85, 1.12);
        let face = Vec3::new(rng.range(-0.25, 0.25), 1.0, rng.range(-0.25, 0.25));
        disc(
            wood,
            place(offset, rng.range(0.012, 0.03)),
            face,
            rng.range(0.012, 0.03),
            color,
        );
    }
    // Charcoal lumps: short fat straps that lie at any angle.
    for _ in 0..70 {
        let offset = rng.range(0.0, 1.0).sqrt() * 0.42 * Vec2::from_angle(rng.range(0.0, TAU));
        let at = place(offset, rng.range(0.03, 0.05));
        let along = Vec3::new(
            rng.range(-1.0, 1.0),
            rng.range(-0.25, 0.25),
            rng.range(-1.0, 1.0),
        )
        .normalize_or(Vec3::X)
            * rng.range(0.012, 0.03);
        let color = CHAR * rng.range(1.0, 3.5);
        strap(
            wood,
            at - along,
            at + along,
            Vec3::Y,
            rng.range(0.018, 0.034),
            color,
        );
    }
    // Live coals gather under the lay, in the middle of the bed.
    for _ in 0..110 {
        let offset = rng.range(0.0, 1.0).powf(0.8) * 0.26 * Vec2::from_angle(rng.range(0.0, TAU));
        let heat = (1.0 - offset.length() / 0.26) * rng.range(0.35, 1.0);
        let face = Vec3::new(rng.range(-0.5, 0.5), 1.0, rng.range(-0.5, 0.5));
        disc(
            glow,
            place(offset, rng.range(0.035, 0.075)),
            face,
            rng.range(0.008, 0.022),
            ember(heat),
        );
    }
}

/// The color of a coal, from dull red to orange-yellow.
fn ember(heat: f32) -> Vec3 {
    EMBER * Vec3::new(heat, heat * heat, heat * heat * heat) * 1.25
}

/// Leans charred logs together over the coals, with their outer ends near
/// the stones still in bark.
fn fire_lay(
    wood: &mut Vec<GaussianSegment>,
    glow: &mut Vec<GaussianSegment>,
    site: Vec2,
    rng: &mut Rng,
) {
    let base = ground(site);
    let count = 6;
    let start = rng.range(0.0, TAU);
    for index in 0..count {
        let angle = start + TAU * (index as f32 + rng.range(-0.22, 0.22)) / count as f32;
        let out_dir = Vec2::from_angle(angle);
        let radius = rng.range(0.034, 0.05);
        let foot = out_dir * rng.range(0.36, 0.44);
        let a = Vec3::new(foot.x, ground(site + foot) - base + radius + 0.02, foot.y);
        // The inner end rests on its neighbors, a little past the center.
        let top = -out_dir * rng.range(0.02, 0.10) + out_dir.perp() * rng.range(-0.06, 0.06);
        let b = Vec3::new(top.x, rng.range(0.15, 0.25), top.y);
        let burn = rng.range(0.25, 0.45);
        let mut log_rng = rng.fork();
        log(
            wood,
            &mut log_rng,
            (a, b),
            radius,
            |along, rng| {
                // Bark survives near the stones. The rest is checked char
                // with patches of ash that thicken toward the flames.
                let charred = crate::landscape::smoothstep(burn - 0.15, burn + 0.1, along);
                let bark = BARK * rng.range(0.6, 1.3);
                let ash = rng.next() < 0.08 + 0.3 * along * along;
                let char = if ash {
                    ASH * rng.range(0.5, 1.1)
                } else {
                    CHAR * rng.range(0.6, 2.4)
                };
                bark.lerp(char, charred)
            },
            [SAPWOOD * 0.8, CHAR * 2.0],
        );
        // Glowing checks in the char on the inner half of each log.
        let axis = (b - a).normalize();
        let u = axis.any_orthonormal_vector();
        let v = axis.cross(u);
        for _ in 0..16 {
            let along = rng.range(0.5, 0.98);
            let around = rng.range(0.0, TAU);
            let out_from_axis = u * around.cos() + v * around.sin();
            // Coals glow most on the underside, which faces the bed.
            let heat = (0.65 - 0.35 * out_from_axis.y) * rng.range(0.4, 1.0) * along;
            let at = a.lerp(b, along) + out_from_axis * (radius * 1.07);
            let half = axis * rng.range(0.006, 0.02);
            strap(
                glow,
                at - half,
                at + half,
                out_from_axis,
                rng.range(0.005, 0.011),
                ember(heat),
            );
        }
    }
}

/// Stacks split and round logs beside the ring: three below and two above.
fn firewood(wood: &mut Vec<GaussianSegment>, site: Vec2, stack: Vec2, away: Vec2, rng: &mut Rng) {
    let base = ground(site);
    let along = away.perp();
    let mut rows = Vec::new();
    for row in 0..3 {
        rows.push((row as f32 - 1.0, 0, rng.range(0.050, 0.062)));
    }
    for row in 0..2 {
        rows.push((row as f32 - 0.5, 1, rng.range(0.046, 0.056)));
    }
    for (column, tier, radius) in rows {
        let length = rng.range(0.40, 0.50);
        let skew = rng.range(-0.09, 0.09);
        let center = stack + away * (column * 0.118) + along * rng.range(-0.04, 0.04);
        let direction = (along + away * skew).normalize();
        let lift = ground(center) - base + radius + tier as f32 * 0.098;
        let middle = Vec3::new(center.x - site.x, lift, center.y - site.y);
        let half = Vec3::new(direction.x, 0.0, direction.y) * (0.5 * length);
        let tone = rng.range(0.75, 1.2);
        let mut log_rng = rng.fork();
        log(
            wood,
            &mut log_rng,
            (middle - half, middle + half),
            radius,
            |_, rng| {
                // Mostly bark, with a few pale scars where it has peeled.
                if rng.next() < 0.05 {
                    SAPWOOD * rng.range(0.5, 0.8)
                } else {
                    BARK * tone * rng.range(0.6, 1.35)
                }
            },
            [SAPWOOD * tone, HEARTWOOD.lerp(SAPWOOD, 0.5) * tone],
        );
    }
}

/// Soft patches of light over the coal bed.
fn coal_glow(rng: &mut Rng) -> Vec<GaussianSegment> {
    (0..22)
        .map(|_| {
            let offset = rng.range(0.0, 1.0).sqrt() * 0.2 * Vec2::from_angle(rng.range(0.0, TAU));
            let at = Vec3::new(offset.x, rng.range(0.0, 0.05), offset.y);
            let sigma = rng.range(0.03, 0.06);
            GaussianSegment::new(
                at,
                at,
                Vec3::new(sigma, sigma * 0.5, sigma),
                Quat::IDENTITY,
                // A peak optical depth near 0.35.
                0.35 * TAU * sigma * sigma,
                Vec3::new(0.95, 0.24, 0.03) * rng.range(0.6, 1.2),
            )
        })
        .collect()
}

/// Upright flame segments that start over a disc of the bed.
fn flames(
    rng: &mut Rng,
    count: usize,
    radius: f32,
    lengths: std::ops::Range<f32>,
    widths: std::ops::Range<f32>,
    color: Vec3,
) -> Vec<GaussianSegment> {
    (0..count)
        .map(|_| {
            let offset =
                rng.range(0.0, 1.0).sqrt() * radius * Vec2::from_angle(rng.range(0.0, TAU));
            let foot = Vec3::new(offset.x, rng.range(0.0, 0.07), offset.y);
            let length = rng.range(lengths.start, lengths.end);
            let sigma = rng.range(widths.start, widths.end);
            let lean = Vec3::new(rng.range(-0.2, 0.2), 1.0, rng.range(-0.2, 0.2)) * length;
            // Hotter, yellower flames mix with cooler orange ones.
            let heat = rng.range(0.0, 1.0);
            GaussianSegment::new(
                foot,
                foot + lean,
                Vec3::splat(sigma),
                Quat::IDENTITY,
                // A peak optical depth near 0.4 across the segment.
                0.4 * 2.5066 * sigma * length,
                color * Vec3::new(1.0, 0.72 + 0.38 * heat, 0.5 + 0.9 * heat) * rng.range(0.7, 1.1),
            )
        })
        .collect()
}

fn sparks(rng: &mut Rng) -> Vec<GaussianSegment> {
    (0..18)
        .map(|_| {
            let offset = rng.range(0.0, 1.0).sqrt() * 0.12 * Vec2::from_angle(rng.range(0.0, TAU));
            let at = Vec3::new(offset.x, rng.range(0.15, 0.4), offset.y);
            let sigma = rng.range(0.002, 0.0035);
            GaussianSegment::new(
                at,
                // A short streak along the rise.
                at + Vec3::Y * rng.range(0.01, 0.03),
                Vec3::splat(sigma),
                Quat::IDENTITY,
                2.0 * 2.5066 * sigma * 0.02,
                Vec3::new(2.4, 0.9, 0.2) * rng.range(0.5, 1.2),
            )
        })
        .collect()
}

fn smoke(rng: &mut Rng) -> Vec<GaussianSegment> {
    (0..96)
        .map(|_| {
            let offset = rng.range(0.0, 1.0).sqrt() * 0.10 * Vec2::from_angle(rng.range(0.0, TAU));
            let at = Vec3::new(offset.x, rng.range(0.32, 0.6), offset.y);
            let sigma = rng.range(0.07, 0.11);
            // Wood smoke is blue-gray where it is thin and brown where thick.
            let gray = rng.range(0.24, 0.36);
            GaussianSegment::new(
                at,
                at,
                Vec3::new(sigma, sigma * rng.range(0.8, 1.3), sigma),
                Quat::IDENTITY,
                // A peak optical depth near 0.45 before the puff swells.
                0.45 * TAU * sigma * sigma,
                Vec3::new(gray, gray * 1.02, gray * rng.range(1.02, 1.12)),
            )
        })
        .collect()
}
