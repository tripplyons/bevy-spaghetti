//! Builders for the kernels that the cutout renderer draws as solid shapes.
//!
//! The renderer cuts each kernel where its optical depth falls below one.
//! The masses here put that edge at the stated width or radius.

use bevy::prelude::*;
use bevy_spaghetti::GaussianSegment;

/// The cutout edge of a strap sits this many standard deviations from its axis.
const STRAP_EDGE: f32 = 1.525;
/// The cutout edge of a disc sits this many standard deviations from its center.
const DISC_EDGE: f32 = 1.52;

fn band(
    a: Vec3,
    b: Vec3,
    face: Vec3,
    width: f32,
    depth: f32,
    color: Vec3,
) -> Option<GaussianSegment> {
    let length = a.distance(b);
    if length < 1e-5 {
        return None;
    }
    let tangent = (b - a) / length;
    let normal = (face - tangent * face.dot(tangent))
        .try_normalize()
        .unwrap_or_else(|| tangent.any_orthonormal_vector());
    let sigma = 0.5 * width / STRAP_EDGE;
    let extent = Vec3::new(sigma, sigma * 0.5, sigma * depth);
    Some(GaussianSegment::new(
        a,
        b,
        extent,
        Quat::from_mat3(&Mat3::from_cols(tangent.cross(normal), tangent, normal)),
        8.0 * sigma * length.max(2.5066 * extent.y),
        color,
    ))
}

/// A flat strap from `a` to `b` that faces `face` and is `width` wide.
pub fn strap(
    out: &mut Vec<GaussianSegment>,
    a: Vec3,
    b: Vec3,
    face: Vec3,
    width: f32,
    color: Vec3,
) {
    out.extend(band(a, b, face, width, 0.3, color));
}

/// A flat strap that narrows to `taper` times its width at `b`.
pub fn tapered(
    out: &mut Vec<GaussianSegment>,
    (a, b): (Vec3, Vec3),
    face: Vec3,
    width: f32,
    taper: f32,
    color: Vec3,
) {
    out.extend(band(a, b, face, width, 0.3, color).map(|strap| strap.with_cutout_taper(taper)));
}

/// A round rod from `a` to `b`. It has the same width from every side, and
/// the light falls on it as on a surface that faces `face`.
pub fn rod(out: &mut Vec<GaussianSegment>, a: Vec3, b: Vec3, face: Vec3, radius: f32, color: Vec3) {
    out.extend(band(a, b, face, 2.0 * radius, 1.0, color));
}

/// A round disc of visible radius `radius` that faces `face`.
pub fn disc(out: &mut Vec<GaussianSegment>, at: Vec3, face: Vec3, radius: f32, color: Vec3) {
    blob(out, at, face, radius, 0.4, color);
}

/// A disc with a thickness of `depth` times its radius. A depth of one gives
/// a ball.
pub fn blob(
    out: &mut Vec<GaussianSegment>,
    at: Vec3,
    face: Vec3,
    radius: f32,
    depth: f32,
    color: Vec3,
) {
    let sigma = radius / DISC_EDGE;
    out.push(GaussianSegment::new(
        at,
        at,
        Vec3::new(sigma, sigma, sigma * depth),
        Quat::from_rotation_arc(Vec3::Z, face.normalize_or(Vec3::Y)),
        sigma * sigma * 21.0,
        color,
    ));
}
