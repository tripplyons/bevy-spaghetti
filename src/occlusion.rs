//! Projected grids for a small, fixed set of parallel ray directions.

use std::f64::consts::{FRAC_1_SQRT_2, TAU};

use bevy_math::{DVec2, DVec3};

use crate::GaussianSegment;

// Omit negligible tails outside the projected grid and truncate the
// quadrature interval to where the Gaussian contributes.
const SUPPORT: f64 = 3.5;
const AMBIENT_RAYS: usize = 6;

pub(crate) fn bake(
    segments: &[GaussianSegment],
    sun: DVec3,
    density: f64,
    output: &mut [[f32; 2]],
) {
    for ray in 0..AMBIENT_RAYS {
        // Equal solid angle samples on the local upper hemisphere.
        let y = (ray as f64 + 0.5) / AMBIENT_RAYS as f64;
        let angle = ray as f64 * 2.399963229728653; // Golden angle.
        let horizontal = (1.0 - y * y).sqrt();
        let direction = DVec3::new(horizontal * angle.cos(), y, horizontal * angle.sin());
        trace(segments, direction, density, |index, transmission| {
            output[index][0] += transmission / AMBIENT_RAYS as f32;
        });
    }
    trace(segments, sun, density, |index, transmission| {
        output[index][1] = transmission;
    });
}

struct ProjectedSegment {
    start: DVec3,
    min: DVec3,
    max: DVec3,
    // Maps projected XY separation into the kernel's whitened plane.
    column_x: DVec2,
    column_y: DVec2,
    // Maps XYZ separation to the conditional mean depth in sigma units.
    depth_gradient: DVec3,
    axis: DVec2,
    length: f64,
    depth_span: f64,
    peak: f64,
}

impl ProjectedSegment {
    fn new(segment: &GaussianSegment, x: DVec3, y: DVec3, ray: DVec3) -> Self {
        let project = |p: DVec3| DVec3::new(x.dot(p), y.dot(p), ray.dot(p));
        let start = project(segment.start().as_dvec3());
        let delta = project(segment.end().as_dvec3() - segment.start().as_dvec3());
        let end = start + delta;
        let sigma = segment.sigma().as_dvec3();
        let unrotate = segment.rotation().as_dquat().normalize().conjugate();
        let kernel_x = unrotate * x;
        let kernel_y = unrotate * y;
        let kernel_ray = unrotate * ray;
        let support = SUPPORT
            * DVec3::new(
                (kernel_x * sigma).length(),
                (kernel_y * sigma).length(),
                (kernel_ray * sigma).length(),
            );

        let wx = kernel_x / sigma;
        let wy = kernel_y / sigma;
        let wz = kernel_ray / sigma;
        let depth_axis = wz.normalize();
        let u = depth_axis.any_orthonormal_vector();
        let v = depth_axis.cross(u);
        let column_x = DVec2::new(u.dot(wx), v.dot(wx));
        let column_y = DVec2::new(u.dot(wy), v.dot(wy));
        let along = column_x * delta.x + column_y * delta.y;
        let length = along.length();
        let depth_gradient = DVec3::new(depth_axis.dot(wx), depth_axis.dot(wy), wz.length());
        Self {
            start,
            min: start.min(end) - support,
            max: start.max(end) + support,
            column_x,
            column_y,
            depth_gradient,
            axis: if length > 1e-5 {
                along / length
            } else {
                DVec2::X
            },
            length,
            depth_span: depth_gradient.dot(delta),
            peak: segment.mass() as f64 / (TAU * sigma.x * sigma.y * sigma.z * wz.length()),
        }
    }

    fn optical_depth(&self, origin: DVec3) -> f64 {
        if origin.z > self.max.z
            || origin.x < self.min.x
            || origin.x > self.max.x
            || origin.y < self.min.y
            || origin.y > self.max.y
        {
            return 0.0;
        }
        let offset = origin - self.start;
        let projected = self.column_x * offset.x + self.column_y * offset.y;
        let along = self.axis.dot(projected);
        let across = self.axis.perp_dot(projected);
        if across.abs() > SUPPORT {
            return 0.0;
        }

        let length = self.length;
        let (lo, hi) = if length < 1e-5 {
            if along.abs() > SUPPORT {
                return 0.0;
            }
            (0.0, 1.0)
        } else {
            (
                ((along - SUPPORT) / length).max(0.0),
                ((along + SUPPORT) / length).min(1.0),
            )
        };
        if hi <= lo {
            return 0.0;
        }

        // At each point on the source segment the half-ray integral is a
        // projected Gaussian times a normal CDF for its forward half.
        let depth = -self.depth_gradient.dot(offset);
        let depth_at = |t: f64| depth + self.depth_span * t;
        let full_line = |a: f64, b: f64| {
            if length < 1e-5 {
                (b - a) * (-0.5 * along * along).exp()
            } else {
                (std::f64::consts::FRAC_PI_2.sqrt() / length)
                    * (normal_cdf(length * b - along) - normal_cdf(length * a - along))
                    * 2.0
            }
        };
        let front = depth_at(lo).min(depth_at(hi));
        let back = depth_at(lo).max(depth_at(hi));
        let integral = if front >= SUPPORT {
            full_line(lo, hi)
        } else if back <= -SUPPORT {
            return 0.0;
        } else if self.depth_span.abs() < 0.1 {
            full_line(lo, hi) * normal_cdf(depth_at((lo + hi) * 0.5))
        } else {
            let enters = ((-SUPPORT - depth) / self.depth_span).clamp(lo, hi);
            let leaves = ((SUPPORT - depth) / self.depth_span).clamp(lo, hi);
            let (a, b, full) = if self.depth_span > 0.0 {
                (enters, leaves, full_line(leaves, hi))
            } else {
                (leaves, enters, full_line(lo, leaves))
            };
            // Only the portion crossing the ray origin needs numerical
            // integration over the segment. The ray integral is closed form.
            let center = (a + b) * 0.5;
            let step = (b - a) * 0.5;
            let point = |t: f64| {
                let distance = length * t - along;
                (-0.5 * distance * distance).exp() * normal_cdf(depth_at(t))
            };
            let node = step * 0.7745966692414834;
            full + step
                * (8.0 * point(center) + 5.0 * (point(center - node) + point(center + node)))
                / 9.0
        };
        self.peak * (-0.5 * across * across).exp() * integral
    }
}

fn normal_cdf(x: f64) -> f64 {
    // Abramowitz and Stegun 7.1.26, also used for projected segment coverage.
    let x_abs = x.abs() * FRAC_1_SQRT_2;
    let t = (1.0 + 0.3275911 * x_abs).recip();
    let tail = 0.5
        * (-x_abs * x_abs).exp()
        * t
        * ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592);
    if x < 0.0 { tail } else { 1.0 - tail }
}

fn trace(
    segments: &[GaussianSegment],
    ray: DVec3,
    density: f64,
    mut record: impl FnMut(usize, f32),
) {
    let x = ray.any_orthonormal_vector();
    let y = ray.cross(x);
    let kernels: Vec<_> = segments
        .iter()
        .map(|segment| ProjectedSegment::new(segment, x, y, ray))
        .collect();
    let mut min = DVec2::splat(f64::INFINITY);
    let mut max = DVec2::splat(f64::NEG_INFINITY);
    let mut total_size = DVec2::ZERO;
    for kernel in &kernels {
        min = min.min(kernel.min.truncate());
        max = max.max(kernel.max.truncate());
        total_size += (kernel.max - kernel.min).truncate();
    }
    let size = max - min;
    let resolution = ((size * kernels.len() as f64) / total_size)
        .min(DVec2::splat((kernels.len() as f64).sqrt().min(128.0)))
        .max(DVec2::ONE)
        .as_uvec2();
    let scale = resolution.as_dvec2() / size.max(DVec2::splat(f64::MIN_POSITIVE));
    let cell = |point: DVec3| (((point.truncate() - min) * scale).as_uvec2()).min(resolution - 1);
    let mut grid = vec![Vec::<u32>::new(); (resolution.x * resolution.y) as usize];
    for (index, kernel) in kernels
        .iter()
        .enumerate()
        .filter(|(_, kernel)| kernel.peak > 0.0)
    {
        let low = cell(kernel.min);
        let high = cell(kernel.max);
        for y in low.y..=high.y {
            for x in low.x..=high.x {
                grid[(y * resolution.x + x) as usize].push(index as u32);
            }
        }
    }
    for (index, segment) in segments.iter().enumerate() {
        let center = (segment.start().as_dvec3() + segment.end().as_dvec3()) * 0.5;
        let origin = DVec3::new(x.dot(center), y.dot(center), ray.dot(center));
        let cell = cell(origin);
        let mut depth = 0.0;
        for &other in &grid[(cell.y * resolution.x + cell.x) as usize] {
            if other as usize != index {
                depth += density * kernels[other as usize].optical_depth(origin);
                if depth >= 8.0 {
                    break;
                }
            }
        }
        record(index, (-depth).exp() as f32);
    }
}
