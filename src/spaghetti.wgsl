// Gaussian segment splatting, Shramko et al., sections 3.2–3.5.
struct Splat {
    start_mass: vec4<f32>,
    end: vec4<f32>,
    sigma: vec4<f32>,
    rotation: vec4<f32>,
    color: vec4<f32>,
}

struct View {
    clip_from_local: mat4x4<f32>,
    viewport: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

struct Raster {
    @builtin(position) position: vec4<f32>,
    @location(0) offset: vec2<f32>,
    @location(1) @interpolate(flat) segment: vec2<f32>,
    @location(2) @interpolate(flat) depth: vec2<f32>,
    @location(3) @interpolate(flat) color_mass: vec4<f32>,
}

fn rotate(q: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, p);
    return p + q.w * t + cross(q.xyz, t);
}

// Derivative of homogeneous divide, in pixels per local coordinate.
fn project_axis(axis: vec4<f32>, midpoint: vec4<f32>, half_viewport: vec2<f32>) -> vec2<f32> {
    return (axis.xy - midpoint.xy / midpoint.w * axis.w) * half_viewport / midpoint.w;
}

// Restrict the parameter interval to the positive half-space of a clip plane.
fn clip_interval(range: vec2<f32>, a: f32, b: f32) -> vec2<f32> {
    if a < 0.0 && b < 0.0 {
        return vec2(1.0, 0.0);
    }
    if a < 0.0 {
        return vec2(max(range.x, a / (a - b)), range.y);
    }
    if b < 0.0 {
        return vec2(range.x, min(range.y, a / (a - b)));
    }
    return range;
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance: u32) -> Raster {
    let splat = splats[order[instance]];
    let transform = view.clip_from_local;
    let start = transform * vec4(splat.start_mass.xyz, 1.0);
    let end = transform * vec4(splat.end.xyz, 1.0);

    var out: Raster;
    // Bevy uses reverse Z. Clip the centerline to 0 <= z <= w before dividing.
    let near = clip_interval(vec2(0.0, 1.0), start.w - start.z, end.w - end.z);
    let span = clip_interval(near, start.z, end.z);
    let a = mix(start, end, span.x);
    let b = mix(start, end, span.y);
    if span.x >= span.y || min(a.w, b.w) <= 0.0 || splat.start_mass.w == 0.0 {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }

    let half_viewport = view.viewport.xy * 0.5;
    let midpoint = (a + b) * 0.5;
    let jx = project_axis(transform[0], midpoint, half_viewport);
    let jy = project_axis(transform[1], midpoint, half_viewport);
    let jz = project_axis(transform[2], midpoint, half_viewport);
    let j0 = vec3(jx.x, jy.x, jz.x);
    let j1 = vec3(jx.y, jy.y, jz.y);

    // Each projected principal axis contributes an outer product to J Σ Jᵀ.
    let q = splat.rotation;
    let s = splat.sigma.xyz;
    let axis_x = rotate(q, vec3(s.x, 0.0, 0.0));
    let axis_y = rotate(q, vec3(0.0, s.y, 0.0));
    let axis_z = rotate(q, vec3(0.0, 0.0, s.z));
    let row_x = vec3(dot(j0, axis_x), dot(j0, axis_y), dot(j0, axis_z));
    let row_y = vec3(dot(j1, axis_x), dot(j1, axis_y), dot(j1, axis_z));
    let xx = dot(row_x, row_x);
    let xy = dot(row_x, row_y);
    let yy = dot(row_y, row_y);
    // Add a pixel-space Gaussian filter. Normalizing by the filtered
    // determinant preserves optical mass as a splat shrinks below one pixel.
    let pixel_variance = 0.25;
    let cxx = xx + pixel_variance;
    let cyy = yy + pixel_variance;
    let cross_row = cross(row_x, row_y);
    let determinant = dot(cross_row, cross_row) + pixel_variance * (xx + yy + pixel_variance);
    let sigma_x = sqrt(cxx);
    let sigma_y_given_x = sqrt(determinant / cxx);

    let a_ndc = a.xyz / a.w;
    let b_ndc = b.xyz / b.w;
    let a_px = a_ndc.xy * half_viewport;
    let segment = (b_ndc.xy - a_ndc.xy) * half_viewport;
    let length_px = length(segment);
    let tangent = select(vec2(1.0, 0.0), segment / max(length_px, 1e-6), length_px > 1e-6);
    let perpendicular = vec2(-tangent.y, tangent.x);
    let cov_t = cxx * tangent.x * tangent.x + 2.0 * xy * tangent.x * tangent.y + cyy * tangent.y * tangent.y;
    let cov_p = cxx * perpendicular.x * perpendicular.x + 2.0 * xy * perpendicular.x * perpendicular.y
        + cyy * perpendicular.y * perpendicular.y;
    let extent = 4.0 * sqrt(max(vec2(cov_t, cov_p), vec2(pixel_variance)));

    let corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
    );
    let corner = corners[vertex_index];
    let v = tangent * (max(corner.x, 0.0) * length_px + corner.x * extent.x)
          + perpendicular * (corner.y * extent.y);

    // Cholesky whitening lets the fragment shader integrate an isotropic
    // unit Gaussian along the whitened centerline.
    let shear = xy / cxx;
    out.offset = vec2(v.x / sigma_x, (v.y - shear * v.x) / sigma_y_given_x);
    out.segment = vec2(segment.x / sigma_x, (segment.y - shear * segment.x) / sigma_y_given_x);
    out.depth = vec2(a_ndc.z, b_ndc.z);
    let area_scale = length(cross(j0, j1));
    let peak = (splat.start_mass.w * (span.y - span.x) * area_scale) / (6.28318530718 * sqrt(determinant));
    out.color_mass = vec4(splat.color.rgb, peak);
    out.position = vec4((a_px + v) / half_viewport, 0.5, 1.0);
    return out;
}

// Abramowitz & Stegun 7.1.26, in complementary form to keep same-tail
// differences accurate when both endpoints are beyond the Gaussian center.
fn erfc_positive(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t;
    return polynomial * exp(-x * x);
}

fn erf_difference(lo: f32, hi: f32) -> f32 {
    if lo >= 0.0 {
        return erfc_positive(lo) - erfc_positive(hi);
    }
    if hi <= 0.0 {
        return erfc_positive(-hi) - erfc_positive(-lo);
    }
    return 2.0 - erfc_positive(-lo) - erfc_positive(hi);
}

struct Fragment {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fragment(in: Raster) -> Fragment {
    let length_sigma = length(in.segment);
    var along = 0.5;
    var average: f32;
    if length_sigma < 0.01 {
        // Point limit of equation 19, also avoids cancellation for short segments.
        let displacement = in.offset - 0.5 * in.segment;
        average = exp(-0.5 * dot(displacement, displacement));
    } else {
        let axis = in.segment / length_sigma;
        let projected = dot(in.offset, axis);
        let normal = dot(in.offset, vec2(-axis.y, axis.x));
        along = clamp(projected / length_sigma, 0.0, 1.0);
        let lo = -projected * 0.70710678118;
        let hi = (length_sigma - projected) * 0.70710678118;
        average = exp(-0.5 * normal * normal) * (1.25331413732 / length_sigma) * erf_difference(lo, hi);
    }
    let optical_depth = in.color_mass.a * max(average, 0.0);
    let alpha = 1.0 - exp(-optical_depth);
    if alpha < 0.0005 {
        discard;
    }
    return Fragment(vec4(in.color_mass.rgb * alpha, alpha), clamp(mix(in.depth.x, in.depth.y, along), 0.0, 1.0));
}
