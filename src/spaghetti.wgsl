// Gaussian segment splatting, Shramko et al., sections 3.2–3.5.
struct Splat {
    start_mass: vec4<f32>,
    end: vec4<f32>,
    sigma: vec4<f32>,
    rotation: vec4<f32>,
    color: vec4<f32>,
}

struct FogLight {
    direction: vec4<f32>,
    color: vec4<f32>,
}

struct Fog {
    color: vec4<f32>,
    extinction: vec3<f32>,
    mode: u32,
    inscattering: vec3<f32>,
    directional_exponent: f32,
    light_count: vec4<u32>,
    lights: array<FogLight, 10>,
}

struct View {
    clip_from_local: mat4x4<f32>,
    viewport: vec4<f32>,
    normal_from_local: mat3x3<f32>,
    direction_to_light: vec4<f32>,
    ambient: vec4<f32>,
    directional: vec4<f32>,
    eye_relative_from_local: mat4x4<f32>,
    fog: Fog,
    world_from_local: mat4x4<f32>,
    local_from_world: mat3x3<f32>,
    pusher: vec4<f32>,
    motion: vec4<f32>,
    detail_depths: array<vec4<f32>, 3>,
    detail_ends: vec4<u32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read> occlusion: array<u32>;

override SHADOW_QUAD: bool = false;
override BAKED_OCCLUSION: bool = false;

struct SunShadows {
    matrices: array<mat4x4<f32>, 4>,
    far_bounds: vec4<f32>,
    normal_bias: vec4<f32>,
    view_depth: vec4<f32>,
    depth_bias: vec4<f32>,
    layers: vec4<u32>,
}

@group(1) @binding(0) var<uniform> sun_shadow: SunShadows;
@group(1) @binding(1) var sun_depth: texture_depth_2d_array;
@group(1) @binding(2) var sun_compare: sampler_comparison;

fn scene_shadow(midpoint: vec3<f32>, normal: vec3<f32>) -> f32 {
    let world = view.world_from_local * vec4(midpoint, 1.0);
    let depth = dot(sun_shadow.view_depth, world);
    let beyond = select(vec4<u32>(0u), vec4<u32>(1u), vec4(depth) >= sun_shadow.far_bounds);
    let cascade = beyond.x + beyond.y + beyond.z + beyond.w;
    if cascade >= sun_shadow.layers.y {
        return 1.0;
    }
    let offset = normal * sun_shadow.normal_bias[cascade] + sun_shadow.depth_bias.xyz;
    // Directional cascades use an orthographic transform and reverse Z.
    let ndc = (sun_shadow.matrices[cascade] * vec4(world.xyz + offset, 1.0)).xyz;
    if any(abs(ndc.xy) >= vec2(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let uv = ndc.xy * vec2(0.5, -0.5) + 0.5;
    return textureSampleCompareLevel(sun_depth, sun_compare, uv, sun_shadow.layers.x + cascade, ndc.z);
}

fn segment_light(index: u32, midpoint: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    var light = vec2(1.0);
    if BAKED_OCCLUSION {
        light = unpack2x16unorm(occlusion[index]);
    }
    let facing = dot(normal, view.direction_to_light.xyz);
    // Thin foliage passes part of the light through from its back face.
    let incident = max(0.0, facing) + view.ambient.w * max(0.0, -facing);
    if SHADOW_QUAD && incident > 0.0 {
        light.y *= scene_shadow(midpoint, select(-normal, normal, facing >= 0.0));
    }
    return light.x * view.ambient.rgb + (incident * light.y) * view.directional.rgb;
}

struct Raster {
    @builtin(position) position: vec4<f32>,
    @location(0) offset: vec2<f32>,
    @location(1) @interpolate(flat) segment: vec2<f32>,
    @location(2) @interpolate(flat) depth: vec2<f32>,
    @location(3) @interpolate(flat, first) color_mass: vec4<f32>,
    @location(4) @interpolate(flat) coverage: f32,
    @location(5) @interpolate(flat) detail: vec2<f32>,
}

// At each transition, the finer level keeps the thresholds below its
// coverage and the coarser level keeps the rest.
fn detail_interval(index: u32, depth: f32) -> vec2<f32> {
    var lower = 0.0;
    for (var i = 0u; i < view.detail_ends.w; i += 1u) {
        let edges = view.detail_depths[i].xy;
        var fine = 1.0;
        if depth >= edges.y {
            fine = 0.0;
        } else if depth > edges.x {
            fine = 1.0 - smoothstep(edges.x, edges.y, depth);
        }
        if index < view.detail_ends[i] {
            return vec2(lower, fine);
        }
        lower = fine;
    }
    return vec2(lower, 1.0);
}

fn screen_threshold(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(floor(pixel), vec2(0.06711056, 0.00583715))));
}

fn detail_visible(pixel: vec2<f32>, interval: vec2<f32>) -> bool {
    if interval.x == 0.0 && interval.y == 1.0 {
        return true;
    }
    // Offset the pattern from the cutout's visibility-range mask.
    let threshold = screen_threshold(pixel + vec2(17.0, 29.0));
    return threshold >= interval.x && threshold < interval.y;
}

// Opacity factor for segments close to the camera; viewport.zw hold the fade
// start and end view depths, and equal values turn it off.
fn near_fade(depth: f32) -> f32 {
    if view.viewport.w <= view.viewport.z {
        return 1.0;
    }
    return smoothstep(view.viewport.z, view.viewport.w, depth);
}

fn rotate(q: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, p);
    return p + q.w * t + cross(q.xyz, t);
}

fn grass_hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2(0.1031, 0.1030));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

fn gust_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let w = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(grass_hash(cell), grass_hash(cell + vec2(1.0, 0.0)), w.x),
        mix(grass_hash(cell + vec2(0.0, 1.0)), grass_hash(cell + vec2(1.0, 1.0)), w.x),
        w.y,
    );
}

fn blade_profile(packed: f32) -> vec2<f32> {
    let bits = bitcast<u32>(packed);
    if (bits & 0xff800000u) != 0x3f800000u {
        return vec2(0.0, 1.0);
    }
    return vec2(f32((bits >> 10u) & 8191u) * 0.001, f32(bits & 1023u) / 1023.0);
}

fn bend_endpoint(local: vec3<f32>, rise: f32, height: f32) -> vec3<f32> {
    if view.motion.y == 0.0 || !(rise > 0.0) {
        return local;
    }
    let world = (view.world_from_local * vec4(local, 1.0)).xyz;
    let at = world.xz;
    let time = view.motion.x;
    let front = sin(dot(at, vec2(-0.29, 0.19)) + time * 1.8);
    let pocket = gust_noise(at * 0.12 - vec2(0.82, 0.32) * time);
    let sway = vec2(-0.30, 0.70) * (0.10 * sin(time * 1.34 + at.x * 0.22));
    let gust = vec2(0.82, 0.47) * (0.14 + 0.50 * pocket * pocket + 0.17 * front) + sway;
    let flutter = sin(time * 6.3 + grass_hash(floor(at * 9.0)) * 6.28) * 0.06;
    let fraction = clamp(rise / max(height, 1e-5), 0.0, 1.0);
    var slope = (gust * 0.55 + vec2(flutter, flutter * 0.6)) * fraction;

    if view.pusher.w > 0.0 {
        let away = at - view.pusher.xz;
        let distance = length(away);
        let contact = max(1.0 - distance / view.pusher.w, 0.0);
        let below = step(world.y - max(height, rise) - 0.25, view.pusher.y);
        slope += away / max(distance, 0.05) * contact * contact * 1.4 * below;
    }
    let length_slope = length(slope);
    let reach = min(length_slope, 0.92);
    let offset = slope * (reach / max(length_slope, 1e-5));
    let down = 1.0 - sqrt(1.0 - reach * reach);
    return local + view.local_from_world * (vec3(offset.x, -down, offset.y) * rise);
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

// Bevy PBR's distance falloff, evaluated at the clipped centerline midpoint.
fn apply_distance_fog(color: vec3<f32>, local_midpoint: vec3<f32>) -> vec3<f32> {
    let eye_to_splat = (view.eye_relative_from_local * vec4(local_midpoint, 1.0)).xyz;
    let distance = length(eye_to_splat);
    var fog_color = view.fog.color.rgb;
    if distance > 0.0 && view.fog.light_count.x > 0u {
        let ray = eye_to_splat / distance;
        for (var i = 0u; i < view.fog.light_count.x; i += 1u) {
            let light = view.fog.lights[i];
            let alignment = max(dot(ray, light.direction.xyz), 0.0);
            fog_color += pow(alignment, view.fog.directional_exponent) * light.color.rgb;
        }
    }
    let strength = view.fog.color.a;
    let be = view.fog.extinction;
    switch view.fog.mode {
        case 1u: {
            let amount = 1.0 - clamp((be.y - distance) / (be.y - be.x), 0.0, 1.0);
            return mix(color, fog_color, amount * strength);
        }
        case 2u: {
            return mix(color, fog_color, (1.0 - exp(-distance * be.x)) * strength);
        }
        case 3u: {
            let scaled = distance * be.x;
            return mix(color, fog_color, (1.0 - exp(-scaled * scaled)) * strength);
        }
        case 4u: {
            let extinction = 1.0 - exp(-distance * be);
            let inscattering = 1.0 - exp(-distance * view.fog.inscattering);
            return color * (1.0 - extinction * strength) + fog_color * inscattering * strength;
        }
        default: {
            return color;
        }
    }
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance: u32) -> Raster {
    let index = order[instance];
    let splat = splats[index];
    let transform = view.clip_from_local;
    var local_start = splat.start_mass.xyz;
    var local_end = splat.end.xyz;
    if view.motion.y > 0.0 {
        let height = blade_profile(splat.color.w).x;
        local_start = bend_endpoint(local_start, splat.end.w, height);
        local_end = bend_endpoint(local_end, splat.sigma.w, height);
    }
    let start = transform * vec4(local_start, 1.0);
    let end = transform * vec4(local_end, 1.0);

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

    let midpoint = (a + b) * 0.5;
    out.detail = detail_interval(index, midpoint.w);
    if out.detail.x >= out.detail.y {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }
    let half_viewport = view.viewport.xy * 0.5;
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

    let corners = array<vec2<f32>, 4>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0),
        vec2(-1.0, 1.0), vec2(1.0, 1.0)
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
    out.color_mass = vec4(0.0);
    if !SHADOW_QUAD || vertex_index == 1u {
        let area_scale = length(cross(j0, j1));
        let peak = (splat.start_mass.w * (span.y - span.x) * area_scale) / (6.28318530718 * sqrt(determinant));
        var color = splat.color.rgb;
        if view.direction_to_light.w > 0.0 {
            let normal = view.normal_from_local * rotate(q, vec3(0.0, 0.0, 1.0));
            let unit = normal * inverseSqrt(max(1e-10, dot(normal, normal)));
            color *= segment_light(index, (local_start + local_end) * 0.5, unit);
        }
        if view.fog.mode != 0u {
            let local_midpoint = mix(local_start, local_end, (span.x + span.y) * 0.5);
            color = apply_distance_fog(color, local_midpoint);
        }
        // The fragment stage premultiplies the fogged color by optical alpha.
        out.color_mass = vec4(color, peak * near_fade(midpoint.w));
    }
    out.coverage = view.motion.z;
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
    if !detail_visible(in.position.xy, in.detail) {
        discard;
    }
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
    let alpha = (1.0 - exp(-optical_depth)) * in.coverage;
    if alpha < 0.0005 {
        discard;
    }
    return Fragment(vec4(in.color_mass.rgb * alpha, alpha), clamp(mix(in.depth.x, in.depth.y, along), 0.0, 1.0));
}

struct CutoutRaster {
    @builtin(position) position: vec4<f32>,
    @location(0) coordinate: vec2<f32>,
    @location(1) @interpolate(flat) shape: vec4<f32>,
    @location(2) @interpolate(flat, first) color: vec3<f32>,
    @location(3) @interpolate(flat) profile: vec4<f32>,
    @location(4) @interpolate(flat) detail: vec2<f32>,
}

// Evaluate a Gaussian cutout in projected segment coordinates. Fixed-function
// depth interpolates along the segment, so hidden grass can be rejected early.
@vertex
fn cutout_vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance: u32,
) -> CutoutRaster {
    let splat = splats[instance];
    let profile = blade_profile(splat.color.w);
    let local_a = bend_endpoint(splat.start_mass.xyz, splat.end.w, profile.x);
    let local_b = bend_endpoint(splat.end.xyz, splat.sigma.w, profile.x);
    let transform = view.clip_from_local;
    let start = transform * vec4(local_a, 1.0);
    let end = transform * vec4(local_b, 1.0);
    let near = clip_interval(vec2(0.0, 1.0), start.w - start.z, end.w - end.z);
    let span = clip_interval(near, start.z, end.z);
    let a = mix(start, end, span.x);
    let b = mix(start, end, span.y);

    var out: CutoutRaster;
    if span.x >= span.y || min(a.w, b.w) <= 0.0 || splat.start_mass.w == 0.0 {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }
    let midpoint = (a + b) * 0.5;
    out.detail = detail_interval(instance, midpoint.w);
    if out.detail.x >= out.detail.y {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }
    let a_ndc = a.xyz / a.w;
    let b_ndc = b.xyz / b.w;
    let half_viewport = view.viewport.xy * 0.5;
    let line = (b_ndc.xy - a_ndc.xy) * half_viewport;
    let length_px = length(line);
    let point = length_px < 1e-4;
    let tangent = select(line / max(length_px, 1e-4), vec2(1.0, 0.0), point);
    let across = vec2(-tangent.y, tangent.x);
    let q = splat.rotation;
    let local_to_kernel = vec4(-q.xyz, q.w);
    let jx = project_axis(transform[0], midpoint, half_viewport);
    let jy = project_axis(transform[1], midpoint, half_viewport);
    let jz = project_axis(transform[2], midpoint, half_viewport);
    let cross_axis = vec3(dot(across, jx), dot(across, jy), dot(across, jz));
    let along_axis = vec3(dot(tangent, jx), dot(tangent, jy), dot(tangent, jz));
    let cross_sigma = length(rotate(local_to_kernel, cross_axis) * splat.sigma.xyz);
    let along_sigma = length(rotate(local_to_kernel, along_axis) * splat.sigma.xyz);
    // The cross section's isocontour is where optical depth falls below one.
    let area = 2.506628 * splat.sigma.x
        * max(distance(splat.start_mass.xyz, splat.end.xyz), 2.506628 * splat.sigma.y);
    let radius = sqrt(max(0.0, 2.0 * log(splat.start_mass.w / area)));
    if radius <= 0.0 {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }

    let tip = (vertex_index & 1u) != 0u;
    let edge = select(-1.0, 1.0, (vertex_index & 2u) != 0u);
    let width_scale = mix(1.0, profile.y, select(span.x, span.y, tip));
    let clip_w = select(a.w, b.w, tip);
    let width = max(cross_sigma * radius * width_scale * midpoint.w / clip_w, 0.42);
    let cap = clamp(along_sigma * radius, 0.45, max(cross_sigma * radius, 0.45) * 1.4);
    let along = select(-cap, cap, tip);
    let endpoint = select(a_ndc, b_ndc, tip);
    out.position = vec4(
        endpoint.xy + (across * edge * width + tangent * along) / half_viewport,
        endpoint.z,
        1.0,
    );
    out.coordinate = vec2(select(0.0, length_px, tip) + along, edge);
    // Close segments dissolve through the same dither as the LOD fades.
    let coverage = view.motion.z * near_fade(midpoint.w);
    if coverage <= 0.0 {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }
    out.shape = vec4(
        length_px, cap, cross_sigma * radius,
        select(coverage, -coverage, view.motion.w > 0.0),
    );
    out.profile = vec4(profile, mix(splat.end.w, splat.sigma.w, span.x), mix(splat.end.w, splat.sigma.w, span.y));

    out.color = vec3(0.0);
    if !SHADOW_QUAD || vertex_index == 1u {
        var color = splat.color.rgb;
        if view.direction_to_light.w > 0.0 {
            let normal = rotate(q, vec3(0.0, 0.0, 1.0));
            let unit = normalize(view.normal_from_local * normal);
            color *= segment_light(instance, (local_a + local_b) * 0.5, unit);
        }
        if view.fog.mode != 0u {
            color = apply_distance_fog(color, (local_a + local_b) * 0.5);
        }
        out.color = color;
    }
    return out;
}

@fragment
fn cutout_fragment(in: CutoutRaster) -> @location(0) vec4<f32> {
    if !detail_visible(in.position.xy, in.detail) {
        discard;
    }
    let along = in.coordinate.x;
    let across = in.coordinate.y;
    let beyond = max(0.0, max(-along, along - in.shape.x)) / in.shape.y;
    if dot(vec2(beyond, across), vec2(beyond, across)) > 1.0 {
        discard;
    }
    if abs(in.shape.w) < 1.0 {
        // A shared screen-space threshold makes paired LOD fades complementary.
        let threshold = screen_threshold(in.position.xy);
        if select(threshold >= in.shape.w, threshold < 1.0 + in.shape.w, in.shape.w < 0.0) {
            discard;
        }
    }
    // Tapered leaf kernels have a pale midrib and thinner, warmer edges.
    var color = in.color;
    if in.profile.y < 1.0 && in.shape.z >= 1.0 {
        let rib = max(1.0 - abs(across) * 3.0, 0.0);
        let edge = across * across;
        // Suppress vein contrast as the blade shrinks below three pixels.
        let visible_detail = smoothstep(1.0, 3.0, in.shape.z);
        let t = mix(in.profile.z, in.profile.w, clamp(along / max(in.shape.x, 1e-4), 0.0, 1.0));
        let seed = in.profile.x * 127.0;
        let vein = 0.93 + 0.07 * sin(across * 23.0 + seed);
        let streak = 0.95 + 0.06 * sin(across * 9.0 + sin(t * 3.8 / in.profile.x + seed));
        let fold = 0.90 + 0.10 * smoothstep(-0.10, 0.10, across);
        color *= (vec3(fold) + 0.23 * rib + vec3(0.05, 0.01, -0.12) * edge)
            * mix(1.0, vein * streak, visible_detail);
    }
    return vec4(color, 1.0);
}
