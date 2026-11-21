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
    ground_ambient_delta: vec4<f32>,
    eye_relative_from_local: mat4x4<f32>,
    world_from_local: mat4x4<f32>,
    local_from_world: mat3x3<f32>,
    pusher: vec4<f32>,
    motion: vec4<f32>,
    detail_depths: array<vec4<f32>, 3>,
    detail_ends: vec4<u32>,
    eye_world: vec4<f32>,
    // Specular strength, exponent, strand fraction, additive flag.
    material: vec4<f32>,
    // Local travel over one loop, then elapsed loops.
    flow_span: vec4<f32>,
    // Local turbulence reach, flicker depth, spread, growth.
    flow_shape: vec4<f32>,
    // Fade-in and fade-out fractions, time, enabled flag.
    flow_fade: vec4<f32>,
    // Color factor at the end of a loop.
    flow_tint: vec4<f32>,
    // World position and range of the local light. A zero range turns it off.
    lamp: vec4<f32>,
    // Color at one unit of distance, then the squared source radius.
    lamp_color: vec4<f32>,
    // Offset of this draw's visible clusters in `order`.
    clusters: vec4<u32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
// Sorted draws read segment indices back to front. Cutouts read packed
// visible clusters: the first segment above seven bits of count minus one.
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read> occlusion: array<u32>;

override SCENE_SHADOWS: bool = false;
override BAKED_OCCLUSION: bool = false;
// Segments per culling cluster, matching the renderer.
const CLUSTER_SIZE: u32 = 128u;

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
@group(1) @binding(3) var<uniform> view_fog: Fog;

fn scene_shadow(world: vec3<f32>, normal: vec3<f32>) -> f32 {
    let depth = dot(sun_shadow.view_depth, vec4(world, 1.0));
    let beyond = select(vec4<u32>(0u), vec4<u32>(1u), vec4(depth) >= sun_shadow.far_bounds);
    let cascade = beyond.x + beyond.y + beyond.z + beyond.w;
    if cascade >= sun_shadow.layers.y {
        return 1.0;
    }
    var shadow = cascade_shadow(cascade, world, normal);
    // Blend into the next cascade across the overlap, as Bevy does for
    // meshes, so the switch does not sweep through crowns as a hard band.
    let next = cascade + 1u;
    if next < sun_shadow.layers.y {
        let far = sun_shadow.far_bounds[cascade];
        let near = (1.0 - sun_shadow.depth_bias.w) * far;
        if depth >= near {
            let blend = (depth - near) / max(far - near, 1e-6);
            shadow = mix(shadow, cascade_shadow(next, world, normal), blend);
        }
    }
    return shadow;
}

// The lit fraction from one cascade. Four bilinear compare taps half a texel
// apart give a 3x3 tent, which averages the stochastic splat shadows enough
// that one segment's sample does not flip between lit and dark.
fn cascade_shadow(cascade: u32, world: vec3<f32>, normal: vec3<f32>) -> f32 {
    let offset = normal * sun_shadow.normal_bias[cascade] + sun_shadow.depth_bias.xyz;
    // Directional cascades use an orthographic transform and reverse Z.
    let ndc = (sun_shadow.matrices[cascade] * vec4(world + offset, 1.0)).xyz;
    if any(abs(ndc.xy) >= vec2(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let uv = ndc.xy * vec2(0.5, -0.5) + 0.5;
    let texel = 0.5 / vec2<f32>(textureDimensions(sun_depth));
    let layer = sun_shadow.layers.x + cascade;
    return 0.25 * (
        textureSampleCompareLevel(sun_depth, sun_compare, uv + vec2(-texel.x, -texel.y), layer, ndc.z)
        + textureSampleCompareLevel(sun_depth, sun_compare, uv + vec2(texel.x, -texel.y), layer, ndc.z)
        + textureSampleCompareLevel(sun_depth, sun_compare, uv + vec2(-texel.x, texel.y), layer, ndc.z)
        + textureSampleCompareLevel(sun_depth, sun_compare, uv + vec2(texel.x, texel.y), layer, ndc.z)
    );
}

// One highlight value for a segment. `normal` is its world-space surface
// normal and `axis` is its local centerline, which may be zero for a point.
fn segment_highlight(world: vec3<f32>, normal: vec3<f32>, axis: vec3<f32>) -> f32 {
    let to_eye = normalize(view.eye_world.xyz - world);
    let to_light = view.direction_to_light.xyz;
    let half_vector = normalize(to_light + to_eye);
    // Splats are two-sided, so the face toward the eye reflects the light.
    let toward = select(-normal, normal, dot(normal, to_eye) >= 0.0);
    let surface = pow(max(dot(toward, half_vector), 0.0), view.material.y)
        * smoothstep(0.0, 0.12, dot(toward, to_light));
    if view.material.z <= 0.0 {
        return surface;
    }
    let tangent = (view.world_from_local * vec4(axis, 0.0)).xyz;
    let length_squared = dot(tangent, tangent);
    if length_squared < 1e-12 {
        return surface;
    }
    // Kajiya-Kay: a thin fiber reflects a cone of light around its axis.
    let along = dot(tangent, half_vector) * inverseSqrt(length_squared);
    let strand = pow(sqrt(max(1.0 - along * along, 0.0)), view.material.y)
        * smoothstep(-0.5, 0.1, dot(normal, to_light));
    return mix(surface, strand, view.material.z);
}

// Light from the local source on a segment. It follows the inverse square
// law and a window brings it smoothly to zero at the range.
fn lamp_light(world: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let to_lamp = view.lamp.xyz - world;
    let distance_squared = dot(to_lamp, to_lamp);
    let fraction = distance_squared / (view.lamp.w * view.lamp.w);
    if fraction >= 1.0 {
        return vec3(0.0);
    }
    let facing = dot(normal, to_lamp) * inverseSqrt(max(distance_squared, 1e-12));
    let incident = max(0.0, facing) + view.ambient.w * max(0.0, -facing);
    let window = 1.0 - fraction * fraction;
    return view.lamp_color.rgb
        * (incident * window * window / (distance_squared + view.lamp_color.w));
}

fn segment_light(
    index: u32,
    color: vec3<f32>,
    world: vec3<f32>,
    normal: vec3<f32>,
    axis: vec3<f32>,
) -> vec3<f32> {
    var light = vec2(1.0);
    if BAKED_OCCLUSION {
        light = unpack2x16unorm(occlusion[index]);
    }
    let facing = dot(normal, view.direction_to_light.xyz);
    // Thin foliage passes part of the light through from its back face.
    let incident = max(0.0, facing) + view.ambient.w * max(0.0, -facing);
    var highlight = 0.0;
    if view.material.x > 0.0 {
        highlight = view.material.x * segment_highlight(world, normal, axis);
    }
    if SCENE_SHADOWS && (incident > 0.0 || highlight > 0.0) {
        light.y *= scene_shadow(world, select(-normal, normal, facing >= 0.0));
    }
    let ground_weight = 0.5 - 0.5 * normal.y;
    let ambient = view.ambient.rgb + ground_weight * view.ground_ambient_delta.rgb;
    var local = vec3(0.0);
    if view.lamp.w > 0.0 {
        local = lamp_light(world, normal);
    }
    // Baked ambient occlusion also dims the local light inside a dense cloud.
    return color * (light.x * (ambient + local) + (incident * light.y) * view.directional.rgb)
        + (highlight * light.y) * view.directional.rgb;
}

// Three uniform random numbers per segment (PCG, then two LCG steps).
fn segment_random(index: u32) -> vec3<f32> {
    let state = index * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let a = (word >> 22u) ^ word;
    let b = a * 1664525u + 1013904223u;
    let c = b * 1664525u + 1013904223u;
    return vec3(f32(a >> 8u), f32(b >> 8u), f32(c >> 8u)) * (1.0 / 16777216.0);
}

struct Flow {
    offset: vec3<f32>,
    // Factors on optical mass and kernel size.
    mass: f32,
    size: f32,
    tint: vec3<f32>,
}

// Each segment loops from its authored place along the flow span, with its
// own start time, a smooth wander, and a fade at both ends of the loop.
fn segment_flow(index: u32, center: vec3<f32>) -> Flow {
    let random = segment_random(index);
    let loops = view.flow_span.w;
    let t = fract(loops + random.x);
    let span = view.flow_span.xyz;
    var radial = center;
    let span_squared = dot(span, span);
    if span_squared > 0.0 {
        radial -= span * (dot(center, span) / span_squared);
    }
    let rate = 0.6 + random.yzx;
    let wander = sin(6.2831853 * (loops * rate + random.zxy));
    var flow: Flow;
    flow.offset = span * t + radial * (view.flow_shape.z * t) + view.flow_shape.x * wander;
    let fade = view.flow_fade.xy;
    flow.mass = 1.0;
    if fade.x > 0.0 {
        flow.mass *= clamp(t / fade.x, 0.0, 1.0);
    }
    if fade.y > 0.0 {
        flow.mass *= clamp((1.0 - t) / fade.y, 0.0, 1.0);
    }
    let pulse = 0.5 + 0.5 * sin(view.flow_fade.z * (9.0 + 13.0 * random.y) + 6.2831853 * random.z);
    flow.mass *= 1.0 - view.flow_shape.y * pulse;
    flow.size = max(1.0 + view.flow_shape.w * t, 0.05);
    flow.tint = mix(vec3(1.0), view.flow_tint.xyz, t);
    return flow;
}

struct Raster {
    @builtin(position) position: vec4<f32>,
    @location(0) offset: vec2<f32>,
    @location(1) @interpolate(flat, first) color_mass: vec4<f32>,
    // Detail fade interval, whitened segment length, and sqrt(pi/2) / length.
    @location(2) @interpolate(flat) detail_len: vec4<f32>,
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
    if view.motion.y == 2.0 {
        return local + view.pusher.xyz * rise;
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
//
// The derivative is a linear stand-in for the projection around the
// midpoint. Far outside the frame and close to the camera plane, its depth
// term grows without limit, and a soft kernel there would smear across the
// whole view. Evaluating the depth term no further out than 1.3 frame
// half-widths keeps such a kernel off screen and leaves kernels in view
// unchanged.
fn project_axis(axis: vec4<f32>, midpoint: vec4<f32>, half_viewport: vec2<f32>) -> vec2<f32> {
    let center = clamp(midpoint.xy / midpoint.w, vec2(-1.3), vec2(1.3));
    return (axis.xy - center * axis.w) * half_viewport / midpoint.w;
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
    var fog_color = view_fog.color.rgb;
    if distance > 0.0 && view_fog.light_count.x > 0u {
        let ray = eye_to_splat / distance;
        for (var i = 0u; i < view_fog.light_count.x; i += 1u) {
            let light = view_fog.lights[i];
            let alignment = max(dot(ray, light.direction.xyz), 0.0);
            fog_color += pow(alignment, view_fog.directional_exponent) * light.color.rgb;
        }
    }
    let strength = view_fog.color.a;
    let be = view_fog.extinction;
    switch view_fog.mode {
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
            let inscattering = 1.0 - exp(-distance * view_fog.inscattering);
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
    var mass = splat.start_mass.w;
    var s = splat.sigma.xyz;
    var tint = vec3(1.0);
    if view.flow_fade.w > 0.0 {
        let flow = segment_flow(index, (local_start + local_end) * 0.5);
        local_start += flow.offset;
        local_end += flow.offset;
        mass *= flow.mass;
        s *= flow.size;
        tint = flow.tint;
    }
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
    if span.x >= span.y || min(a.w, b.w) <= 0.0 || mass <= 0.0 {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }

    let midpoint = (a + b) * 0.5;
    let detail = detail_interval(index, midpoint.w);
    if detail.x >= detail.y {
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
    let local_to_kernel = vec4(-q.xyz, q.w);
    let row_x = rotate(local_to_kernel, j0) * s;
    let row_y = rotate(local_to_kernel, j1) * s;
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
    let area_scale = length(cross(j0, j1));
    let peak = (mass * (span.y - span.x) * area_scale) / (6.28318530718 * sqrt(determinant));
    let eff_peak = peak * near_fade(midpoint.w);
    let bound_sigma = min(4.0, sqrt(max(0.0, 2.0 * log(eff_peak * view.motion.z * 2000.0))));
    if bound_sigma <= 0.0 {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }
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
    let extent = bound_sigma * sqrt(max(vec2(cov_t, cov_p), vec2(pixel_variance)));
    let pad = (abs(tangent) * extent.x + abs(perpendicular) * extent.y) / half_viewport;
    if max(a_ndc.x, b_ndc.x) + pad.x < -1.0 || min(a_ndc.x, b_ndc.x) - pad.x > 1.0
        || max(a_ndc.y, b_ndc.y) + pad.y < -1.0 || min(a_ndc.y, b_ndc.y) - pad.y > 1.0
    {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }

    let corners = array<vec2<f32>, 4>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0),
        vec2(-1.0, 1.0), vec2(1.0, 1.0)
    );
    let corner = corners[vertex_index];
    let v = tangent * (max(corner.x, 0.0) * length_px + corner.x * extent.x)
          + perpendicular * (corner.y * extent.y);

    // Cholesky whitening lets the fragment shader integrate an isotropic
    // unit Gaussian along the whitened centerline. Pre-rotate into segment
    // coordinates so the fragment stage reads (projected, normal) directly.
    let shear = xy / cxx;
    let w_off = vec2(v.x / sigma_x, (v.y - shear * v.x) / sigma_y_given_x);
    let w_seg = vec2(segment.x / sigma_x, (segment.y - shear * segment.x) / sigma_y_given_x);
    let length_sigma = length(w_seg);
    if length_sigma < 0.01 {
        out.offset = w_off - 0.5 * w_seg;
    } else {
        let axis = w_seg / length_sigma;
        out.offset = vec2(dot(w_off, axis), dot(w_off, vec2(-axis.y, axis.x)));
    }
    out.detail_len = vec4(detail, length_sigma, 1.25331413732 / max(length_sigma, 0.01));
    out.color_mass = vec4(0.0);
    if vertex_index == 1u {
        var color = splat.color.rgb * tint;
        if view.direction_to_light.w > 0.0 {
            let normal = view.normal_from_local * rotate(q, vec3(0.0, 0.0, 1.0));
            let unit = normal * inverseSqrt(max(1e-10, dot(normal, normal)));
            let world = (view.world_from_local * vec4((local_start + local_end) * 0.5, 1.0)).xyz;
            color = segment_light(index, color, world, unit, local_end - local_start);
        }
        if view_fog.mode != 0u {
            let local_midpoint = mix(local_start, local_end, (span.x + span.y) * 0.5);
            let fogged = apply_distance_fog(color, local_midpoint);
            if view.material.w > 0.0 {
                // Added light has no surface for the haze to cover, so the
                // fog only dims it.
                color = max(fogged - apply_distance_fog(vec3(0.0), local_midpoint), vec3(0.0));
            } else {
                color = fogged;
            }
        }
        // The fragment stage premultiplies the fogged color by optical alpha.
        out.color_mass = vec4(color, eff_peak);
    }
    let z = clamp(select(a_ndc.z, b_ndc.z, corner.x > 0.0), 0.0, 1.0);
    out.position = vec4((a_px + v) / half_viewport, z, 1.0);
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

@fragment
fn fragment(in: Raster) -> @location(0) vec4<f32> {
    if !detail_visible(in.position.xy, in.detail_len.xy) {
        discard;
    }
    let length_sigma = in.detail_len.z;
    var average: f32;
    if length_sigma < 0.01 {
        // Point limit of equation 19, also avoids cancellation for short segments.
        average = exp(-0.5 * dot(in.offset, in.offset));
    } else {
        let projected = in.offset.x;
        let normal = in.offset.y;
        let lo = -projected * 0.70710678118;
        let hi = (length_sigma - projected) * 0.70710678118;
        average = exp(-0.5 * normal * normal) * in.detail_len.w * erf_difference(lo, hi);
    }
    let optical_depth = in.color_mass.a * max(average, 0.0);
    let alpha = (1.0 - exp(-optical_depth)) * view.motion.z;
    if alpha < 0.0005 {
        discard;
    }
    return vec4(in.color_mass.rgb * alpha, alpha);
}

struct CutoutRaster {
    @builtin(position) position: vec4<f32>,
    // Across coordinate and leaf streak phase.
    @location(0) leaf: vec2<f32>,
    // Shaded color, plus packed (leaf seed phase, vein visibility) when close.
    @location(1) @interpolate(flat, first) color_leaf: vec4<f32>,
}

fn hash_unit(value: u32) -> f32 {
    var h = value;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// Evaluate a Gaussian cutout in projected segment coordinates with no fragment
// discard, enabling hardware early-Z and TBDR hidden-surface removal. End caps
// are shortened by pi/4 to match the area of rounded capsule caps, and LOD /
// range / near fades drop segments stochastically before projection and shading.
@vertex
fn cutout_vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) slot: u32,
) -> CutoutRaster {
    var out: CutoutRaster;
    // Each visible cluster owns `1 << shift` instances, and a short or thinned
    // cluster leaves the rest empty.
    let shift = view.clusters.y;
    let cluster_slot = slot >> shift;
    let offset = slot & ((1u << shift) - 1u);
    let cluster = order[view.clusters.x + cluster_slot];
    if offset > (cluster & (CLUSTER_SIZE - 1u)) {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }
    let instance = (cluster >> 7u) + offset;
    let splat = splats[instance];
    let profile = blade_profile(splat.color.w);
    var local_a = splat.start_mass.xyz;
    var local_b = splat.end.xyz;
    if view.motion.y > 0.0 {
        local_a = bend_endpoint(local_a, splat.end.w, profile.x);
        local_b = bend_endpoint(local_b, splat.sigma.w, profile.x);
    }
    let transform = view.clip_from_local;
    let start = transform * vec4(local_a, 1.0);
    let end = transform * vec4(local_b, 1.0);
    let near = clip_interval(vec2(0.0, 1.0), start.w - start.z, end.w - end.z);
    let span = clip_interval(near, start.z, end.z);
    let a = mix(start, end, span.x);
    let b = mix(start, end, span.y);

    if span.x >= span.y || min(a.w, b.w) <= 0.0 || splat.start_mass.w == 0.0 {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }
    let midpoint = (a + b) * 0.5;
    if view.viewport.w > view.viewport.z {
        let coverage = near_fade(midpoint.w);
        if coverage <= 0.0 {
            out.position = vec4(2.0, 2.0, 0.0, 1.0);
            return out;
        }
        if coverage < 1.0 {
            let threshold = hash_unit(instance * 0xcb1ab31fu ^ 0x9e3779b9u);
            if select(threshold >= coverage, (threshold < 1.0 - coverage), view.motion.w > 0.0) {
                out.position = vec4(2.0, 2.0, 0.0, 1.0);
                return out;
            }
        }
    }
    let a_ndc = a.xyz / a.w;
    let b_ndc = b.xyz / b.w;
    let half_viewport = view.viewport.xy * 0.5;
    let line = (b_ndc.xy - a_ndc.xy) * half_viewport;
    let length_sq = dot(line, line);
    let inv_len = inverseSqrt(max(length_sq, 1e-8));
    let tangent = select(line * inv_len, vec2(1.0, 0.0), length_sq < 1e-8);
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
    let peak = splat.start_mass.w / area;
    let radius = sqrt(max(0.0, 2.0 * log(peak)));
    if radius <= 0.0 {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }

    let cross_r = cross_sigma * radius;
    let cap = clamp(along_sigma * radius, 0.45, max(cross_r, 0.45) * 1.4) * 0.78539816;
    let max_width = max(cross_r * midpoint.w / min(a.w, b.w), 0.42);
    let pad = (abs(across) * max_width + abs(tangent) * cap) / half_viewport;
    if max(a_ndc.x, b_ndc.x) + pad.x < -1.0 || min(a_ndc.x, b_ndc.x) - pad.x > 1.0
        || max(a_ndc.y, b_ndc.y) + pad.y < -1.0 || min(a_ndc.y, b_ndc.y) - pad.y > 1.0
    {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }

    let tip = (vertex_index & 1u) != 0u;
    let edge = select(-1.0, 1.0, (vertex_index & 2u) != 0u);
    let width_scale = mix(1.0, profile.y, select(span.x, span.y, tip));
    let clip_w = select(a.w, b.w, tip);
    let width = max(cross_r * width_scale * midpoint.w / clip_w, 0.42);
    let along = select(-cap, cap, tip);
    let endpoint = select(a_ndc, b_ndc, tip);
    out.position = vec4(
        endpoint.xy + (across * edge * width + tangent * along) / half_viewport,
        endpoint.z,
        1.0,
    );
    let t_vertex = mix(splat.end.w, splat.sigma.w, select(span.x, span.y, tip));
    let streak_phase = select(0.0, t_vertex * 3.8 / max(profile.x, 1e-5) + profile.x * 127.0, profile.x > 0.0);
    out.leaf = vec2(edge, streak_phase);

    out.color_leaf = vec4(0.0);
    if vertex_index == 1u {
        var color = splat.color.rgb;
        let local_mid = (local_a + local_b) * 0.5;
        if view.direction_to_light.w > 0.0 {
            let normal = rotate(q, vec3(0.0, 0.0, 1.0));
            let unit = normalize(view.normal_from_local * normal);
            let world = (view.world_from_local * vec4(local_mid, 1.0)).xyz;
            color = segment_light(instance, color, world, unit, local_b - local_a);
        }
        if view_fog.mode != 0u {
            color = apply_distance_fog(color, local_mid);
        }
        var packed_leaf = 0.0;
        if profile.x > 0.0 && profile.y < 1.0 && cross_r >= 1.0 {
            let seed_frac = fract(profile.x * (127.0 / 6.28318530718));
            let visible_detail = smoothstep(1.0, 3.0, cross_r);
            packed_leaf = round(seed_frac * 4095.0) * 4096.0
                + (1.0 + round(visible_detail * 4094.0));
        }
        out.color_leaf = vec4(color, packed_leaf);
    }
    return out;
}

@fragment
fn cutout_fragment(in: CutoutRaster) -> @location(0) vec4<f32> {
    let across = in.leaf.x;
    // Tapered leaf kernels have a pale midrib and thinner, warmer edges.
    var color = in.color_leaf.rgb;
    if in.color_leaf.w > 0.0 {
        let bits = u32(in.color_leaf.w);
        let seed = f32(bits >> 12u) * (6.28318530718 / 4095.0);
        let visible_detail = f32((bits & 4095u) - 1u) * (1.0 / 4094.0);
        let rib = max(1.0 - abs(across) * 3.0, 0.0);
        let edge = across * across;
        let vein = 0.93 + 0.07 * sin(across * 23.0 + seed);
        let streak = 0.95 + 0.06 * sin(across * 9.0 + sin(in.leaf.y));
        let fold = 0.90 + 0.10 * smoothstep(-0.10, 0.10, across);
        color *= (vec3(fold) + 0.23 * rib + vec3(0.05, 0.01, -0.12) * edge)
            * mix(1.0, vein * streak, visible_detail);
    }
    return vec4(color, 1.0);
}
