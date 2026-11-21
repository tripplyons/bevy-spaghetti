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
    fog: Fog,
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
    // The pixel that holds the world origin. Shadow views set it so their
    // random patterns stay fixed to the ground when the cascade moves.
    pixel_origin: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
// Sorted draws read segment indices back to front. Cutouts read packed
// visible clusters: the first segment above seven bits of count minus one.
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read> occlusion: array<u32>;

override SCENE_SHADOWS: bool = false;
override BAKED_OCCLUSION: bool = false;
// Draws depth-only cutouts into an orthographic directional shadow cascade.
override SHADOW_CASTER: bool = false;
// The optical depth at the edge of a shadow caster's footprint.
const SHADOW_EDGE_DEPTH: f32 = 0.03;
// The narrowest shadow strip in texels before optical depth is spread.
const SHADOW_MIN_WIDTH: f32 = 1.0;
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

fn scene_shadow(midpoint: vec3<f32>, normal: vec3<f32>) -> f32 {
    let world = view.world_from_local * vec4(midpoint, 1.0);
    let depth = dot(sun_shadow.view_depth, world);
    let beyond = select(vec4<u32>(0u), vec4<u32>(1u), vec4(depth) >= sun_shadow.far_bounds);
    let cascade = beyond.x + beyond.y + beyond.z + beyond.w;
    if cascade >= sun_shadow.layers.y {
        return 1.0;
    }
    var shadow = cascade_shadow(cascade, world.xyz, normal);
    // Blend into the next cascade across the overlap, as Bevy does for
    // meshes, so the switch does not sweep through crowns as a hard band.
    let next = cascade + 1u;
    if next < sun_shadow.layers.y {
        let far = sun_shadow.far_bounds[cascade];
        let near = (1.0 - sun_shadow.depth_bias.w) * far;
        if depth >= near {
            let blend = (depth - near) / max(far - near, 1e-6);
            shadow = mix(shadow, cascade_shadow(next, world.xyz, normal), blend);
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
fn segment_highlight(midpoint: vec3<f32>, normal: vec3<f32>, axis: vec3<f32>) -> f32 {
    let world = (view.world_from_local * vec4(midpoint, 1.0)).xyz;
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
fn lamp_light(midpoint: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let world = (view.world_from_local * vec4(midpoint, 1.0)).xyz;
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
    midpoint: vec3<f32>,
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
        highlight = view.material.x * segment_highlight(midpoint, normal, axis);
    }
    if SCENE_SHADOWS && (incident > 0.0 || highlight > 0.0) {
        light.y *= scene_shadow(midpoint, select(-normal, normal, facing >= 0.0));
    }
    let ground_weight = 0.5 - 0.5 * normal.y;
    let ambient = view.ambient.rgb + ground_weight * view.ground_ambient_delta.rgb;
    var local = vec3(0.0);
    if view.lamp.w > 0.0 {
        local = lamp_light(midpoint, normal);
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
    if vertex_index == 1u {
        let area_scale = length(cross(j0, j1));
        let peak = (mass * (span.y - span.x) * area_scale) / (6.28318530718 * sqrt(determinant));
        var color = splat.color.rgb * tint;
        if view.direction_to_light.w > 0.0 {
            let normal = view.normal_from_local * rotate(q, vec3(0.0, 0.0, 1.0));
            let unit = normal * inverseSqrt(max(1e-10, dot(normal, normal)));
            color = segment_light(index, color, (local_start + local_end) * 0.5, unit, local_end - local_start);
        }
        if view.fog.mode != 0u {
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
    @builtin(instance_index) slot: u32,
) -> CutoutRaster {
    var out: CutoutRaster;
    // Each visible cluster owns CLUSTER_SIZE instances, and a short cluster
    // leaves the rest empty.
    // clusters.y holds the slots per cluster, fewer in thinned cascades.
    let cluster = order[view.clusters.x + slot / view.clusters.y];
    let offset = slot % view.clusters.y;
    if offset > (cluster & (CLUSTER_SIZE - 1u)) {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }
    let instance = (cluster >> 7u) + offset;
    let splat = splats[instance];
    let profile = blade_profile(splat.color.w);
    let local_a = bend_endpoint(splat.start_mass.xyz, splat.end.w, profile.x);
    let local_b = bend_endpoint(splat.end.xyz, splat.sigma.w, profile.x);
    let transform = view.clip_from_local;
    let start = transform * vec4(local_a, 1.0);
    let end = transform * vec4(local_b, 1.0);
    var near = vec2(0.0, 1.0);
    if !SHADOW_CASTER {
        near = clip_interval(near, start.w - start.z, end.w - end.z);
    }
    let span = clip_interval(near, start.z, end.z);
    let a = mix(start, end, span.x);
    let b = mix(start, end, span.y);

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
    var peak = splat.start_mass.w / area;
    // A shadow caster draws out to a faint optical depth, because sparse
    // coarse levels never reach one but still dim the light.
    let edge_depth = select(1.0, SHADOW_EDGE_DEPTH, SHADOW_CASTER);
    let radius = sqrt(max(0.0, 2.0 * log(peak / edge_depth)));
    if radius <= 0.0 {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
        return out;
    }

    let tip = (vertex_index & 1u) != 0u;
    let edge = select(-1.0, 1.0, (vertex_index & 2u) != 0u);
    let width_scale = mix(1.0, profile.y, select(span.x, span.y, tip));
    let clip_w = select(a.w, b.w, tip);
    var width = max(cross_sigma * radius * width_scale * midpoint.w / clip_w, 0.42);
    if SHADOW_CASTER {
        // Spread a needle thinner than a texel over a wider, fainter strip so
        // that its integrated optical depth is kept.
        let exact = cross_sigma * radius * width_scale * midpoint.w / clip_w;
        width = max(exact, SHADOW_MIN_WIDTH);
        peak *= exact / width;
    }
    let cap = clamp(along_sigma * radius, 0.45, max(cross_sigma * radius, 0.45) * 1.4);
    let along = select(-cap, cap, tip);
    let endpoint = select(a_ndc, b_ndc, tip);
    var depth = endpoint.z;
    if SHADOW_CASTER {
        // A caster between the sun and the cascade still shades it, so it
        // flattens onto the near plane instead of clipping.
        depth = min(depth, 1.0);
    }
    out.position = vec4(
        endpoint.xy + (across * edge * width + tangent * along) / half_viewport,
        depth,
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
    if SHADOW_CASTER {
        // The shadow fragment needs no color profile. It needs the optical
        // depth falloff and a per-segment seed for its random test. A float
        // holds the index exactly, where a bitcast could flush to zero.
        out.profile = vec4(f32(instance), peak, radius * radius, 0.0);
    }

    out.color = vec3(0.0);
    if vertex_index == 1u && !SHADOW_CASTER {
        var color = splat.color.rgb;
        if view.direction_to_light.w > 0.0 {
            let normal = rotate(q, vec3(0.0, 0.0, 1.0));
            let unit = normalize(view.normal_from_local * normal);
            color = segment_light(instance, color, (local_a + local_b) * 0.5, unit, local_b - local_a);
        }
        if view.fog.mode != 0u {
            color = apply_distance_fog(color, (local_a + local_b) * 0.5);
        }
        out.color = color;
    }
    return out;
}

// Whether a fragment lies inside the cutout and survives the fade dithers.
fn cutout_covers(in: CutoutRaster) -> bool {
    let pixel = in.position.xy - view.pixel_origin.xy;
    if !detail_visible(pixel, in.detail) {
        return false;
    }
    let along = in.coordinate.x;
    let across = in.coordinate.y;
    let beyond = max(0.0, max(-along, along - in.shape.x)) / in.shape.y;
    if dot(vec2(beyond, across), vec2(beyond, across)) > 1.0 {
        return false;
    }
    if abs(in.shape.w) < 1.0 {
        // A shared screen-space threshold makes paired LOD fades complementary.
        let threshold = screen_threshold(pixel);
        if select(threshold >= in.shape.w, threshold < 1.0 + in.shape.w, in.shape.w < 0.0) {
            return false;
        }
    }
    return true;
}

// Each segment blocks a texel with probability 1 - exp(-optical depth).
// Independent seeds per segment make overlapping segments multiply their
// transmittance, and shadow filtering averages the texels into partial shade.
@fragment
fn cutout_shadow_fragment(in: CutoutRaster) {
    if !cutout_covers(in) {
        discard;
    }
    let beyond = max(0.0, max(-in.coordinate.x, in.coordinate.x - in.shape.x)) / in.shape.y;
    let distance_squared = beyond * beyond + in.coordinate.y * in.coordinate.y;
    let depth = in.profile.y * exp(-0.5 * in.profile.z * distance_squared);
    let pixel = bitcast<vec2<u32>>(vec2<i32>(floor(in.position.xy - view.pixel_origin.xy)));
    let seed = u32(in.profile.x);
    // A thinned cascade keeps a fraction of the segments, and each one
    // blocks that much more often, so the expected transmittance holds
    // wherever no single segment is nearly opaque.
    let kept = bitcast<f32>(view.clusters.z);
    let block = min(1.0, (1.0 - exp(-depth)) / kept);
    if hash_unit(pixel.x * 0x8da6b343u ^ pixel.y * 0xd8163841u ^ seed * 0xcb1ab31fu) >= block {
        discard;
    }
}

// A uniform value in [0, 1) from a well-mixed 32-bit hash.
fn hash_unit(value: u32) -> f32 {
    var h = value;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

@fragment
fn cutout_fragment(in: CutoutRaster) -> @location(0) vec4<f32> {
    if !cutout_covers(in) {
        discard;
    }
    let along = in.coordinate.x;
    let across = in.coordinate.y;
    // Tapered leaf kernels have a pale midrib and thinner, warmer edges.
    var color = in.color;
    if in.profile.x > 0.0 && in.profile.y < 1.0 && in.shape.z >= 1.0 {
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
