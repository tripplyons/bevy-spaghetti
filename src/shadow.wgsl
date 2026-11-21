// Depth-only stochastic geometry shadow pass for Gaussian segment splats.
//
// Each segment's optical depth profile tau(d) = peak * exp(-0.5 * d^2) is
// sampled once via an exponential threshold E = -ln(1 - u), precomputed on
// the asset as r2_base = 2 * ln(peak / E). At any texel at normalized
// distance d, P(d <= r) = 1 - exp(-k * tau(d)) matches Beer-Lambert
// transmittance without a fragment shader or discard, enabling hardware
// early-Z and TBDR hidden-surface removal. Each surviving segment emits one
// area-matched triangle (3 vertices, non-indexed).

struct ShadowSegment {
    // Packed unorm16 endpoints in asset-AABB [0, 1]^3:
    //   x: (start.x, start.y), y: (start.z, end.x), z: (end.y, end.z),
    //   w: f16x2 (precomputed 2 * ln(peak / (-ln(1 - u))), sigma_max / asset_scale).
    geom: vec4<u32>,
    // Packed covariance and deformation attributes:
    //   x: snorm8x4 (s00, s01, s02, s11) of Sigma / sigma_max^2,
    //   y: snorm8x4 (s12, s22, taper, 0),
    //   z: f16x2 (bend_start, bend_end),
    //   w: f16x2 (blade_height, 0).
    attr: vec4<u32>,
}

// Compact 256-byte uniform for shadow draws (4x smaller than DrawUniform).
struct ShadowView {
    // clip_from_unit: maps asset-AABB [0, 1]^3 directly to orthographic clip space.
    clip_from_unit: mat4x4<f32>,
    // Rows 0..2 of world_from_unit (used only when procedural wind motion.y == 1.0 is active).
    world_from_unit_r0: vec4<f32>,
    world_from_unit_r1: vec4<f32>,
    world_from_unit_r2: vec4<f32>,
    // Rows 0..2 of orthographic clip_from_world (used only when motion.y == 1.0).
    clip_from_world_r0: vec4<f32>,
    clip_from_world_r1: vec4<f32>,
    clip_from_world_r2: vec4<f32>,
    // j0 * asset_scale (px per asset_scale along clip X), then half_viewport.x in w.
    j0_half_x: vec4<f32>,
    // j1 * asset_scale (px per asset_scale along clip Y), then half_viewport.y in w.
    j1_half_y: vec4<f32>,
    // Pre-transformed clip-space sway per meter (motion.y == 2.0) or world pusher (motion.y == 1.0).
    pusher: vec4<f32>,
    // Time, wind mode, range coverage, reverse range dither.
    motion: vec4<f32>,
    // x: offset in cluster list, y: log2(keep), z: bitcast<u32>(2 * ln(CLUSTER_SIZE / keep)),
    // w: unused.
    clusters: vec4<u32>,
    // xy: NDC pad scale per asset_scale.
    pad_scale: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: ShadowView;
@group(0) @binding(1) var<storage, read> segments: array<ShadowSegment>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

const CLUSTER_SIZE: u32 = 128u;
const MAX_SHADOW_RADIUS_SQ: f32 = 16.0;

fn hash_unit(value: u32) -> f32 {
    var h = value;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return f32(h >> 8u) * (1.0 / 16777216.0);
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

fn bend_world(world: vec3<f32>, rise: f32, height: f32) -> vec3<f32> {
    if !(rise > 0.0) {
        return world;
    }
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
    return world + vec3(offset.x, -down, offset.y) * rise;
}

@vertex
fn shadow_vertex(
    @builtin(vertex_index) raw_vertex_index: u32,
) -> @builtin(position) vec4<f32> {
    let slot = raw_vertex_index / 3u;
    let vertex_index = raw_vertex_index - slot * 3u;
    let shift = view.clusters.y;
    let cluster_slot = slot >> shift;
    let offset = slot & ((1u << shift) - 1u);
    let cluster = order[view.clusters.x + cluster_slot];
    if offset > (cluster & (CLUSTER_SIZE - 1u)) {
        return vec4(2.0, 2.0, 0.0, 1.0);
    }
    let instance = (cluster >> 7u) + offset;

    if view.motion.z < 1.0 {
        let threshold = hash_unit(instance * 0xcb1ab31fu ^ 0x9e3779b9u);
        if select(threshold >= view.motion.z, (threshold < 1.0 - view.motion.z), view.motion.w > 0.0) {
            return vec4(2.0, 2.0, 0.0, 1.0);
        }
    }

    let geom = segments[instance].geom;
    let r2_sigma = unpack2x16float(geom.w);
    let r2 = r2_sigma.x + bitcast<f32>(view.clusters.z);
    if r2 <= 0.0 {
        return vec4(2.0, 2.0, 0.0, 1.0);
    }

    let g0 = unpack2x16unorm(geom.x);
    let g1 = unpack2x16unorm(geom.y);
    let g2 = unpack2x16unorm(geom.z);
    let unit_a = vec3(g0.x, g0.y, g1.x);
    let unit_b = vec3(g1.y, g2.x, g2.y);

    let attr = segments[instance].attr;
    var start: vec3<f32>;
    var end: vec3<f32>;
    if view.motion.y == 1.0 {
        let bend = unpack2x16float(attr.z);
        let height = unpack2x16float(attr.w).x;
        let ua4 = vec4(unit_a, 1.0);
        let ub4 = vec4(unit_b, 1.0);
        let world_a = bend_world(
            vec3(dot(view.world_from_unit_r0, ua4), dot(view.world_from_unit_r1, ua4), dot(view.world_from_unit_r2, ua4)),
            bend.x,
            height,
        );
        let world_b = bend_world(
            vec3(dot(view.world_from_unit_r0, ub4), dot(view.world_from_unit_r1, ub4), dot(view.world_from_unit_r2, ub4)),
            bend.y,
            height,
        );
        let wa4 = vec4(world_a, 1.0);
        let wb4 = vec4(world_b, 1.0);
        start = vec3(dot(view.clip_from_world_r0, wa4), dot(view.clip_from_world_r1, wa4), dot(view.clip_from_world_r2, wa4));
        end = vec3(dot(view.clip_from_world_r0, wb4), dot(view.clip_from_world_r1, wb4), dot(view.clip_from_world_r2, wb4));
    } else {
        let m = view.clip_from_unit;
        start = (m * vec4(unit_a, 1.0)).xyz;
        end = (m * vec4(unit_b, 1.0)).xyz;
        if view.motion.y == 2.0 {
            let bend = unpack2x16float(attr.z);
            start += view.pusher.xyz * bend.x;
            end += view.pusher.xyz * bend.y;
        }
    }

    if start.z < 0.0 && end.z < 0.0 {
        return vec4(2.0, 2.0, 0.0, 1.0);
    }
    var a = start;
    var b = end;
    var span = vec2(0.0, 1.0);
    if start.z < 0.0 {
        let t = start.z / (start.z - end.z);
        a = mix(start, end, t);
        span.x = t;
    } else if end.z < 0.0 {
        let t = start.z / (start.z - end.z);
        b = mix(start, end, t);
        span.y = t;
    }

    let max_reach = r2_sigma.y * sqrt(min(r2, MAX_SHADOW_RADIUS_SQ));
    let pad = view.pad_scale.xy * max_reach;
    if max(a.x, b.x) + pad.x < -1.0 || min(a.x, b.x) - pad.x > 1.0
        || max(a.y, b.y) + pad.y < -1.0 || min(a.y, b.y) - pad.y > 1.0
    {
        return vec4(2.0, 2.0, 0.0, 1.0);
    }

    let half_viewport = vec2(view.j0_half_x.w, view.j1_half_y.w);
    let line = (b.xy - a.xy) * half_viewport;
    let length_sq = dot(line, line);
    let inv_len = inverseSqrt(max(length_sq, 1e-8));
    let length_px = length_sq * inv_len;
    let tangent = select(line * inv_len, vec2(1.0, 0.0), length_sq < 1e-8);
    let across = vec2(-tangent.y, tangent.x);

    let c0 = unpack4x8snorm(attr.x);
    let c1 = unpack4x8snorm(attr.y);
    let j0 = view.j0_half_x.xyz;
    let j1 = view.j1_half_y.xyz;
    let u = across.x * j0 + across.y * j1;
    let v = tangent.x * j0 + tangent.y * j1;
    let su = vec3(
        u.x * c0.x + u.y * c0.y + u.z * c0.z,
        u.x * c0.y + u.y * c0.w + u.z * c1.x,
        u.x * c0.z + u.y * c1.x + u.z * c1.y,
    );
    let sv = vec3(
        v.x * c0.x + v.y * c0.y + v.z * c0.z,
        v.x * c0.y + v.y * c0.w + v.z * c1.x,
        v.x * c0.z + v.y * c1.x + v.z * c1.y,
    );
    let cross_r = max_reach * sqrt(max(dot(u, su), 0.0));
    let along_r = min(max_reach * sqrt(max(dot(v, sv), 0.0)), cross_r * 1.4);

    // One area-matched triangle per segment (base at a - c_a, apex at b + c_b).
    // For a point splat (length_px == 0), this forms a centered equilateral triangle
    // with area pi * cross_r * along_r.
    let taper = max(c1.z, 0.0);
    let w0 = mix(1.0, taper, span.x);
    let w1 = mix(1.0, taper, span.y);
    let c_a = 0.777605 * along_r;
    let c_b = 1.555210 * along_r;
    let height = length_px + c_a + c_b;
    let half_w = cross_r * (w0 + w1) * (length_px + 1.5707963 * along_r) / max(height, 1e-4);

    if vertex_index == 2u {
        return vec4(b.xy + (tangent * c_b) / half_viewport, min(b.z, 1.0), 1.0);
    }
    let side = select(-1.0, 1.0, vertex_index == 1u);
    return vec4(
        a.xy + (across * (side * half_w) - tangent * c_a) / half_viewport,
        min(a.z, 1.0),
        1.0,
    );
}
