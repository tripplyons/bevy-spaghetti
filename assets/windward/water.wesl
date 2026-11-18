import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
};

struct Lake {
    reflected_clip_from_world: mat4x4<f32>,
    seconds: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var<uniform> lake: Lake;
@group(constants::MATERIAL_BIND_GROUP) @binding(101)
var reflected_scene: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(102)
var reflected_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(103)
var depth_map: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104)
var depth_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(105)
var sediment: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(106)
var sediment_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(107)
var bank_color: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(108)
var bank_sampler: sampler;

const WIND: vec2<f32> = vec2(0.50, 0.866);

fn hash_gradient(cell: vec2<f32>) -> vec2<f32> {
    var h = bitcast<vec2<u32>>(vec2<i32>(cell));
    h = h * vec2(1664525u, 1013904223u) + vec2(h.y, h.x) * 22695477u;
    h ^= h >> vec2(16u);
    h *= vec2(0x7feb352du, 0x846ca68bu);
    h ^= h >> vec2(15u);
    let angle = f32(h.x ^ h.y) * (6.2831853 / 4294967296.0);
    return vec2(cos(angle), sin(angle));
}

// Gradient noise and its analytic derivative, as (value, d/dx, d/dy).
fn gradient_noise(p: vec2<f32>) -> vec3<f32> {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let ga = hash_gradient(cell);
    let gb = hash_gradient(cell + vec2(1.0, 0.0));
    let gc = hash_gradient(cell + vec2(0.0, 1.0));
    let gd = hash_gradient(cell + vec2(1.0, 1.0));
    let va = dot(ga, f);
    let vb = dot(gb, f - vec2(1.0, 0.0));
    let vc = dot(gc, f - vec2(0.0, 1.0));
    let vd = dot(gd, f - vec2(1.0, 1.0));
    let k = va - vb - vc + vd;
    let value = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
    let derivative = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
        + du * (u.yx * k + vec2(vb, vc) - va);
    return vec3(value, derivative);
}

// Slope of one wind-driven ripple octave. Octaves finer than a pixel fade out.
fn ripple(p: vec2<f32>, pixel: f32, frequency: f32, height: f32, speed: f32, turn: f32) -> vec2<f32> {
    let c = cos(turn);
    let s = sin(turn);
    let rotation = mat2x2(c, s, -s, c);
    let q = rotation * (p * frequency) - WIND * (lake.seconds * speed * frequency);
    let footprint = pixel * frequency;
    let attenuation = exp(-0.6 * footprint * footprint);
    // The chain rule brings back the frequency and the rotation.
    return transpose(rotation) * gradient_noise(q).yz * (height * frequency * attenuation);
}

// Interleaved gradient noise: a cheap, well-spread per-pixel threshold.
fn screen_noise(pixel: vec2<f32>, seed: f32) -> f32 {
    let p = pixel + seed * vec2(47.0, 17.0);
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Surface-gradient contribution of a deep-water traveling wave. Suppress
// waves that are smaller than a pixel in the projected surface.
fn wave(
    p: vec2<f32>, pixel: f32, direction: vec2<f32>,
    k: f32, slope: f32, phase: f32,
) -> vec2<f32> {
    let displacement = dot(p, direction) * k - sqrt(9.81 * k) * lake.seconds + phase;
    let footprint = pixel * k;
    let attenuation = exp(-0.35 * footprint * footprint);
    return direction * (cos(displacement) * slope * attenuation);
}

@fragment
fn fragment(vertex: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var surface = pbr_input_from_standard_material(vertex, front);
    let p = vertex.world_position.xz;
    let pixel = max(length(dpdx(p)), length(dpdy(p)));
    let front_phase = sin(dot(p, vec2(0.23, -0.15)) - lake.seconds * 0.41);
    let shear = vec2(front_phase * 0.17, 0.13 * sin(p.x * 0.31 + p.y * 0.41));
    let q = p + shear;
    let slope =
        wave(q, pixel, vec2(0.45, 0.893), 2.52, 0.030, 1.4) +
        wave(q, pixel, vec2(0.843, 0.538), 4.90, 0.043, 2.8) +
        wave(q, pixel, vec2(0.122, 0.993), 6.74, 0.025, 5.5) +
        wave(q, pixel, vec2(0.705, -0.709), 8.73, 0.019, 3.6) +
        wave(q, pixel, vec2(-0.258, 0.966), 13.36, 0.024, 0.5) +
        wave(q, pixel, vec2(0.98, 0.20), 21.17, 0.012, 3.3);
    let uv = (p - vec2(10.0, -14.0)) / vec2(28.0, 42.0) + 0.5;
    // Signed depth in meters; it reaches zero on the waterline.
    let depth = max(textureSample(depth_map, depth_sampler, uv).r, 0.0);
    let near_bank = 1.0 - smoothstep(0.04, 0.62, depth);
    // Wind ripples. Slow patches of stronger wind ("cat's paws") drift
    // across the lake and modulate them.
    let wind_patch = gradient_noise(p * 0.085 - WIND * lake.seconds * 0.06).x;
    let paws = 0.45 + 1.1 * smoothstep(-0.25, 0.35, wind_patch);
    let ripples = paws * (
        ripple(p, pixel, 1.9, 0.020, 0.34, 0.3) +
        ripple(p, pixel, 4.3, 0.0085, 0.22, 1.9) +
        ripple(p, pixel, 9.7, 0.0032, 0.15, 4.1)
    );
    // Waves need open water to build, and they die in the shallows.
    let calm = smoothstep(0.0, 0.25, depth);
    let gust = (0.74 + 0.22 * front_phase) * (1.0 - 0.52 * near_bank) * calm;
    // A sheltered tarn at dusk: gentle slopes keep the reflection readable.
    let slope_total = (slope + ripples) * gust * 0.5;
    surface.N = normalize(vec3(-slope_total.x, 1.0, -slope_total.y));
    let cos_theta = clamp(dot(surface.V, surface.N), 0.0, 1.0);
    // Refraction keeps the underwater ray closer to vertical than the viewing
    // ray. Absorption removes red first as its path through the water grows.
    let ratio = 1.0 / 1.333;
    let transmitted_cos = sqrt(1.0 - ratio * ratio * (1.0 - cos_theta * cos_theta));
    let travel = depth / max(0.2, transmitted_cos);
    let transmission = exp(-vec3(2.7, 1.55, 1.1) * travel);
    let refracted = p - surface.V.xz * (travel * ratio);
    let sediment_color = textureSample(sediment, sediment_sampler, refracted * 0.49).rgb * 0.44;
    // At the waterline the bed is the same soil as the bank: the terrain's
    // albedo tile times its wet vertex tint (see `landscape::terrain`).
    let soil = textureSample(bank_color, bank_sampler, refracted * 0.41).rgb * vec3(0.44, 0.42, 0.35);
    // Soaked soil is darker, and the darkening starts within a few centimeters.
    let soaked = smoothstep(0.0, 0.025, depth);
    let bed = mix(soil, sediment_color, smoothstep(0.03, 0.45, depth)) * (1.0 - 0.38 * soaked);
    let pool = vec3(0.016, 0.038, 0.037);
    surface.material.base_color = vec4(mix(pool, bed, transmission), 1.0);
    // Blend from the bank's matte soil into a glossy wet film, then open water.
    let film = smoothstep(0.0, 0.035, depth);
    let water_roughness = 0.105 + 0.028 * gust + 0.065 * near_bank;
    surface.material.perceptual_roughness = mix(0.67, water_roughness, film);
    surface.material.reflectance = vec3(mix(0.29, 0.36, film));
    var color = apply_pbr_lighting(surface);
    let projected = lake.reflected_clip_from_world * vertex.world_position;
    let mirror_uv = projected.xy / max(0.01, projected.w) * vec2(0.5, -0.5) + 0.5;
    let wave_offset = 0.092 * slope_total;
    let sample_uv = mirror_uv + wave_offset;
    let border = min(sample_uv, vec2(1.0) - sample_uv);
    let valid = smoothstep(0.0, 0.045, min(border.x, border.y));
    // Rotate a short blur kernel per pixel and jitter it by a fraction of a
    // texel. The noise hides upsampling steps in smooth sky gradients.
    let texel = 1.0 / vec2<f32>(textureDimensions(reflected_scene));
    let turn = 6.2831853 * screen_noise(vertex.position.xy, 0.0);
    let jitter = (vec2(screen_noise(vertex.position.xy, 1.0), screen_noise(vertex.position.xy, 2.0)) - 0.5) * texel;
    let spread = texel * (0.9 + 1.4 * near_bank);
    let axis = vec2(cos(turn), sin(turn)) * spread;
    let center_uv = sample_uv + jitter;
    let reflected_color =
        textureSample(reflected_scene, reflected_sampler, center_uv).rgb * 0.40 +
        textureSample(reflected_scene, reflected_sampler, center_uv + axis).rgb * 0.15 +
        textureSample(reflected_scene, reflected_sampler, center_uv - axis).rgb * 0.15 +
        textureSample(reflected_scene, reflected_sampler, center_uv + vec2(-axis.y, axis.x)).rgb * 0.15 +
        textureSample(reflected_scene, reflected_sampler, center_uv - vec2(-axis.y, axis.x)).rgb * 0.15;
    // Looking down into the shallows, the bed shows through and the
    // reflection fades out toward the waterline. At grazing angles even a
    // thin film stays a mirror, and the rough soil's sky sheen would read
    // as a pale rim along the far bank.
    let grazing = smoothstep(0.6, 0.92, 1.0 - cos_theta);
    // The grazing mirror starts only once the water is a few centimetres deep.
    let sheet = mix(smoothstep(0.0, 0.12, depth), 1.0, grazing * smoothstep(0.02, 0.08, depth));
    let fresnel = (0.0204 + 0.9796 * pow(1.0 - cos_theta, 5.0)) * sheet;
    color = vec4(mix(color.rgb, reflected_color, valid * fresnel * 0.94), color.a);
    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(surface, color);
    return out;
}
