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
    let depth = textureSample(depth_map, depth_sampler, uv).r * 2.0;
    let near_bank = 1.0 - smoothstep(0.04, 0.62, depth);
    let gust = (0.74 + 0.22 * front_phase) * (1.0 - 0.52 * near_bank);
    surface.N = normalize(vec3(-slope.x * gust, 1.0, -slope.y * gust));
    let cos_theta = clamp(dot(surface.V, surface.N), 0.0, 1.0);
    // Refraction keeps the underwater ray closer to vertical than the viewing
    // ray. Absorption removes red first as its path through the water grows.
    let ratio = 1.0 / 1.333;
    let transmitted_cos = sqrt(1.0 - ratio * ratio * (1.0 - cos_theta * cos_theta));
    let travel = depth / max(0.2, transmitted_cos);
    let transmission = exp(-vec3(2.7, 1.55, 1.1) * travel);
    let refracted = p - surface.V.xz * (travel * ratio);
    let bed = textureSample(sediment, sediment_sampler, refracted * 0.49).rgb;
    let pool = vec3(0.016, 0.038, 0.037);
    surface.material.base_color = vec4(mix(pool, bed * 0.44, transmission), 1.0);
    surface.material.perceptual_roughness = 0.105 + 0.028 * gust + 0.065 * near_bank;
    var color = apply_pbr_lighting(surface);
    let projected = lake.reflected_clip_from_world * vertex.world_position;
    let mirror_uv = projected.xy / max(0.01, projected.w) * vec2(0.5, -0.5) + 0.5;
    let wave_offset = 0.092 * vec2(slope.x, slope.y) * gust;
    let sample_uv = mirror_uv + wave_offset;
    let border = min(sample_uv, vec2(1.0) - sample_uv);
    let valid = smoothstep(0.0, 0.045, min(border.x, border.y));
    let softness = vec2(0.002, 0.0012) * (0.6 + near_bank);
    let reflected_color =
        textureSample(reflected_scene, reflected_sampler, sample_uv).rgb * 0.52 +
        textureSample(reflected_scene, reflected_sampler, sample_uv + softness).rgb * 0.24 +
        textureSample(reflected_scene, reflected_sampler, sample_uv - softness).rgb * 0.24;
    let fresnel = 0.0204 + 0.9796 * pow(1.0 - cos_theta, 5.0);
    color = vec4(mix(color.rgb, reflected_color, valid * fresnel * 0.94), color.a);
    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(surface, color);
    return out;
}
