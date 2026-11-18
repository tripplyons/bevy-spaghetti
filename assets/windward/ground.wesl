import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, calculate_tbn_mikktspace, main_pass_post_lighting_processing},
};
import super::shore::{GRAVEL_TINT, gravel_coverage, gravel_normal};

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var trail_color: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(101)
var tile_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(102)
var trail_normal: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103)
var litter_color: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104)
var litter_surface: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(105)
var<uniform> water_level: f32;

@fragment
fn fragment(vertex: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var surface = pbr_input_from_standard_material(vertex, front);
    let wet = vertex.uv_b.y;
    let worn = vertex.uv_b.x;
    let dx = dpdx(vertex.uv);
    let dy = dpdy(vertex.uv);
    surface.N = normalize(mix(vertex.world_normal, surface.N, 0.64));
    surface.material.base_color.a = 1.0;
    surface.material.perceptual_roughness = mix(0.93, 0.67, wet);

    // The scanned tile covers two meters. Blend back to the terrain tint as
    // the projected footprint grows, and skip the samples beyond that point.
    let litter_scale = 0.5 / 0.41;
    let footprint = litter_scale * max(length(dx), length(dy));
    let litter = smoothstep(0.08, 0.90, vertex.color.a)
        * (1.0 - worn) * (1.0 - smoothstep(0.10, 0.24, footprint));
    if litter > 0.001 {
        let uv = vertex.uv * litter_scale;
        let albedo = textureSampleGrad(litter_color, tile_sampler, uv, dx * litter_scale, dy * litter_scale).rgb;
        let maps = textureSampleGrad(litter_surface, tile_sampler, uv, dx * litter_scale, dy * litter_scale);
        // Packed linear channels are normal XY, roughness and occlusion.
        let relief = (maps.rg * 2.0 - 1.0) * 1.06;
        let local_normal = vec3(relief, sqrt(max(0.0, 1.0 - dot(relief, relief))));
        let tangent = calculate_tbn_mikktspace(vertex.world_normal, vertex.world_tangent);
        let tint = clamp(vertex.color.rgb / vec3(0.40, 0.27, 0.18), vec3(0.45), vec3(1.25));
        let mean = vec3(0.35, 0.23, 0.14);
        let needles = max(vec3(0.0), mean + (albedo - mean) * 1.5);
        surface.material.base_color = vec4(
            mix(surface.material.base_color.rgb, needles * tint * 0.55, 0.72 * litter),
            1.0
        );
        surface.N = normalize(mix(surface.N, tangent * local_normal, 0.65 * litter));
        surface.material.perceptual_roughness = mix(surface.material.perceptual_roughness, maps.b, litter);
        surface.diffuse_occlusion *= 1.0 - 0.34 * litter * (1.0 - maps.a);
    }
    let shore_height = vertex.world_position.y - water_level;
    var gravel = 0.0;
    if shore_height < 0.35 && shore_height > -0.13 {
        let coverage = gravel_coverage(vertex.world_position.xz);
        gravel = coverage * (1.0 - smoothstep(0.025, 0.22 + 0.13 * coverage, shore_height));
    }
    if max(worn, gravel) > 0.001 {
        let tile_uv = vertex.uv * 1.11;
        let albedo = textureSampleGrad(trail_color, tile_sampler, tile_uv, dx * 1.11, dy * 1.11).rgb;
        let map = textureSampleGrad(trail_normal, tile_sampler, tile_uv, dx * 1.11, dy * 1.11);
        let tangent = calculate_tbn_mikktspace(vertex.world_normal, vertex.world_tangent);
        let rock_normal = normalize(tangent * gravel_normal(map.rgb));
        let bank = mix(surface.material.base_color.rgb, albedo * GRAVEL_TINT, gravel);
        surface.material.base_color = vec4(
            mix(bank, albedo * (0.85 - 0.37 * wet), worn),
            1.0
        );
        surface.N = normalize(mix(surface.N, rock_normal, mix(gravel, 1.0, worn)));
    }
    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(surface, apply_pbr_lighting(surface));
    return out;
}
