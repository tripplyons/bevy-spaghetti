import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, calculate_tbn_mikktspace, main_pass_post_lighting_processing},
};

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var trail_color: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(101)
var tile_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(102)
var trail_normal: texture_2d<f32>;

@fragment
fn fragment(vertex: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var surface = pbr_input_from_standard_material(vertex, front);
    let wet = vertex.uv_b.y;
    let worn = vertex.uv_b.x;
    let tile_uv = vertex.uv * 1.11;
    let dx = dpdx(tile_uv);
    let dy = dpdy(tile_uv);
    surface.N = normalize(mix(vertex.world_normal, surface.N, 0.64));
    if worn > 0.001 {
        let albedo = textureSampleGrad(trail_color, tile_sampler, tile_uv, dx, dy).rgb;
        let map = textureSampleGrad(trail_normal, tile_sampler, tile_uv, dx, dy);
        let tangent = calculate_tbn_mikktspace(vertex.world_normal, vertex.world_tangent);
        let detail = (map.rgb * 2.0 - 1.0) * vec3(0.57, 0.57, 1.0);
        let rock_normal = normalize(tangent * detail);
        surface.material.base_color = vec4(
            mix(surface.material.base_color.rgb, albedo * (0.85 - 0.37 * wet), worn),
            1.0
        );
        surface.N = normalize(mix(surface.N, rock_normal, worn));
    }
    surface.material.perceptual_roughness = mix(0.93, 0.67, wet);
    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(surface, apply_pbr_lighting(surface));
    return out;
}
