import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
};

@if(DISTANCE_FOG)
import bevy_pbr::render::{
    mesh_view_bindings::{fog, view},
    mesh_view_types::FOG_MODE_EXPONENTIAL_SQUARED,
    pbr_functions::apply_fog,
};

@fragment
fn fragment(vertex: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var surface = pbr_input_from_standard_material(vertex, front);
    surface.material.base_color = alpha_discard(
        surface.material.flags, surface.material.alpha_cutoff, surface.material.base_color,
    );
    var color = apply_pbr_lighting(surface);
    @if(DISTANCE_FOG) {
        var air = fog;
        if air.mode == FOG_MODE_EXPONENTIAL_SQUARED {
            // The valley's aerosol layer thins with altitude. Approximate the
            // mean density along the ray through an exponential layer. The
            // constant share accounts for haze above the valley.
            let height = max(surface.world_position.y - 9.0, 0.0) / 38.0;
            let mean_density = 1.0 / (1.0 + height * (0.5 + height / 12.0));
            air.be.x *= 0.42 + 0.58 * mean_density;
        }
        color = apply_fog(
            air, color, surface.world_position.xyz, view.world_position.xyz, surface.frag_coord.xy,
        );
    }
    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(surface, color);
    return out;
}
