import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    mesh_view_bindings::{lights, view},
    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
    shadows::fetch_directional_shadow,
};

// Radius of the sun-visibility blur on the hiker, in meters. Grass and crown
// splat shadows are stochastic at cascade-texel scale; sampled as-is on a
// body they read as camouflage blobs rather than dappled shade.
const SOFT_SHADOW_RADIUS: f32 = 0.10;
const SOFT_SHADOW_TAPS: u32 = 12u;

// Sun visibility averaged over a disc in the surface's tangent plane. Taps
// stay on the local surface plane, so a tilted surface does not compare
// against its own depth and shadow itself.
fn soft_directional_shadow(
    light_id: u32,
    world_position: vec4<f32>,
    normal: vec3<f32>,
    view_z: f32,
    frag_coord: vec2<f32>,
) -> f32 {
    let helper = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
    let tangent = normalize(cross(helper, normal));
    let bitangent = cross(normal, tangent);
    var sum = fetch_directional_shadow(light_id, world_position, normal, view_z, frag_coord);
    for (var i = 0u; i < SOFT_SHADOW_TAPS; i++) {
        // Vogel disc: even coverage with a fixed pattern, so no added noise.
        let r = SOFT_SHADOW_RADIUS * sqrt((f32(i) + 0.5) / f32(SOFT_SHADOW_TAPS));
        let theta = f32(i) * 2.39996;
        let offset = (tangent * cos(theta) + bitangent * sin(theta)) * r;
        sum += fetch_directional_shadow(
            light_id,
            vec4(world_position.xyz + offset, world_position.w),
            normal,
            view_z,
            frag_coord,
        );
    }
    return sum / f32(SOFT_SHADOW_TAPS + 1u);
}

struct HikerUniform {
    // xyz: unit direction toward the sun, w: hiker feet world Y.
    sun_and_feet: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var<uniform> hiker: HikerUniform;

@fragment
fn fragment(vertex: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var surface = pbr_input_from_standard_material(vertex, front);
    let base_rgb = surface.material.base_color.rgb;
    let luma = dot(base_rgb, vec3(0.2126, 0.7152, 0.0722));
    let local_y = vertex.world_position.y - hiker.sun_and_feet.w;
    let sun = hiker.sun_and_feet.xyz;

    // Classify garment and skin zones from height above feet and scan albedo.
    let boot_mask = 1.0 - smoothstep(0.19, 0.24, local_y);
    let head_mask = smoothstep(1.47, 1.53, local_y);
    let hand_zone = smoothstep(0.52, 0.60, local_y) * (1.0 - smoothstep(1.04, 1.12, local_y));
    let skin_chroma = smoothstep(0.020, 0.065, base_rgb.r - base_rgb.b)
        * smoothstep(0.004, 0.030, base_rgb.g - base_rgb.b)
        * (1.0 - smoothstep(1.55, 1.95, base_rgb.r / max(base_rgb.g, 0.01)))
        * smoothstep(0.08, 0.16, luma);
    let skin_mask = clamp((head_mask + hand_zone) * skin_chroma, 0.0, 1.0);
    let upper_mask = smoothstep(0.88, 0.94, local_y) * (1.0 - head_mask);
    let shirt_mask = upper_mask * (1.0 - skin_mask);
    let pants_mask = (1.0 - boot_mask) * (1.0 - smoothstep(0.88, 0.94, local_y)) * (1.0 - skin_mask);
    let cloth_mask = clamp(shirt_mask + pants_mask, 0.0, 1.0);

    // Per-material roughness (matte flannel/twill, natural skin, weathered leather boots).
    let roughness = mix(
        mix(0.92, 0.88, pants_mask),
        0.58,
        boot_mask,
    );
    surface.material.perceptual_roughness = mix(roughness, 0.56, skin_mask);

    // Gently rein in blown-out scan skin highlights and ground the boot soles.
    let sole_damp = 1.0 - 0.18 * (1.0 - smoothstep(0.0, 0.07, local_y));
    let skin_balance = mix(1.0, 0.86, skin_mask);
    surface.material.base_color = vec4(
        base_rgb * sole_damp * skin_balance,
        surface.material.base_color.a,
    );
    surface.diffuse_occlusion *= mix(0.88, 1.0, smoothstep(0.0, 0.14, local_y));

    // Split direct directional light out of Bevy's result so its shadow can be
    // replaced with a wider average: `no_sun` keeps ambient and other lights,
    // `unshadowed` adds directional light at full visibility.
    let receives_shadows = (surface.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u;
    var no_sun_input = surface;
    no_sun_input.directional_shadow_factor = 0.0;
    let no_sun = apply_pbr_lighting(no_sun_input);
    var unshadowed_input = surface;
    unshadowed_input.flags &= ~MESH_FLAGS_SHADOW_RECEIVER_BIT;
    let color = apply_pbr_lighting(unshadowed_input);
    var visibility = 1.0;
    if receives_shadows {
        let view_z = dot(
            vec4(view.view_from_world[0].z, view.view_from_world[1].z,
                view.view_from_world[2].z, view.view_from_world[3].z),
            surface.world_position,
        );
        // Weight each shadowed light by its diffuse share of the direct term.
        var weighted = 0.0;
        var weight_sum = 0.0;
        for (var i = 0u; i < lights.n_directional_lights; i++) {
            let light = lights.directional_lights[i];
            if (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) == 0u {
                continue;
            }
            let w = max(dot(surface.N, light.direction_to_light), 0.0)
                * dot(light.color.rgb, vec3(0.2126, 0.7152, 0.0722)) + 1e-6;
            weighted += w * soft_directional_shadow(
                i, surface.world_position, surface.world_normal, view_z, surface.frag_coord.xy,
            );
            weight_sum += w;
        }
        visibility = select(1.0, weighted / weight_sum, weight_sum > 0.0)
            * surface.directional_shadow_factor;
    }
    var lit = no_sun.rgb + (color.rgb - no_sun.rgb) * visibility;

    let n_dot_l = dot(surface.N, sun);
    let n_dot_v = clamp(dot(surface.N, surface.V), 0.0, 1.0);

    // Warm subsurface wrap-lighting softens the shadow terminator on face and hands.
    if skin_mask > 0.01 {
        let wrap = pow(clamp((n_dot_l + 0.44) / 1.44, 0.0, 1.0), 1.65)
            * (1.0 - clamp(n_dot_l, 0.0, 1.0));
        lit += surface.material.base_color.rgb * vec3(0.46, 0.17, 0.09) * wrap * skin_mask * 0.40;
    }

    // Soft grazing fabric sheen (brushed flannel nap and heavy twill rim).
    if cloth_mask > 0.01 {
        let grazing = pow(1.0 - n_dot_v, 3.2);
        let sky_lift = 0.40 + 0.60 * clamp(n_dot_l * 0.5 + 0.5, 0.0, 1.0);
        let sheen_weight = (shirt_mask * 0.09 + pants_mask * 0.07)
            * (0.35 + 0.65 * smoothstep(0.02, 0.18, luma));
        let sheen_tint = mix(
            vec3(0.44, 0.50, 0.62),
            surface.material.base_color.rgb * 1.4 + vec3(0.05),
            0.60,
        );
        lit += sheen_tint * grazing * sky_lift * sheen_weight;
    }

    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(surface, vec4(lit, color.a));
    return out;
}
