import bevy_pbr::render::{
    forward_io::{Vertex, VertexOutput, decompress_vertex},
    mesh_functions,
    mesh_view_bindings::globals,
    view_transformations::position_world_to_clip,
};

// Hiker feet in xyz, push radius in w.
@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var<uniform> hiker: vec4<f32>;

fn hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2(0.1031, 0.1030));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let w = f * f * (3.0 - 2.0 * f);
    let a = hash(cell);
    let b = hash(cell + vec2(1.0, 0.0));
    let c = hash(cell + vec2(0.0, 1.0));
    let d = hash(cell + vec2(1.0, 1.0));
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

// The same gust field that drives the reed springs, as a tip slope.
fn wind(at: vec2<f32>, time: f32) -> vec2<f32> {
    let front = sin(dot(at, vec2(-0.29, 0.19)) + time * 1.8);
    let pocket = value_noise(at * 0.12 - vec2(0.82, 0.32) * time);
    let sway = vec2(-0.30, 0.70) * (0.10 * sin(time * 1.34 + at.x * 0.22));
    return vec2(0.82, 0.47) * (0.14 + 0.50 * pocket * pocket + 0.17 * front) + sway;
}

@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    let vertex = decompress_vertex(vertex_in, vertex_in.instance_index);
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);
    var world = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(vertex.position, 1.0));

    let height = vertex.uv.x;
    let t = vertex.uv.y;
    let rise = height * t;
    let time = globals.time;
    let at = world.xz;
    let flutter = sin(time * 6.3 + hash(floor(at * 9.0)) * 6.28) * 0.06;
    var offset = (wind(at, time) * 0.55 + vec2(flutter, flutter * 0.6)) * rise * t;

    // Blades within the push radius lean away from the hiker's legs.
    let away = at - hiker.xz;
    let distance = length(away);
    let contact = max(1.0 - distance / hiker.w, 0.0);
    let below = step(world.y - height - 0.25, hiker.y);
    offset += away / max(distance, 0.05) * contact * contact * rise * 1.4 * below;

    let reach = min(length(offset), 0.92 * rise);
    offset = offset / max(length(offset), 1e-5) * reach;
    world.x += offset.x;
    world.z += offset.y;
    // Shorten vertically so the bent blade keeps roughly the same length.
    world.y -= rise - sqrt(max(rise * rise - reach * reach, 0.0));

    var out: VertexOutput;
    out.world_position = world;
    out.position = position_world_to_clip(world.xyz);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex_in.instance_index);
    out.uv = vertex.uv;
    out.color = vertex.color;
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex_in.instance_index;
@if(VISIBILITY_RANGE_DITHER)
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        vertex_in.instance_index, world_from_local[3]);
    return out;
}
