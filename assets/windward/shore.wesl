const BANK_SOIL_TINT: vec3<f32> = vec3(0.44, 0.42, 0.35);
const GRAVEL_TINT: vec3<f32> = vec3(1.48, 1.44, 1.32);
const GRAVEL_SCALE: f32 = 0.41 * 1.11;

// Break up the sediment margin with patches of silt between the pebbles.
fn gravel_coverage(p: vec2<f32>) -> f32 {
    let pockets = sin(dot(p, vec2(2.3, 1.2))) * cos(dot(p, vec2(-1.1, 1.8)));
    return 0.70 + 0.19 * pockets;
}

fn gravel_normal(map: vec3<f32>) -> vec3<f32> {
    return (map * 2.0 - 1.0) * vec3(0.57, 0.57, 1.0);
}
