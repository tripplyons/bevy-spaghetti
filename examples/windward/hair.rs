//! Splat hair for the hiker.
//!
//! The scan paints dark brown hair onto a smooth matte scalp. In the low sun
//! that scalp lights up evenly and reads as a tan helmet. Short Gaussian
//! strands over the painted hair area give it a broken, fibrous silhouette
//! and self-shade, like real hair.
//!
//! The strands are built from the skinned mesh's own scalp vertices in bind
//! pose, so they sit on the head exactly. They ride on the `Head` joint, whose
//! inverse bind matrix maps the mesh's bind space into joint space.

use bevy::{
    mesh::{
        VertexAttributeValues,
        skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
    },
    prelude::*,
    world_serialization::WorldInstanceReady,
};
use bevy_spaghetti::{
    GaussianSegment, Spaghetti3d, SpaghettiAsset, SpaghettiLighting, SpaghettiReceiveShadows,
    SpaghettiSpecular,
};

use crate::landscape::{SUN, hash01, reflected_layers};

/// A small counter-based stream over `hash01`.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9);
        hash01(self.0)
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next()
    }
}

/// The crown whorl in the scan's frame, with the face toward +Z. Hair flows away from it over the scalp.
const CROWN: Vec3 = Vec3::new(0.0, 1.775, -0.035);

/// Whether a scalp vertex, in the scan's frame, lies under the painted hair. Measured
/// from the albedo: the front hairline is at about 1.71 m, and the hair
/// reaches lower toward the nape. Ears and face stay bare.
fn is_hair(at: Vec3) -> bool {
    let head = at.y > 1.52 && at.x.abs() < 0.1 && (-0.11..0.14).contains(&at.z);
    // The hairline drops from the forehead to the nape.
    let hairline = (1.60 + (at.z + 0.02) * 1.15).clamp(1.585, 1.712);
    let ear = at.x.abs() > 0.068 && at.y < 1.685;
    head && at.y > hairline && !ear
}

pub fn grow_hair(
    ready: On<WorldInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    skinned: Query<(&SkinnedMesh, &Mesh3d)>,
    names: Query<&Name>,
    meshes: Res<Assets<Mesh>>,
    bindposes: Res<Assets<SkinnedMeshInverseBindposes>>,
    mut clouds: ResMut<Assets<SpaghettiAsset>>,
) {
    let Some((skin, mesh)) = children
        .iter_descendants(ready.entity)
        .find_map(|entity| skinned.get(entity).ok())
    else {
        return;
    };
    let Some((index, &head)) = skin
        .joints
        .iter()
        .enumerate()
        .find(|(_, joint)| names.get(**joint).is_ok_and(|name| name.as_str() == "Head"))
    else {
        warn!("the hiker has no Head joint, so it gets no hair");
        return;
    };
    let (Some(mesh), Some(bindposes)) =
        (meshes.get(&mesh.0), bindposes.get(&skin.inverse_bindposes))
    else {
        return;
    };
    let (
        Some(VertexAttributeValues::Float32x3(positions)),
        Some(VertexAttributeValues::Float32x3(normals)),
    ) = (
        mesh.attribute(Mesh::ATTRIBUTE_POSITION),
        mesh.attribute(Mesh::ATTRIBUTE_NORMAL),
    )
    else {
        warn!("the hiker mesh has no main-world vertex data, so it gets no hair");
        return;
    };
    let Some(indices) = mesh.indices() else {
        return;
    };
    let triangles: Vec<usize> = indices.iter().collect();
    let strands = strands(positions, normals, &triangles);
    if strands.is_empty() {
        return;
    }
    let Ok(cloud) = SpaghettiAsset::new(strands) else {
        return;
    };
    commands.entity(head).with_child((
        Name::new("Hair"),
        Spaghetti3d(clouds.add(cloud)),
        SpaghettiLighting {
            direction_to_light: SUN,
            ambient: Vec3::new(0.26, 0.28, 0.31),
            directional: Vec3::new(0.80, 0.68, 0.52),
        },
        SpaghettiReceiveShadows,
        // Strands reflect the sun in a band across the crown, not a spot.
        SpaghettiSpecular {
            strength: 0.085,
            shininess: 70.0,
            strand: 1.0,
        },
        reflected_layers(),
        Transform::from_matrix(bindposes[index]),
    ));
}

/// Strand roots per square meter of scalp.
const DENSITY: f32 = 160_000.0;

fn strands(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    triangles: &[usize],
) -> Vec<GaussianSegment> {
    // The loader may turn the scan to face -Z. The nose reaches farther from
    // the neck axis than the back of the skull, so it shows which way is front.
    let (mut back, mut front) = (0.0_f32, 0.0_f32);
    for at in positions.iter().map(|&p| Vec3::from_array(p)) {
        if at.y > 1.6 && at.y < 1.72 && at.x.abs() < 0.03 {
            back = back.min(at.z);
            front = front.max(at.z);
        }
    }
    let facing = if front >= -back { 1.0 } else { -1.0 };
    // Work in the scan's frame, with the face toward +Z.
    let frame = Vec3::new(facing, 1.0, facing);

    // The scan's vertices are uneven, with bare patches on the scalp, so
    // scatter roots over the triangles by area instead.
    let mut rng = Rng(0x4a17);
    let mut roots = Vec::new();
    for &corners in triangles.as_chunks::<3>().0 {
        let at = corners.map(|i| Vec3::from_array(positions[i]));
        let centroid = (at[0] + at[1] + at[2]) / 3.0;
        if !is_hair(centroid * frame) {
            continue;
        }
        let area = 0.5 * (at[1] - at[0]).cross(at[2] - at[0]).length();
        let count = (area * DENSITY + rng.next()) as usize;
        for _ in 0..count {
            let (mut u, mut v) = (rng.next(), rng.next());
            if u + v > 1.0 {
                (u, v) = (1.0 - u, 1.0 - v);
            }
            let weights = [1.0 - u - v, u, v];
            let mut root = Vec3::ZERO;
            let mut normal = Vec3::ZERO;
            for (corner, weight) in corners.into_iter().zip(weights) {
                root += Vec3::from_array(positions[corner]) * weight;
                normal += Vec3::from_array(normals[corner]) * weight;
            }
            roots.push((root, normal.normalize_or(Vec3::Y)));
        }
    }

    let mut out = Vec::new();
    for (root, normal) in roots {
        // Comb away from the whorl and a little down, along the scalp.
        let away = (root - CROWN * frame + Vec3::new(0.0, -0.03, 0.0)).normalize_or(Vec3::NEG_Y);
        let flow = (away - normal * away.dot(normal)).normalize_or(normal.any_orthonormal_vector());
        let side = normal.cross(flow) * rng.range(-0.35, 0.35);
        let scan = root * frame;
        // Short back and sides, a little longer on top, and a short fringe
        // so the hair stops near the painted hairline.
        let top = ((scan.y - 1.66) / 0.1).clamp(0.0, 1.0);
        let fringe = 1.0 - 0.6 * crate::landscape::smoothstep(0.03, 0.09, scan.z);
        let length = rng.range(0.012, 0.022) * (0.5 + 0.5 * top) * fringe;
        let lift = rng.range(0.0022, 0.0038);
        let start = root + normal * lift;
        let middle = start + (flow + side) * 0.5 * length + normal * 0.0012;
        // Each strand bends back toward the scalp as it lies down.
        let end = start + (flow + side) * length;
        // Matches the scan's ginger-brown hair and beard.
        let shade = rng.range(0.6, 1.45);
        let color = Vec3::new(0.075, 0.044, 0.024) * shade;
        let width = rng.range(0.0010, 0.0017);
        out.push(strand(start, middle, normal, width, color));
        out.push(strand(middle, end, normal, width * 0.8, color * 1.25));
    }
    out
}

/// A thin round kernel along `start` → `end`.
fn strand(start: Vec3, end: Vec3, normal: Vec3, width: f32, color: Vec3) -> GaussianSegment {
    let along = (end - start).normalize_or(Vec3::Y);
    let across = normal
        .cross(along)
        .try_normalize()
        .unwrap_or_else(|| along.any_orthonormal_vector());
    let up = along.cross(across);
    GaussianSegment::new(
        start,
        end,
        Vec3::new(width * 0.6, width, width * 0.8),
        Quat::from_mat3(&Mat3::from_cols(along, across, up)),
        6.0 * width * start.distance(end),
        color,
    )
}
