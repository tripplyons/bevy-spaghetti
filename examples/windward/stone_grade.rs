//! Regrades the rock scans' colors as they load.
//!
//! The scans are warm desert stone, with a red-to-blue ratio of about 2.2.
//! Cascade andesite and basalt are close to neutral grey. Lighting gains can
//! only shift every color the same way, which would also turn the moss blue,
//! so this pass desaturates warm segments and keeps green ones.

use bevy::prelude::*;
use bevy_spaghetti::{GaussianSegment, SpaghettiAsset};

use crate::landscape::smoothstep;

/// Path fragments of the rock clouds that this pass regrades.
const ROCKS: [&str; 3] = ["windward/basalt", "windward/granite", "windward/stone-"];

/// Saturation kept for warm (red-brown) and green (moss) segments.
const WARM_SATURATION: f32 = 0.3;
const MOSS_SATURATION: f32 = 0.85;
/// The neutral part leans a little cool, like weathered andesite.
const NEUTRAL: Vec3 = Vec3::new(0.99, 1.0, 1.02);

/// Index of the color within a segment's twenty floats.
const COLOR: usize = 16;

pub fn grade_stones(
    mut events: MessageReader<AssetEvent<SpaghettiAsset>>,
    server: Res<AssetServer>,
    mut assets: ResMut<Assets<SpaghettiAsset>>,
) {
    for event in events.read() {
        let AssetEvent::LoadedWithDependencies { id } = *event else {
            continue;
        };
        let Some(path) = server.get_path(id) else {
            continue;
        };
        let path = path.path().to_string_lossy().replace('\\', "/");
        if !ROCKS.iter().any(|rock| path.contains(rock)) {
            continue;
        }
        let Some(asset) = assets.get(id) else {
            continue;
        };
        let segments = asset.segments().iter().map(|&s| grade(s)).collect();
        let detail = asset.detail_transitions().to_vec();
        match SpaghettiAsset::with_detail(segments, &detail) {
            Ok(graded) => {
                // Inserting a fresh asset sends `Modified`, not
                // `LoadedWithDependencies`, so this runs once per asset.
                let _ = assets.insert(id, graded);
            }
            Err(error) => warn!("could not regrade {path}: {error:?}"),
        }
    }
}

fn grade(segment: GaussianSegment) -> GaussianSegment {
    let mut fields: [f32; 20] = bytemuck::cast(segment);
    let color = Vec3::from_slice(&fields[COLOR..COLOR + 3]);
    let luma = color.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let warm = smoothstep(-0.004, 0.02, color.x - color.y);
    let saturation = MOSS_SATURATION + (WARM_SATURATION - MOSS_SATURATION) * warm;
    let graded = luma * NEUTRAL + (color - Vec3::splat(luma)) * saturation;
    fields[COLOR..COLOR + 3].copy_from_slice(&graded.max(Vec3::ZERO).to_array());
    bytemuck::cast(fields)
}
