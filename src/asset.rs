use std::{io::Write, sync::Arc};

use bevy_math::{Quat, Vec3};
use bytemuck::{Pod, Zeroable};
use thiserror::Error;

const MAGIC: &[u8; 8] = b"SPAG\x01\0\0\0";
const HEADER_BYTES: usize = 12;
const SEGMENT_BYTES: usize = size_of::<GaussianSegment>();
/// Caps GPU storage at 80 MB per cloud.
const MAX_SEGMENTS: usize = 1_000_000;

/// A finite segment convolved with a Gaussian of constant covariance.
///
/// Endpoints and standard deviations are in asset units. `rotation` maps the
/// kernel's principal axes into asset space. `mass` is optical area in squared
/// asset units, and `color` is linear RGB. Equal endpoints represent a point.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GaussianSegment {
    start_mass: [f32; 4],
    end: [f32; 4],
    sigma: [f32; 4],
    rotation: [f32; 4],
    color: [f32; 4],
}

impl GaussianSegment {
    pub fn new(
        start: Vec3,
        end: Vec3,
        sigma: Vec3,
        rotation: Quat,
        mass: f32,
        color: Vec3,
    ) -> Self {
        Self {
            start_mass: start.extend(mass).to_array(),
            end: end.extend(0.0).to_array(),
            sigma: sigma.extend(0.0).to_array(),
            rotation: rotation.to_array(),
            color: color.extend(0.0).to_array(),
        }
    }

    pub fn start(&self) -> Vec3 {
        Vec3::from_slice(&self.start_mass)
    }

    pub fn end(&self) -> Vec3 {
        Vec3::from_slice(&self.end)
    }

    pub fn sigma(&self) -> Vec3 {
        Vec3::from_slice(&self.sigma)
    }

    pub fn rotation(&self) -> Quat {
        Quat::from_array(self.rotation)
    }

    pub fn mass(&self) -> f32 {
        self.start_mass[3]
    }

    pub fn color(&self) -> Vec3 {
        Vec3::from_slice(&self.color)
    }

    /// Sets the bend height of each endpoint above the root, in world units.
    ///
    /// A renderer with wind enabled pins endpoints with zero bend height.
    pub fn with_bend_heights(mut self, start: f32, end: f32) -> Self {
        self.end[3] = start.max(0.0);
        self.sigma[3] = end.max(0.0);
        self
    }

    #[cfg(feature = "render")]
    pub(crate) fn max_bend_height(&self) -> f32 {
        self.end[3].max(self.sigma[3]).max(0.0)
    }

    /// Sets the blade height, endpoint fractions along it, and the ratio of
    /// end width to start width for a tapered foliage cutout.
    ///
    /// The stored height is rounded to the nearest millimeter, up to 8.191 m.
    /// Use the same height and fraction where two segments meet.
    pub fn with_blade_profile(mut self, height: f32, fractions: [f32; 2], end_width: f32) -> Self {
        let height_mm = (height * 1000.0).round().clamp(1.0, 8191.0) as u32;
        let height = height_mm as f32 * 0.001;
        self = self.with_bend_heights(
            height * fractions[0].clamp(0.0, 1.0),
            height * fractions[1].clamp(0.0, 1.0),
        );
        let width = (end_width.clamp(0.0, 1.0) * 1023.0).round() as u32;
        // Keep the profile in a finite float with a fixed exponent. The
        // mantissa holds 13 height bits followed by 10 taper bits.
        self.color[3] = f32::from_bits(0x3f80_0000 | (height_mm << 10) | width);
        self
    }

    fn is_valid(&self) -> bool {
        self.start().is_finite()
            && self.end().is_finite()
            && self.sigma().is_finite()
            && self.sigma().min_element() > 0.0
            && self.rotation().is_normalized()
            && self.mass().is_finite()
            && self.mass() >= 0.0
            && self.color().is_finite()
            && self.color().min_element() >= 0.0
    }
}

#[derive(Debug, Error)]
pub enum AssetError {
    #[error("expected a version 1 SPAG file")]
    Header,
    #[error("expected between 1 and {MAX_SEGMENTS} segments, found {0}")]
    Count(usize),
    #[error("file length does not match its segment count")]
    Length,
    #[error("segment {0} has invalid endpoints, covariance, mass, or color")]
    InvalidSegment(usize),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Immutable, shareable data for one cloud of segment splats.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "render",
    derive(bevy::asset::Asset, bevy::reflect::TypePath)
)]
pub struct SpaghettiAsset {
    segments: Arc<Vec<GaussianSegment>>,
    bounds: (Vec3, Vec3),
}

impl SpaghettiAsset {
    pub fn new(segments: Vec<GaussianSegment>) -> Result<Self, AssetError> {
        check_count(segments.len())?;
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for (index, segment) in segments.iter().enumerate() {
            // A sphere of radius four times the largest standard deviation
            // bounds the kernel regardless of its principal axes.
            let radius = Vec3::splat(4.0 * segment.sigma().max_element());
            let low = segment.start().min(segment.end()) - radius;
            let high = segment.start().max(segment.end()) + radius;
            if !segment.is_valid() || !low.is_finite() || !high.is_finite() {
                return Err(AssetError::InvalidSegment(index));
            }
            min = min.min(low);
            max = max.max(high);
        }
        Ok(Self {
            segments: Arc::new(segments),
            bounds: (min, max),
        })
    }

    pub fn segments(&self) -> &[GaussianSegment] {
        &self.segments
    }

    /// Local-space minimum and maximum, including four standard deviations
    /// of kernel support.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.bounds
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AssetError> {
        if !bytes.starts_with(MAGIC) || bytes.len() < HEADER_BYTES {
            return Err(AssetError::Header);
        }
        let count = u32::from_le_bytes(bytes[8..HEADER_BYTES].try_into().unwrap()) as usize;
        check_count(count)?;
        if bytes.len() != HEADER_BYTES + count * SEGMENT_BYTES {
            return Err(AssetError::Length);
        }

        let segments = bytes[HEADER_BYTES..]
            .as_chunks::<SEGMENT_BYTES>()
            .0
            .iter()
            .map(|record| {
                let mut fields = [0.0_f32; SEGMENT_BYTES / 4];
                for (field, &bytes) in fields.iter_mut().zip(record.as_chunks::<4>().0) {
                    *field = f32::from_le_bytes(bytes);
                }
                bytemuck::cast(fields)
            })
            .collect();
        Self::new(segments)
    }

    pub fn write(&self, mut writer: impl Write) -> Result<(), AssetError> {
        writer.write_all(MAGIC)?;
        writer.write_all(&(self.segments.len() as u32).to_le_bytes())?;
        for segment in self.segments.iter() {
            let fields: [f32; SEGMENT_BYTES / 4] = bytemuck::cast(*segment);
            let bytes: [[u8; 4]; SEGMENT_BYTES / 4] = fields.map(f32::to_le_bytes);
            writer.write_all(bytemuck::cast_slice(&bytes))?;
        }
        Ok(())
    }
}

fn check_count(count: usize) -> Result<(), AssetError> {
    if (1..=MAX_SEGMENTS).contains(&count) {
        Ok(())
    } else {
        Err(AssetError::Count(count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_round_trip_and_truncation() {
        let segment = GaussianSegment::new(
            Vec3::X,
            Vec3::Y,
            Vec3::new(0.1, 0.2, 0.3),
            Quat::from_rotation_x(0.3),
            0.4,
            Vec3::new(0.1, 0.9, 0.5),
        );
        let asset = SpaghettiAsset::new(vec![segment]).unwrap();
        let mut bytes = Vec::new();
        asset.write(&mut bytes).unwrap();
        assert_eq!(
            SpaghettiAsset::from_bytes(&bytes).unwrap().segments(),
            &[segment]
        );
        assert!(matches!(
            SpaghettiAsset::from_bytes(&bytes[..bytes.len() - 1]),
            Err(AssetError::Length)
        ));
    }
}
