use std::{io::Write, sync::Arc};

use bevy_math::{Quat, Vec3};
use bytemuck::{Pod, Zeroable};
use thiserror::Error;

const MAGIC_V1: &[u8; 8] = b"SPAG\x01\0\0\0";
const MAGIC_V2: &[u8; 8] = b"SPAG\x02\0\0\0";
const MAGIC_V3: &[u8; 8] = b"SPAG\x03\0\0\0";
const HEADER_BYTES: usize = 12;
const SEGMENT_BYTES: usize = size_of::<GaussianSegment>();
/// Caps GPU storage at 80 MB per cloud.
const MAX_SEGMENTS: usize = 1_000_000;
// The renderer sends this small table in the draw uniform.
pub(crate) const MAX_DETAIL_TRANSITIONS: usize = 3;
const TRANSITION_BYTES: usize = 12;
const PROFILE_MARKER: u32 = 0x3f80_0000;
const PROFILE_TAG_MASK: u32 = 0xff80_0000;
const PROFILE_WIDTH_MASK: u32 = 1023;

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

    #[cfg(feature = "render")]
    pub(crate) fn mean_bend_height(&self) -> f32 {
        self.end[3] * 0.5 + self.sigma[3] * 0.5
    }

    /// Sets the end width relative to the start width in the cutout renderer.
    ///
    /// One keeps a constant width and zero gives the narrowest tip. Use this
    /// independently of wind bending and leaf shading.
    pub fn with_cutout_taper(mut self, end_width: f32) -> Self {
        let width = (end_width.clamp(0.0, 1.0) * PROFILE_WIDTH_MASK as f32).round() as u32;
        let bits = self.color[3].to_bits();
        let profile = if bits & PROFILE_TAG_MASK == PROFILE_MARKER {
            bits & !PROFILE_WIDTH_MASK
        } else {
            PROFILE_MARKER
        };
        self.color[3] = f32::from_bits(profile | width);
        self
    }

    /// Returns the stored end-to-start width ratio for a cutout.
    pub fn cutout_end_width(&self) -> f32 {
        let bits = self.color[3].to_bits();
        if bits & PROFILE_TAG_MASK == PROFILE_MARKER {
            (bits & PROFILE_WIDTH_MASK) as f32 / PROFILE_WIDTH_MASK as f32
        } else {
            1.0
        }
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
        // Keep the profile in a finite float with a fixed exponent. The
        // mantissa holds 13 height bits followed by 10 taper bits.
        self = self.with_cutout_taper(end_width);
        let bits = self.color[3].to_bits() & (PROFILE_MARKER | PROFILE_WIDTH_MASK);
        self.color[3] = f32::from_bits(bits | (height_mm << 10));
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
    #[error("expected a version 1, 2, or 3 SPAG file")]
    Header,
    #[error("expected between 1 and {MAX_SEGMENTS} segments, found {0}")]
    Count(usize),
    #[error("file length does not match its segment count")]
    Length,
    #[error("segment {0} has invalid endpoints, covariance, mass, or color")]
    InvalidSegment(usize),
    #[error(
        "invalid detail transitions: expected up to {MAX_DETAIL_TRANSITIONS} ordered transitions between nonempty levels"
    )]
    InvalidDetail,
    #[error("occlusion direction must be finite and nonzero")]
    InvalidOcclusionDirection,
    #[error("occlusion density must be finite and nonnegative")]
    InvalidOcclusionDensity,
    #[error("occlusion factors must be between 0 and 1")]
    InvalidOcclusion,
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
    detail: Arc<[DetailTransition]>,
    occlusion: Arc<[[f32; 2]]>,
}

/// One transition in a cloud whose segments are grouped from finest to coarsest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetailTransition {
    /// Exclusive end index of the finer level's segments in the merged cloud.
    pub finer_end: u32,
    /// Projected pixels per asset unit at which the cross-fade starts and ends.
    /// Only the coarser level draws below the first value. Only the finer level
    /// draws above the second value.
    pub pixels_per_unit: [f32; 2],
}

impl SpaghettiAsset {
    pub fn new(segments: Vec<GaussianSegment>) -> Result<Self, AssetError> {
        Self::with_detail(segments, &[])
    }

    /// Creates a cloud from up to four consecutive levels, finest first. Each
    /// transition separates two adjacent levels. Pixel ranges must descend
    /// without overlap.
    pub fn with_detail(
        segments: Vec<GaussianSegment>,
        detail: &[DetailTransition],
    ) -> Result<Self, AssetError> {
        check_count(segments.len())?;
        if detail.len() > MAX_DETAIL_TRANSITIONS {
            return Err(AssetError::InvalidDetail);
        }
        let mut previous_end = 0;
        let mut previous_min = f32::INFINITY;
        for &DetailTransition {
            finer_end,
            pixels_per_unit: [min, max],
        } in detail
        {
            if finer_end <= previous_end
                || finer_end as usize >= segments.len()
                || !(0.0 < min && min < max && max <= previous_min && max.is_finite())
            {
                return Err(AssetError::InvalidDetail);
            }
            previous_end = finer_end;
            previous_min = min;
        }
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
            detail: Arc::from(detail),
            occlusion: Arc::from([]),
        })
    }

    pub fn segments(&self) -> &[GaussianSegment] {
        &self.segments
    }

    pub fn detail_transitions(&self) -> &[DetailTransition] {
        &self.detail
    }

    /// Bakes ambient and directional transmittance at the segment midpoints.
    ///
    /// The sun direction and upper hemisphere (+Y) are in asset space. Each
    /// detail level is baked independently. Rebake if the direction changes.
    /// The segment at the ray's origin does not occlude itself.
    pub fn bake_occlusion(self, direction_to_light: Vec3) -> Result<Self, AssetError> {
        self.bake_occlusion_with_density(direction_to_light, 1.0)
    }

    /// Bakes transmittance with optical masses scaled for lighting.
    ///
    /// Use a lower density when kernels with substantial overlap represent
    /// a single opaque surface. This scale does not affect raster opacity.
    pub fn bake_occlusion_with_density(
        mut self,
        direction_to_light: Vec3,
        density: f32,
    ) -> Result<Self, AssetError> {
        if !density.is_finite() || density < 0.0 {
            return Err(AssetError::InvalidOcclusionDensity);
        }
        let direction = direction_to_light
            .as_dvec3()
            .try_normalize()
            .ok_or(AssetError::InvalidOcclusionDirection)?;
        let mut factors = vec![[0.0; 2]; self.segments.len()];
        let mut start = 0;
        let ends = self.detail.iter().map(|level| level.finer_end as usize);
        for end in ends.chain([self.segments.len()]) {
            crate::occlusion::bake(
                &self.segments[start..end],
                direction,
                density.into(),
                &mut factors[start..end],
            );
            start = end;
        }
        self.occlusion = factors.into();
        Ok(self)
    }

    /// One pair of ambient and directional transmittance factors per segment.
    /// Empty for clouds without baked occlusion.
    pub fn occlusion(&self) -> &[[f32; 2]] {
        &self.occlusion
    }

    /// Local-space minimum and maximum, including four standard deviations
    /// of kernel support.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.bounds
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AssetError> {
        let has_occlusion = bytes.starts_with(MAGIC_V3);
        let has_detail = has_occlusion || bytes.starts_with(MAGIC_V2);
        if !(has_detail || bytes.starts_with(MAGIC_V1)) || bytes.len() < HEADER_BYTES {
            return Err(AssetError::Header);
        }
        let count = u32::from_le_bytes(bytes[8..HEADER_BYTES].try_into().unwrap()) as usize;
        check_count(count)?;
        let segment_end = HEADER_BYTES + count * SEGMENT_BYTES;
        let Some(trailer) = bytes.get(segment_end..) else {
            return Err(AssetError::Length);
        };
        let (records, occlusion_bytes) = if has_detail {
            let Some((detail_count, data)) = trailer.split_first_chunk::<4>() else {
                return Err(AssetError::Length);
            };
            let detail_count = u32::from_le_bytes(*detail_count) as usize;
            if detail_count > MAX_DETAIL_TRANSITIONS || (!has_occlusion && detail_count == 0) {
                return Err(AssetError::InvalidDetail);
            }
            let Some((detail, occlusion)) = data.split_at_checked(detail_count * TRANSITION_BYTES)
            else {
                return Err(AssetError::Length);
            };
            if occlusion.len() != if has_occlusion { count * 8 } else { 0 } {
                return Err(AssetError::Length);
            }
            (detail.as_chunks::<TRANSITION_BYTES>().0, occlusion)
        } else {
            if !trailer.is_empty() {
                return Err(AssetError::Length);
            }
            (&[][..], &[][..])
        };

        let segments = bytes[HEADER_BYTES..segment_end]
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
        let detail: Vec<_> = records
            .iter()
            .map(|record| {
                let [end, min, max] = bytemuck::cast::<_, [[u8; 4]; 3]>(*record);
                DetailTransition {
                    finer_end: u32::from_le_bytes(end),
                    pixels_per_unit: [f32::from_le_bytes(min), f32::from_le_bytes(max)],
                }
            })
            .collect();
        let mut asset = Self::with_detail(segments, &detail)?;
        let occlusion = occlusion_bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|record| {
                let factors = bytemuck::cast::<_, [[u8; 4]; 2]>(*record).map(f32::from_le_bytes);
                if factors.iter().all(|value| (0.0..=1.0).contains(value)) {
                    Ok(factors)
                } else {
                    Err(AssetError::InvalidOcclusion)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        asset.occlusion = occlusion.into();
        Ok(asset)
    }

    pub fn write(&self, mut writer: impl Write) -> Result<(), AssetError> {
        writer.write_all(if !self.occlusion.is_empty() {
            MAGIC_V3
        } else if self.detail.is_empty() {
            MAGIC_V1
        } else {
            MAGIC_V2
        })?;
        writer.write_all(&(self.segments.len() as u32).to_le_bytes())?;
        for segment in self.segments.iter() {
            let fields: [f32; SEGMENT_BYTES / 4] = bytemuck::cast(*segment);
            let bytes: [[u8; 4]; SEGMENT_BYTES / 4] = fields.map(f32::to_le_bytes);
            writer.write_all(bytemuck::cast_slice(&bytes))?;
        }
        if !self.detail.is_empty() || !self.occlusion.is_empty() {
            writer.write_all(&(self.detail.len() as u32).to_le_bytes())?;
            for transition in &*self.detail {
                writer.write_all(&transition.finer_end.to_le_bytes())?;
                for edge in transition.pixels_per_unit {
                    writer.write_all(&edge.to_le_bytes())?;
                }
            }
        }
        for factors in &*self.occlusion {
            writer.write_all(bytemuck::cast_slice(&factors.map(f32::to_le_bytes)))?;
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
        let tapered = segment.with_cutout_taper(0.25);
        let blade = segment.with_blade_profile(2.0, [0.25, 0.75], 0.5);
        let blade_with_taper = blade.with_cutout_taper(0.25);
        assert_eq!(segment.cutout_end_width(), 1.0);
        assert_eq!(tapered.color[3].to_bits(), 0x3f80_0100);
        assert_eq!(blade.color[3].to_bits(), 0x3f80_0000 | (2000 << 10) | 512);
        assert_eq!(
            blade_with_taper,
            segment.with_blade_profile(2.0, [0.25, 0.75], 0.25)
        );
        let segments = [segment, tapered, blade, blade_with_taper];
        let asset = SpaghettiAsset::new(segments.to_vec()).unwrap();
        let mut bytes = Vec::new();
        asset.write(&mut bytes).unwrap();
        assert!(bytes.starts_with(MAGIC_V1));
        assert_eq!(
            SpaghettiAsset::from_bytes(&bytes).unwrap().segments(),
            &segments
        );
        assert!(matches!(
            SpaghettiAsset::from_bytes(&bytes[..bytes.len() - 1]),
            Err(AssetError::Length)
        ));
    }

    #[test]
    fn buried_midpoint_has_less_light_and_factors_survive_round_trip() {
        let point = |center, mass| {
            GaussianSegment::new(
                center,
                center,
                Vec3::splat(0.1),
                Quat::IDENTITY,
                mass,
                Vec3::ONE,
            )
        };
        let mut segments = vec![point(Vec3::ZERO, 0.0), point(Vec3::splat(2.0), 0.0)];
        for axis in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::NEG_Y,
            Vec3::NEG_Z,
        ] {
            segments.push(point(axis * 0.1, 0.15));
        }
        // A second level can occupy exactly the same space.
        let finer_end = segments.len() as u32;
        segments.push(point(Vec3::ZERO, 1.0));
        let asset = SpaghettiAsset::with_detail(
            segments,
            &[DetailTransition {
                finer_end,
                pixels_per_unit: [40.0, 60.0],
            }],
        )
        .unwrap()
        .bake_occlusion(Vec3::Y)
        .unwrap();
        let factors = asset.occlusion();
        assert!(
            factors[0].iter().all(|value| *value < 0.3),
            "{:?}",
            factors[0]
        );
        assert_eq!(factors[1], [1.0; 2]);
        assert_eq!(factors[finer_end as usize], [1.0; 2]);
        let mut bytes = Vec::new();
        asset.write(&mut bytes).unwrap();
        assert!(bytes.starts_with(MAGIC_V3));
        let read = SpaghettiAsset::from_bytes(&bytes).unwrap();
        assert_eq!(read.occlusion(), factors);
        assert_eq!(read.detail_transitions(), asset.detail_transitions());
    }

    #[test]
    fn directional_point_integrates_the_forward_half_ray() {
        let point = |center, mass| {
            GaussianSegment::new(center, center, Vec3::ONE, Quat::IDENTITY, mass, Vec3::ONE)
        };
        let mass = std::f32::consts::TAU;
        let asset =
            SpaghettiAsset::new(vec![point(Vec3::ZERO, 0.0), point(Vec3::Y, mass)]).unwrap();
        let forward = asset.clone().bake_occlusion(Vec3::Y).unwrap().occlusion()[0][1];
        let backward = asset.bake_occlusion(Vec3::NEG_Y).unwrap().occlusion()[0][1];
        assert!((forward - (-0.8413447_f32).exp()).abs() < 1e-6);
        assert!((backward - (-0.1586553_f32).exp()).abs() < 1e-6);
    }

    #[test]
    fn detail_round_trip_and_validation() {
        let segment = GaussianSegment::new(
            Vec3::ZERO,
            Vec3::Y,
            Vec3::splat(0.1),
            Quat::IDENTITY,
            1.0,
            Vec3::ONE,
        );
        let transitions = [
            DetailTransition {
                finer_end: 1,
                pixels_per_unit: [40.0, 60.0],
            },
            DetailTransition {
                finer_end: 2,
                pixels_per_unit: [10.0, 20.0],
            },
        ];
        let asset = SpaghettiAsset::with_detail(vec![segment; 3], &transitions).unwrap();
        let mut bytes = Vec::new();
        asset.write(&mut bytes).unwrap();
        assert!(bytes.starts_with(MAGIC_V2));
        let round_trip = SpaghettiAsset::from_bytes(&bytes).unwrap();
        assert_eq!(round_trip.segments(), asset.segments());
        assert_eq!(round_trip.detail_transitions(), &transitions);
        assert!(matches!(
            SpaghettiAsset::from_bytes(&bytes[..bytes.len() - TRANSITION_BYTES]),
            Err(AssetError::Length)
        ));
        assert!(matches!(
            SpaghettiAsset::with_detail(vec![segment; 2], &transitions),
            Err(AssetError::InvalidDetail)
        ));
    }
}
