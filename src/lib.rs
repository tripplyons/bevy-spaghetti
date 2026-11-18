//! Gaussian-convolved segment assets and a Bevy 3D renderer.
//!
#![cfg_attr(
    feature = "render",
    doc = "Add [`SpaghettiPlugin`] after Bevy's default plugins and spawn a +           [`Spaghetti3d`] with a handle to a `.spag` asset."
)]

mod asset;
pub use asset::{AssetError, GaussianSegment, SpaghettiAsset};

#[cfg(feature = "render")]
mod render;
#[cfg(feature = "render")]
pub use render::{
    Spaghetti3d, SpaghettiCutout, SpaghettiLighting, SpaghettiPlugin, SpaghettiTransmission,
    SpaghettiWind,
};
