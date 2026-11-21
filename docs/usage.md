# Add splats to a game

Add this crate as a path dependency to a Bevy 0.20 game.

```toml
bevy-spaghetti = { path = "../bevy-spaghetti" }
```

Copy a `.spag` file into your game's asset directory. Register `SpaghettiPlugin` after `DefaultPlugins` and spawn `Spaghetti3d` like any other 3D entity.

```rust
use bevy::prelude::*;
use bevy_spaghetti::{Spaghetti3d, SpaghettiPlugin};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, SpaghettiPlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    commands.spawn(Spaghetti3d(assets.load("marble_bust_01.spag")));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.25, 0.35, 0.8)
            .looking_at(Vec3::new(0.0, 0.25, 0.0), Vec3::Y),
    ));
}
```

The entity supports transforms, inherited visibility, render layers, frustum culling and per-camera `VisibilityRange`. Each 3D camera gets a separate segment depth order. Splats test against opaque Bevy depth and blend in the transparent pass.

## Build assets in code

To create an asset in Rust, build a `Vec<GaussianSegment>`, pass it to `SpaghettiAsset::new`, and add it to `Assets<SpaghettiAsset>`. Equal endpoints produce a point splat. When splitting a straight segment with the same color and kernel, divide its optical mass in proportion to the child lengths to preserve opacity at the join.

## Components

### Cutouts

Add `SpaghettiCutout` for dense surface splats that should write depth. The cutout pipeline tests a projected Gaussian cross section and draws directly from the segment buffer. Use `GaussianSegment::with_cutout_taper` to set the end width as a fraction of the start width.

### Wind and sway

Use `SpaghettiWind` for procedural gusts and contact, or update `SpaghettiSway` with a precomputed world-space displacement per unit of bend height. Both deform endpoints on the GPU with no asset upload. Set endpoint bend heights with `GaussianSegment::with_bend_heights`. For tapered grass, use `with_blade_profile` to set the blade height, endpoint fractions and end width together.

### Detail levels

Add `SpaghettiDetailScale` to let entities share one multilevel asset while using different detail thresholds. For example, `SpaghettiDetailScale(1.8)` switches to coarser levels at 1.8 times the asset's default projected size.

### Lighting

With `SpaghettiLighting`, add `SpaghettiGroundAmbient` to set a separate linear RGB multiplier for ground bounce. Upward-facing splats receive the sky color from `SpaghettiLighting::ambient`, and side-facing splats receive a blend of sky and ground light.

Add `SpaghettiSpecular` to a lit cloud for a view-dependent highlight. Its `strand` field blends between a highlight off each segment's surface normal, for wet stone, and one off the segment's axis, for hair, needles and blades.

Add `SpaghettiPointLight` to an entity with a transform for a local light source, such as a fire or a lamp. Each lit cloud takes the source that lights its bounds most strongly. The light follows the inverse square law, reaches zero at its range and casts no shadows.

### Additive light and flow

Add `SpaghettiAdditive` to a translucent cloud to add its light to the frame, for flames, sparks and sunlit dust. Update `SpaghettiFlow` to move each segment along a looping path with its own start time, a smooth wander, growth, a color shift and a fade at both ends. Flow animates smoke, mist, fire and drifting particles on the GPU with no asset upload.
