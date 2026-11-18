# Spaghetti splats for Bevy

Render Gaussian-convolved line segments in a Bevy 0.20 3D scene. The crate includes a Rust glTF converter and an orbit viewer with a bundled, textured marble bust. The renderer implements the Gaussian inference method in [Spaghetti Splatting, sections 3.2–3.5](https://arxiv.org/html/2611.12716v1#S3).

## Run the viewer

```sh
cargo run --example viewer
```

Drag to orbit, right-drag to pan, and scroll to zoom. Press **Space** for the turntable or **R** to reset. To view your own conversion, put it in `assets/` and pass its relative path.

```sh
cargo run --example viewer -- my_model.spag
```

The bundled `assets/marble_bust_01.spag` contains 240,000 Gaussian segments converted from the 2K glTF of Marble Bust 01 by Rico Cilliers. The conversion samples the model's color texture and bakes lighting using its normals and normal map. [Source model and CC0 license](https://polyhaven.com/a/marble_bust_01)

## Play Windward

Windward uses splats for wind-driven vegetation, conifer crowns, and scanned props in a third-person wetland scene.

```sh
cargo run --release --example windward
```

Use **WASD** to move, **Shift** to run, and **Space** to jump. Click the window to look with the mouse. Press **Esc** to release the cursor or **R** to reset.

The forest-floor material uses a CC0 scan by Rob Tuytel. [Forest Ground 03](https://polyhaven.com/a/forrest_ground_03)

## Add splats to a game

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

The entity supports transforms, inherited visibility, render layers, frustum culling and per-camera `VisibilityRange`. Each 3D camera gets a separate segment depth order. Splats test against opaque Bevy depth and blend in the transparent pass. To create an asset in Rust, build a `Vec<GaussianSegment>`, pass it to `SpaghettiAsset::new`, and add it to `Assets<SpaghettiAsset>`. Equal endpoints produce a point splat. When splitting a straight segment with the same color and kernel, divide its optical mass in proportion to the child lengths to preserve opacity at the join.

Add `SpaghettiCutout` for dense surface splats that should write depth. The cutout pipeline tests a projected Gaussian cross section and draws directly from the segment buffer. Use `GaussianSegment::with_cutout_taper` to set the end width as a fraction of the start width.

Use `SpaghettiWind` for procedural gusts and contact, or update `SpaghettiSway` with a precomputed world-space displacement per unit of bend height. Both deform endpoints on the GPU with no asset upload. Set endpoint bend heights with `GaussianSegment::with_bend_heights`. For tapered grass, use `with_blade_profile` to set the blade height, endpoint fractions and end width together.

Add `SpaghettiDetailScale` to let entities share one multilevel asset while using different detail thresholds. For example, `SpaghettiDetailScale(1.8)` switches to coarser levels at 1.8 times the asset's default projected size.

With `SpaghettiLighting`, add `SpaghettiGroundAmbient` to set a separate linear RGB multiplier for ground bounce. Upward-facing splats receive the sky color from `SpaghettiLighting::ambient`, and side-facing splats receive a blend of sky and ground light.

## Convert a model

The converter accepts static glTF 2.0 and GLB. Keep external buffers and textures next to the source glTF at their referenced paths.

```sh
cargo run --no-default-features --features convert --bin spaghetti-convert -- \
  path/to/model.glb assets/my_model.spag --samples 240000 --bake-light
```

- Triangles are sampled by world-space area into short segments with flattened Gaussian kernels. Increase `--samples` to retain more fine detail.
- glTF lines, line strips and loops preserve the original endpoints. Points become zero-length segments. Set their Gaussian radius with `--line-sigma` in scene units.
- Base color comes from the material factor, vertex color and base-color texture. `--bake-light` adds a fixed key and fill light to lit triangle materials. Use an exported static pose for skinned or morphed models.

To regenerate the sample, use Poly Haven's 2K glTF download and substitute `marble_bust_01_2k.gltf` and `assets/marble_bust_01.spag` in the command above.

## Rendering scope

The GPU projects each segment's anisotropic kernel with a midpoint EWA Jacobian, adds a half-pixel Gaussian filter in screen space, and evaluates the Gaussian line integral in closed form. Beer–Lambert opacity and back-to-front premultiplied blending accumulate the segments. Segment depth ordering uses midpoints, so crossing translucent segments may sort approximately. Bevy sorts separate clouds and other transparent objects by entity center.

The runtime uses the paper's Gaussian kernel and linear RGB colors. Optional `SpaghettiLighting` and `SpaghettiTransmission` shade the kernel's positive Z axis. This implementation covers rendering and direct geometry conversion. It omits Beta kernels, photo-based training, spherical harmonics and shadow casting. Long segments with strong perspective foreshortening benefit from subdivision. The renderer requires vertex-stage storage buffers. The format caps each asset at one million segments.

## File layout

Version 1 uses the eight bytes `SPAG\x01\x00\x00\x00`, followed by a little-endian `u32` segment count. Each segment stores five little-endian `f32x4` vectors, in order.

| Vector | Fields |
| --- | --- |
| Start | XYZ endpoint and optical mass in squared asset units |
| End | XYZ endpoint and start bend height in world units, or zero |
| Kernel | XYZ principal-axis standard deviations and end bend height in world units, or zero |
| Rotation | Unit quaternion XYZW |
| Color | Linear RGB and optional packed cutout taper or blade profile, or zero |

Version 2 uses `SPAG\x02\x00\x00\x00` and the same segment records, grouped from finest to coarsest. The segment records are followed by a little-endian `u32` transition count (1 to 3). Each transition stores the exclusive end index of the finer level as a `u32` and the lower and upper cross-fade thresholds in projected pixels per asset unit as two `f32` values. The writer uses version 1 for clouds with a single level.

## Checks

```sh
cargo test --all-features
cargo test --test headless -- --ignored --nocapture
```

The offscreen GPU test checks joint continuity, endpoint taper, point splats, transparent depth order, occlusion by an opaque mesh, and rendering the bundled bust. It needs a native GPU adapter and uses texture readback without creating a window or writing images.
