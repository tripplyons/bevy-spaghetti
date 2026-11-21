# Rendering scope

The renderer implements the Gaussian inference method in [Spaghetti Splatting, sections 3.2–3.5](https://arxiv.org/html/2611.12716v1#S3).

The GPU projects each segment's anisotropic kernel with a midpoint EWA Jacobian, adds a half-pixel Gaussian filter in screen space, and evaluates the Gaussian line integral in closed form. Beer–Lambert opacity and back-to-front premultiplied blending accumulate the segments. Segment depth ordering uses midpoints, so crossing translucent segments may sort approximately. Bevy sorts separate clouds and other transparent objects by entity center.

The runtime uses the paper's Gaussian kernel and linear RGB colors. Optional `SpaghettiLighting` and `SpaghettiTransmission` shade the kernel's positive Z axis. This implementation covers rendering and direct geometry conversion. It omits Beta kernels, photo-based training, spherical harmonics and shadow casting. Long segments with strong perspective foreshortening benefit from subdivision. The renderer requires vertex-stage storage buffers. The format caps each asset at one million segments.

## Checks

```sh
cargo test --all-features
cargo test --test headless -- --ignored --nocapture
```

The offscreen GPU test checks joint continuity, endpoint taper, point splats, transparent depth order, occlusion by an opaque mesh, and rendering the bundled bust. It needs a native GPU adapter and uses texture readback without creating a window or writing images.
