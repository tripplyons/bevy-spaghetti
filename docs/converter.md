# Convert a model

The converter accepts static glTF 2.0 and GLB. Keep external buffers and textures next to the source glTF at their referenced paths.

```sh
cargo run --no-default-features --features convert --bin spaghetti-convert -- \
  path/to/model.glb assets/my_model.spag --samples 240000 --bake-light
```

- Triangles are sampled by world-space area into short segments with flattened Gaussian kernels. Increase `--samples` to retain more fine detail.
- glTF lines, line strips and loops preserve the original endpoints. Points become zero-length segments. Set their Gaussian radius with `--line-sigma` in scene units.
- Base color comes from the material factor, vertex color and base-color texture. `--bake-light` adds a fixed key and fill light to lit triangle materials. Use an exported static pose for skinned or morphed models.

To view a conversion, put it in `assets/` and pass its relative path to the viewer.

```sh
cargo run --example viewer -- my_model.spag
```

## Regenerate the sample

The bundled `assets/marble_bust_01.spag` contains 240,000 Gaussian segments converted from the 2K glTF of Marble Bust 01 by Rico Cilliers. The conversion samples the model's color texture and bakes lighting using its normals and normal map. [Source model and CC0 license](https://polyhaven.com/a/marble_bust_01)

To regenerate it, use Poly Haven's 2K glTF download and substitute `marble_bust_01_2k.gltf` and `assets/marble_bust_01.spag` in the command above.
