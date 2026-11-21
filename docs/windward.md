# Windward

Windward is a third-person wetland scene. It uses splats for wind-driven vegetation, conifer crowns, and scanned props. The source is in [`examples/windward`](../examples/windward).

## Play

```sh
cargo run --release --example windward
```

Use **WASD** to move, **Shift** to run, and **Space** to jump. Click the window to look with the mouse. Press **Esc** to release the cursor or **R** to reset.

## Record a clip

The recorder renders a scripted walk to a 60 fps video. The hiker walks past the camp, stops at the lake, runs and jumps down the trail, and climbs a hillside above the valley. `ffmpeg` must be on the `PATH`.

```sh
cargo run --release --example windward -- --record clip.mp4
```

The recorder runs without a window and advances the simulation by exactly one frame per update, so the clip plays back smoothly however long each frame takes to render. A 1080p clip takes about 6 minutes on an M3 Max.

| Option | Default | Effect |
| --- | --- | --- |
| `--size WIDTHxHEIGHT` | `1920x1080` | Output size. Both dimensions must be even. |
| `--shutter N` | `2` | Averages N subframes into each video frame for motion blur. `1` turns blur off. |
| `--crf N` | `15` | x264 quality. Higher numbers make smaller files. |
| `--map PATH.png` | none | Also writes a top-down map of the route, the jumps, and the crowns that block the camera. |

For a quick preview, use `--size 640x360 --shutter 1`.

To change the route, edit `SCRIPT` in [`record.rs`](../examples/windward/record.rs). Each beat lists ground points to walk or run through, the distances at which to jump, and a camera orbit and pitch. Hold beats stand still while the camera moves. The recorder stops with an error if the route enters the lake or the hiker gets stuck.

## Credits

The forest-floor material uses a CC0 scan by Rob Tuytel. [Forest Ground 03](https://polyhaven.com/a/forrest_ground_03)
