# The .spag file format

## Version 1

Version 1 uses the eight bytes `SPAG\x01\x00\x00\x00`, followed by a little-endian `u32` segment count. Each segment stores five little-endian `f32x4` vectors, in order.

| Vector | Fields |
| --- | --- |
| Start | XYZ endpoint and optical mass in squared asset units |
| End | XYZ endpoint and start bend height in world units, or zero |
| Kernel | XYZ principal-axis standard deviations and end bend height in world units, or zero |
| Rotation | Unit quaternion XYZW |
| Color | Linear RGB and optional packed cutout taper or blade profile, or zero |

## Version 2

Version 2 uses `SPAG\x02\x00\x00\x00` and the same segment records, grouped from finest to coarsest. The segment records are followed by a little-endian `u32` transition count (1 to 3). Each transition stores the exclusive end index of the finer level as a `u32` and the lower and upper cross-fade thresholds in projected pixels per asset unit as two `f32` values. The writer uses version 1 for clouds with a single level.
