use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufWriter,
    path::PathBuf,
};

use anyhow::{Context, Result, ensure};
use bevy_math::{IVec3, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
use bevy_spaghetti::{GaussianSegment, SpaghettiAsset};
use clap::Parser;
use gltf::{image, mesh::Mode, texture::WrappingMode};

#[derive(Parser)]
#[command(about = "Convert a static glTF or GLB scene to Gaussian segment splats")]
struct Args {
    input: PathBuf,
    output: PathBuf,

    /// Number of surface samples, distributed by world-space triangle area.
    #[arg(long, default_value = "240000", value_parser = clap::value_parser!(u32).range(1..=1_000_000))]
    samples: u32,

    /// Gaussian standard deviation for line and point primitives, in scene units.
    #[arg(long, default_value_t = 0.005)]
    line_sigma: f32,

    /// Bake a broad key and fill light into lit surface materials.
    #[arg(long)]
    bake_light: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.line_sigma.is_finite() && args.line_sigma > 0.0,
        "--line-sigma must be finite and positive"
    );
    ensure!(
        args.output.extension().is_some_and(|ext| ext == "spag"),
        "output must use the .spag extension"
    );

    let (document, buffers, images) =
        gltf::import(&args.input).with_context(|| format!("importing {}", args.input.display()))?;
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .context("glTF has no scene")?;

    let mut conversion = Conversion {
        args: &args,
        buffers: &buffers,
        images: &images,
        surfaces: Vec::new(),
        segments: Vec::new(),
    };
    for node in scene.nodes() {
        conversion.node(node, Mat4::IDENTITY)?;
    }
    conversion.sample_surfaces();
    let asset = SpaghettiAsset::new(conversion.segments)?;
    let mut writer = BufWriter::new(
        File::create(&args.output)
            .with_context(|| format!("creating {}", args.output.display()))?,
    );
    asset.write(&mut writer)?;
    std::io::Write::flush(&mut writer)?;
    println!(
        "Wrote {} splats to {}",
        asset.segments().len(),
        args.output.display()
    );
    Ok(())
}

struct Conversion<'a> {
    args: &'a Args,
    buffers: &'a [gltf::buffer::Data],
    images: &'a [image::Data],
    surfaces: Vec<Surface<'a>>,
    segments: Vec<GaussianSegment>,
}

#[derive(Clone, Copy)]
struct Vertex {
    position: Vec3,
    normal: Vec3,
    color: Vec4,
    uv: Vec2,
    normal_uv: Vec2,
    tangent: Vec4,
}

struct Surface<'a> {
    vertices: [Vertex; 3],
    material: Material<'a>,
    area: f64,
    frame: Mat3,
    uv_frame: UvFrame,
    normal_tangents: Option<(Vec3, Vec3)>,
}

impl<'a> Conversion<'a> {
    fn node(&mut self, node: gltf::Node<'_>, parent: Mat4) -> Result<()> {
        let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        ensure!(
            transform.is_finite(),
            "node {} has a non-finite transform",
            node.index()
        );
        if let Some(mesh) = node.mesh() {
            ensure!(
                node.skin().is_none(),
                "node {} has a skin. Export a baked mesh first",
                node.index()
            );
            for primitive in mesh.primitives() {
                ensure!(
                    primitive.morph_targets().len() == 0,
                    "mesh {} has morph targets. Export a baked mesh first",
                    mesh.index()
                );
                self.primitive(primitive, transform)
                    .with_context(|| format!("converting mesh {}", mesh.index()))?;
            }
        }
        for child in node.children() {
            self.node(child, transform)?;
        }
        Ok(())
    }

    fn primitive(&mut self, primitive: gltf::Primitive<'_>, transform: Mat4) -> Result<()> {
        let reader = primitive.reader(|buffer| Some(&self.buffers[buffer.index()].0));
        let positions = reader
            .read_positions()
            .context("missing POSITION accessor")?;
        let material = Material::new(primitive.material(), self.images, self.args.bake_light);
        let mut normals = reader.read_normals();
        let mut tangents = reader.read_tangents();
        let mut colors = reader.read_colors(0).map(|colors| colors.into_rgba_f32());
        let mut uvs = material
            .base
            .as_ref()
            .map(|texture| {
                reader
                    .read_tex_coords(texture.uv_set)
                    .context("missing base color TEXCOORD")
            })
            .transpose()?
            .map(|uvs| uvs.into_f32());
        let mut normal_uvs = material
            .normal
            .as_ref()
            .map(|texture| {
                reader
                    .read_tex_coords(texture.uv_set)
                    .context("missing normal map TEXCOORD")
            })
            .transpose()?
            .map(|uvs| uvs.into_f32());
        let linear = Mat3::from_mat4(transform);
        let mirrored = linear.determinant() < 0.0;
        let normal_matrix = linear.inverse().transpose();
        let vertices: Vec<_> = positions
            .map(|position| Vertex {
                position: transform.transform_point3(Vec3::from_array(position)),
                normal: normals
                    .as_mut()
                    .and_then(Iterator::next)
                    .map_or(Vec3::ZERO, |n| {
                        (normal_matrix * Vec3::from_array(n)).normalize_or_zero()
                    }),
                color: colors
                    .as_mut()
                    .and_then(Iterator::next)
                    .map_or(Vec4::ONE, Vec4::from_array),
                uv: uvs
                    .as_mut()
                    .and_then(Iterator::next)
                    .map_or(Vec2::ZERO, Vec2::from_array),
                normal_uv: normal_uvs
                    .as_mut()
                    .and_then(Iterator::next)
                    .map_or(Vec2::ZERO, Vec2::from_array),
                tangent: tangents
                    .as_mut()
                    .and_then(Iterator::next)
                    .map_or(Vec4::ZERO, |tangent| {
                        let tangent = Vec4::from_array(tangent);
                        (linear * tangent.truncate())
                            .normalize_or_zero()
                            .extend(if mirrored { -tangent.w } else { tangent.w })
                    }),
            })
            .collect();
        ensure!(
            vertices.iter().all(|v| v.position.is_finite()),
            "non-finite vertex position"
        );
        let indices: Vec<usize> = reader.read_indices().map_or_else(
            || (0..vertices.len()).collect(),
            |indices| indices.into_u32().map(|i| i as usize).collect(),
        );
        ensure!(
            indices.iter().all(|&i| i < vertices.len()),
            "vertex index out of range"
        );
        let vertex = |i: usize| vertices[indices[i]];
        let count = indices.len();
        let first_surface = self.surfaces.len();
        let mut triangle = |first: usize, mut second: usize, mut third: usize| {
            if mirrored {
                std::mem::swap(&mut second, &mut third);
            }
            if let Some(surface) =
                Surface::new([vertex(first), vertex(second), vertex(third)], material)
            {
                self.surfaces.push(surface);
            }
        };
        match primitive.mode() {
            Mode::Triangles => {
                ensure!(count.is_multiple_of(3), "incomplete triangle");
                for i in (0..count).step_by(3) {
                    triangle(i, i + 1, i + 2);
                }
            }
            Mode::TriangleStrip => {
                for i in 0..count.saturating_sub(2) {
                    triangle(i + i % 2, i + 1 - i % 2, i + 2);
                }
            }
            Mode::TriangleFan => {
                for i in 1..count.saturating_sub(1) {
                    triangle(0, i, i + 1);
                }
            }
            Mode::Points => {
                for i in 0..count {
                    self.line(vertex(i), vertex(i), material);
                }
            }
            Mode::Lines => {
                ensure!(count.is_multiple_of(2), "incomplete line");
                for i in (0..count).step_by(2) {
                    self.line(vertex(i), vertex(i + 1), material);
                }
            }
            Mode::LineLoop | Mode::LineStrip => {
                for i in 1..count {
                    self.line(vertex(i - 1), vertex(i), material);
                }
                if count > 1 && primitive.mode() == Mode::LineLoop {
                    self.line(vertex(count - 1), vertex(0), material);
                }
            }
        }
        if material.base.is_some() || material.normal.is_some() {
            mark_uv_boundaries(&mut self.surfaces[first_surface..]);
        }
        Ok(())
    }

    fn line(&mut self, a: Vertex, b: Vertex, material: Material<'_>) {
        let (color, coverage) = material.color(a.color.lerp(b.color, 0.5), a.uv.lerp(b.uv, 0.5));
        if coverage == 0.0 {
            return;
        }
        let sigma = self.args.line_sigma;
        let length = a.position.distance(b.position);
        // Optical mass is proportional to arc length so splitting a strand
        // does not change its opacity. Points use a finite optical area.
        let mass = if length == 0.0 {
            std::f32::consts::TAU * sigma * sigma
        } else {
            std::f32::consts::TAU.sqrt() * length * sigma
        };
        self.segments.push(GaussianSegment::new(
            a.position,
            b.position,
            Vec3::splat(sigma),
            Quat::IDENTITY,
            3.5 * coverage * mass,
            color,
        ));
    }

    fn sample_surfaces(&mut self) {
        if self.surfaces.is_empty() {
            return;
        }

        let count = self.args.samples as usize;
        let mut area = 0.0;
        let end_areas: Vec<_> = self
            .surfaces
            .iter()
            .map(|surface| {
                area += surface.area;
                area
            })
            .collect();
        let spacing = (area / count as f64).sqrt() as f32;
        let first_separation = 0.76;
        let cell_size = first_separation * f64::from(spacing);
        let adjacent_cells: Vec<_> = (-1..=1)
            .flat_map(|z| (-1..=1).flat_map(move |y| (-1..=1).map(move |x| IVec3::new(x, y, z))))
            .collect();
        let mut grid = HashMap::<IVec3, Vec<(Vec3, Vec3)>>::new();
        let mut accepted = 0;
        let mut i = 0;

        // Reject clusters so narrower footprints still cover the surface.
        // Relax the minimum separation if the first pass cannot fill the budget.
        for (separation, attempts) in [(first_separation, 6), (0.60, 3), (0.0, 1)] {
            let min_distance_squared = (separation as f32 * spacing).powi(2);
            for _ in 0..attempts * count {
                if accepted == count {
                    return;
                }
                let offset = sequence(i, 2) * area;
                let surface = &self.surfaces[end_areas.partition_point(|&end| end <= offset)];
                let u = sequence(i, 3).sqrt() as f32;
                let v = sequence(i, 5) as f32;
                let weights = Vec3::new(1.0 - u, u * (1.0 - v), u * v);
                let angle = std::f32::consts::TAU * sequence(i, 7) as f32;
                i += 1;
                let [a, b, c] = surface.vertices;
                let center =
                    a.position * weights.x + b.position * weights.y + c.position * weights.z;
                let normal = surface.frame.z_axis;
                let cell = (center.as_dvec3() / cell_size)
                    .floor()
                    .as_ivec3()
                    .clamp(IVec3::splat(i32::MIN + 1), IVec3::splat(i32::MAX - 1));
                let too_close = separation > 0.0
                    && adjacent_cells.iter().any(|&offset| {
                        grid.get(&(cell + offset)).is_some_and(|neighbors| {
                            // Opposite sides of a thin sheet need their own samples.
                            neighbors.iter().any(|&(point, facing)| {
                                normal.dot(facing) > 0.0
                                    && center.distance_squared(point) < min_distance_squared
                            })
                        })
                    });
                if too_close {
                    continue;
                }
                grid.entry(cell).or_default().push((center, normal));
                accepted += 1;
                if let Some(segment) = surface.sample(center, weights, angle, spacing) {
                    self.segments.push(segment);
                }
            }
        }
    }
}

fn sequence(mut index: usize, base: usize) -> f64 {
    index += 1;
    let mut value = 0.0;
    let mut place = 1.0 / base as f64;
    while index > 0 {
        value += (index % base) as f64 * place;
        index /= base;
        place /= base as f64;
    }
    value
}

impl<'a> Surface<'a> {
    fn new(vertices: [Vertex; 3], material: Material<'a>) -> Option<Self> {
        let [a, b, c] = vertices;
        let first = b.position - a.position;
        let second = c.position - a.position;
        let cross = first.cross(second);
        let twice_area = cross.length();
        if twice_area == 0.0 || !twice_area.is_finite() {
            return None;
        }
        let normal = cross / twice_area;
        let tangent = first.normalize();
        let frame = Mat3::from_cols(tangent, normal.cross(tangent), normal);

        let delta1 = b.normal_uv - a.normal_uv;
        let delta2 = c.normal_uv - a.normal_uv;
        let determinant = delta1.perp_dot(delta2);
        let normal_tangents = (determinant.abs() > 1e-12).then(|| {
            (
                (first * delta2.y - second * delta1.y) / determinant,
                (second * delta1.x - first * delta2.x) / determinant,
            )
        });
        Some(Self {
            vertices,
            material,
            area: f64::from(twice_area) * 0.5,
            frame,
            uv_frame: UvFrame::new(first.length(), second.dot(tangent), twice_area),
            normal_tangents,
        })
    }

    fn sample(
        &self,
        center: Vec3,
        weights: Vec3,
        angle: f32,
        spacing: f32,
    ) -> Option<GaussianSegment> {
        let [a, b, c] = self.vertices;
        let vertex_color = a.color * weights.x + b.color * weights.y + c.color * weights.z;
        let base = TexturePoint::new(
            self.material.base,
            [a.uv, b.uv, c.uv],
            weights,
            self.uv_frame,
        );
        let normal_map = TexturePoint::new(
            self.material.normal,
            [a.normal_uv, b.normal_uv, c.normal_uv],
            weights,
            self.uv_frame,
        );
        let (along, coherence) = texture_direction(base, normal_map, spacing, angle);
        let across = along.perp();
        let half_length = (0.39 + 0.63 * coherence) * spacing;
        let sigma = Vec3::new(0.34, 0.37 - 0.14 * coherence, 0.085) * spacing;

        // The uniformly weighted centerline adds length²/12 to the variance.
        let footprint = along * (sigma.x * sigma.x + half_length * half_length / 3.0).sqrt();
        let width = across * (0.7 * sigma.y);
        let rgba = base.map_or(Vec4::ONE, |base| {
            if self.material.alpha == gltf::material::AlphaMode::Opaque {
                // A second blur here compounds the renderer's footprint.
                // Recover some contrast from the center-to-neighbor difference.
                base.footprint_sample(footprint, width, true, -0.35)
                    .max(Vec4::ZERO)
            } else {
                base.sample(Vec2::ZERO, true)
            }
        });
        let (mut color, coverage) = self.material.color_from_texel(vertex_color, rgba);
        if coverage == 0.0 {
            return None;
        }

        let smooth_normal = (a.normal * weights.x + b.normal * weights.y + c.normal * weights.z)
            .try_normalize()
            .unwrap_or(self.frame.z_axis);
        let mut normal = smooth_normal;
        if let Some(map) = normal_map {
            let tangent = a.tangent * weights.x + b.tangent * weights.y + c.tangent * weights.z;
            let tangents = if tangent.w != 0.0 {
                Some((
                    tangent.truncate(),
                    smooth_normal.cross(tangent.truncate()) * tangent.w,
                ))
            } else {
                self.normal_tangents
            };
            if let Some((tangent, bitangent)) = tangents {
                normal = self.mapped_normal(
                    smooth_normal,
                    tangent,
                    bitangent,
                    map.footprint_sample(footprint, width, false, 0.25)
                        .truncate()
                        * 2.0
                        - Vec3::ONE,
                );
            }
        }
        if self.material.bake_light {
            let key = normal.dot(Vec3::new(-0.55, 0.78, 0.3)).max(0.0);
            let fill = normal.dot(Vec3::new(0.71, 0.18, -0.68)).max(0.0);
            color *= 0.27 + 0.67 * key + 0.26 * fill;
        }

        let tangent = self.frame.x_axis * along.x + self.frame.y_axis * along.y;
        let offset = tangent * half_length;
        let geometric_frame =
            Mat3::from_cols(tangent, self.frame.z_axis.cross(tangent), self.frame.z_axis);
        // The renderer reads local +Z for lighting. Tilt the covariance frame
        // to carry the shading normal while keeping the centerline on the mesh.
        let rotation =
            Quat::from_rotation_arc(self.frame.z_axis, normal) * Quat::from_mat3(&geometric_frame);
        Some(GaussianSegment::new(
            center - offset,
            center + offset,
            sigma,
            rotation.normalize(),
            4.5 * coverage * spacing * spacing,
            color,
        ))
    }

    fn mapped_normal(&self, normal: Vec3, tangent: Vec3, bitangent: Vec3, sampled: Vec3) -> Vec3 {
        let tangent = (tangent - normal * normal.dot(tangent)).normalize_or_zero();
        let cross = normal.cross(tangent);
        let handedness = cross.dot(bitangent).signum();
        let scale = self.material.normal_scale;
        (tangent * sampled.x * scale + cross * handedness * sampled.y * scale + normal * sampled.z)
            .try_normalize()
            .unwrap_or(normal)
    }
}

fn mark_uv_boundaries(surfaces: &mut [Surface<'_>]) {
    // Match by position and both UV sets so duplicated vertices can still
    // share an edge. An atlas seam has no matching edge with the same UVs.
    let keys = |surface: &Surface<'_>| {
        surface.vertices.map(|v| {
            let [x, y, z] = v.position.to_array();
            [x, y, z, v.uv.x, v.uv.y, v.normal_uv.x, v.normal_uv.y]
                .map(|value| if value == 0.0 { 0 } else { value.to_bits() })
        })
    };
    let edges: HashSet<_> = surfaces
        .iter()
        .flat_map(|surface| {
            let [a, b, c] = keys(surface);
            [(a, b), (b, c), (c, a)]
        })
        .collect();
    for surface in surfaces {
        let [a, b, c] = keys(surface);
        surface.uv_frame.boundaries = [(c, b), (a, c), (b, a)].map(|edge| !edges.contains(&edge));
    }
}

#[derive(Clone, Copy)]
struct UvFrame {
    dx: Vec3,
    dy: Vec3,
    boundaries: [bool; 3],
}

impl UvFrame {
    fn new(ab_length: f32, ac_parallel: f32, twice_area: f32) -> Self {
        // Invert the triangle's two planar edges to get barycentric offsets
        // per unit displacement in the tangent and bitangent directions.
        let dx_b = ab_length.recip();
        let dy_b = -ac_parallel / twice_area;
        let dy_c = ab_length / twice_area;
        Self {
            dx: Vec3::new(-dx_b, dx_b, 0.0),
            dy: Vec3::new(-dy_b - dy_c, dy_b, dy_c),
            boundaries: [false; 3],
        }
    }

    fn clip(self, weights: Vec3, offset: Vec2) -> Vec2 {
        let delta = self.dx * offset.x + self.dy * offset.y;
        let mut fraction = 1.0_f32;
        for i in 0..3 {
            if self.boundaries[i] && delta[i] < -weights[i] {
                fraction = fraction.min(-weights[i] / delta[i]);
            }
        }
        offset * fraction
    }
}

#[derive(Clone, Copy)]
struct TexturePoint<'a> {
    texture: Texture<'a>,
    uv: Vec2,
    dx: Vec2,
    dy: Vec2,
    frame: UvFrame,
    weights: Vec3,
}

impl<'a> TexturePoint<'a> {
    fn new(
        texture: Option<Texture<'a>>,
        [a, b, c]: [Vec2; 3],
        weights: Vec3,
        frame: UvFrame,
    ) -> Option<Self> {
        let texture = texture?;
        let project = |weights: Vec3| a * weights.x + b * weights.y + c * weights.z;
        Some(Self {
            texture,
            uv: project(weights),
            dx: (b - a) * frame.dx.y,
            dy: (b - a) * frame.dy.y + (c - a) * frame.dy.z,
            frame,
            weights,
        })
    }

    fn sample(self, offset: Vec2, srgb: bool) -> Vec4 {
        let offset = self.frame.clip(self.weights, offset);
        self.texture
            .sample(self.uv + self.dx * offset.x + self.dy * offset.y, srgb)
    }

    fn footprint_sample(self, along: Vec2, across: Vec2, srgb: bool, neighbor_weight: f32) -> Vec4 {
        let center = self.sample(Vec2::ZERO, srgb);
        let neighbors = (self.sample(along, srgb)
            + self.sample(-along, srgb)
            + self.sample(across, srgb)
            + self.sample(-across, srgb))
            * 0.25;
        center.lerp(neighbors, neighbor_weight)
    }
}

fn texture_direction(
    base: Option<TexturePoint<'_>>,
    normal: Option<TexturePoint<'_>>,
    spacing: f32,
    angle: f32,
) -> (Vec2, f32) {
    let probe = 0.29 * spacing;
    let offsets = [
        Vec2::new(-probe, -probe),
        Vec2::new(probe, -probe),
        Vec2::new(-probe, probe),
        Vec2::new(probe, probe),
    ];
    let mut tensor = Vec3::ZERO;
    if let Some(base) = base {
        tensor += gradient_tensor(
            offsets.map(|offset| base.sample(offset, true).truncate().max(Vec3::ZERO).sqrt()),
        );
    }
    if let Some(normal) = normal {
        tensor +=
            0.08 * gradient_tensor(offsets.map(|offset| normal.sample(offset, false).truncate()));
    }
    // The major eigenvector is the direction of greatest texture change.
    // Extend the segment along the perpendicular, where texture is steadier.
    let difference = tensor.x - tensor.z;
    let twice_cross = 2.0 * tensor.y;
    let strength = difference.hypot(twice_cross);
    let coherence = strength / (tensor.x + tensor.z + 0.001);
    let direction = if coherence > 0.1 {
        Vec2::from_angle(0.5 * twice_cross.atan2(difference)).perp()
    } else {
        Vec2::from_angle(angle)
    };
    (direction, coherence)
}

fn gradient_tensor([a, b, c, d]: [Vec3; 4]) -> Vec3 {
    let [bottom, top] = [b - a, d - c];
    let [left, right] = [c - a, d - b];
    Vec3::new(
        bottom.length_squared() + top.length_squared(),
        (bottom + top).dot(left + right) * 0.5,
        left.length_squared() + right.length_squared(),
    )
}

#[derive(Clone, Copy)]
struct Material<'a> {
    factor: Vec4,
    base: Option<Texture<'a>>,
    normal: Option<Texture<'a>>,
    normal_scale: f32,
    alpha: gltf::material::AlphaMode,
    cutoff: f32,
    bake_light: bool,
}

impl<'a> Material<'a> {
    fn new(material: gltf::Material<'_>, images: &'a [image::Data], bake_light: bool) -> Self {
        let pbr = material.pbr_metallic_roughness();
        let bake_light = bake_light && !material.unlit();
        let normal = (!material.unlit())
            .then(|| material.normal_texture())
            .flatten();
        Self {
            factor: Vec4::from_array(pbr.base_color_factor()),
            base: pbr
                .base_color_texture()
                .map(|info| Texture::new(info.texture(), info.tex_coord(), images)),
            normal: normal
                .as_ref()
                .map(|info| Texture::new(info.texture(), info.tex_coord(), images)),
            normal_scale: normal.map_or(1.0, |info| info.scale()),
            alpha: material.alpha_mode(),
            cutoff: material.alpha_cutoff().unwrap_or(0.5),
            bake_light,
        }
    }

    fn color(&self, vertex_color: Vec4, uv: Vec2) -> (Vec3, f32) {
        self.color_from_texel(
            vertex_color,
            self.base.map_or(Vec4::ONE, |map| map.sample(uv, true)),
        )
    }

    fn color_from_texel(&self, vertex_color: Vec4, texel: Vec4) -> (Vec3, f32) {
        let rgba = self.factor * vertex_color * texel;
        let coverage = match self.alpha {
            gltf::material::AlphaMode::Opaque => 1.0,
            gltf::material::AlphaMode::Mask => f32::from(rgba.w >= self.cutoff),
            gltf::material::AlphaMode::Blend => rgba.w.clamp(0.0, 1.0),
        };
        (rgba.truncate().max(Vec3::ZERO), coverage)
    }
}

#[derive(Clone, Copy)]
struct Texture<'a> {
    data: &'a image::Data,
    uv_set: u32,
    wrap: [WrappingMode; 2],
}

impl<'a> Texture<'a> {
    fn new(texture: gltf::Texture<'_>, uv_set: u32, images: &'a [image::Data]) -> Self {
        let sampler = texture.sampler();
        Self {
            data: &images[texture.source().index()],
            uv_set,
            wrap: [sampler.wrap_s(), sampler.wrap_t()],
        }
    }

    fn sample(&self, uv: Vec2, srgb: bool) -> Vec4 {
        let [width, height] = [self.data.width as i64, self.data.height as i64];
        let position = uv * Vec2::new(width as f32, height as f32) - Vec2::splat(0.5);
        let x = position.x.floor() as i64;
        let y = position.y.floor() as i64;
        let frac = position - position.floor();
        let pixel = |x, y| {
            let x = wrap_texel(x, width, self.wrap[0]);
            let y = wrap_texel(y, height, self.wrap[1]);
            let rgba = pixel_at(self.data, (y * width + x) as usize);
            if srgb {
                let [r, g, b, a] = rgba.to_array();
                Vec4::new(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a)
            } else {
                rgba
            }
        };
        pixel(x, y)
            .lerp(pixel(x + 1, y), frac.x)
            .lerp(pixel(x, y + 1).lerp(pixel(x + 1, y + 1), frac.x), frac.y)
    }
}

fn wrap_texel(index: i64, size: i64, mode: WrappingMode) -> i64 {
    match mode {
        WrappingMode::ClampToEdge => index.clamp(0, size - 1),
        WrappingMode::Repeat => index.rem_euclid(size),
        WrappingMode::MirroredRepeat => {
            let index = index.rem_euclid(2 * size);
            index.min(2 * size - 1 - index)
        }
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn pixel_at(data: &image::Data, index: usize) -> Vec4 {
    use image::Format::*;
    let (channels, bytes) = match data.format {
        R8 => (1, 1),
        R8G8 => (2, 1),
        R8G8B8 => (3, 1),
        R8G8B8A8 => (4, 1),
        R16 => (1, 2),
        R16G16 => (2, 2),
        R16G16B16 => (3, 2),
        R16G16B16A16 => (4, 2),
        R32G32B32FLOAT => (3, 4),
        R32G32B32A32FLOAT => (4, 4),
    };
    let channel = |offset| {
        let start = (index * channels + offset) * bytes;
        match bytes {
            1 => f32::from(data.pixels[start]) / 255.0,
            2 => {
                f32::from(u16::from_ne_bytes(
                    data.pixels[start..start + 2].try_into().unwrap(),
                )) / 65535.0
            }
            _ => f32::from_ne_bytes(data.pixels[start..start + 4].try_into().unwrap()),
        }
    };
    match channels {
        1 => Vec3::splat(channel(0)).extend(1.0),
        2 => Vec3::splat(channel(0)).extend(channel(1)),
        3 => Vec4::new(channel(0), channel(1), channel(2), 1.0),
        4 => Vec4::new(channel(0), channel(1), channel(2), channel(3)),
        _ => unreachable!(),
    }
}
