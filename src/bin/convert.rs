use std::{fs::File, io::BufWriter, path::PathBuf};

use anyhow::{Context, Result, ensure};
use bevy_math::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
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
}

struct Surface<'a> {
    vertices: [Vertex; 3],
    material: Material<'a>,
    area: f64,
    frame: Mat3,
    tangent_frame: Option<(Vec3, Vec3)>,
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
        let area: f64 = self.surfaces.iter().map(|triangle| triangle.area).sum();
        let area_per_sample = area / count as f64;
        let spacing = area_per_sample.sqrt() as f32;
        let mut surface_index = 0;
        let mut end_area = self.surfaces[0].area;

        // Jitter one sample per equal-area stratum. Independent dimensions
        // select the triangle, barycentric position, and in-plane direction.
        for i in 0..count {
            let offset = (i as f64 + sequence(i, 2)) * area_per_sample;
            while end_area < offset && surface_index + 1 < self.surfaces.len() {
                surface_index += 1;
                end_area += self.surfaces[surface_index].area;
            }
            let u = sequence(i, 3).sqrt() as f32;
            let v = sequence(i, 5) as f32;
            let weights = [1.0 - u, u * (1.0 - v), u * v];
            let angle = std::f32::consts::TAU * sequence(i, 7) as f32;
            if let Some(segment) = self.surfaces[surface_index].sample(weights, angle, spacing) {
                self.segments.push(segment);
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
        let tangent_frame = (determinant.abs() > 1e-12).then(|| {
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
            tangent_frame,
        })
    }

    fn sample(&self, [wa, wb, wc]: [f32; 3], angle: f32, spacing: f32) -> Option<GaussianSegment> {
        let [a, b, c] = self.vertices;
        let (mut color, coverage) = self.material.color(
            a.color * wa + b.color * wb + c.color * wc,
            a.uv * wa + b.uv * wb + c.uv * wc,
        );
        if coverage == 0.0 {
            return None;
        }
        if self.material.bake_light {
            let mut normal = (a.normal * wa + b.normal * wb + c.normal * wc)
                .try_normalize()
                .unwrap_or(self.frame.z_axis);
            if let (Some(map), Some((tangent, bitangent))) =
                (&self.material.normal, self.tangent_frame)
            {
                let sampled = 2.0
                    * map
                        .sample(
                            a.normal_uv * wa + b.normal_uv * wb + c.normal_uv * wc,
                            false,
                        )
                        .truncate()
                    - Vec3::ONE;
                let tangent = (tangent - normal * normal.dot(tangent)).normalize_or_zero();
                let cross = normal.cross(tangent);
                let handedness = cross.dot(bitangent).signum();
                let scale = self.material.normal_scale;
                normal = (tangent * sampled.x * scale
                    + cross * handedness * sampled.y * scale
                    + normal * sampled.z)
                    .try_normalize()
                    .unwrap_or(normal);
            }
            let key = normal.dot(Vec3::new(-0.55, 0.78, 0.3)).max(0.0);
            let fill = normal.dot(Vec3::new(0.71, 0.18, -0.68)).max(0.0);
            color *= 0.27 + 0.67 * key + 0.26 * fill;
        }
        let center = a.position * wa + b.position * wb + c.position * wc;
        let (sin, cos) = angle.sin_cos();
        let offset = (self.frame.x_axis * cos + self.frame.y_axis * sin) * (0.58 * spacing);
        Some(GaussianSegment::new(
            center - offset,
            center + offset,
            Vec3::new(0.70, 0.70, 0.17) * spacing,
            Quat::from_mat3(&self.frame).normalize(),
            4.5 * coverage * spacing * spacing,
            color,
        ))
    }
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
        let normal = bake_light.then(|| material.normal_texture()).flatten();
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
        let rgba =
            self.factor * vertex_color * self.base.map_or(Vec4::ONE, |map| map.sample(uv, true));
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
