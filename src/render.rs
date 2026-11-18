use std::{collections::HashMap, num::NonZeroU64, ops::Range};

use bevy::{
    asset::{AssetLoader, LoadContext, embedded_asset, io::Reader, load_embedded_asset},
    camera::{
        Exposure, MainPassResolutionOverride,
        primitives::{Aabb, Frustum},
        visibility::{
            InheritedVisibility, NoFrustumCulling, RenderLayers, ViewVisibility, VisibilityRange,
        },
    },
    color::ColorToComponents,
    core_pipeline::{
        core_3d::{AlphaMask3d, CORE_3D_DEPTH_FORMAT, Transparent3d, TransparentSortingInfo3d},
        prepass::{OpaqueNoLightmap3dBatchSetKey, OpaqueNoLightmap3dBinKey},
    },
    ecs::{
        query::QueryItem,
        system::{SystemParamItem, lifetimeless::*},
    },
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
        camera::TemporalJitter,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, PhaseItem, PhaseItemExtraIndex,
            RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
            ViewBinnedRenderPhases, ViewSortedRenderPhases,
        },
        render_resource::{binding_types::*, *},
        renderer::{RenderDevice, RenderQueue},
        sync_component::SyncComponent,
        sync_world::{MainEntity, MainEntityHashMap},
        view::{ExtractedView, RetainedViewEntity},
    },
};
use bevy_light::{DirectionalLight, VolumetricLight};
use bevy_math::{Affine3A, Vec3A};
use bevy_pbr::{DistanceFog, FogFalloff, MAX_DIRECTIONAL_LIGHTS};
use bytemuck::{Pod, Zeroable};

use crate::{AssetError, DetailTransition, SpaghettiAsset, asset::MAX_DETAIL_TRANSITIONS};

/// Renders a cloud using the entity's transform, visibility and render layers.
#[derive(Component, Clone, Debug)]
#[require(Transform, Visibility)]
pub struct Spaghetti3d(pub Handle<SpaghettiAsset>);

/// Optional diffuse shading for a cloud with oriented surface splats.
///
/// Each segment's rotated positive Z axis acts as its surface normal. The
/// converter stores normals in this orientation for triangle samples.
/// Supply lighting as linear RGB multipliers on the segment's base color.
#[derive(Component, Clone, Copy, Debug)]
pub struct SpaghettiLighting {
    /// World-space direction from the surface to the light.
    pub direction_to_light: Vec3,
    pub ambient: Vec3,
    pub directional: Vec3,
}

/// Optional light transmission for thin splats such as leaves and blades.
///
/// With [`SpaghettiLighting`], a segment lit from its back face receives this
/// fraction of the directional light, so backlit foliage glows instead of
/// turning black. Values from 0.3 to 0.7 suit grass and needles.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct SpaghettiTransmission(pub f32);

/// Fades segments out as they come close to the camera.
///
/// Segments whose view depth is below `start` metres are hidden, and segments
/// beyond `end` draw at full opacity. Use it on foliage that can pass between a
/// follow camera and its subject, where close splats spread into soft blurs.
#[derive(Component, Clone, Copy, Debug)]
pub struct SpaghettiNearFade {
    pub start: f32,
    pub end: f32,
}

/// Draws dense surface splats as depth-writing Gaussian cutouts.
///
/// Cutouts use a projected cross section and an optical-depth threshold. They
/// draw directly from the asset's segment buffer without sorting the segments.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct SpaghettiCutout;

/// Applies gusts and an optional pusher in world space.
///
/// Set endpoint bend heights with [`crate::GaussianSegment::with_bend_heights`].
/// Update this component to animate a cloud without changing its asset.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct SpaghettiWind {
    /// Time in seconds.
    pub time: f32,
    /// World-space position at the base of the pusher.
    pub pusher: Vec3,
    /// Horizontal radius in world units. Zero disables contact.
    pub push_radius: f32,
}

/// Registers the `.spag` asset loader and the Bevy 3D render pipeline.
///
/// Add this after the plugins that provide Bevy's asset server and 3D renderer.
pub struct SpaghettiPlugin;

impl Plugin for SpaghettiPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<SpaghettiAsset>()
            .init_asset_loader::<SpaghettiLoader>()
            .add_plugins((
                RenderAssetPlugin::<GpuCloud>::default(),
                ExtractComponentPlugin::<Spaghetti3d>::default(),
            ));
        embedded_asset!(app, "spaghetti.wgsl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Transparent3d, (SetItemPipeline, DrawCloud)>()
                .add_render_command::<AlphaMask3d, (SetItemPipeline, DrawCloud)>()
                .init_resource::<SpecializedRenderPipelines<SplatPipeline>>()
                .init_resource::<ViewDraws>()
                .init_resource::<ViewFog>()
                .add_systems(ExtractSchedule, extract_view_fog)
                .add_systems(RenderStartup, init_pipeline)
                .add_systems(Render, queue_clouds.in_set(RenderSystems::QueueMeshes));
        }
    }
}

#[derive(Default, TypePath)]
struct SpaghettiLoader;

impl AssetLoader for SpaghettiLoader {
    type Asset = SpaghettiAsset;
    type Settings = ();
    type Error = AssetError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _: &(),
        _: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        SpaghettiAsset::from_bytes(&bytes)
    }

    fn extensions(&self) -> &[&str] {
        &["spag"]
    }
}

#[derive(Component)]
pub struct ExtractedCloud {
    asset: AssetId<SpaghettiAsset>,
    world_from_local: Mat4,
    layers: RenderLayers,
    disable_culling: bool,
    lighting: Option<SpaghettiLighting>,
    transmission: f32,
    cutout: bool,
    wind: Option<SpaghettiWind>,
    range: Option<VisibilityRange>,
    near_fade: Option<SpaghettiNearFade>,
}

impl SyncComponent<RenderApp> for Spaghetti3d {
    type Target = ExtractedCloud;
}

impl ExtractComponent<RenderApp> for Spaghetti3d {
    type QueryData = (
        &'static Spaghetti3d,
        &'static GlobalTransform,
        &'static InheritedVisibility,
        Option<&'static RenderLayers>,
        Has<NoFrustumCulling>,
        Option<&'static SpaghettiLighting>,
        Option<&'static SpaghettiTransmission>,
        Has<SpaghettiCutout>,
        Option<&'static SpaghettiWind>,
        Option<&'static VisibilityRange>,
        Option<&'static SpaghettiNearFade>,
    );
    type QueryFilter = ();
    type Out = ExtractedCloud;

    fn extract_component(
        (cloud, transform, visibility, layers, disable_culling, lighting, transmission, cutout, wind, range, near_fade):
            QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        visibility.get().then(|| ExtractedCloud {
            asset: cloud.0.id(),
            world_from_local: transform.to_matrix(),
            layers: layers.cloned().unwrap_or_default(),
            disable_culling,
            lighting: lighting.copied(),
            transmission: transmission.map_or(0.0, |t| t.0.max(0.0)),
            cutout,
            wind: wind.copied(),
            range: range.cloned(),
            near_fade: near_fade.copied(),
        })
    }
}

struct GpuCloud {
    segments: Option<Buffer>,
    centers: Box<[Vec3]>,
    bounds: Aabb,
    max_bend_height: f32,
    detail: Box<[DetailTransition]>,
    revision: u64,
}

impl RenderAsset for GpuCloud {
    type SourceAsset = SpaghettiAsset;
    type Param = (SRes<RenderDevice>, SRes<RenderQueue>);

    fn byte_len(asset: &Self::SourceAsset) -> Option<usize> {
        Some(std::mem::size_of_val(asset.segments()))
    }

    fn prepare_asset(
        asset: Self::SourceAsset,
        _: AssetId<Self::SourceAsset>,
        (device, queue): &mut SystemParamItem<Self::Param>,
        previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let data = bytemuck::cast_slice(asset.segments());
        let revision = previous.map_or(0, |cloud| cloud.revision.wrapping_add(1));
        if data.len() > device.limits().max_storage_buffer_binding_size as usize {
            error!(
                "Spaghetti asset requires {} storage bytes, exceeding this GPU's limit",
                data.len()
            );
            return Ok(Self {
                segments: None,
                centers: Box::new([]),
                bounds: Aabb::from_min_max(Vec3::ZERO, Vec3::ZERO),
                max_bend_height: 0.0,
                detail: Box::new([]),
                revision,
            });
        }
        let (min, max) = asset.bounds();
        let centers = asset
            .segments()
            .iter()
            .map(|segment| segment.start() * 0.5 + segment.end() * 0.5)
            .collect();
        let existing = previous
            .and_then(|cloud| cloud.segments.as_ref())
            .filter(|buffer| buffer.size() == data.len() as u64);
        let segments = if let Some(buffer) = existing {
            queue.write_buffer(buffer, 0, data);
            buffer.clone()
        } else {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("spaghetti segment data"),
                contents: data,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            })
        };
        Ok(Self {
            segments: Some(segments),
            centers,
            bounds: Aabb::from_min_max(min, max),
            max_bend_height: asset.segments().iter().map(|segment| segment.max_bend_height()).fold(0.0, f32::max),
            detail: asset.detail_transitions().into(),
            revision,
        })
    }
}

impl GpuCloud {
    // Drop complete levels whose bands are outside the depth interval of the
    // cloud. This also keeps distant draws at the cost of a single coarse level.
    fn select_segments(
        &self,
        clip_from_local: Mat4,
        viewport_height: f32,
        bounds: Aabb,
        uniform: &mut DrawUniform,
    ) -> Range<u32> {
        let count = self.centers.len() as u32;
        if self.detail.is_empty() {
            return 0..count;
        }

        // A local reference length projects to scale / clip.w pixels. The Y row
        // accounts for viewport focal length and asset scale without depending
        // on segment orientation or its distance from the center of the screen.
        let scale = clip_from_local.row(1).truncate().length() * viewport_height * 0.5;
        let depth = clip_from_local.row(3);
        let center = depth.dot(Vec3::from(bounds.center).extend(1.0));
        let radius = depth.truncate().abs().dot(bounds.half_extents.into());
        let padding = (center.abs() + radius + 1.0) * 1e-5;
        let closest = center - radius - padding;
        let farthest = center + radius + padding;
        let mut segments = 0..count;
        for transition in &self.detail {
            let [low, high] = transition.pixels_per_unit;
            let near = scale / high;
            let far = scale / low;
            if closest >= far {
                segments.start = transition.finer_end;
            } else if farthest <= near {
                segments.end = transition.finer_end;
                break;
            } else {
                let slot = uniform.detail_ends[3] as usize;
                uniform.detail_depths[slot] = [near, far, 0.0, 0.0];
                uniform.detail_ends[slot] = transition.finer_end;
                uniform.detail_ends[3] += 1;
            }
        }
        segments
    }
}

#[derive(Resource)]
struct SplatPipeline {
    shader: Handle<Shader>,
    bindings: BindGroupLayoutDescriptor,
}

fn init_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(SplatPipeline {
        shader: load_embedded_asset!(asset_server.as_ref(), "spaghetti.wgsl"),
        bindings: BindGroupLayoutDescriptor::new(
            "spaghetti bindings",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    uniform_buffer_sized(false, NonZeroU64::new(size_of::<DrawUniform>() as u64)),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        ),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct PipelineKey {
    target: TextureFormat,
    samples: u32,
    cutout: bool,
}

impl SpecializedRenderPipeline for SplatPipeline {
    type Key = PipelineKey;

    fn specialize(&self, key: PipelineKey) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("spaghetti volume compositing".into()),
            layout: vec![self.bindings.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some(if key.cutout { "cutout_vertex" } else { "vertex" }.into()),
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some(if key.cutout { "cutout_fragment" } else { "fragment" }.into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target,
                    blend: (!key.cutout).then_some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(key.cutout),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.samples,
                ..default()
            },
            ..default()
        }
    }
}

// Keep this capacity in sync with the array in spaghetti.wgsl.
const FOG_LIGHT_CAPACITY: usize = 10;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FogLight {
    direction: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FogUniform {
    color: [f32; 4],
    extinction: [f32; 3],
    mode: u32,
    inscattering: [f32; 3],
    directional_exponent: f32,
    light_count: [u32; 4],
    lights: [FogLight; FOG_LIGHT_CAPACITY],
}

#[derive(Resource, Default)]
struct ViewFog(MainEntityHashMap<FogUniform>);

fn extract_view_fog(
    mut view_fog: ResMut<ViewFog>,
    views: Extract<
        Query<
            (
                Entity,
                &DistanceFog,
                Option<&Exposure>,
                Option<&RenderLayers>,
            ),
            With<Camera>,
        >,
    >,
    lights: Extract<
        Query<(
            Entity,
            &DirectionalLight,
            &GlobalTransform,
            &ViewVisibility,
            Option<&RenderLayers>,
            Has<VolumetricLight>,
        )>,
    >,
) {
    view_fog.0.clear();
    if views.is_empty() {
        return;
    }
    let default_layers = RenderLayers::default();
    let mut lights: Vec<_> = lights.iter().filter(|light| light.3.get()).collect();
    // Follow Bevy's ordering when more lights are visible than fit in the uniform.
    lights.sort_unstable_by_key(|&(entity, light, _, _, _, volumetric)| {
        (volumetric, light.shadow_maps_enabled, entity)
    });
    for (entity, fog, exposure, layers) in views.iter() {
        let (mode, extinction, inscattering) = match fog.falloff {
            FogFalloff::Linear { start, end } => (1, [start, end, 0.0], [0.0; 3]),
            FogFalloff::Exponential { density } => (2, [density, 0.0, 0.0], [0.0; 3]),
            FogFalloff::ExponentialSquared { density } => (3, [density, 0.0, 0.0], [0.0; 3]),
            FogFalloff::Atmospheric {
                extinction,
                inscattering,
            } => (4, extinction.to_array(), inscattering.to_array()),
        };
        let mut uniform = FogUniform {
            color: fog.color.to_linear().to_f32_array(),
            extinction,
            mode,
            inscattering,
            directional_exponent: fog.directional_light_exponent,
            ..FogUniform::zeroed()
        };
        let tint = fog.directional_light_color.to_linear().to_vec4();
        if tint.w > 0.0 {
            let layers = layers.unwrap_or(&default_layers);
            let exposure = exposure.copied().unwrap_or_default().exposure();
            let tint = tint.truncate() * (tint.w * exposure);
            let view_lights = lights
                .iter()
                .filter(|light| layers.intersects(light.4.unwrap_or(&default_layers)))
                .take(MAX_DIRECTIONAL_LIGHTS);
            for (slot, &(_, light, transform, _, _, _)) in
                uniform.lights.iter_mut().zip(view_lights)
            {
                let color = light.color.to_linear().to_vec4().truncate() * light.illuminance;
                *slot = FogLight {
                    direction: transform.back().extend(0.0).to_array(),
                    color: (color * tint).extend(0.0).to_array(),
                };
                uniform.light_count[0] += 1;
            }
        }
        view_fog.0.insert(MainEntity::from(entity), uniform);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    clip_from_local: [[f32; 4]; 4],
    viewport: [f32; 4],
    normal_from_local: [[f32; 4]; 3],
    direction_to_light: [f32; 4],
    ambient: [f32; 4],
    directional: [f32; 4],
    eye_relative_from_local: [[f32; 4]; 4],
    fog: FogUniform,
    world_from_local: [[f32; 4]; 4],
    local_from_world: [[f32; 4]; 3],
    pusher: [f32; 4],
    // Time, wind enabled, range coverage, reverse range dither.
    motion: [f32; 4],
    // Near and far clip.w for each active transition.
    detail_depths: [[f32; 4]; MAX_DETAIL_TRANSITIONS],
    // Exclusive finer-level end indices, followed by active transition count.
    detail_ends: [u32; 4],
}

impl DrawUniform {
    fn set_lighting(&mut self, lighting: SpaghettiLighting, world_from_local: Mat4) {
        let transform = Mat3::from_mat4(world_from_local).inverse().transpose();
        if transform.is_finite() {
            self.normal_from_local = [transform.x_axis, transform.y_axis, transform.z_axis]
                .map(|axis| axis.extend(0.0).to_array());
        }
        self.direction_to_light = lighting
            .direction_to_light
            .normalize_or_zero()
            .extend(1.0)
            .to_array();
        self.ambient = lighting.ambient.extend(0.0).to_array();
        self.directional = lighting.directional.extend(0.0).to_array();
    }
}

struct CloudDraw {
    main_entity: MainEntity,
    cutout_pipeline: Option<CachedRenderPipelineId>,
    source: Buffer,
    uniform: Buffer,
    order: Option<Buffer>,
    bindings: BindGroup,
    direction: Vec3,
    revision: u64,
    segments: Range<u32>,
    idle_frames: u8,
}

#[derive(Resource, Default)]
struct ViewDraws(HashMap<(RetainedViewEntity, Entity), CloudDraw>);

type SplatView = (
    &'static ExtractedView,
    &'static Msaa,
    Option<&'static Frustum>,
    Option<&'static RenderLayers>,
    Option<&'static TemporalJitter>,
    Option<&'static MainPassResolutionOverride>,
);

#[expect(clippy::too_many_arguments, reason = "Bevy system parameters")]
fn queue_clouds(
    mut draws: ResMut<ViewDraws>,
    gpu_clouds: Res<RenderAssets<GpuCloud>>,
    clouds: Query<(Entity, &MainEntity, &ExtractedCloud)>,
    views: Query<SplatView>,
    view_fog: Res<ViewFog>,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    mut cutout_phases: ResMut<ViewBinnedRenderPhases<AlphaMask3d>>,
    cutout_draw_functions: Res<DrawFunctions<AlphaMask3d>>,
    pipeline: Res<SplatPipeline>,
    cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SplatPipeline>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut depths: Local<Vec<(f32, u32)>>,
    mut order: Local<Vec<u32>>,
) {
    for draw in draws.0.values_mut() {
        draw.idle_frames += 1;
    }

    let draw_function = draw_functions.read().id::<(SetItemPipeline, DrawCloud)>();
    let cutout_draw_function = cutout_draw_functions.read().id::<(SetItemPipeline, DrawCloud)>();
    let bind_group_layout = cache.get_bind_group_layout(&pipeline.bindings);
    let default_layers = RenderLayers::default();
    for (view, msaa, frustum, layers, jitter, resolution) in &views {
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let layers = layers.unwrap_or(&default_layers);
        let mut cutout_phase = cutout_phases.get_mut(&view.retained_view_entity);
        let fog = view_fog.0.get(&view.retained_view_entity.main_entity);
        let size = resolution.map_or(view.viewport.zw(), |resolution| resolution.0);
        if size.min_element() == 0 {
            continue;
        }

        let view_from_world = view.world_from_view.to_matrix().inverse();
        let mut projection = view.clip_from_view;
        let clip_from_world = if let Some(jitter) = jitter {
            jitter.jitter_projection(&mut projection, size.as_vec2());
            projection * view_from_world
        } else {
            view.clip_from_world.unwrap_or(projection * view_from_world)
        };
        let mut pipeline_ids = [None; 2];

        for (entity, main_entity, cloud) in &clouds {
            if !layers.intersects(&cloud.layers) {
                continue;
            }
            let Some(gpu) = gpu_clouds.get(cloud.asset) else {
                continue;
            };
            let Some(source) = &gpu.segments else {
                continue;
            };
            let [coverage, reverse_dither] = cloud.range.as_ref().map_or([1.0, 0.0], |range| {
                let center = if range.use_aabb {
                    cloud.world_from_local.transform_point3(gpu.bounds.center.into())
                } else {
                    cloud.world_from_local.w_axis.truncate()
                };
                let distance = view.world_from_view.translation().distance(center);
                if range.is_culled(distance) {
                    [0.0, 0.0]
                } else if distance < range.start_margin.end {
                    [(distance - range.start_margin.start) / (range.start_margin.end - range.start_margin.start), 0.0]
                } else if distance > range.end_margin.start {
                    [(range.end_margin.end - distance) / (range.end_margin.end - range.end_margin.start), 1.0]
                } else {
                    [1.0, 0.0]
                }
            });
            if coverage <= 0.0 || (cloud.cutout && cutout_phase.is_none()) {
                continue;
            }
            let inverse_basis = cloud.wind.and_then(|_| {
                let inverse = Mat3::from_mat4(cloud.world_from_local).inverse();
                inverse.is_finite().then_some(inverse)
            });
            let mut bounds = gpu.bounds;
            if let Some(inverse) = inverse_basis {
                let rows = inverse.transpose();
                let stretch = Vec3::new(rows.x_axis.length(), rows.y_axis.length(), rows.z_axis.length());
                // Maximum displacement combines horizontal reach and vertical shortening.
                bounds.half_extents += Vec3A::from(stretch * (gpu.max_bend_height * 1.11));
            }
            if !cloud.disable_culling
                && frustum.is_some_and(|frustum| {
                    !frustum.intersects_obb(
                        &bounds,
                        &Affine3A::from_mat4(cloud.world_from_local),
                        true,
                        true,
                    )
                })
            {
                continue;
            }
            let clip_from_local = clip_from_world * cloud.world_from_local;
            let mut uniform = DrawUniform {
                clip_from_local: clip_from_local.to_cols_array_2d(),
                viewport: [size.x as f32, size.y as f32, 0.0, 0.0],
                motion: [0.0, 0.0, coverage, reverse_dither],
                ..DrawUniform::zeroed()
            };
            if let Some(fade) = cloud.near_fade.filter(|fade| fade.end > fade.start) {
                uniform.viewport[2] = fade.start;
                uniform.viewport[3] = fade.end;
            }
            if let (Some(wind), Some(inverse)) = (cloud.wind, inverse_basis) {
                uniform.world_from_local = cloud.world_from_local.to_cols_array_2d();
                uniform.local_from_world = [inverse.x_axis, inverse.y_axis, inverse.z_axis]
                    .map(|axis| axis.extend(0.0).to_array());
                if wind.push_radius > 0.0 {
                    let transform = cloud.world_from_local;
                    let center = transform.transform_point3(gpu.bounds.center.into());
                    let half = gpu.bounds.half_extents;
                    let extent = transform.x_axis.truncate().abs() * half.x
                        + transform.y_axis.truncate().abs() * half.y
                        + transform.z_axis.truncate().abs() * half.z;
                    let outside = ((wind.pusher - center).xz().abs() - extent.xz()).max(Vec2::ZERO);
                    if outside.length_squared() < wind.push_radius * wind.push_radius {
                        uniform.pusher = wind.pusher.extend(wind.push_radius).to_array();
                    }
                }
                uniform.motion[0] = wind.time;
                uniform.motion[1] = 1.0;
            }
            if let Some(lighting) = cloud.lighting {
                uniform.set_lighting(lighting, cloud.world_from_local);
                uniform.ambient[3] = cloud.transmission;
            }
            if let Some(fog) = fog {
                let mut eye_relative = cloud.world_from_local;
                eye_relative.w_axis -= view.world_from_view.translation().extend(0.0);
                uniform.eye_relative_from_local = eye_relative.to_cols_array_2d();
                uniform.fog = *fog;
            }
            let key = (view.retained_view_entity, entity);
            let segments = gpu.select_segments(clip_from_local, size.y as f32, bounds, &mut uniform);
            let new_draw = || make_draw(&device, &bind_group_layout, source, gpu.centers.len(), *main_entity, cloud.cutout);
            let draw = draws.0.entry(key).or_insert_with(new_draw);
            if draw.source.id() != source.id() || draw.order.is_none() != cloud.cutout {
                if draw.cutout_pipeline.is_some() && let Some(phase) = cutout_phase.as_deref_mut() {
                    phase.remove(draw.main_entity);
                }
                *draw = new_draw();
            }
            if let Some(order_buffer) = &draw.order {
                let direction = (view_from_world * cloud.world_from_local)
                    .row(2)
                    .truncate()
                    .normalize_or_zero();
                if draw.direction != direction
                    || draw.revision != gpu.revision
                    || draw.segments != segments
                {
                    // Translation preserves relative depths. Re-sort only if the
                    // view direction, asset data, or submitted levels changed.
                    depths.clear();
                    depths.extend(
                        gpu.centers[segments.start as usize..segments.end as usize]
                            .iter()
                            .enumerate()
                            .map(|(index, &center)| {
                                (center.dot(direction), segments.start + index as u32)
                            }),
                    );
                    radsort::sort_by_key(&mut depths, |entry| entry.0);
                    order.clear();
                    order.extend(depths.iter().map(|&(_, index)| index));
                    queue.write_buffer(order_buffer, 0, bytemuck::cast_slice(&order));
                    draw.direction = direction;
                    draw.revision = gpu.revision;
                }
            }
            draw.segments = segments;
            queue.write_buffer(&draw.uniform, 0, bytemuck::bytes_of(&uniform));
            draw.idle_frames = 0;
            let pipeline_id = *pipeline_ids[usize::from(cloud.cutout)].get_or_insert_with(|| {
                pipelines.specialize(
                    &cache,
                    &pipeline,
                    PipelineKey { target: view.target_format, samples: msaa.samples(), cutout: cloud.cutout },
                )
            });
            if let Some(phase) = cutout_phase.as_deref_mut().filter(|_| cloud.cutout) {
                if draw.cutout_pipeline != Some(pipeline_id) {
                    if draw.cutout_pipeline.replace(pipeline_id).is_some() {
                        phase.remove(*main_entity);
                    }
                    phase.add(
                        OpaqueNoLightmap3dBatchSetKey {
                            pipeline: pipeline_id,
                            draw_function: cutout_draw_function,
                            material_bind_group_index: None,
                            slabs: default(),
                        },
                        OpaqueNoLightmap3dBinKey { asset_id: cloud.asset.untyped() },
                        (entity, *main_entity),
                        default(),
                        BinnedRenderPhaseType::NonMesh,
                    );
                }
                continue;
            }
            phase.add_transient(Transparent3d {
                sorting_info: TransparentSortingInfo3d::Sorted {
                    mesh_center: cloud
                        .world_from_local
                        .transform_point3(gpu.bounds.center.into()),
                    depth_bias: 0.0,
                },
                distance: 0.0,
                pipeline: pipeline_id,
                entity: (entity, *main_entity),
                draw_function,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: false,
            });
        }
    }
    // Binned phases retain draws across frames. Remove entries that fell out
    // of range while keeping their buffers briefly for throttled views.
    draws.0.retain(|&(view, _), draw| {
        if draw.idle_frames > 0
            && draw.cutout_pipeline.take().is_some()
            && let Some(phase) = cutout_phases.get_mut(&view)
        {
            phase.remove(draw.main_entity);
        }
        draw.idle_frames <= 1
    });
}

fn make_draw(
    device: &RenderDevice,
    layout: &BindGroupLayout,
    source: &Buffer,
    count: usize,
    main_entity: MainEntity,
    cutout: bool,
) -> CloudDraw {
    let make_buffer = |label, size: usize, usage| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: usage | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let uniform = make_buffer(
        "spaghetti draw uniform",
        size_of::<DrawUniform>(),
        BufferUsages::UNIFORM,
    );
    let order = (!cutout).then(|| make_buffer(
        "spaghetti depth order",
        count * size_of::<u32>(),
        BufferUsages::STORAGE,
    ));
    let bindings = device.create_bind_group(
        "spaghetti draw bindings",
        layout,
        &BindGroupEntries::sequential((
            uniform.as_entire_binding(),
            source.as_entire_binding(),
            order.as_ref().unwrap_or(source).as_entire_binding(),
        )),
    );
    CloudDraw {
        main_entity,
        cutout_pipeline: None,
        source: source.clone(),
        uniform,
        order,
        bindings,
        direction: Vec3::NAN,
        revision: 0,
        segments: 0..count as u32,
        idle_frames: 0,
    }
}

struct DrawCloud;

impl<P: PhaseItem> RenderCommand<P> for DrawCloud {
    type Param = SRes<ViewDraws>;
    type ViewQuery = Read<ExtractedView>;
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        view: &'w ExtractedView,
        _: Option<()>,
        draws: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let key = (view.retained_view_entity, item.entity());
        let Some(draw) = draws.into_inner().0.get(&key) else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(0, &draw.bindings, &[]);
        let instances = if draw.order.is_some() {
            0..draw.segments.len() as u32
        } else {
            draw.segments.clone()
        };
        pass.draw(0..4, instances);
        RenderCommandResult::Success
    }
}
