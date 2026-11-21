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
            AddRenderCommand, BinnedRenderPhase, BinnedRenderPhaseType, DrawFunctionId,
            DrawFunctions, PhaseItem, PhaseItemExtraIndex,
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
use bevy_pbr::{
    CORE_3D_SHADOW_MAP_FORMAT, DistanceFog, ExtractedDirectionalLight, FogFalloff, LightEntity,
    MAX_DIRECTIONAL_LIGHTS, Shadow, ShadowBatchSetKey, ShadowBinKey,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    AssetError, DetailTransition, GaussianSegment, SpaghettiAsset, asset::MAX_DETAIL_TRANSITIONS,
};

mod shadow;
use shadow::{ShadowBindings, ShadowViewData};

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

/// Adds separate ground bounce to a cloud with [`SpaghettiLighting`].
/// The segment's world-space normal blends this linear RGB multiplier on
/// downward faces with [`SpaghettiLighting::ambient`] on upward faces.
#[derive(Component, Clone, Copy, Debug)]
pub struct SpaghettiGroundAmbient(pub Vec3);

/// Receives Bevy directional shadow maps on a cloud with [`SpaghettiLighting`].
///
/// Uses the shadow-casting light whose direction matches the cloud's light.
/// Outside its cascades or in a view without that light, the cloud stays lit.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct SpaghettiReceiveShadows;

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

/// Casts the cloud's shadow into Bevy's directional shadow maps.
///
/// Each segment blocks a shadow texel with probability 1 - exp(-τ), where τ
/// is the segment's optical depth across that texel. Every segment draws its
/// own random pattern, so overlapping segments multiply their transmittance,
/// and shadow filtering averages the texels into partial shade. Thin or sparse
/// clouds cast light shadows and dense ones cast dark shadows.
///
/// Shadows bend with wind and sway and ignore [`SpaghettiFlow`]. They use a
/// finer detail level than the shadow map's texel size alone selects, because
/// coarse segments are wide and cover more texels. A [`VisibilityRange`] is
/// measured from the camera that owns the cascade.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct SpaghettiCastShadows;

/// Keeps every spaghetti cloud out of this directional light's shadow maps.
///
/// Use it on a light whose shadow map exists for another reason, such as a
/// tiny cascade that only lets volumetric fog see the light. Such a cascade
/// looks along the light through the whole scene, so the few clouds that
/// cross its narrow frustum cover the entire map with fragments.
#[derive(Component, Clone, Copy, Debug, Default, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct NoSpaghettiShadows;

/// Multiplies an asset's projected detail thresholds for this entity.
///
/// Values above one switch to coarser levels sooner. One keeps the asset's
/// schedule. Missing, non-positive, and non-finite values behave as one.
#[derive(Component, Clone, Copy, Debug)]
pub struct SpaghettiDetailScale(pub f32);

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

/// Bends a cloud using a precomputed world-space displacement per meter of
/// bend height. Set heights with [`crate::GaussianSegment::with_bend_heights`].
/// Endpoints with zero height stay fixed. This component takes precedence
/// over [`SpaghettiWind`] when both are present.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct SpaghettiSway(pub Vec3);

/// Adds a view-dependent highlight to a cloud with [`SpaghettiLighting`].
///
/// The highlight uses the lighting's direction and `directional` color, and
/// baked occlusion and received shadows dim it. Each segment gets one
/// highlight value, so dense clouds glint splat by splat.
#[derive(Component, Clone, Copy, Debug)]
pub struct SpaghettiSpecular {
    /// Peak highlight as a multiple of [`SpaghettiLighting::directional`].
    pub strength: f32,
    /// Phong exponent. Higher values give a tighter highlight.
    pub shininess: f32,
    /// Zero reflects light off each segment's surface normal, as wet stone
    /// does. One reflects it off the segment's axis, as hair, needles and
    /// thin blades do. Values between blend the two.
    pub strand: f32,
}

/// A local light source for lit clouds, such as a fire or a lamp.
///
/// Add it to an entity with a transform. A cloud with [`SpaghettiLighting`]
/// takes the one source that lights its bounds most strongly, from the
/// sources that share a render layer with it. The light follows the inverse
/// square law, falls smoothly to zero at `range`, and casts no shadows.
/// [`SpaghettiTransmission`] passes it through thin foliage.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
#[require(Transform, Visibility)]
pub struct SpaghettiPointLight {
    /// Light on a surface that faces the source from one unit away, on the
    /// scale of [`SpaghettiLighting::directional`].
    pub color: Vec3,
    /// Distance at which the light reaches zero.
    pub range: f32,
    /// Radius of the source. It bounds the light on segments close to it.
    pub radius: f32,
}

/// Adds a translucent cloud's light to the frame instead of covering it.
///
/// Use it for flames, sparks, sunlit dust and other glowing media. Draw order
/// has no effect on the result, and distance fog only dims the cloud. This
/// component has no effect on a [`SpaghettiCutout`] cloud.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct SpaghettiAdditive;

/// Moves each segment of a translucent cloud along a looping path.
///
/// A segment starts each loop at its authored place and travels
/// `velocity * period`. Each segment starts its loop at its own random time,
/// so the cloud flows without a visible restart. Update `time` to animate the
/// cloud without changing its asset. This component has no effect on a
/// [`SpaghettiCutout`] cloud, and a non-positive `period` turns it off.
#[derive(Component, Clone, Copy, Debug)]
pub struct SpaghettiFlow {
    /// Time in seconds.
    pub time: f32,
    /// World-space travel per second.
    pub velocity: Vec3,
    /// Length of one loop in seconds.
    pub period: f32,
    /// World-space reach of each segment's smooth random wander.
    pub turbulence: f32,
    /// Growth of the distance from the flow axis over one loop. The axis
    /// passes through the cloud's origin along `velocity`. Negative values
    /// draw segments toward the axis, as a flame narrows to its tip.
    pub spread: f32,
    /// Growth of each segment's kernel over one loop. A grown kernel keeps
    /// its optical mass, so it thins out as smoke does.
    pub growth: f32,
    /// Depth of each segment's random brightness pulsing, from 0 to 1.
    pub flicker: f32,
    /// Fractions of the loop spent fading in and fading out.
    pub fade: [f32; 2],
    /// Color factor that each segment reaches at the end of its loop. The
    /// factor starts at one, so a flame can cool from yellow to red and
    /// smoke can pale as it thins.
    pub tint: Vec3,
}

impl Default for SpaghettiFlow {
    fn default() -> Self {
        Self {
            time: 0.0,
            velocity: Vec3::ZERO,
            period: 1.0,
            turbulence: 0.0,
            spread: 0.0,
            growth: 0.0,
            flicker: 0.0,
            fade: [0.1, 0.3],
            tint: Vec3::ONE,
        }
    }
}

/// Registers the `.spag` asset loader and the Bevy 3D render pipeline.
///
/// Add this after the plugins that provide Bevy's asset server and 3D renderer.
/// The loader accepts `Option<([f32; 3], f32)>` settings for baking at load
/// time. Supply the asset-space sun direction and optical density scale.
pub struct SpaghettiPlugin;

impl Plugin for SpaghettiPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<SpaghettiAsset>()
            .init_asset_loader::<SpaghettiLoader>()
            .add_plugins((
                RenderAssetPlugin::<GpuCloud>::default(),
                ExtractComponentPlugin::<Spaghetti3d>::default(),
                ExtractComponentPlugin::<NoSpaghettiShadows>::default(),
            ));
        embedded_asset!(app, "spaghetti.wgsl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Transparent3d, (SetItemPipeline, DrawCloud)>()
                .add_render_command::<AlphaMask3d, (SetItemPipeline, DrawCloud)>()
                .add_render_command::<Shadow, (SetItemPipeline, DrawCloud)>()
                .init_resource::<SpecializedRenderPipelines<SplatPipeline>>()
                .init_resource::<ViewDraws>()
                .init_resource::<ViewFog>()
                .init_resource::<PointLights>()
                .add_systems(ExtractSchedule, (extract_view_fog, extract_point_lights))
                .add_systems(RenderStartup, init_pipeline)
                .add_systems(Render, queue_clouds.in_set(RenderSystems::QueueMeshes));
        }
    }
}

#[derive(Default, TypePath)]
struct SpaghettiLoader;

impl AssetLoader for SpaghettiLoader {
    type Asset = SpaghettiAsset;
    type Settings = Option<([f32; 3], f32)>;
    type Error = AssetError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        bake: &Self::Settings,
        _: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let asset = SpaghettiAsset::from_bytes(&bytes)?;
        match *bake {
            Some((direction, density)) => {
                asset.bake_occlusion_with_density(Vec3::from_array(direction), density)
            }
            None => Ok(asset),
        }
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
    ground_ambient: Option<SpaghettiGroundAmbient>,
    receive_shadows: bool,
    transmission: f32,
    cutout: bool,
    wind: Option<SpaghettiWind>,
    sway: Option<Vec3>,
    range: Option<VisibilityRange>,
    near_fade: Option<SpaghettiNearFade>,
    detail_scale: f32,
    specular: Option<SpaghettiSpecular>,
    additive: bool,
    flow: Option<SpaghettiFlow>,
    cast_shadows: bool,
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
        Option<&'static SpaghettiGroundAmbient>,
        Has<SpaghettiReceiveShadows>,
        Option<&'static SpaghettiTransmission>,
        Has<SpaghettiCutout>,
        Option<&'static SpaghettiWind>,
        Option<&'static SpaghettiSway>,
        Option<&'static VisibilityRange>,
        (
            Option<&'static SpaghettiNearFade>,
            Option<&'static SpaghettiDetailScale>,
            Option<&'static SpaghettiSpecular>,
            Has<SpaghettiAdditive>,
            Option<&'static SpaghettiFlow>,
            Has<SpaghettiCastShadows>,
        ),
    );
    type QueryFilter = ();
    type Out = ExtractedCloud;

    fn extract_component(
        (
            cloud,
            transform,
            visibility,
            layers,
            disable_culling,
            lighting,
            ground_ambient,
            receive_shadows,
            transmission,
            cutout,
            wind,
            sway,
            range,
            (near_fade, detail_scale, specular, additive, flow, cast_shadows),
        ): QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        visibility.get().then(|| ExtractedCloud {
            asset: cloud.0.id(),
            world_from_local: transform.to_matrix(),
            layers: layers.cloned().unwrap_or_default(),
            disable_culling,
            lighting: lighting.copied(),
            ground_ambient: ground_ambient.copied(),
            receive_shadows,
            transmission: transmission.map_or(0.0, |t| t.0.max(0.0)),
            cutout,
            wind: wind.copied(),
            sway: sway.map(|sway| sway.0).filter(|sway| sway.is_finite()),
            range: range.cloned(),
            near_fade: near_fade.copied(),
            detail_scale: detail_scale
                .map(|scale| scale.0)
                .filter(|scale| scale.is_finite() && *scale > 0.0)
                .unwrap_or(1.0),
            specular: specular
                .copied()
                .filter(|specular| specular.strength > 0.0 && specular.shininess.is_finite()),
            additive: additive && !cutout,
            flow: flow.copied().filter(|flow| {
                !cutout
                    && flow.period > 0.0
                    && flow.period.is_finite()
                    && flow.time.is_finite()
                    && flow.velocity.is_finite()
            }),
            cast_shadows,
        })
    }
}

/// Segments in one culling cluster. The shader packs `count - 1` into the
/// low seven bits of a cluster entry.
const CLUSTER_SIZE: u32 = 128;

/// Spatially close segments of one detail level, culled together.
struct Cluster {
    first: u32,
    count: u32,
    /// Local box around every kernel's faint shadow edge, before bending.
    center: Vec3,
    half_extents: Vec3,
}

struct GpuCloud {
    segments: Option<Buffer>,
    occlusion: Option<Buffer>,
    // XYZ is the segment midpoint. W is its average endpoint bend height.
    centers: Box<[Vec4]>,
    // In segment order. The upload sorts each level along a Morton curve.
    clusters: Box<[Cluster]>,
    bounds: Aabb,
    max_bend_height: f32,
    detail: Box<[DetailTransition]>,
    revision: u64,
}

impl RenderAsset for GpuCloud {
    type SourceAsset = SpaghettiAsset;
    type Param = (SRes<RenderDevice>, SRes<RenderQueue>);

    fn byte_len(asset: &Self::SourceAsset) -> Option<usize> {
        Some(std::mem::size_of_val(asset.segments()) + asset.occlusion().len() * size_of::<u32>())
    }

    fn prepare_asset(
        asset: Self::SourceAsset,
        _: AssetId<Self::SourceAsset>,
        (device, queue): &mut SystemParamItem<Self::Param>,
        previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let (order, clusters) = cluster_segments(&asset);
        let sorted: Vec<GaussianSegment> =
            order.iter().map(|&index| asset.segments()[index as usize]).collect();
        let data: &[u8] = bytemuck::cast_slice(&sorted);
        let revision = previous.map_or(0, |cloud| cloud.revision.wrapping_add(1));
        if data.len() > device.limits().max_storage_buffer_binding_size as usize {
            error!(
                "Spaghetti asset requires {} storage bytes, exceeding this GPU's limit",
                data.len()
            );
            return Ok(Self {
                segments: None,
                occlusion: None,
                centers: Box::new([]),
                clusters: Box::new([]),
                bounds: Aabb::from_min_max(Vec3::ZERO, Vec3::ZERO),
                max_bend_height: 0.0,
                detail: Box::new([]),
                revision,
            });
        }
        let (min, max) = asset.bounds();
        let centers = sorted
            .iter()
            .map(|segment| {
                (segment.start() * 0.5 + segment.end() * 0.5).extend(segment.mean_bend_height())
            })
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
        let occlusion = (!asset.occlusion().is_empty()).then(|| {
            let packed: Vec<_> = order
                .iter()
                .map(|&index| {
                    let [ambient, directional] = asset.occlusion()[index as usize];
                    let pack = |factor: f32| (factor * 65535.0).round() as u32;
                    pack(ambient) | (pack(directional) << 16)
                })
                .collect();
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("spaghetti occlusion attributes"),
                contents: bytemuck::cast_slice(&packed),
                usage: BufferUsages::STORAGE,
            })
        });
        Ok(Self {
            segments: Some(segments),
            occlusion,
            centers,
            clusters,
            bounds: Aabb::from_min_max(min, max),
            max_bend_height: asset.segments().iter().map(|segment| segment.max_bend_height()).fold(0.0, f32::max),
            detail: asset.detail_transitions().into(),
            revision,
        })
    }
}

/// Orders each detail level along a Morton curve and splits it into clusters.
/// Returns the source index of each sorted segment and the clusters.
fn cluster_segments(asset: &SpaghettiAsset) -> (Vec<u32>, Box<[Cluster]>) {
    let segments = asset.segments();
    let ends = asset
        .detail_transitions()
        .iter()
        .map(|transition| transition.finer_end)
        .chain([segments.len() as u32]);
    let midpoint = |segment: &GaussianSegment| (segment.start() + segment.end()) * 0.5;
    // Spreads ten bits so that three of them interleave.
    let spread = |value: u32| {
        let mut x = value & 0x3ff;
        x = (x | (x << 16)) & 0x0300_00ff;
        x = (x | (x << 8)) & 0x0300_f00f;
        x = (x | (x << 4)) & 0x030c_30c3;
        (x | (x << 2)) & 0x0924_9249
    };
    let mut order = Vec::with_capacity(segments.len());
    let mut clusters = Vec::new();
    let mut keyed = Vec::new();
    let mut start = 0;
    for end in ends {
        let level = start..end;
        start = end;
        let (low, high) = segments[level.start as usize..level.end as usize].iter().fold(
            (Vec3::INFINITY, Vec3::NEG_INFINITY),
            |(low, high), segment| (low.min(midpoint(segment)), high.max(midpoint(segment))),
        );
        let scale = 1023.0 / (high - low).max(Vec3::splat(1e-6));
        keyed.clear();
        keyed.extend(level.map(|index| {
            let cell = ((midpoint(&segments[index as usize]) - low) * scale).as_uvec3();
            (spread(cell.x) | (spread(cell.y) << 1) | (spread(cell.z) << 2), index)
        }));
        radsort::sort_by_key(&mut keyed, |entry| entry.0);
        for chunk in keyed.chunks(CLUSTER_SIZE as usize) {
            let (low, high) = chunk.iter().fold(
                (Vec3::INFINITY, Vec3::NEG_INFINITY),
                |(low, high), &(_, index)| {
                    let segment = &segments[index as usize];
                    let sigma = segment.sigma();
                    let length = segment.start().distance(segment.end());
                    let area = 2.506628 * sigma.x * length.max(2.506628 * sigma.y);
                    // The shadow footprint ends where optical depth is 0.03,
                    // the widest edge that any pass draws.
                    let peak = segment.mass() / area.max(1e-12);
                    let radius = (2.0 * (peak / 0.03).max(1.0).ln()).sqrt().max(4.0);
                    let reach = Vec3::splat(radius * sigma.max_element());
                    (
                        low.min(segment.start().min(segment.end()) - reach),
                        high.max(segment.start().max(segment.end()) + reach),
                    )
                },
            );
            clusters.push(Cluster {
                first: order.len() as u32,
                count: chunk.len() as u32,
                center: (low + high) * 0.5,
                half_extents: (high - low) * 0.5,
            });
            // Store each cluster in bit-reversed curve order, so that any
            // prefix of it is an evenly spread subset for thinned shadows.
            let bits = (chunk.len() as u32).next_power_of_two().trailing_zeros();
            let mut spread_order: Vec<u32> = (0..chunk.len() as u32).collect();
            spread_order.sort_by_key(|&slot| slot.reverse_bits() >> (32 - bits.max(1)));
            order.extend(spread_order.iter().map(|&slot| chunk[slot as usize].1));
        }
    }
    (order, clusters.into())
}

impl GpuCloud {
    /// Appends the clusters of `segments` that can draw and returns their
    /// count. A cluster is dropped when it lies outside a frustum plane or
    /// outside the depth band where its detail level shows.
    #[expect(clippy::too_many_arguments, reason = "one draw's culling inputs")]
    fn cull_clusters(
        &self,
        segments: Range<u32>,
        frustum: Option<&Frustum>,
        cull_bands: bool,
        (world_from_local, clip_from_local): (Mat4, Mat4),
        bending: Vec3,
        uniform: &DrawUniform,
        keep: u32,
        visible: &mut Vec<u32>,
    ) -> u32 {
        let first = self.clusters.partition_point(|cluster| cluster.first < segments.start);
        let last = self.clusters.partition_point(|cluster| cluster.first < segments.end);
        let clusters = &self.clusters[first..last];
        // A thinned draw keeps the leading `keep` of every CLUSTER_SIZE.
        let pack = |cluster: &Cluster| {
            let count = (cluster.count * keep).div_ceil(CLUSTER_SIZE).max(1);
            (cluster.first << 7) | (count - 1)
        };
        // Local planes that the whole cloud does not already satisfy. An
        // infinite plane, such as a cascade's pushed-out near plane, never culls.
        let object = Aabb {
            center: self.bounds.center,
            half_extents: self.bounds.half_extents + Vec3A::from(bending),
        };
        let mut planes = [Vec4::ZERO; 6];
        let mut plane_count = 0;
        for half_space in frustum.map_or(&[][..], |frustum| &frustum.half_spaces[..]) {
            let world = half_space.normal_d();
            if !world.is_finite() {
                continue;
            }
            let local = world_from_local.transpose() * world;
            let normal = local.truncate();
            let center = Vec3::from(object.center);
            if normal.dot(center) + local.w - normal.abs().dot(object.half_extents.into()) < 0.0 {
                planes[plane_count] = local;
                plane_count += 1;
            }
        }
        let planes = &planes[..plane_count];
        let bands = if cull_bands { uniform.detail_ends[3] as usize } else { 0 };
        if planes.is_empty() && bands == 0 {
            visible.extend(clusters.iter().map(pack));
            return clusters.len() as u32;
        }
        let depth = clip_from_local.row(3);
        let start = visible.len();
        for cluster in clusters {
            let half = cluster.half_extents + bending;
            if planes.iter().any(|plane| {
                let normal = plane.truncate();
                normal.dot(cluster.center) + plane.w + normal.abs().dot(half) < 0.0
            }) {
                continue;
            }
            if bands > 0 {
                let center = depth.truncate().dot(cluster.center) + depth.w;
                let radius = depth.truncate().abs().dot(half);
                // Transitions before this cluster set its level within the draw.
                let level = uniform.detail_ends[..bands]
                    .iter()
                    .filter(|&&end| end <= cluster.first)
                    .count();
                // A level shows nearer than its own transition's far edge and
                // farther than the previous transition's near edge.
                if (level < bands && center - radius >= uniform.detail_depths[level][1])
                    || (level > 0 && center + radius <= uniform.detail_depths[level - 1][0])
                {
                    continue;
                }
            }
            visible.push(pack(cluster));
        }
        (visible.len() - start) as u32
    }

    // Drop complete levels whose bands are outside the depth interval of the
    // cloud. This also keeps distant draws at the cost of a single coarse level.
    fn select_segments(
        &self,
        clip_from_local: Mat4,
        viewport_height: f32,
        detail_scale: f32,
        bounds: Aabb,
        snap: bool,
        uniform: &mut DrawUniform,
    ) -> Range<u32> {
        let count = self.centers.len() as u32;
        if self.detail.is_empty() {
            return 0..count;
        }

        // A local reference length projects to scale / clip.w pixels. The Y row
        // accounts for viewport focal length and asset scale without depending
        // on segment orientation or its distance from the center of the screen.
        let scale =
            clip_from_local.row(1).truncate().length() * viewport_height * 0.5 / detail_scale;
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
            } else if farthest <= near || snap {
                // A snapped draw takes the finer level instead of fading
                // between two, which would draw both.
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
    shadows: ShadowBindings,
    quad_indices: Buffer,
}

fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
) {
    commands.insert_resource(SplatPipeline {
        shader: load_embedded_asset!(asset_server.as_ref(), "spaghetti.wgsl"),
        bindings: BindGroupLayoutDescriptor::new(
            "spaghetti bindings",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    uniform_buffer_sized(true, NonZeroU64::new(size_of::<DrawUniform>() as u64)),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        ),
        shadows: ShadowBindings::new(&device, &cache),
        // Both triangles use corner 1 as the provoking vertex for flat
        // attributes, so only that corner shades the segment.
        quad_indices: device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("spaghetti quad indices"),
            contents: bytemuck::cast_slice(&[1_u16, 2, 0, 1, 3, 2]),
            usage: BufferUsages::INDEX,
        }),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct PipelineKey {
    target: TextureFormat,
    samples: u32,
    cutout: bool,
    shadowed: bool,
    baked_occlusion: bool,
    additive: bool,
    // Depth-only cutouts for a directional shadow cascade.
    shadow: bool,
}

// Adds premultiplied color and leaves the target's alpha unchanged.
const ADDITIVE_BLENDING: BlendState = BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent {
        src_factor: BlendFactor::Zero,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
};

impl SpecializedRenderPipeline for SplatPipeline {
    type Key = PipelineKey;

    fn specialize(&self, key: PipelineKey) -> RenderPipelineDescriptor {
        if key.shadow {
            return RenderPipelineDescriptor {
                label: Some("spaghetti shadow cutouts".into()),
                layout: vec![self.bindings.clone(), self.shadows.layout.clone()],
                vertex: VertexState {
                    shader: self.shader.clone(),
                    entry_point: Some("cutout_vertex".into()),
                    constants: vec![("SHADOW_CASTER".into(), 1.0)],
                    ..default()
                },
                fragment: Some(FragmentState {
                    shader: self.shader.clone(),
                    entry_point: Some("cutout_shadow_fragment".into()),
                    targets: vec![],
                    ..default()
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    ..default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: CORE_3D_SHADOW_MAP_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(CompareFunction::GreaterEqual),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                ..default()
            };
        }
        RenderPipelineDescriptor {
            label: Some("spaghetti volume compositing".into()),
            layout: vec![self.bindings.clone(), self.shadows.layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some(if key.cutout { "cutout_vertex" } else { "vertex" }.into()),
                constants: vec![
                    ("SCENE_SHADOWS".into(), u32::from(key.shadowed) as f64),
                    ("BAKED_OCCLUSION".into(), u32::from(key.baked_occlusion) as f64),
                ],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some(if key.cutout { "cutout_fragment" } else { "fragment" }.into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target,
                    blend: (!key.cutout).then_some(if key.additive {
                        ADDITIVE_BLENDING
                    } else {
                        BlendState::PREMULTIPLIED_ALPHA_BLENDING
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
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

/// Detail scale of shadow casters relative to the main view. Coarse segments
/// are wide and cover many texels each, so a finer level costs less to draw
/// and stays closer to the full asset's shade.
const SHADOW_DETAIL: f32 = 3.0;
// Segments drawn per cluster of CLUSTER_SIZE in each directional cascade.
// Far cascades have coarse texels where a spread subset with stronger
// blocking casts the same average shade.
const SHADOW_KEEP: [u32; 4] = [128, 48, 32, 32];

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

struct ExtractedPointLight {
    position: Vec3,
    light: SpaghettiPointLight,
    layers: RenderLayers,
}

impl ExtractedPointLight {
    /// The light at `distance_squared` from the source, before the facing
    /// term. The shader evaluates the same expression.
    fn strength(&self, distance_squared: f32) -> f32 {
        let fraction = distance_squared / (self.light.range * self.light.range);
        let window = (1.0 - fraction * fraction).max(0.0);
        window * window / (distance_squared + self.light.radius * self.light.radius)
    }
}

#[derive(Resource, Default)]
struct PointLights(Vec<ExtractedPointLight>);

type PointLightSource = (
    &'static SpaghettiPointLight,
    &'static GlobalTransform,
    &'static InheritedVisibility,
    Option<&'static RenderLayers>,
);

fn extract_point_lights(
    mut extracted: ResMut<PointLights>,
    lights: Extract<Query<PointLightSource>>,
) {
    extracted.0.clear();
    extracted.0.extend(
        lights
            .iter()
            .filter(|(light, transform, visibility, _)| {
                visibility.get()
                    && light.range > 0.0
                    && light.range.is_finite()
                    && light.radius > 0.0
                    && light.color.is_finite()
                    && light.color.max_element() > 0.0
                    && transform.translation().is_finite()
            })
            .map(|(light, transform, _, layers)| ExtractedPointLight {
                position: transform.translation(),
                light: *light,
                layers: layers.cloned().unwrap_or_default(),
            }),
    );
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
    ground_ambient_delta: [f32; 4],
    eye_relative_from_local: [[f32; 4]; 4],
    fog: FogUniform,
    world_from_local: [[f32; 4]; 4],
    local_from_world: [[f32; 4]; 3],
    // Pusher position and radius for procedural wind, or local sway vector.
    pusher: [f32; 4],
    // Time, wind mode, range coverage, reverse range dither.
    motion: [f32; 4],
    // Near and far clip.w for each active transition.
    detail_depths: [[f32; 4]; MAX_DETAIL_TRANSITIONS],
    // Exclusive finer-level end indices, followed by active transition count.
    detail_ends: [u32; 4],
    eye_world: [f32; 4],
    // Specular strength, exponent, strand fraction, additive flag.
    material: [f32; 4],
    // Local travel over one loop, then elapsed loops.
    flow_span: [f32; 4],
    // Local turbulence reach, flicker depth, spread, growth.
    flow_shape: [f32; 4],
    // Fade-in and fade-out fractions, time, enabled flag.
    flow_fade: [f32; 4],
    // Color factor at the end of a loop.
    flow_tint: [f32; 4],
    // World position and range of the local light. A zero range turns it off.
    lamp: [f32; 4],
    // Color at one unit of distance, then the squared source radius.
    lamp_color: [f32; 4],
    // Offset of this draw's visible clusters in the frame's cluster list.
    clusters: [u32; 4],
    // The pixel that holds the world origin, in whole pixels. Shadow views
    // key their random patterns to pixels relative to it.
    pixel_origin: [f32; 4],
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
    // The pipeline of this draw's entry in a retained binned phase.
    cutout_pipeline: Option<CachedRenderPipelineId>,
    // The entry is in a shadow phase rather than the view's cutout phase.
    shadow: bool,
    source: Buffer,
    occlusion: Option<Buffer>,
    order: Option<Buffer>,
    // Built after queueing, once the frame's uniform buffer is large enough.
    // The IDs are the uniform buffer and the buffer at binding 2.
    bindings: Option<(BufferId, BufferId, BindGroup)>,
    // Byte offset of this frame's `DrawUniform` in the shared uniform buffer.
    uniform_offset: u32,
    shadows: BindGroup,
    direction: Vec4,
    revision: u64,
    segments: Range<u32>,
    // Instances to draw. A cutout draws `CLUSTER_SIZE` per visible cluster.
    instances: Range<u32>,
    idle_frames: u8,
}

#[derive(Resource, Default)]
struct ViewDraws {
    draws: HashMap<(RetainedViewEntity, Entity), CloudDraw>,
    // Every draw's uniform for this frame, at a dynamic offset. One upload
    // replaces a staging buffer and a copy for each draw.
    uniforms: Option<Buffer>,
    staging: Vec<u8>,
    // Packed visible clusters of every cutout draw in this frame. Each entry
    // is the cluster's first segment shifted left by seven bits, plus its
    // count minus one.
    clusters: Vec<u32>,
    cluster_buffer: Option<Buffer>,
}

type SplatView = (
    Entity,
    &'static ExtractedView,
    Option<&'static Msaa>,
    Option<&'static Frustum>,
    Option<&'static RenderLayers>,
    Option<&'static TemporalJitter>,
    Option<&'static MainPassResolutionOverride>,
    Option<ShadowViewData>,
    Option<&'static LightEntity>,
);

/// The retained binned phase that holds a view's depth-writing cutouts.
enum CutoutPhase<'a> {
    Main(&'a mut BinnedRenderPhase<AlphaMask3d>, DrawFunctionId),
    Shadow(&'a mut BinnedRenderPhase<Shadow>, DrawFunctionId),
}

impl CutoutPhase<'_> {
    fn add(
        &mut self,
        pipeline: CachedRenderPipelineId,
        asset: AssetId<SpaghettiAsset>,
        entity: (Entity, MainEntity),
    ) {
        match self {
            Self::Main(phase, draw_function) => phase.add(
                OpaqueNoLightmap3dBatchSetKey {
                    pipeline,
                    draw_function: *draw_function,
                    material_bind_group_index: None,
                    slabs: default(),
                },
                OpaqueNoLightmap3dBinKey { asset_id: asset.untyped() },
                entity,
                default(),
                BinnedRenderPhaseType::NonMesh,
            ),
            Self::Shadow(phase, draw_function) => phase.add(
                ShadowBatchSetKey {
                    pipeline,
                    draw_function: *draw_function,
                    material_bind_group_index: None,
                    slabs: default(),
                },
                ShadowBinKey { asset_id: asset.untyped() },
                entity,
                default(),
                BinnedRenderPhaseType::NonMesh,
            ),
        }
    }

    fn remove(&mut self, entity: MainEntity) {
        match self {
            Self::Main(phase, _) => phase.remove(entity),
            Self::Shadow(phase, _) => phase.remove(entity),
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "Bevy system parameters")]
fn queue_clouds(
    mut draws: ResMut<ViewDraws>,
    gpu_clouds: Res<RenderAssets<GpuCloud>>,
    clouds: Query<(Entity, &MainEntity, &ExtractedCloud)>,
    views: Query<SplatView>,
    (view_fog, point_lights): (Res<ViewFog>, Res<PointLights>),
    mut phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    (mut cutout_phases, cutout_draw_functions, mut shadow_phases, shadow_draw_functions): (
        ResMut<ViewBinnedRenderPhases<AlphaMask3d>>,
        Res<DrawFunctions<AlphaMask3d>>,
        ResMut<ViewBinnedRenderPhases<Shadow>>,
        Res<DrawFunctions<Shadow>>,
    ),
    pipeline: Res<SplatPipeline>,
    cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SplatPipeline>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    (mut depths, mut order): (Local<Vec<(f32, u32)>>, Local<Vec<u32>>),
    (cascade_views, sun_lights, shadowless): (
        Query<&LightEntity>,
        Query<&ExtractedDirectionalLight>,
        Query<(), With<NoSpaghettiShadows>>,
    ),
) {
    let draws = draws.as_mut();
    for draw in draws.draws.values_mut() {
        draw.idle_frames += 1;
    }
    let alignment = device.limits().min_uniform_buffer_offset_alignment as usize;
    let stride = size_of::<DrawUniform>().next_multiple_of(alignment);
    draws.staging.clear();
    draws.clusters.clear();

    let draw_function = draw_functions.read().id::<(SetItemPipeline, DrawCloud)>();
    let cutout_draw_function = cutout_draw_functions.read().id::<(SetItemPipeline, DrawCloud)>();
    let shadow_draw_function = shadow_draw_functions.read().id::<(SetItemPipeline, DrawCloud)>();
    let bind_group_layout = cache.get_bind_group_layout(&pipeline.bindings);
    let default_layers = RenderLayers::default();
    // Each directional cascade measures visibility ranges from its camera.
    let eyes: HashMap<MainEntity, Vec3> = views
        .iter()
        .filter(|view| view.8.is_none())
        .map(|view| (view.1.retained_view_entity.main_entity, view.1.world_from_view.translation()))
        .collect();
    for (view_entity, view, msaa, frustum, layers, jitter, resolution, shadow_data, light) in &views {
        let shadow_view = match light {
            Some(&LightEntity::Directional { light_entity, .. }) => {
                if shadowless.contains(light_entity) {
                    continue;
                }
                true
            }
            _ => false,
        };
        let (mut phase, mut cutout_phase, eye) = if shadow_view {
            let Some(phase) = shadow_phases.get_mut(&view.retained_view_entity) else {
                continue;
            };
            let Some(&eye) = view.retained_view_entity.auxiliary_entity.and_then(|camera| eyes.get(&camera)) else {
                continue;
            };
            (None, Some(CutoutPhase::Shadow(phase, shadow_draw_function)), eye)
        } else {
            let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
                continue;
            };
            let cutout_phase = cutout_phases
                .get_mut(&view.retained_view_entity)
                .map(|phase| CutoutPhase::Main(phase, cutout_draw_function));
            (Some(phase), cutout_phase, view.world_from_view.translation())
        };
        let layers = layers.unwrap_or(&default_layers);
        let fog = view_fog.0.get(&view.retained_view_entity.main_entity).filter(|_| !shadow_view);
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
        // Bevy moves a cascade in whole texels and keeps its size, so texels
        // counted from the world origin stay on the same ground as the camera
        // walks. Main views keep plain screen pixels.
        let pixel_origin = if shadow_view {
            let origin = clip_from_world.w_axis;
            let origin = Vec2::new(origin.x + 1.0, 1.0 - origin.y) * 0.5 * size.as_vec2();
            origin.round().extend(0.0).extend(0.0).to_array()
        } else {
            [0.0; 4]
        };
        let mut pipeline_ids = [None; 17];
        let eye_world = view.world_from_view.translation();
        let mut view_shadows = None;

        for (entity, main_entity, cloud) in &clouds {
            if !layers.intersects(&cloud.layers) || (shadow_view && !cloud.cast_shadows) {
                continue;
            }
            let cutout = cloud.cutout || shadow_view;
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
                let distance = eye.distance(center);
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
            if coverage <= 0.0 || (cutout && cutout_phase.is_none()) {
                continue;
            }
            let inverse_basis = (cloud.sway.is_some() || cloud.wind.is_some() || cloud.flow.is_some())
                .then(|| Mat3::from_mat4(cloud.world_from_local).inverse())
                .filter(|inverse| inverse.is_finite());
            let sway = cloud
                .sway
                .zip(inverse_basis)
                .map(|(slope, inverse)| inverse * slope)
                .filter(|sway| sway.is_finite());
            let mut bounds = gpu.bounds;
            if let Some(sway) = sway {
                bounds.half_extents += Vec3A::from(sway.abs() * gpu.max_bend_height);
            } else if let (Some(_), Some(inverse)) = (cloud.wind, inverse_basis) {
                let rows = inverse.transpose();
                let stretch = Vec3::new(rows.x_axis.length(), rows.y_axis.length(), rows.z_axis.length());
                // Maximum displacement combines horizontal reach and vertical shortening.
                bounds.half_extents += Vec3A::from(stretch * (gpu.max_bend_height * 1.11));
            }
            // Local travel per loop and local turbulence reach.
            let flow = cloud.flow.zip(inverse_basis).map(|(flow, inverse)| {
                let rows = inverse.transpose();
                let stretch = Vec3::new(rows.x_axis.length(), rows.y_axis.length(), rows.z_axis.length());
                let span = inverse * (flow.velocity * flow.period);
                let reach = stretch * flow.turbulence.abs();
                let spread = Vec3::from(bounds.half_extents + bounds.center.abs()) * flow.spread.abs();
                bounds.half_extents += Vec3A::from(span.abs() + reach + spread);
                (flow, span, reach.max_element())
            });
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
                world_from_local: cloud.world_from_local.to_cols_array_2d(),
                eye_world: eye_world.extend(0.0).to_array(),
                pixel_origin,
                ..DrawUniform::zeroed()
            };
            uniform.material[3] = f32::from(u8::from(cloud.additive));
            if let Some((flow, span, reach)) = flow {
                uniform.flow_span = span.extend(flow.time / flow.period).to_array();
                uniform.flow_shape = [reach, flow.flicker.clamp(0.0, 1.0), flow.spread, flow.growth];
                let [fade_in, fade_out] = flow.fade.map(|fade| fade.clamp(0.0, 1.0));
                uniform.flow_fade = [fade_in, fade_out, flow.time, 1.0];
                uniform.flow_tint = flow.tint.max(Vec3::ZERO).extend(0.0).to_array();
            }
            if let Some(fade) = cloud.near_fade.filter(|fade| fade.end > fade.start && !shadow_view) {
                uniform.viewport[2] = fade.start;
                uniform.viewport[3] = fade.end;
            }
            if let Some(sway) = sway {
                uniform.motion[1] = 2.0;
                uniform.pusher = sway.extend(0.0).to_array();
            } else if let (Some(wind), Some(inverse)) = (cloud.wind, inverse_basis) {
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
            let mut shadows = &pipeline.shadows.fallback;
            let mut shadowed = false;
            if let Some(lighting) = cloud.lighting.filter(|_| !shadow_view) {
                uniform.set_lighting(lighting, cloud.world_from_local);
                uniform.ambient[3] = cloud.transmission;
                if let Some(specular) = cloud.specular {
                    uniform.material[0] = specular.strength;
                    uniform.material[1] = specular.shininess.max(1.0);
                    uniform.material[2] = specular.strand.clamp(0.0, 1.0);
                }
                if let Some(ground) = cloud.ground_ambient {
                    uniform.ground_ambient_delta =
                        (ground.0 - lighting.ambient).extend(0.0).to_array();
                }
                if !point_lights.0.is_empty() {
                    // A world-aligned box around the cloud.
                    let transform = cloud.world_from_local;
                    let center = transform.transform_point3(bounds.center.into());
                    let half = bounds.half_extents;
                    let extent = transform.x_axis.truncate().abs() * half.x
                        + transform.y_axis.truncate().abs() * half.y
                        + transform.z_axis.truncate().abs() * half.z;
                    let lamp = point_lights.0.iter()
                        .filter(|lamp| lamp.layers.intersects(&cloud.layers))
                        .map(|lamp| {
                            let outside = ((lamp.position - center).abs() - extent).max(Vec3::ZERO);
                            let strength = lamp.strength(outside.length_squared());
                            (lamp, strength * lamp.light.color.max_element())
                        })
                        .filter(|&(_, strength)| strength > 0.0)
                        .max_by(|a, b| a.1.total_cmp(&b.1));
                    if let Some((lamp, _)) = lamp {
                        let radius_squared = lamp.light.radius * lamp.light.radius;
                        uniform.lamp = lamp.position.extend(lamp.light.range).to_array();
                        uniform.lamp_color = lamp.light.color.extend(radius_squared).to_array();
                    }
                }
                if cloud.receive_shadows && let Some(data) = shadow_data {
                    let view_shadows = view_shadows.get_or_insert_with(|| {
                        pipeline.shadows.for_view(
                            &device,
                            &cache,
                            view_entity,
                            data,
                            -view_from_world.row(2),
                            &cascade_views,
                            &sun_lights,
                        )
                    });
                    let direction = lighting.direction_to_light.normalize_or_zero();
                    if let Some((_, binding)) = view_shadows.iter()
                        .filter(|(sun, _)| sun.dot(direction) > 0.999)
                        .max_by(|(a, _), (b, _)| a.dot(direction).total_cmp(&b.dot(direction)))
                    {
                        shadows = binding;
                        shadowed = true;
                    }
                }
            }
            if let Some(fog) = fog {
                let mut eye_relative = cloud.world_from_local;
                eye_relative.w_axis -= view.world_from_view.translation().extend(0.0);
                uniform.eye_relative_from_local = eye_relative.to_cols_array_2d();
                uniform.fog = *fog;
            }
            let key = (view.retained_view_entity, entity);
            let segments = gpu.select_segments(
                clip_from_local,
                size.y as f32,
                cloud.detail_scale * if shadow_view { SHADOW_DETAIL } else { 1.0 },
                bounds,
                // A cascade is orthographic, so every segment sits at one
                // detail depth and a fade would draw two whole levels.
                shadow_view,
                &mut uniform,
            );
            let mut instances = 0..segments.len() as u32;
            if cutout {
                // Flow moves segments out of their clusters' boxes.
                let cull = !cloud.disable_culling && cloud.flow.is_none();
                // Far cascades draw a spread subset of each cluster, and the
                // shader raises the kept segments' blocking to match.
                let keep = match light {
                    Some(&LightEntity::Directional { cascade_index, .. }) => {
                        SHADOW_KEEP[cascade_index.min(SHADOW_KEEP.len() - 1)]
                    }
                    _ => CLUSTER_SIZE,
                };
                uniform.clusters[0] = draws.clusters.len() as u32;
                uniform.clusters[1] = keep;
                uniform.clusters[2] = (keep as f32 / CLUSTER_SIZE as f32).to_bits();
                let visible = gpu.cull_clusters(
                    segments.clone(),
                    cull.then_some(()).and(frustum),
                    cull,
                    (cloud.world_from_local, clip_from_local),
                    Vec3::from(bounds.half_extents - gpu.bounds.half_extents),
                    &uniform,
                    keep,
                    &mut draws.clusters,
                );
                instances = 0..visible * keep;
            }
            let new_draw = || make_draw(&device, source, gpu, *main_entity, cutout, shadow_view, shadows);
            let draw = draws.draws.entry(key).or_insert_with(new_draw);
            if draw.source.id() != source.id()
                || draw.order.is_none() != cutout
                || draw.occlusion.as_ref().map(Buffer::id) != gpu.occlusion.as_ref().map(Buffer::id)
            {
                if draw.cutout_pipeline.is_some() && let Some(phase) = cutout_phase.as_mut() {
                    phase.remove(draw.main_entity);
                }
                *draw = new_draw();
            }
            draw.shadows = shadows.clone();
            if let Some(order_buffer) = &draw.order {
                let direction = (view_from_world * cloud.world_from_local)
                    .row(2)
                    .truncate()
                    .normalize_or_zero();
                let direction = direction.extend(sway.map_or(0.0, |sway| sway.dot(direction)));
                if draw.direction != direction
                    || draw.revision != gpu.revision
                    || draw.segments != segments
                {
                    // Translation preserves relative depths. Re-sort only if the
                    // view direction, sway depth, asset data, or submitted levels changed.
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
            draw.instances = instances;
            draw.uniform_offset = draws.staging.len() as u32;
            draws.staging.extend_from_slice(bytemuck::bytes_of(&uniform));
            draws.staging.resize(draw.uniform_offset as usize + stride, 0);
            draw.idle_frames = 0;
            let baked_occlusion = gpu.occlusion.is_some() && cloud.lighting.is_some() && !shadow_view;
            let variant = if shadow_view {
                16
            } else {
                usize::from(cloud.cutout)
                    + 2 * usize::from(shadowed)
                    + 4 * usize::from(baked_occlusion)
                    + 8 * usize::from(cloud.additive)
            };
            let pipeline_id = *pipeline_ids[variant].get_or_insert_with(|| {
                pipelines.specialize(
                    &cache,
                    &pipeline,
                    PipelineKey {
                        target: view.target_format,
                        samples: msaa.map_or(1, Msaa::samples),
                        cutout,
                        shadowed,
                        baked_occlusion,
                        additive: cloud.additive && !shadow_view,
                        shadow: shadow_view,
                    },
                )
            });
            if let Some(phase) = cutout_phase.as_mut().filter(|_| cutout) {
                if draw.cutout_pipeline != Some(pipeline_id) {
                    if draw.cutout_pipeline.replace(pipeline_id).is_some() {
                        phase.remove(*main_entity);
                    }
                    phase.add(pipeline_id, cloud.asset, (entity, *main_entity));
                }
                continue;
            }
            let Some(phase) = phase.as_deref_mut() else {
                continue;
            };
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
                indexed: true,
            });
        }
    }
    if !draws.staging.is_empty() {
        let size = draws.staging.len() as u64;
        if draws.uniforms.as_ref().is_none_or(|buffer| buffer.size() < size) {
            draws.uniforms = Some(device.create_buffer(&BufferDescriptor {
                label: Some("spaghetti draw uniforms"),
                size: size.next_power_of_two(),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let uniforms = draws.uniforms.as_ref().unwrap();
        queue.write_buffer(uniforms, 0, &draws.staging);
        let uniform = BufferBinding {
            buffer: uniforms,
            offset: 0,
            size: NonZeroU64::new(size_of::<DrawUniform>() as u64),
        };
        let cluster_size = (draws.clusters.len() * size_of::<u32>()).max(256) as u64;
        if draws.cluster_buffer.as_ref().is_none_or(|buffer| buffer.size() < cluster_size) {
            draws.cluster_buffer = Some(device.create_buffer(&BufferDescriptor {
                label: Some("spaghetti visible clusters"),
                size: cluster_size.next_power_of_two(),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let clusters = draws.cluster_buffer.as_ref().unwrap();
        if !draws.clusters.is_empty() {
            queue.write_buffer(clusters, 0, bytemuck::cast_slice(&draws.clusters));
        }
        for draw in draws.draws.values_mut().filter(|draw| draw.idle_frames == 0) {
            // Cutouts read their cluster list where sorted draws read the order.
            let listed = draw.order.as_ref().unwrap_or(clusters);
            if draw.bindings.as_ref().is_some_and(|(uniform_id, listed_id, _)| {
                *uniform_id == uniforms.id() && *listed_id == listed.id()
            }) {
                continue;
            }
            let bindings = device.create_bind_group(
                "spaghetti draw bindings",
                &bind_group_layout,
                &BindGroupEntries::sequential((
                    uniform.clone(),
                    draw.source.as_entire_binding(),
                    listed.as_entire_binding(),
                    draw.occlusion.as_ref().unwrap_or(&draw.source).as_entire_binding(),
                )),
            );
            draw.bindings = Some((uniforms.id(), listed.id(), bindings));
        }
    }
    // Binned phases retain draws across frames. Remove entries that fell out
    // of range while keeping their buffers briefly for throttled views.
    draws.draws.retain(|&(view, _), draw| {
        if draw.idle_frames > 0 && draw.cutout_pipeline.take().is_some() {
            if draw.shadow {
                if let Some(phase) = shadow_phases.get_mut(&view) {
                    phase.remove(draw.main_entity);
                }
            } else if let Some(phase) = cutout_phases.get_mut(&view) {
                phase.remove(draw.main_entity);
            }
        }
        draw.idle_frames <= 1
    });
}

fn make_draw(
    device: &RenderDevice,
    source: &Buffer,
    cloud: &GpuCloud,
    main_entity: MainEntity,
    cutout: bool,
    shadow: bool,
    shadows: &BindGroup,
) -> CloudDraw {
    let count = cloud.centers.len();
    let order = (!cutout).then(|| {
        device.create_buffer(&BufferDescriptor {
            label: Some("spaghetti depth order"),
            size: (count * size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });
    CloudDraw {
        main_entity,
        cutout_pipeline: None,
        shadow,
        source: source.clone(),
        occlusion: cloud.occlusion.clone(),
        order,
        bindings: None,
        uniform_offset: 0,
        shadows: shadows.clone(),
        direction: Vec4::NAN,
        revision: 0,
        segments: 0..count as u32,
        instances: 0..0,
        idle_frames: 0,
    }
}

struct DrawCloud;

impl<P: PhaseItem> RenderCommand<P> for DrawCloud {
    type Param = (SRes<ViewDraws>, SRes<SplatPipeline>);
    type ViewQuery = Read<ExtractedView>;
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        view: &'w ExtractedView,
        _: Option<()>,
        (draws, pipeline): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let key = (view.retained_view_entity, item.entity());
        let Some(draw) = draws.into_inner().draws.get(&key) else {
            return RenderCommandResult::Skip;
        };
        let Some((_, _, bindings)) = &draw.bindings else {
            return RenderCommandResult::Skip;
        };
        if draw.instances.is_empty() {
            return RenderCommandResult::Success;
        }
        pass.set_bind_group(0, bindings, &[draw.uniform_offset]);
        pass.set_bind_group(1, &draw.shadows, &[]);
        let instances = draw.instances.clone();
        pass.set_index_buffer(
            pipeline.into_inner().quad_indices.slice(..),
            IndexFormat::Uint16,
        );
        pass.draw_indexed(0..6, 0, instances);
        RenderCommandResult::Success
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clusters_permute_each_level_and_bound_their_segments() {
        let segment = |i: u32| {
            let at = Vec3::new((i * 37 % 101) as f32, (i * 11 % 53) as f32, (i % 7) as f32) * 0.1;
            GaussianSegment::new(at, at + Vec3::X * 0.05, Vec3::splat(0.01), Quat::IDENTITY, 0.5, Vec3::ONE)
        };
        let detail = [DetailTransition { finer_end: 300, pixels_per_unit: [10.0, 20.0] }];
        let asset = SpaghettiAsset::with_detail((0..420).map(segment).collect(), &detail).unwrap();
        let (order, clusters) = cluster_segments(&asset);

        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..420).collect::<Vec<_>>());
        assert!(order[..300].iter().all(|&index| index < 300));

        let mut next = 0;
        for cluster in &clusters {
            assert_eq!(cluster.first, next);
            assert!((1..=CLUSTER_SIZE).contains(&cluster.count));
            let range = cluster.first..cluster.first + cluster.count;
            assert!(range.end <= 300 || range.start >= 300, "a cluster spans two levels");
            for &index in &order[range.start as usize..range.end as usize] {
                let segment = &asset.segments()[index as usize];
                for point in [segment.start(), segment.end()] {
                    let offset = (point - cluster.center).abs();
                    assert!(offset.cmple(cluster.half_extents).all());
                }
            }
            next = range.end;
        }
        assert_eq!(next, 420);
    }
}
