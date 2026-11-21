use std::{num::NonZeroU64, ops::Range};

use bevy::{
    asset::{AssetLoader, LoadContext, embedded_asset, io::Reader, load_embedded_asset},
    platform::collections::{HashMap, HashSet},
    camera::{
        Exposure, MainPassResolutionOverride,
        primitives::{Aabb, Frustum},
        visibility::{
            InheritedVisibility, NoFrustumCulling, RenderLayers, ViewVisibility, VisibilityRange,
        },
    },
    color::ColorToComponents,
    core_pipeline::core_3d::{
        CORE_3D_DEPTH_FORMAT, Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey, Transparent3d,
        TransparentSortingInfo3d,
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
use shadow::{CachedShadowBinding, ShadowBindings, ShadowViewData};

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
/// The shadow pass runs depth-only without a fragment shader: each segment
/// draws one area-matched triangle at a deterministic stochastic radius whose
/// coverage probability at normalized distance `d` equals `1 - exp(-τ(d))`.
/// Overlapping segments multiply their transmittance and shadow filtering
/// averages texels into partial shade, while hardware early-Z and TBDR HSR
/// reject occluded geometry without fragment-stage discard.
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
            ))
            .add_observer(reprepare_on_flag::<SpaghettiCutout>)
            .add_observer(reprepare_on_flag::<SpaghettiCastShadows>);
        embedded_asset!(app, "spaghetti.wgsl");
        embedded_asset!(app, "shadow.wgsl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Transparent3d, (SetItemPipeline, DrawCloud)>()
                .add_render_command::<Opaque3d, (SetItemPipeline, DrawCloud)>()
                .add_render_command::<Shadow, (SetItemPipeline, DrawCloud)>()
                .init_resource::<SpecializedRenderPipelines<SplatPipeline>>()
                .init_resource::<ViewDraws>()
                .init_resource::<ViewFog>()
                .init_resource::<PointLights>()
                .init_resource::<CloudAssetUsage>()
                .add_systems(
                    ExtractSchedule,
                    (extract_view_fog, extract_point_lights, extract_cloud_asset_usage),
                )
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

#[derive(Resource, Default)]
struct CloudAssetUsage {
    clustered: HashSet<AssetId<SpaghettiAsset>>,
    shadowed: HashSet<AssetId<SpaghettiAsset>>,
}

fn extract_cloud_asset_usage(
    mut usage: ResMut<CloudAssetUsage>,
    clouds: Extract<Query<(&Spaghetti3d, Has<SpaghettiCutout>, Has<SpaghettiCastShadows>)>>,
) {
    usage.clustered.clear();
    usage.shadowed.clear();
    for (cloud, cutout, cast_shadows) in &clouds {
        let id = cloud.0.id();
        if cutout || cast_shadows {
            usage.clustered.insert(id);
        }
        if cast_shadows {
            usage.shadowed.insert(id);
        }
    }
}

fn reprepare_on_flag<C: Component>(
    trigger: On<Add<C>>,
    clouds: Query<&Spaghetti3d>,
    mut assets: ResMut<Assets<SpaghettiAsset>>,
) {
    if let Ok(cloud) = clouds.get(trigger.entity) {
        let _ = assets.get_mut(&cloud.0);
    }
}

#[derive(Component)]
pub struct ExtractedCloud {
    asset: AssetId<SpaghettiAsset>,
    world_from_local: Mat4,
    inverse_basis: Option<Mat3>,
    local_sway: Option<Vec3>,
    layers: RenderLayers,
    disable_culling: bool,
    lighting: Option<SpaghettiLighting>,
    ground_ambient: Option<SpaghettiGroundAmbient>,
    receive_shadows: bool,
    transmission: f32,
    cutout: bool,
    wind: Option<SpaghettiWind>,
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
        visibility.get().then(|| {
            let world_from_local = transform.to_matrix();
            let wind = wind.copied();
            let sway = sway.map(|sway| sway.0).filter(|sway| sway.is_finite());
            let flow = flow.copied().filter(|flow| {
                !cutout
                    && flow.period > 0.0
                    && flow.period.is_finite()
                    && flow.time.is_finite()
                    && flow.velocity.is_finite()
            });
            let inverse_basis = (sway.is_some() || wind.is_some() || flow.is_some())
                .then(|| Mat3::from_mat4(world_from_local).inverse())
                .filter(|inverse| inverse.is_finite());
            let local_sway = sway
                .zip(inverse_basis)
                .map(|(slope, inverse)| inverse * slope)
                .filter(|sway| sway.is_finite());
            ExtractedCloud {
                asset: cloud.0.id(),
                world_from_local,
                inverse_basis,
                local_sway,
                layers: layers.cloned().unwrap_or_default(),
                disable_culling,
                lighting: lighting.copied(),
                ground_ambient: ground_ambient.copied(),
                receive_shadows,
                transmission: transmission.map_or(0.0, |t| t.0.max(0.0)),
                cutout,
                wind,
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
                flow,
                cast_shadows,
            }
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

/// Compact 32-byte per-segment attributes for the depth-only shadow pass.
///
/// Packing quantized AABB-space endpoints, stochastic squared radius base, and
/// normalized symmetric 3x3 covariance cuts shadow vertex bandwidth 2.5x and
/// removes quaternion rotations, logarithms, and distance roots from the GPU.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowSegment {
    /// unorm16 endpoints in `[bounds.min, bounds.max]` across xyz, f16x2 `(r2, sigma_max)` in w.
    geom: [u32; 4],
    /// snorm8x4 `(s00, s01, s02, s11)`, snorm8x4 `(s12, s22, taper, 0)`,
    /// f16x2 `(bend_start, bend_end)`, f16x2 `(blade_height, 0)`.
    attr: [u32; 4],
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | u16::from(mant != 0);
    }
    let half_exp = exp - 112;
    if half_exp >= 31 {
        return sign | 0x7bff;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let shifted = (mant | 0x0080_0000) >> (13 - half_exp);
        return sign | ((shifted + 1) >> 1) as u16;
    }
    let rounded = mant + 0x1000;
    if rounded >= 0x0080_0000 {
        let next_exp = half_exp + 1;
        return sign | if next_exp >= 31 { 0x7bff } else { (next_exp as u16) << 10 };
    }
    sign | ((half_exp as u16) << 10) | ((rounded >> 13) as u16)
}

fn pack_f16x2(a: f32, b: f32) -> u32 {
    u32::from(f32_to_f16_bits(a)) | (u32::from(f32_to_f16_bits(b)) << 16)
}

fn pack_unorm16x2(a: f32, b: f32) -> u32 {
    let pack = |v: f32| (v.clamp(0.0, 1.0) * 65535.0).round() as u16 as u32;
    pack(a) | (pack(b) << 16)
}

fn pack_snorm8x4(a: f32, b: f32, c: f32, d: f32) -> u32 {
    let pack = |v: f32| (v.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8 as u32;
    pack(a) | (pack(b) << 8) | (pack(c) << 16) | (pack(d) << 24)
}

fn hash_phase(mut value: u32) -> f64 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    f64::from(value >> 8) * (1.0 / 16_777_216.0)
}

fn build_shadow_segments(
    sorted: &[GaussianSegment],
    clusters: &[Cluster],
    min: Vec3,
    max: Vec3,
) -> Vec<ShadowSegment> {
    let extent = (max - min).max(Vec3::splat(1e-6));
    let inv_extent = 1.0 / extent;
    let inv_asset_scale = 1.0 / extent.max_element();
    let mut out = Vec::with_capacity(sorted.len());
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        let phase = hash_phase(cluster_index as u32);
        let base = cluster.first as usize;
        for slot in 0..cluster.count as usize {
            let segment = &sorted[base + slot];
            // Stratified golden-ratio sequence per cluster keeps every thinned
            // prefix evenly distributed in (0, 1) with lower variance than white noise.
            let u = ((phase + (slot as f64 + 0.5) * 0.618_033_988_749_894_9).fract())
                .clamp(1e-6, 1.0 - 1e-6) as f32;
            let start = segment.start();
            let end = segment.end();
            let sigma = segment.sigma();
            let area = 2.506_628 * sigma.x * start.distance(end).max(2.506_628 * sigma.y);
            let peak = segment.mass() / area.max(1e-12);
            let r2 = if peak > 0.0 {
                2.0 * (peak / -(1.0 - u).ln()).ln()
            } else {
                -65000.0
            };
            let sigma_max = sigma.max_element().max(1e-12);
            let s = sigma / sigma_max;
            let rot = Mat3::from_quat(segment.rotation());
            let r0 = rot.x_axis;
            let r1 = rot.y_axis;
            let r2_axis = rot.z_axis;
            let s0 = s.x * s.x;
            let s1 = s.y * s.y;
            let s2 = s.z * s.z;
            let s00 = s0 * r0.x * r0.x + s1 * r1.x * r1.x + s2 * r2_axis.x * r2_axis.x;
            let s01 = s0 * r0.x * r0.y + s1 * r1.x * r1.y + s2 * r2_axis.x * r2_axis.y;
            let s02 = s0 * r0.x * r0.z + s1 * r1.x * r1.z + s2 * r2_axis.x * r2_axis.z;
            let s11 = s0 * r0.y * r0.y + s1 * r1.y * r1.y + s2 * r2_axis.y * r2_axis.y;
            let s12 = s0 * r0.y * r0.z + s1 * r1.y * r1.z + s2 * r2_axis.y * r2_axis.z;
            let s22 = s0 * r0.z * r0.z + s1 * r1.z * r1.z + s2 * r2_axis.z * r2_axis.z;
            let [bend_start, bend_end] = segment.bend_heights();
            let profile_bits = segment.profile_bits();
            let (blade_height, taper) = if (profile_bits & 0xff80_0000) == 0x3f80_0000 {
                (
                    ((profile_bits >> 10) & 8191) as f32 * 0.001,
                    (profile_bits & 1023) as f32 / 1023.0,
                )
            } else {
                (0.0, 1.0)
            };
            let ua = (start - min) * inv_extent;
            let ub = (end - min) * inv_extent;
            out.push(ShadowSegment {
                geom: [
                    pack_unorm16x2(ua.x, ua.y),
                    pack_unorm16x2(ua.z, ub.x),
                    pack_unorm16x2(ub.y, ub.z),
                    pack_f16x2(r2, sigma_max * inv_asset_scale),
                ],
                attr: [
                    pack_snorm8x4(s00, s01, s02, s11),
                    pack_snorm8x4(s12, s22, taper, 0.0),
                    pack_f16x2(bend_start, bend_end),
                    pack_f16x2(blade_height, 0.0),
                ],
            });
        }
    }
    out
}

struct GpuCloud {
    segments: Option<Buffer>,
    shadow_segments: Option<Buffer>,
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
    type Param = (SRes<RenderDevice>, SRes<RenderQueue>, SRes<CloudAssetUsage>);

    fn byte_len(asset: &Self::SourceAsset) -> Option<usize> {
        Some(
            std::mem::size_of_val(asset.segments())
                + asset.segments().len() * size_of::<ShadowSegment>()
                + asset.occlusion().len() * size_of::<u32>(),
        )
    }

    fn prepare_asset(
        asset: Self::SourceAsset,
        id: AssetId<Self::SourceAsset>,
        (device, queue, usage): &mut SystemParamItem<Self::Param>,
        previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let need_clusters = usage.clustered.contains(&id);
        let need_shadow = usage.shadowed.contains(&id);
        let revision = previous.map_or(0, |cloud| cloud.revision.wrapping_add(1));
        let byte_len = std::mem::size_of_val(asset.segments());
        if byte_len > device.limits().max_storage_buffer_binding_size as usize {
            error!(
                "Spaghetti asset requires {byte_len} storage bytes, exceeding this GPU's limit"
            );
            return Ok(Self {
                segments: None,
                shadow_segments: None,
                occlusion: None,
                centers: Box::new([]),
                clusters: Box::new([]),
                bounds: Aabb::from_min_max(Vec3::ZERO, Vec3::ZERO),
                max_bend_height: 0.0,
                detail: Box::new([]),
                revision,
            });
        }
        let (order, clusters) = if need_clusters {
            let (order, clusters) = cluster_segments(&asset);
            (Some(order), clusters)
        } else {
            (None, Box::new([]) as Box<[Cluster]>)
        };
        let sorted_storage: Vec<GaussianSegment>;
        let sorted: &[GaussianSegment] = if let Some(order) = &order {
            sorted_storage = order
                .iter()
                .map(|&index| asset.segments()[index as usize])
                .collect();
            &sorted_storage
        } else {
            asset.segments()
        };
        let data: &[u8] = bytemuck::cast_slice(sorted);
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
        let shadow_segments = need_shadow.then(|| {
            let shadow_sorted = build_shadow_segments(sorted, &clusters, min, max);
            let shadow_data: &[u8] = bytemuck::cast_slice(&shadow_sorted);
            let existing_shadow = previous
                .and_then(|cloud| cloud.shadow_segments.as_ref())
                .filter(|buffer| buffer.size() == shadow_data.len() as u64);
            if let Some(buffer) = existing_shadow {
                queue.write_buffer(buffer, 0, shadow_data);
                buffer.clone()
            } else {
                device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("spaghetti shadow segment data"),
                    contents: shadow_data,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                })
            }
        });
        let occlusion = (!asset.occlusion().is_empty()).then(|| {
            let pack = |[ambient, directional]: [f32; 2]| {
                let quantize = |factor: f32| (factor * 65535.0).round() as u32;
                quantize(ambient) | (quantize(directional) << 16)
            };
            let packed: Vec<u32> = if let Some(order) = &order {
                order
                    .iter()
                    .map(|&index| pack(asset.occlusion()[index as usize]))
                    .collect()
            } else {
                asset.occlusion().iter().copied().map(pack).collect()
            };
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("spaghetti occlusion attributes"),
                contents: bytemuck::cast_slice(&packed),
                usage: BufferUsages::STORAGE,
            })
        });
        Ok(Self {
            segments: Some(segments),
            shadow_segments,
            occlusion,
            centers,
            clusters,
            bounds: Aabb::from_min_max(min, max),
            max_bend_height: asset
                .segments()
                .iter()
                .map(|segment| segment.max_bend_height())
                .fold(0.0, f32::max),
            detail: asset.detail_transitions().into(),
            revision,
        })
    }
}

// 7-bit bit-reversed permutation for CLUSTER_SIZE == 128, so any prefix of a
// cluster is an evenly spread subset without per-cluster sorting or allocation.
const BIT_REVERSE_128: [u8; CLUSTER_SIZE as usize] = {
    let mut table = [0u8; CLUSTER_SIZE as usize];
    let mut i = 0;
    while i < CLUSTER_SIZE as usize {
        table[i] = (i as u8).reverse_bits() >> 1;
        i += 1;
    }
    table
};

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
            for &slot in &BIT_REVERSE_128 {
                if let Some(&(_, index)) = chunk.get(slot as usize) {
                    order.push(index);
                }
            }
        }
    }
    (order, clusters.into())
}

impl GpuCloud {
    /// Appends the clusters of `segments` that can draw and returns the
    /// instance count. A cluster is dropped when it lies outside a frustum
    /// plane or outside the depth band where its detail level shows.
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
        let start = visible.len();
        // Local planes that the whole cloud does not already satisfy. An
        // infinite plane, such as a cascade's pushed-out near plane, never culls.
        let object = Aabb {
            center: self.bounds.center,
            half_extents: self.bounds.half_extents + Vec3A::from(bending),
        };
        let mut planes = [(Vec3::ZERO, Vec3::ZERO, 0.0); 6];
        let mut plane_count = 0;
        for half_space in frustum.map_or(&[][..], |frustum| &frustum.half_spaces[..]) {
            let world = half_space.normal_d();
            if !world.is_finite() {
                continue;
            }
            let local = world_from_local.transpose() * world;
            let normal = local.truncate();
            let abs_normal = normal.abs();
            let center = Vec3::from(object.center);
            if normal.dot(center) + local.w - abs_normal.dot(object.half_extents.into()) < 0.0 {
                planes[plane_count] = (normal, abs_normal, local.w);
                plane_count += 1;
            }
        }
        let planes = &planes[..plane_count];
        let bands = if cull_bands { uniform.detail_ends[3] as usize } else { 0 };
        visible.reserve(clusters.len());
        if planes.is_empty() && bands == 0 {
            visible.extend(clusters.iter().map(pack));
        } else {
            let depth = clip_from_local.row(3);
            let depth_xyz = depth.truncate();
            let depth_abs = depth_xyz.abs();
            let mut level = 0;
            for cluster in clusters {
                let half = cluster.half_extents + bending;
                if planes
                    .iter()
                    .any(|&(normal, abs_normal, d)| normal.dot(cluster.center) + d + abs_normal.dot(half) < 0.0)
                {
                    continue;
                }
                if bands > 0 {
                    while level < bands && uniform.detail_ends[level] <= cluster.first {
                        level += 1;
                    }
                    let center = depth_xyz.dot(cluster.center) + depth.w;
                    let radius = depth_abs.dot(half);
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
        }
        // Earlier clusters still need a fixed stride, but no vertex needs
        // to run for empty slots after the last cluster.
        let Some((&last, full_stride)) = visible[start..].split_last() else {
            return 0;
        };
        full_stride.len() as u32 * keep + (last & (CLUSTER_SIZE - 1)) + 1
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
    shadow_shader: Handle<Shader>,
    bindings: BindGroupLayoutDescriptor,
    shadow_bindings: BindGroupLayoutDescriptor,
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
        shadow_shader: load_embedded_asset!(asset_server.as_ref(), "shadow.wgsl"),
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
        shadow_bindings: BindGroupLayoutDescriptor::new(
            "spaghetti shadow bindings",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    uniform_buffer_sized(
                        true,
                        NonZeroU64::new(size_of::<ShadowDrawUniform>() as u64),
                    ),
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
                label: Some("spaghetti shadow depth".into()),
                layout: vec![self.shadow_bindings.clone()],
                vertex: VertexState {
                    shader: self.shadow_shader.clone(),
                    entry_point: Some("shadow_vertex".into()),
                    ..default()
                },
                fragment: None,
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
// Power-of-two segments drawn per cluster of CLUSTER_SIZE in each directional
// cascade so the shadow vertex shader indexes clusters with a shift and mask.
const SHADOW_KEEP: [u32; 4] = [128, 64, 32, 32];
/// Minimum drawn segments per projected shadow-map texel of a cloud's OBB
/// before density-based thinning kicks in. Sparse foliage stays untouched,
/// while dense scanned props and small far-cascade objects thin automatically
/// with optical depth scaled to match.
const TARGET_SHADOW_SEGMENTS_PER_TEXEL: f32 = 2.0;

fn shadow_keep(
    cascade_index: usize,
    segment_count: u32,
    clip_from_local: Mat4,
    bounds: Aabb,
    size: UVec2,
) -> u32 {
    let max_keep = SHADOW_KEEP[cascade_index.min(SHADOW_KEEP.len() - 1)];
    let half = Vec3::from(bounds.half_extents);
    let scale = size.as_vec2() * 0.5;
    let ax = clip_from_local.x_axis.truncate().truncate() * (half.x * scale);
    let ay = clip_from_local.y_axis.truncate().truncate() * (half.y * scale);
    let az = clip_from_local.z_axis.truncate().truncate() * (half.z * scale);
    let area = 4.0 * (ax.perp_dot(ay).abs() + ay.perp_dot(az).abs() + az.perp_dot(ax).abs());
    let target = (area * TARGET_SHADOW_SEGMENTS_PER_TEXEL).max(1.0);
    let ratio = (segment_count as f32 / target).max(1.0);
    let desired = (CLUSTER_SIZE as f32 / ratio).ceil().max(1.0) as u32;
    desired.next_power_of_two().min(max_keep).max(1)
}

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
struct ShadowDrawUniform {
    clip_from_unit: [[f32; 4]; 4],
    world_from_unit_r0: [f32; 4],
    world_from_unit_r1: [f32; 4],
    world_from_unit_r2: [f32; 4],
    clip_from_world_r0: [f32; 4],
    clip_from_world_r1: [f32; 4],
    clip_from_world_r2: [f32; 4],
    j0_half_x: [f32; 4],
    j1_half_y: [f32; 4],
    pusher: [f32; 4],
    motion: [f32; 4],
    clusters: [u32; 4],
    pad_scale: [f32; 4],
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
    // Every main/reflection draw's uniform for this frame, at a dynamic offset.
    uniforms: Option<Buffer>,
    staging: Vec<u8>,
    // Compact 256-byte shadow uniforms for this frame, at a dynamic offset.
    shadow_uniforms: Option<Buffer>,
    shadow_staging: Vec<u8>,
    // Packed visible clusters of every cutout draw in this frame. Each entry
    // is the cluster's first segment shifted left by seven bits, plus its
    // count minus one.
    clusters: Vec<u32>,
    cluster_buffer: Option<Buffer>,
    // Reused per-view shadow uniform buffers and bind groups.
    cached_shadows: HashMap<(Entity, u32), CachedShadowBinding>,
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
    Main(&'a mut BinnedRenderPhase<Opaque3d>, DrawFunctionId),
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
                Opaque3dBatchSetKey {
                    pipeline,
                    draw_function: *draw_function,
                    material_bind_group_index: None,
                    lightmap_slab: None,
                    slabs: default(),
                },
                Opaque3dBinKey { asset_id: asset.untyped() },
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
        ResMut<ViewBinnedRenderPhases<Opaque3d>>,
        Res<DrawFunctions<Opaque3d>>,
        ResMut<ViewBinnedRenderPhases<Shadow>>,
        Res<DrawFunctions<Shadow>>,
    ),
    pipeline: Res<SplatPipeline>,
    cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SplatPipeline>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    (mut depths, mut order): (Local<Vec<(u16, u32)>>, Local<Vec<u32>>),
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
    let shadow_stride = size_of::<ShadowDrawUniform>().next_multiple_of(alignment);
    draws.staging.clear();
    draws.shadow_staging.clear();
    draws.clusters.clear();

    let draw_function = draw_functions.read().id::<(SetItemPipeline, DrawCloud)>();
    let cutout_draw_function = cutout_draw_functions.read().id::<(SetItemPipeline, DrawCloud)>();
    let shadow_draw_function = shadow_draw_functions.read().id::<(SetItemPipeline, DrawCloud)>();
    let bind_group_layout = cache.get_bind_group_layout(&pipeline.bindings);
    let shadow_bind_group_layout = cache.get_bind_group_layout(&pipeline.shadow_bindings);
    let default_layers = RenderLayers::default();
    // Resolve each cloud's GpuCloud once per frame instead of once per view,
    // and split shadow casters so cascade views skip non-casting clouds.
    let active_clouds: Vec<(Entity, MainEntity, &ExtractedCloud, &GpuCloud)> = clouds
        .iter()
        .filter_map(|(entity, main_entity, cloud)| {
            let gpu = gpu_clouds.get(cloud.asset)?;
            Some((entity, *main_entity, cloud, gpu))
        })
        .collect();
    let shadow_clouds: Vec<usize> = active_clouds
        .iter()
        .enumerate()
        .filter_map(|(index, &(_, _, cloud, gpu))| {
            (cloud.cast_shadows && gpu.shadow_segments.is_some()).then_some(index)
        })
        .collect();
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
        let mut pipeline_ids = [None; 17];
        let eye_world = view.world_from_view.translation();
        let mut view_shadows = None;

        if shadow_view {
            let Some(phase) = cutout_phase.as_mut() else {
                continue;
            };
            let cascade_index = match light {
                Some(&LightEntity::Directional { cascade_index, .. }) => cascade_index,
                _ => 0,
            };
            let half_viewport = size.as_vec2() * 0.5;
            let shadow_pipeline_id = *pipeline_ids[16].get_or_insert_with(|| {
                pipelines.specialize(
                    &cache,
                    &pipeline,
                    PipelineKey {
                        target: view.target_format,
                        samples: msaa.map_or(1, Msaa::samples),
                        cutout: true,
                        shadowed: false,
                        baked_occlusion: false,
                        additive: false,
                        shadow: true,
                    },
                )
            });
            let empty_uniform = DrawUniform::zeroed();
            for &cloud_index in &shadow_clouds {
                let (entity, main_entity, cloud, gpu) = active_clouds[cloud_index];
                if !layers.intersects(&cloud.layers) {
                    continue;
                }
                let source = gpu.shadow_segments.as_ref().unwrap();
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
                if coverage <= 0.0 {
                    continue;
                }
                let inverse_basis = cloud.inverse_basis;
                let sway = cloud.local_sway;
                let mut bounds = gpu.bounds;
                if let Some(sway) = sway {
                    bounds.half_extents += Vec3A::from(sway.abs() * gpu.max_bend_height);
                } else if let (Some(_), Some(inverse)) = (cloud.wind, inverse_basis) {
                    let rows = inverse.transpose();
                    let stretch = Vec3::new(rows.x_axis.length(), rows.y_axis.length(), rows.z_axis.length());
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
                let mut scratch_detail = empty_uniform;
                let segments = gpu.select_segments(
                    clip_from_local,
                    size.y as f32,
                    cloud.detail_scale * SHADOW_DETAIL,
                    bounds,
                    true,
                    &mut scratch_detail,
                );
                let keep = shadow_keep(
                    cascade_index,
                    segments.len() as u32,
                    clip_from_local,
                    bounds,
                    size,
                );
                let cluster_offset = draws.clusters.len() as u32;
                let cull = !cloud.disable_culling && cloud.flow.is_none();
                let count = gpu.cull_clusters(
                    segments.clone(),
                    cull.then_some(()).and(frustum),
                    false,
                    (cloud.world_from_local, clip_from_local),
                    Vec3::from(bounds.half_extents - gpu.bounds.half_extents),
                    &empty_uniform,
                    keep,
                    &mut draws.clusters,
                );
                if count == 0 {
                    continue;
                }
                let min = Vec3::from(gpu.bounds.min());
                let extent = (Vec3::from(gpu.bounds.max()) - min).max(Vec3::splat(1e-6));
                let asset_scale = extent.max_element();
                let local_from_unit = Mat4::from_cols(
                    Vec4::new(extent.x, 0.0, 0.0, 0.0),
                    Vec4::new(0.0, extent.y, 0.0, 0.0),
                    Vec4::new(0.0, 0.0, extent.z, 0.0),
                    min.extend(1.0),
                );
                let row0 = clip_from_local.row(0).truncate();
                let row1 = clip_from_local.row(1).truncate();
                let mut shadow_uniform = ShadowDrawUniform {
                    clip_from_unit: (clip_from_local * local_from_unit).to_cols_array_2d(),
                    world_from_unit_r0: [0.0; 4],
                    world_from_unit_r1: [0.0; 4],
                    world_from_unit_r2: [0.0; 4],
                    clip_from_world_r0: [0.0; 4],
                    clip_from_world_r1: [0.0; 4],
                    clip_from_world_r2: [0.0; 4],
                    j0_half_x: (row0 * (half_viewport.x * asset_scale))
                        .extend(half_viewport.x)
                        .to_array(),
                    j1_half_y: (row1 * (half_viewport.y * asset_scale))
                        .extend(half_viewport.y)
                        .to_array(),
                    pusher: [0.0; 4],
                    motion: [0.0, 0.0, coverage, reverse_dither],
                    clusters: [
                        cluster_offset,
                        keep.trailing_zeros(),
                        (2.0 * (CLUSTER_SIZE as f32 / keep as f32).ln()).to_bits(),
                        0,
                    ],
                    pad_scale: [row0.length() * asset_scale, row1.length() * asset_scale, 0.0, 0.0],
                };
                if let Some(sway) = sway {
                    shadow_uniform.motion[1] = 2.0;
                    shadow_uniform.pusher = (clip_from_local * sway.extend(0.0)).to_array();
                } else if let (Some(wind), Some(_)) = (cloud.wind, inverse_basis) {
                    let world_from_unit = cloud.world_from_local * local_from_unit;
                    shadow_uniform.world_from_unit_r0 = world_from_unit.row(0).to_array();
                    shadow_uniform.world_from_unit_r1 = world_from_unit.row(1).to_array();
                    shadow_uniform.world_from_unit_r2 = world_from_unit.row(2).to_array();
                    shadow_uniform.clip_from_world_r0 = clip_from_world.row(0).to_array();
                    shadow_uniform.clip_from_world_r1 = clip_from_world.row(1).to_array();
                    shadow_uniform.clip_from_world_r2 = clip_from_world.row(2).to_array();
                    if wind.push_radius > 0.0 {
                        let transform = cloud.world_from_local;
                        let center = transform.transform_point3(gpu.bounds.center.into());
                        let half = gpu.bounds.half_extents;
                        let extent = transform.x_axis.truncate().abs() * half.x
                            + transform.y_axis.truncate().abs() * half.y
                            + transform.z_axis.truncate().abs() * half.z;
                        let outside =
                            ((wind.pusher - center).xz().abs() - extent.xz()).max(Vec2::ZERO);
                        if outside.length_squared() < wind.push_radius * wind.push_radius {
                            shadow_uniform.pusher = wind.pusher.extend(wind.push_radius).to_array();
                        }
                    }
                    shadow_uniform.motion[0] = wind.time;
                    shadow_uniform.motion[1] = 1.0;
                }
                let key = (view.retained_view_entity, entity);
                let new_draw = || {
                    make_draw(
                        &device,
                        source,
                        gpu,
                        main_entity,
                        true,
                        true,
                        &pipeline.shadows.fallback,
                    )
                };
                let draw = draws.draws.entry(key).or_insert_with(new_draw);
                if draw.source.id() != source.id() || !draw.shadow {
                    if draw.cutout_pipeline.is_some() {
                        phase.remove(draw.main_entity);
                    }
                    *draw = new_draw();
                }
                draw.segments = segments;
                draw.instances = 0..count;
                draw.uniform_offset = draws.shadow_staging.len() as u32;
                draws
                    .shadow_staging
                    .extend_from_slice(bytemuck::bytes_of(&shadow_uniform));
                draws
                    .shadow_staging
                    .resize(draw.uniform_offset as usize + shadow_stride, 0);
                draw.idle_frames = 0;
                if draw.cutout_pipeline != Some(shadow_pipeline_id) {
                    if draw.cutout_pipeline.replace(shadow_pipeline_id).is_some() {
                        phase.remove(main_entity);
                    }
                    phase.add(shadow_pipeline_id, cloud.asset, (entity, main_entity));
                }
            }
            continue;
        }

        for &(entity, main_entity, cloud, gpu) in &active_clouds {
            if !layers.intersects(&cloud.layers) {
                continue;
            }
            let cutout = cloud.cutout;
            let Some(source) = gpu.segments.as_ref() else {
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
            let inverse_basis = cloud.inverse_basis;
            let sway = cloud.local_sway;
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
                            &queue,
                            &cache,
                            &mut draws.cached_shadows,
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
                cloud.detail_scale,
                bounds,
                false,
                &mut uniform,
            );
            let mut instances = 0..segments.len() as u32;
            if cutout {
                // Flow moves segments out of their clusters' boxes.
                let cull = !cloud.disable_culling && cloud.flow.is_none();
                uniform.clusters[0] = draws.clusters.len() as u32;
                uniform.clusters[1] = CLUSTER_SIZE;
                let count = gpu.cull_clusters(
                    segments.clone(),
                    cull.then_some(()).and(frustum),
                    cull,
                    (cloud.world_from_local, clip_from_local),
                    Vec3::from(bounds.half_extents - gpu.bounds.half_extents),
                    &uniform,
                    CLUSTER_SIZE,
                    &mut draws.clusters,
                );
                if count == 0 {
                    continue;
                }
                instances = 0..count;
            }
            let new_draw = || make_draw(&device, source, gpu, main_entity, cutout, false, shadows);
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
                if !(draw.direction.distance_squared(direction) <= 1e-4)
                    || draw.revision != gpu.revision
                    || draw.segments != segments
                {
                    // Translation preserves relative depths. Re-sort only if the
                    // view direction, sway depth, asset data, or submitted levels changed.
                    let dir_xyz = direction.truncate();
                    let center_d = dir_xyz.dot(gpu.bounds.center.into());
                    let radius_d = dir_xyz.abs().dot(gpu.bounds.half_extents.into());
                    let sway_d = gpu.max_bend_height * direction.w;
                    let min_d = center_d - radius_d + sway_d.min(0.0);
                    let max_d = center_d + radius_d + sway_d.max(0.0);
                    let scale = if max_d > min_d { 65535.0 / (max_d - min_d) } else { 0.0 };
                    depths.clear();
                    depths.extend(
                        gpu.centers[segments.start as usize..segments.end as usize]
                            .iter()
                            .enumerate()
                            .map(|(index, &center)| {
                                let key = ((center.dot(direction) - min_d) * scale)
                                    .clamp(0.0, 65535.0) as u16;
                                (key, segments.start + index as u32)
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
                        phase.remove(main_entity);
                    }
                    phase.add(pipeline_id, cloud.asset, (entity, main_entity));
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
                entity: (entity, main_entity),
                draw_function,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
    if !draws.staging.is_empty() || !draws.shadow_staging.is_empty() {
        let uniform = (!draws.staging.is_empty()).then(|| {
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
            (
                uniforms.id(),
                BufferBinding {
                    buffer: uniforms,
                    offset: 0,
                    size: NonZeroU64::new(size_of::<DrawUniform>() as u64),
                },
            )
        });
        let shadow_uniform = (!draws.shadow_staging.is_empty()).then(|| {
            let size = draws.shadow_staging.len() as u64;
            if draws
                .shadow_uniforms
                .as_ref()
                .is_none_or(|buffer| buffer.size() < size)
            {
                draws.shadow_uniforms = Some(device.create_buffer(&BufferDescriptor {
                    label: Some("spaghetti shadow draw uniforms"),
                    size: size.next_power_of_two(),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let shadow_uniforms = draws.shadow_uniforms.as_ref().unwrap();
            queue.write_buffer(shadow_uniforms, 0, &draws.shadow_staging);
            (
                shadow_uniforms.id(),
                BufferBinding {
                    buffer: shadow_uniforms,
                    offset: 0,
                    size: NonZeroU64::new(size_of::<ShadowDrawUniform>() as u64),
                },
            )
        });
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
            if draw.shadow {
                let Some((uniform_id, shadow_binding)) = &shadow_uniform else {
                    continue;
                };
                if draw.bindings.as_ref().is_some_and(|(u_id, listed_id, _)| {
                    *u_id == *uniform_id && *listed_id == listed.id()
                }) {
                    continue;
                }
                let bindings = device.create_bind_group(
                    "spaghetti shadow draw bindings",
                    &shadow_bind_group_layout,
                    &BindGroupEntries::sequential((
                        shadow_binding.clone(),
                        draw.source.as_entire_binding(),
                        listed.as_entire_binding(),
                    )),
                );
                draw.bindings = Some((*uniform_id, listed.id(), bindings));
            } else {
                let Some((uniform_id, uniform_binding)) = &uniform else {
                    continue;
                };
                if draw.bindings.as_ref().is_some_and(|(u_id, listed_id, _)| {
                    *u_id == *uniform_id && *listed_id == listed.id()
                }) {
                    continue;
                }
                let bindings = device.create_bind_group(
                    "spaghetti draw bindings",
                    &bind_group_layout,
                    &BindGroupEntries::sequential((
                        uniform_binding.clone(),
                        draw.source.as_entire_binding(),
                        listed.as_entire_binding(),
                        draw.occlusion.as_ref().unwrap_or(&draw.source).as_entire_binding(),
                    )),
                );
                draw.bindings = Some((*uniform_id, listed.id(), bindings));
            }
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
    draws
        .cached_shadows
        .retain(|&(view_entity, _), _| views.contains(view_entity));
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
        let instances = draw.instances.clone();
        if draw.shadow {
            pass.draw(instances.start * 3..instances.end * 3, 0..1);
        } else {
            pass.set_bind_group(1, &draw.shadows, &[]);
            pass.set_index_buffer(
                pipeline.into_inner().quad_indices.slice(..),
                IndexFormat::Uint16,
            );
            pass.draw_indexed(0..6, 0, instances);
        }
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

    #[test]
    fn cluster_draw_omits_padding_only_after_last_visible_cluster() {
        let cloud = GpuCloud {
            segments: None,
            shadow_segments: None,
            occlusion: None,
            centers: Box::new([]),
            clusters: Box::new([
                Cluster {
                    first: 0,
                    count: 32,
                    center: Vec3::ZERO,
                    half_extents: Vec3::ONE,
                },
                Cluster {
                    first: 32,
                    count: 15,
                    center: Vec3::ZERO,
                    half_extents: Vec3::ONE,
                },
            ]),
            bounds: Aabb::from_min_max(-Vec3::ONE, Vec3::ONE),
            max_bend_height: 0.0,
            detail: Box::new([]),
            revision: 0,
        };
        let mut visible = vec![0];
        let mut cull = |segments, keep| {
            cloud.cull_clusters(
                segments,
                None,
                false,
                (Mat4::IDENTITY, Mat4::IDENTITY),
                Vec3::ZERO,
                &DrawUniform::zeroed(),
                keep,
                &mut visible,
            )
        };
        assert_eq!(cull(0..47, CLUSTER_SIZE), CLUSTER_SIZE + 15);
        assert_eq!(cull(0..47, 48), 48 + 6);
        assert_eq!(cull(47..47, 48), 0);
    }

    fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = (bits >> 10) & 0x1f;
        let mant = (bits & 0x03ff) as u32;
        if exp == 0 {
            if mant == 0 {
                return f32::from_bits(sign);
            }
            let val = (mant as f32) * 2.0_f32.powi(-24);
            return if sign != 0 { -val } else { val };
        }
        if exp == 0x1f {
            return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
        }
        f32::from_bits(sign | (((exp as u32) + 112) << 23) | (mant << 13))
    }

    #[test]
    fn shadow_segment_packing_preserves_covariance_and_f16_precision() {
        for value in [0.0_f32, -0.0, 1.0, -3.5, 0.0025, 1e-5, 15.75, -65000.0] {
            let decoded = f16_bits_to_f32(f32_to_f16_bits(value));
            let tol = value.abs() * 2e-3 + 1e-6;
            assert!(
                (decoded - value).abs() <= tol,
                "f16 round-trip for {value}: got {decoded}"
            );
        }

        let rot = Quat::from_euler(EulerRot::YXZ, 0.7, -0.4, 1.1).normalize();
        let sigma = Vec3::new(0.035, 0.05, 0.011);
        let segment = GaussianSegment::new(
            Vec3::new(-0.2, 0.1, 0.3),
            Vec3::new(0.25, 0.4, -0.1),
            sigma,
            rot,
            0.5,
            Vec3::ONE,
        )
        .with_cutout_taper(0.35);
        let asset = SpaghettiAsset::new(vec![segment]).unwrap();
        let (min, max) = asset.bounds();
        let (_, clusters) = cluster_segments(&asset);
        let packed = build_shadow_segments(asset.segments(), &clusters, min, max);
        let seg = packed[0];
        let unpack_snorm8 = |word: u32, shift: u32| -> f32 {
            ((word >> shift) as u8 as i8 as f32 / 127.0).max(-1.0)
        };
        let s00 = unpack_snorm8(seg.attr[0], 0);
        let s01 = unpack_snorm8(seg.attr[0], 8);
        let s02 = unpack_snorm8(seg.attr[0], 16);
        let s11 = unpack_snorm8(seg.attr[0], 24);
        let s12 = unpack_snorm8(seg.attr[1], 0);
        let s22 = unpack_snorm8(seg.attr[1], 8);
        let taper = unpack_snorm8(seg.attr[1], 16);
        assert!((taper - 0.35).abs() < 0.01);

        let asset_scale = (max - min).max_element();
        let sigma_max = f16_bits_to_f32((seg.geom[3] >> 16) as u16) * asset_scale;
        for dir in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.5, -0.7, 0.5).normalize()] {
            let exact = (rot.conjugate() * dir * sigma).length();
            let su = Vec3::new(
                dir.x * s00 + dir.y * s01 + dir.z * s02,
                dir.x * s01 + dir.y * s11 + dir.z * s12,
                dir.x * s02 + dir.y * s12 + dir.z * s22,
            );
            let approx = sigma_max * dir.dot(su).max(0.0).sqrt();
            assert!(
                (approx - exact).abs() < 0.002,
                "projected sigma mismatch along {dir:?}: exact={exact}, approx={approx}"
            );
        }
    }

    #[test]
    fn shadow_keep_preserves_sparse_clouds_and_thins_dense_props() {
        let size = UVec2::splat(2048);
        // Orthographic 40m wide cascade -> ~51.2 px/m.
        let clip = Mat4::from_scale(Vec3::splat(1.0 / 20.0));
        // Sparse 10m conifer with 4000 segments (~0.015 segs/texel) stays full in cascade 0.
        let conifer = Aabb::from_min_max(Vec3::splat(-4.0), Vec3::splat(4.0));
        assert_eq!(shadow_keep(0, 4_000, clip, conifer, size), 128);
        assert_eq!(shadow_keep(1, 4_000, clip, conifer, size), 64);
        assert_eq!(shadow_keep(2, 4_000, clip, conifer, size), 32);
        // Dense 0.25m stone with 6000 segments in ~160 texels thins strongly to a power of two.
        let stone = Aabb::from_min_max(Vec3::splat(-0.125), Vec3::splat(0.125));
        let kept = shadow_keep(0, 6_000, clip, stone, size);
        assert!(kept.is_power_of_two() && kept <= 16, "expected thin keep, got {kept}");
    }
}
