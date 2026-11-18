use std::{collections::HashMap, num::NonZeroU64};

use bevy::{
    asset::{AssetLoader, LoadContext, embedded_asset, io::Reader, load_embedded_asset},
    camera::{
        MainPassResolutionOverride,
        primitives::{Aabb, Frustum},
        visibility::{InheritedVisibility, NoFrustumCulling, RenderLayers},
    },
    core_pipeline::core_3d::{CORE_3D_DEPTH_FORMAT, Transparent3d, TransparentSortingInfo3d},
    ecs::{
        query::QueryItem,
        system::{SystemParamItem, lifetimeless::*},
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        camera::TemporalJitter,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{binding_types::*, *},
        renderer::{RenderDevice, RenderQueue},
        sync_component::SyncComponent,
        sync_world::MainEntity,
        view::{ExtractedView, RetainedViewEntity},
    },
};
use bevy_math::Affine3A;
use bytemuck::{Pod, Zeroable};

use crate::{AssetError, SpaghettiAsset};

/// Renders a cloud using the entity's transform, visibility and render layers.
#[derive(Component, Clone, Debug)]
#[require(Transform, Visibility)]
pub struct Spaghetti3d(pub Handle<SpaghettiAsset>);

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
                .init_resource::<SpecializedRenderPipelines<SplatPipeline>>()
                .init_resource::<ViewDraws>()
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
    );
    type QueryFilter = ();
    type Out = ExtractedCloud;

    fn extract_component(
        (cloud, transform, visibility, layers, disable_culling): QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        visibility.get().then(|| ExtractedCloud {
            asset: cloud.0.id(),
            world_from_local: transform.to_matrix(),
            layers: layers.cloned().unwrap_or_default(),
            disable_culling,
        })
    }
}

struct GpuCloud {
    segments: Option<Buffer>,
    centers: Box<[Vec3]>,
    bounds: Aabb,
}

impl RenderAsset for GpuCloud {
    type SourceAsset = SpaghettiAsset;
    type Param = SRes<RenderDevice>;

    fn byte_len(asset: &Self::SourceAsset) -> Option<usize> {
        Some(std::mem::size_of_val(asset.segments()))
    }

    fn prepare_asset(
        asset: Self::SourceAsset,
        _: AssetId<Self::SourceAsset>,
        device: &mut SystemParamItem<Self::Param>,
        _: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let data = bytemuck::cast_slice(asset.segments());
        if data.len() > device.limits().max_storage_buffer_binding_size as usize {
            error!(
                "Spaghetti asset requires {} storage bytes, exceeding this GPU's limit",
                data.len()
            );
            return Ok(Self {
                segments: None,
                centers: Box::new([]),
                bounds: Aabb::from_min_max(Vec3::ZERO, Vec3::ZERO),
            });
        }
        let (min, max) = asset.bounds();
        let centers = asset
            .segments()
            .iter()
            .map(|segment| segment.start() * 0.5 + segment.end() * 0.5)
            .collect();
        let segments = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("spaghetti segment data"),
            contents: data,
            usage: BufferUsages::STORAGE,
        });
        Ok(Self {
            segments: Some(segments),
            centers,
            bounds: Aabb::from_min_max(min, max),
        })
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
}

impl SpecializedRenderPipeline for SplatPipeline {
    type Key = PipelineKey;

    fn specialize(&self, key: PipelineKey) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("spaghetti volume compositing".into()),
            layout: vec![self.bindings.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format: key.target,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(false),
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    clip_from_local: [[f32; 4]; 4],
    viewport: [f32; 4],
}

struct CloudDraw {
    source: Buffer,
    uniform: Buffer,
    order: Buffer,
    bindings: BindGroup,
    direction: Vec3,
    count: u32,
    used: bool,
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
    mut phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    pipeline: Res<SplatPipeline>,
    cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SplatPipeline>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut depths: Local<Vec<(f32, u32)>>,
    mut order: Local<Vec<u32>>,
) {
    for draw in draws.0.values_mut() {
        draw.used = false;
    }
    if clouds.is_empty() {
        draws.0.clear();
        return;
    }

    let draw_function = draw_functions.read().id::<(SetItemPipeline, DrawCloud)>();
    let bind_group_layout = cache.get_bind_group_layout(&pipeline.bindings);
    let default_layers = RenderLayers::default();
    for (view, msaa, frustum, layers, jitter, resolution) in &views {
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let layers = layers.unwrap_or(&default_layers);
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
        let pipeline_id = pipelines.specialize(
            &cache,
            &pipeline,
            PipelineKey {
                target: view.target_format,
                samples: msaa.samples(),
            },
        );

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
            if !cloud.disable_culling
                && frustum.is_some_and(|frustum| {
                    !frustum.intersects_obb(
                        &gpu.bounds,
                        &Affine3A::from_mat4(cloud.world_from_local),
                        true,
                        true,
                    )
                })
            {
                continue;
            }
            let view_from_local = view_from_world * cloud.world_from_local;
            let direction = view_from_local.row(2).truncate().normalize_or_zero();
            let uniform = DrawUniform {
                clip_from_local: (clip_from_world * cloud.world_from_local).to_cols_array_2d(),
                viewport: [size.x as f32, size.y as f32, 0.0, 0.0],
            };
            let key = (view.retained_view_entity, entity);
            let draw = draws.0.entry(key).or_insert_with(|| {
                make_draw(&device, &bind_group_layout, source, gpu.centers.len())
            });
            if draw.source.id() != source.id() {
                *draw = make_draw(&device, &bind_group_layout, source, gpu.centers.len());
            }
            if draw.direction != direction {
                // Camera translation adds the same depth offset to every segment.
                // Sort only when the view direction in asset space changes.
                depths.clear();
                depths.extend(
                    gpu.centers
                        .iter()
                        .enumerate()
                        .map(|(index, &center)| (center.dot(direction), index as u32)),
                );
                radsort::sort_by_key(&mut depths, |entry| entry.0);
                order.clear();
                order.extend(depths.iter().map(|&(_, index)| index));
                queue.write_buffer(&draw.order, 0, bytemuck::cast_slice(&order));
                draw.direction = direction;
            }
            queue.write_buffer(&draw.uniform, 0, bytemuck::bytes_of(&uniform));
            draw.used = true;
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
    draws.0.retain(|_, draw| draw.used);
}

fn make_draw(
    device: &RenderDevice,
    layout: &BindGroupLayout,
    source: &Buffer,
    count: usize,
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
    let order = make_buffer(
        "spaghetti depth order",
        count * size_of::<u32>(),
        BufferUsages::STORAGE,
    );
    let bindings = device.create_bind_group(
        "spaghetti draw bindings",
        layout,
        &BindGroupEntries::sequential((
            uniform.as_entire_binding(),
            source.as_entire_binding(),
            order.as_entire_binding(),
        )),
    );
    CloudDraw {
        source: source.clone(),
        uniform,
        order,
        bindings,
        direction: Vec3::NAN,
        count: count as u32,
        used: false,
    }
}

struct DrawCloud;

impl RenderCommand<Transparent3d> for DrawCloud {
    type Param = SRes<ViewDraws>;
    type ViewQuery = Read<ExtractedView>;
    type ItemQuery = ();

    fn render<'w>(
        item: &Transparent3d,
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
        pass.draw(0..6, 0..draw.count);
        RenderCommandResult::Success
    }
}
