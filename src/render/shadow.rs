use std::collections::HashMap;

use bevy::{
    prelude::*,
    render::{
        render_resource::{binding_types::*, *},
        renderer::{RenderDevice, RenderQueue},
    },
};
use bevy_pbr::{ExtractedDirectionalLight, LightEntity, ViewLightEntities, ViewShadowBindings};
use bytemuck::{Pod, Zeroable};

const CASCADE_CAPACITY: usize = 4;

pub(super) type ShadowViewData = (&'static ViewShadowBindings, &'static ViewLightEntities);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowUniform {
    matrices: [[[f32; 4]; 4]; CASCADE_CAPACITY],
    far_bounds: [f32; 4],
    normal_bias: [f32; 4],
    view_depth: [f32; 4],
    depth_bias: [f32; 4],
    // Base array layer and number of cascades.
    layers: [u32; 4],
}

pub(super) struct CachedShadowBinding {
    buffer: Buffer,
    texture_view_id: TextureViewId,
    bind_group: BindGroup,
}

pub(super) struct ShadowBindings {
    pub layout: BindGroupLayoutDescriptor,
    pub fallback: BindGroup,
    sampler: Sampler,
}

impl ShadowBindings {
    pub fn new(device: &RenderDevice, cache: &PipelineCache) -> Self {
        let layout = BindGroupLayoutDescriptor::new(
            "spaghetti shadow bindings",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    uniform_buffer_sized(
                        false,
                        std::num::NonZeroU64::new(size_of::<ShadowUniform>() as u64),
                    ),
                    texture_2d_array(TextureSampleType::Depth),
                    sampler(SamplerBindingType::Comparison),
                ),
            ),
        );
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("spaghetti shadow sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            compare: Some(CompareFunction::GreaterEqual),
            ..default()
        });
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("spaghetti fallback shadow map"),
            size: Extent3d::default(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..default()
        });
        let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("spaghetti fallback shadow cascades"),
            contents: bytemuck::bytes_of(&ShadowUniform::zeroed()),
            usage: BufferUsages::UNIFORM,
        });
        let fallback = device.create_bind_group(
            "spaghetti fallback shadows",
            &cache.get_bind_group_layout(&layout),
            &BindGroupEntries::sequential((buffer.as_entire_binding(), &view, &sampler)),
        );
        Self {
            layout,
            fallback,
            sampler,
        }
    }

    #[expect(clippy::too_many_arguments, reason = "Bevy render view parameters")]
    pub fn for_view(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
        cached: &mut HashMap<(Entity, u32), CachedShadowBinding>,
        entity: Entity,
        (textures, entities): (&ViewShadowBindings, &ViewLightEntities),
        view_depth: Vec4,
        cascade_views: &Query<&LightEntity>,
        lights: &Query<&ExtractedDirectionalLight>,
    ) -> Vec<(Vec3, BindGroup)> {
        let mut bindings = Vec::new();
        let mut directional_layer = 0;
        let texture_view = &textures.directional_light_depth_texture_view;
        let texture_view_id = texture_view.id();
        // Bevy emits directional cascades in texture layer order. Point and
        // spot shadow views in this list don't advance the directional index.
        let mut directional_views = entities
            .lights
            .iter()
            .filter_map(|&view| match cascade_views.get(view).ok()? {
                LightEntity::Directional {
                    light_entity,
                    cascade_index,
                } => Some((*light_entity, *cascade_index)),
                _ => None,
            })
            .peekable();
        while let Some((light, _)) = directional_views.peek().copied() {
            let base = directional_layer;
            while directional_views
                .next_if(|&(entity, _)| entity == light)
                .is_some()
            {
                directional_layer += 1;
            }
            let Ok(light) = lights.get(light) else {
                continue;
            };
            let Some(cascades) = light.cascades.get(&entity) else {
                continue;
            };
            let direction = light.transform.back().as_vec3();
            let mut uniform = ShadowUniform::zeroed();
            uniform.far_bounds = [f32::MAX; CASCADE_CAPACITY];
            uniform.view_depth = view_depth.to_array();
            // The fourth lane holds the share of each cascade that blends
            // into the next one.
            uniform.depth_bias = (direction * light.shadow_depth_bias)
                .extend(light.cascade_shadow_config.overlap_proportion)
                .to_array();
            uniform.layers[0] = base;
            for (index, (cascade, &far)) in cascades
                .iter()
                .zip(&light.cascade_shadow_config.bounds)
                .take((directional_layer - base) as usize)
                .take(CASCADE_CAPACITY)
                .enumerate()
            {
                uniform.matrices[index] = cascade.clip_from_world.to_cols_array_2d();
                uniform.far_bounds[index] = far;
                uniform.normal_bias[index] = light.shadow_normal_bias * cascade.texel_size;
                uniform.layers[1] += 1;
            }
            let entry = cached.entry((entity, base)).or_insert_with(|| {
                let buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("spaghetti view shadow cascades"),
                    size: size_of::<ShadowUniform>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = device.create_bind_group(
                    "spaghetti view shadows",
                    &cache.get_bind_group_layout(&self.layout),
                    &BindGroupEntries::sequential((
                        buffer.as_entire_binding(),
                        texture_view,
                        &self.sampler,
                    )),
                );
                CachedShadowBinding {
                    buffer,
                    texture_view_id,
                    bind_group,
                }
            });
            queue.write_buffer(&entry.buffer, 0, bytemuck::bytes_of(&uniform));
            if entry.texture_view_id != texture_view_id {
                entry.texture_view_id = texture_view_id;
                entry.bind_group = device.create_bind_group(
                    "spaghetti view shadows",
                    &cache.get_bind_group_layout(&self.layout),
                    &BindGroupEntries::sequential((
                        entry.buffer.as_entire_binding(),
                        texture_view,
                        &self.sampler,
                    )),
                );
            }
            bindings.push((direction, entry.bind_group.clone()));
        }
        bindings
    }
}
