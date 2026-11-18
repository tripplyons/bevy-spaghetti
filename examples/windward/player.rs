use std::{f32::consts::PI, time::Duration};

use bevy::{
    animation::{graph::AnimationNodeIndex, transition::AnimationTransitions},
    app::AnimationSystems,
    input::mouse::AccumulatedMouseMotion,
    prelude::*,
    transform::TransformSystems,
    world_serialization::WorldInstanceReady,
};

use crate::landscape::{Obstacles, WATER_LEVEL, ground, trail_x};

const MODEL: &str = "windward/hiker.glb";
const WALK_SPEED: f32 = 1.9;
const RUN_SPEED: f32 = 4.6;
/// The clips hold the upper arms about 25 degrees out from the body. This brings them in
/// so the hands hang beside the thighs without touching them.
const ARM_ADDUCTION: f32 = 0.13;

/// Adds the pose correction that runs after the animation clips.
pub struct HikerPosePlugin;

impl Plugin for HikerPosePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            relax_arms
                .after(AnimationSystems)
                .before(TransformSystems::Propagate),
        );
    }
}

/// An upper-arm bone and the angle it turns toward the body about the hiker's forward axis.
#[derive(Component)]
struct RelaxedArm(f32);

#[derive(Resource, Default)]
pub struct Controls {
    pub movement: Vec2,
    pub run: bool,
    pub jump: bool,
    pub reset: bool,
}

#[derive(Component, Default)]
pub struct Hiker {
    pub velocity: Vec2,
    pub vertical_speed: f32,
    pub grounded: bool,
}

#[derive(Component)]
pub struct HikerCamera {
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
}

impl Default for HikerCamera {
    fn default() -> Self {
        Self {
            yaw: -0.16,
            pitch: 0.10,
        }
    }
}

#[derive(Component, Clone)]
pub(crate) struct HikerClips {
    graph: Handle<AnimationGraph>,
    walk: AnimationNodeIndex,
    run: AnimationNodeIndex,
    idle: AnimationNodeIndex,
}

fn start_position() -> Vec3 {
    let z = 8.5;
    let at = Vec2::new(trail_x(z), z);
    Vec3::new(at.x, ground(at), at.y)
}

impl HikerCamera {
    fn pose(&self, feet: Vec3, obstacles: &Obstacles) -> Transform {
        let target = feet + Vec3::Y * 1.56;
        let rotation = Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(-self.pitch);
        // Frame the hiker off the left of center and leave room to see down the trail.
        let focus = target + rotation * Vec3::new(0.66, 0.40, -1.9);
        let offset = rotation * Vec3::new(0.83, 0.62, 4.95);
        let eye = obstacles.camera_eye(target, target + offset);
        let framing = ((eye.distance(target) - 1.15) / 2.0).clamp(0.0, 1.0);
        Transform::from_translation(eye).looking_at(target.lerp(focus, framing), Vec3::Y)
    }
}

pub fn initial_view(obstacles: &Obstacles) -> Transform {
    HikerCamera::default().pose(start_position(), obstacles)
}

pub fn spawn_hiker(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
) {
    let (graph, indices) = AnimationGraph::from_clips(
        (0..3).map(|index| server.load(GltfAssetLabel::Animation(index).from_asset(MODEL))),
    );
    commands
        .spawn((
            Name::new("Hiker"),
            Hiker {
                grounded: true,
                ..default()
            },
            Transform::from_translation(start_position()).with_rotation(Quat::from_rotation_y(PI)),
            WorldAssetRoot(server.load(GltfAssetLabel::Scene(0).from_asset(MODEL))),
            HikerClips {
                graph: graphs.add(graph),
                walk: indices[0],
                run: indices[1],
                idle: indices[2],
            },
        ))
        .observe(crate::landscape::mark_reflected_meshes)
        .observe(start_animation);
}

fn start_animation(
    ready: On<WorldInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    actors: Query<&HikerClips, With<Hiker>>,
    names: Query<&Name>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let Ok(clips) = actors.get(ready.entity) else {
        return;
    };
    for child in children.iter_descendants(ready.entity) {
        match names.get(child).map(Name::as_str) {
            Ok("LeftArm") => {
                commands.entity(child).insert(RelaxedArm(-ARM_ADDUCTION));
            }
            Ok("RightArm") => {
                commands.entity(child).insert(RelaxedArm(ARM_ADDUCTION));
            }
            _ => {}
        }
        let Ok(mut player) = players.get_mut(child) else {
            continue;
        };
        let mut transitions = AnimationTransitions::new();
        transitions
            .play(&mut player, clips.idle, Duration::ZERO)
            .repeat();
        commands.entity(child).insert((
            AnimationGraphHandle(clips.graph.clone()),
            transitions,
            clips.clone(),
        ));
    }
}

pub fn read_controls(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    delta: Res<AccumulatedMouseMotion>,
    mut controls: ResMut<Controls>,
    camera: Option<Single<&mut HikerCamera>>,
) {
    let axis =
        |positive, negative| f32::from(keys.pressed(positive)) - f32::from(keys.pressed(negative));
    controls.movement = Vec2::new(
        axis(KeyCode::KeyD, KeyCode::KeyA),
        axis(KeyCode::KeyW, KeyCode::KeyS),
    );
    controls.run = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    controls.jump |= keys.just_pressed(KeyCode::Space);
    controls.reset |= keys.just_pressed(KeyCode::KeyR);
    if mouse.pressed(MouseButton::Right)
        && let Some(mut camera) = camera
    {
        camera.yaw = (camera.yaw - delta.delta.x * 0.0035).rem_euclid(std::f32::consts::TAU);
        camera.pitch = (camera.pitch - delta.delta.y * 0.0035).clamp(-0.36, 0.77);
    }
}

pub fn move_hiker(
    time: Res<Time<Fixed>>,
    mut controls: ResMut<Controls>,
    obstacles: Res<Obstacles>,
    mut camera: Single<&mut HikerCamera>,
    mut hiker: Single<(&mut Hiker, &mut Transform)>,
) {
    let (body, transform) = &mut *hiker;
    if std::mem::take(&mut controls.reset) {
        **camera = HikerCamera::default();
        transform.translation = start_position();
        transform.rotation = Quat::from_rotation_y(PI);
        **body = Hiker {
            grounded: true,
            ..default()
        };
        controls.jump = false;
        return;
    }

    let dt = time.delta_secs();
    let basis = Quat::from_rotation_y(camera.yaw);
    let relative = controls.movement.normalize_or_zero();
    let desired = (basis * Vec3::new(relative.x, 0.0, -relative.y)).xz()
        * if controls.run { RUN_SPEED } else { WALK_SPEED };
    let acceleration = if body.grounded { 13.0 } else { 3.2 };
    let response = 1.0 - (-acceleration * dt).exp();
    body.velocity = body.velocity.lerp(desired, response);
    let previous = transform.translation.xz();
    let next =
        (previous + body.velocity * dt).clamp(Vec2::new(-37.0, -62.0), Vec2::new(39.0, 24.0));
    let mut at = obstacles.collide(next, 0.29);
    let floor = ground(at);
    // Keep the hiker on dry ground, including while jumping.
    if floor < WATER_LEVEL - 0.05 {
        let slide_x = Vec2::new(at.x, previous.y);
        let slide_z = Vec2::new(previous.x, at.y);
        at = if ground(slide_x) >= WATER_LEVEL - 0.05 {
            slide_x
        } else if ground(slide_z) >= WATER_LEVEL - 0.05 {
            slide_z
        } else {
            previous
        };
    }
    body.velocity = (at - previous) / dt;
    transform.translation.x = at.x;
    transform.translation.z = at.y;

    let jump = std::mem::take(&mut controls.jump);
    let floor = ground(at);
    if jump && body.grounded {
        body.vertical_speed = 4.15;
        body.grounded = false;
    } else if body.grounded {
        transform.translation.y = floor;
    }
    if !body.grounded {
        body.vertical_speed -= 11.5 * dt;
        transform.translation.y += body.vertical_speed * dt;
        if transform.translation.y <= floor {
            transform.translation.y = floor;
            body.grounded = true;
            body.vertical_speed = 0.0;
        }
    }

    let heading = if body.velocity.length_squared() > 0.15 {
        body.velocity
    } else {
        desired
    };
    if heading.length_squared() > 0.15 {
        let target = Quat::from_rotation_y(heading.x.atan2(heading.y));
        transform.rotation = transform.rotation.slerp(target, 1.0 - (-11.0 * dt).exp());
    }
}

pub fn animate_hiker(
    hiker: Single<&Hiker>,
    mut animations: Query<(&HikerClips, &mut AnimationPlayer, &mut AnimationTransitions)>,
) {
    let speed = hiker.velocity.length();
    for (clips, mut player, mut transitions) in &mut animations {
        let target = if speed < 0.20 {
            clips.idle
        } else if speed >= 2.8 {
            clips.run
        } else {
            clips.walk
        };
        let playback = if target == clips.idle {
            1.0
        } else if target == clips.run {
            (speed / RUN_SPEED).clamp(0.64, 1.25)
        } else {
            (speed / WALK_SPEED).clamp(0.50, 1.32)
        };
        if transitions.get_main_animation() != Some(target) {
            transitions
                .play(&mut player, target, Duration::from_millis(230))
                .repeat();
        }
        if let Some(animation) = player.animation_mut(target) {
            animation.set_speed(if hiker.grounded {
                playback
            } else {
                playback * 0.25
            });
        }
    }
}

/// Every clip writes the arm rotations each frame, so this offset does not accumulate.
fn relax_arms(
    hiker: Single<&GlobalTransform, With<Hiker>>,
    parents: Query<&GlobalTransform>,
    mut arms: Query<(&RelaxedArm, &ChildOf, &mut Transform)>,
) {
    let forward = hiker.rotation() * Vec3::Z;
    for (arm, child_of, mut transform) in &mut arms {
        let Ok(parent) = parents.get(child_of.parent()) else {
            continue;
        };
        let axis = (parent.rotation().inverse() * forward).normalize();
        transform.rotation = Quat::from_axis_angle(axis, arm.0) * transform.rotation;
    }
}

pub fn follow_hiker(
    time: Res<Time>,
    obstacles: Res<Obstacles>,
    hiker: Single<&Transform, (With<Hiker>, Without<HikerCamera>)>,
    mut camera: Single<(&mut Transform, &HikerCamera), Without<Hiker>>,
) {
    let (transform, rig) = &mut *camera;
    let ideal = rig.pose(hiker.translation, &obstacles);
    let blend = 1.0 - (-9.5 * time.delta_secs()).exp();
    if transform.translation.distance_squared(ideal.translation) > 20.0_f32.powi(2) {
        **transform = ideal;
    } else {
        transform.translation = transform.translation.lerp(ideal.translation, blend);
        transform.rotation = transform.rotation.slerp(ideal.rotation, blend);
    }
    transform.translation =
        obstacles.camera_eye(hiker.translation + Vec3::Y * 1.56, transform.translation);
}
