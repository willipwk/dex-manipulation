import csv
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import nimblephysics as nimble
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from dataset import DexYCBVideoDataset
from dex_retargeting import yourdfpy as urdf


# Nimble automatically prepends a 6-DOF rootJoint to every loaded skeleton.
# DOF order within the joint: rotations first, then translations.
ROOT_JOINT_DOF_NAMES = [
    "rootJoint_rot_x",
    "rootJoint_rot_y",
    "rootJoint_rot_z",
    "rootJoint_pos_x",
    "rootJoint_pos_y",
    "rootJoint_pos_z",
]


@dataclass
class GraspOptimizationConfig:
    time_step: float = 0.001
    horizon: int = 20
    num_attempts: int = 4
    max_iters_per_attempt: int = 30
    learning_rate: float = 0.05
    torque_limit: float = 0.3
    displacement_threshold: float = 0.01
    pose_rot_weight: float = 0.03
    control_weight: float = 1e-4
    render_initial_pose: bool = True
    render_result: bool = True


@dataclass
class JointTrackingConfig:
    time_step: float = 0.001
    num_attempts: int = 3
    max_iters_per_attempt: int = 500
    learning_rate: float = 0.05
    torque_limit: float = 3.0
    joint_position_weight: float = 1.0
    control_weight: float = 1e-4
    render_result: bool = True
    frame_start: int = 15
    num_frames: int = 72
    # Path to write per-iteration loss log (CSV). None → no logging.
    loss_log_path: Path = None


@dataclass
class TrajectoryTrackingConfig:
    time_step: float = 0.001
    num_attempts: int = 3
    max_iters_per_attempt: int = 500
    learning_rate: float = 0.05
    torque_limit: float = 3.0
    position_weight: float = 1.0
    rotation_weight: float = 0.03
    control_weight: float = 1e-4
    render_result: bool = True
    # Roll out ID-initialised controls before optimisation to verify hand tracking.
    verify_id_tracking: bool = True
    # Dataset frame range to optimize over
    frame_start: int = 15
    num_frames: int = 72
    object_id: int = 1


@dataclass
class SceneAssets:
    hand_urdf: Path
    ground_urdf: Path
    object_urdf: Path = None  # optional — omit for hand-only scenes


def create_world(time_step: float):
    world: nimble.simulation.World = nimble.simulation.World()
    world.setGravity([0.0, 0.0, -9.8])
    world.setTimeStep(time_step)
    return world


def load_active_joint_names(joint_names_path: Path):
    with joint_names_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def create_mesh_object_urdf(object_name: str, mesh_file: Path, urdf_dir: Path):
    """Create a URDF for a mesh object.

    Nimble's collision detector cannot generate contact points from triangle-mesh
    geometry (getNumContacts() == 0 for mesh collision shapes).  We therefore use
    the mesh's axis-aligned bounding box as the collision geometry — a box
    primitive that Nimble handles correctly — while keeping the full mesh for
    the visual element.
    """
    import trimesh

    urdf_path = urdf_dir / f"{object_name}.urdf"
    mesh_path = mesh_file.resolve().as_posix()

    loaded = trimesh.load(str(mesh_file), force="mesh")
    extents = loaded.bounding_box.extents       # (3,) side lengths
    centroid = loaded.bounding_box.centroid     # (3,) centre offset from origin
    cx, cy, cz = (float(v) for v in centroid)
    sx, sy, sz = (float(v) for v in extents)

    urdf_contents = f"""<?xml version="1.0"?>
<robot name="{object_name}">
  <link name="{object_name}_link">
    <inertial>
      <origin xyz="{cx} {cy} {cz}" rpy="0 0 0"/>
      <mass value="0.10"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="{cx} {cy} {cz}" rpy="0 0 0"/>
      <geometry>
        <box size="{sx} {sy} {sz}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    urdf_path.write_text(urdf_contents, encoding="utf-8")
    return urdf_path


def create_free_joint_urdf(source_urdf: Path, urdf_dir: Path, output_name: str):
    robot_urdf = urdf.URDF.load(str(source_urdf), add_dummy_free_joints=True, build_scene_graph=False)
    output_path = urdf_dir / output_name
    robot_urdf.write_xml_file(str(output_path))
    return output_path


def create_ground_urdf(
    urdf_dir: Path,
    ground_name: str = "ground",
    size_xyz=(4.0, 4.0, 0.02),
    top_z: float = 0.0,
):
    urdf_path = urdf_dir / f"{ground_name}.urdf"
    size_x, size_y, size_z = (float(v) for v in size_xyz)
    center_z = top_z - size_z / 2.0
    urdf_contents = f"""<?xml version="1.0"?>
<robot name="{ground_name}">
  <link name="{ground_name}_link">
    <inertial>
      <origin xyz="0 0 {center_z}" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {center_z}" rpy="0 0 0"/>
      <geometry>
        <box size="{size_x} {size_y} {size_z}"/>
      </geometry>
      <material name="ground_gray">
        <color rgba="0.6 0.6 0.6 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {center_z}" rpy="0 0 0"/>
      <geometry>
        <box size="{size_x} {size_y} {size_z}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    urdf_path.write_text(urdf_contents, encoding="utf-8")
    return urdf_path


def compute_object_pose(object_pose_quat_init: np.ndarray, camera_extrinsics_inv: np.ndarray):
    object_pose_matrix = np.eye(4)
    object_pose_matrix[0:3, 0:3] = R.from_quat(object_pose_quat_init[0:4], scalar_first=False).as_matrix()
    object_pose_matrix[0:3, 3] = object_pose_quat_init[4:7]
    object_pose_matrix = camera_extrinsics_inv @ object_pose_matrix
    object_position = object_pose_matrix[0:3, 3]
    object_rotvec = R.from_matrix(object_pose_matrix[0:3, 0:3]).as_rotvec()
    return object_position, object_rotvec


def get_dof_indices(skeleton: nimble.dynamics.Skeleton, dof_names) -> np.ndarray:
    """Return skeleton-local DOF indices looked up by DOF name.

    Works for both multi-DOF joints (e.g. rootJoint) and single-DOF joints.
    """
    indices = []
    for name in dof_names:
        dof = skeleton.getDof(name)
        if dof is None:
            raise ValueError(f"DOF '{name}' not found in skeleton '{skeleton.getName()}'.")
        indices.append(dof.getIndexInSkeleton())
    return np.asarray(indices, dtype=int)


def set_dof_positions(skeleton: nimble.dynamics.Skeleton, dof_names, dof_values):
    """Set skeleton DOF positions by DOF name."""
    for name, value in zip(dof_names, dof_values):
        dof = skeleton.getDof(name)
        if dof is None:
            raise ValueError(f"DOF '{name}' not found in skeleton '{skeleton.getName()}'.")
        skeleton.setPosition(dof.getIndexInSkeleton(), float(value))


def torch_wrap_to_pi(values: torch.Tensor):
    return torch.remainder(values + torch.pi, 2.0 * torch.pi) - torch.pi


def add_world_axes(gui: nimble.NimbleGUI, axis_length: float = 0.2):
    origin = np.zeros(3)
    axes = [
        ("world_x_axis", np.array([axis_length, 0.0, 0.0]), [1.0, 0.0, 0.0, 1.0]),
        ("world_y_axis", np.array([0.0, axis_length, 0.0]), [0.0, 1.0, 0.0, 1.0]),
        ("world_z_axis", np.array([0.0, 0.0, axis_length]), [0.0, 0.0, 1.0, 1.0]),
    ]
    for key, endpoint, color in axes:
        gui.nativeAPI().createLine(key=key, points=[origin, endpoint], color=color)


def prepare_scene_assets(
    temp_dir_path: Path,
    hand_urdf: Path,
    object_mesh_file: Path,
    object_name: str,
):
    # Load original URDFs directly — nimble attaches rootJoint automatically to
    # every skeleton, so no dummy free-joint wrapper is needed.
    return SceneAssets(
        hand_urdf=hand_urdf,
        ground_urdf=create_ground_urdf(temp_dir_path),
        object_urdf=create_mesh_object_urdf(object_name, object_mesh_file, temp_dir_path),
    )


def build_grasp_scene(
    assets: SceneAssets,
    config,
    hand_finger_joint_names,
    hand_qpos_t0: np.ndarray,
    object_pose_t0: np.ndarray,
):
    """Construct a nimble world at time 0.

    Args:
        hand_finger_joint_names: DOF names for the finger joints (excludes rootJoint).
        hand_qpos_t0: (6 + N_fingers,) array.  First 6 values are rootJoint DOFs
            in ROOT_JOINT_DOF_NAMES order [rot_x, rot_y, rot_z, pos_x, pos_y, pos_z];
            remaining values are finger DOFs in hand_finger_joint_names order.
        object_pose_t0: (6,) array in ROOT_JOINT_DOF_NAMES order
            [rot_x, rot_y, rot_z, pos_x, pos_y, pos_z].
    """
    world = create_world(config.time_step)
    ground = world.loadSkeleton(str(assets.ground_urdf))
    ground.setMobile(False)

    leap = world.loadSkeleton(str(assets.hand_urdf))
    set_dof_positions(leap, ROOT_JOINT_DOF_NAMES, hand_qpos_t0[:6])
    set_dof_positions(leap, hand_finger_joint_names, hand_qpos_t0[6:])
    leap.setVelocities(np.zeros(leap.getNumDofs()))

    obj = world.loadSkeleton(str(assets.object_urdf))
    set_dof_positions(obj, ROOT_JOINT_DOF_NAMES, object_pose_t0)
    obj.setVelocities(np.zeros(obj.getNumDofs()))

    return world, leap, obj


def build_hand_only_scene(
    hand_urdf: Path,
    ground_urdf: Path,
    config,
    hand_finger_joint_names,
    hand_qpos_t0: np.ndarray,
):
    """Construct a nimble world with ground and hand only (no object)."""
    world = create_world(config.time_step)
    ground = world.loadSkeleton(str(ground_urdf))
    ground.setMobile(False)
    leap = world.loadSkeleton(str(hand_urdf))
    set_dof_positions(leap, ROOT_JOINT_DOF_NAMES, hand_qpos_t0[:6])
    set_dof_positions(leap, hand_finger_joint_names, hand_qpos_t0[6:])
    leap.setVelocities(np.zeros(leap.getNumDofs()))
    return world, leap


def rollout_grasp_torch(
    assets: SceneAssets,
    config: GraspOptimizationConfig,
    hand_state_joint_names,
    hand_state_qpos: np.ndarray,
    hand_actuated_joint_names,
    object_position: np.ndarray,
    object_euler: np.ndarray,
    finger_controls: torch.Tensor,
):
    world, leap, obj = build_grasp_scene(
        assets=assets,
        config=config,
        hand_state_joint_names=hand_state_joint_names,
        hand_state_qpos=hand_state_qpos,
        object_position=object_position,
        object_euler=object_euler,
    )

    world_dofs = world.getNumDofs()
    hand_indices_np = get_joint_dof_indices(leap, hand_actuated_joint_names)
    hand_indices = torch.tensor(hand_indices_np.tolist(), dtype=torch.long)
    object_position_slice = slice(leap.getNumDofs(), leap.getNumDofs() + len(DUMMY_FREE_JOINT_NAMES))

    state = torch.tensor(world.getState(), dtype=torch.float32)
    states = [state.detach().clone()]

    for step_idx in range(config.horizon):
        action = torch.zeros((world_dofs,), dtype=torch.float32)
        action[hand_indices] = finger_controls[step_idx]
        state = nimble.timestep(world, state, action)
        states.append(state.detach().clone())

    initial_object_pose = states[0][:world_dofs][object_position_slice]
    final_object_pose = state[:world_dofs][object_position_slice]
    position_delta = final_object_pose[:3] - initial_object_pose[:3]
    rotation_delta = torch_wrap_to_pi(final_object_pose[3:] - initial_object_pose[3:])
    loss = (
        torch.sum(position_delta * position_delta)
        + config.pose_rot_weight * torch.sum(rotation_delta * rotation_delta)
        + config.control_weight * torch.mean(finger_controls * finger_controls)
    )
    displacement = torch.linalg.norm(position_delta)

    return {
        "world": world,
        "states": states,
        "loss": loss,
        "displacement": displacement,
        "position_delta": position_delta,
        "rotation_delta": rotation_delta,
        "initial_object_pose": initial_object_pose,
        "final_object_pose": final_object_pose,
    }


def optimize_grasp_control(
    assets: SceneAssets,
    config: GraspOptimizationConfig,
    hand_state_joint_names,
    hand_state_qpos: np.ndarray,
    hand_actuated_joint_names,
    object_position: np.ndarray,
    object_euler: np.ndarray,
):
    finger_dofs = len(hand_actuated_joint_names)
    global_best = None

    for attempt_idx in range(config.num_attempts):
        finger_controls = torch.nn.Parameter(
            0.03 * (attempt_idx + 1) * torch.randn((config.horizon, finger_dofs), dtype=torch.float32)
        )
        optimizer = torch.optim.Adam([finger_controls], lr=config.learning_rate)
        best_attempt_result = None

        for iter_idx in range(config.max_iters_per_attempt):
            optimizer.zero_grad()
            controls = torch.clamp(finger_controls, -config.torque_limit, config.torque_limit)
            result = rollout_grasp_torch(
                assets=assets,
                config=config,
                hand_state_joint_names=hand_state_joint_names,
                hand_state_qpos=hand_state_qpos,
                hand_actuated_joint_names=hand_actuated_joint_names,
                object_position=object_position,
                object_euler=object_euler,
                finger_controls=controls,
            )
            result["loss"].backward()
            optimizer.step()
            with torch.no_grad():
                finger_controls.clamp_(-config.torque_limit, config.torque_limit)

            scalar_loss = float(result["loss"].detach().item())
            scalar_displacement = float(result["displacement"].detach().item())

            if best_attempt_result is None or scalar_loss < best_attempt_result["loss"]:
                best_attempt_result = {
                    "world": result["world"],
                    "states": [state.detach().clone() for state in result["states"]],
                    "loss": scalar_loss,
                    "displacement": scalar_displacement,
                    "position_delta": result["position_delta"].detach().clone(),
                    "rotation_delta": result["rotation_delta"].detach().clone(),
                    "initial_object_pose": result["initial_object_pose"].detach().clone(),
                    "final_object_pose": result["final_object_pose"].detach().clone(),
                    "controls": controls.detach().clone(),
                }

            print(
                f"attempt={attempt_idx + 1}/{config.num_attempts} "
                f"iter={iter_idx + 1}/{config.max_iters_per_attempt} "
                f"loss={scalar_loss:.6f} "
                f"disp={scalar_displacement:.6f}"
            )
            if scalar_displacement <= config.displacement_threshold:
                break

        if global_best is None or best_attempt_result["loss"] < global_best["loss"]:
            global_best = best_attempt_result
        if global_best["displacement"] <= config.displacement_threshold:
            break

    return global_best


def load_problem_setup(root_dir: Path):
    dataset = DexYCBVideoDataset("/localhome/wpa15/Projects/realart/dex-retargeting/DexYCB", hand_type="right")
    data_id = 4
    sample = dataset[data_id]
    for key, value in sample.items():
        if "pose" not in key:
            print(f"{key}: {value}")

    hand_qpos_all = np.load(root_dir / "leap_hand_retarget_qpos.npy")
    active_joint_names = load_active_joint_names(root_dir / "leap_hand_retarget_active_joint_names.txt")
    hand_step = 50
    object_step = hand_step + 5
    print("hand pose len:", len(hand_qpos_all))
    print("object pose len:", len(sample["object_pose"]))
    object_id = 1

    camera_extrinsics_inv = np.linalg.inv(sample["extrinsics"])
    object_pose_quat_init = sample["object_pose"][object_step][object_id]
    object_position, object_euler = compute_object_pose(object_pose_quat_init, camera_extrinsics_inv)

    return {
        "hand_state_joint_names": active_joint_names,
        "hand_state_qpos": hand_qpos_all[hand_step],
        "hand_actuated_joint_names": active_joint_names[len(DUMMY_FREE_JOINT_NAMES) :],
        "object_mesh_file": Path(sample["object_mesh_file"][object_id]),
        "object_name": f"ycb_{sample['ycb_ids'][object_id]}_{object_id}",
        "object_position": object_position,
        "object_euler": object_euler,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory tracking: optimize control over the full manipulation sequence
# ──────────────────────────────────────────────────────────────────────────────


def compute_object_trajectory(
    object_pose_seq: np.ndarray,
    camera_extrinsics_inv: np.ndarray,
    object_id: int,
    frame_start: int,
    frame_end: int,
) -> np.ndarray:
    """Convert a contiguous range of object poses from camera frame to world frame.

    Returns an array of shape (T, 6) in ROOT_JOINT_DOF_NAMES order:
    [rotvec_x, rotvec_y, rotvec_z, pos_x, pos_y, pos_z] per frame.
    Rotation is represented as a rotation vector (exponential map) matching
    Nimble's FreeJoint convention.
    """
    poses = []
    for t in range(frame_start, frame_end):
        pos, euler = compute_object_pose(object_pose_seq[t][object_id], camera_extrinsics_inv)
        poses.append(np.concatenate([euler, pos]))  # rootJoint DOF order: rot first, then pos
    return np.stack(poses)


def rollout_trajectory_torch(
    assets: SceneAssets,
    config: TrajectoryTrackingConfig,
    hand_finger_joint_names,
    hand_qpos_seq: np.ndarray,
    desired_object_traj: np.ndarray,
    finger_controls: torch.Tensor,
    collect_states: bool = False,
    gui: nimble.NimbleGUI = None,
):
    """Roll out the full manipulation sequence and return the trajectory tracking loss.

    The hand's rootJoint (6 DOFs) is driven kinematically at each step to follow
    ``hand_qpos_seq[:, :6]``.  Finger joints receive the learnable
    ``finger_controls`` torques.  The loss is the sum of squared distances between
    the simulated object pose and ``desired_object_traj`` at every step.

    Args:
        hand_finger_joint_names: DOF names for the finger joints (excludes rootJoint).
        hand_qpos_seq: (T, 6+N) array.  Columns 0:6 are rootJoint DOFs in
            ROOT_JOINT_DOF_NAMES order [rot, pos]; columns 6: are finger DOFs.
        desired_object_traj: (T, 6) in ROOT_JOINT_DOF_NAMES order [rot, pos].
        finger_controls: (T-1, N) learnable torque tensor.
        collect_states: if True, all intermediate states are stored in the result.

    Returns:
        dict with keys "world", "loss", "states" (list if collect_states else None).
    """
    T = len(hand_qpos_seq)

    world, leap, obj = build_grasp_scene(
        assets=assets,
        config=config,
        hand_finger_joint_names=hand_finger_joint_names,
        hand_qpos_t0=hand_qpos_seq[0],
        object_pose_t0=desired_object_traj[0],
    )

    world_dofs = world.getNumDofs()
    hand_num_dofs = leap.getNumDofs()
    obj_num_dofs = obj.getNumDofs()  # 6 (rootJoint only)

    # The world state concatenates skeleton DOFs in load order: ground, leap, object.
    # Ground may still occupy slots even when mobile=False, so derive the leap
    # world-level offset from DOF counts rather than assuming it starts at 0.
    hand_dof_offset = world_dofs - hand_num_dofs - obj_num_dofs
    object_dof_start = hand_dof_offset + hand_num_dofs

    # World-level DOF indices for rootJoint and finger joints.
    root_indices_world = get_dof_indices(leap, ROOT_JOINT_DOF_NAMES) + hand_dof_offset
    finger_indices_world = get_dof_indices(leap, hand_finger_joint_names) + hand_dof_offset
    finger_indices = torch.tensor(finger_indices_world.tolist(), dtype=torch.long)

    # Kinematic override mask: 1 at rootJoint DOF positions, 0 elsewhere.
    free_mask = torch.zeros(world_dofs, dtype=torch.float32)
    free_mask[root_indices_world] = 1.0
    dyn_mask = 1.0 - free_mask

    state = torch.tensor(world.getState(), dtype=torch.float32)
    states = [state.detach().clone()] if collect_states else None
    total_loss = torch.tensor(0.0, dtype=torch.float32)

    for t in range(T - 1):
        if gui is not None:
            gui.displayState(state)
        # Apply learned finger torques; rootJoint is left unactuated.
        action = torch.zeros(world_dofs, dtype=torch.float32)
        action[finger_indices] = finger_controls[t]

        state = nimble.timestep(world, state, action)

        # Kinematically advance rootJoint to the reference pose at t+1.
        desired_root = torch.tensor(hand_qpos_seq[t + 1, :6], dtype=torch.float32)
        prev_root = torch.tensor(hand_qpos_seq[t, :6], dtype=torch.float32)
        desired_root_vel = (desired_root - prev_root) / config.time_step

        desired_root_world = torch.zeros(world_dofs, dtype=torch.float32)
        desired_root_world[root_indices_world] = desired_root
        desired_root_vel_world = torch.zeros(world_dofs, dtype=torch.float32)
        desired_root_vel_world[root_indices_world] = desired_root_vel

        # Replace rootJoint entries; keep physics-simulated finger and object DOFs.
        new_pos = state[:world_dofs] * dyn_mask + desired_root_world * free_mask
        new_vel = state[world_dofs:] * dyn_mask + desired_root_vel_world * free_mask
        state = torch.cat([new_pos, new_vel])

        if collect_states:
            states.append(state.detach().clone())

        # Object tracking loss at step t+1.
        # State layout matches rootJoint DOF order: [rot_x, rot_y, rot_z, pos_x, pos_y, pos_z]
        obj_pose_sim = state[object_dof_start : object_dof_start + 6]
        desired_obj_pose = torch.tensor(desired_object_traj[t + 1], dtype=torch.float32)
        rot_loss = torch.sum(torch_wrap_to_pi(obj_pose_sim[:3] - desired_obj_pose[:3]) ** 2)
        pos_loss = torch.sum((obj_pose_sim[3:] - desired_obj_pose[3:]) ** 2)
        total_loss = total_loss + config.rotation_weight * rot_loss + config.position_weight * pos_loss

    total_loss = total_loss + config.control_weight * torch.mean(finger_controls * finger_controls)

    return {"world": world, "loss": total_loss, "states": states}


def rollout_hand_tracking(
    assets: SceneAssets,
    config: TrajectoryTrackingConfig,
    hand_finger_joint_names,
    hand_qpos_seq: np.ndarray,
    desired_object_traj: np.ndarray,
    finger_controls: torch.Tensor,
    gui: nimble.NimbleGUI = None,
) -> dict:
    """Simulate the hand under the given finger controls and measure how well it
    tracks the reference finger joint trajectory.  The object is moved
    kinematically along ``desired_object_traj`` so the GUI shows both the
    simulated hand motion and the target object path together.

    Runs in numpy (no gradients) — fast, no side-effects on the optimiser.

    Args:
        hand_qpos_seq: (T, 6+N) reference trajectory. Columns 0:6 are rootJoint
            DOFs; columns 6: are finger DOFs in ``hand_finger_joint_names`` order.
        desired_object_traj: (T, 6) desired object poses in ROOT_JOINT_DOF_NAMES
            order [rot, pos].  The object is set to these poses each step so it
            appears at the correct location in the GUI.
        finger_controls: (T-1, N) control tensor (e.g. from ID init or optimizer).
        gui: if provided, ``loopStates`` is called with the collected states.

    Returns:
        dict with keys:
          "finger_pos_sim"  – (T, N) simulated finger positions
          "finger_pos_ref"  – (T, N) reference finger positions (hand_qpos_seq[:, 6:])
          "error_per_step"  – (T,)   mean absolute error per step across all finger DOFs
          "states"          – list of (2*world_dofs,) numpy arrays for GUI playback
    """
    T = len(hand_qpos_seq)
    controls_np = finger_controls.detach().cpu().numpy()

    # Build world with object placed at the first reference pose.
    world, leap, obj = build_grasp_scene(
        assets=assets,
        config=config,
        hand_finger_joint_names=hand_finger_joint_names,
        hand_qpos_t0=hand_qpos_seq[0],
        object_pose_t0=desired_object_traj[0],
    )

    world_dofs = world.getNumDofs()
    hand_num_dofs = leap.getNumDofs()
    obj_num_dofs = obj.getNumDofs()  # 6
    hand_dof_offset = world_dofs - hand_num_dofs - obj_num_dofs
    object_dof_start = hand_dof_offset + hand_num_dofs

    root_idx_world = get_dof_indices(leap, ROOT_JOINT_DOF_NAMES) + hand_dof_offset
    finger_idx_world = get_dof_indices(leap, hand_finger_joint_names) + hand_dof_offset
    finger_idx_local = get_dof_indices(leap, hand_finger_joint_names)
    # Object rootJoint DOF indices in world state (positions only).
    obj_root_idx_world = np.arange(obj_num_dofs) + object_dof_start

    state = world.getState().copy()
    all_states = [state.copy()]
    finger_pos_sim = np.zeros((T, len(hand_finger_joint_names)))
    finger_pos_sim[0] = hand_qpos_seq[0, 6:]

    for t in range(T - 1):
        if gui is not None:
            state_torch = torch.tensor(state, dtype=torch.float32)
            gui.displayState(state_torch)
        action = np.zeros(world_dofs)
        action[finger_idx_world] = controls_np[t]

        world.setState(state)
        # world.setAction(action)
        # world.step()
        state = world.getState().copy()

        # Kinematically override hand rootJoint.
        desired_root = hand_qpos_seq[t + 1, :6]
        prev_root = hand_qpos_seq[t, :6]
        state[root_idx_world] = desired_root
        state[world_dofs + root_idx_world] = (desired_root - prev_root) / config.time_step

        # Kinematically set object to reference pose at t+1 so it appears
        # correctly in the GUI alongside the simulated hand.
        desired_obj = desired_object_traj[t + 1]
        prev_obj = desired_object_traj[t]
        state[obj_root_idx_world] = desired_obj
        state[world_dofs + obj_root_idx_world] = (desired_obj - prev_obj) / config.time_step

        all_states.append(state.copy())

        # world.setState(state)
        # finger_pos_sim[t + 1] = leap.getPositions()[finger_idx_local]

    finger_pos_ref = hand_qpos_seq[:, 6:]
    error = np.abs(finger_pos_sim - finger_pos_ref)
    error_per_step = error.mean(axis=1)

    print(f"Hand tracking — mean error: {error_per_step.mean():.4f} rad  "
          f"max error: {error_per_step.max():.4f} rad  "
          f"(over {T} steps, {len(hand_finger_joint_names)} finger DOFs)")

    if gui is not None:
        gui.loopStates([torch.tensor(s, dtype=torch.float32) for s in all_states])

    return {
        "finger_pos_sim": finger_pos_sim,
        "finger_pos_ref": finger_pos_ref,
        "error_per_step": error_per_step,
        "states": all_states,
    }


def rollout_joint_tracking_torch(
    hand_urdf: Path,
    ground_urdf: Path,
    config: JointTrackingConfig,
    hand_finger_joint_names,
    hand_qpos_seq: np.ndarray,
    finger_controls: torch.Tensor,
    collect_states: bool = False,
    gui: nimble.NimbleGUI = None,
):
    """Roll out the hand under finger torques and return the joint tracking loss.

    The hand's rootJoint (6 DOFs) is driven kinematically at each step to follow
    ``hand_qpos_seq[:, :6]``.  Finger joints receive the learnable torques.
    The loss is the sum of squared errors between simulated and reference finger
    joint positions at every step.  No object is involved.

    Args:
        hand_qpos_seq: (T, 6+N) reference trajectory.  Columns 0:6 are rootJoint
            DOFs in ROOT_JOINT_DOF_NAMES order; columns 6: are finger DOFs.
        finger_controls: (T-1, N) learnable torque tensor.

    Returns:
        dict with keys "world", "loss", "states" (list if collect_states else None),
        and "per_step_metrics" (list of dicts with joint_rmse, tau_rms
        if collect_states else None).
    """
    T = len(hand_qpos_seq)
    world, leap = build_hand_only_scene(hand_urdf, ground_urdf, config, hand_finger_joint_names, hand_qpos_seq[0])

    world_dofs = world.getNumDofs()
    hand_num_dofs = leap.getNumDofs()
    hand_dof_offset = world_dofs - hand_num_dofs

    root_indices_world = get_dof_indices(leap, ROOT_JOINT_DOF_NAMES) + hand_dof_offset
    finger_indices_world = get_dof_indices(leap, hand_finger_joint_names) + hand_dof_offset
    finger_indices = torch.tensor(finger_indices_world.tolist(), dtype=torch.long)

    free_mask = torch.zeros(world_dofs, dtype=torch.float32)
    free_mask[root_indices_world] = 1.0
    dyn_mask = 1.0 - free_mask

    state = torch.tensor(world.getState(), dtype=torch.float32)
    states = [state.detach().clone()] if collect_states else None
    per_step_metrics = [] if collect_states else None
    total_loss = torch.tensor(0.0, dtype=torch.float32)

    for t in range(T - 1):
        if gui is not None:
            gui.displayState(state)
            time.sleep(0.02)
        action = torch.zeros(world_dofs, dtype=torch.float32)
        action[finger_indices] = finger_controls[t]
        state = nimble.timestep(world, state, action)

        # Kinematically advance rootJoint to reference pose at t+1.
        desired_root = torch.tensor(hand_qpos_seq[t + 1, :6], dtype=torch.float32)
        prev_root = torch.tensor(hand_qpos_seq[t, :6], dtype=torch.float32)
        desired_root_vel = (desired_root - prev_root) / config.time_step

        desired_root_world = torch.zeros(world_dofs, dtype=torch.float32)
        desired_root_world[root_indices_world] = desired_root
        desired_root_vel_world = torch.zeros(world_dofs, dtype=torch.float32)
        desired_root_vel_world[root_indices_world] = desired_root_vel

        new_pos = state[:world_dofs] * dyn_mask + desired_root_world * free_mask
        new_vel = state[world_dofs:] * dyn_mask + desired_root_vel_world * free_mask
        state = torch.cat([new_pos, new_vel])

        if collect_states:
            states.append(state.detach().clone())

        # Finger joint position tracking loss at step t+1.
        finger_pos_sim = state[finger_indices]
        finger_pos_ref = torch.tensor(hand_qpos_seq[t + 1, 6:], dtype=torch.float32)
        joint_loss = torch.sum((finger_pos_sim - finger_pos_ref) ** 2)
        total_loss = total_loss + config.joint_position_weight * joint_loss

        # Per-step metrics (joint_rmse, tau_rms)
        if collect_states:
            joint_rmse = float(((finger_pos_sim - finger_pos_ref) ** 2).mean() ** 0.5)
            tau_rms = float((finger_controls[t] ** 2).mean() ** 0.5)
            per_step_metrics.append({
                "frame": t + 1,
                "joint_rmse": joint_rmse,
                "tau_rms": tau_rms,
            })

    total_loss = total_loss + config.control_weight * torch.mean(finger_controls * finger_controls)

    return {"world": world, "loss": total_loss, "states": states, "per_step_metrics": per_step_metrics}


def optimize_joint_tracking(
    hand_urdf: Path,
    ground_urdf: Path,
    config: JointTrackingConfig,
    hand_finger_joint_names,
    hand_qpos_sequence: np.ndarray,
    gui: nimble.NimbleGUI = None,
):
    """Optimize finger torques to track the reference finger joint positions.

    Returns a dict with keys "loss" and "controls".
    Writes a CSV loss log to ``config.loss_log_path`` if set.
    """
    T = len(hand_qpos_sequence)
    finger_dofs = len(hand_finger_joint_names)
    global_best = None

    log_file = None
    log_writer = None
    if config.loss_log_path is not None:
        log_file = open(config.loss_log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["attempt", "iter", "loss"])
        log_file.flush()
        print(f"Loss log → {config.loss_log_path}")

    try:
        for attempt_idx in range(config.num_attempts):
            finger_controls = torch.nn.Parameter(
                0.01 * (attempt_idx + 1) * torch.randn((T - 1, finger_dofs), dtype=torch.float32)
            )
            optimizer = torch.optim.Adam([finger_controls], lr=config.learning_rate)
            best_attempt = None

            for iter_idx in range(config.max_iters_per_attempt):
                optimizer.zero_grad()
                controls = torch.clamp(finger_controls, -config.torque_limit, config.torque_limit)
                result = rollout_joint_tracking_torch(
                    hand_urdf=hand_urdf,
                    ground_urdf=ground_urdf,
                    config=config,
                    hand_finger_joint_names=hand_finger_joint_names,
                    hand_qpos_seq=hand_qpos_sequence,
                    finger_controls=controls,
                    gui=gui,
                )
                result["loss"].backward()
                optimizer.step()
                with torch.no_grad():
                    finger_controls.clamp_(-config.torque_limit, config.torque_limit)

                scalar_loss = float(result["loss"].detach().item())
                print(
                    f"attempt={attempt_idx + 1}/{config.num_attempts} "
                    f"iter={iter_idx + 1}/{config.max_iters_per_attempt} "
                    f"loss={scalar_loss:.6f}"
                )
                if log_writer is not None:
                    log_writer.writerow([attempt_idx + 1, iter_idx + 1, scalar_loss])
                    log_file.flush()

                if best_attempt is None or scalar_loss < best_attempt["loss"]:
                    best_attempt = {
                        "loss": scalar_loss,
                        "controls": controls.detach().clone(),
                    }

            if global_best is None or best_attempt["loss"] < global_best["loss"]:
                global_best = best_attempt
    finally:
        if log_file is not None:
            log_file.close()

    return global_best


def compute_id_initial_controls(
    assets: SceneAssets,
    config: TrajectoryTrackingConfig,
    hand_finger_joint_names,
    hand_qpos_seq: np.ndarray,
) -> torch.Tensor:
    """Initialise finger controls via nimble's inverse-dynamics API.

    Uses ``skeleton.getMultipleContactInverseDynamicsOverTime`` which processes
    the whole position sequence at once, adds temporal smoothing across steps,
    and minimises total torque magnitude — giving a better warm-start than
    computing each timestep independently.

    The hand's rootJoint is kinematically controlled during optimisation, so we
    pass no external contact bodies; the resulting root-DOF torques are simply
    discarded and only finger-DOF entries are kept.

    Returns:
        (T-1, N_fingers) float32 tensor, clamped to ``config.torque_limit``.
    """
    T = len(hand_qpos_seq)

    # Build a temporary world — only the leap skeleton is needed.
    _, leap, _ = build_grasp_scene(
        assets=assets,
        config=config,
        hand_finger_joint_names=hand_finger_joint_names,
        hand_qpos_t0=hand_qpos_seq[0],
        object_pose_t0=np.zeros(6),
    )

    # Skeleton-local DOF indices (no world offset needed here).
    root_local = get_dof_indices(leap, ROOT_JOINT_DOF_NAMES)
    finger_local = get_dof_indices(leap, hand_finger_joint_names)

    # Build the (T, N_skeleton_dofs) position array in skeleton-local DOF order.
    positions = np.zeros((T, leap.getNumDofs()), dtype=np.float64)
    positions[:, root_local] = hand_qpos_seq[:, :6]
    positions[:, finger_local] = hand_qpos_seq[:, 6:]
    print("positions shape:", positions.shape)

    # Full-sequence inverse dynamics with temporal smoothing.
    # No contact bodies: the root is kinematic so no external contact wrench is
    # needed to explain the motion.
    # velocityPenalty(v) penalises large joint velocities in the ID solution;
    # a small constant keeps the problem well-conditioned without over-constraining.
    leap_hand_nodes = [leap.getBodyNode(i) for i in range(leap.getNumBodyNodes())]
    result = leap.getMultipleContactInverseDynamicsOverTime(
        positions.T,
        leap_hand_nodes,              # contactBodies
        smoothingWeight=1.0,            # smoothingWeight
        minTorqueWeight=1.0,            # minTorqueWeight
        velocityPenalty=lambda v: 1e-4 * v,            # velocityPenalty
    )

    # result.jointTorques[t, 0] is the full skeleton torque vector at step t.
    print(type(result.jointTorques))
    print(result.jointTorques.shape)
    # controls = np.stack(
    #     [result.jointTorques[t, 0][finger_local] for t in range(T - 1)]
    # )
    controls = result.jointTorques.T
    controls = controls[:, finger_local]
    controls = np.vstack([controls, controls[-1:]])  # repeat last step to get T rows
    controls = np.clip(controls, -config.torque_limit, config.torque_limit)
    return torch.tensor(controls, dtype=torch.float32)


def optimize_trajectory_control(
    assets: SceneAssets,
    config: TrajectoryTrackingConfig,
    hand_finger_joint_names,
    hand_qpos_sequence: np.ndarray,
    desired_object_traj: np.ndarray,
    gui: nimble.NimbleGUI = None,
):
    """Optimize finger torques over the full manipulation sequence.

    Returns a dict with keys "loss" and "controls".
    """
    T = len(hand_qpos_sequence)
    finger_dofs = len(hand_finger_joint_names)
    global_best = None

    # Compute inverse-dynamics warm-start once; reused for the first attempt.
    print("Computing inverse-dynamics initialisation...")
    id_controls = compute_id_initial_controls(
        assets=assets,
        config=config,
        hand_finger_joint_names=hand_finger_joint_names,
        hand_qpos_seq=hand_qpos_sequence,
    )
    print(f"  ID controls: mean={id_controls.abs().mean():.4f}  max={id_controls.abs().max():.4f}")

    for attempt_idx in range(config.num_attempts):
        if attempt_idx == 0:
            # Warm-start from inverse dynamics.
            finger_controls = torch.nn.Parameter(id_controls.clone())
        else:
            # Subsequent attempts explore with increasing random perturbations.
            finger_controls = torch.nn.Parameter(
                id_controls + 0.01 * attempt_idx * torch.randn((T - 1, finger_dofs), dtype=torch.float32)
            )
        optimizer = torch.optim.Adam([finger_controls], lr=config.learning_rate)
        best_attempt = None

        for iter_idx in range(config.max_iters_per_attempt):
            optimizer.zero_grad()
            controls = torch.clamp(finger_controls, -config.torque_limit, config.torque_limit)
            result = rollout_trajectory_torch(
                assets=assets,
                config=config,
                hand_finger_joint_names=hand_finger_joint_names,
                hand_qpos_seq=hand_qpos_sequence,
                desired_object_traj=desired_object_traj,
                finger_controls=controls,
                gui=gui
            )
            result["loss"].backward()
            optimizer.step()
            with torch.no_grad():
                finger_controls.clamp_(-config.torque_limit, config.torque_limit)

            scalar_loss = float(result["loss"].detach().item())
            print(
                f"attempt={attempt_idx + 1}/{config.num_attempts} "
                f"iter={iter_idx + 1}/{config.max_iters_per_attempt} "
                f"loss={scalar_loss:.6f}"
            )

            if best_attempt is None or scalar_loss < best_attempt["loss"]:
                best_attempt = {
                    "loss": scalar_loss,
                    "controls": controls.detach().clone(),
                }

        if global_best is None or best_attempt["loss"] < global_best["loss"]:
            global_best = best_attempt

    return global_best


def load_sequence_setup(root_dir: Path, config: TrajectoryTrackingConfig):
    """Load the hand qpos sequence and desired object trajectory for a dataset clip."""
    dataset = DexYCBVideoDataset("/localhome/wpa15/Projects/realart/dex-retargeting/DexYCB", hand_type="right")
    data_id = 4
    sample = dataset[data_id]

    hand_qpos_all = np.load(root_dir / "leap_hand_retarget_qpos.npy")
    active_joint_names = load_active_joint_names(root_dir / "leap_hand_retarget_active_joint_names.txt")
    # Stored dummy free-joint DOF order: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
    # Nimble's FreeJoint rootJoint uses a rotation vector (exponential map) for DOFs 0-2,
    # NOT Euler angles.  Convert stored XYZ Euler angles → rotation vector.
    finger_joint_names = active_joint_names[6:]

    frame_start = config.frame_start
    frame_end = min(frame_start + config.num_frames, len(hand_qpos_all), len(sample["object_pose"]))
    print(f"Sequence frames: {frame_start} to {frame_end} ({frame_end - frame_start} steps)")

    start_index = len(sample["object_pose"]) - len(hand_qpos_all)
    print(f"hand_qpos_all len: {len(hand_qpos_all)}, object_pose len: {len(sample['object_pose'])}, start_index: {start_index}")
    raw_seq = hand_qpos_all[frame_start - start_index:frame_end - start_index]
    euler_xyz = raw_seq[:, 3:6]
    rot_rotvec = R.from_euler("XYZ", euler_xyz).as_rotvec()
    hand_qpos_seq = np.concatenate([rot_rotvec, raw_seq[:, 0:3], raw_seq[:, 6:]], axis=1)

    camera_extrinsics_inv = np.linalg.inv(sample["extrinsics"])
    desired_object_traj = compute_object_trajectory(
        sample["object_pose"], camera_extrinsics_inv, config.object_id, frame_start, frame_end
    )

    assert len(hand_qpos_seq) == len(desired_object_traj), "Hand and object trajectories must have the same length. got {} and {}, respectively.".format(len(hand_qpos_seq), len(desired_object_traj))

    return {
        "hand_finger_joint_names": finger_joint_names,
        "hand_qpos_sequence": hand_qpos_seq,
        "desired_object_traj": desired_object_traj,
        "object_mesh_file": Path(sample["object_mesh_file"][config.object_id]),
        "object_name": f"ycb_{sample['ycb_ids'][config.object_id]}_{config.object_id}",
    }


def load_hand_sequence(root_dir: Path, config: JointTrackingConfig):
    """Load the hand qpos sequence and object trajectory from saved retargeting data.

    Stored format (dummy free-joint DOFs):
      cols 0:3 → [pos_x, pos_y, pos_z]  (prismatic dummy joints)
      cols 3:6 → [rot_x, rot_y, rot_z]  (XYZ Euler angles from revolute dummy joints)
      cols 6:  → finger joint positions

    Nimble's FreeJoint rootJoint uses a rotation vector (exponential map) for its
    first 3 DOFs, NOT Euler angles.  We therefore convert the stored XYZ Euler
    angles to a rotation vector before packing the Nimble state.

    Resulting hand_qpos_seq columns:
      0:3 → rotation vector  (Nimble rootJoint DOFs 0-2: rot_x, rot_y, rot_z)
      3:6 → position         (Nimble rootJoint DOFs 3-5: pos_x, pos_y, pos_z)
      6:  → finger positions

    Also loads the corresponding object trajectory from the DexYCB dataset.
    object_traj shape: (T, 6) in ROOT_JOINT_DOF_NAMES order [rotvec_x/y/z, pos_x/y/z].
    """
    dataset = DexYCBVideoDataset("/localhome/wpa15/Projects/realart/dex-retargeting/DexYCB", hand_type="right")
    sample = dataset[4]

    hand_qpos_all = np.load(root_dir / "leap_hand_retarget_qpos.npy")
    active_joint_names = load_active_joint_names(root_dir / "leap_hand_retarget_active_joint_names.txt")
    finger_joint_names = active_joint_names[6:]

    frame_start = config.frame_start
    frame_end = min(frame_start + config.num_frames, len(hand_qpos_all))
    print(f"Sequence frames: {frame_start} to {frame_end} ({frame_end - frame_start} steps)")

    hand_seq_offset = len(sample["object_pose"]) - len(hand_qpos_all)
    raw_seq = hand_qpos_all[frame_start - hand_seq_offset:frame_end - hand_seq_offset]
    # Convert XYZ Euler angles → rotation vector for Nimble's FreeJoint.
    euler_xyz = raw_seq[:, 3:6]
    rot_rotvec = R.from_euler("XYZ", euler_xyz).as_rotvec()
    hand_qpos_seq = np.concatenate([rot_rotvec, raw_seq[:, 0:3], raw_seq[:, 6:]], axis=1)

    # Object trajectory: camera → world, already returns rotation vector.
    camera_extrinsics_inv = np.linalg.inv(sample["extrinsics"])
    object_id = 1
    object_traj = compute_object_trajectory(
        sample["object_pose"], camera_extrinsics_inv, object_id, frame_start, frame_end
    )

    return {
        "hand_finger_joint_names": finger_joint_names,
        "hand_qpos_sequence": hand_qpos_seq,
        "object_traj": object_traj,
        "object_mesh_file": Path(sample["object_mesh_file"][object_id]),
        "object_name": f"ycb_{sample['ycb_ids'][object_id]}_{object_id}",
    }


def debug_visualize_reference_sequence(
    hand_urdf: Path,
    ground_urdf: Path,
    config: JointTrackingConfig,
    hand_finger_joint_names,
    hand_qpos_seq: np.ndarray,
    object_urdf: Path = None,
    object_traj: np.ndarray = None,
):
    """Kinematically replay the reference joint (and object) trajectory in the GUI.

    No simulation is run — all bodies are set directly to their reference poses at
    each frame.  Pass ``object_urdf`` and ``object_traj`` to also show the object.

    Args:
        object_traj: (T, 6) array in ROOT_JOINT_DOF_NAMES order
            [rotvec_x, rotvec_y, rotvec_z, pos_x, pos_y, pos_z].
    """
    T = len(hand_qpos_seq)
    show_object = object_urdf is not None and object_traj is not None

    # Build world — hand only first, then conditionally add object.
    world, leap = build_hand_only_scene(
        hand_urdf=hand_urdf,
        ground_urdf=ground_urdf,
        config=config,
        hand_finger_joint_names=hand_finger_joint_names,
        hand_qpos_t0=hand_qpos_seq[0],
    )
    if show_object:
        obj = world.loadSkeleton(str(object_urdf))
        set_dof_positions(obj, ROOT_JOINT_DOF_NAMES, object_traj[0])
        obj.setVelocities(np.zeros(obj.getNumDofs()))

    world_dofs = world.getNumDofs()
    hand_num_dofs = leap.getNumDofs()
    obj_num_dofs = obj.getNumDofs() if show_object else 0
    hand_dof_offset = world_dofs - hand_num_dofs - obj_num_dofs

    root_indices_world = get_dof_indices(leap, ROOT_JOINT_DOF_NAMES) + hand_dof_offset
    finger_indices_world = get_dof_indices(leap, hand_finger_joint_names) + hand_dof_offset
    if show_object:
        object_dof_start = hand_dof_offset + hand_num_dofs
        obj_indices_world = np.arange(obj_num_dofs) + object_dof_start

    # Print root-pose summary for every frame.
    print(f"\n{'t':>4}  {'rotvec_x':>9} {'rotvec_y':>9} {'rotvec_z':>9}  "
          f"{'euler_x°':>9} {'euler_y°':>9} {'euler_z°':>9}  "
          f"{'pos_x':>8} {'pos_y':>8} {'pos_z':>8}")
    print("-" * 100)
    for t in range(T):
        rv = hand_qpos_seq[t, :3]
        pos = hand_qpos_seq[t, 3:6]
        euler_deg = np.degrees(R.from_rotvec(rv).as_euler("XYZ"))
        print(f"{t:>4}  {rv[0]:>9.4f} {rv[1]:>9.4f} {rv[2]:>9.4f}  "
              f"{euler_deg[0]:>9.2f} {euler_deg[1]:>9.2f} {euler_deg[2]:>9.2f}  "
              f"{pos[0]:>8.4f} {pos[1]:>8.4f} {pos[2]:>8.4f}")

    finger_seq = hand_qpos_seq[:, 6:]
    print(f"\nFinger joint ranges over {T} frames:")
    for i, name in enumerate(hand_finger_joint_names):
        lo, hi = finger_seq[:, i].min(), finger_seq[:, i].max()
        print(f"  {name:<40} [{lo:>8.4f}, {hi:>8.4f}]")

    if show_object:
        print(f"\nObject position range over {T} frames:")
        for i, label in enumerate(["pos_x", "pos_y", "pos_z"]):
            lo, hi = object_traj[:, 3 + i].min(), object_traj[:, 3 + i].max()
            print(f"  {label}: [{lo:>8.4f}, {hi:>8.4f}]")

    # Build world states by directly setting all bodies to their reference poses.
    states = []
    state = world.getState().copy()
    for t in range(T):
        state[root_indices_world] = hand_qpos_seq[t, :6]
        state[finger_indices_world] = hand_qpos_seq[t, 6:]
        if show_object:
            state[obj_indices_world] = object_traj[t]
        # Zero all velocities for a clean pose-only playback.
        state[world_dofs:] = 0.0
        states.append(torch.tensor(state, dtype=torch.float32))

    gui = nimble.NimbleGUI(world)
    gui.serve(8000)
    add_world_axes(gui)
    gui.loopStates(states)
    obj_note = " + object" if show_object else ""
    print(f"\nDebug GUI open at http://localhost:8000 — looping {T} reference frames (hand{obj_note}).")
    _block_until_quit(gui)


def render_states_to_video(
    world: nimble.simulation.World,
    states,
    assets: SceneAssets,
    output_path: Path,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    camera_eye: np.ndarray = None,
    camera_target: np.ndarray = None,
    camera_up: np.ndarray = None,
):
    """Render a sequence of Nimble world states to a video file.

    Uses pyrender for offscreen rendering and imageio for video encoding.
    Nimble has no built-in video recorder; this function drives the DART FK
    API (getWorldTransform) directly so no GUI or browser is required.

    Args:
        world: The nimble world (used for skeleton structure and FK only;
               its state is overwritten for each frame during rendering).
        states: List of (2*world_dofs,) numpy arrays or torch tensors.
        assets: SceneAssets with hand/ground URDF paths; object_urdf is optional.
        output_path: Destination video file (.mp4 recommended).
        fps: Frames per second of the output video.
        width, height: Render resolution in pixels.
        camera_eye: Camera position in world frame  (default: [0.3, -0.4, 0.4]).
        camera_target: Point the camera looks at   (default: [0.0, 0.0, 0.1]).
        camera_up: Camera up vector                (default: [0.0, 0.0, 1.0]).
    """
    try:
        import pyrender
        import trimesh as tm
        import imageio
    except ImportError as e:
        raise ImportError(
            "render_states_to_video requires pyrender and imageio. "
            f"Install them with: pip install pyrender imageio[ffmpeg]  ({e})"
        )

    if camera_eye is None:
        camera_eye = np.array([0.3, -0.4, 0.4])
    if camera_target is None:
        camera_target = np.array([0.0, 0.0, 0.1])
    if camera_up is None:
        camera_up = np.array([0.0, 0.0, 1.0])

    # ── 1.  Parse URDF visual geometry ─────────────────────────────────────
    def _parse_urdf_visuals(urdf_path: Path, default_color):
        """Return {link_name: [(trimesh.Trimesh, offset_4x4, color), ...]}."""
        from xml.etree import ElementTree as ET

        tree = ET.parse(str(urdf_path))
        base_dir = urdf_path.parent
        result = {}

        for link in tree.getroot().findall("link"):
            link_name = link.get("name")
            visuals = []
            for visual in link.findall("visual"):
                origin_el = visual.find("origin")
                if origin_el is not None:
                    xyz = [float(v) for v in (origin_el.get("xyz") or "0 0 0").split()]
                    rpy = [float(v) for v in (origin_el.get("rpy") or "0 0 0").split()]
                else:
                    xyz, rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

                T_offset = np.eye(4)
                T_offset[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
                T_offset[:3, 3] = xyz

                geom = visual.find("geometry")
                if geom is None:
                    continue

                mesh: "tm.Trimesh | None" = None
                mesh_el = geom.find("mesh")
                box_el = geom.find("box")
                sphere_el = geom.find("sphere")
                capsule_el = geom.find("capsule")

                if mesh_el is not None:
                    fname = mesh_el.get("filename", "")
                    scale_vals = [float(v) for v in (mesh_el.get("scale") or "1 1 1").split()]
                    mesh_file = (base_dir / fname).resolve()
                    if mesh_file.exists():
                        try:
                            loaded = tm.load(str(mesh_file), force="mesh")
                            if isinstance(loaded, tm.Trimesh):
                                mesh = loaded
                                if not np.allclose(scale_vals, 1.0):
                                    mesh = mesh.copy()
                                    mesh.apply_scale(scale_vals)
                        except Exception as exc:
                            print(f"  Warning: could not load mesh {mesh_file}: {exc}")
                elif box_el is not None:
                    size = [float(v) for v in (box_el.get("size") or "0.1 0.1 0.1").split()]
                    mesh = tm.creation.box(extents=size)
                elif sphere_el is not None:
                    radius = float(sphere_el.get("radius") or 0.01)
                    mesh = tm.creation.icosphere(radius=radius)
                elif capsule_el is not None:
                    radius = float(capsule_el.get("radius") or 0.01)
                    length = float(capsule_el.get("length") or 0.1)
                    mesh = tm.creation.capsule(radius=radius, height=length)

                if mesh is not None:
                    visuals.append((mesh, T_offset, default_color))

            if visuals:
                result[link_name] = visuals

        return result

    # Per-URDF colours: hand = skin, ground = gray, object = amber.
    urdf_color_map = {
        assets.hand_urdf:   [0.85, 0.65, 0.45, 1.0],
        assets.ground_urdf: [0.55, 0.55, 0.55, 1.0],
    }
    if assets.object_urdf is not None:
        urdf_color_map[assets.object_urdf] = [0.90, 0.70, 0.20, 1.0]

    all_visuals: dict = {}  # link_name → [(trimesh, offset_4x4, color), ...]
    for urdf_path, color in urdf_color_map.items():
        all_visuals.update(_parse_urdf_visuals(urdf_path, color))

    # ── 2.  Build pyrender scene ────────────────────────────────────────────
    pr_scene = pyrender.Scene(
        ambient_light=[0.4, 0.4, 0.4, 1.0],
        bg_color=[0.82, 0.85, 0.90, 1.0],
    )

    # Camera pose matrix (column vectors = camera axes in world frame).
    z_ax = camera_eye - camera_target
    z_ax = z_ax / np.linalg.norm(z_ax)
    x_ax = np.cross(camera_up, z_ax)
    x_ax = x_ax / np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = x_ax
    cam_pose[:3, 1] = y_ax
    cam_pose[:3, 2] = z_ax
    cam_pose[:3, 3] = camera_eye
    pr_scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=width / height), pose=cam_pose)

    # Key light from camera direction + fill from upper-left.
    pr_scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.5), pose=cam_pose)
    fill_pose = np.eye(4)
    fill_pose[:3, 3] = camera_eye * np.array([-1.0, -1.0, 1.2])
    pr_scene.add(pyrender.DirectionalLight(color=[0.7, 0.8, 1.0], intensity=1.5), pose=fill_pose)

    # One pyrender node per (skeleton body, visual mesh) pair.
    node_handles = []  # (skel_idx, body_idx, offset_4x4, pyrender.Node)
    for skel_idx in range(world.getNumSkeletons()):
        skel = world.getSkeleton(skel_idx)
        for bn_idx in range(skel.getNumBodyNodes()):
            bn = skel.getBodyNode(bn_idx)
            link_name = bn.getName()
            if link_name not in all_visuals:
                continue
            for (tri_mesh, T_off, color) in all_visuals[link_name]:
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=color,
                    metallicFactor=0.05,
                    roughnessFactor=0.85,
                )
                pr_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material, smooth=False)
                node = pr_scene.add(pr_mesh, pose=np.eye(4))
                node_handles.append((skel_idx, bn_idx, T_off, node))

    if not node_handles:
        raise RuntimeError(
            "render_states_to_video: no visual meshes were mapped to skeleton body nodes. "
            "Check that URDF link names match skeleton body node names."
        )
    print(f"render_states_to_video: {len(node_handles)} visual mesh nodes mapped.")

    # ── 3.  Render each state ───────────────────────────────────────────────
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    frames = []

    print(f"Rendering {len(states)} frames at {width}×{height} @ {fps} fps ...")
    for frame_idx, state in enumerate(states):
        state_np = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state, dtype=np.float64)
        world.setState(state_np)

        for (skel_idx, bn_idx, T_off, node) in node_handles:
            bn = world.getSkeleton(skel_idx).getBodyNode(bn_idx)
            iso = bn.getWorldTransform()
            T_world = np.eye(4)
            T_world[:3, :3] = iso.rotation()
            T_world[:3, 3] = iso.translation()
            pr_scene.set_pose(node, T_world @ T_off)

        color_img, _ = renderer.render(pr_scene)
        frames.append(color_img)

        if (frame_idx + 1) % max(1, len(states) // 10) == 0:
            print(f"  {frame_idx + 1}/{len(states)}")

    renderer.delete()

    # ── 4.  Encode video ────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Video saved → {output_path}  ({len(frames)} frames, {fps} fps)")


def _block_until_quit(gui: nimble.NimbleGUI):
    """Block while the GUI server runs; Ctrl-C cleanly shuts it down."""
    print("GUI running at http://localhost:8000  —  press Ctrl-C to quit.")
    try:
        gui.blockWhileServing()
    except KeyboardInterrupt:
        print("\nGUI dismissed.")


def render_world(world: nimble.simulation.World):
    gui = nimble.NimbleGUI(world)
    gui.serve(8000)
    add_world_axes(gui)
    gui.nativeAPI().renderWorld(world)
    _block_until_quit(gui)


def render_trajectory(world: nimble.simulation.World, states):
    gui = nimble.NimbleGUI(world)
    gui.serve(8000)
    add_world_axes(gui)
    gui.loopStates([state.detach() for state in states])
    _block_until_quit(gui)


def main():
    root_dir = Path(__file__).resolve().parent
    hand_urdf = root_dir.parent.parent / "assets" / "robots" / "hands" / "leap_hand" / "leap_hand_right.urdf"
    config = JointTrackingConfig(
        loss_log_path=root_dir / "joint_tracking_loss_log.csv",
    )
    problem = load_hand_sequence(root_dir, config)

    with TemporaryDirectory(prefix="nimble_joint_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        ground_urdf = create_ground_urdf(temp_dir_path)

        object_urdf = create_mesh_object_urdf(
            problem["object_name"], problem["object_mesh_file"], temp_dir_path
        )

        assets = SceneAssets(
            hand_urdf=hand_urdf,
            ground_urdf=ground_urdf,
            # Joint tracking has no object in the sim, but we keep the path for
            # rendering so the object appears in the output video at its reference pose.
            object_urdf=object_urdf,
        )

        # # Debug: inspect reference trajectory before running any simulation.
        # debug_visualize_reference_sequence(
        #     hand_urdf=hand_urdf,
        #     ground_urdf=ground_urdf,
        #     config=config,
        #     hand_finger_joint_names=problem["hand_finger_joint_names"],
        #     hand_qpos_seq=problem["hand_qpos_sequence"],
        #     object_urdf=object_urdf,
        #     object_traj=problem["object_traj"],
        # )

        gui_world, _ = build_hand_only_scene(
            hand_urdf=hand_urdf,
            ground_urdf=ground_urdf,
            config=config,
            hand_finger_joint_names=problem["hand_finger_joint_names"],
            hand_qpos_t0=problem["hand_qpos_sequence"][0],
        )
        gui = nimble.NimbleGUI(gui_world)
        gui.serve(8000)
        add_world_axes(gui)
        print("GUI open at http://localhost:8000 — will update after optimization.")

        # best = optimize_joint_tracking(
        #     hand_urdf=hand_urdf,
        #     ground_urdf=ground_urdf,
        #     config=config,
        #     hand_finger_joint_names=problem["hand_finger_joint_names"],
        #     hand_qpos_sequence=problem["hand_qpos_sequence"],
        #     gui=gui,
        # )

        # np.save("best_controls3.npy", best["controls"].cpu().numpy())
        # print(f"best_loss: {best['loss']:.6f}")

        best_controls = torch.from_numpy(np.load(root_dir / "best_controls3.npy")).float()

        if config.render_result:
            with torch.no_grad():
                result = rollout_joint_tracking_torch(
                    hand_urdf=hand_urdf,
                    ground_urdf=ground_urdf,
                    config=config,
                    hand_finger_joint_names=problem["hand_finger_joint_names"],
                    hand_qpos_seq=problem["hand_qpos_sequence"],
                    finger_controls=best_controls,
                    collect_states=True,
                )
            states = result["states"]
            gui.loopStates([s.detach() for s in states])

            # Save per-step metrics to CSV
            metrics = result["per_step_metrics"]
            if metrics:
                metrics_path = root_dir / "nimble_single_metrics.csv"
                with open(metrics_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["frame", "joint_rmse", "tau_rms"])
                    for m in metrics:
                        writer.writerow([m["frame"], m["joint_rmse"], m["tau_rms"]])
                print(f"Saved metrics -> {metrics_path}")

                joint_rmses = [m["joint_rmse"] for m in metrics]
                tau_rmses = [m["tau_rms"] for m in metrics]
                print(f"\n--- Summary ---")
                print(f"  Mean joint RMSE: {np.mean(joint_rmses):.4f}")
                print(f"  Mean tau RMS:    {np.mean(tau_rmses):.4f}")

            # cam_eye = np.array([1.0, 0.5, 0.3])
            # cam_target = np.array([0.0, 0.0, 0.0])

            # # Render to video (offscreen, no browser needed).
            # render_states_to_video(
            #     world=gui_world,
            #     states=states,
            #     assets=assets,
            #     output_path=root_dir / "result_joint_tracking3.mp4",
            #     fps=15,
            #     width=1920,
            #     height=1080,
            #     camera_eye=cam_eye,
            #     camera_target=cam_target,
            # )

            # _block_until_quit(gui)


if __name__ == "__main__":
    main()
