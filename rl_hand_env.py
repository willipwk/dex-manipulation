"""
RL environment for Leap Hand force-control tracking.

The agent controls 16 finger joint torques to track a reference trajectory
from DexYCB retargeting while keeping the manipulated object close to its
target position.

Observation (64-dim):
    joint positions (16), joint velocities (16), reference joint positions (16),
    fingertip-to-object distances (4*3=12), object position error (3),
    trajectory phase (1) [normalised frame index]

Action (16-dim):
    joint torques clipped to actuator limits

Reward (same cost structure as mppi_hand_force.py, negated):
    - joint tracking:   exp(-w_joint * ||q - q_ref||^2)
    - object tracking:  exp(-w_obj * ||obj_pos - obj_target||^2)
    - fingertip contact: exp(-w_contact * mean(||tip_k - obj_pos||))
    - effort penalty:   -w_effort * ||tau||^2
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import genesis as gs

from dataset import DexYCBVideoDataset

# ---------------------------------------------------------------------------
# Paths / constants (shared with mppi_hand_force.py)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
ROBOT_URDF = _HERE.parents[1] / "assets/robots/hands/leap_hand/leap_hand_right.urdf"
QPOS_PATH = _HERE / "leap_hand_retarget_qpos.npy"
JOINT_NAMES_PATH = _HERE / "leap_hand_retarget_active_joint_names.txt"

FINGERTIP_LINK_NAMES = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
DATA_ID = 4
OBJECT_IDX = 1
OBJECT_FRAME_OFFSET = 5
N_FINGER_DOFS = 16

# Settling PD gains (used during reset to drive hand to initial pose)
KP_SETTLE = 1.5
KD_SETTLE = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_joint_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _object_pose_in_world(pose_quat: np.ndarray, cam_inv: np.ndarray) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(pose_quat[:4], scalar_first=False).as_matrix()
    mat[:3, 3] = pose_quat[4:7]
    return cam_inv @ mat


def _build_dof_indices(leap, joint_names_finger: List[str]) -> List[int]:
    idx = []
    for name in joint_names_finger:
        j = leap.get_joint(name)
        idx.extend(j.dofs_idx_local)
    return idx


def _broadcast(val: np.ndarray, n: int, device) -> torch.Tensor:
    return torch.tensor(val, dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class LeapHandTrackingEnv:
    """RL environment for Leap Hand force-control tracking (rsl-rl-lib 2.2.4 compatible)."""

    # q(16) + dq(16) + q_ref(16) + tip_to_obj(12) + obj_err(3) + phase(1)
    NUM_OBS = 64

    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_obs = self.NUM_OBS
        self.num_privileged_obs = None
        self.num_actions = N_FINGER_DOFS
        self.device = gs.device

        self.ctrl_dt = env_cfg.get("ctrl_dt", 4e-3)
        self.substeps = env_cfg.get("substeps", 10)
        self.sim_steps_per_action = env_cfg.get("sim_steps_per_action", 10)

        # Reward weights
        self.w_joint = reward_cfg.get("joint_weight", 5.0)
        self.w_obj = reward_cfg.get("obj_weight", 20.0)
        self.w_contact = reward_cfg.get("contact_weight", 5.0)
        self.w_effort = reward_cfg.get("effort_weight", 0.005)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg

        # Load reference trajectory
        dexycb_dir = Path(env_cfg["dexycb_dir"]).resolve()
        dataset = DexYCBVideoDataset(dexycb_dir, hand_type="right")
        data = dataset[DATA_ID]
        self.qpos_ref = np.load(QPOS_PATH).astype(np.float32)  # (n_frames, 22)
        self.n_frames = len(self.qpos_ref)
        self.max_episode_length = self.n_frames

        joint_names = _load_joint_names(JOINT_NAMES_PATH)
        self.joint_names_finger = joint_names[6:]

        # Precompute object target positions for each frame
        object_pose_list = data["object_pose"]
        cam_inv = np.linalg.inv(data["extrinsics"])
        self.mesh_file = data["object_mesh_file"][OBJECT_IDX]

        self.obj_targets = np.zeros((self.n_frames, 3), dtype=np.float32)
        for t in range(self.n_frames):
            obj_frame = min(t + OBJECT_FRAME_OFFSET, len(object_pose_list) - 1)
            world_mat = _object_pose_in_world(
                object_pose_list[obj_frame][OBJECT_IDX], cam_inv
            )
            self.obj_targets[t] = world_mat[:3, 3]

        # Build Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=self.substeps),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, self.num_envs)))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(0.5, -0.5, 0.5),
                camera_lookat=(0.0, -0.5, 0.0),
                camera_fov=45,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())

        qpos0 = self.qpos_ref[0]
        self.leap = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(ROBOT_URDF),
                fixed=True,
                scale=1.0,
                pos=qpos0[:3].tolist(),
                euler=np.degrees(qpos0[3:6]).tolist(),
                batch_fixed_verts=True,
            )
        )

        # Object entity
        world_mat_init = _object_pose_in_world(
            object_pose_list[OBJECT_FRAME_OFFSET][OBJECT_IDX], cam_inv
        )
        obj_quat_init = R.from_matrix(world_mat_init[:3, :3]).as_quat(scalar_first=True)
        self.obj_ent = self.scene.add_entity(
            gs.morphs.Mesh(
                file=self.mesh_file,
                pos=world_mat_init[:3, 3].tolist(),
                quat=obj_quat_init.tolist(),
                scale=1.0,
            )
        )

        self.scene.build(n_envs=self.num_envs)

        # Cache DOF info
        self.tau_lower, self.tau_upper = self.leap.get_dofs_force_range()
        self.dof_indices = _build_dof_indices(self.leap, self.joint_names_finger)
        self.fingertip_links = [self.leap.get_link(name) for name in FINGERTIP_LINK_NAMES]

        # Convert reference data to tensors
        self.qpos_ref_tensor = torch.tensor(
            self.qpos_ref, dtype=torch.float32, device=self.device
        )
        self.obj_targets_tensor = torch.tensor(
            self.obj_targets, dtype=torch.float32, device=self.device
        )

        # Action scaling: normalise actions from [-1, 1] to torque range
        self.tau_range = (self.tau_upper - self.tau_lower) / 2.0
        self.tau_mid = (self.tau_upper + self.tau_lower) / 2.0

        # Buffers
        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Reset specified environments (by integer index) to the start of the trajectory."""
        if len(envs_idx) == 0:
            return

        self.frame_idx[envs_idx] = 0
        self.episode_length_buf[envs_idx] = 0

        # Set base pose and finger joints to reference frame 0
        num_reset = len(envs_idx)
        qpos0 = self.qpos_ref[0]
        q0 = torch.tensor(qpos0[6:], dtype=torch.float32, device=self.device)
        base_pos = _broadcast(qpos0[:3], num_reset, self.device)
        base_quat = _broadcast(
            R.from_euler("XYZ", qpos0[3:6]).as_quat(scalar_first=True),
            num_reset, self.device,
        )

        self.leap.set_pos(base_pos, envs_idx=envs_idx)
        self.leap.set_quat(base_quat, envs_idx=envs_idx)
        self.leap.set_dofs_position(
            q0.unsqueeze(0).expand(num_reset, -1),
            dofs_idx_local=self.dof_indices,
            envs_idx=envs_idx,
        )
        self.leap.zero_all_dofs_velocity(envs_idx=envs_idx)
        self.obj_ent.zero_all_dofs_velocity(envs_idx=envs_idx)

        # Settling: PD loop to drive hand to initial reference
        for _ in range(20):
            q_now = self.leap.get_dofs_position()
            dq_now = self.leap.get_dofs_velocity()
            q_ref_b = q0.unsqueeze(0).expand(self.num_envs, -1)
            tau = torch.clamp(
                KP_SETTLE * (q_ref_b - q_now) - KD_SETTLE * dq_now,
                self.tau_lower, self.tau_upper,
            )
            self.leap.control_dofs_force(tau, dofs_idx_local=self.dof_indices)
            self.scene.step()

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, extras = self.get_observations()
        return obs, extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment.

        Parameters
        ----------
        actions : (num_envs, 16) tensor in [-1, 1], mapped to torque range.

        Returns
        -------
        obs, reward, reset_buf, extras
        """
        # Update time
        self.episode_length_buf += 1

        # Map actions from [-1, 1] to torque range
        torques = actions * self.tau_range + self.tau_mid
        torques = torch.clamp(torques, self.tau_lower, self.tau_upper)

        # Get current frame reference for base pose
        frame_np = self.frame_idx.cpu().numpy()
        for t_val in np.unique(frame_np):
            mask = (self.frame_idx == int(t_val)).nonzero(as_tuple=True)[0]
            qpos_t = self.qpos_ref[int(t_val)]
            base_pos = _broadcast(qpos_t[:3], len(mask), self.device)
            base_quat = _broadcast(
                R.from_euler("XYZ", qpos_t[3:6]).as_quat(scalar_first=True),
                len(mask), self.device,
            )
            self.leap.set_pos(base_pos, envs_idx=mask)
            self.leap.set_quat(base_quat, envs_idx=mask)

        # Apply torques for multiple sim steps
        for _ in range(self.sim_steps_per_action):
            self.leap.control_dofs_force(torques, dofs_idx_local=self.dof_indices)
            self.scene.step()

        # Compute reward
        reward = self._compute_reward(torques)

        # Advance frame
        self.frame_idx = torch.clamp(self.frame_idx + 1, max=self.n_frames - 1)

        # Check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        obs, self.extras = self.get_observations()
        return obs, reward, self.reset_buf, self.extras

    def is_episode_complete(self) -> torch.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = time_out_buf

        time_out_idx = time_out_buf.nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def _compute_reward(self, torques: torch.Tensor) -> torch.Tensor:
        """Compute per-env reward."""
        # Current state
        q_now = self.leap.get_dofs_position()       # (N, 16)
        obj_pos = self.obj_ent.get_pos()             # (N, 3)

        # Per-env reference
        q_ref = self.qpos_ref_tensor[self.frame_idx, 6:]           # (N, 16)
        obj_target = self.obj_targets_tensor[self.frame_idx]       # (N, 3)

        # Joint tracking reward
        joint_err_sq = ((q_now - q_ref) ** 2).sum(dim=-1)           # (N,)
        r_joint = torch.exp(-self.w_joint * joint_err_sq)

        # Object tracking reward
        obj_err_sq = ((obj_pos - obj_target) ** 2).sum(dim=-1)      # (N,)
        r_obj = torch.exp(-self.w_obj * obj_err_sq)

        # Fingertip contact reward
        tip_dists = torch.stack(
            [torch.norm(tip.get_pos() - obj_pos, dim=-1) for tip in self.fingertip_links],
            dim=1,
        )  # (N, 4)
        r_contact = torch.exp(-self.w_contact * tip_dists.mean(dim=-1))

        # Effort penalty
        effort_penalty = self.w_effort * (torques ** 2).sum(dim=-1)

        reward = r_joint + r_obj + r_contact - effort_penalty
        return reward

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        q_now = self.leap.get_dofs_position()        # (N, 16)
        dq_now = self.leap.get_dofs_velocity()       # (N, 16)
        q_ref = self.qpos_ref_tensor[self.frame_idx, 6:]  # (N, 16)

        obj_pos = self.obj_ent.get_pos()             # (N, 3)
        obj_target = self.obj_targets_tensor[self.frame_idx]  # (N, 3)

        # Fingertip-to-object relative positions (4 tips * 3 = 12)
        tip_rel = torch.cat(
            [tip.get_pos() - obj_pos for tip in self.fingertip_links],
            dim=-1,
        )  # (N, 12)

        obj_err = obj_pos - obj_target               # (N, 3)

        # Trajectory phase [0, 1]
        phase = (self.frame_idx.float() / max(self.n_frames - 1, 1)).unsqueeze(-1)  # (N, 1)

        self.obs_buf = torch.cat([q_now, dq_now, q_ref, tip_rel, obj_err, phase], dim=-1)
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> None:
        return None
