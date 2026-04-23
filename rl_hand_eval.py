"""
Evaluate a trained Leap Hand tracking policy and record a video.

Loads the latest PPO checkpoint, rolls out the policy through the full
reference trajectory in a single-env Genesis scene, and saves a video.

Usage
-----
  # Evaluate with video recording
  python rl_hand_eval.py --dexycb_dir /path/to/dexycb

  # Evaluate with live viewer (no video)
  python rl_hand_eval.py --dexycb_dir /path/to/dexycb --no_record

  # Specify a checkpoint
  python rl_hand_eval.py --dexycb_dir /path/to/dexycb --ckpt logs/leap_hand_tracking_rl/model_500.pt
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from dataset import DexYCBVideoDataset

# ---------------------------------------------------------------------------
# Paths / constants (shared with rl_hand_env.py / mppi_hand_force.py)
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
NUM_OBS = 64

KP_SETTLE = 1.5
KD_SETTLE = 0.1

VIDEO_PATH = _HERE / "rl_policy_visualization.mp4"
LOSS_LOG_PATH = _HERE / "rl_policy_loss_log.csv"


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


def find_latest_checkpoint(log_dir: Path) -> Path:
    ckpt_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    return max(ckpt_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))


# ---------------------------------------------------------------------------
# Dummy env for loading the runner (provides shape info only)
# ---------------------------------------------------------------------------

class _ShimEnv:
    """Minimal shim that satisfies OnPolicyRunner's constructor without building a scene."""

    def __init__(self):
        self.num_envs = 1
        self.num_obs = NUM_OBS
        self.num_privileged_obs = None
        self.num_actions = N_FINGER_DOFS
        self.device = gs.device
        self.extras = {"observations": {}}

    def get_observations(self):
        obs = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.extras["observations"]["critic"] = obs
        return obs, self.extras

    def get_privileged_observations(self):
        return None

    def reset(self):
        return self.get_observations()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Leap Hand tracking policy")
    parser.add_argument("--dexycb_dir", type=str, required=True, help="Path to DexYCB dataset")
    parser.add_argument("-e", "--exp_name", type=str, default="leap_hand_tracking")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (default: latest)")
    parser.add_argument("--no_record", action="store_true", help="Skip video recording")
    parser.add_argument("--sim_steps", type=int, default=10, help="Sim steps per policy step")
    args = parser.parse_args()

    dexycb_dir = Path(args.dexycb_dir).resolve()

    # -- Find checkpoint --
    log_dir = Path("logs") / f"{args.exp_name}_rl"
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = find_latest_checkpoint(log_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    # -- Init Genesis (single env, CPU for viewer) --
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    # -- Load trained policy via OnPolicyRunner + shim env --
    from rl_hand_train import get_train_cfg
    train_cfg = get_train_cfg(args.exp_name, max_iterations=0)
    shim_env = _ShimEnv()
    runner = OnPolicyRunner(shim_env, train_cfg, log_dir, device=gs.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=gs.device)
    print("Policy loaded.")

    # -- Load reference data --
    dataset = DexYCBVideoDataset(dexycb_dir, hand_type="right")
    data = dataset[DATA_ID]
    qpos_ref = np.load(QPOS_PATH).astype(np.float32)
    joint_names = _load_joint_names(JOINT_NAMES_PATH)
    joint_names_finger = joint_names[6:]

    n_frames = len(qpos_ref)
    object_pose_list = data["object_pose"]
    cam_inv = np.linalg.inv(data["extrinsics"])
    mesh_file = data["object_mesh_file"][OBJECT_IDX]

    # Precompute object targets
    obj_targets = np.zeros((n_frames, 3), dtype=np.float32)
    for t in range(n_frames):
        obj_frame = min(t + OBJECT_FRAME_OFFSET, len(object_pose_list) - 1)
        world_mat = _object_pose_in_world(object_pose_list[obj_frame][OBJECT_IDX], cam_inv)
        obj_targets[t] = world_mat[:3, 3]
    obj_targets_t = torch.tensor(obj_targets, dtype=torch.float32, device=gs.device)
    qpos_ref_t = torch.tensor(qpos_ref, dtype=torch.float32, device=gs.device)

    # -- Build single-env Genesis scene --
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    qpos0 = qpos_ref[0]
    leap = scene.add_entity(
        gs.morphs.URDF(
            file=str(ROBOT_URDF),
            fixed=True,
            scale=1.0,
            pos=qpos0[:3].tolist(),
            euler=np.degrees(qpos0[3:6]).tolist(),
        )
    )

    world_mat_init = _object_pose_in_world(
        object_pose_list[OBJECT_FRAME_OFFSET][OBJECT_IDX], cam_inv
    )
    obj_quat_init = R.from_matrix(world_mat_init[:3, :3]).as_quat(scalar_first=True)
    obj_ent = scene.add_entity(
        gs.morphs.Mesh(
            file=mesh_file,
            pos=world_mat_init[:3, 3].tolist(),
            quat=obj_quat_init.tolist(),
            scale=1.0,
        )
    )

    # Recording camera
    hand_pos = qpos0[:3]
    cam = scene.add_camera(
        res=(1920, 1080),
        pos=(hand_pos[0] + 0.5, hand_pos[1] - 0.5, hand_pos[2] + 0.5),
        lookat=(hand_pos[0], hand_pos[1] - 0.5, hand_pos[2]),
        fov=45,
        GUI=False,
    )

    scene.build()

    tau_lower, tau_upper = leap.get_dofs_force_range()
    tau_range = (tau_upper - tau_lower) / 2.0
    tau_mid = (tau_upper + tau_lower) / 2.0
    dof_indices = _build_dof_indices(leap, joint_names_finger)
    fingertip_links = [leap.get_link(name) for name in FINGERTIP_LINK_NAMES]

    # -- Settle hand at frame 0 --
    q0 = qpos_ref[0, 6:].astype(np.float32)
    base_quat0 = R.from_euler("XYZ", qpos0[3:6]).as_quat(scalar_first=True)
    leap.set_pos(qpos0[:3])
    leap.set_quat(base_quat0)
    leap.set_dofs_position(q0, dofs_idx_local=dof_indices)
    leap.zero_all_dofs_velocity()

    for _ in range(50):
        leap.set_pos(qpos0[:3])
        leap.set_quat(base_quat0)
        q_now = leap.get_dofs_position()
        dq_now = leap.get_dofs_velocity()
        tau = torch.clamp(
            KP_SETTLE * (torch.tensor(q0, device=gs.device) - q_now) - KD_SETTLE * dq_now,
            tau_lower, tau_upper,
        )
        leap.control_dofs_force(tau, dofs_idx_local=dof_indices)
        scene.step()

    # -- Rollout loop --
    record = not args.no_record
    if record:
        print(f"Recording video to {VIDEO_PATH} ...")
        cam.start_recording()

    loss_rows = []
    print(f"Running policy for {n_frames} frames. Close the viewer to exit.")

    for t in range(n_frames):
        # Set base pose kinematically
        base_quat = R.from_euler("XYZ", qpos_ref[t, 3:6]).as_quat(scalar_first=True)
        leap.set_pos(qpos_ref[t, :3])
        leap.set_quat(base_quat)

        # Build observation (single env, no batch dim needed for policy input)
        q_now = leap.get_dofs_position()       # (16,)
        dq_now = leap.get_dofs_velocity()      # (16,)
        q_ref = qpos_ref_t[t, 6:]             # (16,)
        obj_pos = obj_ent.get_pos()            # (3,)
        obj_target = obj_targets_t[t]          # (3,)

        tip_rel = torch.cat([link.get_pos() - obj_pos for link in fingertip_links], dim=-1)  # (12,)
        obj_err = obj_pos - obj_target         # (3,)
        phase = torch.tensor([t / max(n_frames - 1, 1)], dtype=torch.float32, device=gs.device)

        obs = torch.cat([q_now, dq_now, q_ref, tip_rel, obj_err, phase], dim=-1)
        obs = obs.unsqueeze(0)  # (1, 64)

        # Query policy
        with torch.inference_mode():
            action = policy(obs)  # (1, 16)

        # Map to torques
        torques = (action.squeeze(0) * tau_range + tau_mid).clamp(tau_lower, tau_upper)

        # Step physics
        for _ in range(args.sim_steps):
            leap.control_dofs_force(torques, dofs_idx_local=dof_indices)
            scene.step()

        # Record frame
        if record:
            cam.render()

        # Log metrics
        q_after = leap.get_dofs_position()
        joint_rmse = float(((q_after - q_ref) ** 2).mean() ** 0.5)
        obj_pos_after = obj_ent.get_pos()
        obj_err_val = float(torch.norm(obj_pos_after - obj_target).item())
        tau_rms = float((torques ** 2).mean() ** 0.5)
        print(
            f"[{t:3d}/{n_frames}]  joint_RMSE={joint_rmse:.4f}  "
            f"obj_pos_err={obj_err_val:.4f}  tau_rms={tau_rms:.4f}"
        )
        loss_rows.append((t, joint_rmse, obj_err_val, tau_rms))

    # -- Save outputs --
    if record:
        cam.stop_recording(save_to_filename=str(VIDEO_PATH), fps=30)
        print(f"Saved video -> {VIDEO_PATH}")

    with open(LOSS_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "joint_rmse", "obj_pos_err", "tau_rms"])
        writer.writerows(loss_rows)
    print(f"Saved loss log -> {LOSS_LOG_PATH}")

    # Print summary
    joint_rmses = [r[1] for r in loss_rows]
    obj_errs = [r[2] for r in loss_rows]
    print(f"\n--- Summary ---")
    print(f"  Mean joint RMSE:    {np.mean(joint_rmses):.4f}")
    print(f"  Mean obj pos error: {np.mean(obj_errs):.4f}")


if __name__ == "__main__":
    main()
