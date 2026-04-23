"""
Force-control MPPI for Leap Hand finger optimization.

Identical problem setup to mppi_hand.py but the control signal is joint
*torque* τ ∈ R^16 rather than position targets.  Genesis's
``control_dofs_force`` is used throughout so the full contact dynamics are
included in every rollout.

Control design
--------------
* State   : q (joint angle) + dq (joint velocity) – evolved by physics
* Control : τ ∈ [τ_min, τ_max]^16  (joint torques, effort limit ±0.95 Nm)
* Nominal sequence initialised each frame with PD-equivalent torques:
      τ_init = kp_init · (q_ref – q_current) – kd_init · dq_current
  clipped to the torque limits.  This warm-starts MPPI around the
  retargeting solution expressed as a torque command.

Cost (same structure as position-control version)
-------------------------------------------------
  c = w_joint   · ‖q – q_ref‖²
    + w_obj     · ‖pos_obj_sim – pos_obj_target‖²
    + w_contact · mean_k ‖pos_tip_k – pos_obj_sim‖

Two modes
---------
optimize  : run MPPI offline, save controls to ``mppi_force_controls.npy``
visualize : replay saved torque commands in the Genesis viewer

Usage
-----
  conda run -n genesis python mppi_hand_force.py \\
      --dexycb_dir /path/to/dexycb --mode optimize
  conda run -n genesis python mppi_hand_force.py \\
      --dexycb_dir /path/to/dexycb --mode visualize
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import tyro

from dataset import DexYCBVideoDataset

import genesis as gs

# numpy version compatibility shims – only apply on numpy < 2.0
if tuple(int(x) for x in np.__version__.split(".")[:2]) < (2, 0):
    np.bool = bool        # type: ignore[attr-defined]
    np.int = int          # type: ignore[attr-defined]
    np.float = float      # type: ignore[attr-defined]
    np.str = str          # type: ignore[attr-defined]
    np.complex = complex  # type: ignore[attr-defined]
    np.object = object    # type: ignore[attr-defined]
    np.unicode = np.unicode_  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
ROBOT_URDF = _HERE.parents[1] / "assets/robots/hands/leap_hand/leap_hand_right.urdf"
QPOS_PATH = _HERE / "leap_hand_retarget_qpos.npy"
JOINT_NAMES_PATH = _HERE / "leap_hand_retarget_active_joint_names.txt"
MPPI_CONTROLS_PATH = _HERE / "mppi_force_controls.npy"
MPPI_LOSS_LOG_PATH = _HERE / "mppi_force_loss_log.csv"
MPPI_VIDEO_PATH = _HERE / "mppi_force_visualization.mp4"

FINGERTIP_LINK_NAMES = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
DATA_ID = 4
OBJECT_IDX = 1          # index of the manipulated object within the data sample
OBJECT_FRAME_OFFSET = 5
N_FINGER_DOFS = 16

# Soft PD gains used to initialise the nominal torque sequence each frame
# and to settle the hand before optimisation begins.
# Chosen so that a 0.5 rad error produces ~0.4 Nm (well within ±0.95 Nm limit).
KP_INIT = 0.8
KD_INIT = 0.05
# Gains for the settling PD loop (stronger, drives the hand to reference quickly)
KP_SETTLE = 1.5
KD_SETTLE = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_joint_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _object_pose_in_world(pose_quat: np.ndarray, cam_inv: np.ndarray) -> np.ndarray:
    """Convert object pose (camera frame, 7-vec qxyzw + txyz) to a 4x4 world matrix."""
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


def _broadcast_pos(pos: np.ndarray, n: int, device) -> torch.Tensor:
    return torch.tensor(pos, dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1)


def _broadcast_quat(quat: np.ndarray, n: int, device) -> torch.Tensor:
    return torch.tensor(quat, dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1)


def _pd_torque(
    q: torch.Tensor,      # (..., n_dofs)  current positions
    dq: torch.Tensor,     # (..., n_dofs)  current velocities
    q_ref: torch.Tensor,  # (n_dofs,) or (..., n_dofs)
    kp: float,
    kd: float,
    tau_lower: torch.Tensor,
    tau_upper: torch.Tensor,
) -> torch.Tensor:
    """Compute PD-equivalent torques clipped to actuator limits."""
    tau = kp * (q_ref - q) - kd * dq
    return torch.clamp(tau, tau_lower, tau_upper)


# ---------------------------------------------------------------------------
# MPPI controller (force/torque control)
# ---------------------------------------------------------------------------

class MPPIForceController:
    """
    MPPI optimiser for Leap Hand finger joints using joint torque control.

    Parameters
    ----------
    n_dofs         : number of finger DOFs (16)
    tau_lower      : lower torque limits, shape (n_dofs,)
    tau_upper      : upper torque limits, shape (n_dofs,)
    n_samples      : N  parallel rollout samples
    horizon        : H  planning horizon (sim steps)
    temperature    : lambda  MPPI temperature
    noise_sigma    : sigma  std-dev of Gaussian torque perturbations (Nm)
    joint_weight   : cost weight for joint-angle tracking
    obj_weight     : cost weight for object-position tracking
    contact_weight : cost weight for fingertip-to-object proximity
    effort_weight  : cost weight penalising large torques (regularisation)
    device         : torch device
    """

    def __init__(
        self,
        n_dofs: int,
        tau_lower: torch.Tensor,
        tau_upper: torch.Tensor,
        n_samples: int = 64,
        horizon: int = 5,
        temperature: float = 0.05,
        noise_sigma: float = 0.15,
        joint_weight: float = 2.0,
        obj_weight: float = 10.0,
        contact_weight: float = 3.0,
        effort_weight: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.n_dofs = n_dofs
        self.N = n_samples
        self.H = horizon
        self.lam = temperature
        self.sigma = noise_sigma
        self.w_joint = joint_weight
        self.w_obj = obj_weight
        self.w_contact = contact_weight
        self.w_effort = effort_weight
        self.device = device

        self.tau_lower = tau_lower.to(device)  # (n_dofs,)
        self.tau_upper = tau_upper.to(device)  # (n_dofs,)

        # Nominal torque sequence (H, n_dofs); initialised by reset()
        self.U: torch.Tensor | None = None

    def reset(
        self,
        q_current: torch.Tensor,   # (n_dofs,) current joint positions
        q_ref: torch.Tensor,       # (n_dofs,) reference joint positions
        dq_current: torch.Tensor | None = None,  # (n_dofs,) current velocities
    ) -> None:
        """
        Warm-start nominal torque sequence with PD-equivalent torques toward q_ref.
        This centres the MPPI perturbations around a torque that drives the
        hand toward the retargeting reference solution.
        """
        tau_init = _pd_torque(
            q_current.to(self.device),
            dq_current.to(self.device) if dq_current is not None
            else torch.zeros_like(q_current, device=self.device),
            q_ref.to(self.device),
            KP_INIT, KD_INIT,
            self.tau_lower, self.tau_upper,
        )
        self.U = tau_init.unsqueeze(0).expand(self.H, -1).clone()

    def _clamp(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tau, self.tau_lower, self.tau_upper)

    @torch.no_grad()
    def step(
        self,
        scene,
        leap,
        fingertip_links: list,
        dof_indices: List[int],
        obj_entities: list,
        saved_state,
        q_ref: torch.Tensor,          # (n_dofs,)
        obj_target_positions: list,   # list of (3,) tensors
    ) -> torch.Tensor:
        """
        Execute one MPPI update with torque control.

        All N envs are reset to saved_state before returning.
        Returns the best first-step torque τ₀ ∈ R^{n_dofs}.
        """
        assert self.U is not None, "Call reset() before step()."

        # Sample perturbations ε ~ N(0, σ²I), shape (N, H, n_dofs)
        eps = torch.randn(self.N, self.H, self.n_dofs, device=self.device) * self.sigma

        # Perturbed torque sequences clipped to actuator limits, (N, H, n_dofs)
        U_pert = self._clamp(self.U.unsqueeze(0) + eps)

        # Parallel rollout
        costs = torch.zeros(self.N, device=self.device)
        q_ref_b = q_ref.to(self.device).unsqueeze(0)  # (1, n_dofs)
        targets_b = [t.to(self.device).unsqueeze(0) for t in obj_target_positions]

        for h in range(self.H):
            # Apply torques directly to all N envs
            leap.control_dofs_force(U_pert[:, h, :], dofs_idx_local=dof_indices)
            scene.step()

            # Joint-tracking cost
            q_now = leap.get_dofs_position()    # (N, n_dofs)
            costs += self.w_joint * ((q_now - q_ref_b) ** 2).sum(dim=-1)

            # Per-object costs
            for obj_ent, tgt_b in zip(obj_entities, targets_b):
                obj_pos_now = obj_ent.get_pos()   # (N, 3)

                # Object-position-tracking cost
                costs += self.w_obj * ((obj_pos_now - tgt_b) ** 2).sum(dim=-1)

                # Contact cost: fingertips near the simulated object
                tip_dists = torch.stack(
                    [torch.norm(tip.get_pos() - obj_pos_now, dim=-1)
                     for tip in fingertip_links],
                    dim=1,
                )  # (N, 4)
                costs += self.w_contact * tip_dists.mean(dim=-1)

            # Effort regularisation: penalise large torques
            costs += self.w_effort * (U_pert[:, h, :] ** 2).sum(dim=-1)

        # Reset all envs to the pre-rollout state
        scene.reset(saved_state)

        # MPPI weight update
        beta = costs.min()
        weights = torch.softmax(-(costs - beta) / self.lam, dim=0)  # (N,)

        delta = (weights.view(self.N, 1, 1) * eps).sum(dim=0)       # (H, n_dofs)
        self.U = self._clamp(self.U + delta)

        # Extract first torque, shift horizon (receding-horizon warm start)
        tau0 = self.U[0].clone()
        self.U = torch.cat([self.U[1:], self.U[-1:]], dim=0)

        return tau0  # (n_dofs,)


# ---------------------------------------------------------------------------
# Optimize mode
# ---------------------------------------------------------------------------

def run_optimize(
    dexycb_dir: Path,
    n_samples: int,
    horizon: int,
    mppi_iters: int,
    joint_weight: float,
    obj_weight: float,
    contact_weight: float,
    effort_weight: float,
    noise_sigma: float,
    temperature: float,
) -> None:
    """Run force-control MPPI offline and save the optimised torques."""

    # Load dataset and reference trajectory
    dataset = DexYCBVideoDataset(dexycb_dir, hand_type="right")
    data = dataset[DATA_ID]
    qpos_list = np.load(QPOS_PATH)          # (n_frames, 22)
    joint_names = _load_joint_names(JOINT_NAMES_PATH)
    joint_names_finger = joint_names[6:]

    n_frames = len(qpos_list)
    object_pose_list = data["object_pose"]
    cam_inv = np.linalg.inv(data["extrinsics"])
    mesh_file = data["object_mesh_file"][OBJECT_IDX]

    # Genesis scene – no viewer, N parallel envs
    backend = gs.gpu if torch.cuda.is_available() else gs.cpu
    gs.init(backend=backend)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    qpos0 = qpos_list[0]
    leap = scene.add_entity(
        gs.morphs.URDF(
            file=str(ROBOT_URDF),
            fixed=True,
            scale=1.0,
            pos=qpos0[:3].tolist(),
            euler=np.degrees(qpos0[3:6]).tolist(),
        )
    )

    # Manipulated object – free-floating rigid body
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
    obj_entities = [obj_ent]

    scene.build(n_envs=n_samples)

    tau_lower, tau_upper = leap.get_dofs_force_range()
    device = tau_lower.device

    fingertip_links = [leap.get_link(name) for name in FINGERTIP_LINK_NAMES]
    dof_indices = _build_dof_indices(leap, joint_names_finger)

    # MPPI controller
    mppi = MPPIForceController(
        n_dofs=N_FINGER_DOFS,
        tau_lower=tau_lower,
        tau_upper=tau_upper,
        n_samples=n_samples,
        horizon=horizon,
        temperature=temperature,
        noise_sigma=noise_sigma,
        joint_weight=joint_weight,
        obj_weight=obj_weight,
        contact_weight=contact_weight,
        effort_weight=effort_weight,
        device=device,
    )

    # Hard-reset all envs to the reference base pose and joint values at t=0
    q0 = torch.tensor(qpos0[6:], dtype=torch.float32, device=device)
    base_pos0 = _broadcast_pos(qpos0[:3], n_samples, device)
    base_quat0 = _broadcast_quat(
        R.from_euler("XYZ", qpos0[3:6]).as_quat(scalar_first=True), n_samples, device
    )
    leap.set_pos(base_pos0)
    leap.set_quat(base_quat0)
    leap.set_dofs_position(
        q0.unsqueeze(0).expand(n_samples, -1), dofs_idx_local=dof_indices
    )
    leap.zero_all_dofs_velocity()
    for obj_ent_i in obj_entities:
        obj_ent_i.zero_all_dofs_velocity()

    # Settling: PD torque loop – drives hand to reference and builds contact
    for _ in range(50):
        leap.set_pos(base_pos0)
        leap.set_quat(base_quat0)
        q_now = leap.get_dofs_position()   # (N, 16)
        dq_now = leap.get_dofs_velocity()  # (N, 16)
        q0_b = q0.unsqueeze(0).expand(n_samples, -1)
        tau_settle = _pd_torque(
            q_now, dq_now, q0_b, KP_SETTLE, KD_SETTLE, tau_lower, tau_upper
        )
        leap.control_dofs_force(tau_settle, dofs_idx_local=dof_indices)
        scene.step()

    # Initialise nominal sequence with PD-equivalent torques toward reference
    q_current = leap.get_dofs_position()[0].detach()
    dq_current = leap.get_dofs_velocity()[0].detach()
    mppi.reset(q_current, q0, dq_current)

    # Main optimisation loop
    best_controls = np.zeros((n_frames, N_FINGER_DOFS), dtype=np.float32)
    loss_rows: list = []  # accumulated per-frame loss metrics for CSV logging

    for t in range(n_frames):
        q_ref = torch.tensor(qpos_list[t, 6:], dtype=torch.float32, device=device)

        # Update base pose for all N envs to match reference at frame t
        base_pos_t = qpos_list[t, :3]
        base_quat_t = R.from_euler("XYZ", qpos_list[t, 3:6]).as_quat(scalar_first=True)
        leap.set_pos(_broadcast_pos(base_pos_t, n_samples, device))
        leap.set_quat(_broadcast_quat(base_quat_t, n_samples, device))

        # Object target position at frame t
        obj_frame = min(t + OBJECT_FRAME_OFFSET, len(object_pose_list) - 1)
        world_mat = _object_pose_in_world(
            object_pose_list[obj_frame][OBJECT_IDX], cam_inv
        )
        obj_target_positions = [
            torch.tensor(world_mat[:3, 3], dtype=torch.float32, device=device)
        ]

        # Bring all N envs to current state and zero velocities
        leap.set_dofs_position(
            q_current.unsqueeze(0).expand(n_samples, -1), dofs_idx_local=dof_indices
        )
        leap.zero_all_dofs_velocity()

        # One step to make the state consistent before saving
        q_now = leap.get_dofs_position()
        dq_now = leap.get_dofs_velocity()
        q_ref_b = q_ref.unsqueeze(0).expand(n_samples, -1)
        tau_init_step = _pd_torque(
            q_now, dq_now, q_ref_b, KP_SETTLE, KD_SETTLE, tau_lower, tau_upper
        )
        leap.control_dofs_force(tau_init_step, dofs_idx_local=dof_indices)
        scene.step()

        # Re-seed nominal torque sequence with PD torques toward current q_ref
        q_s = leap.get_dofs_position()[0].detach()
        dq_s = leap.get_dofs_velocity()[0].detach()
        mppi.reset(q_s, q_ref, dq_s)

        # MPPI refinement iterations
        tau0 = _pd_torque(
            q_s, dq_s, q_ref, KP_SETTLE, KD_SETTLE, tau_lower, tau_upper
        )
        for _ in range(mppi_iters):
            saved_state = scene.get_state()
            tau0 = mppi.step(
                scene, leap, fingertip_links, dof_indices,
                obj_entities, saved_state, q_ref, obj_target_positions,
            )

        # Apply winning torques for multiple real physics steps
        for _ in range(10):
            leap.control_dofs_force(
                tau0.unsqueeze(0).expand(n_samples, -1), dofs_idx_local=dof_indices
            )
            scene.step()

        # Record executed state from env 0
        q_current = leap.get_dofs_position()[0].detach()
        best_controls[t] = tau0.cpu().numpy()

        joint_rmse = float(((q_current - q_ref) ** 2).mean() ** 0.5)
        obj_pos_now = obj_ent.get_pos()[0]
        obj_err = float(torch.norm(obj_pos_now - obj_target_positions[0]).item())
        tau_rms = float((tau0 ** 2).mean() ** 0.5)
        print(
            f"[{t:3d}/{n_frames}]  joint_RMSE={joint_rmse:.4f}  "
            f"obj_pos_err={obj_err:.4f}  "
            f"tau_rms={tau_rms:.4f}"
        )
        loss_rows.append((t, joint_rmse, obj_err, tau_rms))

    np.save(MPPI_CONTROLS_PATH, best_controls)
    print(f"\nSaved optimised torques -> {MPPI_CONTROLS_PATH}")

    # Save per-frame loss metrics to CSV
    with open(MPPI_LOSS_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "joint_rmse", "obj_pos_err", "tau_rms"])
        writer.writerows(loss_rows)
    print(f"Saved loss log         -> {MPPI_LOSS_LOG_PATH}")


# ---------------------------------------------------------------------------
# Visualize mode
# ---------------------------------------------------------------------------

def run_visualize(dexycb_dir: Path) -> None:
    """
    Replay MPPI-optimised torques in the Genesis viewer.

    The hand base pose tracks the reference kinematically; finger torques are
    applied directly via force control.  The object moves through contact physics.
    """

    for p in (MPPI_CONTROLS_PATH, QPOS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run --mode optimize first.")

    best_controls = np.load(MPPI_CONTROLS_PATH)  # (n_frames, 16) – torques in Nm
    qpos_list = np.load(QPOS_PATH)               # (n_frames, 22)
    joint_names = _load_joint_names(JOINT_NAMES_PATH)
    joint_names_finger = joint_names[6:]

    dataset = DexYCBVideoDataset(dexycb_dir, hand_type="right")
    data = dataset[DATA_ID]
    object_pose_list = data["object_pose"]
    cam_inv = np.linalg.inv(data["extrinsics"])
    mesh_file = data["object_mesh_file"][OBJECT_IDX]

    n_frames = len(best_controls)

    # Genesis scene – with viewer, single env
    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())

    qpos0 = qpos_list[0]
    leap = scene.add_entity(
        gs.morphs.URDF(
            file=str(ROBOT_URDF),
            fixed=True,
            scale=1.0,
            pos=qpos0[:3].tolist(),
            euler=np.degrees(qpos0[3:6]).tolist(),
        )
    )

    # Manipulated object – free-floating, driven by contact physics only
    world_mat_init = _object_pose_in_world(
        object_pose_list[OBJECT_FRAME_OFFSET][OBJECT_IDX], cam_inv
    )
    obj_quat_init = R.from_matrix(world_mat_init[:3, :3]).as_quat(scalar_first=True)
    scene.add_entity(
        gs.morphs.Mesh(
            file=mesh_file,
            pos=world_mat_init[:3, 3].tolist(),
            quat=obj_quat_init.tolist(),
            scale=1.0,
        )
    )

    # Camera for recording – positioned to view the hand and object
    hand_pos = qpos0[:3]
    cam = scene.add_camera(
        res=(1920, 1080),
        pos=(hand_pos[0]+ 0.5, hand_pos[1] - 0.5, hand_pos[2] + 0.5),
        lookat=(hand_pos[0], hand_pos[1] - 0.5, hand_pos[2]),
        fov=45,
        GUI=False,
    )

    scene.build()

    tau_lower, tau_upper = leap.get_dofs_force_range()
    dof_indices = _build_dof_indices(leap, joint_names_finger)

    # Hard-reset hand to reference pose and joint values at t=0
    q0 = qpos_list[0, 6:].astype(np.float32)
    base_quat0 = R.from_euler("XYZ", qpos0[3:6]).as_quat(scalar_first=True)
    leap.set_pos(qpos0[:3])
    leap.set_quat(base_quat0)
    leap.set_dofs_position(q0, dofs_idx_local=dof_indices)
    leap.zero_all_dofs_velocity()

    # Settling: PD torques drive the hand to the reference and build contact
    tau_lower_np = tau_lower.cpu().numpy()
    tau_upper_np = tau_upper.cpu().numpy()
    for _ in range(50):
        leap.set_pos(qpos0[:3])
        leap.set_quat(base_quat0)
        q_now = leap.get_dofs_position().cpu().numpy()   # (16,)
        dq_now = leap.get_dofs_velocity().cpu().numpy()  # (16,)
        tau = np.clip(
            KP_SETTLE * (q0 - q_now) - KD_SETTLE * dq_now,
            tau_lower_np, tau_upper_np,
        )
        leap.control_dofs_force(tau, dofs_idx_local=dof_indices)
        scene.step()

    print("Visualising force-control trajectory. Close the viewer window to exit.")
    print(f"Recording first pass to {MPPI_VIDEO_PATH} ...")
    first_pass = True
    while True:
        if first_pass:
            cam.start_recording()
        for t in range(n_frames):
            # Base pose tracks reference kinematically
            base_quat = R.from_euler("XYZ", qpos_list[t, 3:6]).as_quat(scalar_first=True)
            leap.set_pos(qpos_list[t, :3])
            leap.set_quat(base_quat)

            # Feed MPPI torques to the actuators; object responds via contact physics
            for _ in range(10):
                leap.control_dofs_force(best_controls[t], dofs_idx_local=dof_indices)
                scene.step()

            # Capture one frame per trajectory timestep
            if first_pass:
                cam.render()

        if first_pass:
            cam.stop_recording(save_to_filename=str(MPPI_VIDEO_PATH), fps=30)
            print(f"Saved recording -> {MPPI_VIDEO_PATH}")
            first_pass = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    dexycb_dir: str,
    mode: Literal["optimize", "visualize"] = "optimize",
    n_samples: int = 64,
    horizon: int = 5,
    mppi_iters: int = 5,
    joint_weight: float = 2.0,
    obj_weight: float = 10.0,
    contact_weight: float = 3.0,
    effort_weight: float = 0.01,
    noise_sigma: float = 0.15,
    temperature: float = 0.05,
) -> None:
    """
    Force-control MPPI for Leap Hand finger joint optimisation.

    Args:
        dexycb_dir:     Path to the DexYCB dataset root directory.
        mode:           'optimize' runs MPPI and saves torques;
                        'visualize' replays saved torques in the viewer.
        n_samples:      Number of MPPI parallel rollout samples.
        horizon:        MPPI planning horizon in simulation steps.
        mppi_iters:     MPPI refinement iterations per dataset frame.
        joint_weight:   Cost weight for joint-angle tracking.
        obj_weight:     Cost weight for object-position tracking.
        contact_weight: Cost weight for fingertip-to-object proximity.
        effort_weight:  Cost weight penalising large torques (regularisation).
        noise_sigma:    Std-dev of Gaussian torque perturbations (Nm).
        temperature:    MPPI temperature lambda.
    """
    data_root = Path(dexycb_dir).resolve()
    if not data_root.exists():
        raise ValueError(f"DexYCB dir not found: {data_root}")

    if mode == "optimize":
        run_optimize(
            data_root, n_samples, horizon, mppi_iters,
            joint_weight, obj_weight, contact_weight, effort_weight,
            noise_sigma, temperature,
        )
    else:
        run_visualize(data_root)


if __name__ == "__main__":
    tyro.cli(main)
