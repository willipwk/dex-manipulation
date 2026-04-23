"""
Train a PPO policy for Leap Hand force-control tracking using rsl_rl.

Usage
-----
  # Train
  python rl_hand_train.py --dexycb_dir /path/to/dexycb

  # Train with viewer
  python rl_hand_train.py --dexycb_dir /path/to/dexycb --vis

  # Resume from checkpoint
  python rl_hand_train.py --dexycb_dir /path/to/dexycb --resume
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
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

from rl_hand_env import LeapHandTrackingEnv


def get_train_cfg(exp_name: str, max_iterations: int) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "init_noise_std": 0.5,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 50,
        "empirical_normalization": None,
        "seed": 1,
    }


def get_env_cfg(dexycb_dir: str, num_envs: int) -> tuple[dict, dict]:
    env_cfg = {
        "num_envs": num_envs,
        "num_obs": LeapHandTrackingEnv.NUM_OBS,
        "num_actions": 16,
        "dexycb_dir": dexycb_dir,
        "ctrl_dt": 4e-3,
        "substeps": 10,
        "sim_steps_per_action": 10,
    }
    reward_cfg = {
        "joint_weight": 5.0,
        "obj_weight": 20.0,
        "contact_weight": 5.0,
        "effort_weight": 0.005,
    }
    return env_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser(description="Train Leap Hand tracking policy with PPO")
    parser.add_argument("--dexycb_dir", type=str, required=True, help="Path to DexYCB dataset")
    parser.add_argument("-e", "--exp_name", type=str, default="leap_hand_tracking")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        performance_mode=True,
    )

    env_cfg, reward_cfg = get_env_cfg(args.dexycb_dir, args.num_envs)
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    log_dir = Path("logs") / f"{args.exp_name}_rl"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_cfg, train_cfg), f)

    env = LeapHandTrackingEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if args.resume:
        import re
        ckpt_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
        if ckpt_files:
            last_ckpt = max(ckpt_files, key=lambda f: int(re.search(r"\d+", f.stem).group()))
            runner.load(last_ckpt)
            print(f"Resumed from {last_ckpt}")
        else:
            print("No checkpoint found, starting from scratch")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
