#!/usr/bin/env python3
"""Closed-loop Areumii PickPlace evaluation client for a remote ACT policy.

Run this file in the IsaacLab environment.
Run `act_policy_server_lerobot_eval.py` in the LeRobot environment first.

Each rollout:
  1. resets the IsaacLab environment,
  2. resets the ACT action queue on the server,
  3. evaluates until success/termination/horizon,
  4. optionally saves the rollout through the server as a LeRobot episode.

The initial cube XYZ is included in the result JSON so success rate can later
be analyzed as a function of cube spawn position.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI / Isaac Sim launch
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Evaluate Areumii PickPlace using a remote ACT policy."
)
parser.add_argument("--task", required=True, type=str)
parser.add_argument(
    "--task_module",
    default=None,
    type=str,
    help="Optional module that registers a custom Gym task.",
)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--num_rollouts", type=int, default=30)
parser.add_argument("--horizon", type=int, default=150)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument(
    "--policy_period",
    type=float,
    default=0.05,
    help="Action period in simulation seconds. 0.05 = 20 Hz.",
)
parser.add_argument(
    "--record_task",
    default="Areumii ACT pick and place evaluation",
    help="Metadata label only. ACT itself does not use language.",
)
parser.add_argument(
    "--cube_asset_name",
    default="cube_1",
    help="Scene asset used to log initial cube XYZ.",
)
parser.add_argument("--results", default="act_eval_results_remote.json")
parser.add_argument("--disable_fabric", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

launcher_cfg = {
    "headless": False,
    "livestream": 0,
    "xr": False,
    "enable_cameras": True,
    "visualizer": ["kit"],
    "device": getattr(args_cli, "device", "cuda:0"),
    "experience": "isaaclab.python.rendering.kit",
}

app_launcher = AppLauncher(launcher_cfg)
simulation_app = app_launcher.app

print("\n[GUI CHECK]")
print("headless       =", app_launcher._headless)
print("livestream     =", app_launcher._livestream)
print("xr             =", app_launcher._xr)
print("enable_cameras =", app_launcher._enable_cameras)
print("visualizer     = kit")
print("experience     =", app_launcher._sim_experience_file)
print()


# -----------------------------------------------------------------------------
# Imports after AppLauncher
# -----------------------------------------------------------------------------

import importlib
import json
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.areumii_pick_place_one.config.areumii_c1.random_pick_place_record_env_cfg import (
    AreumiiPickPlaceRecordEnvCfg,
)


HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 128 * 1024 * 1024
REQUIRED_OBS = (
    "joint_pos",
    "front_camera",
    "left_wrist_camera",
    "right_wrist_camera",
)


# -----------------------------------------------------------------------------
# Socket protocol
# -----------------------------------------------------------------------------

def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Policy server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Any:
    header = recv_exact(sock, HEADER.size)
    (size,) = HEADER.unpack(header)
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"Incoming message too large: {size} bytes")
    return pickle.loads(recv_exact(sock, size))


def send_message(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


class PolicyClient:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=10.0)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.request({"type": "ping"})

    def request(self, payload: dict) -> dict:
        send_message(self.sock, payload)
        response = recv_message(self.sock)
        if not response.get("ok", False):
            raise RuntimeError(
                "ACT policy server error:\n" + response.get("error", "unknown error")
            )
        return response

    def start_episode(self, rollout_index: int, record_task: str) -> None:
        self.request(
            {
                "type": "episode_start",
                "rollout_index": int(rollout_index),
                "record_task": record_task,
            }
        )

    def infer(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        response = self.request(
            {
                "type": "infer",
                "observation": observation,
            }
        )
        return (
            np.asarray(response["action"], dtype=np.float32),
            float(response["inference_ms"]),
        )

    def end_episode(self, result: dict) -> dict:
        return self.request({"type": "episode_end", "result": result})

    def finalize_dataset(self) -> dict:
        return self.request({"type": "finalize_dataset"})

    def close(self) -> None:
        try:
            self.request({"type": "close"})
        except Exception:
            pass
        self.sock.close()


# -----------------------------------------------------------------------------
# IsaacLab helpers
# -----------------------------------------------------------------------------

def first_env_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim > 0:
            tensor = tensor[0]
        return np.ascontiguousarray(tensor.cpu().numpy())

    array = np.asarray(value)
    if array.ndim > 0:
        array = array[0]
    return np.ascontiguousarray(array)


def build_remote_observation(policy_obs: dict) -> dict[str, np.ndarray]:
    missing = [key for key in REQUIRED_OBS if key not in policy_obs]
    if missing:
        raise KeyError(
            f"Missing policy observations: {missing}. "
            f"Available: {sorted(policy_obs.keys())}"
        )

    return {
        key: first_env_numpy(policy_obs[key])
        for key in REQUIRED_OBS
    }


def tensor_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    return bool(value)


def action_dim(action_space) -> int | None:
    shape = getattr(action_space, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[-1])


def try_get_initial_cube_xyz(env) -> list[float] | None:
    try:
        cube = env.scene[args_cli.cube_asset_name]
        pos = cube.data.root_pos_w[0].detach().cpu().numpy().reshape(-1)
        if pos.size >= 3:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception as exc:
        print(
            f"[WARN] Could not read initial cube XYZ from "
            f"{args_cli.cube_asset_name!r}: {exc}"
        )
    return None


def rollout(
    env,
    client: PolicyClient,
    success_term,
    action_repeat: int,
    rollout_index: int,
) -> dict:
    # This reset is performed for EVERY rollout.
    obs_dict, _ = env.reset()

    initial_cube_xyz = try_get_initial_cube_xyz(env)
    print(f"[RESET] rollout={rollout_index} initial_cube_xyz={initial_cube_xyz}")

    # This also resets ACT's internal action queue for every rollout.
    client.start_episode(rollout_index, args_cli.record_task)

    inference_times = []
    success = False
    reason = "horizon"
    env_steps = 0
    num_policy_steps = 0

    if "policy" not in obs_dict:
        raise KeyError(
            f"Observation has no 'policy' group. Available: {list(obs_dict.keys())}"
        )

    for policy_step in range(args_cli.horizon):
        if not simulation_app.is_running():
            reason = "sim_closed"
            break

        remote_obs = build_remote_observation(obs_dict["policy"])
        action_np, inference_ms = client.infer(remote_obs)

        inference_times.append(inference_ms)
        num_policy_steps += 1

        expected_dim = action_dim(env.action_space)
        if expected_dim is not None and action_np.size != expected_dim:
            raise ValueError(
                f"ACT returned {action_np.size} action values but "
                f"env.action_space expects {expected_dim}."
            )

        if not np.isfinite(action_np).all():
            raise RuntimeError(f"ACT returned NaN/Inf: {action_np}")

        env_action = torch.as_tensor(
            action_np,
            device=env.device,
            dtype=torch.float32,
        ).reshape(1, -1)

        done = False
        for _ in range(action_repeat):
            obs_dict, _, terminated, truncated, _ = env.step(env_action)
            env_steps += 1

            if success_term is not None:
                success_now = success_term.func(env, **success_term.params)
                if tensor_bool(success_now):
                    success = True
                    reason = "success"
                    done = True
                    break

            if tensor_bool(terminated):
                reason = "terminated"
                done = True
                break

            if tensor_bool(truncated):
                reason = "truncated"
                done = True
                break

        if args_cli.print_every > 0 and (
            policy_step == 0
            or (policy_step + 1) % args_cli.print_every == 0
            or done
        ):
            print(
                f"[STEP {policy_step + 1:04d}] "
                f"infer={inference_ms:7.1f} ms "
                f"action={np.array2string(action_np, precision=3, suppress_small=True)}"
            )

        if done:
            if success:
                print(
                    f"[SUCCESS] policy_step={policy_step + 1}, "
                    f"env_steps={env_steps}"
                )
            break

    return {
        "rollout_index": int(rollout_index),
        "success": success,
        "policy_steps": num_policy_steps,
        "env_steps": env_steps,
        "reason": reason,
        "mean_inference_ms": (
            float(np.mean(inference_times)) if inference_times else None
        ),
        "max_inference_ms": (
            float(np.max(inference_times)) if inference_times else None
        ),
        "initial_cube_xyz": initial_cube_xyz,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    if args_cli.task_module:
        print(f"[INFO] Importing custom task module: {args_cli.task_module}")
        importlib.import_module(args_cli.task_module)

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    print("[INFO] Building AreumiiPickPlaceRecordEnvCfg directly...", flush=True)
    env_cfg = AreumiiPickPlaceRecordEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args_cli.seed

    env_cfg.observations.policy.concatenate_terms = False

    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None

    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is not None:
        # Count success ourselves so Gym does not auto-reset before we save result.
        env_cfg.terminations.success = None
    else:
        print("[WARN] No termination term named 'success' was found.")

    print(f"[INFO] Creating Gym environment: {args_cli.task}", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print("[INFO] Gym environment created successfully.", flush=True)

    env_step_dt = float(getattr(env, "step_dt", args_cli.policy_period))
    action_repeat = max(
        1,
        int(round(args_cli.policy_period / env_step_dt)),
    )
    effective_period = action_repeat * env_step_dt

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Env device: {env.device}")
    print(f"[INFO] Env action space: {env.action_space}")
    print(f"[INFO] env.step_dt: {env_step_dt:.6f} s")
    print(f"[INFO] Policy target period: {args_cli.policy_period:.6f} s")
    print(
        f"[INFO] Holding each ACT action for {action_repeat} env.step() "
        f"call(s) -> {effective_period:.6f} s"
    )

    first_obs, _ = env.reset()
    available = sorted(first_obs.get("policy", {}).keys())
    print(f"[CHECK] Policy observation keys: {available}")
    _ = build_remote_observation(first_obs["policy"])
    print("[CHECK] Camera/state observations are present.")

    print(
        f"[INFO] Connecting to ACT server at "
        f"{args_cli.host}:{args_cli.port} ..."
    )
    try:
        client = PolicyClient(args_cli.host, args_cli.port)
    except OSError as exc:
        raise RuntimeError(
            f"Could not connect to ACT server at {args_cli.host}:{args_cli.port}. "
            "Start act_policy_server_lerobot_eval.py in env_lerobot first."
        ) from exc

    print("[INFO] ACT server connected.")

    results = []
    finalized_info = None

    try:
        for trial in range(args_cli.num_rollouts):
            if not simulation_app.is_running():
                break

            print(
                f"\n[INFO] ===== Rollout "
                f"{trial + 1}/{args_cli.num_rollouts} ====="
            )

            result = rollout(
                env,
                client,
                success_term,
                action_repeat,
                rollout_index=trial,
            )

            dataset_result = client.end_episode(result)
            if dataset_result.get("recorded", False):
                result["lerobot_episode_index"] = int(
                    dataset_result["episode_index"]
                )

            results.append(result)

            if result["mean_inference_ms"] is not None:
                print(
                    f"[RESULT] success={result['success']} "
                    f"reason={result['reason']} "
                    f"policy_steps={result['policy_steps']} "
                    f"mean_inference={result['mean_inference_ms']:.1f} ms "
                    f"cube_xyz={result['initial_cube_xyz']}"
                )
            else:
                print(
                    f"[RESULT] success={result['success']} "
                    f"reason={result['reason']} "
                    f"cube_xyz={result['initial_cube_xyz']}"
                )

        finalized_info = client.finalize_dataset()

        if finalized_info.get("recorded", False):
            print("\n[DATASET] Evaluation dataset finalized")
            print(f"[DATASET] root    : {finalized_info.get('root')}")
            print(f"[DATASET] repo_id : {finalized_info.get('repo_id')}")
            print(f"[DATASET] pushed  : {finalized_info.get('pushed_to_hub', False)}")

    finally:
        client.close()
        env.close()

    num_success = sum(int(result["success"]) for result in results)
    success_rate = num_success / len(results) if results else 0.0

    summary = {
        "policy": "ACT",
        "task": args_cli.task,
        "policy_period_s": args_cli.policy_period,
        "env_step_dt_s": env_step_dt,
        "action_repeat": action_repeat,
        "horizon_policy_steps": args_cli.horizon,
        "num_rollouts": len(results),
        "num_success": num_success,
        "success_rate": success_rate,
        "lerobot_dataset": finalized_info,
        "rollouts": results,
    }

    print("\n[SUMMARY]")
    print(f"Successful rollouts: {num_success}/{len(results)}")
    print(f"Success rate: {success_rate:.3f}")

    path = Path(args_cli.results).expanduser()
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Saved: {path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
