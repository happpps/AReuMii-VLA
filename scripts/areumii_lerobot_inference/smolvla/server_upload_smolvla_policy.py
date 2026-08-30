#!/usr/bin/env python3
"""Local SmolVLA inference + LeRobot evaluation recorder for Areumii.

Run this file inside the *LeRobot* environment.
Run `play_smolvla_areumii_remote_lerobot_eval.py` inside the *IsaacLab*
environment.

Why recording lives here:
- IsaacLab and LeRobot stay in separate Python environments.
- The IsaacLab client already sends the exact observations used for inference.
- This server has both those observations and the 16-D policy action, so it can
  write a native LeRobotDataset directly without an HDF5 intermediate.

Protocol: length-prefixed Python pickle over localhost TCP.
Do not expose this port to untrusted networks.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import socket
import struct
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference


HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 128 * 1024 * 1024

STATE_NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_yaw",
    "left_wrist_pitch",
    "left_gripper_left_finger",
    "left_gripper_right_finger",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_yaw",
    "right_wrist_pitch",
    "right_gripper_left_finger",
    "right_gripper_right_finger",
]

ACTION_NAMES = [
    "right_ee_x",
    "right_ee_y",
    "right_ee_z",
    "right_ee_qw",
    "right_ee_qx",
    "right_ee_qy",
    "right_ee_qz",
    "right_gripper",
    "left_ee_x",
    "left_ee_y",
    "left_ee_z",
    "left_ee_qw",
    "left_ee_qx",
    "left_ee_qy",
    "left_ee_qz",
    "left_gripper",
]

CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Any:
    header = recv_exact(sock, HEADER.size)
    (size,) = HEADER.unpack(header)
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"Incoming message too large: {size} bytes")
    payload = recv_exact(sock, size)
    return pickle.loads(payload)


def send_message(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


def as_hwc_image(array: np.ndarray, name: str) -> np.ndarray:
    """Ensure an image is contiguous HWC RGB."""
    image = np.asarray(array)
    if image.ndim != 3:
        raise ValueError(f"{name}: expected 3-D image, got {image.shape}")

    if image.shape[-1] in (3, 4):
        return np.ascontiguousarray(image[..., :3])

    if image.shape[0] in (3, 4):
        image = np.moveaxis(image[:3], 0, -1)
        return np.ascontiguousarray(image)

    raise ValueError(f"{name}: cannot identify RGB channel dimension from shape {image.shape}")


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert HWC RGB to uint8 without darkening normalized float images."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    image = image.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]
    max_value = float(finite.max()) if finite.size else 0.0
    if max_value <= 1.5:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def make_lerobot_observation(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = ("joint_pos", "front_camera", "left_wrist_camera", "right_wrist_camera")
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"Missing observation(s): {missing}. Available: {sorted(raw.keys())}")

    return {
        "observation.state": np.asarray(raw["joint_pos"], dtype=np.float32).reshape(-1),
        "observation.images.front": as_hwc_image(raw["front_camera"], "front_camera"),
        "observation.images.left_wrist": as_hwc_image(raw["left_wrist_camera"], "left_wrist_camera"),
        "observation.images.right_wrist": as_hwc_image(raw["right_wrist_camera"], "right_wrist_camera"),
    }


def vector_names(dim: int, preferred: list[str], prefix: str) -> list[str]:
    if dim == len(preferred):
        return preferred
    return [f"{prefix}_{i}" for i in range(dim)]


class LeRobotEvalRecorder:
    """Incrementally write policy rollouts as native LeRobot episodes."""

    def __init__(
        self,
        repo_id: str | None,
        root: str,
        fps: int,
        robot_type: str,
        overwrite: bool,
        image_writer_threads: int,
        push_to_hub: bool,
    ):
        self.enabled = repo_id is not None
        self.repo_id = repo_id
        self.root = Path(root).expanduser().resolve()
        self.fps = int(fps)
        self.robot_type = robot_type
        self.overwrite = overwrite
        self.image_writer_threads = int(image_writer_threads)
        self.push_to_hub = push_to_hub

        self.dataset: LeRobotDataset | None = None
        self.current_rollout_index: int | None = None
        self.current_frame_count = 0
        self.finalized = False

        if not self.enabled:
            return

        if self.fps <= 0:
            raise ValueError("--dataset_fps must be positive")

        if self.root.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Dataset root already exists: {self.root}\n"
                    "Use --overwrite_dataset if you really want to replace it."
                )
            print(f"[DATASET] Removing existing dataset root: {self.root}")
            shutil.rmtree(self.root)

        print("[DATASET] Direct LeRobot recording enabled")
        print(f"[DATASET] repo_id : {self.repo_id}")
        print(f"[DATASET] root    : {self.root}")
        print(f"[DATASET] fps     : {self.fps}")
        print("[DATASET] The schema will be created from the first inference frame.")

    def _ensure_dataset(self, obs: dict[str, np.ndarray], action: np.ndarray) -> None:
        if not self.enabled or self.dataset is not None:
            return

        state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state.size,),
                "names": vector_names(state.size, STATE_NAMES, "state"),
            },
            "action": {
                "dtype": "float32",
                "shape": (action.size,),
                "names": vector_names(action.size, ACTION_NAMES, "action"),
            },
        }

        for key in CAMERA_KEYS:
            image = as_hwc_image(obs[key], key)
            h, w, c = image.shape
            features[key] = {
                "dtype": "video",
                "shape": (c, h, w),
                "names": ["channels", "height", "width"],
            }

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=features,
            root=self.root,
            robot_type=self.robot_type,
            use_videos=True,
            image_writer_threads=self.image_writer_threads,
            image_writer_processes=0,
            batch_encoding_size=1,
        )

        print("[DATASET] Created LeRobotDataset")
        for key, spec in features.items():
            print(f"          {key}: {spec['dtype']} {spec['shape']}")

    def start_episode(self, rollout_index: int) -> None:
        if not self.enabled:
            return
        if self.finalized:
            raise RuntimeError("Dataset was already finalized. Restart the server for a new recording.")

        if self.dataset is not None and self.dataset.has_pending_frames():
            print("[DATASET][WARN] Discarding unsaved frames from an interrupted previous rollout.")
            self.dataset.clear_episode_buffer()

        self.current_rollout_index = int(rollout_index)
        self.current_frame_count = 0

    def add_frame(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        task: str,
    ) -> None:
        if not self.enabled:
            return
        if self.current_rollout_index is None:
            raise RuntimeError("Received inference frame before episode_start.")

        self._ensure_dataset(obs, action)
        assert self.dataset is not None

        frame = {
            "observation.state": np.ascontiguousarray(
                np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
            ),
            "action": np.ascontiguousarray(
                np.asarray(action, dtype=np.float32).reshape(-1)
            ),
            "task": task,
        }

        for key in CAMERA_KEYS:
            frame[key] = to_uint8_rgb(as_hwc_image(obs[key], key))

        self.dataset.add_frame(frame)
        self.current_frame_count += 1

    def end_episode(self, result: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"recorded": False}

        if self.dataset is None or self.current_frame_count == 0:
            raise RuntimeError("episode_end received but no frames were recorded.")

        episode_index = int(self.dataset.meta.total_episodes)
        self.dataset.save_episode(parallel_encoding=False)

        record = {
            "episode_index": episode_index,
            "rollout_index": int(self.current_rollout_index),
            "success": bool(result.get("success", False)),
            "cube_x": result.get("cube_x"),
            "cube_y": result.get("cube_y"),
            "cube_z": result.get("cube_z"),
            "cube_side": result.get("cube_side"),
            "reason": str(result.get("reason", "")),
            "policy_steps": int(result.get("policy_steps", self.current_frame_count)),
            "env_steps": int(result.get("env_steps", 0)),
            "mean_inference_ms": result.get("mean_inference_ms"),
        }

        results_path = self.root / "eval_results.csv"
        write_header = not results_path.exists() or results_path.stat().st_size == 0
        with results_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        print(
            f"[DATASET] Saved episode {episode_index}: "
            f"rollout={self.current_rollout_index}, "
            f"frames={self.current_frame_count}, "
            f"cube_side={record['cube_side']}, "
            f"success={record['success']}"
        )

        self.current_rollout_index = None
        self.current_frame_count = 0
        return {"recorded": True, **record}

    def finalize(self) -> dict[str, Any]:
        if not self.enabled:
            return {"recorded": False, "finalized": False}
        if self.finalized:
            return {
                "recorded": True,
                "finalized": True,
                "repo_id": self.repo_id,
                "root": str(self.root),
            }

        if self.dataset is not None and self.dataset.has_pending_frames():
            print("[DATASET][WARN] Clearing incomplete rollout before finalization.")
            self.dataset.clear_episode_buffer()

        if self.dataset is not None:
            print("[DATASET] Finalizing LeRobotDataset...")
            self.dataset.finalize()

            if self.push_to_hub:
                print(f"[DATASET] Pushing to Hugging Face Hub: {self.repo_id}")
                self.dataset.push_to_hub()
                print("[DATASET] Hub upload complete.")

        self.finalized = True
        return {
            "recorded": self.dataset is not None,
            "finalized": True,
            "repo_id": self.repo_id,
            "root": str(self.root),
            "pushed_to_hub": bool(self.push_to_hub and self.dataset is not None),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a fine-tuned SmolVLA policy and optionally record rollouts directly as LeRobotDataset."
    )
    parser.add_argument("--model", default="1ys1/areumii-smolvla-pickplace-v1")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=5,
        help="Number of queued actions to execute before SmolVLA replans. Default: 5.",
    )

    # If --dataset_repo_id is omitted, the server behaves as inference-only.
    parser.add_argument(
        "--dataset_repo_id",
        default=None,
        help=(
            "Enable direct LeRobot rollout recording, e.g. "
            "1ys1/areumii-pickplace-eval-v3. Omit for inference-only."
        ),
    )
    parser.add_argument(
        "--dataset_root",
        default="./lerobot_datasets/areumii_pickplace_eval",
        help="Local root for the evaluation LeRobotDataset.",
    )
    parser.add_argument(
        "--dataset_fps",
        type=int,
        default=20,
        help="FPS written into LeRobot metadata. Use 20 for policy_period=0.05 s.",
    )
    parser.add_argument("--image_writer_threads", type=int, default=4)
    parser.add_argument(
        "--overwrite_dataset",
        action="store_true",
        help="Delete an existing --dataset_root before recording.",
    )
    parser.add_argument(
        "--push_dataset_to_hub",
        action="store_true",
        help="Push the finalized evaluation dataset to Hugging Face Hub.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    print(f"[SERVER] Loading SmolVLA: {args.model}")
    print(f"[SERVER] Device: {device}")
    policy = SmolVLAPolicy.from_pretrained(args.model)

    chunk_size = int(getattr(policy.config, "chunk_size", args.n_action_steps))
    if args.n_action_steps < 1 or args.n_action_steps > chunk_size:
        raise ValueError(
            f"--n_action_steps must be between 1 and chunk_size={chunk_size}, "
            f"got {args.n_action_steps}"
        )

    if hasattr(policy.config, "n_action_steps"):
        print(
            f"[SERVER] Checkpoint action config: "
            f"chunk_size={chunk_size}, "
            f"n_action_steps={policy.config.n_action_steps}"
        )
        policy.config.n_action_steps = args.n_action_steps
        print(
            f"[SERVER] Runtime action config: "
            f"chunk_size={chunk_size}, "
            f"n_action_steps={policy.config.n_action_steps}"
        )
    else:
        print("[SERVER][WARN] Policy config has no n_action_steps field; runtime override skipped.")

    policy = policy.to(device)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        args.model,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    recorder = LeRobotEvalRecorder(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        fps=args.dataset_fps,
        robot_type="areumii_c1",
        overwrite=args.overwrite_dataset,
        image_writer_threads=args.image_writer_threads,
        push_to_hub=args.push_dataset_to_hub,
    )

    print("[SERVER] Policy input features:")
    for key, feature in policy.config.input_features.items():
        print(f"         {key}: {feature}")
    print("[SERVER] Policy output features:")
    for key, feature in policy.config.output_features.items():
        print(f"         {key}: {feature}")

    first_inference = True

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"[SERVER] Listening on {args.host}:{args.port}")
        print("[SERVER] Start play_smolvla_areumii_remote_lerobot_eval.py in the IsaacLab environment.")

        while True:
            conn, addr = server.accept()
            print(f"[SERVER] Client connected: {addr}")
            policy.reset()

            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    try:
                        request = recv_message(conn)
                    except ConnectionError:
                        print("[SERVER] Client disconnected")
                        break

                    try:
                        msg_type = request.get("type")

                        if msg_type == "ping":
                            send_message(conn, {"ok": True, "type": "pong"})
                            continue

                        if msg_type == "reset":
                            # Backwards-compatible inference-only reset.
                            policy.reset()
                            send_message(conn, {"ok": True, "type": "reset"})
                            continue

                        if msg_type == "episode_start":
                            policy.reset()
                            recorder.start_episode(int(request["rollout_index"]))
                            send_message(
                                conn,
                                {
                                    "ok": True,
                                    "type": "episode_start",
                                    "rollout_index": int(request["rollout_index"]),
                                },
                            )
                            continue

                        if msg_type == "episode_end":
                            saved = recorder.end_episode(request.get("result", {}))
                            send_message(conn, {"ok": True, "type": "episode_end", **saved})
                            continue

                        if msg_type == "finalize_dataset":
                            info = recorder.finalize()
                            send_message(conn, {"ok": True, "type": "finalize_dataset", **info})
                            continue

                        if msg_type == "close":
                            send_message(conn, {"ok": True, "type": "close"})
                            print("[SERVER] Client requested close")
                            break

                        if msg_type != "infer":
                            raise ValueError(f"Unknown message type: {msg_type!r}")

                        raw_obs = request["observation"]
                        task = request.get("task", "")
                        robot_type = request.get("robot_type", "")
                        obs_np = make_lerobot_observation(raw_obs)

                        # IMPORTANT:
                        # prepare_observation_for_inference() may convert/mutate the
                        # observation dict values into Torch tensors on the target
                        # device. Keep an untouched CPU NumPy copy for LeRobotDataset
                        # recording before passing the dict to the inference helper.
                        record_obs = {
                            key: np.array(value, copy=True)
                            for key, value in obs_np.items()
                        }

                        if first_inference:
                            first_inference = False
                            print("[SERVER] First observation:")
                            for key, value in obs_np.items():
                                print(f"         {key}: shape={value.shape}, dtype={value.dtype}")
                            cfg_state = policy.config.input_features.get("observation.state")
                            print(f"[SERVER] Policy config observation.state: {cfg_state}")

                        frame = prepare_observation_for_inference(
                            obs_np,
                            device=device,
                            task=task,
                            robot_type=robot_type,
                        )

                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        t0 = time.perf_counter()

                        with torch.inference_mode():
                            model_input = preprocessor(frame)
                            action = policy.select_action(model_input)
                            action = postprocessor(action)

                        if not isinstance(action, torch.Tensor):
                            action = torch.as_tensor(action)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        inference_ms = (time.perf_counter() - t0) * 1000.0

                        action_np = (
                            action.detach()
                            .float()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                            .astype(np.float32)
                        )
                        if not np.isfinite(action_np).all():
                            raise RuntimeError(f"Policy returned NaN/Inf action: {action_np}")

                        # Record the original CPU NumPy observation paired with
                        # the action returned for that observation.
                        recorder.add_frame(record_obs, action_np, task)

                        send_message(
                            conn,
                            {
                                "ok": True,
                                "action": action_np,
                                "inference_ms": inference_ms,
                            },
                        )

                    except Exception as exc:
                        error_text = "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        )
                        print("[SERVER] Error:\n" + error_text)
                        try:
                            send_message(conn, {"ok": False, "error": error_text})
                        except Exception:
                            pass


if __name__ == "__main__":
    main()