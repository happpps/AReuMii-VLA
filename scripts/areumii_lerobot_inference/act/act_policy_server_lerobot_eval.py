#!/usr/bin/env python3
"""Local ACT inference server + optional LeRobot rollout recorder for Areumii.

Run this file in the LeRobot environment.
Run `play_act_areumii_remote_lerobot_eval.py` in the IsaacLab environment.

The split keeps IsaacLab/Isaac Sim dependencies separate from LeRobot.
The ACT policy receives:
  - observation.state
  - observation.images.front
  - observation.images.left_wrist
  - observation.images.right_wrist

and returns the same 16-D action used by the Areumii IK-Abs environment.

If --dataset_repo_id is given, each evaluation rollout is also saved directly
as a native LeRobotDataset episode for later visualization.
"""

from __future__ import annotations

import argparse
import json
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
import torch.nn.functional as F

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference


HEADER = struct.Struct("!Q")
MAX_MESSAGE_BYTES = 128 * 1024 * 1024

CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)

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


# -----------------------------------------------------------------------------
# Socket protocol
# -----------------------------------------------------------------------------

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
    return pickle.loads(recv_exact(sock, size))


def send_message(sock: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


# -----------------------------------------------------------------------------
# Observation helpers
# -----------------------------------------------------------------------------

def as_hwc_rgb(array: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim != 3:
        raise ValueError(f"{name}: expected 3-D image, got {image.shape}")

    if image.shape[-1] in (3, 4):
        return np.ascontiguousarray(image[..., :3])

    if image.shape[0] in (3, 4):
        image = np.moveaxis(image[:3], 0, -1)
        return np.ascontiguousarray(image)

    raise ValueError(f"{name}: cannot identify RGB channel dimension from shape {image.shape}")


def resize_hwc(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize HWC image with bilinear interpolation while preserving value scale."""
    image = np.ascontiguousarray(image)
    if image.shape[:2] == (height, width):
        return image

    original_dtype = image.dtype
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor = F.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    tensor = tensor.squeeze(0).permute(1, 2, 0)

    if original_dtype == np.uint8:
        tensor = tensor.round().clamp(0, 255)
        out = tensor.to(torch.uint8).cpu().numpy()
    else:
        out = tensor.cpu().numpy().astype(original_dtype, copy=False)

    return np.ascontiguousarray(out)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    image = image.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]
    max_value = float(finite.max()) if finite.size else 0.0

    # IsaacLab cameras may be float [0, 1] or uint8-like [0, 255].
    if max_value <= 1.5:
        image = image * 255.0

    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def make_raw_lerobot_observation(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = ("joint_pos", "front_camera", "left_wrist_camera", "right_wrist_camera")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Missing observation(s): {missing}. Available: {sorted(raw.keys())}")

    return {
        "observation.state": np.asarray(raw["joint_pos"], dtype=np.float32).reshape(-1),
        "observation.images.front": as_hwc_rgb(raw["front_camera"], "front_camera"),
        "observation.images.left_wrist": as_hwc_rgb(raw["left_wrist_camera"], "left_wrist_camera"),
        "observation.images.right_wrist": as_hwc_rgb(raw["right_wrist_camera"], "right_wrist_camera"),
    }


def make_act_model_observation(
    raw_obs: dict[str, np.ndarray],
    image_height: int,
    image_width: int,
) -> dict[str, np.ndarray]:
    """Create the observation ACT sees.

    Training used a deterministic Resize([240, 320]) transform, so inference
    must use the same image size. The evaluation recorder still stores the
    original camera resolutions.
    """
    obs = {
        "observation.state": np.array(raw_obs["observation.state"], copy=True),
    }

    for key in CAMERA_KEYS:
        obs[key] = resize_hwc(
            as_hwc_rgb(raw_obs[key], key),
            height=image_height,
            width=image_width,
        )

    return obs


def reset_if_available(obj: Any) -> None:
    reset = getattr(obj, "reset", None)
    if callable(reset):
        reset()


# -----------------------------------------------------------------------------
# Optional LeRobot evaluation recorder
# -----------------------------------------------------------------------------

def generic_names(dim: int, prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(dim)]


class LeRobotEvalRecorder:
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
        self.current_task = "ACT evaluation"
        self.finalized = False

        if not self.enabled:
            return

        if self.root.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Dataset root already exists: {self.root}\n"
                    "Pass --overwrite_dataset to replace it."
                )
            print(f"[DATASET] Removing existing root: {self.root}")
            shutil.rmtree(self.root)

        print("[DATASET] Direct LeRobot recording enabled")
        print(f"[DATASET] repo_id : {self.repo_id}")
        print(f"[DATASET] root    : {self.root}")
        print(f"[DATASET] fps     : {self.fps}")

    def _ensure_dataset(self, obs: dict[str, np.ndarray], action: np.ndarray) -> None:
        if not self.enabled or self.dataset is not None:
            return

        state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state.size,),
                # Generic names avoid claiming an incorrect joint ordering.
                "names": generic_names(state.size, "joint"),
            },
            "action": {
                "dtype": "float32",
                "shape": (action.size,),
                "names": ACTION_NAMES if action.size == len(ACTION_NAMES)
                else generic_names(action.size, "action"),
            },
        }

        for key in CAMERA_KEYS:
            image = as_hwc_rgb(obs[key], key)
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

    def start_episode(self, rollout_index: int, task: str) -> None:
        if not self.enabled:
            return
        if self.finalized:
            raise RuntimeError("Dataset already finalized. Restart the server.")

        if self.dataset is not None and self.dataset.has_pending_frames():
            print("[DATASET][WARN] Clearing unfinished previous episode.")
            self.dataset.clear_episode_buffer()

        self.current_rollout_index = int(rollout_index)
        self.current_frame_count = 0
        self.current_task = task or "ACT evaluation"

    def add_frame(
        self,
        raw_obs: dict[str, np.ndarray],
        action: np.ndarray,
    ) -> None:
        if not self.enabled:
            return
        if self.current_rollout_index is None:
            raise RuntimeError("Inference frame received before episode_start.")

        self._ensure_dataset(raw_obs, action)
        assert self.dataset is not None

        frame = {
            "observation.state": np.ascontiguousarray(
                np.asarray(raw_obs["observation.state"], dtype=np.float32).reshape(-1)
            ),
            "action": np.ascontiguousarray(
                np.asarray(action, dtype=np.float32).reshape(-1)
            ),
            "task": self.current_task,
        }

        # Save original camera resolution for easier visual inspection.
        for key in CAMERA_KEYS:
            frame[key] = to_uint8_rgb(as_hwc_rgb(raw_obs[key], key))

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
            "reason": str(result.get("reason", "")),
            "policy_steps": int(result.get("policy_steps", self.current_frame_count)),
            "env_steps": int(result.get("env_steps", 0)),
            "mean_inference_ms": result.get("mean_inference_ms"),
            "max_inference_ms": result.get("max_inference_ms"),
            "initial_cube_xyz": result.get("initial_cube_xyz"),
        }

        results_path = self.root / "eval_results.jsonl"
        with results_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"[DATASET] Saved episode {episode_index}: "
            f"rollout={self.current_rollout_index}, "
            f"frames={self.current_frame_count}, "
            f"success={record['success']}, "
            f"cube_xyz={record['initial_cube_xyz']}"
        )

        self.current_rollout_index = None
        self.current_frame_count = 0
        return {"recorded": True, **record}

    def finalize(self) -> dict[str, Any]:
        if not self.enabled:
            return {"recorded": False, "finalized": False}

        if self.finalized:
            return {
                "recorded": self.dataset is not None,
                "finalized": True,
                "repo_id": self.repo_id,
                "root": str(self.root),
            }

        if self.dataset is not None and self.dataset.has_pending_frames():
            print("[DATASET][WARN] Clearing incomplete rollout before finalize.")
            self.dataset.clear_episode_buffer()

        if self.dataset is not None:
            print("[DATASET] Finalizing...")
            self.dataset.finalize()

            if self.push_to_hub:
                print(f"[DATASET] Pushing to Hub: {self.repo_id}")
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


# -----------------------------------------------------------------------------
# Main server
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve an ACT checkpoint to IsaacLab over localhost TCP."
    )
    parser.add_argument(
        "--model",
        default="1ys1/areumii-act-pickplace-v4",
        help="Hub repo or local ACT pretrained_model directory.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Override checkpoint n_action_steps. Omit to use the saved value.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=240,
        help="ACT inference image height. Match the training Resize transform.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=320,
        help="ACT inference image width. Match the training Resize transform.",
    )

    parser.add_argument(
        "--dataset_repo_id",
        default=None,
        help="Optional eval dataset repo, e.g. 1ys1/eval_act_areumii-pickplace-v4.",
    )
    parser.add_argument(
        "--dataset_root",
        default="./lerobot_datasets/eval_act_areumii_pickplace_v4",
    )
    parser.add_argument("--dataset_fps", type=int, default=20)
    parser.add_argument("--image_writer_threads", type=int, default=4)
    parser.add_argument("--overwrite_dataset", action="store_true")
    parser.add_argument("--push_dataset_to_hub", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    print(f"[SERVER] Loading ACT: {args.model}")
    print(f"[SERVER] Device: {device}")
    policy = ACTPolicy.from_pretrained(args.model)

    chunk_size = int(policy.config.chunk_size)
    saved_n = int(policy.config.n_action_steps)
    runtime_n = saved_n if args.n_action_steps is None else int(args.n_action_steps)

    if runtime_n < 1 or runtime_n > chunk_size:
        raise ValueError(
            f"n_action_steps must be 1..{chunk_size}, got {runtime_n}"
        )

    print(
        f"[SERVER] ACT action config: chunk_size={chunk_size}, "
        f"saved n_action_steps={saved_n}, runtime n_action_steps={runtime_n}"
    )
    policy.config.n_action_steps = runtime_n

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

    required_policy_inputs = {
        "observation.state",
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    missing = required_policy_inputs - set(policy.config.input_features)
    if missing:
        raise RuntimeError(
            "ACT checkpoint input features do not match the Areumii dataset. "
            f"Missing from policy config: {sorted(missing)}"
        )

    print(
        f"[SERVER] ACT inference images will be resized to "
        f"{args.image_height}x{args.image_width}"
    )

    first_inference = True

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)

        print(f"[SERVER] Listening on {args.host}:{args.port}")
        print("[SERVER] Start play_act_areumii_remote_lerobot_eval.py in IsaacLab.")

        while True:
            conn, addr = server.accept()
            print(f"[SERVER] Client connected: {addr}")

            policy.reset()
            reset_if_available(preprocessor)
            reset_if_available(postprocessor)

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

                        if msg_type == "episode_start":
                            policy.reset()
                            reset_if_available(preprocessor)
                            reset_if_available(postprocessor)

                            recorder.start_episode(
                                int(request["rollout_index"]),
                                str(request.get("record_task", "ACT evaluation")),
                            )
                            send_message(conn, {"ok": True, "type": "episode_start"})
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

                        raw = request["observation"]
                        raw_obs = make_raw_lerobot_observation(raw)

                        # Preserve original CPU NumPy data for eval dataset recording.
                        record_obs = {
                            key: np.array(value, copy=True)
                            for key, value in raw_obs.items()
                        }

                        # ACT training used Resize([240, 320]); reproduce it here.
                        model_obs = make_act_model_observation(
                            raw_obs,
                            image_height=args.image_height,
                            image_width=args.image_width,
                        )

                        if first_inference:
                            first_inference = False
                            print("[SERVER] First raw observation:")
                            for key, value in raw_obs.items():
                                print(f"         {key}: shape={value.shape}, dtype={value.dtype}")
                            print("[SERVER] First ACT model observation:")
                            for key, value in model_obs.items():
                                print(f"         {key}: shape={value.shape}, dtype={value.dtype}")

                        frame = prepare_observation_for_inference(
                            model_obs,
                            device=device,
                            task=None,
                            robot_type=None,
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
                            raise RuntimeError(f"ACT returned NaN/Inf: {action_np}")

                        recorder.add_frame(record_obs, action_np)

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
