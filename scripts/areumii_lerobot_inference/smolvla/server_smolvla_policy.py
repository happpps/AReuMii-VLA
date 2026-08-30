#!/usr/bin/env python3
"""Local SmolVLA inference server for Areumii.

Run this file inside the *LeRobot* environment, not the IsaacLab environment.
It listens only on localhost by default and receives observations from
`play_smolvla_areumii_remote.py`.

Protocol: length-prefixed Python pickle over TCP. This is intended only for a
trusted localhost connection. Do not expose the port to untrusted networks.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time
import traceback
from typing import Any

import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference


HEADER = struct.Struct("!Q")  # unsigned 64-bit payload length
MAX_MESSAGE_BYTES = 128 * 1024 * 1024


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
    """Ensure an image is HWC RGB(A), as expected by LeRobot's inference helper."""
    image = np.asarray(array)
    if image.ndim != 3:
        raise ValueError(f"{name}: expected 3-D image, got {image.shape}")

    # Normal case from IsaacLab camera observations.
    if image.shape[-1] in (3, 4):
        return np.ascontiguousarray(image[..., :3])

    # Tolerate CHW if an IsaacLab version/config returns that instead.
    if image.shape[0] in (3, 4):
        image = np.moveaxis(image[:3], 0, -1)
        return np.ascontiguousarray(image)

    raise ValueError(f"{name}: cannot identify RGB channel dimension from shape {image.shape}")


def make_lerobot_observation(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = ("joint_pos", "front_camera", "left_wrist_camera", "right_wrist_camera")
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"Missing observation(s): {missing}. Available: {sorted(raw.keys())}")

    return {
        # Keep the original dataset keys. The checkpoint's saved preprocessor
        # applies the same rename_map that was used during training.
        "observation.state": np.asarray(raw["joint_pos"], dtype=np.float32).reshape(-1),
        "observation.images.front": as_hwc_image(raw["front_camera"], "front_camera"),
        "observation.images.left_wrist": as_hwc_image(raw["left_wrist_camera"], "left_wrist_camera"),
        "observation.images.right_wrist": as_hwc_image(raw["right_wrist_camera"], "right_wrist_camera"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a fine-tuned SmolVLA policy to IsaacLab over localhost TCP.")
    parser.add_argument("--model", default="1ys1/areumii-smolvla-pickplace-v1")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
    "--n_action_steps",
    type=int,
    default=5,
    help="Number of predicted actions to execute before replanning.",
    )
    
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    print(f"[SERVER] Loading SmolVLA: {args.model}")
    print(f"[SERVER] Device: {device}")

    policy = SmolVLAPolicy.from_pretrained(args.model)

    print(
        f"[SERVER] Checkpoint config: "
        f"chunk_size={policy.config.chunk_size}, "
        f"n_action_steps={policy.config.n_action_steps}"
    )

    if args.n_action_steps < 1 or args.n_action_steps > policy.config.chunk_size:
        raise ValueError(
            f"--n_action_steps must be between 1 and "
            f"{policy.config.chunk_size}, got {args.n_action_steps}"
        )

    policy.config.n_action_steps = args.n_action_steps
    policy.reset()

    print(
        f"[SERVER] Inference config: "
        f"chunk_size={policy.config.chunk_size}, "
        f"n_action_steps={policy.config.n_action_steps}"
    )

    policy = policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        args.model,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
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
        print("[SERVER] Start play_smolvla_areumii_remote.py in the IsaacLab environment.")

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
                            policy.reset()
                            send_message(conn, {"ok": True, "type": "reset"})
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

                        if first_inference:
                            first_inference = False
                            print("[SERVER] First observation:")
                            for key, value in obs_np.items():
                                print(f"         {key}: shape={value.shape}, dtype={value.dtype}")
                            cfg_state = policy.config.input_features.get("observation.state")
                            print(f"[SERVER] Policy config observation.state: {cfg_state}")

                        # LeRobot v0.6.1's official inference helper converts HWC NumPy
                        # images -> batched CHW float tensors and adds task/robot_type.
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

                        action_np = action.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
                        if not np.isfinite(action_np).all():
                            raise RuntimeError(f"Policy returned NaN/Inf action: {action_np}")

                        send_message(
                            conn,
                            {
                                "ok": True,
                                "action": action_np,
                                "inference_ms": inference_ms,
                            },
                        )

                    except Exception as exc:
                        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                        print("[SERVER] Inference error:\n" + error_text)
                        try:
                            send_message(conn, {"ok": False, "error": error_text})
                        except Exception:
                            pass


if __name__ == "__main__":
    main()
