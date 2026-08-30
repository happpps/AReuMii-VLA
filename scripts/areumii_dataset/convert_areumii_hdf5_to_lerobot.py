#!/usr/bin/env python3
"""
Convert an Isaac Lab Areumii HDF5 demonstration dataset to LeRobotDataset v3.

Tested against the LeRobot v0.6.1 public API shape:
    LeRobotDataset.create(...)
    dataset.add_frame(...)
    dataset.save_episode(...)
    dataset.finalize()
    dataset.push_to_hub()

Default mapping
---------------
HDF5:
    /data/demo_N/obs/joint_pos
        -> observation.state        (18D)

    /data/demo_N/actions
        -> action                   (16D)
           [right EE pose 7, right grip 1,
            left EE pose 7, left grip 1]

    /data/demo_N/obs/front_camera
        -> observation.images.front

    /data/demo_N/obs/left_wrist_camera
        -> observation.images.left_wrist

    /data/demo_N/obs/right_wrist_camera
        -> observation.images.right_wrist

The converter automatically uses whichever of the three cameras are present.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets import LeRobotDataset


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

RAW_ACTION_NAMES = [
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

CAMERA_MAP = {
    "front_camera": "observation.images.front",
    "left_wrist_camera": "observation.images.left_wrist",
    "right_wrist_camera": "observation.images.right_wrist",
}


def demo_sort_key(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 10**9


def inspect_hdf5(h5: h5py.File, action_source: str):
    if "data" not in h5:
        raise KeyError("HDF5 has no /data group.")

    demo_names = sorted(
        [k for k in h5["data"].keys() if k.startswith("demo_")],
        key=demo_sort_key,
    )
    if not demo_names:
        raise ValueError("No /data/demo_* episodes found.")

    first = h5["data"][demo_names[0]]
    if "obs/joint_pos" not in first:
        raise KeyError(f"{demo_names[0]} has no obs/joint_pos.")
    if action_source not in first:
        raise KeyError(f"{demo_names[0]} has no {action_source}.")

    state_shape = tuple(first["obs/joint_pos"].shape)
    action_shape = tuple(first[action_source].shape)

    if len(state_shape) != 2:
        raise ValueError(f"Expected joint_pos to be (T,D), got {state_shape}")
    if len(action_shape) != 2:
        raise ValueError(f"Expected actions to be (T,D), got {action_shape}")

    camera_specs = {}
    obs = first["obs"]
    for hdf5_key, lerobot_key in CAMERA_MAP.items():
        if hdf5_key not in obs:
            continue
        shape = tuple(obs[hdf5_key].shape)
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(
                f"{hdf5_key} must be (T,H,W,3), got {shape}"
            )
        _, h, w, c = shape
        camera_specs[hdf5_key] = {
            "lerobot_key": lerobot_key,
            "h": h,
            "w": w,
            "c": c,
        }

    return demo_names, state_shape[1], action_shape[1], camera_specs


def build_features(state_dim: int, action_dim: int, camera_specs: dict, action_source: str):
    if state_dim == len(STATE_NAMES):
        state_names = STATE_NAMES
    else:
        state_names = [f"state_{i}" for i in range(state_dim)]

    if action_source == "actions" and action_dim == len(RAW_ACTION_NAMES):
        action_names = RAW_ACTION_NAMES
    else:
        action_names = [f"action_{i}" for i in range(action_dim)]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names,
        },
    }

    # LeRobot canonical video feature shape is C,H,W.
    # add_frame() accepts H,W,C numpy arrays as well.
    for spec in camera_specs.values():
        features[spec["lerobot_key"]] = {
            "dtype": "video",
            "shape": (spec["c"], spec["h"], spec["w"]),
            "names": ["channels", "height", "width"],
        }

    return features


def convert(args):
    source = Path(args.hdf5).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {root}\n"
                "Use --overwrite if you really want to replace it."
            )
        print(f"[WARN] Removing existing output: {root}")
        shutil.rmtree(root)

    with h5py.File(source, "r") as h5:
        demo_names, state_dim, action_dim, camera_specs = inspect_hdf5(
            h5, args.action_source
        )

        if not camera_specs:
            raise ValueError("No supported RGB camera observations were found.")

        print("Source dataset")
        print(f"  episodes   : {len(demo_names)}")
        print(f"  state dim  : {state_dim}")
        print(f"  action dim : {action_dim}")
        print(f"  cameras    : {list(camera_specs)}")
        print(f"  fps        : {args.fps}")
        print(f"  task       : {args.task}")
        print()

        features = build_features(
            state_dim, action_dim, camera_specs, args.action_source
        )

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=features,
            root=root,
            robot_type="areumii_c1",
            use_videos=True,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=0,
            batch_encoding_size=1,
        )

        total_frames = 0

        for episode_idx, demo_name in enumerate(demo_names):
            demo = h5["data"][demo_name]

            state = np.asarray(
                demo["obs/joint_pos"][:], dtype=np.float32
            )
            action = np.asarray(
                demo[args.action_source][:], dtype=np.float32
            )

            camera_arrays = {
                hdf5_key: np.asarray(
                    demo[f"obs/{hdf5_key}"][:], dtype=np.uint8
                )
                for hdf5_key in camera_specs
            }

            original_T = state.shape[0]

            # Drop the first N frames from every synchronized modality.
            # At 20 Hz, --trim-start 2 removes the first 0.10 s.
            if args.trim_start < 0:
                raise ValueError("--trim-start must be >= 0.")
            if args.trim_start >= original_T:
                raise ValueError(
                    f"{demo_name}: --trim-start={args.trim_start} "
                    f"is not smaller than episode length {original_T}."
                )

            if args.trim_start > 0:
                state = state[args.trim_start:]
                action = action[args.trim_start:]
                camera_arrays = {
                    key: frames[args.trim_start:]
                    for key, frames in camera_arrays.items()
                }

            T = state.shape[0]
            lengths = {
                "state": state.shape[0],
                "action": action.shape[0],
                **{k: v.shape[0] for k, v in camera_arrays.items()},
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"{demo_name}: frame counts do not match: {lengths}"
                )

            print(
                f"[{episode_idx + 1}/{len(demo_names)}] "
                f"{demo_name}: {original_T} -> {T} frames "
                f"(trim_start={args.trim_start})"
            )

            for t in range(T):
                frame = {
                    "observation.state": np.ascontiguousarray(state[t]),
                    "action": np.ascontiguousarray(action[t]),
                    "task": args.task,
                }

                for hdf5_key, frames in camera_arrays.items():
                    lerobot_key = camera_specs[hdf5_key]["lerobot_key"]
                    frame[lerobot_key] = np.ascontiguousarray(frames[t])

                dataset.add_frame(frame)

            # Sequential encoding is slower but more robust for a small,
            # multi-camera conversion job.
            dataset.save_episode(parallel_encoding=False)
            total_frames += T

        print("\nFinalizing LeRobotDataset...")
        dataset.finalize()

    print("\nConversion complete")
    print(f"  local root   : {root}")
    print(f"  repo id      : {args.repo_id}")
    print(f"  episodes     : {len(demo_names)}")
    print(f"  total frames : {total_frames}")

    if args.push_to_hub:
        print("\nPushing to Hugging Face Hub...")
        dataset.push_to_hub()
        print(f"Uploaded: {args.repo_id}")
    else:
        print("\nHub upload skipped. Add --push-to-hub when ready.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hdf5",
        help="Isaac Lab HDF5 file, e.g. ./datasets/areumii_stack_v1.hdf5",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id, e.g. USER/areumii-stack-v1",
    )
    parser.add_argument(
        "--root",
        default="./lerobot_datasets/areumii_stack_v1",
        help="Local LeRobot output directory.",
    )
    parser.add_argument(
        "--task",
        default="Stack the blue, red, and green cubes in order.",
        help="Natural-language task instruction stored in every frame.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help=(
            "LeRobot dataset FPS. Default 20 because the current IsaacLab "
            "environment step-size is 0.05 s."
        ),
    )
    parser.add_argument(
        "--action-source",
        choices=["actions", "processed_actions"],
        default="actions",
        help="HDF5 action dataset to use. Default: raw env actions (16D).",
    )
    parser.add_argument(
        "--trim-start",
        type=int,
        default=3,
        help=(
            "Drop this many frames from the start of every episode before "
            "saving to LeRobot. Default: 2 frames (=0.10 s at 20 Hz). "
            "Use 0 to disable trimming."
        ),
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the finalized dataset to Hugging Face Hub.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing --root before conversion.",
    )
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()