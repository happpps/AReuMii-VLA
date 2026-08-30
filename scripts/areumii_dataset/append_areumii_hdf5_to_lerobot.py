#!/usr/bin/env python3
"""
Append selected Isaac Lab Areumii HDF5 episodes to an EXISTING
LeRobotDataset v3 repository on Hugging Face Hub.

Intended use case
-----------------
Existing Hub dataset ends at episode_index 160, and the HDF5 contains
/data/demo_161 ... /data/demo_200. This script appends only those demos.

Important
---------
- It does NOT recreate the existing dataset.
- LeRobotDataset.resume() reads the existing metadata and continues episode,
  parquet, and video indices from the existing dataset.
- A clean local work directory is used for the append session.
- By default it checks that the Hub dataset currently ends at episode_index 160.

Example
-------
python append_areumii_hdf5_to_lerobot.py \
    ./datasets/areumii_pickplace_v4.hdf5 \
    --repo-id 1ys1/areumii_pickplace-v4 \
    --task "Pick red cube and place it on blue basket" \
    --demo-start 161 \
    --demo-end 200 \
    --push-to-hub
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets import LeRobotDataset


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


def demo_index(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid demo name: {name}") from exc


def prepare_work_root(root: Path, overwrite_root: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite_root:
            raise FileExistsError(
                f"Local work root is not empty: {root}\n"
                "Use a new --root, or pass --overwrite-root to delete ONLY the "
                "local append work directory before starting."
            )
        print(f"[WARN] Removing local append work directory: {root}")
        shutil.rmtree(root)

    root.mkdir(parents=True, exist_ok=True)


def select_demos(h5: h5py.File, start: int, end: int) -> list[str]:
    if "data" not in h5:
        raise KeyError("HDF5 has no /data group.")

    all_demos = sorted(
        [name for name in h5["data"].keys() if name.startswith("demo_")],
        key=demo_sort_key,
    )
    if not all_demos:
        raise ValueError("No /data/demo_* episodes found.")

    selected = [
        name for name in all_demos
        if start <= demo_index(name) <= end
    ]

    expected = end - start + 1
    if len(selected) != expected:
        available_indices = [demo_index(name) for name in all_demos]
        raise ValueError(
            f"Requested demo_{start} ... demo_{end} ({expected} demos), "
            f"but found only {len(selected)}.\n"
            f"Available demo index range: {min(available_indices)} ... "
            f"{max(available_indices)}\n"
            "If this HDF5 contains only the NEW 40 episodes and they are named "
            "demo_0 ... demo_39, run with --demo-start 0 --demo-end 39. "
            "They will still be appended to the Hub dataset's next episode indices."
        )

    return selected


def inspect_selected_hdf5(
    h5: h5py.File,
    demo_names: list[str],
    action_source: str,
) -> tuple[int, int, dict[str, tuple[int, int, int]]]:
    first_name = demo_names[0]
    first = h5["data"][first_name]

    if "obs/joint_pos" not in first:
        raise KeyError(f"{first_name} has no obs/joint_pos.")
    if action_source not in first:
        raise KeyError(f"{first_name} has no {action_source}.")

    state_shape = tuple(first["obs/joint_pos"].shape)
    action_shape = tuple(first[action_source].shape)

    if len(state_shape) != 2:
        raise ValueError(f"Expected joint_pos=(T,D), got {state_shape}")
    if len(action_shape) != 2:
        raise ValueError(f"Expected {action_source}=(T,D), got {action_shape}")

    camera_shapes: dict[str, tuple[int, int, int]] = {}
    obs = first["obs"]
    for hdf5_key in CAMERA_MAP:
        if hdf5_key not in obs:
            continue
        shape = tuple(obs[hdf5_key].shape)
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f"{hdf5_key} must be (T,H,W,3), got {shape}")
        _, h, w, c = shape
        camera_shapes[hdf5_key] = (c, h, w)

    if not camera_shapes:
        raise ValueError("No supported RGB camera observations were found.")

    # Check that every selected demo has the same modalities/dimensions.
    for name in demo_names:
        demo = h5["data"][name]
        if "obs/joint_pos" not in demo or action_source not in demo:
            raise KeyError(f"{name} is missing state or action data.")

        if demo["obs/joint_pos"].shape[1] != state_shape[1]:
            raise ValueError(f"{name}: state dimension differs from {first_name}.")
        if demo[action_source].shape[1] != action_shape[1]:
            raise ValueError(f"{name}: action dimension differs from {first_name}.")

        for hdf5_key, chw in camera_shapes.items():
            if f"obs/{hdf5_key}" not in demo:
                raise KeyError(f"{name} has no obs/{hdf5_key}.")
            shape = tuple(demo[f"obs/{hdf5_key}"].shape)
            current_chw = (shape[3], shape[1], shape[2])
            if current_chw != chw:
                raise ValueError(
                    f"{name}: {hdf5_key} shape differs. "
                    f"Expected C,H,W={chw}, got {current_chw}."
                )

    return state_shape[1], action_shape[1], camera_shapes


def validate_against_existing_dataset(
    dataset: LeRobotDataset,
    state_dim: int,
    action_dim: int,
    camera_shapes: dict[str, tuple[int, int, int]],
) -> None:
    features = dataset.meta.features

    required = ["observation.state", "action"]
    required += [CAMERA_MAP[key] for key in camera_shapes]
    missing = [key for key in required if key not in features]
    if missing:
        raise ValueError(
            "Existing Hub dataset is missing features required by the new HDF5: "
            + ", ".join(missing)
        )

    existing_state_shape = tuple(features["observation.state"]["shape"])
    existing_action_shape = tuple(features["action"]["shape"])

    if existing_state_shape != (state_dim,):
        raise ValueError(
            f"State shape mismatch: Hub={existing_state_shape}, HDF5={(state_dim,)}"
        )
    if existing_action_shape != (action_dim,):
        raise ValueError(
            f"Action shape mismatch: Hub={existing_action_shape}, HDF5={(action_dim,)}"
        )

    for hdf5_key, expected_chw in camera_shapes.items():
        lerobot_key = CAMERA_MAP[hdf5_key]
        existing_shape = tuple(features[lerobot_key]["shape"])
        if existing_shape != expected_chw:
            raise ValueError(
                f"Camera shape mismatch for {lerobot_key}: "
                f"Hub={existing_shape}, HDF5={expected_chw}"
            )


def append(args: argparse.Namespace) -> None:
    source = Path(args.hdf5).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(source)
    if args.demo_start < 0 or args.demo_end < args.demo_start:
        raise ValueError("Invalid --demo-start/--demo-end range.")
    if args.trim_start < 0:
        raise ValueError("--trim-start must be >= 0.")

    prepare_work_root(root, args.overwrite_root)

    with h5py.File(source, "r") as h5:
        demo_names = select_demos(h5, args.demo_start, args.demo_end)
        state_dim, action_dim, camera_shapes = inspect_selected_hdf5(
            h5, demo_names, args.action_source
        )

        print("New HDF5 selection")
        print(f"  source       : {source}")
        print(f"  demos        : {demo_names[0]} ... {demo_names[-1]}")
        print(f"  count        : {len(demo_names)}")
        print(f"  state dim    : {state_dim}")
        print(f"  action dim   : {action_dim}")
        print(f"  cameras      : {list(camera_shapes)}")
        print()

        print("Loading existing Hub metadata and opening append writer...")
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=root,
            force_cache_sync=True,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=0,
            batch_encoding_size=1,
        )

        existing_count = dataset.meta.total_episodes
        existing_last_index = existing_count - 1
        next_episode_index = existing_count

        print("Existing LeRobot dataset")
        print(f"  repo id             : {args.repo_id}")
        print(f"  existing episodes   : {existing_count}")
        print(f"  last episode_index  : {existing_last_index}")
        print(f"  next episode_index  : {next_episode_index}")
        print(f"  fps                  : {dataset.meta.fps}")
        print()

        if args.expected_last_index is not None:
            if existing_last_index != args.expected_last_index:
                raise RuntimeError(
                    "Safety check failed.\n"
                    f"Expected existing last episode_index="
                    f"{args.expected_last_index}, but Hub metadata says "
                    f"{existing_last_index}.\n"
                    "Nothing has been appended yet. Check the repo and rerun with "
                    "the correct --expected-last-index."
                )

        if next_episode_index != 161 and args.expected_last_index == 160:
            raise RuntimeError(
                f"Expected the new append to start at episode_index 161, "
                f"but it would start at {next_episode_index}. Aborting before writing."
            )

        validate_against_existing_dataset(
            dataset, state_dim, action_dim, camera_shapes
        )

        print(
            f"Appending {len(demo_names)} HDF5 demos as LeRobot episode_index "
            f"{next_episode_index} ... {next_episode_index + len(demo_names) - 1}"
        )
        print()

        total_frames = 0

        for i, demo_name in enumerate(demo_names):
            demo = h5["data"][demo_name]

            state = np.asarray(demo["obs/joint_pos"][:], dtype=np.float32)
            action = np.asarray(demo[args.action_source][:], dtype=np.float32)
            camera_arrays = {
                hdf5_key: np.asarray(
                    demo[f"obs/{hdf5_key}"][:], dtype=np.uint8
                )
                for hdf5_key in camera_shapes
            }

            original_T = state.shape[0]
            if args.trim_start >= original_T:
                raise ValueError(
                    f"{demo_name}: --trim-start={args.trim_start} is not smaller "
                    f"than episode length {original_T}."
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
                **{key: frames.shape[0] for key, frames in camera_arrays.items()},
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"{demo_name}: synchronized frame counts do not match: {lengths}"
                )

            target_episode_index = next_episode_index + i
            print(
                f"[{i + 1:02d}/{len(demo_names)}] {demo_name} -> "
                f"episode_index {target_episode_index}: "
                f"{original_T} -> {T} frames"
            )

            for t in range(T):
                frame = {
                    "observation.state": np.ascontiguousarray(state[t]),
                    "action": np.ascontiguousarray(action[t]),
                    "task": args.task,
                }

                for hdf5_key, frames in camera_arrays.items():
                    frame[CAMERA_MAP[hdf5_key]] = np.ascontiguousarray(frames[t])

                dataset.add_frame(frame)

            dataset.save_episode(parallel_encoding=False)
            total_frames += T

        print("\nFinalizing appended dataset metadata/files...")
        dataset.finalize()

    final_last = next_episode_index + len(demo_names) - 1
    print("\nAppend complete locally")
    print(f"  work root          : {root}")
    print(f"  added episodes     : {len(demo_names)}")
    print(f"  added index range  : {next_episode_index} ... {final_last}")
    print(f"  added frames       : {total_frames}")

    if args.push_to_hub:
        print("\nPushing appended files + updated metadata to Hugging Face Hub...")
        dataset.push_to_hub()
        print(f"Uploaded: {args.repo_id}")
    else:
        print("\nHub upload skipped. Re-run with --push-to-hub when ready.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append selected Areumii HDF5 demos to an existing LeRobot Hub dataset."
    )
    parser.add_argument(
        "hdf5",
        help="HDF5 containing the new Isaac Lab demos.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="EXISTING Hugging Face dataset repo, e.g. 1ys1/areumii_pickplace-v4",
    )
    parser.add_argument(
        "--root",
        default="./lerobot_datasets/append_work",
        help="Clean local work directory used by LeRobotDataset.resume().",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task instruction to store for the appended frames. Prefer exactly the same wording as the existing dataset.",
    )
    parser.add_argument(
        "--demo-start",
        type=int,
        default=161,
        help="First HDF5 demo number to append. Default: 161",
    )
    parser.add_argument(
        "--demo-end",
        type=int,
        default=200,
        help="Last HDF5 demo number to append, inclusive. Default: 200",
    )
    parser.add_argument(
        "--expected-last-index",
        type=int,
        default=160,
        help=(
            "Safety check for the existing Hub dataset. Default: 160, so the "
            "new first episode_index must be 161."
        ),
    )
    parser.add_argument(
        "--action-source",
        choices=["actions", "processed_actions"],
        default="actions",
        help="HDF5 action dataset to use. Default: actions",
    )
    parser.add_argument(
        "--trim-start",
        type=int,
        default=5,
        help="Drop this many synchronized frames from each new demo. Default: 5.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the appended dataset files/metadata to the existing Hub repo.",
    )
    parser.add_argument(
        "--overwrite-root",
        action="store_true",
        help="Delete the LOCAL --root work directory before starting. Does not delete the Hub repo.",
    )

    args = parser.parse_args()
    append(args)


if __name__ == "__main__":
    main()
