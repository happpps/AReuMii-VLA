#!/usr/bin/env python3
import argparse
from pathlib import Path
import h5py
import numpy as np

OBS_JOINT_NAMES = [
    "L_shoulder_pitch","L_shoulder_roll","L_shoulder_yaw","L_elbow",
    "L_wrist_roll","L_wrist_yaw","L_wrist_pitch",
    "L_gripper_left","L_gripper_right",
    "R_shoulder_pitch","R_shoulder_roll","R_shoulder_yaw","R_elbow",
    "R_wrist_roll","R_wrist_yaw","R_wrist_pitch",
    "R_gripper_left","R_gripper_right",
]

PROCESSED_ACTION_NAMES = [
    "R_shoulder_pitch_cmd","R_shoulder_roll_cmd","R_shoulder_yaw_cmd","R_elbow_cmd",
    "R_wrist_roll_cmd","R_wrist_yaw_cmd","R_wrist_pitch_cmd",
    "R_gripper_left_cmd","R_gripper_right_cmd",
    "L_shoulder_pitch_cmd","L_shoulder_roll_cmd","L_shoulder_yaw_cmd","L_elbow_cmd",
    "L_wrist_roll_cmd","L_wrist_yaw_cmd","L_wrist_pitch_cmd",
    "L_gripper_left_cmd","L_gripper_right_cmd",
]

def save_csv(path, array, names):
    steps = np.arange(array.shape[0], dtype=np.int64)[:, None]
    data = np.concatenate([steps, array], axis=1)
    header = ",".join(["step"] + list(names))
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.7g")

def save_mp4(path, frames, fps):
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(frame)
        return "imageio"
    except Exception as imageio_error:
        try:
            import cv2
            h, w = frames.shape[1:3]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter could not open output file")
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return "opencv"
        except Exception as cv2_error:
            raise RuntimeError(
                "Could not write MP4.\n"
                f"imageio error: {imageio_error}\n"
                f"opencv error: {cv2_error}"
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--demo", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    # Save under: <hdf5 filename without extension>/demo_<n>
    out_dir = Path(args.out_dir) if args.out_dir else dataset.parent / dataset.stem / f"demo_{args.demo}"
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_key = f"data/demo_{args.demo}"
    with h5py.File(dataset, "r") as f:
        if demo_key not in f:
            raise KeyError(f"{demo_key} not found")
        demo = f[demo_key]
        obs = demo["obs"]

        camera_keys = [
            "front_camera",
            "left_wrist_camera",
            "right_wrist_camera",
        ]
        camera_frames = {
            key: obs[key][:]
            for key in camera_keys
            if key in obs
        }

        joint_pos = obs["joint_pos"][:]
        processed = demo["processed_actions"][:]
        raw_actions = demo["actions"][:]

    print(f"demo_{args.demo}")
    for key, frames in camera_frames.items():
        print(f"  {key:19s}: {frames.shape} {frames.dtype}")
    print(f"  joint_pos         : {joint_pos.shape} {joint_pos.dtype}")
    print(f"  raw actions       : {raw_actions.shape} {raw_actions.dtype}")
    print(f"  processed_actions : {processed.shape} {processed.dtype}")

    joint_csv = out_dir / f"demo_{args.demo}_joint_pos.csv"
    action_csv = out_dir / f"demo_{args.demo}_processed_actions.csv"
    raw_csv = out_dir / f"demo_{args.demo}_raw_actions.csv"

    video_results = []
    for key, frames in camera_frames.items():
        mp4_path = out_dir / f"demo_{args.demo}_{key}.mp4"
        backend = save_mp4(mp4_path, frames, args.fps)
        video_results.append((key, mp4_path, backend))
    save_csv(joint_csv, joint_pos, OBS_JOINT_NAMES)
    save_csv(action_csv, processed, PROCESSED_ACTION_NAMES)
    save_csv(raw_csv, raw_actions, [f"action_{i}" for i in range(raw_actions.shape[1])])

    print("\nJoint position range:")
    for name, lo, hi in zip(OBS_JOINT_NAMES, joint_pos.min(axis=0), joint_pos.max(axis=0)):
        print(f"  {name:24s} {lo: .4f} .. {hi: .4f}")

    print("\nProcessed action range:")
    for name, lo, hi in zip(PROCESSED_ACTION_NAMES, processed.min(axis=0), processed.max(axis=0)):
        print(f"  {name:24s} {lo: .4f} .. {hi: .4f}")

    print("\nCamera check:")
    if not camera_frames:
        print("  [WARN] No camera observations found.")
    else:
        for key, frames in camera_frames.items():
            if len(frames) > 1:
                frame_delta = np.mean(
                    np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
                )
                repeated = np.mean(
                    np.all(frames[1:] == frames[:-1], axis=(1, 2, 3))
                )
                print(f"  {key}:")
                print(f"    mean adjacent-frame abs difference : {frame_delta:.3f}")
                print(f"    exactly repeated adjacent frames   : {repeated * 100:.1f}%")
            else:
                print(f"  {key}: only {len(frames)} frame(s)")

    print("\nSaved:")
    for key, mp4_path, backend in video_results:
        print(f"  {mp4_path} ({backend})")
    print(f"  {joint_csv}")
    print(f"  {action_csv}")
    print(f"  {raw_csv}")
    print("\nIMPORTANT:")
    print("  obs/joint_pos order     = LEFT -> RIGHT")
    print("  processed_actions order = RIGHT -> LEFT")
    print("  Reorder before pairing columns for training.")

if __name__ == "__main__":
    main()