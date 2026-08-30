#!/usr/bin/env python3
"""
Visual safety test for Areumii bimanual real-teleop + gripper commands.

Run in env_isaaclab. This process does NOT read Quest/OpenXR.

Command protocol on UDP 5010 (matches areumii_bimanual_teleop_gripper.py):
    L,q0,...,q6
    R,q0,...,q6
    GL,<real_gripper_motor_rad>
    GR,<real_gripper_motor_rad>

Feedback on UDP 5011 (32 CSV values, matching the real bridge):
    measured LEFT7, RIGHT7,
    control  LEFT7, RIGHT7,
    measured gripper LEFT, RIGHT,
    control  gripper LEFT, RIGHT

Real gripper motor endpoints:
    LEFT  open=-1.24 rad, close=0.92 rad
    RIGHT open=-1.22 rad, close=0.82 rad

For visualization only, real motor angle is linearly mapped to the IsaacLab
prismatic fingers:
    open  -> [+0.03, -0.03] m
    close -> [ 0.00,  0.00] m
This adapter does not assume the physical motor-to-linear mechanism is linear.
"""

from __future__ import annotations

import argparse
import socket
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Apply external UDP bimanual arm + gripper commands to Areumii IsaacLab."
)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--cmd-port", type=int, default=5010)
parser.add_argument("--feedback-port", type=int, default=5011)
parser.add_argument("--feedback-hz", type=float, default=100.0)
parser.add_argument("--first-command-max-delta", type=float, default=0.20)
parser.add_argument("--command-max-step", type=float, default=0.25)
parser.add_argument("--watchdog-s", type=float, default=0.30)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli, enable_cameras=True)
simulation_app = app_launcher.app

# Isaac/third-party imports must come after AppLauncher.
import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _ensure_replicator_loaded() -> None:
    import omni.kit.app
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("omni.replicator.core", True)


ARM_JOINT_NAMES = {
    "left": [
        "shoulder_pitch_joint_L",
        "shoulder_roll_joint_L",
        "shouler_yaw_joint_L",
        "elbow_joint_L",
        "wrist_roll_joint_L",
        "wrist_yaw_joint_L",
        "wrist_pitch_joint_L",
    ],
    "right": [
        "shoulder_pitch_joint_R",
        "shoulder_roll_joint_R",
        "shoulder_yaw_joint_R",
        "elbow_joint_R",
        "wrist_roll_joint_R",
        "wrist_yaw_joint_R",
        "wrist_pitch_joint_R",
    ],
}

GRIPPER_JOINT_NAMES = {
    "left": [
        "gripper_left_finger_joint_L",
        "gripper_right_finger_joint_L",
    ],
    "right": [
        "gripper_left_finger_joint_R",
        "gripper_right_finger_joint_R",
    ],
}

TARGET_FRAME_NAMES = {
    "left": "gripper_hand_link_L_1",
    "right": "gripper_hand_link_R_1",
}

GRIPPER_OPEN_RAD = {"left": -1.24, "right": -1.22}
GRIPPER_CLOSE_RAD = {"left": 0.92, "right": 0.82}

SIM_GRIPPER_OPEN = np.array([0.03, -0.03], dtype=np.float64)
SIM_GRIPPER_CLOSE = np.array([0.0, 0.0], dtype=np.float64)


def encode_csv(values: np.ndarray) -> bytes:
    return ",".join(f"{float(v):.12g}" for v in values).encode("ascii")


def parse_packet(data: bytes):
    """Return ('arm', side, q7) or ('gripper', side, motor_rad), else None."""
    try:
        parts = data.decode("ascii").strip().split(",")
    except UnicodeDecodeError:
        return None

    if not parts:
        return None

    tag = parts[0]

    if tag in ("L", "R"):
        if len(parts) != 8:
            return None
        try:
            q = np.asarray([float(v) for v in parts[1:]], dtype=np.float64)
        except ValueError:
            return None
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            return None
        return "arm", ("left" if tag == "L" else "right"), q

    if tag in ("GL", "GR"):
        if len(parts) != 2:
            return None
        try:
            q = float(parts[1])
        except ValueError:
            return None
        if not np.isfinite(q):
            return None
        return "gripper", ("left" if tag == "GL" else "right"), q

    # Pre-IK target debug packet from the teleop process:
    # TL/TR,raw_x,raw_y,raw_z,cmd_x,cmd_y,cmd_z
    if tag in ("TL", "TR"):
        if len(parts) != 7:
            return None
        try:
            p = np.asarray([float(v) for v in parts[1:]], dtype=np.float64)
        except ValueError:
            return None
        if p.shape != (6,) or not np.all(np.isfinite(p)):
            return None
        side = "left" if tag == "TL" else "right"
        return "target", side, (p[:3].copy(), p[3:].copy())

    return None


def motor_rad_to_sim_fingers(side: str, motor_rad: float) -> np.ndarray:
    q_open = GRIPPER_OPEN_RAD[side]
    q_close = GRIPPER_CLOSE_RAD[side]
    alpha = (motor_rad - q_open) / (q_close - q_open)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * SIM_GRIPPER_OPEN + alpha * SIM_GRIPPER_CLOSE


def sim_fingers_to_motor_rad(side: str, fingers: np.ndarray) -> float:
    fingers = np.asarray(fingers, dtype=np.float64)
    opening = float(np.clip(fingers[0], 0.0, 0.03))
    alpha = 1.0 - opening / 0.03
    return GRIPPER_OPEN_RAD[side] + alpha * (
        GRIPPER_CLOSE_RAD[side] - GRIPPER_OPEN_RAD[side]
    )


def get_joint_q(robot, joint_ids) -> np.ndarray:
    return (
        robot.data.joint_pos[0, joint_ids]
        .detach().cpu().numpy().astype(np.float64)
    )


def set_joint_target(robot, env, joint_ids, q: np.ndarray) -> None:
    target = torch.as_tensor(q, dtype=torch.float32, device=env.device).unsqueeze(0)
    robot.set_joint_position_target(target, joint_ids=joint_ids)


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q=(w,x,y,z), NumPy only."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[0]
    xyz = q[1:]
    return (
        (w * w - np.dot(xyz, xyz)) * v
        + 2.0 * xyz * np.dot(xyz, v)
        + 2.0 * w * np.cross(xyz, v)
    )


def base_point_to_world(robot, p_base: np.ndarray) -> np.ndarray:
    """Convert URDF/articulation-root-frame target position to world coordinates."""
    try:
        root_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
        root_quat = robot.data.root_link_quat_w[0].detach().cpu().numpy()
    except AttributeError:
        # Compatibility with older IsaacLab versions.
        root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
        root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()

    return root_pos + quat_rotate_wxyz(root_quat, p_base)


def make_ee_debug_markers() -> VisualizationMarkers:
    """Create raw-target, IK-target, and actual-EE sphere prototypes."""
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/AreumiiEETargetDebug",
        markers={
            # Clutch.rebase() target before clamp_ee_step().
            "raw_target": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.25, 0.05),
                ),
            ),
            # Position actually passed into IK after clamp_ee_step().
            "ik_target": sim_utils.SphereCfg(
                radius=0.018,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 1.0, 0.1),
                ),
            ),
            # Actual simulated EE link position.
            "actual_ee": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 0.35, 1.0),
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def main() -> None:
    print("[SIM BRIDGE] Loading task:", args_cli.task)
    _ensure_replicator_loaded()

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    arm_ids = {}
    gripper_ids = {}

    for side in ("left", "right"):
        ids, names = robot.find_joints(ARM_JOINT_NAMES[side], preserve_order=True)
        if len(ids) != 7:
            raise RuntimeError(f"Expected 7 {side} arm joints, got {len(ids)}: {names}")
        arm_ids[side] = ids

        gids, gnames = robot.find_joints(GRIPPER_JOINT_NAMES[side], preserve_order=True)
        if len(gids) != 2:
            raise RuntimeError(f"Expected 2 {side} gripper joints, got {len(gids)}: {gnames}")
        gripper_ids[side] = gids

        print(f"[SIM BRIDGE] {side} arm ids:", ids)
        print(f"[SIM BRIDGE] {side} arm names:", names)
        print(f"[SIM BRIDGE] {side} gripper ids:", gids)
        print(f"[SIM BRIDGE] {side} gripper names:", gnames)

    ee_body_ids = {}
    for side in ("left", "right"):
        body_ids, body_names = robot.find_bodies(
            [TARGET_FRAME_NAMES[side]],
            preserve_order=True,
        )
        if len(body_ids) != 1:
            raise RuntimeError(
                f"Expected EE body {TARGET_FRAME_NAMES[side]!r}, "
                f"got ids={body_ids}, names={body_names}"
            )
        ee_body_ids[side] = body_ids[0]
        print(f"[SIM BRIDGE] {side} EE marker body:", body_names)

    ee_markers = make_ee_debug_markers()

    # Latest target coordinates are in the URDF/base frame because that is the
    # frame used by RobotKinematics in the teleop process.
    target_raw_base = {"left": None, "right": None}
    target_cmd_base = {"left": None, "right": None}

    # Emulate the real bridge's current control targets.
    q_hold = {
        side: get_joint_q(robot, arm_ids[side]).copy()
        for side in ("left", "right")
    }

    gripper_motor_hold = {}
    for side in ("left", "right"):
        gripper_motor_hold[side] = sim_fingers_to_motor_rad(
            side, get_joint_q(robot, gripper_ids[side])
        )

    last_accepted_q = {"left": None, "right": None}
    last_command_time = {"left": None, "right": None}
    watchdog_reported = {"left": False, "right": False}
    last_gripper_print = {"left": 0.0, "right": 0.0}

    print(
        "[SIM BRIDGE] reset arm q:"
        f" L={np.round(q_hold['left'], 4)}"
        f" R={np.round(q_hold['right'], 4)}"
    )
    print(
        "[SIM BRIDGE] reset gripper equivalent motor rad:"
        f" L={gripper_motor_hold['left']:.4f}"
        f" R={gripper_motor_hold['right']:.4f}"
    )
    print(f"[SIM BRIDGE] command  : UDP 127.0.0.1:{args_cli.cmd_port}")
    print(f"[SIM BRIDGE] feedback : UDP 127.0.0.1:{args_cli.feedback_port}")
    print("[SIM BRIDGE] arm format    : L,q0..q6 / R,q0..q6")
    print("[SIM BRIDGE] gripper format: GL,<rad> / GR,<rad>")
    print("[SIM BRIDGE] target debug  : TL/TR,raw_xyz,cmd_xyz")
    print("[SIM MARKER] orange=raw target, green=IK target, blue=actual EE")
    print("[SIM BRIDGE] feedback      : 32 CSV values")

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind(("127.0.0.1", args_cli.cmd_port))
    rx.setblocking(False)

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    feedback_addr = ("127.0.0.1", args_cli.feedback_port)

    physics_dt = float(env.physics_dt)
    feedback_period = 1.0 / max(args_cli.feedback_hz, 1.0)
    next_feedback_time = time.monotonic()

    try:
        while True:
            wall_start = time.perf_counter()

            newest_arm = {"left": None, "right": None}
            newest_gripper = {"left": None, "right": None}
            newest_target = {"left": None, "right": None}

            while True:
                try:
                    data, _ = rx.recvfrom(4096)
                except BlockingIOError:
                    break

                parsed = parse_packet(data)
                if parsed is None:
                    print("[SIM BRIDGE] malformed UDP packet dropped:", repr(data[:120]))
                    continue

                kind, side, value = parsed
                if kind == "arm":
                    newest_arm[side] = value
                elif kind == "gripper":
                    newest_gripper[side] = value
                else:
                    newest_target[side] = value

            # Target marker data is accepted independently of IK/arm commands.
            # This is the key diagnostic: if IK fails, the green/orange markers
            # can continue moving while the actual blue EE/robot stops.
            for side in ("left", "right"):
                if newest_target[side] is not None:
                    raw_pos, cmd_pos = newest_target[side]
                    target_raw_base[side] = raw_pos.copy()
                    target_cmd_base[side] = cmd_pos.copy()

            for side in ("left", "right"):
                q_new = newest_arm[side]
                if q_new is None:
                    continue

                current_q = get_joint_q(robot, arm_ids[side])
                last_t = last_command_time[side]
                stream_was_idle = (
                    last_t is None
                    or time.monotonic() - last_t > args_cli.watchdog_s
                )

                if last_accepted_q[side] is None or stream_was_idle:
                    reference = current_q
                    max_delta = args_cli.first_command_max_delta
                else:
                    reference = last_accepted_q[side]
                    max_delta = args_cli.command_max_step

                delta = np.abs(q_new - reference)
                if np.any(delta > max_delta):
                    bad = np.where(delta > max_delta)[0].tolist()
                    print(
                        f"[SIM SAFETY][{side.upper()}] arm command dropped: jump too large",
                        f"joints={bad}",
                        "target=", np.round(q_new, 4),
                        "reference=", np.round(reference, 4),
                    )
                    continue

                limits_ok = True
                try:
                    limits = (
                        robot.data.soft_joint_pos_limits[0, arm_ids[side]]
                        .detach().cpu().numpy()
                    )
                    lo = limits[:, 0]
                    hi = limits[:, 1]
                    if np.any(q_new < lo) or np.any(q_new > hi):
                        print(
                            f"[SIM SAFETY][{side.upper()}] arm command outside "
                            "IsaacLab soft joint limits; dropped:",
                            np.round(q_new, 4),
                        )
                        limits_ok = False
                except Exception:
                    pass

                if limits_ok:
                    q_hold[side] = q_new.copy()
                    last_accepted_q[side] = q_new.copy()
                    last_command_time[side] = time.monotonic()
                    watchdog_reported[side] = False

            # Gripper commands are real motor angles. Reject anything outside
            # the measured physical endpoint range before mapping to sim fingers.
            for side in ("left", "right"):
                g_new = newest_gripper[side]
                if g_new is None:
                    continue

                q_min = GRIPPER_OPEN_RAD[side]
                q_max = GRIPPER_CLOSE_RAD[side]
                if g_new < q_min or g_new > q_max:
                    print(
                        f"[SIM SAFETY][{side.upper()} GRIP] command outside measured "
                        f"motor range: {g_new:.4f} not in [{q_min:.4f}, {q_max:.4f}]"
                    )
                    continue

                gripper_motor_hold[side] = float(g_new)

                now_print = time.monotonic()
                if now_print - last_gripper_print[side] >= 0.25:
                    print(
                        f"[SIM {side.upper()} GRIP] motor={g_new:.3f} rad -> "
                        f"fingers={np.round(motor_rad_to_sim_fingers(side, g_new), 4)} m"
                    )
                    last_gripper_print[side] = now_print

            # Direct articulation control: do not call env.step(), because the
            # task ActionManager could overwrite these targets.
            for side in ("left", "right"):
                set_joint_target(robot, env, arm_ids[side], q_hold[side])
                set_joint_target(
                    robot,
                    env,
                    gripper_ids[side],
                    motor_rad_to_sim_fingers(side, gripper_motor_hold[side]),
                )

            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(physics_dt)

            # --------------------------------------------------------------
            # EE TARGET MARKERS
            # orange: raw Clutch target before EE step limiting
            # green : target position actually passed to IK
            # blue  : actual simulated EE link position
            #
            # Marker order is L(raw,cmd,actual), R(raw,cmd,actual).
            # If no target packet has arrived for one side yet, place its raw/cmd
            # markers on the actual EE so startup does not show stray markers.
            # --------------------------------------------------------------
            marker_positions = []
            marker_indices = []
            for side in ("left", "right"):
                actual_w = (
                    robot.data.body_pos_w[0, ee_body_ids[side]]
                    .detach().cpu().numpy().astype(np.float64)
                )

                if target_raw_base[side] is None:
                    raw_w = actual_w
                    cmd_w = actual_w
                else:
                    raw_w = base_point_to_world(robot, target_raw_base[side])
                    cmd_w = base_point_to_world(robot, target_cmd_base[side])

                marker_positions.extend([raw_w, cmd_w, actual_w])
                marker_indices.extend([0, 1, 2])

            ee_markers.visualize(
                translations=np.asarray(marker_positions, dtype=np.float32),
                marker_indices=np.asarray(marker_indices, dtype=np.int32),
            )

            now = time.monotonic()
            if now >= next_feedback_time:
                measured_left = get_joint_q(robot, arm_ids["left"])
                measured_right = get_joint_q(robot, arm_ids["right"])

                measured_grip_left = sim_fingers_to_motor_rad(
                    "left", get_joint_q(robot, gripper_ids["left"])
                )
                measured_grip_right = sim_fingers_to_motor_rad(
                    "right", get_joint_q(robot, gripper_ids["right"])
                )

                feedback = np.concatenate([
                    measured_left,
                    measured_right,
                    q_hold["left"],
                    q_hold["right"],
                    np.asarray([
                        measured_grip_left,
                        measured_grip_right,
                        gripper_motor_hold["left"],
                        gripper_motor_hold["right"],
                    ], dtype=np.float64),
                ])

                if feedback.shape != (32,):
                    raise RuntimeError(f"Internal feedback layout error: {feedback.shape}")

                tx.sendto(encode_csv(feedback), feedback_addr)
                next_feedback_time = now + feedback_period

            for side in ("left", "right"):
                last_t = last_command_time[side]
                if (
                    last_t is not None
                    and not watchdog_reported[side]
                    and now - last_t > args_cli.watchdog_s
                ):
                    print(
                        f"[SIM BRIDGE][{side.upper()}] arm command stream idle "
                        "-> holding last target"
                    )
                    watchdog_reported[side] = True

            elapsed = time.perf_counter() - wall_start
            if elapsed < physics_dt:
                time.sleep(physics_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[SIM BRIDGE] Ctrl-C")
    finally:
        rx.close()
        tx.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
