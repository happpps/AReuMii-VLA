#!/usr/bin/env python3
"""
Areumii right-arm XR teleoperation shared by simulation and the real robot.

Common path:
    Quest / CloudXR
      -> Isaac Teleop XRController
      -> Squeeze: engage-relative Clutch
      -> Areumii URDF IK (Placo via LeRobot RobotKinematics)
      -> q_target [rad]
      -> localhost UDP

Backend:
    --mode sim  : UDP 5010/5011 -> IsaacLab test_real_teleop_in_sim.py
    --mode real : UDP 5005/5006 -> areumii_udp_shm_bridge -> Areum2_can -> CAN

READY/DOWN motion is intentionally NOT handled here. On the real robot use
areumii_move_pose before/after teleoperation.

This intentionally controls ONLY the right 7-DOF arm. Gripper is not implemented.
The script is dry-run unless --enable-command is supplied.
"""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from lerobot.model.kinematics import RobotKinematics

from isaac_teleop.clutch import Clutch
from isaac_teleop.config_isaac_teleop import XRControllerConfig
from isaac_teleop.teleop_xr_controller import XRController

RIGHT_JOINT_NAMES = [
    "shoulder_pitch_joint_R",
    "shoulder_roll_joint_R",
    "shoulder_yaw_joint_R",
    "elbow_joint_R",
    "wrist_roll_joint_R",
    "wrist_yaw_joint_R",
    "wrist_pitch_joint_R",
]

# From the same Areumii URDF / IsaacLab model used in simulation.
# Order MUST match RIGHT_JOINT_NAMES.
RIGHT_JOINT_LOWER = np.array(
    [-3.142, -0.262, -1.396, -0.873, -1.745, -0.524, -1.571],
    dtype=float,
)
RIGHT_JOINT_UPPER = np.array(
    [ 3.142,  2.967,  1.396,  2.094,  1.745,  0.524,  1.571],
    dtype=float,
)
JOINT_LIMIT_MARGIN_RAD = 0.01

MODE_PORTS = {
    "sim": (5010, 5011),
    "real": (5005, 5006),
}


class RadianKinematics:
    """Wrap LeRobot RobotKinematics, whose public API is degree-based, as radians."""

    def __init__(self, urdf_path: str, target_frame_name: str, joint_names: list[str]):
        self._kin = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame_name,
            joint_names=joint_names,
        )

    def forward_kinematics(self, q_rad: np.ndarray) -> np.ndarray:
        return self._kin.forward_kinematics(np.rad2deg(np.asarray(q_rad, dtype=float)))

    def inverse_kinematics(self, q_seed_rad: np.ndarray, desired_T: np.ndarray, orientation_weight: float) -> np.ndarray:
        q_deg = self._kin.inverse_kinematics(
            np.rad2deg(np.asarray(q_seed_rad, dtype=float)),
            desired_T,
            orientation_weight=orientation_weight,
        )
        return np.deg2rad(np.asarray(q_deg, dtype=float))


class UdpJointBridge:
    """Receive actual right-arm q from C++ bridge and send desired q_target."""

    def __init__(self, cmd_port: int = 5005, feedback_port: int = 5006):
        self.cmd_addr = ("127.0.0.1", cmd_port)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("127.0.0.1", feedback_port))
        self.rx.setblocking(False)
        self.latest_q: np.ndarray | None = None
        self.latest_feedback_time = 0.0

    @staticmethod
    def _decode_q(data: bytes) -> np.ndarray | None:
        try:
            values = [float(x) for x in data.decode("ascii").strip().split(",")]
        except (UnicodeDecodeError, ValueError):
            return None
        if len(values) != 7:
            return None
        q = np.asarray(values, dtype=float)
        return q if np.all(np.isfinite(q)) else None

    def poll_feedback(self) -> np.ndarray | None:
        while True:
            try:
                data, _ = self.rx.recvfrom(4096)
            except BlockingIOError:
                break
            q = self._decode_q(data)
            if q is not None:
                self.latest_q = q
                self.latest_feedback_time = time.monotonic()
        return None if self.latest_q is None else self.latest_q.copy()

    def wait_for_feedback(self, timeout_s: float) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            q = self.poll_feedback()
            if q is not None:
                return q
            time.sleep(0.01)
        raise TimeoutError(f"No joint feedback on UDP 127.0.0.1:{self.rx.getsockname()[1]} within {timeout_s:.1f}s")

    def feedback_is_fresh(self, max_age_s: float) -> bool:
        self.poll_feedback()
        return self.latest_q is not None and (time.monotonic() - self.latest_feedback_time) <= max_age_s

    def send_q(self, q_rad: np.ndarray) -> None:
        msg = ",".join(f"{v:.12g}" for v in np.asarray(q_rad, dtype=float)).encode("ascii")
        self.tx.sendto(msg, self.cmd_addr)


def clamp_vector_step(current: np.ndarray, target: np.ndarray, max_abs_step: float) -> np.ndarray:
    delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    return np.asarray(current, dtype=float) + np.clip(delta, -max_abs_step, max_abs_step)


def clamp_ee_translation(last_pos: np.ndarray | None, target_pos: np.ndarray, max_step_m: float) -> np.ndarray:
    target_pos = np.asarray(target_pos, dtype=float)
    if last_pos is None:
        return target_pos
    d = target_pos - last_pos
    n = float(np.linalg.norm(d))
    if n <= max_step_m or n == 0.0:
        return target_pos
    return last_pos + d * (max_step_m / n)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True)
    p.add_argument("--target-frame", default="gripper_hand_link_R_1")
    p.add_argument("--hand-side", choices=["left", "right"], default="right")
    p.add_argument("--cloudxr-env", default=None)
    # SIM is deliberately the default so omitting --mode never targets real hardware.
    p.add_argument("--mode", choices=["sim", "real"], default="sim")
    p.add_argument("--cmd-port", type=int, default=None, help="Optional mode-port override.")
    p.add_argument("--feedback-port", type=int, default=None, help="Optional mode-port override.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--clutch-threshold", type=float, default=0.5)
    p.add_argument("--orientation-weight", type=float, default=0.0)
    p.add_argument("--max-ee-step-m", type=float, default=0.005)
    p.add_argument("--max-joint-step-rad", type=float, default=0.02)
    p.add_argument("--feedback-timeout-s", type=float, default=0.20)
    p.add_argument("--enable-command", action="store_true", help="Actually send q_target. Default is DRY RUN.")
    p.add_argument("--initial-q", default=None, help="Dry-run fallback: 7 comma-separated q values [rad].")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.urdf).exists():
        raise FileNotFoundError(args.urdf)

    default_cmd_port, default_feedback_port = MODE_PORTS[args.mode]
    cmd_port = default_cmd_port if args.cmd_port is None else args.cmd_port
    feedback_port = default_feedback_port if args.feedback_port is None else args.feedback_port

    bridge = UdpJointBridge(cmd_port=cmd_port, feedback_port=feedback_port)

    if args.enable_command:
        q_measured = bridge.wait_for_feedback(5.0)
    else:
        try:
            q_measured = bridge.wait_for_feedback(1.0)
        except TimeoutError:
            if args.initial_q is None:
                raise RuntimeError("No feedback in dry-run. Pass --initial-q 'q1,...,q7' [rad] or start the bridge.")
            vals = [float(x) for x in args.initial_q.split(",")]
            if len(vals) != 7:
                raise ValueError("--initial-q must contain exactly 7 values")
            q_measured = np.asarray(vals, dtype=float)

    lo_safe = RIGHT_JOINT_LOWER + JOINT_LIMIT_MARGIN_RAD
    hi_safe = RIGHT_JOINT_UPPER - JOINT_LIMIT_MARGIN_RAD

    if np.any(q_measured < lo_safe) or np.any(q_measured > hi_safe):
        raise RuntimeError(
            "Initial/measured q is outside Areumii joint limits "
            f"(with {JOINT_LIMIT_MARGIN_RAD:.3f} rad margin): {q_measured}"
        )

    kin = RadianKinematics(args.urdf, args.target_frame, RIGHT_JOINT_NAMES)
    home_T_ee = kin.forward_kinematics(q_measured)
    clutch = Clutch(home_T_ee)

    teleop_cfg = XRControllerConfig(
        app_name="AreumiiTeleop",
        hand_side=args.hand_side,
        clutch_threshold=args.clutch_threshold,
        cloudxr_env_file=args.cloudxr_env,
        auto_launch_cloudxr=True,
    )
    xr = XRController(teleop_cfg)

    print("[AREUMII] Connecting IsaacTeleop / CloudXR...")
    xr.connect()
    try:
        print("[AREUMII] Waiting for XR controller tracking...")
        while True:
            xr.get_action()
            if xr.is_tracking:
                break
            time.sleep(0.05)

        print("[AREUMII] XR tracked. SQUEEZE = clutch/deadman. Trigger unused.")
        print(
            f"[AREUMII] MODE={args.mode.upper()} "
            f"cmd=127.0.0.1:{cmd_port} feedback=127.0.0.1:{feedback_port} "
            f"{'COMMAND ENABLED' if args.enable_command else 'DRY RUN'}"
        )

        prev_enabled = False
        q_seed = q_measured.copy()
        q_last_cmd = q_measured.copy()
        last_ee_pos = home_T_ee[:3, 3].copy()

        # Debug references captured at each clutch engage.
        # controller_delta is expressed in XRController's robot-base frame.
        # ee_delta is expressed in the Areumii URDF base frame.
        clutch_controller_origin = None
        clutch_ee_origin = home_T_ee[:3, 3].copy()

        last_print = 0.0
        period = 1.0 / args.fps

        while True:
            t0 = time.perf_counter()

            q_live = bridge.poll_feedback()
            if q_live is not None:
                q_measured = q_live

            a = xr.get_action()
            tracked = xr.is_tracking
            grip_pos = np.asarray(a["grip_pos"], dtype=float)
            grip_quat = np.asarray(a["grip_quat"], dtype=float)
            squeeze = float(a["squeeze"])

            # XR can occasionally report is_tracking=True while the controller pose
            # for that frame is invalid (for example grip_quat=[0,0,0,0]).
            # Never pass an invalid quaternion into Clutch/Rotation.
            quat_norm = float(np.linalg.norm(grip_quat))
            pose_valid = (
                tracked
                and np.all(np.isfinite(grip_pos))
                and np.all(np.isfinite(grip_quat))
                and quat_norm > 1.0e-6
            )

            if pose_valid:
                # Normalize valid controller quaternions before using them.
                grip_quat = grip_quat / quat_norm
            elif prev_enabled:
                print(
                    "[XR SAFETY] controller pose lost/invalid -> clutch released, "
                    "holding last robot target"
                )

            enabled = pose_valid and squeeze > args.clutch_threshold
            engage_edge = enabled and not prev_enabled

            if engage_edge:
                if args.enable_command and not bridge.feedback_is_fresh(args.feedback_timeout_s):
                    print("[SAFETY] Refusing engage: feedback stale")
                    enabled = False
                else:
                    measured_T = kin.forward_kinematics(q_measured)
                    clutch.engage(grip_pos, grip_quat, measured_base_T_ee=measured_T)
                    q_seed = q_measured.copy()
                    q_last_cmd = q_measured.copy()
                    last_ee_pos = measured_T[:3, 3].copy()

                    clutch_controller_origin = grip_pos.copy()
                    clutch_ee_origin = measured_T[:3, 3].copy()

                    print("[AREUMII] clutch ENGAGED q=", np.round(q_measured, 4))
                    print(
                        "[AXIS DEBUG] origin controller xyz=",
                        np.round(clutch_controller_origin, 4),
                        " ee xyz=",
                        np.round(clutch_ee_origin, 4),
                    )

            if enabled:
                if args.enable_command and not bridge.feedback_is_fresh(args.feedback_timeout_s):
                    print("[SAFETY] Feedback timeout -> command stream stopped")
                    enabled = False
                else:
                    ee_pos_raw, ee_quat = clutch.rebase(grip_pos, grip_quat)

                    # Raw relative deltas are the useful values for checking axis mapping.
                    controller_delta = (
                        grip_pos - clutch_controller_origin
                        if clutch_controller_origin is not None
                        else np.zeros(3, dtype=float)
                    )
                    ee_delta_raw = ee_pos_raw - clutch_ee_origin

                    # This is the target actually passed to IK after our safety step clamp.
                    ee_pos = clamp_ee_translation(last_ee_pos, ee_pos_raw, args.max_ee_step_m)
                    ee_delta_cmd = ee_pos - clutch_ee_origin
                    last_ee_pos = ee_pos.copy()

                    desired_T = np.eye(4, dtype=float)
                    desired_T[:3, :3] = Rotation.from_quat(ee_quat).as_matrix()
                    desired_T[:3, 3] = ee_pos

                    # Keep IK continuous from the previous commanded solution.
                    # Do NOT re-seed IK from q_measured every frame: on the real arm,
                    # gravity/low gain can make measured q lag behind the commanded q.
                    q_ik = kin.inverse_kinematics(q_seed, desired_T, args.orientation_weight)
                    if not np.all(np.isfinite(q_ik)):
                        print("[SAFETY] non-finite IK; dropped")
                    elif np.any(q_ik < lo_safe) or np.any(q_ik > hi_safe):
                        bad = np.where((q_ik < lo_safe) | (q_ik > hi_safe))[0].tolist()
                        print(
                            "[SAFETY] IK outside Areumii joint limits; dropped:",
                            "joints=", bad,
                            "q=", np.round(q_ik, 3),
                        )
                    else:
                        # Rate-limit from the previous command, not from encoder feedback.
                        q_cmd = clamp_vector_step(q_last_cmd, q_ik, args.max_joint_step_rad)
                        q_seed = q_cmd.copy()
                        q_last_cmd = q_cmd.copy()
                        if args.enable_command:
                            bridge.send_q(q_cmd)

                        now = time.monotonic()
                        if now - last_print >= 0.20:
                            print(
                                "[AXIS DEBUG]",
                                "controller_dxyz[m]=",
                                np.round(controller_delta, 4),
                                "| ee_raw_dxyz[m]=",
                                np.round(ee_delta_raw, 4),
                                "| ee_cmd_dxyz[m]=",
                                np.round(ee_delta_cmd, 4),
                            )
                            print(
                                "[q_target rad]",
                                np.round(q_cmd, 4),
                                " measured=",
                                np.round(q_measured, 4),
                            )
                            last_print = now

            if prev_enabled and not enabled:
                if pose_valid:
                    print("[AREUMII] clutch RELEASED -> last target remains held")
            prev_enabled = enabled

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        print("\n[AREUMII] Ctrl-C; stopping command updates")
    finally:
        xr.disconnect()


if __name__ == "__main__":
    main()
