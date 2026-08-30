#!/usr/bin/env python3
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

from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
from isaacteleop.retargeting_engine.tensor_types import TransformMatrix
from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex


JOINT_NAMES = {
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

TARGET_FRAME = {
    "left": "gripper_hand_link_L_1",
    "right": "gripper_hand_link_R_1",
}

JOINT_LOWER = np.array(
    [-3.142, -0.262, -1.396, -0.873, -1.745, -0.524, -1.571]
)
JOINT_UPPER = np.array(
    [3.142, 2.967, 1.396, 2.094, 1.745, 0.524, 1.571]
)
JOINT_MARGIN = 0.01

MODE_PORTS = {
    "sim": (5010, 5011),
    "real": (5005, 5006),
}


class BimanualXRController(XRController):
    """기존 XRController에서 controller output만 좌/우 둘 다 꺼냄."""

    def _build_pipeline(self):
        controllers = ControllersSource(name="controllers")
        xform = ValueInput("base_T_anchor", TransformMatrix())
        transformed = controllers.transformed(xform.output("value"))

        return OutputCombiner({
            "left": transformed.output("controller_left"),
            "right": transformed.output("controller_right"),
        })

    def get_action(self):
        result = self._step(
            execution_events=self._running_events(),
            external_inputs=self._external_inputs,
        )

        actions = {}
        for side in ("left", "right"):
            c = result[side]

            if getattr(c, "is_none", False):
                actions[side] = None
                continue

            try:
                actions[side] = {
                    "grip_pos": np.asarray(
                        c[ControllerInputIndex.GRIP_POSITION],
                        dtype=np.float32,
                    ),
                    "grip_quat": np.asarray(
                        c[ControllerInputIndex.GRIP_ORIENTATION],
                        dtype=np.float32,
                    ),
                    "squeeze": float(c[ControllerInputIndex.SQUEEZE_VALUE]),
                }
            except (IndexError, KeyError, TypeError, ValueError):
                actions[side] = None

        return actions


class Kinematics:
    def __init__(self, urdf, frame, names):
        self.kin = RobotKinematics(
            urdf_path=urdf,
            target_frame_name=frame,
            joint_names=names,
        )

    def fk(self, q):
        return self.kin.forward_kinematics(np.rad2deg(q))

    def ik(self, q_seed, T, orientation_weight):
        q_deg = self.kin.inverse_kinematics(
            np.rad2deg(q_seed),
            T,
            orientation_weight=orientation_weight,
        )
        return np.deg2rad(np.asarray(q_deg, dtype=float))


class UdpBridge:
    def __init__(self, cmd_port, fb_port):
        self.cmd_addr = ("127.0.0.1", cmd_port)

        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("127.0.0.1", fb_port))
        self.rx.setblocking(False)

        self.latest_q = None
        self.latest_time = 0.0

    def poll(self):
        while True:
            try:
                data, _ = self.rx.recvfrom(4096)
            except BlockingIOError:
                break

            try:
                q = np.asarray(
                    [float(x) for x in data.decode().strip().split(",")],
                    dtype=float,
                )
            except Exception:
                continue

            if len(q) == 28 and np.all(np.isfinite(q)):
                self.latest_q = q
                self.latest_time = time.monotonic()

        return None if self.latest_q is None else self.latest_q.copy()

    def wait(self, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            q = self.poll()
            if q is not None:
                return q
            time.sleep(0.01)
        raise TimeoutError("No 28D feedback: measured L7,R7 + control L7,R7")

    def fresh(self, max_age):
        self.poll()
        return (
            self.latest_q is not None
            and time.monotonic() - self.latest_time <= max_age
        )

    def send(self, side, q):
        tag = "L" if side == "left" else "R"
        msg = tag + "," + ",".join(f"{v:.12g}" for v in q)
        self.tx.sendto(msg.encode(), self.cmd_addr)


def clamp_joint_step(current, target, max_step):
    return current + np.clip(target - current, -max_step, max_step)


def clamp_ee_step(current, target, max_step):
    d = target - current
    n = np.linalg.norm(d)

    if n <= max_step or n == 0:
        return target

    return current + d * (max_step / n)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True)
    p.add_argument("--mode", choices=["sim", "real"], default="sim")
    p.add_argument("--cmd-port", type=int)
    p.add_argument("--feedback-port", type=int)
    p.add_argument("--cloudxr-env", default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--clutch-threshold", type=float, default=0.5)
    p.add_argument("--orientation-weight", type=float, default=0.01)
    p.add_argument("--max-ee-step-m", type=float, default=0.005)
    p.add_argument("--max-joint-step-rad", type=float, default=0.02)
    p.add_argument("--feedback-timeout-s", type=float, default=0.2)
    p.add_argument("--enable-command", action="store_true")
    return p.parse_args()



def warn_throttled(st, message, period=0.25):
    now = time.monotonic()
    if now - st["last_warn"] >= period:
        print(message)
        st["last_warn"] = now


def main():
    args = parse_args()

    if not Path(args.urdf).exists():
        raise FileNotFoundError(args.urdf)

    cmd_default, fb_default = MODE_PORTS[args.mode]
    cmd_port = args.cmd_port or cmd_default
    fb_port = args.feedback_port or fb_default

    bridge = UdpBridge(cmd_port, fb_port)
    q14 = bridge.wait()

    q_measured = {
        "left": q14[:7].copy(),
        "right": q14[7:14].copy(),
    }
    q_control = {
        "left": q14[14:21].copy(),
        "right": q14[21:28].copy(),
    }

    lo = JOINT_LOWER + JOINT_MARGIN
    hi = JOINT_UPPER - JOINT_MARGIN

    arms = {}

    for side in ("left", "right"):
        kin = Kinematics(
            args.urdf,
            TARGET_FRAME[side],
            JOINT_NAMES[side],
        )

        T = kin.fk(q_measured[side])

        arms[side] = {
            "kin": kin,
            "clutch": Clutch(T),
            # IK state follows actual/measured robot geometry.
            "q_seed": q_measured[side].copy(),
            # Motor command state follows the CURRENT CAN/SHM control target.
            "q_cmd": q_control[side].copy(),
            "last_ee": T[:3, 3].copy(),
            "enabled": False,
            "last_warn": 0.0,
        }

    xr = BimanualXRController(
        XRControllerConfig(
            app_name="AreumiiBimanualTeleop",
            hand_side="right",  # subclass에서는 실제 side 선택에 사용 안 함
            clutch_threshold=args.clutch_threshold,
            cloudxr_env_file=args.cloudxr_env,
            auto_launch_cloudxr=True,
        )
    )

    print("[AREUMII] Connecting XR...")
    xr.connect()
    print("[AREUMII] LEFT squeeze = left arm / RIGHT squeeze = right arm")

    period = 1.0 / args.fps

    try:
        while True:
            t0 = time.perf_counter()

            q_live = bridge.poll()
            if q_live is not None:
                q_measured["left"] = q_live[:7]
                q_measured["right"] = q_live[7:14]
                q_control["left"] = q_live[14:21]
                q_control["right"] = q_live[21:28]

            actions = xr.get_action()

            for side in ("left", "right"):
                st = arms[side]
                a = actions[side]

                pose_valid = False

                if a is not None:
                    pos = np.asarray(a["grip_pos"], dtype=float)
                    quat = np.asarray(a["grip_quat"], dtype=float)
                    quat_norm = np.linalg.norm(quat)

                    pose_valid = (
                        np.all(np.isfinite(pos))
                        and np.all(np.isfinite(quat))
                        and quat_norm > 1e-6
                    )

                    if pose_valid:
                        quat = quat / quat_norm

                enabled = (
                    pose_valid
                    and a["squeeze"] > args.clutch_threshold
                )

                if enabled and not st["enabled"]:
                    if (
                        args.enable_command
                        and not bridge.fresh(args.feedback_timeout_s)
                    ):
                        warn_throttled(
                            st,
                            f"[{side.upper()} STOP] feedback stale on clutch engage"
                        )
                        enabled = False
                    else:
                        T = st["kin"].fk(q_measured[side])

                        st["clutch"].engage(
                            pos,
                            quat,
                            measured_base_T_ee=T,
                        )

                        # IMPORTANT handoff:
                        #   measured q -> FK / IK geometry
                        #   control q  -> motor command continuity
                        # Do not overwrite the current hold command with measured q.
                        st["q_seed"] = q_measured[side].copy()
                        st["q_cmd"] = q_control[side].copy()
                        st["last_ee"] = T[:3, 3].copy()

                        print(
                            f"[{side.upper()}] clutch ENGAGED "
                            f"measured={np.round(q_measured[side], 3)} "
                            f"control={np.round(q_control[side], 3)}"
                        )

                if enabled:
                    if (
                        args.enable_command
                        and not bridge.fresh(args.feedback_timeout_s)
                    ):
                        warn_throttled(
                            st,
                            f"[{side.upper()} STOP] feedback stale"
                        )
                        enabled = False
                    else:
                        ee_pos, ee_quat = st["clutch"].rebase(pos, quat)

                        ee_pos = clamp_ee_step(
                            st["last_ee"],
                            np.asarray(ee_pos, dtype=float),
                            args.max_ee_step_m,
                        )
                        st["last_ee"] = ee_pos.copy()

                        T = np.eye(4)
                        T[:3, :3] = Rotation.from_quat(ee_quat).as_matrix()
                        T[:3, 3] = ee_pos

                        q_ik = st["kin"].ik(
                            st["q_seed"],
                            T,
                            args.orientation_weight,
                        )

                        if not np.all(np.isfinite(q_ik)):
                            warn_throttled(
                                st,
                                f"[{side.upper()} STOP] IK non-finite: "
                                f"{np.round(q_ik, 3)}"
                            )

                        elif np.any(q_ik < lo) or np.any(q_ik > hi):
                            bad = np.where((q_ik < lo) | (q_ik > hi))[0]
                            warn_throttled(
                                st,
                                f"[{side.upper()} STOP] IK joint limit "
                                f"joints={bad.tolist()} q={np.round(q_ik, 3)}"
                            )

                        else:
                            # q_ik lives in measured/IK coordinates, while q_cmd
                            # carries the existing CAN hold offset. Apply only the
                            # IK joint DELTA to the existing motor command.
                            ik_delta = q_ik - st["q_seed"]
                            ik_delta = np.clip(
                                ik_delta,
                                -args.max_joint_step_rad,
                                args.max_joint_step_rad,
                            )
                            q_cmd = st["q_cmd"] + ik_delta

                            st["q_seed"] = q_ik.copy()
                            st["q_cmd"] = q_cmd.copy()

                            if args.enable_command:
                                bridge.send(side, q_cmd)

                if st["enabled"] and not enabled:
                    if not pose_valid:
                        print(f"[{side.upper()} STOP] XR pose/tracking invalid")
                    else:
                        print(f"[{side.upper()}] clutch RELEASED")

                st["enabled"] = enabled

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        print("\n[AREUMII] stop")

    finally:
        xr.disconnect()


if __name__ == "__main__":
    main()
