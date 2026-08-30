# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.mdp import franka_stack_events
from isaaclab_tasks.contrib.stack.stack_env_cfg import (
    StackEnvCfg,
    apply_default_semantics,
    make_ee_frame_cfg,
)

from isaaclab_assets.robots.areumii_c1 import AREUMII_C1_CFG  # isort: skip

from . import areumii_stack_terminations


# Workspace layout for Areumii C1
GROUND_Z = -1.05
TABLE_SCALE = 0.4

# In the original Isaac Lab stack scene, the table top is around z=0 while
# the ground plane is at z=-1.05. With uniform 0.4 scaling, keep the bottom
# near the ground by lowering the table top accordingly.
TABLE_TOP_Z = GROUND_Z + 1.05 * TABLE_SCALE  # -0.63
CUBE_HALF_HEIGHT = 0.0203
CUBE_CENTER_Z = TABLE_TOP_Z + CUBE_HALF_HEIGHT  # -0.6097

# On-board RGB camera mounted to base_link.
# Initial estimate from the robot geometry; tune these after checking the view.
CAMERA_POS = (0.08, 0.0, 0.7)
CAMERA_ROT = (0.664, -0.664, 0.242, -0.242)  # wxyz, ROS camera convention
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_UPDATE_PERIOD = 0.05  # 20 Hz

# Wrist cameras. The gripper extends roughly along local -Z, so the camera
# optical +Z axis is rotated to look along the gripper's -Z direction.
WRIST_CAMERA_POS = (0.05, 0.0, 0.00)
WRIST_CAMERA_ROT = (0.0, 0.9848, 0.0, -0.1736)
WRIST_CAMERA_WIDTH = 320
WRIST_CAMERA_HEIGHT = 240
WRIST_CAMERA_UPDATE_PERIOD = 0.05  # 20 Hz


@configclass
class EventCfg:
    """Configuration for events."""

    # Robot default joint pose is defined in ArticulationCfg.init_state below.
    randomize_areumii_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.45),
                "y": (-0.16, 0.16),
                "z": (CUBE_CENTER_Z, CUBE_CENTER_Z),
                "yaw": (-1.0, 1.0),
            },
            "min_separation": 0.1,
            "asset_cfgs": [
                SceneEntityCfg("cube_1"),
                SceneEntityCfg("cube_2"),
                SceneEntityCfg("cube_3"),
            ],
        },
    )


@configclass
class DualArmActionsCfg:
    """Action specifications for bimanual Areumii control.

    Declare all four terms at class level so Isaac Lab's ActionManager
    reliably discovers them from cfg.__dict__.
    """

    # Keep the original Stack names for the RIGHT arm.
    arm_action: ActionTermCfg = MISSING
    gripper_action: ActionTermCfg = MISSING

    # Additional LEFT-arm terms.
    left_arm_action: ActionTermCfg = MISSING
    left_gripper_action: ActionTermCfg = MISSING


@configclass
class AreumiiCubeStackEnvCfg(StackEnvCfg):
    """Configuration for the dual-arm Areumii Cube Stack environment.

    Compatibility choice:
    - Keep the inherited `arm_action` / `gripper_action` names for the RIGHT arm.
      This preserves the original Stack task's assumptions.
    - Add `left_arm_action` / `left_gripper_action` as extra action terms.

    Expected action order:
        right arm (7)
        right gripper (1)
        left arm (7)
        left gripper (1)
    Total: 16
    """

    # IMPORTANT:
    # The upstream StackEnvCfg.ActionsCfg only declares arm_action and
    # gripper_action.  We must replace it with a config that explicitly
    # declares all four bimanual action terms.
    actions: DualArmActionsCfg = DualArmActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()

        self.scene.robot = AREUMII_C1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

        # Single source of truth for the robot reset pose.
        # This is used by normal env.reset(), record_demos.py resets, and
        # randomize_joint_by_gaussian_offset via asset.data.default_joint_pos.
        self.scene.robot.init_state.joint_pos = {
            "shoulder_pitch_joint_L": 0.0,
            "shoulder_roll_joint_L": 0.0,
            "shouler_yaw_joint_L": 0.0,
            "elbow_joint_L": 1.5,
            "wrist_roll_joint_L": 0.0,
            "wrist_yaw_joint_L": 0.0,
            "wrist_pitch_joint_L": 0.0,

            "shoulder_pitch_joint_R": 0.0,
            "shoulder_roll_joint_R": 0.0,
            "shoulder_yaw_joint_R": 0.0,
            "elbow_joint_R": 1.5,
            "wrist_roll_joint_R": 0.0,
            "wrist_yaw_joint_R": 0.0,
            "wrist_pitch_joint_R": 0.0,

            "gripper_left_finger_joint_L": 0.0,
            "gripper_right_finger_joint_L": 0.0,
            "gripper_left_finger_joint_R": 0.0,
            "gripper_right_finger_joint_R": 0.0,
        }

        # Spawn an actual RGB camera under base_link.
        # The existing "camera-looking" geometry in the STL is only visual mesh.
        self.scene.robot_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera",
            update_period=CAMERA_UPDATE_PERIOD,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=CAMERA_POS,
                rot=CAMERA_ROT,
                convention="ros",
            ),
        )

        # Left wrist RGB camera.
        self.scene.left_wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_hand_link_L_1/wrist_camera",
            update_period=WRIST_CAMERA_UPDATE_PERIOD,
            height=WRIST_CAMERA_HEIGHT,
            width=WRIST_CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=8.0,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=WRIST_CAMERA_POS,
                rot=WRIST_CAMERA_ROT,
                convention="ros",
            ),
        )

        # Right wrist RGB camera.
        self.scene.right_wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_hand_link_R_1/wrist_camera",
            update_period=WRIST_CAMERA_UPDATE_PERIOD,
            height=WRIST_CAMERA_HEIGHT,
            width=WRIST_CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=8.0,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=WRIST_CAMERA_POS,
                rot=WRIST_CAMERA_ROT,
                convention="ros",
            ),
        )

        # Put the robot base on the ground plane.
        self.scene.robot.init_state.pos = (0.0, 0.0, GROUND_Z)

        # Shrink the inherited Seattle Lab table and lower it so its feet remain
        # approximately on the same ground plane.
        self.scene.table.spawn.scale = (TABLE_SCALE, TABLE_SCALE, TABLE_SCALE)
        self.scene.table.init_state.pos = (0.35, -0.08, TABLE_TOP_Z)
        self.scene.table.init_state.rot = (0.0, 0.0, 0.0, 1.0)

        # Keep the cubes' initial render poses on the resized/lowered table too.
        # Reset randomization will still move x/y afterward.
        self.scene.cube_1.init_state.pos = (0.30, -0.08, CUBE_CENTER_Z)
        self.scene.cube_2.init_state.pos = (0.40,  0.02, CUBE_CENTER_Z)
        self.scene.cube_3.init_state.pos = (0.48, -0.16, CUBE_CENTER_Z)

        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        apply_default_semantics(self.scene)

        # The cubes now sit around z=-0.61, so the inherited dropping threshold
        # (designed for a table near z=0) must also be lowered.
        if self.terminations is not None:
            for term_name in ("cube_1_dropping", "cube_2_dropping", "cube_3_dropping"):
                term = getattr(self.terminations, term_name, None)
                if term is not None:
                    term.params["minimum_height"] = -0.90

            # Areumii uses opposite signs for the two finger joints when open,
            # so use a robot-specific stack success condition.
            if getattr(self.terminations, "success", None) is not None:
                self.terminations.success.func = areumii_stack_terminations.cubes_stacked_areumii

        # ---------------------------------------------------------------------
        # RIGHT ARM
        # Keep the inherited Stack action names for compatibility.
        # ---------------------------------------------------------------------
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pitch_joint_R",
                "shoulder_roll_joint_R",
                "shoulder_yaw_joint_R",
                "elbow_joint_R",
                "wrist_roll_joint_R",
                "wrist_yaw_joint_R",
                "wrist_pitch_joint_R",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "gripper_left_finger_joint_R",
                "gripper_right_finger_joint_R",
            ],
            open_command_expr={
                "gripper_left_finger_joint_R": 0.03,
                "gripper_right_finger_joint_R": -0.03,
            },
            close_command_expr={
                "gripper_left_finger_joint_R": 0.0,
                "gripper_right_finger_joint_R": 0.0,
            },
        )

        # ---------------------------------------------------------------------
        # LEFT ARM
        # These are additional action terms. Isaac Lab's ActionManager will
        # append them after arm_action / gripper_action.
        # ---------------------------------------------------------------------
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pitch_joint_L",
                "shoulder_roll_joint_L",
                "shouler_yaw_joint_L",
                "elbow_joint_L",
                "wrist_roll_joint_L",
                "wrist_yaw_joint_L",
                "wrist_pitch_joint_L",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "gripper_left_finger_joint_L",
                "gripper_right_finger_joint_L",
            ],
            open_command_expr={
                "gripper_left_finger_joint_L": 0.03,
                "gripper_right_finger_joint_L": -0.03,
            },
            close_command_expr={
                "gripper_left_finger_joint_L": 0.0,
                "gripper_right_finger_joint_L": 0.0,
            },
        )

        # ---------------------------------------------------------------------
        # Keep the original Stack task metadata tied to the RIGHT gripper.
        # This avoids changing grasp observations / success logic yet.
        # ---------------------------------------------------------------------
        self.gripper_joint_names = [
            "gripper_left_finger_joint_R",
            "gripper_right_finger_joint_R",
        ]
        self.gripper_open_val = 0.03
        self.gripper_threshold = 0.002

        # Optional metadata for future dual-arm observations/recording.
        self.left_gripper_joint_names = [
            "gripper_left_finger_joint_L",
            "gripper_right_finger_joint_L",
        ]
        self.left_gripper_open_val = 0.03
        self.left_gripper_threshold = 0.002

        # Keep the inherited Stack task's ee_frame on the RIGHT hand for now.
        # The left arm IK does not require a FrameTransformer sensor; it uses
        # `body_name` directly in the IK action configuration.
        self.scene.ee_frame = make_ee_frame_cfg(
            base_prim_path="{ENV_REGEX_NS}/Robot/base_link",
            target_specs=[
                ("{ENV_REGEX_NS}/Robot/gripper_hand_link_R_1", "end_effector", (0.0, 0.0, 0.0)),
                ("{ENV_REGEX_NS}/Robot/gripper_rigft_finger_link_R_1", "tool_rightfinger", (0.0, 0.0, 0.0)),
                ("{ENV_REGEX_NS}/Robot/gripper_left_finger_link_R_1", "tool_leftfinger", (0.0, 0.0, 0.0)),
            ],
            marker_scale=(0.1, 0.1, 0.1),
        )