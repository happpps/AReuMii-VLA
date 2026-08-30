# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
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


# Workspace layout for Areumii C1
GROUND_Z = -1.05
TABLE_SCALE = 0.4

# In the original Isaac Lab stack scene, the table top is around z=0 while
# the ground plane is at z=-1.05. With uniform 0.4 scaling, keep the bottom
# near the ground by lowering the table top accordingly.
TABLE_TOP_Z = GROUND_Z + 1.05 * TABLE_SCALE  # -0.63
CUBE_HALF_HEIGHT = 0.0203
CUBE_CENTER_Z = TABLE_TOP_Z + CUBE_HALF_HEIGHT  # -0.6097


@configclass
class EventCfg:
    """Configuration for events."""

    init_areumii_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, 0.0, 
                             0.0, 0.0, 
                             0.0, 0.0, 
                             1.4, 1.4,
                             0.0, 0.0,
                             0.0, 0.0,
                             0.0, 0.0,],
        },
    )

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
            "pose_range": {"x": (0.25, 0.5), "y": (-0.15, 0.15), "z": (CUBE_CENTER_Z, CUBE_CENTER_Z), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class AreumiiCubeStackEnvCfg(StackEnvCfg):
    """Configuration for the Areumii Cube Stack environment."""

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()

        self.scene.robot = AREUMII_C1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

        # Put the robot base on the ground plane.
        self.scene.robot.init_state.pos = (0.0, 0.0, GROUND_Z)

        # Shrink the inherited Seattle Lab table and lower it so its feet remain
        # approximately on the same ground plane.
        self.scene.table.spawn.scale = (TABLE_SCALE, TABLE_SCALE, TABLE_SCALE)
        self.scene.table.init_state.pos = (0.4, 0.0, TABLE_TOP_Z)
        self.scene.table.init_state.rot = (0.0, 0.0, 0.0, 1.0)

        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        apply_default_semantics(self.scene)

        # The cubes now sit around z=-0.61, so the inherited dropping threshold
        # (designed for a table near z=0) must also be lowered.
        if self.terminations is not None:
            for term_name in ("cube_1_dropping", "cube_2_dropping", "cube_3_dropping"):
                term = getattr(self.terminations, term_name, None)
                if term is not None:
                    term.params["minimum_height"] = -0.90

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
            joint_names=["gripper_left_finger_joint_R", "gripper_right_finger_joint_R"],
            open_command_expr={
                "gripper_left_finger_joint_R": 0.03,
                "gripper_right_finger_joint_R": -0.03,
            },
            close_command_expr={
                "gripper_left_finger_joint_R": 0.0,
                "gripper_right_finger_joint_R": 0.0,
            },
        )
        self.gripper_joint_names = ["gripper_left_finger_joint_R", "gripper_right_finger_joint_R"]
        self.gripper_open_val = 0.03
        self.gripper_threshold = 0.002

        self.scene.ee_frame = make_ee_frame_cfg(
            base_prim_path="{ENV_REGEX_NS}/Robot/base_link",
            target_specs=[
                ("{ENV_REGEX_NS}/Robot/gripper_hand_link_R_1", "end_effector", (0.0, 0.0, 0.0)),
                ("{ENV_REGEX_NS}/Robot/gripper_rigft_finger_link_R_1", "tool_rightfinger", (0.0, 0.0, 0.0)),
                ("{ENV_REGEX_NS}/Robot/gripper_left_finger_link_R_1", "tool_leftfinger", (0.0, 0.0, 0.0)),
            ],
            marker_scale=(0.1, 0.1, 0.1),
        )