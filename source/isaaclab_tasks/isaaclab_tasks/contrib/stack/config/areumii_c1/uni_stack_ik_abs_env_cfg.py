# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_teleop import IsaacTeleopCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils.configclass import configclass

from . import uni_stack_joint_pos_env_cfg


def _build_areumii_stack_pipeline():
    """Build the Isaac Teleop retargeting pipeline for the Areumii right arm."""
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    # Keep the standard Isaac Teleop XR transform input.
    # target_frame_prim_path below rebases the XR/controller pose
    # into the robot base frame.
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(
        transform_input.output(ValueInput.VALUE)
    )

    se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=-90.0,
    )

    se3 = Se3AbsRetargeter(se3_cfg, name="ee_pose")
    connected_se3 = se3.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    gripper_cfg = GripperRetargeterConfig(hand_side="right")
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    connected_gripper = gripper.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # Isaac Teleop SE(3) output is xyzw.
    ee_pose_elements = [
        "pos_x",
        "pos_y",
        "pos_z",
        "quat_x",
        "quat_y",
        "quat_z",
        "quat_w",
    ]
    gripper_elements = ["gripper_value"]

    reorderer = TensorReorderer(
        input_config={
            "ee_pose": ee_pose_elements,
            "gripper_command": gripper_elements,
        },
        # Keep the quaternion reorder that already stabilized this task.
        output_order=[
            "pos_x",
            "pos_y",
            "pos_z",
            "quat_w",
            "quat_x",
            "quat_y",
            "quat_z",
            "gripper_value",
        ],
        name="action_reorderer",
        input_types={
            "ee_pose": "array",
            "gripper_command": "scalar",
        },
    )

    connected_reorderer = reorderer.connect(
        {
            "ee_pose": connected_se3.output("ee_pose"),
            "gripper_command": connected_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner(
        {
            "action": connected_reorderer.output("output"),
        }
    )


@configclass
class AreumiiCubeStackEnvCfg(uni_stack_joint_pos_env_cfg.AreumiiCubeStackEnvCfg):
    """Areumii cube-stack environment with absolute XR teleoperation."""

    _TELEOP_AVAILABLE = True

    def __post_init__(self):
        super().__post_init__()

        # XR 시작 위치 조정
        self.xr.anchor_pos = (-0.2, -0.0, -1.9)

        # IMPORTANT:
        # Do not replace self.scene.robot here.
        # The parent stack_joint_pos_env_cfg owns the final robot root pose,
        # so moving the robot there is preserved in this IK environment.

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
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
            body_name="gripper_hand_link_R_1",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
        )

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_areumii_stack_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,

            # Feed controller poses to IK in the robot base frame rather than world frame.
            target_frame_prim_path="/World/envs/env_0/Robot/base_link",
        )