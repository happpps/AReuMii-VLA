# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_teleop import IsaacTeleopCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils.configclass import configclass

from .pick_place_env_cfg import AreumiiPickPlaceEnvCfg


def _build_areumii_pick_place_pipeline():
    # Fixed yaw calibration in the robot-base frame.
    # Tune only these two values if the neutral orientation is slightly off.
    RIGHT_HEADING_REF_DEG = 90.0
    LEFT_HEADING_REF_DEG = 90.0
    """Build the bimanual Isaac Teleop pipeline for Areumii C1.

    Output action order MUST match the ActionManager term order:
        [right EE pose 7,
         right gripper 1,
         left EE pose 7,
         left gripper 1]
    Total: 16D
    """
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import (
        BaseRetargeter,
        OptionalType,
        OutputCombiner,
        RetargeterIOType,
        TensorGroupType,
        ValueInput,
    )
    from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
    from isaacteleop.retargeting_engine.tensor_types import (
        ControllerInput,
        ControllerInputIndex,
        DLDataType,
        NDArrayType,
        TransformMatrix,
    )
    import numpy as np
    from scipy.spatial.transform import Rotation

    class ControllerYawMirrorSe3AbsRetargeter(BaseRetargeter):
        """Controller-only absolute SE(3) retargeter with mirrored heading yaw.

        Important: yaw is corrected BEFORE the fixed target rotation offset.

        We decompose the controller orientation as

            R_controller = Rz(heading) @ R_residual

        where R_residual contains the current pitch/roll/tilt relative to the
        heading frame.  Only the heading displacement from the startup pose is
        mirrored:

            heading_out = heading_ref - wrap(heading - heading_ref)

        Then the original Isaac Teleop target offset is applied:

            R_final = R_corrected_controller @ R_target_offset

        This keeps the startup orientation unchanged and avoids the axis mixing
        caused by editing Euler angles after target_offset_yaw=-90 deg.
        """

        def __init__(
            self,
            config: Se3RetargeterConfig,
            name: str,
            heading_ref_deg: float,
        ):
            self._config = config
            self._target_offset_pos = np.array(
                [
                    config.target_offset_x,
                    config.target_offset_y,
                    config.target_offset_z,
                ],
                dtype=np.float64,
            )
            self._target_offset_rot = Rotation.from_euler(
                "XYZ",
                [
                    config.target_offset_roll,
                    config.target_offset_pitch,
                    config.target_offset_yaw,
                ],
                degrees=True,
            )
            self._heading_ref = np.deg2rad(heading_ref_deg)
            self._last_pose = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            super().__init__(name=name)

        def input_spec(self) -> RetargeterIOType:
            return {
                self._config.input_device: OptionalType(ControllerInput())
            }

        def output_spec(self) -> RetargeterIOType:
            return {
                "ee_pose": TensorGroupType(
                    "ee_pose",
                    [
                        NDArrayType(
                            "pose",
                            shape=(7,),
                            dtype=DLDataType.FLOAT,
                            dtype_bits=32,
                        )
                    ],
                )
            }

        @staticmethod
        def _wrap(angle: float) -> float:
            return float(np.arctan2(np.sin(angle), np.cos(angle)))

        @staticmethod
        def _heading_from_rotation(rot: Rotation) -> float:
            # Standard heading about the target/world Z axis.
            # For R = Rz(yaw) Ry(pitch) Rx(roll):
            # yaw = atan2(R[1, 0], R[0, 0]).
            mat = rot.as_matrix()
            return float(np.arctan2(mat[1, 0], mat[0, 0]))

        def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
            if context.execution_events.reset:
                self._last_pose = np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                )

            ee_pose = outputs["ee_pose"]
            inp = inputs[self._config.input_device]

            if inp.is_none:
                ee_pose[0] = self._last_pose
                return

            if not bool(inp[ControllerInputIndex.GRIP_IS_VALID]):
                ee_pose[0] = self._last_pose
                return

            position = np.from_dlpack(
                inp[ControllerInputIndex.GRIP_POSITION]
            ).astype(np.float64, copy=True)
            grip_quat = np.from_dlpack(
                inp[ControllerInputIndex.GRIP_ORIENTATION]
            ).astype(np.float64, copy=True)

            # Controller orientation is XYZW, matching SciPy.
            base_rot = Rotation.from_quat(grip_quat)

            # Extract current heading about world/robot-base Z.
            heading = self._heading_from_rotation(base_rot)

            # Split current orientation into heading and residual tilt/roll.
            heading_rot = Rotation.from_euler("Z", heading, degrees=False)
            residual_rot = heading_rot.inv() * base_rot

            # Mirror ONLY the heading displacement around startup heading.
            heading_delta = self._wrap(heading - self._heading_ref)
            corrected_heading = self._heading_ref - heading_delta

            corrected_base_rot = (
                Rotation.from_euler("Z", corrected_heading, degrees=False)
                * residual_rot
            )

            # Same position-offset convention as Isaac Teleop:
            # position offset is expressed in the controller/wrist frame.
            position = position + base_rot.apply(self._target_offset_pos)

            # Same fixed target orientation offset as Isaac Teleop.
            final_rot = corrected_base_rot * self._target_offset_rot

            final_pose = np.concatenate(
                [position, final_rot.as_quat()]
            ).astype(np.float32)

            self._last_pose = final_pose
            ee_pose[0] = final_pose

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    # The IsaacTeleop device supplies world_T_anchor.
    # target_frame_prim_path below rebases both controllers into the robot base frame.
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(
        transform_input.output(ValueInput.VALUE)
    )

    # -------------------------------------------------------------------------
    # RIGHT ARM SE(3)
    # -------------------------------------------------------------------------
    right_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=-10.0,
        target_offset_yaw=-90.0,
    )

    right_se3 = ControllerYawMirrorSe3AbsRetargeter(
        right_se3_cfg,
        name="right_ee_pose",
        heading_ref_deg=RIGHT_HEADING_REF_DEG,
    )
    connected_right_se3 = right_se3.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    right_gripper_cfg = GripperRetargeterConfig(hand_side="right")
    right_gripper = GripperRetargeter(right_gripper_cfg, name="right_gripper")
    connected_right_gripper = right_gripper.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # LEFT ARM SE(3)
    # -------------------------------------------------------------------------
    left_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=-10.0,
        target_offset_yaw=-90.0,
    )

    left_se3 = ControllerYawMirrorSe3AbsRetargeter(
        left_se3_cfg,
        name="left_ee_pose",
        heading_ref_deg=LEFT_HEADING_REF_DEG,
    )
    connected_left_se3 = left_se3.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
        }
    )

    left_gripper_cfg = GripperRetargeterConfig(hand_side="left")
    left_gripper = GripperRetargeter(left_gripper_cfg, name="left_gripper")
    connected_left_gripper = left_gripper.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
        }
    )

    # -------------------------------------------------------------------------
    # Flatten into one 16D action tensor.
    #
    # The labels below are intentionally side-prefixed so every tensor element
    # has a unique name inside TensorReorderer.
    #
    # Isaac Teleop SE(3) retargeter output itself is xyzw.  The environment's
    # DifferentialInverseKinematicsAction expects pose commands in wxyz, so we
    # preserve the quaternion reorder that already worked for the single arm.
    # -------------------------------------------------------------------------
    right_ee_elements = [
        "right_pos_x",
        "right_pos_y",
        "right_pos_z",
        "right_quat_x",
        "right_quat_y",
        "right_quat_z",
        "right_quat_w",
    ]
    left_ee_elements = [
        "left_pos_x",
        "left_pos_y",
        "left_pos_z",
        "left_quat_x",
        "left_quat_y",
        "left_quat_z",
        "left_quat_w",
    ]

    reorderer = TensorReorderer(
        input_config={
            "right_ee_pose": right_ee_elements,
            "right_gripper_command": ["right_gripper_value"],
            "left_ee_pose": left_ee_elements,
            "left_gripper_command": ["left_gripper_value"],
        },
        output_order=[
            # RIGHT arm_action (7)
            "right_pos_x",
            "right_pos_y",
            "right_pos_z",
            "right_quat_w",
            "right_quat_x",
            "right_quat_y",
            "right_quat_z",
            # RIGHT gripper_action (1)
            "right_gripper_value",
            # LEFT left_arm_action (7)
            "left_pos_x",
            "left_pos_y",
            "left_pos_z",
            "left_quat_w",
            "left_quat_x",
            "left_quat_y",
            "left_quat_z",
            # LEFT left_gripper_action (1)
            "left_gripper_value",
        ],
        name="bimanual_action_reorderer",
        input_types={
            "right_ee_pose": "array",
            "right_gripper_command": "scalar",
            "left_ee_pose": "array",
            "left_gripper_command": "scalar",
        },
    )

    connected_reorderer = reorderer.connect(
        {
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "right_gripper_command": connected_right_gripper.output("gripper_command"),
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "left_gripper_command": connected_left_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner(
        {
            "action": connected_reorderer.output("output"),
        }
    )


@configclass
class AreumiiPickPlaceIKAbsEnvCfg(AreumiiPickPlaceEnvCfg):
    """Areumii one-cube Pick & Place with local bimanual absolute XR teleoperation.

    The parent environment uses DualArmActionsCfg, so the ActionManager should
    report exactly four active terms and total_action_dim == 16.
    """

    _TELEOP_AVAILABLE = True

    def __post_init__(self):
        super().__post_init__()

        # Keep your currently working XR start position.
        self.xr.anchor_pos = (-0.1, -0.0, -1.7)

        # ---------------------------------------------------------------------
        # RIGHT ARM IK
        #
        # Keep the name `arm_action` so the original Stack task remains
        # compatible with code that expects that action term.
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # LEFT ARM IK
        # ---------------------------------------------------------------------
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
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
            body_name="gripper_hand_link_L_1",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
        )

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_areumii_pick_place_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,

            # Both controller poses are expressed relative to the common robot base.
            target_frame_prim_path="/World/envs/env_0/Robot/base_link",
        )