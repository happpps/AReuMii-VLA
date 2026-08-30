"""Recording config: 18D state + 3 RGB cameras + target color id."""

import isaaclab.envs.mdp as base_mdp

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ... import mdp as task_mdp
from .color_pick_place_ik_abs_env_cfg import AreumiiColorPickPlaceIKAbsEnvCfg


AREUMII_JOINT_NAMES = [
    "shoulder_pitch_joint_L",
    "shoulder_roll_joint_L",
    "shouler_yaw_joint_L",
    "elbow_joint_L",
    "wrist_roll_joint_L",
    "wrist_yaw_joint_L",
    "wrist_pitch_joint_L",
    "gripper_left_finger_joint_L",
    "gripper_right_finger_joint_L",
    "shoulder_pitch_joint_R",
    "shoulder_roll_joint_R",
    "shoulder_yaw_joint_R",
    "elbow_joint_R",
    "wrist_roll_joint_R",
    "wrist_yaw_joint_R",
    "wrist_pitch_joint_R",
    "gripper_left_finger_joint_R",
    "gripper_right_finger_joint_R",
]


@configclass
class RecordingObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=AREUMII_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )

        target_color_id = ObsTerm(func=task_mdp.target_color_id)

        front_camera = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("robot_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        left_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        right_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class AreumiiColorPickPlaceRecordEnvCfg(AreumiiColorPickPlaceIKAbsEnvCfg):
    observations: RecordingObservationsCfg = RecordingObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.image_obs_list = [
            "front_camera",
            "left_wrist_camera",
            "right_wrist_camera",
        ]
        self.num_rerenders_on_reset = 3
