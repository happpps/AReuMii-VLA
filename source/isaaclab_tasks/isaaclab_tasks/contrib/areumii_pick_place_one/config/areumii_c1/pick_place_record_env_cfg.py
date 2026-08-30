"""Recording configuration for one-cube Areumii Pick & Place."""

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from .pick_place_ik_abs_env_cfg import AreumiiPickPlaceIKAbsEnvCfg


AREUMII_JOINT_NAMES = [
    # LEFT arm + gripper
    "shoulder_pitch_joint_L",
    "shoulder_roll_joint_L",
    "shouler_yaw_joint_L",
    "elbow_joint_L",
    "wrist_roll_joint_L",
    "wrist_yaw_joint_L",
    "wrist_pitch_joint_L",
    "gripper_left_finger_joint_L",
    "gripper_right_finger_joint_L",
    # RIGHT arm + gripper
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

WRIST_CAMERA_POS = (0.04, 0.0, -0.01)
WRIST_CAMERA_ROT = (0.0, 0.9848, 0.0, -0.1736)  # validated ~20 degree tilt


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
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class AreumiiPickPlaceRecordEnvCfg(AreumiiPickPlaceIKAbsEnvCfg):
    """Same local IK as PickPlace + recording cameras/observations."""

    observations: RecordingObservationsCfg = RecordingObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Parent PickPlace env overwrites observations with the lightweight
        # non-recording group. Restore recording observations here.
        self.observations = RecordingObservationsCfg()

        # Keep the existing front camera from stack_joint_pos_env_cfg,
        # but use the resolution already validated for the SmolVLA dataset.
        if getattr(self.scene, "robot_camera", None) is not None:
            self.scene.robot_camera.height = 480
            self.scene.robot_camera.width = 640
            self.scene.robot_camera.update_period = 1/30

        # Wrist cameras are defined locally here; no Stack IK/record env import.
        self.scene.left_wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_hand_link_L_1/left_wrist_camera",
            update_period=1/30,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.02, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=WRIST_CAMERA_POS,
                rot=WRIST_CAMERA_ROT,
                convention="ros",
            ),
        )

        self.scene.right_wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_hand_link_R_1/right_wrist_camera",
            update_period=1/30,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.02, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=WRIST_CAMERA_POS,
                rot=WRIST_CAMERA_ROT,
                convention="ros",
            ),
        )

        self.image_obs_list = [
            "front_camera",
            "left_wrist_camera",
            "right_wrist_camera",
        ]
        self.num_rerenders_on_reset = 3