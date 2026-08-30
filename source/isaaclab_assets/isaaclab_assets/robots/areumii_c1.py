# areumii_c1.py

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "areumii_c1"

AREUMII_C1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "areumii_c1_realcolor.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
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

            "gripper_left_finger_joint_L": 0.03,
            "gripper_right_finger_joint_L": -0.03,
            "gripper_left_finger_joint_R": 0.03,
            "gripper_right_finger_joint_R": -0.03,
        },
    ),

    actuators={
        "areumii_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_joint_[LR]",
                "shoulder_roll_joint_[LR]",
                "shouler_yaw_joint_L",
                "shoulder_yaw_joint_R",
                "elbow_joint_[LR]",
                "wrist_roll_joint_[LR]",
                "wrist_yaw_joint_[LR]",
                "wrist_pitch_joint_[LR]",
            ],
            velocity_limit_sim=100.0,
            effort_limit_sim=100.0,
            stiffness=80.0,
            damping=4.0,
            armature=1e-3,
        ),
        "areumii_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "gripper_left_finger_joint_L",
                "gripper_right_finger_joint_L",
                "gripper_left_finger_joint_R",
                "gripper_right_finger_joint_R",
            ],
            velocity_limit_sim=100.0,
            effort_limit_sim=100.0,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

"""Configuration for the Areumii C1 robot."""
