"""One-cube Pick & Place scene for Areumii C1.

IMPORTANT ARCHITECTURE
----------------------
This task DOES reuse the working simulation environment:
    stack_joint_pos_env_cfg.AreumiiCubeStackEnvCfg

It DOES NOT reuse or inherit the Stack IK environment.
The local `pick_place_ik_abs_env_cfg.py` defines its own IsaacTeleop/IK pipeline.

So:
    Stack simulation/robot/table/camera config -> reused
    Stack IK/retargeting config                -> NOT imported
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.stack import mdp as stack_mdp
from isaaclab_tasks.contrib.stack.config.areumii_c1 import stack_joint_pos_env_cfg
from isaaclab_tasks.contrib.stack.mdp import franka_stack_events

from ... import mdp as pick_mdp


TABLE_TOP_Z = stack_joint_pos_env_cfg.TABLE_TOP_Z
CUBE_CENTER_Z = stack_joint_pos_env_cfg.CUBE_CENTER_Z

# Cube spawn area.
CUBE_X_RANGE = (0.25, 0.35)
# CUBE_Y_RANGE = (0.2, 0.3)
CUBE_Y_RANGE = (-0.3, -0.2)


# Basket geometry.
BASKET_CENTER_X = 0.30
BASKET_CENTER_Y = 0.0

BASKET_OUTER_X = 0.1
BASKET_OUTER_Y = 0.1
BASKET_WALL_T = 0.005
BASKET_WALL_H = 0.05
BASKET_FLOOR_T = 0.005

BASKET_INNER_X = BASKET_OUTER_X - 2.0 * BASKET_WALL_T
BASKET_INNER_Y = BASKET_OUTER_Y - 2.0 * BASKET_WALL_T

# The cube sits on top of the basket floor, not directly on the table.
BASKET_CUBE_TARGET_Z = CUBE_CENTER_Z + BASKET_FLOOR_T

# The Seattle table visual mesh did not reliably stop the object in the new task,
# so add a thin static collision surface exactly at tabletop height.
SUPPORT_CENTER_X = 0.35
SUPPORT_CENTER_Y = 0.0
SUPPORT_SIZE_X = 0.50
SUPPORT_SIZE_Y = 0.80
SUPPORT_THICKNESS = 0.012


def _workspace_support_cfg():
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PickPlaceTableCollision",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                SUPPORT_CENTER_X,
                SUPPORT_CENTER_Y,
                TABLE_TOP_Z - SUPPORT_THICKNESS * 0.5,
            )
        ),
        spawn=sim_utils.CuboidCfg(
            size=(SUPPORT_SIZE_X, SUPPORT_SIZE_Y, SUPPORT_THICKNESS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.32, 0.32, 0.32),
                roughness=0.7,
            ),
        ),
    )


def _basket_floor_cfg():
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketFloor",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                BASKET_CENTER_X,
                BASKET_CENTER_Y,
                TABLE_TOP_Z + BASKET_FLOOR_T * 0.5,
            )
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BASKET_OUTER_X, BASKET_OUTER_Y, BASKET_FLOOR_T),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.25, 0.95),
                metallic=0.0,
                roughness=0.4,
            ),
        ),
    )


def _basket_wall_cfg(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.25, 0.95),
                metallic=0.0,
                roughness=0.4,
            ),
        ),
    )


def _basket_left_wall_cfg():
    return _basket_wall_cfg(
        name="BasketWallLeft",
        size=(BASKET_WALL_T, BASKET_OUTER_Y, BASKET_WALL_H),
        pos=(
            BASKET_CENTER_X - BASKET_OUTER_X * 0.5 + BASKET_WALL_T * 0.5,
            BASKET_CENTER_Y,
            TABLE_TOP_Z + BASKET_FLOOR_T + BASKET_WALL_H * 0.5,
        ),
    )


def _basket_right_wall_cfg():
    return _basket_wall_cfg(
        name="BasketWallRight",
        size=(BASKET_WALL_T, BASKET_OUTER_Y, BASKET_WALL_H),
        pos=(
            BASKET_CENTER_X + BASKET_OUTER_X * 0.5 - BASKET_WALL_T * 0.5,
            BASKET_CENTER_Y,
            TABLE_TOP_Z + BASKET_FLOOR_T + BASKET_WALL_H * 0.5,
        ),
    )


def _basket_front_wall_cfg():
    return _basket_wall_cfg(
        name="BasketWallFront",
        size=(BASKET_INNER_X, BASKET_WALL_T, BASKET_WALL_H),
        pos=(
            BASKET_CENTER_X,
            BASKET_CENTER_Y - BASKET_OUTER_Y * 0.5 + BASKET_WALL_T * 0.5,
            TABLE_TOP_Z + BASKET_FLOOR_T + BASKET_WALL_H * 0.5,
        ),
    )


def _basket_back_wall_cfg():
    return _basket_wall_cfg(
        name="BasketWallBack",
        size=(BASKET_INNER_X, BASKET_WALL_T, BASKET_WALL_H),
        pos=(
            BASKET_CENTER_X,
            BASKET_CENTER_Y + BASKET_OUTER_Y * 0.5 - BASKET_WALL_T * 0.5,
            TABLE_TOP_Z + BASKET_FLOOR_T + BASKET_WALL_H * 0.5,
        ),
    )


# -----------------------------------------------------------------------------
# Deterministic robot reset
# -----------------------------------------------------------------------------

# User-facing arm order:
#   LEFT  = [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#            wrist_roll, wrist_yaw, wrist_pitch]
#   RIGHT = [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#            wrist_roll, wrist_yaw, wrist_pitch]
#
# Desired pose for BOTH arms:
#   [0, 0, 0, 1.5, 0, 0, 0]
#
# The actual articulation joint order is interleaved L/R, so indices 0..13 are:
#   L_shoulder_pitch, R_shoulder_pitch,
#   L_shoulder_roll,  R_shoulder_roll,
#   L_shoulder_yaw,   R_shoulder_yaw,
#   L_elbow,          R_elbow,
#   L_wrist_roll,     R_wrist_roll,
#   L_wrist_yaw,      R_wrist_yaw,
#   L_wrist_pitch,    R_wrist_pitch.
FIXED_ARM_JOINT_POS = (
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    1.5, 1.5,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
)


def reset_areumii_fixed_pose(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the complete Areumii articulation to the same pose every episode.

    This deliberately replaces the inherited Gaussian reset helper.  That helper
    only writes a subset of joints, which can leave the remaining joints at the
    previous episode's final state.

    Arm pose:
        LEFT  = [0, 0, 0, 1.5, 0, 0, 0]
        RIGHT = [0, 0, 0, 1.5, 0, 0, 0]

    Both grippers are reset open.
    """

    robot = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # Build a full 18-joint state instead of resetting only a subset.
    num_joints = robot.data.joint_pos.shape[1]
    if num_joints != 18:
        raise RuntimeError(
            f"Expected Areumii C1 to have 18 joints, but found {num_joints}. "
            "Update reset_areumii_fixed_pose() if the robot articulation changes."
        )

    # IsaacLab 3.0 can expose a backend tensor wrapper whose `.dtype`
    # is not a torch.dtype.  Use an explicit torch dtype here.
    joint_pos = torch.zeros(
        (len(env_ids), num_joints),
        device=env.device,
        dtype=torch.float32,
    )
    joint_vel = torch.zeros_like(joint_pos)

    # Arm joints (actual articulation order is interleaved L/R).
    fixed_arm = torch.tensor(
        FIXED_ARM_JOINT_POS,
        device=env.device,
        dtype=joint_pos.dtype,
    )
    joint_pos[:, :14] = fixed_arm

    # Grippers open:
    #   14: gripper_left_finger_joint_L
    #   15: gripper_right_finger_joint_L
    #   16: gripper_left_finger_joint_R
    #   17: gripper_right_finger_joint_R
    joint_pos[:, 14] = 0.03
    joint_pos[:, 15] = -0.03
    joint_pos[:, 16] = 0.03
    joint_pos[:, 17] = -0.03

    # Keep the requested pose inside the articulation limits.
    limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

    # Reset BOTH controller targets and the actual physics state.
    # This prevents the previous episode pose from being retained by the PD
    # targets after write_joint_state_to_sim().
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


@configclass
class PickPlaceEventCfg(stack_joint_pos_env_cfg.EventCfg):
    """PickPlace reset events with a deterministic full-joint robot reset."""

    # Disable the inherited helper that only changes default_joint_pos.
    init_areumii_arm_pose = None

    # Replace the inherited Gaussian/subset reset with an exact 18-joint reset.
    randomize_areumii_joint_state = EventTerm(
        func=reset_areumii_fixed_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Cube position remains randomized.
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": CUBE_X_RANGE,
                "y": CUBE_Y_RANGE,
                "z": (CUBE_CENTER_Z, CUBE_CENTER_Z),
                "yaw": (-1.0, 1.0),
            },
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("cube_1")],
        },
    )


@configclass
class PickPlaceObservationsCfg:
    """No Stack-specific cube_2/cube_3 observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class PickPlaceTerminationsCfg:
    time_out = DoneTerm(func=stack_mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={
            "minimum_height": -0.90,
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )

    success = DoneTerm(
        func=pick_mdp.cube_on_target,
        params={
            "cube_name": "cube_1",
            "target_center_xy": (BASKET_CENTER_X, BASKET_CENTER_Y),
            "target_size_xy": (BASKET_INNER_X, BASKET_INNER_Y),
            "target_z": BASKET_CUBE_TARGET_Z,
            "z_tolerance": 0.035,
            "max_linear_speed": 0.12,
        },
    )


@configclass
class AreumiiPickPlaceEnvCfg(stack_joint_pos_env_cfg.AreumiiCubeStackEnvCfg):
    """Simulation env inheritance is intentional; IK inheritance is not."""

    def __post_init__(self):
        # Reuse the working Areumii robot/table/cameras/action declarations.
        super().__post_init__()

        # cube_1 -> red
        self.scene.cube_1.spawn.usd_path = (
            f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd"
        )
        self.scene.cube_1.spawn.scale = (0.8, 0.8, 0.8)

        # Keep cube_1's exact inherited USD/rigid-body/collision configuration.
        self.scene.cube_1.init_state.pos = (0.43, -0.10, CUBE_CENTER_Z)

        # One-object task: remove unused Stack cubes.
        self.scene.cube_2 = None
        self.scene.cube_3 = None

        # Add basket + reliable tabletop physics surface.
        self.scene.pick_place_table_collision = _workspace_support_cfg()
        self.scene.basket_floor = _basket_floor_cfg()
        self.scene.basket_wall_left = _basket_left_wall_cfg()
        self.scene.basket_wall_right = _basket_right_wall_cfg()
        self.scene.basket_wall_front = _basket_front_wall_cfg()
        self.scene.basket_wall_back = _basket_back_wall_cfg()

        # Replace task-level managers.
        self.events = PickPlaceEventCfg()
        self.observations = PickPlaceObservationsCfg()
        self.terminations = PickPlaceTerminationsCfg()

        self.episode_length_s = 45.0