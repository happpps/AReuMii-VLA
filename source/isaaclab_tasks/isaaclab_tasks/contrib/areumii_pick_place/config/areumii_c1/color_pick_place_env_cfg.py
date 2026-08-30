"""Areumii C1 color-conditioned cube pick-and-place environment.

Key design choice:
- Reuse the exact cube_1 / cube_2 / cube_3 RigidObjectCfg objects from the
  already-working Areumii Stack task.
- Only swap their USD colors and change task logic.
- Therefore cube collision / rigid-body behavior stays identical to the proven
  Stack environment.

Color mapping:
    cube_1 = red
    cube_2 = yellow
    cube_3 = green
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack import mdp as stack_mdp
from isaaclab_tasks.contrib.stack.config.areumii_c1 import stack_joint_pos_env_cfg
from isaaclab_tasks.contrib.stack.mdp import franka_stack_events

from ... import mdp as task_mdp


# -----------------------------------------------------------------------------
# Workspace constants
# -----------------------------------------------------------------------------

GROUND_Z = stack_joint_pos_env_cfg.GROUND_Z
TABLE_TOP_Z = stack_joint_pos_env_cfg.TABLE_TOP_Z
CUBE_CENTER_Z = stack_joint_pos_env_cfg.CUBE_CENTER_Z

# Basket center on the table.
BASKET_CENTER_X = 0.30
BASKET_CENTER_Y = 0.20

BASKET_OUTER_X = 0.1
BASKET_OUTER_Y = 0.1
BASKET_WALL_T = 0.005
BASKET_WALL_H = 0.05
BASKET_FLOOR_T = 0.005

BASKET_INNER_X = BASKET_OUTER_X - 2.0 * BASKET_WALL_T
BASKET_INNER_Y = BASKET_OUTER_Y - 2.0 * BASKET_WALL_T


def _static_blue_box_piece(name: str, size, pos):
    """Create one static collision piece of the open-top blue basket."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.25, 0.85),
                metallic=0.0,
                roughness=0.45,
            ),
        ),
    )


@configclass
class ColorPickPlaceEventCfg(stack_joint_pos_env_cfg.EventCfg):
    """Keep the working robot reset event; customize cube reset + target color."""

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            # Keep the three cubes away from the basket at +Y.
            "pose_range": {
                "x": (0.32, 0.50),
                "y": (-0.18, 0.05),
                "z": (CUBE_CENTER_Z, CUBE_CENTER_Z),
                "yaw": (-1.0, 1.0),
            },
            "min_separation": 0.08,
            "asset_cfgs": [
                SceneEntityCfg("cube_1"),
                SceneEntityCfg("cube_2"),
                SceneEntityCfg("cube_3"),
            ],
        },
    )

    randomize_target_color = EventTerm(
        func=task_mdp.randomize_target_color,
        mode="reset",
    )


@configclass
class ColorPickPlaceObservationsCfg:
    """Minimal observations for scene / teleop testing."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos)
        target_color_id = ObsTerm(func=task_mdp.target_color_id)

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ColorPickPlaceTerminationsCfg:
    """Termination terms for the task."""

    time_out = DoneTerm(func=stack_mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={
            "minimum_height": -0.90,
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )
    cube_2_dropping = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={
            "minimum_height": -0.90,
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )
    cube_3_dropping = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={
            "minimum_height": -0.90,
            "asset_cfg": SceneEntityCfg("cube_3"),
        },
    )

    success = DoneTerm(
        func=task_mdp.target_object_in_basket,
        params={
            # 0=red=cube_1, 1=yellow=cube_2, 2=green=cube_3
            "object_names": ("cube_1", "cube_2", "cube_3"),
            "basket_center_xy": (BASKET_CENTER_X, BASKET_CENTER_Y),
            "basket_inner_size_xy": (BASKET_INNER_X, BASKET_INNER_Y),
            "z_range": (
                TABLE_TOP_Z + BASKET_FLOOR_T,
                TABLE_TOP_Z + BASKET_FLOOR_T + BASKET_WALL_H + 0.03,
            ),
            "max_linear_speed": 0.15,
        },
    )


@configclass
class AreumiiColorPickPlaceEnvCfg(
    stack_joint_pos_env_cfg.AreumiiCubeStackEnvCfg
):
    """Pick the commanded colored cube and put it into the blue basket."""

    def __post_init__(self):
        # This creates the proven Areumii robot, cameras, table and all 3 cubes.
        super().__post_init__()

        self.scene.table_collision = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TableCollision",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.35, -0.08, TABLE_TOP_Z - 0.01),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.50, 0.40, 0.01),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.3, 0.3, 0.3),
                ),
            ),
        )

        # ------------------------------------------------------------------
        # Keep the SAME cube configs/physics from Stack.
        # Only change visual USD assets:
        #   cube_1 -> red
        #   cube_2 -> yellow
        #   cube_3 -> green
        # ------------------------------------------------------------------
        self.scene.cube_1.spawn.usd_path = (
            f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd"
        )
        self.scene.cube_2.spawn.usd_path = (
            f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/yellow_block.usd"
        )
        self.scene.cube_3.spawn.usd_path = (
            f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd"
        )

        # Keep initial render poses on the tabletop.
        self.scene.cube_1.init_state.pos = (0.36, -0.10, CUBE_CENTER_Z)
        self.scene.cube_2.init_state.pos = (0.44, -0.02, CUBE_CENTER_Z)
        self.scene.cube_3.init_state.pos = (0.50, -0.14, CUBE_CENTER_Z)

        # ------------------------------------------------------------------
        # Static open-top basket = floor + four collision walls.
        # ------------------------------------------------------------------
        floor_z = TABLE_TOP_Z + BASKET_FLOOR_T * 0.5
        wall_z = TABLE_TOP_Z + BASKET_FLOOR_T + BASKET_WALL_H * 0.5

        self.scene.basket_floor = _static_blue_box_piece(
            "BasketFloor",
            (BASKET_OUTER_X, BASKET_OUTER_Y, BASKET_FLOOR_T),
            (BASKET_CENTER_X, BASKET_CENTER_Y, floor_z),
        )
        self.scene.basket_wall_left = _static_blue_box_piece(
            "BasketWallLeft",
            (BASKET_WALL_T, BASKET_OUTER_Y, BASKET_WALL_H),
            (
                BASKET_CENTER_X - (BASKET_OUTER_X - BASKET_WALL_T) * 0.5,
                BASKET_CENTER_Y,
                wall_z,
            ),
        )
        self.scene.basket_wall_right = _static_blue_box_piece(
            "BasketWallRight",
            (BASKET_WALL_T, BASKET_OUTER_Y, BASKET_WALL_H),
            (
                BASKET_CENTER_X + (BASKET_OUTER_X - BASKET_WALL_T) * 0.5,
                BASKET_CENTER_Y,
                wall_z,
            ),
        )
        self.scene.basket_wall_front = _static_blue_box_piece(
            "BasketWallFront",
            (BASKET_OUTER_X, BASKET_WALL_T, BASKET_WALL_H),
            (
                BASKET_CENTER_X,
                BASKET_CENTER_Y - (BASKET_OUTER_Y - BASKET_WALL_T) * 0.5,
                wall_z,
            ),
        )
        self.scene.basket_wall_back = _static_blue_box_piece(
            "BasketWallBack",
            (BASKET_OUTER_X, BASKET_WALL_T, BASKET_WALL_H),
            (
                BASKET_CENTER_X,
                BASKET_CENTER_Y + (BASKET_OUTER_Y - BASKET_WALL_T) * 0.5,
                wall_z,
            ),
        )

        # Replace Stack-specific reset / observation / success logic.
        self.events = ColorPickPlaceEventCfg()
        self.observations = ColorPickPlaceObservationsCfg()
        self.terminations = ColorPickPlaceTerminationsCfg()

        self.episode_length_s = 45.0
        self.viewer.eye = [1.3, -1.0, 0.7]
        self.viewer.lookat = [0.35, 0.0, TABLE_TOP_Z]
