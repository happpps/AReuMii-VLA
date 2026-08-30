from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cubes_stacked_areumii(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg | None = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    left_open_val: float = 0.03,
    right_open_val: float = -0.03,
    atol: float = 0.002,
) -> torch.Tensor:
    """Success condition for Areumii cube stacking.

    Stack order:
        cube_1 (bottom) -> cube_2 -> cube_3

    Unlike Franka, Areumii's two finger joints have opposite signs when open.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    # cube_1 below cube_2
    diff_12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    stacked = torch.norm(diff_12[:, :2], dim=1) < xy_threshold
    stacked &= torch.abs(torch.norm(diff_12[:, 2:], dim=1) - height_diff) < height_threshold
    stacked &= diff_12[:, 2] < 0.0

    # cube_2 below cube_3
    if cube_3_cfg is not None:
        cube_3: RigidObject = env.scene[cube_3_cfg.name]
        diff_23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

        stacked &= torch.norm(diff_23[:, :2], dim=1) < xy_threshold
        stacked &= torch.abs(torch.norm(diff_23[:, 2:], dim=1) - height_diff) < height_threshold
        stacked &= diff_23[:, 2] < 0.0

    # Areumii right gripper: +0.03 / -0.03 when open
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(gripper_joint_ids) != 2:
        raise ValueError("Areumii stack success expects exactly two right-gripper finger joints.")

    left_open = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[0]],
        torch.tensor(left_open_val, device=env.device),
        atol=atol,
        rtol=0.0,
    )
    right_open = torch.isclose(
        robot.data.joint_pos[:, gripper_joint_ids[1]],
        torch.tensor(right_open_val, device=env.device),
        atol=atol,
        rtol=0.0,
    )

    return stacked & left_open & right_open
