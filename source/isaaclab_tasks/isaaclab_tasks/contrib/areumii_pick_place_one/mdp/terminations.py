from __future__ import annotations
import torch


def cube_on_target(
    env,
    cube_name: str = "cube_1",
    target_center_xy: tuple[float, float] = (0.30, 0.05),
    target_size_xy: tuple[float, float] = (0.12, 0.12),
    target_z: float = -0.6097,
    z_tolerance: float = 0.03,
    max_linear_speed: float = 0.12,
) -> torch.Tensor:
    cube = env.scene[cube_name]
    pos = cube.data.root_pos_w[:, :3] - env.scene.env_origins
    vel = cube.data.root_lin_vel_w[:, :3]

    cx, cy = target_center_xy
    sx, sy = target_size_xy

    inside_xy = (
        (torch.abs(pos[:, 0] - cx) <= sx * 0.5)
        & (torch.abs(pos[:, 1] - cy) <= sy * 0.5)
    )
    correct_z = torch.abs(pos[:, 2] - target_z) <= z_tolerance
    settled = torch.linalg.norm(vel, dim=-1) <= max_linear_speed
    return inside_xy & correct_z & settled
