"""Success functions for color cube pick-and-place."""

from __future__ import annotations

import torch


def target_object_in_basket(
    env,
    object_names: tuple[str, str, str] = ("cube_1", "cube_2", "cube_3"),
    basket_center_xy: tuple[float, float] = (0.30, 0.20),
    basket_inner_size_xy: tuple[float, float] = (0.13, 0.13),
    z_range: tuple[float, float] = (-0.62, -0.53),
    max_linear_speed: float = 0.15,
) -> torch.Tensor:
    """Success when the commanded cube is stably inside the basket."""
    if not hasattr(env, "target_color_id"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    positions = torch.stack(
        [env.scene[name].data.root_pos_w[:, :3] for name in object_names],
        dim=1,
    )
    velocities = torch.stack(
        [env.scene[name].data.root_lin_vel_w[:, :3] for name in object_names],
        dim=1,
    )

    env_ids = torch.arange(env.num_envs, device=env.device)
    target_ids = env.target_color_id.to(dtype=torch.long)

    pos = positions[env_ids, target_ids]
    vel = velocities[env_ids, target_ids]

    cx, cy = basket_center_xy
    sx, sy = basket_inner_size_xy
    z_min, z_max = z_range

    inside = (
        (torch.abs(pos[:, 0] - cx) <= sx * 0.5)
        & (torch.abs(pos[:, 1] - cy) <= sy * 0.5)
        & (pos[:, 2] >= z_min)
        & (pos[:, 2] <= z_max)
    )
    settled = torch.linalg.norm(vel, dim=-1) <= max_linear_speed

    return inside & settled
