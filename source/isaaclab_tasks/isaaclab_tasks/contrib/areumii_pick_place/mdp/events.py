"""Reset/event functions for color cube pick-and-place."""

from __future__ import annotations

import os
import torch

_COLOR_NAME_TO_ID = {"red": 0, "yellow": 1, "green": 2}


def randomize_target_color(env, env_ids: torch.Tensor | None):
    """Choose target color: 0=red, 1=yellow, 2=green."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if not hasattr(env, "target_color_id"):
        env.target_color_id = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    requested = os.getenv("AREUMII_TARGET_COLOR", "random").strip().lower()

    if requested in _COLOR_NAME_TO_ID:
        env.target_color_id[env_ids] = _COLOR_NAME_TO_ID[requested]
    else:
        env.target_color_id[env_ids] = torch.randint(
            low=0,
            high=3,
            size=(len(env_ids),),
            dtype=torch.long,
            device=env.device,
        )

    if len(env_ids) == 1:
        color_id = int(env.target_color_id[env_ids[0]].item())
        color_name = ("RED", "YELLOW", "GREEN")[color_id]
        print(
            f"[ColorPickPlace] TARGET = {color_name}: "
            f"pick the {color_name.lower()} cube and place it in the blue basket."
        )
