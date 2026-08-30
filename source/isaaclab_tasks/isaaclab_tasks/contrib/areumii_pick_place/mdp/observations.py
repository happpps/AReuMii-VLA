"""Observation functions for color cube pick-and-place."""

from __future__ import annotations

import torch


def target_color_id(env) -> torch.Tensor:
    """Return target color id as (num_envs, 1) float32."""
    if not hasattr(env, "target_color_id"):
        return torch.zeros(
            (env.num_envs, 1), dtype=torch.float32, device=env.device
        )
    return env.target_color_id.to(dtype=torch.float32).unsqueeze(-1)
