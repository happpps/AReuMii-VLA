"""Gym registrations for Areumii color cube pick-and-place."""

import gymnasium as gym


gym.register(
    id="IsaacContrib-Color-Pick-Place-Areumii-IK-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.color_pick_place_ik_abs_env_cfg:"
            "AreumiiColorPickPlaceIKAbsEnvCfg"
        ),
    },
)


gym.register(
    id="IsaacContrib-Color-Pick-Place-Areumii-IK-Abs-Record",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.color_pick_place_record_env_cfg:"
            "AreumiiColorPickPlaceRecordEnvCfg"
        ),
    },
)
