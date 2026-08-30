"""Gym registration for one-cube Areumii Pick & Place."""

import gymnasium as gym


gym.register(
    id="IsaacContrib-Pick-Place-Areumii-IK-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pick_place_ik_abs_env_cfg:"
            "AreumiiPickPlaceIKAbsEnvCfg"
        ),
    },
)


gym.register(
    id="IsaacContrib-Pick-Place-Areumii-IK-Abs-Record",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pick_place_record_env_cfg:"
            "AreumiiPickPlaceRecordEnvCfg"
        ),
    },
)
