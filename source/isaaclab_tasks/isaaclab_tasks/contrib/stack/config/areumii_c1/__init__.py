# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Stack-Cube-Areumii-IK-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_abs_env_cfg:AreumiiCubeStackEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Stack-Cube-Aruemii-IK-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_abs_env_cfg:AreumiiCubeStackEnvCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="IsaacContrib-Stack-Cube-Aruemii-IK-Abs-Record",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_abs_record_env_cfg:AreumiiCubeStackRecordEnvCfg",
    },
    disable_env_checker=True,
)
