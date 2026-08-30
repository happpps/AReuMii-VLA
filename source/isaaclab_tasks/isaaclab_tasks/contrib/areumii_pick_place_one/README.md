# Areumii one-cube Pick & Place — corrected architecture

This version fixes the architecture confusion:

- `pick_place_env_cfg.py`
  - **inherits** `stack_joint_pos_env_cfg.AreumiiCubeStackEnvCfg`
  - reuses the working robot/table/camera/reset base
  - changes the task to one cube + blue target

- `pick_place_ik_abs_env_cfg.py`
  - **does not import or inherit `stack_ik_abs_env_cfg`**
  - subclasses the local `AreumiiPickPlaceEnvCfg`
  - contains a local copy of the user's known-good bimanual IsaacTeleop pipeline

Control settings preserved from the known-good IK:
- right/left heading ref = 90 deg
- target pitch = -10 deg
- target yaw = -90 deg
- XYZW -> WXYZ action reorder
- action order = right 7 + right grip 1 + left 7 + left grip 1
- XR anchor = (-0.1, 0.0, -1.7)
- target frame = robot base_link

Task:
> Pick up the cube and place it on the blue target.

Install this folder as:

```text
source/isaaclab_tasks/isaaclab_tasks/contrib/areumii_pick_place_one/
```

Test:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task IsaacContrib-Pick-Place-Areumii-IK-Abs \
  --viz kit \
  --num_envs 1 \
  --xr
```

Record:

```bash
./isaaclab.sh -p scripts/record_demos_areumii_standalone.py \
  --task IsaacContrib-Pick-Place-Areumii-IK-Abs-Record \
  --viz kit \
  --xr \
  --device cuda:0 \
  --dataset_file ./datasets/areumii_pick_place_v1.hdf5 \
  --num_demos 50
```
