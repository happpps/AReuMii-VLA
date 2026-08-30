# Areumii Color Cube Pick & Place

This version intentionally reuses the exact rigid cubes from the already-working
Areumii Stack task. No custom sphere/collision setup is used.

## Object mapping

- `cube_1` = red cube
- `cube_2` = yellow cube
- `cube_3` = green cube
- blue basket = fixed

The inherited Stack cube physics/collision are preserved. Only each cube's USD
path is changed to the desired color.

## Destination

Copy this entire folder to:

```text
source/isaaclab_tasks/isaaclab_tasks/contrib/color_pick_place/
```

## Task IDs

- `IsaacContrib-Color-Pick-Place-Areumii-IK-Abs`
- `IsaacContrib-Color-Pick-Place-Areumii-IK-Abs-Record`

## Target selection

Random:

```bash
export AREUMII_TARGET_COLOR=random
```

Fixed:

```bash
export AREUMII_TARGET_COLOR=red
# yellow / green
```

## Language mapping for LeRobot

HDF5 stores `obs/target_color_id`:

- 0 -> `Pick up the red cube and put it in the blue basket.`
- 1 -> `Pick up the yellow cube and put it in the blue basket.`
- 2 -> `Pick up the green cube and put it in the blue basket.`

## First test

Before recording:
1. verify red/yellow/green cubes stay on the table,
2. verify the blue basket position,
3. press/reset and verify cube randomization,
4. verify `TARGET = RED/YELLOW/GREEN` is printed,
5. put only the commanded cube in the basket and verify success.

The IK/XR pipeline is imported from the already-tuned Stack task, so controller
retargeting should remain unchanged.
