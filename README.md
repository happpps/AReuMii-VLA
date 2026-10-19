# AReuMii-VLA

Dual-arm robot manipulation project for **AReuMii**, integrating simulation, XR teleoperation, imitation learning, and real-robot control.

This repository is based on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) and contains AReuMii-specific environments, dataset tools, VLA inference pipelines, and real-robot teleoperation code.

---

## Overview

The current pipeline covers:

- AReuMii dual-arm simulation in Isaac Lab
- XR teleoperation using NVIDIA Isaac Teleop
- Demonstration recording and LeRobot dataset conversion
- SmolVLA / ACT policy inference
- Simulation evaluation
- Real-robot joint teleoperation
- UDP / shared-memory communication with the CAN interface

---

## AReuMii Code Structure

Most AReuMii-specific code can be found in the following directories.

### Simulation

Robot configuration and assets:

```text
source/isaaclab_assets/isaaclab_assets/robots/
├── areumii_c1.py
└── areumii_c1/
    ├── areumii_c1.urdf
    └── areumii_c1_realcolor.urdf
```

Isaac Lab environments:

```text
source/isaaclab_tasks/isaaclab_tasks/contrib/
├── areumii_pick_place/
├── areumii_pick_place_one/
└── stack/config/areumii_c1/
```

### Dataset Collection and Conversion

```text
scripts/areumii_dataset/
```

Main recording script:

```text
scripts/areumii_dataset/record_demos_areumii.py
```

### VLA / Imitation Learning Inference

```text
scripts/areumii_lerobot_inference/
├── smolvla/
└── act/
```

### Real Robot Teleoperation

Standalone XR teleoperation, URDF configuration, UDP bridges, and Docker setup
are maintained in [AReuMii-Teleop](https://github.com/happpps/AReuMii-Teleop).
Use that repository for ongoing real-robot teleoperation development.
The local `real_teleop/` directory retains the existing integration scripts.

```text
real_teleop/
```

Main teleoperation script:

```text
real_teleop/areumii_bimanual_teleop_gripper.py
```

Real robot pipeline:

```text
XR Controller
     ↓
AReuMii Teleop (Python)
     ↓
UDP / Shared Memory
     ↓
CAN Bridge (C++)
     ↓
Motor Controllers
     ↓
Real AReuMii
```

### Evaluation Results

```text
results/
```

Large datasets, HDF5 demonstrations, model checkpoints, and training outputs are not stored in this repository.

---

## Dependencies

- NVIDIA Isaac Lab
- NVIDIA Isaac Sim
- NVIDIA Isaac Teleop
- Hugging Face LeRobot
- PyTorch
- OpenXR / Meta Quest

---

## Robot Assets

The AReuMii URDF files are included in this repository.

The generated Isaac Sim USD asset is not included.
The simulation expects:

```text
source/isaaclab_assets/isaaclab_assets/robots/areumii_c1/areumii_c1_realcolor.usd
```

Generate/import the USD from the provided URDF before running the AReuMii environments.

---

## Third-Party Code

Some files under:

```text
real_teleop/isaac_teleop/
```

are derived from NVIDIA Isaac Teleop / Hugging Face LeRobot integration code and retain their original copyright notices.

---

## Current Status

- [x] AReuMii dual-arm Isaac Lab environment
- [x] XR teleoperation in simulation
- [x] Demonstration recording
- [x] LeRobot dataset conversion
- [x] SmolVLA inference in simulation
- [x] ACT inference experiments
- [x] Real-robot XR joint teleoperation
- [x] Gripper teleoperation
- [ ] VLA inference on the real robot

---

## Upstream

- NVIDIA Isaac Lab
  https://github.com/isaac-sim/IsaacLab

- NVIDIA Isaac Teleop
  https://github.com/NVIDIA/IsaacTeleop

- Hugging Face LeRobot
  https://github.com/huggingface/lerobot
