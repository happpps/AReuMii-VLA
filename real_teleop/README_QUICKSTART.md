# Areumii XR Real Teleop — minimal right-arm test

This is a minimal first-hardware-test port of the official LeRobot
`examples/isaac_teleop_to_so101` XR-controller architecture.

It does **not** use Isaac Sim or ROS in the real-robot command path:

```text
Quest / CloudXR
  -> XRController
  -> clutch (squeeze = deadman)
  -> Areumii URDF IK
  -> q_target [rad]
  -> localhost UDP
  -> areumii_udp_shm_bridge
  -> existing Areum2_can shared memory
  -> existing Areum2_can / SocketCAN
  -> right-arm motors
```

## Scope

- Right arm only: 7 DOF
- No gripper yet
- q commands are radians
- Dry-run by default
- Position-only IK by default (`--orientation-weight 0.0`)
- Conservative EE and joint step limits

## 0. Prerequisites

Use a **LeRobot source checkout** because the official Isaac Teleop -> SO101 example
lives in `examples/` and is not included in the normal pip package.

From the LeRobot repo root, the official example currently documents:

```bash
uv pip install -e ".[kinematics]"
uv pip install "isaacteleop[cloudxr,retargeters-lite]~=1.3.131" "scipy>=1.14"
python -m isaacteleop.cloudxr --accept-eula
```

You also need the Areumii URDF and the existing `Areum2_can` repo.

## 1. Build the UDP -> shared-memory bridge

Copy `areumii_udp_shm_bridge.cpp` into the **Areum2_can repo root**, then:

```bash
g++ -O2 -std=c++17 -I./inc \
  areumii_udp_shm_bridge.cpp \
  -o areumii_udp_shm_bridge
```

## 2. Tonight: XR + IK dry-run

Use a known-valid right-arm pose in radians:

```bash
cd <lerobot_source_root>

python /path/to/areumii_real_teleop.py \
  --urdf /path/to/areumii.urdf \
  --initial-q '0,0,0,0,0,0,0'
```

Do **not** add `--enable-command` tonight unless you intentionally want real commands.
Replace the example zeros with a pose valid for your URDF.

Expected:
1. CloudXR launches.
2. Connect Quest.
3. XR tracking is detected.
4. Hold squeeze to engage.
5. Moving the controller prints 7D `q_target` in radians.

## 3. Tomorrow: hardware process order

Terminal 1 — existing CAN program:

```bash
cd <Areum2_can>
./start.sh
```

Confirm feedback looks sane first.

Terminal 2 — bridge:

```bash
cd <Areum2_can>
./areumii_udp_shm_bridge
```

Terminal 3 — teleop:

```bash
cd <lerobot_source_root>
python /path/to/areumii_real_teleop.py \
  --urdf /path/to/areumii.urdf \
  --enable-command
```

The Python script refuses clutch engagement if joint feedback is stale.

## 4. First physical test

Before squeezing:
- Have hardware power cutoff / E-stop immediately reachable.
- Verify measured q values look plausible and match the physical arm.
- Verify URDF joint zero/sign conventions against the real arm.
- Keep the arm clear of table/self-collision.

Then:
1. Hold controller still.
2. Squeeze to engage.
3. The clutch anchors to the **measured physical EE pose**.
4. Move only a few millimeters.
5. Release squeeze immediately if direction is wrong.

Conservative defaults:

```text
max EE translation step: 0.005 m/frame
max joint step:          0.02 rad/frame
absolute command guard:  +/-2 rad
```

## 5. Orientation

First test defaults to position-only IK:

```bash
--orientation-weight 0.0
```

After position mapping is verified, try a small value such as:

```bash
--orientation-weight 0.02
```

The engage-relative clutch preserves the current measured EE pose on engage, so this
real-robot path does not need the same absolute controller orientation offset used by the
current IsaacLab sim retargeter just to avoid an initial wrist jump.

## 6. CloudXR headset picker

The official SO101 real example does not route commands through ROS/IsaacROS. Its
`XRController` launches/uses CloudXR directly. This Areumii port does the same, with
shared-memory/CAN as the robot backend.

If you already have a Quest CloudXR `.env` that works, pass it:

```bash
--cloudxr-env /path/to/profile.env
```

Otherwise `XRControllerConfig` uses Isaac Teleop's default WebRTC input profile.

## TODO after first success

- Left arm
- Gripper rotary-to-linear mapping
- Camera observation / LeRobot dataset recording
- Robot-specific joint limits
- Confirm actual joint sign/offset conventions
- Tune orientation and workspace bounds
