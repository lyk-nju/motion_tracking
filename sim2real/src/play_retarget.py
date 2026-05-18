"""Play a retargeted G1 reference NPZ directly in MuJoCo (bypasses policy/DDS).

Usage:
  cd /home/lai/大学/text2motion/motion_tracking
  PYTHONPATH=sim2real/src python sim2real/src/play_retarget.py \
    /path/to/reference_xxx.npz [--loop]
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from paths import ASSETS_DIR

XML_PATH = ASSETS_DIR / "g1" / "g1.xml"

DATASET_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    dof_pos = data["dof_pos"].astype(np.float32)
    root_pos = data["root_pos"].astype(np.float32)
    joint_names_raw = data.get("joint_names", None)
    if joint_names_raw is not None:
        joint_names = []
        for n in joint_names_raw.tolist():
            if isinstance(n, (bytes, np.bytes_)):
                joint_names.append(n.decode("utf-8"))
            else:
                joint_names.append(str(n))
    else:
        joint_names = DATASET_JOINT_ORDER
    print(f"Loaded: {dof_pos.shape[0]} frames, {dof_pos.shape[1]} dofs")
    print(f"Joint names: {joint_names}")
    return {"dof_pos": dof_pos, "root_pos": root_pos, "joint_names": joint_names}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <reference.npz> [--loop]")
        sys.exit(1)

    path = sys.argv[1]
    loop = "--loop" in sys.argv
    data = load_npz(path)
    dof_pos = data["dof_pos"]
    root_pos = data["root_pos"]
    joint_names = data["joint_names"]

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    mjdata = mujoco.MjData(model)

    mj_joint_ids = []
    for name in joint_names:
        try:
            mj_joint_ids.append(model.joint(name).id)
        except KeyError:
            print(f"WARNING: joint '{name}' not found in MuJoCo model")
            mj_joint_ids.append(-1)

    fps = 30
    dt = 1.0 / fps
    frame_idx = [0]

    with mujoco.viewer.launch_passive(model, mjdata) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.lookat = [0, 0, 1.0]
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        last_time = time.perf_counter()

        while viewer.is_running():
            now = time.perf_counter()
            if now - last_time < dt:
                time.sleep(0.001)
                continue
            last_time = now

            f = frame_idx[0]
            if f >= dof_pos.shape[0]:
                if loop:
                    frame_idx[0] = 0
                    f = 0
                else:
                    time.sleep(0.01)
                    continue

            # Set root free joint
            mjdata.qpos[0:3] = root_pos[f]
            mjdata.qpos[3:7] = [1, 0, 0, 0]

            # Set joint positions
            for i, jid in enumerate(mj_joint_ids):
                if jid >= 0:
                    mjdata.qpos[model.jnt_qposadr[jid]] = dof_pos[f, i]

            mujoco.mj_forward(model, mjdata)
            viewer.sync()
            frame_idx[0] += 1

            if frame_idx[0] % 60 == 0:
                print(f"Frame {frame_idx[0]}/{dof_pos.shape[0]}")


if __name__ == "__main__":
    main()
