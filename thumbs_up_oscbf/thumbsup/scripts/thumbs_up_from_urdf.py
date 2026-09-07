from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = PROJECT_ROOT / "models/a7v2_arm_hand/urdf/a7v2_latest_stand_single_arm_right_hand.urdf"

ARM_JOINTS = [f"A{i}_joint" for i in range(1, 8)]
FOUR_FINGERS = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class Keyframe:
    name: str
    duration: float
    qpos: dict[str, float]


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def load_urdf_model(urdf: Path) -> mujoco.MjModel:
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    old_cwd = Path.cwd()
    try:
        os.chdir(urdf.parent)
        return mujoco.MjModel.from_xml_path(urdf.name)
    finally:
        os.chdir(old_cwd)


def joint_qpos_addresses(model: mujoco.MjModel) -> dict[str, int]:
    addresses: dict[str, int] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            addresses[name] = int(model.jnt_qposadr[joint_id])
    return addresses


def joint_ranges(model: mujoco.MjModel) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            lo, hi = model.jnt_range[joint_id]
            ranges[name] = (float(lo), float(hi))
    return ranges


def clipped_qpos(model: mujoco.MjModel, targets: dict[str, float]) -> np.ndarray:
    ranges = joint_ranges(model)
    qpos = np.zeros(model.nq, dtype=np.float64)
    for name, value in targets.items():
        if name not in ranges:
            raise KeyError(f"Joint not found in URDF model: {name}")
        qpos[joint_qpos_addresses(model)[name]] = value
    return qpos


def open_hand() -> dict[str, float]:
    hand = {
        "rh_thumb_cmc_roll": 0.0,
        "rh_thumb_cmc_yaw": 0.0,
        "rh_thumb_cmc_pitch": 0.0,
        "rh_thumb_mcp": 0.0,
        "rh_thumb_ip": 0.0,
    }
    for finger in FOUR_FINGERS:
        if finger != "middle":
            hand[f"rh_{finger}_mcp_roll"] = 0.0
        hand[f"rh_{finger}_mcp_pitch"] = 0.0
        hand[f"rh_{finger}_pip"] = 0.0
        hand[f"rh_{finger}_dip"] = 0.0
    return hand


def thumbs_up_hand() -> dict[str, float]:
    hand = {
        # Thumb yaw opens it away from the palm, while the bend joints stay straight.
        "rh_thumb_cmc_roll": 0.0,
        "rh_thumb_cmc_yaw": 0.64,
        "rh_thumb_cmc_pitch": 0.0,
        "rh_thumb_mcp": 0.0,
        "rh_thumb_ip": 0.0,
        "rh_index_mcp_roll": 0.06,
        "rh_ring_mcp_roll": 0.08,
        "rh_pinky_mcp_roll": 0.16,
    }
    for finger in FOUR_FINGERS:
        hand[f"rh_{finger}_mcp_pitch"] = 1.18
        hand[f"rh_{finger}_pip"] = 1.58
        hand[f"rh_{finger}_dip"] = 1.35 if finger == "index" else 0.55
    return hand


def make_keyframes() -> list[Keyframe]:
    vertical_arm = dict(zip(ARM_JOINTS, [0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # Keep A4_joint straight while the arm raises.
    raised_x_arm = dict(zip(ARM_JOINTS, [-1.00, 1.50, 1.20, 0.00, 0.00, 0.00, 0.00]))

    return [
        Keyframe("1/3 vertical_arm_open_hand", 1.2, {**vertical_arm, **open_hand()}),
        Keyframe("2/3 raise_arm_pose", 2.2, {**raised_x_arm, **open_hand()}),
        Keyframe("3/3 curl_four_fingers_thumb_up", 2.0, {**raised_x_arm, **thumbs_up_hand()}),
        Keyframe("hold_final_pose", 2.5, {**raised_x_arm, **thumbs_up_hand()}),
    ]


def apply_qpos(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> None:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def hold_initial_pose(model: mujoco.MjModel, realtime: bool, viewer: bool) -> None:
    hold_arm_pose(model, realtime, viewer, [0.0] * len(ARM_JOINTS), "initial_arm_zero_pose")


def hold_arm_pose(
    model: mujoco.MjModel,
    realtime: bool,
    viewer: bool,
    arm_values: list[float],
    label: str = "custom_arm_pose",
) -> None:
    data = mujoco.MjData(model)
    pose = {**open_hand(), **dict(zip(ARM_JOINTS, arm_values))}
    qpos = clipped_qpos(model, pose)
    apply_qpos(model, data, qpos)
    print(f"{label}:")
    print(" ".join(f"{joint}={value:.2f}" for joint, value in zip(ARM_JOINTS, arm_values)))

    if viewer:
        with mujoco.viewer.launch_passive(model, data) as handle:
            handle.cam.azimuth = -135
            handle.cam.elevation = -18
            handle.cam.distance = 1.35
            handle.cam.lookat[:] = np.array([0.40, -0.62, 1.20])
            while handle.is_running():
                apply_qpos(model, data, qpos)
                handle.sync()
                if realtime:
                    time.sleep(1.0 / 60.0)


def run_demo(model: mujoco.MjModel, realtime: bool, viewer: bool, repeat: bool) -> None:
    data = mujoco.MjData(model)
    keyframes = make_keyframes()
    frames = [clipped_qpos(model, frame.qpos) for frame in keyframes]
    apply_qpos(model, data, frames[0])

    fps = 60.0
    dt = 1.0 / fps

    def play_segment(
        start_qpos: np.ndarray,
        end_qpos: np.ndarray,
        duration: float,
        label: str,
        sync=None,
        is_running=None,
    ) -> bool:
        print(label)
        steps = max(1, int(round(duration * fps)))
        start_time = time.perf_counter()
        for step in range(steps):
            if is_running is not None and not is_running():
                return False
            alpha = smoothstep((step + 1) / steps)
            apply_qpos(model, data, (1.0 - alpha) * start_qpos + alpha * end_qpos)
            if sync is not None:
                sync()
            if realtime:
                target_time = (step + 1) * dt
                elapsed = time.perf_counter() - start_time
                if target_time > elapsed:
                    time.sleep(target_time - elapsed)
        return True

    if viewer:
        with mujoco.viewer.launch_passive(model, data) as handle:
            handle.cam.azimuth = -135
            handle.cam.elevation = -18
            handle.cam.distance = 1.35
            handle.cam.lookat[:] = np.array([0.58, -0.52, 0.90])
            while handle.is_running():
                for prev, nxt, frame in zip(frames, frames[1:], keyframes[1:]):
                    if not play_segment(prev, nxt, frame.duration, frame.name, handle.sync, handle.is_running):
                        break
                if not repeat or not handle.is_running():
                    break
                play_segment(frames[-1], frames[0], 1.0, "reset_to_vertical_open_hand", handle.sync, handle.is_running)
            while not repeat and handle.is_running():
                apply_qpos(model, data, frames[-1])
                handle.sync()
                if realtime:
                    time.sleep(dt)
    else:
        loops = 2 if repeat else 1
        for loop in range(loops):
            if repeat:
                print(f"loop {loop + 1}/{loops}")
            for prev, nxt, frame in zip(frames, frames[1:], keyframes[1:]):
                play_segment(prev, nxt, frame.duration, frame.name)
            if repeat and loop + 1 < loops:
                play_segment(frames[-1], frames[0], 1.0, "reset_to_vertical_open_hand")


def print_pose_check(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    for frame in make_keyframes()[:2]:
        qpos = clipped_qpos(model, frame.qpos)
        apply_qpos(model, data, qpos)
        a1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A1_Link")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A7_Link")
        arm_angles = " ".join(f"{joint}={frame.qpos.get(joint, 0.0):.2f}" for joint in ARM_JOINTS)
        elbow = frame.qpos.get("A4_joint", 0.0)
        if a1_id >= 0 and body_id >= 0:
            a1 = data.xpos[a1_id]
            pos = data.xpos[body_id]
            print(
                f"{frame.name}: "
                f"{arm_angles} "
                f"A1_Link xyz=({a1[0]:.3f}, {a1[1]:.3f}, {a1[2]:.3f}) "
                f"A7_Link xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                f"A4_elbow={elbow:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the specified URDF directly in MuJoCo and show a +X arm raise with a thumbs-up hand pose."
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="URDF file to load directly.")
    parser.add_argument("--no-viewer", action="store_true", help="Run the motion headlessly for validation.")
    parser.add_argument("--fast", action="store_true", help="Skip realtime sleeps.")
    parser.add_argument("--print-pose-check", action="store_true", help="Print A7_Link positions for keyframes.")
    parser.add_argument("--repeat", action="store_true", help="Loop the motion. In viewer mode it repeats until closed.")
    parser.add_argument("--hold-initial", action="store_true", help="Show only the initial all-zero arm pose.")
    parser.add_argument(
        "--hold-arm-pose",
        nargs=7,
        type=float,
        metavar=("A1", "A2", "A3", "A4", "A5", "A6", "A7"),
        help="Show only one frame with the provided A1-A7 joint angles.",
    )
    args = parser.parse_args()

    model = load_urdf_model(args.urdf)
    if args.print_pose_check:
        print_pose_check(model)
    if args.hold_arm_pose is not None:
        hold_arm_pose(model, realtime=not args.fast, viewer=not args.no_viewer, arm_values=args.hold_arm_pose)
        return
    if args.hold_initial:
        hold_initial_pose(model, realtime=not args.fast, viewer=not args.no_viewer)
        return
    run_demo(model, realtime=not args.fast, viewer=not args.no_viewer, repeat=args.repeat)


if __name__ == "__main__":
    main()
