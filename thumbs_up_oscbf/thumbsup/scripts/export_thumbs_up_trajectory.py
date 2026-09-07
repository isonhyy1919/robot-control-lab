from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from thumbs_up_from_urdf import (  # noqa: E402
    ARM_JOINTS,
    DEFAULT_URDF,
    apply_qpos,
    clipped_qpos,
    load_urdf_model,
    make_keyframes,
    smoothstep,
)


HAND_JOINTS = [
    "rh_thumb_cmc_roll",
    "rh_thumb_cmc_yaw",
    "rh_thumb_cmc_pitch",
    "rh_thumb_mcp",
    "rh_thumb_ip",
    "rh_index_mcp_roll",
    "rh_index_mcp_pitch",
    "rh_index_pip",
    "rh_index_dip",
    "rh_middle_mcp_pitch",
    "rh_middle_pip",
    "rh_middle_dip",
    "rh_ring_mcp_roll",
    "rh_ring_mcp_pitch",
    "rh_ring_pip",
    "rh_ring_dip",
    "rh_pinky_mcp_roll",
    "rh_pinky_mcp_pitch",
    "rh_pinky_pip",
    "rh_pinky_dip",
]

DEFAULT_OUT = ROOT / "outputs" / "thumbs_up_trajectory.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "thumbs_up_trajectory_summary.txt"


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> tuple[float, float, float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return (float("nan"), float("nan"), float("nan"))
    pos = data.xpos[body_id]
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def named_qpos(model: mujoco.MjModel, qpos: np.ndarray, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return float("nan")
    return float(qpos[model.jnt_qposadr[joint_id]])


def add_sample(
    rows: list[dict[str, float | str]],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    t: float,
    phase: str,
    phase_progress: float,
) -> None:
    apply_qpos(model, data, qpos)
    a7_pos = body_position(model, data, "A7_Link")
    thumb_pos = body_position(model, data, "rh_thumb_distal")
    middle_pos = body_position(model, data, "rh_middle_distal")

    row: dict[str, float | str] = {
        "time_s": round(t, 6),
        "phase": phase,
        "phase_progress": round(phase_progress, 6),
        "A7_x": a7_pos[0],
        "A7_y": a7_pos[1],
        "A7_z": a7_pos[2],
        "thumb_tip_x": thumb_pos[0],
        "thumb_tip_y": thumb_pos[1],
        "thumb_tip_z": thumb_pos[2],
        "middle_tip_x": middle_pos[0],
        "middle_tip_y": middle_pos[1],
        "middle_tip_z": middle_pos[2],
    }
    for joint in ARM_JOINTS + HAND_JOINTS:
        row[joint] = named_qpos(model, qpos, joint)
    rows.append(row)


def sample_segment(
    rows: list[dict[str, float | str]],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    start_qpos: np.ndarray,
    end_qpos: np.ndarray,
    t0: float,
    duration: float,
    fps: int,
    phase: str,
) -> float:
    steps = max(1, int(round(duration * fps)))
    for step in range(1, steps + 1):
        raw_progress = step / steps
        alpha = smoothstep(raw_progress)
        qpos = (1.0 - alpha) * start_qpos + alpha * end_qpos
        t = t0 + step / fps
        add_sample(rows, model, data, qpos, t, phase, raw_progress)
    return t0 + duration


def build_trajectory(model: mujoco.MjModel, fps: int, include_reset: bool) -> list[dict[str, float | str]]:
    data = mujoco.MjData(model)
    keyframes = make_keyframes()
    qpos_frames = [clipped_qpos(model, keyframe.qpos) for keyframe in keyframes]

    rows: list[dict[str, float | str]] = []
    t = 0.0
    add_sample(rows, model, data, qpos_frames[0], t, keyframes[0].name, 0.0)
    for start_qpos, end_qpos, keyframe in zip(qpos_frames, qpos_frames[1:], keyframes[1:]):
        t = sample_segment(rows, model, data, start_qpos, end_qpos, t, keyframe.duration, fps, keyframe.name)

    if include_reset:
        t = sample_segment(
            rows,
            model,
            data,
            qpos_frames[-1],
            qpos_frames[0],
            t,
            1.0,
            fps,
            "reset_to_initial",
        )
    return rows


def write_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_s",
        "phase",
        "phase_progress",
        *ARM_JOINTS,
        *HAND_JOINTS,
        "A7_x",
        "A7_y",
        "A7_z",
        "thumb_tip_x",
        "thumb_tip_y",
        "thumb_tip_z",
        "middle_tip_x",
        "middle_tip_y",
        "middle_tip_z",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, float | str]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    # End of arm raise is the last row in phase 2; final is the last thumb-up row before reset.
    raise_rows = [row for row in rows if row["phase"] == "2/3 raise_arm_pose"]
    thumb_rows = [row for row in rows if row["phase"] == "3/3 curl_four_fingers_thumb_up"]
    raised = raise_rows[-1] if raise_rows else rows[-1]
    final = thumb_rows[-1] if thumb_rows else rows[-1]

    def arm_line(row: dict[str, float | str]) -> str:
        return " ".join(f"{joint}={float(row[joint]):.3f}" for joint in ARM_JOINTS)

    text = [
        "Thumb-up demo trajectory",
        "",
        "Phase 1: initial pose, arm set to its starting joint angles and hand open.",
        f"  {arm_line(first)}",
        "",
        "Phase 2: arm raise, only arm joints interpolate from initial to raised pose.",
        f"  {arm_line(raised)}",
        "",
        "Phase 3: hand gesture, arm stays fixed and the four fingers/thumb interpolate to thumbs-up.",
        f"  thumb_yaw={float(final['rh_thumb_cmc_yaw']):.3f}",
        f"  index_mcp_pitch={float(final['rh_index_mcp_pitch']):.3f}",
        f"  middle_mcp_pitch={float(final['rh_middle_mcp_pitch']):.3f}",
        "",
        "Cartesian check from sampled MuJoCo forward kinematics:",
        f"  A7 initial = ({float(first['A7_x']):.3f}, {float(first['A7_y']):.3f}, {float(first['A7_z']):.3f})",
        f"  A7 raised  = ({float(raised['A7_x']):.3f}, {float(raised['A7_y']):.3f}, {float(raised['A7_z']):.3f})",
        f"  thumb final = ({float(final['thumb_tip_x']):.3f}, {float(final['thumb_tip_y']):.3f}, {float(final['thumb_tip_z']):.3f})",
    ]
    summary_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the current thumbs-up demo trajectory to CSV.")
    parser.add_argument("--urdf", type=str, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--include-reset", action="store_true")
    args = parser.parse_args()

    urdf = Path(args.urdf) if args.urdf else DEFAULT_URDF
    model = load_urdf_model(urdf)
    rows = build_trajectory(model, fps=args.fps, include_reset=args.include_reset)
    write_csv(rows, args.out)
    write_summary(rows, args.summary)
    print(f"csv={args.out}")
    print(f"summary={args.summary}")
    print(f"samples={len(rows)}")


if __name__ == "__main__":
    main()
