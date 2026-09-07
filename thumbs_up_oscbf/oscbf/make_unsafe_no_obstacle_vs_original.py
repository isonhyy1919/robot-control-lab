from __future__ import annotations

import csv
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from run_oscbf_trajectory import (
    ARM_JOINTS,
    DEFAULT_TRAJECTORY,
    DEFAULT_URDF,
    Obstacle,
    Trajectory,
    build_index,
    load_model_with_obstacle,
    set_state,
)


ROOT = Path(__file__).resolve().parent
SOURCE_RECORDS = ROOT / "outputs" / "oscbf_thumbsup" / "nominal_rollout.csv"
ORIGINAL_VIDEO = ROOT / "outputs" / "oscbf_thumbsup" / "oscbf_obstacle_avoidance.mp4"
OUTPUT_DIR = ROOT / "outputs" / "a4_joint_limit_comparison"
LEFT_VIDEO = OUTPUT_DIR / "no_obstacle_unsafe_a4_limit_violation.mp4"
COMPARISON_VIDEO = OUTPUT_DIR / "no_obstacle_unsafe_vs_original_obstacle.mp4"
CONTACT_SHEET = OUTPUT_DIR / "no_obstacle_unsafe_vs_original_contact_sheet.png"
OBSTACLE = Obstacle(np.asarray([0.70, -0.315, 0.82]), 0.105, 0.025)
FPS = 30


def load_records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise RuntimeError(f"No records in {path}")
    return records


def make_legacy_render_trajectory() -> Trajectory:
    """Reconstruct the old 7.7 s hand timing and final reset for rendering."""
    current = Trajectory.load(DEFAULT_TRAJECTORY)
    times = np.arange(0.0, 7.7000001, 0.005)
    q_values = []
    phases = []
    initial_q, _, _ = current.sample(0.0)
    final_q, _, _ = current.sample(float(current.time[-1]))
    reset_start = 6.735
    reset_end = 7.7
    for time_s in times:
        if time_s <= current.time[-1]:
            q, _, phase = current.sample(float(time_s))
        elif time_s < reset_start:
            q = final_q.copy()
            phase = "hold_final_pose"
        else:
            progress = np.clip((time_s - reset_start) / (reset_end - reset_start), 0.0, 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            q = (1.0 - smooth) * final_q + smooth * initial_q
            phase = "reset_to_initial"
        q_values.append(q)
        phases.append(phase)
    q_array = np.asarray(q_values)
    qd_array = np.gradient(q_array, times, axis=0, edge_order=2)
    return Trajectory(times, phases, list(current.joint_names), q_array, qd_array)


def hide_obstacle_and_cbf_visuals(model: mujoco.MjModel, index) -> None:
    for site_id, _, _ in index.collision_sites:
        model.site_rgba[site_id, 3] = 0.0
    for geom_name in ("cbf_obstacle_safety_zone", "cbf_obstacle_physical"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            model.geom_rgba[geom_id, 3] = 0.0


def record_unsafe_left_video(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    records: list[dict[str, str]],
) -> None:
    index = build_index(model)
    hide_obstacle_and_cbf_visuals(model, index)
    data = mujoco.MjData(model)
    record_times = np.asarray([float(row["time_s"]) for row in records])
    arm_history = np.asarray([
        [float(row[name]) for name in ARM_JOINTS] for row in records
    ])
    duration = float(record_times[-1] - record_times[0])
    frames = int(round(duration * FPS)) + 1

    camera = mujoco.MjvCamera()
    camera.azimuth = -130
    camera.elevation = -18
    camera.distance = 1.45
    camera.lookat[:] = np.asarray([0.68, -0.35, 0.82])
    renderer = mujoco.Renderer(model, height=480, width=640)
    writer = imageio.get_writer(
        str(LEFT_VIDEO), fps=FPS, codec="libx264", quality=8, macro_block_size=None
    )
    try:
        for frame_index in range(frames):
            local_t = min(frame_index / FPS, duration)
            arm_q = np.asarray([
                np.interp(local_t, record_times, arm_history[:, joint_index])
                for joint_index in range(len(ARM_JOINTS))
            ])
            desired_all, _, phase = trajectory.sample(local_t)
            set_state(model, data, index, trajectory, arm_q, desired_all)
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame, "RGBA")
            a4 = float(arm_q[3])
            violating = a4 < -1e-6
            draw.rounded_rectangle((12, 12, 468, 88), radius=8, fill=(0, 0, 0, 165))
            draw.text(
                (24, 22), "No obstacle | OSCBF OFF | unsafe trajectory executed",
                fill=(255, 255, 255, 255),
            )
            draw.text(
                (24, 48),
                f"A4 actual = {a4:+.3f} rad   URDF limit = [0.0, 2.2] rad",
                fill=(255, 225, 120, 255) if violating else (220, 240, 255, 255),
            )
            if violating:
                draw.rounded_rectangle((342, 61, 458, 83), radius=5, fill=(210, 20, 35, 235))
                draw.text((352, 65), "LIMIT VIOLATION", fill=(255, 255, 255, 255))
            draw.text(
                (24, 68), f"{local_t:4.1f}/{duration:.1f} s   {phase}",
                fill=(230, 238, 248, 255),
            )
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
        renderer.close()


def compose_with_original(left_path: Path, right_path: Path) -> None:
    left_reader = imageio.get_reader(left_path)
    right_reader = imageio.get_reader(right_path)
    left_meta = left_reader.get_meta_data()
    right_meta = right_reader.get_meta_data()
    if float(left_meta["fps"]) != float(right_meta["fps"]):
        raise RuntimeError("Video frame rates do not match")
    frames = min(left_reader.count_frames(), right_reader.count_frames())
    writer = imageio.get_writer(
        str(COMPARISON_VIDEO), fps=FPS, codec="libx264", quality=8,
        macro_block_size=None,
    )
    try:
        for frame_index in range(frames):
            left = Image.fromarray(left_reader.get_data(frame_index)).convert("RGB")
            right = Image.fromarray(right_reader.get_data(frame_index)).convert("RGB")
            if right.size != left.size:
                right = right.resize(left.size, Image.Resampling.LANCZOS)
            width, height = left.size
            canvas = Image.new("RGB", (width * 2, height + 50), (18, 22, 30))
            canvas.paste(left, (0, 50))
            canvas.paste(right, (width, 50))
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.rectangle((0, 0, width, 50), fill=(174, 36, 52, 255))
            draw.rectangle((width, 0, width * 2, 50), fill=(28, 104, 153, 255))
            draw.text((18, 16), "LEFT  No obstacle | OSCBF OFF | A4 limit violation", fill="white")
            draw.text((width + 18, 16), "RIGHT  Original obstacle + OSCBF video (unchanged)", fill="white")
            draw.rectangle((width - 1, 0, width + 1, height + 50), fill=(255, 255, 255, 220))
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        left_reader.close()
        right_reader.close()


def create_contact_sheet() -> None:
    reader = imageio.get_reader(COMPARISON_VIDEO)
    metadata = reader.get_meta_data()
    frame_count = reader.count_frames()
    times = (0.0, 1.2, 2.0, 5.5, 7.7)
    frames = []
    for time_s in times:
        index = min(round(time_s * float(metadata["fps"])), frame_count - 1)
        frames.append(Image.fromarray(reader.get_data(index)).convert("RGB"))
    reader.close()
    sheet = Image.new("RGB", (1280 * len(frames), 530), "white")
    for index, frame in enumerate(frames):
        sheet.paste(frame, (1280 * index, 0))
    sheet.save(CONTACT_SHEET)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(SOURCE_RECORDS)
    minimum_a4 = min(float(row["A4_joint"]) for row in records)
    if minimum_a4 >= 0.0:
        raise RuntimeError("Source nominal rollout does not contain an A4 limit violation")
    model = load_model_with_obstacle(DEFAULT_URDF, OBSTACLE)
    render_trajectory = make_legacy_render_trajectory()
    record_unsafe_left_video(model, render_trajectory, records)
    compose_with_original(LEFT_VIDEO, ORIGINAL_VIDEO)
    create_contact_sheet()
    print(
        {
            "left_video": str(LEFT_VIDEO),
            "comparison_video": str(COMPARISON_VIDEO),
            "right_video": str(ORIGINAL_VIDEO),
            "minimum_executed_a4_rad": minimum_a4,
            "urdf_a4_lower_limit_rad": 0.0,
            "duration_s": float(records[-1]["time_s"]),
        }
    )


if __name__ == "__main__":
    main()
