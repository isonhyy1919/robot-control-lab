from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_oscbf_trajectory import (
    ARM_JOINTS,
    DEFAULT_TRAJECTORY,
    DEFAULT_URDF,
    Obstacle,
    Trajectory,
    build_index,
    load_model_with_obstacle,
    percentile,
    record_rollout_video,
    run_rollout,
    summarize,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "a4_joint_limit_comparison"
REFERENCE_VIDEO = ROOT / "outputs" / "oscbf_thumbsup" / "oscbf_obstacle_avoidance.mp4"
OBSTACLE = Obstacle(np.asarray([0.70, -0.315, 0.82]), 0.105, 0.025)
DT = 0.005
ALPHA = 10.0
TRACKING_GAIN = 7.0
JOINT_MARGIN = 0.0
VELOCITY_SCALE = 0.25
VIDEO_FPS = 30


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path(r"C:\Windows\Fonts") / filename
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def draw_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x: np.ndarray,
    series: list[tuple[str, np.ndarray, tuple[int, int, int], int]],
    horizontal_lines: list[tuple[float, tuple[int, int, int]]] | None = None,
) -> None:
    left, top, right, bottom = box
    plot = (left + 82, top + 52, right - 24, bottom - 55)
    px0, py0, px1, py1 = plot
    all_y = [np.asarray(item[1], dtype=float) for item in series]
    if horizontal_lines:
        all_y.extend(np.asarray([value]) for value, _ in horizontal_lines)
    y_min = min(float(np.min(item)) for item in all_y)
    y_max = max(float(np.max(item)) for item in all_y)
    y_pad = max((y_max - y_min) * 0.10, 0.02)
    y_min -= y_pad
    y_max += y_pad
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    def map_x(value: float) -> int:
        return int(px0 + (value - x_min) / max(x_max - x_min, 1e-12) * (px1 - px0))

    def map_y(value: float) -> int:
        return int(py1 - (value - y_min) / max(y_max - y_min, 1e-12) * (py1 - py0))

    draw.rounded_rectangle(box, radius=12, fill=(250, 252, 255), outline=(195, 204, 216), width=2)
    draw.text((left + 18, top + 14), title, font=font(23, True), fill=(25, 38, 55))
    for tick in range(6):
        x_value = x_min + (x_max - x_min) * tick / 5
        x_pixel = map_x(x_value)
        draw.line((x_pixel, py0, x_pixel, py1), fill=(225, 230, 238), width=1)
        draw.text((x_pixel - 15, py1 + 12), f"{x_value:.1f}", font=font(15), fill=(70, 78, 90))
    for tick in range(5):
        y_value = y_min + (y_max - y_min) * tick / 4
        y_pixel = map_y(y_value)
        draw.line((px0, y_pixel, px1, y_pixel), fill=(225, 230, 238), width=1)
        draw.text((left + 7, y_pixel - 8), f"{y_value:.2f}", font=font(14), fill=(70, 78, 90))
    draw.line((px0, py0, px0, py1, px1, py1), fill=(60, 68, 80), width=2)
    if horizontal_lines:
        for value, color in horizontal_lines:
            y_pixel = map_y(value)
            for start in range(px0, px1, 18):
                draw.line((start, y_pixel, min(start + 9, px1), y_pixel), fill=color, width=2)
    for label, y, color, width in series:
        points = [(map_x(float(xv)), map_y(float(yv))) for xv, yv in zip(x, y)]
        draw.line(points, fill=color, width=width, joint="curve")
    legend_x = px0 + 10
    legend_y = py0 + 8
    for label, _, color, width in series:
        draw.line((legend_x, legend_y + 8, legend_x + 28, legend_y + 8), fill=color, width=max(width, 3))
        draw.text((legend_x + 36, legend_y), label, font=font(14), fill=(45, 52, 63))
        legend_y += 21


def make_a4_violating_trajectory(
    source: Trajectory, lower: np.ndarray, target_rad: float = -0.5
) -> Trajectory:
    """Inject a smooth A4 violation while preserving all other references."""
    q = source.q.copy()
    a1_column = source.joint_names.index("A1_joint")
    a4_column = source.joint_names.index("A4_joint")
    a1_motion = np.abs(q[:, a1_column] - q[0, a1_column])
    motion_scale = a1_motion / max(float(np.max(a1_motion)), 1e-12)
    q[:, a4_column] = target_rad * np.clip(motion_scale, 0.0, 1.0)
    if float(np.min(q[:, a4_column])) >= float(lower[3]):
        raise RuntimeError("Injected A4 trajectory did not violate the URDF lower limit")
    qd = np.gradient(q, source.time, axis=0, edge_order=2)
    return Trajectory(source.time.copy(), list(source.phase), list(source.joint_names), q, qd)


def values(records: list[dict[str, float | str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in records])


def scenario_metrics(
    trajectory: Trajectory,
    records: list[dict[str, float | str]],
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, object]:
    a4_column = trajectory.joint_names.index("A4_joint")
    a4_desired = trajectory.q[:, a4_column]
    nominal = values(records, "A4_joint_nominal_velocity")
    command = values(records, "A4_joint_command_velocity")
    arm_intervention = np.column_stack([
        values(records, f"{name}_command_velocity")
        - values(records, f"{name}_nominal_velocity")
        for name in ARM_JOINTS
    ])
    result = summarize(records)
    result.update(
        {
            "upstream_a4_min_rad": float(np.min(a4_desired)),
            "upstream_a4_max_rad": float(np.max(a4_desired)),
            "executed_a4_min_rad": float(np.min(values(records, "A4_joint"))),
            "executed_a4_max_rad": float(np.max(values(records, "A4_joint"))),
            "a4_lower_limit_rad": float(lower[3]),
            "a4_upper_limit_rad": float(upper[3]),
            "a4_nominal_velocity_min_rad_s": float(np.min(nominal)),
            "a4_safe_velocity_min_rad_s": float(np.min(command)),
            "a4_max_velocity_intervention_rad_s": float(np.max(np.abs(command - nominal))),
            "mean_arm_velocity_intervention_norm_rad_s": float(
                np.mean(np.linalg.norm(arm_intervention, axis=1))
            ),
            "p95_arm_velocity_intervention_norm_rad_s": percentile(
                np.linalg.norm(arm_intervention, axis=1), 95
            ),
            "a4_lower_cbf_active_steps_1e-4": int(
                np.sum(values(records, "A4_joint_lower_h") < 1e-4)
            ),
        }
    )
    return result


def comparison_metrics(
    valid: list[dict[str, float | str]],
    violating: list[dict[str, float | str]],
) -> dict[str, object]:
    joint_difference = np.column_stack([
        values(violating, name) - values(valid, name) for name in ARM_JOINTS
    ])
    ee_valid = np.column_stack([values(valid, f"A7_{axis}") for axis in "xyz"])
    ee_violating = np.column_stack([values(violating, f"A7_{axis}") for axis in "xyz"])
    ee_difference = np.linalg.norm(ee_violating - ee_valid, axis=1)
    return {
        "maximum_per_joint_output_difference_rad": {
            name: float(np.max(np.abs(joint_difference[:, i])))
            for i, name in enumerate(ARM_JOINTS)
        },
        "maximum_whole_arm_output_difference_norm_rad": float(
            np.max(np.linalg.norm(joint_difference, axis=1))
        ),
        "maximum_end_effector_position_difference_m": float(np.max(ee_difference)),
        "final_end_effector_position_difference_m": float(ee_difference[-1]),
    }


def plot_joint_trajectories(
    valid: list[dict[str, float | str]],
    violating: list[dict[str, float | str]],
    lower: np.ndarray,
    upper: np.ndarray,
    output_path: Path,
) -> None:
    time_s = values(valid, "time_s")
    image = Image.new("RGB", (2400, 1900), (238, 242, 247))
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "Obstacle-present OSCBF: valid vs A4-limit-violating upstream command", font=font(34, True), fill=(20, 35, 55))
    valid_color = (22, 119, 184)
    violating_color = (209, 73, 91)
    for joint_index, joint_name in enumerate(ARM_JOINTS):
        row, column = divmod(joint_index, 2)
        box = (60 + column * 1180, 100 + row * 445, 1160 + column * 1180, 510 + row * 445)
        series = [
            ("valid upstream -> OSCBF", values(valid, joint_name), valid_color, 4),
            ("violating upstream -> OSCBF", values(violating, joint_name), violating_color, 4),
        ]
        if joint_name == "A4_joint":
            series.extend([
                ("valid desired", values(valid, f"{joint_name}_desired"), (70, 160, 92), 3),
                ("violating desired", values(violating, f"{joint_name}_desired"), (125, 35, 56), 3),
            ])
        draw_chart(
            draw, box, f"{joint_name} position [rad]", time_s, series,
            [(float(lower[joint_index]), (25, 25, 25)), (float(upper[joint_index]), (25, 25, 25))],
        )
    box = (1240, 1435, 2340, 1845)
    draw_chart(
        draw, box, "A7 end-effector Z position [m]",
        time_s,
        [
            ("valid upstream", values(valid, "A7_z"), valid_color, 4),
            ("A4 violating upstream", values(violating, "A7_z"), violating_color, 4),
        ],
    )
    image.save(output_path)


def plot_intervention(
    valid: list[dict[str, float | str]],
    violating: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    time_s = values(valid, "time_s")
    image = Image.new("RGB", (2200, 1300), (238, 242, 247))
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "How OSCBF reacts to the A4 URDF-limit violation", font=font(34, True), fill=(20, 35, 55))
    boxes = [(60, 100, 1070, 660), (1130, 100, 2140, 660), (60, 710, 1070, 1260), (1130, 710, 2140, 1260)]
    draw_chart(draw, boxes[0], "A4 position: unsafe request is not executed", time_s, [
        ("violating desired", values(violating, "A4_joint_desired"), (125, 35, 56), 4),
        ("executed after OSCBF", values(violating, "A4_joint"), (209, 73, 91), 4),
        ("valid desired", values(valid, "A4_joint_desired"), (70, 160, 92), 3),
    ], [(0.0, (20, 20, 20))])
    draw_chart(draw, boxes[1], "A4 velocity filtering at the lower boundary", time_s, [
        ("nominal velocity", values(violating, "A4_joint_nominal_velocity"), (125, 35, 56), 4),
        ("OSCBF command", values(violating, "A4_joint_command_velocity"), (209, 73, 91), 4),
    ], [(0.0, (20, 20, 20))])
    draw_chart(draw, boxes[2], "Minimum URDF joint-limit barrier", time_s, [
        ("valid upstream", values(valid, "min_joint_limit_h"), (22, 119, 184), 4),
        ("violating upstream", values(violating, "min_joint_limit_h"), (209, 73, 91), 4),
    ], [(0.0, (20, 20, 20))])
    draw_chart(draw, boxes[3], "Minimum obstacle barrier (obstacle in both)", time_s, [
        ("valid upstream", values(valid, "min_obstacle_h"), (22, 119, 184), 4),
        ("violating upstream", values(violating, "min_obstacle_h"), (209, 73, 91), 4),
    ], [(0.0, (20, 20, 20))])
    image.save(output_path)


def make_side_by_side_video(left_path: Path, right_path: Path, output_path: Path) -> None:
    left_reader = imageio.get_reader(left_path)
    right_reader = imageio.get_reader(right_path)
    left_meta = left_reader.get_meta_data()
    right_meta = right_reader.get_meta_data()
    fps = float(left_meta["fps"])
    if abs(fps - float(right_meta["fps"])) > 1e-6:
        raise RuntimeError("Comparison videos have different frame rates")
    frame_count = min(left_reader.count_frames(), right_reader.count_frames())
    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    try:
        for frame_index in range(frame_count):
            left = Image.fromarray(left_reader.get_data(frame_index)).convert("RGB")
            right = Image.fromarray(right_reader.get_data(frame_index)).convert("RGB")
            if left.size != right.size:
                right = right.resize(left.size, Image.Resampling.LANCZOS)
            width, height = left.size
            canvas = Image.new("RGB", (width * 2, height + 50), (18, 22, 30))
            canvas.paste(left, (0, 50))
            canvas.paste(right, (width, 50))
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.rectangle((0, 0, width, 50), fill=(18, 93, 145, 255))
            draw.rectangle((width, 0, width * 2, 50), fill=(164, 46, 63, 255))
            draw.text((18, 16), "A  Valid upstream A4 command + obstacle + OSCBF", fill="white")
            draw.text((width + 18, 16), "B  Injected A4=-0.5 rad command + obstacle + OSCBF", fill="white")
            draw.rectangle((width - 1, 0, width + 1, height + 50), fill=(255, 255, 255, 210))
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        left_reader.close()
        right_reader.close()


def main() -> None:
    if not REFERENCE_VIDEO.exists():
        raise FileNotFoundError(f"Reference video not found: {REFERENCE_VIDEO}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_trajectory = Trajectory.load(DEFAULT_TRAJECTORY)
    model = load_model_with_obstacle(DEFAULT_URDF, OBSTACLE)
    index = build_index(model)
    valid_trajectory = source_trajectory
    violating_trajectory = make_a4_violating_trajectory(
        source_trajectory, index.arm_lower, target_rad=-0.5
    )

    valid_records = run_rollout(
        model, valid_trajectory, OBSTACLE, True, DT, ALPHA, TRACKING_GAIN,
        JOINT_MARGIN, VELOCITY_SCALE,
    )
    violating_records = run_rollout(
        model, violating_trajectory, OBSTACLE, True, DT, ALPHA, TRACKING_GAIN,
        JOINT_MARGIN, VELOCITY_SCALE,
    )
    write_csv(OUTPUT / "valid_upstream_oscbf_rollout.csv", valid_records)
    write_csv(OUTPUT / "a4_violating_upstream_oscbf_rollout.csv", violating_records)

    report = {
        "method": "velocity-level task-consistent OSCBF with obstacle and URDF joint-limit CBFs",
        "comparison_definition": {
            "valid_upstream": "current thumbs-up trajectory; A4_joint remains at 0 rad and obeys the URDF range",
            "violating_upstream": "same current trajectory with only A4_joint smoothly injected to -0.5 rad",
            "reference_video": str(REFERENCE_VIDEO),
            "reference_use": "camera, obstacle appearance, and OSCBF visualization reference; it is not overwritten",
            "obstacle_center_m": OBSTACLE.center.tolist(),
            "obstacle_radius_m": OBSTACLE.radius,
            "safety_margin_m": OBSTACLE.margin,
            "control_dt_s": DT,
            "alpha": ALPHA,
        },
        "urdf_arm_joint_limits_rad": {
            name: [float(index.arm_lower[i]), float(index.arm_upper[i])]
            for i, name in enumerate(ARM_JOINTS)
        },
        "valid_upstream": scenario_metrics(
            valid_trajectory, valid_records, index.arm_lower, index.arm_upper
        ),
        "a4_violating_upstream": scenario_metrics(
            violating_trajectory, violating_records, index.arm_lower, index.arm_upper
        ),
        "output_trajectory_difference": comparison_metrics(valid_records, violating_records),
    }
    (OUTPUT / "comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_joint_trajectories(
        valid_records, violating_records, index.arm_lower, index.arm_upper,
        OUTPUT / "joint_trajectory_comparison.png",
    )
    plot_intervention(
        valid_records, violating_records, OUTPUT / "barrier_and_intervention.png"
    )
    valid_video = OUTPUT / "valid_upstream_with_obstacle.mp4"
    record_rollout_video(
        model, valid_trajectory, valid_records, valid_video, VIDEO_FPS, 1
    )
    violating_video = OUTPUT / "a4_violating_upstream_with_obstacle.mp4"
    record_rollout_video(
        model, violating_trajectory, violating_records, violating_video, VIDEO_FPS, 1
    )
    make_side_by_side_video(
        valid_video, violating_video, OUTPUT / "a4_limit_side_by_side_comparison.mp4"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Results written to: {OUTPUT}")


if __name__ == "__main__":
    main()
