from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import mujoco
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
    run_rollout,
    set_state,
    write_csv,
)


OUTPUT_DIR = Path("outputs/joint_limit_comparison")
DT = 0.005
ALPHA = 10.0
TRACKING_GAIN = 7.0
VELOCITY_SCALE = 0.25
FPS = 30
A4 = "A4_joint"


def make_valid_reference(
    trajectory: Trajectory, lower: np.ndarray, upper: np.ndarray
) -> Trajectory:
    """Create the control reference by clipping arm commands to URDF limits.

    The original trajectory only violates A4's lower limit.  Clipping rather
    than shifting keeps the same initial pose and changes only the invalid
    portion of the upstream command, making the comparison easy to interpret.
    """
    q = trajectory.q.copy()
    columns = {name: i for i, name in enumerate(trajectory.joint_names)}
    for joint_number, name in enumerate(ARM_JOINTS):
        column = columns[name]
        q[:, column] = np.clip(q[:, column], lower[joint_number], upper[joint_number])
    qd = np.gradient(q, trajectory.time, axis=0, edge_order=2)
    return replace(trajectory, q=q, qd=qd)


def make_invalid_a4_reference(
    trajectory: Trajectory, target_rad: float = -0.5
) -> Trajectory:
    """Inject a smooth A4 lower-limit violation into an upstream copy.

    A4 starts at 0 rad, ramps smoothly to ``target_rad`` during the arm-raise
    phase, and remains there during the gesture and hold phases.  The source
    trajectory file on disk is never modified.
    """
    q = trajectory.q.copy()
    column = trajectory.joint_names.index(A4)
    raise_indices = [
        index for index, phase in enumerate(trajectory.phase)
        if phase == "2/3 raise_arm_pose"
    ]
    if not raise_indices:
        raise ValueError("The upstream trajectory has no arm-raise phase")
    ramp_start = float(trajectory.time[raise_indices[0]])
    ramp_end = float(trajectory.time[raise_indices[-1]])
    progress = np.clip(
        (trajectory.time - ramp_start) / max(ramp_end - ramp_start, 1e-9), 0.0, 1.0
    )
    smooth_progress = progress * progress * (3.0 - 2.0 * progress)
    q[:, column] = target_rad * smooth_progress
    qd = np.gradient(q, trajectory.time, axis=0, edge_order=2)
    return replace(trajectory, q=q, qd=qd)


def write_reference(path: Path, trajectory: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "phase", *trajectory.joint_names])
        for index, time_s in enumerate(trajectory.time):
            writer.writerow(
                [float(time_s), trajectory.phase[index], *trajectory.q[index].tolist()]
            )


def values(records: list[dict[str, float | str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in records])


def first_intervention_time(records: list[dict[str, float | str]]) -> float | None:
    nominal = np.column_stack(
        [values(records, f"{name}_nominal_velocity") for name in ARM_JOINTS]
    )
    command = np.column_stack(
        [values(records, f"{name}_command_velocity") for name in ARM_JOINTS]
    )
    indices = np.flatnonzero(np.linalg.norm(command - nominal, axis=1) > 1e-4)
    return None if len(indices) == 0 else float(records[int(indices[0])]["time_s"])


def rollout_summary(
    records: list[dict[str, float | str]], lower: np.ndarray, upper: np.ndarray
) -> dict[str, object]:
    actual = np.column_stack([values(records, name) for name in ARM_JOINTS])
    desired = np.column_stack(
        [values(records, f"{name}_desired") for name in ARM_JOINTS]
    )
    nominal = np.column_stack(
        [values(records, f"{name}_nominal_velocity") for name in ARM_JOINTS]
    )
    command = np.column_stack(
        [values(records, f"{name}_command_velocity") for name in ARM_JOINTS]
    )
    correction = command - nominal
    a4_index = ARM_JOINTS.index(A4)
    minimum_h = np.minimum(actual - lower, upper - actual)
    return {
        "minimum_joint_limit_h_rad": float(np.min(minimum_h)),
        "joint_limit_unsafe_steps": int(np.sum(np.min(minimum_h, axis=1) < -1e-6)),
        "a4_desired_min_rad": float(np.min(desired[:, a4_index])),
        "a4_actual_min_rad": float(np.min(actual[:, a4_index])),
        "a4_lower_limit_rad": float(lower[a4_index]),
        "a4_max_abs_tracking_error_rad": float(
            np.max(np.abs(desired[:, a4_index] - actual[:, a4_index]))
        ),
        "maximum_all_joint_tracking_error_rad": float(
            np.max(np.linalg.norm(desired - actual, axis=1))
        ),
        "maximum_filter_correction_norm_rad_s": float(
            np.max(np.linalg.norm(correction, axis=1))
        ),
        "maximum_a4_filter_correction_rad_s": float(
            np.max(np.abs(correction[:, a4_index]))
        ),
        "maximum_non_a4_filter_correction_rad_s": float(
            np.max(np.abs(np.delete(correction, a4_index, axis=1)))
        ),
        "first_filter_intervention_time_s": first_intervention_time(records),
        "maximum_qp_constraint_residual": float(
            np.max(values(records, "qp_max_violation"))
        ),
    }


def plot_a4_response(
    valid_records: list[dict[str, float | str]],
    invalid_records: list[dict[str, float | str]],
    lower_limit: float,
    output: Path,
) -> None:
    t = values(invalid_records, "time_s")
    image = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (800, 30), "OSCBF response to an upstream A4 joint-limit violation",
        anchor="ma", font=get_font(30, bold=True), fill="#172033",
    )
    panels = [(70, 100, 770, 505), (830, 100, 1530, 505), (70, 565, 770, 970), (830, 565, 1530, 970)]
    draw_panel(
        draw, panels[0], t,
        [
            (np.full_like(t, lower_limit), "#111827", "URDF lower limit", False),
            (values(valid_records, f"{A4}_desired"), "#f59e0b", "Upstream A4 reference", True),
            (values(valid_records, A4), "#2563eb", "Executed A4 after OSCBF", False),
        ],
        "Valid upstream reference", "A4 position [rad]", (-0.6, 0.15), unsafe_below=lower_limit,
    )
    draw_panel(
        draw, panels[1], t,
        [
            (np.full_like(t, lower_limit), "#111827", "URDF lower limit", False),
            (values(invalid_records, f"{A4}_desired"), "#ef4444", "Upstream A4 reference", True),
            (values(invalid_records, A4), "#2563eb", "Executed A4 after OSCBF", False),
        ],
        "Invalid upstream reference", "A4 position [rad]", (-0.6, 0.15), unsafe_below=lower_limit,
    )
    nominal_velocity = values(invalid_records, f"{A4}_nominal_velocity")
    safe_velocity = values(invalid_records, f"{A4}_command_velocity")
    velocity_range = padded_range(np.concatenate([nominal_velocity, safe_velocity, [0.0]]))
    draw_panel(
        draw, panels[2], t,
        [
            (nominal_velocity, "#ef4444", "Nominal A4 velocity", True),
            (safe_velocity, "#16a34a", "OSCBF-safe A4 velocity", False),
            (np.zeros_like(t), "#111827", "Zero velocity", False),
        ],
        "How OSCBF modifies the unsafe command", "A4 velocity [rad/s]", velocity_range,
    )
    valid_h = values(valid_records, f"{A4}_lower_h")
    invalid_h = values(invalid_records, f"{A4}_lower_h")
    barrier_range = padded_range(np.concatenate([valid_h, invalid_h, [0.0]]), minimum_span=0.04)
    draw_panel(
        draw, panels[3], t,
        [
            (np.zeros_like(t), "#dc2626", "Safety boundary h=0", False),
            (valid_h, "#2563eb", "Valid-reference h_lower", False),
            (invalid_h, "#f97316", "Invalid-reference h_lower", False),
        ],
        "A4 lower-limit barrier", "h_lower [rad]", barrier_range,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def plot_joint_difference(
    valid_records: list[dict[str, float | str]],
    invalid_records: list[dict[str, float | str]],
    output: Path,
) -> None:
    t = values(invalid_records, "time_s")
    colors = ["#2563eb", "#16a34a", "#9333ea", "#dc2626", "#f59e0b", "#0891b2", "#64748b"]
    series = []
    all_values = [np.zeros_like(t)]
    for name, color in zip(ARM_JOINTS, colors):
        difference = values(invalid_records, name) - values(valid_records, name)
        all_values.append(difference)
        series.append((difference, color, name, False))
    series.append((np.zeros_like(t), "#111827", "zero", False))
    image = Image.new("RGB", (1600, 820), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (800, 35), "Executed joint-trajectory change caused by the invalid A4 reference",
        anchor="ma", font=get_font(30, bold=True), fill="#172033",
    )
    draw_panel(
        draw, (80, 105, 1520, 775), t, series,
        "Invalid case minus valid case", "Joint-position difference [rad]",
        padded_range(np.concatenate(all_values), minimum_span=0.1), legend_columns=4,
    )
    image.save(output)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    for path in (Path("C:/Windows/Fonts") / font_name, Path(font_name)):
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def padded_range(values_array: np.ndarray, minimum_span: float = 0.2) -> tuple[float, float]:
    low = float(np.min(values_array))
    high = float(np.max(values_array))
    span = max(high - low, minimum_span)
    center = 0.5 * (low + high)
    return center - 0.6 * span, center + 0.6 * span


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    times: np.ndarray,
    series: list[tuple[np.ndarray, str, str, bool]],
    title: str,
    y_label: str,
    y_range: tuple[float, float],
    unsafe_below: float | None = None,
    legend_columns: int = 2,
) -> None:
    left, top, right, bottom = box
    plot_left, plot_top = left + 88, top + 50
    plot_right, plot_bottom = right - 24, bottom - 74
    draw.rounded_rectangle(box, radius=12, fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.text(((left + right) // 2, top + 17), title, anchor="ma", font=get_font(20, True), fill="#172033")
    x_min, x_max = float(times[0]), float(times[-1])
    y_min, y_max = y_range

    def px(x: float) -> float:
        return plot_left + (x - x_min) / max(x_max - x_min, 1e-12) * (plot_right - plot_left)

    def py(y: float) -> float:
        return plot_bottom - (y - y_min) / max(y_max - y_min, 1e-12) * (plot_bottom - plot_top)

    if unsafe_below is not None and unsafe_below > y_min:
        cutoff = min(max(py(unsafe_below), plot_top), plot_bottom)
        draw.rectangle((plot_left, cutoff, plot_right, plot_bottom), fill="#fee2e2")
    for grid_index in range(6):
        x_value = x_min + (x_max - x_min) * grid_index / 5
        x_pixel = px(x_value)
        draw.line((x_pixel, plot_top, x_pixel, plot_bottom), fill="#dbe3ee", width=1)
        draw.text((x_pixel, plot_bottom + 8), f"{x_value:.1f}", anchor="ma", font=get_font(13), fill="#475569")
        y_value = y_min + (y_max - y_min) * grid_index / 5
        y_pixel = py(y_value)
        draw.line((plot_left, y_pixel, plot_right, y_pixel), fill="#dbe3ee", width=1)
        draw.text((plot_left - 8, y_pixel), f"{y_value:+.2f}", anchor="rm", font=get_font(13), fill="#475569")
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#64748b", width=1)

    for data, color, _, dashed in series:
        points = [(px(float(x)), py(float(y))) for x, y in zip(times, data)]
        if dashed:
            for index in range(0, len(points) - 1, 8):
                segment = points[index:min(index + 5, len(points))]
                if len(segment) > 1:
                    draw.line(segment, fill=color, width=3)
        else:
            draw.line(points, fill=color, width=3)

    draw.text(((plot_left + plot_right) // 2, bottom - 24), "Time [s]", anchor="mm", font=get_font(15), fill="#334155")
    draw.text((left + 13, (plot_top + plot_bottom) // 2), y_label, anchor="lm", font=get_font(14), fill="#334155")
    legend_y = plot_bottom + 33
    column_width = max((plot_right - plot_left) // legend_columns, 1)
    for index, (_, color, label, dashed) in enumerate(series):
        row = index // legend_columns
        column = index % legend_columns
        x = plot_left + column * column_width
        y = legend_y + row * 19
        if dashed:
            draw.line((x, y, x + 22, y), fill=color, width=3)
            draw.line((x + 29, y, x + 42, y), fill=color, width=3)
        else:
            draw.line((x, y, x + 42, y), fill=color, width=3)
        draw.text((x + 48, y), label, anchor="lm", font=get_font(12), fill="#334155")


def record_comparison_video(
    model: mujoco.MjModel,
    valid_trajectory: Trajectory,
    invalid_trajectory: Trajectory,
    valid_records: list[dict[str, float | str]],
    invalid_records: list[dict[str, float | str]],
    lower_limit: float,
    output: Path,
) -> None:
    index = build_index(model)
    valid_data = mujoco.MjData(model)
    invalid_data = mujoco.MjData(model)
    times = values(invalid_records, "time_s")
    duration = float(times[-1] - times[0])
    valid_history = np.column_stack([values(valid_records, name) for name in ARM_JOINTS])
    invalid_history = np.column_stack([values(invalid_records, name) for name in ARM_JOINTS])
    valid_desired = values(valid_records, f"{A4}_desired")
    invalid_desired = values(invalid_records, f"{A4}_desired")

    original_site_rgba = model.site_rgba.copy()
    original_geom_rgba = model.geom_rgba.copy()
    model.site_rgba[:, 3] = 0.0
    for geom_name in ("cbf_obstacle_safety_zone", "cbf_obstacle_physical"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            model.geom_rgba[geom_id, 3] = 0.0

    camera = mujoco.MjvCamera()
    camera.azimuth = -130
    camera.elevation = -18
    camera.distance = 1.45
    camera.lookat[:] = np.asarray([0.68, -0.35, 0.82])
    renderer = mujoco.Renderer(model, height=480, width=640)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output), fps=FPS, codec="libx264", quality=8, macro_block_size=None
    )
    try:
        for frame_index in range(int(round(duration * FPS)) + 1):
            time_s = min(frame_index / FPS, duration)
            valid_q = np.asarray(
                [np.interp(time_s, times, valid_history[:, i]) for i in range(7)]
            )
            invalid_q = np.asarray(
                [np.interp(time_s, times, invalid_history[:, i]) for i in range(7)]
            )
            valid_all, _, valid_phase = valid_trajectory.sample(time_s)
            invalid_all, _, invalid_phase = invalid_trajectory.sample(time_s)

            set_state(model, valid_data, index, valid_trajectory, valid_q, valid_all)
            renderer.update_scene(valid_data, camera=camera)
            left = Image.fromarray(renderer.render())
            set_state(model, invalid_data, index, invalid_trajectory, invalid_q, invalid_all)
            renderer.update_scene(invalid_data, camera=camera)
            right = Image.fromarray(renderer.render())

            frame = Image.new("RGB", (1280, 480), "black")
            frame.paste(left, (0, 0))
            frame.paste(right, (640, 0))
            draw = ImageDraw.Draw(frame, "RGBA")
            draw.rectangle((0, 0, 1280, 82), fill=(0, 0, 0, 170))
            valid_actual = float(np.interp(time_s, times, valid_history[:, 3]))
            invalid_actual = float(np.interp(time_s, times, invalid_history[:, 3]))
            valid_ref = float(np.interp(time_s, times, valid_desired))
            invalid_ref = float(np.interp(time_s, times, invalid_desired))
            draw.text((20, 14), "VALID UPSTREAM: A4 command stays inside URDF limit", fill=(134, 239, 172, 255))
            draw.text((660, 14), "INVALID UPSTREAM: original A4 command reaches -0.5 rad", fill=(252, 165, 165, 255))
            draw.text((20, 39), f"A4 ref={valid_ref:+.3f}  executed={valid_actual:+.3f}  lower={lower_limit:+.3f}", fill="white")
            draw.text((660, 39), f"A4 ref={invalid_ref:+.3f}  executed={invalid_actual:+.3f}  lower={lower_limit:+.3f}", fill="white")
            draw.text((20, 61), f"t={time_s:4.1f}/{duration:.1f}s  {valid_phase}", fill=(220, 230, 245, 255))
            draw.text((660, 61), f"t={time_s:4.1f}/{duration:.1f}s  {invalid_phase}", fill=(220, 230, 245, 255))
            draw.line((639, 0, 639, 480), fill=(255, 255, 255, 160), width=2)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
        renderer.close()
        model.site_rgba[:] = original_site_rgba
        model.geom_rgba[:] = original_geom_rgba


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    obstacle = Obstacle(np.asarray([10.0, 10.0, 10.0]), 0.1, 0.0)
    model = load_model_with_obstacle(DEFAULT_URDF, obstacle)
    index = build_index(model)
    source_trajectory = Trajectory.load(DEFAULT_TRAJECTORY)
    valid_trajectory = make_valid_reference(
        source_trajectory, index.arm_lower, index.arm_upper
    )
    invalid_trajectory = make_invalid_a4_reference(valid_trajectory, target_rad=-0.5)

    write_reference(OUTPUT_DIR / "upstream_invalid_a4_reference.csv", invalid_trajectory)
    write_reference(OUTPUT_DIR / "upstream_valid_reference.csv", valid_trajectory)

    valid_records = run_rollout(
        model, valid_trajectory, obstacle, True, DT, ALPHA, TRACKING_GAIN,
        0.0, VELOCITY_SCALE,
    )
    invalid_records = run_rollout(
        model, invalid_trajectory, obstacle, True, DT, ALPHA, TRACKING_GAIN,
        0.0, VELOCITY_SCALE,
    )
    write_csv(OUTPUT_DIR / "oscbf_valid_upstream_rollout.csv", valid_records)
    write_csv(OUTPUT_DIR / "oscbf_invalid_a4_rollout.csv", invalid_records)

    a4_index = ARM_JOINTS.index(A4)
    report = {
        "experiment": "URDF joint-limit CBF response to an invalid upstream A4 command",
        "isolation_choice": (
            "The obstacle is placed far outside the workspace so the comparison "
            "isolates the joint-limit CBF. Both cases use identical controller settings."
        ),
        "valid_reference_construction": (
            "Use the current source trajectory after clipping to the URDF intervals. "
            "The current source A4 command is already 0 rad throughout."
        ),
        "invalid_reference_construction": (
            "Without modifying the source CSV, inject a smooth A4 command from 0 to "
            "-0.5 rad during the arm-raise phase and hold -0.5 rad afterward."
        ),
        "urdf_limits_rad": {
            name: [float(index.arm_lower[i]), float(index.arm_upper[i])]
            for i, name in enumerate(ARM_JOINTS)
        },
        "cbf": {
            "lower": "qdot_i >= -alpha * (q_i - q_i,min)",
            "upper": "-qdot_i >= -alpha * (q_i,max - q_i)",
            "alpha": ALPHA,
            "control_dt_s": DT,
        },
        "valid_upstream": rollout_summary(valid_records, index.arm_lower, index.arm_upper),
        "invalid_a4_upstream": rollout_summary(invalid_records, index.arm_lower, index.arm_upper),
        "upstream_a4_ranges_rad": {
            "valid": [
                float(np.min(valid_trajectory.q[:, valid_trajectory.joint_names.index(A4)])),
                float(np.max(valid_trajectory.q[:, valid_trajectory.joint_names.index(A4)])),
            ],
            "invalid": [
                float(np.min(invalid_trajectory.q[:, invalid_trajectory.joint_names.index(A4)])),
                float(np.max(invalid_trajectory.q[:, invalid_trajectory.joint_names.index(A4)])),
            ],
            "urdf": [float(index.arm_lower[a4_index]), float(index.arm_upper[a4_index])],
        },
    }
    (OUTPUT_DIR / "joint_limit_comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_a4_response(
        valid_records, invalid_records, float(index.arm_lower[a4_index]),
        OUTPUT_DIR / "a4_joint_limit_response.png",
    )
    plot_joint_difference(
        valid_records, invalid_records,
        OUTPUT_DIR / "all_joint_trajectory_difference.png",
    )
    record_comparison_video(
        model, valid_trajectory, invalid_trajectory, valid_records, invalid_records,
        float(index.arm_lower[a4_index]),
        OUTPUT_DIR / "a4_joint_limit_comparison.mp4",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Results written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
