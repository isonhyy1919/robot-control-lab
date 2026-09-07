from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_URDF = PROJECT_ROOT / "models/a7v2_arm_hand/urdf/a7v2_latest_stand_single_arm_right_hand.urdf"
DEFAULT_TRAJECTORY = PROJECT_ROOT / "thumbsup/outputs/thumbs_up_trajectory.csv"
ARM_JOINTS = tuple(f"A{i}_joint" for i in range(1, 8))

# Conservative sphere chain attached to the arm.  These are control geometry,
# not MuJoCo contact geometry.  Multiple spheres per long link make the CBF see
# the whole arm rather than only the joint origins.
COLLISION_SPHERES = (
    ("A1_Link", "cbf_A1_0", (0.0, 0.0, 0.000), 0.075),
    ("A1_Link", "cbf_A1_1", (0.0, 0.0, 0.035), 0.070),
    ("A2_Link", "cbf_A2_0", (0.0, 0.0, 0.020), 0.070),
    ("A2_Link", "cbf_A2_1", (0.0, 0.0, 0.080), 0.065),
    ("A3_Link", "cbf_A3_0", (0.0, 0.0, 0.035), 0.070),
    ("A3_Link", "cbf_A3_1", (0.0, 0.0, 0.125), 0.065),
    ("A4_Link", "cbf_A4_0", (0.0, 0.0, 0.025), 0.065),
    ("A4_Link", "cbf_A4_1", (0.0, 0.0, 0.085), 0.060),
    ("A5_Link", "cbf_A5_0", (0.0, 0.0, 0.018), 0.065),
    ("A5_Link", "cbf_A5_1", (0.0, 0.0, 0.060), 0.060),
    ("A6_Link", "cbf_A6_0", (0.0, 0.0, 0.015), 0.060),
    ("A6_Link", "cbf_A6_1", (0.0, 0.0, 0.050), 0.058),
    ("A7_Link", "cbf_A7_0", (0.0, 0.0, 0.035), 0.072),
    ("A7_Link", "cbf_A7_1", (0.0, 0.0, 0.105), 0.080),
)


@dataclass(frozen=True)
class Obstacle:
    center: np.ndarray
    radius: float
    margin: float


@dataclass
class Trajectory:
    time: np.ndarray
    phase: list[str]
    joint_names: list[str]
    q: np.ndarray
    qd: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "Trajectory":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Trajectory is empty: {path}")
        joint_names = [name for name in rows[0] if name not in {
            "time_s", "phase", "phase_progress", "A7_x", "A7_y", "A7_z",
            "thumb_tip_x", "thumb_tip_y", "thumb_tip_z",
            "middle_tip_x", "middle_tip_y", "middle_tip_z",
        }]
        time_values = np.asarray([float(row["time_s"]) for row in rows])
        q = np.asarray([[float(row[name]) for name in joint_names] for row in rows])
        qd = np.gradient(q, time_values, axis=0, edge_order=2)
        return cls(time_values, [row["phase"] for row in rows], joint_names, q, qd)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, str]:
        t = float(np.clip(t, self.time[0], self.time[-1]))
        q = np.asarray([np.interp(t, self.time, self.q[:, i]) for i in range(self.q.shape[1])])
        qd = np.asarray([np.interp(t, self.time, self.qd[:, i]) for i in range(self.qd.shape[1])])
        index = min(int(np.searchsorted(self.time, t, side="right") - 1), len(self.phase) - 1)
        return q, qd, self.phase[max(index, 0)]


@dataclass
class ModelIndex:
    joint_qpos: dict[str, int]
    joint_dof: dict[str, int]
    arm_qpos: np.ndarray
    arm_dof: np.ndarray
    arm_lower: np.ndarray
    arm_upper: np.ndarray
    arm_velocity: np.ndarray
    collision_sites: list[tuple[int, str, float]]
    ee_body: int


def load_model_with_obstacle(urdf: Path, obstacle: Obstacle) -> mujoco.MjModel:
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    old_cwd = Path.cwd()
    try:
        os.chdir(urdf.parent)
        spec = mujoco.MjSpec.from_file(urdf.name)
        for body_name, site_name, position, radius in COLLISION_SPHERES:
            body = spec.body(body_name)
            if body is None:
                raise KeyError(f"Body not found while creating CBF sphere: {body_name}")
            body.add_site(
                name=site_name,
                pos=position,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0.0, 0.0],
                rgba=[0.1, 0.65, 1.0, 0.22],
                group=3,
            )
        # The translucent outer volume shows the environment-side CBF safety
        # boundary.  Since each blue robot collision sphere is also rendered,
        # non-overlap between a blue sphere and this volume is exactly h >= 0.
        spec.worldbody.add_geom(
            name="cbf_obstacle_safety_zone",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=obstacle.center,
            size=[obstacle.radius + obstacle.margin, 0.0, 0.0],
            rgba=[1.0, 0.82, 0.0, 0.24],
            # URDF compilation discards geoms whose contype and conaffinity
            # are both zero. Bit 2 keeps this visual geom in the model, while
            # remaining disjoint from the robot's bit-1 collision masks.
            contype=2,
            conaffinity=0,
            group=0,
        )
        spec.worldbody.add_geom(
            name="cbf_obstacle_physical",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=obstacle.center,
            size=[obstacle.radius, 0.0, 0.0],
            rgba=[1.0, 0.02, 0.02, 1.0],
            contype=2,
            conaffinity=0,
            group=0,
        )
        return spec.compile()
    finally:
        os.chdir(old_cwd)


def build_index(model: mujoco.MjModel) -> ModelIndex:
    joint_qpos: dict[str, int] = {}
    joint_dof: dict[str, int] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            joint_qpos[name] = int(model.jnt_qposadr[joint_id])
            joint_dof[name] = int(model.jnt_dofadr[joint_id])
    missing = [name for name in ARM_JOINTS if name not in joint_qpos]
    if missing:
        raise KeyError(f"Arm joints missing from MuJoCo model: {missing}")
    arm_joint_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS
    ])
    arm_qpos = np.asarray([joint_qpos[name] for name in ARM_JOINTS], dtype=int)
    arm_dof = np.asarray([joint_dof[name] for name in ARM_JOINTS], dtype=int)
    collision_sites = []
    radii = {name: radius for _, name, _, radius in COLLISION_SPHERES}
    for site_name, radius in radii.items():
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise KeyError(f"CBF site missing after model compilation: {site_name}")
        collision_sites.append((site_id, site_name, radius))
    ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A7_Link")
    return ModelIndex(
        joint_qpos=joint_qpos,
        joint_dof=joint_dof,
        arm_qpos=arm_qpos,
        arm_dof=arm_dof,
        arm_lower=model.jnt_range[arm_joint_ids, 0].copy(),
        arm_upper=model.jnt_range[arm_joint_ids, 1].copy(),
        arm_velocity=np.asarray([10.47, 10.47, 9.1, 9.1, 9.1, 9.1, 9.1]),
        collision_sites=collision_sites,
        ee_body=ee_body,
    )


def set_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: ModelIndex,
    trajectory: Trajectory,
    arm_q: np.ndarray,
    desired_all: np.ndarray,
    arm_velocity: np.ndarray | None = None,
) -> None:
    for name, value in zip(trajectory.joint_names, desired_all):
        address = index.joint_qpos.get(name)
        if address is not None:
            data.qpos[address] = value
    data.qpos[index.arm_qpos] = arm_q
    data.qvel[:] = 0.0
    if arm_velocity is not None:
        data.qvel[index.arm_dof] = arm_velocity
    mujoco.mj_forward(model, data)


def arm_jacobian_for_body(model: mujoco.MjModel, data: mujoco.MjData, index: ModelIndex) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, index.ee_body)
    return np.vstack([jacp[:, index.arm_dof], jacr[:, index.arm_dof]])


def task_consistent_metric(model: mujoco.MjModel, data: mujoco.MjData, index: ModelIndex) -> np.ndarray:
    jacobian = arm_jacobian_for_body(model, data, index)
    jacobian_hash = np.linalg.pinv(jacobian, rcond=1e-5)
    null_projection = np.eye(len(ARM_JOINTS)) - jacobian_hash @ jacobian
    task_weights = np.diag(np.asarray([4.0, 4.0, 4.0, 0.45, 0.45, 0.45]) ** 2)
    joint_weights = np.eye(len(ARM_JOINTS)) * 0.35**2
    return (
        jacobian.T @ task_weights @ jacobian
        + null_projection.T @ joint_weights @ null_projection
        + np.eye(len(ARM_JOINTS)) * 1e-3
    )


def add_row(rows: list[np.ndarray], bounds: list[float], row: np.ndarray, bound: float) -> None:
    rows.append(np.asarray(row, dtype=float))
    bounds.append(float(bound))


def cbf_constraints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: ModelIndex,
    obstacle: Obstacle,
    alpha: float,
    joint_margin: float,
    velocity_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # All constraints use A @ qdot >= b.
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    h_values: list[float] = []
    for site_id, _, robot_radius in index.collision_sites:
        delta = data.site_xpos[site_id] - obstacle.center
        distance = float(np.linalg.norm(delta))
        normal = delta / max(distance, 1e-9)
        h = distance - robot_radius - obstacle.radius - obstacle.margin
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        gradient = normal @ jacp[:, index.arm_dof]
        add_row(rows, bounds, gradient, -alpha * h)
        h_values.append(h)

    q = data.qpos[index.arm_qpos]
    identity = np.eye(len(ARM_JOINTS))
    lower_h = q - (index.arm_lower + joint_margin)
    upper_h = (index.arm_upper - joint_margin) - q
    for i in range(len(ARM_JOINTS)):
        add_row(rows, bounds, identity[i], -alpha * lower_h[i])
        add_row(rows, bounds, -identity[i], -alpha * upper_h[i])
        h_values.extend([float(lower_h[i]), float(upper_h[i])])

    max_velocity = index.arm_velocity * velocity_scale
    for i in range(len(ARM_JOINTS)):
        add_row(rows, bounds, identity[i], -max_velocity[i])
        add_row(rows, bounds, -identity[i], -max_velocity[i])
    return np.vstack(rows), np.asarray(bounds), np.asarray(h_values)


def solve_projection_qp(
    nominal: np.ndarray,
    metric: np.ndarray,
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
    max_iterations: int = 1500,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, float, int]:
    """Solve min 0.5*(x-nominal)'P*(x-nominal), A*x >= b.

    Hildreth dual coordinate descent is sufficient here because the QP has
    seven variables and a few dozen linear inequalities.  It avoids adding a
    heavyweight external QP dependency to the MuJoCo demo.
    """
    p_inv = np.linalg.pinv(metric, rcond=1e-10)
    dual_hessian = a_matrix @ p_inv @ a_matrix.T
    violation_at_nominal = b_vector - a_matrix @ nominal
    multipliers = np.zeros(len(b_vector))
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        max_change = 0.0
        for i in range(len(multipliers)):
            diagonal = dual_hessian[i, i]
            if diagonal <= 1e-14:
                continue
            residual = violation_at_nominal[i] - dual_hessian[i] @ multipliers
            updated = max(0.0, multipliers[i] + residual / diagonal)
            max_change = max(max_change, abs(updated - multipliers[i]))
            multipliers[i] = updated
        if max_change < tolerance:
            break
    solution = nominal + p_inv @ a_matrix.T @ multipliers
    max_violation = float(np.max(b_vector - a_matrix @ solution))
    return solution, max_violation, iterations


def body_position(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return data.xpos[body_id].copy()


def run_rollout(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    obstacle: Obstacle,
    use_cbf: bool,
    dt: float,
    alpha: float,
    tracking_gain: float,
    joint_margin: float,
    velocity_scale: float,
    viewer: bool = False,
) -> list[dict[str, float | str]]:
    index = build_index(model)
    data = mujoco.MjData(model)
    desired_all, _, _ = trajectory.sample(trajectory.time[0])
    trajectory_column = {name: i for i, name in enumerate(trajectory.joint_names)}
    arm_columns = np.asarray([trajectory_column[name] for name in ARM_JOINTS], dtype=int)
    arm_q = desired_all[arm_columns].copy()
    set_state(model, data, index, trajectory, arm_q, desired_all)
    records: list[dict[str, float | str]] = []

    handle = None
    if viewer:
        from mujoco import viewer as mj_viewer
        handle = mj_viewer.launch_passive(model, data)
        handle.cam.azimuth = -130
        handle.cam.elevation = -18
        handle.cam.distance = 1.45
        handle.cam.lookat[:] = np.asarray([0.68, -0.35, 0.82])

    try:
        steps = int(np.ceil((trajectory.time[-1] - trajectory.time[0]) / dt)) + 1
        wall_start = time.perf_counter()
        for step in range(steps):
            t = min(trajectory.time[0] + step * dt, trajectory.time[-1])
            desired_all, desired_velocity_all, phase = trajectory.sample(t)
            desired_arm = desired_all[arm_columns]
            desired_arm_velocity = desired_velocity_all[arm_columns]
            nominal = desired_arm_velocity + tracking_gain * (desired_arm - arm_q)
            set_state(model, data, index, trajectory, arm_q, desired_all, nominal)
            a_matrix, b_vector, h_values = cbf_constraints(
                model, data, index, obstacle, alpha, joint_margin, velocity_scale
            )
            start = time.perf_counter_ns()
            if use_cbf:
                metric = task_consistent_metric(model, data, index)
                command, qp_violation, qp_iterations = solve_projection_qp(
                    nominal, metric, a_matrix, b_vector
                )
            else:
                max_velocity = index.arm_velocity * velocity_scale
                command = np.clip(nominal, -max_velocity, max_velocity)
                qp_violation = float(np.max(b_vector - a_matrix @ command))
                qp_iterations = 0
            qp_us = (time.perf_counter_ns() - start) / 1000.0
            if step + 1 < steps:
                # Do not hard-clip q here: position limits are CBF constraints in
                # the safe rollout.  Leaving the nominal rollout unclipped also
                # exposes invalid upstream references instead of hiding them.
                arm_q = arm_q + dt * command
            set_state(model, data, index, trajectory, arm_q, desired_all, command)
            ee_position = body_position(data, index.ee_body)
            obstacle_h = []
            for site_id, _, robot_radius in index.collision_sites:
                distance = np.linalg.norm(data.site_xpos[site_id] - obstacle.center)
                obstacle_h.append(distance - robot_radius - obstacle.radius - obstacle.margin)
            record: dict[str, float | str] = {
                "time_s": t,
                "phase": phase,
                "A7_x": ee_position[0],
                "A7_y": ee_position[1],
                "A7_z": ee_position[2],
                "min_obstacle_h": float(np.min(obstacle_h)),
                "min_joint_limit_h": float(
                    min(
                        np.min(arm_q - (index.arm_lower + joint_margin)),
                        np.min((index.arm_upper - joint_margin) - arm_q),
                    )
                ),
                "min_all_h": float(np.min(h_values)),
                "active_obstacle_cbfs": int(np.sum(np.asarray(obstacle_h) < 0.04)),
                "joint_tracking_error": float(np.linalg.norm(desired_arm - arm_q)),
                "qp_max_violation": qp_violation,
                "qp_iterations": qp_iterations,
                "qp_time_us": qp_us,
            }
            lower_joint_h = arm_q - (index.arm_lower + joint_margin)
            upper_joint_h = (index.arm_upper - joint_margin) - arm_q
            for joint_number, (name, value) in enumerate(zip(ARM_JOINTS, arm_q)):
                record[name] = float(value)
                record[f"{name}_desired"] = float(desired_arm[joint_number])
                record[f"{name}_nominal_velocity"] = float(nominal[joint_number])
                record[f"{name}_command_velocity"] = float(command[joint_number])
                record[f"{name}_lower_h"] = float(lower_joint_h[joint_number])
                record[f"{name}_upper_h"] = float(upper_joint_h[joint_number])
            records.append(record)
            if handle is not None:
                if not handle.is_running():
                    break
                handle.sync()
                target = wall_start + (step + 1) * dt
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    finally:
        if handle is not None:
            handle.close()
    return records


def view_repeated_rollout(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    records: list[dict[str, float | str]],
    dt: float,
    repeats: int,
) -> None:
    """Replay a computed rollout repeatedly in one passive MuJoCo viewer.

    ``repeats <= 0`` means continue until the viewer window is closed.
    """
    from mujoco import viewer as mj_viewer

    index = build_index(model)
    data = mujoco.MjData(model)
    arm_q = np.asarray([float(records[0][name]) for name in ARM_JOINTS])
    desired_all, _, _ = trajectory.sample(float(records[0]["time_s"]))
    set_state(model, data, index, trajectory, arm_q, desired_all)
    handle = mj_viewer.launch_passive(model, data)
    handle.cam.azimuth = -130
    handle.cam.elevation = -18
    handle.cam.distance = 1.45
    handle.cam.lookat[:] = np.asarray([0.68, -0.35, 0.82])

    cycle = 0
    try:
        while handle.is_running() and (repeats <= 0 or cycle < repeats):
            wall_start = time.perf_counter()
            for step, record in enumerate(records):
                if not handle.is_running():
                    return
                t = float(record["time_s"])
                desired_all, _, _ = trajectory.sample(t)
                arm_q = np.asarray([float(record[name]) for name in ARM_JOINTS])
                set_state(model, data, index, trajectory, arm_q, desired_all)
                handle.sync()
                delay = wall_start + (step + 1) * dt - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            cycle += 1
    finally:
        handle.close()


def record_rollout_video(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    records: list[dict[str, float | str]],
    output_path: Path,
    fps: int,
    repeats: int,
    show_safety_visuals: bool = True,
) -> None:
    """Render a rollout to an H.264 MP4 without opening the viewer."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    if repeats <= 0:
        raise ValueError("Video repeat count must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = build_index(model)
    data = mujoco.MjData(model)
    record_times = np.asarray([float(row["time_s"]) for row in records])
    arm_history = np.asarray([
        [float(row[name]) for name in ARM_JOINTS] for row in records
    ])
    duration = float(record_times[-1] - record_times[0])
    frames_per_cycle = int(round(duration * fps)) + 1

    camera = mujoco.MjvCamera()
    camera.azimuth = -130
    camera.elevation = -18
    camera.distance = 1.45
    camera.lookat[:] = np.asarray([0.68, -0.35, 0.82])
    renderer = mujoco.Renderer(model, height=480, width=640)
    original_site_rgba = model.site_rgba.copy()
    original_geom_rgba = model.geom_rgba.copy()
    if not show_safety_visuals:
        for site_id, _, _ in index.collision_sites:
            model.site_rgba[site_id, 3] = 0.0
        for geom_name in ("cbf_obstacle_safety_zone", "cbf_obstacle_physical"):
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if geom_id >= 0:
                model.geom_rgba[geom_id, 3] = 0.0
    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=None,
    )
    try:
        for cycle in range(repeats):
            for frame_index in range(frames_per_cycle):
                local_t = min(frame_index / fps, duration)
                t = record_times[0] + local_t
                arm_q = np.asarray([
                    np.interp(t, record_times, arm_history[:, joint_index])
                    for joint_index in range(len(ARM_JOINTS))
                ])
                desired_all, _, phase = trajectory.sample(t)
                set_state(model, data, index, trajectory, arm_q, desired_all)
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame, "RGBA")
                draw.rounded_rectangle((12, 12, 356, 74), radius=8, fill=(0, 0, 0, 150))
                if show_safety_visuals:
                    draw.ellipse((24, 25, 38, 39), fill=(255, 5, 5, 255))
                    draw.text((45, 23), "Obstacle", fill=(255, 255, 255, 255))
                    draw.ellipse((126, 25, 140, 39), fill=(255, 210, 0, 170))
                    draw.text((147, 23), "CBF safety boundary", fill=(255, 255, 255, 255))
                    motion_label = "OSCBF safe motion"
                else:
                    draw.text(
                        (24, 23), "Original thumbs-up motion (no obstacle)",
                        fill=(255, 255, 255, 255),
                    )
                    motion_label = "Unfiltered nominal motion"
                draw.text(
                    (24, 48),
                    f"{motion_label}   {local_t:4.1f}/{duration:.1f} s   {phase}",
                    fill=(235, 242, 255, 255),
                )
                writer.append_data(np.asarray(frame))
    finally:
        writer.close()
        renderer.close()
        model.site_rgba[:] = original_site_rgba
        model.geom_rgba[:] = original_geom_rgba


def write_csv(path: Path, records: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def percentile(values: Iterable[float], q: float) -> float:
    return float(np.percentile(np.asarray(list(values), dtype=float), q))


def summarize(records: list[dict[str, float | str]]) -> dict[str, float | int]:
    h = np.asarray([float(row["min_obstacle_h"]) for row in records])
    joint_h = np.asarray([float(row["min_joint_limit_h"]) for row in records])
    tracking = np.asarray([float(row["joint_tracking_error"]) for row in records])
    qp_time = np.asarray([float(row["qp_time_us"]) for row in records])
    qp_violation = np.asarray([float(row["qp_max_violation"]) for row in records])
    return {
        "steps": len(records),
        "minimum_obstacle_h_m": float(np.min(h)),
        "obstacle_unsafe_steps": int(np.sum(h < -1e-6)),
        "minimum_joint_limit_h_rad": float(np.min(joint_h)),
        "joint_limit_unsafe_steps": int(np.sum(joint_h < -1e-6)),
        "maximum_joint_tracking_error_rad": float(np.max(tracking)),
        "mean_joint_tracking_error_rad": float(np.mean(tracking)),
        "mean_qp_time_us": float(np.mean(qp_time)),
        "p95_qp_time_us": percentile(qp_time, 95),
        "maximum_qp_constraint_residual": float(np.max(qp_violation)),
    }


def svg_polyline(points: np.ndarray, map_x, map_y, color: str, width: float = 2.2) -> str:
    coords = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def write_svg(
    path: Path,
    baseline: list[dict[str, float | str]],
    safe: list[dict[str, float | str]],
    obstacle: Obstacle,
) -> None:
    width, height = 1000, 760
    left, right = 85, 955
    top1, bottom1 = 60, 385
    top2, bottom2 = 470, 700
    baseline_xz = np.asarray([[float(r["A7_x"]), float(r["A7_z"])] for r in baseline])
    safe_xz = np.asarray([[float(r["A7_x"]), float(r["A7_z"])] for r in safe])
    safety_radius = obstacle.radius + obstacle.margin
    all_x = np.concatenate([baseline_xz[:, 0], safe_xz[:, 0], [obstacle.center[0] - safety_radius, obstacle.center[0] + safety_radius]])
    all_z = np.concatenate([baseline_xz[:, 1], safe_xz[:, 1], [obstacle.center[2] - safety_radius, obstacle.center[2] + safety_radius]])
    x_min, x_max = float(all_x.min() - 0.04), float(all_x.max() + 0.04)
    z_min, z_max = float(all_z.min() - 0.04), float(all_z.max() + 0.04)
    map_x = lambda x: left + (x - x_min) / (x_max - x_min) * (right - left)
    map_z = lambda z: bottom1 - (z - z_min) / (z_max - z_min) * (bottom1 - top1)
    times = np.asarray([float(r["time_s"]) for r in safe])
    h_base = np.asarray([float(r["min_obstacle_h"]) for r in baseline])
    h_safe = np.asarray([float(r["min_obstacle_h"]) for r in safe])
    h_min = min(float(h_base.min()), float(h_safe.min()), -0.01)
    h_max = max(float(h_base.max()), float(h_safe.max()), 0.02)
    map_t = lambda t: left + (t - times[0]) / (times[-1] - times[0]) * (right - left)
    map_h = lambda h: bottom2 - (h - h_min) / (h_max - h_min) * (bottom2 - top2)
    obstacle_px_radius = abs(map_x(obstacle.center[0] + obstacle.radius) - map_x(obstacle.center[0]))
    safety_px_radius = abs(map_x(obstacle.center[0] + safety_radius) - map_x(obstacle.center[0]))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#222}.title{font-size:24px;font-weight:600}.label{font-size:16px}.small{font-size:14px}</style>',
        '<text x="500" y="32" text-anchor="middle" class="title">MuJoCo thumbs-up trajectory with OSCBF obstacle avoidance</text>',
        f'<rect x="{left}" y="{top1}" width="{right-left}" height="{bottom1-top1}" fill="#fafafa" stroke="#bbb"/>',
        '<text x="90" y="82" class="label">A7 trajectory, X-Z projection</text>',
        f'<circle cx="{map_x(obstacle.center[0]):.2f}" cy="{map_z(obstacle.center[2]):.2f}" r="{safety_px_radius:.2f}" fill="#ff8c1a" fill-opacity="0.14" stroke="#f07800" stroke-width="2" stroke-dasharray="7 5"/>',
        f'<circle cx="{map_x(obstacle.center[0]):.2f}" cy="{map_z(obstacle.center[2]):.2f}" r="{obstacle_px_radius:.2f}" fill="#d7191c" fill-opacity="0.72" stroke="#9d0000" stroke-width="2"/>',
        svg_polyline(baseline_xz, map_x, map_z, "#777777", 2.0),
        svg_polyline(safe_xz, map_x, map_z, "#0877d1", 2.7),
        f'<text x="{left}" y="{bottom1+25}" class="small">X: {x_min:.2f} to {x_max:.2f} m</text>',
        f'<text x="{right}" y="{bottom1+25}" text-anchor="end" class="small">dark red: obstacle; orange: safety-margin boundary</text>',
        f'<rect x="{left}" y="{top2}" width="{right-left}" height="{bottom2-top2}" fill="#fafafa" stroke="#bbb"/>',
        '<text x="90" y="492" class="label">Minimum sphere-obstacle barrier h(t)</text>',
        f'<line x1="{left}" y1="{map_h(0):.2f}" x2="{right}" y2="{map_h(0):.2f}" stroke="#d62728" stroke-dasharray="7 5"/>',
        svg_polyline(np.column_stack([times, h_base]), map_t, map_h, "#777777", 2.0),
        svg_polyline(np.column_stack([times, h_safe]), map_t, map_h, "#0877d1", 2.7),
        f'<text x="{left}" y="{bottom2+24}" class="small">0 s</text>',
        f'<text x="{right}" y="{bottom2+24}" text-anchor="end" class="small">{times[-1]:.2f} s</text>',
        '<line x1="660" y1="35" x2="700" y2="35" stroke="#777" stroke-width="3"/><text x="708" y="41" class="small">nominal</text>',
        '<line x1="790" y1="35" x2="830" y2="35" stroke="#0877d1" stroke-width="3"/><text x="838" y="41" class="small">OSCBF</text>',
        '</svg>',
    ]
    path.write_text("\n".join(elements), encoding="utf-8")


def parse_vector(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def trajectory_diagnostics(trajectory: Trajectory, model: mujoco.MjModel) -> dict[str, object]:
    index = build_index(model)
    columns = {name: i for i, name in enumerate(trajectory.joint_names)}
    violations: dict[str, dict[str, float]] = {}
    for i, name in enumerate(ARM_JOINTS):
        values = trajectory.q[:, columns[name]]
        below = max(0.0, float(index.arm_lower[i] - np.min(values)))
        above = max(0.0, float(np.max(values) - index.arm_upper[i]))
        if below > 1e-9 or above > 1e-9:
            violations[name] = {
                "urdf_lower_rad": float(index.arm_lower[i]),
                "urdf_upper_rad": float(index.arm_upper[i]),
                "trajectory_min_rad": float(np.min(values)),
                "trajectory_max_rad": float(np.max(values)),
                "maximum_violation_rad": max(below, above),
            }
    return {
        "duration_s": float(trajectory.time[-1] - trajectory.time[0]),
        "samples": int(len(trajectory.time)),
        "joint_limit_violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the thumb-up joint trajectory in MuJoCo and filter the seven arm joints with a velocity-level OSCBF."
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/oscbf_thumbsup")
    parser.add_argument("--obstacle", nargs=3, type=float, default=[0.70, -0.315, 0.82], metavar=("X", "Y", "Z"))
    parser.add_argument("--obstacle-radius", type=float, default=0.105)
    parser.add_argument("--safety-margin", type=float, default=0.025)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--tracking-gain", type=float, default=7.0)
    parser.add_argument("--joint-margin", type=float, default=0.0)
    parser.add_argument("--velocity-scale", type=float, default=0.25)
    parser.add_argument("--viewer", action="store_true", help="After the headless comparison, replay the OSCBF rollout in the MuJoCo viewer.")
    parser.add_argument(
        "--repeat", type=int, default=0,
        help="Viewer playback count; 0 repeats until the viewer is closed (default).",
    )
    parser.add_argument(
        "--record-video", type=Path, default=None, metavar="MP4",
        help="Render the OSCBF rollout to an H.264 MP4 file.",
    )
    parser.add_argument(
        "--record-nominal-video", type=Path, default=None, metavar="MP4",
        help=(
            "Render the original unfiltered thumb-up rollout without the "
            "obstacle, safety boundary, or CBF collision-sphere visuals."
        ),
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-repeats", type=int, default=1)
    args = parser.parse_args()

    obstacle = Obstacle(parse_vector(args.obstacle), args.obstacle_radius, args.safety_margin)
    trajectory = Trajectory.load(args.trajectory)
    model = load_model_with_obstacle(args.urdf, obstacle)
    baseline = run_rollout(
        model, trajectory, obstacle, False, args.dt, args.alpha, args.tracking_gain,
        args.joint_margin, args.velocity_scale,
    )
    safe = run_rollout(
        model, trajectory, obstacle, True, args.dt, args.alpha, args.tracking_gain,
        args.joint_margin, args.velocity_scale,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "nominal_rollout.csv", baseline)
    write_csv(args.output_dir / "oscbf_rollout.csv", safe)
    report = {
        "method": "velocity-level task-consistent OSCBF",
        "urdf": str(args.urdf),
        "trajectory": str(args.trajectory),
        "obstacle_center_m": obstacle.center.tolist(),
        "obstacle_radius_m": obstacle.radius,
        "safety_margin_m": obstacle.margin,
        "control_dt_s": args.dt,
        "alpha": args.alpha,
        "collision_spheres": len(COLLISION_SPHERES),
        "trajectory_diagnostics": trajectory_diagnostics(trajectory, model),
        "baseline": summarize(baseline),
        "oscbf": summarize(safe),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_svg(args.output_dir / "trajectory_comparison.svg", baseline, safe, obstacle)
    if args.record_video is not None:
        record_rollout_video(
            model, trajectory, safe, args.record_video,
            args.video_fps, args.video_repeats,
        )
        print(f"Video written to: {args.record_video.resolve()}")
    if args.record_nominal_video is not None:
        record_rollout_video(
            model, trajectory, baseline, args.record_nominal_video,
            args.video_fps, args.video_repeats, show_safety_visuals=False,
        )
        print(f"Nominal video written to: {args.record_nominal_video.resolve()}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Results written to: {args.output_dir.resolve()}")

    if args.viewer:
        view_repeated_rollout(model, trajectory, safe, args.dt, args.repeat)


if __name__ == "__main__":
    main()
