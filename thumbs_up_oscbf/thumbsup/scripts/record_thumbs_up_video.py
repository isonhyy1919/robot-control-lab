from __future__ import annotations

from pathlib import Path
import argparse
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from thumbs_up_from_urdf import (  # noqa: E402
    DEFAULT_URDF,
    apply_qpos,
    clipped_qpos,
    load_urdf_model,
    make_keyframes,
    smoothstep,
)


DEFAULT_OUT = ROOT / "outputs" / "thumbs_up_arm_hand_stand.mp4"
DEFAULT_PREVIEW_DIR = ROOT / "outputs" / "thumbs_up_preview"


def make_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    # Oblique front-side view: keeps the hand clear of the stand while retaining the stand full outline.
    camera.azimuth = 58.0
    camera.elevation = -13.0
    camera.distance = 2.35
    camera.lookat[:] = np.array([0.52, -0.56, 0.86], dtype=np.float64)
    return camera


def make_frames(model: mujoco.MjModel, fps: int) -> list[np.ndarray]:
    keyframes = make_keyframes()
    qpos_frames = [clipped_qpos(model, frame.qpos) for frame in keyframes]
    frames: list[np.ndarray] = [qpos_frames[0]]

    for start_qpos, end_qpos, keyframe in zip(qpos_frames, qpos_frames[1:], keyframes[1:]):
        steps = max(1, int(round(keyframe.duration * fps)))
        for step in range(steps):
            alpha = smoothstep((step + 1) / steps)
            frames.append((1.0 - alpha) * start_qpos + alpha * end_qpos)

    # Add a short final hold so the thumb-up pose is easy to inspect in the video.
    for _ in range(int(round(1.0 * fps))):
        frames.append(qpos_frames[-1])
    return frames


def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    qpos: np.ndarray,
) -> np.ndarray:
    apply_qpos(model, data, qpos)
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def save_previews(
    model: mujoco.MjModel,
    frames: list[np.ndarray],
    out_dir: Path,
    width: int,
    height: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = make_camera()
    try:
        preview_indices = [0, len(frames) // 2, len(frames) - 1]
        for i, frame_id in enumerate(preview_indices, start=1):
            image = render_frame(model, data, renderer, camera, frames[frame_id])
            imageio.imwrite(out_dir / f"preview_{i:02d}.png", image)
    finally:
        renderer.close()


def save_video(
    model: mujoco.MjModel,
    frames: list[np.ndarray],
    out_path: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = make_camera()
    try:
        with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for qpos in frames:
                writer.append_data(render_frame(model, data, renderer, camera, qpos))
    finally:
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the current arm-hand thumbs-up demo to MP4.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--previews-only", action="store_true")
    args = parser.parse_args()

    model = load_urdf_model(args.urdf)
    frames = make_frames(model, args.fps)
    save_previews(model, frames, args.preview_dir, args.width, args.height)
    print(f"preview_dir={args.preview_dir}")

    if not args.previews_only:
        save_video(model, frames, args.out, args.fps, args.width, args.height)
        print(f"video={args.out}")


if __name__ == "__main__":
    main()
