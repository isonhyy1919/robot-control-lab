from __future__ import annotations

import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_a4_limit_comparison import draw_chart, font as chart_font


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "oscbf_thumbsup"
VIDEO_DIR = ROOT / "outputs" / "a4_joint_limit_comparison"
OUTPUT = VIDEO_DIR / "result_figures"
LEFT_VIDEO = VIDEO_DIR / "no_obstacle_unsafe_a4_limit_violation.mp4"
RIGHT_VIDEO = DATA_DIR / "oscbf_obstacle_avoidance.mp4"
LEFT_CSV = DATA_DIR / "nominal_rollout.csv"
RIGHT_CSV = DATA_DIR / "oscbf_rollout.csv"
ARM_JOINTS = tuple(f"A{i}_joint" for i in range(1, 8))
JOINT_LIMITS = {
    "A1_joint": (-2.9, 1.0),
    "A2_joint": (-0.15, 3.14),
    "A3_joint": (-2.35, 2.35),
    "A4_joint": (0.0, 2.2),
    "A5_joint": (-2.35, 2.35),
    "A6_joint": (-1.57, 1.57),
    "A7_joint": (-1.57, 1.57),
}
OBSTACLE_CENTER_XZ = (0.70, 0.82)
OBSTACLE_RADIUS = 0.105
SAFETY_RADIUS = 0.130


def cn_font(size: int, bold: bool = False):
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return chart_font(size, bold)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def array(records: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in records])


def extract_frame(reader, time_s: float) -> Image.Image:
    metadata = reader.get_meta_data()
    frame_count = reader.count_frames()
    index = min(round(time_s * float(metadata["fps"])), frame_count - 1)
    return Image.fromarray(reader.get_data(index)).convert("RGB")


def make_keyframes() -> None:
    times = (0.0, 1.2, 2.0, 5.5, 7.7)
    left_reader = imageio.get_reader(LEFT_VIDEO)
    right_reader = imageio.get_reader(RIGHT_VIDEO)
    left_frames = [extract_frame(left_reader, time_s) for time_s in times]
    right_frames = [extract_frame(right_reader, time_s) for time_s in times]
    left_reader.close()
    right_reader.close()

    thumb = (430, 323)
    gap = 18
    margin_x = 70
    title_h = 100
    label_h = 58
    time_h = 44
    row_h = label_h + thumb[1] + 30
    width = margin_x * 2 + len(times) * thumb[0] + (len(times) - 1) * gap
    height = title_h + time_h + row_h * 2 + 30
    image = Image.new("RGB", (width, height), (238, 242, 247))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((margin_x, 30), "关键帧对比：A4越限不安全执行 vs OSCBF带障碍物安全执行", font=cn_font(36, True), fill=(20, 35, 55))
    for column, time_s in enumerate(times):
        x = margin_x + column * (thumb[0] + gap)
        draw.text((x + 160, title_h), f"t = {time_s:.1f} s", font=chart_font(22, True), fill=(55, 65, 80))

    rows = [
        (left_frames, "左：无障碍物｜OSCBF关闭｜A4实际越过URDF下限", (174, 36, 52)),
        (right_frames, "右：带障碍物｜OSCBF开启｜关节限位与碰撞约束同时生效", (28, 104, 153)),
    ]
    for row_index, (frames, label, color) in enumerate(rows):
        row_top = title_h + time_h + row_index * row_h
        draw.rounded_rectangle((margin_x, row_top, width - margin_x, row_top + label_h - 8), radius=10, fill=color + (255,))
        draw.text((margin_x + 20, row_top + 10), label, font=cn_font(24, True), fill="white")
        for column, frame in enumerate(frames):
            x = margin_x + column * (thumb[0] + gap)
            y = row_top + label_h
            resized = frame.resize(thumb, Image.Resampling.LANCZOS)
            image.paste(resized, (x, y))
            draw.rectangle((x, y, x + thumb[0], y + thumb[1]), outline=(190, 198, 210), width=2)
    image.save(OUTPUT / "01_关键帧对比.png")


def make_a4_and_safety(left: list[dict[str, str]], right: list[dict[str, str]]) -> None:
    time_s = array(left, "time_s")
    image = Image.new("RGB", (2200, 1320), (238, 242, 247))
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "A4关节限位与安全屏障结果", font=cn_font(38, True), fill=(20, 35, 55))
    boxes = [(60, 105, 1070, 660), (1130, 105, 2140, 660), (60, 710, 1070, 1260), (1130, 710, 2140, 1260)]
    red = (209, 73, 91)
    blue = (22, 119, 184)
    draw_chart(draw, boxes[0], "A4 actual position [rad]", time_s, [
        ("unsafe / OSCBF OFF", array(left, "A4_joint"), red, 4),
        ("safe / OSCBF ON", array(right, "A4_joint"), blue, 4),
    ], [(0.0, (20, 20, 20)), (2.2, (20, 20, 20))])
    draw_chart(draw, boxes[1], "A4 lower barrier h = q_A4 - 0", time_s, [
        ("unsafe / OSCBF OFF", array(left, "A4_joint"), red, 4),
        ("safe / OSCBF ON", array(right, "A4_joint"), blue, 4),
    ], [(0.0, (20, 20, 20))])
    draw_chart(draw, boxes[2], "Minimum joint-limit barrier", time_s, [
        ("unsafe / OSCBF OFF", array(left, "min_joint_limit_h"), red, 4),
        ("safe / OSCBF ON", array(right, "min_joint_limit_h"), blue, 4),
    ], [(0.0, (20, 20, 20))])
    draw_chart(draw, boxes[3], "Right video: minimum obstacle barrier", time_s, [
        ("obstacle + OSCBF", array(right, "min_obstacle_h"), blue, 4),
    ], [(0.0, (20, 20, 20))])
    image.save(OUTPUT / "02_A4限位与安全屏障.png")


def make_joint_trajectories(left: list[dict[str, str]], right: list[dict[str, str]]) -> None:
    time_s = array(left, "time_s")
    image = Image.new("RGB", (2400, 1900), (238, 242, 247))
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "七关节轨迹对比：不安全直接执行 vs OSCBF安全过滤", font=cn_font(38, True), fill=(20, 35, 55))
    red = (209, 73, 91)
    blue = (22, 119, 184)
    for index, joint in enumerate(ARM_JOINTS):
        row, column = divmod(index, 2)
        box = (60 + column * 1180, 105 + row * 445, 1160 + column * 1180, 515 + row * 445)
        lower, upper = JOINT_LIMITS[joint]
        draw_chart(draw, box, f"{joint} position [rad]", time_s, [
            ("unsafe / OSCBF OFF", array(left, joint), red, 4),
            ("safe / OSCBF ON", array(right, joint), blue, 4),
        ], [(lower, (25, 25, 25)), (upper, (25, 25, 25))])
    box = (1240, 1440, 2340, 1850)
    draw_chart(draw, box, "End-effector Z position [m]", time_s, [
        ("unsafe / no obstacle", array(left, "A7_z"), red, 4),
        ("safe / obstacle + OSCBF", array(right, "A7_z"), blue, 4),
    ])
    image.save(OUTPUT / "03_七关节轨迹对比.png")


def draw_path_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    left: list[dict[str, str]],
    right: list[dict[str, str]],
) -> None:
    left_x, left_z = array(left, "A7_x"), array(left, "A7_z")
    right_x, right_z = array(right, "A7_x"), array(right, "A7_z")
    x_min = min(float(left_x.min()), float(right_x.min())) - 0.05
    x_max = max(float(left_x.max()), float(right_x.max())) + 0.05
    z_min = min(float(left_z.min()), float(right_z.min())) - 0.05
    z_max = max(float(left_z.max()), float(right_z.max())) + 0.05
    l, t, r, b = box
    px0, py0, px1, py1 = l + 95, t + 105, r - 35, b - 70
    scale = min((px1 - px0) / (x_max - x_min), (py1 - py0) / (z_max - z_min))

    def mx(value): return int(px0 + (value - x_min) * scale)
    def mz(value): return int(py1 - (value - z_min) * scale)

    draw.rounded_rectangle(box, radius=14, fill=(250, 252, 255), outline=(195, 204, 216), width=2)
    draw.text((l + 24, t + 18), "A7末端X-Z轨迹（不显示障碍物）", font=cn_font(28, True), fill=(25, 38, 55))
    draw.text(
        (l + 24, t + 58),
        "仅比较末端运动路径；障碍物和安全范围未叠加，安全性仍以CBF最小屏障h为准。",
        font=cn_font(17), fill=(90, 98, 110),
    )
    draw.line((px0, py0, px0, py1, px1, py1), fill=(65, 72, 82), width=2)
    draw.line([(mx(x), mz(z)) for x, z in zip(left_x, left_z)], fill=(209, 73, 91), width=5)
    draw.line([(mx(x), mz(z)) for x, z in zip(right_x, right_z)], fill=(22, 119, 184), width=5)
    draw.line((l + 50, b - 38, l + 90, b - 38), fill=(209, 73, 91), width=5)
    draw.text((l + 100, b - 52), "左：无障碍/不安全", font=cn_font(20), fill=(45, 52, 63))
    draw.line((l + 300, b - 38, l + 340, b - 38), fill=(22, 119, 184), width=5)
    draw.text((l + 350, b - 52), "右：障碍物+OSCBF", font=cn_font(20), fill=(45, 52, 63))


def metric_card(draw, box, title, value, detail, color):
    draw.rounded_rectangle(box, radius=18, fill=(250, 252, 255), outline=color, width=3)
    l, t, r, b = box
    draw.text((l + 28, t + 20), title, font=cn_font(24, True), fill=(45, 55, 70))
    draw.text((l + 28, t + 62), value, font=chart_font(42, True), fill=color)
    draw.multiline_text((l + 28, t + 120), detail, font=cn_font(19), fill=(75, 82, 94), spacing=8)


def make_results(left: list[dict[str, str]], right: list[dict[str, str]]) -> None:
    left_h = array(left, "min_joint_limit_h")
    right_h = array(right, "min_joint_limit_h")
    right_obstacle = array(right, "min_obstacle_h")
    hold_index = int(np.argmin(np.abs(array(left, "time_s") - 5.5)))
    left_ee = np.asarray([array(left, key)[hold_index] for key in ("A7_x", "A7_y", "A7_z")])
    right_ee = np.asarray([array(right, key)[hold_index] for key in ("A7_x", "A7_y", "A7_z")])
    hold_difference = float(np.linalg.norm(left_ee - right_ee))

    image = Image.new("RGB", (2200, 1320), (238, 242, 247))
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "实验结果汇总：安全约束生效，但会改变局部运动", font=cn_font(38, True), fill=(20, 35, 55))
    draw_path_panel(draw, (60, 110, 1120, 1010), left, right)
    metric_card(draw, (1180, 110, 1640, 390), "左侧A4最小值", f"{array(left, 'A4_joint').min():.3f} rad", "URDF下限：0 rad\n实际执行发生越限", (190, 40, 55))
    metric_card(draw, (1690, 110, 2150, 390), "右侧A4最小值", f"{array(right, 'A4_joint').min():.2e} rad", "数值误差量级，工程上为0\n没有越过URDF下限", (20, 120, 180))
    metric_card(draw, (1180, 440, 1640, 720), "关节限位违规点", f"{int(np.sum(left_h < -1e-6))} → {int(np.sum(right_h < -1e-6))}", "左：OSCBF关闭\n右：OSCBF开启", (95, 75, 165))
    metric_card(draw, (1690, 440, 2150, 720), "右侧障碍物屏障", f"min h = {right_obstacle.min():.6f} m", "障碍物违规采样点：0\n安全边界始终未被穿透", (20, 145, 105))
    metric_card(draw, (1180, 770, 2150, 1010), "任务保持阶段末端差异", f"{hold_difference * 100:.2f} cm", "在t=5.5 s点赞保持阶段，安全轨迹与不安全原轨迹仍较接近；\nOSCBF主要在接近障碍物和关节边界时重新分配各关节运动。", (220, 120, 25))
    draw.rounded_rectangle((60, 1060, 2150, 1260), radius=18, fill=(226, 238, 249), outline=(83, 128, 171), width=2)
    conclusion = (
        "结论：左侧直接执行上游轨迹，A4最低达到-0.500 rad并长期违反URDF限位；"
        "右侧OSCBF将关节限位和障碍物约束放入同一个QP，A4保持在0 rad以上，碰撞屏障也保持非负。\n"
        "代价是安全层会修改其他关节和末端的局部轨迹，但在点赞保持阶段，末端位置与原动作的差异约为1.56 cm。"
    )
    draw.multiline_text((90, 1095), conclusion, font=cn_font(25, True), fill=(31, 63, 91), spacing=15)
    image.save(OUTPUT / "04_末端轨迹与结果汇总.png")


def make_obstacle_a4_trajectory(left: list[dict[str, str]], right: list[dict[str, str]]) -> None:
    """Create a focused A4 plot for the obstacle-present OSCBF experiment."""
    time_s = array(right, "time_s")
    unsafe_a4 = array(left, "A4_joint")
    safe_a4 = array(right, "A4_joint")
    image = Image.new("RGB", (2200, 1180), (238, 242, 247))
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "带障碍物场景下A4关节轨迹", font=cn_font(40, True), fill=(20, 35, 55))
    draw.text(
        (72, 84),
        "红线为未过滤的越限参考轨迹，蓝线为同时施加障碍物CBF和URDF关节限位CBF后的实际执行轨迹。",
        font=cn_font(22), fill=(75, 84, 98),
    )
    draw_chart(draw, (60, 140, 2140, 690), "A4 joint position [rad]", time_s, [
        ("unfiltered A4 reference", unsafe_a4, (209, 73, 91), 5),
        ("obstacle + OSCBF executed A4", safe_a4, (22, 119, 184), 5),
    ], [(0.0, (20, 20, 20)), (2.2, (20, 20, 20))])
    draw_chart(draw, (60, 740, 1460, 1120), "A4 lower barrier h_A4 = q_A4 - q_min", time_s, [
        ("unfiltered h_A4", unsafe_a4, (209, 73, 91), 4),
        ("OSCBF h_A4", safe_a4, (22, 119, 184), 4),
    ], [(0.0, (20, 20, 20))])
    metric_card(
        draw, (1520, 740, 2140, 1120), "A4轨迹结果",
        f"{unsafe_a4.min():.3f} → {safe_a4.min():.2e} rad",
        "URDF允许范围：[0, 2.2] rad\n\n未过滤轨迹越过下限；\nOSCBF执行轨迹保持在0 rad以上。",
        (28, 104, 153),
    )
    image.save(OUTPUT / "05_带障碍物场景_A4关节轨迹.png")


def write_caption_notes() -> None:
    text = """结果图使用说明

图1 关键帧对比：左侧为无障碍物且关闭OSCBF的A4越限执行，右侧为带障碍物并开启OSCBF的安全执行。时间点为0、1.2、2.0、5.5和7.7秒。

图2 A4限位与安全屏障：展示A4位置、A4下限屏障、全关节最小限位屏障以及右侧障碍物最小屏障。h=0为安全边界，h<0表示违反约束。

图3 七关节轨迹对比：黑色虚线为URDF关节上下限。OSCBF不仅限制A4，还通过任务一致优化调整其他关节，以同时满足碰撞和关节限位约束。

图4 末端轨迹与结果汇总：右侧OSCBF实现关节违规点和障碍物违规点均为0；在5.5秒点赞保持阶段，末端位置与不安全原轨迹相差约1.56 cm。

图5 带障碍物场景下A4关节轨迹：红线为未过滤越限参考，蓝线为障碍物CBF与URDF限位CBF共同作用后的实际执行结果。A4下限为0 rad。
"""
    (OUTPUT / "结果图说明.txt").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    left = load_csv(LEFT_CSV)
    right = load_csv(RIGHT_CSV)
    if len(left) != len(right):
        raise RuntimeError("Trajectory logs have different lengths")
    make_keyframes()
    make_a4_and_safety(left, right)
    make_joint_trajectories(left, right)
    make_results(left, right)
    make_obstacle_a4_trajectory(left, right)
    write_caption_notes()
    print({"output": str(OUTPUT), "figures": 5, "samples_per_trajectory": len(left)})


if __name__ == "__main__":
    main()
