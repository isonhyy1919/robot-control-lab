from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "a4_joint_limit_comparison" / "result_figures"
TOP = FIGURE_DIR / "04_末端轨迹与结果汇总.png"
BOTTOM = FIGURE_DIR / "05_带障碍物场景_A4关节轨迹.png"
OUTPUT = FIGURE_DIR / "06_末端轨迹与A4关节轨迹_拼接图.png"
GAP = 36


top = Image.open(TOP).convert("RGB")
bottom = Image.open(BOTTOM).convert("RGB")
if top.width != bottom.width:
    raise RuntimeError(f"Figure widths differ: {top.width} vs {bottom.width}")

canvas = Image.new("RGB", (top.width, top.height + GAP + bottom.height), (224, 230, 238))
canvas.paste(top, (0, 0))
canvas.paste(bottom, (0, top.height + GAP))
draw = ImageDraw.Draw(canvas)
separator_y = top.height + GAP // 2
draw.line((70, separator_y, top.width - 70, separator_y), fill=(145, 157, 173), width=3)
canvas.save(OUTPUT, optimize=True)

print({"output": str(OUTPUT), "size": canvas.size})
