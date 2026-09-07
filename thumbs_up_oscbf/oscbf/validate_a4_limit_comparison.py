from __future__ import annotations

import csv
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "a4_joint_limit_comparison"
VIDEOS = {
    "valid": OUTPUT / "valid_upstream_with_obstacle.mp4",
    "violating": OUTPUT / "a4_violating_upstream_with_obstacle.mp4",
    "comparison": OUTPUT / "a4_limit_side_by_side_comparison.mp4",
}


metadata = {}
for name, path in VIDEOS.items():
    reader = imageio.get_reader(path)
    info = reader.get_meta_data()
    metadata[name] = {
        "fps": float(info["fps"]),
        "size": tuple(info["size"]),
        "duration_s": float(info["duration"]),
        "frames": int(reader.count_frames()),
    }
    reader.close()

assert metadata["valid"]["fps"] == metadata["violating"]["fps"] == 30.0
assert metadata["valid"]["size"] == metadata["violating"]["size"] == (640, 480)
assert metadata["comparison"]["size"] == (1280, 530)
assert metadata["valid"]["frames"] == metadata["violating"]["frames"]

with (OUTPUT / "comparison_summary.json").open(encoding="utf-8") as stream:
    summary = json.load(stream)
assert summary["valid_upstream"]["upstream_a4_min_rad"] >= 0.0
assert summary["a4_violating_upstream"]["upstream_a4_min_rad"] == -0.5
assert summary["a4_violating_upstream"]["executed_a4_min_rad"] > -1e-6
assert summary["a4_violating_upstream"]["joint_limit_unsafe_steps"] == 0
assert summary["a4_violating_upstream"]["obstacle_unsafe_steps"] == 0

csv_rows = {}
for name in ("valid_upstream_oscbf_rollout.csv", "a4_violating_upstream_oscbf_rollout.csv"):
    with (OUTPUT / name).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    csv_rows[name] = len(rows)
    assert len(rows) == metadata["valid"]["frames"] * 0 + 1341
assert len(set(csv_rows.values())) == 1

reader = imageio.get_reader(VIDEOS["comparison"])
fps = metadata["comparison"]["fps"]
times = (0.0, 1.2, 2.0, 5.5, 6.7)
frames = []
for time_s in times:
    frame_index = min(round(time_s * fps), metadata["comparison"]["frames"] - 1)
    frame = Image.fromarray(reader.get_data(frame_index)).convert("RGB")
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rounded_rectangle((1134, 488, 1268, 518), radius=6, fill=(0, 0, 0, 165))
    draw.text((1145, 496), f"t = {time_s:.1f} s", fill="white")
    frames.append(frame)
reader.close()

sheet = Image.new("RGB", (1280 * len(frames), 530), "white")
for index, frame in enumerate(frames):
    sheet.paste(frame, (1280 * index, 0))
sheet_path = OUTPUT / "a4_limit_comparison_contact_sheet.png"
sheet.save(sheet_path)

print({"metadata": metadata, "csv_rows": csv_rows, "contact_sheet": str(sheet_path)})
