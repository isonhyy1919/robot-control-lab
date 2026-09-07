from __future__ import annotations

import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "joint_limit_comparison"
SUMMARY = OUTPUT / "joint_limit_comparison_summary.json"
VALID_CSV = OUTPUT / "oscbf_valid_upstream_rollout.csv"
INVALID_CSV = OUTPUT / "oscbf_invalid_a4_rollout.csv"
VIDEO = OUTPUT / "a4_joint_limit_comparison.mp4"
CONTACT_SHEET = OUTPUT / "a4_joint_limit_video_contact_sheet.png"


report = json.loads(SUMMARY.read_text(encoding="utf-8"))
valid_summary = report["valid_upstream"]
invalid_summary = report["invalid_a4_upstream"]
assert report["upstream_a4_ranges_rad"]["valid"] == [0.0, 0.0]
assert report["upstream_a4_ranges_rad"]["invalid"][0] == -0.5
assert valid_summary["joint_limit_unsafe_steps"] == 0
assert invalid_summary["joint_limit_unsafe_steps"] == 0
assert invalid_summary["a4_desired_min_rad"] == -0.5
assert invalid_summary["a4_actual_min_rad"] >= -1e-9
assert invalid_summary["maximum_a4_filter_correction_rad_s"] > 1.0
assert invalid_summary["maximum_non_a4_filter_correction_rad_s"] > 0.1
assert invalid_summary["maximum_qp_constraint_residual"] < 1e-8


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


valid_rows = read_rows(VALID_CSV)
invalid_rows = read_rows(INVALID_CSV)
assert len(valid_rows) == len(invalid_rows) == 1341
required_columns = {
    "A4_joint",
    "A4_joint_desired",
    "A4_joint_nominal_velocity",
    "A4_joint_command_velocity",
    "A4_joint_lower_h",
    "A4_joint_upper_h",
}
assert required_columns.issubset(invalid_rows[0])
assert min(float(row["A4_joint_desired"]) for row in invalid_rows) == -0.5
assert min(float(row["A4_joint"]) for row in invalid_rows) >= -1e-9
assert min(float(row["A4_joint_lower_h"]) for row in invalid_rows) >= -1e-9

reader = imageio.get_reader(VIDEO)
metadata = reader.get_meta_data()
fps = float(metadata["fps"])
assert metadata["size"] == (1280, 480)
assert abs(fps - 30.0) < 1e-9
sample_times = (0.0, 1.2, 2.2, 4.5, 6.7)
frames = []
for time_s in sample_times:
    frame = Image.fromarray(reader.get_data(round(time_s * fps)))
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rounded_rectangle((1120, 442, 1266, 474), radius=6, fill=(0, 0, 0, 170))
    draw.text((1134, 451), f"t = {time_s:.1f} s", fill="white")
    frames.append(frame)
reader.close()
sheet = Image.new("RGB", (1280, 480 * len(frames)), "white")
for index, frame in enumerate(frames):
    sheet.paste(frame, (0, 480 * index))
sheet.save(CONTACT_SHEET)

print(
    {
        "samples_per_rollout": len(valid_rows),
        "video_fps": fps,
        "video_size": metadata["size"],
        "video_duration_s": metadata.get("duration"),
        "invalid_a4_reference_min_rad": invalid_summary["a4_desired_min_rad"],
        "invalid_a4_executed_min_rad": invalid_summary["a4_actual_min_rad"],
        "unsafe_steps": invalid_summary["joint_limit_unsafe_steps"],
        "contact_sheet": str(CONTACT_SHEET),
    }
)
