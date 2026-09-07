from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "outputs" / "oscbf_thumbsup" / "original_thumbsup_same_view.mp4"
CONTACT_SHEET = ROOT / "outputs" / "oscbf_thumbsup" / "original_thumbsup_contact_sheet.png"
TIMES = (0.0, 1.2, 2.0, 5.5, 7.7)


reader = imageio.get_reader(VIDEO)
metadata = reader.get_meta_data()
fps = float(metadata["fps"])
frames = []
for time_s in TIMES:
    frame = reader.get_data(round(time_s * fps))
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((494, 444, 628, 472), radius=6, fill=(0, 0, 0, 150))
    draw.text((506, 451), f"t = {time_s:.1f} s", fill=(255, 255, 255, 255))
    frames.append(image)
reader.close()

sheet = Image.new("RGB", (640 * len(frames), 480), "white")
for index, frame in enumerate(frames):
    sheet.paste(frame, (640 * index, 0))
sheet.save(CONTACT_SHEET)

print(
    {
        "video": str(VIDEO),
        "fps": fps,
        "size": metadata.get("size"),
        "duration_s": metadata.get("duration"),
        "contact_sheet": str(CONTACT_SHEET),
    }
)
