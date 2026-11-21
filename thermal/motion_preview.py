import subprocess
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

video = Path(
    "data/raw/Nov19_Multimodal/"
    "20261119_135653_drone_approaching/thermal.mkv"
)

raw = subprocess.check_output([
    "ffmpeg", "-v", "error",
    "-i", str(video),
    "-f", "rawvideo",
    "-pix_fmt", "gray16le", "-"
])

frames = np.frombuffer(raw, dtype="<u2").reshape(-1, 120, 160)
print("Decoded frames:", len(frames))

# Median background suppresses transient objects.
background = np.median(frames, axis=0)

times = [3, 9, 15, 21, 27]
canvas = Image.new("RGB", (480 * 5, 360), "black")
draw = ImageDraw.Draw(canvas)

for col, second in enumerate(times):
    idx = min(round(second * 9), len(frames) - 1)
    frame = frames[idx].astype(np.float32)

    # Only display pixels hotter/brighter than the
    # estimated background.
    difference = np.maximum(frame - background, 0)

    upper = max(np.percentile(difference, 99.9), 1)
    preview = np.uint8(
        np.clip(difference / upper * 255, 0, 255)
    )

    image = Image.fromarray(preview).resize(
        (480, 330), Image.Resampling.NEAREST
    ).convert("RGB")

    canvas.paste(image, (col * 480, 0))
    draw.text((col * 480 + 12, 337), f"{second} seconds", fill="white")

output = Path("data/thermal/previews/approaching_motion.png")
output.parent.mkdir(parents=True, exist_ok=True)
canvas.save(output)

print("PASS: Motion preview generated")
print("Saved:", output)
