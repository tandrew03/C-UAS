
import csv
import random
import re
import wave
from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA / "cache" / "dads_16k"
MANIFEST = DATA / "dads_subset_manifest.csv"

DADS_ROOT = Path(
    "/mnt/c/Users/andre/.cache/huggingface/hub/"
    "datasets--geronimobasso--drone-audio-detection-samples/"
    "snapshots"
)

SEED = 42
PER_CLASS = 4000
MIN_SAMPLES = 8000

FIELDS = [
    "file_path", "label", "source", "group_id",
    "start_sec", "end_sec", "split", "notes"
]


def inspect_wav(data):
    with wave.open(BytesIO(data), "rb") as wav:
        if (
            wav.getframerate() != 16000
            or wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or wav.getcomptype() != "NONE"
        ):
            return None

        frames = wav.getnframes()
        return frames if frames >= MIN_SAMPLES else None


def iter_records(files):
    for filename in files:
        pf = pq.ParquetFile(filename)

        for batch in pf.iter_batches(
            batch_size=256,
            columns=["audio", "label"]
        ):
            for row in batch.to_pylist():
                yield row


def main():
    files = sorted(DADS_ROOT.glob("*/data/*.parquet"))
    if not files:
        raise FileNotFoundError("DADS Parquet files missing")

    print("Scanning eligible DADS recordings...", flush=True)

    eligible = {0: [], 1: []}
    skipped = Counter()

    # First pass: collect filenames only, not audio data.
    for row in iter_records(files):
        label = int(row["label"])
        audio = row["audio"]
        name = audio["path"]

        if label not in eligible:
            raise ValueError(f"Unexpected label: {label}")

        if (
            not re.fullmatch(r"(?:drone|no-drone)-\d+\.wav", name)
            or audio["bytes"] is None
        ):
            skipped["invalid_name_or_data"] += 1
            continue

        frames = inspect_wav(audio["bytes"])
        if frames is None:
            skipped["short_or_unsupported"] += 1
            continue

        eligible[label].append(name)

    rng = random.Random(SEED)
    selected = set()

    for label in (0, 1):
        available = eligible[label]
        if len(available) < PER_CLASS:
            raise ValueError(
                f"Class {label}: only {len(available)} eligible "
                f"recordings; need {PER_CLASS}"
            )

        selected.update(rng.sample(available, PER_CLASS))

    print(
        "Eligible:", {k: len(v) for k, v in eligible.items()},
        flush=True,
    )
    print("Skipped:", dict(skipped), flush=True)
    print("Selected:", len(selected), flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # Second pass: write only selected audio.
    for row in iter_records(files):
        audio = row["audio"]
        name = audio["path"]

        if name not in selected:
            continue

        label = int(row["label"])
        frames = inspect_wav(audio["bytes"])

        if frames is None:
            raise ValueError(f"Invalid selected audio: {name}")

        destination = OUTPUT_DIR / name

        # Original encoded PCM WAV; no conversion necessary.
        if not destination.exists():
            destination.write_bytes(audio["bytes"])

        rows.append({
            "file_path": str(destination.relative_to(PROJECT_ROOT)),
            "label": label,
            "source": "dads",
            # Unique ID for tracking; NOT proof of independence.
            "group_id": f"dads_{name.removesuffix('.wav')}",
            "start_sec": 0,
            "end_sec": f"{frames / 16000:.6f}",
            "split": "train",
            "notes": "DADS training only; provenance unverified",
        })

        selected.remove(name)

        if len(rows) % 1000 == 0:
            print(f"Extracted {len(rows)} recordings", flush=True)

    if selected:
        raise ValueError(
            f"Failed to extract {len(selected)} selected recordings"
        )

    rows.sort(key=lambda r: (r["label"], r["file_path"]))

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(r["label"] for r in rows)

    print("\nDADS extraction complete")
    print("Extracted:", len(rows))
    print("Labels:", dict(counts))
    print("Manifest:", MANIFEST)

    assert counts == {0: PER_CLASS, 1: PER_CLASS}
    print("PASS: DADS training subset created.")


if __name__ == "__main__":
    main()
