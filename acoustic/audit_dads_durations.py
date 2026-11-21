import csv
import wave
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq

SOURCE = Path("data/dads_parquet_index.csv")
OUTPUT = Path("data/dads_parquet_index_with_durations.csv")
MIN_SAMPLES = 8000

with SOURCE.open(newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    records = list(reader)

groups = defaultdict(list)
for i, r in enumerate(records):
    key = (r["parquet_path"], int(r["row_group"]))
    groups[key].append((i, int(r["row_index"])))

sample_counts = [None] * len(records)
current_path = None
parquet = None

for number, ((path, group_id), entries) in enumerate(
    sorted(groups.items()), 1
):
    if path != current_path:
        parquet = pq.ParquetFile(path)
        current_path = path

    audio_rows = parquet.read_row_group(
        group_id, columns=["audio"]
    ).column("audio").to_pylist()

    for index, row_index in entries:
        with wave.open(
            BytesIO(audio_rows[row_index]["bytes"]), "rb"
        ) as wav:
            if wav.getframerate() != 16000:
                raise ValueError("Unexpected sample rate")
            sample_counts[index] = wav.getnframes()

    del audio_rows

    if number % 20 == 0:
        print(
            f"Processed {number}/{len(groups)} row groups",
            flush=True
        )

assert all(n is not None for n in sample_counts)

summary = Counter()
bins = defaultdict(Counter)

for r, n in zip(records, sample_counts):
    label = r["label"]
    category = "short" if n < MIN_SAMPLES else "eligible"
    summary[(label, category)] += 1

    duration = n / 16000
    if duration < 0.1:
        bucket = "<0.1s"
    elif duration < 0.25:
        bucket = "0.1-0.25s"
    elif duration < 0.5:
        bucket = "0.25-0.5s"
    else:
        bucket = ">=0.5s"

    bins[label][bucket] += 1

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=fieldnames + ["n_samples", "duration_sec"]
    )
    writer.writeheader()
    for r, n in zip(records, sample_counts):
        writer.writerow({
            **r,
            "n_samples": n,
            "duration_sec": f"{n / 16000:.6f}"
        })

print("\n===== DADS DURATION AUDIT =====")
for label in ("0", "1"):
    name = "background" if label == "0" else "drone"
    print(f"\n{name.upper()}")
    for bucket, count in sorted(bins[label].items()):
        print(f"  {bucket}: {count}")
    print("  Eligible:", summary[(label, "eligible")])
    print("  Short:", summary[(label, "short")])

print("\nTotal eligible:", sum(
    summary[(label, "eligible")] for label in ("0", "1")
))
print("Total short:", sum(
    summary[(label, "short")] for label in ("0", "1")
))
print("Saved:", OUTPUT)
print("PASS: DADS duration audit complete")
