
import csv
import hashlib
import subprocess
import wave
from collections import Counter
from pathlib import Path

from preprocessing import (
    load_manifest,
    PROJECT_ROOT,
    SAMPLE_RATE,
)

CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "acoustic_16k"
CACHE_MANIFEST = PROJECT_ROOT / "data" / "acoustic_cache_manifest.csv"


def cache_path(source_path):
    identifier = hashlib.sha256(
        str(Path(source_path).resolve()).lower().encode("utf-8")
    ).hexdigest()[:20]

    return CACHE_DIR / f"{identifier}.wav"


def convert(source, destination):
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v", "error",
        "-i", source,
        "-map", "0:a:0",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(destination),
    ]

    subprocess.run(command, check=True)


def main():
    records = load_manifest(
    PROJECT_ROOT / "data" / "acoustic_unified_manifest_v2.csv"
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    output_rows = []
    converted = 0
    reused = 0

    for index, record in enumerate(records, 1):
        source = record["file_path"]
        destination = cache_path(source)

        if destination.is_file() and destination.stat().st_size > 44:
            reused += 1
        else:
            convert(source, destination)
            converted += 1

        with wave.open(str(destination), "rb") as wav:
            if wav.getframerate() != SAMPLE_RATE:
                raise ValueError(f"Invalid sample rate: {destination}")

            if wav.getnchannels() != 1:
                raise ValueError(f"Not mono: {destination}")

            if wav.getsampwidth() != 2:
                raise ValueError(f"Not PCM16: {destination}")

            duration = wav.getnframes() / SAMPLE_RATE

        if duration + 0.01 < record["end_sec"]:
            raise ValueError(
                f"Cached audio shorter than annotation: {source}"
            )

        output_rows.append({
            **record,
            "cache_path": str(destination.resolve()),
            "cache_duration_sec": round(duration, 6),
        })

        if index % 25 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)} recordings")

    fields = list(output_rows[0].keys())

    with CACHE_MANIFEST.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nCache complete")
    print("Converted:", converted)
    print("Reused:", reused)
    print("Entries:", len(output_rows))
    print("Sources:", dict(Counter(r["source"] for r in output_rows)))
    print("Cache manifest:", CACHE_MANIFEST)


if __name__ == "__main__":
    main()
