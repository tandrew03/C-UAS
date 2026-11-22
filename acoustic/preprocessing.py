
# C-UAS Acoustic Preprocessing
# Virginia Tech M.Eng Capstone
#
# Stage 1: Validated manifest loading

from __future__ import annotations

import csv
from pathlib import Path
from collections import Counter
import subprocess
from pathlib import PureWindowsPath

SAMPLE_RATE = 16000
N_SAMPLES = 8000
SPLITS = ("train", "val", "test")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_ROOT / "acoustic_unified_manifest_v3.csv"
REQUIRED_COLUMNS = {
    "file_path",
    "label",
    "source",
    "group_id",
    "start_sec",
    "end_sec",
    "split",
}


def resolve_audio_path(path):
    """
    Resolve audio paths on Windows or WSL.

    Converts C:\\Users\\... to /mnt/c/Users/...
    when running under Linux.
    """
    import os

    path = str(path)

    if os.name != "nt" and PureWindowsPath(path).drive:
        windows_path = PureWindowsPath(path)
        drive = windows_path.drive.rstrip(":").lower()

        return (
            Path("/mnt")
            / drive
            / Path(*windows_path.parts[1:])
        ).resolve()

    return Path(path).resolve()


def load_manifest(path=MANIFEST_PATH):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Manifest missing: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])

        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        rows = list(reader)

    # A relocated training dataset may retain original paths as
    # identifiers while reading audio exclusively from validated WAV caches.
    cache_manifest = DATA_ROOT / "acoustic_cache_manifest.csv"
    cached_sources = {}

    if cache_manifest.is_file():
        with cache_manifest.open(newline="", encoding="utf-8-sig") as f:
            for cache_row in csv.DictReader(f):
                original = str(
                    resolve_audio_path(cache_row["file_path"])
                ).lower()
                cached = resolve_audio_path(cache_row["cache_path"])
                cached_sources[original] = cached

    validated = []
    seen = set()

    for index, row in enumerate(rows, start=2):
        audio_path = resolve_audio_path(row["file_path"])

        if not audio_path.is_absolute():
            audio_path = PROJECT_ROOT / audio_path

        if not audio_path.is_file():
            # Training may use the cached WAV without its original file.
            # Never accept a missing original without a real cache file.
            cached = cached_sources.get(str(audio_path).lower())
            if cached is None or not cached.is_file():
                raise FileNotFoundError(
                    f"Row {index}: original missing and no valid cache "
                    f"available for {audio_path}"
                )

        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Row {index}: invalid label")

        start = float(row["start_sec"])
        end = float(row["end_sec"])

        if start < 0 or end <= start:
            raise ValueError(f"Row {index}: invalid interval")

        group = row["group_id"].strip()
        if not group:
            raise ValueError(f"Row {index}: missing group_id")

        split = row["split"].strip().lower()
        if split and split not in SPLITS:
            raise ValueError(f"Row {index}: invalid split")

        key = (str(audio_path.resolve()).lower(), start, end)

        if key in seen:
            raise ValueError(f"Row {index}: duplicate interval")

        seen.add(key)

        validated.append({
            "file_path": str(audio_path.resolve()),
            "label": label,
            "source": row["source"],
            "group_id": group,
            "start_sec": start,
            "end_sec": end,
            "split": split,
        })

    # Detect leakage if splits have already been assigned.
    group_splits = {}
    for row in validated:
        if not row["split"]:
            continue

        group_splits.setdefault(row["group_id"], set()).add(
            row["split"]
        )

    leaking = {
        group: splits
        for group, splits in group_splits.items()
        if len(splits) > 1
    }

    if leaking:
        raise ValueError(f"Group leakage detected: {leaking}")

    return validated


def create_window_index(records, window_sec=0.5):
    """
    Build fixed-length, non-overlapping audio windows.

    Each window inherits:
      - label
      - source
      - group_id
      - split

    Windows never extend beyond their annotated interval.
    """
    windows = []

    for record in records:
        start = record["start_sec"]
        end = record["end_sec"]

        n_windows = int((end - start + 1e-9) / window_sec)

        for i in range(n_windows):
            window_start = round(start + i * window_sec, 6)

            windows.append({
                "file_path": record["file_path"],
                "start_sec": window_start,
                "duration_sec": window_sec,
                "label": record["label"],
                "source": record["source"],
                "group_id": record["group_id"],
                "split": record["split"],
            })

    return windows


def decode_audio_window(window, sample_rate=SAMPLE_RATE):
    """
    Decode one indexed audio window using FFmpeg.

    Returns a list of float samples in [-1.0, 1.0].
    Multichannel input is downmixed to mono.
    """
    command = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-i", window["file_path"],
        "-ss", str(window["start_sec"]),
        "-t", str(window["duration_sec"]),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    from array import array
    import sys

    samples = array("f")
    samples.frombytes(result.stdout)

    if sys.byteorder != "little":
        samples.byteswap()

    expected = round(
        window["duration_sec"] * sample_rate
    )

    if len(samples) != expected:
        raise ValueError(
            f"Expected {expected} samples, "
            f"received {len(samples)} from "
            f"{window['file_path']}"
        )

    return samples



class DroneAudio:
    """
    PyTorch-compatible drone detection dataset.

    Reads 0.5-second windows from cached PCM16 WAVs.
    Original multichannel recordings remain untouched.
    """

    def __init__(self, windows, split=None):
        import wave

        if split is not None and split not in SPLITS:
            raise ValueError(f"Invalid split: {split}")

        cache_manifest = DATA_ROOT / "acoustic_cache_manifest.csv"

        with cache_manifest.open(
            newline="", encoding="utf-8-sig"
        ) as f:
            cache_rows = list(csv.DictReader(f))

        
        self.cache_lookup = {
            str(resolve_audio_path(r["file_path"])).lower():
                str(resolve_audio_path(r["cache_path"]))
            for r in cache_rows
        }


        self.windows = [
            w for w in windows
            if split is None or w["split"] == split
        ]

        # Verify cached files and their audio formats once.
        for path in set(self.cache_lookup.values()):
            with wave.open(path, "rb") as wav:
                if (
                    wav.getframerate() != SAMPLE_RATE
                    or wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                ):
                    raise ValueError(f"Invalid cache format: {path}")

        print(
            f"DroneAudio initialized: "
            f"{len(self.windows)} windows"
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        import torch
        import wave
        from array import array
        import sys

        window = self.windows[index]

        source = str(
            resolve_audio_path(window["file_path"])
        ).lower()

        if source not in self.cache_lookup:
            raise KeyError(f"Missing cached audio: {source}")

        cached_path = self.cache_lookup[source]

        start_frame = round(
            window["start_sec"] * SAMPLE_RATE
        )
        frames_needed = round(
            window["duration_sec"] * SAMPLE_RATE
        )

        with wave.open(cached_path, "rb") as wav:
            if start_frame + frames_needed > wav.getnframes():
                raise ValueError(
                    f"Window exceeds cached audio: {cached_path}"
                )

            wav.setpos(start_frame)
            raw = wav.readframes(frames_needed)

        samples = array("h")
        samples.frombytes(raw)

        if sys.byteorder != "little":
            samples.byteswap()

        if len(samples) != frames_needed:
            raise ValueError(
                f"Expected {frames_needed} samples, "
                f"received {len(samples)}"
            )

        audio = torch.tensor(
            samples, dtype=torch.float32
        ) / 32768.0

        label = torch.tensor(
            window["label"], dtype=torch.long
        )

        return {
            "audio": audio,
            "label": label,
            "group_id": window["group_id"],
            "source": window["source"],
        }


def create_balanced_sampler(dataset, seed=42):
    """
    Construct a class-balanced sampler for training.

    Uses inverse-frequency weights so each class
    has approximately equal sampling probability.

    Never use this sampler for validation or testing.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = [w["label"] for w in dataset.windows]
    counts = Counter(labels)

    if set(counts) != {0, 1}:
        raise ValueError(
            f"Training requires both classes: {dict(counts)}"
        )

    weights = torch.tensor(
        [1.0 / counts[label] for label in labels],
        dtype=torch.double,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


if __name__ == "__main__":
    records = load_manifest()
    windows = create_window_index(records)

    print("Recordings:", len(records))
    print("Windows:", len(windows))
    print("Window duration:", 0.5, "seconds")

    print("Window labels:",
          dict(Counter(w["label"] for w in windows)))

    print("Windows by source:",
          dict(Counter(w["source"] for w in windows)))

    print("Recording groups:",
          len(set(w["group_id"] for w in windows)))

    print("Example window:", windows[0])
