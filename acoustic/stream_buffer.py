import argparse
import wave
from collections import deque
from pathlib import Path

import numpy as np


def stream_wav(path, window_sec=0.5, hop_sec=0.1):
    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()

        if width != 4:
            raise ValueError(f"Expected 32-bit PCM, got {width * 8}-bit")
        if channels != 16:
            raise ValueError(f"Expected 16 channels, got {channels}")

        window_samples = round(sr * window_sec)
        hop_samples = round(sr * hop_sec)

        buffer = deque(maxlen=window_samples)
        samples_seen = 0

        while True:
            raw = wav.readframes(hop_samples)
            if not raw:
                break

            chunk = np.frombuffer(raw, dtype="<i4")
            if chunk.size % channels:
                raise ValueError("Incomplete multichannel frame")

            chunk = chunk.reshape(-1, channels)
            buffer.extend(chunk)
            samples_seen += len(chunk)

            if len(buffer) == window_samples:
                window = np.asarray(buffer, dtype=np.int32)
                yield {
                    "start_sec": (samples_seen - window_samples) / sr,
                    "end_sec": samples_seen / sr,
                    "audio": window.copy(),
                    "sample_rate": sr,
                    "channels": channels,
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    args = parser.parse_args()

    count = 0
    first = last = None

    for event in stream_wav(args.wav):
        count += 1
        if first is None:
            first = event
        last = event

    print("Windows:", count)
    if first:
        print("First:", first["start_sec"], first["end_sec"])
        print("Last:", last["start_sec"], last["end_sec"])
        print("Window shape:", first["audio"].shape)
        print("Sample rate:", first["sample_rate"])
