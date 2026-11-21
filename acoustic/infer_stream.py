import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acoustic.model import DroneDetectionModel
from acoustic.stream_buffer import stream_wav


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    model = DroneDetectionModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Checkpoint validation:", checkpoint.get("val", "unavailable"))
    return model


@torch.inference_mode()
def run_inference(model, wav_path, device, threshold=0.5, channel=None):
    results = []

    for window in stream_wav(wav_path):
        audio = window["audio"].astype(np.float32)

        # Convert signed 32-bit PCM to [-1, 1].
        audio /= 2147483648.0

        # Preserve all channels in stream_buffer.
        # Downmix only for the classification model.
        mono = audio.mean(axis=1) if channel is None else audio[:, channel]

        # 48 kHz -> 16 kHz
        resampled = resample_poly(mono, up=1, down=3)

        if len(resampled) != 8000:
            raise ValueError(
                f"Expected 8000 samples, got {len(resampled)}"
            )

        x = torch.from_numpy(
            resampled.astype(np.float32)
        ).unsqueeze(0).to(device)

        lengths = torch.tensor(
            [x.shape[1]], device=device, dtype=torch.long
        )

        start = time.perf_counter()
        logit = model(x, lengths).reshape(-1)
        probability = torch.sigmoid(logit).item()
        latency_ms = (time.perf_counter() - start) * 1000

        results.append({
            "start_sec": round(window["start_sec"], 3),
            "end_sec": round(window["end_sec"], 3),
            "probability": round(probability, 6),
            "detected": int(probability >= threshold),
            "latency_ms": round(latency_ms, 3),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/v4/best.pt"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--channel", type=int, choices=range(16),
                        default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/acoustic_stream_results.csv"
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    model = load_model(args.checkpoint, device)

    start = time.perf_counter()
    results = run_inference(
        model, args.wav, device, args.threshold, args.channel
    )
    elapsed = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    probabilities = np.array(
        [r["probability"] for r in results]
    )
    latencies = np.array(
        [r["latency_ms"] for r in results]
    )

    print("Windows:", len(results))
    print("Mean probability:", round(probabilities.mean(), 4))
    print("Median probability:", round(np.median(probabilities), 4))
    print("Detected windows:", sum(r["detected"] for r in results))
    print("Median inference latency:", round(np.median(latencies), 2), "ms")
    print("95th percentile latency:", round(np.percentile(latencies, 95), 2), "ms")
    print("Total processing time:", round(elapsed, 2), "sec")
    print("CSV saved:", args.output)


if __name__ == "__main__":
    main()
