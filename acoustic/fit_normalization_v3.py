
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from preprocessing import (
    load_manifest,
    create_window_index,
    DroneAudio,
)
from feature_extraction import FeatureExtraction


def main():
    torch.manual_seed(42)

    records = load_manifest()
    windows = create_window_index(records)

    train = DroneAudio(windows, split="train")

    # No balanced sampler: fit statistics on the actual
    # training distribution, without replacement.
    loader = DataLoader(
        train,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    extractor = FeatureExtraction()
    extractor.eval()

    n_mels = extractor.n_mels

    sums = torch.zeros(n_mels, dtype=torch.float64)
    sums_sq = torch.zeros(n_mels, dtype=torch.float64)
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            waveforms = batch["audio"]

            # Shape: (batch, 80, time)
            features = extractor.compute_log_mel(
                waveforms
            ).double()

            sums += features.sum(dim=(0, 2))
            sums_sq += features.square().sum(dim=(0, 2))

            count += features.shape[0] * features.shape[2]

            if i % 50 == 0:
                print(
                    f"Processed {i}/{len(loader)} batches",
                    flush=True,
                )

    mean = sums / count
    variance = (sums_sq / count) - mean.square()
    std = variance.clamp_min(1e-12).sqrt()

    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()

    extractor.set_normalization_stats(
        mean.float(),
        std.float(),
    )

    output = Path("checkpoints/v3_mel_stats.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "mel_mean": extractor.mel_mean.cpu(),
            "mel_std": extractor.mel_std.cpu(),
            "training_windows": len(train),
            "feature_frames_per_band": count,
            "manifest": "acoustic_unified_manifest_v3.csv",
        },
        output,
    )

    print("\nNormalization complete")
    print("Training windows:", len(train))
    print("Feature frames per mel band:", count)
    print("Mean range:",
          round(mean.min().item(), 4),
          "to",
          round(mean.max().item(), 4))
    print("Std range:",
          round(std.min().item(), 4),
          "to",
          round(std.max().item(), 4))
    print("Saved:", output)
    print("PASS: Training-only normalization fitted.")


if __name__ == "__main__":
    main()
