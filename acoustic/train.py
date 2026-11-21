
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from preprocessing import (
    load_manifest,
    create_window_index,
    DroneAudio,
    create_balanced_sampler,
)
from model import DroneDetectionModel



def make_loader(dataset, batch_size, training):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=(
            create_balanced_sampler(dataset)
            if training else None
        ),
        shuffle=False,
        num_workers=0,
    )



def select_balanced(dataset, limit, seed=42):
    if limit <= 0:
        return dataset

    labels = dataset["label"]
    generator = torch.Generator().manual_seed(seed)

    selected = []

    for label in (0, 1):
        indices = [
            i for i, y in enumerate(labels)
            if y == label
        ]

        if not indices:
            raise ValueError(f"No examples with label {label}")

        order = torch.randperm(
            len(indices), generator=generator
        )

        count = min(limit // 2, len(indices))
        selected.extend(
            indices[j] for j in order[:count].tolist()
        )

    return dataset.select(selected)


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None

    model.train(training)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    context = (
        torch.enable_grad()
        if training else torch.no_grad()
    )

    with context:
        
        for batch in loader:
            waveforms = batch["audio"].to(device)

            labels = batch["label"].to(
                device=device,
                dtype=torch.float32,
            )

            lengths = torch.full(
                (waveforms.shape[0],),
                waveforms.shape[1],
                dtype=torch.long,
                device=device,
            )


            logits = model(waveforms, lengths)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                optimizer.step()

            count = labels.numel()
            total_loss += loss.item() * count

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += count

    return {
        "loss": total_loss / total,
        "accuracy": correct / total
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    
    print("Loading acoustic v3 manifest...")

    records = load_manifest()
    windows = create_window_index(records)

    train_data = DroneAudio(windows, split="train")
    val_data = DroneAudio(windows, split="val")

    train_loader = make_loader(
        train_data, args.batch_size, True
    )

    val_loader = make_loader(
        val_data, args.batch_size, False
    )

    train_groups = {
        w["group_id"] for w in train_data.windows
    }
    val_groups = {
        w["group_id"] for w in val_data.windows
    }

    assert not train_groups.intersection(val_groups)

    print("Training windows:", len(train_data))
    print("Validation windows:", len(val_data))
    print("Group overlap: 0")


    model = DroneDetectionModel()

    # Restore training-only log-mel normalization statistics.
    stats_path = Path("checkpoints/v3_mel_stats.pt")

    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Missing normalization statistics: {stats_path}"
        )

    stats = torch.load(
        stats_path,
        map_location="cpu",
        weights_only=True,
    )

    model.features.set_normalization_stats(
        stats["mel_mean"].reshape(-1),
        stats["mel_std"].reshape(-1),
    )

    print("Loaded v3 training normalization statistics")


    

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    checkpoint_dir = Path("checkpoints/v3")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):

        train_metrics = run_epoch(
            model, train_loader, optimizer, device
        )

        val_metrics = run_epoch(
            model, val_loader, None, device
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"Train: {train_metrics} "
            f"Val: {val_metrics}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": best_val_loss,
                    "seed": 42,
                    "num_blocks": 2,
                    "manifest": "acoustic_unified_manifest_v3.csv",
                    "training_windows": len(train_data),
                    "validation_windows": len(val_data),
                },
                checkpoint_dir / "best.pt",
            )

            print("Saved checkpoints/v3/best.pt")



if __name__ == "__main__":
    main()
