
from collections import Counter

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from preprocessing import (
    load_manifest,
    create_window_index,
    DroneAudio,
)
from dads_dataset import DADSAudio


def build_combined_training(seed=42, samples_per_epoch=20000):
    windows = create_window_index(load_manifest())

    local_train = DroneAudio(windows, split="train")
    dads_train = DADSAudio(training=True)

    combined = ConcatDataset([local_train, dads_train])

    metadata = [
        (int(w["label"]), w["source"])
        for w in local_train.windows
    ] + [
        (int(r["label"]), "dads")
        for r in dads_train.records
    ]

    counts = Counter(metadata)
    sources_by_label = {
        label: {
            source for (y, source) in counts
            if y == label
        }
        for label in (0, 1)
    }

    weights = [
        1.0 / (
            2
            * len(sources_by_label[label])
            * counts[(label, source)]
        )
        for label, source in metadata
    ]

    generator = torch.Generator().manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )

    return combined, sampler, metadata


if __name__ == "__main__":
    combined, sampler, metadata = build_combined_training()

    sampled = Counter(metadata[index] for index in sampler)

    print("\nCombined examples:", len(combined))
    print("Samples per epoch:", len(sampler))

    print("\nSampled distribution:")
    for (label, source), count in sorted(sampled.items()):
        print(f"label={label} source={source:12} count={count}")

    class_counts = Counter()
    for (label, source), count in sampled.items():
        class_counts[label] += count

    print("\nClass totals:", dict(class_counts))

    assert len(combined) == 192975
    assert 0.47 <= class_counts[1] / len(sampler) <= 0.53

    print("\nPASS: Combined source-aware sampler works.")
