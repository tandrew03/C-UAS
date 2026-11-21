import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from augmentation import random_gain, mix_at_snr


class TrainingAugmentation(Dataset):
    """Training-only gain and background mixing."""

    def __init__(
        self,
        dataset,
        metadata,
        mix_probability=0.5,
        min_snr_db=-15.0,
        max_snr_db=15.0,
        min_gain_db=-20.0,
        max_gain_db=20.0,
    ):
        if len(dataset) != len(metadata):
            raise ValueError("Dataset and metadata lengths differ")
        if not 0.0 <= mix_probability <= 1.0:
            raise ValueError("Invalid mixing probability")
        if min_snr_db > max_snr_db:
            raise ValueError("Invalid SNR range")
        if min_gain_db > max_gain_db:
            raise ValueError("Invalid gain range")

        self.dataset = dataset
        self.metadata = metadata
        self.mix_probability = mix_probability
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        self.negative_by_source = defaultdict(list)

        for idx, (label, source, category) in enumerate(metadata):
            if int(label) == 0:
                self.negative_by_source[source].append(idx)

        self.background_sources = sorted(self.negative_by_source)

        if mix_probability > 0 and not self.background_sources:
            raise ValueError("No training negatives available for mixing")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Do not mutate the underlying dataset's returned dictionary.
        example = dict(self.dataset[index])
        audio = example["audio"].float().clone()
        label = int(example["label"])

        if label == 1 and random.random() < self.mix_probability:
            # Choose a source first so large datasets do not
            # automatically dominate background selection.
            source = random.choice(self.background_sources)
            bg_index = random.choice(self.negative_by_source[source])

            background = self.dataset[bg_index]["audio"].float()

            if background.shape != audio.shape:
                raise ValueError(
                    f"Audio shapes differ: {audio.shape}, "
                    f"{background.shape}"
                )

            if (
                audio.square().mean().sqrt().item() >= 1e-8
                and background.square().mean().sqrt().item() >= 1e-8
            ):
                snr_db = random.uniform(
                    self.min_snr_db, self.max_snr_db
                )
                audio, _ = mix_at_snr(audio, background, snr_db)

        audio, _ = random_gain(
            audio,
            min_db=self.min_gain_db,
            max_db=self.max_gain_db,
        )

        example["audio"] = audio
        return example
