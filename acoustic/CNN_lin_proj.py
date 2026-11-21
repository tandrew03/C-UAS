# CNN_lin_proj.py learns the short-range time-frequency relationships in the spectrogram. 
# The file also reduces the number of time steps before self-attention
# Define convolution kernels
# activations
# paddings
# output shape
# flattening order
# projection

import torch
import torch.nn as nn

from feature_extraction import FeatureExtraction
from torch.utils.data import DataLoader



class CNNProjection(nn.Module):
    def __init__(self, n_mels=80, d_model=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

        reduced_mels = (n_mels + 1) // 2
        reduced_mels = (reduced_mels + 1) // 2

        self.projection = nn.Linear(
            32 * reduced_mels,
            d_model,
        )

    def compute_output_lengths(self, lengths):
        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2

        return lengths

    def forward(self, features, lengths):
        # features: (batch, mel bands, time)
        x = features.unsqueeze(1)

        # x: (batch, 1, mel bands, time)
        x = self.conv(x)

        # x: (batch, channels, reduced mel bands, reduced time)
        x = x.permute(0, 3, 1, 2)

        # x: (batch, reduced time, channels, reduced mel bands)
        x = x.flatten(start_dim=2)

        # x: (batch, reduced time, channels * reduced mel bands)
        x = self.projection(x)

        # x: (batch, reduced time, d_model)
        output_lengths = self.compute_output_lengths(lengths)

        return x, output_lengths


from feature_extraction import FeatureExtraction
from torch.utils.data import DataLoader


def main():
    # 1. Load the dataset and recreate your splits.
    print("Loading DADS...")
    dataset = load_dataset(DATASET_PATH)
    splits = create_splits(dataset)

    # 2. Prepare fixed-length waveforms.
    # training=False selects repeatable center crops.
    train_dataset = DroneAudio(
        splits["train"],
        n_samples=8000,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # 3. Restore the feature extractor, including the
    # normalization statistics you calculated earlier.
    extractor = FeatureExtraction()

    extractor.load_state_dict(
        torch.load(
            "feature_extractor.pt",
            map_location="cpu",
            weights_only=True,
        )
    )

    # 4. Create the CNN and projection layers.
    cnn = CNNProjection(
        n_mels=extractor.n_mels,
        d_model=128,
    )

    extractor.eval()
    cnn.eval()

    # 5. Get one batch from preprocessing.py.
    waveforms, labels, valid_lengths = next(iter(train_loader))

    # 6. Run that batch through both modules.
    with torch.no_grad():
        features, feature_lengths = extractor(
            waveforms,
            valid_lengths,
        )

        embeddings, embedding_lengths = cnn(
            features,
            feature_lengths,
        )

    # 7. Inspect the connections.
    print("\nWaveform batch:", waveforms.shape)
    print("Labels:", labels)
    print("Valid waveform lengths:", valid_lengths)

    print("\nLog-mel features:", features.shape)
    print("Valid feature lengths:", feature_lengths)

    print("\nCNN embeddings:", embeddings.shape)
    print("Valid embedding lengths:", embedding_lengths)


if __name__ == "__main__":
    main()
