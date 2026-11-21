
import torch
from torch import nn

from feature_extraction import FeatureExtraction
from CNN_lin_proj import CNNProjection
from conformer import SmallConformerEncoder
from temporal_pooling import MaskedMeanPooling
from classification import BinaryDetectionHead


class DroneDetectionModel(nn.Module):

    def __init__(self, d_model=128, num_blocks=2):
        super().__init__()

        self.features = FeatureExtraction()

        self.cnn = CNNProjection(
            n_mels=80,
            d_model=d_model
        )

        self.encoder = SmallConformerEncoder(
            n_mels=80,
            dim=d_model,
            num_blocks=num_blocks
        )

        self.pool = MaskedMeanPooling()

        self.classifier = BinaryDetectionHead(
            input_dim=d_model
        )

    def forward(self, waveforms, valid_lengths):

        features, lengths = self.features(
            waveforms, valid_lengths
        )

        embeddings, lengths = self.cnn(
            features, lengths
        )

        encoded, lengths = self.encoder.forward_embeddings(
            embeddings, lengths
        )

        pooled = self.pool(
            encoded, lengths
        )

        logits = self.classifier(pooled)

        return logits


if __name__ == "__main__":
    model = DroneDetectionModel()

    waveforms = torch.randn(4, 8000)
    lengths = torch.tensor([8000, 8000, 5081, 6400])

    logits = model(waveforms, lengths)

    print("Logits:", logits.shape)
    print("Finite:", torch.isfinite(logits).all().item())

    loss = nn.BCEWithLogitsLoss()(
        logits,
        torch.tensor([0., 1., 0., 1.])
    )

    loss.backward()
    print("Backward pass successful")
