
from torch import nn


class BinaryDetectionHead(nn.Module):
    """Produces one raw logit per audio window."""

    def __init__(self, input_dim: int = 128):
        super().__init__()

        self.output = nn.Linear(input_dim, 1)

    def forward(self, pooled):
        # [B, 128] -> [B]
        # Matches float binary labels of shape [B]
        return self.output(pooled).squeeze(-1)
