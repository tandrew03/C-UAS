
import torch
from torch import nn


class MaskedMeanPooling(nn.Module):
    """Aggregate [B,T,D] features using valid sequence lengths."""

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                f"Expected [B,T,D], got {tuple(x.shape)}"
            )

        lengths = lengths.to(
            device=x.device,
            dtype=torch.long
        )

        if torch.any(lengths < 1) or torch.any(lengths > x.shape[1]):
            raise ValueError(
                "All lengths must be between 1 and T"
            )

        mask = (
            torch.arange(x.shape[1], device=x.device)[None, :]
            < lengths[:, None]
        )

        return (
            (x * mask.unsqueeze(-1)).sum(dim=1)
            / lengths.unsqueeze(-1)
        )
