import torch
from torch import nn




def feed_forward(dim, dropout):
    return nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.SiLU(),
        nn.Dropout(dropout), nn.Linear(4 * dim, dim), nn.Dropout(dropout)
    )



class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim=128,
        heads=4,
        kernel_size=15,
        dropout=0.2,
        ffn_expansion=2
    ):
        super().__init__()

        def make_ffn():
            return nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, ffn_expansion * dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_expansion * dim, dim),
                nn.Dropout(dropout)
            )

        self.ffn1 = make_ffn()
        self.ffn2 = make_ffn()

        self.attention_norm = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_dropout = nn.Dropout(dropout)

        self.conv_norm = nn.LayerNorm(dim)

        self.convolution = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, kernel_size=1),
            nn.GLU(dim=1),

            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim
            ),

            nn.GroupNorm(1, dim),
            nn.SiLU(),

            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x, lengths=None):
        valid_mask = None
        padding_mask = None

        if lengths is not None:
            positions = torch.arange(
                x.shape[1],
                device=x.device
            )

            valid_mask = (
                positions.unsqueeze(0)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1)

            padding_mask = ~valid_mask.squeeze(-1)

            x = x * valid_mask

        # First feed-forward module
        x = x + 0.5 * self.ffn1(x)

        if valid_mask is not None:
            x = x * valid_mask

        # Multi-head self-attention
        z = self.attention_norm(x)

        z, _ = self.attention(
            z, z, z,
            key_padding_mask=padding_mask,
            need_weights=False
        )

        x = x + self.attention_dropout(z)

        if valid_mask is not None:
            x = x * valid_mask

        # Convolution module
        z = self.conv_norm(x).transpose(1, 2)

        x = x + self.convolution(z).transpose(1, 2)

        if valid_mask is not None:
            x = x * valid_mask

        # Second feed-forward module
        x = x + 0.5 * self.ffn2(x)

        x = self.output_norm(x)

        if valid_mask is not None:
            x = x * valid_mask

        return x




class SmallConformerEncoder(nn.Module):
    def __init__(self, n_mels=80, dim=128, num_blocks=2):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, 512, dim) * 0.02)
        self.blocks = nn.ModuleList(
        ConformerBlock(
            dim=dim,
            heads=4,
            kernel_size=15,
            dropout=0.2,
            ffn_expansion=2
        )
        for _ in range(num_blocks)
    )

    def forward(self, log_mel):
        x = self.cnn(log_mel.unsqueeze(1))    
        x = x.permute(0, 3, 1, 2).flatten(2)
        x = self.projection(x)
        if x.shape[1] > self.position.shape[1]:
            raise ValueError("Use shorter clips or increase the positional table length")
        x = x + self.position[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return x 

    
    def forward_embeddings(self, x, lengths):
        if x.shape[1] > self.position.shape[1]:
            raise ValueError("Sequence exceeds positional table")

        x = x + self.position[:, :x.shape[1]]

        for block in self.blocks:
            x = block(x, lengths)

        return x, lengths



if __name__ == "__main__":
    waveform = torch.randn(1, 16000 * 2)
    features = AudioFeatures()
    encoder = SmallConformerEncoder()
    encoder.eval()
    with torch.no_grad():
        stft, mel, log_mel = features(waveform)
        encoded = encoder(log_mel)
    print("STFT:", tuple(stft.shape))
    print("Mel spectrogram:", tuple(mel.shape))
    print("Log-mel:", tuple(log_mel.shape))
    print("Encoder output:", tuple(encoded.shape))