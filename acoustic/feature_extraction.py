# file aims to convert each waveform into a time-frequency representation for the Conformer
# featuere extraction is STFT -> Mel -> log
# 1. Compute the STFT using a Hann window, 512-point FFT and 160-sample hop
# 2. Calculate the power spectrograms
# 3. Apply 80 mel filters
# 4. Apply log compression with numerical protection against zero values
# 5. Normalize the features using training-derived statistics
# 6. return the spectrogram and its valid sequence length

# compute_stft()
# compute_log_mel()
# normalize_features()
# forward()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



class TorchMelSpectrogram(nn.Module):
    """
    Native PyTorch replacement for torchaudio's
    default MelSpectrogram configuration.

    Uses a Hann window and HTK-scale mel filters.
    """

    def __init__(self, sample_rate, n_fft, hop_length, n_mels):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer(
            "window",
            torch.hann_window(n_fft)
        )

        # HTK mel scale, matching torchaudio defaults.
        mel_min = 0.0
        mel_max = 2595.0 * torch.log10(
            torch.tensor(1.0 + (sample_rate / 2) / 700.0)
        ).item()

        mel_points = torch.linspace(
            mel_min, mel_max, n_mels + 2
        )

        hz_points = 700.0 * (
            torch.pow(10.0, mel_points / 2595.0) - 1.0
        )

        frequencies = torch.linspace(
            0, sample_rate / 2, n_fft // 2 + 1
        )

        lower = hz_points[:-2].unsqueeze(1)
        center = hz_points[1:-1].unsqueeze(1)
        upper = hz_points[2:].unsqueeze(1)

        rising = (
            (frequencies - lower) /
            (center - lower)
        )

        falling = (
            (upper - frequencies) /
            (upper - center)
        )

        filterbank = torch.clamp(
            torch.minimum(rising, falling),
            min=0.0
        )

        self.register_buffer("filterbank", filterbank)

    def forward(self, waveforms):
        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

        power = stft.abs().square()

        return torch.matmul(self.filterbank, power)


class FeatureExtraction(nn.Module):

    def __init__(
            self,
            sample_rate =16000,
            n_fft=512,
            hop_length = 160,
            n_mels = 80,
            eps = 1e-6
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps
        self.register_buffer(
            "mel_mean",
            torch.zeros(1, n_mels, 1)
        )
        self.register_buffer(
            "mel_std",
            torch.ones(1, n_mels, 1)
        )

        self.mel_transform = TorchMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )


    def compute_log_mel(self, waveforms):
        mel = self.mel_transform(waveforms)

        log_mel = torch.log(mel + self.eps)

        return log_mel

    
    def compute_feature_lengths(self, valid_len):
        f_len = (valid_len // self.hop_length) + 1

        return f_len


    def set_normalization_stats(self, mean, std):
        if mean.numel() != self.n_mels:
            raise ValueError("Incorrect mean dimensions")

        if std.numel() != self.n_mels:
            raise ValueError("Incorrect std dimensions")

        self.mel_mean.copy_(
            mean.reshape(1, self.n_mels, 1)
        )

        self.mel_std.copy_(
            std.reshape(1, self.n_mels, 1).clamp_min(self.eps)
        )


    def normalize_features(self,features):
        return (features - self.mel_mean) / self.mel_std


    def forward(self, waveforms, valid_len):
        features = self.compute_log_mel(waveforms)

        f_len = self.compute_feature_lengths(valid_len)

        f_len = f_len.clamp(max = features.shape[-1])

        features = self.normalize_features(features)

        return features, f_len


@torch.no_grad()
def fit_normalization(extractor, train_loader):

    n_mels = extractor.n_mels

    sums = torch.zeros(n_mels, dtype=torch.float64)
    sums_sq = torch.zeros(n_mels,dtype = torch.float64)
    count = 0

    for waveform, labels, valid_len in train_loader:

        features = extractor.compute_log_mel(waveform)
        feature_len = extractor.compute_feature_lengths(valid_len).clamp(max=features.shape[-1])

        time_indicies = torch.arange(features.shape[-1]).unsqueeze(0)
        mask = time_indicies < feature_len.unsqueeze(1)

        mask = mask.unsqueeze(1)

        features = features.double()

        sums += (features * mask).sum(dim=(0,2))
        sums_sq += (features.square() * mask).sum(dim=(0,2))

        count += mask.sum().item()

    if count == 0:
        raise ValueError("No valid frames found")

    mean = sums / count

    var = sums_sq / count - mean.square()

    std = torch.sqrt(var.clamp_min(1e-12))

    extractor.set_normalization_stats(
        mean.float(),
        std.float()
    )

    return mean.float(), std.float()


def main():
    print("Loading DADS...")
    dataset = load_dataset(DATASET_PATH)

    print("Creating dataset splits...")
    splits = create_splits(dataset)

    # We use the training partition only.
    # Deterministic segmentation for reproducible stats.
    train_dataset = DroneAudio(
        splits["train"],
        n_samples=8000,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    extractor = FeatureExtraction()

    print("Computing training normalization statistics...")

    mean, std = fit_normalization(
        extractor,
        train_loader
    )

    print("Mean shape:", mean.shape)
    print("Std shape:", std.shape)

    # Save the extractor's buffers and configuration-dependent state
    torch.save(
        extractor.state_dict(),
        "feature_extractor.pt"
    )

    print("Saved feature_extractor.pt")


if __name__ == "__main__":
    main()