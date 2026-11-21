import math
import random

import torch


def random_gain(
    waveform: torch.Tensor,
    min_db: float = -20.0,
    max_db: float = 20.0,
):
    """
    Apply random amplitude gain to a normalized waveform.

    Args:
        waveform: 1D float tensor.
        min_db: Minimum gain in dB.
        max_db: Maximum gain in dB.

    Returns:
        Augmented waveform and applied gain in dB.
    """
    if waveform.ndim != 1:
        raise ValueError("Expected mono waveform [samples]")

    gain_db = random.uniform(min_db, max_db)
    gain = 10.0 ** (gain_db / 20.0)

    augmented = waveform * gain

    # Prevent clipping without independently RMS-normalizing
    # every input.
    peak = augmented.abs().max()

    if peak > 1.0:
        augmented = augmented / peak

    return augmented, gain_db


def rms_dbfs(waveform: torch.Tensor):
    rms = torch.sqrt(torch.mean(waveform.float().square()))
    return 20.0 * torch.log10(rms.clamp_min(1e-12)).item()


def mix_at_snr(
    drone: torch.Tensor,
    background: torch.Tensor,
    snr_db: float,
    eps: float = 1e-8,
):
    """
    Mix drone audio with background audio at a target SNR.

    Returns:
        Mixed waveform, actual SNR before final peak limiting.
    """
    if drone.ndim != 1 or background.ndim != 1:
        raise ValueError("Expected mono waveforms")

    if drone.shape != background.shape:
        raise ValueError("Waveforms must have matching shapes")

    drone_rms = torch.sqrt(drone.square().mean())
    background_rms = torch.sqrt(background.square().mean())

    if drone_rms < eps or background_rms < eps:
        raise ValueError("Cannot mix silent waveforms at a defined SNR")

    target_ratio = 10.0 ** (snr_db / 20.0)

    background_scale = (
        drone_rms / (background_rms * target_ratio)
    )

    scaled_background = background * background_scale
    mixed = drone + scaled_background

    # Uniform scaling preserves the relative SNR.
    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak

    actual_snr = 20.0 * torch.log10(
        drone_rms / scaled_background.square().mean().sqrt()
    ).item()

    return mixed, actual_snr
