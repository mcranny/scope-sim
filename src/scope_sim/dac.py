from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quantize(values: np.ndarray, bits: int, full_scale: float) -> np.ndarray:
    if bits <= 0:
        raise ValueError("bits must be positive")
    if full_scale <= 0:
        raise ValueError("full_scale must be positive")
    lsb = 2.0 * full_scale / (2**bits)
    clipped = np.clip(values, -full_scale, full_scale - lsb)
    return np.round(clipped / lsb) * lsb


def enob_noise_std(full_scale: float, enob: float) -> float:
    if full_scale <= 0:
        raise ValueError("full_scale must be positive")
    if enob <= 0:
        raise ValueError("enob must be positive")
    lsb = 2.0 * full_scale / (2**enob)
    return lsb / np.sqrt(12.0)


@dataclass
class DACModel:
    """Convert ideal digital voltages to a non-ideal analog output."""

    bits: int = 14
    full_scale: float = 5.0
    gain_error: float = 0.0
    offset_error: float = 0.0
    enob: float | None = None
    dnl: float = 0.0
    inl: float = 0.0
    seed: int | None = None

    def convert(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=float) * (1.0 + self.gain_error) + self.offset_error
        values = quantize(values, self.bits, self.full_scale)

        if self.inl:
            values = values + self.inl * np.sin(np.pi * values / self.full_scale)
        if self.dnl:
            lsb = 2.0 * self.full_scale / (2**self.bits)
            values = values + self.dnl * lsb * np.sin(2.0 * np.pi * values / max(lsb, np.finfo(float).eps))
        if self.enob is not None:
            rng = np.random.default_rng(self.seed)
            values = values + rng.normal(0.0, enob_noise_std(self.full_scale, self.enob), size=values.shape)

        return values
