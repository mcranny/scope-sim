from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class AnalogPath:
    """Apply analog impairments before digitization."""

    sample_rate: float
    gaussian_noise_std: float = 0.0
    thermal_drift_std: float = 0.0
    gain_variation_std: float = 0.0
    cable_attenuation: float = 1.0
    lowpass_cutoff: float | None = None
    lowpass_order: int = 4
    delay: float = 0.0
    seed: int | None = None

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        values = np.asarray(samples, dtype=float).copy()
        rng = np.random.default_rng(self.seed)

        values *= self.cable_attenuation

        if self.gain_variation_std:
            steps = rng.normal(0.0, self.gain_variation_std, size=values.shape)
            gain = 1.0 + np.cumsum(steps) / max(1, values.size)
            values *= gain

        if self.thermal_drift_std:
            drift_steps = rng.normal(0.0, self.thermal_drift_std, size=values.shape)
            values += np.cumsum(drift_steps) / np.sqrt(max(1, values.size))

        if self.gaussian_noise_std:
            values += rng.normal(0.0, self.gaussian_noise_std, size=values.shape)

        if self.lowpass_cutoff is not None:
            values = self._lowpass(values)

        if self.delay:
            values = self._fractional_delay(values, self.delay)

        return values

    def _lowpass(self, values: np.ndarray) -> np.ndarray:
        nyquist = self.sample_rate / 2.0
        if not 0.0 < self.lowpass_cutoff < nyquist:
            raise ValueError("lowpass_cutoff must be between 0 and Nyquist")
        sos = signal.butter(self.lowpass_order, self.lowpass_cutoff, btype="low", fs=self.sample_rate, output="sos")
        return signal.sosfiltfilt(sos, values) if values.size > 3 * self.lowpass_order else signal.sosfilt(sos, values)

    @staticmethod
    def _fractional_delay(values: np.ndarray, delay_samples: float) -> np.ndarray:
        index = np.arange(values.size, dtype=float)
        return np.interp(index - delay_samples, index, values, left=values[0], right=values[-1])
