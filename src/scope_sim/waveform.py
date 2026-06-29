from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import erf, erfinv


@dataclass(frozen=True)
class WaveformGenerator:
    """Generate ideal digital voltage waveforms."""

    kind: str
    frequency: float = 1_000.0
    amplitude: float = 1.0
    offset: float = 0.0
    duty_cycle: float = 0.5
    rise_time: float = 0.0
    fall_time: float = 0.0
    arbitrary_samples: Iterable[float] | None = None

    def generate(self, n_samples: int, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        time = np.arange(n_samples, dtype=float) / sample_rate
        kind = self.kind.lower()

        if kind == "sine":
            values = np.sin(2.0 * np.pi * self.frequency * time)
        elif kind == "square":
            values = self._pulse_train(time, bipolar=True)
        elif kind == "pulse":
            values = self._pulse_train(time, bipolar=False)
        elif kind == "triangle":
            phase = self._phase(time)
            values = 4.0 * np.abs(phase - 0.5) - 1.0
            values = -values
        elif kind == "sawtooth":
            values = 2.0 * self._phase(time) - 1.0
        elif kind == "arbitrary":
            values = self._arbitrary(n_samples)
        else:
            raise ValueError(f"unsupported waveform kind: {self.kind}")

        return time, self.offset + self.amplitude * values

    def _phase(self, time: np.ndarray) -> np.ndarray:
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")
        return np.mod(time * self.frequency, 1.0)

    def _pulse_train(self, time: np.ndarray, *, bipolar: bool) -> np.ndarray:
        if not 0.0 < self.duty_cycle < 1.0:
            raise ValueError("duty_cycle must be between 0 and 1")
        phase_seconds = self._phase(time) / self.frequency
        period = 1.0 / self.frequency
        high_width = self.duty_cycle * period

        if self.rise_time <= 0.0 and self.fall_time <= 0.0:
            high = phase_seconds < high_width
            return np.where(high, 1.0, -1.0 if bipolar else 0.0)

        ten_to_ninety_scale = 2.0 * erfinv(0.8)
        rise = max(self.rise_time / ten_to_ninety_scale, np.finfo(float).eps)
        fall = max(self.fall_time / ten_to_ninety_scale, np.finfo(float).eps)
        rising = 0.5 * (1.0 + erf((phase_seconds - 0.0) / rise))
        falling = 0.5 * (1.0 + erf((phase_seconds - high_width) / fall))
        unipolar = np.clip(rising - falling, 0.0, 1.0)
        return 2.0 * unipolar - 1.0 if bipolar else unipolar

    def _arbitrary(self, n_samples: int) -> np.ndarray:
        if self.arbitrary_samples is None:
            raise ValueError("arbitrary_samples is required for arbitrary waveforms")
        samples = np.asarray(list(self.arbitrary_samples), dtype=float)
        if samples.size == 0:
            raise ValueError("arbitrary_samples cannot be empty")
        if samples.size == n_samples:
            return samples.copy()
        source = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        target = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        return np.interp(target, source, samples, period=1.0)
