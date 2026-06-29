from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dac import enob_noise_std, quantize


@dataclass
class ADCModel:
    """Sample and quantize an analog voltage waveform."""

    sample_rate: float
    bits: int = 12
    full_scale: float = 5.0
    gain_error: float = 0.0
    offset_error: float = 0.0
    enob: float | None = None
    aperture_jitter: float = 0.0
    seed: int | None = None

    def convert(
        self,
        analog_time: np.ndarray,
        analog_samples: np.ndarray,
        *,
        duration: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        analog_time = np.asarray(analog_time, dtype=float)
        analog_samples = np.asarray(analog_samples, dtype=float)
        if analog_time.ndim != 1 or analog_samples.ndim != 1:
            raise ValueError("analog_time and analog_samples must be one-dimensional")
        if analog_time.size != analog_samples.size:
            raise ValueError("analog_time and analog_samples must have the same length")
        if analog_time.size < 2:
            raise ValueError("at least two analog samples are required")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        start = analog_time[0]
        stop = analog_time[-1] if duration is None else start + duration
        n_samples = int(np.floor((stop - start) * self.sample_rate))
        if n_samples <= 1:
            raise ValueError("duration is too short for the ADC sample rate")

        sample_time = start + np.arange(n_samples, dtype=float) / self.sample_rate
        rng = np.random.default_rng(self.seed)
        interp_time = sample_time
        if self.aperture_jitter:
            interp_time = sample_time + rng.normal(0.0, self.aperture_jitter, size=sample_time.shape)

        values = np.interp(interp_time, analog_time, analog_samples, left=analog_samples[0], right=analog_samples[-1])
        values = values * (1.0 + self.gain_error) + self.offset_error
        if self.enob is not None:
            values = values + rng.normal(0.0, enob_noise_std(self.full_scale, self.enob), size=values.shape)
        values = quantize(values, self.bits, self.full_scale)
        return sample_time, values
