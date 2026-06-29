from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class AcquisitionRecord:
    samples: np.ndarray
    time: np.ndarray
    sample_rate: float
    trigger_index: int
    trigger_level: float
    trigger_edge: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OscilloscopeEngine:
    """Simulate timebase, bandwidth limiting, edge triggering, and acquisition memory."""

    time_per_div: float
    divisions: int = 10
    trigger_level: float = 0.0
    trigger_edge: str = "rising"
    pre_trigger_fraction: float = 0.5
    bandwidth_limit: float | None = None
    bandwidth_order: int = 4
    acquisitions: list[AcquisitionRecord] = field(default_factory=list)

    def acquire(self, samples: np.ndarray, sample_rate: float) -> AcquisitionRecord:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.time_per_div <= 0:
            raise ValueError("time_per_div must be positive")
        if self.divisions <= 0:
            raise ValueError("divisions must be positive")
        if not 0.0 <= self.pre_trigger_fraction <= 1.0:
            raise ValueError("pre_trigger_fraction must be between 0 and 1")

        values = np.asarray(samples, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("samples must be a one-dimensional array with at least two values")
        if self.bandwidth_limit is not None:
            values = self._bandwidth_limit(values, sample_rate)

        window_samples = max(2, int(round(self.time_per_div * self.divisions * sample_rate)))
        trigger_index = self._find_trigger(values)
        pre_samples = int(round(window_samples * self.pre_trigger_fraction))
        start = trigger_index - pre_samples
        end = start + window_samples

        padded = values
        pad_left = max(0, -start)
        pad_right = max(0, end - values.size)
        if pad_left or pad_right:
            padded = np.pad(values, (pad_left, pad_right), mode="edge")
            start += pad_left
            trigger_index += pad_left

        captured = padded[start : start + window_samples]
        local_trigger = trigger_index - start
        time = (np.arange(captured.size, dtype=float) - local_trigger) / sample_rate
        record = AcquisitionRecord(
            samples=captured,
            time=time,
            sample_rate=sample_rate,
            trigger_index=local_trigger,
            trigger_level=self.trigger_level,
            trigger_edge=self.trigger_edge,
            metadata={
                "time_per_div": self.time_per_div,
                "divisions": self.divisions,
                "bandwidth_limit": self.bandwidth_limit,
            },
        )
        self.acquisitions.append(record)
        return record

    def _find_trigger(self, values: np.ndarray) -> int:
        edge = self.trigger_edge.lower()
        previous = values[:-1]
        current = values[1:]
        if edge == "rising":
            crossings = np.flatnonzero((previous < self.trigger_level) & (current >= self.trigger_level))
        elif edge == "falling":
            crossings = np.flatnonzero((previous > self.trigger_level) & (current <= self.trigger_level))
        else:
            raise ValueError("trigger_edge must be 'rising' or 'falling'")
        if crossings.size == 0:
            return int(np.argmin(np.abs(values - self.trigger_level)))
        return int(crossings[0] + 1)

    def _bandwidth_limit(self, values: np.ndarray, sample_rate: float) -> np.ndarray:
        nyquist = sample_rate / 2.0
        if not 0.0 < self.bandwidth_limit < nyquist:
            raise ValueError("bandwidth_limit must be between 0 and Nyquist")
        sos = signal.butter(self.bandwidth_order, self.bandwidth_limit, btype="low", fs=sample_rate, output="sos")
        return signal.sosfiltfilt(sos, values) if values.size > 3 * self.bandwidth_order else signal.sosfilt(sos, values)
