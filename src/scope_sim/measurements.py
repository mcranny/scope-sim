from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scope import AcquisitionRecord


@dataclass
class MeasurementEngine:
    """Compute oscilloscope-style measurements from an acquisition."""

    record: AcquisitionRecord

    def average(self) -> float:
        return float(np.mean(self.record.samples))

    def rms(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.record.samples))))

    def vpp(self) -> float:
        samples = self.record.samples
        return float(np.max(samples) - np.min(samples))

    def frequency(self) -> float:
        samples = self.record.samples - np.mean(self.record.samples)
        if np.allclose(samples, 0.0):
            return float("nan")
        crossings = self._rising_crossing_times(np.mean(self.record.samples))
        if crossings.size >= 2:
            periods = np.diff(crossings)
            periods = periods[periods > 0]
            if periods.size:
                return float(1.0 / np.median(periods))

        window = np.hanning(samples.size)
        spectrum = np.fft.rfft(samples * window)
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / self.record.sample_rate)
        if freqs.size <= 1:
            return float("nan")
        peak = np.argmax(np.abs(spectrum[1:])) + 1
        return float(freqs[peak])

    def period(self) -> float:
        frequency = self.frequency()
        return float(1.0 / frequency) if frequency > 0 else float("nan")

    def duty_cycle(self) -> float:
        samples = self.record.samples
        threshold = 0.5 * (np.max(samples) + np.min(samples))
        high = samples >= threshold
        crossings = self._rising_crossing_times(threshold)
        if crossings.size >= 2:
            first = np.searchsorted(self.record.time, crossings[0])
            second = np.searchsorted(self.record.time, crossings[1])
            if second > first:
                return float(np.mean(high[first:second]))
        return float(np.mean(high))

    def rise_time(self) -> float:
        low, high = self._ten_ninety_levels()
        return self._transition_time(low, high, rising=True)

    def fall_time(self) -> float:
        low, high = self._ten_ninety_levels()
        return self._transition_time(high, low, rising=False)

    def overshoot(self) -> float:
        samples = self.record.samples
        final = float(np.mean(samples[int(0.9 * samples.size) :]))
        initial = float(np.mean(samples[: max(1, int(0.1 * samples.size))]))
        step = final - initial
        if np.isclose(step, 0.0):
            return 0.0
        peak = float(np.max(samples) if step > 0 else np.min(samples))
        return float(max(0.0, (peak - final) / abs(step) * np.sign(step)))

    def settling_time(self, tolerance: float = 0.02) -> float:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        samples = self.record.samples
        final = float(np.mean(samples[int(0.9 * samples.size) :]))
        band = max(abs(final) * tolerance, tolerance * max(self.vpp(), np.finfo(float).eps))
        settled = np.abs(samples - final) <= band
        for idx in range(samples.size):
            if np.all(settled[idx:]):
                return float(self.record.time[idx] - self.record.time[0])
        return float("nan")

    def all(self) -> dict[str, float]:
        return {
            "frequency": self.frequency(),
            "period": self.period(),
            "rms": self.rms(),
            "average": self.average(),
            "vpp": self.vpp(),
            "duty_cycle": self.duty_cycle(),
            "rise_time": self.rise_time(),
            "fall_time": self.fall_time(),
            "overshoot": self.overshoot(),
            "settling_time": self.settling_time(),
        }

    def _rising_crossing_times(self, threshold: float) -> np.ndarray:
        samples = self.record.samples
        prev = samples[:-1]
        curr = samples[1:]
        idx = np.flatnonzero((prev < threshold) & (curr >= threshold))
        times: list[float] = []
        for i in idx:
            dv = curr[i] - prev[i]
            frac = 0.0 if np.isclose(dv, 0.0) else (threshold - prev[i]) / dv
            times.append(float(self.record.time[i] + frac * (self.record.time[i + 1] - self.record.time[i])))
        return np.asarray(times)

    def _ten_ninety_levels(self) -> tuple[float, float]:
        samples = self.record.samples
        low = float(np.min(samples) + 0.1 * self.vpp())
        high = float(np.min(samples) + 0.9 * self.vpp())
        return low, high

    def _transition_time(self, start_level: float, stop_level: float, *, rising: bool) -> float:
        samples = self.record.samples
        if rising:
            start_idx = np.flatnonzero((samples[:-1] < start_level) & (samples[1:] >= start_level))
            stop_idx = np.flatnonzero((samples[:-1] < stop_level) & (samples[1:] >= stop_level))
        else:
            start_idx = np.flatnonzero((samples[:-1] > start_level) & (samples[1:] <= start_level))
            stop_idx = np.flatnonzero((samples[:-1] > stop_level) & (samples[1:] <= stop_level))
        if start_idx.size == 0 or stop_idx.size == 0:
            return float("nan")
        first_start = int(start_idx[0])
        later_stops = stop_idx[stop_idx >= first_start]
        if later_stops.size == 0:
            return float("nan")
        start_time = self._crossing_time(first_start, start_level)
        stop_time = self._crossing_time(int(later_stops[0]), stop_level)
        return float(stop_time - start_time)

    def _crossing_time(self, index: int, threshold: float) -> float:
        samples = self.record.samples
        dv = samples[index + 1] - samples[index]
        frac = 0.0 if np.isclose(dv, 0.0) else (threshold - samples[index]) / dv
        return float(self.record.time[index] + frac * (self.record.time[index + 1] - self.record.time[index]))
