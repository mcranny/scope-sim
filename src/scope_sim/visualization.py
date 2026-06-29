from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .measurements import MeasurementEngine
from .scope import AcquisitionRecord


def plot_acquisition(record: AcquisitionRecord, output_path: str | Path | None = None):
    """Create a static oscilloscope-style plot with measurements."""

    measurements = MeasurementEngine(record).all()
    fig, (ax_wave, ax_text) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        gridspec_kw={"height_ratios": [4, 1.2]},
        constrained_layout=True,
    )

    ax_wave.plot(record.time, record.samples, color="#2b6cb0", linewidth=1.4)
    ax_wave.axhline(record.trigger_level, color="#c53030", linestyle="--", linewidth=1.0, label="trigger level")
    ax_wave.axvline(record.time[record.trigger_index], color="#2f855a", linestyle="--", linewidth=1.0, label="trigger")
    ax_wave.set_title("Virtual Oscilloscope Capture")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Voltage (V)")
    ax_wave.grid(True, alpha=0.35)
    ax_wave.legend(loc="upper right")

    ax_text.axis("off")
    rows = [
        f"sample rate: {record.sample_rate:.6g} S/s",
        f"time/div: {record.metadata.get('time_per_div', float('nan')):.6g} s",
        f"frequency: {measurements['frequency']:.6g} Hz",
        f"period: {measurements['period']:.6g} s",
        f"rms: {measurements['rms']:.6g} V",
        f"vpp: {measurements['vpp']:.6g} V",
        f"duty: {measurements['duty_cycle'] * 100.0:.3g} %",
    ]
    ax_text.text(0.01, 0.85, "\n".join(rows), family="monospace", va="top")

    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    return fig
