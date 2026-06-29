import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scope_sim_matplotlib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scope_sim import ADCModel, AnalogPath, DACModel, MeasurementEngine, OscilloscopeEngine, WaveformGenerator, plot_acquisition


def main() -> None:
    sample_rate = 5_000_000
    generator = WaveformGenerator(kind="square", frequency=25_000, amplitude=1.0, duty_cycle=0.4, rise_time=2e-6, fall_time=2e-6)
    time, ideal = generator.generate(n_samples=100_000, sample_rate=sample_rate)

    dac = DACModel(bits=14, full_scale=2.5, gain_error=0.0005, offset_error=0.001, enob=12.5, seed=2)
    analog = dac.convert(ideal)
    analog = AnalogPath(sample_rate=sample_rate, gaussian_noise_std=0.002, lowpass_cutoff=1_000_000, delay=2.4, seed=3).process(analog)

    adc_time, digitized = ADCModel(sample_rate=sample_rate, bits=12, full_scale=2.5, enob=11.5, aperture_jitter=2e-9, seed=4).convert(time, analog)
    record = OscilloscopeEngine(time_per_div=10e-6, trigger_level=0.0, pre_trigger_fraction=0.25, bandwidth_limit=2_000_000).acquire(digitized, sample_rate)

    print(MeasurementEngine(record).all())
    plot_acquisition(record, Path(__file__).with_name("basic_capture.png"))


if __name__ == "__main__":
    main()
