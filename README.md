# scope-sim

A physics-oriented virtual oscilloscope and signal acquisition simulator. The package models a complete acquisition path:

1. Configurable waveform generation
2. DAC gain, offset, quantization, ENOB noise, and optional nonlinearity
3. Analog path noise, drift, attenuation, low-pass filtering, and fractional delay
4. ADC sample timing, aperture jitter, gain/offset error, quantization, and ENOB noise
5. Oscilloscope timebase, bandwidth limiting, edge triggering, and acquisition buffers
6. Standard oscilloscope measurements and static visualization

The simulator is intended as a technically defensible instrumentation artifact rather than a commercial scope UI clone.

## Quick start

```bash
python3 -m unittest discover -s tests -v
MPLBACKEND=Agg PYTHONPATH=src python3 examples/basic_capture.py
```

The example writes `examples/basic_capture.png`.

## Minimal usage

```python
from scope_sim import WaveformGenerator, OscilloscopeEngine, MeasurementEngine

generator = WaveformGenerator(kind="sine", frequency=1_000, amplitude=1.0)
time, signal = generator.generate(n_samples=100_000, sample_rate=1_000_000)

scope = OscilloscopeEngine(
    time_per_div=100e-6,
    divisions=10,
    trigger_level=0.0,
    trigger_edge="rising",
    pre_trigger_fraction=0.5,
)
record = scope.acquire(signal, sample_rate=1_000_000)

measurements = MeasurementEngine(record)
print(measurements.frequency())
print(measurements.vpp())
```

## Validation focus

The test suite validates waveform FFT placement, DAC gain/offset behavior, analog low-pass rolloff, ADC ENOB/SNR behavior, trigger stability, sample-rate reconstruction error, bandwidth rolloff, jitter impact, and measurement accuracy.
