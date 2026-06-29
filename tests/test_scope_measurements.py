import unittest

import numpy as np

from scope_sim import AcquisitionRecord, MeasurementEngine, OscilloscopeEngine, WaveformGenerator


class ScopeAndMeasurementTests(unittest.TestCase):
    def test_trigger_places_crossing_at_requested_pretrigger_fraction(self):
        sample_rate = 1_000_000.0
        _, samples = WaveformGenerator(kind="sine", frequency=1_000, amplitude=1.0).generate(20_000, sample_rate)
        scope = OscilloscopeEngine(time_per_div=100e-6, divisions=10, trigger_level=0.0, pre_trigger_fraction=0.25)

        record = scope.acquire(samples, sample_rate)

        self.assertEqual(record.samples.size, 1000)
        self.assertAlmostEqual(record.trigger_index / record.samples.size, 0.25, delta=0.01)
        self.assertGreaterEqual(record.samples[record.trigger_index], 0.0)
        self.assertLess(record.samples[record.trigger_index - 1], 0.0)
        self.assertFalse(record.metadata["trigger_fallback"])

    def test_trigger_fallback_is_marked_when_no_edge_crosses(self):
        samples = np.full(1000, 0.25)
        record = OscilloscopeEngine(time_per_div=100e-6, trigger_level=1.0).acquire(samples, 1_000_000.0)

        self.assertTrue(record.metadata["trigger_fallback"])

    def test_acquisition_history_is_capped_and_clearable(self):
        _, samples = WaveformGenerator(kind="sine", frequency=1_000, amplitude=1.0).generate(20_000, 1_000_000.0)
        scope = OscilloscopeEngine(time_per_div=100e-6, max_history=2)

        for _ in range(5):
            scope.acquire(samples, 1_000_000.0)

        self.assertEqual(len(scope.acquisitions), 2)
        scope.clear_history()
        self.assertEqual(scope.acquisitions, [])

    def test_bandwidth_validation_rolls_off_signal_above_scope_bandwidth(self):
        sample_rate = 1_000_000_000.0
        n = 100_000
        _, low = WaveformGenerator(kind="sine", frequency=90_000_000, amplitude=1.0).generate(n, sample_rate)
        _, high = WaveformGenerator(kind="sine", frequency=200_000_000, amplitude=1.0).generate(n, sample_rate)
        scope = OscilloscopeEngine(time_per_div=5e-6, divisions=10, bandwidth_limit=100_000_000)

        low_record = scope.acquire(low, sample_rate)
        high_record = scope.acquire(high, sample_rate)

        low_rms = MeasurementEngine(low_record).rms()
        high_rms = MeasurementEngine(high_record).rms()
        self.assertLess(high_rms, low_rms * 0.45)

    def test_sample_rate_validation_higher_rate_reduces_reconstruction_error(self):
        reference_rate = 100_000_000.0
        frequency = 1_000_000.0
        duration = 100e-6
        ref_time, reference = WaveformGenerator(kind="sine", frequency=frequency, amplitude=1.0).generate(int(duration * reference_rate), reference_rate)

        errors = []
        for rate in (5_000_000.0, 50_000_000.0):
            sample_count = int(duration * rate)
            sample_time, sampled = WaveformGenerator(kind="sine", frequency=frequency, amplitude=1.0).generate(sample_count, rate)
            reconstructed = np.interp(ref_time, sample_time, sampled)
            errors.append(float(np.sqrt(np.mean((reference - reconstructed) ** 2))))

        self.assertGreater(errors[0], errors[1] * 20.0)

    def test_measurement_accuracy_for_known_square_wave(self):
        sample_rate = 1_000_000.0
        frequency = 1_000.0
        _, samples = WaveformGenerator(kind="square", frequency=frequency, amplitude=0.5, duty_cycle=0.5).generate(20_000, sample_rate)
        record = OscilloscopeEngine(time_per_div=200e-6, divisions=10, trigger_level=0.0).acquire(samples, sample_rate)
        measurements = MeasurementEngine(record)

        self.assertAlmostEqual(measurements.frequency(), frequency, delta=frequency * 0.01)
        self.assertAlmostEqual(measurements.vpp(), 1.0, delta=0.01)
        self.assertAlmostEqual(measurements.duty_cycle(), 0.5, delta=0.01)
        self.assertAlmostEqual(measurements.rms(), 0.5, delta=0.01)

    def test_frequency_rejects_noisy_threshold_chatter(self):
        sample_rate = 10_000.0
        time = np.arange(int(3.0 * sample_rate)) / sample_rate
        rng = np.random.default_rng(12)
        samples = np.sin(2.0 * np.pi * 1.0 * time) + rng.normal(0.0, 0.003, size=time.shape)
        record = AcquisitionRecord(samples, time, sample_rate, 0, 0.0, "rising")

        self.assertAlmostEqual(MeasurementEngine(record).frequency(), 1.0, delta=0.01)

    def test_noisy_trigger_stability_over_multiple_acquisitions(self):
        sample_rate = 2_000_000.0
        _, clean = WaveformGenerator(kind="sine", frequency=10_000, amplitude=1.0).generate(20_000, sample_rate)
        trigger_times = []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            noisy = clean + rng.normal(0.0, 0.02, size=clean.shape)
            record = OscilloscopeEngine(time_per_div=10e-6, divisions=10, trigger_level=0.0).acquire(noisy, sample_rate)
            trigger_times.append(record.time[record.trigger_index])

        self.assertLess(np.std(trigger_times), 1.0 / sample_rate)

    def test_rise_and_fall_time_on_erf_square_wave(self):
        sample_rate = 50_000_000.0
        _, samples = WaveformGenerator(
            kind="square",
            frequency=10_000,
            amplitude=1.0,
            duty_cycle=0.5,
            rise_time=2e-6,
            fall_time=2e-6,
        ).generate(200_000, sample_rate)
        record = OscilloscopeEngine(time_per_div=20e-6, divisions=10, trigger_level=0.0, pre_trigger_fraction=0.1).acquire(samples, sample_rate)
        measurements = MeasurementEngine(record)

        self.assertGreater(measurements.rise_time(), 0.5e-6)
        self.assertGreater(measurements.fall_time(), 0.5e-6)
        self.assertLess(measurements.rise_time(), 4.0e-6)
        self.assertLess(measurements.fall_time(), 4.0e-6)

    def test_settling_time_detects_step_entering_final_band(self):
        sample_rate = 10_000.0
        time = np.arange(1000) / sample_rate
        samples = np.ones_like(time)
        samples[:100] = 0.0
        tail = np.exp(-np.linspace(0.0, 8.0, 900))
        samples[100:] = 1.0 - tail
        record = AcquisitionRecord(samples=samples, time=time, sample_rate=sample_rate, trigger_index=100, trigger_level=0.5, trigger_edge="rising")

        settling = MeasurementEngine(record).settling_time(tolerance=0.02)

        self.assertGreater(settling, 0.04)
        self.assertLess(settling, 0.08)

    def test_settling_time_scales_linearly_for_large_records(self):
        sample_rate = 1_000_000.0
        time = np.arange(300_000) / sample_rate
        samples = np.ones_like(time)
        samples[:250_000] = 0.0
        record = AcquisitionRecord(samples=samples, time=time, sample_rate=sample_rate, trigger_index=0, trigger_level=0.5, trigger_edge="rising")

        settling = MeasurementEngine(record).settling_time(tolerance=0.01)

        self.assertAlmostEqual(settling, time[250_000])

    def test_overshoot_for_positive_and_negative_steps(self):
        sample_rate = 1_000.0
        time = np.arange(1000) / sample_rate
        rising = np.ones_like(time)
        rising[:100] = 0.0
        rising[100:200] = 1.2
        falling = np.zeros_like(time)
        falling[:100] = 1.0
        falling[100:200] = -0.2

        rising_record = AcquisitionRecord(rising, time, sample_rate, 100, 0.5, "rising")
        falling_record = AcquisitionRecord(falling, time, sample_rate, 100, 0.5, "falling")

        self.assertAlmostEqual(MeasurementEngine(rising_record).overshoot(), 0.2)
        self.assertAlmostEqual(MeasurementEngine(falling_record).overshoot(), 0.2)


if __name__ == "__main__":
    unittest.main()
