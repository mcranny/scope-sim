import unittest

import numpy as np

from scope_sim import WaveformGenerator


class WaveformGeneratorTests(unittest.TestCase):
    def test_sine_fft_peak_lands_on_expected_bin(self):
        sample_rate = 1024.0
        frequency = 64.0
        _, samples = WaveformGenerator(kind="sine", frequency=frequency, amplitude=1.0).generate(1024, sample_rate)

        spectrum = np.fft.rfft(samples)
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
        peak = freqs[np.argmax(np.abs(spectrum[1:])) + 1]

        self.assertEqual(peak, frequency)
        self.assertLess(np.abs(np.mean(samples)), 1e-12)

    def test_square_wave_duty_cycle_and_levels(self):
        _, samples = WaveformGenerator(kind="square", frequency=10.0, amplitude=2.0, duty_cycle=0.25).generate(1000, 1000.0)

        self.assertAlmostEqual(np.mean(samples > 0), 0.25, delta=0.01)
        self.assertAlmostEqual(np.max(samples), 2.0)
        self.assertAlmostEqual(np.min(samples), -2.0)

    def test_triangle_and_sawtooth_ranges_are_normalized(self):
        _, triangle = WaveformGenerator(kind="triangle", frequency=10.0).generate(1000, 1000.0)
        _, sawtooth = WaveformGenerator(kind="sawtooth", frequency=10.0).generate(1000, 1000.0)

        self.assertAlmostEqual(float(np.max(triangle)), 1.0)
        self.assertAlmostEqual(float(np.min(triangle)), -1.0)
        self.assertAlmostEqual(float(np.max(sawtooth)), 0.98)
        self.assertAlmostEqual(float(np.min(sawtooth)), -1.0)

    def test_pulse_waveform_is_unipolar_with_requested_duty_cycle(self):
        _, samples = WaveformGenerator(kind="pulse", frequency=10.0, amplitude=3.0, duty_cycle=0.2).generate(1000, 1000.0)

        self.assertAlmostEqual(float(np.min(samples)), 0.0)
        self.assertAlmostEqual(float(np.max(samples)), 3.0)
        self.assertAlmostEqual(float(np.mean(samples > 1.5)), 0.2, delta=0.01)

    def test_arbitrary_waveform_resamples_periodically(self):
        _, samples = WaveformGenerator(kind="arbitrary", arbitrary_samples=[0, 1, 0, -1]).generate(8, 8.0)

        np.testing.assert_allclose(samples, [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
