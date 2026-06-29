import unittest

import numpy as np

from scope_sim import ADCModel, AnalogPath, DACModel, WaveformGenerator


def fft_amplitude(samples, sample_rate, frequency):
    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    bin_index = np.argmin(np.abs(freqs - frequency))
    return 2.0 * np.abs(spectrum[bin_index]) / samples.size


class ModelTests(unittest.TestCase):
    def test_dac_dc_gain_and_offset_are_applied_before_quantization(self):
        samples = np.full(4096, 1.25)
        dac = DACModel(bits=16, full_scale=5.0, gain_error=0.002, offset_error=0.003)

        output = dac.convert(samples)

        self.assertAlmostEqual(float(np.mean(output)), 1.25 * 1.002 + 0.003, delta=2e-4)

    def test_dac_dnl_perturbs_quantization_codes_within_lsb_scale(self):
        samples = np.linspace(-0.9, 0.9, 1000)
        ideal = DACModel(bits=8, full_scale=1.0, dnl=0.0).convert(samples)
        nonlinear = DACModel(bits=8, full_scale=1.0, dnl=0.5).convert(samples)

        delta = nonlinear - ideal
        lsb = 2.0 / (2**8)
        self.assertGreater(float(np.std(delta)), lsb * 0.1)
        self.assertLessEqual(float(np.max(np.abs(delta))), lsb * 0.5 + 1e-12)

    def test_analog_lowpass_rolls_off_above_cutoff(self):
        sample_rate = 1_000_000.0
        duration = 0.05
        time = np.arange(int(duration * sample_rate)) / sample_rate
        low = np.sin(2 * np.pi * 10_000 * time)
        high = np.sin(2 * np.pi * 200_000 * time)
        mixed = low + high

        filtered = AnalogPath(sample_rate=sample_rate, lowpass_cutoff=50_000, lowpass_order=6).process(mixed)

        low_amp = fft_amplitude(filtered, sample_rate, 10_000)
        high_amp = fft_amplitude(filtered, sample_rate, 200_000)
        self.assertGreater(low_amp, 0.85)
        self.assertLess(high_amp, 0.08)

    def test_adc_snr_is_consistent_with_enob(self):
        analog_rate = 5_000_000.0
        adc_rate = 1_000_000.0
        frequency = 10_000.0
        duration = 0.05
        time, signal = WaveformGenerator(kind="sine", frequency=frequency, amplitude=1.0).generate(int(duration * analog_rate), analog_rate)
        adc = ADCModel(sample_rate=adc_rate, bits=16, full_scale=2.0, enob=10.0, seed=11)

        adc_time, quantized = adc.convert(time, signal)
        ideal = np.sin(2 * np.pi * frequency * adc_time)
        noise = quantized - ideal
        snr = 20.0 * np.log10(np.sqrt(np.mean(ideal**2)) / np.sqrt(np.mean(noise**2)))

        expected = 6.02 * 10.0 + 1.76
        self.assertGreater(snr, expected - 8.0)
        self.assertLess(snr, expected + 5.0)

    def test_adc_jitter_increases_high_frequency_error(self):
        analog_rate = 20_000_000.0
        adc_rate = 2_000_000.0
        frequency = 500_000.0
        duration = 0.01
        time, signal = WaveformGenerator(kind="sine", frequency=frequency, amplitude=1.0).generate(int(duration * analog_rate), analog_rate)

        clean_t, clean = ADCModel(sample_rate=adc_rate, bits=16, full_scale=2.0, aperture_jitter=0.0).convert(time, signal)
        jitter_t, jittered = ADCModel(sample_rate=adc_rate, bits=16, full_scale=2.0, aperture_jitter=80e-9, seed=22).convert(time, signal)

        clean_ideal = np.sin(2 * np.pi * frequency * clean_t)
        jitter_ideal = np.sin(2 * np.pi * frequency * jitter_t)
        clean_error = np.sqrt(np.mean((clean - clean_ideal) ** 2))
        jitter_error = np.sqrt(np.mean((jittered - jitter_ideal) ** 2))
        self.assertGreater(jitter_error, clean_error * 10.0)


if __name__ == "__main__":
    unittest.main()
