import unittest

from scope_sim.viewer import run_acquisition


class ViewerTests(unittest.TestCase):
    def test_run_acquisition_returns_trace_and_measurements(self):
        payload = run_acquisition(
            {
                "kind": "sine",
                "frequency": "1000",
                "adc_rate_msps": "1",
                "time_div_us": "100",
                "scope_bw_mhz": "0",
                "analog_lpf_mhz": "0",
            }
        )

        self.assertGreater(len(payload["time"]), 100)
        self.assertEqual(len(payload["time"]), len(payload["samples"]))
        self.assertAlmostEqual(payload["measurements"]["frequency"], 1000.0, delta=20.0)
        self.assertFalse(payload["trigger_fallback"])

    def test_run_acquisition_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            run_acquisition({"frequency": "0"})


if __name__ == "__main__":
    unittest.main()
