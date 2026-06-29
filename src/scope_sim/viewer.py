from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from urllib.parse import parse_qs, urlparse

import numpy as np

from .adc import ADCModel
from .analog import AnalogPath
from .dac import DACModel
from .measurements import MeasurementEngine
from .scope import OscilloscopeEngine
from .waveform import WaveformGenerator


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scope-sim viewer</title>
<style>
:root {
  color-scheme: dark;
  --bg: #121417;
  --panel: #1c2127;
  --line: #343c45;
  --text: #e6edf3;
  --muted: #9da9b5;
  --accent: #35d07f;
  --warn: #f5b942;
  --trace: #6eb6ff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(320px, 380px) minmax(0, 1fr);
}
aside {
  border-right: 1px solid var(--line);
  background: #171b20;
  overflow: auto;
  max-height: 100vh;
}
.brand {
  height: 56px;
  display: flex;
  align-items: center;
  padding: 0 18px;
  border-bottom: 1px solid var(--line);
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0;
}
.controls {
  display: grid;
  gap: 14px;
  padding: 16px;
}
fieldset {
  border: 1px solid var(--line);
  border-radius: 6px;
  margin: 0;
  padding: 12px;
  background: var(--panel);
}
legend {
  padding: 0 5px;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}
label {
  display: grid;
  grid-template-columns: 1fr minmax(92px, 128px);
  align-items: center;
  gap: 12px;
  min-height: 34px;
  font-size: 13px;
  color: var(--muted);
}
input, select {
  width: 100%;
  min-width: 0;
  height: 30px;
  border: 1px solid #47515c;
  border-radius: 5px;
  background: #0f1216;
  color: var(--text);
  padding: 4px 8px;
}
.stage {
  min-width: 0;
  display: grid;
  grid-template-rows: minmax(360px, 1fr) auto;
}
.scope {
  position: relative;
  min-height: 360px;
}
canvas {
  width: 100%;
  height: 100%;
  display: block;
  background: #080a0d;
}
.status {
  position: absolute;
  left: 16px;
  top: 14px;
  color: var(--muted);
  font: 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.readout {
  border-top: 1px solid var(--line);
  background: #171b20;
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 1px;
}
.metric {
  min-height: 58px;
  padding: 10px 12px;
  background: var(--panel);
}
.metric span {
  display: block;
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
}
.metric strong {
  display: block;
  margin-top: 3px;
  font: 18px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: nowrap;
}
@media (max-width: 900px) {
  main { grid-template-columns: 1fr; }
  aside { max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }
  .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .readout { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
}
@media (max-width: 620px) {
  .controls { grid-template-columns: 1fr; }
  label { grid-template-columns: 1fr; gap: 4px; }
}
</style>
</head>
<body>
<main>
  <aside>
    <div class="brand">scope-sim</div>
    <div class="controls">
      <fieldset>
        <legend>Waveform</legend>
        <label>Type <select id="kind"><option>sine</option><option>square</option><option>triangle</option><option>sawtooth</option><option>pulse</option></select></label>
        <label>Frequency Hz <input id="frequency" type="number" value="25000" min="1" step="100"></label>
        <label>Amplitude V <input id="amplitude" type="number" value="1" min="0.001" step="0.1"></label>
        <label>Offset V <input id="offset" type="number" value="0" step="0.1"></label>
        <label>Duty <input id="duty_cycle" type="number" value="0.4" min="0.01" max="0.99" step="0.01"></label>
        <label>Rise us <input id="rise_us" type="number" value="2" min="0" step="0.1"></label>
        <label>Fall us <input id="fall_us" type="number" value="2" min="0" step="0.1"></label>
      </fieldset>
      <fieldset>
        <legend>Converters</legend>
        <label>DAC bits <input id="dac_bits" type="number" value="14" min="4" max="24" step="1"></label>
        <label>DAC ENOB <input id="dac_enob" type="number" value="12.5" min="1" max="24" step="0.1"></label>
        <label>ADC rate MS/s <input id="adc_rate_msps" type="number" value="5" min="0.01" step="0.1"></label>
        <label>ADC bits <input id="adc_bits" type="number" value="12" min="4" max="24" step="1"></label>
        <label>ADC ENOB <input id="adc_enob" type="number" value="11.5" min="1" max="24" step="0.1"></label>
        <label>Jitter ns <input id="jitter_ns" type="number" value="2" min="0" step="0.1"></label>
      </fieldset>
      <fieldset>
        <legend>Analog Path</legend>
        <label>Noise mV <input id="noise_mv" type="number" value="2" min="0" step="0.1"></label>
        <label>Attenuation <input id="attenuation" type="number" value="1" min="0" step="0.05"></label>
        <label>LPF MHz <input id="analog_lpf_mhz" type="number" value="1" min="0" step="0.1"></label>
        <label>Delay samples <input id="delay_samples" type="number" value="2.4" step="0.1"></label>
      </fieldset>
      <fieldset>
        <legend>Scope</legend>
        <label>Time/div us <input id="time_div_us" type="number" value="10" min="0.01" step="0.1"></label>
        <label>Trigger V <input id="trigger_level" type="number" value="0" step="0.05"></label>
        <label>Edge <select id="trigger_edge"><option>rising</option><option>falling</option></select></label>
        <label>Pre-trigger <input id="pre_trigger" type="number" value="0.25" min="0" max="1" step="0.05"></label>
        <label>Scope BW MHz <input id="scope_bw_mhz" type="number" value="2" min="0" step="0.1"></label>
      </fieldset>
    </div>
  </aside>
  <section class="stage">
    <div class="scope">
      <canvas id="plot"></canvas>
      <div id="status" class="status">acquiring</div>
    </div>
    <div id="readout" class="readout"></div>
  </section>
</main>
<script>
const ids = ["kind","frequency","amplitude","offset","duty_cycle","rise_us","fall_us","dac_bits","dac_enob","adc_rate_msps","adc_bits","adc_enob","jitter_ns","noise_mv","attenuation","analog_lpf_mhz","delay_samples","time_div_us","trigger_level","trigger_edge","pre_trigger","scope_bw_mhz"];
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const readout = document.getElementById("readout");
let timer = 0;

function params() {
  const query = new URLSearchParams();
  for (const id of ids) query.set(id, document.getElementById(id).value);
  return query;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
}

function fmt(value, unit = "") {
  if (value === null || Number.isNaN(value)) return "n/a";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 1e5 || abs < 1e-3)) return `${value.toExponential(3)}${unit}`;
  return `${value.toPrecision(5)}${unit}`;
}

function draw(data) {
  resizeCanvas();
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#080a0d";
  ctx.fillRect(0, 0, w, h);

  const left = 54, top = 28, right = 22, bottom = 40;
  const pw = w - left - right, ph = h - top - bottom;
  const times = data.time, samples = data.samples;
  const minY = Math.min(...samples), maxY = Math.max(...samples);
  const padY = Math.max((maxY - minY) * 0.12, 0.05);
  const y0 = minY - padY, y1 = maxY + padY;
  const x0 = times[0], x1 = times[times.length - 1];

  ctx.strokeStyle = "#25303a";
  ctx.lineWidth = 1;
  ctx.font = `${12 * (window.devicePixelRatio || 1)}px ui-monospace, Menlo, Consolas`;
  ctx.fillStyle = "#9da9b5";
  for (let i = 0; i <= 10; i++) {
    const x = left + (i / 10) * pw;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.stroke();
  }
  for (let i = 0; i <= 8; i++) {
    const y = top + (i / 8) * ph;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + pw, y); ctx.stroke();
  }

  const sx = v => left + ((v - x0) / (x1 - x0 || 1)) * pw;
  const sy = v => top + ph - ((v - y0) / (y1 - y0 || 1)) * ph;
  const triggerY = sy(data.trigger_level);
  const triggerX = sx(0);

  ctx.setLineDash([8, 6]);
  ctx.strokeStyle = "#f5b942";
  ctx.beginPath(); ctx.moveTo(left, triggerY); ctx.lineTo(left + pw, triggerY); ctx.stroke();
  ctx.strokeStyle = "#35d07f";
  ctx.beginPath(); ctx.moveTo(triggerX, top); ctx.lineTo(triggerX, top + ph); ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#6eb6ff";
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.beginPath();
  for (let i = 0; i < samples.length; i++) {
    const x = sx(times[i]), y = sy(samples[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = "#9da9b5";
  ctx.fillText(`${fmt(y1, " V")}`, 8, top + 8);
  ctx.fillText(`${fmt(y0, " V")}`, 8, top + ph);
  ctx.fillText(`${fmt(x0, " s")}`, left, h - 12);
  ctx.fillText(`${fmt(x1, " s")}`, left + pw - 92, h - 12);
}

function renderMetrics(data) {
  const m = data.measurements;
  const items = [
    ["frequency", fmt(m.frequency, " Hz")],
    ["period", fmt(m.period, " s")],
    ["rms", fmt(m.rms, " V")],
    ["vpp", fmt(m.vpp, " V")],
    ["duty", m.duty_cycle === null ? "n/a" : `${(m.duty_cycle * 100).toPrecision(4)} %`],
    ["rise", fmt(m.rise_time, " s")],
    ["fall", fmt(m.fall_time, " s")],
    ["overshoot", m.overshoot === null ? "n/a" : `${(m.overshoot * 100).toPrecision(4)} %`],
    ["settling", fmt(m.settling_time, " s")],
    ["trigger", data.trigger_fallback ? "fallback" : "edge"],
  ];
  readout.innerHTML = items.map(([k, v]) => `<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join("");
}

async function acquire() {
  statusEl.textContent = "acquiring";
  try {
    const response = await fetch(`/api/acquire?${params().toString()}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    draw(data);
    renderMetrics(data);
    statusEl.textContent = `${data.sample_rate.toPrecision(4)} S/s`;
  } catch (err) {
    statusEl.textContent = err.message;
  }
}

for (const id of ids) {
  document.getElementById(id).addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(acquire, 120);
  });
}
window.addEventListener("resize", acquire);
acquire();
</script>
</body>
</html>
"""


def run_acquisition(params: dict[str, str]) -> dict[str, object]:
    waveform_frequency = _float(params, "frequency", 25_000.0, minimum=1.0)
    adc_rate = _float(params, "adc_rate_msps", 5.0, minimum=0.01) * 1_000_000.0
    time_per_div = _float(params, "time_div_us", 10.0, minimum=0.01) * 1e-6
    divisions = 10

    source_rate = max(adc_rate * 8.0, waveform_frequency * 200.0, 250_000.0)
    duration = max(time_per_div * divisions * 3.0, 5.0 / waveform_frequency)
    n_samples = int(np.clip(np.ceil(duration * source_rate), 2_000, 300_000))
    source_rate = n_samples / duration

    generator = WaveformGenerator(
        kind=params.get("kind", "sine"),
        frequency=waveform_frequency,
        amplitude=_float(params, "amplitude", 1.0, minimum=0.0),
        offset=_float(params, "offset", 0.0),
        duty_cycle=_float(params, "duty_cycle", 0.5, minimum=0.001, maximum=0.999),
        rise_time=_float(params, "rise_us", 0.0, minimum=0.0) * 1e-6,
        fall_time=_float(params, "fall_us", 0.0, minimum=0.0) * 1e-6,
    )
    time, ideal = generator.generate(n_samples, source_rate)

    dac = DACModel(
        bits=_int(params, "dac_bits", 14, minimum=1),
        full_scale=5.0,
        enob=_float(params, "dac_enob", 12.5, minimum=0.1),
        seed=1,
    )
    analog = dac.convert(ideal)

    analog_lpf = _float(params, "analog_lpf_mhz", 1.0, minimum=0.0) * 1_000_000.0
    analog = AnalogPath(
        sample_rate=source_rate,
        gaussian_noise_std=_float(params, "noise_mv", 2.0, minimum=0.0) / 1000.0,
        cable_attenuation=_float(params, "attenuation", 1.0, minimum=0.0),
        lowpass_cutoff=analog_lpf if 0.0 < analog_lpf < source_rate / 2.0 else None,
        delay_samples=_float(params, "delay_samples", 0.0),
        seed=2,
    ).process(analog)

    adc_time, digitized = ADCModel(
        sample_rate=adc_rate,
        bits=_int(params, "adc_bits", 12, minimum=1),
        full_scale=5.0,
        enob=_float(params, "adc_enob", 11.5, minimum=0.1),
        aperture_jitter=_float(params, "jitter_ns", 0.0, minimum=0.0) * 1e-9,
        seed=3,
    ).convert(time, analog)

    scope_bw = _float(params, "scope_bw_mhz", 2.0, minimum=0.0) * 1_000_000.0
    record = OscilloscopeEngine(
        time_per_div=time_per_div,
        divisions=divisions,
        trigger_level=_float(params, "trigger_level", 0.0),
        trigger_edge=params.get("trigger_edge", "rising"),
        pre_trigger_fraction=_float(params, "pre_trigger", 0.25, minimum=0.0, maximum=1.0),
        bandwidth_limit=scope_bw if 0.0 < scope_bw < adc_rate / 2.0 else None,
        max_history=1,
    ).acquire(digitized, adc_rate)

    measurements = {key: _json_number(value) for key, value in MeasurementEngine(record).all().items()}
    time_out, samples_out = _decimate(record.time, record.samples, max_points=2400)
    return {
        "time": time_out.tolist(),
        "samples": samples_out.tolist(),
        "sample_rate": adc_rate,
        "trigger_level": record.trigger_level,
        "trigger_fallback": bool(record.metadata["trigger_fallback"]),
        "measurements": measurements,
    }


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            if urlparse(self.path).path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(HTML.encode())))
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, HTML.encode(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/acquire":
                try:
                    query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                    payload = json.dumps(run_acquisition(query)).encode()
                    self._send(200, payload, "application/json")
                except Exception as exc:
                    self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
            self._send(404, b"not found", "text/plain")

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving scope-sim viewer at http://{host}:{server.server_port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scope-sim interactive viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    serve(args.host, args.port)


def _float(params: dict[str, str], key: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    value = float(params.get(key, default))
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be <= {maximum}")
    return value


def _int(params: dict[str, str], key: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    value = int(float(params.get(key, default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be <= {maximum}")
    return value


def _json_number(value: float) -> float | None:
    return float(value) if isfinite(value) else None


def _decimate(time: np.ndarray, samples: np.ndarray, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if samples.size <= max_points:
        return time, samples
    step = int(np.ceil(samples.size / max_points))
    return time[::step], samples[::step]


if __name__ == "__main__":
    main()
