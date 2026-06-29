"""Virtual oscilloscope and signal acquisition simulator."""

from .adc import ADCModel
from .analog import AnalogPath
from .dac import DACModel
from .measurements import MeasurementEngine
from .scope import AcquisitionRecord, OscilloscopeEngine
from .waveform import WaveformGenerator

__all__ = [
    "ADCModel",
    "AcquisitionRecord",
    "AnalogPath",
    "DACModel",
    "MeasurementEngine",
    "OscilloscopeEngine",
    "WaveformGenerator",
    "plot_acquisition",
]


def __getattr__(name: str):
    if name == "plot_acquisition":
        from .visualization import plot_acquisition

        return plot_acquisition
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
