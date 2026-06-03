"""Human vs AI generated text detection."""

from .data import make_demo_dataset
from .pipeline import DetectorConfig, run_detector_experiment

__all__ = ["DetectorConfig", "make_demo_dataset", "run_detector_experiment"]
