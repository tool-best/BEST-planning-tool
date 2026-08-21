"""modechoice: a small, transparent MNL mode-share prediction toolkit."""

from .config import MODES
from .generate_data import generate
from .mnl import (
    estimate,
    predicted_shares,
    choice_probabilities,
    apply_adjustments,
    sensitivity,
    MNLResult,
)

__all__ = [
    "MODES",
    "generate",
    "estimate",
    "predicted_shares",
    "choice_probabilities",
    "apply_adjustments",
    "sensitivity",
    "MNLResult",
]

__version__ = "0.1.0"
