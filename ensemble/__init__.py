"""Ensemble methods used by MIL_BASELINE."""

from .spe import (
    ArchitectureWeightFit,
    architecture_disagreement,
    fit_architecture_weights,
    patient_equal_sample_weights,
)

__all__ = [
    "ArchitectureWeightFit",
    "architecture_disagreement",
    "fit_architecture_weights",
    "patient_equal_sample_weights",
]
