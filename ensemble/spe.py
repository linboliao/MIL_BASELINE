"""Core mathematics for the Stability-Prioritized Ensemble (SPE).

The paper defines a three-level hierarchy:

1. average validation-stable checkpoint states inside each architecture/fold;
2. average the five fold trajectories inside each architecture;
3. combine architecture probabilities using weights learned exclusively from
   patient-level out-of-fold (OOF) development predictions.

The supplied manuscript does not state the promised Supplementary Methods
weighting equation.  This implementation therefore makes that step explicit:
it minimizes patient-equal binary cross-entropy plus a configurable quadratic
penalty on the weighted correlation of architecture residual errors.  Weights
are constrained to the probability simplex (non-negative and summing to one).
No independent-test labels or predictions enter weight fitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


EPSILON = 1e-7


@dataclass(frozen=True)
class ArchitectureWeightFit:
    """Result and diagnostics from development-only architecture weighting."""

    architecture_names: tuple[str, ...]
    weights: tuple[float, ...]
    diversity_lambda: float
    objective: float
    patient_equal_log_loss: float
    residual_similarity_penalty: float
    optimizer: str
    converged: bool
    iterations: int

    def as_dict(self) -> dict:
        return asdict(self)


def _as_1d(values: Sequence, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
    return array


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            "probabilities must have shape [samples, architectures], "
            f"got {array.shape}."
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("probabilities cannot be empty.")
    if not np.isfinite(array).all():
        raise ValueError("probabilities contain NaN or infinite values.")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1].")
    return array


def patient_equal_sample_weights(patient_ids: Sequence) -> np.ndarray:
    """Give every patient equal total influence regardless of slide count."""
    ids = _as_1d(patient_ids, "patient_ids").astype(str)
    if len(ids) == 0:
        raise ValueError("patient_ids cannot be empty.")
    unique_ids, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= float(len(unique_ids))
    if not np.isclose(weights.sum(), 1.0):
        raise AssertionError("Patient-equal sample weights do not sum to one.")
    return weights


def residual_similarity_matrix(
    probabilities: np.ndarray,
    labels: Sequence,
    sample_weights: Sequence,
) -> np.ndarray:
    """Return the weighted residual correlation matrix across architectures."""
    probabilities = _validate_probabilities(probabilities)
    labels = _as_1d(labels, "labels").astype(np.float64)
    weights = _as_1d(sample_weights, "sample_weights").astype(np.float64)
    if len(labels) != probabilities.shape[0] or len(weights) != len(labels):
        raise ValueError("Labels/sample weights do not match probability rows.")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("SPE currently supports binary labels 0/1 only.")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("sample_weights must be non-negative and sum to one.")

    residuals = probabilities - labels[:, None]
    means = np.sum(weights[:, None] * residuals, axis=0)
    centered = residuals - means[None, :]
    covariance = centered.T @ (weights[:, None] * centered)
    scales = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(scales, scales)
    similarity = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > EPSILON,
    )
    np.fill_diagonal(similarity, 1.0)
    return np.clip((similarity + similarity.T) / 2.0, -1.0, 1.0)


def _project_probability_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w) = 1}."""
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(values) + 1)
    valid = ordered - cumulative / indices > 0
    if not valid.any():
        return np.full_like(values, 1.0 / len(values))
    rho = indices[valid][-1]
    theta = cumulative[rho - 1] / rho
    projected = np.maximum(values - theta, 0.0)
    return projected / projected.sum()


def _loss_terms(
    architecture_weights: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    similarity: np.ndarray,
    diversity_lambda: float,
) -> tuple[float, float, float]:
    ensemble_probability = np.clip(
        probabilities @ architecture_weights,
        EPSILON,
        1.0 - EPSILON,
    )
    log_loss = -np.sum(
        sample_weights
        * (
            labels * np.log(ensemble_probability)
            + (1.0 - labels) * np.log(1.0 - ensemble_probability)
        )
    )
    residual_penalty = float(architecture_weights @ similarity @ architecture_weights)
    objective = float(log_loss + diversity_lambda * residual_penalty)
    return objective, float(log_loss), residual_penalty


def _projected_gradient_fit(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    similarity: np.ndarray,
    diversity_lambda: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, bool, int]:
    architecture_count = probabilities.shape[1]
    weights = np.full(architecture_count, 1.0 / architecture_count)
    current, _, _ = _loss_terms(
        weights,
        probabilities,
        labels,
        sample_weights,
        similarity,
        diversity_lambda,
    )
    for iteration in range(1, max_iterations + 1):
        ensemble_probability = np.clip(
            probabilities @ weights, EPSILON, 1.0 - EPSILON
        )
        loss_gradient = probabilities.T @ (
            sample_weights
            * (ensemble_probability - labels)
            / (ensemble_probability * (1.0 - ensemble_probability))
        )
        gradient = loss_gradient + 2.0 * diversity_lambda * similarity @ weights
        step = 1.0
        candidate = weights
        candidate_objective = current
        for _ in range(40):
            candidate = _project_probability_simplex(weights - step * gradient)
            candidate_objective, _, _ = _loss_terms(
                candidate,
                probabilities,
                labels,
                sample_weights,
                similarity,
                diversity_lambda,
            )
            if candidate_objective <= current + 1e-12:
                break
            step *= 0.5
        delta = np.linalg.norm(candidate - weights, ord=1)
        improvement = current - candidate_objective
        weights = candidate
        current = candidate_objective
        if delta <= tolerance or abs(improvement) <= tolerance:
            return weights, True, iteration
    return weights, False, max_iterations


def fit_architecture_weights(
    probabilities: np.ndarray,
    labels: Sequence,
    patient_ids: Sequence,
    architecture_names: Sequence[str] | None = None,
    diversity_lambda: float = 0.05,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
) -> tuple[ArchitectureWeightFit, np.ndarray]:
    """Fit non-negative architecture weights from development OOF data only.

    Slides from the same patient share one unit of total loss weight.  The
    diversity term is the quadratic form of the patient-weighted residual
    correlation matrix, so correlated errors are penalized while negatively
    correlated (complementary) errors are retained.
    """
    probabilities = _validate_probabilities(probabilities)
    labels_array = _as_1d(labels, "labels").astype(np.float64)
    patients = _as_1d(patient_ids, "patient_ids").astype(str)
    if len(labels_array) != probabilities.shape[0] or len(patients) != len(labels_array):
        raise ValueError("Labels/patient IDs do not match probability rows.")
    if not np.isin(labels_array, [0.0, 1.0]).all():
        raise ValueError("SPE currently supports binary labels 0/1 only.")
    if diversity_lambda < 0:
        raise ValueError("diversity_lambda must be non-negative.")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("Invalid optimizer settings.")

    if architecture_names is None:
        names = tuple(f"architecture_{index}" for index in range(probabilities.shape[1]))
    else:
        names = tuple(str(name) for name in architecture_names)
        if len(names) != probabilities.shape[1]:
            raise ValueError("architecture_names do not match probability columns.")
        if len(set(names)) != len(names):
            raise ValueError("architecture_names must be unique.")

    sample_weights = patient_equal_sample_weights(patients)
    similarity = residual_similarity_matrix(probabilities, labels_array, sample_weights)

    optimizer_name = "projected-gradient"
    try:
        from scipy.optimize import minimize

        def objective(weight_vector: np.ndarray) -> float:
            return _loss_terms(
                weight_vector,
                probabilities,
                labels_array,
                sample_weights,
                similarity,
                diversity_lambda,
            )[0]

        start = np.full(probabilities.shape[1], 1.0 / probabilities.shape[1])
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * probabilities.shape[1],
            constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
            options={"maxiter": max_iterations, "ftol": tolerance, "disp": False},
        )
        if result.success and np.isfinite(result.x).all():
            fitted = _project_probability_simplex(result.x)
            converged = True
            iterations = int(result.nit)
            optimizer_name = "scipy-slsqp"
        else:
            fitted, converged, iterations = _projected_gradient_fit(
                probabilities,
                labels_array,
                sample_weights,
                similarity,
                diversity_lambda,
                max_iterations,
                tolerance,
            )
    except ImportError:
        fitted, converged, iterations = _projected_gradient_fit(
            probabilities,
            labels_array,
            sample_weights,
            similarity,
            diversity_lambda,
            max_iterations,
            tolerance,
        )

    objective, log_loss, penalty = _loss_terms(
        fitted,
        probabilities,
        labels_array,
        sample_weights,
        similarity,
        diversity_lambda,
    )
    fit = ArchitectureWeightFit(
        architecture_names=names,
        weights=tuple(float(value) for value in fitted),
        diversity_lambda=float(diversity_lambda),
        objective=objective,
        patient_equal_log_loss=log_loss,
        residual_similarity_penalty=penalty,
        optimizer=optimizer_name,
        converged=bool(converged),
        iterations=int(iterations),
    )
    return fit, similarity


def architecture_disagreement(
    probabilities: np.ndarray,
    architecture_weights: Sequence,
) -> np.ndarray:
    """Weighted per-slide SD around the final architecture probability."""
    probabilities = _validate_probabilities(probabilities)
    weights = _as_1d(architecture_weights, "architecture_weights").astype(np.float64)
    if len(weights) != probabilities.shape[1]:
        raise ValueError("architecture_weights do not match probability columns.")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("architecture_weights must be non-negative and sum to one.")
    mean = probabilities @ weights
    variance = np.sum(weights[None, :] * (probabilities - mean[:, None]) ** 2, axis=1)
    return np.sqrt(np.clip(variance, 0.0, None))
