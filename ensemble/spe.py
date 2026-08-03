"""Core mathematics for the Stability-Prioritized Ensemble (SPE).

The paper defines a three-level hierarchy:

1. average validation-stable checkpoint states inside each architecture/fold;
2. average the five fold trajectories inside each architecture;
3. select accurate, complementary architectures using fold-held-out OOF
   development predictions;
4. combine selected architecture probabilities using weights learned
   exclusively from patient-level OOF development predictions.

This implementation is fully dynamic and supports any number of architectures (N >= 1).
It minimizes patient-equal binary cross-entropy plus a configurable quadratic
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


@dataclass(frozen=True)
class ArchitectureSelection:
    """Development-only greedy architecture selection diagnostics."""

    candidate_names: tuple[str, ...]
    eligible_names: tuple[str, ...]
    selected_names: tuple[str, ...]
    individual_log_losses: tuple[float, ...]
    null_log_loss: float
    cross_validated_log_loss: float
    min_cv_improvement: float
    require_better_than_null: bool
    max_individual_log_loss_gap: float | None
    steps: tuple[dict, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _as_1d(values: Sequence, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
    return np.ascontiguousarray(array, dtype=np.float64)


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
    return np.ascontiguousarray(array)


def patient_equal_sample_weights(patient_ids: Sequence) -> np.ndarray:
    """Give every patient equal total influence regardless of slide count."""
    ids = np.asarray(patient_ids).astype(str)
    if len(ids) == 0:
        raise ValueError("patient_ids cannot be empty.")
    unique_ids, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= float(len(unique_ids))
    if not np.isclose(weights.sum(), 1.0):
        raise AssertionError("Patient-equal sample weights do not sum to one.")
    return np.ascontiguousarray(weights)


def residual_similarity_matrix(
    probabilities: np.ndarray,
    labels: Sequence,
    sample_weights: Sequence,
) -> np.ndarray:
    """Return the weighted residual correlation matrix across architectures."""
    probabilities = _validate_probabilities(probabilities)
    labels = _as_1d(labels, "labels")
    weights = _as_1d(sample_weights, "sample_weights")

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


def _objective_and_gradient(
    architecture_weights: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    similarity: np.ndarray,
    diversity_lambda: float,
) -> tuple[float, np.ndarray]:
    """Compute both objective value and its gradient for fast SLSQP optimization."""
    ensemble_probability = np.clip(
        probabilities @ architecture_weights, EPSILON, 1.0 - EPSILON
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

    # Gradient of log_loss
    loss_gradient = probabilities.T @ (
        sample_weights
        * (ensemble_probability - labels)
        / (ensemble_probability * (1.0 - ensemble_probability))
    )
    # Gradient of quadratic penalty
    gradient = loss_gradient + 2.0 * diversity_lambda * similarity @ architecture_weights
    return objective, np.asarray(gradient)


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
    weights = np.full(architecture_count, 1.0 / architecture_count, dtype=np.float64)
    current, _ = _objective_and_gradient(
        weights, probabilities, labels, sample_weights, similarity, diversity_lambda
    )
    for iteration in range(1, max_iterations + 1):
        _, gradient = _objective_and_gradient(
            weights, probabilities, labels, sample_weights, similarity, diversity_lambda
        )
        step = 1.0
        candidate = weights
        candidate_objective = current
        for _ in range(40):
            candidate = _project_probability_simplex(weights - step * gradient)
            candidate_objective, _ = _objective_and_gradient(
                candidate, probabilities, labels, sample_weights, similarity, diversity_lambda
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
    labels_array = _as_1d(labels, "labels")
    patients = np.asarray(patient_ids, dtype=str)

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

        start = np.full(probabilities.shape[1], 1.0 / probabilities.shape[1], dtype=np.float64)
        result = minimize(
            _objective_and_gradient,
            start,
            args=(probabilities, labels_array, sample_weights, similarity, diversity_lambda),
            method="SLSQP",
            jac=True,
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


def _weighted_log_loss(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return float(
        -np.sum(
            sample_weights
            * (
                labels * np.log(clipped)
                + (1.0 - labels) * np.log(1.0 - clipped)
            )
        )
    )


def select_architectures(
    probabilities: np.ndarray,
    labels: Sequence,
    patient_ids: Sequence,
    fold_ids: Sequence,
    architecture_names: Sequence[str],
    diversity_lambda: float = 0.05,
    min_members: int = 1,
    max_members: int | None = None,
    min_cv_improvement: float = 1e-3,
    require_better_than_null: bool = True,
    max_individual_log_loss_gap: float | None = 0.10,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
) -> ArchitectureSelection:
    """Select accurate, complementary MIL members using development OOF data.

    Individual patient-equal log loss provides an accuracy gate.  Forward
    selection then adds a member only when refitting the ensemble on all other
    folds improves held-out-fold log loss.  Thus neither independent-test
    predictions nor labels influence membership.
    """
    probabilities = _validate_probabilities(probabilities)
    labels_array = _as_1d(labels, "labels")
    patients = np.asarray(patient_ids, dtype=str)
    folds = np.asarray(fold_ids)
    names = tuple(str(name) for name in architecture_names)
    sample_count, architecture_count = probabilities.shape

    if any(len(values) != sample_count for values in (labels_array, patients, folds)):
        raise ValueError("Labels/patient IDs/fold IDs do not match probability rows.")
    if len(names) != architecture_count or len(set(names)) != architecture_count:
        raise ValueError("architecture_names must uniquely match probability columns.")
    if not np.isin(labels_array, [0.0, 1.0]).all():
        raise ValueError("SPE currently supports binary labels 0/1 only.")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("Architecture selection requires at least two OOF folds.")
    patient_fold_counts = np.asarray(
        [len(np.unique(folds[patients == patient])) for patient in np.unique(patients)]
    )
    if (patient_fold_counts > 1).any():
        raise ValueError("A patient cannot occur in more than one OOF fold.")
    if min_members < 1 or min_members > architecture_count:
        raise ValueError("min_members must be between 1 and the candidate count.")
    if max_members is None:
        max_members = architecture_count
    if max_members < min_members or max_members > architecture_count:
        raise ValueError("max_members must be between min_members and candidate count.")
    if min_cv_improvement < 0:
        raise ValueError("min_cv_improvement must be non-negative.")
    if max_individual_log_loss_gap is not None and max_individual_log_loss_gap < 0:
        raise ValueError("max_individual_log_loss_gap must be non-negative or null.")

    sample_weights = patient_equal_sample_weights(patients)
    individual_losses = np.asarray(
        [
            _weighted_log_loss(probabilities[:, index], labels_array, sample_weights)
            for index in range(architecture_count)
        ]
    )
    prevalence = float(np.sum(sample_weights * labels_array))
    null_probability = np.full(sample_count, prevalence, dtype=np.float64)
    null_loss = _weighted_log_loss(null_probability, labels_array, sample_weights)
    best_individual = float(individual_losses.min())
    eligible_mask = np.ones(architecture_count, dtype=bool)
    if require_better_than_null:
        eligible_mask &= individual_losses < null_loss
    if max_individual_log_loss_gap is not None:
        eligible_mask &= individual_losses <= best_individual + max_individual_log_loss_gap
    eligible = [index for index in range(architecture_count) if eligible_mask[index]]
    if len(eligible) < min_members:
        raise ValueError(
            f"Only {len(eligible)} architectures passed the accuracy gate, fewer "
            f"than selection.min_members={min_members}. Relax the locked gate or "
            "do not construct an ensemble."
        )

    score_cache: dict[tuple[int, ...], float] = {}

    def cross_validated_score(indices: Sequence[int]) -> float:
        key = tuple(sorted(indices))
        if key in score_cache:
            return score_cache[key]
        cross_fitted = np.empty(sample_count, dtype=np.float64)
        # Pre-slice columns to avoid repeated advanced indexing overhead
        sub_probs = probabilities[:, key]
        for fold in unique_folds:
            validation_mask = folds == fold
            training_mask = ~validation_mask
            fit, _ = fit_architecture_weights(
                sub_probs[training_mask],
                labels_array[training_mask],
                patients[training_mask],
                architecture_names=[names[index] for index in key],
                diversity_lambda=diversity_lambda,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            cross_fitted[validation_mask] = (
                sub_probs[validation_mask] @ np.asarray(fit.weights)
            )
        score = _weighted_log_loss(cross_fitted, labels_array, sample_weights)
        score_cache[key] = score
        return score

    first = min(eligible, key=lambda index: (individual_losses[index], names[index]))
    selected = [first]
    current_score = cross_validated_score(selected)
    steps: list[dict] = [
        {
            "step": 1,
            "added": names[first],
            "cross_validated_log_loss": current_score,
            "improvement": None,
        }
    ]
    while len(selected) < max_members:
        remaining = [index for index in eligible if index not in selected]
        if not remaining:
            break
        scored = [
            (cross_validated_score([*selected, index]), names[index], index)
            for index in remaining
        ]
        candidate_score, _, candidate = min(scored)
        improvement = current_score - candidate_score
        if len(selected) >= min_members and improvement < min_cv_improvement:
            break
        selected.append(candidate)
        current_score = candidate_score
        steps.append(
            {
                "step": len(selected),
                "added": names[candidate],
                "cross_validated_log_loss": current_score,
                "improvement": improvement,
            }
        )

    return ArchitectureSelection(
        candidate_names=names,
        eligible_names=tuple(names[index] for index in eligible),
        selected_names=tuple(names[index] for index in selected),
        individual_log_losses=tuple(float(value) for value in individual_losses),
        null_log_loss=null_loss,
        cross_validated_log_loss=current_score,
        min_cv_improvement=float(min_cv_improvement),
        require_better_than_null=bool(require_better_than_null),
        max_individual_log_loss_gap=(
            None
            if max_individual_log_loss_gap is None
            else float(max_individual_log_loss_gap)
        ),
        steps=tuple(steps),
    )


def architecture_disagreement(
    probabilities: np.ndarray,
    architecture_weights: Sequence,
) -> np.ndarray:
    """Weighted per-slide SD around the final architecture probability."""
    probabilities = _validate_probabilities(probabilities)
    weights = _as_1d(architecture_weights, "architecture_weights")
    if len(weights) != probabilities.shape[1]:
        raise ValueError("architecture_weights do not match probability columns.")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("architecture_weights must be non-negative and sum to one.")
    mean = probabilities @ weights
    variance = np.sum(weights[None, :] * (probabilities - mean[:, None]) ** 2, axis=1)
    return np.sqrt(np.clip(variance, 0.0, None))