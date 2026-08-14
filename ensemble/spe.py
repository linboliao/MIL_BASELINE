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
from itertools import combinations
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


@dataclass(frozen=True)
class ConstrainedStackingFit:
    """Diagnostics for the shrinkage-constrained linear OOF stacker."""

    architecture_names: tuple[str, ...]
    weights: tuple[float, ...]
    diversity_lambda: float
    shrinkage_lambda: float
    max_weight: float
    min_effective_members: float
    effective_members: float
    objective: float
    patient_equal_log_loss: float
    residual_similarity_penalty: float
    shrinkage_penalty: float
    optimizer: str
    converged: bool
    iterations: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BalancedAccuracySelection:
    """Fold-held-out architecture selection driven by Balanced Accuracy."""

    candidate_names: tuple[str, ...]
    eligible_names: tuple[str, ...]
    selected_names: tuple[str, ...]
    individual_balanced_accuracies: tuple[float, ...]
    cross_validated_balanced_accuracy: float
    classification_threshold: float
    min_cv_improvement: float
    max_individual_bacc_gap: float | None
    steps: tuple[dict, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MacroF1Selection:
    """Fold-held-out architecture selection driven by binary macro F1."""

    candidate_names: tuple[str, ...]
    eligible_names: tuple[str, ...]
    selected_names: tuple[str, ...]
    individual_macro_f1_scores: tuple[float, ...]
    cross_validated_macro_f1: float
    classification_threshold: float
    min_cv_improvement: float
    max_individual_macro_f1_gap: float | None
    steps: tuple[dict, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DevelopmentRobustAnchorSelection:
    """Development-only anchor selected under accuracy and stability constraints."""

    strategy: str
    candidate_names: tuple[str, ...]
    accuracy_eligible_names: tuple[str, ...]
    eligible_names: tuple[str, ...]
    selected_name: str
    best_bacc_name: str
    best_balanced_accuracy: float
    bacc_noninferiority_margin: float
    max_sensitivity_drop: float | None
    calibration_tolerance: float
    state_variance_tolerance: float
    individual_balanced_accuracies: tuple[float, ...]
    individual_sensitivities: tuple[float, ...]
    individual_specificities: tuple[float, ...]
    class_patient_equal_log_losses: tuple[float, ...]
    mean_state_variances: tuple[float | None, ...]
    fold_mean_balanced_accuracies: tuple[float, ...]
    fold_bacc_standard_deviations: tuple[float, ...]
    worst_fold_balanced_accuracies: tuple[float, ...]
    state_variance_used: bool
    selection_order: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Top1AnchorFallbackSelection:
    """Risk-controlled Top1-anchored ensemble selected on development OOF data."""

    candidate_names: tuple[str, ...]
    anchor_name: str
    fixed_anchor: bool
    anchor_selection_strategy: str
    anchor_selection_diagnostics: dict | None
    proposed_selected_names: tuple[str, ...]
    deployed_names: tuple[str, ...]
    individual_balanced_accuracies: tuple[float, ...]
    individual_sensitivities: tuple[float, ...]
    proposed_weights: tuple[float, ...]
    deployed_weights: tuple[float, ...]
    anchor_min_weight: float
    anchor_confidence_low: float | None
    anchor_confidence_high: float | None
    min_override_agreement: int
    anchor_balanced_accuracy: float
    ensemble_balanced_accuracy: float
    balanced_accuracy_gain: float
    min_oof_bacc_gain: float | None
    bacc_noninferiority_margin: float | None
    anchor_sensitivity: float
    ensemble_sensitivity: float
    sensitivity_gain: float
    max_oof_sensitivity_drop: float | None
    anchor_specificity: float
    ensemble_specificity: float
    specificity_gain: float
    max_oof_specificity_drop: float | None
    anchor_macro_f1: float
    ensemble_macro_f1: float
    anchor_roc_auc: float
    ensemble_roc_auc: float
    anchor_average_precision: float
    ensemble_average_precision: float
    selection_objective: str
    non_decreasing_folds: int
    min_non_decreasing_folds: int
    non_decreasing_sensitivity_folds: int
    min_non_decreasing_sensitivity_folds: int | None
    fallback_triggered: bool
    fallback_reasons: tuple[str, ...]
    fold_diagnostics: tuple[dict, ...]
    steps: tuple[dict, ...]

    @property
    def selected_names(self) -> tuple[str, ...]:
        """Names actually deployed after applying the automatic guardrail."""
        return self.deployed_names

    def as_dict(self) -> dict:
        values = asdict(self)
        values["selected_names"] = self.selected_names
        values["strategy"] = "top1_anchor_with_automatic_fallback"
        return values


@dataclass(frozen=True)
class DiversityVetoSelection:
    """OOF-only selection for a specificity-oriented two-member ensemble."""

    candidate_names: tuple[str, ...]
    eligible_veto_names: tuple[str, ...]
    anchor_name: str
    veto_name: str
    individual_balanced_accuracies: tuple[float, ...]
    anchor_veto_residual_similarity: float
    max_veto_bacc_gap: float
    classification_threshold: float

    @property
    def selected_names(self) -> tuple[str, str]:
        return (self.anchor_name, self.veto_name)

    def as_dict(self) -> dict:
        values = asdict(self)
        values["selected_names"] = self.selected_names
        values["aggregation"] = "minimum_probability"
        return values


@dataclass(frozen=True)
class SensitivityConstrainedSelection:
    """OOF-only equal-weight subset selected under a sensitivity floor."""

    candidate_names: tuple[str, ...]
    selected_names: tuple[str, ...]
    reference_sensitivity: float
    sensitivity_floor: float
    cross_validated_balanced_accuracy: float
    cross_validated_sensitivity: float
    cross_validated_specificity: float
    classification_threshold: float
    min_members: int
    max_members: int
    evaluated_subsets: int

    def as_dict(self) -> dict:
        values = asdict(self)
        values["aggregation"] = "equal_probability_mean"
        return values


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


def class_patient_equal_sample_weights(
    labels: Sequence, patient_ids: Sequence
) -> np.ndarray:
    """Give each class half the loss weight while retaining patient equality."""
    labels_array = _as_1d(labels, "labels")
    if not np.isin(labels_array, [0.0, 1.0]).all():
        raise ValueError("Class-balanced weighting requires binary labels 0/1.")
    if np.unique(labels_array).size != 2:
        raise ValueError("Class-balanced weighting requires both classes.")
    weights = patient_equal_sample_weights(patient_ids)
    for label in (0.0, 1.0):
        mask = labels_array == label
        weights[mask] *= 0.5 / weights[mask].sum()
    if not np.isclose(weights.sum(), 1.0):
        raise AssertionError("Class/patient-equal sample weights do not sum to one.")
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


def _project_capped_simplex(values: np.ndarray, upper: float) -> np.ndarray:
    """Project onto {sum(w)=1, 0<=w<=upper} by scalar bisection."""
    values = np.asarray(values, dtype=np.float64)
    if upper * len(values) < 1.0 - 1e-12:
        raise ValueError("max_weight is infeasible for the architecture count.")
    low = float(np.min(values) - upper)
    high = float(np.max(values))
    for _ in range(100):
        midpoint = (low + high) / 2.0
        projected = np.clip(values - midpoint, 0.0, upper)
        if projected.sum() > 1.0:
            low = midpoint
        else:
            high = midpoint
    projected = np.clip(values - high, 0.0, upper)
    # The bisection error is tiny; this correction retains the box constraint.
    residual = 1.0 - float(projected.sum())
    if abs(residual) > 1e-12:
        free = np.flatnonzero(
            projected < upper - 1e-12 if residual > 0 else projected > 1e-12
        )
        for index in free:
            room = upper - projected[index] if residual > 0 else projected[index]
            delta = np.sign(residual) * min(abs(residual), room)
            projected[index] += delta
            residual -= delta
            if abs(residual) <= 1e-12:
                break
    return projected


def _enforce_effective_members(
    weights: np.ndarray, min_effective_members: float
) -> np.ndarray:
    """Shrink a simplex vector toward uniform until its effective size is valid."""
    if min_effective_members <= 1.0:
        return weights
    uniform = np.full_like(weights, 1.0 / len(weights))
    maximum_l2 = 1.0 / min_effective_members
    if float(weights @ weights) <= maximum_l2 + 1e-12:
        return weights
    direction = weights - uniform
    squared_norm = float(direction @ direction)
    allowed = max(0.0, maximum_l2 - 1.0 / len(weights))
    alpha = min(1.0, np.sqrt(allowed / max(squared_norm, EPSILON)))
    return uniform + alpha * direction


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
    class_balance: bool = False,
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

    sample_weights = (
        class_patient_equal_sample_weights(labels_array, patients)
        if class_balance
        else patient_equal_sample_weights(patients)
    )
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


def fit_constrained_linear_stacking(
    probabilities: np.ndarray,
    labels: Sequence,
    patient_ids: Sequence,
    architecture_names: Sequence[str] | None = None,
    diversity_lambda: float = 0.02,
    shrinkage_lambda: float = 0.10,
    max_weight: float | None = None,
    min_effective_members: float | None = None,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
    class_balance: bool = True,
) -> tuple[ConstrainedStackingFit, np.ndarray]:
    """Fit a conservative linear stacker using development OOF predictions.

    The objective combines patient/class-balanced BCE, residual-correlation
    regularization, and L2 shrinkage toward uniform averaging.  A per-member
    weight cap and a lower bound on ``1 / sum(w**2)`` prevent collapse onto a
    small architecture subset.  The returned model is therefore learnable but
    remains close to the manuscript's equal-weight reliability prior.
    """
    probabilities = _validate_probabilities(probabilities)
    labels_array = _as_1d(labels, "labels")
    patients = np.asarray(patient_ids, dtype=str)
    architecture_count = probabilities.shape[1]
    if len(labels_array) != len(probabilities) or len(patients) != len(labels_array):
        raise ValueError("Labels/patient IDs do not match probability rows.")
    if not np.isin(labels_array, [0.0, 1.0]).all():
        raise ValueError("Constrained stacking supports binary labels 0/1 only.")
    if diversity_lambda < 0 or shrinkage_lambda < 0:
        raise ValueError("Regularization strengths must be non-negative.")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("Invalid optimizer settings.")

    names = (
        tuple(f"architecture_{index}" for index in range(architecture_count))
        if architecture_names is None
        else tuple(str(name) for name in architecture_names)
    )
    if len(names) != architecture_count or len(set(names)) != len(names):
        raise ValueError("architecture_names must uniquely match probability columns.")

    cap = 1.0 if max_weight is None else float(max_weight)
    if cap <= 0 or cap > 1 or cap * architecture_count < 1.0 - 1e-12:
        raise ValueError("max_weight must be feasible and lie in (0, 1].")
    minimum_effective = (
        1.0 if min_effective_members is None else float(min_effective_members)
    )
    if minimum_effective < 1.0 or minimum_effective > architecture_count:
        raise ValueError("min_effective_members must lie in [1, architecture_count].")

    sample_weights = (
        class_patient_equal_sample_weights(labels_array, patients)
        if class_balance
        else patient_equal_sample_weights(patients)
    )
    similarity = residual_similarity_matrix(probabilities, labels_array, sample_weights)
    uniform = np.full(architecture_count, 1.0 / architecture_count, dtype=np.float64)

    def objective_and_gradient(weights: np.ndarray) -> tuple[float, np.ndarray]:
        ensemble = np.clip(probabilities @ weights, EPSILON, 1.0 - EPSILON)
        log_loss = _weighted_log_loss(ensemble, labels_array, sample_weights)
        diversity = float(weights @ similarity @ weights)
        shrinkage = float((weights - uniform) @ (weights - uniform))
        gradient = probabilities.T @ (
            sample_weights
            * (ensemble - labels_array)
            / (ensemble * (1.0 - ensemble))
        )
        gradient += 2.0 * diversity_lambda * similarity @ weights
        gradient += 2.0 * shrinkage_lambda * (weights - uniform)
        return (
            float(log_loss + diversity_lambda * diversity + shrinkage_lambda * shrinkage),
            np.asarray(gradient),
        )

    optimizer_name = "projected-gradient"
    converged = False
    iterations = 0
    fitted = uniform.copy()
    try:
        from scipy.optimize import minimize

        constraints: list[dict] = [
            {"type": "eq", "fun": lambda value: float(value.sum() - 1.0)}
        ]
        if minimum_effective > 1.0:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda value: float(
                        1.0 / minimum_effective - value @ value
                    ),
                }
            )
        result = minimize(
            objective_and_gradient,
            uniform,
            method="SLSQP",
            jac=True,
            bounds=[(0.0, cap)] * architecture_count,
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": tolerance, "disp": False},
        )
        if result.success and np.isfinite(result.x).all():
            fitted = _project_capped_simplex(result.x, cap)
            fitted = _enforce_effective_members(fitted, minimum_effective)
            converged = True
            iterations = int(result.nit)
            optimizer_name = "scipy-slsqp"
    except ImportError:
        pass

    if not converged:
        current, _ = objective_and_gradient(fitted)
        for iterations in range(1, max_iterations + 1):
            _, gradient = objective_and_gradient(fitted)
            step = 1.0
            candidate = fitted
            candidate_objective = current
            for _ in range(40):
                candidate = _project_capped_simplex(fitted - step * gradient, cap)
                candidate = _enforce_effective_members(candidate, minimum_effective)
                candidate_objective, _ = objective_and_gradient(candidate)
                if candidate_objective <= current + 1e-12:
                    break
                step *= 0.5
            delta = float(np.linalg.norm(candidate - fitted, ord=1))
            improvement = current - candidate_objective
            fitted, current = candidate, candidate_objective
            if delta <= tolerance or abs(improvement) <= tolerance:
                converged = True
                break

    ensemble = np.clip(probabilities @ fitted, EPSILON, 1.0 - EPSILON)
    log_loss = _weighted_log_loss(ensemble, labels_array, sample_weights)
    diversity = float(fitted @ similarity @ fitted)
    shrinkage = float((fitted - uniform) @ (fitted - uniform))
    objective = float(
        log_loss + diversity_lambda * diversity + shrinkage_lambda * shrinkage
    )
    effective = float(1.0 / np.sum(fitted**2))
    fit = ConstrainedStackingFit(
        architecture_names=names,
        weights=tuple(float(value) for value in fitted),
        diversity_lambda=float(diversity_lambda),
        shrinkage_lambda=float(shrinkage_lambda),
        max_weight=cap,
        min_effective_members=minimum_effective,
        effective_members=effective,
        objective=objective,
        patient_equal_log_loss=float(log_loss),
        residual_similarity_penalty=diversity,
        shrinkage_penalty=shrinkage,
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


def _balanced_accuracy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> float:
    predictions = probabilities >= threshold
    positive = labels == 1.0
    negative = labels == 0.0
    if not positive.any() or not negative.any():
        return float("nan")
    sensitivity = float(predictions[positive].mean())
    specificity = float((~predictions[negative]).mean())
    return (sensitivity + specificity) / 2.0


def _sensitivity(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    """Return binary sensitivity, or NaN when no positive examples exist."""
    positive = labels == 1.0
    if not positive.any():
        return float("nan")
    return float((probabilities[positive] >= threshold).mean())


def _specificity(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    """Return binary specificity, or NaN when no negative examples exist."""
    negative = labels == 0.0
    if not negative.any():
        return float("nan")
    return float((probabilities[negative] < threshold).mean())


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute binary AUROC using average ranks for tied probabilities."""
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive_count = int(np.sum(labels == 1.0))
    negative_count = int(np.sum(labels == 0.0))
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    ranks = np.empty(len(probabilities), dtype=np.float64)
    start = 0
    while start < len(probabilities):
        end = start + 1
        while (
            end < len(probabilities)
            and sorted_probabilities[end] == sorted_probabilities[start]
        ):
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(np.sum(ranks[labels == 1.0]))
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute binary average precision from descending score thresholds."""
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive_count = int(np.sum(labels == 1.0))
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(labels):
        end = start + 1
        while (
            end < len(labels)
            and sorted_probabilities[end] == sorted_probabilities[start]
        ):
            end += 1
        group_positive = int(np.sum(sorted_labels[start:end] == 1.0))
        true_positives += group_positive
        false_positives += end - start - group_positive
        recall = true_positives / positive_count
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(average_precision)


def apply_anchor_confidence_guard(
    anchor_probability: Sequence,
    ensemble_probability: Sequence,
    confidence_low: float | None,
    confidence_high: float | None,
) -> np.ndarray:
    """Keep high-confidence anchor predictions and blend only uncertain rows."""
    anchor = _as_1d(anchor_probability, "anchor_probability")
    ensemble = _as_1d(ensemble_probability, "ensemble_probability")
    if len(anchor) != len(ensemble):
        raise ValueError("Anchor and ensemble probabilities must have equal length.")
    if confidence_low is None and confidence_high is None:
        return ensemble.copy()
    if confidence_low is None or confidence_high is None:
        raise ValueError("Both confidence_low and confidence_high must be configured.")
    if not 0.0 <= confidence_low < confidence_high <= 1.0:
        raise ValueError(
            "Anchor confidence bounds must satisfy 0 <= low < high <= 1."
        )
    protected = (anchor <= confidence_low) | (anchor >= confidence_high)
    return np.where(protected, anchor, ensemble)


def apply_anchor_decision_guard(
    anchor_probability: Sequence,
    ensemble_probability: Sequence,
    complement_probabilities: np.ndarray,
    classification_threshold: float,
    confidence_low: float | None,
    confidence_high: float | None,
    min_override_agreement: int = 0,
) -> np.ndarray:
    """Protect anchor decisions unless enough active complements support a flip."""
    anchor = _as_1d(anchor_probability, "anchor_probability")
    guarded = apply_anchor_confidence_guard(
        anchor,
        ensemble_probability,
        confidence_low,
        confidence_high,
    )
    complements = np.asarray(complement_probabilities, dtype=np.float64)
    if complements.ndim != 2 or complements.shape[0] != len(anchor):
        raise ValueError(
            "complement_probabilities must be a row-aligned two-dimensional matrix."
        )
    if min_override_agreement < 0:
        raise ValueError("min_override_agreement must be non-negative.")
    if min_override_agreement == 0:
        return guarded
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must lie strictly between 0 and 1.")

    anchor_prediction = anchor >= classification_threshold
    guarded_prediction = guarded >= classification_threshold
    flips = anchor_prediction != guarded_prediction
    if not flips.any():
        return guarded
    if complements.shape[1] < min_override_agreement:
        guarded[flips] = anchor[flips]
        return guarded
    complement_predictions = complements >= classification_threshold
    agreement = np.sum(
        complement_predictions == guarded_prediction[:, None], axis=1
    )
    blocked = flips & (agreement < min_override_agreement)
    guarded[blocked] = anchor[blocked]
    return guarded


def binary_macro_f1(
    labels: Sequence,
    probabilities: Sequence,
    threshold: float = 0.5,
) -> float:
    """Return the unweighted mean of class-0 and class-1 F1 scores.

    Both classes are always included, matching ``sklearn.metrics.f1_score``
    with ``labels=[0, 1], average="macro", zero_division=0``. This explicit
    implementation keeps SPE's core math independent of scikit-learn.
    """
    labels_array = _as_1d(labels, "labels")
    probabilities_array = _as_1d(probabilities, "probabilities")
    if len(labels_array) != len(probabilities_array):
        raise ValueError("Labels and probabilities must have the same length.")
    if not np.isin(labels_array, [0.0, 1.0]).all():
        raise ValueError("Macro-F1 requires binary labels 0/1.")
    if not np.isfinite(probabilities_array).all():
        raise ValueError("probabilities contain NaN or infinite values.")
    if ((probabilities_array < 0.0) | (probabilities_array > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1].")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between 0 and 1.")

    predictions = (probabilities_array >= threshold).astype(np.int8)
    scores: list[float] = []
    for class_id in (0, 1):
        true_positive = int(
            np.sum((predictions == class_id) & (labels_array == class_id))
        )
        false_positive = int(
            np.sum((predictions == class_id) & (labels_array != class_id))
        )
        false_negative = int(
            np.sum((predictions != class_id) & (labels_array == class_id))
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def select_architectures_for_macro_f1(
    probabilities: np.ndarray,
    labels: Sequence,
    patient_ids: Sequence,
    fold_ids: Sequence,
    architecture_names: Sequence[str],
    diversity_lambda: float = 0.05,
    classification_threshold: float = 0.5,
    min_members: int = 1,
    max_members: int | None = None,
    min_cv_improvement: float = 1e-3,
    max_individual_macro_f1_gap: float | None = 0.10,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
) -> MacroF1Selection:
    """Forward-select members using fold-held-out development macro F1.

    For each candidate subset, architecture weights are fitted on four OOF
    folds with class/patient-balanced BCE and applied to the untouched fifth
    fold. The pooled cross-fitted predictions are scored by macro F1 at the
    fixed classification threshold. Independent-test data never enter this
    procedure.
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
        raise ValueError("Macro-F1 selection requires binary labels 0/1.")
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must lie strictly between 0 and 1.")
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
    if (
        max_individual_macro_f1_gap is not None
        and max_individual_macro_f1_gap < 0
    ):
        raise ValueError(
            "max_individual_macro_f1_gap must be non-negative or null."
        )

    individual_scores = np.asarray(
        [
            binary_macro_f1(
                labels_array,
                probabilities[:, index],
                classification_threshold,
            )
            for index in range(architecture_count)
        ]
    )
    best_individual = float(individual_scores.max())
    eligible_mask = individual_scores > 0.0
    if max_individual_macro_f1_gap is not None:
        eligible_mask &= (
            individual_scores >= best_individual - max_individual_macro_f1_gap
        )
    eligible = [index for index in range(architecture_count) if eligible_mask[index]]
    if len(eligible) < min_members:
        raise ValueError(
            f"Only {len(eligible)} architectures passed the macro-F1 gate, fewer "
            f"than selection.min_members={min_members}."
        )

    score_cache: dict[tuple[int, ...], float] = {}

    def cross_validated_score(indices: Sequence[int]) -> float:
        key = tuple(sorted(indices))
        if key in score_cache:
            return score_cache[key]
        cross_fitted = np.empty(sample_count, dtype=np.float64)
        sub_probabilities = probabilities[:, key]
        for fold in unique_folds:
            validation_mask = folds == fold
            training_mask = ~validation_mask
            fit, _ = fit_architecture_weights(
                sub_probabilities[training_mask],
                labels_array[training_mask],
                patients[training_mask],
                architecture_names=[names[index] for index in key],
                diversity_lambda=diversity_lambda,
                max_iterations=max_iterations,
                tolerance=tolerance,
                class_balance=True,
            )
            cross_fitted[validation_mask] = (
                sub_probabilities[validation_mask] @ np.asarray(fit.weights)
            )
        score = binary_macro_f1(
            labels_array, cross_fitted, classification_threshold
        )
        score_cache[key] = score
        return score

    first = min(eligible, key=lambda index: (-individual_scores[index], names[index]))
    selected = [first]
    current_score = cross_validated_score(selected)
    steps: list[dict] = [
        {
            "step": 1,
            "added": names[first],
            "cross_validated_macro_f1": current_score,
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
        candidate_score, _, candidate = min(
            scored, key=lambda item: (-item[0], item[1])
        )
        improvement = candidate_score - current_score
        if len(selected) >= min_members and improvement < min_cv_improvement:
            break
        selected.append(candidate)
        current_score = candidate_score
        steps.append(
            {
                "step": len(selected),
                "added": names[candidate],
                "cross_validated_macro_f1": current_score,
                "improvement": improvement,
            }
        )

    return MacroF1Selection(
        candidate_names=names,
        eligible_names=tuple(names[index] for index in eligible),
        selected_names=tuple(names[index] for index in selected),
        individual_macro_f1_scores=tuple(float(value) for value in individual_scores),
        cross_validated_macro_f1=float(current_score),
        classification_threshold=float(classification_threshold),
        min_cv_improvement=float(min_cv_improvement),
        max_individual_macro_f1_gap=(
            None
            if max_individual_macro_f1_gap is None
            else float(max_individual_macro_f1_gap)
        ),
        steps=tuple(steps),
    )


def select_architectures_for_balanced_accuracy(
    probabilities: np.ndarray,
    labels: Sequence,
    patient_ids: Sequence,
    fold_ids: Sequence,
    architecture_names: Sequence[str],
    diversity_lambda: float = 0.05,
    classification_threshold: float = 0.5,
    min_members: int = 1,
    max_members: int | None = None,
    min_cv_improvement: float = 1e-3,
    max_individual_bacc_gap: float | None = 0.10,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
) -> BalancedAccuracySelection:
    """Forward-select members by fold-held-out Balanced Accuracy.

    Weights inside every four-fold training split use class/patient-balanced
    binary cross-entropy plus the residual-correlation penalty. Candidate
    membership is then scored only on the untouched fifth-fold probabilities.
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
        raise ValueError("Balanced Accuracy selection requires binary labels 0/1.")
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must lie strictly between 0 and 1.")
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
    if max_individual_bacc_gap is not None and max_individual_bacc_gap < 0:
        raise ValueError("max_individual_bacc_gap must be non-negative or null.")

    individual_scores = np.asarray(
        [
            _balanced_accuracy(
                labels_array,
                probabilities[:, index],
                classification_threshold,
            )
            for index in range(architecture_count)
        ]
    )
    best_individual = float(np.nanmax(individual_scores))
    eligible_mask = individual_scores > 0.5
    if max_individual_bacc_gap is not None:
        eligible_mask &= individual_scores >= best_individual - max_individual_bacc_gap
    eligible = [index for index in range(architecture_count) if eligible_mask[index]]
    if len(eligible) < min_members:
        raise ValueError(
            f"Only {len(eligible)} architectures passed the BAcc gate, fewer "
            f"than selection.min_members={min_members}."
        )

    score_cache: dict[tuple[int, ...], float] = {}

    def cross_validated_score(indices: Sequence[int]) -> float:
        key = tuple(sorted(indices))
        if key in score_cache:
            return score_cache[key]
        cross_fitted = np.empty(sample_count, dtype=np.float64)
        sub_probabilities = probabilities[:, key]
        for fold in unique_folds:
            validation_mask = folds == fold
            training_mask = ~validation_mask
            fit, _ = fit_architecture_weights(
                sub_probabilities[training_mask],
                labels_array[training_mask],
                patients[training_mask],
                architecture_names=[names[index] for index in key],
                diversity_lambda=diversity_lambda,
                max_iterations=max_iterations,
                tolerance=tolerance,
                class_balance=True,
            )
            cross_fitted[validation_mask] = (
                sub_probabilities[validation_mask] @ np.asarray(fit.weights)
            )
        score = _balanced_accuracy(
            labels_array, cross_fitted, classification_threshold
        )
        score_cache[key] = score
        return score

    first = min(eligible, key=lambda index: (-individual_scores[index], names[index]))
    selected = [first]
    current_score = cross_validated_score(selected)
    steps: list[dict] = [
        {
            "step": 1,
            "added": names[first],
            "cross_validated_balanced_accuracy": current_score,
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
        candidate_score, _, candidate = min(
            scored, key=lambda item: (-item[0], item[1])
        )
        improvement = candidate_score - current_score
        if len(selected) >= min_members and improvement < min_cv_improvement:
            break
        selected.append(candidate)
        current_score = candidate_score
        steps.append(
            {
                "step": len(selected),
                "added": names[candidate],
                "cross_validated_balanced_accuracy": current_score,
                "improvement": improvement,
            }
        )

    return BalancedAccuracySelection(
        candidate_names=names,
        eligible_names=tuple(names[index] for index in eligible),
        selected_names=tuple(names[index] for index in selected),
        individual_balanced_accuracies=tuple(float(value) for value in individual_scores),
        cross_validated_balanced_accuracy=float(current_score),
        classification_threshold=float(classification_threshold),
        min_cv_improvement=float(min_cv_improvement),
        max_individual_bacc_gap=(
            None
            if max_individual_bacc_gap is None
            else float(max_individual_bacc_gap)
        ),
        steps=tuple(steps),
    )


def select_development_robust_anchor(
    probabilities: np.ndarray,
    labels: Sequence,
    patient_ids: Sequence,
    fold_ids: Sequence,
    architecture_names: Sequence[str],
    state_variances: np.ndarray | None = None,
    classification_threshold: float = 0.5,
    bacc_noninferiority_margin: float = 0.002,
    max_sensitivity_drop: float | None = 0.005,
    calibration_tolerance: float = 0.001,
    state_variance_tolerance: float = 1.0e-5,
    selection_strategy: str = "bacc_equivalent_calibrated_stable",
) -> DevelopmentRobustAnchorSelection:
    """Choose a general-purpose anchor from development OOF predictions.

    Balanced Accuracy defines an equivalence set rather than forcing the
    numerically best architecture to become the anchor.  A sensitivity floor
    is applied inside that set. ``bacc_equivalent_calibrated_stable`` ranks by
    calibration and checkpoint stability. ``bacc_equivalent_maximin_stable``
    ranks by worst-fold BAcc, mean-fold BAcc, fold dispersion, checkpoint-state
    variance, and calibration. Calibration and state variance are discretized
    at configured resolutions to avoid treating numerical noise as meaningful.

    ``state_variances`` must contain the per-sample variance across validation-
    selected checkpoint states.  Independent-test labels or metrics are never
    accepted by this function.
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
    if not np.isin(labels_array, [0.0, 1.0]).all() or np.unique(labels_array).size != 2:
        raise ValueError("Robust anchor selection requires both binary labels 0/1.")
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must lie strictly between 0 and 1.")
    if bacc_noninferiority_margin < 0:
        raise ValueError("bacc_noninferiority_margin must be non-negative.")
    if max_sensitivity_drop is not None and max_sensitivity_drop < 0:
        raise ValueError("max_sensitivity_drop must be non-negative or null.")
    if calibration_tolerance <= 0 or state_variance_tolerance <= 0:
        raise ValueError("Anchor metric tolerances must be positive.")
    allowed_strategies = {
        "bacc_equivalent_calibrated_stable",
        "bacc_equivalent_maximin_stable",
    }
    if selection_strategy not in allowed_strategies:
        raise ValueError(
            f"selection_strategy must be one of {sorted(allowed_strategies)}."
        )

    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("Robust anchor selection requires at least two OOF folds.")
    patient_fold_counts = np.asarray(
        [len(np.unique(folds[patients == patient])) for patient in np.unique(patients)]
    )
    if (patient_fold_counts > 1).any():
        raise ValueError("A patient cannot occur in more than one OOF fold.")

    state_array = None
    if state_variances is not None:
        state_array = np.asarray(state_variances, dtype=np.float64)
        if state_array.shape != probabilities.shape:
            raise ValueError(
                "state_variances must have the same [samples, architectures] "
                f"shape as probabilities, got {state_array.shape}."
            )
        if not np.isfinite(state_array).all() or (state_array < 0.0).any():
            raise ValueError("state_variances must be finite and non-negative.")

    individual_bacc = np.asarray(
        [
            _balanced_accuracy(
                labels_array, probabilities[:, index], classification_threshold
            )
            for index in range(architecture_count)
        ],
        dtype=np.float64,
    )
    individual_sensitivity = np.asarray(
        [
            _sensitivity(labels_array, probabilities[:, index], classification_threshold)
            for index in range(architecture_count)
        ],
        dtype=np.float64,
    )
    individual_specificity = np.asarray(
        [
            _specificity(labels_array, probabilities[:, index], classification_threshold)
            for index in range(architecture_count)
        ],
        dtype=np.float64,
    )
    metric_weights = class_patient_equal_sample_weights(labels_array, patients)
    log_losses = np.asarray(
        [
            _weighted_log_loss(
                probabilities[:, index], labels_array, metric_weights
            )
            for index in range(architecture_count)
        ],
        dtype=np.float64,
    )
    if state_array is None:
        mean_state_variances = np.full(architecture_count, np.nan, dtype=np.float64)
    else:
        mean_state_variances = np.sum(
            metric_weights[:, None] * state_array, axis=0
        )

    fold_scores = np.full(
        (len(unique_folds), architecture_count), np.nan, dtype=np.float64
    )
    for fold_index, fold in enumerate(unique_folds):
        mask = folds == fold
        for architecture_index in range(architecture_count):
            fold_scores[fold_index, architecture_index] = _balanced_accuracy(
                labels_array[mask],
                probabilities[mask, architecture_index],
                classification_threshold,
            )
    if not np.isfinite(fold_scores).all():
        raise ValueError(
            "Every OOF fold must contain both classes for robust anchor selection."
        )
    fold_bacc_std = np.std(fold_scores, axis=0)
    fold_bacc_mean = np.mean(fold_scores, axis=0)
    worst_fold_bacc = np.min(fold_scores, axis=0)

    best_index = min(
        range(architecture_count),
        key=lambda index: (-individual_bacc[index], names[index]),
    )
    best_bacc = float(individual_bacc[best_index])
    accuracy_eligible = tuple(
        index
        for index in range(architecture_count)
        if individual_bacc[index] + bacc_noninferiority_margin
        >= best_bacc - 1.0e-12
    )
    if max_sensitivity_drop is None:
        eligible = accuracy_eligible
    else:
        sensitivity_floor = (
            float(individual_sensitivity[best_index]) - max_sensitivity_drop
        )
        eligible = tuple(
            index
            for index in accuracy_eligible
            if individual_sensitivity[index] >= sensitivity_floor - 1.0e-12
        )

    def resolution_bucket(value: float, resolution: float) -> int:
        return int(np.floor(value / resolution + 1.0e-12))

    def selection_key(index: int) -> tuple:
        variance_bucket = (
            resolution_bucket(mean_state_variances[index], state_variance_tolerance)
            if state_array is not None
            else 0
        )
        if selection_strategy == "bacc_equivalent_maximin_stable":
            return (
                -float(worst_fold_bacc[index]),
                -float(fold_bacc_mean[index]),
                float(fold_bacc_std[index]),
                variance_bucket,
                resolution_bucket(log_losses[index], calibration_tolerance),
                (
                    float(mean_state_variances[index])
                    if state_array is not None
                    else 0.0
                ),
                float(log_losses[index]),
                -float(individual_bacc[index]),
                names[index],
            )
        return (
            resolution_bucket(log_losses[index], calibration_tolerance),
            variance_bucket,
            float(fold_bacc_std[index]),
            -float(worst_fold_bacc[index]),
            float(log_losses[index]),
            (
                float(mean_state_variances[index])
                if state_array is not None
                else 0.0
            ),
            -float(individual_bacc[index]),
            names[index],
        )

    selected_index = min(eligible, key=selection_key)
    return DevelopmentRobustAnchorSelection(
        strategy=selection_strategy,
        candidate_names=names,
        accuracy_eligible_names=tuple(names[index] for index in accuracy_eligible),
        eligible_names=tuple(names[index] for index in eligible),
        selected_name=names[selected_index],
        best_bacc_name=names[best_index],
        best_balanced_accuracy=best_bacc,
        bacc_noninferiority_margin=float(bacc_noninferiority_margin),
        max_sensitivity_drop=(
            None if max_sensitivity_drop is None else float(max_sensitivity_drop)
        ),
        calibration_tolerance=float(calibration_tolerance),
        state_variance_tolerance=float(state_variance_tolerance),
        individual_balanced_accuracies=tuple(float(value) for value in individual_bacc),
        individual_sensitivities=tuple(
            float(value) for value in individual_sensitivity
        ),
        individual_specificities=tuple(
            float(value) for value in individual_specificity
        ),
        class_patient_equal_log_losses=tuple(float(value) for value in log_losses),
        mean_state_variances=tuple(
            None if not np.isfinite(value) else float(value)
            for value in mean_state_variances
        ),
        fold_mean_balanced_accuracies=tuple(
            float(value) for value in fold_bacc_mean
        ),
        fold_bacc_standard_deviations=tuple(
            float(value) for value in fold_bacc_std
        ),
        worst_fold_balanced_accuracies=tuple(
            float(value) for value in worst_fold_bacc
        ),
        state_variance_used=state_array is not None,
        selection_order=(
            (
                "bacc_noninferiority_set",
                "sensitivity_floor",
                "worst_fold_bacc",
                "mean_fold_bacc",
                "fold_bacc_standard_deviation",
                "checkpoint_state_variance",
                "class_patient_equal_log_loss",
            )
            if selection_strategy == "bacc_equivalent_maximin_stable"
            else (
                "bacc_noninferiority_set",
                "sensitivity_floor",
                "class_patient_equal_log_loss",
                "checkpoint_state_variance",
                "fold_bacc_standard_deviation",
                "worst_fold_bacc",
            )
        ),
    )


def select_top1_anchor_with_fallback(
    probabilities: np.ndarray,
    labels: Sequence,
    patient_ids: Sequence,
    fold_ids: Sequence,
    architecture_names: Sequence[str],
    fixed_anchor_name: str | None = None,
    anchor_selection_strategy: str = "max_bacc",
    state_variances: np.ndarray | None = None,
    anchor_bacc_noninferiority_margin: float = 0.002,
    anchor_max_sensitivity_drop: float | None = 0.005,
    anchor_calibration_tolerance: float = 0.001,
    anchor_state_variance_tolerance: float = 1.0e-5,
    diversity_lambda: float = 0.05,
    classification_threshold: float = 0.5,
    anchor_min_weight: float = 0.70,
    weight_grid_step: float = 0.01,
    min_members: int = 2,
    max_members: int = 3,
    min_cv_member_improvement: float = 0.0,
    max_individual_bacc_gap: float | None = 0.05,
    max_individual_sensitivity_gap: float | None = None,
    min_oof_bacc_gain: float = 0.005,
    bacc_noninferiority_margin: float | None = None,
    max_oof_sensitivity_drop: float | None = None,
    max_oof_specificity_drop: float | None = None,
    min_non_decreasing_folds: int = 4,
    min_non_decreasing_sensitivity_folds: int | None = None,
    fold_tolerance: float = 0.0,
    fold_sensitivity_tolerance: float = 0.0,
    anchor_confidence_low: float | None = None,
    anchor_confidence_high: float | None = None,
    min_override_agreement: int = 0,
    secondary_metric_tolerance: float = 0.001,
) -> tuple[Top1AnchorFallbackSelection, np.ndarray, np.ndarray]:
    """Select a small Top1-anchored ensemble and fall back when it is unstable.

    The best individual development OOF architecture is the default immutable
    anchor. ``bacc_equivalent_calibrated_stable`` instead chooses a near-best,
    well-calibrated and trajectory-stable development anchor.
    ``fixed_anchor_name`` may lock a prespecified architecture.
    Every fold-held-out candidate blend reserves at least ``anchor_min_weight``
    for that anchor.  Complementary members share only the remaining risk
    budget.  The proposed blend is deployed only when its pooled OOF BAcc gain
    and its number of non-decreasing folds both satisfy locked guardrails.

    Returns the diagnostics, the probability used for OOF reporting after the
    fallback decision, and the proposed cross-fitted ensemble probability.
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
    allowed_anchor_strategies = {
        "max_bacc",
        "bacc_equivalent_calibrated_stable",
        "bacc_equivalent_maximin_stable",
    }
    if anchor_selection_strategy not in allowed_anchor_strategies:
        raise ValueError(
            "anchor_selection_strategy must be one of "
            f"{sorted(allowed_anchor_strategies)}."
        )
    if fixed_anchor_name is not None and anchor_selection_strategy != "max_bacc":
        raise ValueError(
            "fixed_anchor_name cannot be combined with an automatic anchor "
            "selection strategy."
        )
    if not np.isin(labels_array, [0.0, 1.0]).all():
        raise ValueError("Top1-anchor selection requires binary labels 0/1.")
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must lie strictly between 0 and 1.")
    if not 0.5 <= anchor_min_weight <= 1.0:
        raise ValueError("anchor_min_weight must lie in [0.5, 1.0].")
    if anchor_min_weight < 1.0 and not (
        0.0 < weight_grid_step <= 1.0 - anchor_min_weight
    ):
        raise ValueError(
            "weight_grid_step must be positive and no larger than the complement budget."
        )
    if min_members < 1 or min_members > architecture_count:
        raise ValueError("min_members must be between 1 and the candidate count.")
    if max_members < min_members or max_members > architecture_count:
        raise ValueError("max_members must be between min_members and candidate count.")
    if max_members > 3:
        raise ValueError("Top1-anchor grid search supports at most three members.")
    if min_cv_member_improvement < 0 or min_oof_bacc_gain < 0:
        raise ValueError("BAcc improvement requirements must be non-negative.")
    if max_individual_bacc_gap is not None and max_individual_bacc_gap < 0:
        raise ValueError("max_individual_bacc_gap must be non-negative or null.")
    if max_individual_sensitivity_gap is not None and max_individual_sensitivity_gap < 0:
        raise ValueError(
            "max_individual_sensitivity_gap must be non-negative or null."
        )
    if max_oof_sensitivity_drop is not None and max_oof_sensitivity_drop < 0:
        raise ValueError("max_oof_sensitivity_drop must be non-negative or null.")
    if bacc_noninferiority_margin is not None and bacc_noninferiority_margin < 0:
        raise ValueError("bacc_noninferiority_margin must be non-negative or null.")
    if max_oof_specificity_drop is not None and max_oof_specificity_drop < 0:
        raise ValueError("max_oof_specificity_drop must be non-negative or null.")
    if secondary_metric_tolerance <= 0:
        raise ValueError("secondary_metric_tolerance must be positive.")
    if fold_sensitivity_tolerance < 0:
        raise ValueError("fold_sensitivity_tolerance must be non-negative.")
    if min_override_agreement < 0:
        raise ValueError("min_override_agreement must be non-negative.")
    # Validate the optional confidence guard once, including paired bounds.
    apply_anchor_confidence_guard(
        np.asarray([0.5]),
        np.asarray([0.5]),
        anchor_confidence_low,
        anchor_confidence_high,
    )
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("Top1-anchor selection requires at least two OOF folds.")
    if not 1 <= min_non_decreasing_folds <= len(unique_folds):
        raise ValueError(
            "min_non_decreasing_folds must be between 1 and the OOF fold count."
        )
    if min_non_decreasing_sensitivity_folds is not None and not (
        1 <= min_non_decreasing_sensitivity_folds <= len(unique_folds)
    ):
        raise ValueError(
            "min_non_decreasing_sensitivity_folds must be between 1 and the "
            "OOF fold count, or null."
        )
    patient_fold_counts = np.asarray(
        [len(np.unique(folds[patients == patient])) for patient in np.unique(patients)]
    )
    if (patient_fold_counts > 1).any():
        raise ValueError("A patient cannot occur in more than one OOF fold.")

    individual_scores = np.asarray(
        [
            _balanced_accuracy(
                labels_array, probabilities[:, index], classification_threshold
            )
            for index in range(architecture_count)
        ]
    )
    individual_sensitivities = np.asarray(
        [
            _sensitivity(labels_array, probabilities[:, index], classification_threshold)
            for index in range(architecture_count)
        ]
    )
    robust_anchor_selection = None
    if fixed_anchor_name is not None:
        if fixed_anchor_name not in names:
            raise ValueError(
                f"fixed_anchor_name={fixed_anchor_name!r} is not a candidate architecture."
            )
        anchor = names.index(fixed_anchor_name)
        effective_anchor_strategy = "fixed"
    elif anchor_selection_strategy in {
        "bacc_equivalent_calibrated_stable",
        "bacc_equivalent_maximin_stable",
    }:
        robust_anchor_selection = select_development_robust_anchor(
            probabilities,
            labels=labels_array,
            patient_ids=patients,
            fold_ids=folds,
            architecture_names=names,
            state_variances=state_variances,
            classification_threshold=classification_threshold,
            bacc_noninferiority_margin=anchor_bacc_noninferiority_margin,
            max_sensitivity_drop=anchor_max_sensitivity_drop,
            calibration_tolerance=anchor_calibration_tolerance,
            state_variance_tolerance=anchor_state_variance_tolerance,
            selection_strategy=anchor_selection_strategy,
        )
        anchor = names.index(robust_anchor_selection.selected_name)
        effective_anchor_strategy = anchor_selection_strategy
    else:
        anchor = min(
            range(architecture_count),
            key=lambda index: (-individual_scores[index], names[index]),
        )
        effective_anchor_strategy = "max_bacc"
    anchor_probability = probabilities[:, anchor].copy()
    anchor_score = float(individual_scores[anchor])
    anchor_sensitivity = float(individual_sensitivities[anchor])
    anchor_specificity = _specificity(
        labels_array, anchor_probability, classification_threshold
    )

    def probability_metrics(
        metric_labels: np.ndarray,
        metric_probabilities: np.ndarray,
        metric_patients: np.ndarray,
    ) -> dict[str, float]:
        patient_weights = class_patient_equal_sample_weights(
            metric_labels, metric_patients
        )
        return {
            "balanced_accuracy": _balanced_accuracy(
                metric_labels, metric_probabilities, classification_threshold
            ),
            "sensitivity": _sensitivity(
                metric_labels, metric_probabilities, classification_threshold
            ),
            "specificity": _specificity(
                metric_labels, metric_probabilities, classification_threshold
            ),
            "macro_f1": binary_macro_f1(
                metric_labels, metric_probabilities, classification_threshold
            ),
            "roc_auc": _roc_auc(metric_labels, metric_probabilities),
            "average_precision": _average_precision(
                metric_labels, metric_probabilities
            ),
            "patient_equal_log_loss": _weighted_log_loss(
                metric_probabilities, metric_labels, patient_weights
            ),
        }

    anchor_metrics = probability_metrics(
        labels_array, anchor_probability, patients
    )

    def is_noninferior(
        metrics: dict[str, float], reference: dict[str, float] | None = None
    ) -> bool:
        reference = anchor_metrics if reference is None else reference
        if bacc_noninferiority_margin is not None and (
            metrics["balanced_accuracy"] + bacc_noninferiority_margin
            < reference["balanced_accuracy"] - 1e-12
        ):
            return False
        if max_oof_sensitivity_drop is not None and (
            metrics["sensitivity"] + max_oof_sensitivity_drop
            < reference["sensitivity"] - 1e-12
        ):
            return False
        if max_oof_specificity_drop is not None and (
            metrics["specificity"] + max_oof_specificity_drop
            < reference["specificity"] - 1e-12
        ):
            return False
        return True

    def secondary_key(metrics: dict[str, float], anchor_weight: float) -> tuple:
        tolerance = secondary_metric_tolerance

        def bucket(value: float) -> int:
            return int(np.floor(value / tolerance + 1e-12))

        return (
            -bucket(metrics["macro_f1"]),
            -bucket(metrics["roc_auc"]),
            -bucket(metrics["average_precision"]),
            metrics["patient_equal_log_loss"],
            -anchor_weight,
        )

    eligible = [index for index in range(architecture_count) if index != anchor]
    if max_individual_bacc_gap is not None:
        eligible = [
            index
            for index in eligible
            if individual_scores[index] >= anchor_score - max_individual_bacc_gap
        ]
    if max_individual_sensitivity_gap is not None:
        eligible = [
            index
            for index in eligible
            if individual_sensitivities[index]
            >= anchor_sensitivity - max_individual_sensitivity_gap
        ]

    def anchored_weights(
        indices: tuple[int, ...], training_mask: np.ndarray
    ) -> np.ndarray:
        weights = np.zeros(architecture_count, dtype=np.float64)
        if len(indices) == 1 or anchor_min_weight == 1.0:
            weights[anchor] = 1.0
            return weights
        # BAcc is discontinuous, so use a small deterministic grid over the
        # at-most-30% complement budget. Log loss plus residual similarity is
        # only a tie-breaker; it cannot trade away the primary endpoint.
        complement_indices = tuple(index for index in indices if index != anchor)
        residual_budget = 1.0 - anchor_min_weight
        budget_units = max(1, int(round(residual_budget / weight_grid_step)))
        actual_step = residual_budget / budget_units
        train_probabilities = probabilities[training_mask]
        train_labels = labels_array[training_mask]
        train_patients = patients[training_mask]
        sample_weights = class_patient_equal_sample_weights(
            train_labels, train_patients
        )
        train_anchor_metrics = probability_metrics(
            train_labels, train_probabilities[:, anchor], train_patients
        )
        subset_similarity = residual_similarity_matrix(
            train_probabilities[:, indices], train_labels, sample_weights
        )
        best_key = None
        best_weights = None
        for allocation in np.ndindex(
            *((budget_units + 1,) * len(complement_indices))
        ):
            if sum(allocation) > budget_units:
                continue
            candidate = np.zeros(architecture_count, dtype=np.float64)
            for index, units in zip(complement_indices, allocation):
                candidate[index] = units * actual_step
            candidate[anchor] = 1.0 - float(candidate.sum())
            candidate_probability = train_probabilities @ candidate
            active_complements = [
                index
                for index in complement_indices
                if candidate[index] > 1e-12
            ]
            candidate_probability = apply_anchor_decision_guard(
                train_probabilities[:, anchor],
                candidate_probability,
                train_probabilities[:, active_complements],
                classification_threshold,
                anchor_confidence_low,
                anchor_confidence_high,
                min_override_agreement,
            )
            bacc = _balanced_accuracy(
                train_labels, candidate_probability, classification_threshold
            )
            subset_weights = candidate[np.asarray(indices, dtype=int)]
            tie_objective = _weighted_log_loss(
                candidate_probability, train_labels, sample_weights
            ) + diversity_lambda * float(
                subset_weights @ subset_similarity @ subset_weights
            )
            if bacc_noninferiority_margin is not None:
                metrics = probability_metrics(
                    train_labels, candidate_probability, train_patients
                )
                if not is_noninferior(metrics, train_anchor_metrics):
                    continue
                # BAcc, sensitivity and specificity define feasibility. Among
                # feasible candidates, optimize secondary endpoints in the
                # prespecified order and prefer a simpler anchor-heavy blend.
                key = (
                    *secondary_key(metrics, float(candidate[anchor])),
                    float(tie_objective),
                    allocation,
                )
            else:
                # Legacy mode: prefer higher BAcc, then the smooth objective.
                key = (
                    -float(bacc),
                    float(tie_objective),
                    -candidate[anchor],
                    allocation,
                )
            if best_key is None or key < best_key:
                best_key = key
                best_weights = candidate
        if best_weights is None:
            raise AssertionError("Top1-anchor weight grid produced no candidates.")
        return best_weights

    score_cache: dict[tuple[int, ...], tuple[dict[str, float], np.ndarray]] = {}

    def cross_fitted_score(
        indices: Sequence[int],
    ) -> tuple[dict[str, float], np.ndarray]:
        ordered = (anchor, *sorted(index for index in indices if index != anchor))
        if ordered in score_cache:
            return score_cache[ordered]
        cross_fitted = np.empty(sample_count, dtype=np.float64)
        for held_out_fold in unique_folds:
            held_out_mask = folds == held_out_fold
            training_mask = ~held_out_mask
            weights = anchored_weights(ordered, training_mask)
            held_out_probability = probabilities[held_out_mask] @ weights
            active_complements = [
                index
                for index in ordered
                if index != anchor and weights[index] > 1e-12
            ]
            cross_fitted[held_out_mask] = apply_anchor_decision_guard(
                anchor_probability[held_out_mask],
                held_out_probability,
                probabilities[held_out_mask][:, active_complements],
                classification_threshold,
                anchor_confidence_low,
                anchor_confidence_high,
                min_override_agreement,
            )
        metrics = probability_metrics(
            labels_array, cross_fitted, patients
        )
        score_cache[ordered] = (metrics, cross_fitted)
        return score_cache[ordered]

    selected = [anchor]
    current_score = anchor_score
    steps: list[dict] = [
        {
            "step": 1,
            "added": names[anchor],
            "cross_validated_balanced_accuracy": anchor_score,
            "improvement": None,
            "role": "anchor",
        }
    ]
    if bacc_noninferiority_margin is not None:
        feasible_subsets: list[tuple[tuple, tuple[int, ...], dict[str, float], np.ndarray]] = []
        for member_count in range(min_members, max_members + 1):
            for complement_subset in combinations(eligible, member_count - 1):
                indices = (anchor, *complement_subset)
                metrics, cross_fitted = cross_fitted_score(indices)
                if not is_noninferior(metrics):
                    continue
                subset_bacc_folds = 0
                subset_sensitivity_folds = 0
                for fold in unique_folds:
                    fold_mask = folds == fold
                    fold_anchor_bacc = _balanced_accuracy(
                        labels_array[fold_mask],
                        anchor_probability[fold_mask],
                        classification_threshold,
                    )
                    fold_candidate_bacc = _balanced_accuracy(
                        labels_array[fold_mask],
                        cross_fitted[fold_mask],
                        classification_threshold,
                    )
                    subset_bacc_folds += int(
                        np.isfinite(fold_anchor_bacc)
                        and np.isfinite(fold_candidate_bacc)
                        and fold_candidate_bacc + fold_tolerance
                        >= fold_anchor_bacc
                    )
                    fold_anchor_sensitivity = _sensitivity(
                        labels_array[fold_mask],
                        anchor_probability[fold_mask],
                        classification_threshold,
                    )
                    fold_candidate_sensitivity = _sensitivity(
                        labels_array[fold_mask],
                        cross_fitted[fold_mask],
                        classification_threshold,
                    )
                    subset_sensitivity_folds += int(
                        np.isfinite(fold_anchor_sensitivity)
                        and np.isfinite(fold_candidate_sensitivity)
                        and fold_candidate_sensitivity
                        + fold_sensitivity_tolerance
                        >= fold_anchor_sensitivity
                    )
                if subset_bacc_folds < min_non_decreasing_folds:
                    continue
                if (
                    min_non_decreasing_sensitivity_folds is not None
                    and subset_sensitivity_folds
                    < min_non_decreasing_sensitivity_folds
                ):
                    continue
                key = (
                    *secondary_key(metrics, anchor_min_weight),
                    member_count,
                    tuple(names[index] for index in indices),
                )
                feasible_subsets.append((key, indices, metrics, cross_fitted))
        if feasible_subsets:
            _, selected_tuple, selected_metrics, proposed_oof_probability = min(
                feasible_subsets, key=lambda item: item[0]
            )
            selected = list(selected_tuple)
            current_score = float(selected_metrics["balanced_accuracy"])
            steps.extend(
                {
                    "step": step,
                    "added": names[index],
                    "cross_validated_balanced_accuracy": current_score,
                    "cross_validated_macro_f1": float(
                        selected_metrics["macro_f1"]
                    ),
                    "cross_validated_roc_auc": float(
                        selected_metrics["roc_auc"]
                    ),
                    "role": "complement",
                }
                for step, index in enumerate(selected_tuple[1:], start=2)
            )
        else:
            selected_tuple = (anchor,)
            selected_metrics = anchor_metrics
            proposed_oof_probability = anchor_probability.copy()
    else:
        while len(selected) < max_members:
            remaining = [index for index in eligible if index not in selected]
            if not remaining:
                break
            scored = []
            for index in remaining:
                metrics, _ = cross_fitted_score([*selected, index])
                scored.append(
                    (metrics["balanced_accuracy"], names[index], index)
                )
            candidate_score, _, candidate = min(
                scored, key=lambda item: (-item[0], item[1])
            )
            improvement = candidate_score - current_score
            if (
                len(selected) >= min_members
                and improvement < min_cv_member_improvement
            ):
                break
            selected.append(candidate)
            current_score = candidate_score
            steps.append(
                {
                    "step": len(selected),
                    "added": names[candidate],
                    "cross_validated_balanced_accuracy": float(current_score),
                    "improvement": float(improvement),
                    "role": "complement",
                }
            )
        selected_tuple = (
            anchor,
            *sorted(index for index in selected if index != anchor),
        )
        selected_metrics, proposed_oof_probability = cross_fitted_score(
            selected_tuple
        )

    ensemble_score = float(selected_metrics["balanced_accuracy"])
    fold_diagnostics: list[dict] = []
    non_decreasing_folds = 0
    non_decreasing_sensitivity_folds = 0
    for fold in unique_folds:
        mask = folds == fold
        fold_anchor = _balanced_accuracy(
            labels_array[mask], anchor_probability[mask], classification_threshold
        )
        fold_ensemble = _balanced_accuracy(
            labels_array[mask], proposed_oof_probability[mask], classification_threshold
        )
        non_decreasing = bool(
            np.isfinite(fold_anchor)
            and np.isfinite(fold_ensemble)
            and fold_ensemble + fold_tolerance >= fold_anchor
        )
        non_decreasing_folds += int(non_decreasing)
        fold_anchor_sensitivity = _sensitivity(
            labels_array[mask], anchor_probability[mask], classification_threshold
        )
        fold_ensemble_sensitivity = _sensitivity(
            labels_array[mask], proposed_oof_probability[mask], classification_threshold
        )
        sensitivity_non_decreasing = bool(
            np.isfinite(fold_anchor_sensitivity)
            and np.isfinite(fold_ensemble_sensitivity)
            and fold_ensemble_sensitivity + fold_sensitivity_tolerance
            >= fold_anchor_sensitivity
        )
        non_decreasing_sensitivity_folds += int(sensitivity_non_decreasing)
        fold_diagnostics.append(
            {
                "fold": int(fold) if np.issubdtype(type(fold), np.integer) else str(fold),
                "anchor_balanced_accuracy": float(fold_anchor),
                "ensemble_balanced_accuracy": float(fold_ensemble),
                "gain": float(fold_ensemble - fold_anchor),
                "non_decreasing": non_decreasing,
                "anchor_sensitivity": float(fold_anchor_sensitivity),
                "ensemble_sensitivity": float(fold_ensemble_sensitivity),
                "sensitivity_gain": float(
                    fold_ensemble_sensitivity - fold_anchor_sensitivity
                ),
                "sensitivity_non_decreasing": sensitivity_non_decreasing,
            }
        )

    gain = float(ensemble_score - anchor_score)
    ensemble_sensitivity = _sensitivity(
        labels_array, proposed_oof_probability, classification_threshold
    )
    sensitivity_gain = float(ensemble_sensitivity - anchor_sensitivity)
    ensemble_specificity = _specificity(
        labels_array, proposed_oof_probability, classification_threshold
    )
    specificity_gain = float(ensemble_specificity - anchor_specificity)
    fallback_reasons = []
    if (
        bacc_noninferiority_margin is not None
        and gain + bacc_noninferiority_margin < -1e-12
    ):
        fallback_reasons.append(
            f"OOF BAcc gain {gain:.6f} violates non-inferiority margin "
            f"{-bacc_noninferiority_margin:.6f}"
        )
    elif bacc_noninferiority_margin is None and gain + 1e-12 < min_oof_bacc_gain:
        fallback_reasons.append(
            f"OOF BAcc gain {gain:.6f} < required {min_oof_bacc_gain:.6f}"
        )
    if non_decreasing_folds < min_non_decreasing_folds:
        fallback_reasons.append(
            f"non-decreasing folds {non_decreasing_folds} < required "
            f"{min_non_decreasing_folds}"
        )
    if (
        max_oof_sensitivity_drop is not None
        and sensitivity_gain + max_oof_sensitivity_drop < -1e-12
    ):
        fallback_reasons.append(
            f"OOF sensitivity drop {-sensitivity_gain:.6f} > allowed "
            f"{max_oof_sensitivity_drop:.6f}"
        )
    if (
        max_oof_specificity_drop is not None
        and specificity_gain + max_oof_specificity_drop < -1e-12
    ):
        fallback_reasons.append(
            f"OOF specificity drop {-specificity_gain:.6f} > allowed "
            f"{max_oof_specificity_drop:.6f}"
        )
    if (
        min_non_decreasing_sensitivity_folds is not None
        and non_decreasing_sensitivity_folds
        < min_non_decreasing_sensitivity_folds
    ):
        fallback_reasons.append(
            "non-decreasing sensitivity folds "
            f"{non_decreasing_sensitivity_folds} < required "
            f"{min_non_decreasing_sensitivity_folds}"
        )
    if len(selected_tuple) < min_members:
        fallback_reasons.append(
            f"eligible members {len(selected_tuple)} < required {min_members}"
        )
    fallback_triggered = bool(fallback_reasons)

    full_mask = np.ones(sample_count, dtype=bool)
    proposed_weights = anchored_weights(selected_tuple, full_mask)
    deployed_weights = proposed_weights.copy()
    if fallback_triggered:
        deployed_weights.fill(0.0)
        deployed_weights[anchor] = 1.0
        deployed_names = (names[anchor],)
        deployed_oof_probability = anchor_probability
    else:
        deployed_names = tuple(names[index] for index in selected_tuple)
        deployed_oof_probability = proposed_oof_probability

    result = Top1AnchorFallbackSelection(
        candidate_names=names,
        anchor_name=names[anchor],
        fixed_anchor=fixed_anchor_name is not None,
        anchor_selection_strategy=effective_anchor_strategy,
        anchor_selection_diagnostics=(
            None
            if robust_anchor_selection is None
            else robust_anchor_selection.as_dict()
        ),
        proposed_selected_names=tuple(names[index] for index in selected_tuple),
        deployed_names=deployed_names,
        individual_balanced_accuracies=tuple(float(value) for value in individual_scores),
        individual_sensitivities=tuple(
            float(value) for value in individual_sensitivities
        ),
        proposed_weights=tuple(float(value) for value in proposed_weights),
        deployed_weights=tuple(float(value) for value in deployed_weights),
        anchor_min_weight=float(anchor_min_weight),
        anchor_confidence_low=anchor_confidence_low,
        anchor_confidence_high=anchor_confidence_high,
        min_override_agreement=int(min_override_agreement),
        anchor_balanced_accuracy=anchor_score,
        ensemble_balanced_accuracy=float(ensemble_score),
        balanced_accuracy_gain=gain,
        min_oof_bacc_gain=(
            None
            if bacc_noninferiority_margin is not None
            else float(min_oof_bacc_gain)
        ),
        bacc_noninferiority_margin=bacc_noninferiority_margin,
        anchor_sensitivity=anchor_sensitivity,
        ensemble_sensitivity=float(ensemble_sensitivity),
        sensitivity_gain=sensitivity_gain,
        max_oof_sensitivity_drop=max_oof_sensitivity_drop,
        anchor_specificity=float(anchor_specificity),
        ensemble_specificity=float(ensemble_specificity),
        specificity_gain=specificity_gain,
        max_oof_specificity_drop=max_oof_specificity_drop,
        anchor_macro_f1=float(anchor_metrics["macro_f1"]),
        ensemble_macro_f1=float(selected_metrics["macro_f1"]),
        anchor_roc_auc=float(anchor_metrics["roc_auc"]),
        ensemble_roc_auc=float(selected_metrics["roc_auc"]),
        anchor_average_precision=float(anchor_metrics["average_precision"]),
        ensemble_average_precision=float(selected_metrics["average_precision"]),
        selection_objective=(
            "bacc_noninferiority_then_macro_f1_roc_auc_average_precision_log_loss"
            if bacc_noninferiority_margin is not None
            else "balanced_accuracy"
        ),
        non_decreasing_folds=int(non_decreasing_folds),
        min_non_decreasing_folds=int(min_non_decreasing_folds),
        non_decreasing_sensitivity_folds=int(non_decreasing_sensitivity_folds),
        min_non_decreasing_sensitivity_folds=min_non_decreasing_sensitivity_folds,
        fallback_triggered=fallback_triggered,
        fallback_reasons=tuple(fallback_reasons),
        fold_diagnostics=tuple(fold_diagnostics),
        steps=tuple(steps),
    )
    return result, deployed_oof_probability, proposed_oof_probability


def select_diversity_veto(
    probabilities: np.ndarray,
    labels: Sequence,
    patient_ids: Sequence,
    architecture_names: Sequence[str],
    classification_threshold: float = 0.5,
    max_veto_bacc_gap: float = 0.03,
) -> DiversityVetoSelection:
    """Choose an accurate anchor and a complementary negative-veto member.

    The anchor is the architecture with the highest development OOF Balanced
    Accuracy.  Among architectures whose OOF BAcc is close to the anchor, the
    veto member is the one with the least-correlated residual errors.  At
    inference the ensemble probability is ``min(anchor_prob, veto_prob)``.
    Consequently, a positive decision requires agreement while either member
    may veto a likely false positive.  Independent-test data are not used.
    """
    probabilities = _validate_probabilities(probabilities)
    labels_array = _as_1d(labels, "labels")
    patients = np.asarray(patient_ids, dtype=str)
    names = tuple(str(name) for name in architecture_names)
    sample_count, architecture_count = probabilities.shape

    if architecture_count < 2:
        raise ValueError("Diversity veto requires at least two architectures.")
    if len(labels_array) != sample_count or len(patients) != sample_count:
        raise ValueError("Labels/patient IDs do not match probability rows.")
    if len(names) != architecture_count or len(set(names)) != architecture_count:
        raise ValueError("architecture_names must uniquely match probability columns.")
    if not np.isin(labels_array, [0.0, 1.0]).all():
        raise ValueError("Diversity veto requires binary labels 0/1.")
    if not 0.0 < classification_threshold < 1.0:
        raise ValueError("classification_threshold must lie strictly between 0 and 1.")
    if max_veto_bacc_gap < 0:
        raise ValueError("max_veto_bacc_gap must be non-negative.")

    individual_scores = np.asarray(
        [
            _balanced_accuracy(
                labels_array, probabilities[:, index], classification_threshold
            )
            for index in range(architecture_count)
        ]
    )
    anchor = min(
        range(architecture_count),
        key=lambda index: (-individual_scores[index], names[index]),
    )
    minimum_score = float(individual_scores[anchor] - max_veto_bacc_gap)
    eligible = [
        index
        for index in range(architecture_count)
        if index != anchor and individual_scores[index] >= minimum_score
    ]
    if not eligible:
        raise ValueError(
            "No veto member passed the OOF BAcc gate; increase "
            "aggregation.max_veto_bacc_gap."
        )

    sample_weights = class_patient_equal_sample_weights(labels_array, patients)
    similarity = residual_similarity_matrix(
        probabilities, labels_array, sample_weights
    )
    veto = min(
        eligible,
        key=lambda index: (similarity[anchor, index], names[index]),
    )
    return DiversityVetoSelection(
        candidate_names=names,
        eligible_veto_names=tuple(names[index] for index in eligible),
        anchor_name=names[anchor],
        veto_name=names[veto],
        individual_balanced_accuracies=tuple(
            float(value) for value in individual_scores
        ),
        anchor_veto_residual_similarity=float(similarity[anchor, veto]),
        max_veto_bacc_gap=float(max_veto_bacc_gap),
        classification_threshold=float(classification_threshold),
    )


def _sensitivity_specificity(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[float, float]:
    predictions = np.asarray(predictions, dtype=bool)
    positive = labels == 1.0
    negative = labels == 0.0
    if not positive.any() or not negative.any():
        raise ValueError("Sensitivity/specificity require both binary classes.")
    return (
        float(predictions[positive].mean()),
        float((~predictions[negative]).mean()),
    )


def _highest_threshold_at_sensitivity(
    positive_probabilities: np.ndarray, sensitivity_floor: float
) -> float:
    """Highest observed-data threshold retaining the requested sensitivity."""
    ordered = np.sort(np.asarray(positive_probabilities, dtype=np.float64))
    allowed_false_negatives = int(
        np.floor((1.0 - sensitivity_floor) * len(ordered) + 1e-12)
    )
    allowed_false_negatives = min(max(allowed_false_negatives, 0), len(ordered) - 1)
    return float(np.nextafter(ordered[allowed_false_negatives], -np.inf))


def select_sensitivity_constrained_subset(
    probabilities: np.ndarray,
    labels: Sequence,
    architecture_names: Sequence[str],
    reference_probabilities: Sequence,
    reference_threshold: float = 0.5,
    sensitivity_margin: float = 0.0,
    min_members: int = 1,
    max_members: int | None = 8,
) -> SensitivityConstrainedSelection:
    """Maximize OOF specificity/BAcc without lowering reference sensitivity.

    Every candidate is an equal-probability architecture subset. Its threshold
    is the largest positive-class OOF order statistic that satisfies the
    sensitivity floor. This makes sensitivity a constraint rather than a
    post-hoc trade-off and leaves independent-test labels entirely untouched.
    """
    probabilities = _validate_probabilities(probabilities)
    labels_array = _as_1d(labels, "labels")
    reference = _as_1d(reference_probabilities, "reference_probabilities")
    names = tuple(str(name) for name in architecture_names)
    sample_count, architecture_count = probabilities.shape

    if len(labels_array) != sample_count or len(reference) != sample_count:
        raise ValueError("Labels/reference probabilities do not match probability rows.")
    if len(names) != architecture_count or len(set(names)) != architecture_count:
        raise ValueError("architecture_names must uniquely match probability columns.")
    if not np.isin(labels_array, [0.0, 1.0]).all():
        raise ValueError("Sensitivity-constrained selection requires labels 0/1.")
    if ((reference < 0.0) | (reference > 1.0)).any():
        raise ValueError("reference_probabilities must lie in [0, 1].")
    if not 0.0 < reference_threshold < 1.0:
        raise ValueError("reference_threshold must lie strictly between 0 and 1.")
    if sensitivity_margin < 0.0 or sensitivity_margin >= 1.0:
        raise ValueError("sensitivity_margin must lie in [0, 1).")
    if min_members < 1 or min_members > architecture_count:
        raise ValueError("min_members must be between 1 and the candidate count.")
    if max_members is None:
        max_members = architecture_count
    max_members = min(int(max_members), architecture_count)
    if max_members < min_members:
        raise ValueError("max_members must be at least min_members.")

    reference_sensitivity, _ = _sensitivity_specificity(
        labels_array, reference >= reference_threshold
    )
    sensitivity_floor = max(0.0, reference_sensitivity - sensitivity_margin)
    positive = labels_array == 1.0
    best: tuple[tuple[float, float, float, int], tuple[int, ...], float] | None = None
    evaluated = 0
    for member_count in range(min_members, max_members + 1):
        for indices in combinations(range(architecture_count), member_count):
            evaluated += 1
            ensemble_probability = probabilities[:, indices].mean(axis=1)
            candidate_threshold = _highest_threshold_at_sensitivity(
                ensemble_probability[positive], sensitivity_floor
            )
            predictions = ensemble_probability >= candidate_threshold
            sensitivity, specificity = _sensitivity_specificity(
                labels_array, predictions
            )
            if sensitivity + 1e-12 < sensitivity_floor:
                continue
            balanced_accuracy = (sensitivity + specificity) / 2.0
            # Specificity is the improvement target after sensitivity is locked.
            # Then prefer BAcc, sensitivity, and a smaller subset in that order.
            objective = (
                specificity,
                balanced_accuracy,
                sensitivity,
                -member_count,
            )
            if best is None or objective > best[0]:
                best = (objective, indices, candidate_threshold)

    if best is None:
        raise ValueError("No architecture subset satisfied the sensitivity floor.")
    _, selected_indices, selected_threshold = best
    selected_probability = probabilities[:, selected_indices].mean(axis=1)
    sensitivity, specificity = _sensitivity_specificity(
        labels_array, selected_probability >= selected_threshold
    )
    return SensitivityConstrainedSelection(
        candidate_names=names,
        selected_names=tuple(names[index] for index in selected_indices),
        reference_sensitivity=float(reference_sensitivity),
        sensitivity_floor=float(sensitivity_floor),
        cross_validated_balanced_accuracy=float((sensitivity + specificity) / 2.0),
        cross_validated_sensitivity=float(sensitivity),
        cross_validated_specificity=float(specificity),
        classification_threshold=float(selected_threshold),
        min_members=int(min_members),
        max_members=int(max_members),
        evaluated_subsets=int(evaluated),
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
