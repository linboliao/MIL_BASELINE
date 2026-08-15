"""Class-Pareto Adaptive Weight Averaging (CP-AWA).

CP-AWA is a training-free checkpoint-soup builder. It gates states using only
fold-validation metrics, assigns a trajectory/class-stability prior, and then
greedily accepts parameter-space averages under class-wise validation guards.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CPMetric:
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    macro_f1: float
    log_loss: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CPCandidate:
    epoch: int
    checkpoint: str
    balanced_accuracy: float
    macro_f1: float
    sensitivity: float
    specificity: float
    log_prior: float
    prior_weight: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CPBuildResult:
    state: Any
    accepted_candidates: tuple[CPCandidate, ...]
    normalized_weights: tuple[float, ...]
    reference_metrics: CPMetric
    final_metrics: CPMetric
    trace: tuple[dict[str, Any], ...]


def _finite_metric(record: Mapping[str, Any], name: str) -> float | None:
    value = record.get("val_metrics", {}).get(name)
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return None


def class_recalls(record: Mapping[str, Any]) -> tuple[float, float] | None:
    confusion = record.get("val_metrics", {}).get("confusion_mat")
    try:
        tn, fp = float(confusion[0][0]), float(confusion[0][1])
        fn, tp = float(confusion[1][0]), float(confusion[1][1])
    except (IndexError, TypeError, ValueError):
        return None
    if tn + fp <= 0 or tp + fn <= 0:
        return None
    return tp / (tp + fn), tn / (tn + fp)


def pareto_candidates(
    manifest: Mapping[str, Any],
    *,
    metric_tolerance: float = 0.005,
    macro_f1_tolerance: float = 0.005,
    sensitivity_drop_tolerance: float = 0.010,
    specificity_drop_tolerance: float = 0.010,
    temperature: float = 0.0025,
    class_temperature: float = 0.020,
    max_candidates_to_evaluate: int = 10,
) -> tuple[list[CPCandidate], CPCandidate]:
    """Return performance/class-gated candidates and the best-BAcc reference."""
    tolerances = {
        "metric_tolerance": metric_tolerance,
        "macro_f1_tolerance": macro_f1_tolerance,
        "sensitivity_drop_tolerance": sensitivity_drop_tolerance,
        "specificity_drop_tolerance": specificity_drop_tolerance,
    }
    if any(value < 0 for value in tolerances.values()):
        raise ValueError("CP-AWA tolerances must be non-negative.")
    if temperature <= 0 or class_temperature <= 0:
        raise ValueError("CP-AWA temperatures must be positive.")
    if max_candidates_to_evaluate < 1:
        raise ValueError("max_candidates_to_evaluate must be positive.")

    parsed: list[dict[str, Any]] = []
    for record in manifest.get("epochs", []):
        checkpoint = record.get("checkpoint")
        bacc = _finite_metric(record, "bacc")
        macro_f1 = _finite_metric(record, "macro_f1")
        recalls = class_recalls(record)
        if not checkpoint or bacc is None or macro_f1 is None or recalls is None:
            continue
        sensitivity, specificity = recalls
        parsed.append(
            {
                "epoch": int(record["epoch"]),
                "checkpoint": str(checkpoint),
                "balanced_accuracy": bacc,
                "macro_f1": macro_f1,
                "sensitivity": sensitivity,
                "specificity": specificity,
            }
        )
    if not parsed:
        raise ValueError(
            "No checkpoint records contain bacc, macro_f1, and a binary confusion matrix."
        )

    reference_record = min(
        parsed,
        key=lambda item: (
            -item["balanced_accuracy"],
            -item["macro_f1"],
            item["epoch"],
        ),
    )
    best_bacc = max(item["balanced_accuracy"] for item in parsed)
    best_macro_f1 = max(item["macro_f1"] for item in parsed)
    gated = [
        item
        for item in parsed
        if item["balanced_accuracy"] >= best_bacc - metric_tolerance - 1e-12
        and item["macro_f1"] >= best_macro_f1 - macro_f1_tolerance - 1e-12
        and item["sensitivity"]
        >= reference_record["sensitivity"] - sensitivity_drop_tolerance - 1e-12
        and item["specificity"]
        >= reference_record["specificity"] - specificity_drop_tolerance - 1e-12
    ]
    if reference_record not in gated:
        # The independently best macro-F1 can be incompatible with the best
        # BAcc state. The reference always remains a safe one-state fallback.
        gated.append(reference_record)

    for item in gated:
        class_drift = abs(
            item["sensitivity"] - reference_record["sensitivity"]
        ) + abs(item["specificity"] - reference_record["specificity"])
        item["log_prior"] = (
            (item["balanced_accuracy"] - best_bacc) / temperature
            - class_drift / class_temperature
        )
    max_log_prior = max(item["log_prior"] for item in gated)
    for item in gated:
        item["prior_weight"] = float(np.exp(item["log_prior"] - max_log_prior))

    gated.sort(
        key=lambda item: (
            item["checkpoint"] != reference_record["checkpoint"],
            -item["prior_weight"],
            -item["balanced_accuracy"],
            item["epoch"],
        )
    )
    gated = gated[:max_candidates_to_evaluate]
    candidates = [CPCandidate(**item) for item in gated]
    reference = next(
        candidate
        for candidate in candidates
        if candidate.checkpoint == reference_record["checkpoint"]
    )
    return candidates, reference


def normalized_weights(candidates: Sequence[CPCandidate]) -> tuple[float, ...]:
    values = np.asarray([candidate.prior_weight for candidate in candidates], dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("CP-AWA candidate weights must be finite and positive.")
    values /= values.sum()
    return tuple(float(value) for value in values)


def weighted_state_average(states: Sequence[Any], weights: Sequence[float]) -> Any:
    """Recursively average regular or nested (e.g. DTFD) state dictionaries."""
    if len(states) == 0 or len(states) != len(weights):
        raise ValueError("states and weights must have the same non-zero length.")
    numeric_weights = np.asarray(weights, dtype=float)
    if not np.isfinite(numeric_weights).all() or (numeric_weights < 0).any():
        raise ValueError("weights must be finite and non-negative.")
    if numeric_weights.sum() <= 0:
        raise ValueError("weights must have a positive sum.")
    numeric_weights /= numeric_weights.sum()

    def average(values: Sequence[Any], path: str) -> Any:
        first = values[0]
        if torch.is_tensor(first):
            if any(not torch.is_tensor(value) for value in values):
                raise TypeError(f"Mixed tensor/non-tensor values at {path}.")
            if any(value.shape != first.shape for value in values):
                raise ValueError(f"Tensor shapes differ at {path}.")
            if first.is_floating_point() or first.is_complex():
                accumulation_dtype = (
                    torch.float64 if first.dtype == torch.float64 else torch.float32
                )
                result = torch.zeros_like(first, dtype=accumulation_dtype, device="cpu")
                for weight, value in zip(numeric_weights, values):
                    result.add_(value.detach().to(device="cpu", dtype=accumulation_dtype), alpha=float(weight))
                return result.to(dtype=first.dtype)
            return first.detach().cpu().clone()
        if isinstance(first, Mapping):
            keys = list(first.keys())
            if any(list(value.keys()) != keys for value in values):
                raise ValueError(f"State-dict keys differ at {path}.")
            return type(first)(
                (key, average([value[key] for value in values], f"{path}.{key}"))
                for key in keys
            )
        if isinstance(first, tuple):
            if any(len(value) != len(first) for value in values):
                raise ValueError(f"Tuple lengths differ at {path}.")
            return type(first)(
                average([value[index] for value in values], f"{path}[{index}]")
                for index in range(len(first))
            )
        if isinstance(first, list):
            if any(len(value) != len(first) for value in values):
                raise ValueError(f"List lengths differ at {path}.")
            return [
                average([value[index] for value in values], f"{path}[{index}]")
                for index in range(len(first))
            ]
        return first

    return average(states, "state")


def cp_metric_from_probabilities(labels: Sequence[int], probabilities: Sequence[float]) -> CPMetric:
    labels_array = np.asarray(labels, dtype=np.int8)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if labels_array.ndim != 1 or probability_array.shape != labels_array.shape:
        raise ValueError("labels and probabilities must be aligned one-dimensional arrays.")
    if not np.isin(labels_array, [0, 1]).all() or np.unique(labels_array).size != 2:
        raise ValueError("CP-AWA validation requires both binary classes.")
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities contain non-finite values.")
    probability_array = np.clip(probability_array, 1e-7, 1.0 - 1e-7)
    predictions = probability_array >= 0.5
    positive = labels_array == 1
    negative = ~positive
    sensitivity = float(np.mean(predictions[positive]))
    specificity = float(np.mean(~predictions[negative]))
    class_f1: list[float] = []
    for class_id in (0, 1):
        predicted_class = predictions == bool(class_id)
        true_class = labels_array == class_id
        tp = int(np.sum(predicted_class & true_class))
        fp = int(np.sum(predicted_class & ~true_class))
        fn = int(np.sum(~predicted_class & true_class))
        denominator = 2 * tp + fp + fn
        class_f1.append(0.0 if denominator == 0 else 2 * tp / denominator)
    log_loss = -float(
        np.mean(
            labels_array * np.log(probability_array)
            + (1 - labels_array) * np.log(1.0 - probability_array)
        )
    )
    return CPMetric(
        balanced_accuracy=(sensitivity + specificity) / 2.0,
        sensitivity=sensitivity,
        specificity=specificity,
        macro_f1=float(np.mean(class_f1)),
        log_loss=log_loss,
    )


def greedy_cp_awa(
    candidates: Sequence[CPCandidate],
    reference: CPCandidate,
    *,
    load_state: Callable[[CPCandidate], Any],
    evaluate_state: Callable[[Any, tuple[CPCandidate, ...]], CPMetric],
    max_checkpoints: int = 5,
    bacc_drop_tolerance: float = 0.001,
    sensitivity_drop_tolerance: float = 0.005,
    specificity_drop_tolerance: float = 0.005,
    log_loss_increase_tolerance: float = 0.005,
) -> CPBuildResult:
    """Greedily construct a class-protected parameter soup."""
    if reference not in candidates:
        raise ValueError("reference must be present in candidates.")
    if max_checkpoints < 1:
        raise ValueError("max_checkpoints must be positive.")
    guard_values = (
        bacc_drop_tolerance,
        sensitivity_drop_tolerance,
        specificity_drop_tolerance,
        log_loss_increase_tolerance,
    )
    if any(value < 0 for value in guard_values):
        raise ValueError("CP-AWA greedy guard tolerances must be non-negative.")

    ordered = [reference, *(candidate for candidate in candidates if candidate != reference)]
    accepted = [reference]
    states = [load_state(reference)]
    state = weighted_state_average(states, normalized_weights(accepted))
    reference_metrics = evaluate_state(state, tuple(accepted))
    current_metrics = reference_metrics
    trace: list[dict[str, Any]] = [
        {
            "epoch": reference.epoch,
            "checkpoint": reference.checkpoint,
            "decision": "reference",
            "metrics": reference_metrics.as_dict(),
        }
    ]

    def passes(value: float, current: float, baseline: float, tolerance: float) -> bool:
        return value >= current - tolerance - 1e-12 and value >= baseline - tolerance - 1e-12

    for candidate in ordered[1:]:
        if len(accepted) >= max_checkpoints:
            break
        candidate_state = load_state(candidate)
        proposed_candidates = [*accepted, candidate]
        proposed_states = [*states, candidate_state]
        proposed_state = weighted_state_average(
            proposed_states, normalized_weights(proposed_candidates)
        )
        metrics = evaluate_state(proposed_state, tuple(proposed_candidates))
        accepted_by_guard = (
            passes(
                metrics.balanced_accuracy,
                current_metrics.balanced_accuracy,
                reference_metrics.balanced_accuracy,
                bacc_drop_tolerance,
            )
            and passes(
                metrics.sensitivity,
                current_metrics.sensitivity,
                reference_metrics.sensitivity,
                sensitivity_drop_tolerance,
            )
            and passes(
                metrics.specificity,
                current_metrics.specificity,
                reference_metrics.specificity,
                specificity_drop_tolerance,
            )
            and metrics.log_loss
            <= current_metrics.log_loss + log_loss_increase_tolerance + 1e-12
            and metrics.log_loss
            <= reference_metrics.log_loss + log_loss_increase_tolerance + 1e-12
        )
        trace.append(
            {
                "epoch": candidate.epoch,
                "checkpoint": candidate.checkpoint,
                "decision": "accepted" if accepted_by_guard else "rejected",
                "metrics": metrics.as_dict(),
            }
        )
        if accepted_by_guard:
            accepted = proposed_candidates
            states = proposed_states
            state = proposed_state
            current_metrics = metrics

    final_weights = normalized_weights(accepted)
    final_state = weighted_state_average(states, final_weights)
    return CPBuildResult(
        state=final_state,
        accepted_candidates=tuple(accepted),
        normalized_weights=final_weights,
        reference_metrics=reference_metrics,
        final_metrics=current_metrics,
        trace=tuple(trace),
    )
