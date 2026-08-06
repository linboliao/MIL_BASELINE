"""Lightweight reliability-aware learnable aggregation for SPE.

The meta-learner consumes architecture-level OOF probabilities rather than
WSI features.  It is deliberately small, permutation-compatible at the token
encoder level, and regularized toward uniform averaging.  Cross-fitted OOF
predictions are returned for honest development diagnostics; a final model is
then fitted on all OOF samples and locked for independent cohorts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ensemble.spe import class_patient_equal_sample_weights, patient_equal_sample_weights


@dataclass(frozen=True)
class RASPEFit:
    architecture_names: tuple[str, ...]
    hidden_dim: int
    epochs: int
    learning_rate: float
    weight_decay: float
    member_dropout: float
    uniform_kl_lambda: float
    consistency_lambda: float
    group_dro_temperature: float
    max_weight: float
    state_variance_used: bool
    state_variance_scale: float
    seed: int
    cross_fitted_balanced_accuracy: float
    cross_fitted_sensitivity: float
    cross_fitted_specificity: float

    def as_dict(self) -> dict:
        return asdict(self)


class ReliabilityAwareAggregator(nn.Module):
    """Shared token encoder plus sample-specific architecture gating."""

    def __init__(
        self,
        architecture_count: int,
        hidden_dim: int = 16,
        max_weight: float = 0.30,
        state_variance_scale: float = 1.0,
        use_state_variance: bool = True,
    ) -> None:
        super().__init__()
        if architecture_count < 2:
            raise ValueError("RA-SPE requires at least two architectures.")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2.")
        if max_weight < 1.0 / architecture_count or max_weight > 1.0:
            raise ValueError("max_weight is infeasible for the architecture count.")
        self.architecture_count = int(architecture_count)
        self.hidden_dim = int(hidden_dim)
        self.max_weight = float(max_weight)
        self.use_state_variance = bool(use_state_variance)
        self.register_buffer(
            "state_variance_scale",
            torch.tensor(max(float(state_variance_scale), 1e-8), dtype=torch.float32),
        )
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        encoded_dim = hidden_dim // 2
        self.gate = nn.Sequential(
            nn.Linear(encoded_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.architecture_bias = nn.Parameter(torch.zeros(architecture_count))

    def _features(
        self, probabilities: torch.Tensor, state_variances: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clipped = probabilities.clamp(1e-5, 1.0 - 1e-5)
        logits = torch.logit(clipped)
        entropy = -(clipped * clipped.log() + (1.0 - clipped) * (1.0 - clipped).log())
        disagreement = (clipped - clipped.mean(dim=1, keepdim=True)).abs()
        if self.use_state_variance:
            if state_variances is None:
                raise ValueError("state_variances are required by this RA-SPE model.")
            stability = torch.log1p(
                state_variances.clamp_min(0.0) / self.state_variance_scale
            )
        else:
            stability = torch.zeros_like(clipped)
        return torch.stack((logits, entropy, disagreement, stability), dim=-1), logits

    def forward(
        self,
        probabilities: torch.Tensor,
        state_variances: torch.Tensor | None = None,
        member_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if probabilities.ndim != 2 or probabilities.shape[1] != self.architecture_count:
            raise ValueError("probabilities must have shape [samples, architectures].")
        features, member_logits = self._features(probabilities, state_variances)
        encoded = self.encoder(features)
        if member_mask is None:
            member_mask = torch.ones_like(probabilities, dtype=torch.bool)
        active = member_mask.to(encoded.dtype)
        counts = active.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (encoded * active.unsqueeze(-1)).sum(dim=1) / counts
        centered = (encoded - mean.unsqueeze(1)) * active.unsqueeze(-1)
        std = torch.sqrt((centered.square().sum(dim=1) / counts).clamp_min(1e-8))
        context = torch.cat((mean, std), dim=1).unsqueeze(1).expand(-1, encoded.shape[1], -1)
        gate_logits = self.gate(torch.cat((encoded, context), dim=-1)).squeeze(-1)
        gate_logits = gate_logits + self.architecture_bias.unsqueeze(0)
        gate_logits = gate_logits.masked_fill(~member_mask, -1e9)
        soft_weights = torch.softmax(gate_logits, dim=1)

        # Convexly mix dynamic gates with the active-member uniform prior.  The
        # resulting weights are normalized and bounded without post-hoc clipping.
        uniform = active / counts
        feasible_cap = torch.maximum(
            torch.full_like(counts, self.max_weight), 1.0 / counts
        )
        denominator = (1.0 - 1.0 / counts).clamp_min(1e-8)
        alpha = ((feasible_cap - 1.0 / counts) / denominator).clamp(0.0, 1.0)
        alpha = torch.where(counts <= 1.0, torch.zeros_like(alpha), alpha)
        weights = uniform + alpha * (soft_weights - uniform)
        aggregate_logit = (weights * member_logits).sum(dim=1)
        return aggregate_logit, weights


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float]:
    positive = labels == 1
    negative = labels == 0
    sensitivity = float(np.mean(predictions[positive] == 1))
    specificity = float(np.mean(predictions[negative] == 0))
    return (sensitivity + specificity) / 2.0, sensitivity, specificity


def _dropout_mask(shape: tuple[int, int], probability: float, generator: torch.Generator) -> torch.Tensor:
    mask = torch.rand(shape, generator=generator) >= probability
    # Keep at least two members so dropout cannot turn training into a singleton expert.
    for row in range(shape[0]):
        if int(mask[row].sum()) < 2:
            indices = torch.randperm(shape[1], generator=generator)[:2]
            mask[row, indices] = True
    return mask


def _train_model(
    probabilities: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    fold_ids: np.ndarray,
    state_variances: np.ndarray | None,
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    member_dropout: float,
    uniform_kl_lambda: float,
    consistency_lambda: float,
    group_dro_temperature: float,
    max_weight: float,
    class_balance: bool,
    seed: int,
) -> ReliabilityAwareAggregator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    positive_variances = (
        state_variances[state_variances > 0] if state_variances is not None else np.array([])
    )
    variance_scale = float(np.median(positive_variances)) if positive_variances.size else 1.0
    model = ReliabilityAwareAggregator(
        probabilities.shape[1],
        hidden_dim=hidden_dim,
        max_weight=max_weight,
        state_variance_scale=variance_scale,
        use_state_variance=state_variances is not None,
    )
    probability_tensor = torch.as_tensor(probabilities, dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32)
    variance_tensor = (
        None if state_variances is None else torch.as_tensor(state_variances, dtype=torch.float32)
    )
    numpy_weights = (
        class_patient_equal_sample_weights(labels, patient_ids)
        if class_balance
        else patient_equal_sample_weights(patient_ids)
    )
    sample_weights = torch.as_tensor(numpy_weights, dtype=torch.float32)
    fold_tensor = torch.as_tensor(fold_ids)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 1729)
    unique_folds = torch.unique(fold_tensor)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        aggregate, gates = model(probability_tensor, variance_tensor)
        per_sample = nn.functional.binary_cross_entropy_with_logits(
            aggregate, label_tensor, reduction="none"
        )
        group_risks = []
        for fold in unique_folds:
            mask = fold_tensor == fold
            local_weights = sample_weights[mask]
            local_weights = local_weights / local_weights.sum()
            group_risks.append((local_weights * per_sample[mask]).sum())
        risks = torch.stack(group_risks)
        if group_dro_temperature > 0:
            primary_loss = group_dro_temperature * torch.logsumexp(
                risks / group_dro_temperature, dim=0
            )
        else:
            primary_loss = risks.max()
        uniform_kl = (
            gates * (gates.clamp_min(1e-8).log() + np.log(probabilities.shape[1]))
        ).sum(dim=1).mean()
        consistency = torch.zeros((), dtype=torch.float32)
        if member_dropout > 0 and consistency_lambda > 0:
            mask_one = _dropout_mask(probabilities.shape, member_dropout, generator)
            mask_two = _dropout_mask(probabilities.shape, member_dropout, generator)
            dropped_one, _ = model(probability_tensor, variance_tensor, mask_one)
            dropped_two, _ = model(probability_tensor, variance_tensor, mask_two)
            consistency = nn.functional.mse_loss(
                torch.sigmoid(dropped_one), torch.sigmoid(dropped_two)
            )
        loss = (
            primary_loss
            + uniform_kl_lambda * uniform_kl
            + consistency_lambda * consistency
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    return model.eval()


def predict_ra_spe(
    model: ReliabilityAwareAggregator,
    probabilities: np.ndarray,
    state_variances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-class probabilities and sample-specific member weights."""
    with torch.no_grad():
        probability_tensor = torch.as_tensor(probabilities, dtype=torch.float32)
        variance_tensor = (
            None if state_variances is None else torch.as_tensor(state_variances, dtype=torch.float32)
        )
        logits, weights = model(probability_tensor, variance_tensor)
        return torch.sigmoid(logits).numpy(), weights.numpy()


def fit_ra_spe(
    probabilities: np.ndarray,
    labels: Sequence,
    patient_ids: Sequence,
    fold_ids: Sequence,
    architecture_names: Sequence[str],
    state_variances: np.ndarray | None = None,
    *,
    hidden_dim: int = 16,
    epochs: int = 300,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-3,
    member_dropout: float = 0.20,
    uniform_kl_lambda: float = 0.05,
    consistency_lambda: float = 0.10,
    group_dro_temperature: float = 0.10,
    max_weight: float = 0.30,
    class_balance: bool = True,
    seed: int = 2026,
) -> tuple[RASPEFit, ReliabilityAwareAggregator, np.ndarray, np.ndarray]:
    """Cross-fit RA-SPE for OOF diagnostics, then fit its locked final model."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    patients = np.asarray(patient_ids, dtype=str)
    folds = np.asarray(fold_ids)
    names = tuple(str(name) for name in architecture_names)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(names):
        raise ValueError("architecture_names do not match probability columns.")
    if len(probabilities) != len(labels_array) or len(patients) != len(labels_array):
        raise ValueError("Labels/patient IDs do not match probability rows.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite and lie in [0, 1].")
    if len(np.unique(folds)) < 2:
        raise ValueError("RA-SPE cross-fitting requires at least two folds.")
    if state_variances is not None:
        state_variances = np.asarray(state_variances, dtype=np.float64)
        if state_variances.shape != probabilities.shape:
            raise ValueError("state_variances must match probabilities.")
    if not 0 <= member_dropout < 1:
        raise ValueError("member_dropout must lie in [0, 1).")

    kwargs = dict(
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        member_dropout=member_dropout,
        uniform_kl_lambda=uniform_kl_lambda,
        consistency_lambda=consistency_lambda,
        group_dro_temperature=group_dro_temperature,
        max_weight=max_weight,
        class_balance=class_balance,
    )
    cross_fitted = np.empty(len(labels_array), dtype=np.float64)
    cross_fitted_weights = np.empty_like(probabilities)
    for offset, held_out_fold in enumerate(np.unique(folds)):
        train_mask = folds != held_out_fold
        validation_mask = ~train_mask
        model = _train_model(
            probabilities[train_mask],
            labels_array[train_mask],
            patients[train_mask],
            folds[train_mask],
            None if state_variances is None else state_variances[train_mask],
            seed=seed + offset,
            **kwargs,
        )
        predicted, gates = predict_ra_spe(
            model,
            probabilities[validation_mask],
            None if state_variances is None else state_variances[validation_mask],
        )
        cross_fitted[validation_mask] = predicted
        cross_fitted_weights[validation_mask] = gates

    final_model = _train_model(
        probabilities,
        labels_array,
        patients,
        folds,
        state_variances,
        seed=seed + 1000,
        **kwargs,
    )
    predictions = (cross_fitted >= 0.5).astype(np.int64)
    bacc, sensitivity, specificity = _balanced_accuracy(labels_array, predictions)
    fit = RASPEFit(
        architecture_names=names,
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        member_dropout=float(member_dropout),
        uniform_kl_lambda=float(uniform_kl_lambda),
        consistency_lambda=float(consistency_lambda),
        group_dro_temperature=float(group_dro_temperature),
        max_weight=float(max_weight),
        state_variance_used=state_variances is not None,
        state_variance_scale=float(final_model.state_variance_scale.item()),
        seed=int(seed),
        cross_fitted_balanced_accuracy=bacc,
        cross_fitted_sensitivity=sensitivity,
        cross_fitted_specificity=specificity,
    )
    return fit, final_model, cross_fitted, cross_fitted_weights


def ra_spe_checkpoint(
    fit: RASPEFit, model: ReliabilityAwareAggregator
) -> dict:
    """Build a portable torch checkpoint with architecture order and settings."""
    return {
        "format_version": 1,
        "fit": fit.as_dict(),
        "state_dict": model.state_dict(),
    }


def load_ra_spe_checkpoint(
    path: str, map_location: str | torch.device = "cpu"
) -> tuple[RASPEFit, ReliabilityAwareAggregator]:
    """Restore a locked RA-SPE model for inference on another cohort."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location=map_location)
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("Unsupported RA-SPE checkpoint format.")
    fit = RASPEFit(**payload["fit"])
    model = ReliabilityAwareAggregator(
        architecture_count=len(fit.architecture_names),
        hidden_dim=fit.hidden_dim,
        max_weight=fit.max_weight,
        state_variance_scale=fit.state_variance_scale,
        use_state_variance=fit.state_variance_used,
    )
    model.load_state_dict(payload["state_dict"])
    return fit, model.eval()
