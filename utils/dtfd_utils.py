"""Shared DTFD grouping and instance distillation for train and evaluation."""
from numbers import Integral
import torch

def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"DTFD {name} must be a positive integer, got {value!r}")
    return int(value)

def dtfd_groups(bag, num_group, total_instance, distill):
    """Return nonempty, balanced groups and each group's per-extreme budget.

    AFS ignores total_instance. MaxS budgets total_instance across the actual
    groups. MaxMinS uses the same budget for EACH extreme (at most twice that
    many selected patches); overlapping top/bottom selections are not repeated.
    Budgets exceeding a group's size are capped without repeating patches.
    """
    num_group = _positive_integer(num_group, 'num_Group')
    if distill not in ('AFS', 'MaxS', 'MaxMinS'):
        raise ValueError(f"Unknown DTFD distillation: {distill!r}")
    if bag.ndim == 3:
        if bag.shape[0] != 1:
            raise ValueError("DTFD accepts one WSI per batch")
        bag = bag.squeeze(0)
    if bag.ndim != 2 or bag.shape[0] == 0:
        raise ValueError("DTFD requires a nonempty [patches, features] bag")
    groups = torch.tensor_split(bag, min(num_group, bag.shape[0]), dim=0)
    if distill == 'AFS':
        return groups, [None]*len(groups)
    budget = _positive_integer(total_instance, 'total_instance')
    if budget < len(groups):
        raise ValueError(f"DTFD {distill}: total_instance={budget} is smaller than "
                         f"the {len(groups)} actual groups; increase the budget, "
                         "reduce num_Group, or use AFS")
    base, remainder = divmod(budget, len(groups))
    return groups, [base+(i < remainder) for i in range(len(groups))]

def dtfd_distill_features(models, bag, num_group, total_instance, distill):
    classifier, attention, reduction, _ = models
    groups, budgets = dtfd_groups(bag, num_group, total_instance, distill)
    pseudo, sub_predictions = [], []
    for chunk, budget in zip(groups, budgets):
        reduced = reduction(chunk)
        weights = attention(reduced).squeeze(0)
        weighted = torch.einsum('ns,n->ns', reduced, weights)

        pooled = weighted.sum(dim=0, keepdim=True)
        sub_predictions.append(classifier(pooled))
        if distill == 'AFS':
            pseudo.append(pooled)
            continue
        classifier_weight = list(classifier.parameters())[-2]
        patch_logits = torch.einsum('nf,cf->nc', weighted, classifier_weight)
        rank = torch.softmax(patch_logits, dim=1)[:, -1].argsort(descending=True)
        k = min(budget, len(rank))
        selected = rank[:k]
        if distill == 'MaxMinS':
            # Rank indices outside the top set; -0 slicing is never used.
            bottom_k = min(k, len(rank)-k)
            if bottom_k:
                selected = torch.cat((selected, rank[-bottom_k:]))
        pseudo.append(reduced.index_select(0, selected))
    return torch.cat(pseudo, dim=0), torch.cat(sub_predictions, dim=0)
