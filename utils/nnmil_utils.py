from .prediction_utils import loss_target
"""Data batching, training and evaluation utilities for NN_MIL."""

import math
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Sampler

from .loop_utils import cal_scores


class BalancedBatchSampler(Sampler):
    """Yield approximately class-balanced batches, recycling a class only when needed."""

    def __init__(self, labels, batch_size, seed=2024):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.labels = [int(label) for label in labels]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.class_indices = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.class_indices[label].append(index)
        if len(self.class_indices) < 2:
            raise ValueError("BalancedBatchSampler requires at least two classes")

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + torch.initial_seed() % (2 ** 31 - 1))
        classes = sorted(self.class_indices)
        pools = {label: torch.tensor(indices)[torch.randperm(len(indices), generator=generator)].tolist()
                 for label, indices in self.class_indices.items()}
        positions = {label: 0 for label in classes}

        def next_index(label):
            if positions[label] >= len(pools[label]):
                base = torch.tensor(self.class_indices[label])
                pools[label] = base[torch.randperm(len(base), generator=generator)].tolist()
                positions[label] = 0
            selected = pools[label][positions[label]]
            positions[label] += 1
            return selected

        for batch_number in range(len(self)):
            size = min(self.batch_size, len(self.labels) - batch_number * self.batch_size)
            targets = [classes[position % len(classes)] for position in range(size)]
            order = torch.randperm(size, generator=generator).tolist()
            yield [next_index(targets[position]) for position in order]


def resolve_fixed_bag_size(dataset, configured_size="auto", factor=0.5):
    """Resolve paper-style M: a fraction of the train-bag median."""
    if isinstance(configured_size, str) and configured_size.lower() == "auto":
        lengths = [int(dataset[index][0].shape[0]) for index in range(len(dataset))]
        if not lengths or min(lengths) < 1:
            raise ValueError("All NN_MIL training bags must contain at least one patch")
        median = float(torch.tensor(lengths, dtype=torch.float32).median().item())
        return max(1, int(median * float(factor))), lengths
    size = int(configured_size)
    if size < 1:
        raise ValueError("fixed_bag_size must be a positive integer or 'auto'")
    return size, None


def fixed_bag_collate(batch, bag_size):
    """Randomly subsample or zero-pad feature bags and return a valid-patch mask."""
    features, labels, masks = [], [], []
    for feature, label in batch:
        if feature.ndim == 3 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        if feature.ndim != 2:
            raise ValueError("Each NN_MIL feature bag must have shape [patches, features]")
        patch_count = feature.shape[0]
        if patch_count < 1:
            raise ValueError("NN_MIL cannot train on an empty patch bag")
        if patch_count > bag_size:
            selected = torch.randperm(patch_count)[:bag_size]
            feature = feature.index_select(0, selected)
            mask = torch.ones(bag_size, dtype=torch.bool)
        elif patch_count < bag_size:
            padding = torch.zeros(bag_size - patch_count, feature.shape[1], dtype=feature.dtype)
            feature = torch.cat([feature, padding], dim=0)
            mask = torch.zeros(bag_size, dtype=torch.bool)
            mask[:patch_count] = True
        else:
            mask = torch.ones(bag_size, dtype=torch.bool)
        features.append(feature)
        labels.append(label)
        masks.append(mask)
    return torch.stack(features), torch.stack(labels).long(), torch.stack(masks)


def nnmil_train_loop(device, model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for bags, labels, valid_masks in loader:
        bags, labels = bags.to(device).float(), labels.to(device).long()
        valid_masks = valid_masks.to(device)
        optimizer.zero_grad()
        logits = model(bags, valid_mask=valid_masks)["logits"]
        loss = criterion(logits, loss_target(criterion, logits, labels))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
    return total_loss / max(1, len(loader))


@torch.no_grad()

def nnmil_val_loop(device, num_classes, model, loader, criterion, return_predictions=False,
                   slide_paths=None, prediction_path=None):
    from .prediction_utils import PredictionRecorder, loss_target, class_probabilities
    model.to(device).eval()
    enabled = return_predictions or prediction_path is not None
    recorder = PredictionRecorder(loader, enabled, slide_paths)
    total_loss, labels, probabilities = 0., [], []
    for batch in loader:
        bags, target = batch[0].to(device).float(), batch[1].to(device).long()
        mask = torch.ones(bags.shape[:2], device=device, dtype=torch.bool)
        output = model(bags, valid_mask=mask)
        logits = output['logits']
        total_loss += criterion(logits, loss_target(criterion, logits, target)).item()
        probs = class_probabilities(logits, criterion)
        probabilities.extend(probs.cpu().tolist())
        labels.extend(target.cpu().tolist())
        extras = []
        if enabled:
            for j in range(len(target)):
                row = {k: float(output[k][j]) for k in
                       ('total_entropy', 'aleatoric_entropy', 'mutual_information')}
                row['num_subspaces'] = int(output['chunk_logits'].shape[0])
                row.update({f'probability_variance_class_{k}': float(v) for k,v in enumerate(output['probability_variance'][j])})
                row.update({f'probability_class_{k}': float(v) for k,v in enumerate(probs[j])})
                extras.append(row)
        recorder.add(batch, logits, probs, target, extras)
    if not labels:
        raise ValueError("Empty NN_MIL evaluation loader")
    return recorder.finish(total_loss/len(loader), cal_scores(probabilities, labels, num_classes),
                           return_predictions, prediction_path)
