"""Scalar prediction records shared by all MIL evaluation paths."""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import SequentialSampler

def loss_target(criterion, logits, labels):
    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        if logits.shape[-1] == 1:
            return labels.to(logits.dtype).reshape_as(logits)
        return torch.nn.functional.one_hot(labels.long(), logits.shape[-1]).to(logits.dtype)
    return labels.long()

def class_probabilities(logits, criterion):
    if isinstance(criterion, torch.nn.BCEWithLogitsLoss) and logits.shape[-1] == 1:
        p = torch.sigmoid(logits)
        return torch.cat((1-p, p), dim=-1)
    return torch.softmax(logits, dim=-1)


class PredictionRecorder:
    def __init__(self, loader, enabled=False, slide_paths=None):
        self.loader, self.enabled, self.rows = loader, enabled, []
        self.offset = 0
        self.slide_paths = slide_paths
        sampler = getattr(loader, 'sampler', None)
        self.sequential = isinstance(sampler, SequentialSampler)
        if slide_paths is not None and not self.sequential:
            raise ValueError("Positional slide_paths require a sequential evaluation loader")

    def add(self, batch, logits, probabilities, labels, extras=None):
        if not self.enabled:
            return
        logits = logits.detach().cpu().reshape(len(labels), -1)
        probabilities = probabilities.detach().cpu().reshape(len(labels), -1)
        labels = labels.detach().cpu().reshape(-1)
        explicit = len(batch) > 2 and isinstance(batch[2], (list, tuple)) and all(isinstance(x, str) for x in batch[2])
        ds = getattr(self.loader, 'dataset', None)
        paths = self.slide_paths if self.slide_paths is not None else getattr(ds, 'slide_path_list', None)
        if not explicit and not self.sequential:
            raise ValueError("Prediction identity requires sequential evaluation or explicit sample IDs in batch[2]")
        for j, label in enumerate(labels):
            idx = self.offset+j
            path = str(paths[idx]) if paths is not None and self.sequential else ''
            if explicit:
                sample_id = batch[2][j]
            elif hasattr(ds, 'sample_id_list'):
                sample_id = str(ds.sample_id_list[idx])
            elif path:
                sample_id = path
            else:
                raise ValueError("Prediction output requires dataset paths or explicit sample IDs")
            row = {'sample_index': idx, 'sample_id': sample_id,
                   'slide_id': Path(path or sample_id).stem, 'slide_path': path,
                   'label': int(label), 'prediction': int(probabilities[j].argmax())}
            row.update({f'prob_{k}': float(v) for k,v in enumerate(probabilities[j])})
            row.update({f'logit_{k}': float(v) for k,v in enumerate(logits[j])})
            if extras:
                row.update(extras[j])
            self.rows.append(row)
        self.offset += len(labels)

    def finish(self, loss, metrics, return_predictions=False, prediction_path=None):
        if not self.enabled:
            return loss, metrics
        frame = pd.DataFrame(self.rows)
        if frame.empty:
            raise ValueError("Cannot save predictions for an empty evaluation set")
        if prediction_path is not None:
            p = Path(prediction_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('x', encoding='utf-8', newline='') as f:
                frame.to_csv(f, index=False)
        return (loss, metrics, frame) if return_predictions else (loss, metrics)

def prediction_path(args, split, epoch):
    if not args.General.get('save_predictions', False):
        return None
    return Path(args.Logs.now_log_dir)/'predictions'/f'{split}_epoch_{epoch:04d}.csv'
