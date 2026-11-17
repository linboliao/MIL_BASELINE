"""Common evaluation contract; algorithm-specific losses remain explicit."""
import numpy as np
import torch
from .prediction_utils import PredictionRecorder, loss_target, class_probabilities

def _dtfd_forward(models, bag, num_group, total_instance, distill, feature=False, attention=False):
    from .dtfd_utils import dtfd_distill_features
    pseudo, _ = dtfd_distill_features(models, bag, num_group, total_instance, distill)
    return models[3](pseudo, return_WSI_feature=feature, return_WSI_attn=attention)

@torch.no_grad()
def evaluate_mil(device, num_classes, model, loader, criterion, *, kind='generic',
                 bag_weight=None, n_token=None, num_group=None, total_instance=None,
                 distill=None, return_feature=False, return_attention=False,
                 return_predictions=False, prediction_path=None, slide_paths=None):
    from .loop_utils import cal_scores
    if return_feature and return_attention:
        raise ValueError("Select either feature or attention extraction")
    if (return_feature or return_attention) and (return_predictions or prediction_path is not None):
        raise ValueError("Prediction output and feature/attention extraction are separate operations")
    models = list(model) if kind == 'dtfd' else [model]


    for m in models:
        m.to(device).eval()
    recorder = PredictionRecorder(loader, return_predictions or prediction_path is not None, slide_paths)
    features, attentions, probabilities, labels = [], [], [], []
    total_loss = 0.
    for batch in loader:
        bag, target = batch[0].to(device).float(), batch[1].to(device).long()
        kwargs = {}
        if return_feature:
            kwargs['return_WSI_feature'] = True
        if return_attention:
            kwargs['return_WSI_attn'] = True
        if kind == 'dtfd':
            output = _dtfd_forward(models, bag, num_group, total_instance, distill,
                                   return_feature, return_attention)
        elif kind == 'clam':
            output = model(bag, label=target, **kwargs)
        elif kind == 'sc':
            output = model(bag[..., :-2], coords=bag[..., -2:], **kwargs)
        else:
            output = model(bag, **kwargs)
        if return_feature:
            features.append(output['WSI_feature'].detach().cpu())
            continue
        if return_attention:
            attentions.append(output['WSI_attn'].detach().cpu())
            continue
        logits = output['logits']
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        loss = criterion(logits, loss_target(criterion, logits, target))
        if kind == 'clam':
            if bag_weight is None or not 0 <= float(bag_weight) <= 1:
                raise ValueError("CLAM bag_weight must be between zero and one")
            if 'instance_loss' not in output:
                raise ValueError("CLAM output lacks instance_loss")
            loss = loss*float(bag_weight)+output['instance_loss']*(1-float(bag_weight))
        elif kind == 'ds':
            max_logits = output['max_prediction']
            loss = .5*loss+.5*criterion(max_logits, loss_target(criterion, max_logits, target))
        elif kind == 'ac':
            tokens = int(n_token)
            if tokens > 1:
                sub = output['sub_preds']
                repeated = target.repeat_interleave(tokens)
                loss = loss+criterion(sub, loss_target(criterion, sub, repeated))
                attns = torch.softmax(output['attns'], dim=-1)
                penalty = logits.new_zeros(())
                for i in range(tokens):
                    for j in range(i+1, tokens):
                        penalty += torch.cosine_similarity(attns[:, i], attns[:, j], dim=-1).mean()/(tokens*(tokens-1)/2)
                loss = loss+penalty
        probs = class_probabilities(logits, criterion)
        probabilities.extend(probs.cpu().tolist())
        labels.extend(target.cpu().reshape(-1).tolist())
        total_loss += float(loss)
        recorder.add(batch, logits, probs, target)
    if return_feature:
        if not features:
            raise ValueError("Empty feature extraction loader")
        return torch.cat(features, dim=0).numpy()
    if return_attention:
        return attentions
    if not labels:
        raise ValueError("Empty evaluation loader")
    metrics = cal_scores(probabilities, labels, num_classes)
    return recorder.finish(total_loss/len(loader), metrics, return_predictions, prediction_path)
