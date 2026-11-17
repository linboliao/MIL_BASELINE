from .prediction_utils import loss_target
import torch
import time
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,roc_curve,precision_recall_fscore_support,balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import time
import torch.nn as nn


def cal_scores(probs, labels, num_classes, input_kind='probabilities'):
    """Classification scores; never apply softmax to probabilities again."""
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if input_kind == 'logits':
        p = torch.softmax(torch.as_tensor(p), dim=-1).numpy()
    elif input_kind != 'probabilities':
        raise ValueError("input_kind must be probabilities or logits")
    if p.shape != (len(y), num_classes) or not len(y):
        raise ValueError("Expected nonempty [samples, num_classes] probabilities")
    if not np.isfinite(p).all() or (p < -1e-6).any() or (p > 1+1e-6).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    if not np.allclose(p.sum(axis=1), 1., atol=1e-5):
        raise ValueError("Class probabilities must sum to one")
    if not np.isin(y, np.arange(num_classes)).all():
        raise ValueError("Labels are outside the configured class range")
    predicted = p.argmax(axis=1)
    supported = len(np.unique(y)) == num_classes
    if num_classes == 2:
        macro_auc = roc_auc_score(y, p[:, 1]) if supported else float('nan')
        micro_auc = weighted_auc = macro_auc
    elif supported:
        macro_auc = roc_auc_score(y, p, average='macro', multi_class='ovr')
        weighted_auc = roc_auc_score(y, p, average='weighted', multi_class='ovr')
        binary_labels = np.eye(num_classes)[y]
        micro_auc = roc_auc_score(binary_labels.reshape(-1), p.reshape(-1))
    else:
        macro_auc = micro_auc = weighted_auc = float('nan')
    metrics = {'acc': accuracy_score(y, predicted), 'bacc': balanced_accuracy_score(y, predicted),
               'macro_auc': macro_auc, 'micro_auc': micro_auc, 'weighted_auc': weighted_auc}
    for average in ('macro', 'micro', 'weighted'):
        metrics[average+'_f1'] = f1_score(y, predicted, average=average, zero_division=0)
        metrics[average+'_recall'] = recall_score(y, predicted, average=average, zero_division=0)
        metrics[average+'_pre'] = precision_score(y, predicted, average=average, zero_division=0)
    metrics['quadratic_kappa'] = cohen_kappa_score(y, predicted, weights='quadratic')
    metrics['linear_kappa'] = cohen_kappa_score(y, predicted, weights='linear')
    metrics['confusion_mat'] = confusion_matrix(y, predicted, labels=np.arange(num_classes))
    return metrics

def train_loop(device,model,loader,criterion,optimizer,scheduler):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        train_logits = model(bag)['logits']
        train_loss = criterion(train_logits, loss_target(criterion, train_logits, label))
        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def val_loop(device,num_classes,model,loader,criterion,retrun_WSI_feature = False,return_WSI_attn=False, return_predictions=False, prediction_path=None, slide_paths=None):
    from .evaluation_utils import evaluate_mil
    return evaluate_mil(device, num_classes, model, loader, criterion, kind='generic',
                        return_feature=retrun_WSI_feature, return_attention=return_WSI_attn,
                        return_predictions=return_predictions, prediction_path=prediction_path, slide_paths=slide_paths)

def ac_train_loop(device,model,loader,criterion,optimizer,scheduler,n_token):
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        forward_return = model(bag)
        train_logits = forward_return['logits']
        sub_preds = forward_return['sub_preds']
        attns = forward_return['attns']
        if n_token > 1:
            loss0 = criterion(sub_preds, loss_target(criterion, sub_preds, label.repeat_interleave(n_token)))
        else:
            loss0 = torch.tensor(0.)
        diff_loss = torch.tensor(0).to(device, dtype=torch.float)
        attns = torch.softmax(attns, dim=-1)

        for i in range(n_token):
            for j in range(i + 1, n_token): 
                diff_loss += torch.cosine_similarity(attns[:, i], attns[:, j], dim=-1).mean() / (
                            n_token * (n_token - 1) / 2)
        train_loss = criterion(train_logits, loss_target(criterion, train_logits, label))
        train_loss = diff_loss + loss0 + train_loss
        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def ac_val_loop(device,num_classes,model,loader,criterion,n_token,retrun_WSI_feature = False,return_WSI_attn=False, return_predictions=False, prediction_path=None, slide_paths=None):
    from .evaluation_utils import evaluate_mil
    return evaluate_mil(device, num_classes, model, loader, criterion, kind='ac', n_token=n_token,
                        return_feature=retrun_WSI_feature, return_attention=return_WSI_attn,
                        return_predictions=return_predictions, prediction_path=prediction_path, slide_paths=slide_paths)


# clam has instance-loss defferent from other mil models
def clam_train_loop(device,model,loader,criterion,optimizer,scheduler,bag_weight):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        forward_return = model(bag,label=label)
        instance_loss = forward_return['instance_loss']
        train_logits = forward_return['logits']
        train_loss = criterion(train_logits, loss_target(criterion, train_logits, label))
        total_loss = train_loss * bag_weight + instance_loss * (1-bag_weight)
        train_loss_log += total_loss.item()
        total_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def tripleloss(golabal,p_center,nc_center):
    golabal = golabal.squeeze(0)
    n_globallesionrepresente, _ = golabal.shape
    p_center = p_center.repeat(n_globallesionrepresente, 1)
    nc_center = nc_center.repeat(n_globallesionrepresente, 1)
    triple_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y) ,margin=1)
    loss = triple_loss(golabal,p_center,nc_center)
    return loss

def dgr_train_loop(device,model,loader,criterion,optimizer,scheduler,now_epoch,epoch_des,n_lesion):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        if torch.argmax(label)==0:
            forward_return = model(bag,bag_mode='normal')
            train_logits, A,H,p_center,nc_center,lesion = forward_return['logits'],forward_return['A'],forward_return['H'],forward_return['postivecenter'],forward_return['normalcenter'],forward_return['lesion_enhacing']
        else:
            train_logits, A,H,p_center,nc_center,lesion= model(bag,bag_mode='abnormal')
        if now_epoch < epoch_des:
            train_loss = criterion(train_logits, loss_target(criterion, train_logits, label))
        else:
            train_loss = criterion(train_logits, loss_target(criterion, train_logits, label))
            lesion_norm = lesion.squeeze(0)
            lesion_norm = torch.nn.functional.normalize(lesion_norm)
            div_loss = -torch.logdet(lesion_norm@lesion_norm.T+1e-10*torch.eye(n_lesion).to(device))
            sim_loss = tripleloss(lesion,p_center,nc_center)
            train_loss = train_loss + 0.1*div_loss + 0.1*sim_loss 

        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time

def clam_val_loop(device,num_classes,model,loader,criterion,bag_weight,retrun_WSI_feature = False,return_WSI_attn=False, return_predictions=False, prediction_path=None, slide_paths=None):
    from .evaluation_utils import evaluate_mil
    return evaluate_mil(device, num_classes, model, loader, criterion, kind='clam', bag_weight=bag_weight,
                        return_feature=retrun_WSI_feature, return_attention=return_WSI_attn,
                        return_predictions=return_predictions, prediction_path=prediction_path, slide_paths=slide_paths)

def ds_train_loop(device,model,loader,criterion,optimizer,scheduler):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    model = model.to(device)
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        forward_return = model(bag)
        max_prediction = forward_return['max_prediction']
        train_logits = forward_return['logits']
        loss_bag = criterion(train_logits, loss_target(criterion, train_logits, label))
        loss_max = criterion(max_prediction, loss_target(criterion, max_prediction, label))
        train_loss = 0.5*loss_bag + 0.5*loss_max
        train_loss_log += train_loss.item()

        train_loss.backward()

        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time

def ds_val_loop(device,num_classes,model,loader,criterion,retrun_WSI_feature = False,return_WSI_attn=False, return_predictions=False, prediction_path=None, slide_paths=None):
    from .evaluation_utils import evaluate_mil
    return evaluate_mil(device, num_classes, model, loader, criterion, kind='ds',
                        return_feature=retrun_WSI_feature, return_attention=return_WSI_attn,
                        return_predictions=return_predictions, prediction_path=prediction_path, slide_paths=slide_paths)

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def dtfd_train_loop(device, model_list, loader, criterion, optimizer_list, scheduler_list, num_Group, grad_clipping,distill,total_instance):
    train_loss_log = 0
    start = time.time()
    # Unpack model list
    classifier, attention, dimReduction, attCls = model_list
    classifier.train()
    attention.train()
    dimReduction.train()
    attCls.train()

    # Unpack optimizer and scheduler lists
    optimizer_A, optimizer_B = optimizer_list
    scheduler_A, scheduler_B = scheduler_list

    total_loss = 0
    for i, data in enumerate(loader):
        label = data[1].long().to(device)
        bag = data[0].to(device).float()

        from .dtfd_utils import dtfd_distill_features
        slide_pseudo_feat, slide_sub_preds = dtfd_distill_features(
            model_list, bag, num_Group, total_instance, distill)
        slide_sub_labels = label.repeat(slide_sub_preds.shape[0])

        # Calculate and backpropagate loss for the first tier
        loss_A = criterion(slide_sub_preds, loss_target(criterion, slide_sub_preds, slide_sub_labels))
        optimizer_A.zero_grad()
        loss_A.backward(retain_graph=True)
        total_loss += loss_A.item()

        # Clip gradients and update weights
        torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), grad_clipping)
        torch.nn.utils.clip_grad_norm_(attention.parameters(), grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clipping)


        # Second tier optimization
        gSlidePred = attCls(slide_pseudo_feat)['logits']
        loss_B = criterion(gSlidePred, loss_target(criterion, gSlidePred, label)).mean()
        optimizer_B.zero_grad()
        loss_B.backward()
        total_loss += loss_B.item()

        # Clip gradients and update weights
        torch.nn.utils.clip_grad_norm_(attCls.parameters(), grad_clipping)
        optimizer_A.step()
        optimizer_B.step()

    # Step schedulers
    if scheduler_A is not None:
        scheduler_A.step()
    if scheduler_B is not None:
        scheduler_B.step()

    end = time.time()
    total_loss /= len(loader)
    total_time = end - start

    return total_loss, total_time


def dtfd_val_loop(device,num_classes,model_list,loader,criterion,num_Group,grad_clipping,distill,total_instance,retrun_WSI_feature = False,return_WSI_attn=False, return_predictions=False, prediction_path=None, slide_paths=None):
    from .evaluation_utils import evaluate_mil
    return evaluate_mil(device, num_classes, model_list, loader, criterion, kind='dtfd', num_group=num_Group, total_instance=total_instance, distill=distill,
                        return_feature=retrun_WSI_feature, return_attention=return_WSI_attn,
                        return_predictions=return_predictions, prediction_path=prediction_path, slide_paths=slide_paths)


# ============================================
# Mixup Training Loop Utilities
# ============================================
import numpy as np

def train_loop_with_mixup(device, model, dataloader, criterion, optimizer, scheduler, mixup_config, mix_fn):
    """
    Generic training loop with mixup augmentation
    Args:
        device: torch device
        model: MIL model
        dataloader: training dataloader
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler
        mixup_config: dict with mixup parameters (prob, alpha, etc.)
        mix_fn: mixing function from the specific module (e.g., mixup_data, insmix_data, etc.)
    Returns:
        train_loss, cost_time
    """
    model.train()
    train_loss = 0.
    start_time = time.time()
    
    all_features = []
    all_labels = []
    
    for idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            _, data, label = batch
        else:
            data, label = batch
        all_features.append(data.squeeze(0).to(device))
        all_labels.append(label.to(device))
    
    prob = mixup_config.get('prob', 0.5)
    n_samples = len(all_features)
    perm = torch.randperm(n_samples)
    
    for i in range(n_samples):
        optimizer.zero_grad()
        
        if np.random.rand() < prob:
            j = perm[i].item()
            if i != j:
                # Call the specific mix function with appropriate kwargs
                mix_kwargs = {k: v for k, v in mixup_config.items() if k != 'prob'}
                # For rankmix, we need to pass the model
                if 'model' in mix_fn.__code__.co_varnames:
                    mixed_feat, mixed_label, _ = mix_fn(
                        all_features[i], all_labels[i],
                        all_features[j], all_labels[j],
                        model, **mix_kwargs
                    )
                else:
                    mixed_feat, mixed_label, _ = mix_fn(
                        all_features[i], all_labels[i],
                        all_features[j], all_labels[j],
                        **mix_kwargs
                    )
            else:
                mixed_feat, mixed_label = all_features[i], all_labels[i]
        else:
            mixed_feat, mixed_label = all_features[i], all_labels[i]
        
        forward_return = model(mixed_feat.unsqueeze(0))
        logits = forward_return['logits']
        
        if mixed_label.dim() == 0 or mixed_label.size(0) == 1:
            if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                label_for_loss = F.one_hot(mixed_label.long(), num_classes=logits.size(1)).float()
            else:
                label_for_loss = mixed_label.long()
        else:
            label_for_loss = mixed_label.float()
        
        if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
            loss = criterion(logits, label_for_loss)
        else:
            loss = criterion(logits, label_for_loss.argmax(dim=-1) if label_for_loss.dim() > 1 else label_for_loss)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if scheduler is not None:
        scheduler.step()
    
    train_loss /= n_samples
    cost_time = time.time() - start_time
    
    return train_loss, cost_time
