"""One-anchor Panel A/B flip and error detection."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def read_panel(root, cfg):
    path = cfg['panel']
    pred = pd.read_csv(root/path['reference_predictions.csv'], dtype={'stem': str})
    pairs = pd.read_csv(root/path['reviewed_pair_manifest.csv'], dtype={'left_stem': str, 'right_stem': str})
    slides = pd.read_csv(root/path['reviewed_slide_manifest.csv'], dtype={'stem': str})
    truth = slides.assign(panel=slides.cohort).set_index(['panel','stem'])
    pred = pred[pred.model_id.isin(cfg['primary_models'])].copy()
    if set(pred.model_id) != set(cfg['primary_models']) or not truth.index.is_unique:
        raise ValueError('Incomplete model or whitelist')
    for _, frame in pred.groupby('model_id'):
        a = frame.set_index(['panel','stem'])
        if not a.index.is_unique or set(a.index) != set(truth.index):
            raise ValueError('Predictions do not cover the whitelist')
        for field in ('case_id','manual_label','identity_union_group','quality_usable_for_primary'):
            if not a[field].reindex(truth.index).equals(truth[field]):
                raise ValueError(f'Prediction provenance mismatch: {field}')
    if not (pred.threshold.eq(.5).all() and pred.decision.eq(pred.prob_pos.ge(.5)).all()):
        raise ValueError('Unexpected Panel working point')
    return pred, pairs


def summarize(pred, pairs, draws, seed):
    output = []
    for (model, panel), frame in pred.groupby(['model_id','panel']):
        frame = frame[frame.quality_usable_for_primary & frame.manual_label.notna()].set_index('stem').copy()
        partners = {s: [] for s in frame.index}
        selected = pairs[(pairs.panel == panel) & pairs.manual_pair_quality_eligible &
                         pairs.manual_truth_relation.eq('same')]
        for pair in selected.itertuples(index=False):
            partners[pair.left_stem].append(pair.right_stem)
            partners[pair.right_stem].append(pair.left_stem)
        frame['flip'] = [any(frame.at[s,'decision'] != frame.at[t,'decision'] for t in partners[s])
                         if partners[s] else np.nan for s in frame.index]
        frame['slide_error'] = frame.decision.ne(frame.manual_label)
        consistent = frame.groupby('case_id').decision.transform('nunique').eq(1)
        frame['consistent_error'] = consistent & frame.slide_error
        frame['prob_margin'] = -abs(frame.prob_pos - .5)
        frame['logit_margin'] = -abs(frame.logit_1 - frame.logit_0)
        for endpoint in ('flip','slide_error','consistent_error'):
            eligible = frame[frame[endpoint].notna()].copy()
            sizes = eligible.groupby(['identity_union_group','case_id'])[endpoint].transform('size')
            cases = eligible.groupby('identity_union_group').case_id.transform('nunique')
            weights = 1 / (sizes * cases)
            groups, inverse = np.unique(eligible.identity_union_group, return_inverse=True)
            y = eligible[endpoint].astype(int)
            rng = np.random.default_rng(seed)
            # Use the same group multiplicities for both margin scores.
            multiplicities = rng.multinomial(len(groups), np.full(len(groups),1/len(groups)), size=draws)
            for metric in ('prob_margin','logit_margin'):
                score = eligible[metric]
                auc = roc_auc_score(y, score, sample_weight=weights) if y.nunique()==2 else None
                values = []
                if auc is not None:
                    for count in multiplicities:
                        w = weights * count[inverse]
                        if w[y.eq(0)].sum() and w[y.eq(1)].sum():
                            values.append(roc_auc_score(y, score, sample_weight=w))
                ci = np.quantile(values,[.025,.975]) if values else (None,None)
                output.append(dict(model=model,panel=panel,endpoint=endpoint,score=metric,
                                   anchors=len(y),groups=len(groups),events=int(y.sum()),auroc=auc,
                                   ci_low=ci[0],ci_high=ci[1],valid_draws=len(values)))
    return output
