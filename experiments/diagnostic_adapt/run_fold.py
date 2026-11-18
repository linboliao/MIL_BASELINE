"""Execute one fold: train, lock the action, then score evaluation inputs."""
import collections
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

from .artifacts import put, require, sha, utc, write_csv
from .inputs import FeatureStore, Inputs, ReadAudit
from .scoring import aggregate, batch_forward, rank_auc, score_one


def gradients_finite(net):
    import torch
    checks = [torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None]
    return bool(torch.stack(checks).all().item()) if checks else True


def network(device):
    import torch
    from modules.AB_MIL.ab_mil import AB_MIL
    torch.manual_seed(42)
    return AB_MIL(in_dim=1536, L=512, D=128, num_classes=2, dropout=.1,
                  act=torch.nn.ReLU(), rrt=None).to(device)


class BudgetStop(RuntimeError):
    pass


class Fold:
    def __init__(self, c, policy, slug, out, device, run_started, used_seconds, config_sha, prepath, amendment):
        import torch
        self.c, self.policy, self.slug, self.out = c, policy, slug, out
        self.device, self.run_started, self.used_seconds = device, run_started, used_seconds
        self.config_sha, self.prepath, self.amendment = config_sha, prepath, amendment
        self.audit = ReadAudit(out)
        self.inputs = Inputs(c, policy, slug, self.audit, config_sha)
        self.store = FeatureStore(self.inputs, policy)
        self.log = (out / 'events.jsonl').open('x')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        self.net = network(device)

    def event(self, **values):
        self.log.write(json.dumps(dict(at=utc(), **values), ensure_ascii=False, allow_nan=False) + '\n')
        self.log.flush()

    def boundary(self):
        elapsed = self.used_seconds + time.monotonic() - self.run_started
        if elapsed >= 24 * 3600:
            raise BudgetStop('24 allocated GPU-hour ceiling reached at safe phase boundary')

    def train(self):
        import numpy as np
        import torch
        self.audit.phase = 'train'
        rows = self.inputs.read('source')
        train = [r for r in rows if r['role'] == 'source_design']
        groups = collections.defaultdict(list)
        for r in train:
            require(int(r['source_wsi_label']) in (0, 1), 'invalid source label')
            groups[r['patient_group_id']].append(r)
        patients = sorted(groups)
        for p in patients:
            groups[p].sort(key=lambda r: r['stem'])
        require(bool(patients), 'empty source_design')
        preload = self.store.preload(train)
        put(self.out / 'PRELOAD_COMPLETE.json', dict(**preload, fold=self.slug,
                                                     scientific_config_sha256=self.config_sha, io_amendment=str(self.amendment),
                                                     io_amendment_sha256=sha(self.amendment)))
        self.event(kind='preload_complete', **preload)
        self.boundary()
        rng = np.random.default_rng(self.policy.stable_seed(42, self.slug, 'patient_plan'))
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-4, weight_decay=1e-5,
                               betas=(.9, .999), eps=1e-8, amsgrad=False)
        start = time.monotonic()
        plan_log = (self.out / 'training_plan.jsonl').open('x')
        try:
            self.net.train()
            for step in range(1, 301):
                selected = rng.choice(len(patients), 16, replace=True)
                batch = [groups[patients[int(i)]][int(rng.integers(len(groups[patients[int(i)]])))] for i in selected]
                bags = []
                for draw, r in enumerate(batch):
                    seed = self.policy.stable_seed(42, int(r['io_row_id']), f'mve_train|{step}|{draw}')
                    bags.append(self.store.bag(r, 1200, seed)[0])
                plan_log.write(json.dumps(dict(step=step, stems=[r['stem'] for r in batch])) + '\n')
                plan_log.flush()
                opt.zero_grad(set_to_none=True)
                torch.manual_seed(self.policy.stable_seed(42, self.slug, step, 'dropout'))
                z = batch_forward(self.net, bags, self.device)
                y = torch.tensor([int(r['source_wsi_label']) for r in batch], device=self.device, dtype=torch.float32)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
                require(torch.isfinite(z).all().item() and torch.isfinite(loss).item(), 'nonfinite loss/logits')
                loss.backward()
                require(gradients_finite(self.net), 'nonfinite gradient')
                opt.step()
                self.event(kind='train_update', step=step, loss=float(loss.detach()), seconds=time.monotonic() - start)
                if step % 25 == 0:
                    print(self.slug, 'TRAIN', step, 'seconds', round(time.monotonic() - start, 1), flush=True)
        finally:
            plan_log.close()
            self.store.release_preload()
        with (self.out / 'checkpoint_step300.pt').open('xb') as f:
            torch.save(self.net.state_dict(), f)
        self.checkpoint_hash = sha(self.out / 'checkpoint_step300.pt')
        put(self.out / 'training_complete.json', dict(updates=300, patients=len(patients), source_design_WSI=len(train),
                                                      checkpoint_sha256=self.checkpoint_hash, config_sha256=self.config_sha, seconds=time.monotonic() - start,
                                                      trained_centers=sorted({r['center'] for r in train}), outer_target=self.c['centers'][self.slug],
                                                      target_used_for_training=False, checkpoint_selection='fixed_step300_no_selection'))
        self.boundary()

    def score(self, rows, name, diagnostics):
        scores = []
        start = time.monotonic()
        with (self.out / (name + '_progress.jsonl')).open('x') as progress:
            for i, r in enumerate(sorted(rows, key=lambda r: (r['patient_group_id'], r['stem'])), 1):
                s = score_one(r, diagnostics, store=self.store, policy=self.policy,
                              net=self.net, device=self.device, inputs=self.inputs,
                              audit=self.audit, out_dir=self.out, prepath=self.prepath)
                progress.write(json.dumps(s, ensure_ascii=False, allow_nan=False) + '\n')
                progress.flush()
                scores.append(s)
                self.boundary()
                if i % 25 == 0:
                    print(self.slug, name, i, '/', len(rows), flush=True)
        require(len(scores) == len(rows), 'incomplete WSI prediction coverage')
        write_csv(self.out / (name + '_WSI_scores.csv'), scores)
        self.event(kind='scoring_complete', name=name, WSI=len(rows), seconds=time.monotonic() - start)
        self.boundary()
        patients = aggregate(scores)
        write_csv(self.out / (name + '_patient_scores_NO_LABELS.csv'),
                  [dict(patient_group_id=k, **v) for k, v in sorted(patients.items())])
        if diagnostics:
            import numpy as np
            fields = ['patches', 'eligible_patches', 'foreground_fraction', 'thumbnail_mask_fraction',
                      'thumbnail_S_median', 'thumbnail_V_median', 'original_sampling_jaccard',
                      'foreground_sampling_jaccard', 'physical_input_jaccard']
            summary = {}
            for field in fields:
                values = [r[field] for r in scores if r.get(field) is not None]
                summary[field] = dict(count=len(values), minimum=min(values) if values else None,
                                      median=float(np.median(values)) if values else None,
                                      maximum=max(values) if values else None)
            put(self.out / (name + '_quality_summary.json'), dict(WSI=len(scores), patients=len(patients),
                                                                  WSI_fallback=sum(r['foreground_fallback'] for r in scores),
                                                                  patients_any_fallback=sum(r['fallback_any'] for r in patients.values()), diagnostics=summary,
                                                                  affects_selection=False, score_stability_test=False))
        return patients

    def calibrate_and_select(self):
        self.audit.phase = 'calibration'
        source = [r for r in self.inputs.read('source') if r['role'] == 'source_calibration']
        cal = self.score(source, 'source_calibration', True)
        labels = {}
        for r in source:
            labels[r['patient_group_id']] = max(labels.get(r['patient_group_id'], 0), int(r['source_wsi_label']))
        keys = sorted(cal)
        thresholds = {}
        for input_name in ('original', 'foreground'):
            t = self.policy.threshold([cal[k][input_name] for k in keys], [labels[k] for k in keys], .95)
            require(t is not None, 'no source calibration positive; stop fold')
            thresholds[input_name] = t
        put(self.out / 'source_thresholds.json', dict(thresholds=thresholds, config_sha256=self.config_sha,
                                                      calibration_patients=len(keys), selection_inputs_used=False))
        self.audit.phase = 'budget'
        budget_rows = self.inputs.read('budget_wsi')
        truth = self.inputs.read('budget_labels')
        budget = self.score(budget_rows, 'target_budget', True)
        require(len(truth) == 32 and len({r['patient_group_id'] for r in truth}) == 32, 'wrong budget')
        label_map = {r['patient_group_id']: int(r['patient_label']) for r in truth}
        require(set(budget) == set(label_map), 'incomplete budget patient coverage')
        fit = sorted(r['patient_group_id'] for r in truth if r['role'] == 'target_fit')
        sel = sorted(r['patient_group_id'] for r in truth if r['role'] == 'target_select')
        require(len(fit) == len(sel) == 16 and not set(fit) & set(sel), 'wrong fit/select')
        acts = {'A': dict(input='original', threshold=thresholds['original']),
                'B': dict(input='foreground', threshold=thresholds['foreground'])}
        for action, inp in [('C', 'original'), ('D', 'foreground')]:
            acts[action] = dict(input=inp, threshold=self.policy.threshold([budget[k][inp] for k in fit], [label_map[k] for k in fit]))
        candidates = {a: dict(scores=[budget[k][x['input']] for k in sel], threshold=x['threshold']) for a, x in acts.items()}
        selected = self.policy.select(candidates, [label_map[k] for k in sel])
        selected['evidence_insufficient'] = selected['fallback']
        strong_t = self.policy.threshold([budget[k]['original'] for k in sorted(budget)], [label_map[k] for k in sorted(budget)])
        select_metrics = {}
        for a, x in acts.items():
            zs = candidates[a]['scores']
            ys = [label_map[k] for k in sel]
            select_metrics[a] = dict(available=x['threshold'] is not None, AUROC=rank_auc(zs, ys),
                                     working_point=None if x['threshold'] is None else self.policy.working_point(zs, ys, x['threshold']))
        put(self.out / 'target_select_diagnostics.json', select_metrics)
        lock = dict(complete=True, at=utc(), fold=self.slug, repeat=0, config_sha256=self.config_sha,
                    checkpoint_sha256=self.checkpoint_hash, actions=acts, selection=selected,
                    controls={'ERM_fixed05': dict(input='original', threshold=0.),
                              'target_fit16_simple_calibration': 'alias C',
                              'budget32_simple_calibration': dict(input='original', threshold=strong_t,
                                                                  unavailable=strong_t is None, fallback_action='A' if strong_t is None else None)},
                    input_hashes={k: v['sha256'] for k, v in self.c['inputs'][self.slug].items()},
                    selected_without_evaluation_labels=True, adaptation_label_budget=32)
        path = self.out / 'action_lock.json'
        put(path, lock)
        put(self.out / 'action_lock_sha256.json', dict(sha256=sha(path), at=utc()))
        self.inputs.authorize_evaluation(path, self.checkpoint_hash)
        self.boundary()
        return lock

    def evaluate_unlabelled(self, lock):
        self.audit.phase = 'evaluation_inference'
        rows = self.inputs.read('evaluation_wsi')
        scores = self.score(rows, 'evaluation_NO_LABELS', False)
        out = []
        actions = lock['actions']
        for patient, s in sorted(scores.items()):
            for name in ('A', 'B', 'C', 'D', 'automatic', 'ERM_fixed05', 'target_fit16_simple_calibration', 'budget32_simple_calibration'):
                a = lock['selection']['action'] if name == 'automatic' else 'C' if name == 'target_fit16_simple_calibration' else name
                x = actions[a] if a in actions else lock['controls'][name]
                unavailable = x['threshold'] is None
                effective = actions['A'] if unavailable else x
                z = s[effective['input']]
                out.append(dict(patient_group_id=patient, method=name, logit=z, threshold=effective['threshold'],
                                prediction=int(z >= effective['threshold']), action_unavailable=unavailable,
                                fallback_to_A=unavailable or (name == 'automatic' and lock['selection']['fallback']),
                                foreground_fallback_any=s['fallback_any'], WSI=s['WSI']))
        write_csv(self.out / 'evaluation_patient_predictions_NO_LABELS.csv', out)
        put(self.out / 'feature_witnesses.json', self.store.witnesses)
        put(self.out / 'FOLD_COMPLETE.json', dict(fold=self.slug, at=utc(), evaluation_patients=len(scores),
                                                  prediction_methods=8, action_lock_sha256=self.inputs.lock_hash, config_sha256=self.config_sha,
                                                  target_evaluation_truth_read=False, stage3_metrics_computed=False))

    def close(self):
        self.audit.close()
        self.log.close()


def run_fold(args, *, c, policy, config_sha, prepath, amendment, code_manifest):
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'exactly one visible CUDA GPU required')
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    (out / 'indices').mkdir()
    put(out / 'IO_EXECUTION_AMENDMENT.json', dict(amendment_path=str(amendment),
                                                  amendment_sha256=sha(amendment), source='user requested single-fold preload, serial',
                                                  scientific_config_sha256=config_sha))
    put(out / 'fold_start.json', dict(at=utc(), fold=args.fold, config_sha256=config_sha, host=socket.gethostname(),
                                      python=sys.executable, gpu=os.environ['CUDA_VISIBLE_DEVICES'], code_manifest=code_manifest))
    fold = None
    try:
        fold = Fold(c, policy, args.fold, out, 'cuda:0', time.monotonic(), args.used_seconds,
                    config_sha, prepath, amendment)
        fold.train()
        lock = fold.calibrate_and_select()
        fold.evaluate_unlabelled(lock)
    except BaseException as exc:
        put(out / 'FAILURE.json', dict(at=utc(), type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if fold is not None:
            fold.close()
