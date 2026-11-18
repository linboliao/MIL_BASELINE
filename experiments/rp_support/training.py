"""Native A/D/B fixed-update training and checkpoint selection."""
import hashlib,json,math,os,random,socket,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from modules.AB_MIL.ab_mil import AB_MIL
from .artifacts import csv_x,event,json_x,manifest,read,sha,utc,verify_manifest

def seed_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def make_model(run, dim):
    return AB_MIL(in_dim=dim, L=512, D=128, num_classes=2, dropout=.1, act=torch.nn.ReLU()).cuda()


@torch.no_grad()
def predict(model, frame, data):
    model.eval(); records = []
    for r in frame.to_dict('records'):
        logits = model(data[r['feature_path']].unsqueeze(0).cuda())['logits']
        score = float(logits.softmax(1)[0, 1].item())
        loss = float(torch.nn.functional.cross_entropy(logits, torch.tensor([int(r['label'])], device='cuda')).item())
        assert math.isfinite(score) and math.isfinite(loss)
        records.append({k: r[k] for k in ['pool_dir', 'stem', 'canonical_slide_id', 'leakage_group', 'recorded_patient_id', 'type', 'center', 'label']}
                       | {'prob_1': score, 'loss': loss})
    return pd.DataFrame(records)


def cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return obj


def lr_at(step):
    if step <= 1000:
        return .0002*step/1000
    t = (step-1000)/(30000-1000)
    return .00002 + (.0002-.00002)*.5*(1+math.cos(math.pi*t))


def completed_attempt(base):
    for path in sorted(base.glob('attempt_*'), reverse=True):
        if (path/'completion.json').is_file():
            complete = json.loads((path/'completion.json').read_text())
            assert complete['optimizer_updates'] == 30000
            assert sha(path/'artifacts_manifest.csv') == complete['artifacts_manifest_sha256']
            verify_manifest(path, path/'artifacts_manifest.csv')
            return path
    return None


def resume_checkpoint(base, config_hash):
    for attempt in sorted(base.glob('attempt_*'), reverse=True):
        if not (attempt/'execution_meta.json').is_file():
            continue
        meta = json.loads((attempt/'execution_meta.json').read_text())
        assert meta['config_sha256'] == config_hash
        for record in sorted(attempt.glob('validation_step_*.json'), reverse=True):
            r = json.loads(record.read_text()); p = Path(r['checkpoint'])
            if p.is_file() and sha(p) == r['checkpoint_sha256']:
                return p
    return None


def train_job(run, config_path, data, gpu):
    cfg = json.loads(config_path.read_text()); base = Path(cfg['output_base'])
    done = completed_attempt(base)
    if done:
        event(run, 'training_already_complete', job=cfg['job_id'], attempt=str(done)); return
    base.mkdir(parents=True, exist_ok=True)
    parent_checkpoint = resume_checkpoint(base, sha(config_path))
    attempt = base/f'attempt_{len(list(base.glob("attempt_*")))+1:03d}'
    attempt.mkdir(exist_ok=False); (attempt/'checkpoints').mkdir(); (attempt/'selection_predictions').mkdir()
    started = time.time(); log = (attempt/'runtime.log').open('x', buffering=1)
    def emit(kind, **details):
        r = {'event': kind, 'utc': utc(), 'job': cfg['job_id'], 'gpu': gpu, **details}
        line = json.dumps(r, ensure_ascii=False, allow_nan=False); print(line, flush=True); log.write(line+'\n')
    try:
        seed_gpu(gpu); model = make_model(run, cfg['dimension'])
        ih = hashlib.sha256()
        for key, val in model.state_dict().items():
            ih.update(key.encode()); ih.update(val.detach().cpu().numpy().tobytes())
        json_x(attempt/'execution_meta.json', {'job_id': cfg['job_id'], 'config_sha256': sha(config_path),
            'code_sha256': {str(p): sha(p) for p in (
                Path(__file__), Path(__file__).with_name('inputs.py'), Path(__file__).with_name('artifacts.py'),
                Path(__file__).resolve().parents[2] / 'modules/AB_MIL/ab_mil.py')},
            'train_csv_sha256': sha(cfg['train_csv']), 'selection_csv_sha256': sha(cfg['selection_csv']),
            'hostname': socket.gethostname(), 'python': sys.executable, 'torch': torch.__version__,
            'gpu_index': gpu, 'GPU': torch.cuda.get_device_name(0), 'seed': 42, 'initial_state_sha256': ih.hexdigest(),
            'resume_checkpoint': str(parent_checkpoint) if parent_checkpoint else None, 'started_utc': utc()})
        train = read(cfg['train_csv']); val = read(cfg['selection_csv'])
        assert set(train.feature_path) <= set(data) and set(val.feature_path) <= set(data)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0002, weight_decay=.00001, betas=(.9,.999), eps=1e-8)
        generator = torch.Generator().manual_seed(42)
        step = 0; passes = 0; patch_exposure = 0; cursor = 0; order = []
        visits = np.zeros(len(train), dtype=np.int64); history = []; selected = None; loss_sum = 0.
        if parent_checkpoint:
            state = torch.load(parent_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            step = state['step']; passes = state['passes']; patch_exposure = state['patch_exposure']
            cursor = state['cursor']; order = state['order']; visits = state['visits']
            history = state['history']; selected = state['selected']
            generator.set_state(state['generator_state']); random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state(state['cuda_rng']); del state
            emit('resume_full_state', step=step, checkpoint=str(parent_checkpoint))
        emit('training_start', step=step, train_slides=len(train), selection_slides=len(val), preloaded=True)
        while step < 30000:
            if cursor == len(order):
                order = torch.randperm(len(train), generator=generator).tolist(); cursor = 0; passes += 1
            i = order[cursor]; cursor += 1; step += 1
            row = train.iloc[i]; bag = data[row.feature_path]
            optimizer.param_groups[0]['lr'] = lr_at(step)
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(bag.unsqueeze(0).cuda())['logits'], torch.tensor([int(row.label)], device='cuda'))
            assert bool(torch.isfinite(loss)), ('nonfinite loss', step)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.item()); visits[i] += 1; patch_exposure += bag.shape[0]
            if step % 100 == 0:
                emit('progress', step=step, total=30000, seconds=time.time()-started)
            if step % 600:
                continue
            predictions = predict(model, val, data)
            f1 = float(f1_score(predictions.label.astype(int), predictions.prob_1.ge(.5).astype(int), labels=[0,1], average='macro', zero_division=0))
            ce = float(predictions.loss.mean())
            checkpoint = attempt/'checkpoints'/f'step_{step:06d}.pth'
            record = {'step': step, 'val_macro_f1': f1, 'val_loss': ce, 'train_loss_interval': loss_sum/600,
                'lr': lr_at(step), 'passes_started': passes, 'equivalent_data_passes': step/len(train),
                'patch_exposure': int(patch_exposure), 'elapsed_seconds_current_attempt': time.time()-started,
                'checkpoint': str(checkpoint), 'GPU_peak_bytes': torch.cuda.max_memory_allocated()}
            if selected is None or f1 > selected['val_macro_f1']+1e-12 or (abs(f1-selected['val_macro_f1']) <= 1e-12 and ce < selected['val_loss']-1e-12):
                selected = dict(record)
            history.append(dict(record))
            state = {'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                'step': step, 'passes': passes, 'patch_exposure': patch_exposure, 'cursor': cursor, 'order': order,
                'visits': visits, 'history': history, 'selected': selected, 'generator_state': generator.get_state(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'config_sha256': sha(config_path)}
            with checkpoint.open('xb') as f:
                torch.save(state, f)
            del state
            record['checkpoint_sha256'] = sha(checkpoint)
            history[-1] = dict(record)
            if selected['step'] == step:
                selected = dict(record)
            csv_x(attempt/'selection_predictions'/f'step_{step:06d}.csv', predictions)
            json_x(attempt/f'validation_step_{step:06d}.json', record)
            loss_sum = 0.; emit('validation', **record)
        # A restored checkpoint's self hash is deliberately external to the saved state.
        for record in history:
            if 'checkpoint_sha256' not in record:
                record['checkpoint_sha256'] = sha(record['checkpoint'])
        if 'checkpoint_sha256' not in selected:
            selected['checkpoint_sha256'] = sha(selected['checkpoint'])
        assert step == 30000 and len(history) == 50 and int(visits.sum()) == 30000
        exposure = train.copy(); exposure['visits'] = visits
        exposure['patches_per_visit'] = exposure.feature_path.map(lambda s: data[s].shape[0])
        csv_x(attempt/'training_exposure.csv', exposure); csv_x(attempt/'training_history.csv', pd.DataFrame(history))
        json_x(attempt/'checkpoint_selection.json', {'selected': selected, 'final': history[-1], 'calibration_or_test_used': False})
        emit('training_complete', step=step, selected_step=selected['step'], seconds=time.time()-started)
        log.close()
        manifest(attempt, attempt/'artifacts_manifest.csv')
        json_x(attempt/'completion.json', {'status': 'completed', 'optimizer_updates': step, 'validation_checks': 50,
            'artifacts_manifest_sha256': sha(attempt/'artifacts_manifest.csv'), 'finished_utc': utc()})
    except BaseException as error:
        emit('training_failed', error=repr(error), traceback=traceback.format_exc())
        log.close(); json_x(attempt/'failure.json', {'status': 'failed', 'error': repr(error), 'utc': utc(), 'traceback': traceback.format_exc()})
        raise
