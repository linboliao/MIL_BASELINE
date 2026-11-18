"""Validate and load the two training roles only."""
from pathlib import Path
import torch
from .artifacts import event,read

def memory_available():
    lines = Path('/proc/meminfo').read_text().splitlines()
    return next(int(s.split()[1])*1024 for s in lines if s.startswith('MemAvailable:'))


def tensor_load(path, dim, mmap=False):
    x = torch.load(path, map_location='cpu', weights_only=True, mmap=mmap)
    assert isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[0] > 0 and x.shape[1] == dim, str(path)
    assert x.dtype == torch.float32, (str(path), str(x.dtype), 'precision conversion forbidden')
    return x


def finite(x):
    flat = x.reshape(-1)
    return all(bool(torch.isfinite(flat[i:i+1024**2]).all()) for i in range(0, flat.numel(), 1024**2))

def preload(paths, dim, run, model, phase):
    unique = {r['feature_path']: r for df in paths for r in df.to_dict('records')}
    required = sum(int(r['feature_size_bytes']) for r in unique.values())
    available = memory_available()
    assert available > required + 32*1024**3, ('insufficient RAM for full preload', model, phase, required, available)
    event(run, 'preload_start', model=model, phase=phase, files=len(unique), bytes=required, available_bytes=available)
    data = {}
    for i, (path, r) in enumerate(sorted(unique.items()), 1):
        x = tensor_load(path, dim)
        data[path] = x
        if i % 100 == 0:
            event(run, 'preload_progress', model=model, phase=phase, files=i, total=len(unique))
    event(run, 'preload_complete', model=model, phase=phase, files=len(data), bytes=required)
    return data

def validate(cfg):
    train, selection = (read(cfg[k]) for k in ('train_csv', 'selection_csv'))
    for frame in (train, selection):
        if frame.empty or set(frame.label) != {'0', '1'} or not frame.leakage_group.str.strip().all():
            raise ValueError('Each training role needs case groups and both classes')
        if not frame.feature_path.is_unique:
            raise ValueError('Duplicate feature paths in a training role')
    if set(train.leakage_group) & set(selection.leakage_group):
        raise ValueError('Training and Selection overlap by case group')
    if not set(selection.type) <= {'CNB', 'TURP'}:
        raise ValueError('Selection must be RP-free')
    if ('RP' in set(train.type)) != (cfg['arm'] in ('D', 'B')):
        raise ValueError('Training specimen mix does not match the arm')
    return train, selection
