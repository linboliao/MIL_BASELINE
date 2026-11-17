"""Scoped dataset configuration and cache-aware DataLoader construction."""
from contextlib import contextmanager
from contextvars import ContextVar
import warnings
from torch.utils.data import DataLoader as TorchDataLoader
from .repro_utils import deterministic_requested, seed_worker
_OPTIONS = ContextVar('mil_dataset_options', default={})

@contextmanager
def dataset_options(options):
    token = _OPTIONS.set(dict(options))
    try:
        yield
    finally:
        _OPTIONS.reset(token)

def dataset_defaults():
    return _OPTIONS.get()

def DataLoader(dataset, *args, **kwargs):
    # Default fork and spawn both become predictable with no cache replicas.
    # Users can disable preload if they need multi-process disk loading.
    if getattr(dataset, 'preload_enabled', False):
        if kwargs.get('num_workers', 0):
            warnings.warn("Preloaded dataset: using num_workers=0 to avoid cache replication")
        kwargs['num_workers'] = 0
        kwargs.pop('prefetch_factor', None)
        kwargs['persistent_workers'] = False
    if deterministic_requested():
        kwargs.setdefault('worker_init_fn', seed_worker)
    return TorchDataLoader(dataset, *args, **kwargs)
