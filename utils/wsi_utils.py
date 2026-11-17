"""WSI feature datasets with optional, bounded CPU tensor caching."""
import math
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from .data_loading import dataset_defaults

def _read_feature_file(path):
    coords = None
    if str(path).lower().endswith('.h5'):
        with h5py.File(path, 'r') as f:
            feat = torch.from_numpy(f['features'][:])
            if 'coords' in f:
                coords = torch.from_numpy(f['coords'][:])
    else:
        value = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(value, dict):
            if 'feats' in value:
                feat = value['feats']
            elif 'features' in value:
                feat = value['features']
            else:
                raise ValueError(f"Feature dict has no feats/features key: {path}")
            coords = value.get('coords')
        else:
            feat = value
    feat = torch.as_tensor(feat).detach().cpu()
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat.squeeze(0)
    if feat.ndim != 2 or feat.shape[0] == 0:
        raise ValueError(f"Expected nonempty [patches, features] at {path}, got {tuple(feat.shape)}")
    if coords is not None:
        coords = torch.as_tensor(coords).detach().cpu()
        if coords.ndim == 3 and coords.shape[0] == 1:
            coords = coords.squeeze(0)
    return feat, coords

def _append_coords(feat, coords, path):
    if coords is None:
        return feat
    if coords.ndim != 2 or coords.shape[0] != feat.shape[0] or coords.shape[1] < 2:
        raise ValueError(f"Feature/coordinate mismatch: {path}")
    return torch.cat((feat, coords[:, :2].to(dtype=feat.dtype)), dim=-1)

class WSI_Dataset(Dataset):

    def __init__(self, dataset_info_csv_path, group, preload=None, mem_map=None,
                 preload_mem_gb=None, preload_workers=None, return_metadata=False):
        if group not in ('train', 'val', 'test'):
            raise ValueError(f"Unknown dataset group: {group}")
        self.dataset_info_csv_path = str(dataset_info_csv_path)
        self.dataset_df = pd.read_csv(dataset_info_csv_path)
        pairs = self.dataset_df[[group+'_slide_path', group+'_label']]
        if pairs.isna().any(axis=1).ne(pairs.isna().all(axis=1)).any():
            raise ValueError(f"Unpaired path/label values in {group}: {dataset_info_csv_path}")
        valid = pairs.notna().all(axis=1)
        pairs = pairs.loc[valid]
        labels = pairs.iloc[:, 1]
        if not np.isfinite(labels.astype(float)).all() or not np.equal(labels.astype(float), labels.astype(int)).all():
            raise ValueError("Class-index labels must be finite integers")
        self.slide_path_list = pairs.iloc[:, 0].astype(str).tolist()
        self.labels_list = labels.astype(int).tolist()
        id_col = group+'_sample_id'
        if id_col in self.dataset_df:
            identifiers = self.dataset_df.loc[valid, id_col]
            if identifiers.isna().any():
                raise ValueError(f"Missing {id_col}")
            self.sample_id_list = identifiers.astype(str).tolist()
        else:
            self.sample_id_list = list(self.slide_path_list)
        if len(set(self.sample_id_list)) != len(self.sample_id_list):
            raise ValueError(f"Duplicate sample IDs/paths in {group}; provide unique {id_col}")
        opts = dataset_defaults()
        enabled = opts.get('preload', False) if preload is None else preload
        if isinstance(enabled, dict):
            enabled = enabled.get(group, False)
        if not isinstance(enabled, bool):
            raise ValueError("preload must be a boolean or per-split boolean mapping")
        self.preload_enabled = enabled
        budget = preload_mem_gb if preload_mem_gb is not None else mem_map
        if budget is None:
            budget = opts.get('preload_mem_gb', 1.)
        if isinstance(budget, dict):
            budget = budget.get(group, 0.)
        if isinstance(budget, bool) or not math.isfinite(float(budget)) or float(budget) < 0:
            raise ValueError("preload_mem_gb must be finite and nonnegative")
        self.max_memory = int(float(budget)*1024**3)
        workers = opts.get('preload_workers', 1) if preload_workers is None else preload_workers
        if isinstance(workers, bool) or int(workers) != workers or not 1 <= int(workers) <= 4:
            raise ValueError("preload_workers must be an integer in [1, 4]")
        self.preload_workers = int(workers)
        self.return_metadata = bool(return_metadata)
        self._cache = {}
        self._cache_prepared = False
        self.used_memory = self.cache_hits = self.cache_misses = 0

    def __len__(self):
        return len(self.slide_path_list)

    def _load_features(self, idx):
        return _read_feature_file(self.slide_path_list[idx])[0]

    def _cache_one(self, idx, feat):
        size = feat.numel()*feat.element_size()
        if idx not in self._cache and self.used_memory+size <= self.max_memory:
            # Clone so a tensor view cannot retain a much larger storage.
            self._cache[idx] = feat.clone()
            self.used_memory += size

    def _ensure_cache(self):
        if self._cache_prepared or not self.preload_enabled:
            return

        if not self.max_memory:
            self._cache_prepared = True
            return
        indices = sorted(range(len(self)), key=lambda i: os.path.getsize(self.slide_path_list[i]))
        # Limit outstanding reads to one batch of workers. Budget governs retained
        # tensor bytes, not total RSS; in-flight reads and cloning need headroom.
        with ThreadPoolExecutor(max_workers=self.preload_workers) as pool:
            for start in range(0, len(indices), self.preload_workers):
                if self.used_memory >= self.max_memory:
                    break
                chunk = indices[start:start+self.preload_workers]
                for idx, feat in zip(chunk, pool.map(self._load_features, chunk)):
                    self._cache_one(idx, feat)
        self._cache_prepared = True
        print(f"{self.__class__.__name__}: cached {len(self._cache)}/{len(self)} samples, "
              f"{self.used_memory/1024**3:.3f}/{self.max_memory/1024**3:.3f} GiB retained tensors")
        if len(self._cache) < len(self):
            warnings.warn("Partial feature cache: uncached samples are loaded from disk on demand")

    def __getitem__(self, idx):
        self._ensure_cache()
        if idx in self._cache:
            self.cache_hits += 1
            feat = self._cache[idx]
        else:
            self.cache_misses += 1
            feat = self._load_features(idx)
            if self.preload_enabled:
                self._cache_one(idx, feat)
        result = (feat, torch.tensor(self.labels_list[idx], dtype=torch.long))
        return result+(self.sample_id_list[idx],) if self.return_metadata else result

    def cache_info(self):
        return {'cached_samples': len(self._cache), 'total_samples': len(self),
                'tensor_bytes': self.used_memory, 'budget_bytes': self.max_memory,
                'hits': self.cache_hits, 'misses': self.cache_misses,
                'complete': self._cache_prepared and len(self._cache) == len(self)}

    def is_None_Dataset(self):
        return len(self) == 0

    def is_with_labels(self):
        return bool(self.labels_list)

    def get_balanced_sampler(self, replacement=True):
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        counts = Counter(self.labels_list)
        return WeightedRandomSampler([1/counts[y] for y in self.labels_list],
                                     len(self), replacement=replacement)

class WSI_Coord_Dataset(WSI_Dataset):
    def _load_features(self, idx):
        feat, coords = _read_feature_file(self.slide_path_list[idx])
        return _append_coords(feat, coords, self.slide_path_list[idx])

class CDP_MIL_WSI_Dataset(WSI_Dataset):
    def __init__(self, dataset_info_csv_path, BeyesGuassian_pt_dir, group, **kwargs):
        super().__init__(dataset_info_csv_path, group, **kwargs)
        self.slide_path_list = [os.path.join(BeyesGuassian_pt_dir,
            Path(p).stem+'_bayesian_gaussian.pt') for p in self.slide_path_list]

class LONG_MIL_WSI_Dataset(WSI_Dataset):
    def __init__(self, dataset_info_csv_path, h5_csv_path, group, **kwargs):
        super().__init__(dataset_info_csv_path, group, **kwargs)
        self.h5_path_list = pd.read_csv(h5_csv_path)['h5_path'].dropna().tolist()
        self._h5_paths = {Path(p).stem:p for p in self.h5_path_list}

    def _find_h5_path_by_slide_name(self, slide_name, h5_paths_list):
        return self._h5_paths.get(slide_name)

    def _load_features(self, idx):
        path = self.slide_path_list[idx]
        feat, _ = _read_feature_file(path)
        h5_path = self._h5_paths.get(Path(path).stem)
        if h5_path is None:
            raise FileNotFoundError(f"No coordinates for {path}")
        with h5py.File(h5_path, 'r') as f:
            coords = torch.from_numpy(f['coords'][:])
        if coords.ndim == 3 and coords.shape[0] == 1:
            coords = coords.squeeze(0)
        return _append_coords(feat, coords, path)

class SC_MIL_WSI_Dataset(LONG_MIL_WSI_Dataset):
    def __init__(self, dataset_info_csv_path, h5_csv_path=None, group='train',
                 use_dummy_coords=True, **kwargs):
        WSI_Dataset.__init__(self, dataset_info_csv_path, group, **kwargs)
        self._h5_paths = {}
        if h5_csv_path is not None and os.path.exists(h5_csv_path):
            self.h5_path_list = pd.read_csv(h5_csv_path)['h5_path'].dropna().tolist()
            self._h5_paths = {Path(p).stem:p for p in self.h5_path_list}
        self.use_dummy_coords = not bool(self._h5_paths)
        if self.use_dummy_coords:
            warnings.warn("SC_MIL uses pseudo-grid coordinates; these are not measured spatial coordinates")

    def _generate_dummy_coords(self, num_patches):
        grid = int(np.ceil(np.sqrt(num_patches)))
        idx = np.arange(num_patches)
        return np.stack((idx % grid, idx // grid), axis=1).astype(np.float32)

    def _load_features(self, idx):
        path = self.slide_path_list[idx]
        if Path(path).stem in self._h5_paths:
            return super()._load_features(idx)
        feat, _ = _read_feature_file(path)
        return _append_coords(feat, torch.from_numpy(self._generate_dummy_coords(len(feat))), path)
