import pandas as pd
import math
import os
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from threading import Event


class WSI_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_info_csv_path, group, preload=True):
        assert group in ['train', 'val', 'test'], "group must be in [train, val, test]"
        mem_map = {'train': 150, 'val': 40, 'test': 40}
        self.max_memory = (mem_map[group]) * (1024 ** 3)

        df = pd.read_csv(dataset_info_csv_path)
        self.slide_path_list = df[group + '_slide_path'].dropna().tolist()
        self.labels_list = df[group + '_label'].dropna().tolist()

        self.preloaded_data = [None] * len(self.slide_path_list)
        self.used_memory = 0
        self.stop_signal = Event()

        if preload and self.slide_path_list:
            self.parallel_preload(min(32, os.cpu_count() or 4))

    def load_item(self, idx, check_memory=False):
        """
        核心加载逻辑
        :param check_memory: 为True时受内存上限和熔断机制控制；为False时强制加载（用于__getitem__）
        """
        # 并行模式下，若已熔断则直接返回
        if check_memory and self.stop_signal.is_set():
            return None

        path = self.slide_path_list[idx]
        try:
            # 加载特征
            if path.endswith('.h5'):
                with h5py.File(path, 'r') as f:
                    feat = torch.from_numpy(f['features'][:])
            else:
                feat = torch.load(path)
                if isinstance(feat, dict):
                    feat = feat.get('feats') or feat.get('features')

            feat = feat.squeeze(0) if feat.dim() == 3 else feat

            # 内存检查逻辑
            if check_memory:
                mem = feat.numel() * feat.element_size()
                if self.used_memory + mem >= self.max_memory:
                    self.stop_signal.set()  # 触发熔断
                    return None
                self.used_memory += mem

            return feat, torch.tensor(int(self.labels_list[idx])), Path(path).stem
        except Exception as e:
            print(f"\n[Error] Index {idx} | Path: {path} | {e}")
            return None

    def parallel_preload(self, num_workers):
        """并行预加载：按文件从小到大排序，确保内存利用率最大化"""
        indices = sorted(range(len(self.slide_path_list)),
                         key=lambda i: os.path.getsize(self.slide_path_list[i]) if os.path.exists(self.slide_path_list[i]) else 0)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            # 注意：此处传递 check_memory=True
            futures = {pool.submit(self.load_item, i, True): i for i in indices}
            for f in tqdm(as_completed(futures), total=len(indices), desc="Preloading", ncols=100):
                idx = futures[f]
                res = f.result()
                if res:
                    self.preloaded_data[idx] = res

        loaded_count = sum(1 for x in self.preloaded_data if x is not None)
        print(f"Preload Done! Loaded {loaded_count}/{len(self.slide_path_list)} ({(self.used_memory / 1e9):.2f}GB)")

    def __getitem__(self, idx):
        res = self.preloaded_data[idx]
        if res is not None:
            return res

        res = self.load_item(idx, check_memory=False)

        if res is None:
            raise RuntimeError(f"Failed to load sample: {self.slide_path_list[idx]}")
        return res

    def __len__(self):
        return len(self.slide_path_list)

    def is_None_Dataset(self):
        return self.__len__() == 0

    def is_with_labels(self):
        return len(self.labels_list) > 0

    def get_balanced_sampler(self, replacement=True):
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        counts = Counter(self.labels_list)
        weights = [1.0 / counts[label] for label in self.labels_list]
        return WeightedRandomSampler(weights, len(self.labels_list), replacement=replacement)


class CDP_MIL_WSI_Dataset(WSI_Dataset):
    def __init__(self, dataset_info_csv_path, BeyesGuassian_pt_dir, group):
        super(CDP_MIL_WSI_Dataset, self).__init__(dataset_info_csv_path, group)
        self.slide_path_list = [os.path.join(BeyesGuassian_pt_dir, os.path.basename(slide_path).replace('.pt', '_bayesian_gaussian.pt')) for slide_path in self.slide_path_list]


class LONG_MIL_WSI_Dataset(WSI_Dataset):
    def __init__(self, dataset_info_csv_path, h5_csv_path, group):
        super(LONG_MIL_WSI_Dataset, self).__init__(dataset_info_csv_path, group)
        self.h5_path_list = pd.read_csv(h5_csv_path)['h5_path'].dropna().values

    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        slide_name = os.path.basename(slide_path).replace('.pt', '')
        h5_path = self._find_h5_path_by_slide_name(slide_name, self.h5_path_list)
        print(h5_path)
        h5_file = h5py.File(h5_path, 'r')
        coords = torch.from_numpy(np.array(h5_file['coords']))
        label = int(self.labels_list[idx])
        label = torch.tensor(label)
        feat = torch.load(slide_path)
        if len(feat.shape) == 3:
            feat = feat.squeeze(0)  # (N,D)
        if len(coords.shape) == 3:
            coords = coords.squeeze(0)  # (N,2)
        feat_with_coords = torch.cat([feat, coords], dim=-1)  # (N,D+2)
        return feat_with_coords, label

    def _find_h5_path_by_slide_name(self, slide_name, h5_paths_list):
        h5_dict = {os.path.basename(h5_path).replace('.h5', ''): h5_path for h5_path in h5_paths_list}
        return h5_dict.get(slide_name, None)


class WSI_MM_Dataset(Dataset):
    def __init__(self, dataset_info_csv_path, group):
        assert group in ['train', 'val', 'test']
        df = pd.read_csv(dataset_info_csv_path)
        # 过滤并重置索引
        self.data_df = df[[f'{group}_slide_path', f'{group}_label']].dropna().reset_index(drop=True)
        self.slide_paths = self.data_df[f'{group}_slide_path'].to_list()
        self.labels = self.data_df[f'{group}_label'].astype(int).to_list()

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        he_path = self.slide_paths[idx]
        modalities = ['HE', 'DHR/CD31', 'DHR/CD34', 'DHR/MASSON']
        feat_list = []

        for m in modalities:
            # 动态生成路径并检查是否存在
            m_path = Path(he_path.replace('/HE/', f'/{m}/'))
            if m_path.exists():
                f = torch.load(m_path)
                # 统一形状为 (n, 1024)
                if isinstance(f, torch.Tensor):
                    f = f.squeeze()  # 去掉可能存在的 batch 维度
                    if f.dim() == 1: f = f.unsqueeze(0)  # 确保是 2D
                    feat_list.append(f)

        combined_feat = torch.cat(feat_list, dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return combined_feat, label, Path(self.slide_paths[idx]).stem

    def is_None_Dataset(self):
        return (self.__len__() == 0)

    def is_with_labels(self):
        return (len(self.labels_list) != 0)
