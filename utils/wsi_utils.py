import os
import numpy as np
import pandas as pd
import h5py
import torch
import concurrent.futures
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

import threading

import os
import pandas as pd
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class WSI_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_info_csv_path, group, preload=True, num_workers=None, max_memory_gb=50):
        assert group in ['train', 'val', 'test']

        df = pd.read_csv(dataset_info_csv_path)
        self.slide_path_list = df[group + '_slide_path'].dropna().tolist()
        self.labels_list = df[group + '_label'].dropna().tolist()
        self.preloaded_data = [None] * len(self.slide_path_list)
        self.max_memory = max_memory_gb * (1024 ** 3)
        self.used_memory = 0

        if preload and self.slide_path_list:
            self._preload(num_workers or min(32, os.cpu_count() or 4))

    def _preload(self, num_workers):
        """并行预加载数据"""
        print(f"开始预加载数据 (内存上限: {self.max_memory / (1024 ** 3):.1f}GB)...")

        sorted_indices = self._sorted_indices()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._load_item, i): i for i in sorted_indices}

            for future in tqdm(as_completed(futures), total=len(futures), ncols=100, desc="预加载进度"):
                idx = futures[future]
                data = future.result()

                if data and self.used_memory < self.max_memory:
                    self.preloaded_data[idx] = data
                    self.used_memory += data[0].numel() * data[0].element_size()

        loaded_count = sum(1 for item in self.preloaded_data if item is not None)
        print(f"预加载完成! 成功加载 {loaded_count}/{len(self.slide_path_list)} 个样本")
        print(f"内存占用: {self.used_memory / (1024 ** 3):.2f}GB")

    def _sorted_indices(self):
        """返回按文件大小排序的索引"""
        sizes = [(os.path.getsize(p) if os.path.exists(p) else 0, i)
                 for i, p in enumerate(self.slide_path_list)]
        return [i for _, i in sorted(sizes, key=lambda x: x[0])]

    def _load_item(self, idx):
        """加载单个样本"""
        try:
            feat = torch.load(self.slide_path_list[idx])
            label = torch.tensor(int(self.labels_list[idx]))
            if feat.dim() == 3:
                feat = feat.squeeze(0)
            return (feat, label, Path(self.slide_path_list[idx]).stem)
        except Exception as e:
            return None

    def __getitem__(self, idx):
        if self.preloaded_data[idx] is not None:
            return self.preloaded_data[idx]
        return self._load_item(idx) or (torch.zeros(1), torch.tensor(-1), '')

    def __len__(self):
        return len(self.slide_path_list)

    def is_None_Dataset(self):
        return (self.__len__() == 0)

    def is_with_labels(self):
        return (len(self.labels_list) != 0)


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
        modalities = ['HE', 'CD31', 'CD34', 'MASSON']
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

