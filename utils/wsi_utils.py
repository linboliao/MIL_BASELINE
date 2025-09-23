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


class WSI_Dataset(Dataset):
    def __init__(self, dataset_info_csv_path, group, preload=True, num_workers=None, max_memory_gb=100):
        assert group in ['train', 'val', 'test'], 'group must be in [train,val,test]'

        self.dataset_info_csv_path = dataset_info_csv_path
        self.dataset_df = pd.read_csv(self.dataset_info_csv_path)
        self.slide_path_list = self.dataset_df[group + '_slide_path'].dropna().to_list()
        self.labels_list = self.dataset_df[group + '_label'].dropna().to_list()
        self.preloaded = False
        self.preloaded_data = []  # 存储预加载数据
        self.max_memory_bytes = max_memory_gb * 1024 ** 3  # 50GB上限
        self.total_loaded_size = 0  # 已加载内存统计
        self.stop_preload = False  # 停止预加载标志
        self.preload_lock = threading.Lock()  # 内存统计锁

        # 按文件大小排序（优先加载小文件）
        self._sort_by_file_size()

        self.num_workers = num_workers or min(32, (os.cpu_count() or 4))

        if preload and not self.is_None_Dataset():
            self._parallel_preload()

    def _sort_by_file_size(self):
        """按文件大小升序排序，优先加载小文件以最大化利用内存"""
        sizes_and_indices = []
        for idx, path in enumerate(self.slide_path_list):
            try:
                size = os.path.getsize(path)
                sizes_and_indices.append((size, idx))
            except Exception:
                sizes_and_indices.append((0, idx))

        # 按文件大小升序排序
        sizes_and_indices.sort(key=lambda x: x[0])
        self.sorted_indices = [idx for _, idx in sizes_and_indices]

    def _parallel_preload(self):
        """并行预加载（含内存上限控制）"""
        print(f"开始并行预加载（上限: {self.max_memory_bytes / (1024 ** 3):.1f}GB）...")

        # 初始化预加载数组（保持原始索引位置）
        self.preloaded_data = [None] * len(self.slide_path_list)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 按排序顺序提交任务
            future_to_idx = {}
            for idx in self.sorted_indices:
                if self.stop_preload:
                    break  # 提前终止
                future = executor.submit(self._load_single_item, idx)
                future_to_idx[future] = idx

            for future in tqdm(concurrent.futures.as_completed(future_to_idx),
                               total=len(future_to_idx),
                               desc="预加载进度"):
                idx = future_to_idx[future]
                try:
                    item = future.result()
                    if item is not None:
                        self._update_memory_usage(idx, item)
                except Exception as e:
                    print(f"加载索引 {idx} 失败: {e}")

        self.preloaded = True
        success_count = sum(1 for item in self.preloaded_data if item is not None)
        print(f"预加载完成! 成功加载 {success_count}/{len(self.slide_path_list)} 个样本")
        print(f"内存占用: {self.total_loaded_size / (1024 ** 3):.2f}GB")

    def _load_single_item(self, idx):
        """带内存检查的单个样本加载"""
        # 检查全局停止标志
        if self.stop_preload:
            return None

        slide_path = self.slide_path_list[idx]
        label = int(self.labels_list[idx])

        try:
            file_size = os.path.getsize(slide_path)
            with self.preload_lock:
                if self.total_loaded_size + file_size > self.max_memory_bytes:
                    return None

            # 实际加载数据
            feat = torch.load(slide_path)
            if len(feat.shape) == 3:
                feat = feat.squeeze(0)

            label_tensor = torch.tensor(label)
            return feat, label_tensor, Path(slide_path).stem

        except Exception as e:
            print(f"文件加载失败: {slide_path}, {e}")
            return None

    def _update_memory_usage(self, idx, item):
        """更新内存统计并存入预加载数组"""
        feat, label_tensor, _ = item
        sample_size = feat.numel() * feat.element_size() + label_tensor.numel() * label_tensor.element_size()

        with self.preload_lock:
            if self.total_loaded_size + sample_size <= self.max_memory_bytes:
                self.preloaded_data[idx] = item
                self.total_loaded_size += sample_size
            else:
                self.stop_preload = True  # 触发全局停止

    def __getitem__(self, idx):
        if self.preloaded and self.preloaded_data[idx] is not None:
            return self.preloaded_data[idx]
        else:
            return self._load_single_item(idx)

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
