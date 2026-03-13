import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WSI_Dataset(Dataset):
    def __init__(self, dataset_info_csv_path, group):
        assert group in ['train', 'val', 'test'], 'group must be in [train,val,test]'
        self.dataset_info_csv_path = dataset_info_csv_path
        self.dataset_df = pd.read_csv(self.dataset_info_csv_path)
        self.slide_path_list = self.dataset_df[group + '_slide_path'].dropna().to_list()
        self.labels_list = self.dataset_df[group + '_label'].dropna().to_list()

    def __len__(self):
        return len(self.slide_path_list)

    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        label = int(self.labels_list[idx])
        label = torch.tensor(label)
        feat = torch.load(slide_path)
        if len(feat.shape) == 3:
            feat = feat.squeeze(0)
        return feat, label, Path(slide_path).stem

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


class WSI_Clinical_Dataset(WSI_Dataset):

    def __init__(self, dataset_info_csv_path, group, normalize_clinical=True):
        assert group in ['train', 'val', 'test'], 'group must be in [train,val,test]'

        self.dataset_info_csv_path = dataset_info_csv_path
        self.group = group
        self.dataset_df = pd.read_csv(self.dataset_info_csv_path)

        self.slide_path_list = self.dataset_df[group + '_slide_path'].dropna().to_list()
        self.labels_list = self.dataset_df[group + '_label'].dropna().astype(int).to_list()

        self.clinical_cols = [f'{group}_MLH-1', f'{group}_MSH-2', f'{group}_MSH-6', f'{group}_PMS-2', f'{group}_Age', f'{group}_Sex', f'{group}_family_history', f'{group}_tumor_location']

        self.clinical_features = self.dataset_df[self.clinical_cols].values.astype(np.float32)

        self.clinical_features = np.nan_to_num(self.clinical_features, nan=0.0)

        if normalize_clinical:
            self._normalize_features()

    def _normalize_features(self):
        # Age标准化
        age_idx = 4
        age_data = self.clinical_features[:, age_idx]
        if np.std(age_data) > 1e-8:
            mean = np.mean(age_data)
            std = np.std(age_data)
            self.clinical_features[:, age_idx] = (age_data - mean) / std

        # tumor_location归一化到[0,1]
        tumor_idx = 7
        tumor_data = self.clinical_features[:, tumor_idx]
        max_val = np.max(tumor_data)
        if max_val > 0:
            self.clinical_features[:, tumor_idx] = tumor_data / max_val

    def __len__(self):
        return len(self.slide_path_list)

    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        feat = torch.load(slide_path)
        if len(feat.shape) == 3:
            feat = feat.squeeze(0)

        clinical = torch.tensor(self.clinical_features[idx], dtype=torch.float32)

        label = torch.tensor(self.labels_list[idx], dtype=torch.long)

        return feat, clinical, label

    def is_None_Dataset(self):
        return len(self) == 0
