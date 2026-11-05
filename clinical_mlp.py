import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # 确保卷积操作也是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置Python哈希种子
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子已设置为: {seed}")


# 配置参数
class Config:
    seed = 44
    batch_size = 32
    learning_rate = 0.001
    epochs = 50
    dropout = 0.3
    clinical_dim = 8


# 设置全局随机种子
set_seed(Config.seed)


class ClinicalDataset(Dataset):
    def __init__(self, features, labels, slide_ids=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.slide_ids = slide_ids if slide_ids is not None else []

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.slide_ids:
            return self.features[idx], self.labels[idx], self.slide_ids[idx]
        return self.features[idx], self.labels[idx]


class AugmentedDataset(Dataset):
    def __init__(self, features, labels, slide_ids=None, noise_level=0.01, seed=Config.seed):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.slide_ids = slide_ids if slide_ids is not None else []
        self.noise_level = noise_level
        # 为数据增强设置随机种子
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        # 添加轻微噪声，使用固定的随机状态
        if self.rng.random() > 0.5:
            # 使用固定的numpy随机状态生成噪声
            noise = torch.tensor(self.rng.randn(*feature.shape) * self.noise_level, dtype=torch.float32)
            feature = feature + noise

        if self.slide_ids:
            return feature, self.labels[idx], self.slide_ids[idx]
        return feature, self.labels[idx]


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ClinicalClassifier(nn.Module):
    def __init__(self, clinical_dim, dropout=0.3):
        super(ClinicalClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),

            nn.Linear(64, 32),
            nn.ReLU(),  # 最后一层不加dropout
        )

        self.classifier = nn.Linear(32, 2)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        features = self.network(x)
        return self.classifier(features)


def load_fold_data_with_ids(fold_num):
    """加载指定折数的数据，包含slide_id"""
    df = pd.read_csv(f'datasets/LS/clinical/dataset_002/5fold/Total_5-fold_5fold_{fold_num}fold.csv')

    # 检查是否存在slide_id列
    slide_id_col = None
    for col in df.columns:
        if 'slide_id' in col.lower() or 'id' in col.lower():
            slide_id_col = col
            break

    if slide_id_col is None:
        # 如果没有slide_id列，创建一个基于索引的ID
        slide_id_col = 'Index'
        df[slide_id_col] = [f'ID_{i:04d}' for i in range(len(df))]
        print(f"警告: 未找到slide_id列，已自动创建索引ID列")

    # 提取特征列
    feature_cols = ['MLH-1', 'MSH-2', 'MSH-6', 'PMS-2', 'Age', 'Sex', 'family_history', 'tumor_location']

    def extract_data(prefix):
        features = []
        slide_ids = []
        labels_list = []

        for i in range(len(df)):
            # 检查是否是该分割的数据
            label_col = f'{prefix}label'
            if pd.isna(df[label_col].iloc[i]):
                continue

            # 提取特征
            sample = []
            valid_sample = True
            for col in feature_cols:
                feature_value = df[f'{prefix}{col}'].iloc[i]
                if pd.isna(feature_value):
                    valid_sample = False
                    break
                sample.append(feature_value)

            if valid_sample and len(sample) == len(feature_cols):
                features.append(sample)
                slide_ids.append(df[f'{prefix}slide_id'].iloc[i])
                labels_list.append(df[label_col].iloc[i])

        return np.array(features), np.array(labels_list), slide_ids

    train_features, train_labels, train_ids = extract_data('train_')
    val_features, val_labels, val_ids = extract_data('val_')
    test_features, test_labels, test_ids = extract_data('test_')

    print(f"训练集: {len(train_ids)}个样本, 验证集: {len(val_ids)}个样本, 测试集: {len(test_ids)}个样本")

    # 标准化
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    return (train_features, train_labels, train_ids), (val_features, val_labels, val_ids), (test_features, test_labels, test_ids)


def train_and_eval_fold(fold_num):
    """训练并评估单折，使用macro_f1挑选最佳模型"""
    print(f"\n=== 训练第 {fold_num} 折 ===")

    fold_seed = Config.seed
    set_seed(fold_seed)

    (train_f, train_l, train_slide_ids), (val_f, val_l, val_slide_ids), (test_f, test_l, test_slide_ids) = load_fold_data_with_ids(fold_num)

    # 创建数据加载器
    train_loader = DataLoader(
        AugmentedDataset(train_f, train_l, train_slide_ids, seed=fold_seed),
        batch_size=Config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(fold_seed)  # DataLoader的随机种子
    )
    val_loader = DataLoader(
        ClinicalDataset(val_f, val_l, val_slide_ids),
        batch_size=Config.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        ClinicalDataset(test_f, test_l, test_slide_ids),
        batch_size=Config.batch_size,
        shuffle=False
    )

    # 初始化模型
    model = ClinicalClassifier(Config.clinical_dim, Config.dropout)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    best_macro_auc = 0
    best_model_state = None
    best_epoch = 0
    best_val_results = None

    for epoch in range(Config.epochs):
        # 训练阶段
        model.train()
        for batch_idx, (x, y, _) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # 验证阶段 - 使用macro_auc评估
        model.eval()
        val_preds, val_probs, val_targets, val_slide_ids_list = [], [], [], []
        with torch.no_grad():
            for x, y, slide_ids in val_loader:
                outputs = model(x)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_probs.extend(outputs[:, 1].cpu().numpy())
                val_targets.extend(y.cpu().numpy())
                val_slide_ids_list.extend(slide_ids)

        # 计算macro_auc
        val_macro_auc = roc_auc_score(val_targets, val_preds)

        # 保存最佳模型（基于macro_auc）
        if val_macro_auc > best_macro_auc:
            best_macro_auc = val_macro_auc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            # 保存验证集结果用于调试
            best_val_results = {
                'slide_ids': val_slide_ids_list,
                'labels': val_targets,
                'probs': val_probs,
                'predictions': val_preds
            }

    print(f"最佳模型在第 {best_epoch} 轮，验证集 macro_auc: {best_macro_auc:.4f}")

    # 使用最佳模型进行测试
    model.load_state_dict(best_model_state)
    model.eval()

    test_preds, test_probs, test_targets, test_slide_ids_list = [], [], [], []
    with torch.no_grad():
        for x, y, slide_ids in test_loader:
            outputs = model(x)
            test_preds.extend(outputs.argmax(1).cpu().numpy())
            test_probs.extend(outputs[:, 1].cpu().numpy())
            test_targets.extend(y.cpu().numpy())
            test_slide_ids_list.extend(slide_ids)

    # 保存测试结果到CSV
    test_results_df = pd.DataFrame({
        'slide_id': test_slide_ids_list,
        'label': test_targets,
        'probs': [f'[{1 - p} {p}]' for p in test_probs],  # 类别0的概率
        'prediction': test_preds
    })

    # 添加预测概率
    test_results_df['probability'] = test_probs

    # 保存到文件
    test_results_file = os.path.join(f'results/test_results_fold{fold_num}_seed{Config.seed}.csv')
    test_results_df.to_csv(test_results_file, index=False)
    print(f"测试结果已保存到: {test_results_file}")

    # 计算测试集指标
    acc = accuracy_score(test_targets, test_preds)
    try:
        auc = roc_auc_score(test_targets, test_probs)
    except:
        auc = 0.5
    precision = precision_score(test_targets, test_preds, average='binary', zero_division=0)
    recall = recall_score(test_targets, test_preds, average='binary', zero_division=0)
    macro_f1 = f1_score(test_targets, test_preds, average='macro')
    kappa = cohen_kappa_score(test_targets, test_preds)

    print(f"第 {fold_num} 折测试结果: precision={precision:.4f}, recall={recall:.4f} ACC={acc:.4f}, AUC={auc:.4f}, Macro-F1={macro_f1:.4f}, Kappa={kappa:.4f}")

    return precision, recall, acc, auc, macro_f1, kappa


if __name__ == "__main__":
    # 设置主随机种子
    set_seed(Config.seed)

    all_metrics = []
    fold_results = []

    for fold in range(1, 6):  # 1到5折
        metrics = train_and_eval_fold(fold)
        all_metrics.append(metrics)
        fold_results.append({
            'fold': fold,
            'precision': metrics[0],
            'recall': metrics[1],
            'acc': metrics[2],
            'auc': metrics[3],
            'macro_f1': metrics[4],
            'kappa': metrics[5]
        })

    # 计算平均指标
    all_metrics = np.array(all_metrics)
    mean_pre = np.mean(all_metrics[:, 0])
    mean_recall = np.mean(all_metrics[:, 1])
    mean_acc = np.mean(all_metrics[:, 2])
    mean_auc = np.mean(all_metrics[:, 3])
    mean_macro_f1 = np.mean(all_metrics[:, 4])
    mean_kappa = np.mean(all_metrics[:, 5])

    std_pre = np.std(all_metrics[:, 0])
    std_rec = np.std(all_metrics[:, 1])
    std_acc = np.std(all_metrics[:, 2])
    std_auc = np.std(all_metrics[:, 3])
    std_macro_f1 = np.std(all_metrics[:, 4])
    std_kappa = np.std(all_metrics[:, 5])

    # 打印最终结果
    print("\n" + "=" * 50)
    print(f"随机种子: {Config.seed}")
    print("五折交叉验证平均结果:")
    print("=" * 50)
    print(f"precision (ACC):  {mean_pre:.4f} ± {std_pre:.4f}")
    print(f"recall (ACC):  {mean_recall:.4f} ± {std_rec:.4f}")
    print(f"准确率 (ACC):  {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"AUC:          {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Macro-F1:     {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(f"Kappa:        {mean_kappa:.4f} ± {std_kappa:.4f}")
    print("=" * 50)

    # 打印每折详细结果
    # print("\n各折详细结果:")
    # for i, (acc, auc, macro_f1, kappa) in enumerate(all_metrics, 1):
    #     print(f"折{i}: ACC={acc:.4f}, AUC={auc:.4f}, Macro-F1={macro_f1:.4f}, Kappa={kappa:.4f}")

    # 保存结果到文件
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f'results_seed_{Config.seed}.csv', index=False)
    print(f"\n详细结果已保存到: results_seed_{Config.seed}.csv")
