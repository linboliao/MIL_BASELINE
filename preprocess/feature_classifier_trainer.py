import argparse
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
import os
import glob

from tqdm import tqdm


class PTDataset(Dataset):
    def __init__(self, root_dir, preload, max_workers=None, transform=None):
        """
        初始化数据集并并行预加载所有数据到内存

        Args:
            root_dir: 根目录路径，子文件夹名为标签名
            max_workers: 并行加载的线程数
            transform: 数据变换函数
        """
        self.root_dir = root_dir
        self.preload = preload
        self.transform = transform
        self.max_workers = max_workers if max_workers is not None else os.cpu_count()

        self.file_paths = []
        self.labels = []
        self._setup_data()

        self.preloaded_data = [None] * len(self.file_paths)
        if self.preload:
            self._parallel_preload()

    def _setup_data(self):
        """遍历目录结构，收集.pt文件路径和标签"""
        label_dict = {'NORM': 0, 'TUM': 1}
        tum_count = 0
        norm_count = 0
        for label_name in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label_name)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    if file_name.endswith('.pt'):
                        file_path = os.path.join(label_dir, file_name)
                        label = label_dict[label_name]

                        # if not label and norm_count > tum_count:
                        #     continue
                        tum_count += 1 if label else 0
                        norm_count += 1 if not label else 0
                        self.file_paths.append(file_path)
                        self.labels.append(label_dict[label_name])
        print(f'TUM nums: {tum_count}, NORM nums: {norm_count}')

    def _load_single_item(self, idx: int):
        """加载单个数据项"""
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            tensor = torch.load(file_path)
            tensor = tensor.view(-1)  # 展平张量

            if self.transform:
                tensor = self.transform(tensor)

            return tensor, label
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            return None

    def _parallel_preload(self):
        """使用多线程并行预加载所有数据，并显示进度条"""
        print(f"开始并行预加载 {len(self.file_paths)} 个数据项...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有加载任务
            future_to_idx = {
                executor.submit(self._load_single_item, idx): idx
                for idx in range(len(self.file_paths))
            }

            # 使用tqdm显示进度
            with tqdm(total=len(self.file_paths), desc="加载数据") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if result is not None:
                            self.preloaded_data[idx] = result
                    except Exception as e:
                        print(f"加载索引 {idx} 的数据时发生错误: {e}")
                    finally:
                        pbar.update(1)

        print("数据预加载完成!")

    def __getitem__(self, idx: int):
        """从内存缓存中获取数据项"""
        if self.preload:
            return self.preloaded_data[idx]
        else:
            return self._load_single_item(idx)

    def __len__(self) -> int:
        return len(self.file_paths)


# 3. 定义模型
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=1536, hidden_dims=[512, 256], dropout_rate=0.1):
        super(BinaryClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # 输出层：二分类，输出维度为1（使用Sigmoid激活）或2（使用Softmax）
        # 这里选择输出维度为1，配合BCEWithLogitsLoss或BCELoss
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        # 如果选择输出维度为2，配合CrossEntropyLoss，则注释上一行，取消下一行注释
        # layers.append(nn.Linear(prev_dim, 2))
        # layers.append(nn.Sigmoid(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 4. 训练和评估函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, feature_model):
    model.train()
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            # data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if batch_idx % 100 == 0:
            #     print('Epoch: %d, batch id: %5d loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        val_acc = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.8f}, Val Accuracy: {val_acc:.8f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'nct_best-{feature_model}.pth')
    torch.save(model.state_dict(), f'nct_last-{feature_model}.pth')

    return train_losses, val_accuracies


def evaluate_model(model, data_loader, device):
    model.eval()

    all_probs = []  # 存储预测概率
    all_targets = []  # 存储真实标签

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            probs = output.squeeze()

            # probs = torch.sigmoid(output).squeeze()
            # predicted = (probs >= 0.0034).float()  # 应用阈值进行二分类判断
            # 如果使用CrossEntropyLoss且输出维度为2，则使用下面这行获取预测结果
            # _, predicted = torch.max(output.data, 1)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    _auc = roc_auc_score(all_targets, all_probs)
    # threshold = obtain_optimal_threshold(all_targets, all_probs)
    threshold = 0.5

    predictions = (all_probs >= threshold).astype(np.int64)  # 确保类型一致
    accuracy = 100.0 * accuracy_score(all_targets, predictions)

    print(f'ACC: {accuracy:.4f}%')
    print(f"AUC值: {_auc:.4f}")
    return accuracy


def obtain_optimal_threshold(targets, probs):
    fpr, tpr, thresholds = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    # 使用约登指数(Youden's J statistic)寻找最佳阈值 J = tpr - fpr，寻找使J最大的阈值
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]

    predictions = (probs >= optimal_threshold).astype(int)
    accuracy = 100.0 * accuracy_score(targets, predictions)

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label=f'Optimal threshold (={optimal_threshold:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"最佳阈值: {optimal_threshold:.4f}")
    print(f"在此阈值下的准确率: {accuracy:.2f}%")
    print(f"AUC值: {roc_auc:.4f}")

    return optimal_threshold


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, help='path to MIL-yaml file')
parser.add_argument('--feature_model', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--features_dim', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = f'{args.data_root}/{args.feature_model}/train'
    val_data = f'{args.data_root}/{args.feature_model}/val'
    test_data = f'{args.data_root}/{args.feature_model}/test'

    train_dataset = PTDataset(train_data, preload=True)
    val_dataset = PTDataset(val_data, preload=True)
    test_dataset = PTDataset(test_data, preload=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8 if torch.cuda.is_available() else 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8 if torch.cuda.is_available() else 0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8 if torch.cuda.is_available() else 0)

    # 初始化模型、损失函数和优化器
    model = BinaryClassifier(args.features_dim).to(device)
    # 使用BCEWithLogitsLoss（结合了Sigmoid和BCELoss，数值稳定性更好）
    criterion = nn.BCELoss()
    # 如果使用CrossEntropyLoss且输出维度为2，则使用下面这行，并将target转换为long tensor且不需要 unsqueeze
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练模型
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, args.feature_model)

    # 加载最佳模型并在验证集上做最终评估
    model.load_state_dict(torch.load(f'nct_best-{args.feature_model}.pth'))
    evaluate_model(model, test_loader, device)
