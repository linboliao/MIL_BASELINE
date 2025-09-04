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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tumor_path = '/NAS2/Data1/lbliao/Data/CRC/协和/cls/geojson/tumor'
normal_path = '/NAS2/Data1/lbliao/Data/CRC/协和/cls/geojson/normal'
tumor_list = [os.path.splitext(p)[0] for p in os.listdir(tumor_path)]
normal_list = [os.path.splitext(p)[0] for p in os.listdir(normal_path)]


class TensorDataset(Dataset):
    def __init__(self, file_paths, labels, preload=False, transform=None, max_memory_gb=50, num_workers=80):
        """
        优化后的自定义数据集类，支持预加载与内存控制
        :param file_paths: .pt文件路径列表
        :param labels: 对应的标签列表
        :param preload: 是否在初始化时进行预加载
        :param transform: 可选的数据增强/变换函数（应用时机：__getitem__）
        :param max_memory_gb: 预加载时的最大内存限制 (GB)
        :param num_workers: 预加载时使用的并行工作线程数
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.preload = preload

        # 预加载相关属性
        self.preloaded_data = []
        self.total_loaded_size = 0
        self.max_memory_bytes = max_memory_gb * 1024 ** 3  # 转换为字节
        self.stop_preload = False  # 停止预加载标志
        self.preload_lock = threading.Lock()  # 内存统计锁
        self.num_workers = num_workers or min(32, (os.cpu_count() or 4))  # 工作线程数

        # 如果启用预加载，则进行并行预加载
        if self.preload:
            self._parallel_preload()
        else:
            # 如果不预加载，初始化为空列表，后续在__getitem__中加载
            self.preloaded_data = [None] * len(self.file_paths)

    def _parallel_preload(self):
        """并行预加载数据，并遵守内存上限"""
        print(f"开始并行预加载数据（内存上限: {self.max_memory_bytes / (1024 ** 3):.1f} GB）...")
        self.preloaded_data = [None] * len(self.file_paths)  # 初始化列表，保持原始索引

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 按原始顺序提交任务（不再按文件大小排序）
            future_to_index = {
                executor.submit(self._load_single_item, idx): idx
                for idx in range(len(self.file_paths))
            }

            # 使用tqdm创建进度条
            for future in tqdm(
                    concurrent.futures.as_completed(future_to_index),
                    total=len(future_to_index),
                    desc="预加载进度"
            ):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        tensor, label, computed_size = result
                        # 安全地更新内存使用情况并存储数据
                        with self.preload_lock:
                            if not self.stop_preload and self.total_loaded_size + computed_size <= self.max_memory_bytes:
                                self.preloaded_data[idx] = (tensor, label)
                                self.total_loaded_size += computed_size
                            else:
                                # 超出内存限制，停止后续加载
                                self.stop_preload = True
                                # 可以选择取消剩余任务
                                # for f in future_to_index:
                                #     f.cancel()
                except Exception as e:
                    os.remove(self.file_paths[idx])
                    print(f"加载索引 {idx} 的文件 {self.file_paths[idx]} 时出错: {e}")

        success_count = sum(1 for item in self.preloaded_data if item is not None)
        print(f"预加载完成! 成功加载 {success_count}/{len(self.file_paths)} 个样本")
        print(f"总内存占用: {self.total_loaded_size / (1024 ** 3):.2f} GB")

    def _load_single_item(self, idx):
        """加载单个数据项并计算其内存占用"""
        if self.stop_preload:
            return None

        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # 加载张量
            tensor = torch.load(file_path)
            tensor = tensor.view(-1)  # 展平

            # 计算这个样本将占用的内存大小（张量数据 + 标签张量）
            tensor_size = tensor.numel() * tensor.element_size()
            label_size = 4  # 假设标签是整数，约占4字节
            total_size = tensor_size + label_size

            return tensor, label, total_size

        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            return None

    def __getitem__(self, idx):
        """
        获取数据项
        如果已预加载，则直接从内存返回；否则，从磁盘加载。
        """
        if self.preload:
            # 如果预加载了，直接返回内存中的数据
            tensor, label = self.preloaded_data[idx]
        else:
            # 否则，从磁盘加载
            load_result = self._load_single_item(idx)
            if load_result is not None:
                tensor, label, _ = load_result
            else:
                # 处理加载失败，例如返回一个空张量或抛出异常
                raise RuntimeError(f"无法加载索引 {idx} 的数据")

        # 应用变换（如果有）
        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label

    def __len__(self):
        return len(self.file_paths)


def process_single_file(file_path, tumor_list, normal_list):
    """
    处理单个文件并返回文件路径和标签
    :param file_path: .pt文件路径
    :param tumor_list: 肿瘤前缀列表
    :param normal_list: 正常组织前缀列表
    :return: (file_path, label) 或 None（如果文件不符合要求）
    """
    # 获取文件名（不含路径和扩展名）
    filename = os.path.splitext(os.path.basename(file_path))[0]
    # 提取第一个下划线前的部分
    prefix = filename.split('_')[0]

    # 根据前缀分配标签
    if prefix in tumor_list:
        label = 1
    elif prefix in normal_list:
        label = 0
    else:
        # 如果前缀既不在tumor_list也不在normal_list，返回None
        return None

    return (file_path, label)


def prepare_data(data_dir, max_workers=None):
    """
    从指定目录并行读取所有.pt文件，并根据文件名分配标签
    :param data_dir: 包含.pt文件的目录
    :param tumor_list: 肿瘤前缀列表
    :param normal_list: 正常组织前缀列表
    :param max_workers: 最大线程数，默认为None（让ThreadPoolExecutor自动选择）
    :return: 文件路径列表和标签列表
    """
    # 查找所有.pt文件
    pt_files = glob.glob(os.path.join(data_dir, '*.pt'))

    file_paths = []
    labels = []
    tumor_count, normal_count = 0, 0

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_file, file_path, tumor_list, normal_list): file_path
            for file_path in pt_files
        }

        # 使用tqdm显示进度条
        for future in tqdm(as_completed(future_to_file), total=len(pt_files), desc="Processing files"):
            result = future.result()
            if result is not None:
                file_path, label = result
                file_paths.append(file_path)
                labels.append(label)

                if label == 1:
                    tumor_count += 1
                else:
                    normal_count += 1

    print(f'Tumor count: {tumor_count}; Normal count: {normal_count}')
    return file_paths, labels


# 3. 定义模型
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], dropout_rate=0.1):
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
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
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
            torch.save(model.state_dict(), 'best_model.pth')
    torch.save(model.state_dict(), 'last_model.pth')

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
    threshold = obtain_optimal_threshold(all_targets, all_probs)
    _auc = roc_auc_score(all_targets, all_probs)

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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = '/NAS2/Data1/lbliao/Data/CRC/协和/cls/feats/train'
    val_data = '/NAS2/Data1/lbliao/Data/CRC/协和/cls/feats/val'
    test_data = '/NAS2/Data1/lbliao/Data/CRC/协和/cls/feats/test'
    batch_size = 1024
    num_epochs = 50
    learning_rate = 0.001

    # train_files, train_labels = prepare_data(train_data)
    # print(f"Total train samples: {len(train_files)}")
    # val_files, val_labels = prepare_data(val_data)
    # print(f"Total val samples: {len(val_files)}")
    test_files, test_labels = prepare_data(test_data)
    print(f"Total test samples: {len(test_files)}")

    # train_dataset = TensorDataset(train_files, train_labels, preload=True)
    # val_dataset = TensorDataset(val_files, val_labels, preload=True)
    test_dataset = TensorDataset(test_files, test_labels, preload=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8 if torch.cuda.is_available() else 0)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8 if torch.cuda.is_available() else 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8 if torch.cuda.is_available() else 0)

    # 初始化模型、损失函数和优化器
    model = BinaryClassifier().to(device)
    # 使用BCEWithLogitsLoss（结合了Sigmoid和BCELoss，数值稳定性更好）
    criterion = nn.BCELoss()
    # 如果使用CrossEntropyLoss且输出维度为2，则使用下面这行，并将target转换为long tensor且不需要 unsqueeze
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    # train_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # 加载最佳模型并在验证集上做最终评估
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, test_loader, device)
    # final_val_acc = evaluate_model_with_optimal_threshold(model, test_loader, device)
