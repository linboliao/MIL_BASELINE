import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from preprocess.dataset import ImageDataset

# 配置日志文件（追加模式，包含时间、级别和消息）
logging.basicConfig(
    filename='NCT-CRC-HE-100K.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
# 1. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms.Normalize(mean=[0.7406, 0.5331, 0.7059], std=[0.1279, 0.1606, 0.1191])
])

train_folder = '/NAS2/Data4/llb/Data/cls/NCT-CRC-HE_含数据集介绍/NCT-CRC-HE-100K'
nc = 0

# 统计类别数量
with os.scandir(train_folder) as entries:
    for entry in entries:
        if entry.is_dir():
            nc += 1

print(f'number of training classes: {nc}')
test_folder = '/NAS2/Data4/llb/Data/cls/NCT-CRC-HE_含数据集介绍/CRC-VAL-HE-7K'
train_dataset = ImageDataset(train_folder, transform=transform)
print("类别列表:", train_dataset.classes)
print("类别到索引的字典:", train_dataset.class_to_idx)
print("首张图像路径及标签:", train_dataset.imgs[0])
test_dataset = ImageDataset(test_folder, transform=transform)

# 3. 计算划分比例 (70%训练, 15%验证, 15%测试)
total_size = len(train_dataset)
train_size = int(0.7 * total_size)
val_size = total_size - train_size
test_size = total_size - train_size - val_size  # 防止因舍入误差导致样本缺失

# 4. 随机划分数据集
train_dataset, val_dataset = random_split(
    train_dataset,
    [train_size, val_size]
)
# 5. 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=64 * torch.cuda.device_count(),
    shuffle=True,  # 仅训练集打乱
    num_workers=4,
    pin_memory=True  # 加速GPU传输
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32 * torch.cuda.device_count(),
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64 * torch.cuda.device_count(),
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

nc = 0

# 统计类别数量
with os.scandir(train_folder) as entries:
    for entry in entries:
        if entry.is_dir():
            nc += 1
print(f'number of training classes: {nc}')
# 4. 模型初始化
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, nc)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6])  # 显式指定GPU
model = model.to(device)

total_epoch = 100


# 定义测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    all_labels = []  # 存储所有真实标签
    all_probs = []  # 存储所有预测概率

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing', ncols=100, colour='blue')
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 获取预测结果和概率
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)  # 转换为概率分布[3,5](@ref)

            # 收集数据用于AUC计算
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())  # 存储每个类别的概率[6](@ref)

            # 计算准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_bar.set_postfix(acc=f"{100 * correct / total:.2f}%")

    # 计算平均损失和准确率
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    # 计算多分类AUC（One-vs-Rest策略）
    auc_score = roc_auc_score(
        np.array(all_labels),
        np.array(all_probs),
        multi_class='ovr',  # 多分类采用One-vs-Rest策略
        average='macro'  # 对各类别的AUC取平均
    )

    logger.info(f'Val Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | AUC: {auc_score:.4f}')
    return accuracy, auc_score  # 同时返回准确率和AUC


def val():
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():  # 禁用梯度计算
        val_bar = tqdm(val_loader, desc='Validating', ncols=100, colour='blue')
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 实时更新测试进度
            tmp_loss = val_loss / (val_bar.n + 1e-8)
            val_bar.set_postfix(acc=f"{100 * correct / total:.2f}%", loss=f"{tmp_loss:.4f}")

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    logger.info(f'val Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')
    return accuracy


# 5. 训练函数
def train(epoch):
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epoch}', ncols=100, colour='green')

    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (train_bar.n + 1e-8)
        train_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

    return total_loss / len(train_loader)


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 正样本权重平衡因子
        self.gamma = gamma  # 难样本聚焦参数
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: 模型原始输出 [batch_size, num_classes]
        targets: 类别索引 [batch_size]（值范围：0 ~ num_classes-1）
        """
        # 计算交叉熵损失（未加权）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # shape=[192]

        # 计算当前预测概率 p_t
        pt = torch.exp(-ce_loss)  # p_t = e^(-CE_loss)

        # 应用Focal系数：alpha * (1-p_t)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # 按reduction要求聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# 6. 主训练循环
# criterion = nn.CrossEntropyLoss()
criterion = MultiClassFocalLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

best_acc = 0.0  # 记录最佳测试准确率

for epoch in range(total_epoch):
    epoch_loss = train(epoch)
    scheduler.step()

    logger.info(f'Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}')

    current_acc = test()

    # 保存最佳模型
    if current_acc[0] > best_acc:
        best_acc = current_acc[0]
        torch.save(model.state_dict(), 'resnet50_100k_best.pth')
        print(f'Best model saved with accuracy: {best_acc:.2f}%')

# 保存最终模型
torch.save(model.state_dict(), 'resnet50_100k_final.pth')
test()
