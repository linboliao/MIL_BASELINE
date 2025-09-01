import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
from sklearn.model_selection import train_test_split

tumor_path = '/NAS3/lbliao/Data/CRC/协和/cls/geojson/tumor'
normal_path = '/NAS3/lbliao/Data/CRC/协和/cls/geojson/normal'
tumor_list = [os.path.splitext(p)[0] for p in os.listdir(tumor_path)]
normal_list = [os.path.splitext(p)[0] for p in os.listdir(normal_path)]


# 1. 自定义Dataset类
class CustomTensorDataset(Dataset):
    def __init__(self, file_paths, labels, preload=False, transform=None):
        """
        优化后的自定义数据集类
        :param file_paths: .pt文件路径列表
        :param labels: 对应的标签列表
        :param preload: 是否在初始化时将所有数据预加载到内存
        :param transform: 可选的数据增强/变换函数
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.preload = preload

        # 如果启用预加载，将所有数据加载到内存中
        if self.preload:
            self.data = []
            for path in file_paths:
                tensor = torch.load(path)
                self.data.append(tensor.view(-1))  # 展平后存储

        # 创建内存映射文件缓存（可选，用于极大数据集）
        self.memory_mapped = False
        # 示例：可以在这里实现内存映射逻辑

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.preload:
            # 直接从内存中获取数据
            tensor = self.data[idx]
        else:
            # 原始加载方式（仍可优化）
            tensor = torch.load(self.file_paths[idx])
            tensor = tensor.view(-1)  # 展平

        label = self.labels[idx]

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label


# 2. 准备数据：获取文件路径和标签
def prepare_data(data_dir):
    """
    从指定目录读取所有.pt文件，并根据文件名分配标签
    :param data_dir: 包含.pt文件的目录
    :return: 文件路径列表和标签列表
    """
    file_paths = []
    labels = []
    # 查找所有.pt文件
    pt_files = glob.glob(os.path.join(data_dir, '*.pt'))

    for file_path in pt_files:
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
            # 如果前缀既不在tumor_list也不在normal_list，可以跳过或视为某种默认类别
            # 这里我们选择跳过该文件
            continue

        file_paths.append(file_path)
        labels.append(label)

    return file_paths, labels


# 3. 定义模型
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], dropout_rate=0.3):
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
        # 如果选择输出维度为2，配合CrossEntropyLoss，则注释上一行，取消下一行注释
        # layers.append(nn.Linear(prev_dim, 2))

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
            data, target = data.to(device), target.to(device).float().unsqueeze(1)  # 调整target形状以匹配输出

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        val_acc = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    return train_losses, val_accuracies


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 将输出通过Sigmoid函数并四舍五入到0或1
            predicted = torch.round(torch.sigmoid(output))
            # 如果使用CrossEntropyLoss且输出维度为2，则使用下面这行获取预测结果
            # _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted.squeeze() == target).sum().item()  # 调整形状匹配
    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数设置
    data_dir = '/NAS3/lbliao/Data/CRC/协和/cls/feat/train'  # 请替换为您的.pt文件存放目录
    batch_size = 512
    num_epochs = 50
    learning_rate = 0.001

    # 准备数据
    file_paths, labels = prepare_data(data_dir)
    print(f"Total samples: {len(file_paths)}")

    # 划分训练集和验证集
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 创建数据集和数据加载器
    train_dataset = CustomTensorDataset(train_files, train_labels, preload=True)
    val_dataset = CustomTensorDataset(val_files, val_labels, preload=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8 if torch.cuda.is_available() else 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8 if torch.cuda.is_available() else 0)

    # 初始化模型、损失函数和优化器
    model = BinaryClassifier().to(device)
    # 使用BCEWithLogitsLoss（结合了Sigmoid和BCELoss，数值稳定性更好）
    criterion = nn.BCEWithLogitsLoss()
    # 如果使用CrossEntropyLoss且输出维度为2，则使用下面这行，并将target转换为long tensor且不需要unsqueeze
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # 加载最佳模型并在验证集上做最终评估
    model.load_state_dict(torch.load('best_model.pth'))
    final_val_acc = evaluate_model(model, val_loader, device)
    print(f'Final Validation Accuracy of the best model: {final_val_acc:.2f}%')
