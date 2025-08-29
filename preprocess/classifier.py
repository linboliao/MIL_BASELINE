import os
import shutil
import time
import h5py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 解除图像尺寸限制
Image.MAX_IMAGE_PIXELS = None

# 多卡配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 根据实际GPU数量调整
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 常量设置
WEIGHT_PATH = 'ckpts/resnet50_100k_best.pth'
THRESHOLD_CLASS = 8
BATCH_SIZE = 1024  # 根据GPU显存调整
nc = 9  # 分类类别数
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')  # 支持的图片格式

# 全局配置
images_dir = '/NAS2/Data1/lbliao/Data/CRC/协和/level1/stains'
coords_dir = '/NAS2/Data1/lbliao/Data/CRC/协和/level1/patches/patches'
processed_folders = set()  # 记录已处理的文件夹
check_interval = 60  # 全量扫描间隔(秒)

# 加载模型
print("初始化模型...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, nc)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model = model.to(device)
model.eval()
print("模型加载完成")

# 预处理流程（与训练一致）
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 确保图像尺寸统一
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7406, 0.5331, 0.7059],
                         std=[0.1279, 0.1606, 0.1191])
])


def process_image(src_folder):
    """批量处理文件夹中的图片，分类结果不为8则移动到目标文件夹"""
    # 获取所有支持的图片文件
    image_files = [
        f for f in os.listdir(src_folder)
        if os.path.isfile(os.path.join(src_folder, f))
           and f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not image_files:
        print(f"在文件夹 {src_folder} 中未找到图片文件")
        return 0

    print(f"找到 {len(image_files)} 张图片，开始处理...")
    moved_count = 0

    # 创建进度条
    pbar = tqdm(total=len(image_files), desc="处理进度", unit="img")

    # 批量处理
    for i in range(0, len(image_files), BATCH_SIZE):
        batch_files = image_files[i:i + BATCH_SIZE]
        batch_images = []
        batch_paths = []

        # 加载当前批次图片
        for file in batch_files:
            src_path = os.path.join(src_folder, file)
            try:
                img = Image.open(src_path).convert('RGB')
                batch_images.append(img)
                batch_paths.append(src_path)
            except Exception as e:
                print(f"无法加载图片 {file}: {e}")

        if not batch_images:
            pbar.update(len(batch_files))
            continue

        # 预处理并推理
        with torch.no_grad(), torch.cuda.amp.autocast():
            input_batch = torch.stack([preprocess(img) for img in batch_images])
            input_batch = input_batch.half().to(device)
            outputs = model(input_batch)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        # 处理分类结果
        for j, pred in enumerate(predictions):
            if pred != THRESHOLD_CLASS:
                src_path = batch_paths[j]

                try:
                    os.remove(src_path)
                    moved_count += 1
                except Exception as e:
                    print(f"删除文件失败: {src_path} : {str(e)}")

        pbar.update(len(batch_images))

    pbar.close()
    return moved_count


class FolderEventHandler(FileSystemEventHandler):
    """文件夹事件处理器"""

    def on_created(self, event):
        if not event.is_directory:
            return
        folder_path = event.src_path
        folder_name = os.path.basename(folder_path)
        print(f"检测到新文件夹: {folder_name}")
        check_single_folder(folder_name, folder_path)

    def on_modified(self, event):
        if not event.is_directory:
            return
        folder_path = event.src_path
        folder_name = os.path.basename(folder_path)
        print(f"文件夹更新: {folder_name}")
        check_single_folder(folder_name, folder_path)


def check_single_folder(folder_name, folder_path):
    """检查单个文件夹是否满足条件"""
    if folder_name in processed_folders:
        return

    h5_file = os.path.join(coords_dir, f"{folder_name}.h5")
    if not os.path.isfile(h5_file):
        print(f"跳过 {folder_name}: 未找到.h5文件")
        return

    try:
        with h5py.File(h5_file, 'r') as f:
            coords_length = f['coords'].shape[0]

        file_count = sum(1 for _ in os.scandir(folder_path) if _.is_file())

        if file_count >= coords_length:
            print(f"条件满足: {folder_name} (文件:{file_count} >= 坐标:{coords_length})")

            # 处理图片并获取移动数量
            moved_count = process_image(folder_path)

            # 检查处理结果
            remaining_files = sum(1 for _ in os.scandir(folder_path) if _.is_file())
            print(f"处理完成: 移动 {moved_count} 文件 | 剩余 {remaining_files} 文件")

            # 安全删除源文件夹
            if remaining_files == 0:
                try:
                    shutil.rmtree(folder_path)
                    print(f"已清空并删除文件夹: {folder_path}")
                except Exception as e:
                    print(f"删除文件夹失败: {folder_path}: {str(e)}")
            processed_folders.add(folder_name)
        else:
            print(f"待处理: {folder_name} (文件:{file_count} < 坐标:{coords_length})")

    except Exception as e:
        print(f"处理 {folder_name} 出错: {str(e)}")


def full_scan():
    """全量扫描所有文件夹"""
    print("\n" + "=" * 50)
    print("执行全量文件夹扫描...")
    for folder_entry in os.scandir(images_dir):
        if folder_entry.is_dir():
            check_single_folder(folder_entry.name, folder_path=folder_entry.path)
    print("扫描完成\n" + "=" * 50)


def start_monitoring():
    """启动监控系统"""
    # 初始化监控器
    event_handler = FolderEventHandler()
    observer = Observer()
    observer.schedule(event_handler, images_dir, recursive=False)
    observer.start()
    print(f"启动文件夹监控: {images_dir}")

    try:
        # 初始全量扫描
        full_scan()

        # 定时全量扫描（应对可能遗漏的事件）
        while True:
            time.sleep(check_interval)
            full_scan()

    except KeyboardInterrupt:
        observer.stop()
        print("监控已停止")
    finally:
        observer.join()
        print("监控器已关闭")


if __name__ == "__main__":
    start_monitoring()
