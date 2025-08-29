"""
全切片图像（WSI）肿瘤区域检测与可视化模块

该模块使用预训练的ResNet50模型对WSI切片的小块图像进行分类，
标记出肿瘤区域并生成可视化掩码图像。
"""
import argparse
import os

import numpy as np
import staintools
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch import nn
from tqdm import tqdm

from preprocess.wsi import WSIOperator

# 防止图像过大导致处理问题
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 设置GPU环境
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 常量定义
WEIGHT_PATH = "ckpts/resnet50_100k_best.pth"
THRESHOLD_CLASS = 8
BATCH_SIZE = 128
NUM_CLASSES = 9
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
PATCH_SIZE = 224
WSI_LEVEL = 1
MASK_LEVEL = 4  # 缩略图层级


class StainNormalizer:
    """
    染色归一化处理器

    使用Macenko方法对病理图像进行染色归一化处理，
    确保不同切片间的染色一致性[3](@ref)。

    Attributes:
        normalizer (staintools.StainNormalizer): 染色归一化器实例
    """

    def __init__(self):
        """初始化染色归一化器并加载参考图像"""
        print("初始化染色归一化参考图像...")
        ref_img_path = "TUM-AAALPREY.tif"

        try:
            reference_image = staintools.read_image(ref_img_path)
            reference_image = staintools.LuminosityStandardizer.standardize(reference_image)
            normalizer = staintools.StainNormalizer(method="macenko")
            normalizer.fit(reference_image)
            self.normalizer = normalizer
            print("染色归一化参考图像已加载")
        except Exception as error:
            print(f"参考图像加载失败: {error}")
            raise

    def __call__(self, image):
        """
        对输入图像进行染色归一化处理

        Args:
            image (PIL.Image): 输入图像

        Returns:
            PIL.Image: 归一化后的图像
        """
        image_array = np.array(image)
        try:
            image_array = self.normalizer.transform(image_array)
        except Exception as error:
            print(f"染色归一化失败: {error}")
        return Image.fromarray(image_array)


def initialize_model():
    """
    初始化并加载预训练模型

    Returns:
        torch.nn.Module: 加载好权重的模型
    """
    print("初始化模型...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("模型加载完成")
    return model


# 初始化模型
MODEL = initialize_model()

# 预处理流程
PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        StainNormalizer(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.7406, 0.5331, 0.7059], std=[0.1279, 0.1606, 0.1191]
        ),
    ]
)


def process_batch(batch_files, source_folder, mask, mask_factor):
    """
    处理一批图像文件并进行预测

    Args:
        batch_files (list): 批处理文件列表
        source_folder (str): 源文件夹路径
        mask (numpy.ndarray): 掩码图像数组
        mask_factor (float): 掩码层级缩放因子
    """
    batch_images = []
    batch_paths = []
    batch_coordinates = []

    for file in batch_files:
        source_path = os.path.join(source_folder, file)
        try:
            image = Image.open(source_path).convert("RGB")
            batch_images.append(image)
            batch_paths.append(source_path)

            # 从文件名解析坐标（假设格式为x_y_patch.png）
            coordinates = file[:-12].split("_")
            x_coord, y_coord = int(coordinates[0]), int(coordinates[1])
            batch_coordinates.append((x_coord, y_coord))
        except Exception as error:
            print(f"处理文件 {file} 失败: {error}")
            batch_coordinates.append((None, None))

    if not batch_images:
        return

    # 模型推理
    with torch.no_grad(), torch.cuda.amp.autocast():
        input_batch = torch.stack([PREPROCESS(img) for img in batch_images])
        input_batch = input_batch.to(DEVICE)
        outputs = MODEL(input_batch)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # 处理结果
    for index, prediction in enumerate(predictions):
        if prediction == THRESHOLD_CLASS:
            x_coord, y_coord = batch_coordinates[index]
            if x_coord is None or y_coord is None:
                continue

            # 坐标转换到标记层级
            x_pos = int(x_coord / mask_factor / 2)
            y_pos = int(y_coord / mask_factor / 2)
            patch_size = max(1, int(PATCH_SIZE / mask_factor))  # 确保至少1像素

            # 边界检查
            y_end = min(y_pos + patch_size, mask.shape[0])
            x_end = min(x_pos + patch_size, mask.shape[1])

            if y_pos < y_end and x_pos < x_end:
                # 创建红色半透明层
                red_layer = np.zeros((y_end - y_pos, x_end - x_pos, 4), dtype=np.uint8)
                red_layer[..., 0] = 255  # R通道
                red_layer[..., 3] = 128  # Alpha通道

                # Alpha混合
                area = mask[y_pos:y_end, x_pos:x_end]
                blend_ratio = red_layer[..., 3:] / 255.0
                mask[y_pos:y_end, x_pos:x_end, :3] = (
                        red_layer[..., :3] * blend_ratio + area[..., :3] * (1 - blend_ratio)
                )


def create_tumor_mask(wsi_path, source_folder, destination_folder):
    """
    创建肿瘤区域可视化掩码

    Args:
        wsi_path (str): WSI文件路径
        source_folder (str): 源图像文件夹路径
        destination_folder (str): 结果保存文件夹路径
    """
    wsi_operator = WSIOperator(wsi_path)
    mask_factor = wsi_operator.level_downsamples[MASK_LEVEL] // wsi_operator.level_downsamples[WSI_LEVEL]
    width, height = wsi_operator.level_dimensions[MASK_LEVEL]
    mask = np.array(wsi_operator.read_region((0, 0), MASK_LEVEL, (width, height)))

    os.makedirs(destination_folder, exist_ok=True)

    # 获取支持的图像文件
    image_files = [
        file
        for file in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, file))
           and file.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not image_files:
        print(f"在文件夹 {source_folder} 中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")
    progress_bar = tqdm(total=len(image_files), desc="处理进度", unit="img")

    # 批量处理图像
    for i in range(0, len(image_files), BATCH_SIZE):
        batch = image_files[i: i + BATCH_SIZE]
        process_batch(batch, source_folder, mask, mask_factor)
        progress_bar.update(len(batch))

    # 保存标记图
    base_name = os.path.splitext(os.path.basename(wsi_path))[0]
    mask_image = Image.fromarray(mask)
    mask_image.save(os.path.join(destination_folder, f"{base_name}_tumor_mask.png"))

    progress_bar.close()
    print(f"处理完成！标记图已保存至 {destination_folder}")


parser = argparse.ArgumentParser(description='癌区ROI可视化')
parser.add_argument('--wsi_path', type=str, default='')
parser.add_argument('--image_dir', type=str, default='')
parser.add_argument('--mask_dir', type=str, default='')
if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.mask_dir, exist_ok=True)
    create_tumor_mask(args.wsi_path, args.image_dir, args.mask_dir)
