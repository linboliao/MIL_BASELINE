import os
import shutil

import cv2
import h5py
import staintools
import concurrent.futures
from tqdm import tqdm
from PIL import Image
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 初始化全局参考图像（每个进程独立加载）
REFERENCE_IMG = cv2.imread('TUM-AAALPREY.tif')[:, :, ::-1]  # BGR转RGB

normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(REFERENCE_IMG)  # 每个进程独立拟合参考图像


def process_image(src_folder, dst_folder):
    """单张图像处理函数"""
    for img in os.listdir(src_folder):
        img_path = os.path.join(src_folder, img)
        # 读取并转换颜色空间
        target_img = cv2.imread(img_path)[:, :, ::-1]  # BGR转RGB
        # 执行归一化
        normalized_img = normalizer.transform(target_img)
        # 保存结果
        dst_path = os.path.join(dst_folder, img)
        Image.fromarray(normalized_img).save(dst_path)
        print(f"Success: {img}")


def process_single_image(img_path, dst_folder):
    """处理单张图片（并行任务单元）"""
    try:
        # 读取并转换颜色空间
        target_img = cv2.imread(img_path)[:, :, ::-1]  # BGR转RGB
        if target_img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        # 执行归一化
        normalized_img = normalizer.transform(target_img)

        # 保存结果
        img_name = os.path.basename(img_path)
        dst_path = os.path.join(dst_folder, img_name)
        Image.fromarray(normalized_img).save(dst_path)
        return True, img_name
    except Exception as e:
        return False, f"{img_path} | 错误: {str(e)}"


def parallel_process_images(src_folder, dst_folder, max_workers=5):
    """并行处理图像并显示进度条"""
    # 获取所有图片路径
    img_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]

    # 创建目标目录
    os.makedirs(dst_folder, exist_ok=True)

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_image, img, dst_folder)
                   for img in img_files]

        # 使用tqdm显示进度条
        success_count = 0
        with tqdm(total=len(img_files), desc="处理进度", unit="img") as pbar:
            for future in concurrent.futures.as_completed(futures):
                status, message = future.result()
                if status:
                    pbar.set_postfix_str(f"成功: {message}", refresh=False)
                    success_count += 1
                else:
                    pbar.set_postfix_str(f"失败: {message}", refresh=False)
                pbar.update(1)


# 全局配置
images_dir = '/NAS2/Data1/lbliao/Data/CRC/协和/level1/images_1'
coords_dir = '/NAS2/Data1/lbliao/Data/CRC/协和/level1/patches/patches'
processed_folders = set()  # 记录已处理的文件夹
check_interval = 60  # 全量扫描间隔(秒)


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
            output = folder_path.replace('images_1', 'stains_1')
            os.makedirs(output, exist_ok=True)
            if os.path.isdir(output):
                print(f"跳过 {folder_name}: 文件已处理")
                return
            parallel_process_images(folder_path, output)
            cp_count = sum(1 for _ in os.scandir(output) if _.is_file())
            if cp_count >= coords_length:
                shutil.rmtree(folder_path)
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
            check_single_folder(folder_entry.name, folder_entry.path)
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
    observer.join()


if __name__ == "__main__":
    start_monitoring()
