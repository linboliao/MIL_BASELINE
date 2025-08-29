from torchvision import datasets
import concurrent.futures

from tqdm import tqdm


class ImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, preload=True):
        """
        初始化数据集
        :param preload: 是否预加载所有数据到内存
        """
        super().__init__(root, transform, target_transform, loader)
        self.preloaded_images = []  # 存储预加载的图像数据
        self.preloaded = False

        if preload:
            self._preload_data()

    def _preload_data(self):
        """预加载所有图像数据到内存"""
        self.preloaded_images = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交所有加载任务
            futures = {executor.submit(self.loader, path): (path, label) for path, label in self.samples}
            # 异步获取结果并排序
            for future in tqdm(concurrent.futures.as_completed(futures), desc="预加载进度"):
                path, label = futures[future]
                try:
                    image = future.result()
                    self.preloaded_images.append((image, label))
                except Exception as e:
                    print(f"加载失败 {path}: {e}")
        self.preloaded = True
        print(f"预加载完成！共加载 {len(self.preloaded_images)} 张图像")

    def __getitem__(self, index):
        """
        从内存中获取数据项，应用预处理
        """
        if self.preloaded:
            # 从预加载数据中直接获取[3](@ref)
            image, label = self.preloaded_images[index]
        else:
            # 回退到原始磁盘加载方式
            path, label = self.samples[index]
            image = self.loader(path)

        # 应用预处理变换[1,7](@ref)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.samples)
