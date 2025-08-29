import os
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import openslide
from PIL import Image
from openslide import lowlevel

sys.path.insert(0, r'../../aslide')
from aslide import Aslide

sys.path.insert(1, r'../../opensdpc/')
from opensdpc.opensdpc import OpenSdpc


def is_background(img, threshold=20):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    diff = np.ptp(img, axis=2)  # ptp直接计算max-min
    return (diff > threshold).mean() < 0.05


class WSIOperator(openslide.OpenSlide):
    def __init__(self, filename):
        if isinstance(filename, str):
            filename = Path(filename)
        suffix = filename.suffix.lower()
        self.suffix = suffix
        if self.suffix == '.kfb':
            slide = Aslide(str(filename))
            self.mpp = slide.mpp
        elif self.suffix == '.sdpc':
            slide = OpenSdpc(str(filename))
            self.mpp = slide.mpp
        else:
            slide = openslide.OpenSlide(str(filename))
            self.mpp = int(slide.properties.get('aperio.AppMag', '20'))
        self.wsi = slide

    def read_region(self, location, level, size, check_background=False):
        """统一区域读取接口"""
        img = self.wsi.read_region(location, level, size)
        if check_background and is_background(img) and random.random() < 0.75:
            return None
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        return img

    @property
    def level_dimensions(self):
        """获取原始尺寸"""
        return self.wsi.level_dimensions

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        return self.wsi.get_thumbnail(size)

    @property
    def level_downsamples(self):
        return self.wsi.level_downsamples
