import numpy as np
import random
from random import sample
from PIL import Image, ImageEnhance, ImageOps
#Resolution
color_resolution = {'x1': 256, 'x8': 128, 'x64': 64, 'x512': 32}
color_resolution_v2 = {'x1': 256, 'x2': 128, 'x4': 64, 'x8': 32}

class ChannelSplit():
    def __init__(self, res, skip, prob):
        self.res = res
        self.skip = skip
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            img = self._color_global(img, color_resolution[self.res], choice=1, skip=self.skip)
        return img
    def _color_global(self, image, resolution=128, choice=2, skip=False):
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        _skip = 0
        if resolution == 256:
            _skip = -2
        result = []
        for r in range(int(255 / resolution) + 1):
            f_r = np.multiply(image[0], (resolution * r <= image[0]) & ((resolution * (r + 1) - 1) >= image[0]))
            for g in range(int(255 / resolution) + 1):
                f_g = np.multiply(image[1], (resolution * g <= image[1]) & ((resolution * (g + 1) - 1) >= image[1]))
                for b in range(int(255 / resolution) + 1):
                    _skip += 1
                    if skip and (_skip == 1 or _skip == (int(255 / resolution) + 1) ** 3):
                        continue
                    f_b = np.multiply(image[2], (resolution * b <= image[2]) & ((resolution * (b + 1) - 1) >= image[2]))
                    result.append(np.stack((f_r, f_g, f_b)))
        result = np.array(sample(result, choice), dtype=np.uint8)
        result = np.transpose(np.squeeze(result, axis=0), (1, 2, 0))
        result = Image.fromarray(result)
        return result

class ChannelSplit2():
    def __init__(self, res, choice, skip, prob):
        self.res = res
        self.choice = choice
        self.skip = skip
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            img = self._color_global(img, color_resolution_v2[self.res], choice=self.choice, skip=self.skip, sum=True)
        return img
    def _color_global(self, image, resolution=128, choice=2, skip=False, sum=False):
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        _skip = 0
        if resolution == 256 or resolution == 128:
            _skip = -2

        result = []
        f_r = []
        f_g = []
        f_b = []
        for r in range(int(255 / resolution) + 1):
            f_r.append(np.multiply(image[0], (resolution * r <= image[0]) & ((resolution * (r + 1) - 1) >= image[0])))
        for g in range(int(255 / resolution) + 1):
            f_g.append(np.multiply(image[1], (resolution * g <= image[1]) & ((resolution * (g + 1) - 1) >= image[1])))
        for b in range(int(255 / resolution) + 1):
            f_b.append(np.multiply(image[2], (resolution * b <= image[2]) & ((resolution * (b + 1) - 1) >= image[2])))
        for f in range(int(255 / resolution) + 1):
            _skip += 1
            if skip and (_skip == 1 or _skip == (int(255 / resolution) + 1)):
                continue
            result.append(np.stack((np.array(f_r[f]), np.array(f_g[f]), np.array(f_b[f]))))

        result = np.array(sample(result, choice), dtype=np.uint8)
        if sum:
            result = result.sum(axis=0)
            result = np.expand_dims(result, axis=0)
        result = np.transpose(np.squeeze(result, axis=0), (1, 2, 0))
        result = Image.fromarray(result.astype(np.uint8))
        return result