import numpy as np
import random
from PIL import Image

class Puzzle():
    def __call__(self, img, prob=0.5):
        if random.random() < prob:
            img = self._puzzle(img)
        return img
    def _puzzle(self, img):
        #pillow image (H, W, C) => pillow image (C, H, W)
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))

        h = 16
        h2 = 32
        a = []
        a.append(img[:, 0:h, 0:h])
        a.append(img[:, 0:h, h:h2])
        a.append(img[:, h:h2, 0:h])
        a.append(img[:, h:h2, h:h2])

        _img = np.zeros_like(img)
        b = np.random.permutation(4)
        _img[:, 0:h, 0:h] = a[b[0]]
        _img[:, 0:h, h:h2] = a[b[1]]
        _img[:, h:h2, 0:h] = a[b[2]]
        _img[:, h:h2, h:h2] = a[b[3]]

        _img = (_img*255).astype(np.uint8)
        _img = np.transpose(_img, (1, 2, 0))
        return Image.fromarray(_img)