import numpy as np
from PIL import Image

class ChannelClip():
    def __init__(self, min=50, max=150):
        self.min = min
        self.max = max
    def __call__(self, img):
        #pillow image (H, W, C) => pillow image (C, H, W)
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        img = np.clip(img, self.min, self.max)
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)
        return img