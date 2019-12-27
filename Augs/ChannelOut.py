import numpy as np
import torch
from PIL import Image

class ChannelOut():
    def __init__(self):
        pass
    def __call__(self, img):
        #pillow image (H, W, C) => pillow image (C, H, W)
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        rand = np.random.randint(0, 3)
        img[rand, :, :] = 0
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)
        return img