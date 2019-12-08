import numpy as np
import random
from random import sample
from PIL import Image, ImageEnhance, ImageOps

class Decalcomanie():
    def __call__(self, img):
        if random.random() < 0.5:
            img = self._decalcomanie(img)
        return img
    def _decalcomanie(self, image):
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        lam = np.random.randint(4, size=1)
        if lam == 0:
            #print("0")
            #a = np.transpose(image.numpy(), (1, 2, 0))
            image[:, :16, :] = np.flip(image, 0)[:, :16, :]
        elif lam == 1:
            #print("1")
            #a = np.transpose(a.numpy(), (1, 2, 0))
            image[:, 16:, :] = np.flip(image, 0)[:, 16:, :]
        elif lam == 2:
            #("2")
            #a = np.transpose(a.numpy(), (1, 2, 0))
            image[:, :, :16] = np.flip(image, 1)[:, :, :16]
        elif lam == 3:
            #print("3")
            #a = np.transpose(a.numpy(), (1, 2, 0))
            image[:, :, 16:] = np.flip(image, 1)[:, :, 16:]
        #result = np.array(sample(result, choise), dtype=np.uint8)
        result = np.transpose(image, (1, 2, 0))
        result = Image.fromarray(result)
        return result