import numpy as np
import random
from PIL import Image

class Puzzle():
    def __call__(self, img):
        if random.random() < 0.5:
            img = self._puzzle(img)
        return img
    def _puzzle(self, image):
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))

        h = 16
        h2 = 32
        a = []
        a.append(image[:, 0:h, 0:h])
        a.append(image[:, 0:h, h:h2])
        a.append(image[:, h:h2, 0:h])
        a.append(image[:, h:h2, h:h2])

        _image = np.zeros_like(image)
        b = np.random.permutation(4)
        _image[:, 0:h, 0:h] = a[b[0]]
        _image[:, 0:h, h:h2] = a[b[1]]
        _image[:, h:h2, 0:h] = a[b[2]]
        _image[:, h:h2, h:h2] = a[b[3]]

        _image = (_image*255).astype(np.uint8)
        _image = np.transpose(_image, (1, 2, 0))
        return Image.fromarray(_image)