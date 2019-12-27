import numpy as np
import torch

#this method can not use in the pytorch transforms.
class ChannelMix():
    def __init__(self):
        pass
    def __call__(self, imgs, targets):
        #torch tensor => torch tensor
        #input is torch tensor, ex(B, C, H, W) and this values is between 0 to 1
        rand = np.random.permutation(imgs.size(0))
        crand = np.random.randint(0, 3)
        imgs[:, crand, :, :] = imgs[rand, crand, :, :]
        return imgs, targets, targets[rand]