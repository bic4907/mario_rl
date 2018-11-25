import cv2, numpy as np
import torch.nn as nn

def rgb2dataset(rgb_data):
    gray_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_data, (84, 84))
    downsampled = resized / 255.0
    return np.array(downsampled)

def initialize(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()
    elif type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight, a=1.0)
        m.bias.data.zero_()