import torch.nn as nn
import cv2

def rgb2dataset(rgb_data):
    gray_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2GRAY)
    cropped = gray_data[16:240, 16:240]
    resized = cv2.resize(cropped, (84, 84))
    downsampled = resized / 255.0
    return downsampled

def initialize(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
