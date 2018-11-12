import cv2
import torch.nn as nn

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

def save(self, g_episode, g_step, net):
    state = {
        'global_episode': g_episode,
        'global_step': g_step,
        'main_net': net.state_dict(),
    }
    torch.save(state, 'saved_model/' + ("%07d" % (g_episode)) + '.pt')
    print('[ Model ] Saved model : ' + ("%07d" % (g_episode)) + '.pt')

def load(self, path, shared_net):
    data = torch.load('saved_model/' + path)
    episode = data['global_episode']
    step = data['global_step']
    shared_net.load_state_dict(data['main_net'])
    print('[ Model ] Loaded model : ' + path)
    return g_episode, g_step