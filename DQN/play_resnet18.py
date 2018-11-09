from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import torch
import torch.nn as nn
import torchvision.models as models
from tensorboardX import SummaryWriter
import numpy as np

import cv2
import random, logging, datetime
from collections import namedtuple, deque
import itertools
import msvcrt as m

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RENDER = True
MODEL_FILENAME = '0000100.pt'

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.cnn = models.resnet18(pretrained=False)
        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, a_dim)
        )

    def forward(self, x):
        x = self.cnn(x)

        f = x.cpu().detach().numpy().reshape([50, 20])
        print(x.shape)
        cv2.imshow('CNN Feature', f)
        cv2.waitKey(0)

        x = x.reshape([-1, 1000])
        q_value = self.fc(x)
        raw = q_value.cpu().data[0].numpy()
        print(
            "NOP : %.2f | RIGHT : %.2f | RIGHT+A : %.2f | RIGHT+B : %.2f | RIGHT+A+B : %.2f | A : %.2f | LEFT : %.2f" % (
            raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6]))
        return q_value


class DQN:
    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.main_net = Net(s_dim, a_dim).to(device)
        self.target_net = Net(s_dim, a_dim).to(device)


    def get_action(self, s, is_random=False):
        q_value = self.main_net.forward(torch.Tensor([s]).to(device))
        action = q_value.argmax().item()
        return action

    def load(self, path):
        data = torch.load('saved_model/resnet18(pre)/' + path)
        self.episode = data['global_episode']
        self.step = data['global_step']
        self.main_net.load_state_dict(data['main_net'])
        self.target_net.load_state_dict(data['target_net'])
        print('LOADED MODEL : ' + path)


def main():
    model = DQN(env.observation_space.shape, env.action_space.n)
    model.load(MODEL_FILENAME)

    while True:

        state = env.reset()
        state = rgb2dataset(state)
        model.episode += 1
        accum_reward = 0

        while True:
            action = model.get_action(state, is_random=False)

            state_, reward, done, info = env.step(action)
            state_ = rgb2dataset(state_)

            accum_reward += reward
            state = state_

            if RENDER:
                env.render()

            if done:
                print("accum_reward : %7d" % (accum_reward))
                break

    env.close()


def rgb2dataset(rgb_data):
    cropped = rgb_data[16:240, 16:240]
    downsampled = cropped / 255.0
    return np.array(cv2.split(downsampled))

if __name__ == '__main__':
    main()