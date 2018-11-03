from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import torch
import torch.nn as nn
import numpy as np
import cv2
import random
import logging
import time, datetime

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)


RENDER = True
MAX_EPISODE = 1000000000
MEMORY_SIZE = 8000
UPDATE_INTERVAL = 100
GAMMA = 0.9
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_LENGTH = 100000 # 해당프레임 동안 epsilon 감소
LEARNING_RATE = 0.001

MODEL_FILE = '0000200.pt'

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=s_dim[2], out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=4),
            nn.ReLU()
        ).cuda()
        self.fc = nn.Sequential(
            nn.Linear(1872, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, a_dim),
        ).cuda()


    def forward(self, s):
        s = s.cuda()
        f = self.cnn(s)
        f_flatten = f.reshape([-1, 1872])
        q_value = self.fc(f_flatten).cpu()
        return q_value


class DQN:
    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.episode = 0
        self.step = 0
        self.replay_buffer = []

        self.main_net = Net(s_dim, a_dim)
        self.target_net = Net(s_dim, a_dim)
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=LEARNING_RATE)

        initialize(self.main_net)
        initialize(self.target_net)

    def get_action(self, s):
        q_value = self.main_net.forward(torch.Tensor([s]))
        action = q_value.argmax().item()

        return action



    def train(self):
        batches = random.sample(self.replay_buffer, 36)
        loss = self.get_loss(batches)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load(self, path):
        global EPSILON
        data = torch.load('saved_model/' + path)
        self.episode = data['global_episode']
        self.step = data['global_step']
        self.main_net.load_state_dict(data['main_net'])
        self.target_net.load_state_dict(data['target_net'])
        EPSILON = data['epsilon']
        print('LOADED MODEL : ' + path)


def main():
    global EPSILON
    model = DQN(env.observation_space.shape, env.action_space.n)

    try:
        model.load(MODEL_FILE)
    except:
        pass


    accum_reward = 0

    state = env.reset()
    state = rgb2dataset(state)

    while model.episode < MAX_EPISODE:

        action = model.get_action(state)
        state_, reward, done, info = env.step(action)
        accum_reward += reward

        if RENDER:
            env.render()

        state = rgb2dataset(state_)


        if done:
            state = env.reset()
            state = rgb2dataset(state)
            print("episode : %5d\t\tsteps : %10d\t\taccum_reward : %7d\t\tepsilon : %.3f" % (model.episode, model.step, accum_reward, EPSILON))
            accum_reward = 0

    env.close()


def rgb2dataset(rgb_data):
    return np.array(cv2.split(rgb_data))

def initialize(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    main()