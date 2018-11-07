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

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RENDER = True

# Hyper Parameters
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 36
GAMMA = 0.99
TAU = float(1)

EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_LENGTH = 100000 # 해당프레임 동안 epsilon 감소

MAX_EPISODE = 10000
TRAIN_START_STEP = int(1e5)
LEARNING_RATE = float(1e-3)
UPDATE_INTERVAL = 2000

MODEL_FILE = '0000900.pt'

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.cnn = models.resnet18(pretrained=False)
        self.fc = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, a_dim)
        )


    def forward(self, s):
        f = self.cnn(s)
        f_flatten = f.reshape([-1, 1000])
        q_value = self.fc(f_flatten)
        return q_value



class DQN:
    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.episode = 0
        self.step = 0

        self.main_net = Net(s_dim, a_dim).to(device)
        self.target_net = Net(s_dim, a_dim).to(device)

    def get_action(self, s):

        q_value = self.main_net.forward(torch.Tensor([s]).to(device))
        action = q_value.argmax().item()


        return action

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

    model = DQN(env.observation_space.shape, env.action_space.n)
    model.load(MODEL_FILE)

    while True:

        state = env.reset()
        state = rgb2dataset(state)
        model.episode += 1
        accum_reward = 0

        while True:
            action = model.get_action(state)
            state_, reward, done, info = env.step(action)
            state_ = rgb2dataset(state_)

            accum_reward += reward
            model.step += 1
            state = state_

            if RENDER:
                env.render()

            if done:
                print("episode : %5d\t\tsteps : %10d\t\taccum_reward : %7d\t\tepsilon : %.3f" % (model.episode, model.step, accum_reward, EPSILON))

                break

    env.close()


def rgb2dataset(rgb_data):
    # Use this for imshow
    # rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)

    # Raw Image : (240, 256, 3)
    cropped = rgb_data[16:240, 16:240]
    # Cropped Image : (224, 224, 3)
    return np.array(cv2.split(cropped))
    # DataSet Image : (3, 240, 240)

def initialize(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    main()