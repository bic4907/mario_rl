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
MODEL_FILENAME = '0000900.pt'

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=[8, 8], stride=[4, 4], padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[4, 4], stride=[2, 2], padding=0),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, a_dim)
        )

    def forward(self, x):
        x = self.cnn(x)



        channels =  x.cpu().detach().numpy()[0]
        for i, f in enumerate(channels):
            resized = cv2.resize(f, (90, 90))
            cv2.imshow('CNN Feature' + str(i), resized)
            cv2.moveWindow('CNN Feature' + str(i), i * 90, 100)
        cv2.waitKey(0)


        x = x.reshape([-1, 2592])
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
        if is_random:
            action = random.choice(np.arange(self.a_dim))
        else:
            q_value = self.main_net.forward(torch.Tensor([s]).to(device))
            action = q_value.argmax().item()
        return action

    def load(self, path):
        data = torch.load('saved_model2/' + path)
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

        # Transition
        transition = []
        transition.append(state)

        model.episode += 1
        accum_reward = 0

        while True:
            if len(transition) == 4:
                action = model.get_action(transition, is_random=False)
            else:
                action = model.get_action(transition, is_random=True)


            state_, reward, done, info = env.step(action)
            state_ = rgb2dataset(state_)

            accum_reward += reward
            state = state_

            # Transition
            transition.append(state)
            if len(transition) > 4:
                transition.pop(0)

            if RENDER:
                env.render()

            if done:
                print("accum_reward : %7d" % (accum_reward))
                break

    env.close()


def rgb2dataset(rgb_data):
    gray_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2GRAY)
    # Grayed Image : (240, 256, 1)
    cropped = gray_data[16:240, 16:240]
    # Cropped Image : (224, 224, 3)
    resized = cv2.resize(cropped, (84, 84))
    # Resized Image : (84 84, 3)
    downsampled = resized / 255.0
    #cv2.imshow('Window', cropped)
    #cv2.waitKey(0)
    return downsampled

if __name__ == '__main__':
    main()