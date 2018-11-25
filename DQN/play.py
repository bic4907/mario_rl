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
from model import DQN

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RENDER = True
MODEL_FILENAME = '0001800.pt'





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