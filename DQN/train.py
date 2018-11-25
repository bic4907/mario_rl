from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np

import cv2
import random, logging, datetime

from model import DQN
from utils import rgb2dataset

if __name__ == '__main__':

    ####### Env Settings ##########
    env_id = 'SuperMarioBros-v2'
    env = gym_super_mario_bros.make(env_id)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    s_dim = 4
    a_dim = env.action_space.n
    ###############################

    ##### Etc Settings ########################
    max_episode = 1000000
    use_cuda = True
    is_render = False
    save_model = True
    ###########################################

    ####### DQN Setting #######################
    replay_buffer_size = 500000
    train_start_step = 10000
    batch_size = 36
    gamma = 0.9
    lr=0.9
    train_step_interval = 100
    target_update_interval = 1000
    ###########################################

    model = DQN(env.observation_space.shape, env.action_space.n,
                gamma=gamma,
                epsilon_start=1,
                epsilon_end=0.01,
                epsilon_length=300000,
                use_cuda=use_cuda,
                lr=lr,
                replay_buffer_size=replay_buffer_size,
                train_start_step=train_start_step,
                batch_size=batch_size,
                target_update_interval=target_update_interval,
                train_step_interval=train_step_interval
                )

    writer = SummaryWriter(log_dir='runs/ddqn')

    while model.episode < max_episode:

        state = env.reset()
        state = rgb2dataset(state)
        model.episode += 1
        accum_reward = 0

        # Transition
        transition = []
        transition.append(state)

        while True:
            if len(transition) == 4:
                action = model.get_action(transition, is_random=False)
            else:
                action = model.get_action(transition, is_random=True)

            state_, reward, done, info = env.step(action)

            # Reward Shaping
            reward += -0.2
            if info['flag_get']:
                reward += 100

            state_ = rgb2dataset(state_)

            model.memory(state, action, reward, done)
            accum_reward += reward
            model.step += 1
            state = state_

            # Transition
            transition.append(state)
            if len(transition) > 4:
                transition.pop(0)

            if model.step > model.train_start_step and model.step % model.train_step_interval:
                model.train()
                if model.step % model.target_update_interval == 0:
                    model.update_target()

            if is_render:
                env.render()

            if done:

                writer.add_scalar('reward/accum', accum_reward, model.step)
                writer.add_scalar('data/epsilon', model.epsilon, model.step)
                writer.add_scalar('data/x_pos', info['x_pos'], model.step)
                print("Episode : %5d\t\tSteps : %10d\t\tReward : %7d\t\tX_step : %4d\t\tEpsilon : %.3f" % (model.episode, model.step, accum_reward, info['x_pos'], model.epsilon))

                if save_model and model.episode % 100 == 0:
                    model.save()

                break

    env.close()
