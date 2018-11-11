from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter

from multiprocessing import Process, Pipe, Queue
from models import A2C
from utils import rgb2dataset
import copy

class MarioEnv(Process):
    def __init__(self, env_id, idx, child_conn, queue, n_step, is_render=False):
        super(MarioEnv, self).__init__()

        self.idx = idx
        self.env_id = env_id

        self.child_conn = child_conn
        self.queue = queue
        self.is_render = is_render
        self.n_step = n_step
        self.steps = 0
        self.episodes = 0
        self.accum_reward = 0
        self.transition = []

    def run(self):
        super(MarioEnv, self).run()

        self.env = gym_super_mario_bros.make(self.env_id)
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)
        self.reset()

        print('[ Worker ',self.idx, ']', 'Playing <', self.env_id, '>')

        self.request_action(0, False)

        while True:
            action = self.child_conn.recv()
            next_state, reward, done, info = self.env.step(action)
            self.steps += 1
            self.accum_reward += reward
            next_state = rgb2dataset(next_state)

            if self.is_render and self.idx == 0:
                self.env.render()

            # make a transition
            self.transition.append(next_state)
            if len(self.transition) > 4:
                self.transition.pop(0)

            if done:
                self.send_result(info['x_pos'])
                self.reset()
                self.request_action(reward, True)
            else:
                self.request_action(reward, False)

    def reset(self):
        state = self.env.reset()
        state = rgb2dataset(state)
        self.transition.clear()
        self.transition.append(state)

        self.steps = 0
        self.episodes += 1
        self.accum_reward = 0

    def request_action(self, reward, done):
        self.queue.put([self.idx, "OnStep", [self.transition, reward, done]])

    def send_result(self, x_pos):
        self.queue.put([self.idx, "Result", [self.episodes, self.steps, self.accum_reward, x_pos]])

if __name__ == '__main__':
    writer = SummaryWriter('runs/Vanilla')

    ####### Env Settings ##########
    env_id = 'SuperMarioBros-v2'
    env = gym_super_mario_bros.make(env_id)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    s_dim = 4 # transition
    a_dim = env.action_space.n
    env.close()
    ###############################


    ####### MultiProcessing Settings ##########
    num_worker = 1
    workers = []
    parent_conns = []
    queue = Queue()
    ###########################################

    ##### Etc Settings ########################
    max_episode = 10000
    n_step = 10
    use_cuda = True
    is_render = False
    save_model = True
    ###########################################

    buffer_state = [[] for _ in range(num_worker)]
    buffer_action = [[] for _ in range(num_worker)]
    buffer_reward = [[] for _ in range(num_worker)]
    buffer_next_state = [[] for _ in range(num_worker)]

    model = A2C(s_dim, a_dim, num_worker,
                gamma=0.95,
                epsilon_start=1.0,
                epsilon_end=0.1,
                epsilon_length=100000,
                use_cuda=use_cuda,
                n_step = n_step,
                lr=0.001
                )

    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        worker = MarioEnv(env_id, idx, child_conn, queue, n_step, is_render)
        worker.start()
        workers.append(worker)
        parent_conns.append(parent_conn)


    max_prob = 0
    while model.g_episode < max_episode:

        while queue.empty(): # Wait for worker's state
            continue

        # Received some data
        idx, command, parameter = queue.get()

        if command == "OnStep":
            transition, reward, done = parameter


            if len(transition) != 4:
                action, prob = model.get_action(transition, is_random=True)
            else:
                action, prob = model.get_action(transition, is_random=False)

                buffer_state[idx].append(np.array(transition))
                buffer_action[idx].append(action)
                buffer_reward[idx].append(reward)

            if max_prob < prob:
                max_prob = prob

            # n-step을 위한 데이터들이 다 모였을 시
            if len(buffer_state[idx]) > n_step:
                model.train(buffer_state[idx], buffer_action[idx], buffer_reward[idx], done)

                # 가장 오래된 데이터부터 삭제
                buffer_state[idx].pop(0)
                buffer_action[idx].pop(0)
                buffer_reward[idx].pop(0)

            if done:
                buffer_state[idx].clear()
                buffer_action[idx].clear()
                buffer_reward[idx].clear()

            parent_conns[idx].send(action)


        elif command == "Result":
            episode, step, reward, x_pos = parameter
            model.g_episode += 1
            model.g_step += step

            print('[ Worker', idx, '] ', end='')
            print("Episode : %5d\tStep : %5d\tReward : %5d\tMax_prob : %.3f\tEpsilon : %3f\tX_pos : %5d" % (model.g_episode, step, reward, max_prob, model.epsilon, x_pos))

            writer.add_scalar('perf/x_pos', x_pos, model.g_step)
            writer.add_scalar('perf/reward', reward, model.g_step)
            writer.add_scalar('data/epsilon', model.epsilon, model.g_step)
            writer.add_scalar('data/max_prob', max_prob, model.g_step)

            if model.g_episode % 50 == 0:
                model.save()

            max_prob = 0