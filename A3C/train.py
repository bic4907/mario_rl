from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe, Queue
from model import A3C, Net
from utils import rgb2dataset
import copy

class MarioEnv(Process):
    def __init__(self, env_id, idx, child_conn, queue, s_dim, a_dim, g_net, g_opt, update_iter=10, is_render=False, use_cuda=False):
        super(MarioEnv, self).__init__()

        self.idx = idx
        self.env_id = env_id

        self.child_conn = child_conn
        self.queue = queue
        self.is_render = is_render
        # self.n_step = n_step
        self.update_iter=update_iter
        self.steps = 0
        self.episodes = 0
        self.accum_reward = 0
        self.transition = []

        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.g_net = g_net
        self.g_opt = g_opt

        self.buffer_state = []
        self.buffer_action = []
        self.buffer_reward = []


    def run(self):
        super(MarioEnv, self).run()

        self.model = A3C(self.s_dim, self.a_dim,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_length=100000,
            use_cuda=self.use_cuda,
        )
        self.model.l_net.load_state_dict(self.g_net.state_dict())

        self.env = gym_super_mario_bros.make(self.env_id)
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)
        self.reset()
        print('[ Worker %2d ] ' % (self.idx), end='')
        print('Playing <', self.env_id, '>')

        while True:
            if len(self.transition) != 4:
                action = self.model.get_action(self.transition, is_random=True)
            else:
                action = self.model.get_action(self.transition, is_random=False)


            next_state, reward, done, info = self.env.step(action)
            self.steps += 1
            self.accum_reward += reward
            next_state = rgb2dataset(next_state)

            if self.is_render and self.idx == 0:
                self.env.render()

            self.buffer_state.append(self.transition)
            self.buffer_action.append(action)
            self.buffer_reward.append(reward)

            if len(self.buffer_state) > 0 and self.steps % self.update_iter == 0:
                next_transition = self.transition[1:]
                next_transition.append(next_state)

                self.train(next_transition, done)

                self.buffer_state.clear()
                self.buffer_action.clear()
                self.buffer_reward.clear()


            # make a transition
            self.transition.append(next_state)
            if len(self.transition) > 4:
                self.transition.pop(0)


            if done:
                self.send_result(info['x_pos'])
                self.reset()

    def reset(self):
        state = self.env.reset()
        state = rgb2dataset(state)
        self.transition.clear()
        self.transition.append(state)

        self.steps = 0
        self.episodes += 1
        self.accum_reward = 0

    def send_result(self, x_pos):
        self.queue.put([self.idx, "Result", [self.episodes, self.steps, self.accum_reward, x_pos]])

    def train(self, next_transition, done):
        if done:
            v_s_ = 0.
        else:
            _, v = self.model.l_net.forward(torch.Tensor([next_transition]).to(self.device))
            v_s_ = v.cpu().detach().numpy()[0][0]

        prob, v = self.model.l_net.forward(torch.Tensor(self.buffer_state).to(self.device))

        buffer_v_target = []
        for r in self.buffer_reward[::-1]:
            v_s_ = r + self.model.gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        buffer_v_target = torch.Tensor(np.array(buffer_v_target)).to(self.device)
        buffer_action = torch.Tensor(np.array(self.buffer_action)).to(self.device)


        # LOSS 함수 구성
        td_error = buffer_v_target - v
        loss_critic = td_error.pow(2)

        dist = torch.distributions.Categorical(prob)
        loss_actor = -dist.log_prob(buffer_action) * td_error.detach()

        loss = (loss_critic + loss_actor).mean()

        self.g_opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.model.l_net.parameters(), self.g_net.parameters()):
            gp._grad = lp.grad.clone().cpu()
        self.g_opt.step()

        self.model.l_net.load_state_dict(self.g_net.state_dict())


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
    max_episode = 1000000
    # n_step = 10
    use_cuda = True
    is_render = False
    save_model = True
    ###########################################

    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        try:
            mp.set_start_method('spawn')
        except:
            pass

    ######## Variable for A3C #################
    from SharedAdam import SharedAdam
    g_net = Net(s_dim, a_dim).share_memory()
    g_opt = SharedAdam(g_net.parameters(), lr=0.001)
    ###########################################

    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        worker = MarioEnv(env_id, idx, child_conn, queue, s_dim, a_dim, g_net, g_opt, update_iter=10, is_render=is_render, use_cuda=use_cuda)
        # (self, env_id, idx, child_conn, queue, s_dim, a_dim, g_net, g_opt, update_iter=10, is_render=False, use_cuda=False):
        worker.start()
        workers.append(worker)
        parent_conns.append(parent_conn)

    g_episode = 0
    g_step = 0
    while g_episode < max_episode:

        while queue.empty(): # Wait for worker's state
            continue

        # Received some data
        idx, command, parameter = queue.get()

        if command == "Result":
            episode, step, reward, x_pos = parameter
            g_episode += 1
            g_step += step

            print('[ Worker %2d ] '% (idx), end='')
            print("Episode : %5d\tStep : %5d\tReward : %5d\t\tX_pos : %5d" % (g_episode, g_step, reward, x_pos))

            writer.add_scalar('perf/x_pos', x_pos, g_step)
            writer.add_scalar('perf/reward', reward, g_step)

            if g_episode % 100 == 0:
                save(g_episode, g_step, g_net)