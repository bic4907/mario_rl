import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
from collections import namedtuple, deque
import numpy as np
import random

from utils import initialize

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=[8, 8], stride=[4, 4], padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=0),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, a_dim)
        )

        initialize(self.cnn)
        initialize(self.cnn)

    def forward(self, s):
        f = F.relu(self.cnn(s))
        f_flatten = f.reshape([-1, 3136])
        q_value = self.fc(f_flatten)
        return q_value


class DQN:
    def __init__(self, s_dim, a_dim,
                 gamma=0.95,
                 epsilon_start=1,
                 epsilon_end=0.1,
                 epsilon_length=10000,
                 use_cuda=False,
                 lr=0.00001,
                 replay_buffer_size=10000,
                 train_start_step=10000,
                 batch_size=36,
                 target_update_interval=1000,
                 train_step_interval = 100
                ):

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.episode = 0
        self.step = 0

        self.replay_buffer_size = replay_buffer_size
        self.train_start_step = train_start_step
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.train_step_interval = train_step_interval
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.replay_buffer = ReplayBuffer(replay_buffer_size, batch_size, self.device)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_length = epsilon_length

        self.main_net = Net(s_dim, a_dim).to(self.device)
        self.target_net = Net(s_dim, a_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=lr)
        self.update_target()

    def get_action(self, s, is_random=False):
        if np.random.uniform(0, 1) < self.epsilon  or is_random:  # Exploration
            action = random.choice(np.arange(self.a_dim))
        else:
            q_value = self.main_net.forward(torch.Tensor([s]).to(self.device))
            action = q_value.argmax().item()

        if self.epsilon > self.epsilon_end:
            self.epsilon -= (1 - self.epsilon_end) / self.epsilon_length

        return action

    def memory(self, s, a, r, done):
        self.replay_buffer.add(s, a, r, done)

    def get_loss(self, batches):
        states, actions, rewards, next_states, dones = batches

        action_ddqn = torch.argmax(self.main_net.forward(next_states), dim=-1, keepdim=True)
        q_next = self.target_net.forward(next_states).gather(1, action_ddqn).detach()

        for i in range(len(dones)):
            if dones[i]:
                q_next[i] = 0

        target = (rewards + self.gamma * q_next)

        main = self.main_net.forward(states).gather(1, actions)

        return torch.nn.MSELoss()(main, target)

    def train(self):
        if len(self.replay_buffer) >= self.replay_buffer_size:
            batches = self.replay_buffer.sample()

            loss = self.get_loss(batches)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

    def save(self):
        global EPSILON
        state = {
            'global_episode': self.episode,
            'global_step': self.step,
            'main_net': self.main_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(state, 'saved_model/' + ("%07d" % (self.episode)) + '.pt')
        print('[ Model ] Saved model : ' + ("%07d" % (self.episode)) + '.pt')


    def load(self, path):
        global EPSILON
        data = torch.load('saved_model/' + path)
        self.episode = data['global_episode']
        self.step = data['global_step']
        self.main_net.load_state_dict(data['main_net'])
        self.target_net.load_state_dict(data['target_net'])
        self.epsilon = data['epsilon']
        print('[ Model ] Loaded model : ' + path)

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "done"])
        self.device = device

    def add(self, state, action, reward, done):
        # e = self.experience(state, action, reward, next_state, done)
        e = self.experience(state, action, reward, done)
        self.memory.append(e)

    def sample(self):

        # 1. 배치사이즈 만큼 랜덤 수 뽑기 ( range : 5 ~ batch_size, 4 transition + 1 next_state )
        rand_idx = np.random.uniform(5, len(self.memory) - 1, size=(self.batch_size)).astype(np.int)

        # 2. 배치사이즈 만큼 루프돌면서 샘플 추출
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in rand_idx:

            # transition 중에 done이 된 장면이 있으면 그 배치는 사용하지 않고 스킵함
            is_skip = False
            for sub_idx in range(idx - 4, idx):
                if self.memory[sub_idx].done == True:
                    is_skip = True
                    break
            if is_skip:
                continue

            # Make transition
            short_exp = list(self.memory)[idx - 4:idx]
            current_transition = [e.state for e in short_exp]
            short_exp = list(self.memory)[idx - 3:idx + 1]
            next_transition = [e.state for e in short_exp]

            states.append(current_transition)
            actions.append(self.memory[idx].action)
            rewards.append(self.memory[idx].reward)
            next_states.append(next_transition)
            dones.append(self.memory[idx].done)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device).reshape(-1, 1)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device).reshape(-1, 1)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
