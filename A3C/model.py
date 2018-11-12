import torch
import torch.nn as nn

import numpy as np
import random
import cv2

from utils import initialize

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
        self.actor = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, a_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        initialize(self.cnn)
        initialize(self.actor)
        initialize(self.critic)

    def forward(self, x):
        x = self.cnn(x)
        '''
        # cnn debug
        channels =  x.cpu().detach().numpy()[0]
        for i, f in enumerate(channels):
            resized = cv2.resize(f, (90, 90))
            cv2.imshow('CNN Feature' + str(i), resized)
            cv2.moveWindow('CNN Feature' + str(i), int(i % 6) * 90, int(i / 6) * 100)
        cv2.waitKey(0)
        '''
        x = x.reshape([-1, 2592])
        p = self.actor(x)
        v = self.critic(x)
        return p, v

class A3C:
    def __init__(self, s_dim, a_dim,
                 gamma=0.95,
                 epsilon_start=1,
                 epsilon_end=0.1,
                 epsilon_length=10000,
                 use_cuda=False,
                 n_step=10,
                 lr=0.001):

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # self.n_step = n_step

        self.gamma = gamma

        self.g_episode = 0
        self.g_step = 0

        self.l_net = Net(s_dim, a_dim).to(self.device)

    def get_action(self, s, is_random=False):
        if is_random:
            action = random.choice(np.arange(self.a_dim))
        else:
            probs, _ = self.l_net.forward(torch.Tensor([s]).to(self.device))
            action = torch.distributions.Categorical(probs).sample().cpu().numpy()[0]

        return action

    def train(self, buffer_state, buffer_action, buffer_reward, done):

        states = torch.Tensor(np.array(buffer_state)).to(self.device)
        prob, value = self.net.forward(states)

        td_error = tg_critic - value
        loss_critic = td_error.pow(2)                                   # td에서 MSE로 critic loss function

        dist = torch.distributions.Categorical(prob[:-1])               # 마지막 장면 prob 삭제
        probs = dist.log_prob(buffer_action)

        loss_actor = -dist.log_prob(buffer_action) * td_error.detach()
        loss = (loss_critic + loss_actor).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

