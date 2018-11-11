import torch
import torch.nn as nn

import numpy as np
import random

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
            nn.Linear(256, a_dim),
        )

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

class A2C:
    def __init__(self, s_dim, a_dim, num_worker,
                 gamma=0.95,
                 epsilon_start=1,
                 epsilon_end=0.1,
                 epsilon_length=10000,
                 use_cuda=True,
                 n_step=10,
                 lr=0.001):

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.num_worker = num_worker
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.n_step = n_step

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_length = epsilon_length

        self.g_episode = 0
        self.g_step = 0

        self.net = Net(s_dim, a_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def get_action(self, s, is_random=False):
        if np.random.uniform(0, 1) < self.epsilon or is_random:  # Exploration
            action = random.choice(np.arange(self.a_dim))
        else:
            probs, _ = self.net.forward(torch.Tensor([s]).to(self.device))
            action = torch.distributions.Categorical(probs).sample().cpu().numpy()[0]

        if self.epsilon > self.epsilon_end:
            self.epsilon -= (1 - self.epsilon_end) / self.epsilon_length

        return action

    def train(self, buffer_state, buffer_action, buffer_reward, current_state, done):

        # TODO 손실함수 계산

        states = torch.Tensor(np.array(buffer_state)).to(self.device)

        prob, value = self.net.forward(states)

        print(prob.shape, value.shape)



        return

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self):
        state = {
            'global_episode': self.episode,
            'global_step': self.step,
            'main_net': self.main_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(state, 'saved_model/transition/' + ("%07d" % (self.episode)) + '.pt')
        print('[Model] Saved model : ' + path)

    def load(self, path):
        data = torch.load('saved_model/transition/' + path)
        self.episode = data['global_episode']
        self.step = data['global_step']
        self.main_net.load_state_dict(data['main_net'])
        self.target_net.load_state_dict(data['target_net'])
        self.epsilon = data['epsilon']
        print('[Model] Loaded model : ' + path)