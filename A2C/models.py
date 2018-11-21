import torch
import torch.nn as nn

import numpy as np
import random

import cv2

import time

from utils import initialize

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=[8, 8], stride=[4, 4], padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=0),
            nn.LeakyReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.LeakyReLU(),
            nn.Linear(512, a_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(3136, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )
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
        x = x.reshape([-1, 3136])
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
        self.a_optimizer = torch.optim.Adam(self.net.actor.parameters(), lr=0.0003)
        self.c_optimizer = torch.optim.Adam(self.net.critic.parameters(), lr=0.005)
    def get_action(self, s, is_random=False):
        if np.random.uniform(0, 1) < self.epsilon or is_random:  # Exploration
            action = random.choice(np.arange(self.a_dim))
        else:
            probs, _ = self.net.forward(torch.Tensor([s]).to(self.device))

            action = torch.distributions.Categorical(probs).sample().cpu().numpy()[0]
            # print((probs.cpu().detach().numpy() * 100).astype(int))

        if self.epsilon > self.epsilon_end:
            self.epsilon -= (1 - self.epsilon_end) / self.epsilon_length

        return action

    def train(self, buffer_state, buffer_action, buffer_reward, done):

#        print(buffer_action)
#        print(buffer_reward)
#        print(done)

        display_list = buffer_state[-1]

        image = np.array(display_list[0])
        image = cv2.hconcat((image, display_list[1], display_list[2], display_list[3]))
        image = cv2.resize(image, (800, 200))

        cv2.imshow(str('Transition'), image)
        cv2.waitKey(0)


        states = torch.Tensor(np.array(buffer_state)).to(self.device)
        prob, value = self.net.forward(states)
#        print(value)
        tg_value = value.cpu().detach().numpy()
        tg_critic = []                                                  # td를 구할때 baseline으로 쓰임
        if done:
            tg_critic.append(0.)
        else:
            tg_critic.append(tg_value[-1])                              # 마지막 장면이 next_state
        tg_value = tg_value[:-1]                                        # 마지막 장면 삭제
        buffer_reward = buffer_reward[1:]                               # Reward가 한번 밀려서 들어오므로 첫번째 필요없음

        for r in buffer_reward[::-1]:                                   # 거꾸로 가면서 n-step
            tg_critic.append(r + self.gamma * tg_critic[-1])

        tg_critic.pop(0)                                                # 가장 첫번째 데이터는 next_state 이기 때문에 삭제
        tg_critic.reverse()                                             # 뒤에서 부터 n-step이므로 다시 뒤집기
        tg_critic = torch.Tensor(tg_critic).to(self.device)
        value = value[:-1]                                              # value의 마지막 값은 next_state이기 때문에 삭제

        buffer_action = buffer_action[:-1]                              # 마지막 action은 next_state의 action 이기 때문에 삭제
        buffer_action = torch.Tensor(buffer_action).to(self.device)

        td_error = tg_critic - value
        loss_critic = td_error.pow(2)                                   # td에서 MSE로 critic loss function

        dist = torch.distributions.Categorical(prob[:-1])               # 마지막 장면 prob 삭제
        probs = dist.log_prob(buffer_action)

        loss_actor = -dist.log_prob(buffer_action) * td_error.detach()
        loss = (loss_critic + loss_actor).mean()

        self.a_optimizer.zero_grad()
        self.c_optimizer.zero_grad()
        loss.backward()
        self.a_optimizer.step()
        self.c_optimizer.step()
    def save(self):
        state = {
            'global_episode': self.g_episode,
            'global_step': self.g_step,
            'main_net': self.net.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(state, 'saved_model/' + ("%07d" % (self.g_episode)) + '.pt')
        print('[Model] Saved model : ' + ("%07d" % (self.g_episode)) + '.pt')

    def load(self, path):
        data = torch.load('saved_model/' + path)
        self.episode = data['global_episode']
        self.step = data['global_step']
        self.net.load_state_dict(data['main_net'])
        self.epsilon = data['epsilon']
        print('[Model] Loaded model : ' + path)