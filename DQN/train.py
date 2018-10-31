from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import torch
import torch.nn as nn
import numpy as np
import cv2
import random

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)


RENDER = True
MAX_EPISODE = 1000000000
MEMORY_SIZE = 2000
UPDATE_INTERVAL = 100
GAMMA = 0.9
EPSILON = 1
EPSILON_MIN = 0.001
LEARNING_RATE = 0.01


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=s_dim[2], out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=4),
            nn.ReLU()
        ).cuda()
        self.fc = nn.Sequential(
            nn.Linear(1872, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, a_dim),
        ).cuda()


    def forward(self, s):
        s = s.cuda()
        f = self.cnn(s)
        f_flatten = f.reshape([-1, 1872])
        q_value = self.fc(f_flatten).cpu()
        return q_value


class DQN:
    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.episode = 0
        self.step = 0
        self.replay_buffer = []

        self.main_net = Net(s_dim, a_dim)
        self.target_net = Net(s_dim, a_dim)
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=LEARNING_RATE)

        initialize(self.main_net)
        initialize(self.target_net)

    def get_action(self, s):
        global EPSILON, EPSILON_MIN
        if random.uniform(0, 1) <= EPSILON:  # Exploration
            action = random.randint(0, self.a_dim - 1)
        else:
            q_value = self.main_net.forward(torch.Tensor([s]))
            action = q_value.argmax().item()

        if EPSILON > EPSILON_MIN:
            EPSILON *= 0.9999

        return action

    def memory(self, s, a, r, s_, done):
        s = s.reshape(1, -1)
        a = np.array([a]).reshape(1, -1)
        r = np.array([r]).reshape(1, -1)
        s_ = s_.reshape(1, -1)
        done = np.array([done]).reshape(1, -1)
        if len(self.replay_buffer) >= MEMORY_SIZE:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(np.hstack((s, a, r, s_, done)))

    def get_loss(self, batches):
        memory = np.vstack(batches)
        batch_s = torch.Tensor(memory[:, :184320]).reshape(-1, 3, 240, 256)
        batch_a = torch.LongTensor(memory[:, 184320]).reshape(-1, 1)
        batch_r = torch.Tensor(memory[:, 184320 + 1]).reshape(-1, 1)
        batch_s_ = torch.Tensor(memory[:, 184320 + 2: -1]).reshape(-1, 3, 240, 256)
        batch_done = torch.Tensor(memory[:, -1]).reshape(-1, 1)

        q_next = self.target_net.forward(batch_s_).max(dim=-1)[0].detach().reshape(-1, 1)

        for i in range(len(batch_done)):
            if batch_done[i]:
                q_next[i] = 0

        target = (batch_r + GAMMA * q_next)
        main = self.main_net.forward(batch_s).gather(1, batch_a)

        return torch.nn.MSELoss()(main, target)


    def train(self):
        batches = random.sample(self.replay_buffer, 36)
        loss = self.get_loss(batches)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.main_net.state_dict())


    def save(self):
        pass

    def load(self, path):
        pass




def main():

    model = DQN(env.observation_space.shape, env.action_space.n)

    done = True


    while model.episode < MAX_EPISODE:
        if done:
            state = env.reset()
            state = rgb2dataset(state)
            model.episode += 1
            print(model.episode)

        action = model.get_action(state)
        state_, reward, done, info = env.step(action)
        model.step += 1
        model.memory(state, action, reward, state_, done)

        if model.step > 1000:
            model.train()

        if model.step % 2000 == 0:
            model.update_target()

        state = rgb2dataset(state_)

        if RENDER:
            env.render()


    env.close()


def rgb2dataset(rgb_data):
    return np.array(cv2.split(rgb_data))

def initialize(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    main()