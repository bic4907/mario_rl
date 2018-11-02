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


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

RENDER = True
MAX_EPISODE = 10000
MEMORY_SIZE = 8000
UPDATE_INTERVAL = 100
GAMMA = 0.99
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_LENGTH = 100000 # 해당프레임 동안 epsilon 감소
LEARNING_RATE = 0.01

SAVE_MODEL = True

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.cnn = models.resnet18(pretrained=True).cuda()
        self.fc = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, a_dim)
        ).cuda()
        initialize(self.fc)

    def forward(self, s):
        s = s.cuda()
        f = self.cnn(s)
        f_flatten = f.reshape([-1, 1000])
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

    def get_action(self, s):
        global EPSILON, EPSILON_MIN
        if random.uniform(0, 1) <= EPSILON:  # Exploration
            action = random.randint(0, self.a_dim - 1)
        else:
            q_value = self.main_net.forward(torch.Tensor([s]))
            action = q_value.argmax().item()

        if EPSILON > EPSILON_MIN:
            EPSILON -= (EPSILON - EPSILON_MIN) / EPSILON_LENGTH

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
        count_pixel = 224 * 224 * 3
        memory = np.vstack(batches)

        batch_s = torch.Tensor(memory[:, :count_pixel]).reshape(-1, 3, 224, 224)
        batch_a = torch.LongTensor(memory[:, count_pixel]).reshape(-1, 1)
        batch_r = torch.Tensor(memory[:, count_pixel + 1]).reshape(-1, 1)
        batch_s_ = torch.Tensor(memory[:, count_pixel + 2: -1]).reshape(-1, 3, 224, 224)
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
        global EPSILON
        state = {
            'global_episode': self.episode,
            'global_step': self.step,
            'main_net': self.main_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': EPSILON
        }
        torch.save(state, 'saved_model/' + ("%07d" % (self.episode)) + '.pt')

    def load(self, path):
        global EPSILON
        data = torch.load('save_model/' + path)
        self.episode = data['global_episode']
        self.step = data['global_step']
        self.main_net.load_state_dict(data['main_net'])
        self.target_net.load_state_dict(data['target_net'])
        EPSILON = data['epsilon']
        print('LOADED MODEL : ' + path)


def main():
    global EPSILON
    model = DQN(env.observation_space.shape, env.action_space.n)


    while model.episode < MAX_EPISODE:
        if done:
            state = env.reset()
            state = rgb2dataset(state)
            model.episode += 1
            logging.info("episode : %5d\tsteps : %10d\taccum_reward : %7d\tepsilon : %.3f" % (model.episode, model.step, accum_reward, EPSILON))
            print("episode : %5d\t\tsteps : %10d\t\taccum_reward : %7d\t\tepsilon : %.3f" % (model.episode, model.step, accum_reward, EPSILON))

            if SAVE_MODEL and model.episode % 100 == 0:
                model.save()

            action = model.get_action(state)
            state_, reward, done, info = env.step(action)
            state_ = rgb2dataset(state_)

            accum_reward += reward
            model.step += 1
            if RENDER:
                env.render()

            model.memory(state, action, reward, state_, done)

            if model.step > 100000:
                model.train()

            if model.step % 5000 == 0:
                model.update_target()

            state = state_

            if done:
                logging.info("episode : %5d\tsteps : %10d\taccum_reward : %7d\tepsilon : %.3f" % (model.episode, model.step, accum_reward, EPSILON))
                writer.add_scalar('reward/accum', accum_reward, model.step)
                print("episode : %5d\t\tsteps : %10d\t\taccum_reward : %7d\t\tepsilon : %.3f" % (model.episode, model.step, accum_reward, EPSILON))
                break

    env.close()


def rgb2dataset(rgb_data):
    # Use this for imshow
    # rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)

    # Raw Image : (240, 256, 3)
    cropped = rgb_data[16:240, 16:240]
    # Cropped Image : (224, 224, 3)
    return np.array(cv2.split(cropped))
    # DataSet Image : (3, 240, 240)

def initialize(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    logging.basicConfig(filename=('logs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'), filemode='a', level=logging.DEBUG)
    main()