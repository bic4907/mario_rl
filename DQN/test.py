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

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RENDER = False
SAVE_MODEL = True

# Hyper Parameters
BUFFER_SIZE = int(1e5)
#BUFFER_SIZE = 100
BATCH_SIZE = 36
GAMMA = 0.99
TAU = float(1)

EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_LENGTH = 100000 # 해당프레임 동안 epsilon 감소

MAX_EPISODE = 10000
TRAIN_START_STEP = int(1e5)
#TRAIN_START_STEP = 100
LEARNING_RATE = float(1e-3)
UPDATE_INTERVAL = 2000

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

        self.fc = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, a_dim)
        )
        initialize(self.cnn)
        initialize(self.fc)

    def forward(self, s):
        f = self.cnn(s)
        f_flatten = f.reshape([-1, 2592])
        q_value = self.fc(f_flatten)
        return q_value


class DQN:
    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.episode = 0
        self.step = 0
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        self.main_net = Net(s_dim, a_dim).to(device)
        self.target_net = Net(s_dim, a_dim).to(device)
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=LEARNING_RATE)
        self.update_target()

    def get_action(self, s, is_random=False):
        global EPSILON, EPSILON_MIN

        if np.random.uniform(0, 1) < EPSILON or is_random:  # Exploration
            action = random.choice(np.arange(self.a_dim))
        else:
            q_value = self.main_net.forward(torch.Tensor([s]).to(device))
            action = q_value.argmax().item()

        if EPSILON > EPSILON_MIN:
            EPSILON -= (1 - EPSILON_MIN) / EPSILON_LENGTH

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

        target = (rewards + GAMMA * q_next)

        main = self.main_net.forward(states).gather(1, actions)

        return torch.nn.MSELoss()(main, target)

    def train(self):
        if len(self.replay_buffer) >= BUFFER_SIZE:
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
    writer = SummaryWriter(log_dir='runs/DQN_181107_DDQN (Input Changed)')

    while model.episode < MAX_EPISODE:

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
            state_ = rgb2dataset(state_)

            model.memory(state, action, reward, done)
            accum_reward += reward
            model.step += 1
            state = state_

            # Transition
            transition.append(state)
            if len(transition) > 4:
                transition.pop(0)

            if model.step > TRAIN_START_STEP:
                model.train()
                if model.step % UPDATE_INTERVAL == 0:
                    model.update_target()

            if RENDER:
                env.render()

            if done:
                logging.info("episode : %5d\tsteps : %10d\taccum_reward : %7d\tepsilon : %.3f" % (model.episode, model.step, accum_reward, EPSILON))
                writer.add_scalar('reward/accum', accum_reward, model.step)
                writer.add_scalar('epsilon', EPSILON, model.step)
                print("episode : %5d\t\tsteps : %10d\t\taccum_reward : %7d\t\tepsilon : %.3f" % (model.episode, model.step, accum_reward, EPSILON))

                if SAVE_MODEL and model.episode % 100 == 0:
                    model.save()

                break

    env.close()

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "done"])

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

        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.from_numpy(np.array(actions)).long().to(device).reshape(-1, 1)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device).reshape(-1, 1)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)



def rgb2dataset(rgb_data):
    # Use this for imshow
    # rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
    # Raw Image : (240, 256, 3)

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
    # DataSet Image : (3, 240, 240)

def initialize(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    logging.basicConfig(filename=('logs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'), filemode='a', level=logging.DEBUG)
    main()