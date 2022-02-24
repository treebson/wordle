import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque

import config
import data

num_actions = data.n_words
num_features = 512

# state encoder
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(data.n_words + 1, config.word_embedding_size)
        self.letter_embedding = nn.Embedding(data.n_letters + 1, config.letter_embedding_size)
        self.dense1 = nn.Linear(745, num_features)
        self.dense2 = nn.Linear(num_features, num_features)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = x.view(-1, 5, 11)
        # words
        x_words = x[:, :, 0]
        x_words = self.word_embedding(x_words)
        x_words = torch.flatten(x_words, 1, -1)
        # letters
        x_letters = x[:, :, 1:6]
        x_letters = self.letter_embedding(x_letters)
        x_letters = torch.flatten(x_letters, 1, -1)
        # clues
        x_clues = x[:, :, 6:]
        x_clues = torch.flatten(x_clues, 1, -1)
        # combine
        x = torch.cat((x_words, x_letters, x_clues), -1)
        # dense layers
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

# intrinsic curiosity module
class ICM(nn.Module):

    def __init__(self):
        super(ICM, self).__init__()
        self.encoder = Encoder()
        self.inverse_net = nn.Sequential(
            nn.Linear(num_features * 2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, num_actions)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(num_features + num_actions, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, num_features)
        )

    def forward(self, state, next_state, action):
        state_encoded = self.encoder(state)
        next_state_encoded = self.encoder(next_state)
        action_logits = self.inverse_net(torch.cat((state_encoded, next_state_encoded), 1))
        next_state_encoded_pred = self.forward_net(torch.cat((state_encoded, action), 1))
        return action_logits, next_state_encoded_pred, next_state_encoded

# deep q network
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.encoder = Encoder()
        self.relu = nn.LeakyReLU()
        self.output = nn.Linear(num_features, num_actions)

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions)) # transpose batch
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        return state_batch, action_batch, reward_batch, next_state_batch
    
    def __len__(self):
        return len(self.memory)