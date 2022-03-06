import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque

import config
import words

num_actions = config.num_words
num_features = config.num_features

# state encoder
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(num_actions + 1, config.word_embedding_size)
        self.letter_embedding = nn.Embedding(26 + 1, config.letter_embedding_size) # 26 letters in alphabet + 1 empty case
        self.dense1 = nn.Linear(1065, num_features)
        self.relu = nn.LeakyReLU()
        self.dense2 = nn.Linear(num_features, num_features)
    
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

# actor critic
class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.encoder = Encoder()
        self.relu = nn.LeakyReLU()
        self.actor = nn.Linear(num_features, num_actions)
        self.critic = nn.Linear(num_features, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        x = self.encoder(state)
        x = self.relu(x)
        action = self.actor(x)
        action = action
        return self.actor(x), self.critic(x)

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
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state, next_state, action):
        state_encoded = self.encoder(state)
        next_state_encoded = self.encoder(next_state)
        action_logits = self.inverse_net(torch.cat((state_encoded, next_state_encoded), 1))
        next_state_encoded_pred = self.forward_net(torch.cat((state_encoded, action), 1))
        return action_logits, next_state_encoded_pred, next_state_encoded