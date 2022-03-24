import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque

import config
import words

num_actions = words.num_words
input_size = config.input_size
num_features = config.num_features
num_rounds = config.num_rounds_per_game

# state encoder
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.dense1 = nn.Linear(input_size, num_features)
        self.relu = nn.LeakyReLU()
        self.dense2 = nn.Linear(num_features, num_features)
    
    def forward(self, x):
        (x_keyboard, x_position, x_possible) = x
        # flatten and combine
        x_keyboard = torch.flatten(x_keyboard)
        x_position = torch.flatten(x_position)
        x_possible = torch.flatten(x_possible)
        x = torch.cat((x_keyboard, x_position, x_possible))
        x = x.view(1, -1).float()
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
        x = self.encoder(state).float()
        x = self.relu(x)
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