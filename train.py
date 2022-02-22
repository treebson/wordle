import random
import math
import string
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple, deque

from agent import DQN
from wordle import Wordle

import config
import data

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    # save a transition
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(config.replay_memory_size)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = config.eps_start + (config.eps_start - config.eps_end) * math.exp(-1. * steps_done / config.eps_decay)
    steps_done += 1
    # explore (random)
    if sample > eps_threshold:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].item() + 1
    # exploit (net)
    else:
        action = random.randrange(data.n_words) + 1
    action = torch.tensor([[action]], dtype=torch.int64)
    return action

def optimize_model():
    if len(memory) < config.batch_size:
        return
    transitions = memory.sample(config.batch_size)
    batch = Transition(*zip(*transitions)) # transpose batch

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(config.batch_size)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * config.gamma) + reward_batch

    # compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    # TODO: is gradient clipping necessary?
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return float(loss)

def print_progress_bar(iteration, total, prefix="", suffix="", length=30, fill="=", head=">", track="."):
    iteration += 1
    filled_length = int(length * iteration // total)
    if filled_length == 0:
        bar = track * length
    elif filled_length == 1:
        bar = head + track * (length - 1)
    elif filled_length == length:
        bar = fill * filled_length
    else:
        bar = fill * (filled_length-1) + ">" + "." * (length-filled_length)
    print("\r" + prefix + "[" + bar + "] " + str(iteration) + "/" + str(total), suffix, end = "\r")
    if iteration == total: 
        print()

# main training loop
env = Wordle()
for e in range(config.n_epochs):
    for g in range(config.n_games_per_epoch):
        env.reset()
        state = env.state()
        action_states = []
        # play game
        for score in range(config.n_rounds_per_game):
            action = select_action(state)
            next_state, done = env.step(action.item())
            action_states.append((state, action, next_state))
            state = next_state
            if done:
                break
        # calculate reward
        reward = config.n_rounds_per_game - env.score
        reward = torch.tensor([[reward]])
        # store in replay memory
        for state, action, next_state in action_states:
            memory.push(state, action, next_state, reward)
        # train
        loss = optimize_model()
        # update target net every N games
        if g % config.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print_progress_bar(g, config.n_games_per_epoch, 
            prefix=f'Training epoch {e+1}/{config.n_epochs}: ', 
            suffix=f"loss={loss}, secret={env.secret}")