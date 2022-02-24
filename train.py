import random
import math
import string
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from wordle import Wordle
from agent import DQN, ICM, ReplayMemory, Transition

import config
import data

torch.manual_seed(42)

# instantiate agent components
target_net = DQN()
policy_net = DQN()
icm = ICM()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.train()
icm.train()

# loss functions
inv_criterion = nn.CrossEntropyLoss()
fwd_criterion = nn.MSELoss()
dqn_criterion = nn.SmoothL1Loss()

# unified optimizer
optimizer = optim.Adam(list(policy_net.parameters()) + list(icm.parameters()), lr=config.learning_rate)

# my favourite diagnostic function
def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='=', head='>', track='.'):
    iteration += 1
    filled_length = int(length * iteration // total)
    if filled_length == 0:
        bar = track * length
    elif filled_length == 1:
        bar = head + track * (length - 1)
    elif filled_length == length:
        bar = fill * filled_length
    else:
        bar = fill * (filled_length-1) + '>' + '.' * (length-filled_length)
    print(f'\r{prefix} [{bar}] {iteration}/{total} {suffix}', end = '\r')
    if iteration == total: 
        print()

# select an action using epsilon greedy policy
steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = config.eps_start + (config.eps_start - config.eps_end) * math.exp(-1. * steps_done / config.eps_decay)
    steps_done += 1
    # exploit (action with highest expected reward)
    if sample > eps_threshold:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].item()
    # explore (random action)
    else:
        action = random.randrange(data.n_words)
        action = torch.tensor([action])
    action = F.one_hot(action, num_classes=data.n_words).float()
    return action

# main training loop
env = Wordle()
memory = ReplayMemory(config.replay_memory_size)
# TODO: consider removing concept of epochs completely, replace with running step count
for epoch in range(config.n_epochs):
    for game in range(config.n_games_per_epoch):
        env.reset()
        state = env.state()
        replay_buffer = []
        inv_losses = []
        fwd_losses = []
        # play game
        for score in range(config.n_rounds_per_game):
            # select action (epsilon greedy)
            action = select_action(state)
            guess = torch.argmax(action, dim=1).item() + 1
            # receive next state from environment
            next_state, done = env.step(guess)
            # forward pass (icm)
            action_logits, next_state_encoded_pred, next_state_encoded = icm(state, next_state, action)
            inv_loss = inv_criterion(action_logits, action)
            fwd_loss = fwd_criterion(next_state_encoded_pred, next_state_encoded) / 2
            intrinsic_reward = config.intrinsic_coeff * fwd_loss.detach()
            # add to buffers
            replay_buffer.append((state, action, next_state, intrinsic_reward))
            inv_losses.append(inv_loss)
            fwd_losses.append(fwd_loss)
            # step to next state
            state = next_state
            if done:
                break
        # reward from environment
        extrinsic_reward = config.n_rounds_per_game - env.score
        # add to experience replay
        for state, action, next_state, intrinsic_reward in replay_buffer:
            reward = intrinsic_reward + extrinsic_reward
            memory.push(state, action, next_state, reward)
        # sample batch from replay memory
        if len(memory) < config.batch_size:
            continue
        transitions = memory.sample(config.batch_size)
        batch = Transition(*zip(*transitions)) # transpose batch
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # forward pass (dqn)
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(config.batch_size)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * config.gamma) + reward_batch
        # calculate loss
        dqn_loss = dqn_criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        c = config.curiosity_coeff
        curiosity_loss = sum([(1 - c) * inv + c * fwd for inv, fwd in inv_losses.zip(fwd_losses)])
        combined_loss = dqn_loss + curiosity_loss
        # optimise model
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        # update target net every n games
        if game % config.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        # diagnostics
        print_progress_bar(g, config.n_games_per_epoch, prefix=f'Epoch {epoch+1}/{config.n_epochs}:', suffix=f'secret={env.secret}')