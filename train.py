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
dqn_policy = DQN()
dqn_target = DQN()
icm = ICM()
dqn_target.load_state_dict(dqn_policy.state_dict())
dqn_policy.train()
dqn_target.eval()
icm.train()

# loss functions
inv_criterion = nn.CrossEntropyLoss()
fwd_criterion = nn.MSELoss()
dqn_criterion = nn.SmoothL1Loss()

# unified optimizer
optimizer = optim.Adam(list(dqn_policy.parameters()) + list(icm.parameters()), lr=config.learning_rate)

# favourite diagnostic function
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
            action = dqn_policy(state).max(1)[1].item()
    # explore (random action)
    else:
        action = random.randrange(data.n_words)
    action = torch.tensor([action])
    return action

# main training loop
env = Wordle()
memory = ReplayMemory(config.replay_memory_size)
# TODO: consider removing concept of epochs completely, replace with running step count
for epoch in range(config.n_epochs):
    running_score = 0
    running_curiosity_loss = 0
    running_policy_loss = 0
    skips = 0
    for game in range(config.n_games_per_epoch):
        env.reset()
        state = env.state()
        replay_buffer = []
        inv_losses = []
        fwd_losses = []
        # play game
        for score in range(config.n_rounds_per_game):
            # select action (epsilon greedy)
            action_index = select_action(state)
            action_onehot = F.one_hot(action_index, num_classes=data.n_words).float()
            word_guessed = torch.argmax(action_onehot, dim=1).item() + 1
            # receive next state from environment
            next_state, done = env.step(word_guessed)
            if not done:
                # forward pass (icm)
                action_logits, next_state_encoded_pred, next_state_encoded = icm(state, next_state, action_onehot)
                inv_loss = inv_criterion(action_logits, action_onehot)
                fwd_loss = fwd_criterion(next_state_encoded_pred, next_state_encoded) / 2
                intrinsic_reward = config.intrinsic_coeff * fwd_loss.detach()
                # add to buffers
                replay_buffer.append((state, action_index, next_state, intrinsic_reward))
                inv_losses.append(inv_loss)
                fwd_losses.append(fwd_loss)
                # step to next state
                state = next_state
            else:
                break
        running_score += env.score
        # reward from environment
        extrinsic_reward = config.n_rounds_per_game - env.score
        extrinsic_reward = torch.tensor([extrinsic_reward], dtype=torch.float)
        # add to experience replay
        for state, action, next_state, intrinsic_reward in replay_buffer:
            reward = intrinsic_reward + extrinsic_reward
            memory.push(state, action, next_state, reward)
        # sample batch from replay memory
        if len(memory) < config.batch_size:
            skips += 1
            continue
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(config.batch_size)
        # forward pass (dqn)
        q_values = dqn_policy(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = dqn_target(next_state_batch).detach().max(1)[0]
        target_q_values = reward_batch + (config.discount_factor * next_q_values)
        target_q_values = target_q_values.unsqueeze(1)
        # calculate loss
        policy_loss = dqn_criterion(q_values, target_q_values)
        c = config.curiosity_coeff
        curiosity_loss = sum([(1 - c) * inv + c * fwd for inv, fwd in zip(inv_losses, fwd_losses)])
        combined_loss = policy_loss + curiosity_loss
        # optimise model
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        # update target net every n games
        if game % config.target_update == 0:
            dqn_target.load_state_dict(dqn_policy.state_dict())
        # diagnostics
        running_curiosity_loss += float(curiosity_loss)
        running_policy_loss += float(policy_loss)
        ave_score = running_score / (game + 1)
        ave_curiosity_loss = running_curiosity_loss / (game - skips + 1)
        ave_policy_loss = running_policy_loss / (game - skips + 1)
        metrics = {
            'secret': env.secret,
            'ave_score': round(ave_score, 6),
            'curiosity_loss': round(ave_curiosity_loss, 6),
            'policy_loss': round(ave_policy_loss, 6)
        }
        prefix = f'Epoch {epoch+1}/{config.n_epochs}:'
        suffix = f'games played, metrics: {metrics}'
        print_progress_bar(game, config.n_games_per_epoch, prefix=prefix, suffix=suffix)