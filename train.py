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
    message = f'\r{prefix} [{bar}] {iteration}/{total} {suffix}'
    print(message, end = '\r')
    if iteration == total: 
        print()
    return message

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
        strategy = 'exploit'
    # explore (random action)
    else:
        action = random.randrange(data.n_words)
        strategy = 'explore'
    action = torch.tensor([action])
    return action, strategy

import os
if os.path.isdir('output'):
    os.rmdir('output')
os.mkdir('output')

# main training loop
env = Wordle()
memory = ReplayMemory(config.replay_memory_size)
# TODO: consider removing concept of epochs completely, replace with running step count
for epoch in range(config.n_epochs):
    running_score = 0
    running_reward = 0
    losses = {'combined': 0, 'curiosity': 0, 'policy': 0}
    strategies = {'explore': 0, 'exploit': 0}
    guesses = {}
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
            action_index, strategy = select_action(state)
            strategies[strategy] += 1
            action_onehot = F.one_hot(action_index, num_classes=data.n_words).float()
            guess_index = torch.argmax(action_onehot, dim=1).item() + 1
            guess = data.idx2word[guess_index]
            if guess in guesses:
                guesses[guess] += 1
            else:
                guesses[guess] = 1
            # receive next state from environment
            next_state, done = env.step(guess_index)
            if not done:
                # forward pass (icm)
                pred_logits, pred_phi, phi = icm(state, next_state, action_onehot)
                inv_loss = inv_criterion(pred_logits, action_onehot)
                fwd_loss = fwd_criterion(pred_phi, phi) / 2
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
            running_reward += float(reward)
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
        losses['combined'] += float(combined_loss)
        losses['curiosity'] += float(curiosity_loss)
        losses['policy'] += float(policy_loss)
        # optimise model
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        # update target net every n games
        if game % config.target_update == 0:
            dqn_target.load_state_dict(dqn_policy.state_dict())
        # diagnostics
        ave_score = running_score / (game + 1)
        ave_reward = running_reward / (game + 1)
        ave_combined_loss = losses['combined'] / (game - skips + 1)
        ave_curiosity_loss = losses['curiosity'] / (game - skips + 1)
        ave_policy_loss = losses['policy'] / (game - skips + 1)
        top_guesses = dict(sorted(guesses.items(), key=lambda x: x[1], reverse=True)[:5])
        metrics = {
            'avg_score': round(ave_score, 6),
            'avg_reward': round(ave_reward, 6),
            'explore': round(strategies['explore'] / (strategies['explore'] + strategies['exploit']), 3),
            'total_loss': round(ave_combined_loss, 6),
            'icm_loss': round(ave_curiosity_loss, 6),
            'dqn_loss': round(ave_policy_loss, 6),
            'top_guesses': top_guesses
        }
        prefix = f'Epoch {epoch+1}/{config.n_epochs}:'
        suffix = f'games played, metrics: {metrics}'
        message = print_progress_bar(game, config.n_games_per_epoch, prefix=prefix, suffix=suffix)
    with open('output/log.txt', 'a') as log_file:
        log_file.write(message)
    torch.save(dqn_policy.state_dict(), 'output/dqn.pth')
    torch.save(icm.state_dict(), 'output/icm.pth')
    
        