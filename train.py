import random
import math
import string
import os
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from wordle import Wordle
from agent import ICM, Policy, SavedAction

import config
import data

torch.manual_seed(42)

# instantiate agent components
policy = Policy()
icm = ICM()
policy.train()
icm.train()

# loss functions
policy_criterion = nn.SmoothL1Loss()
inv_criterion = nn.CrossEntropyLoss()
fwd_criterion = nn.MSELoss()

# unified optimizer
optimizer = optim.Adam(list(policy.parameters()) + list(icm.parameters()), lr=config.learning_rate)

# constants
eps = np.finfo(np.float32).eps.item()
c = config.curiosity_coeff

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

def reset_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

# select an action using epsilon greedy policy
def select_action(state):
    probs, state_value = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    action = torch.tensor([action.item()])
    return action

# training loop
def main():
    reset_directory('output')
    env = Wordle()
    # run infinitely many episodes
    for epoch in range(config.n_epochs):
        running_reward = 0.0
        running_score = 0.0
        losses = {'total': 0.0, 'actor': 0.0, 'critic': 0.0, 'icm': 0.0}
        guesses = {}
        for game in range(config.n_games_per_epoch):
            state = env.reset()
            curiosity_losses = []
            intrinsic_rewards = []
            # play a game
            for score in range(config.n_rounds_per_game):
                action_index = select_action(state)
                action_onehot = F.one_hot(action_index, num_classes=data.n_words).float()
                guess_index = torch.argmax(action_onehot, dim=1).item() + 1
                guessed_word = data.idx2word[guess_index]
                # guesses[guess] += 1 if guess in guesses else guesses[guess] = 1
                next_state, done = env.step(guess_index)
                if done:
                    break
                # intrinsic reward
                pred_logits, pred_phi, phi = icm(state, next_state, action_onehot)
                inv_loss = inv_criterion(pred_logits, action_onehot)
                fwd_loss = fwd_criterion(pred_phi, phi) / 2
                intrinsic_reward = config.intrinsic_coeff * fwd_loss.detach()
                curiosity_loss = (1 - c) * inv_loss + c * fwd_loss
                curiosity_losses.append(curiosity_loss)
                intrinsic_rewards.append(intrinsic_reward)
                state = next_state
            running_score += env.score
            # extrinsic reward
            extrinsic = config.n_rounds_per_game - env.score
            extrinsic = torch.tensor([extrinsic], dtype=torch.float)
            for intrinsic in intrinsic_rewards:
                reward = intrinsic + extrinsic
                policy.saved_rewards.append(reward)
                running_reward += float(reward)
            # calculate true values using rewards from environment
            R = 0
            returns = [] # true values
            for r in policy.saved_rewards[::-1]:
                R = r * config.discount_factor * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
            # calculate loss
            policy_losses = [] # actor (policy) loss
            value_losses = [] # critic (value) loss
            for (log_prob, value), R in zip(policy.saved_actions, returns):
                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage)
                value_losses.append(policy_criterion(value.squeeze(1), torch.tensor([R])))
            actor_loss = torch.stack(policy_losses).sum()
            critic_loss = torch.stack(value_losses).sum()
            icm_loss = torch.stack(curiosity_losses).sum()
            # optimize model
            optimizer.zero_grad()
            combined_loss = actor_loss + critic_loss + icm_loss
            combined_loss.backward()
            optimizer.step()
            # TODO: move out of model
            del policy.saved_rewards[:]
            del policy.saved_actions[:]
            # diagnostics
            # avg score, avg reward, 
            losses['total'] += float(combined_loss)
            losses['actor'] += float(actor_loss)
            losses['critic'] += float(critic_loss)
            losses['icm'] += float(icm_loss)
            ave_score = running_score / (game + 1)
            ave_reward = running_reward / (game + 1)
            ave_total_loss = losses['total'] / (game + 1)
            ave_actor_loss = losses['actor'] / (game + 1)
            ave_critic_loss = losses['critic'] / (game + 1)
            ave_icm_loss = losses['icm'] / (game + 1)
            # top_guesses = dict(sorted(guesses.items(), key=lambda x: x[1], reverse=True)[:5])
            metrics = {
                'avg_score': round(ave_score, 6),
                'avg_reward': round(ave_reward, 6),
                'total_loss': round(ave_total_loss, 6),
                'actor_loss': round(ave_actor_loss, 6),
                'critic_loss': round(ave_critic_loss, 6),
                'icm_loss': round(ave_icm_loss, 6)
            }
            prefix = f'Epoch {epoch+1}/{config.n_epochs}:'
            suffix = f'games played, metrics: {metrics}'
            message = print_progress_bar(game, config.n_games_per_epoch, prefix=prefix, suffix=suffix)
        with open('output/log.txt', 'a') as log_file:
            log_file.write(message)
        torch.save(policy.state_dict(), 'output/policy.pth')
        torch.save(icm.state_dict(), 'output/icm.pth')
    
if __name__ == '__main__':
    main()