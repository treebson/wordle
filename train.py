import random
import math
import string
import os
import shutil
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from wordle import Wordle

import config
import data
import agent

# fix manual seed
torch.manual_seed(42)

# instantiate agent
a2c = agent.ActorCritic()
icm = agent.ICM()
a2c.train()
icm.train()

# define loss functions
inv_criterion = nn.CrossEntropyLoss()
fwd_criterion = nn.MSELoss()

# define ptimizer
optimizer = optim.Adam(list(a2c.parameters()) + list(icm.parameters()), lr=config.learning_rate)

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

# reset output directory
def reset_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

# training loop
def main():
    reset_directory('output')
    env = Wordle()
    seen_words = set()
    for epoch in range(config.n_epochs):
        # diagnostic variables
        start = time.time()
        running_reward = 0.0
        running_score = 0.0
        losses = {'total': 0.0, 'actor': 0.0, 'critic': 0.0, 'entropy': 0.0, 'a2c': 0.0, 'icm': 0.0}
        guesses = {}
        seen_word_count = len(seen_words)
        # begin epoch
        for game in range(config.n_games_per_epoch):
            # reset game environment
            state = env.reset()
            # game metrics
            log_policies = []
            values = []
            rewards = [] 
            entropies = []
            inv_losses = []
            fwd_losses = []
            # play a game
            for score in range(config.n_rounds_per_game):
                # forward - actor critic
                logits, value = a2c(state)
                policy = F.softmax(logits, dim=1)
                log_policy = F.log_softmax(logits, dim=1)
                entropy = -(policy * log_policy).sum(1, keepdim=True)
                # determine next action
                m = Categorical(policy)
                action = m.sample()
                action_onehot = F.one_hot(action, num_classes=data.n_words).float()
                action_index = action.item() + 1
                # save guess for logging
                guess = data.idx2word[action_index]
                if guess in guesses:
                    guesses[guess] += 1
                else:
                    guesses[guess] = 1
                seen_words.add(guess)
                # step environment
                next_state, done = env.step(action_index)
                # exit if correct guess
                if done:
                    break
                # intrinsic reward
                pred_logits, pred_phi, phi = icm(state, next_state, action_onehot)
                inv_loss = inv_criterion(pred_logits, action_onehot)
                fwd_loss = fwd_criterion(pred_phi, phi) / 2
                intrinsic_reward = config.intrinsic_coeff * fwd_loss.detach()
                # add to lists
                values.append(value)
                rewards.append(intrinsic_reward)
                log_policies.append(log_policy[0, action])
                entropies.append(entropy)
                inv_losses.append(inv_loss)
                fwd_losses.append(fwd_loss)
                # update state
                state = next_state
            # game finished
            running_score += env.score
            # calculate reward (intrinsic + extrinsic)
            extrinsic = config.n_rounds_per_game - env.score
            extrinsic = torch.tensor([extrinsic], dtype=torch.float)
            rewards = [intrinsic + extrinsic for intrinsic in rewards]
            running_reward += float(sum(rewards))
            # calculate loss
            _, R = a2c(state)
            gae = torch.zeros((1, 1), dtype=torch.float)
            actor_loss = 0
            critic_loss = 0
            entropy_loss = 0
            icm_loss = 0
            next_value = R
            for value, log_policy, reward, entropy, inv, fwd in list(zip(values, log_policies, rewards, entropies, inv_losses, fwd_losses))[::-1]:
                gae = gae * config.discount_factor * config.gae_coeff
                gae = gae + reward + config.discount_factor * next_value.detach() - value.detach()
                next_value = value
                actor_loss = actor_loss + log_policy * gae
                R = R * config.discount_factor + reward
                critic_loss = critic_loss + (R - value) ** 2 / 2
                entropy_loss = entropy_loss + entropy
                icm_loss = icm_loss + (1 - config.curiosity_coeff) * inv + config.curiosity_coeff * fwd
            a2c_loss = config.a2c_loss_coeff * (-actor_loss + critic_loss - config.entropy_coeff * entropy_loss)
            total_loss = a2c_loss + icm_loss
            # optimize model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # print diagnostics
            losses['total'] += float(total_loss)
            losses['actor'] += float(actor_loss)
            losses['critic'] += float(critic_loss)
            losses['entropy'] += float(entropy_loss)
            losses['a2c'] += float(a2c_loss)
            losses['icm'] += float(icm_loss)
            games_played = game + 1
            ave_score = running_score / games_played
            ave_reward = running_reward / games_played
            ave_total_loss = losses['total'] / games_played
            ave_actor_loss = losses['actor'] / games_played
            ave_critic_loss = losses['critic'] / games_played
            ave_entropy_loss = losses['entropy'] / games_played
            ave_a2c_loss = losses['a2c'] / games_played
            ave_icm_loss = losses['icm'] / games_played
            top_words = list(dict(sorted(guesses.items(), key=lambda x: x[1], reverse=True)[:3]).keys())
            new_words = len(seen_words) - seen_word_count
            time_taken = f'{(time.time() - start):.1f}'
            metrics = {
                'avg_score': round(ave_score, 3),
                'new_words': new_words,
                'top_words': top_words,
                'avg_reward': round(ave_reward, 4),
                'total_loss': round(ave_total_loss, 5),
                'actor_loss': round(ave_actor_loss, 5),
                'critic_loss': round(ave_critic_loss, 5),
                'entropy_loss': round(ave_entropy_loss, 5),
                'a2c_loss': round(ave_a2c_loss, 5),
                'icm_loss': round(ave_icm_loss, 5),    
            }

            prefix = f'Epoch {epoch+1}/{config.n_epochs}:'
            suffix = f'games played in {time_taken}s, metrics: {metrics}'
            message = print_progress_bar(game, config.n_games_per_epoch, prefix=prefix, suffix=suffix)
        # log final message of epoch
        with open('output/log.txt', 'a') as log_file:
            log_file.write(message)
        # save model weights
        torch.save(a2c.state_dict(), 'output/a2c.pth')
        torch.save(icm.state_dict(), 'output/icm.pth')
    
if __name__ == '__main__':
    main()