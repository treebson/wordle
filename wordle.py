import string
import torch
import numpy as np
import pandas as pd

import data
import config

class Wordle:
    def __init__(self):
        self.secret = None
        self.guesses = []
        self.clues = []
        self.score = 0

    # Encode state into array
    def state(self):
        # 5 rows for each previous state
        # 11 columns - 1 word, 5 letters, 5 clues
        x = np.zeros((config.n_rounds_per_game - 1, 11))
        for i in range(len(self.clues)):
            # encode index
            guess = self.guesses[i]
            word_idx = data.word2idx[guess]
            letter_idxs = [data.letter2idx[letter] for letter in guess]
            clues = [int(clue) for clue in self.clues[i]]
            x[i] = [word_idx] + letter_idxs + clues
        x = torch.from_numpy(x).type(torch.int64)
        return x

    def step(self, action):
        self.score += 1
        if self.score < config.n_rounds_per_game:
            guess = data.idx2word[action]
            clue = self.check(guess)
            self.guesses.append(guess)
            self.clues.append(clue)
            done = clue == '33333'
            next_state = self.state()
            return next_state, clue, done
        else:
            return None, None, True

    # TODO: fix incomplete logic
    def check(self, guess):
        clue = ''
        for i in range(5):
            if guess[i] == self.secret[i]:
                clue += '3'
            elif guess[i] in self.secret:
                clue += '2'
            else:
                clue += '1'
        return clue

    def reward_clue(self, clue):
        reward = 0
        for x in list(clue):
            if x == '2':
                reward += 1
            if x == '3':
                reward += 3
        reward = reward * config.reward_clue_coeff
        return reward

    def reset(self):
        self.secret = np.random.choice(data.df.word, p=data.df.freq)
        self.guesses = []
        self.clues = []
        self.score = 0
        return self.state()