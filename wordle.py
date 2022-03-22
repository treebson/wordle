import string
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import words
import config

yellow = 1
green = 3
max_reward = 5 * green

clue_state = {'?' : 0, 'B': 1, 'Y': 2, 'G': 3}

class Wordle:
    def __init__(self):
        self.answer = None
        self.guesses = []
        self.clues = []
        self.score = 1
        self.keyboard_state = {c: '?' for c in string.ascii_lowercase}
        self.position_state = [string.ascii_lowercase for i in range(5)]
        self.action_mask = np.ones((1, words.num_words))
        self.reset()

    # retrieve game state as tensor
    def state(self):
        # one-hot encode keyboard state (26x4)
        x_keyboard = torch.tensor([clue_state[self.keyboard_state[c]] for c in string.ascii_lowercase])
        x_keyboard = F.one_hot(x_keyboard, num_classes=4)
        # multi-hot encode letters by position (5x26)
        x_position = [[1 if c in ps.lower() else 0 for c in string.ascii_lowercase] for ps in self.position_state]
        x_position = torch.tensor(x_position)
        return (x_keyboard, x_position)

    def step(self, guess):
        clue = self.check(guess)
        self.update_keyboard_state(guess, clue)
        self.update_position_state(guess, clue)
        self.update_action_mask(guess)
        self.guesses.append(guess)
        self.clues.append(clue)
        done = clue == ['G'] * 5
        if not done:
            self.score += 1
        return self.state(), clue, done

    def update_keyboard_state(self, guess, clue):
        ks = self.keyboard_state
        for i, (x, y) in enumerate(zip(list(guess), list(clue))):
            if clue_state[y] > clue_state[ks[x]]:
                ks[x] = y
        self.keyboard_state = ks

    def update_position_state(self, guess, clue):
        ps = self.position_state
        for i, (x, y) in enumerate(zip(list(guess), list(clue))):
            if y == 'G':
                for j in range(5):
                    ps[j] = ps[j].replace(x, '')
                    ps[j] = ps[j].replace(x.upper(), '')
                ps[i] = x.upper()
            elif y == 'B':
                for j in range(5):
                    ps[j] = ps[j].replace(x, '')
            else:
                for j in range(5):
                    ps[j] = ps[j].replace(x, x.upper())
                ps[i] = ps[i].replace(x, '')
                ps[i] = ps[i].replace(x.upper(), '')
        self.position_state = ps

    def update_action_mask(self, guess):
        idx = words.word2idx[guess] - 1
        self.action_mask[0, idx] = 0
        
    def check(self, guess):
        clue = []
        for i in range(5):
            letter = guess[i]
            if letter == self.answer[i]:
                clue.append('G')
            elif letter in self.answer:
                occurences_in_guess = guess.count(letter)
                occurences_in_answer = self.answer.count(letter)
                if occurences_in_guess > occurences_in_answer and guess[:i].count(letter) == occurences_in_answer:
                    clue.append('B')
                else:
                    clue.append('Y')
            else:
                clue.append('B')
        return clue

    def reward_clue(self, clue):
        reward = 0
        for x in list(clue):
            if x == 'Y':
                reward += yellow
            if x == 'G':
                reward += green
        reward = reward / max_reward
        return reward

    def reset(self, answer=None):
        if answer != None:
            self.answer = answer
        elif config.sample_by_freq:
            self.answer = np.random.choice(words.df.word, p=words.df.freq)
        else:
            self.answer = np.random.choice(words.words)
        self.guesses = []
        self.clues = []
        self.score = 1
        self.keyboard_state = {c: '?' for c in string.ascii_lowercase}
        self.position_state = [string.ascii_lowercase for i in range(5)]
        self.action_mask = np.ones((1, words.num_words))
        return self.state()
