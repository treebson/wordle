'''
IDEAS:
* sample words by frequency
* train set: test frequency
* use all words as test set

PROBLEMS:
* need to speed up training somehow

'''

import os
import pickle
import random
import string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed = 42
random.seed(42)

def read_text_file(file_name):
    with open(file_name, 'r') as f:
        words = f.readlines()
    words = [word.strip() for word in words]
    return words

allowed_words = read_text_file('allowed_words.txt')
word2idx = {word: i+1 for i, word in enumerate(allowed_words)}
idx2word = {i+1: word for i, word in enumerate(allowed_words)}
word2idx[0] = None
n_words = len(allowed_words)
print(f'\nPlaying wordle with {n_words} known words.')

letter2idx = {letter: i+1 for i, letter in enumerate(string.ascii_lowercase)}
idx2letter = {i+1: letter for i, letter in enumerate(string.ascii_lowercase)}
letter2idx[0] = None
n_letters = len(string.ascii_lowercase)

# read word frequencies
word_freqs = read_text_file('word_frequency.txt')
word_freqs = [(wf.split(' ')[0], float(wf.split(' ')[1])) for wf in word_freqs]
min_freq = min([freq for _, freq in word_freqs if freq > 0])
word_freqs = [(word, freq) if freq > 0 else (word, min_freq) for word, freq in word_freqs]
df_word_freq = pd.DataFrame(word_freqs, columns=['word','freq'])
sum_freq = sum(df_word_freq.freq)
df_word_freq['freq'] = df_word_freq['freq'] / sum_freq
print(df_word_freq.sort_values('freq', ascending=False)[0:20])
print('...')

# join frequencies onto words
word_data = [(i+1, word) for i, word in enumerate(allowed_words)]
df_word = pd.DataFrame(word_data, columns=['index', 'word'])
df = df_word.merge(df_word_freq, on='word', how='inner')

'''
input: (5, 11)
- 5 rows for each prior prediction (init to -1)
- 11 columns:
    - 1 neuron for word guessed (passed into word embedding)
    - 5 neurons for each letter (passed into letter embedding)
    - 5 neurons for state (clues returned for each word, 0 = none, 1 = grey, 2 = yellow, 3 = green)

output, softmax of 12972 words
'''
class Agent(nn.Module):
    # define model elements
    def __init__(self):
        super(Agent, self).__init__()
        self.word_embedding = nn.Embedding(n_words+1, 64)
        self.letter_embedding = nn.Embedding(n_letters+1, 8)
        self.hidden1 = nn.Linear(545, 512)
        self.relu1 = nn.ReLU()
        self.output = nn.Linear(512, n_words+1)
        self.softmax = nn.Softmax(dim = 0)

     # forward propagate input
    def forward(self, X):
        # words
        X_words = X[:, 0]
        X_words = self.word_embedding(X_words)
        X_words = torch.flatten(X_words)
        # letters
        X_letters = X[:, 1:6]
        X_letters = self.letter_embedding(X_letters)
        X_letters = torch.flatten(X_letters)
        # state
        X_state = X[:, 6:]
        X_state = torch.flatten(X_state)
        # combine
        X = torch.cat((X_words, X_letters, X_state), 0)
        # hidden layers
        X = self.hidden1(X)
        X = self.relu1(X)
        # softmax output
        X = self.output(X)
        X = self.softmax(X)
        X = X.unsqueeze(0) # add additional dimension
        return X

class Wordle:
    def __init__(self, secret):
        self.secret = secret
        # guess/clues better?
        self.words = []
        self.states = []

    # TODO: seems like theres a key error 0 issue, this should not be predictable
    # TODO: create version that is actually limited ot 6 rounds
    def self_play(self, agent, optimizer, criterion):
        # play game (forward)
        score = 0.0
        match = False
        running_loss = 0
        while not match:
            score += 1
            X, label = self.encode()
            # forward + backward + optimize
            probs = agent(X)
            optimizer.zero_grad()
            loss = criterion(probs, label)
            loss.backward()
            optimizer.step()
            # valid_probabilities
            y_argmax = int(torch.argmax(probs))
            word = idx2word[y_argmax]
            # fetch best word
            state = self.check_guess(word)
            # only store buffer of 5 items
            self.words.append(word)
            self.states.append(state)
            if len(self.words) > 5:
                self.words = self.words[1:]
                self.states = self.states[1:]
            # exit game if correct word guessed
            if state == '33333':
                match = True
            running_loss += float(loss)
        return score, running_loss / score

    def encode(self):
        assert(len(self.words) == len(self.states))
        X = []
        for i in range(len(self.words)):
            row = []
            row.append(word2idx[self.words[i]])
            for l in list(self.words[i]):
                row.append(letter2idx[l])
            for s in self.states[i]:
                row.append(int(s))
            X.append(row)
        for _ in range(5 - len(self.words)):
            X.append([0] * 11)
        X = np.array(X)
        X = torch.from_numpy(X)
        y = torch.tensor([word2idx[self.secret]])
        # encode label
        y = nn.functional.one_hot(y, num_classes=n_words+1)
        y = y.type('torch.FloatTensor')
        return X, y

    def possible_words(self):
        possible_words = []
        for word, state in self.words.zip(self.states):
            print(word, state)

    def check_guess(self, guess):
        clue = ''
        for i in range(5):
            if guess[i] == self.secret[i]:
                clue += '3'
            elif guess[i] in self.secret:
                clue += '2'
            else:
                clue += '1'
        return clue


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

# hyperparameters
agent = Agent()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(agent.parameters())
n_epochs = 10
games_per_epoch = 1000

# training loop
print(f'\nPlaying self {n_epochs * games_per_epoch} times.')
for e in range(n_epochs):
    running_score = 0
    running_loss = 0
    for i in range(games_per_epoch):
        # sample word for game
        secret = np.random.choice(df.word, p=df.freq)
        # play wordle (forward)
        game = Wordle(secret)
        score, loss = game.self_play(agent, optimizer, criterion)
        # diagnostics
        running_score += score
        running_loss += loss
        avg_score = running_score / (i+1)
        print_progress_bar(i, games_per_epoch, prefix=f'Training epoch {e+1}/{n_epochs}: ', suffix=f"ave_score={running_score/(i+1):.2f}, loss={loss}, secret={secret}")
    # save
    torch.save(agent.state_dict(), 'agent.pth')