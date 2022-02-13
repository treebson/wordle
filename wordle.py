'''
IDEAS:
* sample words by frequency
* train set: test frequency
* use all words as test set

'''

import os
import pickle
import random
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed = 42
random.seed(42)

def read_words(file_name):
    with open(file_name, 'r') as f:
        words = f.readlines()
    words = [word.strip() for word in words]
    return words

allowed_words = read_words('allowed_words.txt')
word2idx = {word: i+1 for i, word in enumerate(allowed_words)}
idx2word = {i+1: word for i, word in enumerate(allowed_words)}
word2idx[0] = None
n_words = len(allowed_words)

letter2idx = {letter: i+1 for i, letter in enumerate(string.ascii_lowercase)}
idx2letter = {i+1: letter for i, letter in enumerate(string.ascii_lowercase)}
letter2idx[0] = None
n_letters = len(string.ascii_lowercase)

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
        self.word_embedding = nn.Embedding(n_words+1, 32)
        self.letter_embedding = nn.Embedding(n_letters+1, 8)
        self.hidden1 = nn.Linear(385, 512)
        self.relu1 = nn.ReLU()
        self.output = nn.Linear(512, n_words+1)
        self.softmax = nn.Softmax(dim = -1)

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
        return X

class Wordle:
    def __init__(self, secret, criterion, optimizer):
        self.secret = secret
        self.criterion = criterion
        self.optimizer = optimizer
        # guess/clues better?
        self.words = []
        self.states = []

    # TODO: update to play infinite rounds
    # TODO: seems like theres a key error 0 issue, this should not be predictable
    def self_play(self, agent):
        # play game (forward)
        score = 0.0
        match = False
        guessed = []
        while not match:
            score += 1
            X, label = self.encode()
            probabilities = agent(X)
            # get 1 hot encoding for guessed
            valid_indices = np.ones(n_words+1)
            valid_indices[0] = 0 # don't predict blank
            # TODO: this could be more efficient
            for i in guessed:
                valid_indices[i] = 0
            valid_indices = torch.from_numpy(valid_indices)
            # valid_probabilities
            probabilities = probabilities * valid_indices
            y_argmax = int(torch.argmax(probabilities))
            word = idx2word[y_argmax]
            guessed.append(y_argmax)
            # fetch best word
            state = self.check_guess(word)
            self.words.append(word)
            self.states.append(state)
            # only store buffer of 5 items
            if len(self.words) > 5:
                self.words = self.words[-5:]
                self.states = self.states[-5:]
            if state == '33333':
                match = True
        # encode score
        target = torch.tensor([[1.0]], requires_grad=True)
        actual = torch.tensor([[score]], requires_grad=True)
        # backward + optimize
        optimizer.zero_grad()
        loss = criterion(target, actual)
        loss.backward()
        optimizer.step() 
        return agent, score, float(loss)

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
        y = torch.flatten(y)
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

# MAIN

agent = Agent()
criterion = nn.MSELoss()
optimizer = optim.Adam(agent.parameters(), lr=3e-4)
# optimizer = optim.SGD(agent.parameters(), lr=0.001, momentum=0.9)

# train
n_epochs = 10
for e in range(n_epochs):
    # shuffle words
    shuffled_words = allowed_words.copy()
    random.shuffle(shuffled_words)
    # diagnostics
    running_score = 0
    running_loss = 0
    # play wordle
    # TODO: Don't play for every single word...this should be the test set
    #       Instead perhaps sample from natural word occurence frequency
    for i, secret in enumerate(shuffled_words):
        game = Wordle(secret, criterion, optimizer)
        agent, score, loss = game.self_play(agent)
        running_score += score
        running_loss += loss
        avg_score = running_score / (i+1)
        print_progress_bar(i, n_words, prefix=f'Training epoch {e+1}/{n_epochs}: ', suffix=f'loss={loss:.6f}, ave_score={running_score/(i+1):.3f}')

# save
torch.save(agent.state_dict(), 'agent.pth')
