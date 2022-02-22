import torch
import torch.nn as nn

import config
import data

# deep q network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.word_embedding = nn.Embedding(data.n_words + 1, 64)
        self.letter_embedding = nn.Embedding(data.n_letters + 1, 8)
        self.hidden1 = nn.Linear(545, 512)
        self.relu1 = nn.ReLU()
        self.output = nn.Linear(512, data.n_words + 1)
        self.softmax = nn.Softmax(dim = -1)

     # forward propagate input
    def forward(self, X):
        X = X.view(-1, 5, 11)
        # words
        X_words = X[:, :, 0]
        X_words = self.word_embedding(X_words)
        X_words = torch.flatten(X_words, 1, -1)
        # letters
        X_letters = X[:, :, 1:6]
        X_letters = self.letter_embedding(X_letters)
        X_letters = torch.flatten(X_letters, 1, -1)
        # clue
        X_clue = X[:, :, 6:]
        X_clue = torch.flatten(X_clue, 1, -1)
        # combine
        X = torch.cat((X_words, X_letters, X_clue), -1)
        # hidden layers
        X = self.hidden1(X)
        X = self.relu1(X)
        # softmax output
        X = self.output(X)
        X = self.softmax(X)
        return X