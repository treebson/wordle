import random
import math
import string
import numpy as np
import pandas as pd
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim

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
# potentially log frequency
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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Memory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    # save a transition
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.word_embedding = nn.Embedding(n_words+1, 64)
        self.letter_embedding = nn.Embedding(n_letters+1, 8)
        self.hidden1 = nn.Linear(545, 512)
        self.relu1 = nn.ReLU()
        self.output = nn.Linear(512, n_words)
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
        X = np.zeros((N_ROUNDS_PER_GAME - 1, 11))
        for i in range(len(self.clues)):
            # encode index
            guess = self.guesses[i]
            word_idx = word2idx[guess]
            letter_idxs = [letter2idx[letter] for letter in guess]
            clues = [int(clue) for clue in self.clues[i]]
            X[i] = [word_idx] + letter_idxs + clues
        X = torch.from_numpy(X)
        X = X.type('torch.IntTensor')
        return X

    def step(self, action):
        self.score += 1
        if self.score < N_ROUNDS_PER_GAME:
            guess = idx2word[action]
            clue = self.check(guess)
            self.guesses.append(guess)
            self.clues.append(clue)
            done = clue == '33333'
            next_state = self.state()
            reward = sum([int(x) for x in list(clue)])
            
            reward = torch.tensor([[reward]])
            return reward, next_state, done
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

    def reset(self):
        self.secret = np.random.choice(df.word, p=df.freq)
        self.guesses = []
        self.clues = []
        self.score = 0
        return

# Hyperparameters
# TODO: tweak
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
N_EPOCHS = 1
N_GAMES_PER_EPOCH = 100
N_ROUNDS_PER_GAME = 6

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = Memory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # explore (random)
    if sample > eps_threshold:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].item() + 1
    # exploit (net)
    else:
        action = random.randrange(n_words) + 1
    action = torch.tensor([[action]], dtype=torch.int64)
    return action

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions)) # transpose batch

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    # TODO: is gradient clipping necessary?
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return float(loss)

# main training loop
env = Wordle()
for e in range(N_EPOCHS):
    for g in range(N_GAMES_PER_EPOCH):
        env.reset()
        state = env.state()
        # play game
        for score in range(N_ROUNDS_PER_GAME):
            action = select_action(state)
            reward, next_state, done = env.step(action.item())
            memory.push(state, action, next_state, reward)
            state = next_state
            if done:
                break          
        # train
        loss = optimize_model()
        # update target net every N games
        if g % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())