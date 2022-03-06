import string
import pandas as pd

import config

def read_text_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

def word_frequencies():
    words = read_text_file('data/possible_words.txt')
    word_freqs = read_text_file('data/word_frequency.txt')
    word_freqs = [(wf.split(' ')[0], float(wf.split(' ')[1])) for wf in word_freqs]
    word_freqs = [(w, f) for w, f in word_freqs if w in words]
    min_freq = min([freq for _, freq in word_freqs if freq > 0])
    word_freqs = [(word, freq) if freq > 0 else (word, min_freq) for word, freq in word_freqs]
    word_freqs = pd.DataFrame(word_freqs, columns=['word','freq'])
    sum_freq = sum(word_freqs.freq)
    word_freqs['freq'] = word_freqs['freq'] / sum_freq
    word_data = [(i+1, word) for i, word in enumerate(words)]
    df = pd.DataFrame(word_data, columns=['index', 'word'])
    df = df.merge(word_freqs, on='word', how='inner')
    return df

df = word_frequencies()
words = df[:config.num_words].word.values.tolist()
# word index dictionaries
word2idx = {word: i+1 for i, word in enumerate(words)}
idx2word = {i+1: word for i, word in enumerate(words)}
# letter index dictionaries
letter2idx = {letter: i+1 for i, letter in enumerate(string.ascii_lowercase)}
idx2letter = {i+1: letter for i, letter in enumerate(string.ascii_lowercase)}