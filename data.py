import string
import pandas as pd

def read_text_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

# words
allowed_words = read_text_file('data/allowed_words.txt')
word2idx = {word: i+1 for i, word in enumerate(allowed_words)}
idx2word = {i+1: word for i, word in enumerate(allowed_words)}
n_words = len(allowed_words)

# letters
letter2idx = {letter: i+1 for i, letter in enumerate(string.ascii_lowercase)}
idx2letter = {i+1: letter for i, letter in enumerate(string.ascii_lowercase)}
n_letters = len(string.ascii_lowercase)

# word frequencies
word_freqs = read_text_file('data/word_frequency.txt')
word_freqs = [(wf.split(' ')[0], float(wf.split(' ')[1])) for wf in word_freqs]
min_freq = min([freq for _, freq in word_freqs if freq > 0])
word_freqs = [(word, freq) if freq > 0 else (word, min_freq) for word, freq in word_freqs]
word_freqs = pd.DataFrame(word_freqs, columns=['word','freq'])
sum_freq = sum(word_freqs.freq)
word_freqs['freq'] = word_freqs['freq'] / sum_freq
word_data = [(i+1, word) for i, word in enumerate(allowed_words)]
df = pd.DataFrame(word_data, columns=['index', 'word'])
df = df.merge(word_freqs, on='word', how='inner')