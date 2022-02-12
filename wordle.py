import random

print('\nWORDLE')

def read_words(file_name):
    with open(file_name, 'r') as f:
        words = f.readlines()
    words = [word.strip() for word in words]
    return words

allowed_words = read_words('allowed_words.txt')
possible_words = read_words('possible_words.txt')

secret = random.choice(possible_words)
print(secret)

def receive_guess():
    valid = False
    while not valid:
        guess = input('\n')
        if guess in allowed_words:
            return guess
        else:
            print('error: invalid word')


def give_clue(guess, secret):
    clues = ''
    for i in range(5):
        if guess[i] == secret[i]:
            clues += '1'
        elif guess[i] in secret:
            clues += '2'
        else:
            clues += '0'
    return clues

def play_wordle():
    score = 0
    match = False
    while score < 6:
        score += 1
        guess = receive_guess()
        clue = give_clue(guess, secret)
        print(clue)
        if clue == '11111':
            match = True
            break
    score = score if match else 7
    return score

score = play_wordle()
print(f'\nscore: {score}\n')
