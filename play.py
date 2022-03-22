import config
import words
from wordle import Wordle

env = Wordle()
# env.reset('abide') # test 'speed'

def color(clue):
    if clue == 'G':
        return '🟩'
    elif clue == 'Y':
        return '🟨'
    elif clue == 'B':
        return '⬛️'
    else:
        return '⬜️'

for i in range(config.num_rounds_per_game):
    print(f'-- Round {i+1} --\n')
    while True:
        print('State Input:')
        print('- keyboard:', ', '.join([f'{c}={color(env.keyboard_state[c])}' for c in env.keyboard_state.keys()]))
        print('- position:', env.position_state)
        guess = input('\nGuess: ')
        guess = guess.lower()
        if len(guess) != 5:
            print('Error: guess needs to be a 5 letter word\n')
        elif guess not in words.word2idx.keys():
            print('Error: invalid 5 letter word\n')
        else:
            break
    _, clue_word, done = env.step(guess)
    print(f'Clue: {"".join([color(clue) for clue in list(clue_word)])}\n')
    if done:
        break
# TODO: MAKE SURE THE BOT IS PREDICTING THIS
print('End State')
print('- keyboard:', ', '.join([f'{c}={color(env.keyboard_state[c])}' for c in env.keyboard_state.keys()]))
print('- position:', env.position_state)
print(f'\nAnswer: {env.answer}')
print(f'Score: {env.score}\n')


