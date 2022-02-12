import random

print('\n---- WORDLE ----\n')

def read_words(file_name):
    with open(file_name, 'r') as f:
        words = f.readlines()
    words = [word.strip() for word in words]
    return words

allowed_words = read_words('allowed_words.txt')

class Wordle:
    def __init__(self, secret):
        self.secret = secret

    def play(self, agent):
        score = 0
        match = False
        while score < 6:
            score += 1
            state = None
            guess = agent.guess(state)
            clue = self.check_guess(guess)
            if clue == '11111':
                match = True
                break
        score = score if match else 7
        return score

    def check_guess(self, guess):
        clue = ''
        for i in range(5):
            if guess[i] == self.secret[i]:
                clue += '1'
            elif guess[i] in self.secret:
                clue += '2'
            else:
                clue += '0'
        return clue


class Agent:
    def guess(self, state):
        return random.choice(allowed_words)

    def train(self):
        n_words = len(allowed_words)
        seed = 42
        sum_score = 0
        shuffled_words = allowed_words.copy()
        random.Random(seed).shuffle(shuffled_words)
        for i, secret in enumerate(shuffled_words):
            game = Wordle(secret)
            score = game.play(agent)
            sum_score += score
        print(sum_score/n_words)

agent = Agent()
agent.train()