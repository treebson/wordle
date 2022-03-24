# Wordle
num_rounds_per_game = 6
word_limit = 100
sample_by_freq = False
# Training
random_first_word = False
num_epochs = 10000
num_games_per_epoch = 100
# Neural net
input_size = 334
num_features = 512
# Hyperparameters
learning_rate = 3e-4
discount_factor = 0.9
gae_coeff = 0.25
entropy_coeff = 0.01
intrinsic_coeff = 0.2
curiosity_coeff = 0.2
a2c_loss_coeff = 1.0
