n_epochs = 10000
n_games_per_epoch = 100
n_rounds_per_game = 6

word_embedding_size = 128
letter_embedding_size = 16
num_features = 1024

sample_by_freq = False
debug_actions = False
num_words = 100

learning_rate = 3e-4
discount_factor = 0.9
gae_coeff = 1.0
entropy_coeff = 1e-4
intrinsic_coeff = 0.2
curiosity_coeff = 0.2
a2c_loss_coeff = 0.1