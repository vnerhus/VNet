from rnn import lstm
import os

# -- TEXT IMPORT -- #
def text_import(file_path):
    input_file = os.path.join(file_path)
    with open(input_file, "r", errors='ignore') as f:
        text = f.read()
        text = text[81:]
    return text

text = text_import('./data/simpsons/moes_tavern_lines.txt')










network = lstm(text)
network.setHyperparameters(hyperparameters = {
    "num_epochs": 200,   # Number of training epochs
    "learning_rate": 0.01, # Learning rate
    "batch_size": 128,      # Size of each batch
    "sequence_length": 16,  # Length of sequence
    "embed_dim": 128,       # Embedding dimension size
    "lstm_size": 128,       # Lstm size
    "stack_size": 1,        # Number of stacked LSTM-cells
})
network.train()
network.generate(prime_words=("homer_simpson:", "moe_szyslak:"))
