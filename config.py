import torch

seq_length = 10
input_size = 128
hidden_size = 256
num_layers = 2
dropout = 0.4
weight_decay = 1e-4
batch_size = 256
epochs = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")