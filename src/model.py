import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_len, input_size, hidden_size, num_layers, dropout=0.4):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_len, self.input_size)
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(self.hidden_size, vocab_len)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out, hidden