import math
import torch
import torch.nn as nn
import config
from dataset import get_data_loaders
from model import LSTM

_, test_loader, vocab_len = get_data_loaders()

model = LSTM(
    vocab_len, 
    input_size=config.input_size, 
    hidden_size=config.hidden_size, 
    num_layers=config.num_layers, 
    dropout=config.dropout
)
model = model.to(config.device)

criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load("shakespeare_lstm2.pth"))
model.eval()

with torch.no_grad():
    corr_val = 0
    total_val = 0
    total_loss = 0.0

    for xb, yb in test_loader:
        xb = xb.to(config.device)
        yb = yb.to(config.device)
        output, _ = model(xb)
        loss = criterion(output, yb)

        preds = output.argmax(dim=1)
        total_val += yb.size(0)
        corr_val += (preds == yb).sum().item()
        total_loss += loss.item()

avg_loss = total_loss / len(test_loader)
perplexity = math.exp(avg_loss)

print(f"accuracy = {corr_val/total_val*100}%")
print(f"perplexity = {perplexity}")