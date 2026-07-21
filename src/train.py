import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import get_data_loaders
from model import LSTM

print(f"Using device: {config.device}")

train_loader, test_loader, vocab_len = get_data_loaders()

model = LSTM(
    vocab_len, 
    input_size=config.input_size, 
    hidden_size=config.hidden_size, 
    num_layers=config.num_layers, 
    dropout=config.dropout
)
model = model.to(config.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=config.weight_decay)

best_val_loss = float('inf')

for epoch in range(config.epochs):
    model.train()
    epoch_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(config.device)
        yb = yb.to(config.device)
        optimizer.zero_grad()

        output, _ = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            output, _ = model(xb)
            loss = criterion(output, yb)
            val_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)

    print(f"epoch = {epoch+1} & train_loss = {epoch_loss/len(train_loader)} & val_loss = {val_loss/len(test_loader)}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "shakespeare_lstm2.pth")
        print(f"new best model saved(val_loss={avg_val_loss})")