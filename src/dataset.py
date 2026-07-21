import re
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import config

def get_data_loaders(data_path="tiny-shakespeare.txt"):
    with open(data_path, "r", encoding="UTF-8") as f:
        text = f.read()

    text = text.lower()
    text = re.sub(r"\n+", " ", text)

    tokens = text.split()

    word_counts = Counter(tokens)
    vocab = ["<UNK>"] + sorted([w for w, c in word_counts.items() if c >= 2])
    vocab_len = len(vocab)

    word_to_index = {}
    for idx, word in enumerate(vocab):
        word_to_index[word] = idx

    index_to_word = {idx: word for idx, word in enumerate(vocab)}

    encoded_text = []
    for word in tokens:
        encoded_text.append(word_to_index.get(word, word_to_index["<UNK>"]))

    sequences = []
    targets = []

    for i in range(len(encoded_text) - config.seq_length):
        sequences.append(encoded_text[i : i + config.seq_length])
        targets.append(encoded_text[i + config.seq_length])

    X_tensor = torch.tensor(sequences, dtype=torch.long)
    y_tensor = torch.tensor(targets, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    return train_loader, test_loader, vocab_len