# Word-Level LSTM Text Generator (Shakespeare)

A word-level language model built from scratch in PyTorch that learns to predict the next word in a sequence, trained on the Tiny Shakespeare corpus.

## Overview

This project implements a next-word prediction model using an LSTM (Long Short-Term Memory) network. Given a sequence of 10 words, the model predicts the most likely word to follow. The pipeline covers the full workflow: raw text ingestion, cleaning, tokenization, vocabulary construction, sequence generation, model design, training, and evaluation.

## Dataset

- **Source:** Tiny Shakespeare (`tiny-shakespeare.txt`)
- **Size:** 1,115,394 characters of raw text
- Loaded and read directly in a Colab environment using a file upload workflow.

## Preprocessing

1. **Normalization** — text lowercased and newline characters collapsed into single spaces using regex (`\n+` → `" "`), reducing the corpus to 1,108,171 characters.
2. **Tokenization** — whitespace-based word splitting (`text.split()`).
3. **Vocabulary construction** — a `Counter` was used to compute word frequencies. Words occurring fewer than 2 times were excluded and mapped to a single `<UNK>` token to keep the vocabulary manageable. This produced a vocabulary of **10,113 unique tokens**.
4. **Word ↔ index mappings** — `word_to_index` and `index_to_word` dictionaries built by enumerating the vocabulary.
5. **Encoding** — the full token stream converted into integer indices, falling back to `<UNK>` for out-of-vocabulary words.
6. **Sequence generation** — a sliding window of length 10 was used to build `(sequence, target)` pairs, where each sequence of 10 consecutive word indices is paired with the word that immediately follows it.

## Model Architecture

A custom `LSTM` class built on top of `nn.Module`:

| Component | Configuration |
|---|---|
| Embedding layer | `vocab_len → 128` dimensions |
| LSTM layers | 2 stacked layers, hidden size 256 |
| Dropout | 0.4 (regularization between LSTM layers) |
| Output layer | Fully connected, `256 → vocab_len` |
| Batching | `batch_first=True` |

**Forward pass:** input tokens → embedding lookup → LSTM → take the hidden state of the final timestep → linear projection to vocabulary-sized logits.

## Training Setup

- **Loss function:** `CrossEntropyLoss`
- **Optimizer:** Adam, learning rate default, weight decay `1e-4`
- **Train/test split:** 80% / 20%
- **Batch size:** 256 (train and test)
- **Epochs:** 25
- **Device:** CUDA (T4 GPU), with automatic CPU fallback
- Best model checkpointed based on lowest validation loss (`shakespeare_lstm2.pth`)

## Results

| Metric | Value |
|---|---|
| Final training loss (epoch 25) | 5.14 |
| Best validation loss | 5.77 (epoch 17) |
| Next-word accuracy | 12.79% |
| Perplexity | 320.1 |

**Training curve:** training loss decreased steadily across all 25 epochs (6.77 → 5.14), while validation loss plateaued early and improved only marginally (6.53 → 5.77), suggesting the model's capacity/training regime is close to saturating what it can learn from this setup rather than the classic overfitting pattern of diverging train/val curves.


## How to Run

1. Upload `tiny-shakespeare.txt` (or point to your own text corpus).
2. Run preprocessing cells to clean text and build vocabulary.
3. Run training loop (25 epochs by default).
4. Best model weights are saved to `shakespeare_lstm2.pth`.
5. Load the saved weights and run the evaluation loop to compute accuracy and perplexity on the held-out test set.

## Tech Stack

- Python, PyTorch
- `torch.utils.data` (`TensorDataset`, `DataLoader`)
- NumPy, Pandas (data handling utilities)
- Google Colab (GPU training environment)
