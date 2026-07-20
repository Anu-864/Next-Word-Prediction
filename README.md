# Shakespeare Text Generation with LSTMs

Two LSTM-based text generation models trained on the Tiny Shakespeare dataset — one operating at the **character level**, one at the **word level** — built to understand the full pipeline from raw text to trained model, and to compare how tokenization granularity affects difficulty, overfitting, and results.

## Contents

- `LSTM.ipynb` — full notebook: data preprocessing, both models, training, evaluation
- `shakespeare_lstm1.pth` — trained character-level model weights
- `shakespeare_lstm2.pth` — trained word-level model weights (best checkpoint by validation loss)
- `tiny-shakespeare.txt` — training data

## Dataset

[Tiny Shakespeare](https://github.com/karpathy/char-rnn) — ~1.1 million characters of Shakespeare's plays, lowercased and newline-collapsed during preprocessing.

---

## Model 1: Character-Level LSTM

Predicts the next **character** given the previous 100 characters.

**Architecture:** Embedding → 2-layer LSTM (hidden size 256) → Linear → softmax over vocabulary

| Setting | Value |
|---|---|
| Vocabulary size | ~35 unique characters |
| Sequence length | 100 |
| Batch size | 256 |
| Epochs | 20 |
| Optimizer | Adam |

**Results (held-out test set):**
| Metric | Value |
|---|---|
| Accuracy | 60.04% |
| Perplexity | 3.57 |

A perplexity of 3.57 means the model is typically choosing among just a handful of plausible next characters — a strong result, though expected for character-level prediction, which is an inherently easier task than word-level (far fewer possible outputs, more redundant structure like consistent spelling).

---

## Model 2: Word-Level LSTM

Predicts the next **word** given the previous *N* words. Same architecture family as the character model, extended to handle a much larger vocabulary.

### Key differences from the character model

**Tokenization:** text is split on whitespace into words rather than characters. Raw vocabulary came out to **23,641 unique words** — words appearing fewer than 2 times were folded into a single `<UNK>` token, reducing the effective vocabulary to **10,113**.

**Why this matters:** with ~200,000 word-tokens total spread across 10,000+ possible classes, this is a fundamentally harder learning problem than the character model, which only had ~35 classes to choose from over 1.1 million tokens. This shows up directly in the results below.

### Experiments and overfitting

An initial run (sequence length 20, no dropout, no validation tracking) trained for 25 epochs and looked fine by training loss alone (dropped steadily to ~3.3) — but evaluating afterward revealed severe overfitting: test-set loss corresponded to a perplexity of **3380**, barely better than random guessing (`ln(vocab_size) ≈ 9.2` nats, vs. the model's actual test loss of ~8.1 nats). The model had memorized training sequences without learning anything that generalized.

Fixes applied:
- Added **dropout** between LSTM layers
- Reduced **sequence length** (more training examples from the same text)
- Added **per-epoch validation loss tracking**, so overfitting is visible during training rather than only discovered afterward
- Added **checkpointing**: the model is saved only when validation loss improves, so the final saved weights come from the best-performing epoch rather than simply the last one

### Sequence length comparison

| Sequence length | Test accuracy | Test perplexity |
|---|---|---|
| 20 (no fixes above) | ~10.5% | 3380 |
| 10 | 12.82% | 321.4 |
| 5 | 12.55% | 318.1 |

Shortening the context window from 20 to 10 words produced a dramatic improvement (more training examples, plus dropout). Going further to 5 words made almost no additional difference — suggesting the model was already near its ceiling for this dataset size and vocabulary, rather than sequence length being the dominant remaining bottleneck.

**Final reported model:** sequence length 10, best checkpoint by validation loss.

| Metric | Value |
|---|---|
| Accuracy | 12.82% |
| Perplexity | 321.4 |

### Honest interpretation of these numbers

12.82% exact-match accuracy sounds low in isolation, but the model is choosing from **10,113 possible words** — random guessing would be correct roughly 0.01% of the time, so this is over 1,000x better than chance. Word-level language modeling on a dataset this small (~200K tokens) is a genuinely hard problem; modern word/subword-level language models are typically trained on datasets many orders of magnitude larger. This result reflects a real, expected data-size ceiling rather than a implementation flaw — the character-level model's much higher accuracy (60%) is a fair comparison only once you account for its far smaller output space (35 vs. 10,113 classes).

---

## Running this project

```bash
pip install torch torchvision pandas numpy
jupyter notebook LSTM.ipynb
```

Run cells top to bottom. Training data (`tiny-shakespeare.txt`) is included in this repo.

## Limitations / what I'd improve with more time

- Validation and test sets are currently the same split (no separate held-out validation set), so per-epoch monitoring and final evaluation use the same data — a stricter setup would use a three-way train/val/test split
- No text-generation sampling loop yet (temperature/top-k sampling) — currently the models only do one-step-ahead prediction; generating longer passages is the natural next step
- Word-level vocabulary could be further reduced with stemming/lemmatization, or extended with subword tokenization (e.g. BPE) to reduce the `<UNK>` rate without exploding vocabulary size
- Training on a larger corpus would be the single biggest lever for improving word-level results.
