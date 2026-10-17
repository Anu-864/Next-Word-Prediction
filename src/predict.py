import os
import re
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F

from src.model import LSTM
import config

class NextWordPredictor:
    _instance = None

    def __init__(
        self,
        model_path: str = "shakespeare_lstm.pth",
        vocab_path: str = "vocab.json",
        text_corpus_path: str = "tiny-shakespeare.txt",
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_length = config.seq_length  # 10
        self.input_size = config.input_size  # 128
        self.hidden_size = config.hidden_size  # 256
        self.num_layers = config.num_layers  # 2
        self.dropout = config.dropout  # 0.4

        # Load or generate vocabulary
        self.vocab, self.word_to_index, self.index_to_word, self.top_words = self._load_or_build_vocab(
            vocab_path, text_corpus_path
        )
        self.vocab_len = len(self.vocab)
        self.unk_idx = self.word_to_index.get("<UNK>", 0)

        # Initialize and load model
        self.model = LSTM(
            vocab_len=self.vocab_len,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        resolved_model_path = self._resolve_model_path(model_path)
        if resolved_model_path and os.path.exists(resolved_model_path):
            state_dict = torch.load(resolved_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
        else:
            self.is_loaded = False

    def _resolve_model_path(self, model_path: str) -> Optional[str]:
        candidates = [
            model_path,
            "shakespeare_lstm.pth",
            "shakespeare_lstm2.pth",
            "shakespeare_lstm1.pth",
            os.path.join(os.path.dirname(__file__), "..", "shakespeare_lstm.pth")
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    def _load_or_build_vocab(self, vocab_path: str, text_corpus_path: str):
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                vocab = data["vocab"]
                w2i = data["word_to_index"]
                i2w = {int(i): w for w, i in w2i.items()}
                top_words = data.get("top_words", [])
                return vocab, w2i, i2w, top_words
            except Exception:
                pass

        # Build from corpus
        if not os.path.exists(text_corpus_path):
            raise FileNotFoundError(f"Corpus file '{text_corpus_path}' not found.")

        with open(text_corpus_path, "r", encoding="UTF-8") as f:
            text = f.read()

        text = text.lower()
        text = re.sub(r"\n+", " ", text)
        tokens = text.split()

        word_counts = Counter(tokens)
        vocab = ["<UNK>"] + sorted([w for w, c in word_counts.items() if c >= 2])
        w2i = {word: idx for idx, word in enumerate(vocab)}
        i2w = {idx: word for idx, word in enumerate(vocab)}
        top_words = word_counts.most_common(50)

        # Cache vocab
        try:
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump({
                    "vocab": vocab,
                    "word_to_index": w2i,
                    "top_words": top_words
                }, f)
        except Exception:
            pass

        return vocab, w2i, i2w, top_words

    def tokenize_and_encode(self, text: str) -> Tuple[List[str], List[int], List[Dict]]:
        """Tokenize input text and return (tokens, token_indices, token_debug_info)."""
        clean_text = text.lower().strip()
        tokens = clean_text.split() if clean_text else []
        token_indices = [self.word_to_index.get(w, self.unk_idx) for w in tokens]

        # Inspect the window of 10 tokens fed into LSTM
        if len(token_indices) < self.seq_length:
            pad_count = self.seq_length - len(token_indices)
            window_tokens = ["<PAD>"] * pad_count + tokens
            window_indices = [self.unk_idx] * pad_count + token_indices
            window_status = ["padded"] * pad_count + [
                "known" if w in self.word_to_index else "oov" for w in tokens
            ]
        else:
            window_tokens = tokens[-self.seq_length :]
            window_indices = token_indices[-self.seq_length :]
            window_status = [
                "known" if w in self.word_to_index else "oov" for w in window_tokens
            ]

        debug_info = [
            {"word": w, "index": idx, "status": status}
            for w, idx, status in zip(window_tokens, window_indices, window_status)
        ]

        return tokens, window_indices, debug_info

    def predict_next_word(
        self,
        prompt: str,
        top_k: int = 5,
        filter_unk: bool = True,
        temperature: float = 1.0
    ) -> Dict:
        """
        Predict the top-k next words for a given prompt.
        Returns a dict with tokens, window info, and predictions list.
        """
        tokens, window_indices, debug_info = self.tokenize_and_encode(prompt)
        x = torch.tensor([window_indices], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, _ = self.model(x)  # shape: [1, vocab_len]
            logits = logits[0]  # shape: [vocab_len]

            if filter_unk:
                logits[self.unk_idx] = -float("inf")

            temp = max(float(temperature), 0.01)
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=-1)

            effective_k = min(top_k, self.vocab_len)
            top_probs, top_indices = torch.topk(probs, effective_k)

            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                w_idx = idx.item()
                word_str = self.index_to_word.get(w_idx, "<UNK>")
                p_val = prob.item()
                predictions.append({
                    "word": word_str,
                    "index": w_idx,
                    "probability": p_val,
                    "percent": round(p_val * 100, 2)
                })

        return {
            "prompt": prompt,
            "raw_tokens": tokens,
            "token_count": len(tokens),
            "window_debug": debug_info,
            "predictions": predictions,
            "top_word": predictions[0]["word"] if predictions else ""
        }

    def generate_sequence(
        self,
        seed_text: str,
        num_words: int = 15,
        temperature: float = 0.8,
        top_k: int = 20,
        filter_unk: bool = True,
        strategy: str = "temperature"  # 'temperature', 'greedy', 'top_k'
    ) -> Dict:
        """
        Generates a sequence of words starting from seed_text.
        """
        current_text = seed_text.strip()
        generated_words = []

        for _ in range(num_words):
            _, window_indices, _ = self.tokenize_and_encode(current_text)
            x = torch.tensor([window_indices], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits, _ = self.model(x)
                logits = logits[0]

                if filter_unk:
                    logits[self.unk_idx] = -float("inf")

                if strategy == "greedy":
                    chosen_idx = torch.argmax(logits).item()
                elif strategy == "top_k":
                    k = min(top_k, self.vocab_len)
                    top_vals, top_inds = torch.topk(logits, k)
                    probs = F.softmax(top_vals / max(temperature, 0.01), dim=-1)
                    sample = torch.multinomial(probs, 1).item()
                    chosen_idx = top_inds[sample].item()
                else:  # temperature sampling
                    probs = F.softmax(logits / max(temperature, 0.01), dim=-1)
                    chosen_idx = torch.multinomial(probs, 1).item()

                next_word = self.index_to_word.get(chosen_idx, "<UNK>")
                generated_words.append(next_word)
                current_text = f"{current_text} {next_word}"

        return {
            "seed": seed_text,
            "generated_words": generated_words,
            "full_text": current_text
        }

    def search_vocab(self, query: str, limit: int = 20) -> List[Dict]:
        """Search vocabulary for words containing query substring."""
        q = query.lower().strip()
        if not q:
            return []
        matches = []
        for word, idx in self.word_to_index.items():
            if q in word:
                matches.append({"word": word, "index": idx})
                if len(matches) >= limit:
                    break
        return matches


# Singleton instance helper
_PREDICTOR: Optional[NextWordPredictor] = None

def get_predictor() -> NextWordPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = NextWordPredictor()
    return _PREDICTOR
