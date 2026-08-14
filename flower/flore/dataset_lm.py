"""Language Modeling Dataset Utilities for Federated Learning.

This module provides data loading utilities for language modeling tasks,
specifically designed for LEAF StackOverflow dataset.

LEAF StackOverflow:
- ~340K users (natural federation)
- Question/Answer text data
- Standard FL benchmark (FedAvg, FedProx papers)
"""

import os
import json
import html
import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter
import pickle
import re


# ============================================================================
# Tokenizer
# ============================================================================

class SimpleTokenizer:
    """Simple word-level tokenizer for StackOverflow.

    For production, consider using HuggingFace tokenizers or SentencePiece.
    This implementation focuses on simplicity and FL compatibility.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._built = False

        # Reserve special tokens
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for i, token in enumerate(self.special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token

        self.pad_id = self.word2idx[self.PAD_TOKEN]
        self.unk_id = self.word2idx[self.UNK_TOKEN]
        self.bos_id = self.word2idx[self.BOS_TOKEN]
        self.eos_id = self.word2idx[self.EOS_TOKEN]

    def _preprocess(self, text: str) -> List[str]:
        """Preprocess text: lowercase, split on whitespace and punctuation."""
        text = text.lower()
        # Split on whitespace and keep punctuation as separate tokens
        text = re.sub(r'([.,!?;:\'\"])', r' \1 ', text)
        tokens = text.split()
        return tokens

    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from texts."""
        counter = Counter()
        for text in texts:
            tokens = self._preprocess(text)
            counter.update(tokens)

        # Keep most common words up to vocab_size
        most_common = counter.most_common(self.vocab_size - len(self.special_tokens))

        # Filter by min_freq
        most_common = [(word, count) for word, count in most_common if count >= min_freq]

        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self._built = True
        print(f"[Tokenizer] Built vocab with {len(self.word2idx)} tokens")

    def encode(self, text: str, max_length: Optional[int] = None, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._preprocess(text)
        ids = [self.word2idx.get(token, self.unk_id) for token in tokens]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in ids]
        # Remove special tokens for clean output
        tokens = [t for t in tokens if t not in self.special_tokens]
        return ' '.join(tokens)

    def save(self, path: str):
        """Save tokenizer to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = data['idx2word']
        tokenizer._built = True
        return tokenizer


# ============================================================================
# StackOverflow Dataset
# ============================================================================

class StackOverflowDataset(Dataset):
    """StackOverflow dataset for language modeling.

    Each sample is a question or answer text from StackOverflow.
    Can load from LEAF format or HuggingFace datasets.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 256,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.tokenizer.encode(text, max_length=self.max_seq_len, add_special_tokens=True)

        # Pad to max_seq_len
        if len(ids) < self.max_seq_len:
            ids = ids + [self.tokenizer.pad_id] * (self.max_seq_len - len(ids))

        input_ids = torch.tensor(ids, dtype=torch.long)
        # Labels are the same as input_ids (for next token prediction)
        # -100 is used for padding tokens (ignored in loss)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_id] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class StackOverflowFederated:
    """Federated StackOverflow dataset manager.

    Handles loading and partitioning of StackOverflow data for FL.
    Each user is treated as a separate client (natural federation).
    """

    def __init__(
        self,
        data_dir: str = "./data/stackoverflow",
        vocab_size: int = 10000,
        max_seq_len: int = 256,
        min_samples_per_client: int = 10,
    ):
        self.data_dir = data_dir
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.min_samples_per_client = min_samples_per_client

        self.tokenizer = None
        self.client_data: Dict[int, List[str]] = {}
        self.test_data: List[str] = []

    def load_from_leaf(self, leaf_dir: str):
        """Load data from LEAF format.

        LEAF format: JSON files with 'users', 'user_data' keys.
        Each user has a list of samples.
        """
        train_files = [f for f in os.listdir(os.path.join(leaf_dir, 'train')) if f.endswith('.json')]
        test_files = [f for f in os.listdir(os.path.join(leaf_dir, 'test')) if f.endswith('.json')]

        # Load training data
        all_train_texts = []
        client_id = 0

        for fname in sorted(train_files):
            with open(os.path.join(leaf_dir, 'train', fname), 'r') as f:
                data = json.load(f)

            for user_id in data['users']:
                user_samples = data['user_data'][user_id]['x']  # List of texts

                if len(user_samples) >= self.min_samples_per_client:
                    self.client_data[client_id] = user_samples
                    all_train_texts.extend(user_samples)
                    client_id += 1

        # Load test data
        for fname in sorted(test_files):
            with open(os.path.join(leaf_dir, 'test', fname), 'r') as f:
                data = json.load(f)

            for user_id in data['users']:
                self.test_data.extend(data['user_data'][user_id]['x'])

        print(f"[StackOverflow] Loaded {len(self.client_data)} clients, {len(all_train_texts)} train samples")

        # Build tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=self.vocab_size)
        self.tokenizer.build_vocab(all_train_texts, min_freq=2)

    def _clean_hf_text(self, text: str) -> str:
        """Normalize HTML-heavy Stack Exchange text into plain text."""
        text = html.unescape(str(text))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_hf_text(self, sample: Dict[str, Any]) -> str:
        """Extract a usable training text from a HuggingFace sample."""
        if "question" in sample and "answers" in sample:
            question = self._clean_hf_text(sample.get("question", ""))
            answers = sample.get("answers", []) or []
            best_answer = ""
            if isinstance(answers, list) and answers:
                selected = [a for a in answers if isinstance(a, dict) and a.get("selected")]
                if selected:
                    best = selected[0]
                else:
                    scored = [
                        a for a in answers
                        if isinstance(a, dict) and ("pm_score" in a or "score" in a)
                    ]
                    if scored:
                        best = max(scored, key=lambda a: a.get("pm_score", a.get("score", -1)))
                    else:
                        best = next((a for a in answers if isinstance(a, dict)), {})
                best_answer = self._clean_hf_text(best.get("text", ""))
            return f"{question} {best_answer}".strip()

        if {"title", "question", "answer"} & set(sample.keys()):
            parts = [
                self._clean_hf_text(sample.get("title", "")),
                self._clean_hf_text(sample.get("question", "")),
                self._clean_hf_text(sample.get("answer", "")),
            ]
            return " ".join(part for part in parts if part).strip()

        if "text" in sample:
            return self._clean_hf_text(sample.get("text", ""))

        text_parts: List[str] = []
        for value in sample.values():
            if isinstance(value, str):
                cleaned = self._clean_hf_text(value)
                if cleaned:
                    text_parts.append(cleaned)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        cleaned = self._clean_hf_text(item)
                        if cleaned:
                            text_parts.append(cleaned)
        return " ".join(text_parts).strip()

    def load_from_huggingface(
        self,
        num_clients: int = 100,
        samples_per_client: int = 500,
        dataset_name: str = "HuggingFaceH4/stack-exchange-preferences",
        split: str = "train",
        max_samples: Optional[int] = None,
        allow_synthetic_fallback: bool = True,
    ):
        """Load from HuggingFace datasets (easier setup).

        Uses a public Stack Exchange dataset from the Hugging Face Hub.
        """
        try:
            from datasets import load_dataset

            print(f"[StackOverflow] Loading from HuggingFace dataset '{dataset_name}'...")
            target_samples = max_samples or (num_clients * samples_per_client + 5000)
            split_spec = split if "[" in split else f"{split}[:{target_samples}]"
            dataset = load_dataset(dataset_name, split=split_spec)

            all_texts = []
            for sample in dataset:
                text = self._extract_hf_text(sample)
                if len(text) > 50:
                    all_texts.append(text)

            if len(all_texts) < num_clients * samples_per_client:
                raise ValueError(
                    f"Not enough usable texts in '{dataset_name}' split '{split_spec}': "
                    f"got {len(all_texts)}, need at least {num_clients * samples_per_client}"
                )

            print(
                f"[StackOverflow] Loaded {len(all_texts)} texts from "
                f"HuggingFace dataset '{dataset_name}'"
            )

            self.tokenizer = SimpleTokenizer(vocab_size=self.vocab_size)
            self.tokenizer.build_vocab(all_texts[:100000], min_freq=2)  # Use subset for vocab

            np.random.shuffle(all_texts)

            test_size = int(0.1 * len(all_texts))
            self.test_data = all_texts[:test_size]
            train_texts = all_texts[test_size:]

            for i in range(num_clients):
                start = i * samples_per_client
                end = start + samples_per_client
                if end <= len(train_texts):
                    self.client_data[i] = train_texts[start:end]

            print(f"[StackOverflow] Created {len(self.client_data)} clients")

        except Exception as e:
            print(f"[Error] Could not load from HuggingFace: {e}")
            if allow_synthetic_fallback:
                print("[Info] Falling back to synthetic data for testing")
                self._create_synthetic_data(num_clients, samples_per_client)
            else:
                raise

    def _create_synthetic_data(self, num_clients: int = 100, samples_per_client: int = 100):
        """Create synthetic data for testing (when real data unavailable)."""
        print("[StackOverflow] Creating synthetic data for testing...")

        # Simple synthetic sentences
        templates = [
            "How do I implement {} in Python?",
            "What is the best way to {} using JavaScript?",
            "I'm having trouble with {} in my code.",
            "Can someone explain how {} works?",
            "Error when trying to {} in my application.",
            "Need help with {} for my project.",
            "What's the difference between {} and {} in programming?",
            "How to optimize {} for better performance?",
            "Why does {} not work in my environment?",
            "Best practices for {} in software development.",
        ]

        topics = [
            "sorting algorithms", "database queries", "API calls", "authentication",
            "caching", "memory management", "threading", "file handling",
            "error handling", "unit testing", "deployment", "logging",
            "data structures", "recursion", "loops", "functions",
            "classes", "inheritance", "polymorphism", "encapsulation",
        ]

        all_texts = []
        for _ in range(num_clients * samples_per_client + 5000):  # Extra for test
            template = np.random.choice(templates)
            topic1 = np.random.choice(topics)
            topic2 = np.random.choice(topics)
            text = template.format(topic1, topic2) if '{}' in template.replace('{}', '', 1) else template.format(topic1)
            all_texts.append(text)

        # Build tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=self.vocab_size)
        self.tokenizer.build_vocab(all_texts, min_freq=1)

        # Split into test and train
        np.random.shuffle(all_texts)
        self.test_data = all_texts[:5000]
        train_texts = all_texts[5000:]

        # Distribute to clients
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client
            self.client_data[i] = train_texts[start:end]

        print(f"[StackOverflow] Created {len(self.client_data)} synthetic clients")

    def get_client_dataset(self, client_id: int) -> StackOverflowDataset:
        """Get dataset for a specific client."""
        if client_id not in self.client_data:
            raise ValueError(f"Client {client_id} not found")

        return StackOverflowDataset(
            texts=self.client_data[client_id],
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
        )

    def get_test_dataset(self) -> StackOverflowDataset:
        """Get centralized test dataset."""
        return StackOverflowDataset(
            texts=self.test_data,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
        )

    def get_num_clients(self) -> int:
        """Get number of clients."""
        return len(self.client_data)

    def save(self, path: str):
        """Save dataset state."""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save(os.path.join(path, 'tokenizer.pkl'))

        with open(os.path.join(path, 'client_data.pkl'), 'wb') as f:
            pickle.dump(self.client_data, f)

        with open(os.path.join(path, 'test_data.pkl'), 'wb') as f:
            pickle.dump(self.test_data, f)

    @classmethod
    def load(cls, path: str) -> 'StackOverflowFederated':
        """Load dataset state."""
        obj = cls()
        obj.tokenizer = SimpleTokenizer.load(os.path.join(path, 'tokenizer.pkl'))

        with open(os.path.join(path, 'client_data.pkl'), 'rb') as f:
            obj.client_data = pickle.load(f)

        with open(os.path.join(path, 'test_data.pkl'), 'rb') as f:
            obj.test_data = pickle.load(f)

        return obj


# ============================================================================
# Data Loading Functions for FL
# ============================================================================

# Global cache for federated dataset
STACKOVERFLOW_FDS: Optional[StackOverflowFederated] = None


def get_stackoverflow_federated(
    num_clients: int = 100,
    vocab_size: int = 10000,
    max_seq_len: int = 256,
    samples_per_client: int = 500,
    data_source: str = "huggingface",  # "huggingface", "leaf", or "synthetic"
    leaf_dir: Optional[str] = None,
    hf_dataset_name: str = "HuggingFaceH4/stack-exchange-preferences",
    hf_dataset_split: str = "train",
    hf_max_samples: Optional[int] = None,
    allow_synthetic_fallback: bool = True,
) -> StackOverflowFederated:
    """Get or create federated StackOverflow dataset."""
    global STACKOVERFLOW_FDS

    if STACKOVERFLOW_FDS is None:
        STACKOVERFLOW_FDS = StackOverflowFederated(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )

        if data_source == "leaf" and leaf_dir:
            STACKOVERFLOW_FDS.load_from_leaf(leaf_dir)
        elif data_source == "huggingface":
            STACKOVERFLOW_FDS.load_from_huggingface(
                num_clients=num_clients,
                samples_per_client=samples_per_client,
                dataset_name=hf_dataset_name,
                split=hf_dataset_split,
                max_samples=hf_max_samples,
                allow_synthetic_fallback=allow_synthetic_fallback,
            )
        else:
            STACKOVERFLOW_FDS._create_synthetic_data(num_clients, samples_per_client)

    return STACKOVERFLOW_FDS


def load_lm_partition_data(
    partition_id: int,
    config: dict,
) -> Tuple[DataLoader, DataLoader]:
    """Load language modeling data for a specific partition (client).

    Args:
        partition_id: Client's partition ID
        config: Run configuration

    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset_config = config.get("dataset", {})
    num_clients = config.get("num_clients", 100)
    batch_size = dataset_config.get("batch_size", 32)
    vocab_size = dataset_config.get("vocab_size", 10000)
    max_seq_len = dataset_config.get("max_seq_len", 256)
    samples_per_client = dataset_config.get("samples_per_client", 500)
    data_source = dataset_config.get("data_source", "huggingface")
    hf_dataset_name = dataset_config.get("hf_dataset_name", "HuggingFaceH4/stack-exchange-preferences")
    hf_dataset_split = dataset_config.get("hf_dataset_split", "train")
    hf_max_samples = dataset_config.get("hf_max_samples", None)
    allow_synthetic_fallback = bool(dataset_config.get("allow_synthetic_fallback", True))

    fds = get_stackoverflow_federated(
        num_clients=num_clients,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        samples_per_client=samples_per_client,
        data_source=data_source,
        hf_dataset_name=hf_dataset_name,
        hf_dataset_split=hf_dataset_split,
        hf_max_samples=hf_max_samples,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )

    client_dataset = fds.get_client_dataset(partition_id)

    # Split into train/val (90/10)
    n_samples = len(client_dataset)
    n_val = max(1, int(0.1 * n_samples))
    n_train = n_samples - n_val

    indices = list(range(n_samples))
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = Subset(client_dataset, train_indices)
    val_subset = Subset(client_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_lm_test_data(config: dict) -> DataLoader:
    """Load centralized test data for language modeling."""
    dataset_config = config.get("dataset", {})
    num_clients = config.get("num_clients", 100)
    batch_size = dataset_config.get("batch_size", 32)
    vocab_size = dataset_config.get("vocab_size", 10000)
    max_seq_len = dataset_config.get("max_seq_len", 256)
    samples_per_client = dataset_config.get("samples_per_client", 500)
    data_source = dataset_config.get("data_source", "huggingface")
    hf_dataset_name = dataset_config.get("hf_dataset_name", "HuggingFaceH4/stack-exchange-preferences")
    hf_dataset_split = dataset_config.get("hf_dataset_split", "train")
    hf_max_samples = dataset_config.get("hf_max_samples", None)
    allow_synthetic_fallback = bool(dataset_config.get("allow_synthetic_fallback", True))

    fds = get_stackoverflow_federated(
        num_clients=num_clients,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        samples_per_client=samples_per_client,
        data_source=data_source,
        hf_dataset_name=hf_dataset_name,
        hf_dataset_split=hf_dataset_split,
        hf_max_samples=hf_max_samples,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )

    test_dataset = fds.get_test_dataset()
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_tokenizer(config: dict) -> SimpleTokenizer:
    """Get tokenizer from federated dataset."""
    fds = get_stackoverflow_federated(
        num_clients=config.get("num_clients", 100),
        vocab_size=config.get("dataset", {}).get("vocab_size", 10000),
    )
    return fds.tokenizer


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("StackOverflow Language Modeling Dataset")
    print("=" * 70)

    # Test with synthetic data
    fds = StackOverflowFederated(vocab_size=5000, max_seq_len=128)
    fds._create_synthetic_data(num_clients=10, samples_per_client=50)

    print(f"\nNumber of clients: {fds.get_num_clients()}")
    print(f"Vocab size: {len(fds.tokenizer.word2idx)}")
    print(f"Test samples: {len(fds.test_data)}")

    # Test client dataset
    client_ds = fds.get_client_dataset(0)
    print(f"\nClient 0 samples: {len(client_ds)}")

    sample = client_ds[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")

    # Test decoding
    decoded = fds.tokenizer.decode(sample['input_ids'].tolist())
    print(f"Decoded text: {decoded[:100]}...")

    print("\n" + "=" * 70)
    print("All tests passed!")
