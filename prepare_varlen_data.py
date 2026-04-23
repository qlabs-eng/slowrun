"""
Preprocess FineWeb into varlen-friendly train/val splits.

Unlike `prepare_data.py`, this preserves the original token order and document
boundaries. Each output file stores a flat token stream plus the BOS offsets of
all documents, which is suitable for varlen attention and TTT.

The token stream matches `prepare_data.py` exactly before that script applies
artificial row chunking and row shuffling. In particular, if a token budget
cuts through a document, the final document is kept as a partial document.

Usage:
    python prepare_varlen_data.py
    python prepare_varlen_data.py --train_tokens 100_000_000 --val_tokens 10_000_000
    python prepare_varlen_data.py --local_dir fineweb_varlen_data
"""

import os
import gc
import hashlib
import argparse

import numpy as np
import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


LOCAL_DIR = "fineweb_varlen_data"
OUTPUT_FILES = {
    "val": "fineweb_val_varlen.pt",
    "train": "fineweb_train_varlen.pt",
}


def tokenize_documents(dataset_iter, encoder, token_budget):
    """Tokenize documents exactly like prepare_data.py, keeping a partial final doc."""
    bos_id = encoder._special_tokens["<|endoftext|>"]
    tokens = []
    doc_starts = []
    last_doc_partial = False
    pbar = tqdm(total=token_budget, unit="tok")
    for doc in dataset_iter:
        doc_tokens = [bos_id] + encoder.encode_ordinary(doc["text"])
        doc_starts.append(len(tokens))
        remaining = token_budget - len(tokens)
        keep = min(len(doc_tokens), remaining)
        tokens.extend(doc_tokens[:keep])
        pbar.update(keep)
        if len(tokens) >= token_budget:
            last_doc_partial = keep < len(doc_tokens)
            tokens = tokens[:token_budget]
            break
    pbar.close()
    return (
        np.asarray(tokens, dtype=np.uint16),
        np.asarray(doc_starts, dtype=np.int64),
        last_doc_partial,
    )


def doc_lengths(doc_starts, total_tokens):
    ends = np.empty_like(doc_starts)
    ends[:-1] = doc_starts[1:]
    ends[-1] = total_tokens
    return ends - doc_starts


def write_varlen_datafile(filename, tokens, doc_starts, bos_id, vocab_size, last_doc_partial):
    """Write one split as a flat token stream plus BOS positions."""
    if tokens.size == 0:
        raise ValueError(f"refusing to write empty token stream to {filename}")
    if doc_starts.size == 0:
        raise ValueError(f"refusing to write empty doc index to {filename}")
    if doc_starts[0] != 0:
        raise ValueError("expected first document to start at offset 0")
    if not np.all(tokens[doc_starts] == bos_id):
        raise ValueError("doc_starts do not point at BOS tokens")

    data = {
        "tokens": torch.from_numpy(tokens.copy()),
        "doc_starts": torch.from_numpy(doc_starts.copy()),
        "num_tokens": int(tokens.size),
        "num_docs": int(doc_starts.size),
        "bos_id": int(bos_id),
        "vocab_size": int(vocab_size),
        "tokenizer": "gpt2",
        "last_doc_partial": bool(last_doc_partial),
    }
    torch.save(data, filename)


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def print_split_stats(name, tokens, doc_starts, token_budget, last_doc_partial):
    lens = doc_lengths(doc_starts, tokens.size)
    print(f"  {name}: {tokens.size:,} tokens across {doc_starts.size:,} docs")
    print(f"    budget={token_budget:,} kept={tokens.size:,} dropped={token_budget - tokens.size:,}")
    print(f"    doc lens: min={lens.min()} max={lens.max()} mean={lens.mean():.0f}")
    print(f"    last_doc_partial={last_doc_partial}")


def preprocess(train_tokens, val_tokens, local_dir):
    encoder = tiktoken.get_encoding("gpt2")
    bos_id = encoder._special_tokens["<|endoftext|>"]

    print("=" * 60)
    print("Preprocessing FineWeb for varlen attention")
    print("=" * 60)
    print("Format: flat token stream + document starts")
    print("Matches prepare_data.py token selection exactly")
    print("No row chunking or row shuffling; final doc may be partial")
    print(f"Val budget:   {val_tokens:>13,} tokens")
    print(f"Train budget: {train_tokens:>13,} tokens")
    print(f"Output: {local_dir}/")
    print("=" * 60)

    os.makedirs(local_dir, exist_ok=True)

    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    dataset_iter = iter(dataset)

    print(f"\nTokenizing val (budget {val_tokens:,})...")
    val_raw, val_doc_starts, val_last_doc_partial = tokenize_documents(dataset_iter, encoder, val_tokens)
    print_split_stats("val", val_raw, val_doc_starts, val_tokens, val_last_doc_partial)

    print(f"\nTokenizing train (budget {train_tokens:,})...")
    train_raw, train_doc_starts, train_last_doc_partial = tokenize_documents(dataset_iter, encoder, train_tokens)
    print_split_stats("train", train_raw, train_doc_starts, train_tokens, train_last_doc_partial)

    print()
    val_path = os.path.join(local_dir, OUTPUT_FILES["val"])
    train_path = os.path.join(local_dir, OUTPUT_FILES["train"])
    write_varlen_datafile(
        val_path, val_raw, val_doc_starts, bos_id, encoder.n_vocab, val_last_doc_partial
    )
    write_varlen_datafile(
        train_path, train_raw, train_doc_starts, bos_id, encoder.n_vocab, train_last_doc_partial
    )

    del dataset_iter
    del dataset
    gc.collect()

    print(f"Wrote {val_path}")
    print(f"  sha256={sha256_file(val_path)}")
    print(f"Wrote {train_path}")
    print(f"  sha256={sha256_file(train_path)}")
    print(f"\nDone! Files saved to {local_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FineWeb for varlen attention")
    parser.add_argument("--train_tokens", type=int, default=100_000_000)
    parser.add_argument("--val_tokens", type=int, default=10_000_000)
    parser.add_argument("--local_dir", type=str, default=LOCAL_DIR)
    args = parser.parse_args()

    preprocess(
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        local_dir=args.local_dir,
    )
