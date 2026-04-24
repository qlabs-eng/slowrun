"""
Preprocess FineWeb into pre-tokenized train/val splits using the GPT-2 tokenizer.

Usage:
    python prepare_data.py
    python prepare_data.py --train_tokens 100_000_000 --val_tokens 10_000_000 --local_dir fineweb_data
"""

import os
import hashlib
import argparse
import numpy as np
import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

SEQUENCE_LENGTH = 2048
SEQUENCE_SIZE = SEQUENCE_LENGTH + 1

VAL_SHUFFLE_SEED = 42
TRAIN_SHUFFLE_SEED = 43


def tokenize_documents(dataset_iter, encoder, token_budget):
    """Tokenize documents from an iterator, recording document boundaries."""
    bos_id = encoder._special_tokens["<|endoftext|>"]
    tokens = []
    doc_starts = []
    pbar = tqdm(total=token_budget, unit="tok")
    for doc in dataset_iter:
        doc_tokens = [bos_id] + encoder.encode_ordinary(doc["text"])
        doc_starts.append(len(tokens))
        remaining = token_budget - len(tokens)
        keep = min(len(doc_tokens), remaining)
        tokens.extend(doc_tokens[:keep])
        pbar.update(keep)
        if len(tokens) >= token_budget:
            tokens = tokens[:token_budget]
            break
    pbar.close()
    return np.asarray(tokens, dtype=np.uint16), np.asarray(doc_starts, dtype=np.int64)


def write_datafile(filename, tokens, doc_starts, bos_id, shuffle_seed):
    if tokens.size == 0:
        raise ValueError(f"refusing to write empty token stream to {filename}")
    assert doc_starts[0] == 0
    assert np.all(tokens[doc_starts] == bos_id)

    num_seqs = tokens.size // SEQUENCE_SIZE
    print(f"Writing {filename}")
    print(f"  {tokens.size:,} tokens, {doc_starts.size:,} docs, {num_seqs:,} sequences")

    data = {
        "tokens": torch.from_numpy(tokens.copy()),
        "doc_starts": torch.from_numpy(doc_starts.copy()),
        "bos_id": int(bos_id),
        "seq_shuffle_seed": int(shuffle_seed),
        "seq_size": int(SEQUENCE_SIZE),
    }
    torch.save(data, filename)


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


EXPECTED_HASHES = {
    "fineweb_val.pt": "6868ed375b289a89c72c2f9df1ecbdcff700c4b9478ca806435d2dbfad8573b1",
    "fineweb_train.pt": "36e7c95c1e7f6ed952fb002d76a03044e8617fea7e696a68d7dc1ce78465dcaf",
}


def verify_hash(filepath):
    """Check file hash against expected value and assert it matches."""
    basename = os.path.basename(filepath)
    actual = sha256_file(filepath)
    expected = EXPECTED_HASHES.get(basename)
    if expected is None:
        print(f"  Hash for {basename}: {actual}")
        print(f"  (no expected hash set — paste this value into EXPECTED_HASHES to lock it in)")
    else:
        assert actual == expected, (
            f"HASH MISMATCH for {basename}!\n    expected: {expected}\n    actual:   {actual}"
        )
        print(f"  Hash OK for {basename}: {actual}")


def preprocess(train_tokens, val_tokens, local_dir):
    encoder = tiktoken.get_encoding("gpt2")
    bos_id = encoder._special_tokens["<|endoftext|>"]

    print(f"{'='*60}")
    print(f"Preprocessing FineWeb with GPT-2 tokenizer")
    print(f"{'='*60}")
    print(f"Sequence length: {SEQUENCE_LENGTH} (size {SEQUENCE_SIZE})")
    print(f"Val:   {val_tokens:>13,} tokens")
    print(f"Train: {train_tokens:>13,} tokens")
    print(f"Output: {local_dir}/")
    print(f"{'='*60}")

    os.makedirs(local_dir, exist_ok=True)

    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    dataset_iter = iter(dataset)

    print(f"\nTokenizing val ({val_tokens:,} tokens)...")
    val_tokens_arr, val_doc_starts = tokenize_documents(dataset_iter, encoder, val_tokens)
    print(f"  {val_tokens_arr.size:,} tokens, {val_doc_starts.size:,} docs")

    print(f"\nTokenizing train ({train_tokens:,} tokens)...")
    train_tokens_arr, train_doc_starts = tokenize_documents(dataset_iter, encoder, train_tokens)
    print(f"  {train_tokens_arr.size:,} tokens, {train_doc_starts.size:,} docs")

    # Shut down the streaming dataset before Python finalizes to avoid
    # background-thread errors (Bad file descriptor / PyGILState_Release).
    del dataset_iter, dataset

    print()
    val_path = os.path.join(local_dir, "fineweb_val.pt")
    train_path = os.path.join(local_dir, "fineweb_train.pt")
    write_datafile(val_path, val_tokens_arr, val_doc_starts, bos_id, VAL_SHUFFLE_SEED)
    write_datafile(train_path, train_tokens_arr, train_doc_starts, bos_id, TRAIN_SHUFFLE_SEED)

    print()
    verify_hash(val_path)
    verify_hash(train_path)

    print(f"\nDone! Files saved to {local_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FineWeb with GPT-2 tokenizer")
    parser.add_argument("--train_tokens", type=int, default=100_000_000)
    parser.add_argument("--val_tokens", type=int, default=10_000_000)
    parser.add_argument("--local_dir", type=str, default="fineweb_data")
    args = parser.parse_args()

    preprocess(
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        local_dir=args.local_dir,
    )
