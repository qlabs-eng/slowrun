import argparse
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_interpreter import naive_slot_memory, vectorized_slot_memory


def build_inputs(batch, seq_len, slots, mem_dim, device):
    torch.manual_seed(0)
    slot_ids = torch.randint(0, slots, (batch, seq_len), device=device)
    write_slots = torch.nn.functional.one_hot(slot_ids, num_classes=slots).float()
    write_values = torch.randn(batch, seq_len, mem_dim, device=device)
    return write_slots, write_values


def time_fn(fn, *args, iters, warmup=3):
    for _ in range(warmup):
        fn(*args)
    if args[0].is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if args[0].is_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt / iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark the frozen trace interpreter scan")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--mem-dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    write_slots, write_values = build_inputs(
        batch=args.batch,
        seq_len=args.seq_len,
        slots=args.slots,
        mem_dim=args.mem_dim,
        device=device,
    )

    memory_before, _ = vectorized_slot_memory(write_slots, write_values)
    naive = naive_slot_memory(write_slots, write_values)
    max_diff = (memory_before - naive).abs().max().item()

    vectorized_dt = time_fn(
        vectorized_slot_memory,
        write_slots,
        write_values,
        iters=args.iters,
    )
    naive_dt = time_fn(
        naive_slot_memory,
        write_slots,
        write_values,
        iters=max(1, args.iters // 4),
    )

    tokens = args.batch * args.seq_len
    print(f"device: {device}")
    print(f"shape: batch={args.batch}, seq_len={args.seq_len}, slots={args.slots}, mem_dim={args.mem_dim}")
    print(f"max_diff: {max_diff:.8f}")
    print(f"vectorized: {vectorized_dt*1000:.2f} ms  ({tokens/vectorized_dt:,.0f} tok/s)")
    print(f"naive:      {naive_dt*1000:.2f} ms  ({tokens/naive_dt:,.0f} tok/s)")
    print(f"speedup:    {naive_dt / vectorized_dt:.2f}x")


if __name__ == "__main__":
    main()
