"""
Evaluation for Slowrun models.

Supports strided (sliding-window) eval and optional test-time training (TTT).

Usage:
    python eval.py --checkpoint sr-checkpoints/varlen-15.pt --do-strided --do-ttt
    torchrun --nproc_per_node=2 eval.py --checkpoint sr-checkpoints/varlen-15.pt --do-ttt
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"\n?Online softmax is disabled on the fly.*",
    category=UserWarning,
    module=r"torch\._inductor\.lowering",
)
import sys
import math
import time
import fcntl
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import tiktoken
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (
    GPTConfig, GPT, get_dist_info, print0,
    flash_attn, norm, apply_rotary_emb,
    DATA_DIR, VarlenDataLoader, evaluate_bpb,
    MAX_SEQ_LEN, EVAL_TOKENS,
)

# =============================================================================
# Helpers
# =============================================================================

def build_token_bytes_lut(device):
    """Build token-to-bytes lookup table. Returns (lut_tensor, vocab_size)."""
    encoder = tiktoken.get_encoding("gpt2")
    eot_id = encoder._special_tokens['<|endoftext|>']
    token_bytes_list = []
    for i in range(encoder.n_vocab):
        token_bytes_list.append(0 if i == eot_id else len(encoder.decode_single_token_bytes(i)))
    return torch.tensor(token_bytes_list, dtype=torch.int32, device=device), encoder.n_vocab

def load_val_data(val_data_path):
    """Load flat val tokens and doc_starts from varlen data file."""
    data = torch.load(val_data_path, weights_only=True)
    return data["tokens"].long(), data["doc_starts"].long()

def sample_docs(docs, fraction, seed=42):
    """Sample a fraction of docs for evaluation."""
    n_sample = max(1, int(len(docs) * fraction))
    sampled_idx = sorted(random.Random(seed).sample(range(len(docs)), n_sample))
    return [docs[i] for i in sampled_idx]

def load_model(checkpoint_path, vocab_size, device):
    """Load model from checkpoint, set to eval mode. Infers arch from weights."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    n_layer = state_dict["resid_lambdas"].shape[0]
    n_embd = state_dict["lm_head.weight"].shape[1]
    n_head = state_dict["transformer.h.0.attn.attn_gate.weight"].shape[0]
    config = GPTConfig(vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd, n_head=n_head, n_kv_head=n_head, dropout=0.0)
    model = GPT(config).to(device)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(state_dict[name].to(p.device, p.dtype))
    del state_dict
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config

def find_docs(doc_starts, total_tokens):
    """Return (start_offset, length) for each document from doc_starts tensor."""
    starts = doc_starts.tolist()
    docs = []
    for i in range(len(starts)):
        s = starts[i]
        e = starts[i + 1] if i + 1 < len(starts) else total_tokens
        if e - s >= 2:
            docs.append((s, e - s))
    return docs

def compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_start = ci * chunk_size
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len

# =============================================================================
# TTT forward pass (model forward with LoRA on Q/K/V/proj/MLP/lm_head)
# =============================================================================

def forward_ttt(model, idx, targets, lora):
    """Forward with per-batch LoRA deltas. Returns per-token loss (bsz, seq_len)."""
    B, T = idx.size()
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = norm(model.transformer.wte(idx))
    x0 = x
    skip_connections = []
    for i, block in enumerate(model.transformer.h):
        if i >= model.encoder_layers and skip_connections:
            skip = skip_connections.pop()
            x = x + model.skip_weights[i - model.encoder_layers] * skip
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.ve_projs[str(i)](x0) if str(i) in model.ve_projs else None

        x_norm = norm(x)
        attn = block.attn
        q = (attn.c_q(x_norm) + lora.q_loras[i](x_norm)).view(B, T, attn.n_head, attn.head_dim)
        k = (attn.c_k(x_norm) + lora.k_loras[i](x_norm)).view(B, T, attn.n_kv_head, attn.head_dim)
        v = (attn.c_v(x_norm) + lora.v_loras[i](x_norm)).view(B, T, attn.n_kv_head, attn.head_dim)
        if ve is not None:
            ve_r = ve.view(B, T, attn.n_kv_head, attn.head_dim)
            gate = 2 * torch.sigmoid(attn.ve_gate(x_norm[..., :attn.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve_r
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        if getattr(attn, "use_key_offset", False) and T > 1:
            k[:, 1:, :, attn.head_dim // 2:] = k[:, :-1, :, attn.head_dim // 2:].clone()
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=model.window_sizes[i])
        y = y * torch.sigmoid(attn.attn_gate(x_norm[..., :attn.attn_gate_channels])).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        x = x + attn.c_proj(y) + lora.proj_loras[i](y)

        x_mlp = norm(x)
        mlp = block.mlp
        x = x + mlp.c_proj(F.silu(mlp.c_gate(x_mlp)) * (mlp.c_fc(x_mlp) + lora.mlp_loras[i](x_mlp)))
        if i < model.encoder_layers:
            skip_connections.append(x)

    x = norm(x)
    logits = (model.lm_head(x) + lora.lm_head_lora(x))[..., :model.config.vocab_size].float()
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none'
    ).reshape(B, T)

# =============================================================================
# TTT LoRA
# =============================================================================

class BatchedLinearLoRA(nn.Module):
    """LoRA with independent weights per batch element: delta = x @ A^T @ B^T."""
    def __init__(self, bsz, in_features, out_features, rank):
        super().__init__()
        self._bound = 1.0 / math.sqrt(in_features)
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features).uniform_(-self._bound, self._bound))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))

    def reset(self):
        with torch.no_grad():
            self.A.uniform_(-self._bound, self._bound)
            self.B.zero_()

    def forward(self, x):
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

class BatchedTTTLoRA(nn.Module):
    """All LoRA adapters for one TTT batch: Q/K/V/proj per layer + MLP fc + LM head.
    
    Each sequence in the batch has its own LoRA, so there is **0** dependence between
    sequences in the dataset.
    """
    def __init__(self, bsz, model, rank):
        super().__init__()
        self.bsz = bsz
        dim = model.config.n_embd
        vocab = model.lm_head.out_features
        q_dim = model.config.n_head * (dim // model.config.n_head)
        kv_dim = model.config.n_kv_head * (dim // model.config.n_head)
        hidden = model.transformer.h[0].mlp.c_fc.out_features
        n = model.config.n_layer

        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, q_dim, rank) for _ in range(n)])
        self.k_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(n)])
        self.v_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(n)])
        self.mlp_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, hidden, rank) for _ in range(n)])
        self.proj_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(n)])

    def reset(self):
        with torch.no_grad():
            for loras in [self.q_loras, self.k_loras, self.v_loras,
                          self.mlp_loras, self.proj_loras]:
                for lora in loras:
                    lora.reset()
            self.lm_head_lora.reset()

# =============================================================================
# Strided (sliding-window) evaluation
# =============================================================================

def strided_eval(batch, bsz, all_tokens_idx, token_bytes, chunk_size, eval_seq_len, device, forward_fn, lora):
    """Sliding-window eval over a batch of docs. Returns (loss_sum, byte_sum, token_count) as tensors."""
    pred_lens = [doc_len - 1 for _, doc_len in batch]
    num_chunks_list = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
    max_nc = max(num_chunks_list)
    ls = torch.zeros((), device=device, dtype=torch.float64)
    bs = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    with torch.no_grad():
        for ci in range(max_nc):
            active = [ci < nc for nc in num_chunks_list]
            tok_starts = torch.zeros(bsz, dtype=torch.int64)
            tok_wls = torch.zeros(bsz, dtype=torch.int64)
            chunk_offsets_cpu = torch.zeros(bsz, dtype=torch.int64)
            chunk_lens_cpu = torch.zeros(bsz, dtype=torch.int64)
            for b in range(bsz):
                if not active[b]:
                    continue
                doc_start, doc_len = batch[b]
                ws, wl, co, cl = compute_chunk_window(ci, pred_lens[b], num_chunks_list[b], chunk_size, eval_seq_len)
                tok_starts[b] = doc_start + ws
                tok_wls[b] = wl
                chunk_offsets_cpu[b] = co
                chunk_lens_cpu[b] = cl
            _, context_size, _, _ = compute_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
            col_idx = torch.arange(context_size + 1)
            idx = tok_starts.unsqueeze(1) + col_idx.unsqueeze(0)
            idx.clamp_(max=all_tokens_idx.numel() - 1)
            gathered = all_tokens_idx[idx].to(device=device, dtype=torch.int64, non_blocking=True)
            valid = (col_idx[:context_size].unsqueeze(0) < tok_wls.unsqueeze(1)).to(device)
            chunk_offsets_d = chunk_offsets_cpu.to(device)
            chunk_lens_d = chunk_lens_cpu.to(device)
            x = torch.where(valid, gathered[:, :context_size], 0)
            y = torch.where(valid, gathered[:, 1:context_size + 1], 0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_tok_loss = forward_fn(x, y, lora=lora)
            ctx_pos = torch.arange(context_size, device=device).unsqueeze(0)
            mask = (chunk_lens_d.unsqueeze(1) > 0) & \
                   (ctx_pos >= chunk_offsets_d.unsqueeze(1)) & \
                   (ctx_pos < (chunk_offsets_d + chunk_lens_d).unsqueeze(1))
            mask_f64 = mask.to(torch.float64)
            ls += (per_tok_loss.detach().to(torch.float64) * mask_f64).sum()
            bs += (token_bytes[y].to(torch.float64) * mask_f64).sum()
            tc += chunk_lens_d.to(torch.float64).sum()
    return ls, bs, tc

def _finalize_metrics(loss_sum, byte_sum, token_count):
    """Convert raw sums to (bpb, loss)."""
    loss = (loss_sum / token_count).item()
    bpb = (loss_sum / math.log(2) / byte_sum).item()
    return bpb, loss

# =============================================================================
# TTT helpers
# =============================================================================

def build_ttt_global_batches(doc_entries, batch_size):
    """Group docs into fixed-size batches sorted by length, then order batches longest-first.
    
    We want to sort by length because due to the sequential nature of TTT, we want batches
    to have similar length docs. We want longest first because we have a central queue
    that GPUs pull from, so the distribution is more even if we process the longer ones
    first.
    """
    sorted_entries = sorted(doc_entries, key=lambda x: x[1][1])
    batches = [sorted_entries[i:i + batch_size] for i in range(0, len(sorted_entries), batch_size)]
    indexed = list(enumerate(batches))
    indexed.sort(key=lambda ib: -max(dl for _, (_, dl) in ib[1]))
    return indexed

def init_batch_counter(path, start=0):
    """Write initial counter value to a shared file used for multi-GPU batch assignment."""
    with open(path, "wb") as f:
        f.write(start.to_bytes(4, "little"))

def claim_next_batch(counter_path, queue_len):
    """Atomically increment the file counter and return the old value. Returns queue_len if file is gone."""
    try:
        with open(counter_path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            idx = int.from_bytes(f.read(4), "little")
            f.seek(0)
            f.write((idx + 1).to_bytes(4, "little"))
            f.flush()
    except FileNotFoundError:
        return queue_len
    return idx

def build_ttt_optimizer(lora, args):
    """Build AdamW optimizer for all LoRA parameters."""
    return torch.optim.AdamW(
        [{"params": list(lora.parameters()), "lr": args.ttt_lora_lr}],
        betas=(args.ttt_beta1, args.ttt_beta2), eps=1e-10,
        weight_decay=args.ttt_weight_decay, fused=True)

def warmup_ttt(model, forward_fn, docs, val_tokens, args, device):
    """Warmup TTT compilation. Returns nothing, just traces the compiled fn."""
    ds0 = docs[0][0]
    bsz = args.ttt_batch_size
    warmup_lora = BatchedTTTLoRA(bsz, model, args.ttt_lora_rank).to(device)
    warmup_opt = build_ttt_optimizer(warmup_lora, args)
    ctx_lens = [args.stride, args.eval_seq_len]
    for ctx_len in tqdm(ctx_lens, desc="TTT warmup", disable=dist.is_initialized() and dist.get_rank() != 0):
        col = torch.arange(ctx_len + 1)
        idx = (ds0 + col).clamp_(max=val_tokens.numel() - 1)
        row = val_tokens[idx].to(device=device, dtype=torch.int64)
        x_w = row[:ctx_len].unsqueeze(0).expand(bsz, -1).contiguous()
        y_w = row[1:ctx_len + 1].unsqueeze(0).expand(bsz, -1).contiguous()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ptl = forward_fn(x_w, y_w, lora=warmup_lora)
        ptl[:, :min(args.stride, ctx_len)].mean(dim=-1).sum().backward()
        warmup_opt.step()
        warmup_opt.zero_grad(set_to_none=True)
    del warmup_lora, warmup_opt
    torch.cuda.empty_cache()

# =============================================================================
# TTT evaluation
# =============================================================================

def eval_val_ttt(args, model, rank, device, token_bytes, all_tokens_idx, global_batches, forward_ttt_fn):
    """Multi-GPU TTT eval with dynamic batch assignment. Returns (bpb, loss)."""
    chunk_size = args.stride
    eval_seq_len = args.eval_seq_len
    total_batches = len(global_batches)
    print0(f"  Total TTT batches (across all devices): {total_batches}")

    if rank == 0:
        counter_path = f"/tmp/ttt_counter_{os.getpid()}"
        init_batch_counter(counter_path)
    else:
        counter_path = ""
    if dist.is_initialized():
        path_list = [counter_path]
        dist.broadcast_object_list(path_list, src=0)
        counter_path = path_list[0]
        dist.barrier()

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    reusable_lora = None
    reusable_opt = None

    pbar = tqdm(total=total_batches, desc="TTT eval", disable=rank != 0)
    pbar.refresh()
    while True:
        queue_idx = claim_next_batch(counter_path, total_batches)
        if queue_idx >= total_batches:
            break
        _, batch_entries = global_batches[queue_idx]
        batch = [doc for _, doc in batch_entries]
        bsz = len(batch)

        # Reset LoRA if exists + batch size matches, otherwise initialize.
        if reusable_lora is not None and bsz == reusable_lora.bsz:
            reusable_lora.reset()
            for s in reusable_opt.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        v.zero_()
                    elif k == "step":
                        s[k] = 0
            cur_lora = reusable_lora
            cur_opt = reusable_opt
        else:
            cur_lora = BatchedTTTLoRA(bsz, model, args.ttt_lora_rank).to(device)
            cur_opt = build_ttt_optimizer(cur_lora, args)
            reusable_lora = cur_lora
            reusable_opt = cur_opt

        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks_list = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks_list)
        num_chunks_t = torch.tensor(num_chunks_list, dtype=torch.int64, device=device)

        for ci in range(max_nc):
            active = [ci < nc for nc in num_chunks_list]
            needs_train = any(ci < nc - 1 for nc in num_chunks_list)

            # Per-doc sliding window: each doc gets a context window ending at
            # this chunk's end, up to eval_seq_len tokens of prior context.
            # tok_starts/tok_wls = absolute start and length of each doc's window
            # chunk_offsets/chunk_lens = where the new (scoreable) tokens sit within the window
            tok_starts = torch.zeros(bsz, dtype=torch.int64)
            tok_wls = torch.zeros(bsz, dtype=torch.int64)
            chunk_offsets_cpu = torch.zeros(bsz, dtype=torch.int64)
            chunk_lens_cpu = torch.zeros(bsz, dtype=torch.int64)
            for b in range(bsz):
                if not active[b]:
                    continue
                doc_start, doc_len = batch[b]
                ws, wl, co, cl = compute_chunk_window(ci, pred_lens[b], num_chunks_list[b], chunk_size, eval_seq_len)
                tok_starts[b] = doc_start + ws
                tok_wls[b] = wl
                chunk_offsets_cpu[b] = co
                chunk_lens_cpu[b] = cl

            # Max context size for this chunk index (used as the padded sequence length)
            _, context_size, chunk_offset, _ = compute_chunk_window(
                ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)

            # Gather tokens into (bsz, context_size+1) — the +1 gives us targets
            col_idx = torch.arange(context_size + 1)
            idx = tok_starts.unsqueeze(1) + col_idx.unsqueeze(0)
            idx.clamp_(max=all_tokens_idx.numel() - 1)
            gathered = all_tokens_idx[idx].to(device=device, dtype=torch.int64, non_blocking=True)
            valid = (col_idx[:context_size].unsqueeze(0) < tok_wls.unsqueeze(1)).to(device)
            chunk_offsets_d = chunk_offsets_cpu.to(device)
            chunk_lens_d = chunk_lens_cpu.to(device)
            x = torch.where(valid, gathered[:, :context_size], 0)
            y = torch.where(valid, gathered[:, 1:context_size + 1], 0)
            ctx_pos = torch.arange(context_size, device=device)

            # Compute loss for the chunk.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_tok_loss = forward_ttt_fn(x, y, lora=cur_lora)
            with torch.no_grad():
                pos = ctx_pos.unsqueeze(0)
                mask = (chunk_lens_d.unsqueeze(1) > 0) & \
                        (pos >= chunk_offsets_d.unsqueeze(1)) & \
                        (pos < (chunk_offsets_d + chunk_lens_d).unsqueeze(1))
                mask_f64 = mask.to(torch.float64)
                loss_sum += (per_tok_loss.detach().to(torch.float64) * mask_f64).sum()
                byte_sum += (token_bytes[y].to(torch.float64) * mask_f64).sum()
                token_count += chunk_lens_d.to(torch.float64).sum()

            # Train on the chunk (after computing loss).
            if needs_train:
                activate_mask = (num_chunks_t - 1 > ci).float()
                per_doc = per_tok_loss[:, chunk_offset:chunk_offset + chunk_size].mean(dim=-1)
                cur_opt.zero_grad(set_to_none=True)
                (per_doc * activate_mask).sum().backward()
                cur_opt.step()
            else:
                del per_tok_loss

        rl = loss_sum.item() / max(token_count.item(), 1)
        rb = (loss_sum.item() / math.log(2)) / max(byte_sum.item(), 1)
        pbar.n = queue_idx + 1
        pbar.set_postfix(loss=f"{rl:.4f}", bpb=f"{rb:.4f}")
        pbar.refresh()

    pbar.close()
    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    if rank == 0:
        os.remove(counter_path)

    loss = (loss_sum / token_count).item()
    bpb = (loss_sum / math.log(2) / byte_sum).item()
    return bpb, loss

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation (strided + optional TTT)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=os.path.join(DATA_DIR, "fineweb_val_varlen.pt"))
    parser.add_argument("--stride", type=int, default=32, help="Chunk/stride size for sliding-window eval")
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--val-doc-fraction", type=float, default=1.0)

    parser.add_argument("--do-standard", action="store_true", help="Run standard (non-overlapping) eval")
    parser.add_argument("--do-strided", action="store_true", help="Run strided (sliding-window) eval")
    parser.add_argument("--do-ttt", action="store_true", help="Run test-time training eval")

    parser.add_argument("--ttt-lora-rank", type=int, default=96)
    parser.add_argument("--ttt-lora-lr", type=float, default=0.0001)
    parser.add_argument("--ttt-batch-size", type=int, default=8)
    parser.add_argument("--ttt-weight-decay", type=float, default=0.5)
    parser.add_argument("--ttt-beta1", type=float, default=0.0)
    parser.add_argument("--ttt-beta2", type=float, default=0.999)
    args = parser.parse_args()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    master_process = ddp_rank == 0

    if ddp and torch.cuda.is_available():
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(42)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_bytes, vocab_size = build_token_bytes_lut(device)

    t0 = time.perf_counter()
    print0(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, vocab_size, device)
    print0(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params ({time.perf_counter()-t0:.1f}s)")

    # --- Load val data ---
    val_tokens, val_doc_starts = load_val_data(args.val_data)
    print0(f"Val tokens: {val_tokens.numel():,}")

    docs = find_docs(val_doc_starts, val_tokens.numel())
    print0(f"Total documents: {len(docs)}")
    if args.val_doc_fraction < 1.0:
        docs = sample_docs(docs, args.val_doc_fraction)
        print0(f"Sampled {len(docs)} docs ({args.val_doc_fraction*100:.0f}%)")

    doc_lens = [dl for _, dl in docs]
    print0(f"Doc lengths: min={min(doc_lens)} max={max(doc_lens)} mean={sum(doc_lens)/len(doc_lens):.0f}")

    if args.do_standard:
        print0(f"\n=== Standard evaluation (seq_len={MAX_SEQ_LEN}, whole dataset) ===")
        model_compiled = torch.compile(model, dynamic=False, fullgraph=True)
        val_loader = VarlenDataLoader(
            args.val_data, args.ttt_batch_size, MAX_SEQ_LEN,
            device=device, shuffle=False, varlen=True)
        eval_steps = min(EVAL_TOKENS // (args.ttt_batch_size * MAX_SEQ_LEN * ddp_world_size),
                         val_loader.num_steps)
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            std_bpb, std_loss = evaluate_bpb(model_compiled, val_loader, eval_steps, token_bytes, varlen=True)
        print0(f"Standard Loss: {std_loss:.6f} | BPB: {std_bpb:.6f} ({time.perf_counter()-t0:.1f}s)")
        del model_compiled

    # --- Compile TTT forward ---
    torch._dynamo.reset()
    def _fwd_ttt(input_ids, target_ids, lora):
        return forward_ttt(model, input_ids, target_ids, lora)
    fwd_compiled = torch.compile(_fwd_ttt, dynamic=True)

    global_batches = build_ttt_global_batches(list(enumerate(docs)), args.ttt_batch_size)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    all_tokens_idx = val_tokens.to(torch.int32)

    if args.do_strided:
        print0(f"\n=== Strided evaluation (stride={args.stride}, seq_len={args.eval_seq_len}, longest docs first) ===")
        t0 = time.perf_counter()
        base_ls = torch.zeros((), device=device, dtype=torch.float64)
        base_bs = torch.zeros((), device=device, dtype=torch.float64)
        base_tc = torch.zeros((), device=device, dtype=torch.float64)
        zero_lora_cache = {}
        my_batches = list(range(ddp_rank, len(global_batches), world_size))
        pbar = tqdm(my_batches, desc="Strided eval", disable=ddp_rank != 0)
        for qi in pbar:
            _, batch_entries = global_batches[qi]
            batch = [doc for _, doc in batch_entries]
            bsz = len(batch)
            if bsz not in zero_lora_cache:
                zero_lora_cache[bsz] = BatchedTTTLoRA(bsz, model, args.ttt_lora_rank).to(device)
            ls, bs, tc = strided_eval(
                batch, bsz, all_tokens_idx, token_bytes, args.stride, args.eval_seq_len,
                device, fwd_compiled, zero_lora_cache[bsz])
            base_ls += ls
            base_bs += bs
            base_tc += tc
            bpb_so_far, loss_so_far = _finalize_metrics(base_ls, base_bs, base_tc)
            pbar.set_postfix(loss=f"{loss_so_far:.4f}", bpb=f"{bpb_so_far:.4f}")
        del zero_lora_cache
        torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.all_reduce(base_ls, op=dist.ReduceOp.SUM)
            dist.all_reduce(base_bs, op=dist.ReduceOp.SUM)
            dist.all_reduce(base_tc, op=dist.ReduceOp.SUM)

        base_bpb, base_loss = _finalize_metrics(base_ls, base_bs, base_tc)
        print0(f"Strided Loss: {base_loss:.6f} | BPB: {base_bpb:.6f} ({time.perf_counter()-t0:.1f}s)")

    if args.do_ttt:
        print0(f"\n=== TTT evaluation (stride={args.stride}, longest docs first) ===")
        print0(f"TTT config: rank={args.ttt_lora_rank} lr={args.ttt_lora_lr} "
               f"chunk={args.stride} seq_len={args.eval_seq_len} "
               f"batch={args.ttt_batch_size} wd={args.ttt_weight_decay} "
               f"beta1={args.ttt_beta1} beta2={args.ttt_beta2}")

        print0("Warming up TTT compilation...")
        t0 = time.perf_counter()
        warmup_ttt(model, fwd_compiled, docs, val_tokens, args, device)
        print0(f"TTT warmup: {time.perf_counter()-t0:.1f}s")

        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_bpb, ttt_loss = eval_val_ttt(
            args, model, ddp_rank, device, token_bytes,
            all_tokens_idx, global_batches, fwd_compiled)
        torch.cuda.synchronize()
        ttt_time = time.perf_counter() - t_ttt

    print0(f"\n{'='*60}")
    if args.do_standard:
        print0(f"Standard Loss: {std_loss:.6f} | BPB: {std_bpb:.6f}")
    if args.do_strided:
        print0(f"Strided  Loss: {base_loss:.6f} | BPB: {base_bpb:.6f}")
    if args.do_ttt:
        print0(f"TTT      Loss: {ttt_loss:.6f} | BPB: {ttt_bpb:.6f}")
        if args.do_strided:
            print0(f"Delta (strided-TTT): loss={base_loss - ttt_loss:.6f} bpb={base_bpb - ttt_bpb:.6f}")
        print0(f"TTT eval time: {ttt_time:.1f}s")
    print0(f"{'='*60}")

    if dist.is_initialized():
        dist.destroy_process_group()
