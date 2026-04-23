"""
Train a language model on ~100M tokens with val loss evaluation.
Code is based on Nanochat (https://github.com/karpathy/nanochat), with modifications to support the slowrun setting.
Made for the Tiny Track of the NanoGPT Slowrun benchmark.

Usage:
    torchrun --standalone --nproc_per_node=8 tiny/train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_LOGS"] = "recompiles"
import gc
import math
import time
import json
import argparse
import sys
import secrets
import shutil
import string
from types import SimpleNamespace
from functools import partial
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
import wandb
import numpy as np

import tiktoken

_script_start = time.time()

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Train GPT model")
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--num-epochs", type=int, default=16)
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument("--run-name", type=str, default=None,
                    help="Run name under runs/ (default: random 6-char string)")
parser.add_argument("--scalar-lr", type=float, default=0.25)
parser.add_argument("--matrix-lr", type=float, default=0.04)
parser.add_argument("--embedding-lr", type=float, default=0.15)
parser.add_argument("--unembedding-lr", type=float, default=0.001)
parser.add_argument("--weight-decay", type=float, default=0.8)
# WD follows a 3-phase schedule: hold → decay → ramp
#   [0, wd-phase1-epoch]:          hold at --weight-decay
#   [wd-phase1-epoch, wd-phase2-epoch]: decay to --wd-mid
#   [wd-phase2-epoch, num-epochs]:      ramp up to --wd-end
parser.add_argument("--wd-phase1-epoch", type=int, default=2)
parser.add_argument("--wd-phase2-epoch", type=int, default=8)
parser.add_argument("--wd-mid", type=float, default=0.1)
parser.add_argument("--wd-end", type=float, default=1.25)
parser.add_argument("--warmdown-ratio", type=float, default=0.6)
parser.add_argument("--total-batch-size", type=int, default=524288)
parser.add_argument("--n_layer", type=int, default=16)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--n_embd", type=int, default=1024)
parser.add_argument("--lr_multiplier", type=float, default=0.8)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--update-ema-every", type=int, default=10)
parser.add_argument("--ema-decay-per-epoch", type=float, default=0.15)
parser.add_argument("--swa-last-epochs", type=int, default=4,
                    help="SWA: cosine-cycle LR in last N epochs for checkpoint diversity (0=off)")
parser.add_argument("--varlen", action="store_true",
                    help="Use variable-length attention (no cross-document context)")
parser.add_argument("--stride", type=int, default=512,
                    help="Stride for sliding-window eval; 0 to disable (default: 512)")

# =============================================================================
# Default architecture constants (overridden by CLI when run as __main__)
# =============================================================================

MAX_SEQ_LEN = 2048
WINDOW_PATTERN = "SSSL"
DEPTH = 16
N_EMBD = 1024
N_HEAD = 8
HEAD_DIM = N_EMBD // N_HEAD
EVAL_TOKENS = 10_000_000
BOS_ID = 50256  # <|endoftext|>
DATA_DIR = "fineweb_varlen_data"
VARLEN_CU_BUCKET_SIZE = 64
RUNS_DIR = "runs"

# =============================================================================
# Utilities
# =============================================================================

def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def print0(s="", **kwargs):
    if int(os.environ.get('RANK', 0)) == 0:
        print(s, **kwargs)

class DummyWandb:
    def __init__(self): self.summary = {}
    def log(self, *a, **kw): pass
    def finish(self): pass

class TeeStream:
    """Save terminal output to file."""
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")
    def write(self, data):
        for stream in self.streams: stream.write(data)
        return len(data)
    def flush(self):
        for stream in self.streams: stream.flush()
    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)
    def fileno(self):
        return self.streams[0].fileno()

def resolve_run_dir(run_name):
    if run_name:
        actual_run_name = run_name
    else:
        alphabet = string.ascii_lowercase + string.digits
        actual_run_name = "".join(secrets.choice(alphabet) for _ in range(6))
    return actual_run_name, os.path.join(RUNS_DIR, actual_run_name)

# =============================================================================
# Flash Attention (FA3 on Hopper, SDPA fallback elsewhere)
# =============================================================================

def _load_fa3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None

_fa3 = _load_fa3()

def _sdpa_attention(q, k, v, window_size, enable_gqa):
    Tq, Tk = q.size(2), k.size(2)
    window = window_size[0]
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
    if Tq == 1:
        if window >= 0 and window < Tk:
            start = max(0, Tk - (window + 1))
            k, v = k[:, :, start:, :], v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Flash Attention for training. q,k,v: (B, T, H, D)."""
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)

flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)


def flash_attn_varlen_fn(q, k, v, cu_seqlens, max_seqlen, causal=False, window_size=(-1, -1)):
    """Flash Attention varlen. q,k,v: (T, H, D) — no batch dim. Returns (T, H, D)."""
    assert _fa3 is not None, "FA3 required for varlen attention"
    return _fa3.flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        causal=causal, window_size=window_size,
    )


def _build_cu_seqlens(bos_pos, total_len, device, max_doc_len=0, bucket_size=64):
    """Build cu_seqlens from BOS positions, optionally splitting long segments."""
    if not bos_pos or bos_pos[0] != 0:
        bos_pos = [0] + bos_pos
    seg_starts = []
    starts_with_end = bos_pos + [total_len]
    for i in range(len(starts_with_end) - 1):
        start, end = starts_with_end[i], starts_with_end[i + 1]
        if max_doc_len > 0:
            pos = start
            while pos < end:
                seg_starts.append(pos)
                pos += max_doc_len
        else:
            seg_starts.append(start)
    boundaries = seg_starts + [total_len]
    padded_len = ((len(boundaries) + bucket_size - 1) // bucket_size) * bucket_size
    cu = torch.full((padded_len,), total_len, dtype=torch.int32, device=device)
    cu[:len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
    seg_ends = seg_starts[1:] + [total_len]
    max_seqlen = max(end - start for start, end in zip(seg_starts, seg_ends))
    return cu, max_seqlen

# =============================================================================
# GPT Model
# =============================================================================

@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 32768
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_kv_head: int = N_HEAD
    n_embd: int = N_EMBD
    window_pattern: str = WINDOW_PATTERN
    dropout: float = 0.1
    device_batch_size: int = 32

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    """Value Embedding on alternating layers, last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # Per-head attention gate: enables context-based attention no-op
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)
        # Determine if this is a long-window layer for partial key offset
        pattern = config.window_pattern.upper()
        char = pattern[layer_idx % len(pattern)]
        self.use_key_offset = (char == 'L') or (layer_idx == config.n_layer - 1)

    def forward(self, x, ve, cos_sin, window_size, cu_seqlens=None, max_seqlen=0):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # Value residual (ResFormer)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        # Partial key offset: shift stationary dims forward by 1 on long-window layers
        if self.use_key_offset and T > 1:
            stationary = k[..., self.head_dim // 2:].clone()
            k[:, 1:, :, self.head_dim // 2:] = stationary[:, :-1]
            if cu_seqlens is not None:
                # Reset at varlen boundaries so each segment starts from its own token.
                # Use fixed-shape ops to avoid data-dependent shapes that break torch.compile.
                all_b = cu_seqlens[1:].long()
                is_real = (all_b > 0) & (all_b < T)
                safe_idx = torch.where(is_real, all_b, torch.zeros_like(all_b))
                boundary_mask = torch.zeros(T, device=k.device, dtype=torch.bool)
                boundary_mask.scatter_(0, safe_idx, is_real)
                bm = boundary_mask[None, :, None, None]
                k[..., self.head_dim // 2:] = torch.where(bm, stationary, k[..., self.head_dim // 2:])
        if cu_seqlens is not None:
            y = flash_attn_varlen_fn(
                q[0], k[0], v[0], cu_seqlens, max_seqlen,
                causal=True, window_size=window_size,
            )[None]
        else:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # Per-head attention gate (sparse gated attention, zero-init → sigmoid(0)=0.5 at start)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_channels])).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256 * ((8 * config.n_embd // 3 + 255) // 256)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, cu_seqlens=None, max_seqlen=0):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, cu_seqlens, max_seqlen)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict({str(i): nn.Linear(config.n_embd, kv_dim, bias=False) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # U-Net skip connections: encoder layer i → decoder layer (n_layer - 1 - i)
        self.encoder_layers = config.n_layer // 2
        self.skip_weights = nn.Parameter(torch.ones(self.encoder_layers))
        self.rotary_seq_len = config.device_batch_size * config.sequence_len
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        self.resid_lambdas.fill_(1.1)
        self.x0_lambdas.fill_(0.1)
        for proj in self.ve_projs.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)
        self.skip_weights.fill_(1.0)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        # Half-truncated RoPE: only rotate half the dims, leave the rest stationary
        half = head_dim // 4  # number of frequency pairs for the rotated half
        inv_freq = 1.0 / (base ** (torch.arange(0, half * 2, 2, dtype=torch.float32, device=device) / (half * 2)))
        # Pad with zeros for the stationary half
        inv_freq = torch.cat([inv_freq, torch.zeros(head_dim // 2 - half, dtype=torch.float32, device=device)])
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)  # final layer always full context
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device
        
    def _avg_causal_attended_keys(self, window, seq_len):
        if window < 0 or window >= seq_len - 1:
            return (seq_len + 1) / 2
        max_keys = min(window + 1, seq_len)
        return max_keys - max_keys * (max_keys - 1) / (2 * seq_len)

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embedding lookup + elementwise scalars
        nparams_exclude = (self.transformer.wte.weight.numel()
                          + self.resid_lambdas.numel()
                          + self.x0_lambdas.numel()
                          + self.skip_weights.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Exact causal sliding-window attention FLOPs: 12 * h * q * E[keys attended per query]
        attn_flops = sum(12 * h * q * self._avg_causal_attended_keys(w[0], t) for w in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

    def setup_optimizer(self):
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate attn_gate params (small, Adam-optimized) from matrix params (Muon)
        attn_gate_params = [block.attn.attn_gate.weight for block in self.transformer.h]
        attn_gate_ids = {id(p) for p in attn_gate_params}
        all_h_params = list(self.transformer.h.parameters()) + list(self.ve_projs.parameters())
        matrix_params = [p for p in all_h_params if id(p) not in attn_gate_ids]
        embed_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        skip_params = [self.skip_weights]

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=embed_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=resid_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=SCALAR_LR, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=skip_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=attn_gate_params, lr=SCALAR_LR, betas=(0.9, 0.99), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=MATRIX_LR,
                                     momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY))

        optimizer = DistMuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, loss_reduction='mean', cu_seqlens=None, max_seqlen=0):
        # Can encode abs position in batch for varlen bc RoPE is relative.
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        skip_connections = []
        for i, block in enumerate(self.transformer.h):
            if i >= self.encoder_layers and skip_connections:
                skip = skip_connections.pop()
                x = x + self.skip_weights[i - self.encoder_layers] * skip
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = block(x, ve, cos_sin, self.window_sizes[i], cu_seqlens, max_seqlen)
            if i < self.encoder_layers:
                skip_connections.append(x)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)  # softcap
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return logits

# =============================================================================
# Optimizer: MuonAdamW (Muon for matrices, AdamW for embeddings/scalars)
# =============================================================================

# Polar Express coefficients for orthogonalization
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    p.add_(exp_avg / ((exp_avg_sq / bias2).sqrt() + eps_t), alpha=-(lr_t / bias1))

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar Express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class DistMuonAdamW(torch.optim.Optimizer):
    """Distributed MuonAdamW with ZeRO-2 style sharding."""
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0)
        self._adamw_lr_t = torch.tensor(0.0)
        self._adamw_beta1_t = torch.tensor(0.0)
        self._adamw_beta2_t = torch.tensor(0.0)
        self._adamw_eps_t = torch.tensor(0.0)
        self._adamw_wd_t = torch.tensor(0.0)
        self._muon_momentum_t = torch.tensor(0.0)
        self._muon_lr_t = torch.tensor(0.0)
        self._muon_wd_t = torch.tensor(0.0)
        self._muon_beta2_t = torch.tensor(0.0)

    def _reduce_adamw(self, group, world_size):
        infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=infos)

    def _reduce_muon(self, group, world_size):
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        stacked_grads = torch.empty(padded, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(torch.stack([p.grad for p in params]))
        if len(params) < padded:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group, info, gather_list, rank, world_size):
        for p in group['params']:
            pinfo = info['param_infos'][p]
            pinfo['future'].wait()
            state = self.state[p]
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p_slice, pinfo['grad_slice'], state['exp_avg'], state['exp_avg_sq'],
                           self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                           self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group, info, gather_list, rank):
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            s = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(s, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        updated = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        if num_owned > 0:
            owned = torch.stack([params[start_idx + i] for i in range(num_owned)])
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(info['grad_chunk'][:num_owned], owned,
                          state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                          self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                          group["ns_steps"], red_dim)
            updated[:num_owned].copy_(owned)
        if num_owned < chunk_size:
            updated[num_owned:].zero_()
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    @torch.no_grad()
    def step(self):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        reduce_infos = []
        for group in self.param_groups:
            if group['kind'] == 'adamw': reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon': reduce_infos.append(self._reduce_muon(group, world_size))
        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw': self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon': self._compute_muon(group, info, gather_list, rank)
        for info in gather_list:
            info["future"].wait()
            if info.get("params") is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))
# =============================================================================
# Dataloaders
# =============================================================================

class VarlenDataLoader:
    """Flat-token loader for varlen attention with document-level shuffling."""

    def __init__(self, filepath, B, T, device="cuda", *, shuffle=True,
                 varlen=True, cu_bucket_size=VARLEN_CU_BUCKET_SIZE):
        data = torch.load(filepath, weights_only=True)
        if "tokens" not in data or "doc_starts" not in data:
            raise ValueError(
                f"{filepath} is not a varlen data file; expected keys 'tokens' and 'doc_starts'"
            )
        all_tokens = data["tokens"].long()
        raw_doc_starts = data["doc_starts"].long()
        bos_id = int(data["bos_id"])
        assert bos_id == BOS_ID, f"data bos_id {bos_id} != expected {BOS_ID}"

        # Extract individual documents
        n_docs = raw_doc_starts.numel()
        doc_ends = torch.cat([raw_doc_starts[1:], torch.tensor([all_tokens.numel()])])
        self.doc_tokens = [all_tokens[s:e] for s, e in zip(raw_doc_starts.tolist(), doc_ends.tolist())]

        _, rank, _, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.B = B
        self.T = T
        self.varlen = varlen
        self.shuffle = shuffle
        self.cu_bucket_size = cu_bucket_size
        self.tokens_per_rank = B * T
        self.per_rank_span = self.tokens_per_rank + 1
        self.global_span = self.per_rank_span * world_size
        self.total_tokens_available = all_tokens.numel()
        self.epoch = 1
        self._build_stream()

    def _build_stream(self):
        """Concatenate documents (in current order) into a flat stream."""
        self.tokens = torch.cat(self.doc_tokens)
        # Recompute doc_starts from doc lengths
        lengths = torch.tensor([d.numel() for d in self.doc_tokens], dtype=torch.int64)
        self.doc_starts = torch.zeros(len(self.doc_tokens), dtype=torch.int64)
        torch.cumsum(lengths[:-1], dim=0, out=self.doc_starts[1:])
        self.num_steps = self.tokens.numel() // self.global_span
        self.total_tokens = self.num_steps * self.tokens_per_rank * self.world_size
        self.pos = 0

    def __iter__(self):
        return self

    def _shuffle_docs(self):
        if not self.shuffle:
            return
        g = torch.Generator()
        g.manual_seed(self.epoch)
        perm = torch.randperm(len(self.doc_tokens), generator=g)
        self.doc_tokens = [self.doc_tokens[i] for i in perm.tolist()]
        self._build_stream()

    def _local_doc_starts(self, start):
        end = start + self.tokens_per_rank
        lo = torch.searchsorted(self.doc_starts, torch.tensor(start, dtype=torch.int64), right=False).item()
        hi = torch.searchsorted(self.doc_starts, torch.tensor(end, dtype=torch.int64), right=False).item()
        return (self.doc_starts[lo:hi] - start).tolist()

    def __next__(self):
        if self.pos >= self.num_steps:
            self.pos = 0
            self.epoch += 1
            print0(f"Starting epoch {self.epoch}")
            self._shuffle_docs()
        batch_start = self.pos * self.global_span
        self.pos += 1
        local_start = batch_start + self.rank * self.per_rank_span
        buf = self.tokens[local_start:local_start + self.per_rank_span]
        x = buf[:-1].to(self.device, non_blocking=True)
        y = buf[1:].to(self.device, non_blocking=True)
        if self.varlen:
            starts = self._local_doc_starts(local_start)
            cu_seqlens, _ = _build_cu_seqlens(
                starts, self.tokens_per_rank, x.device,
                max_doc_len=self.T, bucket_size=self.cu_bucket_size,
            )
            return x[None], y[None], cu_seqlens, self.T, self.epoch
        return x.view(self.B, self.T), y.view(self.B, self.T), None, self.T, self.epoch

# =============================================================================
# Loss evaluation
# =============================================================================

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, varlen=False):
    """Compute bits per byte and mean cross-entropy loss on a set of batches."""
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, cu_seqlens, max_seqlen, _ = next(batch_iter)
        if varlen:
            loss2d = model(x, y, loss_reduction='none',
                           cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).view(-1)
        else:
            loss2d = model(x, y, loss_reduction='none').view(-1)
        y = y.view(-1)
        mask = y != -1
        total_loss += loss2d[mask].sum()
        total_tokens += mask.sum()
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss


@torch.no_grad()
def evaluate_bpb_strided(model, val_tokens_path, token_bytes, stride, seq_len,
                         batch_size, device, varlen=False):
    """Sliding-window BPB: each token scored with up to seq_len context.

    Slides a window of `seq_len` across the flat val token stream with the given
    stride.  Only the last `stride` tokens of each window are scored (they see
    full seq_len context).  Windows are sharded across DDP ranks.

    If varlen=True, builds per-window cu_seqlens from doc_starts so attention
    does not cross document boundaries.
    """
    data = torch.load(val_tokens_path, weights_only=True)
    tokens = data["tokens"].long()
    doc_starts = data["doc_starts"].long() if varlen else None
    n = tokens.numel()
    if n <= seq_len:
        return float('inf'), float('inf')

    starts = list(range(0, n - seq_len, stride))
    _, rank, _, world_size = get_dist_info()
    my_starts = starts[rank::world_size]

    total_nats = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=device)

    for i in range(0, len(my_starts), batch_size):
        chunk = my_starts[i:i + batch_size]
        if len(chunk) < batch_size:
            break
        x = torch.stack([tokens[s:s + seq_len] for s in chunk]).to(device)
        y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in chunk]).to(device)
        B_actual = len(chunk)
        if varlen:
            losses = []
            for b_idx in range(B_actual):
                s = chunk[b_idx]
                lo = torch.searchsorted(doc_starts, torch.tensor(s, dtype=torch.int64), right=False).item()
                hi = torch.searchsorted(doc_starts, torch.tensor(s + seq_len, dtype=torch.int64), right=False).item()
                local_starts = (doc_starts[lo:hi] - s).tolist()
                cu, _ = _build_cu_seqlens(local_starts, seq_len, device,
                                          max_doc_len=seq_len, bucket_size=VARLEN_CU_BUCKET_SIZE)
                loss_1d = model(x[b_idx:b_idx+1], y[b_idx:b_idx+1],
                                loss_reduction='none',
                                cu_seqlens=cu, max_seqlen=seq_len)
                losses.append(loss_1d.view(seq_len))
            loss_2d = torch.stack(losses)
        else:
            loss_2d = model(x, y, loss_reduction='none').view(B_actual, seq_len)
        scored = loss_2d[:, -stride:]
        scored_y = y[:, -stride:]
        total_nats += scored.double().sum()
        total_bytes += token_bytes[scored_y].sum()
        total_tokens += scored.numel()

    if dist.is_initialized():
        for t in (total_nats, total_bytes, total_tokens):
            dist.all_reduce(t)

    bpb = (total_nats / total_bytes.double() / math.log(2)).item()
    loss = (total_nats / total_tokens.double()).item()
    return bpb, loss


# =============================================================================
# Training
# =============================================================================

if __name__ == "__main__":
    args = parser.parse_args()

    # Resolve output path
    # Override architecture constants from CLI
    DEPTH = args.n_layer if args.n_layer is not None else DEPTH
    N_EMBD = args.n_embd if args.n_embd is not None else N_EMBD
    N_HEAD = args.n_head if args.n_head is not None else N_HEAD
    HEAD_DIM = N_EMBD // N_HEAD
    TOTAL_BATCH_SIZE = args.total_batch_size

    # Optimizer hyperparameters
    _lr_mult = args.lr_multiplier if args.lr_multiplier is not None else 1.0
    MATRIX_LR = args.matrix_lr * _lr_mult
    SCALAR_LR = args.scalar_lr * _lr_mult
    EMBEDDING_LR = args.embedding_lr * _lr_mult
    UNEMBEDDING_LR = args.unembedding_lr * _lr_mult
    WEIGHT_DECAY = args.weight_decay
    ADAM_BETAS = (0.8, 0.95)
    WARMUP_RATIO = 0.0
    WARMDOWN_RATIO = args.warmdown_ratio
    FINAL_LR_FRAC = 0.0

    # Compute init
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    master_process = ddp_rank == 0
    torch.manual_seed(42)

    if ddp and torch.cuda.is_available():
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(42)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_type = device.type
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

    # GPU info for MFU
    gpu_peak_flops = float('inf')
    if device_type == "cuda":
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "h100" in gpu_name: gpu_peak_flops = 989e12
        elif "a100" in gpu_name: gpu_peak_flops = 312e12
        elif "4090" in gpu_name: gpu_peak_flops = 165.2e12

    # FA3 status
    if _fa3 is not None:
        print0("Using Flash Attention 3 (Hopper GPU detected)")
    else:
        print0("Using PyTorch SDPA fallback (no FA3)")

    # Run / logging paths
    run_name, run_dir = resolve_run_dir(args.run_name)
    if dist.is_initialized():
        shared = [run_name]
        dist.broadcast_object_list(shared, src=0)
        run_name = shared[0]
        run_dir = os.path.join(RUNS_DIR, run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    artifact_model_path = os.path.join(run_dir, "model.pt")
    terminal_log_path = os.path.join(run_dir, "terminal.log")
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    artifacts_log_f = None
    result_path = os.path.join(run_dir, "result.json")
    os.makedirs(run_dir, exist_ok=True)
    if master_process:
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "wandb"), exist_ok=True)
        shutil.copy2(__file__, os.path.join(run_dir, "train.py"))
    if dist.is_initialized():
        dist.barrier()
    artifacts_log_f = open(terminal_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, artifacts_log_f)
    sys.stderr = TeeStream(sys.stderr, artifacts_log_f)

    # wandb
    _wandb_kwargs = {"project": "slowrun", "name": run_name}
    if args.wandb_group:
        _wandb_kwargs["group"] = args.wandb_group
    if run_dir:
        _wandb_kwargs["dir"] = os.path.join(run_dir, "wandb")
    wandb_run = DummyWandb() if not master_process else wandb.init(**_wandb_kwargs)
    if master_process:
        wandb_run.log_code(".")

    # Print hyperparameters
    print0(f"--- Hyperparameters ---")
    print0(f"  n_layer={DEPTH}, n_embd={N_EMBD}, n_head={N_HEAD}, head_dim={HEAD_DIM}")
    print0(f"  seq_len={MAX_SEQ_LEN}, window_pattern={WINDOW_PATTERN}")
    print0(f"  total_batch_size={TOTAL_BATCH_SIZE}, device_batch_size={args.device_batch_size}")
    print0(f"  matrix_lr={MATRIX_LR}, scalar_lr={SCALAR_LR}, embedding_lr={EMBEDDING_LR}, unembedding_lr={UNEMBEDDING_LR}")
    print0(f"  weight_decay={WEIGHT_DECAY}, adam_betas={ADAM_BETAS}")
    print0(f"  warmup_ratio={WARMUP_RATIO}, warmdown_ratio={WARMDOWN_RATIO}, final_lr_frac={FINAL_LR_FRAC}")
    print0(f"  num_epochs={args.num_epochs}, patience={args.patience}")
    print0(f"  dropout={args.dropout}")
    print0(f"  run={run_name}")
    print0(f"  run_dir={run_dir}")
    print0(f"  varlen={args.varlen}")
    print0(f"-----------------------")

    # Load GPT-2 tokenizer and compute token_bytes for BPB evaluation
    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.n_vocab  # 50257
    print0(f"Vocab size: {vocab_size:,}")

    eot_id = encoder._special_tokens['<|endoftext|>']
    token_bytes_list = []
    for i in range(vocab_size):
        if i == eot_id:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(encoder.decode_single_token_bytes(i)))
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

    # Build model
    config = GPTConfig(vocab_size=vocab_size, dropout=args.dropout, device_batch_size=args.device_batch_size)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    param_counts = sum(p.numel() for p in model.parameters())
    transformer_params = sum(p.numel() for p in model.transformer.h.parameters())
    ve_params = sum(p.numel() for p in model.ve_projs.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    other_params = param_counts - transformer_params - ve_params - lm_head_params
    num_flops_per_token = model.estimate_flops()
    print0(f"Parameters: {param_counts:,} (transformer: {transformer_params:,}, value_embeds: {ve_params:,}, lm_head: {lm_head_params:,}, other: {other_params:,})")
    print0(f"FLOPs per token: {num_flops_per_token:e}")

    # Compile
    orig_model = model
    model = torch.compile(model, dynamic=False, fullgraph=True)

    default_train_path = os.path.join(DATA_DIR, "fineweb_train_varlen.pt")
    default_val_path = os.path.join(DATA_DIR, "fineweb_val_varlen.pt")
    build_loader = lambda path, shuffle: VarlenDataLoader(
        path, args.device_batch_size, MAX_SEQ_LEN, device=device,
        shuffle=shuffle, varlen=args.varlen,
    )
    _val_path = args.input_val_bin if args.input_val_bin else default_val_path
    build_val_loader = lambda: build_loader(_val_path, False)
    eval_steps = EVAL_TOKENS // (args.device_batch_size * MAX_SEQ_LEN * ddp_world_size)
    _val_probe = build_val_loader()
    eval_steps = min(eval_steps, _val_probe.num_steps)
    del _val_probe

    # Strided eval setup
    do_strided = args.stride > 0
    # Optimizer
    optimizer = model.setup_optimizer()

    # Warmup torch.compile for each cu_seqlens bucket size (avoids recompiles during training)
    if args.varlen:
        total_tokens = args.device_batch_size * MAX_SEQ_LEN
        warmup_cu_buckets = tuple(VARLEN_CU_BUCKET_SIZE * i for i in range(1, 5))
        warmup_iters = 3
        warmup_x = torch.zeros(1, total_tokens, dtype=torch.long, device=device)
        warmup_y = torch.zeros(1, total_tokens, dtype=torch.long, device=device)
        print0(f"Warming up varlen compile buckets: {warmup_cu_buckets} ({warmup_iters} iters each)")
        model.train()
        for i, bucket_len in enumerate(warmup_cu_buckets):
            print0(f"\tBucket {i+1}/{len(warmup_cu_buckets)}: {bucket_len} toks")
            boundaries = list(range(0, total_tokens, MAX_SEQ_LEN))
            if boundaries[-1] != total_tokens:
                boundaries.append(total_tokens)
            cu = torch.full((bucket_len,), total_tokens, dtype=torch.int32, device=device)
            cu[:len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
            for _ in range(warmup_iters):
                with autocast_ctx:
                    wloss = model(warmup_x, warmup_y, cu_seqlens=cu, max_seqlen=MAX_SEQ_LEN)
                wloss.backward()
                optimizer.zero_grad(set_to_none=True)
        del warmup_x, warmup_y, wloss
        torch.cuda.empty_cache()
        print0("Varlen compile warmup done")

    # Dataloaders
    _train_path = args.input_bin if args.input_bin else default_train_path
    train_loader = build_loader(_train_path, True)
    TOKENS_PER_EPOCH = train_loader.total_tokens
    x, y, cu_seqlens, max_seqlen, current_epoch = next(train_loader)

    # Training config
    tokens_per_fwdbwd = args.device_batch_size * MAX_SEQ_LEN * ddp_world_size
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    num_iterations = round(TOKENS_PER_EPOCH * args.num_epochs / TOTAL_BATCH_SIZE)
    wd_phase1_end_step = round(args.wd_phase1_epoch / args.num_epochs * num_iterations)
    wd_phase2_end_step = round(args.wd_phase2_epoch / args.num_epochs * num_iterations)
    print0(f"Batch size: {TOTAL_BATCH_SIZE:,} tokens, grad accum: {grad_accum_steps} steps")
    print0(f"Training for {args.num_epochs} epoch(s) (~{num_iterations} steps estimated)")
    print0(f"Eval set: {EVAL_TOKENS:,} tokens")

    # Schedulers
    def get_lr_multiplier(it):
        warmup = round(WARMUP_RATIO * num_iterations)
        warmdown = round(WARMDOWN_RATIO * num_iterations)
        if it < warmup: return (it + 1) / warmup
        elif it <= num_iterations - warmdown: return 1.0
        else:
            progress = (num_iterations - it) / warmdown
            return progress + (1 - progress) * FINAL_LR_FRAC

    def get_muon_momentum(it):
        return (1 - min(it / 300, 1)) * 0.85 + min(it / 300, 1) * 0.95

    # Training loop
    step = 0
    min_val_bpb = float("inf")
    min_val_loss = float("inf")
    epochs_without_improvement = 0
    smooth_train_loss = 0
    total_training_time = 0
    recent_dts = []
    steps_per_epoch = num_iterations / args.num_epochs
    param_ema_beta = args.ema_decay_per_epoch ** (args.update_ema_every / steps_per_epoch) if args.update_ema_every > 0 else 0
    ema_params = [torch.zeros_like(p) for p in model.parameters()] if args.update_ema_every > 0 else None

    wall_clock_start = time.time()
    _swa_start_step = (num_iterations - args.swa_last_epochs * steps_per_epoch) if args.swa_last_epochs > 0 else -1
    late_ckpt_paths = []

    # Initial val evaluation
    model.eval()
    val_loader = build_val_loader()
    with autocast_ctx:
        val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes, varlen=args.varlen)
    print0(f"Step {step:05d} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
    wandb_run.log({"step": step, "val/bpb": val_bpb, "val/loss": val_loss})
    min_val_bpb = val_bpb
    min_val_loss = val_loss
    model.train()

    while current_epoch <= args.num_epochs:
        # Training step
        synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                if args.varlen:
                    loss = model(x, y, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                else:
                    loss = model(x, y)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, cu_seqlens, max_seqlen, epoch = next(train_loader)

        # Update optimizer
        lrm = get_lr_multiplier(step)
        # SWA: cosine-cycle LR in final epochs for diverse checkpoints to average
        if _swa_start_step >= 0 and step >= _swa_start_step:
            cycle_pos = (step - _swa_start_step) % steps_per_epoch
            swa_base = max(lrm, 0.05)
            lrm = 0.05 + (swa_base - 0.05) * (1 + math.cos(math.pi * cycle_pos / steps_per_epoch)) / 2
        # WD schedule:
        #   [0, wd_phase1_end_step]:              hold at weight_decay
        #   [wd_phase1_end_step, wd_phase2_end_step]: decay to wd_mid
        #   [wd_phase2_end_step, num_iterations]:     ramp up to wd_end
        wd = np.interp(step,
            [0, wd_phase1_end_step, wd_phase2_end_step, num_iterations],
            [args.weight_decay, args.weight_decay, args.wd_mid, args.wd_end])
        # Convert to a scale factor;
        # groups with weight_decay=0.0 (scalar params) correctly stay at zero.
        wd_scale = wd / args.weight_decay if args.weight_decay > 0 else 0.0
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if "initial_wd" not in group:
                group["initial_wd"] = group.get("weight_decay", 0.0)
            group["weight_decay"] = group["initial_wd"] * wd_scale
            if group['kind'] == 'muon':
                group["momentum"] = get_muon_momentum(step)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        if ema_params is not None and step % args.update_ema_every == 0:
            torch._foreach_lerp_(ema_params, list(model.parameters()), 1 - param_ema_beta)
        train_loss_f = train_loss.item()
        synchronize()
        dt = time.time() - t0

        step += 1

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased = smooth_train_loss / (1 - ema_beta**step)
        pct = 100 * step / num_iterations
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / (gpu_peak_flops * ddp_world_size)
        if step > 3:
            total_training_time += dt
            recent_dts.append(dt)
            if len(recent_dts) > 10:
                recent_dts.pop(0)
        eta_str = f" | eta: {(num_iterations - step) * sum(recent_dts) / len(recent_dts) / 60:.1f}m" if recent_dts else ""
        print0(f"step {step:05d} ({pct:.2f}%) | loss: {debiased:.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}%{eta_str}")
        wandb_run.log({"step": step, "train/loss": debiased, "train/mfu": mfu})

        # Synchronize epoch across ranks (different ranks may exhaust data at different steps)
        if ddp:
            epoch_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
            dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MAX)
            epoch = epoch_tensor.item()

        # Epoch boundary: evaluate when the dataloader advances to a new epoch
        if epoch != current_epoch:
            model.eval()
            val_loader = build_val_loader()
            with autocast_ctx:
                val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes, varlen=args.varlen)
            print0(f"Step {step:05d} | Epoch {current_epoch} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
            wandb_run.log({"step": step, "epoch": current_epoch, "val/bpb": val_bpb, "val/loss": val_loss})
            # Save checkpoint for weight averaging
            ckpt_path = os.path.join(checkpoints_dir, f"epoch_{current_epoch:03d}.pt")
            if master_process:
                os.makedirs(checkpoints_dir, exist_ok=True)
                torch.save({n: p.data.float().cpu() for n, p in orig_model.named_parameters()}, ckpt_path)
            late_ckpt_paths.append(ckpt_path)
            if len(late_ckpt_paths) > args.swa_last_epochs:
                old = late_ckpt_paths.pop(0)
                if master_process and os.path.exists(old): os.remove(old)
            # Early stopping
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
                min_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if args.patience >= 0 and epochs_without_improvement >= args.patience:
                    print0(f"Early stopping: no improvement for {args.patience} epoch(s)")
                    break

            model.train()
            current_epoch = epoch

        # GC management
        if step == 1:
            gc.collect(); gc.freeze(); gc.disable()

    # Final EMA evaluation
    if ema_params is not None:
        ema_updates = step // args.update_ema_every
        if ema_updates > 0:
            correction = 1.0 / (1.0 - param_ema_beta ** ema_updates)
            model.eval()
            with torch.no_grad():
                for p, ema in zip(model.parameters(), ema_params):
                    p.copy_(ema * correction)
            val_loader = build_val_loader()
            with autocast_ctx:
                ema_bpb, ema_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes, varlen=args.varlen)
            print0(f"EMA Val BPB: {ema_bpb:.6f} | EMA Val Loss: {ema_loss:.6f}")
            wandb_run.log({"step": step, "val/ema_bpb": ema_bpb, "val/ema_loss": ema_loss})
            val_bpb = ema_bpb
            val_loss = ema_loss
            if ema_bpb < min_val_bpb:
                min_val_bpb = ema_bpb
                min_val_loss = ema_loss

    # Checkpoint weight averaging (recency-weighted)
    if len(late_ckpt_paths) >= 2:
        if ddp: dist.barrier()
        n = len(late_ckpt_paths)
        raw_w = list(range(1, n + 1))
        weights = [w / sum(raw_w) for w in raw_w]
        if master_process:
            ckpts = [torch.load(p, map_location="cpu", weights_only=True) for p in late_ckpt_paths]
            merged = {name: sum(w * ckpts[i][name].float() for i, w in enumerate(weights)) for name in ckpts[0]}
            with torch.no_grad():
                for name, p in orig_model.named_parameters():
                    if name in merged: p.copy_(merged[name].to(p.device, p.dtype))
        if ddp:
            dist.barrier()
            for p in orig_model.parameters(): dist.broadcast(p.data, src=0)
        model.eval()
        val_loader = build_val_loader()
        with autocast_ctx:
            avg_bpb, avg_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes, varlen=args.varlen)
        print0(f"Ckpt avg Val BPB: {avg_bpb:.6f} | Val Loss: {avg_loss:.6f}")
        wandb_run.log({"ckpt_avg/bpb": avg_bpb, "ckpt_avg/loss": avg_loss})
        if avg_loss < min_val_loss:
            min_val_loss, min_val_bpb = avg_loss, avg_bpb

    # Strided eval at end of training
    if do_strided:
        model.eval()
        with autocast_ctx:
            s_bpb, s_loss = evaluate_bpb_strided(
                model, default_val_path, token_bytes,
                args.stride, MAX_SEQ_LEN, args.device_batch_size, device,
                varlen=args.varlen)
        print0(f"Strided (stride={args.stride}) | Val BPB: {s_bpb:.6f} | Val Loss: {s_loss:.6f}")
        wandb_run.log({"strided/bpb": s_bpb, "strided/loss": s_loss})

    # Summary
    wall_clock_time = time.time() - wall_clock_start
    print0(f"Wall clock time: {wall_clock_time/60:.2f}m")
    print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
    print0(f"Total training time: {total_training_time/60:.2f}m")
    final_train_loss = smooth_train_loss / (1 - 0.9**step) if step > 0 else float('inf')
    print0(f"Final train loss: {final_train_loss:.6f}")
    print0(f"Min val BPB: {min_val_bpb:.6f}")
    print0(f"Min val Loss: {min_val_loss:.6f}")
    wandb_run.summary["final_train_loss"] = final_train_loss
    wandb_run.summary["best_val_loss"] = min_val_loss

    if master_process:
        result = {
            "matrix_lr": args.matrix_lr,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "val_loss": val_loss,
            "best_val_loss": min_val_loss,
            "wandb_url": getattr(wandb_run, "url", None),
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print0(f"Result saved to {result_path}")

    # Save final model
    if master_process:
        print0(f"Saving model to {artifact_model_path}")
        os.makedirs(os.path.dirname(artifact_model_path), exist_ok=True)
        torch.save({n: p.data.float().cpu() for n, p in orig_model.named_parameters()}, artifact_model_path)
        print0(f"Model saved to {artifact_model_path}")

    print0(f"Min val BPB: {min_val_bpb:.6f} | Min val Loss: {min_val_loss:.6f}")
    total_wall_time = time.time() - _script_start
    print0(f"Total wall time: {total_wall_time:.2f}s ({total_wall_time/60:.2f}m)")

    wandb_run.finish()
    if dist.is_initialized():
        dist.destroy_process_group()
    if artifacts_log_f is not None:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        artifacts_log_f.close()
