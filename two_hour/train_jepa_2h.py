"""
JEPA v3 for the slowrun competition.

Joint Embedding Predictive Architecture: CE loss + span-masked latent prediction
with EMA target encoder and VICReg anti-collapse regularization.

Ported from parameter-golf v3 to GPT-2 tokenizer and slowrun .pt data format.

Usage:
    torchrun --standalone --nproc_per_node=8 unlimited/train_jepa.py
"""

from __future__ import annotations

import copy
import math
import os
import random
from dotenv import load_dotenv
load_dotenv()
import time
import argparse

import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from tokenizer import get_encoding
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ============================================================================
# CLI args
# ============================================================================

parser = argparse.ArgumentParser(description="JEPA training for slowrun unlimited track")
# Model
parser.add_argument("--n-layer",          type=int,   default=30)
parser.add_argument("--n-embd",           type=int,   default=2048)
parser.add_argument("--n-head",           type=int,   default=16)
parser.add_argument("--n-kv-head",        type=int,   default=16)
parser.add_argument("--mlp-mult",         type=int,   default=4)
parser.add_argument("--vocab-size",       type=int,   default=50257)
parser.add_argument("--tie-embeddings",   action="store_true", default=False)
parser.add_argument("--rope-base",        type=float, default=10000.0)
parser.add_argument("--logit-softcap",    type=float, default=30.0)
parser.add_argument("--qk-gain-init",     type=float, default=4.0)
parser.add_argument("--mlp-leaky-slope",  type=float, default=0.5)
# Training
parser.add_argument("--num-epochs",        type=int,   default=32)
parser.add_argument("--device-batch-size", type=int,   default=2)
parser.add_argument("--total-batch-size",  type=int,   default=524288)
parser.add_argument("--seq-len",           type=int,   default=2048)
parser.add_argument("--seed",              type=int,   default=1337)
parser.add_argument("--warmdown-frac",     type=float, default=0.20)
parser.add_argument("--val-every",         type=int,   default=200)
parser.add_argument("--val-steps",         type=int,   default=50)
parser.add_argument("--val-batch",         type=int,   default=16)
# Optimizer
parser.add_argument("--matrix-lr",         type=float, default=0.04)
parser.add_argument("--scalar-lr",         type=float, default=0.10)
parser.add_argument("--embed-lr",          type=float, default=0.15)
parser.add_argument("--head-lr",           type=float, default=0.002)
parser.add_argument("--tied-embed-lr",     type=float, default=0.05)
parser.add_argument("--muon-momentum",     type=float, default=0.95)
parser.add_argument("--muon-backend-steps",type=int,   default=5)
parser.add_argument("--muon-warmup-steps", type=int,   default=500)
parser.add_argument("--muon-warmup-start", type=float, default=0.85)
parser.add_argument("--beta1",             type=float, default=0.9)
parser.add_argument("--beta2",             type=float, default=0.95)
parser.add_argument("--adam-eps",          type=float, default=1e-8)
parser.add_argument("--grad-clip-norm",    type=float, default=0.0)
parser.add_argument("--dropout",           type=float, default=0.0)
parser.add_argument("--weight-decay",      type=float, default=0.0)
# JEPA
parser.add_argument("--jepa-lambda",        type=float, default=0.12)
parser.add_argument("--jepa-ema-start",     type=float, default=0.9)
parser.add_argument("--jepa-pred-dim",      type=int,   default=256)
parser.add_argument("--jepa-warmup-steps",  type=int,   default=100)
parser.add_argument("--jepa-num-spans",     type=int,   default=4)
parser.add_argument("--jepa-span-len-mean", type=int,   default=16)
parser.add_argument("--jepa-span-len-min",  type=int,   default=4)
parser.add_argument("--jepa-var-weight",    type=float, default=0.15)
parser.add_argument("--jepa-cov-weight",    type=float, default=0.02)
parser.add_argument("--jepa-var-gamma",     type=float, default=1.0)
parser.add_argument("--jepa-var-eps",       type=float, default=1e-4)
parser.add_argument("--bigram-vocab-size",  type=int,   default=0)
# IO
parser.add_argument("--input-bin",      type=str, default=None)
parser.add_argument("--input-val-bin",  type=str, default=None)
parser.add_argument("--checkpoint-path",type=str, default="jepa_checkpoint.pt")
parser.add_argument("--run",            type=str, default=None)
parser.add_argument("--wandb-group",    type=str, default=None)
parser.add_argument("--wandb-offline",  action="store_true")
args = parser.parse_args()

DATA_DIR = "fineweb_data"

# Scalar/control param names — excluded from Muon, routed to Adam
_CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")

# ============================================================================
# Distributed helpers
# ============================================================================

def get_dist_info() -> tuple[bool, int, int, int]:
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return (True, int(os.environ["RANK"]),
                int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]))
    return False, 0, 0, 1


def print0(s: str = "", **kw) -> None:
    if int(os.environ.get("RANK", 0)) == 0:
        print(s, **kw)


class DummyWandb:
    def __init__(self):
        self.summary = {}
    def log(self, *a, **kw): pass
    def finish(self): pass

# ============================================================================
# Muon optimizer
# ============================================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float,
                 backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# ============================================================================
# DataLoader  (slowrun .pt format)
# ============================================================================

class DataLoader:
    """Epoch-based DataLoader for slowrun .pt sequence files."""

    def __init__(self, filepath: str, rank: int, world_size: int, device: torch.device):
        data = torch.load(filepath, weights_only=True)
        tokens = data["tokens"].to(torch.int64)   # uint16 → int64, safe for 0-50256
        seq_size = int(data["seq_size"])           # 2049 = seq_len + 1
        self.seq_len = seq_size - 1
        self.n_seqs = tokens.numel() // seq_size
        self.tokens = tokens
        # start position of each sequence (in flat token index space)
        self.starts = torch.arange(self.n_seqs, dtype=torch.int64) * seq_size
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.epoch = 0
        self.pos = 0
        self._new_epoch()

    def _new_epoch(self) -> None:
        rng = np.random.default_rng(self.epoch + 1337)
        self.order = rng.permutation(self.n_seqs)
        self.pos = 0
        self.epoch += 1

    def next_batch(self, local_batch: int) -> tuple[Tensor, Tensor]:
        """Return (x, y) of shape [local_batch, seq_len] int64 on device."""
        needed = local_batch * self.world_size
        if self.pos + needed > self.n_seqs:
            self._new_epoch()
        idx_global = self.order[self.pos:self.pos + needed]
        self.pos += needed
        my_idx = idx_global[self.rank::self.world_size][:local_batch]
        seqs = torch.stack([
            self.tokens[int(self.starts[i]):int(self.starts[i]) + self.seq_len + 1]
            for i in my_idx
        ])
        seqs = seqs.to(self.device, non_blocking=True)
        return seqs[:, :-1], seqs[:, 1:]

# ============================================================================
# BPB evaluation
# ============================================================================

@torch.no_grad()
def evaluate_bpb(
    raw_model: nn.Module,
    val_seqs: list[Tensor],
    token_bytes_lut: Tensor,
    device: torch.device,
    val_batch: int,
    val_steps: int,
    rank: int,
) -> tuple[float, float]:
    raw_model.eval()
    total_nats  = torch.zeros((), device=device, dtype=torch.float64)
    total_bytes = torch.zeros((), device=device, dtype=torch.float64)
    total_toks  = torch.zeros((), device=device, dtype=torch.float64)

    rng = np.random.default_rng(rank + 9999)
    for _ in range(val_steps):
        idx = rng.integers(0, len(val_seqs), size=val_batch)
        batch = torch.stack([val_seqs[i] for i in idx]).to(device).long()
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            per_tok = raw_model(x, y, loss_reduction="none").double()  # [B*T]
        total_nats  += per_tok.sum()
        total_bytes += token_bytes_lut[y.reshape(-1)].double().sum()
        total_toks  += float(y.numel())

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_nats)
        dist.all_reduce(total_bytes)
        dist.all_reduce(total_toks)

    bpb      = (total_nats / total_bytes / math.log(2)).item()
    val_loss = (total_nats / total_toks).item()
    raw_model.train()
    return val_loss, bpb

# ============================================================================
# Model
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Weights stay fp32; cast to input dtype at matmul time."""
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            is_control = any(pat in name for pat in _CONTROL_PATTERNS)
            if (param.ndim < 2 or is_control) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device,
                dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (self._cos_cached is None or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q   = CastedLinear(dim, dim,    bias=False)
        self.c_k   = CastedLinear(dim, kv_dim, bias=False)
        self.c_v   = CastedLinear(dim, kv_dim, bias=False)
        self.proj  = CastedLinear(dim, dim,    bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, leaky_slope: float = 0.5):
        super().__init__()
        hidden = mlp_mult * dim
        self.leaky_slope = leaky_slope
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim,  bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc(x)
        h = F.leaky_relu(h, self.leaky_slope) if self.leaky_slope > 0.0 else torch.relu(h)
        return self.proj(h.square())


class JEPAPredictor(nn.Module):
    """Residual ReLU² MLP: maps z_context → z_pred at masked positions."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = RMSNorm()
        self.fc   = CastedLinear(dim, hidden_dim, bias=False)
        self.proj = CastedLinear(hidden_dim, dim,  bias=False)
        self.proj._zero_init = True  # starts as identity through residual

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.fc(self.norm(x)))
        return x + self.proj(h.square())


class BigramHashEmbedding(nn.Module):
    """Cantor-hashed bigram lookup table. Disabled by default (--bigram-vocab-size 0)."""
    def __init__(self, bigram_vocab_size: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embedding = nn.Embedding(bigram_vocab_size, model_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, input_ids: Tensor) -> Tensor:
        prev = torch.cat([input_ids.new_zeros(input_ids.shape[0], 1), input_ids[:, :-1]], dim=1)
        a, b = prev.long(), input_ids.long()
        s = a + b
        h = (s * (s + 1) // 2 + b) % self.bigram_vocab_size
        return self.embedding(h)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float, qk_gain_init: float,
                 leaky_slope: float, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp  = MLP(dim, mlp_mult, leaky_slope)
        self.dropout    = nn.Dropout(dropout)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        leaky_slope: float,
        dropout: float = 0.0,
        jepa_pred_dim: int = 0,
        bigram_vocab_size: int = 0,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap  = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )
        block_kw = dict(
            dim=model_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult, rope_base=rope_base,
            qk_gain_init=qk_gain_init, leaky_slope=leaky_slope,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList([Block(**block_kw) for _ in range(num_layers)])
        self.final_norm = RMSNorm()
        self.lm_head: CastedLinear | None = None
        if not tie_embeddings:
            self.lm_head = CastedLinear(model_dim, vocab_size, bias=False)
            self.lm_head._zero_init = True
        self.jepa_predictor: JEPAPredictor | None = (
            JEPAPredictor(model_dim, jepa_pred_dim) if jepa_pred_dim > 0 else None
        )
        self.bigram_hash_emb: BigramHashEmbedding | None = (
            BigramHashEmbedding(bigram_vocab_size, model_dim) if bigram_vocab_size > 0 else None
        )
        # Mask embedding: replaces token embeddings at JEPA target span positions.
        # Zero-init so it starts neutral; trained by the JEPA loss.
        self.jepa_mask_emb = nn.Parameter(torch.zeros(model_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def encode(self, input_ids: Tensor, jepa_mask: Tensor | None = None) -> Tensor:
        """U-Net forward → hidden states [B, T, D].

        jepa_mask: bool [B, T]; True positions have their token embedding
        replaced with jepa_mask_emb. Pass None for CE forward and target encoder.
        """
        x = self.tok_emb(input_ids)
        if self.bigram_hash_emb is not None:
            bigram = self.bigram_hash_emb(input_ids)
            if jepa_mask is not None:
                bigram = bigram.masked_fill(jepa_mask.unsqueeze(-1), 0.0)
            x = x + bigram
        if jepa_mask is not None:
            mask_vec = self.jepa_mask_emb.to(dtype=x.dtype)
            x = torch.where(jepa_mask.unsqueeze(-1), mask_vec, x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def forward(self, input_ids: Tensor, target_ids: Tensor,
                loss_reduction: str = "mean") -> Tensor:
        """CE-only forward. The JEPA path is handled explicitly in the training loop."""
        z = self.encode(input_ids)                        # [B, T, D]
        z_flat  = z.reshape(-1, z.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(z_flat, self.tok_emb.weight)
        else:
            assert self.lm_head is not None
            logits_proj = self.lm_head(z_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction=loss_reduction)

# ============================================================================
# Span masking + VICReg helpers
# ============================================================================

def sample_block_spans(
    seq_len: int, num_spans: int, span_len_mean: int,
    span_len_min: int = 4, device: torch.device | None = None,
) -> Tensor:
    """Sample num_spans non-overlapping spans via geometric length distribution."""
    max_span_len = max(span_len_min, seq_len // (2 * num_spans))
    p = 1.0 / span_len_mean
    spans = []
    for _ in range(num_spans):
        length = int(math.floor(math.log(random.random()) / math.log(1.0 - p))) + 1
        length = min(max_span_len, max(span_len_min, length))
        start  = random.randint(0, seq_len - length)
        spans.append((start, start + length))
    spans.sort()
    resolved, cursor = [], 0
    for s, e in spans:
        s = max(s, cursor)
        if s >= seq_len:
            break
        e = min(e, seq_len)
        if e - s >= span_len_min:
            resolved.append((s, e))
            cursor = e
    while len(resolved) < num_spans and resolved:
        resolved.append(resolved[-1])
    return torch.tensor(resolved[:num_spans], dtype=torch.long, device=device)


def vicreg_var_loss(z: Tensor, gamma: float, eps: float) -> Tensor:
    """Hinge: penalize per-feature std < gamma across the batch of masked tokens."""
    n  = z.shape[0]
    zc = z - z.mean(dim=0)
    std = (zc.pow(2).sum(dim=0) / (n - 1) + eps).sqrt()
    return (gamma - std).clamp(min=0.0).mean()


def vicreg_cov_loss(z: Tensor) -> Tensor:
    """Off-diagonal covariance penalty: decorrelate feature dimensions."""
    n, d = z.shape
    zc  = z - z.mean(dim=0)
    cov = zc.T @ zc / (n - 1)
    off = cov.pow(2)
    off.fill_diagonal_(0.0)
    return off.sum() / d

# ============================================================================
# LR schedule
# ============================================================================

def get_lr_multiplier(it: int, num_iterations: int, warmdown_frac: float) -> float:
    warmdown_start = round((1.0 - warmdown_frac) * num_iterations)
    if it <= warmdown_start:
        return 1.0
    progress = (num_iterations - it) / max(num_iterations - warmdown_start, 1)
    return max(progress, 0.0)

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    global zeropower_via_newtonschulz5
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    ddp, rank, local_rank, world_size = get_dist_info()
    if ddp:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    # --- Wandb (rank 0 only) ---
    run_name = args.run if args.run else f"jepa_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb_kw = {"project": "slowrun", "name": run_name}
    if args.wandb_group:
        wandb_kw["group"] = args.wandb_group
    if args.wandb_offline:
        wandb_kw["mode"] = "offline"
    wandb_run = DummyWandb() if rank != 0 else wandb.init(**wandb_kw)

    # --- Token bytes LUT for BPB ---
    enc = get_encoding("gpt2")
    eot_id = enc._special_tokens["<|endoftext|>"]
    token_bytes_lut = torch.tensor(
        [0 if i == eot_id else len(enc.decode_single_token_bytes(i))
         for i in range(args.vocab_size)],
        dtype=torch.int32, device=device,
    )

    # --- Data ---
    train_path = args.input_bin     or os.path.join(DATA_DIR, "fineweb_train.pt")
    val_path   = args.input_val_bin or os.path.join(DATA_DIR, "fineweb_val.pt")
    print0("Loading data...")
    train_loader = DataLoader(train_path, rank, world_size, device)
    val_data   = torch.load(val_path, weights_only=True)
    val_tokens = val_data["tokens"].to(torch.int64)
    val_seq_sz = int(val_data["seq_size"])
    val_n_seqs = val_tokens.numel() // val_seq_sz
    val_seqs   = [val_tokens[i * val_seq_sz:(i + 1) * val_seq_sz] for i in range(val_n_seqs)]
    print0(f"  Train: {train_loader.n_seqs:,} seqs  |  Val: {val_n_seqs:,} seqs")

    # --- Model ---
    print0(f"\n{'='*60}")
    print0(f"JEPA  {args.n_layer}L  {args.n_embd}d  {args.n_head}h  vocab={args.vocab_size}")
    print0(f"  jepa_lambda={args.jepa_lambda}  ema_start={args.jepa_ema_start}"
           f"  pred_dim={args.jepa_pred_dim}  num_spans={args.jepa_num_spans}")
    print0(f"dropout: {args.dropout}")
    print0(f"weight decay: {args.weight_decay}")
    print0(f"{'='*60}")

    base_model = GPT(
        vocab_size      = args.vocab_size,
        num_layers      = args.n_layer,
        model_dim       = args.n_embd,
        num_heads       = args.n_head,
        num_kv_heads    = args.n_kv_head,
        mlp_mult        = args.mlp_mult,
        tie_embeddings  = args.tie_embeddings,
        logit_softcap   = args.logit_softcap,
        rope_base       = args.rope_base,
        qk_gain_init    = args.qk_gain_init,
        leaky_slope     = args.mlp_leaky_slope,
        dropout         = args.dropout,
        jepa_pred_dim   = args.jepa_pred_dim,
        bigram_vocab_size = args.bigram_vocab_size,
    ).to(device).bfloat16()

    # JEPA EMA target encoder: frozen copy, not saved in checkpoint
    jepa_target_encoder = copy.deepcopy(base_model).to(device).bfloat16()
    for p in jepa_target_encoder.parameters():
        p.requires_grad_(False)

    # Cast CastedLinear weights to fp32; restore control scalars to fp32
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model          = torch.compile(base_model, dynamic=False, fullgraph=True)
    compiled_target_encoder = torch.compile(jepa_target_encoder, dynamic=False, fullgraph=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if ddp else compiled_model
    )

    n_params = sum(p.numel() for p in base_model.parameters())
    print0(f"Parameters: {n_params:,}")

    # --- Optimizer ---
    # matrix params (2D, non-control) → Muon
    # scalar/control params           → Adam (scalar_lr)
    # tok_emb                         → Adam (embed_lr or tied_embed_lr)
    # lm_head                         → Adam (head_lr)
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in _CONTROL_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in _CONTROL_PATTERNS)
    ]
    # params outside base_model.blocks
    scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.jepa_mask_emb)
    if base_model.jepa_predictor is not None:
        for name, p in base_model.jepa_predictor.named_parameters():
            (matrix_params if p.ndim == 2 else scalar_params).append(p)
    if base_model.bigram_hash_emb is not None:
        scalar_params.extend(base_model.bigram_hash_emb.parameters())

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr,
                          momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in optimizer_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
            weight_decay=args.weight_decay,
        )
        optimizers.insert(1, optimizer_head)

    # --- Training schedule ---
    grad_accum_steps = max(
        1, args.total_batch_size // (args.device_batch_size * world_size * args.seq_len)
    )
    tokens_per_step  = args.device_batch_size * world_size * grad_accum_steps * args.seq_len
    total_tokens     = train_loader.n_seqs * args.seq_len
    num_iterations   = round(total_tokens * args.num_epochs / tokens_per_step)
    grad_scale       = 1.0 / grad_accum_steps

    print0(f"grad_accum={grad_accum_steps}  tokens/step={tokens_per_step:,}"
           f"  iterations={num_iterations}  world_size={world_size}")
    print0(f"{'='*60}\n")

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    model.train()
    best_val_bpb = float("inf")
    training_time_ms = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_start = t0

    for step in range(num_iterations + 1):
        is_last = (step == num_iterations)
        if not is_last and time.perf_counter() - run_start >= 2 * 3600:
            print0(f"2h cap reached at step {step} — stopping.")
            is_last = True
        do_val  = is_last or (args.val_every > 0 and step % args.val_every == 0)

        if do_val:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = evaluate_bpb(
                base_model, val_seqs, token_bytes_lut, device,
                args.val_batch, args.val_steps, rank,
            )
            step_avg = training_time_ms / max(step, 1)
            print0(
                f"step:{step}/{num_iterations}  val_loss:{val_loss:.4f}  val_bpb:{val_bpb:.4f}"
                f"  train_time:{training_time_ms:.0f}ms  step_avg:{step_avg:.1f}ms"
            )
            wandb_run.log({"step": step, "val_loss": val_loss, "val_bpb": val_bpb})
            if rank == 0 and val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                torch.save({
                    "model": base_model.state_dict(),
                    "step":  step,
                    "config": {
                        "n_layer": args.n_layer, "n_embd": args.n_embd,
                        "n_head": args.n_head, "n_kv_head": args.n_kv_head,
                        "mlp_mult": args.mlp_mult, "vocab_size": args.vocab_size,
                        "tie_embeddings": args.tie_embeddings,
                        "jepa_pred_dim": args.jepa_pred_dim,
                    },
                }, args.checkpoint_path)
                print0(f"  → checkpoint saved  (val_bpb={best_val_bpb:.4f})")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if is_last:
            break

        # --- LR + momentum schedules ---
        lr_mul = get_lr_multiplier(step, num_iterations, args.warmdown_frac)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_mul

        muon_frac = min(step / max(args.muon_warmup_steps, 1), 1.0)
        muon_mom  = (1 - muon_frac) * args.muon_warmup_start + muon_frac * args.muon_momentum
        for g in optimizer_muon.param_groups:
            g["momentum"] = muon_mom

        jepa_frac = min(step / max(args.jepa_warmup_steps, 1) + 1e-6, 1.0) # Avoid abs. 0 for DDP
        eff_jepa  = args.jepa_lambda * jepa_frac

        zero_grad_all()

        train_ce   = torch.zeros((), device=device)
        train_jmse = torch.zeros((), device=device)
        train_varp = torch.zeros((), device=device)
        train_covp = torch.zeros((), device=device)
        train_jloss= torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            x, y = train_loader.next_batch(args.device_batch_size)

            # CE: through compiled DDP model (gradient all-reduce fires on last micro-step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ce_loss = model(x, y)

            # JEPA: through base_model.encode() directly (intentionally no DDP sync)
            jloss_m = jmse_m = varp_m = covp_m = ce_loss.new_zeros(())
            if eff_jepa > 0.0:
                spans = sample_block_spans(
                    x.shape[1], args.jepa_num_spans,
                    args.jepa_span_len_mean, args.jepa_span_len_min,
                )
                jepa_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)
                for s, e in spans.tolist():
                    jepa_mask[:, s:e] = True

                # Target encoder: full sequence, no grad
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_target = compiled_target_encoder.encode(x).detach()  # [B, T, D]

                # Context encoder: masked sequence, grads flow
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_context = base_model.encode(x, jepa_mask=jepa_mask)  # [B, T, D]
                    z_pred    = base_model.jepa_predictor(z_context)        # [B, T, D]

                z_p = z_pred[jepa_mask].float()    # [N_masked, D]
                z_t = z_target[jepa_mask].float()  # [N_masked, D]

                jmse_m  = F.mse_loss(z_p, z_t)
                varp_m  = vicreg_var_loss(z_p, args.jepa_var_gamma, args.jepa_var_eps)
                covp_m  = vicreg_cov_loss(z_p)
                jloss_m = jmse_m + args.jepa_var_weight * varp_m + args.jepa_cov_weight * covp_m

            loss = ce_loss + eff_jepa * jloss_m
            (loss * grad_scale).backward()

            train_ce    += ce_loss.detach()
            train_jmse  += jmse_m.detach()
            train_varp  += varp_m.detach()
            train_covp  += covp_m.detach()
            train_jloss += jloss_m.detach()

        train_ce    /= grad_accum_steps
        train_jmse  /= grad_accum_steps
        train_varp  /= grad_accum_steps
        train_covp  /= grad_accum_steps
        train_jloss /= grad_accum_steps

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        if args.weight_decay > 0.0:
            wd_factor = 1.0 - args.matrix_lr * lr_mul * args.weight_decay
            for p in matrix_params:
                p.data.mul_(wd_factor)

        # JEPA EMA: momentum ramps from jepa_ema_start → 0.999 over training
        frac_done = step / max(num_iterations, 1)
        ema_mom   = args.jepa_ema_start + (0.999 - args.jepa_ema_start) * frac_done
        with torch.no_grad():
            src = [p.data.to(dtype=tp.dtype)
                   for p, tp in zip(base_model.parameters(), jepa_target_encoder.parameters())]
            tgt = [p.data for p in jepa_target_encoder.parameters()]
            torch._foreach_lerp_(tgt, src, 1.0 - ema_mom)

        if rank == 0 and (step < 10 or step % 100 == 0):
            elapsed = 1000.0 * (time.perf_counter() - t0)
            print0(
                f"step:{step + 1}/{num_iterations}"
                f"  ce:{train_ce.item():.4f}"
                f"  jmse:{train_jmse.item():.4f}"
                f"  var_p:{train_varp.item():.4f}"
                f"  cov_p:{train_covp.item():.4f}"
                f"  ema:{ema_mom:.4f}"
                f"  lr_mul:{lr_mul:.3f}"
                f"  jepa_f:{jepa_frac:.2f}"
                f"  {elapsed:.0f}ms"
            )
            wandb_run.log({
                "step":             step + 1,
                "train_ce_loss":    train_ce.item(),
                "jepa_mse":         train_jmse.item(),
                "jepa_var_p":       train_varp.item(),
                "jepa_cov_p":       train_covp.item(),
                "jepa_lambda_eff":  eff_jepa,
                "jepa_frac":        jepa_frac,
                "jepa_total":       eff_jepa * train_jloss.item(),
                "jmse_contrib":     eff_jepa * train_jmse.item(),
                "var_contrib":      eff_jepa * args.jepa_var_weight * train_varp.item(),
                "cov_contrib":      eff_jepa * args.jepa_cov_weight * train_covp.item(),
                "ema_momentum":     ema_mom,
                "muon_momentum":    muon_mom,
                "lr_multiplier":    lr_mul,
                "lr_matrix":        args.matrix_lr * lr_mul,
                "lr_scalar":        args.scalar_lr * lr_mul,
            })

    # --- Final checkpoint ---
    if rank == 0:
        torch.save({
            "model": base_model.state_dict(),
            "step":  num_iterations,
            "config": {
                "n_layer": args.n_layer, "n_embd": args.n_embd,
                "n_head": args.n_head, "n_kv_head": args.n_kv_head,
                "mlp_mult": args.mlp_mult, "vocab_size": args.vocab_size,
                "tie_embeddings": args.tie_embeddings,
                "jepa_pred_dim": args.jepa_pred_dim,
            },
        }, args.checkpoint_path)
        print0(f"\nFinal checkpoint → {args.checkpoint_path}")
    print0(
        f"peak_memory_allocated:{torch.cuda.max_memory_allocated() // 1024 // 1024}MiB"
        f"  reserved:{torch.cuda.max_memory_reserved() // 1024 // 1024}MiB"
    )
    wandb_run.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
