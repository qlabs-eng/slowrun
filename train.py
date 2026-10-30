"""
Train a language model on ~100M tokens with val loss evaluation.
Code is based on Nanochat (https://github.com/karpathy/nanochat), with modifications to support the slowrun setting.

Usage:
    torchrun --standalone --nproc_per_node=8 train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import json
import base64
import hashlib
import sys
import shutil
import argparse
from types import SimpleNamespace
from functools import partial
from dataclasses import dataclass
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
import wandb
import tiktoken

_script_start = time.time()

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Train GPT model")
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--num-epochs", type=int, default=11)
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument("--run-name", type=str, default=None,
                    help="Run name under runs/ (default: random 6-char string)")
parser.add_argument("--scalar-lr", type=float, default=0.1)
parser.add_argument("--matrix-lr", type=float, default=0.04)
parser.add_argument("--weight-decay", type=float, default=1.3)
parser.add_argument("--total-batch-size", type=int, default=524288)
parser.add_argument("--save-result", type=str, default="")
parser.add_argument("--n_layer", type=int, default=30)
parser.add_argument("--n_head", type=int, default=14)
parser.add_argument("--n_embd", type=int, default=1792)
parser.add_argument("--lr_multiplier", type=float, default=0.25)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--dropout", type=float, default=0.05)   # 0.1 in entry 20
parser.add_argument("--dupe-start-epoch", type=int, default=7,
                    help="Epoch to enable layer duplication")
parser.add_argument("--dupe-layers-start", type=int, default=15,
                    help="First decoder layer to duplicate (inclusive)")
parser.add_argument("--dupe-layers-end", type=int, default=21,
                    help="Last decoder layer to duplicate (exclusive)")
parser.add_argument("--dupe-loops", type=int, default=2,
                    help="Number of extra replay passes through dupe layers")
parser.add_argument("--warmdown-ratio", type=float, default=None,
                    help="Override warmdown ratio (default 0.2)")
parser.add_argument("--logit-cap", type=float, default=10.0,
                    help="Logit soft-capping value (0=disabled)")
parser.add_argument("--logit-avg", type=int, default=5,
                    help="Number of late epoch checkpoints kept on disk; the final ensemble uses the last 5 (0=disabled)")
parser.add_argument("--logit-avg-dir", type=str, default="logit_avg_ckpts",
                    help="Directory to save/load epoch checkpoints for logit averaging")
parser.add_argument("--logit-avg-mode", type=str, default="both",
                    choices=["equal", "weighted", "both"],
                    help="Weight scheme: equal, linear recency weighted, or compare both")
parser.add_argument("--eval-logit-avg", action="store_true",
                    help="Skip training and only run logit-avg eval on saved checkpoints")
parser.add_argument("--swa-last-epochs", type=int, default=3,
                    help="SWA: cosine-cycle LR in last N epochs for checkpoint diversity (0=off)")
parser.add_argument("--stoch-depth", type=float, default=0.0,   # 0.05 in entry 20
                    help="Stochastic depth max drop rate (linear schedule, 0=off)")
parser.add_argument("--mtp-weight", type=float, default=0.3,
                    help="Multi-token prediction weight (0=off)")
parser.add_argument("--iha", action="store_true", default=True,
                    help="Enable Interleaved Head Attention (cross-head Q/K/V mixing)")
parser.add_argument("--no-iha", action="store_false", dest="iha",
                    help="Disable IHA cross-head mixing")
parser.add_argument("--iha-lr", type=float, default=0.02,
                    help="LR for IHA mixing matrices")
parser.add_argument("--no-doc-shuffle", action="store_true",
                    help="Disable per-epoch document reshuffling (still shuffles batch order)")
# --- every-step meta-gradient step (default for this entry; --mg-every 0 --dropout 0.1 --stoch-depth 0.05 = entry 20) ---
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mg-every", type=int, default=1, help="meta-gradient step every N steps (1 = every step, the record; 0 = off = entry 20)")
parser.add_argument("--mg-step-norm", type=float, default=0.5, help="L2 norm of the temporary step on the plastic matrices")
parser.add_argument("--mg-step-schedule", type=str, default="lr", choices=["const", "lr"],
                    help="lr: scale the step norm by the warmdown learning-rate multiplier")
parser.add_argument("--mg-plastic", type=str, default="matrix_all", choices=["matrix_all", "mlp_all"])
parser.add_argument("--schedule-steps", type=int, default=2075,
                    help="Optimizer steps for the LR/SWA schedule; epochs follow the schedule. The hardcoded sketch tables are laid out for 2075 steps")
args = parser.parse_args()

# Resolve output path
if args.output_json and not args.save_result:
    args.save_result = args.output_json

# =============================================================================
# Hardwired d12 (GPT-2 small) hyperparameters
# =============================================================================

# Architecture (defaults = d12 GPT-2 small)
DEPTH = args.n_layer if args.n_layer is not None else 12
N_EMBD = args.n_embd if args.n_embd is not None else 768
N_HEAD = args.n_head if args.n_head is not None else 6
HEAD_DIM = N_EMBD // N_HEAD
MAX_SEQ_LEN = 2048
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = args.total_batch_size
EVAL_TOKENS = 10_000_000
DATA_DIR = "fineweb_data"
BOS_ID = 50256  # <|endoftext|>
RUNS_DIR = "runs"

# Base optimizer hyperparameters
BASE_MATRIX_LR = args.matrix_lr
BASE_SCALAR_LR = args.scalar_lr
BASE_EMBEDDING_LR = 0.15
BASE_UNEMBEDDING_LR = 0.002

# Apply LR multiplier if provided (scales all LRs uniformly)
_lr_mult = args.lr_multiplier if args.lr_multiplier is not None else 1.0
MATRIX_LR = BASE_MATRIX_LR * _lr_mult
UNEMBEDDING_LR = BASE_UNEMBEDDING_LR * _lr_mult
EMBEDDING_LR = BASE_EMBEDDING_LR * _lr_mult
SCALAR_LR = BASE_SCALAR_LR * _lr_mult

WEIGHT_DECAY = args.weight_decay
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = args.warmdown_ratio if args.warmdown_ratio is not None else 0.2
FINAL_LR_FRAC = 0.0
WARMDOWN_POWER = 0.5      # sqrt-shaped warmdown (stays ~41% higher at midpoint than linear)
WD_PRE_HOLD_FRAC = 0.40   # hold at base WD for first 40% of training, then decay to LOW by SWA start
WD_SWA_LOW_FACTOR = 0.65  # WD at start of each SWA epoch (LR is high → less regularization)
WD_SWA_HIGH_FACTOR = 1.50 # WD at end of each SWA epoch (LR has decayed → more regularization)
LOGIT_CAP = args.logit_cap

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
        return run_name, os.path.join(RUNS_DIR, run_name)
    name = time.strftime('%Y%m%d_%H%M%S')
    return name, os.path.join(RUNS_DIR, name)

# =============================================================================
def load_state_dict_into_model(model, state_dict):
    """Load a state dict into model, handling dtype conversion."""
    for name, p in model.named_parameters():
        if name in state_dict:
            p.data.copy_(state_dict[name].to(p.device, dtype=p.dtype))

# =============================================================================
# Flash Attention (FA3 on Hopper)
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
        return get_kernel('kernels-community/flash-attn3', version=1)
    except ImportError:
        print0("Warning: kernels package not found. Install with: pip install -U kernels")
        return None
    except Exception as e:
        print0(f"Warning: Failed to load FA3 kernel: {e}")
        return None

_fa3 = _load_fa3()

def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Flash Attention for training (FA3 only). q,k,v: (B, T, H, D)."""
    return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)

# =============================================================================
# GPT Model
# =============================================================================

@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 50257
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_kv_head: int = N_HEAD
    n_embd: int = N_EMBD
    window_pattern: str = WINDOW_PATTERN
    dropout: float = 0.0
    stoch_depth: float = 0.05
    use_iha: bool = False
    iha_mix_v: bool = True

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
        # Attention gate: per-head gating to enable context-based no-op
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)
        # IHA: cross-head mixing matrices (Interleaved Head Attention)
        # Mixing is fused into projection weights at forward time.
        # Cost: [H,H]@[H,d*C] matmul is negligible vs the [B*T,C]@[C,H*d] projection.
        self.use_iha = config.use_iha
        if self.use_iha:
            self.q_mix = nn.Parameter(torch.zeros(self.n_head, self.n_head))
            self.k_mix = nn.Parameter(torch.zeros(self.n_kv_head, self.n_kv_head))
            self.iha_mix_v = config.iha_mix_v
            if self.iha_mix_v:
                self.v_mix = nn.Parameter(torch.zeros(self.n_kv_head, self.n_kv_head))

    def _fuse_mix(self, weight, mix, H):
        """Fuse mixing matrix into projection weight: W_fused[h] = sum_m mix[h,m]*W[m]."""
        d = self.head_dim
        return (mix @ weight.view(H, d, -1).flatten(1)).view_as(weight)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        if self.use_iha:
            # Fuse mixing into weights then project — grad flows through mix params
            q = F.linear(x, self._fuse_mix(self.c_q.weight, self.q_mix, self.n_head))
            q = q.view(B, T, self.n_head, self.head_dim)
            k = F.linear(x, self._fuse_mix(self.c_k.weight, self.k_mix, self.n_kv_head))
            k = k.view(B, T, self.n_kv_head, self.head_dim)
            if self.iha_mix_v:
                v = F.linear(x, self._fuse_mix(self.c_v.weight, self.v_mix, self.n_kv_head))
                v = v.view(B, T, self.n_kv_head, self.head_dim)
            else:
                v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        else:
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
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # Attention gate: per-head sigmoid gate
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
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.resid_dropout(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        # Stochastic depth: linear schedule from 0 at layer 0 to stoch_depth at last layer
        self.drop_prob = config.stoch_depth * (layer_idx / max(config.n_layer - 1, 1))

    def forward(self, x, ve, cos_sin, window_size):
        # Stochastic depth: blend with identity when dropped (compile-friendly, no graph break)
        if self.training and self.drop_prob > 0:
            keep = (torch.rand((), device=x.device) >= self.drop_prob).to(x.dtype)
            x_in = x
            x = x + self.attn(norm(x), ve, cos_sin, window_size)
            x = x + self.mlp(norm(x))
            x = x_in + keep * (x - x_in)
        else:
            x = x + self.attn(norm(x), ve, cos_sin, window_size)
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
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim, base=10000)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._dupe_layers = None  # (start, end) or None
        self.mtp_weight = args.mtp_weight
        if self.mtp_weight > 0:
            self.mtp_proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
            self.mtp_block = Block(config, config.n_layer)

    def set_dupe_layers(self, start, end, loops=2):
        assert start >= self.encoder_layers, "dupe layers must be decoder-only"
        assert end <= self.config.n_layer
        self._dupe_layers = (start, end)
        self._dupe_loops = loops
        print0(f"Dupe layers {start}-{end-1}: {loops} extra replays ({loops+1} total passes)")

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        normal_std = self.config.n_embd ** -0.5
        all_blocks = list(self.transformer.h)
        if self.mtp_weight > 0:
            all_blocks.append(self.mtp_block)
            torch.nn.init.uniform_(self.mtp_proj.weight, -s, s)
        for block in all_blocks:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=normal_std)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=normal_std)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)
            # IHA: initialize mixing matrices to identity (baseline-equivalent)
            if block.attn.use_iha:
                torch.nn.init.eye_(block.attn.q_mix)
                torch.nn.init.eye_(block.attn.k_mix)
                if block.attn.iha_mix_v:
                    torch.nn.init.eye_(block.attn.v_mix)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for proj in self.ve_projs.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        self.skip_weights.fill_(1.0)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim, base=10000)
        self.cos = cos
        self.sin = sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
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
        # Separate IHA mixing params (small H×H matrices) from large matrix params
        iha_params = []
        iha_param_ids = set()
        all_blocks = list(self.transformer.h)
        if self.mtp_weight > 0:
            all_blocks_for_iha = all_blocks + [self.mtp_block]
        else:
            all_blocks_for_iha = all_blocks
        for block in all_blocks_for_iha:
            if block.attn.use_iha:
                iha_params.append(block.attn.q_mix)
                iha_params.append(block.attn.k_mix)
                iha_param_ids.add(id(block.attn.q_mix))
                iha_param_ids.add(id(block.attn.k_mix))
                if block.attn.iha_mix_v:
                    iha_params.append(block.attn.v_mix)
                    iha_param_ids.add(id(block.attn.v_mix))
        all_h_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in all_h_params if id(p) not in iha_param_ids] + list(self.ve_projs.parameters())
        if self.mtp_weight > 0:
            mtp_params = [p for p in list(self.mtp_block.parameters()) + list(self.mtp_proj.parameters()) if id(p) not in iha_param_ids]
            matrix_params += mtp_params
        ve_params = []
        embed_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        skip_params = [self.skip_weights]

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=embed_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=ve_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=resid_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=SCALAR_LR, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=skip_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
        ]
        # IHA mixing matrices: use AdamW with dedicated or scalar-like LR
        if iha_params:
            iha_lr = args.iha_lr if args.iha_lr is not None else SCALAR_LR
            param_groups.append(dict(kind='adamw', params=iha_params, lr=iha_lr, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0))
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=MATRIX_LR,
                                     momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY))

        optimizer = DistMuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _run_decoder_layers(self, x, x0, encoder_outputs, start, end, T):
        """Run decoder layers [start, end), with U-Net skip connections."""
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for i in range(start, end):
            # Encoder layer j connects to decoder layer (n_layer - 1 - j)
            j = self.config.n_layer - 1 - i
            if 0 <= j < self.encoder_layers:
                x = x + self.skip_weights[i - self.encoder_layers] * encoder_outputs[j]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
        return x

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        x = norm(self.transformer.wte(idx))
        x0 = x
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Encoder half: run layers and collect outputs for skip connections
        encoder_outputs = []
        for i in range(self.encoder_layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
            encoder_outputs.append(x)

        # Decoder half
        dupe = self._dupe_layers
        if dupe is None:
            x = self._run_decoder_layers(x, x0, encoder_outputs,
                                        self.encoder_layers, self.config.n_layer, T)
        else:
            # First pass: encoder boundary through end of dupe range
            x = self._run_decoder_layers(x, x0, encoder_outputs,
                                        self.encoder_layers, dupe[1], T)
            # Extra replays through dupe range
            for _ in range(self._dupe_loops):
                x = self._run_decoder_layers(x, x0, encoder_outputs,
                                            dupe[0], dupe[1], T)
            # Remaining decoder layers
            x = self._run_decoder_layers(x, x0, encoder_outputs,
                                        dupe[1], self.config.n_layer, T)

        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = LOGIT_CAP * torch.tanh(logits / LOGIT_CAP) if LOGIT_CAP > 0 else logits
        if targets is None:
            return logits
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                  ignore_index=-1, reduction=loss_reduction)
        if loss_reduction != 'mean':
            return lm_loss
        if self.mtp_weight <= 0:
            return lm_loss, {'lm_loss': lm_loss}
        mtp_emb = norm(self.transformer.wte(targets[:, :-1].clamp(min=0)))
        combined = self.mtp_proj(torch.cat([x[:, :-1], mtp_emb], dim=-1))
        mT = combined.size(1)
        mtp_out = norm(self.mtp_block(combined, None, (self.cos[:, :mT], self.sin[:, :mT]), (-1, -1)))
        mtp_logits = self.lm_head(mtp_out)[..., :self.config.vocab_size].float()
        if LOGIT_CAP > 0:
            mtp_logits = LOGIT_CAP * torch.tanh(mtp_logits / LOGIT_CAP)
        mtp_loss = F.cross_entropy(mtp_logits.view(-1, mtp_logits.size(-1)),
                                   targets[:, 1:].reshape(-1), ignore_index=-1)
        loss = lm_loss + self.mtp_weight * mtp_loss
        return loss, {'lm_loss': lm_loss, 'mtp_loss': mtp_loss}

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
    # MuonEq-R row normalization
    g /= g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
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
# Dataloader: BOS-aligned best-fit packing
# =============================================================================

class DataLoader:
    """Loads flat tokens , chunks into batches.

    doc_shuffle=False: applies the stored default sequence permutation (bitwise match
    with the old chunked pipeline), shuffles batch order each epoch.
    doc_shuffle=True: reshuffles documents each epoch, re-chunks, re-shuffles sequences.

    Always yields (x, y, epoch).
    """

    def __init__(self, filepath, B, T, device="cuda", doc_shuffle=False):
        data = torch.load(filepath, weights_only=True)
        all_tokens = data["tokens"].long()
        raw_doc_starts = data["doc_starts"].long()
        bos_id = int(data["bos_id"])
        assert bos_id == BOS_ID, f"data bos_id {bos_id} != expected {BOS_ID}"

        doc_ends = torch.cat([raw_doc_starts[1:], torch.tensor([all_tokens.numel()])])
        self.doc_tokens = [all_tokens[s:e] for s, e in zip(raw_doc_starts.tolist(), doc_ends.tolist())]
        self.default_shuffle_seed = data["seq_shuffle_seed"]

        _, rank, _, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.B = B
        self.T = T
        self.seq_size = T + 1
        self.doc_shuffle = doc_shuffle
        self.epoch = 1
        self._build_batches()

    def _build_batches(self):
        tokens = torch.cat(self.doc_tokens)
        num_seqs = len(tokens) // self.seq_size
        all_seqs = tokens[:num_seqs * self.seq_size].view(num_seqs, self.seq_size)
        if self.doc_shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + 1000)
            all_seqs = all_seqs[torch.randperm(num_seqs, generator=g)]
        else:
            perm = np.random.RandomState(self.default_shuffle_seed).permutation(num_seqs)
            all_seqs = all_seqs[torch.from_numpy(perm)]
        seqs_per_step = self.B * self.world_size
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step
        all_seqs = all_seqs[:usable].view(num_steps, self.world_size, self.B, self.seq_size)
        self.rank_data = all_seqs[:, self.rank].contiguous()
        self.num_steps = num_steps
        self.total_tokens = usable * self.T
        self.pos = 0

    def __iter__(self):
        return self

    def _next_epoch(self):
        self.epoch += 1
        print0(f"Starting epoch {self.epoch}")
        if self.doc_shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            perm = torch.randperm(len(self.doc_tokens), generator=g)
            self.doc_tokens = [self.doc_tokens[i] for i in perm.tolist()]
            self._build_batches()
        else:
            self.pos = 0
            g = torch.Generator()
            g.manual_seed(self.epoch)
            self.rank_data = self.rank_data[torch.randperm(self.num_steps, generator=g)]

    def __next__(self):
        if self.pos >= self.num_steps:
            self._next_epoch()
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch

# =============================================================================
# Loss evaluation
# =============================================================================

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """Compute bits per byte and mean cross-entropy loss on a set of batches."""
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)
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
def evaluate_bpb_logit_avg(eval_model, ckpt_paths, weights, steps):
    """Evaluate using probability averaging across checkpoints (proper ensemble).

    Loads each checkpoint from disk once, runs all val batches for it, then
    moves to the next — one CPU->GPU weight transfer per checkpoint, not per batch.
    Accumulates running scalar totals instead of per-token tensors.
    """
    dev = orig_model.get_device()
    V   = orig_model.config.vocab_size

    # Pre-fetch all val batches to CPU (token ids, tiny ~10 MB)
    val_loader = build_val_loader()
    all_x, all_y = [], []
    for _ in range(steps):
        x, y, _ = next(val_loader)
        all_x.append(x.cpu())
        all_y.append(y.cpu())

    BT = all_y[0].numel()

    # Per-batch accumulated weighted target probs, kept on GPU
    # Shape: (steps, BT) — only target-token probs, not full vocab
    batch_target_probs = torch.zeros(steps, BT, dtype=torch.float32, device=dev)

    # Checkpoint-outer, batch-inner: each checkpoint loaded exactly once.
    # A member is a checkpoint path or a callable that writes its weights into orig_model.
    for path, w in zip(ckpt_paths, weights):
        if callable(path):
            path()
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            load_state_dict_into_model(orig_model, ckpt)
            del ckpt
        for i, (x, y) in enumerate(zip(all_x, all_y)):
            y_flat = y.view(-1).to(dev)
            with autocast_ctx:
                logits = eval_model(x.to(dev))
            probs = torch.softmax(logits.view(BT, V).float(), dim=-1)
            tgt   = probs[torch.arange(BT, device=dev), y_flat.clamp_min(0)]
            batch_target_probs[i].add_(tgt, alpha=w)

    # Compute metrics from accumulated target probs using running totals
    total_nats   = torch.tensor(0.0, dtype=torch.float64, device=dev)
    total_bytes  = torch.tensor(0, dtype=torch.int64, device=dev)
    total_loss   = torch.tensor(0.0, dtype=torch.float64, device=dev)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=dev)

    for i, y in enumerate(all_y):
        y_flat = y.view(-1).to(dev)
        mask = y_flat != -1
        log_probs = batch_target_probs[i].clamp_min(1e-40).log()
        num_bytes_batch = token_bytes[y_flat.clamp_min(0)]

        total_nats   += (log_probs.neg() * (num_bytes_batch > 0)).sum().double()
        total_bytes  += num_bytes_batch.sum()
        total_loss   += log_probs[mask].neg().sum().double()
        total_tokens += mask.sum()

    del batch_target_probs

    if dist.is_initialized():
        dist.all_reduce(total_nats,   op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes,  op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss,   op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    bpb  = total_nats.item()  / (math.log(2) * total_bytes.item())  if total_bytes.item()  > 0 else float('inf')
    loss = total_loss.item()  / total_tokens.item()                  if total_tokens.item() > 0 else float('inf')
    return bpb, loss

# =============================================================================
# Training
# =============================================================================

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
master_process = ddp_rank == 0
torch.manual_seed(args.seed)

if ddp and torch.cuda.is_available():
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(args.seed)
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
    raise RuntimeError("Flash Attention 3 is required but not available. A Hopper (sm90) GPU is needed.")

# Run / logging paths
run_name, run_dir = resolve_run_dir(args.run_name)
if dist.is_initialized():
    shared = [run_name]
    dist.broadcast_object_list(shared, src=0)
    run_name = shared[0]
    run_dir = os.path.join(RUNS_DIR, run_name)
checkpoints_dir = os.path.join(run_dir, "checkpoints")
terminal_log_path = os.path.join(run_dir, "terminal.log")
stdout_orig = sys.stdout
stderr_orig = sys.stderr
artifacts_log_f = None
result_path = os.path.join(run_dir, "result.json")
if master_process:
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "wandb"), exist_ok=True)
    shutil.copy2(__file__, os.path.join(run_dir, "train.py"))
    artifacts_log_f = open(terminal_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, artifacts_log_f)
    sys.stderr = TeeStream(sys.stderr, artifacts_log_f)

# wandb
_wandb_kwargs = {"project": "slowrun", "name": run_name, "dir": os.path.join(run_dir, "wandb")}
if args.wandb_group:
    _wandb_kwargs["group"] = args.wandb_group
wandb_run = DummyWandb() if not master_process else wandb.init(**_wandb_kwargs)
if master_process:
    wandb_run.log_code(".")

# Print hyperparameters
print0(f"--- Hyperparameters ---")
print0(f"  n_layer={DEPTH}, n_embd={N_EMBD}, n_head={N_HEAD}, head_dim={HEAD_DIM}")
print0(f"  seq_len={MAX_SEQ_LEN}, window_pattern={WINDOW_PATTERN}")
print0(f"  stoch_depth={args.stoch_depth}")
print0(f"  total_batch_size={TOTAL_BATCH_SIZE}, device_batch_size={args.device_batch_size}")
print0(f"  matrix_lr={MATRIX_LR}, scalar_lr={SCALAR_LR}, embedding_lr={EMBEDDING_LR}, unembedding_lr={UNEMBEDDING_LR}")
print0(f"  weight_decay={WEIGHT_DECAY}, adam_betas={ADAM_BETAS}")
print0(f"  warmup_ratio={WARMUP_RATIO}, warmdown_ratio={WARMDOWN_RATIO}, final_lr_frac={FINAL_LR_FRAC}")
print0(f"  num_epochs={args.num_epochs}, patience={args.patience}")
print0(f"  dropout={args.dropout}, doc_shuffle={not args.no_doc_shuffle}")
print0(f"  run={run_name}")
print0(f"  run_dir={run_dir}")
if args.iha:
    print0(f"  iha=True, iha_lr={args.iha_lr}")
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
config = GPTConfig(vocab_size=vocab_size, dropout=args.dropout,
                   stoch_depth=args.stoch_depth,
                   use_iha=args.iha, iha_mix_v=args.iha)
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
model = torch.compile(model, dynamic=False)

# Optimizer
optimizer = model.setup_optimizer()

# ---- every-step meta-gradient step ----
MG = args.mg_every > 0
mg_params, mg_owned = [], []   # plastic matrices; indices of the ones this rank owns in the Muon step
if MG:
    owned_ids = set()
    for g in optimizer.param_groups:
        if g["kind"] == "muon":   # Muon updates params[rank*chunk:(rank+1)*chunk] on this rank and all-gathers the rest
            chunk = (len(g["params"]) + ddp_world_size - 1) // ddp_world_size
            owned_ids.update(id(p) for p in g["params"][ddp_rank * chunk:(ddp_rank + 1) * chunk])
    muon_ids = {id(p) for g in optimizer.param_groups if g["kind"] == "muon" for p in g["params"]}
    for name, p in orig_model.named_parameters():
        if id(p) in muon_ids and (args.mg_plastic == "matrix_all" or ".mlp." in name):
            mg_params.append(p)
            if id(p) in owned_ids:
                mg_owned.append(len(mg_params) - 1)
    print0(f"[mg] every {args.mg_every} steps, step norm {args.mg_step_norm} ({args.mg_step_schedule}), "
           f"{len(mg_params)} plastic matrices, {sum(p.numel() for p in mg_params):,} parameters; "
           f"rank {ddp_rank} restores {len(mg_owned)} of them")
mg_owned_params = [mg_params[i] for i in mg_owned]

@torch.no_grad()
def mg_temporary_step(step):
    """theta_P <- theta_P - eta * g_P / ||g_P|| with g the gradient accumulated so far on this rank.
    Returns (scaled gradient of the owned matrices, alpha) so the caller can undo the step: the Muon
    step overwrites every matrix this rank does not own with the owner's all-gathered copy, so only
    the owned matrices need restoring."""
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in mg_params]
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))).clamp_min(1e-12)
    eta = args.mg_step_norm * (get_lr_multiplier(step) if args.mg_step_schedule == "lr" else 1.0)
    alpha = -eta / float(norm)
    kept = [grads[i].clone() for i in mg_owned]
    torch._foreach_add_(mg_params, grads, alpha=alpha)
    return kept, alpha

# Dataloaders
_train_path = args.input_bin if args.input_bin else os.path.join(DATA_DIR, "fineweb_train.pt")
_val_path = args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "fineweb_val.pt")
train_loader = DataLoader(_train_path, args.device_batch_size, MAX_SEQ_LEN, device=device, doc_shuffle=not args.no_doc_shuffle)
build_val_loader = lambda: DataLoader(_val_path, args.device_batch_size, MAX_SEQ_LEN, device=device)
TOKENS_PER_EPOCH = train_loader.total_tokens
x, y, current_epoch = next(train_loader)

# Training config
tokens_per_fwdbwd = args.device_batch_size * MAX_SEQ_LEN * ddp_world_size
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
num_iterations = round(TOKENS_PER_EPOCH * args.num_epochs / TOTAL_BATCH_SIZE)  # estimate for LR schedule
if args.schedule_steps > 0:
    # A slightly shorter schedule than 11 full data passes keeps training clear of the time cap.
    print0(f"Schedule: {args.schedule_steps} steps (11 full data passes would be {num_iterations})")
    num_iterations = int(args.schedule_steps)
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
        shaped = progress ** WARMDOWN_POWER  # concave (stays higher longer) when POWER < 1
        return shaped + (1 - shaped) * FINAL_LR_FRAC

def get_muon_momentum(it):
    return (1 - min(it / 300, 1)) * 0.85 + min(it / 300, 1) * 0.95

steps_per_epoch = num_iterations / args.num_epochs
_swa_start_step = (num_iterations - args.swa_last_epochs * steps_per_epoch) if args.swa_last_epochs > 0 else -1

def get_wd_multiplier(it):
    """Anti-phase WD: hold at 1.0 pre-SWA, decay to LOW by SWA start, then sawtooth LOW→HIGH per SWA epoch (anti-phase with the LR cosine cycle)."""
    if _swa_start_step >= 0 and it >= _swa_start_step:
        cycle_pos = (it - _swa_start_step) % steps_per_epoch
        frac = cycle_pos / steps_per_epoch
        return WD_SWA_LOW_FACTOR + (WD_SWA_HIGH_FACTOR - WD_SWA_LOW_FACTOR) * frac
    t = it / num_iterations
    if t < WD_PRE_HOLD_FRAC:
        return 1.0
    swa_start_frac = _swa_start_step / num_iterations if _swa_start_step > 0 else 1.0
    decay_frac = (t - WD_PRE_HOLD_FRAC) / max(swa_start_frac - WD_PRE_HOLD_FRAC, 1e-6)
    decay_frac = min(max(decay_frac, 0.0), 1.0)
    return 1.0 - (1.0 - WD_SWA_LOW_FACTOR) * decay_frac

# =============================================================================
# Snapshot ensemble from sketched weight averages
# =============================================================================
# The final model is a probability ensemble of the last 5 epoch checkpoints and 7 weight
# averages over earlier stretches of the trajectory (windows ending at epochs 7 and 8, and the
# whole last SWA cycle). The averages are never stored as snapshots: training is split into
# blocks, and inside block b only two running sums S_j = sum_t V_b[t, j] * theta_t are kept,
# where theta_t are the weights after step t. At each block end, average p gets
# A_p += sum_j C[p, b, j] * S_j. The block layout, V and C are fixed functions of the LR schedule
# (no data involved). The 12 ensemble weights were fit once on a held-out 2% slice of the
# training set in earlier runs and are frozen here; the validation set is only read for reporting.

_SKETCH_BLOCK_ENDS = [189, 378, 566, 755, 944, 1132, 1321, 1510, 1557, 1604, 1651, 1698, 1745, 1792, 1839, 1887, 1934, 1981, 2028, 2075]
_SKETCH_BASES_SHA256 = "4927bfa8b7cf1f542e01028d2e4c3cd95feb52d17d567a7ffe1b3b18fcb201d8"
_SKETCH_BASES_B64 = (
    "cdioO6A/9Dpt5Kg7ObjxOm3wqDuKL+86cPyoO/Cl7Dp3CKk7lRvqOoAUqTuSkOc6iyCpO/gE5TqZLKk71HjiOqk4qTsw7N86"
    "u0SpOxVf3TrPUKk7itHaOuRcqTuTQ9g6/GipOze11ToVdak7eSbTOjCBqTtdl9A6TI2pO+YHzjprmak7GHjLOoqlqTv158g6"
    "rLGpO39XxjrPvak7usbDOvPJqTumNcE6GdapO0akvjpA4qk7nBK8OmnuqTupgLk6k/qpO27utjq/Bqo77lu0OusSqjsqybE6"
    "Gh+qOyI2rzpKK6o72KKsOns3qjtND6o6rUOqO4J7pzrhT6o7eeekOhZcqjsxU6I6TGiqO62+nzqEdKo77CmdOryAqjvwlJo6"
    "94yqO7n/lzoymao7SWqVOm+lqjuf1JI6rbGqO70+kDrsvao7pKiNOizKqjtTEos6btaqO8t7iDqx4qo7DuWFOvXuqjsbToM6"
    "OvuqO/O2gDqAB6s7Lz98OsgTqzsPEHc6ESCrO4ngcTpbLKs7nbBsOqY4qztMgGc68kSrO5hPYjpAUas7gB5dOo9dqzsG7Vc6"
    "32mrOyu7Ujowdqs774hNOoKCqztTVkg61Y6rO1gjQzopm6s7/u89On+nqztHvDg61rOrOzKIMzouwKs7wVMuOofMqzv1Hik6"
    "4dirO83pIzo85as7S7QeOpjxqztvfhk69v2rOzlIFDpUCqw7qxEPOrQWrDvE2gk6FSOsO4WjBDp3L6w74Nf+Odo7rDsHaPQ5"
    "PkisO4L36TmjVKw7UobfOQlhrDt3FNU5cW2sO/KhyjnZeaw7xS7AOUOGrDvxurU5rZKsO3VGqzkZn6w7VNGgOYarrDuNW5Y5"
    "9LesOyPlizljxKw7FG6BOdPQrDvH7G05RN2sOyH8WDm26aw7OQpEOSn2rDsQFy85nQKtO6ciGjkTD607AS0FOYkbrTs8bOA4"
    "ASitOwB8tjh5NK07UImMOPNArTtgKEU4bk2tO4Vy4jfpWa07bSrqNmZmrTvnzVq35HKtOw6xCrhjf607bLNeuOOLrTtGXZm4"
    "ZJitOzZjw7jmpK07hGvtuGmxrTsWuwu57b2tO5bBILlyyq07Qsk1ufjWrTsY0kq5gOOtOxfcX7kI8K07Pud0uZH8rTvG+YS5"
    "HAmuO4CAj7mnFa47zAeauTQirjurj6S5wS6uOxsYr7lQO647HKG5ud9HrjutKsS5cFSuO8+0zrkBYa47gD/ZuZRtrjvByuO5"
    "KHquO5BW7rm9hq477uL4uVKTrjvttwG66Z+uO6n+BrqBrK47rEUMuhq5rjv1jBG6tMWuO4XUFrpP0q47Whwcuurerjt1ZCG6"
    "h+uuO9asJrol+K47e/UrusQErztmPjG6ZBGvO5aHNroFHq87C9E7uqgqrzvEGkG6SzevO8JkRrrvQ687BK9LupRQrzuK+VC6"
    "Ol2vO1REVrrhaa87YY9buol2rzuy2mC6M4OvO0YmZrrdj687HnJruoicrzs5vnC6NKmvO5YKdrrita87Nld7upDCrzsNUoC6"
    "P8+vO5/4grrv2687U5+FuqHorzsoRoi6U/WvOx3tiroHArA7NJSNursOsDtrO5C6cBuwO8TikronKLA7PYqVut40sDvWMZi6"
    "l0GwO5DZmrpQTrA7a4GdugtbsDtmKaC6xmewO4HRorqCdLA7vXmlukCBsDsYIqi6/o2wO5TKqrq+mrA7MHOtun+nsDvsG7C6"
    "QLSwO8jEsroDwbA7xG21usbNsDvgFri6i9qwOxvAurpQ57A7d2m9uhf0sDvxEsC63wCxO4y8wrqnDbE7RmbFunEasTsfEMi6"
    "PCexOxi6yroHNLE7MGTNutRAsTtoDtC6ok2xO7+40rpwWrE7NWPVukBnsTvKDdi6EXSxO3642rrigLE7UWPdurWNsTtDDuC6"
    "iZqxO1W54rpep7E7hWTlujO0sTvUD+i6CsGxO0G76rrizbE7zmbturvasTt5EvC6lOexO0O+8rpv9LE7K2r1uksBsjsyFvi6"
    "gmSjOxEzoLs6f6M7ih+bu/OZozuxGZa7rrSjO4Yhkbtrz6M7CTeMuyvqozs7Woe77ASkOxyLgruvH6Q7WZN7u3Q6pDvaK3K7"
    "O1WkO7vfaLsEcKQ7/q5fu8+KpDukmVa7nKWkO66fTbtrwKQ7HcFEuzzbpDvy/Tu7D/akOy5WM7vjEKU708kqu7orpTvhWCK7"
    "k0alO1oDGrttYaU7QMkRu0p8pTuUqgm7KZelO1anAbsJsqU7E3/zuuzMpTtd5uO60OelO4+E1Lq2AqY7q1nFup8dpjuzZba6"
    "iTimO62op7p1U6Y7miKZumNupjt+04q6U4mmO7p2ebpFpKY7dLRdujm/pjsxYEK6L9qmO/h5J7on9aY70AENuiEQpzuE7+W5"
    "HCunO6i3srkaRqc7HVyAuRphpzvluR25G3ynO73Rcbgfl6c7GGAMOCSypztb6wA5K82nO20FXTk16Kc7DrObOUADqDujBsg5"
    "TR6oO2V98zlcOag7oAsPOm1UqDsT6iM6gG+oOwFaODqViqg7YltMOqylqDss7l86xMCoO1YSczrf26g77OOCOvv2qDtTB4w6"
    "GhKpO13zlDo6Lak7A6idOl1IqTtCJaY6gWOpOxRrrjqnfqk7dXm2Os+ZqTteUL46+bSpO83vxTol0Kk7ulfNOlPrqTsiiNQ6"
    "gwaqO/6A2zq1Iao7SkLiOug8qjsAzOg6HliqOxse7zpWc6o7ljj1Oo+OqjtrG/s6yqmqO0pjADsIxao7Bx0DO0fgqjvougU7"
    "iPuqO+s8CDvLFqs7DqMKOxAyqztM7Qw7V02rO6UbDzufaKs7FC4RO+qDqzuWJBM7N5+rOyr/FDuFuqs7y70WO9bVqzt3YBg7"
    "KPGrOyvnGTt8DKw75FEbO9MnrDufoBw7K0OsO1jTHTuFXqw7DuoeO+F5rDu85B87P5WsO2DDIDuesKw79oUhOwDMrDt8LCI7"
    "ZOesO+22IjvJAq07SCUjOzEerTuJdyM7mjmtO6ytIzsFVa07r8cjO3JwrTuOxSM74YutO0anIztSp6071GwjO8XCrTs0FiM7"
    "Ot6tO2OjIjux+a07XRQiOyoVrjshaSE7pDCuO6mhIDshTK47870fO59nrjv7vR47H4OuO76hHTuinq47OGkcOya6rjtmFBs7"
    "rNWuO0WjGTs08a470RUYO74MrzsHbBY7SSivO+KlFDvXQ687YMMSO2dfrzt9xBA7+HqvOzapDjuMlq87h3EMOyGyrztsHQo7"
    "uM2vO+GsBztR6a875B8FO+0EsDtwdgI7iiCwOwRh/zopPLA7LZz5OslXsDtTnvM6bHOwO25n7ToRj7A7ePfmOreqsDtoTuA6"
    "YMawOzls2ToK4rA74VDSOrf9sDtZ/Mo6ZRmxO5tuwzoVNbE7nqe7OsdQsTtbp7M6e2yxO8ptqzoxiLE74/qiOumjsTugTpo6"
    "or+xO/dokTpe27E740mIOhv3sTuz4n062xKyO6m+ajqcLrI7lydXOmBKsjtuHUM6JWayOx6gLjrsgbI7l68ZOrWdsjvKSwQ6"
    "gLmyO0zp3DlN1bI7OFSwORzxsjs42II57AyzO1bqKDm/KLM7x6uUOJNEsztCabC3amCzOxx+8LhCfLM7DkBcuRyYszs1CKG5"
    "+bOzOzHY1LnXz7M7DsgEurfrszsMmB+6mQe0OyLcOrp9I7Q7YZRWumI/tDvZwHK6Slu0O82wh7o0d7Q7WzuWuh+TtDsfAKW6"
    "Da+0OyH/s7r8yrQ7aTjDuu7mtDsArNK64QK1O+1Z4rrWHrU7O0Lyus06tTt4MgG7xla1OwthCbvBcrU726wRu76OtTvrFRq7"
    "vaq1O0CcIru9xrU73z8ru8DitTvLADS7xP61OwnfPLvLGrY7ntpFu9M2tjuN80673lK2O9opWLvqbrY7jH1hu/iKtjul7mq7"
    "CKe2Oyl9dLsaw7Y7Hyl+uy7ftjtE+YO7RPu2O7bsiLtcF7c75u6Nu3UztzvY/5K7kU+3O4wfmLuva7c7Bk6du86HtztIi6K7"
    "jWShOyENPT0xh6E7XMAOPdipoTuMGs08gsyhO4Tahzwv76E7J4UZPN8Rojs7JFc7kTSiO8Tr6bpHV6I7uWbCu/95ojsOoBe8"
    "upyiO+jtQbx4v6I7nHVhvDjiojvDdHe8/ASjO+mHgrzCJ6M7h6yFvItKozt3p4W8V22jO+zrgrwmkKM7Vst7vPeyozt07228"
    "zNWjOwz3XLyj+KM7zIZJvH0bpDu7NTS8Wj6kO3WIHbw5YaQ7/PgFvByEpDv569u7AaekOy+8q7vqyaQ7yCJ4u9XspDv3/Rq7"
    "wg+lOy8zgrqzMqU7GZSlOadVpTshUcs6nXilO9ciMTuWm6U796h2O5K+pTuE7Jo7keGlO1QttzuTBKY76gLQO5gnpjuHW+U7"
    "n0qmO5E29zupbaY7sccCPLeQpju9PQg8x7OmO0oEDDzZ1qY78ScOPO/5pjtsuQ48CB2nO7fKDTwjQKc7FXALPEFjpztdwAc8"
    "YoanOwLSAjyGqac7uoL5O63MpzufSes71++nO1c02zsDE6g7XXnJOzI2qDsUS7Y7ZVmoOzTnoTuafKg774KMO9GfqDsqr2w7"
    "DMOoO8g3PztK5qg76v0QO4oJqTsT5cQ6ziypOzH6TzoUUKk7dIu+OF1zqTvMHx26qZapO/S9prr4uak7Bvr7ukndqTvT4Ca7"
    "ngCqO3i5Tbv1I6o7DENyu09HqjteKIq7rGqqOy/OmbsMjqo7ggaou2+xqjvTr7S71dSqO1zCv7s++Ko7ZC7Ju6kbqzvK5tC7"
    "Fz+rO6jp1ruJYqs7Sivbu/2Fqzt+s927dKmrO9V23rvuzKs70oXdu2rwqzt349q76hOsO1+X1rtsN6w7S7TQu/JarDtBQsm7"
    "en6sO25WwLsFoqw7tAa2u5PFrDuJYqq7JOmsO6yGnbu4DK07GIyPu04wrTu5joC76FOtO/9PYbuEd607KOw/uySbrTuQNx27"
    "xr6tO13K8rpr4q07oIqpuhMGrjsnNj66vimuOwh5ILlsTa47JkzcOR1xrju3zYE60JSuO2CEyzqHuK47aewJO0DcrjvfES07"
    "/f+uO/v6Tju8I687/2ZvO35HrzsCEYc7Q2uvO91ylTsLj687AMOiO9ayrzsV5a47o9avOz/DuTt0+q870UbDO0gesDtZWcs7"
    "HkKwOybs0Tv3ZbA7svDWO9SJsDvQVNo7s62wO0sQ3DuV0bA7PCDcO3r1sDsjeto7YhmxOyMb1ztNPbE7OgbSOzthsTt7Pcs7"
    "LIWxO+TJwjsfqbE7/LC4OxbNsTtfAq07D/GxO6DKnzsMFbI7ZiSROws5sju/GYE7DV2yO16fXzsTgbI7O6g6OxulsjtzohM7"
    "JsmyO/F+1To07bI7TruAOkURszuEtKU5WTWzO8H9vrlvWbM7ZyCKuol9szux3OS6pqGzO7a2H7vGxbM73HpMu+jpszscR3i7"
    "Dg60O69nkbs2MrQ7oM6lu2JWtDtCILm7kHq0O4Qvy7vBnrQ7273bu/bCtDs7nOq7Lee0O6uW97tnC7U7ET4BvKQvtTsYiwW8"
    "5FO1O86eCLwneLU7vl4KvG2ctTsrvQq8tsC1O9ueCbwC5bU7RAAHvFEJtjtVxwK8oy22Oznj+bv3UbY7XOrqu092tjuonti7"
    "qpq2O+n7wrsIv7Y7fg+qu2jjtjtQ7I27zAe3O7BPXbsyLLc7BPQYu5xQtztQHJ66CXW3OxkCc7Z4mbc7ExKmOuu9tztZ2Ck7"
    "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"
    "JtGuOwt2jrk966479RmZuVgFrzulwKO5dh+vOx1qrrmYOa87XRa5ub5TrztnxcO5522vOzt3zrkUiK872ivZuUWirztE4+O5"
    "ebyvO3yd7rmx1q87gVr5ue3wrzsqDQK6LAuwO3tuB7puJbA7NdEMurU/sDtXNRK6/1mwO+KaF7pNdLA71gEdup6OsDs0aiK6"
    "86iwO/zTJ7pMw7A7Lz8tuqjdsDvNqzK6CPiwO9cZOLpsErE7TIk9utMssTsu+kK6PkexO31sSLqtYbE7OeBNuh98sTtjVVO6"
    "lpaxO/vLWLoPsbE7AUReuo3LsTt3vWO6DuaxO1w4abqTALI7sLRuuhsbsjt2MnS6qDWyO6yxebo4ULI7UzJ/usxqsjs2WoK6"
    "Y4WyO/sbhbr+n7I7et6Hup26sjuyoYq6QNWyO6Rljbrm77I7UCqQupAKszu375K6PiWzO9e1lbrvP7M7snyYuqVasztIRJu6"
    "XnWzO5kMnrobkLM7ptWgutuqsztun6O6n8WzO/Fpprpo4LM7MTWpujP7szstAay6Axa0O+XNrrrWMLQ7Wpuxuq1LtDuLabS6"
    "iGa0O3o4t7pngbQ7Jgi6ukqctDuQ2Ly6MLe0O7ipv7oa0rQ7nXvCugjttDtBTsW6+Qe1O6MhyLrvIrU7xPXKuug9tTukys26"
    "5Vi1O0Sg0Lrmc7U7onbTuuuOtTvATda686m1O54l2boAxbU7PP7buhDgtTub1966JPu1O7qx4bo8FrY7mozkulcxtjs7aOe6"
    "d0y2O51E6rqaZ7Y7wCHtusGCtjum/++67J22O03e8robubY7t731uk7Utjvjnfi6hO+2O9F++7q/Crc7g2D+uib8Zjt6vYA7"
    "YCNnO/04gDuhSmc7mmh/O+lxZzvTXn47OJlnO6ZUfTuNwGc7EUp8Oyr5qDvHo7463AypOwGNvDqRIKk7bHW6Okk0qTsHXbg6"
    "BUipO9NDtjrEW6k7zym0OoZvqTv6DrI6TIOpO1bzrzoWl6k74NatOuKqqTuZuas6s76pO4GbqTqG0qk7lnynOl3mqTvaXKU6"
    "OPqpO0s8ozoWDqo76hqhOvchqju1+J463DWqO63VnDrESao70bGaOrBdqjshjZg6n3GqO51nljqShao7REGUOoiZqjsWGpI6"
    "ga2qOxLyjzp+wao7OcmNOn/VqjuKn4s6g+mqOwR1iTqK/ao7qEmHOpURqzt1HYU6oyWrO2rwgjq1Oas7iMKAOstNqzucJ306"
    "5GGrO3fIeDoAdqs7oWd0OiCKqzsYBXA6Q56rO92gazpqsqs77zpnOpXGqztN02I6w9qrO/ZpXjr07qs76v5ZOikDrDsnklU6"
    "YhesO68jUTqeK6w7frNMOt0/rDuWQUg6IVSsO/XNQzpnaKw7m1g/OrJ8rDuH4To6/5CsO7hoNjpRpaw7Le4xOqa5rDvncS06"
    "/s2sO+TzKDpa4qw7I3QkOrr2rDuk8h86HQutO2ZvGzqEH607aeoWOu4zrTusYxI6XEitOy7bDTrOXK077lAJOkNxrTvsxAQ6"
    "vIWtOyc3ADo4mq07PE/3ObiurTuhLO45O8OtO30G5TnD1607zdzbOU3srTuQr9I53ACuO8V+yTluFa47akrAOQQqrjt+Erc5"
    "nT6uO//WrTk6U6477JekOdpnrjtEVZs5f3yuOwUPkjkmka47LsWIOdKlrjt77345gbquO2NNbDk0z647EqRZOerjrjuF80Y5"
    "pfiuO7k7NDliDa87q3whOSQirztZtg456TavO4DR9ziyS687uCfSOH9grztWb6w4T3WvO1Sohjgjiq87VqVBOPuerztZues3"
    "1rOvO37aJze1yK87czMIt5jdrzuoW9y3fvKvO1JsOrhpB7A7MmSDuFYcsDsJoam4SDGwO7bsz7g+RrA7Pkf2uDdbsDtTWA65"
    "NHCwO3uUIbk0hbA7Gdg0uTmasDsxI0i5Qa+wO8V1W7lNxLA72c9uuVzZsDu3GIG5cO6wO0XNirmHA7E7l4WUuaIYsTuuQZ65"
    "wC2xO40BqLnjQrE7NcWxuQlYsTumjLu5M22xO+RXxblhgrE77ybPuZOXsTvI+di5yKyxO3LQ4rkBwrE77qrsuT/XsTs+ifa5"
    "f+yxO7E1ALrEAbI7rigFug0XsjuYHQq6WSyyO24UD7qpQbI7Mg0Uuv1WsjvkBxm6VWyyO4UEHrqxgbI7FgMjuhCXsjuYAyi6"
    "dKyyOwsGLbrbwbI7cAoyukbXsjvJEDe6teyyOxUZPLooArM7ViNBup8XszuML0a6GS2zO7k9S7qYQrM73E1QuhpYszv4X1W6"
    "oG2zOwt0Wroqg7M7GIpfuriYszsgomS6Sq6zOyK8abrgw7M7INhuunrZszsb9nO6GO+zOxIWebq5BLQ7CTh+ul8atDv/rYG6"
    "CDC0O/lAhLq1RbQ79NSGumdbtDvvaYm6HHG0O+z/i7rVhrQ76paOupKctDvqLpG6U7K0O+zHk7oZyLQ78WGWuuLdtDv6/Ji6"
    "r/O0OwWZm7p/CbU7FTaeulQftTsp1KC6LTW1O0Jzo7oKS7U7YROmuutgtTuEtKi60Ha1O65Wq7q5jLU73/mtuqaitTsWnrC6"
    "l7i1O1VDs7qMzrU7m+m1uoXktTvqkLi6gfq1O0E5u7qCELY7oeK9uocmtjsKjcC6kDy2O304w7qeUrY7++TFuq9otjuDksi6"
    "xH62OxdBy7rdlLY7tvDNuvqqtjthodC6HMG2OxhT07pB17Y73AXWumrttjutudi6mAO3O41u27rKGbc7eiTeuv8vtzt22+C6"
    "OUa3O4CT47p3XLc7m0zmurlytzvFBum6/4i3O//B67pJn7c7Sn7uupi1tzunO/G66su3OxX687q5ppM7GZaSOqe8kzvpn5I6"
    "m9KTO7ipkjqT6JM7hrOSOpD+kztUvZI6khSUOyHHkjqZKpQ77tCSOqRAlDu62pI6tVaUO4bkkjrKbJQ7Ue6SOuSClDsc+JI6"
    "A5mUO+YBkzonr5Q7rwuTOlDFlDt4FZM6ftuUO0Efkzqw8ZQ7CCmTOugHlTvQMpM6JB6VO5Y8kzpmNJU7XEaTOqxKlTsiUJM6"
    "92CVO+ZZkzpHd5U7q2OTOp2NlTtubZM696OVOzF3kzpWupU79ICTOrrQlTu1ipM6I+eVO3eUkzqR/ZU7N56TOgQUljv3p5M6"
    "eyqWO7axkzr4QJY7dbuTOnpXljszxZM6AW6WO/DOkzqNhJY7rdiTOh6bljtp4pM6tLGWOyXskzpQyJY73/WTOvDeljuZ/5M6"
    "lfWWO1MJlDo/DJc7DBOUOu8ilzvEHJQ6ozmXO3smlDpcUJc7MjCUOhtnlzvoOZQ6332XO51DlDqolJc7Uk2UOnWrlzsGV5Q6"
    "ScKXO7lglDoh2Zc7bGqUOv7vlzsddJQ64AaYO899lDrIHZg7f4eUOrU0mDsvkZQ6p0uYO96alDqeYpg7jKSUOpp5mDs5rpQ6"
    "m5CYO+a3lDqip5g7ksGUOq6+mDs9y5Q6v9WYO+jUlDrV7Jg7kd6UOvEDmTs66JQ6ERuZO+PxlDo3Mpk7ivuUOmJJmTsxBZU6"
    "k2CZO9cOlTrJd5k7fBiVOgOPmTsgIpU6RKaZO8MrlTqJvZk7ZjWVOtTUmTsIP5U6JOyZO6lIlTp5A5o7SVKVOtQamjvpW5U6"
    "NDKaO4hllTqZSZo7JW+VOgRhmjvCeJU6dHiaO1+ClTrpj5o7+ouVOmSnmjuUlZU65L6aOy6flTpp1po7x6iVOvTtmjtfspU6"
    "hAWbO/a7lToaHZs7jMWVOrU0mzshz5U6VUybO7bYlTr7Y5s7SeKVOqZ7mzvc65U6V5ObO271lToNq5s7//6VOsjCmzuPCJY6"
    "idqbOx4SljpQ8ps7rBuWOhwKnDs5JZY67SGcO8YuljrEOZw7UTiWOqBRnDvcQZY6gmmcO2VLljppgZw77lSWOk0mwTs7b6i6"
    "Pz7BO7VlqLo3VsE7L1youjVuwTuqUqi6OIbBOyZJqLpAnsE7oz+ouk62wTshNqi6Ys7BO6AsqLp75sE7ICOoupr+wTuhGai6"
    "vxbCOyMQqLrpLsI7pgaouhhHwjsq/ae6TV/CO6/zp7qId8I7NeqnusmPwju84Ke6D6jCO0XXp7pbwMI7zs2nuqzYwjtYxKe6"
    "A/HCO+O6p7pgCcM7b7GnusMhwzv8p6e6KzrDO4uep7qZUsM7GpWnugxrwzuri6e6hoPDOzyCp7oFnMM7z3inuom0wztib6e6"
    "FM3DO/dlp7qk5cM7jVynujr+wzskU6e61hbEO7xJp7p3L8Q7VUCnuh9IxDvvNqe6zGDEO4stp7p/ecQ7JySnujeSxDvFGqe6"
    "9qrEO2QRp7q6w8Q7BAinuoTcxDul/qa6VPXEO0f1proqDsU76uumugYnxTuP4qa66D/FOzXZprrPWMU728+murxxxTuExqa6"
    "sIrFOy29prqpo8U717Omuqi8xTuDqqa6rdXFOzChprq47sU73pemuskHxjuNjqa63yDGOz6Fprr8OcY773umuh9Txjuicqa6"
    "SGzGO1dpprp2hcY7DGCmuquexjvDVqa65bfGO3tNprom0cY7NESmum3qxjvuOqa6uQPHO6oxproMHcc7ZyimumU2xzsmH6a6"
    "xE/HO+UVpropacc7pgymupSCxztoA6a6BZzHOyz6pbp8tcc78fCluvnOxzu356W6fOjHO3/epboGAsg7R9WlupUbyDsSzKW6"
    "KzXIO93CpbrHTsg7qrmlumloyDt4sKW6EYLIO0inpbq/m8g7GZ6lunO1yDvrlKW6Ls/IO7+Lpbrv6Mg7lIKlurYCyTtreaW6"
    "gxzJO0NwpbpXNsk7HGelujBQyTv3XaW6EGrJO9NUpbr3g8k7sUuluuOdyTuQQqW61rfJO3E5pbo/vZk7FqtzOv7RmTs4sXM6"
    "w+aZO1W3czqO+5k7bb1zOl8QmjuAw3M6NiWaO47JczoTOpo7l89zOvdOmjua1XM64WOaO5jbczrReJo7keFzOseNmjuE53M6"
    "xKKaO3PtczrHt5o7W/NzOtDMmjs/+XM63+GaOx3/czr19po79gR0OhAMmzvJCnQ6MyGbO5cQdDpbNps7YBZ0OopLmzsjHHQ6"
    "v2CbO+AhdDr7dZs7mCd0OjyLmztKLXQ6haCbO/cydDrTtZs7njh0OijLmztAPnQ6hOCbO9xDdDrm9Zs7ckl0Ok4LnDsDT3Q6"
    "vSCcO45UdDoyNpw7E1p0Oq5LnDuSX3Q6MGGcOwxldDq4dpw7gGp0OkeMnDvub3Q63aGcO1Z1dDp5t5w7uHp0OhzNnDsVgHQ6"
    "xeKcO2uFdDp1+Jw7vIp0OisOnTsGkHQ66COdO0uVdDqsOZ07iZp0OnZPnTvCn3Q6R2WdO/SkdDoee507Iap0Ov2QnTtHr3Q6"
    "4aadO2e0dDrNvJ07gbl0Or/SnTuVvnQ6uOidO6LDdDq3/p07qch0Or4UnjuqzXQ6yyqeO6XSdDreQJ47mdd0OvlWnjuH3HQ6"
    "Gm2eO2/hdDpCg547UOZ0OnGZnjsr63Q6p6+eO//vdDrjxZ47zfR0OifcnjuV+XQ6cfKeO1b+dDrCCJ87EAN1OhofnzvEB3U6"
    "eTWfO3EMdTreS587FxF1Oktinzu3FXU6v3ifO1AadTo5j5874h51OrulnztuI3U6Q7yfO/MndTrT0p87cSx1OmnpnzvoMHU6"
    "BgCgO1k1dTqrFqA7wjl1OlYtoDslPnU6CUSgO4FCdTrDWqA71UZ1OoNxoDsjS3U6S4igO2pPdToan6A7qVN1OvC1oDviV3U6"
    "zcygOxNcdTqx46A7PWB1Op36oDtgZHU6jxGhO3xodTqJKKE7kWx1Ooo/oTuecHU6klahO6V0dTqibaE7o3h1OrmEoTubfHU6"
    "15uhO4uAdTr8sqE7c4R1OijKoTtViHU6XOGhOy6MdTqX+KE7AZB1OtoPojvLk3U6IyeiO4+XdTp1PqI7Spt1Os1Vojv+nnU6"
    "LW2iO6qidTqUhKI7T6Z1OgOcojvsqXU6ebOiO4GtdTr3yqI7DrF1OnziojuUtHU6CPqiOxG4dTqcEaM7h7t1Ojgpozv1vnU6"
    "20CjO1vCdTqFWKM7ucV1OjdwozsPyXU68YejO13MdTqyn6M7o891Onu3ozvg0nU6S8+jOxbWdToj56M7Q9l1OgP/oztp3HU6"
    "6hakO4bfdTrZLqQ7muJ1OloExDspCNq6WRzEO6cG2rpfNMQ7KQXaum1MxDuvA9q6gmTEOzoC2rqgfMQ7yQDausWUxDtc/9m6"
    "8azEO/P92bomxcQ7j/zZumPdxDsv+9m6p/XEO9P52brzDcU7fPjZukcmxTsp99m6oz7FO9r12boHV8U7kPTZunNvxTtK89m6"
    "5ofFOwny2bpioMU7zPDZuuW4xTuU79m6cdHFO2Du2boF6sU7MO3ZuqACxjsG7Nm6RBvGO9/q2brvM8Y7venZuqNMxjug6Nm6"
    "X2XGO4jn2bojfsY7dObZuu+Wxjtl5dm6w6/GO1rk2bqfyMY7VOPZuoPhxjtT4tm6cPrGO1bh2bplE8c7XuDZumIsxztr39m6"
    "Z0XHO33e2bp1Xsc7lN3Zuop3xzuv3Nm6qJDHO8/b2brPqcc79NrZuv3Cxzse2tm6NNzHO03Z2bpz9cc7gdjZursOyDu619m6"
    "CyjIO/jW2bpkQcg7OtbZusRayDuC1dm6LnTIO8/U2bqfjcg7INTZuhqnyDt309m6nMDIO9PS2bon2sg7NNLZurvzyDua0dm6"
    "Vw3JOwbR2br8Jsk7dtDZuqpAyTvsz9m6YFrJO2fP2boedMk7587ZuuWNyTtsztm6tafJO/fN2bqOwck7h83Zum/byTsczdm6"
    "WfXJO7fM2bpMD8o7V8zZukcpyjv8y9m6S0PKO6fL2bpYXco7V8vZum53yjsNy9m6jJHKO8jK2brs4qw86lWNvUfyrDwpgla9"
    "qAGtPIR9Gb0OEa08Dq/GvHsgrTxpI0+87i+tPDuSKLtmP608vofFO+VOrTxfv1g8al6tPMp9nDz0ba08JlXCPIV9rTzJgN48"
    "G42tPNmd8Ty4nK08qUv8PFusrTy6K/88A7ytPMXh+jyyy608wRPwPGfbrTzpad88IuutPMSOyTzj+q08KS+vPKsKrjxL+pA8"
    "eBquPHtDXzxMKq489LIXPCU6rjyrX5k7BUquPGI/qLfsWa48YMSbu9hprjySfhq8y3muPGKoZLzEia48yWyVvMOZrjxuw7W8"
    "yKmuPNWP0rzUua48BwfrvObJrjxXW/68/9muPC7eBb0e6q48cSsJvUP6rjx1qgi9bwqvPM7uA72hGq88PhX1vNkqrzwvHdi8"
    "GDuvPNgTsLxdS688Uyl4vKlbrzw04Oy7+2uvPA5lGztUfK88P45hPLOMrzxJyN08GZ2vPBWyLT2Fra88AEl1Pfi9rzwSE6M9"
    "h5qsPLbvHjwirKw8g2IdPMS9rDwc1Bs8bs+sPIBEGjwf4aw8rrMYPNjyrDykIRc8mAStPGGOFTxgFq085PkTPDAorTwsZBI8"
    "BzqtPDfNEDzmS608BDUPPM1drTySmw08u2+tPOAADDyyga087GQKPK+TrTy1xwg8taWtPDkpBzzDt608eIkFPNjJrTxw6AM8"
    "9tutPCBGAjwb7q08haIAPEgArjxB+/07fRKuPN6u+ju6JK484F/3O/82rjxDDvQ7TEmuPAa68DugW648JWPtO/1trjyeCeo7"
    "YoCuPG6t5jvPkq48kk7jO0WlrjwH7d87wreuPMqI3DtHyq482SHZO9XcrjwxuNU7a++uPM9L0jsJAq88r9zOO68UrzzQass7"
    "XievPC32xzsVOq88xX7EO9RMrzyUBME7m1+vPJeHvTtrcq88zAe6O0OFrzwvhbY7JJivPL3/sjsNq688c3evOxPesDz1TOO9"
    "DfGwPNWF470QBLE8477jvcl2mzycfqS8Y4WbPIWJpLwElJs8c5SkvKyimzxln6S8W7GbPFuqpLwRwJs8VbWkvM7OmzxTwKS8"
    "k92bPFbLpLxe7Js8XNakvDH7mzxo4aS8CwqcPHfspLztGJw8i/ekvNUnnDyjAqW8xTacPL8Npby8RZw83xilvLtUnDwEJKW8"
    "wWOcPC0vpbzOcpw8WzqlvOOBnDyNRaW8/5CcPMNQpbwioJw8/VulvE2vnDw8Z6W8gL6cPIBypby6zZw8yH2lvPzcnDwUiaW8"
    "ReycPGSUpbyW+5w8uZ+lvO4KnTwTq6W8ThqdPHG2pby2KZ0808GlvCU5nTw6zaW8nUidPKXYpbwcWJ08FeSlvKJnnTyJ76W8"
    "MXedPAL7pbzHhp08fwamvGWWnTwBEqa8C6adPIgdprxrA/k8roauPSET+TzKg6493yL5POWArj2kMvk8/32uPXJC+TwXe649"
    "SFL5PC94rj0mYvk8RXWuPQxy+Txbcq49+oH5PG9vrj1hHVw8MFV/v6IjXDy8Zwe/5ilcPOA/Ib4uMFw8IMYAPnk2XDwDmag+"
    "yDxcPHVA7D4aQ1w83roHP3BJXDysGAs/yk9cPEYjAj8nVlw8s2HdPodcXDzOBqU+62JcPGEfPT5TaVw8f6jSPL9vXDz+Txi+"
    "LnZcPHeWqr6gfFw8McAFvxeDXDwbPDa/kYlcPIeiZb8OkFw8M3mJv4ij2TxiJQk/yqbZPBcnxT4Pqtk86tqAPlWt2TzG8gw+"
    "nbDZPPmkNj3ns9k8oUnYvDO32TwYJJm9gbrZPHfdzb3Rvdk8YmLVvSPB2TzFCbK9dsTZPHbPTr3Mx9k8fhQ1OyPL2Tww+o49"
    "fc7ZPMeBFj7Y0dk8xWlrPjXV2TylgaA+ldjZPJuuyD7229k80+DqPlnf2TzcqwE/vuLZPBv1Bj8l5tk8sAEDP4/p2Txrn+Y+"
    "+uzZPCpIqj5n8Nk8vyMWPtbz2TxLofW9SPfZPPQH9r67+tk8XBdzvzD+2TyWcMW/RtmXPKKMRTyl8Jc8PKFFPBMImDzdtUU8"
    "kB+YPIfKRTwdN5g8ON9FPLlOmDzy80U8ZWaYPLQIRjwffpg8fR1GPOqVmDxPMkY8xK2YPChHRjyuxZg8ClxGPKjdmDz0cEY8"
    "sfWYPOaFRjzLDZk84JpGPPQlmTzhr0Y8LT6ZPOzERjx3Vpk8/tlGPNFumTwY70Y8O4eZPDoERzy1n5k8ZRlHPEC4mTyYLkc8"
    "JnO+PBpEIbzSi7481y4hvI6kvjyLGSG8W72+PDgEIbw51r483e4gvCfvvjx52SC8Jwi/PA3EILw3Ib88mK4gvFk6vzwcmSC8"
    "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"
    "4UdFPCJdqzwwOUY8e2iqPP5ARzyEVqk8rmFIPBEkqDzgnUk8os2mPHv4SjxVT6U8uXRMPOCkozwnFk48hMmhPLXgTzwAuJ88"
    "v9hRPIVqnTwVA1Q8pNqaPAllVjxCAZg8gQRZPIHWlDwA6Fs8rlGRPLkWXzwraY08pJhiPFISiTyQdmY8YUGEPDq6ajyq0n08"
    "Zm5vPJP3cTye0fM8VISNvLKu9jxEzpS81tf5PKLinLwUVf087talvKiXAD3rwq+8L7gCPd3AuryMEQU9zO3GvMCpBz3QadS8"
    "bYcKPV9Y47znsQ09rODzvEcxET0DFwO9gQ4VPSY4Db14Uxk9MG4YvRkLHj1a1iS9d0EjPeyQMr3pAyk9kMFBvSlhLz2vj1K9"
    "fmk2Pc8mZb023348XP10vSbffjwP/XS9FN9+PLr8dL0A3348Xfx0venefjz1+3S90d5+PIL7dL223n48BPt0vZjefjx4+nS9"
    "dt5+PN35dL1S3n48Mvl0vRFf1zyHPwa9+17XPLY+Br3iXtc8zz0Gvcde1zzQPAa9qF7XPLY7Br2HXtc8fjoGvWJe1zwmOQa9"
    "OV7XPKo3Br0MXtc8BTYGvdpd1zw1NAa9o13XPDUyBr1mXdc8/i8GvSNd1zyNLQa92FzXPNoqBr2GXNc83ycGvSxc1zyUJAa9"
    "x1vXPPIgBr1ZW9c87hwGvd9a1zyAGAa9WFrXPJsTBr3DWdc8NQ4GvZzMujx7/J885su6PKYJoDwey7o8LxigPEHKujw6KKA8"
    "Tcm6PPA5oDw/yLo8fE2gPBbHujwPY6A85LdxPESvAj4PtXE8jbICPvCxcTwttgI+fq5xPC26Aj6xqnE8lr4CPn+mcTx1wwI+"
    "3qFxPNTIAj7CnHE8wc4CPiCXcTxL1QI+pJ2SPJX8Hj46gJI8WmUePslfkjyLvh0+ATySPI8GHT6KFJI8pDscPgTpkTzZWxs+"
    "BLmRPAxlGj4WhJE841QZPrhJkTzHKBg+XQmRPNzdFj5nwpA8/XAVPuJ5yDwo5iM+oSPIPKQqIj6JxMc8skEgPrZbxzywJh4+"
    "KOjGPIbUGz7IaMY8lkUZPmPcxTyzcxY+pkHFPA1YEz4dVNA80ks9PeuYzzwnNi49zMnOPCiSHT2O5c08yDsLPQ3qzDz3De48"
    "7tTLPI2GwTyZo8o8kHaQPDFTyTwA0TQ8Gr/FPKDaubvPJsQ8yCNgvAhlwjyOV7i8lXXAPAn6A73YU748ltIvvb76uzzVHWC9"
    "sWS5PFOnir2Mi7Y8ufGnvYxoszymM8i9QPSvPJy56715Jqw85GsJvjb2pzxi9R6+jlmjPDasNr6dRZ48aMhQvmeumDyZh22+"
    "u4aSPMmWhr4bwIs8cQKYvpRKhDzFL6u+OCl4PNtLwL7UFWY8SYjXvg=="
)
_AVERAGES = [  # (name, ensemble weight, coefficient per block per running sum)
    ("avg_epoch7_half", 0.021397176990490183, [[1.0805897168697412e-05, -7.49341083635946e-05], [-3.0445270125844474e-05, 0.00011261051150719288], [7.807370609783108e-05, -3.314995934803298e-05], [-0.00027844004492319583, -0.00023451671388928389], [-0.0018828843371237356, 0.016050942676289336], [-0.007696548316338139, 0.08132521014559409], [1.0218078444246836, -3.7684776896619385], [-0.010639079159724249, -0.08757681636534931], [-0.001820971064099467, 0.0011558852921067504], [0.00038850886019055044, 0.0010605392395715662], [5.668106272912439e-05, -8.398146794482346e-05], [6.5234475368934535e-06, -9.914229332914418e-07], [-6.451290506152275e-08, -2.439571848104845e-07], [-7.136808742946361e-09, -1.4057554839642838e-09], [1.0238859336699697e-09, -4.030075011724125e-09], [1.4175862174641264e-09, -5.938644283455184e-10], [1.990184348555153e-12, 4.2298381504687665e-12], [1.367860767174073e-13, 3.3206213706755874e-13], [1.536359898920999e-14, -3.4465581527207435e-14], [2.3255087846480515e-14, 3.850147652331566e-15]]),
    ("avg_epoch7_full", 0.01947330670774621, [[-4.520097854256732e-06, 3.134487563416499e-05], [1.273523133075189e-05, -4.710488389566264e-05], [-3.265816672337611e-05, 1.3866600584610075e-05], [0.00011647124061983436, 9.809814755699697e-05], [0.0007876089617612962, -0.006714096053210072], [0.003219459798458342, -0.03401826817740509], [0.9987918318361643, 0.6285852007543229], [-0.002561274180430315, -0.021083426036558252], [-0.0004383843845709309, 0.00027827024405008803], [9.353043600555605e-05, 0.0002553164358266338], [1.3645517606426291e-05, -2.0217874374974046e-05], [1.5704684512800226e-06, -2.3867723757568144e-07], [-1.553097030157322e-08, -5.8730761622935536e-08], [-1.7181315682785024e-09, -3.384252855571175e-10], [2.465437479116113e-10, -9.701733749674627e-10], [3.412258731182483e-10, -1.4294484489749394e-10], [4.721083435865117e-13, 1.0072349360684922e-12], [3.2184124858132886e-14, 8.08592410774333e-14], [2.9153373384405197e-15, -8.38301512182842e-15], [4.3180355840955227e-16, 2.004656225667337e-15]]),
    ("ema61_at_epoch7", 0.012943587647187418, [[-2.7251892243125612e-05, 0.00019749773801833181], [9.382134634921315e-05, -0.00028779055169505895], [-0.00010543821020923392, 0.0001425048044438587], [0.001731653692828557, 0.00021667109000554405], [0.013757904781133458, -0.04720794652821644], [0.09209304761934624, -0.22609059441924595], [0.8696914953805277, -1.6832202140618842], [0.020168885026860667, 0.1660225207253215], [0.00345208034249369, -0.0021912533228350193], [-0.0007365102200475482, -0.002010502381846306], [-0.00010745232946794528, 0.0001592066893318541], [-1.2366734021506481e-05, 1.879476096896394e-06], [1.2229942275653134e-07, 4.6247846292947845e-07], [1.3529520170996297e-08, 2.6649465502458064e-09], [-1.9409951481822288e-09, 7.639971649253909e-09], [-2.6873714667954396e-09, 1.125828797530284e-09], [-3.787722847033468e-12, -8.099645115702019e-12], [-2.6836041271244977e-13, -6.603840636265387e-13], [-2.5502068718343315e-14, 7.371731965681589e-14], [-4.942957318913841e-14, -1.1116042306423565e-14]]),
    ("ema121_at_epoch7", 0.009201432983940439, [[0.0008722233130975315, -0.0005142656166447213], [0.0027948148330563583, 0.0003483033782441096], [0.007615364577128764, 0.001481421487860986], [0.025982598180906667, -0.0009836203222237316], [0.07948884154777637, -0.10516869145997144], [0.21601291940087936, -0.2548191353888454], [0.6622447579289326, -0.4638888997129047], [0.004419640347406417, 0.036380783082474485], [0.0007564599403561823, -0.0004801728793663951], [-0.00016139267393705783, -0.00044056463381165995], [-2.3546202562448963e-05, 3.488721888209397e-05], [-2.709942330076536e-06, 4.1185262207347726e-07], [2.6799664886701327e-08, 1.0134366119252302e-07], [2.9647496991402326e-09, 5.839757163488721e-10], [-4.2532268391033746e-10, 1.6741658389723277e-09], [-5.888922246408644e-10, 2.467114337274953e-10], [-8.344279801743625e-13, -1.7966964325690645e-12], [-6.123676211062487e-14, -1.530542534678642e-13], [-6.0808243956004e-15, 1.7704848190549132e-14], [-1.2109389807458687e-14, -2.749270557021276e-15]]),
    ("avg_epoch8_full", 0.037554068246038186, [[-4.088186327592405e-07, 2.834976091686814e-06], [1.1518334380193707e-06, -4.26038435845225e-06], [-2.9537561840873404e-06, 1.2541597458580136e-06], [1.0534199601360945e-05, 8.872451788344727e-06], [7.123501019102796e-05, -0.0006072540105090128], [0.0002911828873514429, -0.0030767700701080703], [0.004585188855840695, -0.03315804265858578], [0.9981345193222717, 0.525470643577379], [-0.004109770411610216, 0.0026087307296215794], [0.0008768300879255356, 0.0023935431338010754], [0.0001279241384392201, -0.0001895387351306377], [1.472284336588935e-05, -2.2375537599033827e-06], [-1.4559991318994916e-07, -5.50589823816153e-07], [-1.6107152794613697e-08, -3.172669419056518e-09], [2.3108708375233107e-09, -9.095481549264607e-09], [3.199307314575758e-09, -1.3402816470644392e-09], [4.491169124112298e-12, 9.594612525327078e-12], [3.16550470747076e-13, 7.742410709780423e-13], [3.509127505969912e-14, -8.314004372778393e-14], [4.9300552823900394e-14, 1.3374436371429883e-14]]),
    ("ema61_at_epoch8", 0.024028031930144668, [[-1.6785416566027828e-06, 1.2612247308044222e-05], [6.674318023000817e-06, -1.7925383475124944e-05], [-1.6877309870765098e-06, 1.187295310797537e-05], [0.0001607603675181656, -6.350080759730666e-06], [0.0013208795503075628, -0.0032613090292496227], [0.009492130122960806, -0.015028989815726366], [0.08321024522859288, -0.07597429501209871], [0.8929898532674285, -1.7515187833517274], [0.017052166585594806, -0.010824086633329344], [-0.0036381236003506924, -0.009931235120249947], [-0.0005307799469033668, 0.000786429839257091], [-6.108768879961972e-05, 9.284007459005677e-06], [6.041198529116631e-07, 2.2844948440731136e-06], [6.683144814786492e-08, 1.3163972098440997e-08], [-9.588019428570665e-09, 3.7738902173983466e-08], [-1.327464529776787e-08, 5.561163974029678e-09], [-1.868447693303533e-11, -3.990563705206395e-11], [-1.3159686471322461e-12, -3.229463888194082e-12], [-1.3784137675835794e-13, 3.455734490183515e-13], [-2.2501010278197907e-13, -4.849635456321256e-14]]),
    ("avg_last_cycle", 0.10704502766084374, [[2.5596549137517795e-16, -3.8484840314545465e-15], [-8.367307831405593e-16, 5.936733586331585e-16], [1.5845857365146545e-15, 2.5047608265985128e-14], [-4.557346734971205e-15, 2.031334786120308e-15], [-1.1303043602680022e-14, 1.5932500652068224e-13], [-3.493241364598098e-15, 2.85860591977919e-13], [-5.966090710444776e-14, 6.573292377993669e-13], [-1.1683742208591092e-12, 8.877215314543688e-12], [8.309674106985803e-12, 6.016804189334298e-12], [3.8042986770342763e-10, -1.0339645712458141e-09], [1.917027755161717e-08, 2.0241369979917585e-08], [-1.5494692525310283e-07, 7.95894354511083e-09], [-9.732650952886074e-07, 3.2279816325480842e-06], [5.9036895289679605e-06, -9.975791737505298e-07], [0.00017431581223330226, 0.0001324547318477244], [0.00558294626322996, 0.0010134931594083568], [0.24867290268428452, 0.023285637563071403], [0.2533791909759182, 0.10435119059748453], [0.23687240458215694, -0.0001622924104651014], [0.25531344464690253, 0.001434797563664009]]),
]
_CHECKPOINTS = [  # (epoch, ensemble weight)
    (7, 0.011931739669738171),
    (8, 0.023854448304529173),
    (9, 0.2273221075616267),
    (10, 0.2123078019487163),
    (11, 0.2929412703489988),
]


def _sketch_bases():
    raw = base64.b64decode("".join(_SKETCH_BASES_B64))
    assert hashlib.sha256(raw).hexdigest() == _SKETCH_BASES_SHA256, "sketch table corrupted"
    table = np.frombuffer(raw, dtype="<f4").reshape(_SKETCH_BLOCK_ENDS[-1], 2).astype(np.float64)
    starts = [0] + _SKETCH_BLOCK_ENDS[:-1]
    return [table[a:b] for a, b in zip(starts, _SKETCH_BLOCK_ENDS)]


class SketchedAverages:
    """Running block sums of the weight trajectory, folded into fixed weight averages.

    Each rank tracks one contiguous slice of the flattened parameters, so the memory cost is
    (2 + n_averages) fp32 copies of 1/world_size of the model.
    """

    def __init__(self, params, rank, world_size, device):
        self.params = list(params)
        sizes = [p.numel() for p in self.params]
        self.total = sum(sizes)
        self.chunk = -(-self.total // world_size)
        self.lo = min(rank * self.chunk, self.total)
        self.hi = min(self.lo + self.chunk, self.total)
        self.pieces = []  # (param, start, stop, offset in this rank's slice)
        off = 0
        for p, n in zip(self.params, sizes):
            a, b = max(off, self.lo), min(off + n, self.hi)
            if a < b:
                self.pieces.append((p, a - off, b - off, a - self.lo))
            off += n
        n_local = self.chunk
        self.ws = world_size
        self.bases = _sketch_bases()
        self.coefs = torch.tensor([row for _, _, row in _AVERAGES], dtype=torch.float32, device=device)
        self.live = torch.zeros(n_local, dtype=torch.float32, device=device)
        self.sums = torch.zeros(2, n_local, dtype=torch.float32, device=device)
        self.avgs = torch.zeros(len(_AVERAGES), n_local, dtype=torch.float32, device=device)
        self.block, self.block_start = 0, 1

    @torch.no_grad()
    def update(self, step):
        """Consume the weights after 1-based optimizer step ``step``."""
        if self.block >= len(_SKETCH_BLOCK_ENDS):
            return
        for p, a, b, d in self.pieces:
            self.live[d:d + (b - a)].copy_(p.detach().reshape(-1)[a:b])
        v = self.bases[self.block][step - self.block_start]
        self.sums[0].add_(self.live, alpha=float(v[0]))
        self.sums[1].add_(self.live, alpha=float(v[1]))
        if step == _SKETCH_BLOCK_ENDS[self.block]:
            self.avgs.addmm_(self.coefs[:, self.block, :], self.sums)
            self.sums.zero_()
            self.block += 1
            self.block_start = step + 1

    @torch.no_grad()
    def load_average(self, k):
        """Write average k into the model parameters (gathers every rank's slice)."""
        full = torch.empty(self.chunk * self.ws, dtype=torch.float32, device=self.avgs.device)
        if dist.is_initialized():
            dist.all_gather_into_tensor(full, self.avgs[k].contiguous())
        else:
            full.copy_(self.avgs[k])
        off = 0
        for p in self.params:
            n = p.numel()
            p.copy_(full[off:off + n].view_as(p))
            off += n
        del full


sketch = None
if not args.eval_logit_avg:
    assert num_iterations == _SKETCH_BLOCK_ENDS[-1], (
        f"the frozen averages are laid out for a {_SKETCH_BLOCK_ENDS[-1]}-step schedule, got {num_iterations}")
    sketch = SketchedAverages(orig_model.parameters(), ddp_rank, ddp_world_size, device)
    print0(f"Sketched averages: {len(_AVERAGES)} averages over {len(_SKETCH_BLOCK_ENDS)} blocks, "
           f"{(2 + len(_AVERAGES)) * sketch.chunk * 4 / 1e9:.2f} GB/rank")

# Training loop
step = 0
min_val_bpb = float("inf")
min_val_loss = float("inf")
epochs_without_improvement = 0
smooth_train_loss = 0
total_training_time = 0
timed_steps = 0
timing_start_step = 4  # skip first compile + 3 warmup steps
eval_steps = EVAL_TOKENS // (args.device_batch_size * MAX_SEQ_LEN * ddp_world_size)
dupe_active = False

late_checkpoint_paths = []  # paths to saved epoch checkpoints for logit averaging
logit_avg_count = args.logit_avg
if logit_avg_count > 0 and master_process:
    os.makedirs(args.logit_avg_dir, exist_ok=True)
if logit_avg_count > 0:
    print0(f"Logit averaging: saving last {logit_avg_count} epoch checkpoints to {args.logit_avg_dir}/")

if args.eval_logit_avg:
    print0("--eval-logit-avg set: skipping training, loading checkpoints from disk.")
else:
    # Initial val evaluation
    model.eval()
    val_loader = build_val_loader()
    with autocast_ctx:
        val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print0(f"Step {step:05d} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
    wandb_run.log({"step": step, "val/bpb": val_bpb, "val/loss": val_loss})
    min_val_bpb = val_bpb
    min_val_loss = val_loss
    model.train()

while not args.eval_logit_avg and current_epoch <= args.num_epochs:
    if not dupe_active and current_epoch >= args.dupe_start_epoch:
        print0(f"\n=== Enabling dupe-layers at epoch {current_epoch} ===")
        orig_model.set_dupe_layers(args.dupe_layers_start, args.dupe_layers_end, args.dupe_loops)
        model = torch.compile(orig_model, dynamic=False)
        # model = orig_model # replace compile with this line for eager mode
        dupe_active = True
        timing_start_step = step + 4  # skip dupe recompile + 3 warmup steps
        gc.enable(); gc.collect()

    # Training step
    synchronize()
    t0 = time.time()
    if MG and step % args.mg_every == 0 and grad_accum_steps >= 2:
        # split pairing: the first half of this step's micro-batches adapts, the second half is
        # evaluated at the adapted point; the optimizer receives the ordinary full-batch mean gradient
        half = grad_accum_steps // 2
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss, metrics = model(x, y)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)
            if micro_step == half - 1:
                mg_kept, mg_alpha = mg_temporary_step(step)
        with torch.no_grad():
            torch._foreach_add_(mg_owned_params, mg_kept, alpha=-mg_alpha)   # restore the owned matrices
        del mg_kept
    else:
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss, metrics = model(x, y)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)

    # Update optimizer
    lrm = get_lr_multiplier(step)
    # SWA: cosine-cycle LR in final epochs for diverse checkpoints to average
    if _swa_start_step >= 0 and step >= _swa_start_step:
        cycle_pos = (step - _swa_start_step) % steps_per_epoch
        swa_base = max(lrm, 0.05)
        lrm = 0.05 + (swa_base - 0.05) * (1 + math.cos(math.pi * cycle_pos / steps_per_epoch)) / 2
    # WD schedule: pre-SWA decay from base to 0.65×base, then anti-phase sawtooth
    # during SWA's N epochs (0.65→1.50×base per epoch, anti-phase with LR cosine cycle).
    wdm = get_wd_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if "initial_wd" not in group:
            group["initial_wd"] = group.get("weight_decay", 0.0)
        group["weight_decay"] = group["initial_wd"] * wdm
        if group['kind'] == 'muon':
            group["momentum"] = get_muon_momentum(step)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    if sketch is not None:
        sketch.update(step + 1)
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
    if step >= timing_start_step:
        total_training_time += dt
        timed_steps += 1
    eta_str = f" | eta: {(num_iterations - step) * total_training_time / timed_steps / 60:.1f}m" if timed_steps > 0 else ""
    dupe_str = " [DUPE]" if dupe_active else ""
    print0(f"step {step:05d} ({pct:.2f}%) | loss: {debiased:.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}%{dupe_str}{eta_str}")
    wandb_run.log({"step": step, "train/loss": debiased, "train/mfu": mfu,
                   **{f"train/{k}": v.item() for k, v in metrics.items()}})

    # Synchronize epoch across ranks (different ranks may exhaust data at different steps)
    if ddp:
        epoch_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
        dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MAX)
        epoch = epoch_tensor.item()
    if args.schedule_steps > 0:
        # Epochs follow the schedule: epoch k ends at the first step with step * num_epochs >= k * num_iterations.
        epoch = 1 + min(args.num_epochs, (step * args.num_epochs) // num_iterations)

    # Epoch boundary: evaluate when the dataloader advances to a new epoch
    if epoch != current_epoch:
        model.eval()
        val_loader = build_val_loader()
        with autocast_ctx:
            val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Epoch {current_epoch} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
        wandb_run.log({"step": step, "epoch": current_epoch, "val/bpb": val_bpb, "val/loss": val_loss})
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
        # Save checkpoint to disk for logit averaging
        if logit_avg_count > 0:
            ckpt_path = os.path.join(args.logit_avg_dir, f"epoch_{current_epoch:03d}.pt")
            if master_process:
                ckpt = {name: p.data.float().cpu() for name, p in orig_model.named_parameters()}
                torch.save(ckpt, ckpt_path)
                del ckpt
            late_checkpoint_paths.append(ckpt_path)
            if len(late_checkpoint_paths) > logit_avg_count:
                old = late_checkpoint_paths.pop(0)
                if master_process and os.path.exists(old):
                    os.remove(old)
            print0(f"  Saved checkpoint {ckpt_path} ({len(late_checkpoint_paths)}/{logit_avg_count})")

        model.train()
        # Update num_iterations estimate now that we know real steps per epoch
        # steps_per_epoch = step // current_epoch
        # num_iterations = steps_per_epoch * args.num_epochs
        # print0(f"Epoch {current_epoch} took {steps_per_epoch} steps. Updated estimate: {num_iterations} total steps.")
        current_epoch = epoch

    # GC management
    if step == 1:
        gc.collect(); gc.freeze(); gc.disable()

# =============================================================================
# Post-training: evaluate checkpoint averages
# =============================================================================

# Final model: frozen-weight probability ensemble of 5 epoch checkpoints + 7 sketched averages
if sketch is not None:
    ckpt_by_epoch = {int(os.path.basename(p)[6:9]): p for p in late_checkpoint_paths}
    missing = [e for e, _ in _CHECKPOINTS if e not in ckpt_by_epoch]
    assert not missing, f"epoch checkpoints missing for the ensemble: {missing}"
    members = [ckpt_by_epoch[e] for e, _ in _CHECKPOINTS] + \
              [partial(sketch.load_average, k) for k in range(len(_AVERAGES))]
    ens_w = [w for _, w in _CHECKPOINTS] + [w for _, w, _ in _AVERAGES]
    names = [f"epoch_{e:03d}" for e, _ in _CHECKPOINTS] + [n for n, _, _ in _AVERAGES]
    print0(f"\n--- Final ensemble ({len(members)} members) ---")
    for nm, w in zip(names, ens_w):
        print0(f"  {nm:18s} weight {w:.4f}")
    optimizer = None  # free optimizer state before the evaluation passes
    gc.collect(); torch.cuda.empty_cache()
    ens_start = time.time()
    ens_model = torch.compile(orig_model, dynamic=False)
    ens_model.eval()
    ens_bpb, ens_loss = evaluate_bpb_logit_avg(ens_model, members, ens_w, eval_steps)
    print0(f"  Ensemble Val BPB: {ens_bpb:.6f} | Val Loss: {ens_loss:.6f} "
           f"(eval {(time.time() - ens_start) / 60:.2f}m)")
    wandb_run.log({"ensemble/bpb": ens_bpb, "ensemble/loss": ens_loss})
    val_bpb, val_loss = ens_bpb, ens_loss
    if ens_loss < min_val_loss:
        min_val_loss, min_val_bpb = ens_loss, ens_bpb

# Evaluate logit (probability) average of saved checkpoints only (eval-only mode)
if logit_avg_count > 0 and sketch is None:
    # In eval-only mode, discover checkpoints from disk; otherwise use what was saved during training
    if args.eval_logit_avg:
        import glob as _glob
        all_disk = sorted(_glob.glob(os.path.join(args.logit_avg_dir, "epoch_*.pt")))
        ckpt_paths_for_logit = all_disk[-logit_avg_count:]
    else:
        ckpt_paths_for_logit = late_checkpoint_paths

    if len(ckpt_paths_for_logit) >= 2:
        n = len(ckpt_paths_for_logit)
        print0(f"\n--- Evaluating logit avg ({n} checkpoints: {[os.path.basename(p) for p in ckpt_paths_for_logit]}) ---")

        la_model = torch.compile(orig_model, dynamic=False)
        la_model.eval()

        def _run_mode(label, weights):
            print0(f"  [{label}] weights: {[f'{w:.3f}' for w in weights]}")
            bpb, loss = evaluate_bpb_logit_avg(la_model, ckpt_paths_for_logit, weights, eval_steps)
            print0(f"  [{label}] Val BPB: {bpb:.6f} | Val Loss: {loss:.6f}")
            wandb_run.log({f"logit_avg_{label}/bpb": bpb, f"logit_avg_{label}/loss": loss})
            return bpb, loss

        equal_w    = [1.0 / n] * n
        raw_w      = list(range(1, n + 1))
        weighted_w = [w / sum(raw_w) for w in raw_w]

        if args.logit_avg_mode in ("equal", "both"):
            eq_bpb, eq_loss = _run_mode("equal", equal_w)
            if eq_loss < min_val_loss:
                min_val_loss, min_val_bpb = eq_loss, eq_bpb
                print0(f"  ** New best! (logit avg equal weights)")

        if args.logit_avg_mode in ("weighted", "both"):
            wt_bpb, wt_loss = _run_mode("weighted", weighted_w)
            if wt_loss < min_val_loss:
                min_val_loss, min_val_bpb = wt_loss, wt_bpb
                print0(f"  ** New best! (logit avg recency weights)")


# Summary
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
final_train_loss = smooth_train_loss / (1 - 0.9**step) if step > 0 else float('inf')
print0(f"Final train loss: {final_train_loss:.6f}")
print0(f"Min val BPB: {min_val_bpb:.6f}")
print0(f"Min val Loss: {min_val_loss:.6f}")
wandb_run.summary["final_train_loss"] = final_train_loss
wandb_run.summary["best_val_loss"] = min_val_loss

_result_out = args.save_result or result_path
if master_process:
    result = {
        "matrix_lr": args.matrix_lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "val_loss": val_loss,
        "best_val_loss": min_val_loss,
        "wandb_url": getattr(wandb_run, "url", None),
    }
    with open(_result_out, "w") as f:
        json.dump(result, f, indent=2)
    print0(f"Result saved to {_result_out}")

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
