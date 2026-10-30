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
parser.add_argument("--schedule-steps", type=int, default=2085,
                    help="Optimizer steps for the LR/SWA schedule; epochs follow the schedule (0 = derive from the data)")
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

_SKETCH_BLOCK_ENDS = [190, 379, 569, 758, 948, 1138, 1327, 1517, 1564, 1611, 1658, 1706, 1753, 1801, 1848, 1896, 1943, 1990, 2037, 2085]
_SKETCH_BASES_SHA256 = "76fbeff59e831648d237ee9289cd989e6919acfda3a1a4563edf56f93bf33d16"
_SKETCH_BASES_B64 = (
    "jwqoO1kh8rovFqg7E6TvutIhqDuIJe26eS2oOxGm6rojOag72CXoutBEqDv1pOW6f1CoO3kj47owXKg7cqHguuNnqDvqHt66"
    "mHOoO+eb27pPf6g7chjZugiLqDuPlNa6w5aoO0QQ1Lp/oqg7lIvRuj2uqDuEBs+6/bmoOxaBzLq+xag7TvvJuoHRqDsvdce6"
    "Rt2oO7ruxLoM6ag78WfCutP0qDvY4L+6nACpO29ZvbpnDKk7udG6ujMYqTu3Sbi6ACSpO2vBtbrPL6k71Tizup87qTv4r7C6"
    "cUepO9QmrrpDU6k7a52ruhhfqTu9E6m67WqpO82JprrEdqk7mv+jupyCqTsldaG6do6pO3HqnrpRmqk7fF+cui2mqTtJ1Jm6"
    "CrKpO9hIl7rpvak7Kb2UusnJqTs+MZK6qtWpOxalj7qM4ak7tBiNunDtqTsWjIq6VfmpOz7/h7o7Bao7LXKFuiIRqjvj5IK6"
    "Cx2qO2BXgLr1KKo7S5N7uuA0qjtnd3a6zECqOxRbcbq5TKo7VD5suqhYqjsoIWe6mGSqO5ADYrqJcKo7juVcunt8qjsix1e6"
    "boiqO02oUrpilKo7D4lNuligqjtqaUi6T6yqO15JQ7pHuKo77Cg+ukDEqjsVCDm6OtCqO9jmM7o23Ko7N8UuujLoqjszoym6"
    "MPSqO8yAJLovAKs7Al4fui4MqzvXOhq6MBirO0oXFboyJKs7XPMPujUwqzsOzwq6OjyrO2CqBbo/SKs7VIUAukZUqzvQv/a5"
    "TmCrOzx07LlXbKs77SfiuWF4qzvj2te5bISrOx+Nzbl4kKs7oz7DuYWcqztu77i5lKirO4KfrrmjtKs74E6kubTAqzuI/Zm5"
    "xsyrO3urj7nZ2Ks7uliFue3kqzuNCna5AvGrO0BiYbkY/as7j7hMuS8JrDt8DTi5RxWsOwhhI7lhIaw7NLMOuXstrDsFCPS4"
    "lzmsO+imyri0Raw7FEOhuNFRrDsWuW+48F2sO6DmHLgQaqw7lB2UtzF2rDtq5gw2U4KsO9thtzd2jqw7xpguOJqarDv2woA4"
    "wKasOy08qjjmsqw7BrjTOA2/rDt/Nv04NsusO8tbEzlf16w7pB0oOYrjrDvL4Dw5tu+sOz2lUTni+6w7+mpmORAIrTsBMns5"
    "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"
    "BfiyO2noTTp2D7M77aWVOuomszufecM6Yj6zO8op8DrdVbM7K7wNO11tszuMkCI734SzOx90NjtmnLM70UpJO/CzszsF9lo7"
    "fsuzOz9fazsQ47M7gGt6O6X6szvsAoQ7PxK0Ox4MijvbKbQ7aEmPO3xBtDsZsZM7IFm0OwE9lzvIcLQ71+aZO3SItDvpqZs7"
    "I6C0OwuCnDvXt7Q7XG2cO43PtDuIaZs7SOe0O7x3mTsG/7Q7GZiWO8gWtTtczJI7ji61O4kZjjtYRrU7A4OIOyVetTukDoI7"
    "9nW1O4mJdTvLjbU7fVhlO6SltTtHnVM7gL21O6tvQDtg1bU7AuIrO0TttTu9EhY7LAW2O/A0/joXHbY7ci3OOgc1tjv7Tpw6"
    "+ky2O1u7UTrwZLY7wWXQOet8tjsNAVa36ZS2O3nC4LnsrLY7OzdeuvLEtjvZIqa6+9y2O/3x3LoJ9bY7W5gJuxoNtzskRSS7"
    "LyW3O7JSPrtIPbc72pVXu2VVtzv54W+7hm23O9yDg7uqhbc7lG6Ou9Kdtzvampi7/7W3Oxbzobsuzrc7oWCqu2Lmtzv6z7G7"
    "mv63OxEsuLvVFrg7KGK9uxQvuDucX8G7WEe4O2USxLufX7g7NmvFu+l3uDvOWsW7OJC4O+XSw7uLqLg7s8jAu+HAuDsWMry7"
    "O9m4O5oHtruZ8bg7fkKuu/sJuTvI36S7YSK5O9nembvLOrk7skGNuzhTuTtOFn67qmu5O2mKXrsfhLk77/M7u5mcuTujcBa7"
    "FrW5O0ZG3LqXzbk78nSGuhzmuTvaTK+5pf65OzGGzDkxF7o76d+VOsIvujvtxPs6V0i6O3kLMjvvYLo7sQlnO4t5ujvcNI47"
    "LJK6O8XYqDvQqro7RivDO3jDujtS4tw7JNy6O/us9TvU9Lo7apkGPIgNuzuriRE8QCa7O2V0Gzz8Prs7SiIkPLxXuzsXVys8"
    "gHC7O5LTMDxHibs7k1M0PBOiuzvcjjU847q7Ox03NDy207s7r/ovPI7suzuygig8aQW8O9xxHTxJHrw7lmYOPCw3vDvH8vU7"
    "FFC8O7J6xTv/aLw75XyKO++BvDv8EQg74pq8O6Mi7rnZs7w74aBeu9XMvDuSWt671OW8O+mULrzX/rw7enl2vN8XvTu+vqO8"
    "6jC9O0Ql0bz6Sb07keQBvW4DojuX9+m6ZB6iO9a057pgOaI7P3HlumFUojvULOO6aG+iO5Tn4LpziqI7fqHeuoSlojuSWty6"
    "m8CiO9AS2rq226I7OMrXutf2ojvJgNW6/RGjO4Q207opLaM7Z+vQulpIoztzn866kGOjO6hSzLrLfqM7BQXKugyaozuJtse6"
    "UrWjOzZnxbqd0KM7CRfDuu7rozsExsC6RAekOyZ0vrqfIqQ7byG8ugA+pDvezbm6ZlmkO3N5t7rSdKQ7LiS1ukOQpDsOzrK6"
    "uaukOxR3sLo0x6Q7Px+uurXipDuPxqu6PP6kOwRtqbrIGaU7nRKnulk1pTtat6S671ClOztborqLbKU7P/6fui2IpTtnoJ26"
    "06OlO7JBm7qAv6U7H+KYujHbpTuvgZa66PalO2IglLqlEqY7Nr6RumcupjstW4+6LkqmO0T3jLr7ZaY7fZKKus2BpjvXLIi6"
    "pZ2mO1LGhbqDuaY77V6DumXVpjup9oC6TfGmOwgbfbo7Dac7/kZ4ui4ppzszcXO6J0WnO6aZbrolYac7WMBpuil9pztG5WS6"
    "MpmnO3EIYLpBtac72ClbulXRpzt8SVa6b+2nO1pnUbqOCag7c4NMurMlqDvGnUe63UGoO1K2QroNXqg7GM09ukJ6qDsW4ji6"
    "fZaoO031M7q+sqg7ugYvugTPqDtfFiq6T+uoOzokJbqhB6k7SzAguvcjqTuSOhu6VECpOw1DFrq2XKk7vEkRuh15qTugTgy6"
    "ipWpO7ZRB7r9sak7AFMCunbOqTv2pPq59OqpO1Cg8Ll3B6o7DJjmuQEkqjsqjNy5j0CqO6d80rkkXao7hGnIub55qju+Ur65"
    "XpaqO1U4tLkDs6o7SBqqua7PqjuW+J+5X+yqOz3TlbkWCas7PaqLudIlqzuUfYG5lEKrO4SabrlbX6s7ijJauSh8qzs4w0W5"
    "+5irO41MMbnUtas7hc4cubLSqzsgSQi5lu+rO7R457h/DKw7Y1C+uG8prDtJGZW4ZEasO8SmV7hfY6w7Uf0EuF+ArDvF2Ei3"
    "Zp2sO466Ajdyuqw7bWLnN4PXrDuR0UY4m/SsO90HjTi4Ea073rW2ONsurTvQcuA4BEytO1sfBTkzaa07ywwaOWeGrTu6AS85"
    "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"
    "Jh20O/ttLzrrNLQ7udAwOrZMtDtiNDI6hmS0O/aYMzpdfLQ7dv40OjmUtDviZDY6HKy0OzrMNzoExLQ7fzQ5OvPbtDuynTo6"
    "5/O0O9IHPDriC7U733I9OuMjtTvb3j466Tu1O8VLQDr2U7U7n7lBOglstTtnKEM6IoS1Ox+YRDpBnLU7xwhGOma0tTtfekc6"
    "kcy1O+jsSDrC5LU7YWBKOvn8tTvM1Es6NxW2OylKTTp6LbY7d8BOOsRFtju4N1A6E162O+yvUTppdrY7EylTOsWOtjsto1Q6"
    "J6e2OzseVjqPv7Y7PZpXOv7XtjszF1k6cvC2Ox+VWjrtCLc7/xNcOm4htzvVk1069Tm3O6EUXzqCUrc7Y5ZgOhVrtzscGWI6"
    "r4O3O8ycYzpOnLc7cyFlOvS0tzsSp2Y6oc23O6ktaDpT5rc7OLVpOgz/tzvAPWs6yhe4O0HHbDqPMLg7vFFuOltJuDsw3W86"
    "LGK4O59pcToEe7g7CfdyOuKTuDtthXQ6xqy4O80Udjqxxbg7KKV3OqLeuDt/Nnk6mfe4O9PIejqXELk7JFx8OpopuTtx8H06"
    "pEK5O72Ffzq1W7k7A46AOsx0uTunWYE66Y25O8olgjp7q5Y7xruNOo/Eljvsv406qt2WOw/EjTrN9pY7MMiNOvYPlztPzI06"
    "JimXO2vQjTpeQpc7htSNOp1blzue2I0643SXO7TcjTowjpc7x+CNOoSnlzvZ5I0638CXO+jojTpC2pc79OyNOqzzlzv/8I06"
    "HQ2YOwf1jTqVJpg7DfmNOhRAmDsQ/Y06m1mYOxEBjjopc5g7EAWOOr6MmDsNCY46WqaYOwcNjjr+v5g7/xCOOqnZmDv1FI46"
    "W/OYO+gYjjoUDZk72RyOOtUmmTvHII46nUCZO7MkjjpsWpk7nSiOOkN0mTuELI46IY6ZO2kwjjoGqJk7TDSOOvPBmTssOI46"
    "59uZOwo8jjri9Zk75T+OOuUPmju+Q4467ymaO5VHjjoBRJo7aUuOOhlemjs7T446OniaOwpTjjphkpo71laOOpGsmjuhWo46"
    "x8aaO2lejjoF4Zo7LmKOOkv7mjvxZY46lxWbO7FpjjrsL5s7b22OOkhKmzsqcY46q2SbO+N0jjoWf5s7mniOOoiZmztOfI46"
    "ArSbO/9/jjqDzps7roOOOgzpmztah446nAOcOwSLjjo0Hpw7q46OOtQ4nDtPko46e1OcO/GVjjopbpw7kZmOOt+InDsunY46"
    "naOcO8igjjpjvpw7YKSOOi/ZnDv1p446BPScO4erjjrgDp07F6+OOsQpnTulso46r0SdOy+2jjqjX507t7mOOp16nTs8vY46"
    "oJWdO7/AjjqqsJ07P8SOOrzLnTu9x4461eadOzfLjjr2AZ47r86OOh8dnjsl0o46UDieO5fVjjqIU547B9mOOshunjt13I46"
    "EIqeO9/fjjpgpZ47R+OOOrfAnjus5o46FtyeOw7qjjp99547bu2OOuwSnzvL8I46Yi6fOyX0jjrhSZ87fPeOOmdlnzvR+o46"
    "9YCfOyP+jjqLnJ87cgGPOii4nzu+BI86ztOfOwcIjzp77587TguPOjELoDuSDo867iagO9MRjzqzQqA7ERWPOoBeoDtMGI86"
    "VXqgO4UbjzoxlqA7uh6POhayoDvtIY86A86gOx0ljzr36aA7SiiPOvQFoTt0K486+CGhO5sujzoFPqE7vzGPOhpaoTvhNI86"
    "cfu9OxjQrrqVF747/cyuusIzvjvkya669k++O87GrrozbL47vMOuuniIvjuswK66xKS+O5+9rroZwb47lbquunbdvjuOt666"
    "2/m+O4q0rrpIFr87ibGuur0yvzuLrq66Ok+/O5CrrrrAa787maiuuk2Ivzukpa6646S/O7KirrqBwb87w5+uuifevzvXnK66"
    "1fq/O++ZrrqLF8A7CZeuuko0wDsmlK66EFHAO0eRrrrfbcA7ao6uuraKwDuRi666lqfAO7uIrrp9xMA76IWuum3hwDsYg666"
    "Zf7AO0uArrpmG8E7gX2uum84wTu7eq66gFXBO/d3rrqZcsE7N3WuurqPwTt6cq665KzBO8BvrroXysE7CW2uulHnwTtWaq66"
    "lATCO6ZnrrrfIcI7+WSuujM/wjtPYq66j1zCO6hfrrr0ecI7BV2uumCXwjtlWq661rTCO8hXrrpT0sI7LlWuutnvwjuYUq66"
    "aA3DOwVQrrr/KsM7dU2uup5IwzvoSq66RmbDO19Irrr3g8M72UWuurChwztXQ666cb/DO9hArro73cM7XD6uug77wzvjO666"
    "6RjEO245rrrMNsQ7/DauurhUxDuONK66rXLEOyMyrrqqkMQ7uy+uurCuxDtXLa66vszEO/cqrrrV6sQ7mSiuuvUIxTs/Jq66"
    "HSfFO+kjrrpORcU7liGuuohjxTtGH666yoHFO/ocrroVoMU7shquumi+xTtsGK66xdzFOysWrroq+8U77ROuupcZxjuyEa66"
    "DjjGO3sPrrqNVsY7Rw2uuhV1xjsXC666pZPGO+sIrro/ssY7wgauuuHQxjudBK66jO/GO3sCrrpADsc7XQCuuvwsxztC/q26"
    "wkvHOyz8rbqQasc7GPqtumeJxzsJ+K26R6jHO/31rbqdBZo7qmBtupEfmjtSXG26jjmaO/FXbbqTU5o7h1NtuqBtmjsST226"
    "t4eaO5RKbbrVoZo7C0Ztuv27mjt5QW26LdaaO948bbpl8Jo7ODhtuqYKmzuIM2268CSbO88ubbpDP5s7Cyptup5Zmzs9JW26"
    "AXSbO2Ygbbpujps7hBttuuOomzuYFm26YcObO6IRbbro3Zs7ogxtunf4mzuXB226DxOcO4MCbbqwLZw7ZP1sulpInDs6+Gy6"
    "DGOcOwfzbLrIfZw7ye1suoyYnDuB6Gy6WbOcOy7jbLouzpw70d1sug3pnDtp2Gy69QOdO/fSbLrlHp07es1sut45nTvzx2y6"
    "4FSdO2HCbLrsb507xLxsugCLnTsdt2y6HaadO2uxbLpDwZ07rqtsunLcnTvnpWy6qfedOxSgbLrqEp47N5psujQunjtPlGy6"
    "h0meO1yObLrjZJ47XohsukiAnjtWgmy6t5ueO0J8bLout547I3Zsuq7Snjv5b2y6OO6eO8NpbLrKCZ87g2NsumYlnzs4XWy6"
    "C0GfO+FWbLq5XJ87f1BsunB4nzsSSmy6MJSfO5lDbLr6r587FT1sus3LnzuGNmy6qeefO+svbLqOA6A7RSlsun0foDuTImy6"
    "dDugO9YbbLp1V6A7DRVsuoBzoDs5Dmy6lI+gO1kHbLqxq6A7bQBsutfHoDt1+Wu6B+SgO3Lya7pAAKE7Y+truoIcoTtI5Gu6"
    "zjihOyHda7okVaE779VruoJxoTuwzmu6642hO2bHa7pcqqE7D8BrutfGoTusuGu6XOOhOz6xa7rq/6E7w6lruoIcojs8omu6"
    "IzmiO6maa7rOVaI7CZNruoJyojtei2u6QI+iO6WDa7oHrKI74XtrutjIojsQdGu6s+WiOzNsa7qXAqM7SWRruoUfoztTXGu6"
    "fTyjO1BUa7p+WaM7QUxruol2ozslRGu6nZOjO/w7a7q8sKM7xzNruuTNozuFK2u6FuujOzYja7pRCKQ72hprupclpDtxEmu6"
    "5kKkO/wJa7o/YKQ7eQFruqJ9pDvp+Gq6DpukO03warqFuKQ7o+dqugXWpDvs3mq6kPOkOyjWarokEaU7V81qusIupTt5xGq6"
    "akylO427arocaqU7lLJqutiHpTuOqWq6nqWlO3qgarpuw6U7WZdqukjhpTsqjmq6LP+lO+2EaroaHaY7o3tquhI7pjtMcmq6"
    "FFmmO+Zoarogd6Y7c19qujeVpjvzVWq6V7OmO2RMarqC0aY7yEJqurbvpjsdOWq69Q2nO2Uvaro+LKc7nyVqupJKpzvKG2q6"
    "72inO+gRarpXh6c79wdqusmlpzv4/Wm6+j+/O0Ih4jqBXr87TybiOhJ9vztkK+I6rZu/O4Aw4jpSur87ozXiOgLZvzvOOuI6"
    "vPe/O/8/4jqBFsA7OEXiOlA1wDt4SuI6KVTAO8BP4joNc8A7D1XiOvuRwDtlWuI69LDAO8Nf4jr3z8A7KGXiOgXvwDuUauI6"
    "HQ7BOwhw4jpALcE7hHXiOm1MwTsHe+I6pWvBO5GA4jrnisE7I4biOjSqwTu9i+I6jMnBO16R4jru6ME7B5fiOlsIwju4nOI6"
    "0yfCO3Ci4jpVR8I7MKjiOuJmwjv4reI6eobCO8ez4jocpsI7n7niOsnFwjt+v+I6geXCO2XF4jpEBcM7VMviOhIlwztL0eI6"
    "6kTDO0nX4jrNZMM7UN3iOruEwzte4+I6tKTDO3Xp4jq4xMM7lO/iOsfkwzu69eI64QTEO+n74joFJcQ7IALjOjVFxDtfCOM6"
    "b2XEO6YO4zq1hcQ79hTjOgamxDtNG+M6YcbEO60h4zrI5sQ7FSjjOjoHxTuGLuM6tyfFO/804zo/SMU7gDvjOtJoxTsJQuM6"
    "cInFO5tI4zoaqsU7NU/jOs7KxTvYVeM6juvFO4Rc4zpZDMY7OGPjOi8txjv0aeM6EU7GO7lw4zr+bsY7h3fjOvaPxjtdfuM6"
    "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"
    "CyWePFrYNjwfmr086VgvvBy7vTzJPi+8K9y9PKAkL7xO/b08cAovvIQevjw38C68zj++PPXVLrwrYb48q7suvJyCvjxZoS68"
    "IaS+PP6GLry6xb48m2wuvGbnvjwwUi68Jgm/PLw3Lrz6Kr88QB0uvONMvzy8Ai68326/PC/oLbzwkL88mc0tvBWzvzz8si28"
    "T9W/PFaYLbyd9788p30tvAAawDzxYi28dzzAPDFILbwDX8A8ai0tvKSBwDyaEi28WqTAPML3LLyl26c8LJVAPv/4pzyyxrA9"
    "axaoPCRjTDzsM6g8ShQbvYFRqDx6yoy9Km+oPMXKq73mjKg8dW2xvbiqqDz046O9ncioPJqxiL2X5qg8NGBJvaUEqTwmT/C8"
    "yCKpPLB/E7z/QKk8TxcqPEtfqTwAiOI8rX2pPNOyKz0jnKk8+jhXPa66qTwYPnI9TtmpPOoYfD0E+Kk8Kfx0Pc8Wqjx35V09"
    "sDWqPAOFOD2mVKo81SkHPbJzqjzITpk81JKqPHq/QzsLsqo8oylavFnRqjz02O+8vfCqPAgaNL03EKs8pVFnvccvqzyPA4e9"
    "bk+rPJ+Dkr0rb6s8yNCUvf+Oqzyd6Yy96q6rPBpzdL3szqs8mmI5vQXvqzxvgdO8NA+sPFbcAbt8L6w8E5TOPNpPrDxZVV49"
    "UHCsPMReqj3ekKw8X+rfPYOxrDyMkAQ+QNKsPLvtDj4V86w8rCgKPgIUrTyp8eA9CDWtPNVubD0lVq08AQfuvFx3rTxYFiO+"
    "qpitPPWOr764F2Q8v3oCvac+ZDx/jAK9t2VkPEueAr3ojGQ8JLACvTq0ZDwKwgK9rdtkPP3TAr1CA2U8/OUCvfkqZTwI+AK9"
    "0VJlPCIKA73MemU8SBwDvemiZTx7LgO9KctlPLxAA72M82U8CVMDvREcZjxkZQO9ukRmPMx3A72GbWY8QooDvXaWZjzFnAO9"
    "Wl+0POYNErz2c7Q8XlgSvKSItDwNoxK8ZZ20PPPtErw3srQ8EDkTvBzHtDxlhBO8FNy0PPHPE7wf8bQ8tRsUvDwGtTyyZxS8"
    "bBu1POezFLyvMLU8VAAVvAZGtTz7TBW8cFu1PNuZFbztcLU89eYVvH6GtTxJNBa8Ipy1PNeBFrzbsbU8oM8WvKfHtTyjHRe8"
    "h921POFrF7x787U8W7oXvIQJtjwRCRi8Lz8HPYZhpz1ISgc9oFenPWtVBz2zTac9mWAHPb5Dpz3Rawc9wTmnPRR3Bz29L6c9"
    "YYIHPbElpz25jQc9nRunPRyZBz2CEac9sHNrPOJwCD5cfWs8FnUIPhKHazxOeQg+05BrPIl9CD6fmms8yIEIPnekazwLhgg+"
    "Wa5rPFGKCD5HuGs8m44IPkDCazzqkgg+RMxrPDyXCD5U1ms8kZsIPnDgazzrnwg+l+prPEmkCD7K9Gs8qqgIPgr/azwQrQg+"
    "VQlsPHmxCD6sE2w857UIPg8ebDxZugg+fihsPM6+CD76Mmw8SMMIPk7MzzwxvcG9mNHPPC60wb3p1s88IavBvUDczzwNosG9"
    "nuHPPPCYwb0C5888yo/BvW3szzydhsG93vHPPGZ9wb1W9888J3TBvdX8zzzfasG9WgLQPI9hwb3mB9A8NljBvXkN0DzUTsG9"
    "ExPQPGlFwb20GNA89TvBvVwe0Dx4MsG9CyTQPPMowb2609A8LvLFvXjZ0DyW6MW9Pd/QPPXexb0J5dA8StXFvdzq0DyXy8W9"
    "t/DQPNnBxb2a9tA8E7jFvYT80DxDrsW9dgLRPGmkxb1wCNE8hprFvXEO0TyZkMW9pXKaPIx+JLzqkpo8eYskvGSzmjxfmCS8"
    "EtSaPD+lJLz29Jo8F7IkvA8WmzznviS8YDebPK7LJLznWJs8bNgkvKZ6mzwf5SS8npybPMfxJLzPvps8ZP4kvDrhmzzyCiW8"
    "3wOcPHMXJbzAJpw85SMlvNxJnDxGMCW8Nm2cPJU8JbzMkJw80EglvKG0nDz3VCW8tdicPAZhJbwK/Zw8/WwlvJ4hnTzZeCW8"
    "dUadPJeEJbyPa508NpAlvOyQnTyymyW8jbadPAinJbx13J08NbIlvKQCnjw1vSW8GimePATIJbzaT548ntIlvOV2njz+3CW8"
    "O56ePB3nJbzfxZ489/AlvNLtnjyE+iW8FRafPL4DJryqPp88nQwmvJRnnzwWFSa805CfPCIdJrxrup88tSQmvFzknzzDKya8"
    "p+MAPTOXST3++AA9vJVJPYUOAT1xlEk9PyQBPVaTST0tOgE9cZJJPVBQAT3GkUk9q2YBPVyRST0/fQE9OZFJPeinLzRMgKY8"
    "TwY0NJ+QpjyYcTg0rKGmPCXqPDSEs6Y8XXBBNDvGpjyxBEY06NmmPJinSjSj7qY8k1lPNIcEpzwuG1Q0sBunPP3sWDRCNKc8"
    "oc9dNF9OpzzJw2I0MGqnPDHKZzThh6c8peNsNKSnpzwCEXI0r8mnPDlTdzQ+7qc8Tqt8NJQVqDwvDYE0+T+oPNDQgzTAbag8"
    "MqGGNEKfqDwNf4k04dSoPCprjDQND6k8ZmaPND1OqTyxcZI0+ZKpPBKOlTTU3ak8qLyYNHEvqjyx/ps0hoiqPIdVnzTa6ao8"
    "pMKiNEtUqzysR6Y0zMirPHHwcD3VQRi9c/BwPeH7F7118HA9JK8XvXfwcD3qWhe9efBwPWf+Fr178HA9v5gWvX7wcD38KBa9"
    "gPBwPRCuFb2C8HA9zyYVvYXwcD3ukRS9h/BwPQHuE72K8HA9dDkTvY3wcD2KchK9kPBwPViXEb2T8HA9vqUQvZbwcD1lmw+9"
    "mvBwPbd1Dr3uFng8AsFivZsVeDziv2K9hxV4PJK/Yr1xFXg8Or9ivVgVeDzYvmK9PRV4PGy+Yr0fFXg89b1ivf4UeDxyvWK9"
    "2RR4POC8Yr2xFHg8QLxivYQUeDyPu2K9y1bgPE2eBr2wVuA8dZ0GvZJW4DyGnAa9cVbgPH6bBr1NVuA8W5oGvSRW4DwamQa9"
    "+FXgPLeXBr3HVeA8LpYGvZFV4Dx+lAa9VVXgPKCSBr0TVeA8kJAGvcpU4DxJjga9elTgPMaLBr0hVOA8AYkGvb9T4DzxhQa9"
    "U1PgPJCCBr3cUuA81n4GvVhS4Dy4ega9x1HgPC52Br0nUeA8K3EGvXZQ4Dyjawa9wVq9PP54pDzqWb08dYakPP1YvTxQlaQ8"
    "91e9PLSlpDzWVr08ybekPJdVvTy8y6Q8G/VZPGv9BT4T8lk8dAAGPrzuWTzNAwY+DetZPH4HBj795lk8kAsGPoLiWTwOEAY+"
    "kN1ZPAEVBj4e2Fk8dxoGPhvSWTx9IAY+IVaLPIecFD5ROIs8qRQUPnMXizzdfhM+N/OKPLfZEj5Fy4o8piMSPjyfijzxWhE+"
    "sm6KPLJ9ED4wOYo80YkPPjb+iTwDfQ4+Nr2JPL1UDT6SdYk8Ng4MPp8miTxbpgo+4dLOPA7YGj4Ac848DSMZPloJzjyCQRc+"
    "8ZTNPOkuFT6tFM08TOYSPl2HzDwyYhA+revLPJacDT4qQMs81o4KPmzO0Tz+11g9Uv7QPBQHSj0SGdA8cLM5PZAczzzJtyc9"
    "dAbOPFjpEz0p1Mw8GDL8PNeCyzw1Jsw8Xg/KPBM9lzxw8MI8++oovAIuwTzKnZS8DD6/PF1C27z6G708DoUUvcPCujxQVj+9"
    "3Sy4POx5br0xVLU8dy+RvQwysjy8v629Eb+uPO8wzb0k86o86szvvV7Fpjxz8gq+8iuiPBjpH74aHJ08f/s2vv6JlzykX1C+"
    "mWiRPOJQbL6ZqYo8PoiFvkE9gzwXc5a+iCR2PGsQqb43K2Q8lIu9vr9kUDw3FNS+"
)
_AVERAGES = [  # (name, ensemble weight, coefficient per block per running sum)
    ("avg_epoch7_half", 0.021397176990490183, [[1.2444202212321597e-05, 8.655067103605085e-05], [-3.5105318033640164e-05, 0.00012990580280675834], [8.99649483952516e-05, 3.8347872545104736e-05], [-0.000318269000004557, 0.0003361174549381211], [-0.0021060418751409175, -0.018060674578289394], [-0.009858000056948984, -0.0834073915808828], [1.023855518373815, -3.8452745050584918], [-0.010249538088423714, 0.08418878604266662], [-0.001931655251570335, 0.0010858207856688415], [0.000414244244830492, 0.0017447078752834439], [0.00011330145508209265, -0.0001705588164338998], [1.3278018619024358e-05, 2.145123122877058e-06], [-1.3181878110137745e-07, -4.856485966676479e-07], [-1.5076008203572978e-08, -2.826302947196937e-09], [2.2782707301337576e-09, -8.636076770150104e-09], [2.9592472656591184e-09, 1.2320185377513233e-09], [4.108206702780219e-12, -8.969647884117988e-12], [2.879644541952271e-13, 7.782858477593308e-13], [1.515129840625289e-14, -3.919758368122038e-14], [2.512223955285655e-14, 5.976563923786644e-15]]),
    ("avg_epoch7_full", 0.01947330670774621, [[-3.6928248824009344e-06, -2.568396640525994e-05], [1.0417525342002488e-05, -3.854962919873608e-05], [-2.669715535831988e-05, -1.1379755436979396e-05], [9.444652712087902e-05, -9.974306741204728e-05], [0.0006249692589223729, 0.005359516607978418], [0.002925367753925301, 0.024751196222562454], [0.9968724787477357, 0.5296737693022083], [-0.0004378665989952961, 0.0035965969490466794], [-8.252150566540434e-05, 4.6386934756693794e-05], [1.76967700353407e-05, 7.453499826953842e-05], [4.840308128204362e-06, -7.28637806070244e-06], [5.672451897173821e-07, 9.164099178079641e-08], [-5.631383130610157e-09, -2.074719987104349e-08], [-6.440515184024806e-10, -1.2074116968634713e-10], [9.736512933728483e-11, -3.6890940639275523e-10], [1.2638178206575658e-10, 5.261523490597026e-11], [1.6546731863939155e-13, -3.9896186506990705e-13], [1.761134075350956e-14, 5.422741287408447e-14], [2.350233685991393e-15, -4.823905397255176e-15], [8.271828910544589e-16, 7.063535802303397e-16]]),
    ("ema61_at_epoch7", 0.012943587647187418, [[-2.9126697817239063e-05, -0.00021106675151008614], [9.900732412627906e-05, -0.0003080144825883477], [-0.00011994671396259282, -0.0001491280634563521], [0.00180616565953123, -0.0005681462748131177], [0.014190620509143135, 0.051969473489957126], [0.08505994701751188, 0.16562375170002577], [0.8751490067358088, -1.74438373021202], [0.020995068298045837, -0.17245160685739092], [0.003956786499553261, -0.00222418623724847], [-0.0008485344546943567, -0.003573845053787726], [-0.00023208575519871147, 0.00034937125633811795], [-2.7198582514864706e-05, -4.394052299610708e-06], [2.7001647450108966e-07, 9.947985311180072e-07], [3.0881580022239125e-08, 5.789376377071855e-09], [-4.66660915206364e-09, 1.7690199654137332e-08], [-6.061884347305747e-09, -2.523737543637539e-09], [-8.441605262091393e-12, 1.833655811698512e-11], [-5.710015604053367e-13, -1.5447157829543522e-12], [-3.3015565514137695e-14, 7.982585543319301e-14], [-5.067629630277518e-14, -1.1344946688398139e-14]]),
    ("ema121_at_epoch7", 0.009201432983940439, [[0.0009032766366686341, 0.000749262288547714], [0.002675782785079953, 0.000683399611668594], [0.00783560035991315, -0.0013987963310409725], [0.025842333823191392, -0.0008709734723317462], [0.07981778137248886, 0.1128424553939849], [0.20908275969654863, 0.1792454864986254], [0.6673506152190003, -0.49067106681121403], [0.005716111893143225, -0.046951630114047856], [0.0010772736743490335, -0.0006055563727063087], [-0.0002310217723255461, -0.0009730141348366911], [-6.318760795978267e-05, 9.511972838435989e-05], [-7.405079109815111e-06, -1.1963235534498123e-06], [7.351461051283183e-08, 2.708435880764254e-07], [8.407813472144903e-09, 1.5762146840604692e-09], [-1.2704887871255581e-09, 4.816357527122744e-09], [-1.6504495554585805e-09, -6.871302411234629e-10], [-2.301153815025539e-12, 4.9879151767220936e-12], [-1.544745498181498e-13, -4.1379995922201343e-13], [-7.049132412301964e-15, 2.13463607308339e-14], [-1.4460256771438086e-14, -3.3119031687740516e-15]]),
    ("avg_epoch8_full", 0.037554068246038186, [[-3.363286882619699e-07, -2.339199791451252e-06], [9.487892523757707e-07, -3.510956095730575e-06], [-2.431477034985793e-06, -1.0364255468437728e-06], [8.601836337436365e-06, -9.084225408311089e-06], [5.6919861934604035e-05, 0.0004881247212460879], [0.0002664315505923445, 0.002254246351868184], [0.005411808882147607, -0.03867704034471748], [0.9971399848808058, -0.4578158952562608], [-0.004002155759305114, 0.00224968917585065], [0.0008582639106696179, 0.003614823434748625], [0.00023474689414203567, -0.0003533772130864295], [2.7510446664184757e-05, 4.444435341269715e-06], [-2.7311254825435353e-07, -1.006205091956911e-06], [-3.1235669320089906e-08, -5.855757045710062e-09], [4.720198976928124e-09, -1.78929841115977e-08], [6.131308203627368e-09, 2.5526381644812548e-09], [8.519750491976697e-12, -1.8577554960645534e-11], [5.893697739205563e-13, 1.600568285521405e-12], [3.279486942279926e-14, -8.612198635050457e-14], [5.4757442690750714e-14, 1.254427857905592e-14]]),
    ("ema61_at_epoch8", 0.024028031930144668, [[-2.19621473879424e-06, -1.6253878158914437e-05], [8.13796858571117e-06, -2.338307019308855e-05], [-5.424714135031062e-06, -1.3615881260007325e-05], [0.0001785743564824978, -3.410993528479484e-05], [0.001439898144164842, 0.0043059055381205295], [0.008889445568273311, 0.011306225537261875], [0.08222278363071166, -0.06742155209452362], [0.8979508477492173, 1.829698212458707], [0.012939886537063726, -0.007273760550532442], [-0.002774963867186734, -0.011687552386863975], [-0.0007589904936484882, 0.0011425494950686175], [-8.89475771444982e-05, -1.4369877757065981e-05], [8.830354241573316e-07, 3.2532916193213847e-06], [1.0099208876666063e-07, 1.8933008927191607e-08], [-1.526126406521401e-08, 5.785226148068656e-08], [-1.982414951110257e-08, -8.253361513392791e-09], [-2.7576096968877813e-11, 6.000349205561784e-11], [-1.8842658638744967e-12, -5.106177543860842e-12], [-1.1099125942734371e-13, 2.649734112354158e-13], [-1.6933876483248172e-13, -3.687521950742467e-14]]),
    ("avg_last_cycle", 0.10704502766084374, [[3.21937215658232e-16, 4.5341334853343795e-15], [-1.1311849200990881e-15, 9.902650802096817e-16], [2.7066096890352693e-15, -1.9677220405196325e-15], [-9.556467123680973e-15, 1.4879325141845847e-14], [-2.685650394199666e-14, -3.428126402561029e-13], [1.2766561100488964e-14, -4.919927500637009e-13], [-1.4734604867705632e-13, 1.2822488011727232e-12], [-2.4980567775506434e-12, -1.94700330302945e-11], [1.737741322943323e-11, 1.2485933215161928e-11], [2.189691731787858e-10, -5.614473257906034e-10], [1.867749650524762e-08, 1.97606389867945e-08], [-1.4769343782275425e-07, -7.752339100498822e-09], [-9.97054448456425e-07, 3.3546476606039067e-06], [5.848836616530225e-06, -1.0115442867360256e-06], [0.0001692157288941529, 0.0001272463007461312], [0.005605977121018024, -0.0010224914281275863], [0.24721725637596362, -0.05187179643344265], [0.25394588176633065, 0.2552893264386904], [0.2336951173095763, 0.000794376930926856], [0.25936182869831265, 0.001468522811068573]]),
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
