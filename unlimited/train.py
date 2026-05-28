"""
Chain Distillation Snapshot Ensemble training with 1.8B GPT model on 100M tokens.

Key features:
- Chain distillation: each cycle is distilled from the last snapshot as teacher
- Per-node HP diversity: nodes can have slightly different hyperparameters
- Gradient ensemble selection: learn softmax mixture weights over snapshots on a
  held-out fitness slice; pick top-K by weight at end of training
- Weight perturbation: inject noise at cycle boundaries for ensemble diversity
- Dupe layers: replay decoder layers for increased test-time compute (activated
  after --dupe-after-n-models initial models are trained without dupe)
- Multi-node DDP: each node trains one model using intra-node data parallelism;
  inter-node communication only at the final selection step

Multi-node launch (5 nodes, 8 GPUs each — primary usage):
    torchrun --nnodes=5 --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=<MASTER_IP>:29500 \
        train.py --num-models 480

    Training data and checkpoint paths must be on a
    shared filesystem visible to both nodes.

    This trains 5 independent models (one per node, 96 cycles each), with 8 GPUs
    per node collaborating via DDP for faster training. Each model gets a unique
    initialization and data shuffle. The published results use this configuration.

NOTE on single-node usage:
    torchrun --standalone --nproc_per_node=8 train.py

    With 1 node, all 8 GPUs collaborate via DDP on a SINGLE model that runs for
    all 480 cycles sequentially. This is NOT the same as the published results
    (5 independent models x 96 cycles). To reproduce the original 5-independent-
    models behavior on a single node, code changes are required to disable DDP
    and treat each GPU as an independent training stream.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import json
import argparse
from contextlib import nullcontext as nullctx
from types import SimpleNamespace
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import tiktoken

from gradient_select import optimize_alpha as _optimize_alpha_fn

# =============================================================================
# Per-node hyperparameter overrides for diversity (keyed by node_rank 0..N-1)
# =============================================================================

# NODE_OVERRIDES = {
#     6: {"distill_alpha": 0.40},
#     7: {"distill_temperature": 1.3}
# }

## Uncomment to disable per-node HP diversity
NODE_OVERRIDES = {}

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Chain Distillation Snapshot Ensemble (1.8B model, 100M tokens)")
parser.add_argument("--device-batch-size", type=int, default=2)
parser.add_argument("--epochs-per-cycle", type=float, default=2.0,
                    help="Training epochs per snapshot cycle (fractional OK)")
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--scalar-lr", type=float, default=0.1)
parser.add_argument("--matrix-lr", type=float, default=0.04)
parser.add_argument("--weight-decay", type=float, default=1.3)
parser.add_argument("--total-batch-size", type=int, default=524288)
parser.add_argument("--save-result", type=str, default="")
parser.add_argument("--n_layer", type=int, default=30)
parser.add_argument("--n_head", type=int, default=16)
parser.add_argument("--n_embd", type=int, default=2048)
parser.add_argument("--lr_multiplier", type=float, default=0.25)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--embedding-lr", type=float, default=0.15)
parser.add_argument("--unembedding-lr", type=float, default=0.002)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--num-models", type=int, default=480,
                    help="Total ensemble members (must be divisible by num nodes)")
parser.add_argument("--gpus-per-node", type=int, default=8,
                    help="GPUs per node for intra-node DDP")
parser.add_argument("--checkpoint-base", type=str, default="checkpoints")
parser.add_argument("--resume", type=str, default=None, help="Run ID to resume from")
parser.add_argument("--label-smoothing", type=float, default=0.0)
parser.add_argument("--lr-floor", type=float, default=0.08,
                    help="Minimum LR as fraction of peak in cyclic schedule")
parser.add_argument("--wd-schedule", type=str, default="inverse_lr",
                    choices=["flat", "inverse_lr"],
                    help="Per-cycle weight decay schedule. 'inverse_lr' anti-correlates WD "
                         "with LR (low WD when LR high, high WD when LR low) by mapping the "
                         "current LR multiplier through (1 - lr_mult)/(1 - lr_floor). 'flat' "
                         "keeps WD constant at the base value.")
parser.add_argument("--wd-floor", type=float, default=0.7,
                    help="Minimum WD as fraction of base WD when --wd-schedule=inverse_lr "
                         "(default 0.8 → with --weight-decay=1.2, WD swings 0.96 → 1.2 across "
                         "each cycle, peaking when LR hits its floor).")
parser.add_argument("--bs-schedule", type=str, default="triangular",
                    choices=["flat", "triangular"],
                    help="Across-cycles batch size schedule. 'triangular' uses bs_floor for early+late "
                         "cycles and --total-batch-size (peak) for the middle ~50%% of cycles "
                         "(block-triangular). BS is constant within each cycle.")
parser.add_argument("--bs-floor", type=int, default=393216,
                    help="Floor batch size (tokens) for --bs-schedule triangular. Must be a multiple of "
                         "device_batch_size * MAX_SEQ_LEN * gpus_per_node.")
parser.add_argument("--adam-betas", type=float, nargs=2, default=[0.8, 0.95])
parser.add_argument("--muon-momentum", type=float, default=0.95)
parser.add_argument("--ns-steps", type=int, default=5)
parser.add_argument("--k-sweep", type=int, nargs="*", default=[8, 16, 32, 64, 128, 256],
                    help="K values to sweep for gradient ensemble selection (at end of training).")
parser.add_argument("--pgt-steps", type=int, default=16,
                    help="Number of fitness batches for p(gt) caching per snapshot. "
                         "Clamped to available steps; 16 = fitness_seqs/B_cache for the "
                         "default 132K-token fitness budget at B_cache=4.")

# Chain distillation
parser.add_argument("--distill-alpha", type=float, default=0.45,
                    help="Weight of KL distillation loss (0=pure CE, 1=pure KL)")
parser.add_argument("--distill-temperature", type=float, default=1.2)
parser.add_argument("--distill-after-cycles", type=int, default=8,
                    help="Start distillation after this many cycles")

# Weight perturbation
parser.add_argument("--perturb-scale", type=float, default=0.25,
                    help="Max scale of weight perturbation relative to each param's std (0=disabled)")
parser.add_argument("--perturb-scale-min", type=float, default=0.05,
                    help="Min scale of weight perturbation (cosine decay from --perturb-scale to this)")

# Dupe layers
parser.add_argument("--dupe-layers-start", type=int, default=15,
                    help="First decoder layer to duplicate (inclusive)")
parser.add_argument("--dupe-layers-end", type=int, default=25,
                    help="Last decoder layer to duplicate (exclusive)")
parser.add_argument("--dupe-after-n-models", type=int, default=0,
                    help="Number of models to train without dupe layers before "
                         "enabling dupe for all remaining models (0=always on).")

# IHA (Interleaved Head Attention)
parser.add_argument("--iha", action="store_true", default=True,
                    help="Enable Interleaved Head Attention (cross-head Q/K/V mixing)")
parser.add_argument("--no-iha", action="store_false", dest="iha",
                    help="Disable IHA cross-head mixing")
parser.add_argument("--iha-lr", type=float, default=0.02,
                    help="LR for IHA mixing matrices (when --iha is enabled)")

# MTP (Multi-token prediction)
parser.add_argument("--mtp-weight", type=float, default=0.3,
                    help="Multi-token prediction aux loss weight (0=disabled)")

# Document shuffling
parser.add_argument("--no-doc-shuffle", action="store_true",
                    help="Disable per-epoch document reshuffling (default: enabled)")

# Fitness set (held-out training docs for gradient ensemble selection)
parser.add_argument("--fitness-tokens", type=int, default=132_000,
                    help="Approximate token budget for the held-out fitness set "
                         "(carved from the front of the train file at whole-document "
                         "granularity; same docs across every model and every epoch).")


# Gradient ensemble selection
parser.add_argument("--grad-opt-steps", type=int, default=300,
                    help="AdamW steps for alpha optimization.")
parser.add_argument("--grad-lr", type=float, default=0.5,
                    help="AdamW learning rate on alpha logits.")
parser.add_argument("--grad-weighted-eval", action=argparse.BooleanOptionalAction, default=True,
                    help="Ensemble-eval the top-K with the renormalized learned weights "
                         "(off = uniform average across the top-K).")

args = parser.parse_args()

if args.epochs_per_cycle <= 0:
    parser.error("--epochs-per-cycle must be > 0")
if args.output_json and not args.save_result:
    args.save_result = args.output_json

# =============================================================================
# Hyperparameters
# =============================================================================

DEPTH = args.n_layer
N_EMBD = args.n_embd
N_HEAD = args.n_head
HEAD_DIM = N_EMBD // N_HEAD
MAX_SEQ_LEN = 2048
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = args.total_batch_size
DATA_DIR = "fineweb_data"

# =============================================================================
# Utilities
# =============================================================================

def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return 0, 0, 1

def print0(s="", **kwargs):
    if int(os.environ.get('RANK', 0)) == 0:
        print(s, **kwargs)

class DummyWandb:
    def __init__(self): self.summary = {}
    def log(self, *a, **kw): pass
    def finish(self): pass
    def log_code(self, *a, **kw): pass

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
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)

flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)

# =============================================================================
# GPT Model (1.8B architecture from train.py: SwiGLU, attn gate, U-Net, VE projs)
# =============================================================================

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 50257
    n_layer: int = 30
    n_head: int = 16
    n_kv_head: int = 16
    n_embd: int = 2048
    window_pattern: str = "SSSL"
    dropout: float = 0.0
    use_iha: bool = False
    iha_mix_v: bool = True
    mtp_weight: float = 0.0  # >0 enables multi-token prediction aux loss

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
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
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)
        # IHA: cross-head Q/K/V mixing matrices, fused into projection weights at fwd time.
        self.use_iha = config.use_iha
        if self.use_iha:
            self.q_mix = nn.Parameter(torch.zeros(self.n_head, self.n_head))
            self.k_mix = nn.Parameter(torch.zeros(self.n_kv_head, self.n_kv_head))
            self.iha_mix_v = config.iha_mix_v
            if self.iha_mix_v:
                self.v_mix = nn.Parameter(torch.zeros(self.n_kv_head, self.n_kv_head))

    def _fuse_mix(self, weight, mix, H):
        d = self.head_dim
        return (mix @ weight.view(H, d, -1).flatten(1)).view_as(weight)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        if self.use_iha:
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
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
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

    def forward(self, x, ve, cos_sin, window_size):
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
        self.ve_projs = nn.ModuleDict({
            str(i): nn.Linear(config.n_embd, kv_dim, bias=False)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # U-Net skip connections: encoder layer i -> decoder layer (n_layer - 1 - i)
        self.encoder_layers = config.n_layer // 2
        self.skip_weights = nn.Parameter(torch.ones(self.encoder_layers))
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._dupe_layers = None
        self.mtp_weight = config.mtp_weight
        if self.mtp_weight > 0:
            self.mtp_proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
            self.mtp_block = Block(config, config.n_layer)

    def set_dupe_layers(self, start, end):
        assert start >= self.encoder_layers, "dupe layers must be decoder-only"
        assert end <= self.config.n_layer
        self._dupe_layers = (start, end)
        print0(f"Dupe layers {start}-{end-1}: decoder layers repeated with skip connections")

    @torch.no_grad()
    def init_weights(self, convert_embed=True):
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
            # IHA: initialize mixing matrices to identity (baseline-equivalent at init)
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
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if convert_embed and self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
        self._dupe_layers = None

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
        sizes[-1] = (long_w, 0)
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def setup_optimizer(self, optim_config):
        matrix_lr = optim_config['matrix_lr']
        scalar_lr = optim_config['scalar_lr']
        embedding_lr = optim_config['embedding_lr']
        unembedding_lr = optim_config['unembedding_lr']
        weight_decay = optim_config['weight_decay']
        adam_betas = optim_config['adam_betas']
        muon_momentum = optim_config.get('muon_momentum', 0.95)
        ns_steps = optim_config.get('ns_steps', 5)
        iha_lr = optim_config.get('iha_lr', None)
        if iha_lr is None:
            iha_lr = scalar_lr

        # Separate IHA mixing params (small H×H matrices) from large matrix params
        iha_params = []
        iha_param_ids = set()
        all_blocks_for_iha = list(self.transformer.h)
        if self.mtp_weight > 0:
            all_blocks_for_iha = all_blocks_for_iha + [self.mtp_block]
        for block in all_blocks_for_iha:
            if getattr(block.attn, 'use_iha', False):
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
            mtp_params = [p for p in list(self.mtp_block.parameters()) + list(self.mtp_proj.parameters())
                          if id(p) not in iha_param_ids]
            matrix_params += mtp_params
        embed_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        skip_params = [self.skip_weights]

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr, betas=adam_betas, eps=1e-10, weight_decay=weight_decay),
            dict(kind='adamw', params=embed_params, lr=embedding_lr, betas=adam_betas, eps=1e-10, weight_decay=weight_decay),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=skip_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        if iha_params:
            param_groups.append(dict(kind='adamw', params=iha_params, lr=iha_lr,
                                     betas=adam_betas, eps=1e-10, weight_decay=0.0))
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=matrix_lr,
                                     momentum=muon_momentum, ns_steps=ns_steps, beta2=0.95,
                                     weight_decay=weight_decay))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
            group["initial_weight_decay"] = group.get("weight_decay", 0.0)
        return optimizer

    def _run_decoder_layers(self, x, x0, cos_sin, encoder_outputs, start, end):
        for i in range(start, end):
            j = self.config.n_layer - 1 - i
            if 0 <= j < self.encoder_layers:
                x = x + self.skip_weights[i - self.encoder_layers] * encoder_outputs[j]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
        return x

    def _forward_trunk(self, idx):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x

        # Encoder half
        encoder_outputs = []
        for i in range(self.encoder_layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
            encoder_outputs.append(x)

        # Decoder half (with optional dupe layer replay)
        dupe = self._dupe_layers
        if dupe is None:
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                        self.encoder_layers, self.config.n_layer)
        else:
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                        self.encoder_layers, dupe[1])
            for _ in range(4):
                x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                            dupe[0], dupe[1])
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                        dupe[1], self.config.n_layer)

        return norm(x)

    def _primary_logits(self, x):
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        return logits

    def _mtp_loss(self, x, targets):
        mtp_emb = norm(self.transformer.wte(targets[:, :-1].clamp(min=0)))
        combined = self.mtp_proj(torch.cat([x[:, :-1], mtp_emb], dim=-1))
        mT = combined.size(1)
        mtp_out = norm(self.mtp_block(combined, None, (self.cos[:, :mT], self.sin[:, :mT]), (-1, -1)))
        mtp_logits = self.lm_head(mtp_out)[..., :self.config.vocab_size].float()
        mtp_logits = 15 * torch.tanh(mtp_logits / 15)
        return F.cross_entropy(mtp_logits.view(-1, mtp_logits.size(-1)),
                               targets[:, 1:].reshape(-1), ignore_index=-1)

    def forward(self, idx, targets=None, loss_reduction='mean', label_smoothing=0.0,
                distill=False):
        """
        targets=None: returns primary logits.
        distill=True: returns (primary_logits, mtp_loss_tensor).
        Otherwise:
          loss_reduction='none'/'sum': returns lm_loss with that reduction (no MTP).
          loss_reduction='mean': returns total_loss with MTP folded in if enabled,
            plus a metrics dict. (Single-tensor return when MTP is off, to keep
            backward compat with callers that do `loss = compiled_model(x, y)`.)
        """
        x = self._forward_trunk(idx)
        logits = self._primary_logits(x)
        if targets is None:
            return logits
        if distill:
            if self.mtp_weight > 0:
                mtp_loss = self._mtp_loss(x, targets)
            else:
                mtp_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
            return logits, mtp_loss
        if loss_reduction != 'mean':
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction,
                                   label_smoothing=label_smoothing)
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                  ignore_index=-1, reduction='mean',
                                  label_smoothing=label_smoothing)
        if self.mtp_weight <= 0:
            return lm_loss
        mtp_loss = self._mtp_loss(x, targets)
        return lm_loss + self.mtp_weight * mtp_loss

    def forward_logits(self, idx):
        return self.forward(idx, targets=None)

# =============================================================================
# Optimizer: MuonAdamW (single-GPU, no DDP needed for per-rank training)
# =============================================================================

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
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
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

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None: continue
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, p.grad, state['exp_avg'], state['exp_avg_sq'],
                           self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                           self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params: return
        p = params[0]
        state = self.state[p]
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            s = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(s, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params, state["momentum_buffer"],
                       state["second_momentum_buffer"], self._muon_momentum_t, self._muon_lr_t,
                       self._muon_wd_t, self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw': self._step_adamw(group)
            elif group['kind'] == 'muon': self._step_muon(group)

# =============================================================================
# Data loading (flat tokens + doc_starts format from prepare_data_updated.py)
# =============================================================================

BOS_ID = 50256  # GPT-2 <|endoftext|>


def _load_new_format_file(filepath):
    """Parse a flat-format data file into (doc_tokens, default_shuffle_seed).

    doc_tokens is a list of 1-D long tensors, one per document (with BOS already
    baked in by prepare_data_updated.py).
    """
    data = torch.load(filepath, weights_only=True)
    all_tokens = data["tokens"].long()
    raw_doc_starts = data["doc_starts"].long()
    bos_id = int(data["bos_id"])
    assert bos_id == BOS_ID, f"data bos_id {bos_id} != expected {BOS_ID}"
    default_shuffle_seed = int(data["seq_shuffle_seed"])
    doc_ends = torch.cat([raw_doc_starts[1:], torch.tensor([all_tokens.numel()])])
    doc_tokens = [all_tokens[s:e] for s, e in zip(raw_doc_starts.tolist(), doc_ends.tolist())]
    return doc_tokens, default_shuffle_seed


def _carve_fitness_docs(doc_tokens, fitness_tokens):
    """Carve whole documents off the front of the doc list to form a held-out
    fitness set. Same selection across all models/epochs. Returns
    (fitness_docs, remaining_docs, fitness_token_count).

    fitness_tokens <= 0 disables the carve and returns ([], doc_tokens, 0).
    """
    if fitness_tokens <= 0:
        return [], list(doc_tokens), 0
    fitness_docs = []
    fitness_token_count = 0
    split_idx = len(doc_tokens)
    for i, doc in enumerate(doc_tokens):
        if fitness_token_count >= fitness_tokens:
            split_idx = i
            break
        fitness_docs.append(doc)
        fitness_token_count += int(doc.numel())
    if fitness_token_count < fitness_tokens:
        raise ValueError(
            f"Cannot carve out {fitness_tokens:,} fitness tokens — "
            f"file has only {fitness_token_count:,} tokens.")
    return fitness_docs, list(doc_tokens[split_idx:]), fitness_token_count


def _build_seqs_from_docs(doc_tokens, seq_size):
    """Concatenate docs and chunk into (N, seq_size) sequences. Tail tokens
    that don't fill a sequence are dropped."""
    tokens = torch.cat(doc_tokens) if len(doc_tokens) > 1 else doc_tokens[0]
    num_seqs = len(tokens) // seq_size
    return tokens[:num_seqs * seq_size].view(num_seqs, seq_size)


class DataLoader:
    """Training dataloader for the flat tokens+doc_starts format.

    Two modes:
      doc_shuffle=False: chunk-once (using the file's default sequence permutation
        offset by the per-model seed), then reshuffle step order each epoch.
        Sequence contents and chunk boundaries never change across epochs.
      doc_shuffle=True:  reshuffle documents each epoch, re-concat and re-chunk
        into sequences, and re-permute step order. Different document
        ordering — and therefore different chunk boundaries — every epoch.

    With dp_rank/dp_world_size, all GPUs in a node share the same shuffle but
    each takes non-overlapping interleaved steps (zero overlap within an
    optimizer step).

    If `seqs` is provided (a (N, T+1) tensor), data is taken directly from it
    and doc_shuffle is forced off (used by the fitness set, where we have no
    doc boundaries).
    """

    def __init__(self, filepath, B, T, device="cuda", seed=42,
                 dp_rank=0, dp_world_size=1, seqs=None,
                 doc_shuffle=False, doc_tokens=None, default_shuffle_seed=None):
        self.B = B
        self.T = T
        self.seq_size = T + 1
        self.device = device
        self.seed = int(seed)
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.pos = 0
        self.epoch = 1

        if seqs is not None:
            # Fixed pre-chunked sequences (no doc info available)
            self._fixed_seqs = seqs.long()
            self.doc_shuffle = False
            self.doc_tokens = None
            self.default_shuffle_seed = 0
        else:
            if doc_tokens is None:
                doc_tokens, default_shuffle_seed = _load_new_format_file(filepath)
            self.doc_tokens = list(doc_tokens)
            self.default_shuffle_seed = int(default_shuffle_seed if default_shuffle_seed is not None else 0)
            self._fixed_seqs = None
            self.doc_shuffle = bool(doc_shuffle)
        self._build_batches()

    def _shuffle_and_shard(self, all_seqs):
        seqs_per_step = self.B
        num_steps_total = len(all_seqs) // seqs_per_step
        # Make total steps divisible by dp_world_size so all GPUs get equal work
        num_steps_total = (num_steps_total // self.dp_world_size) * self.dp_world_size
        usable = num_steps_total * seqs_per_step
        all_seqs = all_seqs[:usable].view(num_steps_total, seqs_per_step, self.seq_size)
        self.num_steps_total = num_steps_total
        self.num_steps = num_steps_total // self.dp_world_size
        self.total_tokens = usable * self.T
        self.rank_data = all_seqs[self.dp_rank::self.dp_world_size].contiguous()

    def _build_batches(self):
        if self._fixed_seqs is not None:
            all_seqs = self._fixed_seqs
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            all_seqs = all_seqs[torch.randperm(len(all_seqs), generator=g)]
        elif self.doc_shuffle:
            # Reshuffle documents each epoch, then chunk + permute
            g = torch.Generator()
            g.manual_seed(self.seed * 1009 + self.epoch + 1000)
            doc_perm = torch.randperm(len(self.doc_tokens), generator=g)
            self.doc_tokens = [self.doc_tokens[i] for i in doc_perm.tolist()]
            all_seqs = _build_seqs_from_docs(self.doc_tokens, self.seq_size)
            g.manual_seed(self.seed * 1009 + self.epoch + 2000)
            all_seqs = all_seqs[torch.randperm(len(all_seqs), generator=g)]
        else:
            # Chunk once with default-perm + per-model seed; reshuffle step order each epoch
            if not hasattr(self, '_base_seqs'):
                base = _build_seqs_from_docs(self.doc_tokens, self.seq_size)
                g = torch.Generator()
                g.manual_seed((self.default_shuffle_seed + self.seed * 1009) & 0x7FFFFFFF)
                self._base_seqs = base[torch.randperm(len(base), generator=g)]
            g = torch.Generator()
            g.manual_seed(self.seed * 1009 + self.epoch)
            all_seqs = self._base_seqs[torch.randperm(len(self._base_seqs), generator=g)]
        self._shuffle_and_shard(all_seqs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= self.num_steps:
            self.pos = 0
            self.epoch += 1
            self._build_batches()
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch


class DDPValLoader:
    """Val loader: shards seqs across ranks. Document shuffling is intentionally
    NOT supported here — replay across ensemble eval chunks must be bit-identical."""

    def __init__(self, filepath, B, T, rank, world_size, device="cuda", seed=42,
                 seqs=None, doc_tokens=None, default_shuffle_seed=None):
        if seqs is not None:
            all_seqs = seqs.long()
            ds_seed = 0
        else:
            if doc_tokens is None:
                doc_tokens, default_shuffle_seed = _load_new_format_file(filepath)
            all_seqs = _build_seqs_from_docs(doc_tokens, T + 1)
            ds_seed = int(default_shuffle_seed if default_shuffle_seed is not None else 0)
            g0 = torch.Generator()
            g0.manual_seed((ds_seed + int(seed) * 1009) & 0x7FFFFFFF)
            all_seqs = all_seqs[torch.randperm(len(all_seqs), generator=g0)]
        seqs_per_step = B * world_size
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step
        self.all_seqs = all_seqs[:usable]
        self.B = B
        self.world_size = world_size
        self.rank = rank
        self.num_steps = num_steps
        self.device = device
        self.seed = int(seed)
        self.pos = 0
        self.epoch = 1
        self._shuffle_and_shard()

    def _shuffle_and_shard(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(self.all_seqs), generator=g)
        shuffled = self.all_seqs[perm]
        shaped = shuffled.view(self.num_steps, self.world_size, self.B, -1)
        self.rank_data = shaped[:, self.rank].contiguous()

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= self.num_steps:
            self.pos = 0
            self.epoch += 1
            self._shuffle_and_shard()
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch

# =============================================================================
# Evaluation helpers
# =============================================================================

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, device=None, process_group=None):
    if device is None:
        device = model.get_device()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=device)
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
    if process_group is not None:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM, group=process_group)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss


@torch.no_grad()
def cache_pgt_single_model(model, config, device, autocast_ctx, fitness_seqs, pgt_steps=2,
                           local_rank=0, gpus_per_node=1, intra_node_group=None):
    """Cache p(ground truth) for a single already-loaded model on the held-out fitness subset.
    When gpus_per_node > 1, shards work across GPUs and gathers results."""
    B_cache = 4
    T = config.sequence_len
    val_loader = DDPValLoader(None, B_cache, T, rank=local_rank, world_size=gpus_per_node,
                              device=device, seed=0, seqs=fitness_seqs)
    pgt_steps = min(pgt_steps, val_loader.num_steps)
    all_pgt = []
    model.eval()
    for _ in range(pgt_steps):
        x, y, _ = next(val_loader)
        with torch.inference_mode():
            with autocast_ctx:
                logits = model.forward_logits(x).float()
        flat_logits = logits.view(-1, logits.size(-1))
        flat_y = y.view(-1)
        log_denom = torch.logsumexp(flat_logits, dim=-1)
        logit_gt = flat_logits.gather(1, flat_y.unsqueeze(1)).squeeze(1)
        p = torch.exp(logit_gt - log_denom)
        all_pgt.append(p)
    local_pgt = torch.cat(all_pgt) if all_pgt else torch.zeros(0, device=device)
    if gpus_per_node > 1 and intra_node_group is not None:
        gathered = [torch.empty_like(local_pgt) for _ in range(gpus_per_node)]
        dist.all_gather(gathered, local_pgt, group=intra_node_group)
        return torch.cat(gathered).cpu()
    return local_pgt.cpu()

# =============================================================================
# Ensemble evaluation (DDP-aware, with dupe layer support)
# =============================================================================

@torch.no_grad()
def evaluate_ensemble_bpb(checkpoint_paths, config, token_bytes, device, autocast_ctx,
                          val_path, rank, world_size, dupe_layers=None, max_models_in_memory=64,
                          process_group=None, weights=None):
    """Ensemble val loss by mixing p(y_t) across checkpoints. Models sharded across nodes.

    weights: optional list of float, len == len(checkpoint_paths). If None, uses
    uniform 1/num_models (matches CD's uniform average). If given, expected to sum
    to 1 (the caller is responsible for renormalizing after a top-K selection)."""
    num_models = len(checkpoint_paths)
    my_idx = list(range(rank, num_models, world_size))
    my_paths = [checkpoint_paths[i] for i in my_idx]
    if weights is None:
        my_weights = [1.0 / num_models] * len(my_paths)
    else:
        assert len(weights) == num_models, \
            f"weights len {len(weights)} != checkpoint count {num_models}"
        my_weights = [float(weights[i]) for i in my_idx]
    my_count = len(my_paths)
    n_chunks = math.ceil(my_count / max_models_in_memory) if my_count > 0 else 1
    print0(f"  Evaluating ensemble of {num_models} models ({my_count} per node, "
           f"{world_size} nodes, {n_chunks} chunk(s))...")

    B_ensemble = 2
    val_loader = DDPValLoader(val_path, B_ensemble, config.sequence_len,
                              rank=0, world_size=1, device=device, seed=0)
    ensemble_eval_steps = val_loader.num_steps

    def _forward_pgt(model, x, flat_targets):
        with autocast_ctx:
            logits = model.forward_logits(x).float()
        flat_logits = logits.view(-1, logits.size(-1))
        logit_gt = flat_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        log_denom = torch.logsumexp(flat_logits, dim=-1)
        return torch.exp(logit_gt - log_denom)

    compiled_forward_pgt = torch.compile(_forward_pgt, dynamic=False)

    # First pass: collect per-step targets and init accumulators
    step_flat_y = []
    step_p_sums = []
    batch_iter = iter(val_loader)
    for _ in range(ensemble_eval_steps):
        _, y, _ = next(batch_iter)
        flat_y = y.view(-1)
        step_flat_y.append(flat_y)
        step_p_sums.append(torch.zeros_like(flat_y, dtype=torch.float32))

    # Process models in chunks
    for chunk_start in range(0, max(my_count, 1), max_models_in_memory):
        chunk_paths = my_paths[chunk_start:chunk_start + max_models_in_memory]
        if not chunk_paths:
            break
        chunk_models = []
        for ckpt_path in chunk_paths:
            with torch.device("meta"):
                model = GPT(config)
            model.to_empty(device=device)
            model.init_weights()
            model.bfloat16()
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            if dupe_layers:
                model.set_dupe_layers(*dupe_layers)
            model.eval()
            chunk_models.append(model)
            del state_dict

        chunk_weights = my_weights[chunk_start:chunk_start + max_models_in_memory]
        chunk_loader = DDPValLoader(val_path, B_ensemble, config.sequence_len,
                                    rank=0, world_size=1, device=device, seed=0)
        chunk_iter = iter(chunk_loader)
        for step_idx in range(ensemble_eval_steps):
            x, y, _ = next(chunk_iter)
            flat_y_clamped = y.view(-1).clamp(min=0)
            for model, w in zip(chunk_models, chunk_weights):
                step_p_sums[step_idx] += w * compiled_forward_pgt(model, x, flat_y_clamped)

        del chunk_models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # All-reduce and compute final metrics
    total_nats = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=device)

    for step_idx in range(ensemble_eval_steps):
        p_target_sum = step_p_sums[step_idx]
        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(p_target_sum, op=dist.ReduceOp.SUM, group=process_group)
        flat_y = step_flat_y[step_idx]
        # p_target_sum is already a weighted mixture: each contribution was
        # multiplied by its per-model weight (1/num_models if uniform).
        p_avg = p_target_sum
        loss2d = -torch.log(p_avg.clamp(min=1e-12))
        mask = flat_y != -1
        loss2d = loss2d * mask
        total_loss += loss2d[mask].sum().double()
        total_tokens += mask.sum()
        num_bytes2d = token_bytes[flat_y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum().double()
        total_bytes += num_bytes2d.sum()

    del step_flat_y, step_p_sums
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss

# =============================================================================
# End-of-training: gradient ensemble selection
# =============================================================================

def _run_selection(
    pgt_caches, num_models_total, k_values, device,
    node_rank, inter_node_group=None,
    grad_opt_steps=300, grad_lr=0.5,
):
    """Phase 1: gradient ensemble selection. Called only by node leaders (local_rank==0).

    Optimizes alpha (M-dim softmax mixture weights) on the held-out fitness pgt
    cache, then for each K picks the top-K by learned weight and renormalizes.
    Alpha is optimized on node 0 and broadcast to all leaders so every leader
    has bit-identical selections.

    Returns (grad_selections, active_global) where:
        grad_selections: {k: {"global_indices": [...], "weights": [...], "fit_loss": float}}
        active_global:   list of trained-model global indices
    """
    master_process = (node_rank == 0)
    dist.barrier(group=inter_node_group)

    if not pgt_caches:
        N_pgt_local = 0
    else:
        N_pgt_local = next(iter(pgt_caches.values())).numel()

    N_pgt_t = torch.tensor(N_pgt_local, dtype=torch.long, device=device)
    dist.all_reduce(N_pgt_t, op=dist.ReduceOp.MAX, group=inter_node_group)
    N_pgt = int(N_pgt_t.item())

    if N_pgt == 0:
        return {}, []

    pgt_global = torch.zeros(num_models_total, N_pgt, device=device)
    for model_idx, cache in pgt_caches.items():
        pgt_global[model_idx] = cache.to(device)
    dist.all_reduce(pgt_global, op=dist.ReduceOp.SUM, group=inter_node_group)

    active_indicator = torch.zeros(num_models_total, device=device)
    for idx in pgt_caches.keys():
        active_indicator[idx] = 1.0
    dist.all_reduce(active_indicator, op=dist.ReduceOp.SUM, group=inter_node_group)
    active_global = [i for i in range(num_models_total) if active_indicator[i].item() > 0.5]

    if not active_global:
        del pgt_global, active_indicator
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {}, []

    P_active = pgt_global[active_global]
    grad_selections = {}

    print0(f"\n  --- Gradient selection: {grad_opt_steps} steps, lr={grad_lr}, "
           f"M={len(active_global)} ---")
    if master_process:
        alpha_t, _ = _optimize_alpha_fn(
            P_fit=P_active, opt_steps=grad_opt_steps, lr=grad_lr, seed=0,
        )
    else:
        alpha_t = torch.zeros(P_active.shape[0], device=device, dtype=torch.float32)
    if inter_node_group is not None:
        dist.broadcast(alpha_t, src=0, group=inter_node_group)
    w_all = torch.softmax(alpha_t, dim=0)

    for k in k_values:
        k_eff = min(int(k), P_active.shape[0])
        top_idx = torch.topk(w_all, k=k_eff).indices.sort().values
        w_sel = w_all[top_idx]
        w_renorm = (w_sel / w_sel.sum().clamp_min(1e-12)).cpu().tolist()
        sel_local = top_idx.cpu().tolist()
        sel_global = [active_global[i] for i in sel_local]
        p_fit = (torch.tensor(w_renorm, device=device).unsqueeze(1)
                 * P_active[top_idx]).sum(dim=0)
        fit_loss = float((-torch.log(p_fit.clamp_min(1e-12))).mean().item())
        grad_selections[k] = {
            "global_indices": sel_global,
            "weights": [float(w) for w in w_renorm],
            "fit_loss": fit_loss,
        }
        print0(f"    [grad] k={len(sel_global)}: fit_loss={fit_loss:.6f}")

    del P_active, pgt_global, active_indicator
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return grad_selections, active_global


def _run_ensemble_evals(
    grad_selections, active_global,
    checkpoint_dir, config, token_bytes, device, autocast_ctx,
    val_path, rank, world_size, dupe_layers=None,
    timing_stats=None, wandb_run=None,
    grad_weighted_eval=True,
):
    """Evaluate each top-K gradient-selected ensemble on the val set.
    Called by ALL GPUs; models sharded across all GPUs for max parallelism."""
    ensemble_results = {}

    for k, entry in grad_selections.items():
        selected_global = entry["global_indices"]
        weights = entry["weights"] if grad_weighted_eval else None
        selected_paths = [os.path.join(checkpoint_dir, f"model_{i}.pt") for i in selected_global]
        t_eval = time.time()
        bpb, loss = evaluate_ensemble_bpb(
            checkpoint_paths=selected_paths, config=config, token_bytes=token_bytes,
            device=device, autocast_ctx=autocast_ctx, val_path=val_path,
            rank=rank, world_size=world_size, dupe_layers=dupe_layers,
            weights=weights,
        )
        if timing_stats is not None:
            timing_stats["ensemble_eval"] += time.time() - t_eval
        mode = "weighted" if weights is not None else "uniform"
        print0(f"  [FINAL] grad ensemble ({mode}, k={len(selected_global)}) | "
               f"Val BPB: {bpb:.6f} | Val Loss: {loss:.6f} | "
               f"Total active: {len(active_global)} models")
        print0(f"  Models (1-indexed): {[i + 1 for i in selected_global]}")
        if wandb_run is not None:
            wandb_run.log({
                f"grad_ensemble_k{k}/num_models": len(selected_global),
                f"grad_ensemble_k{k}/val_bpb": bpb,
                f"grad_ensemble_k{k}/val_loss": loss,
                f"grad_ensemble_k{k}/fit_loss": entry.get("fit_loss", float("nan")),
                f"grad_ensemble_k{k}/total_active": len(active_global),
            })
        ensemble_results[k] = {
            "selected_indices": selected_global,
            "weights": entry["weights"],
            "bpb": bpb, "loss": loss, "fit_loss": entry.get("fit_loss"),
        }

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ensemble_results

# =============================================================================
# Weight perturbation
# =============================================================================

def perturb_weights(model, scale, seed=None):
    """Perturb model weights with random noise. If seed is given, use it for
    deterministic noise (needed for DDP so all GPUs get identical perturbation)."""
    if seed is not None:
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() > 1:
                if seed is not None:
                    noise_cpu = torch.randn(p.shape, generator=rng, dtype=torch.float32)
                    noise = noise_cpu.to(device=p.device) * (p.float().std() * scale)
                else:
                    noise = torch.randn_like(p) * (p.float().std() * scale)
                p.add_(noise.to(p.dtype))

# =============================================================================
# Snapshot training run (multi-GPU DDP) with chain distillation + diversity + dupe
# =============================================================================

def train_snapshot_run(model_indices, seed, device, config, autocast_ctx, token_bytes,
                       wandb_run, checkpoint_dir, eval_steps, timing_stats,
                       pgt_steps=2, overrides=None,
                       k_values=None,
                       num_nodes=1, num_models_total=None, val_path=None,
                       intra_node_group=None, inter_node_group=None,
                       gpus_per_node=8, node_rank=0):
    rank = int(os.environ.get('RANK', 0))
    local_rank = rank % gpus_per_node
    is_node_leader = (local_rank == 0)
    num_cycles = len(model_indices)
    overrides = overrides or {}

    lr_mult = overrides.get("lr_multiplier", args.lr_multiplier)
    wd = overrides.get("weight_decay", args.weight_decay)
    ls = overrides.get("label_smoothing", args.label_smoothing)
    lr_floor = overrides.get("lr_floor", args.lr_floor)
    warmup_frac = overrides.get("warmup_frac", 0.0)
    warmdown_frac = overrides.get("warmdown_frac", 0.0)

    distill_alpha = args.distill_alpha
    distill_T = args.distill_temperature
    distill_after = args.distill_after_cycles
    perturb_scale_max = args.perturb_scale
    perturb_scale_min = args.perturb_scale_min

    # Compute LRs
    matrix_lr = args.matrix_lr * lr_mult
    scalar_lr = args.scalar_lr * lr_mult
    embedding_lr = getattr(args, 'embedding_lr', 0.15) * lr_mult
    unembedding_lr = getattr(args, 'unembedding_lr', 0.002) * lr_mult

    optim_config = {
        'matrix_lr': matrix_lr, 'scalar_lr': scalar_lr,
        'embedding_lr': embedding_lr, 'unembedding_lr': unembedding_lr,
        'weight_decay': wd, 'adam_betas': tuple(args.adam_betas),
        'muon_momentum': args.muon_momentum, 'ns_steps': args.ns_steps,
        'iha_lr': args.iha_lr,
    }

    # Dupe layer config (activates mid-cycle, not mid-run)

    override_desc = f", overrides={overrides}" if overrides else ""
    if is_node_leader:
        print(f"  [node {node_rank}] lr_mult={lr_mult}, wd={wd}, ls={ls}, floor={lr_floor}, "
              f"warmup={warmup_frac}, warmdown={warmdown_frac}"
              f"{override_desc}")
    if is_node_leader:
        print(f"  [node {node_rank}] Distillation: alpha={distill_alpha}, T={distill_T}, "
              f"after_cycles={distill_after}, teacher=last_snapshot")
        print(f"  [node {node_rank}] Perturbation: scale={perturb_scale_max} -> {perturb_scale_min} (cosine decay)")
        print(f"  [node {node_rank}] WD schedule: {args.wd_schedule}, wd_floor={args.wd_floor} "
              f"(WD swing per cycle: {args.wd_floor*wd:.4f} -> {wd:.4f})")
        print(f"  [node {node_rank}] Dupe layers: {args.dupe_layers_start}-{args.dupe_layers_end}, "
              f"activating after {args.dupe_after_n_models} models")
        print(f"  [node {node_rank}] DDP: {gpus_per_node} GPUs per node")

    # Model init seed: same for all GPUs in a node (identical initial weights)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Build model
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    param_counts = sum(p.numel() for p in model.parameters())
    if is_node_leader:
        print(f"  [node {node_rank}] Parameters: {param_counts:,}")

    # --- Resume: load checkpoint if available ---
    _resume_path = os.path.join(checkpoint_dir, f"resume_node_{node_rank}.pt")
    _resume_ckpt = None
    if args.resume and os.path.exists(_resume_path):
        if is_node_leader:
            print(f"  [node {node_rank}] Loading resume checkpoint...")
        _resume_ckpt = torch.load(_resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(_resume_ckpt.pop("model_state_dict"))
        if _resume_ckpt["dupe_active"]:
            model.set_dupe_layers(args.dupe_layers_start, args.dupe_layers_end)
        if is_node_leader:
            print(f"  [node {node_rank}] Will resume from cycle {_resume_ckpt['cycle']}, step {_resume_ckpt['step']}")

    # Wrap model in DDP for intra-node gradient sync
    ddp_model = DDP(model, device_ids=[local_rank], process_group=intra_node_group)
    compiled_model = torch.compile(ddp_model, dynamic=False)
    optimizer = model.setup_optimizer(optim_config)
    if _resume_ckpt:
        optimizer.load_state_dict(_resume_ckpt.pop("optimizer_state_dict"))

    # Data loader: all GPUs in a node share the same shuffle, each gets a non-overlapping
    # shard. Fitness set is carved off the FRONT at whole-document granularity so CD
    # selection sees training-distribution data the model never trained on. The remaining
    # docs feed the training DataLoader; with doc_shuffle on they get a fresh ordering
    # (and fresh sequence chunk boundaries) every epoch.
    _train_path = args.input_bin if args.input_bin else os.path.join(DATA_DIR, "fineweb_train.pt")
    _all_doc_tokens, _default_shuffle_seed = _load_new_format_file(_train_path)
    _fitness_doc_tokens, _train_doc_tokens, _ = _carve_fitness_docs(
        _all_doc_tokens, args.fitness_tokens)
    _fitness_seqs = _build_seqs_from_docs(_fitness_doc_tokens, MAX_SEQ_LEN + 1)
    del _all_doc_tokens
    doc_shuffle = not args.no_doc_shuffle
    if is_node_leader:
        print(f"  [node {node_rank}] Held out {len(_fitness_doc_tokens)} fitness doc(s) "
              f"({_fitness_seqs.shape[0]} seqs); training stream has "
              f"{len(_train_doc_tokens)} doc(s); doc_shuffle={doc_shuffle}")
    train_loader = DataLoader(None, args.device_batch_size, MAX_SEQ_LEN,
                              device=device, seed=seed,
                              dp_rank=local_rank, dp_world_size=gpus_per_node,
                              doc_tokens=_train_doc_tokens,
                              default_shuffle_seed=_default_shuffle_seed,
                              doc_shuffle=doc_shuffle)
    x, y, _ = next(train_loader)

    tokens_per_fwdbwd_node = args.device_batch_size * MAX_SEQ_LEN * gpus_per_node

    TOKENS_PER_EPOCH = train_loader.total_tokens
    tokens_per_cycle = args.epochs_per_cycle * TOKENS_PER_EPOCH
    total_epochs = num_cycles * args.epochs_per_cycle

    # ---- Batch size per cycle (constant within cycle, varies across cycles) ----
    bs_schedule = args.bs_schedule
    bs_peak = TOTAL_BATCH_SIZE
    bs_floor = args.bs_floor
    assert bs_peak % tokens_per_fwdbwd_node == 0, \
        f"bs_peak ({bs_peak}) must be a multiple of device_batch_size*MAX_SEQ_LEN*gpus_per_node ({tokens_per_fwdbwd_node})"
    assert bs_floor % tokens_per_fwdbwd_node == 0, \
        f"bs_floor ({bs_floor}) must be a multiple of device_batch_size*MAX_SEQ_LEN*gpus_per_node ({tokens_per_fwdbwd_node})"
    assert 0 < bs_floor <= bs_peak

    def bs_for_cycle(c):
        if bs_schedule == "flat" or num_cycles == 1:
            return bs_peak
        # Block-triangular: amp = 1 at middle cycle, 0 at edges; snap to peak in middle 50%
        amp = 1.0 - abs(2.0 * c / (num_cycles - 1) - 1.0)
        return bs_peak if amp >= 0.5 else bs_floor

    cycle_batch_sizes = [bs_for_cycle(c) for c in range(num_cycles)]
    cycle_lengths = [max(1, round(tokens_per_cycle / bs)) for bs in cycle_batch_sizes]
    cycle_grad_accums = [bs // tokens_per_fwdbwd_node for bs in cycle_batch_sizes]
    total_iterations = sum(cycle_lengths)
    dupe_start_step = sum(cycle_lengths[:args.dupe_after_n_models])

    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    _val_path = val_path

    if is_node_leader:
        _n_peak = sum(1 for b in cycle_batch_sizes if b == bs_peak)
        _n_floor = num_cycles - _n_peak
        print(f"  [node {node_rank}] Total: {total_epochs} epochs, {total_iterations} steps "
              f"({args.epochs_per_cycle} epochs/cycle)")
        print(f"  [node {node_rank}] BS schedule: {bs_schedule}, peak={bs_peak:,}, floor={bs_floor:,} "
              f"(x{gpus_per_node} GPUs DDP)")
        print(f"  [node {node_rank}] Cycles at peak BS: {_n_peak}, at floor BS: {_n_floor}")
        print(f"  [node {node_rank}] Dupe activates after {args.dupe_after_n_models} models (step {dupe_start_step})")

    # Cyclic LR schedule (per-cycle: phase splits recomputed from that cycle's length)
    def get_lr_multiplier(step_in_cycle, cycle_length):
        warmup_steps = int(cycle_length * warmup_frac)
        warmdown_steps = int(cycle_length * warmdown_frac)
        decay_steps = cycle_length - warmup_steps - warmdown_steps
        if step_in_cycle < warmup_steps:
            return lr_floor + (1.0 - lr_floor) * step_in_cycle / warmup_steps
        if step_in_cycle < warmup_steps + decay_steps:
            decay_progress = (step_in_cycle - warmup_steps) / decay_steps
            return lr_floor + (1.0 - lr_floor) * (1.0 - decay_progress)
        warmdown_progress = (step_in_cycle - warmup_steps - decay_steps) / warmdown_steps
        return lr_floor * 0.5 * (1.0 + math.cos(math.pi * warmdown_progress))

    # Per-cycle WD schedule: anti-correlated with LR (high LR -> low WD, low LR -> high WD)
    wd_schedule = args.wd_schedule
    wd_floor = args.wd_floor
    _lr_span = 1.0 - lr_floor
    def get_wd_multiplier(step_in_cycle, cycle_length):
        if wd_schedule == "flat" or _lr_span <= 0.0:
            return 1.0
        lrm = get_lr_multiplier(step_in_cycle, cycle_length)
        progress = (1.0 - lrm) / _lr_span
        return wd_floor + (1.0 - wd_floor) * progress

    # State initialization
    last_snapshot_state = None
    teacher_model = None
    compiled_teacher_fwd = None
    distilling = False
    dupe_active = False
    results = {}
    pgt_caches = {}
    last_ensemble_results = {}
    current_cycle = 0
    smooth_train_loss = 0
    start_step = 0
    step_in_cycle = 0
    cur_cycle_length = cycle_lengths[0]
    cur_grad_accum = cycle_grad_accums[0]

    if _resume_ckpt:
        current_cycle = _resume_ckpt["cycle"]
        start_step = _resume_ckpt["step"]
        dupe_active = _resume_ckpt["dupe_active"]
        distilling = _resume_ckpt["distilling"]
        results = _resume_ckpt["results"]
        smooth_train_loss = _resume_ckpt["smooth_train_loss"]
        pgt_caches = _resume_ckpt.get("pgt_caches", {})

        # Verify the BS/cycle-length schedule hasn't changed since the checkpoint
        # was written. If it did, total_iterations and cycle boundaries would
        # shift and start_step would no longer land on a cycle edge.
        saved_sched = _resume_ckpt.get("bs_sched_sig")
        current_sched = {
            "bs_schedule": bs_schedule,
            "bs_floor": bs_floor,
            "bs_peak": bs_peak,
            "epochs_per_cycle": args.epochs_per_cycle,
            "num_cycles": num_cycles,
            "tokens_per_fwdbwd_node": tokens_per_fwdbwd_node,
        }
        if saved_sched is not None and saved_sched != current_sched:
            raise RuntimeError(
                f"[node {node_rank}] Resume schedule mismatch.\n"
                f"  saved:   {saved_sched}\n"
                f"  current: {current_sched}\n"
                f"Re-launch with the original --bs-schedule/--bs-floor/--total-batch-size/"
                f"--epochs-per-cycle/--num-models/--device-batch-size/--gpus-per-node values, "
                f"or delete the resume checkpoint to start over."
            )
        expected_start = sum(cycle_lengths[:current_cycle])
        assert start_step == expected_start, (
            f"[node {node_rank}] Resume step {start_step} does not match cycle boundary "
            f"{expected_start} for cycle {current_cycle}. Refusing to proceed."
        )
        if current_cycle < num_cycles:
            step_in_cycle = 0
            cur_cycle_length = cycle_lengths[current_cycle]
            cur_grad_accum = cycle_grad_accums[current_cycle]

        # Reconstruct last snapshot for teacher from surviving checkpoints
        completed_indices = model_indices[:current_cycle]
        surviving = [idx for idx in completed_indices
                     if os.path.exists(os.path.join(checkpoint_dir, f"model_{idx}.pt"))]
        if surviving:
            last_idx = surviving[-1]
            sd = torch.load(os.path.join(checkpoint_dir, f"model_{last_idx}.pt"),
                           map_location="cpu", weights_only=True)
            last_snapshot_state = {k: v.float() for k, v in sd.items()}
            del sd
            if is_node_leader:
                print(f"  [node {node_rank}] Last snapshot loaded (model_{last_idx})")

        # Reconstruct teacher if distillation was active
        if distilling and last_snapshot_state is not None:
            with torch.device("meta"):
                teacher_model = GPT(config)
            teacher_model.to_empty(device=device)
            teacher_model.init_weights()
            snap_device = {k: v.to(device=device) for k, v in last_snapshot_state.items()}
            teacher_model.load_state_dict(snap_device)
            del snap_device
            teacher_model.set_dupe_layers(args.dupe_layers_start, args.dupe_layers_end)
            teacher_model.bfloat16()
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            compiled_teacher_fwd = torch.compile(teacher_model.forward_logits, dynamic=False)
            if is_node_leader:
                print(f"  [node {node_rank}] Teacher model reconstructed (last snapshot)")

        del _resume_ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if is_node_leader:
            print(f"  [node {node_rank}] Resume ready: cycle {current_cycle}/{num_cycles}, "
                  f"step {start_step}/{total_iterations}, dupe={dupe_active}, distill={distilling}")

    # Verify all nodes are at the same cycle/step before training starts
    if is_node_leader and num_nodes > 1:
        cycle_tensor = torch.tensor([current_cycle], dtype=torch.long, device=device)
        step_tensor = torch.tensor([start_step], dtype=torch.long, device=device)
        all_cycles = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(num_nodes)]
        all_steps = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(num_nodes)]
        dist.all_gather(all_cycles, cycle_tensor, group=inter_node_group)
        dist.all_gather(all_steps, step_tensor, group=inter_node_group)
        cycles = [c.item() for c in all_cycles]
        steps = [s.item() for s in all_steps]
        if len(set(cycles)) > 1 or len(set(steps)) > 1:
            msg = (f"FATAL: Nodes are out of sync! "
                   f"Cycles: {cycles}, Steps: {steps}. "
                   f"Delete the stale resume checkpoint(s) to align all nodes "
                   f"to the earliest cycle, or restart from scratch.")
            print(msg)
            dist.destroy_process_group()
            raise RuntimeError(msg)
        if node_rank == 0:
            print(f"  All {num_nodes} nodes synced at cycle {current_cycle}, step {start_step}")
    # Global barrier: ALL 64 GPUs must sync here (not just leaders).
    # Non-leaders skip the sync check above but still wait here.
    dist.barrier()

    gc.enable()
    gc.collect()

    compiled_model.train()
    for step in range(start_step, total_iterations):
        # Enable dupe layers permanently after the first N models
        if not dupe_active and step >= dupe_start_step:
            model.set_dupe_layers(args.dupe_layers_start, args.dupe_layers_end)
            dupe_active = True

        synchronize()
        t0 = time.time()

        mtp_on = config.mtp_weight > 0
        for micro_step in range(cur_grad_accum):
            # Skip DDP gradient sync on all but last micro-step
            sync_ctx = ddp_model.no_sync() if micro_step < cur_grad_accum - 1 else nullctx()
            with sync_ctx:
                if distilling:
                    with autocast_ctx:
                        # distill=True path returns (student_logits, mtp_loss)
                        student_logits, mtp_loss = compiled_model(x, y, distill=True)
                        with torch.no_grad():
                            teacher_logits = compiled_teacher_fwd(x)

                    flat_s = student_logits.view(-1, student_logits.size(-1)).float()
                    flat_t = teacher_logits.view(-1, teacher_logits.size(-1)).float()
                    flat_y = y.view(-1)
                    mask = flat_y != -1

                    ce_loss = F.cross_entropy(flat_s, flat_y, label_smoothing=ls)
                    kl_loss = F.kl_div(
                        F.log_softmax(flat_s[mask] / distill_T, dim=-1),
                        F.log_softmax(flat_t[mask] / distill_T, dim=-1),
                        reduction='batchmean', log_target=True,
                    ) * (distill_T ** 2)

                    # Fold MTP into the "normal loss" side of the alpha trade-off
                    # so it participates in the distillation balance, matching the
                    # non-distill path where forward() folds MTP into the returned loss.
                    normal_loss = ce_loss
                    if mtp_on:
                        normal_loss = normal_loss + config.mtp_weight * mtp_loss
                    loss = (1 - distill_alpha) * normal_loss + distill_alpha * kl_loss
                else:
                    with autocast_ctx:
                        loss = compiled_model(x, y, label_smoothing=ls)

                train_loss = loss.detach()
                (loss / cur_grad_accum).backward()
            x, y, _ = next(train_loader)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        lrm = get_lr_multiplier(step_in_cycle, cur_cycle_length)
        wdm = get_wd_multiplier(step_in_cycle, cur_cycle_length)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            group["weight_decay"] = group["initial_weight_decay"] * wdm
        optimizer.step()
        compiled_model.zero_grad(set_to_none=True)

        synchronize()
        dt = time.time() - t0
        if timing_stats is not None:
            timing_stats["training"] += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
        debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))

        wandb_run.log({
            "global_step": step,
            f"node_{node_rank}/train_loss": debiased,
            f"node_{node_rank}/lr_mult": lrm,
            f"node_{node_rank}/wd_mult": wdm,
            f"node_{node_rank}/cycle": current_cycle,
            f"node_{node_rank}/distilling": int(distilling),
            f"node_{node_rank}/dupe_active": int(dupe_active),
            f"node_{node_rank}/batch_size": cycle_batch_sizes[current_cycle],
            f"node_{node_rank}/grad_accum": cur_grad_accum,
            f"node_{node_rank}/step_in_cycle": step_in_cycle,
        })

        step_in_cycle += 1

        # =================================================================
        # End of cycle
        # =================================================================
        if step_in_cycle == cur_cycle_length:
            model_idx = model_indices[current_cycle]

            # Eval + p(gt) caching: all GPUs participate (sharded), then leader saves/logs
            # All GPUs have identical weights via DDP; use raw model (not compiled/DDP)
            # to avoid DDP param broadcast that would deadlock
            model.eval()

            _t_val = time.time()
            val_loader = DataLoader(_val_path, args.device_batch_size, MAX_SEQ_LEN,
                                    device=device, seed=model_idx,
                                    dp_rank=local_rank, dp_world_size=gpus_per_node)
            with autocast_ctx:
                val_bpb, val_loss = evaluate_bpb(model, val_loader, val_loader.num_steps,
                                                 token_bytes, device=device,
                                                 process_group=intra_node_group)
            if is_node_leader and timing_stats is not None:
                timing_stats["val_eval"] += time.time() - _t_val

            # Cache p(gt) on held-out fitness set; the gradient selector fits
            # mixture weights on this slice the model never trained on.
            _t_pgt = time.time()
            cached_pgt = cache_pgt_single_model(
                model, config, device, autocast_ctx, _fitness_seqs, pgt_steps=pgt_steps,
                local_rank=local_rank, gpus_per_node=gpus_per_node,
                intra_node_group=intra_node_group,
            )
            if is_node_leader:
                pgt_caches[model_idx] = cached_pgt
                if timing_stats is not None:
                    timing_stats["pgt_caching"] += time.time() - _t_pgt

                # Save snapshot (bf16 to halve disk usage; training stays fp32)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{model_idx}.pt")
                torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, checkpoint_path)

                print(f"  [node {node_rank}] Cycle {current_cycle+1}/{num_cycles} "
                      f"(model {model_idx+1}) | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}"
                      f"{' [distilled]' if distilling else ''}"
                      f"{' [dupe]' if dupe_active else ''}")

                wandb_run.log({
                    "global_step": step,
                    f"model_{model_idx+1}/val_bpb": val_bpb,
                    f"model_{model_idx+1}/val_loss": val_loss,
                    f"model_{model_idx+1}/distilled": int(distilling),
                    f"model_{model_idx+1}/dupe_active": int(dupe_active),
                })
                results[model_idx] = (val_bpb, val_loss)

            # Wait for leader to finish checkpoint save before all GPUs proceed
            dist.barrier(group=intra_node_group)

            # Save last snapshot for teacher (before perturbation) — all GPUs do this to keep state consistent
            last_snapshot_state = {k: v.float().cpu() for k, v in model.state_dict().items()}

            current_cycle += 1

            # === Chain distillation teacher (all GPUs need the teacher for forward) ===
            if current_cycle >= distill_after and distill_alpha > 0:
                if teacher_model is None:
                    with torch.device("meta"):
                        teacher_model = GPT(config)
                    teacher_model.to_empty(device=device)
                    teacher_model.init_weights()
                    snap_device = {k: v.to(device=device) for k, v in last_snapshot_state.items()}
                    teacher_model.load_state_dict(snap_device)
                    del snap_device
                    teacher_model.set_dupe_layers(args.dupe_layers_start, args.dupe_layers_end)
                    teacher_model.bfloat16()
                    teacher_model.eval()
                    teacher_model.requires_grad_(False)
                    compiled_teacher_fwd = torch.compile(teacher_model.forward_logits, dynamic=False)
                    distilling = True
                    if is_node_leader:
                        print(f"  [node {node_rank}] Chain distillation ON at cycle {current_cycle} "
                              f"(last snapshot as teacher, alpha={distill_alpha}, T={distill_T})")
                else:
                    with torch.no_grad():
                        for name, param in teacher_model.named_parameters():
                            param.copy_(last_snapshot_state[name].to(device=device, dtype=param.dtype))
                    if is_node_leader:
                        print(f"  [node {node_rank}] Teacher updated (last snapshot)")

            # === End-of-training selection ===
            is_final = (current_cycle == num_cycles)

            if is_final and k_values:
                # Global sync: nodes train independently so they can drift apart;
                # ensure all ranks across all nodes are here before inter-node work
                dist.barrier()
                dupe_for_eval = (args.dupe_layers_start, args.dupe_layers_end)

                # Phase 1: gradient selection (node leaders only, fast)
                grad_selections = {}
                active_global = []
                if is_node_leader:
                    if node_rank == 0:
                        print(f"\n  Running final gradient selection after "
                              f"{current_cycle}/{num_cycles} cycles (k={k_values})")
                    _t_sel = time.time()
                    grad_selections, active_global = _run_selection(
                        pgt_caches=pgt_caches, num_models_total=num_models_total,
                        k_values=k_values, device=device,
                        node_rank=node_rank, inter_node_group=inter_node_group,
                        grad_opt_steps=args.grad_opt_steps, grad_lr=args.grad_lr,
                    )
                    if timing_stats is not None:
                        timing_stats["selection"] += time.time() - _t_sel

                # Broadcast selection results from node leader to all GPUs in node
                dist.barrier(group=intra_node_group)
                node_leader_global_rank = node_rank * gpus_per_node

                if is_node_leader:
                    grad_idx_parts = []
                    grad_w_parts = []
                    for k, entry in sorted(grad_selections.items()):
                        inds = entry["global_indices"]
                        ws = entry["weights"]
                        grad_idx_parts.extend([k, len(inds)] + inds)
                        grad_w_parts.extend(ws)
                    gidx_t = torch.tensor(grad_idx_parts, dtype=torch.long, device=device)
                    gw_t = torch.tensor(grad_w_parts, dtype=torch.float64, device=device)
                    active_t = torch.tensor(active_global, dtype=torch.long, device=device)
                    sizes_t = torch.tensor(
                        [len(gidx_t), len(gw_t), len(active_t)],
                        dtype=torch.long, device=device,
                    )
                else:
                    sizes_t = torch.zeros(3, dtype=torch.long, device=device)

                dist.broadcast(sizes_t, src=node_leader_global_rank, group=intra_node_group)
                s_gidx, s_gw, s_active = sizes_t.tolist()

                if not is_node_leader:
                    gidx_t = torch.zeros(s_gidx, dtype=torch.long, device=device)
                    gw_t = torch.zeros(s_gw, dtype=torch.float64, device=device)
                    active_t = torch.zeros(s_active, dtype=torch.long, device=device)

                if s_gidx > 0:
                    dist.broadcast(gidx_t, src=node_leader_global_rank, group=intra_node_group)
                if s_gw > 0:
                    dist.broadcast(gw_t, src=node_leader_global_rank, group=intra_node_group)
                if s_active > 0:
                    dist.broadcast(active_t, src=node_leader_global_rank, group=intra_node_group)

                if not is_node_leader:
                    grad_selections = {}
                    gidx_vals = gidx_t.tolist()
                    gw_vals = gw_t.tolist()
                    i = 0
                    w_cursor = 0
                    while i < len(gidx_vals):
                        k_val, n_sel = gidx_vals[i], gidx_vals[i + 1]
                        inds = gidx_vals[i + 2: i + 2 + n_sel]
                        ws = gw_vals[w_cursor: w_cursor + n_sel]
                        grad_selections[k_val] = {
                            "global_indices": inds, "weights": ws,
                        }
                        i += 2 + n_sel
                        w_cursor += n_sel
                    active_global = active_t.tolist()

                if grad_selections:
                    # Phase 2: Ensemble eval (ALL GPUs, models sharded across all)
                    global_rank = int(os.environ.get('RANK', 0))
                    global_world_size = num_nodes * gpus_per_node
                    last_ensemble_results = _run_ensemble_evals(
                        grad_selections=grad_selections,
                        active_global=active_global, checkpoint_dir=checkpoint_dir,
                        config=config, token_bytes=token_bytes, device=device,
                        autocast_ctx=autocast_ctx, val_path=_val_path,
                        rank=global_rank, world_size=global_world_size,
                        dupe_layers=dupe_for_eval, timing_stats=timing_stats,
                        wandb_run=wandb_run,
                        grad_weighted_eval=args.grad_weighted_eval,
                    )

            # Sync all GPUs after end-of-training selection
            dist.barrier(group=intra_node_group)

            # === Perturb weights before next cycle (cosine decay) ===
            if current_cycle < num_cycles and perturb_scale_max > 0:
                # current_cycle goes from 1 (first perturbation) to num_cycles-1 (last)
                progress = (current_cycle - 1) / max(1, num_cycles - 2)
                decayed_scale = perturb_scale_min + (perturb_scale_max - perturb_scale_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
                # Use deterministic seed so all GPUs in the node get identical noise
                perturb_seed = seed * 100003 + current_cycle * 31337
                perturb_weights(model, decayed_scale, seed=perturb_seed)
                # Broadcast from node leader to guarantee bit-exact sync across GPUs
                node_leader_global_rank = node_rank * gpus_per_node
                for p in model.parameters():
                    dist.broadcast(p.data, src=node_leader_global_rank, group=intra_node_group)

            # === Save resume checkpoint (node leader only) ===
            if is_node_leader:
                _rp = os.path.join(checkpoint_dir, f"resume_node_{node_rank}.pt")
                torch.save({
                    "cycle": current_cycle,
                    "step": step + 1,
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dupe_active": dupe_active,
                    "distilling": distilling,
                    "results": dict(results),
                    "smooth_train_loss": smooth_train_loss,
                    "pgt_caches": {k: v.cpu() for k, v in pgt_caches.items()},
                    "bs_sched_sig": {
                        "bs_schedule": bs_schedule,
                        "bs_floor": bs_floor,
                        "bs_peak": bs_peak,
                        "epochs_per_cycle": args.epochs_per_cycle,
                        "num_cycles": num_cycles,
                        "tokens_per_fwdbwd_node": tokens_per_fwdbwd_node,
                    },
                }, _rp)

            # Continue training — set both raw model and DDP wrapper back to train mode,
            # and advance per-cycle batch-size state.
            if current_cycle < num_cycles:
                step_in_cycle = 0
                cur_cycle_length = cycle_lengths[current_cycle]
                cur_grad_accum = cycle_grad_accums[current_cycle]
                model.train()
                compiled_model.train()

        if step == start_step:
            gc.collect(); gc.freeze(); gc.disable()

    # Cleanup
    del model, ddp_model, compiled_model, optimizer, train_loader
    if teacher_model is not None:
        del teacher_model, compiled_teacher_fwd
    gc.enable()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results, last_ensemble_results

# =============================================================================
# Main
# =============================================================================

def main():
    total_start_time = time.time()
    rank, local_rank, world_size = get_dist_info()
    master_process = rank == 0
    gpus_per_node = args.gpus_per_node
    num_nodes = world_size // gpus_per_node
    node_rank = rank // gpus_per_node  # which node (0..num_nodes-1)
    node_local_rank = rank % gpus_per_node  # alias for local_rank within node

    assert world_size % gpus_per_node == 0, (
        f"world_size ({world_size}) must be divisible by gpus_per_node ({gpus_per_node})")
    assert args.num_models >= num_nodes, f"Need at least one model per node"
    assert args.num_models % num_nodes == 0, (
        f"num_models ({args.num_models}) must be divisible by num_nodes ({num_nodes})")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    import datetime
    dist.init_process_group(backend="nccl", device_id=device,
                            timeout=datetime.timedelta(hours=3))
    dist.barrier()

    # --- Process groups for multi-node DDP ---
    pg_timeout = datetime.timedelta(hours=3)

    # Intra-node group: GPUs within the same node (for DDP gradient sync)
    intra_node_group = None
    for node in range(num_nodes):
        ranks_in_node = list(range(node * gpus_per_node, (node + 1) * gpus_per_node))
        group = dist.new_group(ranks_in_node, timeout=pg_timeout)
        if node == node_rank:
            intra_node_group = group

    # Inter-node group: one representative (local_rank==0) per node (for selection)
    inter_node_ranks = list(range(0, world_size, gpus_per_node))
    inter_node_group = dist.new_group(inter_node_ranks, timeout=pg_timeout)
    is_node_leader = (node_local_rank == 0)

    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

    if master_process:
        if _fa3 is not None:
            print("Using Flash Attention 3 (Hopper GPU detected)")
        else:
            print("Using PyTorch SDPA fallback (no FA3)")

    # Run ID + checkpoint dir
    run_id = args.resume if args.resume else time.strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(args.checkpoint_base, run_id)
    if master_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Run: {run_id}, checkpoint_dir: {checkpoint_dir}")
    dist.barrier()

    # wandb
    run_name = args.run if args.run else f"slowrun-main_{run_id}"
    _wandb_kwargs = {"project": "slowrun", "name": run_name}
    if args.wandb_group:
        _wandb_kwargs["group"] = args.wandb_group
    wandb_run = DummyWandb() if not master_process else wandb.init(**_wandb_kwargs)
    if master_process:
        wandb_run.log_code(".")

    # Tokenizer + token_bytes
    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.n_vocab
    eot_id = encoder._special_tokens['<|endoftext|>']
    token_bytes_list = []
    for i in range(vocab_size):
        if i == eot_id:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(encoder.decode_single_token_bytes(i)))
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=DEPTH, n_head=N_HEAD, n_kv_head=N_HEAD,
        n_embd=N_EMBD, window_pattern=WINDOW_PATTERN,
        dropout=args.dropout,
        use_iha=args.iha, iha_mix_v=args.iha,
        mtp_weight=args.mtp_weight,
    )

    _val_path = args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "fineweb_val.pt")

    # Auto-size eval steps from val data
    _tmp_val_loader = DataLoader(_val_path, args.device_batch_size, MAX_SEQ_LEN, device=device, seed=0)
    eval_steps = _tmp_val_loader.num_steps
    del _tmp_val_loader

    # Model assignment: contiguous blocks per node (all GPUs in a node train the same model)
    cycles_per_node = args.num_models // num_nodes
    my_model_indices = list(range(node_rank * cycles_per_node, (node_rank + 1) * cycles_per_node))

    # Per-node overrides
    overrides = NODE_OVERRIDES.get(node_rank, {})

    # K values for gradient ensemble selection at end of training
    k_values = sorted(set(k for k in args.k_sweep if 0 < k < args.num_models))

    if master_process:
        total_epochs = cycles_per_node * args.epochs_per_cycle
        dupe_after_n = args.dupe_after_n_models
        print(f"\n{'='*60}")
        print(f"Chain Distillation Snapshot Ensemble (1.8B model, 100M tokens)")
        print(f"  {args.num_models} models = {num_nodes} nodes x {cycles_per_node} snapshots/node")
        print(f"  {num_nodes} nodes x {gpus_per_node} GPUs/node = {world_size} total GPUs")
        print(f"{'='*60}")
        print(f"  run_id={run_id}")
        print(f"  n_layer={DEPTH}, n_embd={N_EMBD}, n_head={N_HEAD}, head_dim={HEAD_DIM}")
        print(f"  seq_len={MAX_SEQ_LEN}, window_pattern={WINDOW_PATTERN}")
        print(f"  total_batch_size={TOTAL_BATCH_SIZE}, device_batch_size={args.device_batch_size}")
        print(f"  OPTIMAL: lr_multiplier={args.lr_multiplier}, weight_decay={args.weight_decay}")
        print(f"  OPTIMAL: label_smoothing={args.label_smoothing}, lr_floor={args.lr_floor}")
        print(f"  adam_betas={tuple(args.adam_betas)}, muon_momentum={args.muon_momentum}, ns_steps={args.ns_steps}")
        print(f"  epochs_per_cycle={args.epochs_per_cycle}, total_epochs_per_node={total_epochs}")
        print(f"  pgt_steps={args.pgt_steps}")
        print(f"  eval_steps={eval_steps}")
        print(f"  --- Architecture ---")
        print(f"  SwiGLU MLP, attention gating, U-Net skip connections, VE projections, dropout={args.dropout}")
        print(f"  dupe_layers={args.dupe_layers_start}-{args.dupe_layers_end}, "
              f"activating after {dupe_after_n} models (then always on)")
        print(f"  iha={args.iha}, iha_lr={args.iha_lr}")
        print(f"  mtp_weight={args.mtp_weight}")
        print(f"  doc_shuffle={not args.no_doc_shuffle}")
        print(f"  --- Chain Distillation ---")
        print(f"  distill_alpha={args.distill_alpha}")
        print(f"  distill_temperature={args.distill_temperature}")
        print(f"  distill_after_cycles={args.distill_after_cycles}")
        print(f"  teacher=last_snapshot")
        print(f"  --- Batch Size Schedule ---")
        print(f"  bs_schedule={args.bs_schedule}, bs_peak={TOTAL_BATCH_SIZE:,}, bs_floor={args.bs_floor:,}")
        print(f"  --- Weight Perturbation ---")
        print(f"  perturb_scale={args.perturb_scale} -> {args.perturb_scale_min} (cosine decay)")
        print(f"  --- Gradient Ensemble Selection (final cycle) ---")
        print(f"  k_sweep={k_values}")
        print(f"  grad_opt_steps={args.grad_opt_steps}, grad_lr={args.grad_lr}")
        print(f"  grad_weighted_eval={args.grad_weighted_eval}")
        print(f"  checkpoint_dir={checkpoint_dir}")
        print(f"\n  Per-node diversity overrides:")
        for n in range(num_nodes):
            ov = NODE_OVERRIDES.get(n, {})
            if ov:
                print(f"    node {n}: {ov}")
            else:
                print(f"    node {n}: optimal (no overrides)")
        print(f"{'='*60}")

    # =========================================================================
    # Snapshot training
    # =========================================================================
    timing_stats = {"training": 0.0, "val_eval": 0.0, "pgt_caching": 0.0,
                    "ensemble_eval": 0.0, "selection": 0.0}

    local_results, last_ensemble_results = train_snapshot_run(
        model_indices=my_model_indices,
        seed=args.seed + node_rank,       # model init seed: same for all GPUs in node
        device=device, config=config, autocast_ctx=autocast_ctx,
        token_bytes=token_bytes, wandb_run=wandb_run,
        checkpoint_dir=checkpoint_dir, eval_steps=eval_steps,
        timing_stats=timing_stats, pgt_steps=args.pgt_steps,
        overrides=overrides,
        k_values=k_values,
        num_nodes=num_nodes, num_models_total=args.num_models,
        val_path=_val_path,
        intra_node_group=intra_node_group, inter_node_group=inter_node_group,
        gpus_per_node=gpus_per_node, node_rank=node_rank,
    )

    dist.barrier()

    # Gather individual results from node leaders via inter_node_group
    results_tensor = torch.zeros(args.num_models, 2, dtype=torch.float64, device=device)
    if is_node_leader:
        for model_idx, (bpb, loss) in local_results.items():
            results_tensor[model_idx, 0] = bpb
            results_tensor[model_idx, 1] = loss
        dist.all_reduce(results_tensor, op=dist.ReduceOp.SUM, group=inter_node_group)

    individual_results = []
    for i in range(args.num_models):
        individual_results.append({
            "model": i + 1,
            "val_bpb": results_tensor[i, 0].item(),
            "val_loss": results_tensor[i, 1].item(),
        })

    if master_process:
        print(f"\nIndividual model results:")
        for r in individual_results:
            print(f"  Model {r['model']}: BPB={r['val_bpb']:.6f}, Loss={r['val_loss']:.6f}")

        wandb_run.define_metric("single_model_val_loss", step_metric="model_number")
        for i in range(args.num_models):
            wandb_run.log({"model_number": i + 1, "single_model_val_loss": results_tensor[i, 1].item()})

    # =========================================================================
    # Summary + save
    # =========================================================================
    if master_process:
        print(f"\n{'='*60}")
        print(f"Chain Distillation Ensemble Training Complete (1.8B, 100M)")
        print(f"{'='*60}")

        if last_ensemble_results:
            print(f"\n--- Final Gradient Ensemble Summary ---")
            for k, r in sorted(last_ensemble_results.items()):
                print(f"  k={k:4d}: BPB={r['bpb']:.6f}  Loss={r['loss']:.6f}  "
                      f"Models={[i+1 for i in r['selected_indices']]}")

        if args.save_result:
            result = {
                "config": {
                    "n_layer": DEPTH, "n_embd": N_EMBD, "n_head": N_HEAD,
                    "total_batch_size": TOTAL_BATCH_SIZE,
                    "epochs_per_cycle": args.epochs_per_cycle,
                    "num_models": args.num_models,
                    "dupe_layers": f"{args.dupe_layers_start}-{args.dupe_layers_end}",
                    "dupe_after_n_models": args.dupe_after_n_models,
                },
                "chain_distillation": {
                    "alpha": args.distill_alpha, "temperature": args.distill_temperature,
                    "after_cycles": args.distill_after_cycles, "teacher": "last_snapshot",
                },
                "weight_perturbation": {"perturb_scale_max": args.perturb_scale, "perturb_scale_min": args.perturb_scale_min},
                "node_overrides": {str(k): v for k, v in NODE_OVERRIDES.items()},
                "individual_models": individual_results,
            }
            if last_ensemble_results:
                result["grad_sweep"] = []
                for k in sorted(last_ensemble_results.keys()):
                    r = last_ensemble_results[k]
                    result["grad_sweep"].append({
                        "k": k,
                        "selected_models": [i + 1 for i in r["selected_indices"]],
                        "weights": r.get("weights"),
                        "val_bpb": r["bpb"], "val_loss": r["loss"],
                        "fit_loss": r.get("fit_loss"),
                    })
                # Largest-K row is the canonical final ensemble.
                best_k = max(last_ensemble_results.keys())
                r = last_ensemble_results[best_k]
                result["final_ensemble_bpb"] = r["bpb"]
                result["final_ensemble_loss"] = r["loss"]
            with open(args.save_result, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.save_result}")

        progress_path = os.path.join(checkpoint_dir, "progress.json")
        progress = {"individual_models": individual_results}
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    dist.barrier()

    total_elapsed = time.time() - total_start_time
    if master_process:
        hours, remainder = divmod(total_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        print(f"\n--- Timing Breakdown (node 0, gpu 0) ---")
        for label, secs in timing_stats.items():
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            pct = 100 * secs / total_elapsed if total_elapsed > 0 else 0
            print(f"  {label:20s}: {int(h)}h {int(m)}m {s:5.1f}s  ({pct:5.1f}%)")
        tracked = sum(timing_stats.values())
        other = total_elapsed - tracked
        h, rem = divmod(other, 3600)
        m, s = divmod(rem, 60)
        pct = 100 * other / total_elapsed if total_elapsed > 0 else 0
        print(f"  {'other':20s}: {int(h)}h {int(m)}m {s:5.1f}s  ({pct:5.1f}%)")

    wandb_run.finish()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
