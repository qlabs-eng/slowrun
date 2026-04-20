"""
NCA pre-pre-training for the slowrun model.
Trains the GPT model on synthetic Neural Cellular Automata (NCA) sequences
before standard language pre-training.

Paper: "Training Language Models via Neural Cellular Automata" (arXiv:2603.10055)

Usage (sanity check, single GPU, tiny model):
    python pre_pre_train.py --n-layer 4 --n-head 2 --n-embd 128 \
        --num-train-tokens 40960 --num-eval-tokens 4096 \
        --device-batch-size 4 --total-batch-size 4096 --num-epochs 2

Usage (8xH100, ~10M NCA tokens, 10 epochs):
    torchrun --standalone --nproc_per_node=8 pre_pre_train.py \
        --num-train-tokens 10000000 --num-eval-tokens 1000000 --num-epochs 10

NOTE: The GPT model in this file is copied from train.py. Keep the two in sync
if the architecture changes, to ensure checkpoint compatibility.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import io
import gzip
import math
import time
import json
import argparse
from dataclasses import dataclass
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
import wandb

# =============================================================================
# CLI Arguments
# =============================================================================

parser = argparse.ArgumentParser(description="NCA pre-pre-training for slowrun GPT")

# NCA simulation
parser.add_argument("--grid", type=int, default=12, help="NCA grid size (grid x grid)")
parser.add_argument("--patch", type=int, default=2, help="Patch size for tokenization (patch x patch cells → 1 token)")
parser.add_argument("--num-colors", type=int, default=10, help="NCA discrete state space size")
parser.add_argument("--identity-bias", type=float, default=0.0, help="Bias added to current-state logits (encourages state persistence)")
parser.add_argument("--temperature", type=float, default=0.5, help="NCA sampling temperature")
parser.add_argument("--dT", type=int, default=2, help="NCA steps between recorded snapshots")
parser.add_argument("--init-rollout-steps", type=int, default=0, help="Transient warmup NCA steps before recording")

# Data generation — two mutually-exclusive ways to specify training data size:
#   --num-train-tokens T : generate T/num_epochs tokens per epoch (dataset = T/num_epochs tokens, seen num_epochs times)
#   --tokens-per-epoch T : generate exactly T tokens per epoch (dataset = T tokens, seen num_epochs times)
# Either way the dataset is generated ONCE and reshuffled each epoch.
_tok_group = parser.add_mutually_exclusive_group()
_tok_group.add_argument("--num-train-tokens", type=int, default=10_000_000,
                         help="Total training token budget; dataset = this / num-epochs tokens (default)")
_tok_group.add_argument("--tokens-per-epoch", type=int, default=None,
                         help="Tokens per epoch (dataset size); total = this × num-epochs")
parser.add_argument("--num-eval-tokens", type=int, default=1_000_000,
                    help="Evaluation token budget (dataset generated once)")
parser.add_argument("--sims-per-rule", type=int, default=10, help="Simulations per rule (different initial states, same dynamics)")
parser.add_argument("--min-grid", type=int, default=1, help="Context snapshots before loss is computed")
parser.add_argument("--filter-rules", action="store_true", help="Filter NCA rules by gzip complexity")
parser.add_argument("--filter-threshold", type=float, default=0.5, help="Gzip complexity lower bound")
parser.add_argument("--filter-upper-bound", type=float, default=1.0, help="Gzip complexity upper bound")

# Model (must match train.py defaults for checkpoint compatibility)
parser.add_argument("--n-layer", type=int, default=30)
parser.add_argument("--n-head", type=int, default=14)
parser.add_argument("--n-embd", type=int, default=1792)
parser.add_argument("--seq-len", type=int, default=1024, help="Token sequence length for NCA training")
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--window-pattern", type=str, default="SSSL")
parser.add_argument("--logit-cap", type=float, default=10.0)

# Training
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--total-batch-size", type=int, default=131072, help="Total tokens per optimizer step")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--num-epochs", type=int, default=10)
parser.add_argument("--warmup-ratio", type=float, default=0.05)
parser.add_argument("--warmdown-ratio", type=float, default=0.2)
parser.add_argument("--max-train-time", type=float, default=None,
                    help="Max cumulative training step time in minutes (None = unlimited). "
                         "Counts forward+backward+optimizer time per step; excludes eval, checkpointing, and startup.")
parser.add_argument("--max-wall-time", type=float, default=None,
                    help="Max total wall-clock time in minutes (None = unlimited). "
                         "Measured from script start, includes eval and data generation.")

# Data regeneration
parser.add_argument("--regen-data", action="store_true",
                    help="Regenerate training data each epoch (fresh NCA trajectories, same rules). "
                         "Matches the paper's --generate_train behaviour.")

# Output
parser.add_argument("--save-dir", type=str, default="nca_ckpts")
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--wandb-group", type=str, default=None)

args = parser.parse_args()
_script_start = time.time()

# =============================================================================
# NCA Simulation (PyTorch port from nca-pre-pretraining/utils/nca.py)
# Original implementation uses JAX/Flax; ported here to pure PyTorch.
# =============================================================================

class NCANetwork(nn.Module):
    """Small convolutional net defining the NCA transition rule.
    Port of NCANetwork from nca-pre-pretraining/utils/nca.py."""
    def __init__(self, d_state: int = 10):
        super().__init__()
        # 3x3 conv for spatial neighbourhood integration (padding done separately for wrap)
        self.conv1 = nn.Conv2d(d_state, 4, kernel_size=3, padding=0, bias=True)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(16, d_state, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, d_state, H, W)
        x = F.pad(x, (1, 1, 1, 1), mode='circular')  # wrap-around (matches JAX mode='wrap')
        x = self.conv1(x)                              # (B, 4, H, W)
        x = self.conv2(x)                              # (B, 16, H, W)
        x = F.relu(x)
        x = self.conv3(x)                              # (B, d_state, H, W)
        return x


class NCA:
    """Discrete Neural Cellular Automaton substrate."""
    def __init__(self, grid_size: int = 12, d_state: int = 10,
                 identity_bias: float = 0.0, temperature: float = 0.5):
        self.grid_size = grid_size
        self.d_state = d_state
        self.identity_bias = identity_bias
        self.temperature = max(temperature, 1e-8)

    def default_params(self, seed: int) -> NCANetwork:
        """Return an NCANetwork whose weights are seeded by `seed` (defines the 'rule')."""
        net = NCANetwork(d_state=self.d_state)
        gen = torch.Generator()
        gen.manual_seed(int(seed) & 0xFFFFFFFF)
        for p in net.parameters():
            nn.init.normal_(p, generator=gen)
        net.eval()
        return net

    def init_state(self, seed: int) -> Tensor:
        """Random initial grid state: (H, W) integers in [0, d_state)."""
        gen = torch.Generator()
        gen.manual_seed(int(seed) & 0xFFFFFFFF)
        return torch.randint(0, self.d_state, (self.grid_size, self.grid_size), generator=gen)

    @torch.no_grad()
    def step_state_batched(self, states: Tensor, net: NCANetwork) -> Tensor:
        """One NCA step over a batch of grids.
        Args:
            states: (B, H, W) integer tensor
            net: NCANetwork
        Returns: (B, H, W) integer tensor
        """
        B, H, W = states.shape
        # One-hot encode: (B, d_state, H, W)
        state_oh = F.one_hot(states, self.d_state).permute(0, 3, 1, 2).float()
        logits = net(state_oh)                          # (B, d_state, H, W)
        logits = logits + state_oh * self.identity_bias # identity bias
        logits = logits / self.temperature
        probs = F.softmax(logits, dim=1)                # (B, d_state, H, W)
        # Categorical sample: reshape to (B*H*W, d_state), sample, reshape back
        flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, self.d_state)  # (B*H*W, d_state)
        next_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(1)  # (B*H*W,)
        return next_flat.view(B, H, W)


@torch.no_grad()
def generate_nca_dataset(rule_seeds: list, sims_per_rule: int, num_examples: int,
                          dT: int, start_step: int, nca: NCA, epoch: int = 0) -> Tensor:
    """
    Generate NCA trajectory dataset for a set of rules.
    Each rule produces `sims_per_rule` sequences of `num_examples` grid snapshots.

    Returns: (num_rules * sims_per_rule, num_examples, H, W) integer tensor
    """
    all_grids = []
    H = W = nca.grid_size

    for rule_idx, rule_seed in enumerate(rule_seeds):
        net = nca.default_params(rule_seed)

        # Initialize all simulations for this rule in parallel
        init_grids = torch.stack([
            nca.init_state(epoch * 99991 + rule_idx * 997 + sim_idx)
            for sim_idx in range(sims_per_rule)
        ])  # (sims_per_rule, H, W)

        # Seed global RNG for reproducible stochastic transitions
        torch.manual_seed(rule_seed * 10007 + epoch * 997)

        states = init_grids  # (B, H, W)

        # Warmup steps
        for _ in range(start_step):
            states = nca.step_state_batched(states, net)

        # Record snapshots every dT steps
        snapshots = []
        for _ in range(num_examples):
            for _ in range(dT):
                states = nca.step_state_batched(states, net)
            snapshots.append(states.clone())

        # Stack: (num_examples, B, H, W) → (B, num_examples, H, W)
        rule_grids = torch.stack(snapshots, dim=1)
        all_grids.append(rule_grids)

    return torch.cat(all_grids, dim=0)  # (total_sims, num_examples, H, W)


# =============================================================================
# Rule Filtering
# =============================================================================

def gzip_complexity(byte_data: bytes) -> float:
    """Gzip compression ratio as proxy for Kolmogorov complexity (matches reference)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(byte_data)
    return len(buf.getvalue()) / max(len(byte_data), 1)


def score_rule(rule_seed: int, nca: NCA, tokenizer, n_steps: int = 10,
               dT: int = 1) -> float:
    """Score a rule by gzip complexity of its tokenized rollout."""
    grids = generate_nca_dataset([rule_seed], 1, n_steps, dT, 0, nca, epoch=0)
    seq, _ = tokenizer.encode(grids)  # (1, L)
    seq_np = seq[0].numpy().astype(np.int32)
    return gzip_complexity(seq_np.tobytes())


def generate_filtered_rules(nca: NCA, tokenizer, num_rules: int,
                              threshold: float, upper_bound: float,
                              base_seed: int = 0) -> list:
    """Yield rule seeds whose gzip complexity falls in [threshold, upper_bound]."""
    accepted = []
    candidate = base_seed
    while len(accepted) < num_rules:
        s = score_rule(candidate, nca, tokenizer)
        if threshold <= s <= upper_bound:
            accepted.append(candidate)
        candidate += 1
    return accepted


# =============================================================================
# NCA Tokenizer (PyTorch port from nca-pre-pretraining/utils/tokenizers.py)
# =============================================================================

class NCATokenizer:
    """
    Converts NCA grid snapshots to token sequences using patch-based encoding.
    A (patch x patch) tile of cells becomes a single integer token via base-num_colors encoding.
    """
    def __init__(self, patch: int = 2, num_colors: int = 10):
        self.patch = patch
        self.num_colors = num_colors
        self.start_tk = num_colors ** (patch * patch)   # 10000 for defaults
        self.end_tk = self.start_tk + 1                 # 10001 for defaults
        self.vocab_size = self.end_tk + 1               # 10002 for defaults

    def grid_len(self, grid_size: int) -> int:
        """Tokens per snapshot: num_patches + start_token + end_token."""
        return (grid_size // self.patch) ** 2 + 2

    def encode(self, grids: Tensor) -> tuple:
        """
        Encode NCA grid snapshots to flat token sequences.

        Args:
            grids: (B, N, H, W) integer tensor of cell states in [0, num_colors)

        Returns:
            tokens:  (B, N * grid_len) long tensor of token IDs
            targets: (B, N * grid_len) long tensor; -100 at start/end positions
        """
        B, N, H, W = grids.shape
        p = self.patch
        N_H, N_W = H // p, W // p

        # Reshape into non-overlapping patches: (B, N, N_H, N_W, p, p)
        g = grids.reshape(B, N, N_H, p, N_W, p)
        g = g.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, N, N_H, N_W, p, p)
        g = g.reshape(B, N, N_H * N_W, p * p)          # (B, N, num_patches, patch_cells)

        # Base-num_colors encoding: each patch → single integer in [0, num_colors^(p^2))
        powers = (self.num_colors ** torch.arange(p * p, dtype=torch.long))
        patch_tokens = (g.long() * powers).sum(dim=-1)  # (B, N, num_patches)

        # Wrap each snapshot with start/end tokens
        start = torch.full((B, N, 1), self.start_tk, dtype=torch.long)
        end   = torch.full((B, N, 1), self.end_tk,   dtype=torch.long)
        tokens  = torch.cat([start, patch_tokens, end], dim=2)   # (B, N, grid_len)

        # Targets: -100 at start/end positions (don't predict special tokens)
        mask    = torch.full((B, N, 1), -100, dtype=torch.long)
        targets = torch.cat([mask, patch_tokens, mask], dim=2)   # (B, N, grid_len)

        return tokens.reshape(B, -1), targets.reshape(B, -1)


# =============================================================================
# NCA Dataset
# =============================================================================

class NCADataset(torch.utils.data.Dataset):
    """
    Wraps tokenized NCA sequences for causal language modeling.
    Each item is (input_ids, labels) of length max_seq_len.
    """
    def __init__(self, tokens: Tensor, targets: Tensor,
                 max_seq_len: int, min_grid: int, grid_len: int):
        self.tokens  = tokens   # (N_seqs, L)
        self.targets = targets  # (N_seqs, L), -100 at start/end positions
        self.max_seq_len = max_seq_len
        # First min_grid snapshots are used as context; mask their labels
        self.context_mask_len = min_grid * grid_len

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        seq = self.tokens[idx]   # (L,)
        tgt = self.targets[idx]  # (L,)

        # Causal shift: input seq[:-1], labels from tgt[1:]
        T = min(len(seq), self.max_seq_len + 1)
        input_ids = seq[:T - 1]
        labels    = tgt[1:T].clone()

        # Mask context snapshots
        mask_end = min(self.context_mask_len, len(labels))
        if mask_end > 0:
            labels[:mask_end] = -100

        # Pad shorter sequences
        L = len(input_ids)
        if L < self.max_seq_len:
            pad = self.max_seq_len - L
            input_ids = torch.cat([input_ids, torch.zeros(pad, dtype=torch.long)])
            labels    = torch.cat([labels,    torch.full((pad,), -100, dtype=torch.long)])

        return input_ids, labels


# =============================================================================
# GPT Model
# Copied from train.py with two adaptations:
#   1. GPTConfig uses plain defaults (not tied to module-level argparse globals)
#   2. Flash Attention 3 falls back gracefully to SDPA when unavailable
#   3. LOGIT_CAP stored as model instance variable instead of a global
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


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """q,k,v: (B, T, H, D). Uses FA3 on Hopper GPUs, falls back to SDPA otherwise."""
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    # SDPA fallback — full causal attention (window_size ignored; fine for seq_len=1024)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return y.transpose(1, 2)

flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)


@dataclass
class GPTConfig:
    sequence_len:  int   = 1024
    vocab_size:    int   = 10002
    n_layer:       int   = 30
    n_head:        int   = 14
    n_kv_head:     int   = 14
    n_embd:        int   = 1792
    window_pattern: str  = "SSSL"
    dropout:       float = 0.0


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
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd    = config.n_embd
        self.head_dim  = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q    = nn.Linear(self.n_embd, self.n_head    * self.head_dim, bias=False)
        self.c_k    = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v    = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ve_gate_channels   = 32
        self.attn_gate_channels = 12
        self.ve_gate   = (nn.Linear(self.ve_gate_channels,   self.n_kv_head, bias=False)
                          if has_ve(layer_idx, config.n_layer) else None)
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head,    self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve   = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v    = v + gate.unsqueeze(-1) * ve
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
        self.c_fc   = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.resid_dropout(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64,
                 logit_cap: float = 10.0):
        super().__init__()
        self.config    = config
        self.logit_cap = logit_cap
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h":   nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head       = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas    = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim   = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict({
            str(i): nn.Linear(config.n_embd, kv_dim, bias=False)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.encoder_layers = config.n_layer // 2
        self.skip_weights   = nn.Parameter(torch.ones(self.encoder_layers))
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._dupe_layers = None

    def set_dupe_layers(self, start, end, loops=2):
        assert start >= self.encoder_layers
        assert end <= self.config.n_layer
        self._dupe_layers = (start, end)
        self._dupe_loops  = loops

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
        self.resid_lambdas.fill_(1.0)
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
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _avg_causal_attended_keys(self, window, seq_len):
        if window < 0 or window >= seq_len - 1:
            return (seq_len + 1) / 2
        max_keys = min(window + 1, seq_len)
        return max_keys - max_keys * (max_keys - 1) / (2 * seq_len)

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = (self.transformer.wte.weight.numel()
                          + self.resid_lambdas.numel()
                          + self.x0_lambdas.numel()
                          + self.skip_weights.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = sum(12 * h * q * self._avg_causal_attended_keys(w[0], t) for w in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def _run_decoder_layers(self, x, x0, cos_sin, encoder_outputs, start, end):
        for i in range(start, end):
            j = self.config.n_layer - 1 - i
            if 0 <= j < self.encoder_layers:
                x = x + self.skip_weights[i - self.encoder_layers] * encoder_outputs[j]
            x  = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x  = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
        return x

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x  = norm(self.transformer.wte(idx))
        x0 = x

        # Encoder half
        encoder_outputs = []
        for i in range(self.encoder_layers):
            x  = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x  = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
            encoder_outputs.append(x)

        # Decoder half (with optional dupe layers)
        dupe = self._dupe_layers
        if dupe is None:
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                          self.encoder_layers, self.config.n_layer)
        else:
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                          self.encoder_layers, dupe[1])
            for _ in range(self._dupe_loops):
                x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs, dupe[0], dupe[1])
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                          dupe[1], self.config.n_layer)

        x      = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if self.logit_cap > 0:
            logits = self.logit_cap * torch.tanh(logits / self.logit_cap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                    ignore_index=-100, reduction=loss_reduction)
        return logits


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


def get_lr(step: int, total_steps: int, max_lr: float,
           warmup_ratio: float, warmdown_ratio: float) -> float:
    """3-phase LR: linear warmup → constant → linear warmdown (matches train.py schedule)."""
    warmup   = max(1, round(warmup_ratio   * total_steps))
    warmdown = max(1, round(warmdown_ratio * total_steps))
    if step < warmup:
        return max_lr * (step + 1) / warmup
    elif step <= total_steps - warmdown:
        return max_lr
    else:
        progress = (total_steps - step) / warmdown
        return max_lr * max(progress, 0.0)


def save_checkpoint(model: nn.Module, save_dir: str, epoch: int, config: GPTConfig,
                    master_process: bool):
    """Save all model parameters as float32 CPU tensors."""
    if not master_process:
        return
    os.makedirs(save_dir, exist_ok=True)
    raw = model
    if hasattr(raw, 'module'):         # DDP wrapper
        raw = raw.module
    if hasattr(raw, '_orig_mod'):      # torch.compile wrapper
        raw = raw._orig_mod
    state = {name: p.data.float().cpu() for name, p in raw.named_parameters()}
    state['config'] = {
        'n_layer': config.n_layer, 'n_head': config.n_head,
        'n_embd': config.n_embd,   'vocab_size': config.vocab_size,
        'window_pattern': config.window_pattern,
    }
    best_path  = os.path.join(save_dir, 'nca_pretrained_best.pt')
    epoch_path = os.path.join(save_dir, f'nca_pretrained_epoch{epoch:03d}.pt')
    torch.save(state, best_path)
    torch.save(state, epoch_path)
    print0(f"  Checkpoint saved → {best_path}")


@torch.no_grad()
def evaluate_val_loss(model: nn.Module, val_tokens: Tensor, val_targets: Tensor,
                      device: torch.device, autocast_ctx,
                      batch_size: int, actual_seq_len: int,
                      min_grid: int, grid_len: int,
                      ddp: bool, ddp_rank: int, ddp_world_size: int) -> float:
    """
    Evaluate mean cross-entropy loss on validation NCA sequences.
    Only rank 0 does the computation; result is broadcast to all ranks.
    """
    val_loss = torch.zeros(1, device=device)

    if ddp_rank == 0:
        # Use the inner model (without DDP/compile wrappers) on rank 0 for eval
        raw = model
        if hasattr(raw, 'module'):
            raw = raw.module
        raw.eval()
        val_dataset = NCADataset(val_tokens, val_targets, actual_seq_len, min_grid, grid_len)
        val_loader  = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        total_loss, total_valid = 0.0, 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                loss = raw(x, y, loss_reduction='sum')
            total_loss  += loss.item()
            total_valid += (y != -100).sum().item()
        raw.train()
        val_loss[0] = total_loss / max(total_valid, 1)

    if ddp:
        dist.broadcast(val_loss, src=0)

    return val_loss.item()


# =============================================================================
# Training Setup
# =============================================================================

ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
master_process = ddp_rank == 0
torch.manual_seed(42 + ddp_rank)

if ddp and torch.cuda.is_available():
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(42 + ddp_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_type  = device.type
autocast_ctx = (torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
                if device_type == "cuda" else nullcontext())
synchronize  = torch.cuda.synchronize if device_type == "cuda" else lambda: None

if _fa3 is not None:
    print0("Flash Attention 3 (Hopper GPU detected)")
else:
    print0("FA3 not available — using SDPA fallback (window_size ignored; OK for seq_len=1024)")

# NCA and tokenizer
nca       = NCA(grid_size=args.grid, d_state=args.num_colors,
                identity_bias=args.identity_bias, temperature=args.temperature)
tokenizer = NCATokenizer(patch=args.patch, num_colors=args.num_colors)
grid_len        = tokenizer.grid_len(args.grid)     # tokens per snapshot (e.g., 38)
num_examples    = args.seq_len // grid_len           # snapshots per sequence  (e.g., 26)
actual_seq_len  = num_examples * grid_len            # true sequence length    (e.g., 988)

print0(f"NCA: grid={args.grid}, patch={args.patch}, num_colors={args.num_colors}, "
       f"temperature={args.temperature}, dT={args.dT}")
print0(f"Vocab: {tokenizer.vocab_size} tokens "
       f"(padded to {((tokenizer.vocab_size + 63) // 64) * 64})")
print0(f"Sequence: {grid_len} tok/snapshot × {num_examples} snapshots = {actual_seq_len} tokens")

# Compute dataset sizes from token budgets
if args.tokens_per_epoch is not None:
    num_train_seqs = max(1, args.tokens_per_epoch // actual_seq_len)
else:
    num_train_seqs = max(1, (args.num_train_tokens // args.num_epochs) // actual_seq_len)
num_eval_seqs = max(1, args.num_eval_tokens // actual_seq_len)

num_train_rules = max(1, math.ceil(num_train_seqs / args.sims_per_rule))
num_eval_rules  = max(1, math.ceil(num_eval_seqs  / args.sims_per_rule))

tokens_per_epoch = num_train_seqs * actual_seq_len
total_tokens     = tokens_per_epoch * args.num_epochs

print0(f"\nDataset sizing:")
print0(f"  Train: {num_train_seqs} seqs × {actual_seq_len} tok = {tokens_per_epoch:,} tok/epoch "
       f"× {args.num_epochs} epochs = {total_tokens:,} total tokens")
print0(f"  Eval:  {num_eval_seqs} seqs × {actual_seq_len} tok = {num_eval_seqs * actual_seq_len:,} tokens (fixed)")
print0(f"  Rules: {num_train_rules} train, {num_eval_rules} eval "
       f"({args.sims_per_rule} sims/rule)")

# Generate rules (deterministic — all ranks get same result)
print0("\nGenerating NCA rules...")
if args.filter_rules:
    print0(f"  Filtering by gzip complexity ∈ [{args.filter_threshold}, {args.filter_upper_bound}]...")
    train_rules = generate_filtered_rules(
        nca, tokenizer, num_train_rules,
        args.filter_threshold, args.filter_upper_bound, base_seed=0)
    val_rules = generate_filtered_rules(
        nca, tokenizer, num_eval_rules,
        args.filter_threshold, args.filter_upper_bound,
        base_seed=num_train_rules * 10)
else:
    train_rules = list(range(num_train_rules))
    val_rules   = list(range(num_train_rules * 10, num_train_rules * 10 + num_eval_rules))

# Build model
config = GPTConfig(
    sequence_len   = args.seq_len,
    vocab_size     = tokenizer.vocab_size,
    n_layer        = args.n_layer,
    n_head         = args.n_head,
    n_kv_head      = args.n_head,
    n_embd         = args.n_embd,
    window_pattern = args.window_pattern,
    dropout        = args.dropout,
)
with torch.device("meta"):
    model = GPT(config, logit_cap=args.logit_cap)
model.to_empty(device=device)
model.init_weights()

param_counts      = sum(p.numel() for p in model.parameters())
transformer_params = sum(p.numel() for p in model.transformer.h.parameters())
print0(f"\nModel: {param_counts:,} params ({transformer_params:,} in transformer blocks)")

# Compile then DDP-wrap
orig_model = model
model      = torch.compile(model, dynamic=False)
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])

# GPU peak FLOPs for MFU
gpu_peak_flops = float('inf')
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "h100" in gpu_name:   gpu_peak_flops = 989e12
    elif "a100" in gpu_name: gpu_peak_flops = 312e12
    elif "4090" in gpu_name: gpu_peak_flops = 165.2e12
num_flops_per_token = orig_model.estimate_flops()
print0(f"FLOPs per token: {num_flops_per_token:e}")

# Optimizer (standard AdamW — simpler than Muon, fine for NCA phase)
optimizer = torch.optim.AdamW(
    orig_model.parameters(), lr=args.lr,
    weight_decay=args.weight_decay, betas=(0.9, 0.95),
    fused=(device_type == "cuda"),
)

# Gradient accumulation
tokens_per_fwdbwd  = args.device_batch_size * actual_seq_len * ddp_world_size
grad_accum_steps   = max(1, args.total_batch_size // tokens_per_fwdbwd)
effective_batch    = grad_accum_steps * tokens_per_fwdbwd
print0(f"\nBatch: device_bs={args.device_batch_size} × seq={actual_seq_len} × "
       f"world={ddp_world_size} × accum={grad_accum_steps} = {effective_batch:,} tokens/step")

# Estimate total steps for LR schedule
steps_per_epoch  = max(1, (num_train_seqs // ddp_world_size) // args.device_batch_size // grad_accum_steps)
num_total_steps  = steps_per_epoch * args.num_epochs
print0(f"~{steps_per_epoch} steps/epoch × {args.num_epochs} epochs = ~{num_total_steps} total steps")

def generate_train_data(epoch: int):
    grids  = generate_nca_dataset(train_rules, args.sims_per_rule, num_examples,
                                   args.dT, args.init_rollout_steps, nca, epoch=epoch)
    toks, tgts = tokenizer.encode(grids)
    toks, tgts = toks[:num_train_seqs], tgts[:num_train_seqs]
    return toks, tgts

# Generate initial training data (regenerated each epoch if --regen-data)
print0("\nGenerating NCA training data...")
train_tokens, train_tgts = generate_train_data(epoch=0)
print0(f"Train: {train_tokens.shape[0]} sequences, {train_tokens.numel():,} tokens")

print0("Generating NCA validation data (once)...")
val_grids            = generate_nca_dataset(val_rules, args.sims_per_rule, num_examples,
                                             args.dT, args.init_rollout_steps, nca, epoch=999)
val_tokens, val_tgts = tokenizer.encode(val_grids)
val_tokens = val_tokens[:num_eval_seqs]
val_tgts   = val_tgts[:num_eval_seqs]
print0(f"Eval:  {val_tokens.shape[0]} sequences, {val_tokens.numel():,} tokens")

# =============================================================================
# Training Loop
# =============================================================================

print0(f"\n{'='*60}")
print0(f"NCA Pre-Pre-Training")
print0(f"  LR={args.lr}, WD={args.weight_decay}, epochs={args.num_epochs}")
if args.max_train_time:
    print0(f"  Max train time: {args.max_train_time}m")
if args.max_wall_time:
    print0(f"  Max wall time:  {args.max_wall_time}m")
print0(f"{'='*60}\n")

# wandb
run_name = args.run if args.run else time.strftime("%Y%m%d_%H%M%S")
_wandb_kwargs = {"project": "nanochat", "name": run_name}
if args.wandb_group:
    _wandb_kwargs["group"] = args.wandb_group
wandb_run = DummyWandb() if not master_process else wandb.init(**_wandb_kwargs)
if master_process:
    wandb_run.log_code(".")

best_val_loss   = float('inf')
step            = 0
smooth_loss     = 0.0
total_step_time = 0.0
gc_frozen       = False

train_dataset = NCADataset(train_tokens, train_tgts, actual_seq_len, args.min_grid, grid_len)

for epoch in range(args.num_epochs):
    print0(f"--- Epoch {epoch + 1}/{args.num_epochs} ---")

    if args.regen_data and epoch > 0:
        print0(f"  Regenerating training data (epoch {epoch + 1})...")
        train_tokens, train_tgts = generate_train_data(epoch=epoch)
        train_dataset = NCADataset(train_tokens, train_tgts, actual_seq_len, args.min_grid, grid_len)
        print0(f"  Done: {train_tokens.shape[0]} sequences")

    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=ddp_world_size, rank=ddp_rank,
            shuffle=True, seed=epoch)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.device_batch_size, sampler=sampler,
            drop_last=True, pin_memory=(device_type == "cuda"),
            num_workers=2, persistent_workers=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.device_batch_size,
            shuffle=True, drop_last=True, pin_memory=(device_type == "cuda"),
            num_workers=2, persistent_workers=True)

    model.train()
    micro_step = 0
    loss_accum_t = torch.zeros(1, device=device)  # accumulate on GPU; single .item() after sync
    time_limit_hit = False
    t0 = time.time()

    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast_ctx:
            loss = model(x, y)
        (loss / grad_accum_steps).backward()
        loss_accum_t += loss.detach()
        micro_step += 1

        if micro_step % grad_accum_steps == 0:
            lr = get_lr(step, num_total_steps, args.lr, args.warmup_ratio, args.warmdown_ratio)
            for g in optimizer.param_groups:
                g['lr'] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            synchronize()
            dt = time.time() - t0
            total_step_time += dt
            t0 = time.time()

            step += 1
            ema_beta    = 0.9
            step_loss   = loss_accum_t.item() / grad_accum_steps  # single .item() after GPU sync
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * step_loss
            debiased    = smooth_loss / (1 - ema_beta ** step)
            tok_per_sec = int(effective_batch / max(dt, 1e-6))
            mfu         = 100 * num_flops_per_token * effective_batch / max(dt, 1e-6) / (gpu_peak_flops * ddp_world_size)
            train_t_str = f" | train_t: {total_step_time/60:.2f}m"
            wall_t_str  = f" | wall_t: {(time.time() - _script_start)/60:.2f}m"
            print0(f"  step {step:05d} | ep {epoch+1:02d} | loss {debiased:.6f} | "
                   f"lr {lr:.2e} | {tok_per_sec:,} tok/s | bf16_mfu: {mfu:.2f}%{train_t_str}{wall_t_str}")
            wandb_run.log({"step": step, "train/loss": debiased, "train/lr": lr, "train/tok_per_sec": tok_per_sec, "train/mfu": mfu})
            loss_accum_t.zero_()

            if not gc_frozen and step == 1:
                gc.collect(); gc.freeze(); gc.disable()
                gc_frozen = True

            # Time-based stopping
            if args.max_train_time is not None and total_step_time > args.max_train_time * 60:
                print0(f"  Max train time ({args.max_train_time}m) reached at step {step}, stopping.")
                time_limit_hit = True
                break
            if args.max_wall_time is not None and (time.time() - _script_start) > args.max_wall_time * 60:
                print0(f"  Max wall time ({args.max_wall_time}m) reached at step {step}, stopping.")
                time_limit_hit = True
                break

    # Validation
    val_loss = evaluate_val_loss(
        model, val_tokens, val_tgts, device, autocast_ctx,
        args.device_batch_size, actual_seq_len, args.min_grid, grid_len,
        ddp, ddp_rank, ddp_world_size)
    is_best = val_loss < best_val_loss
    print0(f"Epoch {epoch+1:02d} | Val Loss: {val_loss:.6f}{' *** BEST' if is_best else ''}")
    wandb_run.log({"step": step, "epoch": epoch + 1, "val/loss": val_loss})

    if is_best:
        best_val_loss = val_loss
        save_checkpoint(orig_model, args.save_dir, epoch, config, master_process)

    if time_limit_hit:
        break

# =============================================================================
# Summary
# =============================================================================

total_wall = time.time() - _script_start
print0(f"\n{'='*60}")
print0(f"NCA pre-pre-training complete")
print0(f"  Total steps:    {step}")
print0(f"  Best val loss:  {best_val_loss:.6f}")
print0(f"  Wall time:      {total_wall:.1f}s ({total_wall/60:.2f}m)")
print0(f"  Checkpoint:     {os.path.join(args.save_dir, 'nca_pretrained_best.pt')}")
print0(f"{'='*60}")

wandb_run.summary["best_val_loss"] = best_val_loss
wandb_run.finish()

if ddp and dist.is_initialized():
    dist.destroy_process_group()
