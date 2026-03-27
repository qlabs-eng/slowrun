import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _shift_right(x):
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)


def straight_through_one_hot(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    indices = probs.argmax(dim=-1)
    hard = F.one_hot(indices, num_classes=logits.size(-1)).to(probs.dtype)
    return hard + probs - probs.detach()


def exclusive_counter_scan(deltas):
    return _shift_right(torch.cumsum(deltas.float(), dim=1)).to(deltas.dtype)


def soft_slot_memory(write_slots, write_values, write_strength, eps=1e-4):
    """
    Soft overwrite memory used only for optional ablations.

    Args:
        write_slots: (B, T, S), typically one-hot or near one-hot.
        write_values: (B, T, M), value payload written into the chosen slot.
        write_strength: (B, T, 1), write gate in [0, 1).

    Returns:
        memory_before: (B, T, S, M) slot state visible before each step.
        memory_after: (B, T, S, M) slot state after each step.
    """
    gate = (write_slots.float() * write_strength.float()).clamp_(0.0, 1.0 - eps)
    payload = gate.unsqueeze(-1) * write_values.float().unsqueeze(2)
    carry = (1.0 - gate).clamp_min_(eps)
    prefix = torch.cumprod(carry, dim=1)
    normalized = torch.cumsum(payload / prefix.unsqueeze(-1), dim=1)
    memory_after = prefix.unsqueeze(-1) * normalized
    memory_before = _shift_right(memory_after)
    return memory_before.to(write_values.dtype), memory_after.to(write_values.dtype)


def vectorized_slot_memory(write_slots, write_values):
    """
    Exact last-write-wins memory for hard slot assignments.

    Args:
        write_slots: (B, T, S), one-hot slot assignments.
        write_values: (B, T, M), payload written at each step.

    Returns:
        memory_before: (B, T, S, M) slot state visible before each step.
        memory_after: (B, T, S, M) slot state after each step.
    """
    write_mask = write_slots > 0.5
    B, T, S = write_mask.shape
    M = write_values.size(-1)
    times = torch.arange(1, T + 1, device=write_values.device, dtype=torch.long).view(1, T, 1)
    write_pos = times * write_mask.long()
    memory_after_pos = torch.cummax(write_pos, dim=1).values
    memory_before_pos = _shift_right(memory_after_pos)

    padded_values = torch.cat(
        [torch.zeros(B, 1, M, dtype=write_values.dtype, device=write_values.device), write_values],
        dim=1,
    )
    expanded_values = padded_values.unsqueeze(2).expand(B, T + 1, S, M)
    before_idx = memory_before_pos.unsqueeze(-1).expand(B, T, S, M)
    after_idx = memory_after_pos.unsqueeze(-1).expand(B, T, S, M)
    memory_before = torch.gather(expanded_values, 1, before_idx)
    memory_after = torch.gather(expanded_values, 1, after_idx)
    return memory_before, memory_after


def naive_slot_memory(write_slots, write_values):
    slots = write_slots.argmax(dim=-1)
    B, T, S = write_slots.shape
    M = write_values.size(-1)
    state = torch.zeros(B, S, M, dtype=torch.float32, device=write_values.device)
    outputs = []
    for t in range(T):
        outputs.append(state.clone())
        batch_idx = torch.arange(B, device=write_values.device)
        state[batch_idx, slots[:, t]] = write_values[:, t].float()
    memory_before = torch.stack(outputs, dim=1)
    return memory_before.to(write_values.dtype)


class FrozenTraceInterpreter(nn.Module):
    """
    A learnable controller around a frozen append-only memory/counter substrate.

    The read/write/counter semantics are hand-coded and parameter-free. The
    surrounding projections remain trainable.
    """

    def __init__(
        self,
        model_dim,
        num_slots,
        memory_dim,
        counter_dim,
        dropout,
        temperature=1.0,
        hard_slots=True,
        counter_scale=0.25,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.counter_dim = counter_dim
        self.temperature = temperature
        self.hard_slots = hard_slots
        self.counter_scale = counter_scale

        self.read_proj = nn.Linear(model_dim, num_slots, bias=False)
        self.write_proj = nn.Linear(model_dim, num_slots, bias=False)
        self.write_strength_proj = nn.Linear(model_dim, 1, bias=True)
        self.write_value_proj = nn.Linear(model_dim, memory_dim, bias=False)
        self.counter_delta_proj = nn.Linear(model_dim, counter_dim, bias=False)
        self.counter_gate_proj = nn.Linear(model_dim, counter_dim, bias=True)
        self.output_gate_proj = nn.Linear(model_dim, 1, bias=True)
        self.output_proj = nn.Linear(memory_dim + counter_dim, model_dim, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer("slot_ids", torch.arange(num_slots, dtype=torch.float32), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        s = 1.0 / math.sqrt(self.model_dim)
        torch.nn.init.uniform_(self.read_proj.weight, -s, s)
        torch.nn.init.uniform_(self.write_proj.weight, -s, s)
        torch.nn.init.uniform_(self.write_value_proj.weight, -s, s)
        torch.nn.init.uniform_(self.counter_delta_proj.weight, -s, s * 0.5)
        torch.nn.init.normal_(self.output_proj.weight, mean=0.0, std=1e-4)
        torch.nn.init.zeros_(self.counter_gate_proj.weight)
        torch.nn.init.zeros_(self.output_gate_proj.weight)
        torch.nn.init.zeros_(self.write_strength_proj.weight)
        self.write_strength_proj.bias.data.fill_(-3.0)
        self.counter_gate_proj.bias.data.fill_(-1.0)
        self.output_gate_proj.bias.data.fill_(-2.0)

    def _slot_probs(self, logits):
        if self.hard_slots:
            return straight_through_one_hot(logits, self.temperature)
        return F.softmax(logits / self.temperature, dim=-1)

    def forward(self, x):
        h = x.float()
        read_slots = self._slot_probs(self.read_proj(h))
        write_slots = self._slot_probs(self.write_proj(h))
        write_values = self.write_value_proj(h) * torch.sigmoid(self.write_strength_proj(h))
        if self.hard_slots:
            memory_before, _ = vectorized_slot_memory(write_slots, write_values)
        else:
            memory_before, _ = soft_slot_memory(write_slots, write_values, torch.sigmoid(self.write_strength_proj(h)))
        readout = (read_slots.unsqueeze(-1) * memory_before).sum(dim=2)

        counter_deltas = self.counter_scale * torch.tanh(self.counter_delta_proj(h))
        counter_state = exclusive_counter_scan(counter_deltas)
        counter_gate = torch.sigmoid(self.counter_gate_proj(h))
        counter_state = counter_state * counter_gate

        features = torch.cat([readout, counter_state], dim=-1)
        out = self.output_proj(features.to(x.dtype))
        out = out * torch.sigmoid(self.output_gate_proj(h)).to(out.dtype)
        return self.resid_dropout(out)
