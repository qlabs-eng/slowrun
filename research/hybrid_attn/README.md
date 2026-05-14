# Hybrid Attention

Hybrid attention mixes full softmax layers with linear-attention layers inside the same 30-layer trainer. The current trainer supports both GDN and KDA on the same alternating-layer layout, plus the newer hybrid-specific optimizer and dataloader controls that recent experiments depended on.

On this Hopper host, full GDN training should use `FLA_TILELANG=1`.

## Current Leaderboard

| Rank | Experiment | Backend | Best / Final Val Loss | Training Time | Peak Memory | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | EXP096 | GDN square | `3.228005 / 3.239774` | `87.99` min | `57417.80` MiB | current quality-best run, uses `--gdn-head-dim-mode square --muon-eq-r` |
| 2 | EXP087 | GDN | `3.230877 / 3.243014` | `80.38` min | `54820.06` MiB | param-matched quality baseline, adds `--muon-eq-r` |
| 3 | EXP112 | GDN square no-conv | `3.231137 / 3.243791` | `75.57` min | `55562.31` MiB | current square speed-quality record, EXP096 + `--gdn-no-conv` |
| 4 | EXP089 | GDN no-conv | `3.234275 / 3.246534` | `71.93` min | `53583.05` MiB | fastest GDN frontier, EXP087 + `--gdn-no-conv` |
| 5 | EXP081 | KDA | `3.239565 / 3.255612` | `89.28` min | `57735.46` MiB | slower quality reference line |
| 6 | EXP086 | GDN | `3.234646 / 3.247445` | `80.37` min | `54820.06` MiB | matched TileLang GDN control before MuonEq-R |

## Recommended Commands

Current quality-best default, based on EXP096:

```bash
FLA_TILELANG=1 torchrun --standalone --nproc_per_node=8 research/hybrid_attn/train.py \
  --gdn-layers 1,3,5,6,8,10,11,13,15,16,18,20,22,23 \
  --linear-attn-type gdn \
  --gdn-head-dim-mode square \
  --muon-eq-r
```

Current square speed-quality record, based on EXP112:

```bash
FLA_TILELANG=1 torchrun --standalone --nproc_per_node=8 research/hybrid_attn/train.py \
  --gdn-layers 1,3,5,6,8,10,11,13,15,16,18,20,22,23 \
  --linear-attn-type gdn \
  --gdn-head-dim-mode square \
  --muon-eq-r \
  --gdn-no-conv
```

Fastest GDN alternative, based on EXP089:

```bash
FLA_TILELANG=1 torchrun --standalone --nproc_per_node=8 research/hybrid_attn/train.py \
  --gdn-layers 1,3,5,6,8,10,11,13,15,16,18,20,22,23 \
  --linear-attn-type gdn \
  --muon-eq-r \
  --gdn-no-conv
```

Current KDA reference run, based on EXP081:

```bash
torchrun --standalone --nproc_per_node=8 research/hybrid_attn/train.py \
  --gdn-layers 1,3,5,6,8,10,11,13,15,16,18,20,22,23 \
  --linear-attn-type kda \
  --muon-eq-r
```

## What The Trainer Supports Now

- `--linear-attn-type {gdn,kda}` switches the linear-attention block on the selected hybrid layers.
- `--gdn-head-dim-mode {param-matched,square}` switches the GDN key width between `d_head/2` and `d_head`.
- `--gdn-no-conv` disables the GDN short-convolution path and is the current best runtime-saving knob.
- `--muon-eq-r` enables row-normalized Muon updates and is part of the current quality-best hybrid baseline.
- `--muon-ns-schedule {polar-express,deepseek-v4}` switches the Muon Newton-Schulz coefficient table.
- `--no-doc-shuffle` disables per-epoch document reshuffling when using the flat-token dataset format.
- `--grad-clip <value>` enables global gradient-norm clipping before the optimizer step.
- `--gdn-use-recurrent` remains experimental and should not be treated as a stable default.

## Practical Guidance

- Use EXP096 as the control for new quality-focused GDN follow-ups.
- Use EXP112 when runtime matters but the run should remain near the param-matched quality baseline. It saves `12.42` training minutes versus EXP096 and finishes within `0.0008` final loss of EXP087.
- Use EXP089 when runtime matters more than the last `0.003` of validation loss.
- Treat KDA as a reference backend, not the default deployment path on this host.
- For Hopper GDN runs, prefer `FLA_TILELANG=1`; the older non-TileLang GDN path is no longer the right frontier comparison.

## In-Flight Follow-Ups

- No GDN follow-up is required before treating EXP112 as the current square speed-quality record. A repeat seed would be useful if this line is promoted beyond the research track.
