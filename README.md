# NanoGPT Slowrun
![Experiments](val_loss_animation.gif)

NanoGPT Slowrun is a new benchmark for language modeling algorithms in the infinite compute, fixed data regime: 100M tokens from FineWeb, no compute/time limit, lowest validation loss wins.[^1] We call it a Slowrun since the goal is to spend as much time with the data as we need to maximize learning on it. We deliberately choose this setting in contrast to speedruns like modded-nanogpt, which assume infinite data and optimize for wall-clock time on fixed hardware. Loved by [@karpathy](https://x.com/karpathy/status/2027099040073286087) himself! 

<img src="karpathy.png" alt="karpathy" width="600">

When speed is not the binding constraint, the space of promising algorithms changes dramatically--for example, large models trained with heavy regularization, expensive optimizers, and evolutionary search are all fair game. We want leaps like GPT-3, where previously unimaginable compute led to better generalization. That doesn't happen if wall-clock time is your constraint.

The baseline trains in \~47 minutes on 8xH100 (\~$12) and achieves 3.402 val loss. There are two tracks: 
1. a limited compute track capped at a single 8xH100 node for 1 hour (this is 100x the compute used by the Nanochat 1-epoch baseline),
2. and an unlimited compute track with minimal restrictions on hardware or time. 

For now the limited track lives in the root directory, and the unlimited track lives at [unlimited/](unlimited/). Submit an entry by opening a PR.

## Leaderboards

### Limited Compute 

The limited-compute track caps runs at a single 8xH100 node for at most 1 hour. 

| # | Val Loss | Description | Date | Time | Script | Contributors |
| - | - | - | - | - | - | - |
1 | 3.402 | Baseline: 2.7B transformer, Muon, dropout 0.1, weight decay 1.6 | 02/26/26 | \~47 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/0d49316316dc6684049a679e03958c3fb612a8fd/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
2 | 3.376 | Add shuffling every epoch | 02/27/26 | \~47 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/106a290604abb6d8c5b0c3cc94c3b0eb6fe87dff/train.py) | [@kvegesna](https://x.com/karvegas_)

### Unlimited Compute 

| # | Val Loss | Description | Date | Time | Script | Contributors |
| - | - | - | - | - | - | - |
1 | 3.402 | Baseline: 2.7B transformer, Muon, dropout 0.1, weight decay 1.6 | 02/26/26 | \~47 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/0d49316316dc6684049a679e03958c3fb612a8fd/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
2 | 3.264 | Baseline: 8 × 2.7B transformer, Muon, dropout 0.1, weight decay 1.6, logit averaging | 02/27/26 | 6h 44m | [Script](https://github.com/qlabs-eng/slowrun/blob/106a290604abb6d8c5b0c3cc94c3b0eb6fe87dff/unlimited/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)

## Why limited data, unlimited compute? 

The bitter lesson tells us that we should strongly prefer algorithms that scale with compute alone. We can't improve models at the rate compute scales as long as performance is bottlenecked by data.

This repo builds on [Nanochat](https://github.com/karpathy/nanochat), which took many ideas from the modded-nanogpt speedrun contest. To be fair, the speedrun contest did provide real data efficiency gains: using less data is one way to train faster. But because it sets speed as the binding constraint, it filters out an entire class of algorithms that yield learning gains. 

## Baseline Approach 

Following Kim et al. (2025),[^2] we developed the baseline in three steps:

1. **Optimizer selection.** We tested popular optimizers in the data-limited regime, training for multiple epochs on the 100M tokens. Muon outperforms AdamW, SOAP, and MAGMA.

2. **Scaling up.** We increased model size but found diminishing returns due to the limited data. Without appropriate regularization, a 1.4B parameter model outperforms a 2.7B parameter model.

3. **Regularization.** When we scale up parameter size also using heavy weight decay, we recover monotonic improvements with scale. We further find that dropout improves performance on top of weight decay. Our final model is a 2.7B parameter transformer, with 1.2B parameters in the transformer trunk and heavy embedding defaults from Nanochat. It is trained with dropout 0.1 and weight decay 1.6. This weight decay is very large by traditional standards, but consistent with Kim et al. (2025), who find optimal weight decay is up to 30× larger than standard practice in the data-constrained regime.

Given the strong performance by large models that are well regularized, we speculate that larger models have a strong simplicity bias, amplified by regularization.

![Overparametrization](overparametrization.png)
*Figure taken from Andrew Gordon Wilson, ["Deep Learning is Not So Mysterious or Different."](https://arxiv.org/abs/2503.02113)*

## Why 100M tokens? 

We choose 100M tokens because it is small enough to affordably try radically different learning algorithms, while large enough that the winning techniques may work at a larger scale, though the scaling behavior is an open empirical question.

[^1]: For practical purposes, we begin by providing an upper bound on time of 64 H100's for 7 days. For reference, nanogpt can be trained for 1 epoch in 30s, so using this amount of compute would be 100,000x the compute used by that baseline.

[^2]: Konwoo Kim, Suhas Kotha, Percy Liang, and Tatsunori Hashimoto. ["Pre-training under infinite compute."](https://arxiv.org/abs/2509.14786) arXiv:2509.14786, 2025.

