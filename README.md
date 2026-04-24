# NanoGPT Slowrun
![Experiments](val_loss_animation.gif)

NanoGPT Slowrun is a new benchmark for language modeling algorithms in the infinite compute, fixed data regime: 100M tokens from FineWeb, no compute/time limit, lowest validation loss wins.[^1] We call it a Slowrun since the goal is to spend as much time with the data as we need to maximize learning on it. We deliberately choose this setting in contrast to speedruns like modded-nanogpt, which assume infinite data and optimize for wall-clock time on fixed hardware. Loved by [@karpathy](https://x.com/karpathy/status/2027099040073286087) himself! 

<img src="karpathy.png" alt="karpathy" width="600">

When speed is not the binding constraint, the space of promising algorithms changes dramatically--for example, large models trained with heavy regularization, expensive optimizers, and evolutionary search are all fair game. We want leaps like GPT-3, where previously unimaginable compute led to better generalization. That doesn't happen if wall-clock time is your constraint.

The baseline trains in \~47 minutes on 8xH100 (\~$12) and achieves 3.402 val loss. There are four tracks: 
1. a limited compute track capped at a single 8xH100 node for 1 hour (this is 100x the compute used by the Nanochat 1-epoch baseline),
2. a tiny compute track capped at a single 8xH100 node for 15 minutes,
3. a two hour track capped at a single 8xH100 node for two hours,
4. and an unlimited compute track with minimal restrictions on hardware or time. 

For now the limited track lives in the root directory, the tiny track lives at [tiny/](tiny/), the two hour track lives at [two_hour/](two_hour/), and the unlimited track lives at [unlimited/](unlimited/). The approaches pursued vary significantly based on the compute budget. Submit an entry by opening a PR.

## Running the current record 

You can reproduce the limited-compute record by running the following commands: 
```bash 
git clone https://github.com/qlabs-eng/slowrun.git && cd slowrun
pip install -r requirements.txt
python prepare_data.py
torchrun --standalone --nproc_per_node=8 train.py
```

## World Record History

We accept PRs that achieve a new World Record validation loss within the track's time limit, and add an entry below for each improvement.

### Limited Compute Track (1 hour) 

The limited-compute track caps runs at a single 8xH100 node for at most 1 hour. 

| # | Val Loss | Description | Date | Time | Script | Contributors |
| - | - | - | - | - | - | - |
1 | 3.402 | Baseline: 2.7B transformer, Muon, dropout 0.1, weight decay 1.6 | 02/26/26 | \~47 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/0d49316316dc6684049a679e03958c3fb612a8fd/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
2 | 3.376 | Add shuffling every epoch | 02/27/26 | \~47 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/106a290604abb6d8c5b0c3cc94c3b0eb6fe87dff/train.py) | [@kvegesna](https://x.com/karvegas_)
3 | 3.349 | Change value embed tables to projections from x0 | 03/01/26 | \~47 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/b261fba252920582076cf8c77dedf9251fe7f7ed/train.py) | [@ms337](https://x.com/madhavsinghal_)
4 | 3.335 | Use swiglu activation | 03/01/26 | 52.1 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/22d4a24ec53633c16d643779900ac3e9d10643a3/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
5 | 3.314 | Add U-Net architecture | 03/03/26 | 52.3 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/e463653a2b07790e0694bfaa6bdd7e6ee57cef64/train.py) | [@em-see-squared](https://github.com/em-see-squared)
6 | 3.295 | Add gating per attention head  | 03/03/26 | 53.3 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/52e7441f862c3295c0f5695933438dac78f7fc5b/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
7 | 3.285 | Repeat layers 15-20 for last 3 epochs, reduce warmdown | 03/11/26 | 53.3 mins (training time only) | [Script](https://github.com/qlabs-eng/slowrun/blob/7d8e580ab6a339079294562d000df3f7b1ce8c3c/train.py) | [@shmublu](https://x.com/ShmuelBerman)
8 | 3.278 | Run layers 15-20 3 times before layers 21-29 for the last 3 epochs | 03/11/26 | 55.7 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/64be4733075251c7da1d8b25529963520b16cdb8/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
9 | 3.276 | Add [exclusive self attention (XSA)](https://arxiv.org/pdf/2603.09078) | 03/12/26 | 57.7 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/ac968a62c633d75d972afa6d86a59f89e12997b9/train.py) | [@not-nonymous](https://github.com/not-nonymous)
10 | 3.270 | LR tuning, warmdown tuning | 03/16/26 | 55.5 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/28152d9996d9e910d4fb1b4a569fae399c546d6b/train.py) | [@zhiweixx](https://x.com/zhiweixux)
11 | 3.252 | EMA of weights, hyperparameter tuning | 03/18/26 | 59.2 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/3fb4428f67e77e3bec53d4eafbb2cfd6f999b684/train.py) | [@ChinmayK0607](https://x.com/ChinmayKak), [@ms337](https://x.com/madhavsinghal_)
12 | 3.248 | Use weighted average of last 3 epoch checkpoints | 03/23/26 | 58.2 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/20117fb36e62b65e96997268883f0b789a85e75d/train.py) | [@not-nonymous](https://github.com/not-nonymous)
13 | 3.236 | Add [Stochastic Weight Averaging (SWA)](https://arxiv.org/pdf/1803.05407) | 04/01/26 | 58.9 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/5f39cd82fbe58c5a0ab67af2f60a09e6356b88fe/train.py) | [@shmublu](https://x.com/ShmuelBerman)
14 | 3.230 | Switch c_proj init from zero to normal | 04/02/26 | 58.6 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/146042209ac9ffeff7809f721b2c95dbc6ef87d9/train.py) | [@ms337](https://x.com/madhavsinghal_)
15 | 3.227 | Add stochastic depth training | 04/06/26 | 58.5 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/038fa4b5ddcacd0d7232564ef3ff58cc97f759dc/train.py) | [@ChinmayK0607](https://x.com/ChinmayKak)
16 | 3.222 | Add multi-token prediction loss | 04/09/26 | 57.1 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/549997a53e88f9e6fed9ce18af86945e31257077/train.py) | [@clarkkev](https://github.com/clarkkev)
17 | 3.214 | Add [Interleaved Head Attention (IHA)](https://arxiv.org/abs/2602.21371) | 04/13/26 | 58.9 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/bd719090f61d88659bc21e9d7b2e69f781231e75/train.py) | [@ms337](https://x.com/madhavsinghal_)
18 | 3.211 | Add [MuonEq-R](https://arxiv.org/abs/2603.28254) | 04/17/26 | 59.4 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/a2ed9f4530d2dbd80b8cc0b8775659b196053277/train.py) | [@clarkkev](https://github.com/clarkkev)


### Tiny Track (15 minutes)

The tiny track caps runs at a single 8xH100 node for at most 15 minutes. 

| # | Val Loss | Description | Date | Time | Script | Contributors |
| - | - | - | - | - | - | - |
1 | 3.428 | Baseline: 300M transformer, weight decay 0.8, dropout 0.1 | 03/02/26 | 13.7 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/22c1618d843a692384b0329f309ddfb4b5df9ff6/tiny/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
2 | 3.410 | Add swiglu activation | 03/02/26 | 14.4 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/efa7f2ed81ac0b2aa9d5954c9b56ee56786c1934/tiny/train.py) | [@ChinmayK0607](https://x.com/ChinmayKak)
3 | 3.395 | Add U-Net architecture | 03/03/26 | 14.5 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/0e6280bb7f3673cf84e46a9b7cf7818b24511ed6/tiny/train.py) | [@em-see-squared](https://github.com/em-see-squared), [@akshayvegesna](https://x.com/akshayvegesna)
4 | 3.385 | Add gating per attention head | 03/04/26 | 14.6 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/781d005e8e99a8af0ee9ab356a4c543778730f6b/tiny/train.py) | [@ChinmayK0607](https://x.com/ChinmayKak)
5 | 3.383 | Update warmdown ratio | 03/06/26 | 14.6 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/56559aa8526708c107e1e28eb8fc4a1721bd9c67/tiny/train.py) | [@not-nonymous](https://github.com/not-nonymous)
6 | 3.374 | Half truncated RoPE, partial key offset, residual lambdas to 1.1 | 03/06/26 | 14.8 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/ed62160275273197c3a996c4469d735a05c5eedb/tiny/train.py) | [@ChinmayK0607](https://x.com/ChinmayKak)
7 | 3.365 | Add weight decay schedule | 03/15/26 | 14.8 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/42c39127d19bebbb806afd630fa852936da35562/tiny/train.py) | [@shmublu](https://x.com/ShmuelBerman)
8 | 3.353 | Add EMA parameter averaging | 03/18/26 | 14.9 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/0092945e4bf570fdb90fbdf2f1b311361288ea0b/tiny/train.py) | [@clarkkev](https://github.com/clarkkev)
9 | 3.345 | Add [Stochastic Weight Averaging (SWA)](https://arxiv.org/pdf/1803.05407) | 04/01/26 | 14.6 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/e094217e7a1abffae5b3ee06737b6d1747d76a48/tiny/train.py) | [@shmublu](https://x.com/ShmuelBerman)



### Two hour track

The two hour track caps runs at a single 8xH100 node for at most two hours.

| # | Val Loss | Description | Date | Time | Script | Contributors |
| - | - | - | - | - | - | - |
1 | 3.203 | Baseline, extending the 1 hour multi token prediction result | 04/12/26 | 110.6 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/7f7b417c4ea5abc9c63eb2e43495d66a1646fa11/two_hour/train.py) | [@ChinmayK0607](https://x.com/ChinmayKak)
2 | 3.197 | Add [Interleaved Head Attention (IHA)](https://arxiv.org/abs/2602.21371) | 04/14/26 | 115.0 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/f19b938eacddd1fb8731a42fc8b0a0ca29667686/two_hour/train.py) | [@ChinmayK0607](https://x.com/ChinmayKak), [@ms337](https://x.com/madhavsinghal_)
3 | 3.188 | Add [MuonEq-R](https://arxiv.org/abs/2603.28254) | 04/20/26 | 116.3 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/eec06c03996d1a568025624871540ba8cfe85274/two_hour/train.py) | [@ChinmayK0607](https://x.com/ChinmayKak)


### Unlimited Compute Track 

| # | Val Loss | Description | Date | Time | Script | Contributors |
| - | - | - | - | - | - | - |
1 | 3.402 | Baseline: 2.7B transformer, Muon, dropout 0.1, weight decay 1.6 | 02/26/26 | \~47 mins | [Script](https://github.com/qlabs-eng/slowrun/blob/0d49316316dc6684049a679e03958c3fb612a8fd/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
2 | 3.264 | Baseline: 8 × 2.7B transformer, Muon, dropout 0.1, weight decay 1.6, logit averaging | 02/27/26 | 6h 44m | [Script](https://github.com/qlabs-eng/slowrun/blob/106a290604abb6d8c5b0c3cc94c3b0eb6fe87dff/unlimited/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
3 | 3.218 | Use value projections and swiglu activation | 03/02/26 | 6h 54m | [Script](https://github.com/qlabs-eng/slowrun/blob/4681cfd6fa8266fc6cbbf2af947773e188599857/unlimited/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
4 | 3.185 | Add U-Net and Attention Gating | 03/04/26 | 7h 8m | [Script](https://github.com/qlabs-eng/slowrun/blob/bfe12a71d84a4102dcd1a2faaedfbd9aa1c417c0/unlimited/train.py) | [@akshayvegesna](https://x.com/akshayvegesna), [@em-see-squared](https://github.com/em-see-squared)
5 | 3.166 | Train each model for 1.5x longer | 03/05/26 | 10h 35m | [Script](https://github.com/qlabs-eng/slowrun/blob/6848b4a7b4d1373dead2c7ceaaf47927762b86c8/unlimited/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
6 | 3.126 | Train each model in ensemble to distill previous model + usual CE loss | 03/07/26 | 16h 1m | [Script](https://github.com/qlabs-eng/slowrun/blob/4eb2cce258b9edb97862f65349e130507d7c433c/unlimited/train.py) | [@not-nonymous](https://github.com/not-nonymous)
7 | 3.089 | Ensemble of 10 models, looping of layers 15-20, tuned epoch counts, loss weight | 03/13/26 | 19h 18m (2 nodes, 8xH100) | [Script](https://github.com/qlabs-eng/slowrun/blob/5c6ecd540cd789eef50fe894302da82b670fcc93/unlimited/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
8 | 3.081 | Ensemble of 12 models, distill alpha 0.5 | 03/18/26 | 42h 35m (1 node, 8xH100) | [Script](https://github.com/qlabs-eng/slowrun/blob/7dc6c386ae6ccf8bc8fa9b03cc53acf7723c4084/unlimited/train.py) | [@not-nonymous](https://github.com/not-nonymous)
9 | 3.045 | More looping, hyperparam tuning, model size increase | 03/19/26 | ~44h (2 nodes, 8xH100) | [Script](https://github.com/qlabs-eng/slowrun/blob/fcfb67b230afcf2e1702936abc077defcf5f8657/unlimited/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)
10 | 3.024 | Use probability averaging over logit averaging, train 20 models | 03/31/26 | 210 hours (7xH100 node) | [Script](https://github.com/qlabs-eng/slowrun/blob/dceb3e9d5b6153058b8dadf36af84f6a7c622fdf/unlimited/train.py) | [@L-z-Chen](https://x.com/lzchen_ut)
11 | 3.001 | Add MTP, IHA, MuonEq-R, adjust initialization, ensemble more models | 04/23/26 | 68 hours (4 nodes, 8xH100) | [Script](https://github.com/qlabs-eng/slowrun/blob/eed13a509d79d55553b6e0c71d6f7e386c5b3c4c/unlimited/train.py) | [@akshayvegesna](https://x.com/akshayvegesna)




## Why limited data, unlimited compute? 

The bitter lesson tells us that we should strongly prefer algorithms that scale with compute alone. We can't improve models at the rate compute scales as long as performance is bottlenecked by data.

This repo builds on [Nanochat](https://github.com/karpathy/nanochat), which took many ideas from the modded-nanogpt speedrun contest. To be fair, the speedrun contest did provide real data efficiency gains: using less data is one way to train faster. But because it sets speed as the binding constraint, it filters out an entire class of algorithms that yield learning gains. 

## Initial Baseline Approach (02/26/26)

Following Kim et al. (2025),[^2] we developed the initial baseline in three steps:

1. **Optimizer selection.** We tested popular optimizers in the data-limited regime, training for multiple epochs on the 100M tokens. Muon outperforms AdamW, SOAP, and MAGMA.

2. **Scaling up.** We increased model size but found diminishing returns due to the limited data. Without appropriate regularization, a 1.4B parameter model outperforms a 2.7B parameter model.

3. **Regularization.** When we scale up parameter size also using heavy weight decay, we recover monotonic improvements with scale. We further find that dropout improves performance on top of weight decay. Our final model[^3] is a 2.7B parameter transformer, with 1.2B parameters in the transformer trunk and heavy embedding defaults from Nanochat. It is trained with dropout 0.1 and weight decay 1.6. This weight decay is very large by traditional standards, but consistent with Kim et al. (2025), who find optimal weight decay is up to 30× larger than standard practice in the data-constrained regime.

Given the strong performance by large models that are well regularized, we speculate that larger models have a strong simplicity bias, amplified by regularization.

![Overparametrization](overparametrization.png)
*Figure taken from Andrew Gordon Wilson, ["Deep Learning is Not So Mysterious or Different."](https://arxiv.org/abs/2503.02113)*

## Why 100M tokens? 

We choose 100M tokens because it is small enough to affordably try radically different learning algorithms, while large enough that the winning techniques may work at a larger scale, though the scaling behavior is an open empirical question.

[^1]: For practical purposes, we begin by providing an upper bound on time of 64 H100's for 7 days. For reference, nanogpt can be trained for 1 epoch in 30s, so using this amount of compute would be 100,000x the compute used by that baseline.

[^2]: Konwoo Kim, Suhas Kotha, Percy Liang, and Tatsunori Hashimoto. ["Pre-training under infinite compute."](https://arxiv.org/abs/2509.14786) arXiv:2509.14786, 2025.

[^3]: These numbers from 02/26/26 are no longer accurate as of the latest world records. As of 04/08/26, the world record on the 1 hour track uses a 1.4B parameter model.
