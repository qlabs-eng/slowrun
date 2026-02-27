## Ensemble Track Leaderboard

The ensemble track allows aggregating ensembles of different models to minimize validation loss. Ensembling is separated because it provides a distinct advantage worth measuring independently. The baseline trains 8 × 2.7B transformers with different random seeds and averages logits at evaluation time. It takes 6 hours, 44 minutes.

| # | Val Loss | Description | Date | Time | Contributors |
| - | - | - | - | - | - |
1 | 3.264 | Baseline: 8 × 2.7B transformer, Muon, dropout 0.1, weight decay 1.6, logit averaging | 02/27/26 | 6h 44m | @akshayvegesna