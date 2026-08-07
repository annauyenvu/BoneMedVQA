# Experiments & Ablation

## Required ablations

| Experiment | Visual | Textual | Latent |
|------------|-------:|--------:|-------:|
| Baseline | No | No | No |
| V | Yes | No | No |
| T | No | Yes | No |
| L | No | No | Yes |
| V+T | Yes | Yes | No |
| V+L | Yes | No | Yes |
| T+L | No | Yes | Yes |
| V+T+L | Yes | Yes | Yes |

```bash
python scripts/run_ablation.py --experiments Baseline V T L VT VL TL VTL --epochs 1
```

Results CSV: `outputs/ablation/ablation_results.csv`

**Do not fill research metrics until real experiments complete.** Empty cells are intentional.

## Additional comparisons

- Mask vs box vs contour vs blur-background
- Concat vs cross-attention fusion
- With/without token compression
- With/without latent consistency loss
- With/without LoRA
- Knowledge graph / GNN only if relational data exists (disabled by default)
