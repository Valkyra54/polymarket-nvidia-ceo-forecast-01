# polymarket-nvidia-ceo-forecast-01 — v1

A short-horizon quantile forecaster for the Yes prices of Polymarket's
"Next CEO of NVIDIA?" event. For every market and every minute of the 4 hours
after a checkpoint, it writes five quantiles (5%, 25%, 50%, 75%, 95%) of that
minute's price.

## The model

Prediction-market prices are near-martingales, so the median stays at the
checkpoint price. What varies per market is how far the price can wander —
and it wanders in steps (often flat for minutes, then a tick, now and then a
jump), not like a smooth diffusion.

Per market, from its last 360 minutes of one-minute moves:

- the **empirical quantiles** of those moves (type-7 interpolation), and
- a **normal shape** with sigma = 0.5 · RMS(move) + 0.5 · 0.0025 — the second
  term a strong shrinkage toward the house baseline's 0.0025, so a quiet
  market never collapses to a zero-width forecast and a loud one never
  inherits a one-off jump wholesale. Sigma is floored at 0.001 and capped at
  0.012 (the wildest practice market blends to 0.01085, so the cap never
  binds there; it is only a guard against a pathological final case).

At horizon h the two shapes are averaged 50/50 and grown by **h^0.4**, not
√h: spreads grow distinctly slower than a random walk's. The 24 Yes prices
are mutually exclusive outcomes, so each market's median is also pulled a
tenth of the way toward x / S, where S is the sum of the case's last prices,
growing linearly over the horizon — a small, bounded correction.

Everything is read from the case file given: no dates, market ids, names or
checkpoint times are hardcoded. The program is deterministic (no randomness
at all) and runs a practice case in well under a second.

## Practice backtest

A local scorer (in `~/workspace/makefun/polymarket-dev/`, not part of this
repo) fetches the practice answers from Polymarket's public CLOB price
history and scores with the exact pinball formula. It reproduces the house
baseline's published practice numbers exactly (loss 0.363589, median error
0.767503, coverage 0.949642), so the comparison is on the same footing.

| model | loss | median error | 90% coverage |
|---|---|---|---|
| house baseline (persistence, σ=0.0025, √h) | 0.363589 | 0.7675 | 0.9496 |
| **this model (v1, shipped)** | **0.275821** | **0.7460** | **0.9132** |

Per case (v1 vs baseline): p00 0.4638 vs 0.5361, p01 0.3174 vs 0.3804,
p02 0.3480 vs 0.4084, p03 0.2584 vs 0.3686, p04 0.2120 vs 0.2988,
p05 0.1175 vs 0.2382 — v1 beats the baseline in **every** practice case.
The shipped program's own outputs were scored (not just the lab model):
identical to the lab within 0.00002. Runtime: ~0.3 s for all six practice
cases together (~0.05 s per case, far under the 8 s cap); two runs produce
byte-identical outputs for every case.

## Why it should beat the house on the final

The edge is structural, not fitted: per-market volatility from the recent
6 hours (not a single global sigma), step-aware empirical tails (not a pure
normal), and sub-√h spread growth. The choices that would have been
overfitting were tested and rejected, and the plateau is broad: nearby
configurations (window 3–6 h, blend 0.4–0.6, alpha 0.35–0.45) all score
0.277–0.283, so the entry does not depend on one knife-edge parameter.

## What was deliberately NOT fitted

- **No per-market or per-case tuning.** Every parameter is global.
- **No drift from recent trends.** A last-hour / last-4-hour slope, heavily
  shrunk, was tested and made the loss worse (0.55 vs 0.55 on a bad base;
  the median error rose from 0.7675 to 0.80).
- **No MAD / robust sigma.** It underestimates: most minutes are flat, so
  the median absolute move is ~0 and the jumps — which are the signal —
  get discarded (loss 0.32).
- **No 24-hour window.** Stale jumps inflate sigma; the loss jumps to
  ~0.55. The 6-hour window is the freshest that still has enough moves.
- **No full sum-to-one snap.** The sum of the 24 prices itself wanders
  (0.81–1.09 over a case, and it moved away from 1.0 in p01), so only a
  tenth of the way toward x / S is taken — it never hurt a practice case,
  but a full snap would be an anchored bet on a premise the data rejects.
- **No alpha = 0.5.** √h growth is what the baseline assumes; the practice
  says spreads grow slower (alpha 0.3–0.4 beats 0.5 by 0.05–0.06).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` in the input directory and writes `<id>.out` per case:
`{"forecasts":[{"id","q":[[240 rows x 5 numbers]]}]}`. Node 22, standard
library only, no network.
