// polymarket-nvidia-ceo-forecast-01 entry v1 (Node 22, standard library only).
//
// Model: a short-horizon, per-market quantile forecast of each Yes price.
// Prediction-market prices are near-martingales, so the median stays at the
// checkpoint price; what varies per market is how far the price can wander,
// and it wanders in steps (often flat for minutes, then a tick, now and then
// a jump), not like a smooth diffusion.
//
// For each market, from its last 360 minutes of one-minute moves:
//   - the empirical quantiles of those moves (type-7 interpolation), and
//   - a normal shape with sigma = 0.5 * RMS(move) + 0.5 * 0.0025, the second
//     term a strong shrinkage toward the house baseline's 0.0025, so a quiet
//     market never collapses to a zero-width forecast and a loud one never
//     inherits a one-off jump wholesale.
// At horizon h the two shapes are averaged 50/50 and grown by h^0.4, not
// sqrt(h): on the practice, spreads grow distinctly slower than a random
// walk's (alpha 0.5 scores 0.345, alpha 0.4 scores 0.295, alpha 0.3 scores
// 0.283 on the same sigma). The 24 Yes prices are mutually exclusive
// outcomes, so each market's median is also pulled a tenth of the way toward
// x / S, where S is the sum of the case's last prices, growing linearly over
// the horizon: a small, bounded correction (it never hurt a practice case).
//
// Everything is read from the case file given: no dates, market ids, names
// or checkpoint times are hardcoded. The program is deterministic (no
// randomness at all) and fast: one pass over the prices per market.
//
// Practice backtest (local scorer, answers fetched from Polymarket's public
// CLOB history; the scorer reproduces the house baseline's 0.363589 exactly):
//   this model  loss 0.2758, median error 0.7460, 90% coverage 0.9132
//   house baseline loss 0.3636, median error 0.7675, coverage 0.9496
// It beats the baseline in every practice case (p00..p05).
//
// What was deliberately NOT fitted: no per-market or per-case tuning, no
// drift from recent trends (tested: it hurts), no MAD/robust sigma (tested:
// it underestimates, because most minutes are flat and the jumps are the
// signal), no 24-hour window (tested: stale jumps inflate sigma and the loss
// jumps to ~0.55), no full sum-to-one snap (the sum itself wanders between
// 0.81 and 1.09 over a case, so only a tenth of the way is taken).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Minutes of recent history used per market. */
const WINDOW_MINUTES = 360;
/** Weight on the empirical quantiles (1 - this on the normal shape). */
const EMPIRICAL_BLEND = 0.5;
/** Weight on the recent RMS sigma (1 - this on the baseline 0.0025). */
const SIGMA_RECENT_WEIGHT = 0.5;
/** The house baseline's spread, the shrinkage anchor. */
const BASELINE_SIGMA = 0.0025;
/** Floor on the blended sigma, so a flat market keeps some width. */
const SIGMA_FLOOR = 0.001;
/** Cap on the blended sigma: a safety net only. The wildest market in the
 *  practice (Colette Kress, p00) blends to 0.01085, so 0.012 never binds
 *  there; it only stops a pathological final case (a market jumping tens of
 *  points a minute) from producing an absurdly wide forecast. */
const SIGMA_CAP = 0.012;
/** Horizon growth exponent: spreads grow like h^ALPHA, not sqrt(h). */
const GROWTH_ALPHA = 0.4;
/** Fraction of the way each median is pulled toward x / S (sum-to-one). */
const SUM_CORRECTION = 0.1;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0, 1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Inverse of the standard normal CDF (Acklam's rational approximation). */
function normalQuantile(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

/** Type-7 quantile of a sorted array. */
function quantileSorted(sorted: number[], tau: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0];
  const pos = tau * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** One-minute moves of a market's prices, oldest to newest. */
function moves(prices: number[]): number[] {
  const d: number[] = [];
  for (let i = 1; i < prices.length; i++) d.push(prices[i] - prices[i - 1]);
  return d;
}

function rms(values: number[]): number {
  if (values.length === 0) return 0;
  let s = 0;
  for (const v of values) s += v * v;
  return Math.sqrt(s / values.length);
}

/** Enforce non-decreasing quantiles in a row, in place. */
function monotone(row: number[]): number[] {
  for (let j = 1; j < row.length; j++) if (row[j] < row[j - 1]) row[j] = row[j - 1];
  return row;
}

/**
 * The forecast of one market: horizon rows of one quantile per level in
 * `taus`. The median stays at the checkpoint price x; the spread blends the
 * empirical quantiles of the recent moves with a shrunk normal shape.
 */
function forecastMarket(
  m: CaseMarket,
  horizon: number,
  taus: number[],
  sumAtCheckpoint: number,
): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1];
  const recent = moves(m.prices).slice(-WINDOW_MINUTES);
  const sorted = [...recent].sort((a, b) => a - b);
  const empQ = taus.map((t) => quantileSorted(sorted, t));
  const sigma = Math.min(
    SIGMA_CAP,
    Math.max(
      SIGMA_FLOOR,
      SIGMA_RECENT_WEIGHT * rms(recent) + (1 - SIGMA_RECENT_WEIGHT) * BASELINE_SIGMA,
    ),
  );
  const z = taus.map((t) => normalQuantile(t));
  // The sum-to-one pull: toward x / S, linear in h, bounded by the weight.
  const target = sumAtCheckpoint > 0 ? Math.min(1, Math.max(0, x / sumAtCheckpoint)) : x;
  const pull = SUM_CORRECTION * (target - x);

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const growth = Math.pow(h, GROWTH_ALPHA);
    const shift = pull * (h / horizon);
    rows.push(monotone(taus.map((_, j) =>
      price6(x + shift + EMPIRICAL_BLEND * empQ[j] * growth + (1 - EMPIRICAL_BLEND) * z[j] * sigma * growth),
    )));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const sumAtCheckpoint = c.markets.reduce((s, m) => s + m.prices[m.prices.length - 1], 0);
  const forecasts = c.markets.map((m) => ({
    id: m.id,
    q: forecastMarket(m, c.horizon_minutes, c.quantiles, sumAtCheckpoint),
  }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
