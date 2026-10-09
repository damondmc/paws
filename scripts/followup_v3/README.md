# v3 follow-up

0.1% (2F−4) ratio cut and 0.1% H1/L1 window (central 99.9% of the injections) at every stage, per 100 Hz band
(20–100, 100–200, 200–300, 300–400 Hz). All 380 bands, including the non-saturated jobs of 299/302/303/306/307 Hz.
Cuts come from the v2 injections (`injections-v2-0` … `-3`). Stage settings are in `config/stages.yaml`.

| k | stage | Tcoh / order | sky | seeds | earlier runs reused |
|---|---|---|---|---|---|
| 0 | `search-0` | t5 O2 | GC | – | – |
| 1 | `followup-v3-1` | t10 O2 | 1 point | search-0 clustered outliers inside the t5 window | `followup-1` (all) |
| 2 | `followup-v3-2` | t20 O2 | 57-pt grid | `followup-v3-1` clustered | `followup-2`, `followup-v2-2` |
| 3 | `followup-v3-3` | t40 O3 | 57-pt grid | `followup-v3-2` clustered | `followup-v2-3` |

A seed reuses an earlier run when its (freq, f1dot, f2dot) equals one of that run's seeds; the other seeds run in
`followup-v3-k`. The seed → run map of stage k is `results/followup-v3-k/seed_map.npz`.

A seed of stage k passes when its loudest candidate over the stage's sky points has
`(2F_k − 4) / (2F_{k−1} − 4)` ≥ the ratio cut and log10 r_HL, r_HL = (2F_H1 − 4)/(2F_L1 − 4), inside the window.

## Commands (from `paws/`)

```bash
export PAWS_CONFIG_DIR=/home/hoitim.cheung/galacticCenter/config
for k in 0 1 2 3; do                # config/injections-v2-k_hl_window_v3.txt
  uv run paws cuts hl injections-v2-$k --out injections-v2-${k}_hl_window_v3.txt
done
for k in 1 2 3; do                  # config/injections-v2-(k-1)_vs_injections-v2-k_threshold_v3.txt
  uv run paws cuts ratio injections-v2-$((k-1)) injections-v2-$k --out injections-v2-$((k-1))_vs_injections-v2-${k}_threshold_v3.txt
done
S=scripts/followup_v3/followup_v3.py
uv run python $S outliers 1       # followup-v3-1 from the followup-1 results (no jobs)
uv run python $S dag 2            # DAG for the t20 seeds without an earlier result -> submit
uv run python $S outliers 2       # after the DAG
uv run python $S dag 3            # needs followup-v2-3 finished -> submit
uv run python $S outliers 3
```

`--bands f1 f2 …` restricts `dag` / `outliers` to some bands. Loudest rows read from the Weave files are cached in
`results/followup-v3-k/loudest/` (delete to re-read).

## Results

| stage | seeds | pass | clustered |
|---|---|---|---|
| search-0 (t5) H1/L1 window | 647,229 (clustered search-0 outliers) | 71,721 | 71,721 (already one per cluster) |
| followup-v3-1 (t10) | 71,721 | 16,654 | 15,727 |
