# v3 follow-up

0.1% (2F−4) ratio cut and 0.1% H1/L1 window (central 99.9% of the injections) at every stage, per 50 Hz band
(20–50, 50–100, …, 350–400 Hz). All 380 bands, including the non-saturated jobs of 299/302/303/306/307 Hz.
Cuts come from the v2 injections (`injections-v2-0` … `-3`).

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
S=scripts/followup_v3/followup_v3.py
uv run python $S cuts             # config/*_threshold_v3.txt, config/injections-v2-*_hl_window_v3.txt
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
| search-0 (t5) H1/L1 window | 647,229 (clustered search-0 outliers) | 65,796 | 65,796 (already one per cluster) |
| followup-v3-1 (t10) | 65,796 | 12,553 | 11,928 |
