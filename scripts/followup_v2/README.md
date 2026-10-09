# v2 follow-up chain (2026-10)

The `make_*` scripts below were merged into the `paws` CLI; their stage settings are entries of
`config/stages.yaml`. The files as run are in commit 617c672 (`git show 617c672:scripts/followup_v2/<file>`).

Scripts that produced the v2 injection chain and the v2 real-search follow-up, frozen
with the settings each stage actually ran with. The shared scripts one level up
(`make_inj_dag.py`, `make_followup_dag.py`, `make_followup_outlier.py`) keep changing,
so they are not a record. Run everything with `uv run python` from `paws/`.

Why v2: the f4dot coefficient in `PowerLawModel` was wrong (fixed in `paws` commit
5deeb9c), so the whole injection chain was regenerated, with 20 injections per 1 Hz band.

Strategy (same for injections and real search): t5 O2 and t10 O2 at the single GC sky
point; t20 O2 and t40 O3 on the 57-point sky grid `config/gc_sky_grid.txt`; from t10 on,
only the loudest candidate per parent is kept (`num_toplist = 1`).

All DAGs get the OSDF cleanup (PRE script + RETRY + periodic_remove) and periodic_release
from `paws` (`WorkflowManager.make_osdf_cleanup_dag`, `write_search_subfile`); the dag
list's last line is the stage's `all_nodes.dag`. Submit with
`dagFiles/onedagsub.sh <dag list>`.

## Injections (`injections-v2-*`)

| Step | Script | Notes |
|---|---|---|
| t5 O2 DAG | `make_injections_v2_0_dag.py` | h0 from `postprocess/data/*upperlimit-1pc-1skypt*`, looked up by frequency |
| t5 outliers | `make_injections_v2_0_outlier.py` | |
| t10 O2 DAG | `make_injections_v2_1_dag.py` | 1 sky point, 20 runs/job |
| t10 outliers | `make_injections_v2_1_outlier.py` | |
| t5→t10 ratio | `plot_v2_ratio.py 0-1` | → `config/injections-v2-0_vs_injections-v2-1_threshold.txt`, `plots/…_ratio.png` |
| t20 O2 DAG | `make_injections_v2_2_dag.py` | 57-pt grid, 114 runs/job. (This stage's cleanup was hand-made, before it was in `paws`.) |
| t20 outliers | `make_injections_v2_2_outlier.py` | `n_sky = 57` |
| t10→t20 ratio | `plot_v2_ratio.py 1-2` | → `config/injections-v2-1_vs_injections-v2-2_threshold.txt` |
| t40 O3 DAG | `make_injections_v2_3_dag.py` | 57-pt grid, 57 runs/job, 4 GB |
| t20→t40 ratio (preview) | `preview_injections_v2_3_ratio.py` | while 4 of 7,195 jobs were pending: loudest per injection read from the Weave files, those 4 left out → `config/injections-v2-2_vs_injections-v2-3_threshold_preview.txt` (1st pct 1.797/1.716/1.595/1.530) and `results/sat_followup/inj_ratio_v2-2_v2-3_preview.pkl` |
| t40 outliers | `make_injections_v2_3_outlier.py [bands]` | `n_sky = 57`, O3; bands can be split over several processes |
| t20→t40 ratio | `plot_v2_ratio.py 2-3` | → `config/injections-v2-2_vs_injections-v2-3_threshold.txt` (1st pct 1.797/1.716/1.595/1.530) |

Thresholds use the 1st percentile of (2F−4)_next / (2F−4)_prev in 4 bins
(20–100, 100–200, 200–300, 300–400 Hz; 20 Hz bins have too few injections at
20 inj/Hz), written as 19 rows of 20 Hz so `make_followup_outlier.py` can index them.

## Real search, regular candidates

| Step | Script | Notes |
|---|---|---|
| t10 outliers | `make_followup_v2_1_outlier.py` | reads the existing `followup-1` Weave results; v2 t5→t10 cut, top 1 per t5 parent. Written as stage `followup-1`, then moved by `split_followup_v2_1.sh` |
| move to `followup-v2-1` | `split_followup_v2_1.sh <stamp>` | stamp = a file dated when the outlier run **started** (run with `touch -d "2026-10-07 18:05"`); restores `followup-1`'s original outlier files |
| seed check | `check_followup_v2_1_seeds.py` | matches `followup-v2-1` clustered rows to the old `followup-1` rows that `followup-2` ran on → `results/followup-v2-2/seed_map_followup-v2-1_vs_followup-1.npz`. Result: 101,263 exact (reuse `followup-2`), 4,285 changed seeds |
| t20 DAG, changed seeds | `make_followup_v2_2_dag.py` | only changed seeds; `condorFiles/followup-v2-2/GalacticCenter/<f>/changed_rows.txt` gives the `followup-v2-1` row of each block of 57 jobs |
| t20 outliers | `make_followup_v2_2_outlier.py` | reads `followup-2` results for exact seeds (old row j ↔ files j·57+1 … (j+1)·57) and `followup-v2-2` results for changed seeds; row 0 per file, loudest cached in `results/followup-v2-2/loudest/`. Result: 4,150 of 105,548 seeds pass, 4,113 clustered, 82 bands |
| t40 O3 DAG | `make_followup_v2_3_dag.py` | stage `followup-v2-3`: 82 bands, 4,113 Condor jobs × 57 runs, 4 GB |

## Real search, saturated sub-bands

The loudest candidate of each saturated `search-0` Weave job is in the
`SEARCH-0_SAT_OUTLIER` extension of the unclustered `search-0` outlier file
(27,121 jobs = 2.38 % of all t5 jobs). Each is followed up on its own.

| Step | Script | Notes |
|---|---|---|
| t10 O2 DAG | `make_followup_v2_1_sat_dag.py` | job i ↔ row i of the saturated table, no clustering |
| t10 outliers | `make_followup_v2_1_sat_outlier.py` | same v2 t5→t10 cut, top 1 per seed; does not skip the saturated bands. Result: 1,403 of 27,121 seeds pass (5.2 %), 1,390 after clustering. Log in `results/followup-v2-1-sat/` |
| t20 O2 DAG | `make_followup_v2_2_sat_dag.py` | stage `followup-v2-2-sat`: the 1,390 clustered survivors on the 57-pt grid, 114 runs/job → 74 bands, 715 Condor jobs, 79,230 Weave runs |
| t20 outliers | `make_followup_v2_2_sat_outlier.py` | v2 t10→t20 cut, top 1 per seed, `n_sky = 57`, saturated bands not skipped. Result: 21 of 1,390 seeds pass (1.5 %), 21 after clustering, in 11 bands. Log in `results/followup-v2-2-sat/` |
| t40 O3 DAG | `make_followup_v2_3_sat_dag.py` | stage `followup-v2-3-sat`: the 21 clustered t20 survivors on the 57-pt grid, 57 runs/job, 4 GB → 11 bands, 21 Condor jobs, 1,197 Weave runs |
| t40 analysis + H1/L1 veto | `postprocess/sat_band_followup.ipynb` §5–6 | loudest of each seed's 57 files read in the notebook (cached). Preview ratio cut: 5 of 21 pass. Veto: r_HL = (2F_H1−4)/(2F_L1−4) must lie in the central 99 % of the t40 injections of its 100 Hz bin (`inj_t40_preview_det.npz` from `preview_injections_v2_3_ratio.py`); 0 of 21 do → **sat-band follow-up closed, no survivors** (independent of the final t20→t40 threshold) |
