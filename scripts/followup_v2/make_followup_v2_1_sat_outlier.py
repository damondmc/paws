"""t10 outliers of the saturated-job seeds (stage followup-v2-1-sat).

Seeds: the loudest candidate of every saturated search-0 Weave job, stored one row
per job in the SEARCH-0_SAT_OUTLIER extension of the UNclustered search-0 outlier
file. Weave job i of followup-v2-1-sat followed up row i of that table.

Same cut as the regular t10 stage (followup-v2-1): a seed's t10 candidates must reach
(mean2F_t5 - 4) * r + 4, with r the v2 t5->t10 1st percentile of its frequency bin,
and only the loudest candidate per seed is kept (num_toplist = 1). The saturated
bands (299, 302, 303, 306, 307 Hz) are NOT skipped: that is where most seeds are.
"""
import numpy as np
import yaml
from astropy.io import fits
from tqdm import tqdm

from paws.analysis.outlier import ResultAnalysisManager
from paws.filepaths import PathManager

H = "/home/hoitim.cheung/galacticCenter/"
with open(H + "config/config.yaml") as f:
    config = yaml.safe_load(f)
with open(H + "config/gal.yaml") as f:
    target = yaml.safe_load(f)

paths = PathManager(config, target)
result_manager = ResultAnalysisManager(config, target)

#################################################################
fmin, fmax = 20, 400
THREADS = 32
prev_stage, prev_tcoh, prev_freq_deriv_order = "search-0", 5, 2
sat_hdu = "SEARCH-0_SAT_OUTLIER"
stage, tcoh, freq_deriv_order = "followup-v2-1-sat", 10, 2
n_sky = 1
num_toplist = 1  # loudest candidate per seed, as for the injections
threshold_file = H + "config/injections-v2-0_vs_injections-v2-1_threshold.txt"
#################################################################

fs, fe, ratio_th, _ = np.loadtxt(threshold_file).T
band_step = fe[0] - fs[0]

n_seeds = 0
for freq in tqdm(range(fmin, fmax), total=fmax - fmin):
    prev_taskname = f"{target['name']}_{prev_stage}_TCoh{prev_tcoh}_O{prev_freq_deriv_order}_{freq}Hz"
    taskname = f"{target['name']}_{stage}_TCoh{tcoh}_O{freq_deriv_order}_{freq}Hz"

    data_file = paths.outlier_file(freq, prev_taskname, prev_stage, cluster=False)
    seeds = fits.getdata(data_file, extname=sat_hdu) if data_file.is_file() else None
    if seeds is None or len(seeds) == 0:
        continue

    idx = int((freq - fs[0]) // band_step)
    mean2f_th = (seeds["mean2F"].astype(float) - 4) * ratio_th[idx] + 4
    n_seeds += len(seeds)
    print(f"Freq={freq}Hz: {len(seeds)} saturated seeds, ratio={ratio_th[idx]}")

    result_manager.make_outlier(
        taskname,
        freq,
        mean2f_th,
        len(seeds),
        num_toplist=num_toplist,
        stage=stage,
        freq_deriv_order=freq_deriv_order,
        n_sky=n_sky,
        cluster=True,
        work_in_local_dir=False,
        separate_saturated=False,
        is_injection=False,
        max_workers=THREADS,
    )

print(f"Done: {n_seeds} saturated seeds")
