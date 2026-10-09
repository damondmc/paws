import sys
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from tqdm import tqdm

from paws.analysis.outlier import ResultAnalysisManager
from paws.filepaths import PathManager
from paws.workflow.manager import WorkflowManager

# 1. Load Configs
with open("/home/hoitim.cheung/galacticCenter/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("/home/hoitim.cheung/galacticCenter/config/gal.yaml", "r") as f:
    target = yaml.safe_load(f)

paths = PathManager(config, target)
manager = WorkflowManager(config, target)  # PathManager is initialized inside here too
result_manager = ResultAnalysisManager(config, target)

sat_band_list = [299, 302, 303, 306, 307]
THREADS = 16

fmin, fmax = 20, 400
f0_band = config["f0_band"]
cluster = True

#################################################################
is_injection = True
prev_stage = "injections-v2-2"
prev_tcoh = 20
prev_freq_deriv_order = 2

stage = "injections-v2-3"
tcoh = 40
freq_deriv_order = 3

n_sky = 57
# n_sky = 1
n_inj_max = None  # must match make_followup_dag.py (None = all)

if is_injection:
    num_toplist = 1
else:
    num_toplist = 10

# For non-injection runs, threshold on mean2F scaled by the injection-derived
# efficiency ratio between the previous and current follow-up stage instead of
# keeping every candidate.
prev_inj_stage = "injections-v2-0"
now_inj_stage = "injections-v2-1"

#################################################################

if not is_injection:
    threshold_filename = f"/home/hoitim.cheung/galacticCenter/config/{prev_inj_stage}_vs_{now_inj_stage}_threshold.txt"
    fs, fe, ratio_th, _ = np.loadtxt(threshold_filename).T
    band_step = fe[0] - fs[0]


# optional arguments: bands to process (default fmin..fmax)
freqs = [int(a) for a in sys.argv[1:]] or list(range(fmin, fmax))
for freq in tqdm(freqs, total=len(freqs)):
    if freq in sat_band_list:
        print(f"Skipping saturated band {freq} Hz")
        continue

    prev_taskname = f"{target['name']}_{prev_stage}_TCoh{prev_tcoh}_O{prev_freq_deriv_order}_{freq}Hz"

    taskname = f"{target['name']}_{stage}_TCoh{tcoh}_O{freq_deriv_order}_{freq}Hz"

    data_file = paths.outlier_file(freq, prev_taskname, prev_stage, cluster=cluster)

    if not data_file.is_file() or fits.getdata(data_file).size == 0:
        print(f"No outlier for {freq}Hz, skip.")
        continue

    if is_injection:
        mean2f_th = fits.getdata(data_file)["mean2F threshold"]
        mean2f_th = np.zeros_like(mean2f_th)[:n_inj_max]
    else:
        idx = int((freq - fs[0]) // band_step)
        mean2f_th = (fits.getdata(data_file)["mean2F"] - 4) * ratio_th[idx] + 4
        print(f"Freq={freq}Hz: th={ratio_th[idx]}")

    n_jobs = mean2f_th.size
    result_file = result_manager.make_outlier(
        taskname,
        freq,
        mean2f_th,
        n_jobs,
        num_toplist=num_toplist,
        stage=stage,
        freq_deriv_order=freq_deriv_order,
        n_sky=n_sky,
        cluster=cluster,
        work_in_local_dir=False,
        separate_saturated=False,
        is_injection=is_injection,
        max_workers=THREADS,
    )
