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
manager = WorkflowManager(config, target)
result_manager = ResultAnalysisManager(config, target)

THREADS = 32
fmin, fmax = 20, 400
cluster = True

prev_stage = "search-0"
prev_tcoh = 5
prev_freq_deriv_order = 2
stage = "injections-v2-0"
tcoh = 5
freq_deriv_order = 2
n_inj = 20
n_sky = 1

for freq in tqdm(range(fmin, fmax), total=(fmax - fmin)):
    prev_taskname = f"{target['name']}_{prev_stage}_TCoh{prev_tcoh}_O{prev_freq_deriv_order}_{freq}Hz"
    taskname = f"{target['name']}_{stage}_TCoh{tcoh}_O{freq_deriv_order}_{freq}Hz"

    data_file = paths.outlier_file(freq, prev_taskname, prev_stage, cluster=cluster)

    try:
        mean2f_th = fits.getval(data_file, "mean2F_th", 0)
    except FileNotFoundError as e:
        print(f"Error loading previous stage outlier file for {freq} Hz: {e}")
        continue

    result_file = result_manager.make_outlier(
        taskname,
        freq,
        mean2f_th,
        n_inj,
        num_toplist=1,
        stage=stage,
        freq_deriv_order=freq_deriv_order,
        n_sky=n_sky,
        cluster=cluster,
        work_in_local_dir=False,
        separate_saturated=False,
        is_injection=True,
        max_workers=THREADS,
    )
