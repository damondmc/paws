import numpy as np
import yaml
from paws.params.search import SearchParamGenerator
from paws.params.models import PowerLawModel
from paws.workflow.manager import WorkflowManager
from paws.analysis.outlier import ResultAnalysisManager
from paws.filepaths import PathManager
from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from paws.analysis.tools import detection_stat_threshold

# 1. Load Configs
with open("/home/hoitim.cheung/galacticCenter/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("/home/hoitim.cheung/galacticCenter/config/gal.yaml", "r") as f:
    target = yaml.safe_load(f)

paths = PathManager(config, target)
manager = WorkflowManager(config, target)  # PathManager is initialized inside here too
result_manager = ResultAnalysisManager(config, target)

THREADS = 32
fmin, fmax = 20, 400
f0_band = config["f0_band"]

stage = "search-0"
coh_day = 5
freq_deriv_order = 2
n_seg = 107
num_top_list = 2000  # Weave toplist-limit of search-0
n_jobs = np.zeros(fmax - fmin)

for i, freq in enumerate(range(fmin, fmax)):
    if freq < 200:
        age = 300
    else:
        age = 300 + (freq - 199) * 0.5

    tau = 86400 * 365.25 * age

    model = PowerLawModel(nc_min=config["nc_min"], nc_max=config["nc_max"], tau=tau)
    generator = SearchParamGenerator(model, freq_deriv_order)

    params = generator.generate_parameters(
        target["alpha"],
        target["dalpha"],
        target["delta"],
        target["ddelta"],
        freq,
        freq + 1,
        df=f0_band,
        df1=5.0e-10,
        df2=1.0e-17,
    )

    n_jobs[i] = params[freq].data.size
n_jobs = n_jobs.astype(int)


def get_header_val(args):
    freq, taskname, job_index, stage = args
    filename = paths.weave_output_file(freq, taskname, job_index, stage)
    return fits.getheader(filename)["NSEMITPL"]


n_temp = np.zeros(n_jobs.size)

with ThreadPoolExecutor(max_workers=THREADS) as executor:
    for i, freq in tqdm(enumerate(range(fmin, fmax)), total=(fmax - fmin)):
        taskname = (
            f"{target['name']}_{stage}_TCoh{coh_day}_O{freq_deriv_order}_{freq}Hz"
        )

        args_list = [
            (freq, taskname, job_index, stage) for job_index in range(1, n_jobs[i] + 1)
        ]

        results = executor.map(get_header_val, args_list)
        n_temp[i] = sum(results)

        print(f"{freq} Hz: ntemp={n_temp[i]}")


for i, freq in tqdm(enumerate(range(fmin, fmax)), total=(fmax - fmin)):
    taskname = f"{target['name']}_{stage}_TCoh{coh_day}_O{freq_deriv_order}_{freq}Hz"

    mean2f_th = detection_stat_threshold(n_temp[i], n_seg)

    result_file = result_manager.make_outlier(
        taskname,
        freq,
        mean2f_th,
        n_jobs[i],
        num_toplist=num_top_list,
        stage=stage,
        freq_deriv_order=freq_deriv_order,
        cluster=True,
        work_in_local_dir=False,
        separate_saturated=True,
        max_workers=THREADS,
    )
