#!/opt/paws/.venv/bin/python
import numpy as np
import yaml
from astropy.io import fits
from tqdm import tqdm

from paws.definitions import phase_param_name
from paws.filepaths import PathManager
from paws.workflow.manager import WorkflowManager

# 1. Load Configs
config_file = "/home/hoitim.cheung/galacticCenter/config/config.yaml"
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

target_file = "/home/hoitim.cheung/galacticCenter/config/gal.yaml"
with open(target_file, "r") as f:
    target = yaml.safe_load(f)

manager = WorkflowManager(
    config=config, target=target
)  # PathManager is initialized inside here too
paths = PathManager(config=config, target=target)

# OSG worker nodes can't read home_dir directly, so config/target files must
# also be staged on OSDF (like exe/metric_file/image below) and transferred
# via the osdf:// URL rather than the home_dir path used above for loading.
config_file_osdf = paths.to_osdf_url(
    "/osdf/igwn/cit/staging/hoitim.cheung/config/config.yaml"
)
target_file_osdf = paths.to_osdf_url(
    "/osdf/igwn/cit/staging/hoitim.cheung/config/gal.yaml"
)

obs_day = 630
coh_day = 5
stage = "upperlimit-1pc-1skypt"
freq_deriv_order = 2
inj_freq_deriv_order = 4
prev_stage = "search-0"
prev_tcoh = 5
prev_freq_deriv_order = 2
# spacing > sky_radius: a single sky point
sky_radius = 1.25e-4
spacing_alpha = 2.5e-4
spacing_delta = 2.5e-4
n_inj = 200
num_toplist = 1

fmin, fmax = 20, 400
metric_file = "osdf:///igwn/cit/staging/hoitim.cheung/metricSetup/Start1368970000_TCoh432000_N107_Spin2.fts"

exe = None  # None: paws.upperlimit in the image

image = "osdf:///igwn/cit/staging/hoitim.cheung/images/paws.sif"

h0_file = "/home/hoitim.cheung/gc/results/upperLimit-rt-200inj/NGC6544/NGC6544_upperLimit-rt-200inj_20-475Hz_clustered.txt"
freq, h0_arr, _ = np.loadtxt(h0_file).T

_, freq_deriv_param_names = phase_param_name(freq_deriv_order)

dag_list_path = f"/home/hoitim.cheung/galacticCenter/dagFiles/{stage}_{target['name']}_dag{fmin}-{fmax}Hz.txt"

skipped_freqs = []

with open(dag_list_path, "w") as f_daglist:
    for freq, h0_est in tqdm(
        zip(range(fmin, fmax), h0_arr), desc="Generating DAGs", total=fmax - fmin
    ):
        # --- Collect SFT files ---
        sft_files = []
        files = paths.sft_ensemble(freq)
        sft_files.extend(files)  # Use extend, not append, to flatten the list

        data_taskname = f"{target['name']}_{prev_stage}_TCoh{prev_tcoh}_O{prev_freq_deriv_order}_{freq}Hz"

        data_file = paths.outlier_file(freq, data_taskname, prev_stage, cluster=True)

        try:
            non_sat_bands = fits.getdata(data_file, extname="non_sat_band")[
                "non_sat_band"
            ]
        except FileNotFoundError as e:
            print(f"Error loading previous stage outlier file for {freq} Hz: {e}")
            skipped_freqs.append(freq)
            continue

        if len(non_sat_bands) == 0:
            print(
                f"Warning: No non-saturated bands found for {freq} Hz. Skipping DAG generation."
            )
            skipped_freqs.append(freq)
            continue

        df_grid = [
            fits.getval(data_file, param_name, 0)
            for param_name in freq_deriv_param_names
        ]

        mean2f_th = fits.getval(data_file, "mean2F_th", 0)
        # --- Make DAG ---
        # Generate task name
        taskname = (
            f"{target['name']}_{stage}_TCoh{coh_day}_O{freq_deriv_order}_{freq}Hz"
        )

        dagfile = manager.make_upperlimit_dag(
            config_file_osdf,
            target_file_osdf,
            taskname,
            freq,
            stage,
            freq_deriv_order,
            sft_files,
            metric_file,
            mean2f_th,
            non_sat_bands,
            exe,
            df_grid=df_grid,
            inj_freq_deriv_order=inj_freq_deriv_order,
            num_toplist=num_toplist,
            sky_radius=sky_radius,
            spacing_alpha=spacing_alpha,
            spacing_delta=spacing_delta,
            h0_est=h0_est,
            n_inj=n_inj,
            request_memory="16GB",
            request_disk="6GB",
            request_cpu=8,
            cluster=True,
            work_in_local_dir=True,
            save_intermediate=False,
            image=image,
        )

        f_daglist.write(f"{dagfile}\n")

if skipped_freqs:
    print(f"Skipped frequencies due to no non-saturated bands: {skipped_freqs}")
