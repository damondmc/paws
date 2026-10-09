import yaml
from astropy.io import fits
from tqdm import tqdm

from paws.filepaths import PathManager
from paws.workflow.manager import WorkflowManager

# 1. Load Configs
CONFIG_FILE = "/home/hoitim.cheung/galacticCenter/config/config.yaml"
TARGET_FILE = "/home/hoitim.cheung/galacticCenter/config/gal.yaml"

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

with open(TARGET_FILE, "r") as f:
    target = yaml.safe_load(f)

paths = PathManager(config, target)
manager = WorkflowManager(config, target)  # PathManager is initialized inside here too

# OSG worker nodes can't pull files back through a direct connection to the
# access point, so config/target files must also be staged on OSDF (like exe
# /image below) and transferred via the osdf:// URL, not the home_dir path
# used above for loading.
config_file_osdf = paths.to_osdf_url(
    "/osdf/igwn/cit/staging/hoitim.cheung/config/config.yaml"
)
target_file_osdf = paths.to_osdf_url(
    "/osdf/igwn/cit/staging/hoitim.cheung/config/gal.yaml"
)

# freq 298 - 310 not yet ready, so start from 310
sat_band_list = [299, 302, 303, 306, 307]
fmin, fmax = 100, 120
cluster = True

#################################################################
is_injection = True
prev_stage = "injections-2-o3"
prev_tcoh = 20
prev_freq_deriv_order = 3

stage = "injections-3-o3"
tcoh = 40
freq_deriv_order = 3

n_sky = 1

if is_injection:
    num_toplist = 1
else:
    num_toplist = 10

zero_threshold = False  # follow-up stage keeps all candidates instead of re-thresholding
max_workers = 32

request_memory = "8GB"
request_disk = "8GB"
request_cpu = 1

exe = "osdf:///igwn/cit/staging/hoitim.cheung/scripts/make_outlier_cli.py"
image = "osdf:///igwn/cit/staging/hoitim.cheung/images/paws_v10.sif"
#################################################################

home_dir = config["home_dir"]
dag_list_path = f"{home_dir}dagFiles/{stage}_{target['name']}_dag{fmin}-{fmax}Hz.txt"

skipped_freqs = []

with open(dag_list_path, "w") as f_daglist:
    for freq in tqdm(range(fmin, fmax), total=(fmax - fmin)):
        if freq in sat_band_list:
            print(f"Skipping saturated band {freq} Hz")
            continue

        prev_taskname = f"{target['name']}_{prev_stage}_TCoh{prev_tcoh}_O{prev_freq_deriv_order}_{freq}Hz"
        taskname = f"{target['name']}_{stage}_TCoh{tcoh}_O{freq_deriv_order}_{freq}Hz"

        # The previous stage's outlier file lands under home_dir if it was
        # produced by a local run, or under OSDF if produced by the
        # outlier-collection Condor job (its OSG output can't transfer
        # straight back to the access point).
        prev_outlier_file = paths.outlier_file(
            freq, prev_taskname, prev_stage, cluster=cluster
        )
        if not prev_outlier_file.exists():
            prev_outlier_file = paths.outlier_file(
                freq, prev_taskname, prev_stage, cluster=cluster, osdf=True
            )

        try:
            prev_data = fits.getdata(prev_outlier_file)
        except FileNotFoundError as e:
            print(f"Error loading previous stage outlier file for {freq} Hz: {e}")
            skipped_freqs.append(freq)
            continue

        if prev_data.size == 0:
            print(
                f"No outliers in previous stage outlier file for {freq} Hz. Skipping."
            )
            skipped_freqs.append(freq)
            continue

        n_jobs = prev_data["mean2F threshold"].size

        dag_file = manager.make_outlier_dag(
            config_file_osdf,
            target_file_osdf,
            taskname,
            freq,
            stage,
            freq_deriv_order,
            n_jobs,
            prev_outlier_file,
            exe,
            num_toplist=num_toplist,
            n_sky=n_sky,
            zero_threshold=zero_threshold,
            cluster=cluster,
            separate_saturated=False,
            is_injection=is_injection,
            max_workers=max_workers,
            request_memory=request_memory,
            request_disk=request_disk,
            request_cpu=request_cpu,
            image=image,
        )

        f_daglist.write(f"{dag_file}\n")

if skipped_freqs:
    print(f"Skipped frequencies: {skipped_freqs}")
