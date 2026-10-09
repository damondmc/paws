import sys
from pathlib import Path

import numpy as np
SEED_MAP = np.load("/home/hoitim.cheung/galacticCenter/results/followup-v2-2/seed_map_followup-v2-1_vs_followup-1.npz")  # written by check_followup_v2_1_seeds.py
import yaml
from astropy.io import fits
from tqdm import tqdm

from paws.definitions import phase_param_name
from paws.filepaths import PathManager
from paws.params.followup import FollowUpParamGenerator
from paws.params.models import PowerLawModel
from paws.workflow.manager import WorkflowManager

# 1. Load Configs
with open("/home/hoitim.cheung/galacticCenter/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

home_dir = config["home_dir"]

with open(f"{home_dir}config/gal.yaml", "r") as f:
    target = yaml.safe_load(f)

paths = PathManager(config, target)
manager = WorkflowManager(config, target)

fmin = 20
fmax = 400
use_osg = True
use_osdf = True
cluster = True
is_injection = False  # True to carry injections from prev stage into DAG

################################################
prev_stage = "followup-v2-1"
prev_tcoh = 10
prev_freq_deriv_order = 2
################################################

################################################
stage = "followup-v2-2"
tcoh = 20
freq_deriv_order = 2
################################################

################################################
inj_freq_deriv_order = 4
n_seg = 27
tasks_per_job = 114  # 2 seeds x 57 sky points per Condor job (~1 h)
sky_radius = 0
spacing_alpha = None
spacing_delta = None
sky_grid_file = "/home/hoitim.cheung/galacticCenter/config/gc_sky_grid.txt"  # actual Weave sky grid offsets (d_alpha, d_delta)
# sky_grid_file = None  # single sky point
n_inj_max = None  # keep only the first n_inj_max injections per band (None = all)
################################################

extra_stats = "coh2F_det,mean2F,coh2F_det,mean2F_det"
weave_exe = config["executables"]["weave"]
num_top_list = config["num_top_list"]
metric_file = f"osdf:///igwn/cit/staging/hoitim.cheung/metricSetup/o4ab_t{tcoh}_s{freq_deriv_order}.fts"

_, freq_deriv_param_names = phase_param_name(prev_freq_deriv_order)

skipped_freqs = []

# optional arguments: bands to process (default fmin..fmax); a band list gets its own DAG list file
freqs = [int(a) for a in sys.argv[1:]]
dag_list_path = (f"{home_dir}dagFiles/{stage}_{target['name']}_dag{'_'.join(map(str, freqs))}Hz.txt" if freqs
                 else f"{home_dir}dagFiles/{stage}_{target['name']}_dag{fmin}-{fmax}Hz.txt")

with open(dag_list_path, "w") as f_daglist:
    for freq in tqdm(freqs or range(fmin, fmax), desc="Generating DAGs"):
        sft_files = []
        files = paths.sft_ensemble(freq)
        sft_files.extend(files)

        data_taskname = f"{target['name']}_{prev_stage}_TCoh{prev_tcoh}_O{prev_freq_deriv_order}_{freq}Hz"

        data = []
        injection_data = None

        try:
            # The previous stage's outlier file lands under home_dir if it was
            # produced by a local run, or under OSDF if produced by the
            # outlier-collection Condor job (its OSG output can't transfer
            # straight back to the access point).
            data_file = paths.outlier_file(
                freq, data_taskname, prev_stage, cluster=cluster
            )
            if not data_file.exists():
                data_file = paths.outlier_file(
                    freq, data_taskname, prev_stage, cluster=cluster, osdf=True
                )
            data = fits.getdata(data_file, ext=1)
            # Only seeds that changed vs the old followup-1 clustering; exact matches reuse followup-2.
            changed = np.where(SEED_MAP[f"{freq}_exact"] < 0)[0] if f"{freq}_exact" in SEED_MAP.files else np.zeros(0, int)
            assert len(SEED_MAP[f"{freq}_exact"]) == len(data) if f"{freq}_exact" in SEED_MAP.files else True
            data = data[changed]
            if len(changed):
                rows_file = Path(paths.dag_file(freq, "x", stage)).parent / "changed_rows.txt"
                rows_file.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(rows_file, changed, fmt="%d", header="followup-v2-1 clustered row index of each seed, in job-block order (57 jobs per seed)")
            if is_injection:
                injection_data = fits.getdata(data_file, extname="injection")
                # Outlier and injection tables are row-aligned, so slicing
                # both by the same rows keeps the pairing intact.
                if n_inj_max is not None:
                    data = data[:n_inj_max]
                    injection_data = injection_data[:n_inj_max]
        except FileNotFoundError as e:
            print(f"Error loading data for frequency {freq} Hz: {e}")

        if len(data) == 0:
            print(
                f"No outliers found for frequency {freq} Hz. Skipping follow-up generation."
            )
            skipped_freqs.append(freq)
            continue

        df_grid = [
            fits.getval(data_file, param_name, 0)
            for param_name in freq_deriv_param_names
        ]

        _, freq_deriv_names = phase_param_name(prev_freq_deriv_order)
        if len(df_grid) != len(freq_deriv_names):
            raise ValueError(
                f"Length of df_grid ({len(df_grid)}) does not match number of frequency derivative names ({len(freq_deriv_names)})"
            )

        df_grid = {name: df_grid[i] for i, name in enumerate(freq_deriv_names)}

        if freq < 200:
            tau = 86400 * 365.25 * 300
        else:
            tau = 86400 * 365.25 * (300 + (freq - 199) * 0.5)

        model = PowerLawModel(nc_min=config["nc_min"], nc_max=config["nc_max"], tau=tau)
        followup_generator = FollowUpParamGenerator(model)

        search_data = followup_generator.generate_parameter(
            alpha=target["alpha"],
            dalpha=target["dalpha"],
            delta=target["delta"],
            ddelta=target["ddelta"],
            data=data,
            old_freq_deriv_order=prev_freq_deriv_order,
            new_freq_deriv_order=freq_deriv_order,
            spacing=df_grid,
            n_spacing=config["followup_n_spacing"],
            sky_radius=sky_radius,
            spacing_alpha=spacing_alpha,
            spacing_delta=spacing_delta,
        )

        # Tile each outlier across the actual Weave sky grid: (d_alpha, d_delta)
        # offsets centered on the grid's mean, read from sky_grid_file. One of
        # these points is already nearly coincident with the outlier's own
        # (0,0) position, so it is not added separately.
        sky_offsets = None
        if len(search_data.data) > 0 and sky_grid_file:
            d_alpha, d_delta = np.loadtxt(sky_grid_file, unpack=True)
            sky_offsets = (d_alpha, d_delta)

        taskname = f"{target['name']}_{stage}_TCoh{tcoh}_O{freq_deriv_order}_{freq}Hz"

        dag_file = manager.make_search_dag(
            taskname,
            freq,
            search_data.data,
            num_top_list=num_top_list,
            stage=stage,
            freq_deriv_order=freq_deriv_order,
            n_seg=n_seg,
            sft_files=sft_files,
            metric_file=metric_file,
            request_memory="2GB",  # old t20 O2 Weave peak 0.73 GB
            request_disk="4GB",  # ~2.4 GB used
            request_cpu=1,
            use_osg=use_osg,
            use_osdf=use_osdf,
            inj_params=injection_data,
            inj_freq_deriv_order=inj_freq_deriv_order if is_injection else None,
            tasks_per_job=tasks_per_job,
            sky_offsets=sky_offsets,
        )

        f_daglist.write(f"{dag_file}\n")

    # Must be the LAST entry: its ALL_NODES lines apply to every DAG in the list
    if use_osdf:
        f_daglist.write(f"{manager.make_osdf_cleanup_dag(stage)}\n")

if skipped_freqs:
    print(f"Skipped frequencies: {skipped_freqs}")
