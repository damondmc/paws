import numpy as np
import yaml
from astropy.io import fits
from tqdm import tqdm

from paws.definitions import phase_param_name
from paws.filepaths import PathManager
from paws.params.injections import InjectionParamGenerator
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
prev_stage = "search-0"
prev_tcoh = 5
prev_freq_deriv_order = 2
stage = "injections-v2-0"  # v2: regenerated after f4dot coefficient fix
tcoh = 5
freq_deriv_order = 2
inj_freq_deriv_order = 4
sky_radius = 1.25e-4  # injection sky spread; spacing > 2*radius -> 1 search sky point
spacing_alpha = 2.5e-4
spacing_delta = 2.5e-4
n_seg = 107
n_inj = 20  # per 1 Hz band
tasks_per_job = 20  # 1 Condor job/band; ~25-80 s per Weave task (injections-0 logs)
request_memory = "3GB"  # injections-0 Weave peak 1.75 GB at 399 Hz, x1.5 buffer

ul_stage = "upperlimit-1pc-1skypt"
h0_file = f"{home_dir}postprocess/data/{target['name']}_{ul_stage}_TCoh{tcoh}_O{freq_deriv_order}_{fmin}-{fmax}Hz_clustered.txt"
h0_freq, h0_arr, _ = np.loadtxt(h0_file).T
# The h0 file skips saturated bands, so look h0 up by frequency rather than zip by row.
h0_by_freq = dict(zip(h0_freq.astype(int), h0_arr))

extra_stats = "coh2F_det,mean2F,coh2F_det,mean2F_det"
weave_exe = config["executables"]["weave"]
num_top_list = config["num_top_list"]
metric_file = "osdf:///igwn/cit/staging/hoitim.cheung/metricSetup/Start1368970000_TCoh432000_N107_Spin2.fts"
use_osg = True
use_osdf = True

_, freq_deriv_param_names = phase_param_name(freq_deriv_order)

skipped_freqs = []

dag_list_path = f"{home_dir}dagFiles/{stage}_{target['name']}_dag{fmin}-{fmax}Hz.txt"

with open(dag_list_path, "w") as f_daglist:
    for freq in tqdm(range(fmin, fmax), desc="Generating DAGs", total=fmax - fmin):
        if freq not in h0_by_freq:
            print(f"No h0 for {freq} Hz. Skipping DAG generation.")
            skipped_freqs.append(freq)
            continue
        h0 = h0_by_freq[freq]

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

        _, freq_deriv_names = phase_param_name(freq_deriv_order)
        if len(df_grid) != len(freq_deriv_names):
            raise ValueError(
                f"Length of df_grid ({len(df_grid)}) does not match number of frequency derivative names ({len(freq_deriv_names)})"
            )

        df_grid = {name: df_grid[i] for i, name in enumerate(freq_deriv_names)}

        mean2f_th = fits.getval(data_file, "mean2F_th", 0)

        if freq < 200:
            tau = 86400 * 365.25 * 300
        else:
            tau = 86400 * 365.25 * (300 + (freq - 199) * 0.5)

        model = PowerLawModel(nc_min=config["nc_min"], nc_max=config["nc_max"], tau=tau)
        injection_generator = InjectionParamGenerator(
            model=model, ref_time=config["ref_time"], f0_band=config["f0_band"]
        )

        search_data, injection_data = injection_generator.generate_parameters(
            alpha=target["alpha"],
            dalpha=target["dalpha"],
            delta=target["delta"],
            ddelta=target["ddelta"],
            non_sat_bands=non_sat_bands,
            spacing=df_grid,
            h0=h0,
            freq=freq,
            n_inj=n_inj,
            n_spacing=config["followup_n_spacing"],
            inj_freq_deriv_order=inj_freq_deriv_order,
            freq_deriv_order=freq_deriv_order,
            sky_radius=sky_radius,
            spacing_alpha=spacing_alpha,
            spacing_delta=spacing_delta,
        )

        # --- Make DAG ---
        taskname = f"{target['name']}_{stage}_TCoh{tcoh}_O{freq_deriv_order}_{freq}Hz"

        dag_file = manager.make_search_dag(
            taskname,
            freq,
            search_data[str(freq)].data,
            num_top_list=num_top_list,
            stage=stage,
            freq_deriv_order=freq_deriv_order,
            n_seg=n_seg,
            sft_files=sft_files,
            metric_file=metric_file,
            request_memory=request_memory,
            request_disk="4GB",
            request_cpu=1,
            use_osg=use_osg,
            use_osdf=use_osdf,
            inj_params=injection_data[str(freq)].data,
            inj_freq_deriv_order=inj_freq_deriv_order,
            tasks_per_job=tasks_per_job,
        )

        f_daglist.write(f"{dag_file}\n")

    # Must be the LAST entry: its ALL_NODES lines apply to every DAG in the list
    if use_osdf:
        f_daglist.write(f"{manager.make_osdf_cleanup_dag(stage)}\n")

if skipped_freqs:
    print(f"Skipped frequencies due to no non-saturated bands: {skipped_freqs}")
