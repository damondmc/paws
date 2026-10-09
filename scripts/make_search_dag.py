"""
Create search DAG files for Galactic Center search over specified frequency bands.
"""

import yaml

from paws.definitions import task_name
from paws.filepaths import PathManager
from paws.params.models import PowerLawModel
from paws.params.search import SearchParamGenerator
from paws.workflow.manager import WorkflowManager

# 1. Load Configs
with open("/home/hoitim.cheung/galacticCenter/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("/home/hoitim.cheung/galacticCenter/config/gal.yaml", "r") as f:
    target = yaml.safe_load(f)

# 2. Initialize Helpers
# We create the PathManager here to pass it to utility functions if needed
paths = PathManager(config, target)
manager = WorkflowManager(config, target)  # PathManager is initialized inside here too

# Search Setup
coh_day = 5
stage = "search-0"
freq_deriv_order = 2
n_seg = 107
f0_band = config["f0_band"]  # Use config!
num_top_list = 2000
metric_file = "osdf:///igwn/cit/staging/hoitim.cheung/metricSetup/Start1368970000_TCoh432000_N107_Spin2.fts"
use_osg = True
use_osdf = True

# generator = SearchParamGenerator(f0_band, freq_deriv_order, nc_min=config['nc_min'], nc_max=config['nc_max'])

# Frequency Bands to Process
fminList = [20]
fmaxList = [400]
for fmin, fmax in zip(fminList, fmaxList):
    print(f"Working on {fmin}-{fmax} Hz")
    dag_list_path = f"/home/hoitim.cheung/galacticCenter/dagFiles/{stage}_{target['name']}_dag{fmin}-{fmax}Hz.txt"

    with open(dag_list_path, "w") as f_daglist:
        for freq in range(fmin, fmax):
            # for freq in [299, 302, 303, 306, 307]:
            # Generate parameters
            if freq < 200:
                tau = target["age"] * 86400 * 365.25  # Convert years to seconds
            else:
                tau = 86400 * 365.25 * (300 + (freq - 199) * 0.5)
                print("older")
            model = PowerLawModel(
                nc_min=config["nc_min"], nc_max=config["nc_max"], tau=tau
            )
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

            # --- Collect SFT files ---
            sftFiles = []
            files = paths.sft_ensemble(freq)
            sftFiles.extend(files)  # Use extend, not append, to flatten the list

            # --- Make DAG ---
            # Generate task name
            taskname = (
                task_name(target["name"], stage, coh_day, freq_deriv_order)
                + f"_{freq}Hz"
            )

            dag_file = manager.make_search_dag(
                taskname,
                freq,
                params[freq].data,
                num_top_list=num_top_list,
                stage=stage,
                freq_deriv_order=freq_deriv_order,
                n_seg=n_seg,
                sft_files=sftFiles,
                metric_file=metric_file,
                request_memory="16GB",
                request_disk="4GB",
                request_cpu=1,
                use_osg=use_osg,
                use_osdf=use_osdf,
            )

            f_daglist.write(f"{dag_file}\n")

        # Must be the LAST entry: its ALL_NODES lines apply to every DAG in the list
        if use_osdf:
            f_daglist.write(f"{manager.make_osdf_cleanup_dag(stage)}\n")
