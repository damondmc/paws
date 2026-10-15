"""HTCondor DAG and submit files of the Weave, upper-limit and outlier-collection jobs."""

import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from jaxtyping import Float, Shaped
from tqdm import tqdm

from paws.definitions import ext_param_name, phase_param_name
from paws.filepaths import PathManager, make_dir
from paws.fits_io import TableRow
from paws.settings import FixedSettings

from .writer import write_search_dagfile, write_search_subfile

# the Weave runs of one Condor node: parameter rows, or (parameter row, injection row) pairs
Chunk = Union[Shaped[np.ndarray, " n_task"], list[tuple[TableRow, TableRow]]]

# DAGMan PRE script that deletes the OSDF outputs a node left behind before it is
# (re)submitted. A broken output upload leaves truncated objects on the origin, the
# site's PelicanRetry transform reruns the job, and Pelican refuses to overwrite, so
# the rerun is held with "remote object already exists"; periodic_remove (see
# write_search_subfile) turns that hold into a node failure and RETRY resubmits it.
# DAGMan never runs PRE for DONE nodes, so finished outputs are never touched.
# Args: <stage DAG root (condorFiles/<stage>/<target>)> <allowed OSDF dir> <$JOB>
OSDF_CLEANUP_SCRIPT = r"""#!/bin/bash
ROOT=$1
ALLOWED=$2
LOG="$ROOT/cleanup_osdf_outputs.log"

# Multi-DAG node names carry an "<index>." prefix: 239.GalacticCenter_..._259Hz_3
NODE=$(sed 's/^[0-9]\+\.//' <<< "$3")
TASKNAME=${NODE%_*}
FREQ=$(sed -n 's/.*_\([0-9]\+\)Hz$/\1/p' <<< "$TASKNAME")
DAG_FILE="$ROOT/$FREQ/$TASKNAME.dag"

line=$(grep -m1 "^VARS $NODE " "$DAG_FILE" 2>/dev/null)
if [ -z "$line" ]; then
    echo "$(date -Is) $3: no VARS line for $NODE in $DAG_FILE" >> "$LOG"
    exit 1
fi

remaps=$(sed -n 's/.*REMAP_OUTPUT_FILES="\([^"]*\)".*/\1/p' <<< "$line")
IFS=';' read -ra pairs <<< "$remaps"
n_del=0
for pair in "${pairs[@]}"; do
    url="${pair#*=}"
    path="/osdf${url#osdf://}"
    if [[ "$path" != "$ALLOWED"* ]]; then
        echo "$(date -Is) $NODE: refusing to touch $path" >> "$LOG"
        exit 1
    fi
    if [ -e "$path" ]; then
        rm -f -- "$path" || exit 1
        n_del=$((n_del + 1))
    fi
done
[ "$n_del" -gt 0 ] && echo "$(date -Is) $NODE: deleted $n_del existing output(s)" >> "$LOG"
exit 0
"""


class WorkflowManager:
    """
    Central manager for creating HTCondor DAG and SUB files for all
    analysis stages (Search, Follow-up, Upper Limit).
    """

    def __init__(self, settings: FixedSettings) -> None:
        """settings: paws.settings.FixedSettings (or Settings)."""
        self.settings = settings
        self.paths = PathManager(settings)

        # Internal state for search parameter names
        self.freq_param_names: list[str] = []
        self.freq_deriv_param_names: list[str] = []
        self.num_top_list: int = 0

    # =========================================================================
    #  SECTION 1: SEARCH & FOLLOW-UP STAGE
    # =========================================================================

    def _get_execution_kwargs(self, n_seg: int) -> dict[str, Union[str, float, int]]:
        """Helper to generate common keyword arguments for the search executable."""
        extra_stats = "coh2F_det,mean2F,coh2F_det,mean2F_det"
        kwargs = {
            "semi-max-mismatch": self.settings.search.semi_mm,
            "toplist-limit": self.num_top_list,
            "extra-statistics": extra_stats,
        }

        if n_seg != 1:
            kwargs["coh-max-mismatch"] = self.settings.search.coh_mm

        return kwargs

    def _format_injection_str(self, colnames: list[str], inj_param: TableRow) -> str:
        """Formats injection parameters into a semicolon-separated string."""
        return ";".join([f"{col}={inj_param[col]}" for col in colnames])

    def _create_wrapper_script(self, stage: str) -> Path:
        """Creates a bash script that runs multiple Weave commands sequentially."""
        wrapper_path = (
            self.paths.dag_file(0, "wrapper", stage).parent
            / f"run_weave_batch_{stage}.sh"
        )
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)

        with open(wrapper_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("TASK_FILE=$1\n")
            f.write("EXE=$2\n\n")
            f.write("while IFS= read -r weave_args; do\n")
            f.write("    $EXE $weave_args\n")
            f.write("    if [ $? -ne 0 ]; then\n")
            f.write('        echo "Error: Weave failed on args: $weave_args"\n')
            f.write("        exit 1\n")
            f.write("    fi\n")
            f.write('done < "$TASK_FILE"\n')

        return wrapper_path

    def make_osdf_cleanup_dag(self, stage: str, n_retry: int = 3) -> Path:
        """
        Writes the stage's OSDF cleanup PRE script and a node-less DAG holding
        SCRIPT PRE ALL_NODES + RETRY ALL_NODES, and returns that DAG's path.

        In a multi-DAG submission ALL_NODES reaches the nodes of every DAG file, so
        these lines must live in this one file only, listed LAST in the dag list.
        """
        dag_root = self.paths.dag_file(0, "cleanup", stage).parent.parent
        stage_dir = dag_root.parent
        stage_dir.mkdir(parents=True, exist_ok=True)
        allowed = self.paths.osdf_dir / "o4ab" / "results" / stage

        script_path = stage_dir / "cleanup_osdf_outputs.sh"
        with open(script_path, "w") as f:
            f.write(OSDF_CLEANUP_SCRIPT)

        all_nodes_path = stage_dir / "all_nodes.dag"
        with open(all_nodes_path, "w") as f:
            f.write(
                "# Applies to every node of the multi-DAG it is submitted with: list this file LAST.\n"
            )
            f.write(
                f"SCRIPT PRE ALL_NODES /bin/bash {script_path} {dag_root} {allowed}/ $JOB\n"
            )
            f.write(f"RETRY ALL_NODES {n_retry}\n")

        return all_nodes_path

    def _search_batch_args(
        self,
        freq: int,
        stage: str,
        chunk: Chunk,
        taskname: str,
        n_seg: int,
        sft_files: list[str],
        node_index: int,
        use_osg: bool,
        metric_file: str,
        exe: str,
        wrapper_path: Path,
        tasks_per_job: int,
        inj_colnames: Optional[list[str]] = None,
    ) -> str:
        """Generates task files and DAG VARS for a chunk of grouped jobs."""

        # 1. Create a task file for this specific Condor node (saved in the DAG directory)
        task_dir = self.paths.dag_file(freq, taskname, stage).parent / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_file = task_dir / f"{taskname}_task_{node_index}.txt"

        output_files = []
        remap_strings = []
        task_lines = []

        kwargs = self._get_execution_kwargs(n_seg)

        # 2. Build literal command lines for the wrapper script to run
        for i, job_data in enumerate(chunk):
            job_index = (node_index - 1) * tasks_per_job + i + 1

            # Unpack based on injection mode
            if inj_colnames:
                search_param, inj_param = job_data
                inj_str = self._format_injection_str(inj_colnames, inj_param)
            else:
                search_param = job_data
                inj_str = None

            result_file = self.paths.weave_output_file(freq, taskname, job_index, stage)
            make_dir([result_file])

            # OSG expects local filenames (dynamically pulling the correct extension); Local execution expects absolute paths
            local_out_name = (
                f"{freq}Hz_out.fts.{job_index}" if use_osg else str(result_file)
            )
            sft_names = ";".join(
                [Path(s).name if use_osg else str(s) for s in sft_files]
            )
            metric_name = Path(metric_file).name if use_osg else str(metric_file)

            cmd_parts = [
                f"--output-file={local_out_name}",
                f'--sft-files="{sft_names}"',
                f"--setup-file={metric_name}",
            ]

            for key, value in kwargs.items():
                cmd_parts.append(f"--{key}={value}")

            cmd_parts.append(
                f"--alpha={search_param['alpha']}/{search_param['dalpha']}"
            )
            cmd_parts.append(
                f"--delta={search_param['delta']}/{search_param['ddelta']}"
            )

            for key1, key2 in zip(self.freq_param_names, self.freq_deriv_param_names):
                cmd_parts.append(f"--{key1}={search_param[key1]}/{search_param[key2]}")

            if inj_str:
                cmd_parts.append(f"--injections={{{inj_str}}}")

            task_lines.append(" ".join(cmd_parts))

            if use_osg:
                output_files.append(local_out_name)
                remap_strings.append(
                    f"{local_out_name}={self.paths.to_osdf_url(result_file)}"
                )

        # Write task file for the worker node
        with open(task_file, "w") as f:
            f.write("\n".join(task_lines) + "\n")

        # 3. Construct DAG VARS mapping for this node
        cmd_args = f"{task_file.name} {exe}" if use_osg else f"{task_file} {exe}"
        args_list = [f'CMD_ARGS="{cmd_args}"']

        if use_osg:
            args_list.append(f'OUTPUT_FILES="{", ".join(output_files)}"')
            args_list.append(f'REMAP_OUTPUT_FILES="{";".join(remap_strings)}"')

            transfer_files = [str(s) for s in sft_files] + [
                str(metric_file),
                str(task_file),
                str(wrapper_path),
            ]
            args_list.append(f'TRANSFER_FILES="{", ".join(transfer_files)}"')

        return " ".join(args_list) + " "

    def make_search_dag(
        self,
        taskname: str,
        freq: int,
        params: Shaped[np.ndarray, " n_param"],
        num_top_list: int,
        stage: str,
        freq_deriv_order: int,
        n_seg: int,
        sft_files: list[str],
        metric_file: str,
        request_memory: str = "18GB",
        request_disk: str = "5GB",
        request_cpu: int = 1,
        use_osg: bool = False,
        use_osdf: bool = False,
        exe: Optional[str] = None,
        image: Optional[str] = None,
        inj_params: Optional[Shaped[np.ndarray, " n_param"]] = None,
        inj_freq_deriv_order: Optional[int] = None,
        tasks_per_job: int = 10,
        sky_offsets: Optional[
            tuple[Float[np.ndarray, " n_sky"], Float[np.ndarray, " n_sky"]]
        ] = None,
    ) -> Path:
        """
        Creates the DAG and SUB files for the Search/Follow-up stage using grouped batching.

        params: one parameter row per seed (or search cell); with sky_offsets each row becomes n_sky Weave runs.
        inj_params: the injection of each row (injection stages).
        """
        if not use_osg:
            print(
                "Error: CIT cluster mode (use_osg=False) is not supported. Only OSG (use_osg=True) is tested."
            )
            sys.exit(1)

        t0 = time.time()
        is_injection = inj_params is not None and inj_freq_deriv_order is not None
        print(
            f"Generating DAG for {taskname} (Mode: {stage}, Injections: {is_injection}, Batch Size: {tasks_per_job})..."
        )

        if use_osdf and not use_osg:
            print("Warning: SFTs from OSDF requested but not using OSG resources.")

        self.freq_param_names, self.freq_deriv_param_names = phase_param_name(
            freq_deriv_order
        )
        self.num_top_list = num_top_list

        inj_colnames = None
        if is_injection:
            inj_freq_names, _ = phase_param_name(inj_freq_deriv_order)
            # Combine external params + "Freq" + remaining freq derivatives (f1dot, etc.)
            inj_colnames = ext_param_name() + ["Freq"] + inj_freq_names[1:]

        dag_file_path = self.paths.dag_file(freq, taskname, stage)
        dag_file_path.parent.mkdir(parents=True, exist_ok=True)
        dag_file_path.unlink(missing_ok=True)

        cr_files = self.paths.condor_record_files(freq, taskname, stage)
        make_dir(cr_files)

        wrapper_path = self._create_wrapper_script(stage)
        exe = exe if exe else self.paths.weave_executable

        sub_file_path = self.paths.condor_sub_file(freq, taskname, stage)
        sub_file_path.parent.mkdir(parents=True, exist_ok=True)
        sub_file_path.unlink(missing_ok=True)

        # Executable is /bin/bash; arguments point to the wrapper script and dynamic job tasks
        write_search_subfile(
            filename=str(sub_file_path),
            executable_path="/bin/bash",
            transfer_executable=False,
            output_path=str(cr_files[0]),
            error_path=str(cr_files[1]),
            log_path=str(cr_files[2]),
            arg_list_string=f"{wrapper_path.name if use_osg else wrapper_path} $(CMD_ARGS)",
            accounting_group=self.settings.environment.acc_group,
            user=self.settings.environment.user,
            request_memory=request_memory,
            request_disk=request_disk,
            request_cpu=request_cpu,
            use_osg=use_osg,
            use_osdf=use_osdf,
            image=image,
        )

        if sky_offsets is not None:
            d_alpha_sky, d_delta_sky = sky_offsets
            n_sky = len(d_alpha_sky)
        else:
            n_sky = 1

        n_out = len(params)
        n_jobs = n_out * n_sky
        n_chunks = (n_jobs + tasks_per_job - 1) // tasks_per_job

        for node_index in tqdm(range(1, n_chunks + 1), total=n_chunks):
            job_start = (node_index - 1) * tasks_per_job
            job_end = min(job_start + tasks_per_job, n_jobs)

            if sky_offsets is not None:
                out_start = job_start // n_sky
                out_end = min((job_end - 1) // n_sky + 1, n_out)
                n_out_chunk = out_end - out_start
                local_start = job_start - out_start * n_sky
                local_end = local_start + (job_end - job_start)

                row_idx = np.repeat(np.arange(n_out_chunk), n_sky)
                sky_idx = np.tile(np.arange(n_sky), n_out_chunk)
                tiled = params[out_start:out_end][row_idx]
                tiled["alpha"] = tiled["alpha"] + d_alpha_sky[sky_idx]
                tiled["delta"] = tiled["delta"] + d_delta_sky[sky_idx]
                chunk_params = tiled[local_start:local_end]

                if is_injection:
                    chunk_inj = np.repeat(inj_params[out_start:out_end], n_sky)[
                        local_start:local_end
                    ]
                    chunk = list(zip(chunk_params, chunk_inj))
                else:
                    chunk = chunk_params
            else:
                chunk_params = params[job_start:job_end]
                if is_injection:
                    chunk = list(zip(chunk_params, inj_params[job_start:job_end]))
                else:
                    chunk = chunk_params

            arg_list = self._search_batch_args(
                freq,
                stage,
                chunk,
                taskname,
                n_seg,
                sft_files,
                node_index,
                use_osg,
                metric_file,
                exe,
                wrapper_path,
                tasks_per_job,
                inj_colnames,
            )
            write_search_dagfile(
                str(dag_file_path), taskname, str(sub_file_path), node_index, arg_list
            )

        elapsed = time.time() - t0
        print(f"Finished writing {stage} dag files. Time: {elapsed:.2f}s")
        return dag_file_path

    # =========================================================================
    #  SECTION 2: UPPER LIMIT STAGE
    # =========================================================================

    def _ul_args(
        self,
        environment_file: str,
        data_file: str,
        search_file: str,
        taskname: str,
        freq: int,
        stage: str,
        freq_deriv_order: int,
        df_grid: list[float],
        inj_freq_deriv_order: int,
        sft_files_local: str,
        metric_file_local: str,
        n_inj: int,
        num_toplist: int,
        h0_est: float,
        mean2f_th: float,
        non_sat_bands: Float[np.ndarray, " n_subband"],
        sky_radius: float,
        spacing_alpha: Optional[float],
        spacing_delta: Optional[float],
        n_spacing: float,
        cluster: bool,
        work_in_local_dir: bool,
        save_intermediate: bool,
        request_cpu: int,
    ) -> str:
        """Generates command line arguments for the python upper limit script."""

        bands_str = " ".join(map(str, non_sat_bands))
        df_grid_str = " ".join(map(str, df_grid))

        arg_list_string = (
            f"--environment_file {environment_file} --data_file {data_file} --search_file {search_file} "
            f"--taskname {taskname} "
            f"--freq {freq} --stage {stage} --freq_deriv_order {freq_deriv_order} "
            f"--df_grid {df_grid_str} --inj_freq_deriv_order {inj_freq_deriv_order} "
            f"--sft_files '{sft_files_local}' --metric_file {metric_file_local} "
            f"--n_inj {n_inj} --num_toplist {num_toplist} --h0_est {h0_est} "
            f"--mean2f_th {mean2f_th} --non_sat_bands {bands_str} "
            f"--sky_radius {sky_radius} --n_spacing {n_spacing} --n_cpus {request_cpu}"
        )

        if spacing_alpha is not None:
            arg_list_string += (
                f" --spacing_alpha {spacing_alpha} --spacing_delta {spacing_delta}"
            )
        if cluster:
            arg_list_string += " --cluster"
        if work_in_local_dir:
            arg_list_string += " --work_in_local_dir"
        if save_intermediate:
            arg_list_string += " --save_intermediate"

        return arg_list_string

    def _ul_transfer_args(
        self,
        config_files: tuple[str, ...],
        taskname: str,
        freq: int,
        metric_file: str,
        stage: str,
        sft_files: list[str],
        cluster: bool,
        exe: Optional[str],
        image: Optional[str],
    ) -> str:
        """Generates VARS for OSG file transfers for Upper Limits (config_files: the job's settings files)."""

        input_files_list = ([str(exe)] if exe else []) + [
            str(config_file) for config_file in config_files
        ]
        input_files_list.extend([str(s) for s in sft_files])
        input_files_list.append(str(metric_file))

        input_files_str = ", ".join(input_files_list)

        outlier_file_path = self.paths.outlier_file(
            freq, taskname, stage, cluster=cluster, location="home"
        )
        make_dir([outlier_file_path])

        arg_list = (
            f'OUTPUT_FILES="{Path(outlier_file_path).name}" '
            f'REMAP_OUTPUT_FILES="{Path(outlier_file_path).name}={outlier_file_path}" '
            f'TRANSFER_FILES="{input_files_str}" '
        )
        return arg_list

    def make_upperlimit_dag(
        self,
        environment_file: str,
        data_file: str,
        search_file: str,
        spindown_model_file: str,
        taskname: str,
        freq: int,
        stage: str,
        freq_deriv_order: int,
        sft_files: list[str],
        metric_file: str,
        mean2f_th: float,
        non_sat_bands: Float[np.ndarray, " n_subband"],
        exe: Optional[str],
        n_spacing: float,
        df_grid: list[float] = [1e-6, 1e-13, 1e-20],
        inj_freq_deriv_order: int = 4,
        num_toplist: int = 1,
        sky_radius: float = 1e-5,
        spacing_alpha: Optional[float] = None,
        spacing_delta: Optional[float] = None,
        h0_est: float = 6e-26,
        n_inj: int = 64,
        request_memory: str = "4GB",
        request_disk: str = "4GB",
        request_cpu: int = 32,
        cluster: bool = False,
        work_in_local_dir: bool = False,
        save_intermediate: bool = False,
        image: Optional[str] = None,
    ) -> Path:
        """
        Creates the DAG and SUB files for the Upper Limit stage.
        """
        print(f"Generating UPPERLIMIT DAG for {taskname}...")

        dag_file_path = self.paths.dag_file(freq, taskname, stage)
        dag_file_path.parent.mkdir(parents=True, exist_ok=True)
        dag_file_path.unlink(missing_ok=True)

        cr_files = self.paths.condor_record_files(freq, taskname, stage)
        make_dir(cr_files)

        sub_file_path = self.paths.condor_sub_file(freq, taskname, stage)
        sub_file_path.unlink(missing_ok=True)

        sft_files_local = ";".join([Path(s).name for s in sft_files])
        metric_file_local = Path(metric_file).name

        python_args = self._ul_args(
            environment_file=Path(environment_file).name,
            data_file=Path(data_file).name,
            search_file=Path(search_file).name,
            taskname=taskname,
            freq=freq,
            stage=stage,
            freq_deriv_order=freq_deriv_order,
            df_grid=df_grid,
            inj_freq_deriv_order=inj_freq_deriv_order,
            sft_files_local=sft_files_local,
            metric_file_local=metric_file_local,
            n_inj=n_inj,
            num_toplist=num_toplist,
            h0_est=h0_est,
            mean2f_th=mean2f_th,
            non_sat_bands=non_sat_bands,
            sky_radius=sky_radius,
            spacing_alpha=spacing_alpha,
            spacing_delta=spacing_delta,
            n_spacing=n_spacing,
            cluster=cluster,
            work_in_local_dir=work_in_local_dir,
            save_intermediate=save_intermediate,
            request_cpu=request_cpu,
        )

        # exe=None: run paws.jobs.upperlimit from the container image
        script = Path(exe).name if exe else "-m paws.jobs.upperlimit"
        full_arg_string = f"{script} {python_args}"

        write_search_subfile(
            filename=str(sub_file_path),
            executable_path="/opt/paws/.venv/bin/python",
            transfer_executable=False,
            output_path=str(cr_files[0]),
            error_path=str(cr_files[1]),
            log_path=str(cr_files[2]),
            arg_list_string=full_arg_string,
            accounting_group=self.settings.environment.acc_group,
            user=self.settings.environment.user,
            request_memory=request_memory,
            request_disk=request_disk,
            request_cpu=request_cpu,
            use_osg=True,
            use_osdf=True,
            image=image,
        )

        arg_list = self._ul_transfer_args(
            (environment_file, data_file, search_file, spindown_model_file),
            taskname,
            freq,
            metric_file,
            stage,
            sft_files,
            cluster,
            exe,
            image,
        )

        write_search_dagfile(
            str(dag_file_path), taskname, str(sub_file_path), 1, arg_list
        )

        return dag_file_path

    # =========================================================================
    #  SECTION 3: OUTLIER COLLECTION STAGE
    # =========================================================================

    def _outlier_args(
        self,
        environment_file: str,
        data_file: str,
        search_file: str,
        taskname: str,
        freq: int,
        stage: str,
        freq_deriv_order: int,
        prev_outlier_file_local: str,
        num_toplist: int,
        n_sky: int,
        zero_threshold: bool,
        cluster: bool,
        separate_saturated: bool,
        is_injection: bool,
        max_workers: int,
    ) -> str:
        """Generates command line arguments for the python outlier-collection script."""
        arg_list_string = (
            f"--environment_file {environment_file} --data_file {data_file} --search_file {search_file} "
            f"--taskname {taskname} "
            f"--freq {freq} --stage {stage} --freq_deriv_order {freq_deriv_order} "
            f"--prev_outlier_file {prev_outlier_file_local} --num_toplist {num_toplist} "
            f"--n_sky {n_sky} --max_workers {max_workers}"
        )

        if zero_threshold:
            arg_list_string += " --zero_threshold"
        if cluster:
            arg_list_string += " --cluster"
        if separate_saturated:
            arg_list_string += " --separate_saturated"
        if is_injection:
            arg_list_string += " --is_injection"

        return arg_list_string

    def _outlier_transfer_args(
        self,
        config_files: tuple[str, ...],
        taskname: str,
        freq: int,
        stage: str,
        n_jobs: int,
        n_sky: int,
        prev_outlier_file: Path,
        exe: Optional[str],
        cluster: bool,
    ) -> str:
        """Generates VARS for OSG file transfers for the outlier-collection job (config_files: the job's settings
        files)."""

        # This stage's raw Weave result files live on OSDF; pull them via the
        # osdf:// scheme so they transfer over the OSDF plugin like SFTs do.
        weave_files = [
            self.paths.to_osdf_url(
                self.paths.weave_output_file(freq, taskname, job_index, stage)
            )
            for job_index in range(1, n_jobs * n_sky + 1)
        ]

        input_files_list = (
            ([str(exe)] if exe else [])
            + [str(config_file) for config_file in config_files]
            + [str(prev_outlier_file)]
            + weave_files
        )
        input_files_str = ", ".join(input_files_list)

        # OSG execute nodes can't transfer output straight back to the access
        # point's home_dir, so the outlier file is remapped through OSDF
        # instead, same as the raw Weave results. Downstream readers (e.g.
        # make_followup_dag.py) need to check both locations.
        outlier_file_path = self.paths.outlier_file(
            freq, taskname, stage, cluster=False, location="osdf"
        )
        make_dir([outlier_file_path])

        output_names = [Path(outlier_file_path).name]
        remap_strings = [
            f"{Path(outlier_file_path).name}={self.paths.to_osdf_url(outlier_file_path)}"
        ]

        if cluster:
            clustered_file_path = self.paths.outlier_file(
                freq, taskname, stage, cluster=True, location="osdf"
            )
            output_names.append(Path(clustered_file_path).name)
            remap_strings.append(
                f"{Path(clustered_file_path).name}={self.paths.to_osdf_url(clustered_file_path)}"
            )

        arg_list = (
            f'OUTPUT_FILES="{", ".join(output_names)}" '
            f'REMAP_OUTPUT_FILES="{";".join(remap_strings)}" '
            f'TRANSFER_FILES="{input_files_str}" '
        )
        return arg_list

    def make_outlier_dag(
        self,
        environment_file: str,
        data_file: str,
        search_file: str,
        spindown_model_file: str,
        taskname: str,
        freq: int,
        stage: str,
        freq_deriv_order: int,
        n_jobs: int,
        prev_outlier_file: Path,
        exe: Optional[str],
        num_toplist: int = 1000,
        n_sky: int = 1,
        zero_threshold: bool = False,
        cluster: bool = False,
        separate_saturated: bool = False,
        is_injection: bool = False,
        max_workers: int = 32,
        request_memory: str = "8GB",
        request_disk: str = "8GB",
        request_cpu: int = 1,
        image: Optional[str] = None,
    ) -> Path:
        """
        Creates the DAG and SUB files for collecting Weave outputs into an
        outlier FITS file as an OSG Condor job. Transfers in every raw Weave
        result file for this stage/frequency (via OSDF) plus the previous
        stage's outlier file (used to derive mean2F thresholds).
        """
        print(f"Generating OUTLIER DAG for {taskname}...")

        # Job-management paths (DAG/SUB/OUT/ERR/LOG) are keyed only on
        # (freq, taskname, stage). Since this job reuses the same taskname/stage
        # as the underlying Weave search job (to locate its result files), a
        # distinct identifier is needed here so it doesn't overwrite the
        # search job's own DAG/SUB/records.
        job_taskname = f"{taskname}_outlier"

        dag_file_path = self.paths.dag_file(freq, job_taskname, stage)
        dag_file_path.parent.mkdir(parents=True, exist_ok=True)
        dag_file_path.unlink(missing_ok=True)

        cr_files = self.paths.condor_record_files(freq, job_taskname, stage)
        make_dir(cr_files)

        sub_file_path = self.paths.condor_sub_file(freq, job_taskname, stage)
        sub_file_path.unlink(missing_ok=True)

        prev_outlier_file_local = Path(prev_outlier_file).name

        python_args = self._outlier_args(
            environment_file=Path(environment_file).name,
            data_file=Path(data_file).name,
            search_file=Path(search_file).name,
            taskname=taskname,
            freq=freq,
            stage=stage,
            freq_deriv_order=freq_deriv_order,
            prev_outlier_file_local=prev_outlier_file_local,
            num_toplist=num_toplist,
            n_sky=n_sky,
            zero_threshold=zero_threshold,
            cluster=cluster,
            separate_saturated=separate_saturated,
            is_injection=is_injection,
            max_workers=max_workers,
        )

        # exe=None: run paws.jobs.collect_outliers from the container image
        script = Path(exe).name if exe else "-m paws.jobs.collect_outliers"
        full_arg_string = f"{script} {python_args}"

        write_search_subfile(
            filename=str(sub_file_path),
            executable_path="/opt/paws/.venv/bin/python",
            transfer_executable=False,
            output_path=str(cr_files[0]),
            error_path=str(cr_files[1]),
            log_path=str(cr_files[2]),
            arg_list_string=full_arg_string,
            accounting_group=self.settings.environment.acc_group,
            user=self.settings.environment.user,
            request_memory=request_memory,
            request_disk=request_disk,
            request_cpu=request_cpu,
            use_osg=True,
            use_osdf=True,
            image=image,
        )

        arg_list = self._outlier_transfer_args(
            (environment_file, data_file, search_file, spindown_model_file),
            taskname,
            freq,
            stage,
            n_jobs,
            n_sky,
            prev_outlier_file,
            exe,
            cluster,
        )

        write_search_dagfile(
            str(dag_file_path), job_taskname, str(sub_file_path), 1, arg_list
        )
        return dag_file_path
