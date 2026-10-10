import subprocess
from multiprocessing import Pool
from pathlib import Path

from astropy.io import fits

from paws.analysis.outlier import ResultAnalysisManager
from paws.definitions import phase_param_name
from paws.filepaths import PathManager, make_dir


def delete_files(result_file_list):
    """Delete files to release disk storage."""
    for f in result_file_list:
        Path(f).unlink(missing_ok=True)
    # print(f'Deleted {len(result_file_list)} weave result files.\n')


def search_job(
    config,
    target,
    freq_deriv_order,
    n_seg,
    num_toplist,
    sft_files,
    metric_file,
    extra_stats,
    weave_exe,
    search_data,
):
    """
    Worker function to run a single WEAVE search job.
    """
    result_file, param_row = search_data
    make_dir([result_file])

    if Path(result_file).exists():
        return result_file

    # Construct command
    cmd_parts = [
        f"{weave_exe}",
        f"--output-file={result_file}",
        f'--sft-files="{sft_files}"',
        f"--setup-file={metric_file}",
        f"--semi-max-mismatch={config.semi_mm}",
        f"--toplist-limit={num_toplist}",
        f"--extra-statistics={extra_stats}",
        f"--alpha={target.alpha}",
        f"--delta={target.delta}",
    ]

    # Add coherence mismatch if the coherence time is not equal to the total observation time
    if n_seg > 1:
        cmd_parts.append(f"--coh-max-mismatch={config.coh_mm}")

    # Add frequency/derivative parameters
    freq_names, freq_deriv_names = phase_param_name(freq_deriv_order)
    for f_name, df_name in zip(freq_names, freq_deriv_names):
        val = param_row[f_name]
        dval = param_row[df_name]
        cmd_parts.append(f"--{f_name}={val}/{dval}")

    command = " ".join(cmd_parts)

    # Run command
    subprocess.run(command, shell=True, capture_output=True, text=True)
    return result_file


def injection_job(
    config,
    target,
    freq_deriv_order,
    n_seg,
    num_toplist,
    sft_files,
    metric_file,
    extra_stats,
    weave_exe,
    search_data,
    injection_data,
):
    """
    Worker function to run a single WEAVE injection job.
    """
    result_file, param_row = search_data
    make_dir([result_file])

    if Path(result_file).exists():
        return result_file

    # 1. Build Base Search Command
    cmd_parts = [
        f"{weave_exe}",
        f"--output-file={result_file}",
        f'--sft-files="{sft_files}"',
        f"--setup-file={metric_file}",
        f"--semi-max-mismatch={config.semi_mm}",
        f"--toplist-limit={num_toplist}",
        f"--extra-statistics={extra_stats}",
        f"--alpha={target.alpha}",
        f"--delta={target.delta}",
    ]

    if n_seg > 1:
        cmd_parts.append(f"--coh-max-mismatch={config.coh_mm}")

    freq_names, freq_deriv_names = phase_param_name(freq_deriv_order)
    for f_name, df_name in zip(freq_names, freq_deriv_names):
        val = param_row[f_name]
        dval = param_row[df_name]
        cmd_parts.append(f"--{f_name}={val}/{dval}")

    # 2. Build Injection String
    inj_str = (
        f"Alpha={injection_data['Alpha']};Delta={injection_data['Delta']};refTime={injection_data['refTime']};"
        f"aPlus={injection_data['aPlus']};aCross={injection_data['aCross']};psi={injection_data['psi']};"
        f"Freq={injection_data['Freq']};f1dot={injection_data['f1dot']};f2dot={injection_data['f2dot']};"
        f"f3dot={injection_data['f3dot']};f4dot={injection_data['f4dot']}"
    )

    cmd_parts.append(f'--injections="{{{inj_str}}}"')

    command = " ".join(cmd_parts)
    proc = subprocess.run(command, shell=True, capture_output=True, text=True)
    if proc.stdout:
        print(f"[injection_job stdout] {result_file}:\n{proc.stdout}")
    if proc.stderr:
        print(f"[injection_job stderr] {result_file}:\n{proc.stderr}")

    # subprocess.run(command, shell=True, capture_output=True, text=True)

    return result_file


def determine_efficiency(
    taskname,
    stage,
    config,
    target,
    freq,
    freq_deriv_order,
    n_sky,
    n_seg,
    num_toplist,
    sft_files,
    metric_file,
    extra_stats,
    weave_exe,
    search_data,
    injection_data,
    mean2f_th,
    n_cpu,
    cluster,
    work_in_local_dir,
    save_intermediate=False,
):
    """
    Runs injections in parallel and calculates detection efficiency.
    """
    paths = PathManager(config=config, target=target)
    result_manager = ResultAnalysisManager(config=config, target=target)

    # Prepare File Paths
    if work_in_local_dir:
        job_data = [
            (Path(paths.weave_output_file(freq, taskname, i, stage)).name, p)
            for i, p in enumerate(search_data, 1)
        ]
    else:
        job_data = [
            (str(paths.weave_output_file(freq, taskname, i, stage)), p)
            for i, p in enumerate(search_data, 1)
        ]

    # Run Parallel Jobs
    with Pool(processes=n_cpu) as pool:
        results = pool.starmap(
            injection_job,
            [
                (
                    config,
                    target,
                    freq_deriv_order,
                    n_seg,
                    num_toplist,
                    sft_files,
                    metric_file,
                    extra_stats,
                    weave_exe,
                    jd,
                    inj,
                )
                for jd, inj in zip(job_data, injection_data)
            ],
        )

    # Analysis
    # injection_data has n_inj * n_sky rows; recover actual n_inj
    n_inj = injection_data.size // n_sky
    outlier_file_path = result_manager.make_outlier(
        taskname,
        freq,
        mean2f_th,
        n_inj,
        n_processes=1,
        num_toplist=num_toplist,
        stage=stage,
        freq_deriv_order=freq_deriv_order,
        n_sky=n_sky,
        cluster=cluster,
        work_in_local_dir=work_in_local_dir,
        is_injection=True,
    )

    if not save_intermediate:
        delete_files(results)

    # Calculate Efficiency
    n_outlier = fits.getdata(outlier_file_path, 1).size
    eff = n_outlier / n_inj
    print(
        f"{eff * 100:.2f}% ({n_outlier}/{n_inj}) above mean2F threshold. \nSaved to {outlier_file_path}."
    )

    return eff, outlier_file_path
