import subprocess
from pathlib import Path
from multiprocessing import Pool
from itertools import islice
from astropy.io import fits

from paws.io import make_dir
from paws.definitions import phase_param_name
from paws.filepaths import PathManager
from paws.analysis.outlier import ResultAnalysisManager 

def delete_files(result_file_list):
    """Delete files to release disk storage."""
    for f in result_file_list:
        Path(f).unlink(missing_ok=True)
    # print(f'Deleted {len(result_file_list)} weave result files.\n')

def search_job(config, target, freq_deriv_order, n_seg, num_top_list, sft_files, metric_file, 
               extra_stats, weave_exe, search_data):
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
        f"--sft-files=\"{sft_files}\"",
        f"--setup-file={metric_file}",
        f"--semi-max-mismatch={config['semi_mm']}",
        f"--toplist-limit={num_top_list}",
        f"--extra-statistics={extra_stats}",
        f"--alpha={target['alpha']}",
        f"--delta={target['delta']}"
    ]

    # Add coherence mismatch if the coherence time is not equal to the total observation time
    if n_seg > 1:
        cmd_parts.append(f"--coh-max-mismatch={config['coh_mm']}")

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

def injection_job(config, target, freq_deriv_order, n_seg, num_top_list, sft_files, metric_file, 
                  extra_stats, weave_exe, search_data, injection_data):
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
        f"--sft-files=\"{sft_files}\"",
        f"--setup-file={metric_file}",
        f"--semi-max-mismatch={config['semi_mm']}",
        f"--toplist-limit={num_top_list}",
        f"--extra-statistics={extra_stats}",
        f"--alpha={target['alpha']}",
        f"--delta={target['delta']}"
    ]

    if n_seg > 1:
        cmd_parts.append(f"--coh-max-mismatch={config['coh_mm']}")
        
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
    
    cmd_parts.append(f'--injections=\"{{{inj_str}}}\"')
    
    command = " ".join(cmd_parts)
    subprocess.run(command, shell=True, capture_output=True, text=True)
    return result_file

def determine_efficiency(taskname, stage, config, target, freq, freq_deriv_order, n_seg, num_top_list, sft_files, metric_file, 
                         extra_stats, weave_exe, search_data, injection_data, mean2f_th, n_cpu,
                         cluster, work_in_local_dir, save_intermediate=False):
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
        results = pool.starmap(injection_job, [
            (config, target, freq_deriv_order, n_seg, num_top_list, sft_files, metric_file, 
             extra_stats, weave_exe, jd, inj) 
            for jd, inj in zip(job_data, injection_data)
        ])

    # Analysis    
    # Delegate writing results to ResultManager
    n_inj = injection_data.size
    outlier_file_path = result_manager.make_injection_outlier(
        taskname, freq, mean2f_th, n_inj, num_top_list=num_top_list, 
        stage=stage, freq_deriv_order=freq_deriv_order, 
        cluster=cluster, work_in_local_dir=work_in_local_dir
    )

    if not save_intermediate:
        delete_files(results)

    # Calculate Efficiency
    n_outlier = fits.getdata(outlier_file_path, 1).size
    eff = n_outlier / n_inj
    print(f'{eff*100:.2f}% ({n_outlier}/{n_inj}) above mean2F threshold. Saved to {outlier_file_path}.')
    
    return eff, outlier_file_path

# def injection_followup(paths, config, fm, result_manager, target, obs_day, freq, sft_files, 
#                        old_coh_day, old_freq_deriv_order, old_stage, 
#                        new_coh_day, new_freq_deriv_order, new_stage, 
#                        num_top_list, extra_stats, num_cpus, 
#                        cluster=False, work_in_local_dir=False, save_intermediate=False):
#     """
#     Performs injection follow-up to determine the mean2F growth ratio.
#     """
#     print('Doing injection follow-up...')    
    
#     # 1. Generate Follow-up Parameters
#     # fm is the FollowUpManager (or whatever class manages follow-up param generation)
#     sp, ip = fm.gen_followup_param_from_injection_1Hz(
#         old_coh_day, freq, stage=old_stage, 
#         old_freq_deriv_order=old_freq_deriv_order, 
#         new_freq_deriv_order=new_freq_deriv_order, 
#         cluster=cluster, work_in_local_dir=work_in_local_dir
#     )

#     if sp[str(freq)].data.size == 0:
#         print('0 outliers from the previous injection stage: Error!')
#         exit()
#     else:
#         print(f'{sp[str(freq)].data.size} injections to be carried out.')

#     inj_result_file_list = []
    
#     # 2. Setup Paths
#     task_name_str = task_name(target['name'], new_stage, new_coh_day, new_freq_deriv_order, freq)
#     metric_file = paths.weave_setup_file_from_param(obs_day, new_coh_day, new_freq_deriv_order)
    
#     if work_in_local_dir:  
#         metric_file = Path(metric_file).name

#     # 3. Prepare Job Parameters
#     if work_in_local_dir:
#         search_params = [
#             (Path(paths.weave_output_file(freq, task_name_str, i, new_stage)).name, p) 
#             for i, p in enumerate(sp[str(freq)].data, 1)
#         ]
#     else:
#         search_params = [
#             (str(paths.weave_output_file(freq, task_name_str, i, new_stage)), p) 
#             for i, p in enumerate(sp[str(freq)].data, 1)
#         ]

#     inj_params = ip[str(freq)].data
#     weave_exe = str(paths.weave_executable)
#     n_seg = int(obs_day / new_coh_day) if new_coh_day > 0 else 1

#     print("Generated params, running Weave...")

#     # 4. Run Parallel Jobs
#     # Tuple matches injection_job signature:
#     # (target, config, freq_deriv_order, n_seg, params, num_top_list, inj_row, sft_files, metric_file, extra_stats, weave_exe)
#     with Pool(processes=num_cpus) as pool:
#         results = pool.starmap(injection_job, [
#             (target, config, new_freq_deriv_order, n_seg, params, 1000, inj, sft_files, metric_file, 
#              extra_stats, weave_exe) 
#             for params, inj in zip(search_params, inj_params)
#         ])
     
#     inj_result_file_list.extend(results)

#     print('Analyzing injection result.')
    
#     # 5. Analyze Results
#     # Get old mean2F for outlier selection reference
#     task_old = task_name(target['name'], old_stage, old_coh_day, old_freq_deriv_order, freq)
#     outlier_file_old = paths.outlier_file(freq, task_old, old_stage, cluster=cluster)
    
#     if work_in_local_dir:
#         outlier_file_old = Path(outlier_file_old).name
        
#     data = fits.getdata(outlier_file_old, 1) 
#     old_mean2F = data['mean2F']
    
#     # For injection follow-up, ratio is usually set to 0 initially or calculated differently
#     ratio = 0 
    
#     outlier_file_path = result_manager.write_follow_up_result(
#         old_mean2F, new_coh_day, freq, num_top_list=num_top_list, 
#         new_stage=new_stage, new_freq_deriv_order=new_freq_deriv_order, 
#         ratio=ratio, work_in_local_dir=work_in_local_dir, 
#         inj=True, cluster=cluster
#     )
    
#     return outlier_file_path, inj_result_file_list

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

# def search_job(config, target, freq_deriv_order, n_seg, num_top_list, sft_files, metric_file, 
#                extra_stats, weave_exe, search_data):
    

# def determine_efficiency(taskname, stage, config, target, freq, freq_deriv_order, n_seg, num_top_list, sft_files, metric_file, 
#                          extra_stats, weave_exe, search_data, injection_data, mean2f_th, n_cpu,
#                          cluster, work_in_local_dir, save_intermediate=False):


def real_followup(taskname, stage, config, target, freq, freq_deriv_order, n_seg, num_top_list, sft_files, metric_file, 
                  extra_stats, weave_exe, search_data, mean2f_th, n_cpu,
                  cluster, work_in_local_dir, save_intermediate=False):

    """
    Executes the 'real' (non-injection) follow-up stage.
    """
    print('Doing real follow-up...')
    
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
        results = pool.starmap(search_job, [
            (config, target, freq_deriv_order, n_seg, num_top_list, sft_files, metric_file, 
             extra_stats, weave_exe, jd) 
            for jd in job_data
        ])

    # Analysis    
    # Delegate writing results to ResultManager
    outlier_file_path = result_manager.make_followup_outlier(
        taskname, freq, mean2f_th, num_top_list=num_top_list, 
        new_stage=stage, new_freq_deriv_order=freq_deriv_order, 
        cluster=cluster, work_in_local_dir=work_in_local_dir
    )


    if not save_intermediate:
        delete_files(results)

    return outlier_file_path

# def determine_mean2f_ratio(percentile, paths, target, freq, 
#                            old_coh_day, old_freq_deriv_order, old_stage, 
#                            new_coh_day, new_freq_deriv_order, new_stage, 
#                            cluster=False, work_in_local_dir=False):
#     """
#     Calculates the ratio of mean2F between two stages.
#     """
    
#     # Load Old Data
#     task_old = task_name(target['name'], old_stage, old_coh_day, old_freq_deriv_order, freq)
#     file_old = paths.outlier_file(freq, task_old, old_stage, cluster=cluster)
#     if work_in_local_dir:
#         file_old = Path(file_old).name
#     data_old = fits.getdata(file_old, 1)

#     # Load New Data
#     task_new = task_name(target['name'], new_stage, new_coh_day, new_freq_deriv_order, freq)
#     file_new = paths.outlier_file(freq, task_new, new_stage, cluster=cluster)
#     if work_in_local_dir:
#         file_new = Path(file_new).name
#     data_new = fits.getdata(file_new, 1)

#     # Calculate Ratio
#     try:
#         ratio_distribution = np.sort(data_new['mean2F'] / data_old['mean2F'])
        
#         # Get value at specific percentile
#         r = np.percentile(ratio_distribution, percentile) 
#         r = int(r * 100.) / 100. 
        
#         print(f'Ratio = {r} at {(1-percentile)*100:.1f}% percentile (N={ratio_distribution.size}).\n')
#     except Exception as e:
#         print(f'[Error] Could not calculate ratio: {e}')
#         # print(f'Sizes - Before: {data_old.size}, After: {data_new.size}.\n')
#         r = 0    
        
#     return r