import numpy as np
import subprocess
from pathlib import Path
from multiprocessing import Pool
from itertools import islice
from astropy.io import fits

from paws.io import make_dir
from paws.definitions import phase_param_name, task_name

def delete_files(result_file_list):
    """Delete files to release disk storage."""
    for f in result_file_list:
        Path(f).unlink(missing_ok=True)
    # print(f'Deleted {len(result_file_list)} weave result files.\n')

def search_job(target, config, freq_deriv_order, n_seg, params, num_top_list, sft_files, metric_file, 
               extra_stats, weave_exe):
    """
    Worker function to run a single WEAVE search job.
    """
    result_file, param_row = params
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

    # Add coherence mismatch if not observing for the full duration
    # Logic: if n_seg > 1, we are splitting the observation, so mismatch is allowed
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

def injection_job(target, config, freq_deriv_order, n_seg, params, num_top_list, inj_row, sft_files, metric_file, 
                  extra_stats, weave_exe):
    """
    Worker function to run a single WEAVE injection job.
    """
    result_file, param_row = params
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
        f"Alpha={inj_row['Alpha']};Delta={inj_row['Delta']};refTime={inj_row['refTime']};"
        f"aPlus={inj_row['aPlus']};aCross={inj_row['aCross']};psi={inj_row['psi']};"
        f"Freq={inj_row['Freq']};f1dot={inj_row['f1dot']};f2dot={inj_row['f2dot']};"
        f"f3dot={inj_row['f3dot']};f4dot={inj_row['f4dot']}"
    )
    
    cmd_parts.append(f'--injections=\"{{{inj_str}}}\"')
    
    command = " ".join(cmd_parts)
    subprocess.run(command, shell=True, capture_output=True, text=True)
    return result_file

def determine_efficiency(paths, config, sft_files, search_params_chunk, inj_params_chunk, result_manager, 
                         target, task_name_str, freq, n_inj, freq_deriv_order, stage, 
                         num_top_list, extra_stats, num_cpus, cluster, work_in_local_dir, 
                         obs_day, coh_day, save_intermediate=False):
    """
    Runs injections in parallel and calculates detection efficiency.
    """
    
    # Prepare File Paths
    if work_in_local_dir:
        job_list = [
            (Path(paths.weave_output_file(freq, task_name_str, i, stage)).name, p) 
            for i, p in enumerate(search_params_chunk, 1)
        ]
        metric_file = Path(paths.weave_setup_file_from_param(obs_day, coh_day, freq_deriv_order)).name
    else:
        job_list = [
            (str(paths.weave_output_file(freq, task_name_str, i, stage)), p) 
            for i, p in enumerate(search_params_chunk, 1)
        ]
        metric_file = str(paths.weave_setup_file_from_param(obs_day, coh_day, freq_deriv_order))

    weave_exe = str(paths.weave_executable)
    
    # Calculate n_seg (Number of segments)
    # If coh_day equals obs_day, we have 1 segment. Otherwise obs_day / coh_day
    n_seg = int(obs_day / coh_day) if coh_day > 0 else 1

    # Run Parallel Jobs
    # Note: We must construct the tuple EXACTLY as injection_job expects it:
    # (target, config, freq_deriv_order, n_seg, params, num_top_list, inj_row, sft_files, metric_file, extra_stats, weave_exe)
    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(injection_job, [
            (target, config, freq_deriv_order, n_seg, params, num_top_list, inj, sft_files, metric_file, 
             extra_stats, weave_exe) 
            for params, inj in zip(job_list, inj_params_chunk)
        ])

    # Analysis
    # Get threshold from the search stage outlier file
    # Ensure coh_day and freq are correct types for task_name
    search_task_name = task_name(target['name'], 'search', coh_day, freq_deriv_order, int(freq))
    search_outlier_file = paths.outlier_file(freq, search_task_name, 'search', cluster=cluster)
    
    if work_in_local_dir:
        search_outlier_file = Path(search_outlier_file).name
        
    mean2F_th = fits.getheader(search_outlier_file)['HIERARCH mean2F_th']
    
    # Delegate writing results to ResultManager
    outlier_file_path = result_manager.write_injection_result(
        coh_day, freq, mean2F_th, n_inj, num_top_list=num_top_list, 
        stage=stage, freq_deriv_order=freq_deriv_order, 
        cluster=cluster, work_in_local_dir=work_in_local_dir
    )

    if not save_intermediate:
        delete_files(results)

    # Calculate Efficiency
    nout = fits.getdata(outlier_file_path, 1).size
    p = nout / n_inj
    print(f'{p*100:.2f}% ({nout}/{n_inj}) above mean2F threshold. Saved to {outlier_file_path}.')
    
    return p, outlier_file_path

def injection_followup(paths, config, fm, result_manager, target, obs_day, freq, sft_files, 
                       old_coh_day, old_freq_deriv_order, old_stage, 
                       new_coh_day, new_freq_deriv_order, new_stage, 
                       num_top_list, extra_stats, num_cpus, 
                       cluster=False, work_in_local_dir=False, save_intermediate=False):
    """
    Performs injection follow-up to determine the mean2F growth ratio.
    """
    print('Doing injection follow-up...')    
    
    # 1. Generate Follow-up Parameters
    # fm is the FollowUpManager (or whatever class manages follow-up param generation)
    sp, ip = fm.gen_followup_param_from_injection_1Hz(
        old_coh_day, freq, stage=old_stage, 
        old_freq_deriv_order=old_freq_deriv_order, 
        new_freq_deriv_order=new_freq_deriv_order, 
        cluster=cluster, work_in_local_dir=work_in_local_dir
    )

    if sp[str(freq)].data.size == 0:
        print('0 outliers from the previous injection stage: Error!')
        exit()
    else:
        print(f'{sp[str(freq)].data.size} injections to be carried out.')

    inj_result_file_list = []
    
    # 2. Setup Paths
    task_name_str = task_name(target['name'], new_stage, new_coh_day, new_freq_deriv_order, freq)
    metric_file = paths.weave_setup_file_from_param(obs_day, new_coh_day, new_freq_deriv_order)
    
    if work_in_local_dir:  
        metric_file = Path(metric_file).name

    # 3. Prepare Job Parameters
    if work_in_local_dir:
        search_params = [
            (Path(paths.weave_output_file(freq, task_name_str, i, new_stage)).name, p) 
            for i, p in enumerate(sp[str(freq)].data, 1)
        ]
    else:
        search_params = [
            (str(paths.weave_output_file(freq, task_name_str, i, new_stage)), p) 
            for i, p in enumerate(sp[str(freq)].data, 1)
        ]

    inj_params = ip[str(freq)].data
    weave_exe = str(paths.weave_executable)
    n_seg = int(obs_day / new_coh_day) if new_coh_day > 0 else 1

    print("Generated params, running Weave...")

    # 4. Run Parallel Jobs
    # Tuple matches injection_job signature:
    # (target, config, freq_deriv_order, n_seg, params, num_top_list, inj_row, sft_files, metric_file, extra_stats, weave_exe)
    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(injection_job, [
            (target, config, new_freq_deriv_order, n_seg, params, 1000, inj, sft_files, metric_file, 
             extra_stats, weave_exe) 
            for params, inj in zip(search_params, inj_params)
        ])
     
    inj_result_file_list.extend(results)

    print('Analyzing injection result.')
    
    # 5. Analyze Results
    # Get old mean2F for outlier selection reference
    task_old = task_name(target['name'], old_stage, old_coh_day, old_freq_deriv_order, freq)
    outlier_file_old = paths.outlier_file(freq, task_old, old_stage, cluster=cluster)
    
    if work_in_local_dir:
        outlier_file_old = Path(outlier_file_old).name
        
    data = fits.getdata(outlier_file_old, 1) 
    old_mean2F = data['mean2F']
    
    # For injection follow-up, ratio is usually set to 0 initially or calculated differently
    ratio = 0 
    
    outlier_file_path = result_manager.write_follow_up_result(
        old_mean2F, new_coh_day, freq, num_top_list=num_top_list, 
        new_stage=new_stage, new_freq_deriv_order=new_freq_deriv_order, 
        ratio=ratio, work_in_local_dir=work_in_local_dir, 
        inj=True, cluster=cluster
    )
    
    return outlier_file_path, inj_result_file_list

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def real_followup(paths, config, sp, target, obs_day, freq, sft_files, 
                  old_mean2F, mean2F_ratio, 
                  new_coh_day, new_freq_deriv_order, new_stage, 
                  num_top_list, extra_stats, num_cpus, 
                  result_manager, cluster=False, work_in_local_dir=False, 
                  save_intermediate=False):
    """
    Executes the 'real' (non-injection) follow-up stage.
    """
    print('Doing real follow-up...')
     
    # 1. Setup Paths
    t_name = task_name(target['name'], new_stage, new_coh_day, new_freq_deriv_order, freq)
    metric_file = paths.weave_setup_file_from_param(obs_day, new_coh_day, new_freq_deriv_order)
    
    if work_in_local_dir:  
        metric_file = Path(metric_file).name

    # 2. Prepare Job Parameters
    if work_in_local_dir:
        search_params = [
            (Path(paths.weave_output_file(freq, t_name, i, new_stage)).name, p) 
            for i, p in enumerate(sp[str(freq)].data, 1)
        ]
    else:
        search_params = [
            (str(paths.weave_output_file(freq, t_name, i, new_stage)), p) 
            for i, p in enumerate(sp[str(freq)].data, 1)
        ]
       
    # 3. Setup Chunking
    chunk_size = 100  
    total_job_counts = len(search_params)
    chunk_count = int(np.ceil(total_job_counts / chunk_size))
    weave_exe = str(paths.weave_executable)
    n_seg = int(obs_day / new_coh_day) if new_coh_day > 0 else 1
    
    outlier_file_path = None

    print("Generated params, running Weave...")
    
    # 4. Process Chunks
    with Pool(processes=num_cpus) as pool:
        
        for chunk_index, chunk in enumerate(chunked_iterable(search_params, chunk_size)):
            print(f"Processing chunk {chunk_index + 1} out of {chunk_count}...")
            
            # Run Search Jobs in Parallel
            # Tuple must match search_job signature:
            # (target, config, freq_deriv_order, n_seg, params, num_top_list, sft_files, metric_file, extra_stats, weave_exe)
            results = pool.starmap(search_job, [
                (target, config, new_freq_deriv_order, n_seg, params, num_top_list, sft_files, metric_file, 
                 extra_stats, weave_exe)
                for params in chunk
            ])
            
            # Analyze Results Immediately
            if chunk_count == 1:
                outlier_file_path = result_manager.write_follow_up_result(
                    old_mean2F, new_coh_day, freq, num_top_list=num_top_list, 
                    new_stage=new_stage, new_freq_deriv_order=new_freq_deriv_order, 
                    ratio=mean2F_ratio, work_in_local_dir=work_in_local_dir, 
                    inj=False, cluster=cluster
                )
            else:
                outlier_file_path = result_manager.write_follow_up_result(
                    old_mean2F, new_coh_day, freq, num_top_list=num_top_list, 
                    new_stage=new_stage, new_freq_deriv_order=new_freq_deriv_order, 
                    ratio=mean2F_ratio, work_in_local_dir=work_in_local_dir, 
                    inj=False, cluster=cluster,
                    chunk_count=chunk_count, chunk_index=chunk_index, chunk_size=chunk_size
                )
   
            # Cleanup Disk Space
            if not save_intermediate:
                delete_files(results)

    # 5. Final Aggregation
    if chunk_count != 1:
        outlier_file_path = result_manager.ensemble_outlier_chunk(
            total_job_counts, chunk_size, chunk_count, 
            new_coh_day, freq, new_stage, new_freq_deriv_order, 
            cluster, work_in_local_dir
        )
            
    return outlier_file_path

def determine_mean2f_ratio(percentile, paths, target, freq, 
                           old_coh_day, old_freq_deriv_order, old_stage, 
                           new_coh_day, new_freq_deriv_order, new_stage, 
                           cluster=False, work_in_local_dir=False):
    """
    Calculates the ratio of mean2F between two stages.
    """
    
    # Load Old Data
    task_old = task_name(target['name'], old_stage, old_coh_day, old_freq_deriv_order, freq)
    file_old = paths.outlier_file(freq, task_old, old_stage, cluster=cluster)
    if work_in_local_dir:
        file_old = Path(file_old).name
    data_old = fits.getdata(file_old, 1)

    # Load New Data
    task_new = task_name(target['name'], new_stage, new_coh_day, new_freq_deriv_order, freq)
    file_new = paths.outlier_file(freq, task_new, new_stage, cluster=cluster)
    if work_in_local_dir:
        file_new = Path(file_new).name
    data_new = fits.getdata(file_new, 1)

    # Calculate Ratio
    try:
        ratio_distribution = np.sort(data_new['mean2F'] / data_old['mean2F'])
        
        # Get value at specific percentile
        r = np.percentile(ratio_distribution, percentile) 
        r = int(r * 100.) / 100. 
        
        print(f'Ratio = {r} at {(1-percentile)*100:.1f}% percentile (N={ratio_distribution.size}).\n')
    except Exception as e:
        print(f'[Error] Could not calculate ratio: {e}')
        # print(f'Sizes - Before: {data_old.size}, After: {data_new.size}.\n')
        r = 0    
        
    return r