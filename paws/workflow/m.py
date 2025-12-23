import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .writer import write_search_subfile, write_search_dagfile
from paws.io import make_dir
from paws.definitions import phase_param_name, task_name
from paws.filepaths import PathManager

class WorkflowManager:
    """
    Central manager for creating HTCondor DAG and SUB files for all 
    analysis stages (Search, Follow-up, Upper Limit).
    """

    def __init__(self, config, target):
        """
        Initialize the WorkflowManager.

        Parameters:
            target (dict): Target object containing astronomical target info.
            config (dict): Configuration dictionary (user, accounting, etc.).
        """
        self.config = config
        self.target = target
        self.paths = PathManager(config, target)
        self.obs_day = config.get('obs_day', 0) # Assumes config has obs_day, or defaults to 0

        # Internal state for search parameter names
        self.freq_param_names = []
        self.freq_deriv_param_names = []
        self.num_top_list = 0

    # =========================================================================
    #  SECTION 1: SEARCH STAGE
    # =========================================================================

    def _get_execution_kwargs(self, n_seg):
        """Helper to generate common keyword arguments for the search executable."""
        extra_stats = "coh2F_det,mean2F,coh2F_det,mean2F_det"
        kwargs = {
            "semi-max-mismatch": self.config['semi_mm'],
            "toplist-limit": self.num_top_list,
            "extra-statistics": extra_stats
        }
        
        if n_seg != 1:
            kwargs["coh-max-mismatch"] = self.config['coh_mm']
            
        return kwargs

    def _get_weave_arg_string(self, n_seg): 
        """Generates the Submit file argument template using Condor $(VAR) syntax."""
        arg_keys = [
            "output-file", "sft-files", "setup-file", 
            "semi-max-mismatch", "toplist-limit", "extra-statistics"
        ]
        
        if n_seg != 1:
            arg_keys.insert(3, "coh-max-mismatch")
            
        # Build template: --key=$({KEY_UPPER})
        template_parts = [f"--{key}=$({key.replace('-', '').upper()})" for key in arg_keys]
            
        template_parts.append("--alpha=$(ALPHA)/$(DALPHA)")
        template_parts.append("--delta=$(DELTA)/$(DDELTA)")
        
        for key1, key2 in zip(self.freq_param_names, self.freq_deriv_param_names):
            template_parts.append(f"--{key1}=$({key1.upper()})/$({key2.upper()})")
        
        return " ".join(template_parts) + " "

    def _search_dag_args(self, freq, stage, params, task_name, n_seg, sft_files, job_index, use_osg, metric):
        """Generates the argument string (VARS) for a specific Search DAG node."""
        result_file = self.paths.weave_output_file(freq, task_name, job_index, stage)
        make_dir([result_file])
        
        kwargs = self._get_execution_kwargs(n_seg)
        arg_list = ""
        
        if not use_osg:
            # Local Execution
            cmd_parts = [
                f"--output-file={result_file}",
                f"--sft-files={';'.join([str(s) for s in sft_files])}",
                f"--setup-file={metric}"
            ]
            for key, value in kwargs.items():  
                cmd_parts.append(f"--{key}={value}")
            
            cmd_parts.append(f"--alpha={params['alpha']}/{self.target['dalpha']}")
            cmd_parts.append(f"--delta={params['delta']}/{self.target['ddelta']}")
        
            for key1, key2 in zip(self.freq_param_names, self.freq_deriv_param_names): 
                cmd_parts.append(f"--{key1}={params[key1]}/{params[key2]}")
            
            arg_list = f'argList= "{" ".join(cmd_parts)} "'

        else: 
            # OSG Execution
            args = [
                f'OUTPUTFILE="{result_file.name}"',
                f'REMAPOUTPUTFILE="{result_file}"',
                f'SETUPFILE="{Path(metric).name}"',
                f'SFTFILES="{";".join([Path(s).name for s in sft_files])}"'
            ]
            
            # File Transfer List
            input_files = ', '.join([str(s) for s in sft_files]) + ', ' + str(metric)
            args.append(f'TRANSFERFILES="{input_files}"')
            
            for key, value in kwargs.items():
                var_name = key.replace('-', '').upper()
                args.append(f'{var_name}="{value}"')        
            
            args.append(f'ALPHA="{params["alpha"]}" DALPHA="{self.target["dalpha"]}"')
            args.append(f'DELTA="{params["delta"]}" DDELTA="{self.target["ddelta"]}"')
            
            for key1, key2 in zip(self.freq_param_names, self.freq_deriv_param_names): 
                args.append(f'{key1.upper()}="{params[key1]}"')
                args.append(f'{key2.upper()}="{params[key2]}"')
            
            arg_list = " ".join(args) + " "
                
        return arg_list

    def make_search_dag(self, task_name, freq, params, num_top_list, stage, freq_deriv_order, n_seg,
                        sft_files, metric_file, request_memory='18GB', request_disk='5GB', request_cpu=1, 
                        use_osg=False, use_osdf=False, exe=None, image=None):
        """
        Creates the DAG and SUB files for the Search stage (Weave).
        """
        t0 = time.time()
        print(f"Generating SEARCH DAG for {task_name}...")

        if use_osdf and not use_osg:
            print('Warning: SFTs from OSDF requested but not using OSG resources.')

        self.freq_param_names, self.freq_deriv_param_names = phase_param_name(freq_deriv_order)
        self.num_top_list = num_top_list
            
        dag_file_path = self.paths.dag_file(freq, task_name, stage)
        dag_file_path.parent.mkdir(parents=True, exist_ok=True)
        dag_file_path.unlink(missing_ok=True)
        
        cr_files = self.paths.condor_record_files(freq, task_name, stage)
        make_dir(cr_files)

        arg_string = self._get_weave_arg_string(n_seg)
        exe = exe if exe else self.paths.weave_executable
        
        sub_file_path = self.paths.condor_sub_file(freq, task_name, stage)
        sub_file_path.parent.mkdir(parents=True, exist_ok=True)
        sub_file_path.unlink(missing_ok=True)
        
        write_search_subfile(
            filename=str(sub_file_path), executable_path=str(exe), transfer_executable=False, 
            output_path=str(cr_files[0]), error_path=str(cr_files[1]), log_path=str(cr_files[2]), 
            arg_list_string=arg_string, accounting_group=self.config['acc_group'], user=self.config['user'],
            request_memory=request_memory, request_disk=request_disk, request_cpu=request_cpu,
            use_osg=use_osg, use_osdf=use_osdf, image=image
        )   
    
        for job_index, params in tqdm(enumerate(params, 1), total=params.size):
            arg_list = self._search_dag_args(
                freq, stage, params, task_name, n_seg, sft_files, job_index, use_osg, metric_file
            )
            write_search_dagfile(str(dag_file_path), task_name, str(sub_file_path), job_index, arg_list)

        elapsed = time.time() - t0
        print(f'Finished writing {stage} dag files. Time: {elapsed:.2f}s')
        return dag_file_path

    # =========================================================================
    #  SECTION 2: UPPER LIMIT STAGE
    # =========================================================================

    def _upper_limit_args(self, metric, coh_day, freq, stage, freq_deriv_order, num_top_list, n_inj, 
                          sky_uncertainty, h0est, num_cpus, sft_files, cluster, work_in_local_dir, use_osdf):
        """Generates command line arguments for the python upper limit script."""
        if work_in_local_dir:
            metric = Path(metric).name

        sft_files_str = ';'.join([Path(s).name for s in sft_files])
        
        # Construct argument string
        arg_list_string = (
            f'--target {self.target["name"]} --obsDay {self.obs_day} --cohDay {coh_day} '
            f'--freq {freq} --stage {stage} --freqDerivOrder {freq_deriv_order} '
            f'--numTopList {num_top_list} --nInj {n_inj} --sftFiles {sft_files_str} '
            f'--num_cpus {num_cpus} --skyUncertainty {sky_uncertainty} --h0est {h0est} --metric {metric}'
        )

        if cluster:
            arg_list_string += " --cluster"
        if work_in_local_dir:
            arg_list_string += " --workInLocalDir"
        if use_osdf:
            arg_list_string += " --OSDF"
            
        return arg_list_string

    def _ul_transfer_args(self, coh_day, freq, freq_deriv_order, metric, stage, sft_files, cluster, use_osg):
        """Generates VARS for OSG file transfers for Upper Limits."""
        # Find the Search Result file (input for UL)
        search_taskname = task_name(self.target['name'], 'search', coh_day, freq_deriv_order)
        search_result_file = self.paths.outlier_file(freq, search_taskname, 'search', cluster=cluster)
        
        exe = self.paths.upper_limit_executable
        image = self.paths.container_image
        
        # Build Input Files list
        input_files_list = [str(exe), str(image), str(search_result_file)]
        input_files_list.extend([str(s) for s in sft_files])
        input_files_list.append(str(metric))
        
        input_files_str = ", ".join(input_files_list)

        # Output File (The result of the UL analysis)
        ul_taskname = task_name(self.target['name'], stage, coh_day, freq_deriv_order)
        outlier_file_path = self.paths.outlier_file(freq, ul_taskname, stage, cluster=cluster)
        make_dir([outlier_file_path])

        arg_list = (
            f'OUTPUTFILE="{Path(outlier_file_path).name}" '
            f'REMAPOUTPUTFILE="{outlier_file_path}" '
            f'TRANSFERFILES="{input_files_str}" '
        )
        return arg_list

    def make_upper_limit_dag(self, fmin, fmax, coh_day, sft_files, freq_deriv_order=2, metric='', 
                             stage='upperLimit', sky_uncertainty=1e-4, h0est=[1e-25], n_inj=100, 
                             num_top_list=1000, num_cpus=4, request_memory='4GB', request_disk='4GB', 
                             cluster=False, work_in_local_dir=False, use_osg=False, use_osdf=False):
        """
        Creates the DAG and SUB files for the Upper Limit stage.
        """
        ul_task_name = task_name(self.target['name'], stage, coh_day, freq_deriv_order, f'{fmin}-{fmax}')
        dag_file_path = self.paths.dag_file('', ul_task_name, stage) # Empty freq for ensemble dag
        dag_file_path.unlink(missing_ok=True)
        
        print(f"Generating UPPER LIMIT DAG for {ul_task_name}...")

        # Loop through frequency bands
        for job_index, freq in tqdm(enumerate(range(fmin, fmax), 1), total=fmax-fmin):
            
            taskname_freq = task_name(self.target['name'], stage, coh_day, freq_deriv_order, freq)
            
            # Executable setup
            exe = self.paths.upper_limit_executable
            local_exe_name = Path(exe).name
            image = self.paths.container_image
            local_image_name = Path(image).name

            # Paths
            sub_file_path = self.paths.condor_sub_file(freq, taskname_freq, stage)
            sub_file_path.unlink(missing_ok=True)
            cr_files = self.paths.condor_record_files(freq, taskname_freq, stage)
            make_dir(cr_files)
            
            # NOTE: Assuming sft_files passed in are valid for this freq, or using utils to find them if specific
            # Use specific h0est for this frequency band if list provided
            current_h0 = h0est[job_index-1] if isinstance(h0est, list) else h0est

            # Arguments for the python script
            script_args = self._upper_limit_args(
                metric, coh_day, freq, stage, freq_deriv_order, num_top_list, n_inj, 
                sky_uncertainty, current_h0, num_cpus, sft_files, cluster, work_in_local_dir, use_osdf
            )
            
            # Write SUB file
            write_search_subfile(
                filename=str(sub_file_path), executable_path=str(local_exe_name), transfer_executable=True,
                output=str(cr_files[0]), error_path=str(cr_files[1]), log_path=str(cr_files[2]),
                arg_list_string=script_args, request_memory=request_memory, request_disk=request_disk,
                use_osg=use_osg, use_osdf=use_osdf, image=local_image_name,
                accounting_group=self.config['acc_group'], user=self.config['user']
            )
            
            # Arguments for DAG (OSG Transfer Logic)
            dag_vars = self._ul_transfer_args(
                coh_day, freq, freq_deriv_order, metric, stage, sft_files, cluster, use_osg
            )
            
            write_search_dagfile(str(dag_file_path), taskname_freq, str(sub_file_path), job_index, dag_vars)
            
        print(f'Finish writing upper limit dag: {fmin}-{fmax}Hz')
        return dag_file_path

    # =========================================================================
    #  SECTION 3: FOLLOW-UP STAGE
    # =========================================================================

    def _followup_args(self, h0, coh_day, freq, stage, freq_deriv_order, num_top_list, sft_files, 
                       request_cpu, real, inj, cluster, work_in_local_dir): 
        """Generates command line arguments for the python follow-up script."""
        
        sft_files_str = ';'.join([Path(s).name for s in sft_files])
        
        arg_list_string = (
            f'--target {self.target["name"]} --obsDay {self.obs_day} --cohDay {coh_day} '
            f'--freq {freq} --stage {stage} --freqDerivOrder {freq_deriv_order} '
            f'--numTopList {num_top_list} --sftFiles {sft_files_str} --num_cpus {request_cpu} --h0 {h0}'
        )
        
        if real:
            arg_list_string += " --real"
        if inj:
            arg_list_string += " --inj"
        if cluster:
            arg_list_string += " --cluster" 
        if work_in_local_dir:
            arg_list_string += " --workInLocalDir"
        
        return arg_list_string

    def _followup_transfer_args(self, config_file, coh_day, freq, freq_deriv_order, stage, sft_files, 
                                old_stage, cluster, use_osg, use_osdf, from_saturated_band):       
        """Generates VARS for OSG file transfers for Follow-up."""
        
        # Determine previous stage file (Input)
        prev_taskname = task_name(self.target['name'], old_stage, coh_day, freq_deriv_order)
        if not from_saturated_band:
            search_result_file = self.paths.outlier_file(freq, prev_taskname, old_stage, cluster=cluster) 
        else:
            search_result_file = self.paths.outlier_from_saturated_file(freq, prev_taskname, old_stage) # Assuming this method exists in paths
        
        exe = self.paths.followup_executable
        image = self.paths.container_image
        
        # Build Input Files
        input_files_list = [str(exe), str(image), str(search_result_file)]
        input_files_list.extend([str(s) for s in sft_files])
        
        # Add Metrics
        # 1. Initial stage metric
        _, coh_time, n_seg, _, _ = utils.getTimeSetup(self.target['name'], self.obs_day, coh_day)
        metric = self.paths.weave_setup_file(coh_time, n_seg, freq_deriv_order) # Assuming method exists
        input_files_list.append(str(metric))
    
        # 2. Metrics for subsequent follow-up stages defined in config file
        coh_day_list, freq_deriv_list = np.loadtxt(config_file, skiprows=2, dtype=('i4', 'i4')).T 
        # Handle single line case
        if coh_day_list.ndim == 0:
             coh_day_list, freq_deriv_list = [coh_day_list], [freq_deriv_list]

        for _c_day, _f_order in zip(coh_day_list, freq_deriv_list):
            _, _ctime, _nseg, _, _ = utils.getTimeSetup(self.target['name'], self.obs_day, _c_day)
            _metric = self.paths.weave_setup_file(_ctime, _nseg, _f_order)
            input_files_list.append(str(_metric))
            
        input_files_str = ", ".join(input_files_list)

        # Output File
        curr_taskname = task_name(self.target['name'], stage, coh_day, freq_deriv_order)
        outlier_file_path = self.paths.outlier_file(freq, curr_taskname, stage, cluster=cluster)
        make_dir([outlier_file_path])
        
        arg_list = (
            f'OUTPUTFILE="{Path(outlier_file_path).name}" '
            f'REMAPOUTPUTFILE="{outlier_file_path}" '
            f'TRANSFERFILES="{input_files_str}" '
        )
        return arg_list

    def make_followup_dag(self, config_file, fmin, fmax, h0, sft_files, stage='followUp', num_top_list=1000, 
                          request_cpu=4, request_disk='4GB', old_stage='search', real=True, inj=False, 
                          cluster=False, work_in_local_dir=False, use_osg=False, use_osdf=False, 
                          from_saturated_band=False):
        """
        Creates the DAG and SUB files for the Follow-up stage.
        """
        # Read schedule from config file to get initial params
        data = np.loadtxt(config_file)
        if data.ndim == 1:
            coh_day, freq_deriv_order = data[0], data[1]
        else:
            coh_day, freq_deriv_order = data[0][0], data[0][1]

        coh_day = int(coh_day)
        freq_deriv_order = int(freq_deriv_order)
        
        fu_task_name = task_name(self.target['name'], stage, coh_day, freq_deriv_order, f'{fmin}-{fmax}')
        dag_file_path = self.paths.dag_file('', fu_task_name, stage)
        dag_file_path.unlink(missing_ok=True)

        print(f"Generating FOLLOW-UP DAG for {fu_task_name}...")
        
        for job_index, freq in tqdm(enumerate(range(fmin, fmax), 1), total=fmax-fmin):
            taskname_freq = task_name(self.target['name'], stage, coh_day, freq_deriv_order, freq)
            
            # Executable
            exe = self.paths.followup_executable
            local_exe_name = Path(exe).name
            image = self.paths.container_image
            local_image_name = Path(image).name
            
            # Paths
            sub_file_path = self.paths.condor_sub_file(freq, taskname_freq, stage)
            sub_file_path.unlink(missing_ok=True)
            cr_files = self.paths.condor_record_files(freq, taskname_freq, stage)
            make_dir(cr_files)
            
            # Arguments
            script_args = self._followup_args(
                h0, coh_day, freq, stage, freq_deriv_order, num_top_list, sft_files, 
                request_cpu, real, inj, cluster, work_in_local_dir
            )
            
            write_search_subfile(
                filename=str(sub_file_path), executable_path=str(local_exe_name), transfer_executable=True,
                output=str(cr_files[0]), error_path=str(cr_files[1]), log_path=str(cr_files[2]),
                arg_list_string=script_args, request_memory='4GB', request_disk=request_disk, request_cpu=request_cpu,
                use_osg=use_osg, use_osdf=use_osdf, image=local_image_name,
                accounting_group=self.config['acc_group'], user=self.config['user']
            )
            
            # DAG Vars
            dag_vars = self._followup_transfer_args(
                config_file, coh_day, freq, freq_deriv_order, stage, sft_files, 
                old_stage, cluster, use_osg, use_osdf, from_saturated_band
            )
                             
            write_search_dagfile(str(dag_file_path), taskname_freq, str(sub_file_path), job_index, dag_vars)
            
        print(f'Finish writing follow-up dag: {fmin}-{fmax}Hz')
        return dag_file_path
