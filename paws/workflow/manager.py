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

    def _search_dag_args(self, freq, stage, params, taskname, n_seg, sft_files, job_index, use_osg, metric_file):
        """Generates the argument string (VARS) for a specific Search DAG node."""
        result_file = self.paths.weave_output_file(freq, taskname, job_index, stage)
        make_dir([result_file])
        
        kwargs = self._get_execution_kwargs(n_seg)
        arg_list = ""
        
        if not use_osg:
            # Local Execution
            cmd_parts = [
                f"--output-file={result_file}",
                f"--sft-files={';'.join([str(s) for s in sft_files])}",
                f"--setup-file={metric_file}"
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
                f'SETUPFILE="{Path(metric_file).name}"',
                f'SFTFILES="{";".join([Path(s).name for s in sft_files])}"'
            ]
            
            # File Transfer List
            input_files = ', '.join([str(s) for s in sft_files]) + ', ' + str(metric_file)
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

    def make_search_dag(self, taskname, freq, params, num_top_list, stage, freq_deriv_order, n_seg,
                        sft_files, metric_file, request_memory='18GB', request_disk='5GB', request_cpu=1, 
                        use_osg=False, use_osdf=False, exe=None, image=None):
        """
        Creates the DAG and SUB files for the Search stage (Weave).
        """
        t0 = time.time()
        print(f"Generating SEARCH DAG for {taskname}...")

        if use_osdf and not use_osg:
            print('Warning: SFTs from OSDF requested but not using OSG resources.')

        self.freq_param_names, self.freq_deriv_param_names = phase_param_name(freq_deriv_order)
        self.num_top_list = num_top_list
            
        dag_file_path = self.paths.dag_file(freq, taskname, stage)
        dag_file_path.parent.mkdir(parents=True, exist_ok=True)
        dag_file_path.unlink(missing_ok=True)
        
        cr_files = self.paths.condor_record_files(freq, taskname, stage)
        make_dir(cr_files)

        arg_string = self._get_weave_arg_string(n_seg)
        exe = exe if exe else self.paths.weave_executable
        
        sub_file_path = self.paths.condor_sub_file(freq, taskname, stage)
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
                freq, stage, params, taskname, n_seg, sft_files, job_index, use_osg, metric_file
            )
            write_search_dagfile(str(dag_file_path), taskname, str(sub_file_path), job_index, arg_list)

        elapsed = time.time() - t0
        print(f'Finished writing {stage} dag files. Time: {elapsed:.2f}s')
        return dag_file_path

    # =========================================================================
    #  SECTION 2: UPPER LIMIT STAGE
    # =========================================================================

    def _upperlimit_args(self, taskname, freq, stage, freq_deriv_order, n_seg, num_top_list, 
                         sft_files, metric_file, data_file, n_inj, h0_est, sky_uncertainty, request_cpu, 
                         cluster, work_in_local_dir, use_osdf):
        """Generates command line arguments for the python upper limit script."""
        if work_in_local_dir:
            metric_file = Path(metric_file).name

        sft_files = ';'.join([Path(s).name for s in sft_files])
        
        # Construct argument string
        arg_list_string = (
            f'--taskname {taskname} '
            f'--freq {freq} --stage {stage} --freqDerivOrder {freq_deriv_order} '
            f'--nSeg {n_seg} --numTopList {num_top_list} '
            f'--sftFiles {sft_files} --metric_file {Path(metric_file).name} --data_fule {Path(data_file).name} '
            f'--n_inj {n_inj} --h0_est {h0_est} --sky_uncertainty {sky_uncertainty} '
            f'--request_cpu {request_cpu} '
        )

        if cluster:
            arg_list_string += " --cluster"
        if work_in_local_dir:
            arg_list_string += " --workInLocalDir"
        if use_osdf:
            arg_list_string += " --OSDF"
            
        return arg_list_string

    def _upperlimit_dag_args(self, taskname, freq, stage, sft_files, metric_file, data_file,
                             cluster, exe, image):
        """Generates VARS for OSG file transfers for Upper Limits."""
        # Find the Search Result file (input for UL)
        
        # Build Input Files list
        input_files_list = [str(exe), str(image), str(data_file)]
        input_files_list.extend([str(s) for s in sft_files])
        input_files_list.append(str(metric_file))
        
        input_files = ", ".join(input_files_list)

        # Output File (The result of the UL analysis)
        outlier_file = self.paths.outlier_file(freq, taskname, stage, cluster=cluster)
        make_dir([outlier_file])

        arg_list = (
            f'OUTPUTFILE="{Path(outlier_file).name}" '
            f'REMAPOUTPUTFILE="{outlier_file}" '
            f'TRANSFERFILES="{input_files}" '
        )
        return arg_list
        
    def make_upperlimit_dag(self, taskname, freq, h0_est, num_top_list, stage, freq_deriv_order, n_seg,
                            sft_files, metric_file, data_file, sky_uncertainty=1e-4, n_inj=100, cluster=False,
                            request_memory='4GB', request_disk='4GB', request_cpu=4, 
                            use_osg=False, use_osdf=False, exe=None, image=None, 
                            work_in_local_dir=False):
        """
        Creates the DAG and SUB files for the Upper Limit stage.
        """
        t0 = time.time()
        print(f"Generating SEARCH DAG for {taskname}...")

        if use_osdf and not use_osg:
            print('Warning: SFTs from OSDF requested but not using OSG resources.')

        self.freq_param_names, self.freq_deriv_param_names = phase_param_name(freq_deriv_order)
        self.num_top_list = num_top_list
            
        dag_file_path = self.paths.dag_file(freq, taskname, stage)
        dag_file_path.parent.mkdir(parents=True, exist_ok=True)
        dag_file_path.unlink(missing_ok=True)
        
        cr_files = self.paths.condor_record_files(freq, taskname, stage)
        make_dir(cr_files)

        args = self._upperlimit_args(taskname, freq, stage, freq_deriv_order, n_seg, num_top_list, 
                                     sft_files, metric_file, data_file, 
                                     n_inj, h0_est, sky_uncertainty, request_cpu, 
                                     cluster, work_in_local_dir, use_osdf
        )
        exe = exe if exe else self.paths.upper_limit_executable
        
        sub_file_path = self.paths.condor_sub_file(freq, taskname, stage)
        sub_file_path.parent.mkdir(parents=True, exist_ok=True)
        sub_file_path.unlink(missing_ok=True)
        
        write_search_subfile(
            filename=str(sub_file_path), executable_path=str(exe), transfer_executable=False, 
            output_path=str(cr_files[0]), error_path=str(cr_files[1]), log_path=str(cr_files[2]), 
            arg_list_string=args, accounting_group=self.config['acc_group'], user=self.config['user'],
            request_memory=request_memory, request_disk=request_disk, request_cpu=request_cpu,
            use_osg=use_osg, use_osdf=use_osdf, image=image
        )   
    
        for job_index, params in tqdm(enumerate(params, 1), total=params.size):
            arg_list = self._upperlimit_dag_args(
                taskname, freq, stage, sft_files, metric_file, data_file, cluster, exe, image
            )
            write_search_dagfile(str(dag_file_path), taskname, str(sub_file_path), job_index, arg_list)

        elapsed = time.time() - t0
        print(f'Finished writing {stage} dag files. Time: {elapsed:.2f}s')
        return dag_file_path
    
    # =========================================================================
    #  SECTION 3: FOLLOW-UP STAGE
    # =========================================================================

    def _get_followup_arg_string(self, cluster, work_in_local_dir): 
        """Generates the Submit file argument template using Condor $(VAR) syntax."""

        # arg_keys = [
        #     "target_file", "config_file", "coh_day", "freq", "stage", "freq_deriv_order",
        #     "n_seg", "num_top_list", "sft_files", "metric_file", "sky_uncertainty",
        #     "n_cpus", "n_candidate_per_job", "job_index"
        # ]
        arg_keys = [
            "sft_files", "metric_file", "candidate_file",
            "n_cpus", "n_candidate_per_job", "job_index"
        ]

        # Build template: --key=$({KEY_UPPER})
        arg_list_string = [f"--{key}=$({key.replace('_', '').upper()})" for key in arg_keys]
        
        if cluster:
            arg_list_string.append("--cluster") 
        
        if work_in_local_dir:
            arg_list_string.append("--work_in_local_dir")
        
        # Join handles the spaces between arguments automatically
        return " ".join(arg_list_string)

    def _followup_dag_args(self, freq, stage, candiate_file, n_candidate_per_job, taskname, 
                           sft_files, job_index, cluster, n_cpus, use_osg, exe, metric_file, image):
        """Generates the argument string (VARS) for a specific Search DAG node."""
        result_file = self.paths.outlier_file(freq, taskname, stage, cluster=cluster)
        make_dir([result_file])
        
        # arg_keys = [
        #     "target_file", "config_file", "coh_day", "freq", "stage", "freq_deriv_order",
        #     "n_seg", "num_top_list", "sft_files", "metric_file", "sky_uncertainty",
        #     "n_cpus", "n_candidate_per_job", "job_index"
        # ]

        arg_keys = [
            "sft_files", "metric_file", "candidate_file",
            "n_cpus", "n_candidate_per_job", "job_index"
        ]
        
        arg_list = ""
        
        if not use_osg:
            print('Warning: Follow-up stage is designed to run on OSG. Local execution may not function as intended.')
            # Local Execution
            # cmd_parts = [
            #     #f"--output-file={result_file+f'.{job_index}'}",
            #     f"--sft_files={';'.join([str(s) for s in sft_files])}",
            #     f"--metric_file={metric_file}"
            # ]
            # for key, value in kw.items():  
            #     cmd_parts.append(f"--{key}={value}")
                        
            # arg_list = f'argList= "{" ".join(cmd_parts)} "'

        else: 
            # OSG Execution
            args = [
                f'OUTPUTFILE="{result_file.name}"',
                f'REMAPOUTPUTFILE="{result_file}.{job_index}"',
                f'METRICFILE="{Path(metric_file).name}"',
                f'SFTFILES="{";".join([Path(s).name for s in sft_files])}"'
            ]
            
            # File Transfer List
            input_files = ', '.join([str(s) for s in sft_files]) + ', ' + str(metric_file) + ', ' + str(candiate_file) + ', ' + str(exe)  + ', ' + str(image)
            args.append(f'TRANSFERFILES="{input_files}"')
            
            args.append(f'SFTFILES="{";".join([Path(s).name for s in sft_files])}"')
            args.append(f'METRICFILE="{Path(metric_file).name}"')
            args.append(f'CANDIDATEFILE="{Path(candiate_file).name}"')
            args.append(f'NCPUS="{n_cpus}"')
            args.append(f'NCANDIDATEPERJOB="{n_candidate_per_job}"')
            args.append(f'JOBINDEX="{job_index}"')        
            
            arg_list = " ".join(args) + " "
                
        return arg_list


    def make_followup_dag(self, taskname, freq, n_jobs, n_candidate_per_job, 
                          candiate_file,
                          num_top_list, stage, freq_deriv_order,
                          sft_files, metric_file, cluster,
                          request_memory='18GB', request_disk='5GB', request_cpu=1, 
                          use_osg=False, use_osdf=False, exe=None, image=None):

        """
        Creates the DAG and SUB files for the Follow-up stage.
        """
        t0 = time.time()
        print(f"Generating SEARCH DAG for {taskname}...")

        if use_osdf and not use_osg:
            print('Warning: SFTs from OSDF requested but not using OSG resources.')

        self.freq_param_names, self.freq_deriv_param_names = phase_param_name(freq_deriv_order)
        self.num_top_list = num_top_list

            
        dag_file_path = self.paths.dag_file(freq, taskname, stage)
        dag_file_path.parent.mkdir(parents=True, exist_ok=True)
        dag_file_path.unlink(missing_ok=True)
        
        cr_files = self.paths.condor_record_files(freq, taskname, stage)
        make_dir(cr_files)

        arg_string = self._get_followup_arg_string(cluster, work_in_local_dir=True)
        exe = exe if exe else self.paths.follow_up_executable
        
        sub_file_path = self.paths.condor_sub_file(freq, taskname, stage)
        sub_file_path.parent.mkdir(parents=True, exist_ok=True)
        sub_file_path.unlink(missing_ok=True)
        
        write_search_subfile(
            filename=str(sub_file_path), executable_path=Path(exe).name, transfer_executable=False, 
            output_path=str(cr_files[0]), error_path=str(cr_files[1]), log_path=str(cr_files[2]), 
            arg_list_string=arg_string, accounting_group=self.config['acc_group'], user=self.config['user'],
            request_memory=request_memory, request_disk=request_disk, request_cpu=request_cpu,
            use_osg=use_osg, use_osdf=use_osdf, image=Path(image).name
        )   
    
        for job_index in tqdm(range(1, n_jobs + 1), total=n_jobs):
            arg_list = self._followup_dag_args(
                freq, stage, candiate_file, n_candidate_per_job, taskname, sft_files, job_index, cluster,
                request_cpu, use_osg, exe, metric_file, image
            )

            write_search_dagfile(str(dag_file_path), taskname, str(sub_file_path), job_index, arg_list)

        elapsed = time.time() - t0
        print(f'Finished writing {stage} dag files. Time: {elapsed:.2f}s')
        return dag_file_path