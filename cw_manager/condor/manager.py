from pathlib import Path
import time
from tqdm import tqdm

from . import writer as wc
from ..io import make_dir
from ..definitions import phase_param_name
from ..file_paths import PathManager

class CondorManager:
    """
    Manages the creation of HTCondor DAG and SUB files for Weave searches.
    
    This class orchestrates the interface between the user configuration, 
    file system paths, and the writer module to generate workflow files.
    """

    def __init__(self, target, config):
        """
        Initialize the CondorManager.

        Parameters:
            target (dict): Target object containing astronomical target info.
            config (dict): Configuration dictionary (user, accounting, etc.).
        """
        self.config = config
        self.target = target
        self.paths = PathManager(config, target)
        
        # Internal state for parameter names, set during DAG creation
        self.freq_param_names = []
        self.freq_deriv_param_names = []
        self.num_top_list = 0

    def write_sub(self, freq, stage, task_name, cr_files, arg_str, request_memory, request_disk, request_cpu, 
                  use_osg, use_osdf, exe=None, transfer_executable=False, image=None):
        """
        Prepares paths and invokes the writer to create the HTCondor .sub file.
        """
        if exe is None:
            exe = self.paths.weave_executable
        
        sub_file_path = self.paths.condor_sub_file(freq, task_name, stage)
        
        # Ensure directory exists and clean up previous versions
        sub_file_path.parent.mkdir(parents=True, exist_ok=True)
        sub_file_path.unlink(missing_ok=True)
        
        wc.writeSearchSub(
            filename=str(sub_file_path), 
            executable_path=str(exe), 
            transfer_executable=transfer_executable, 
            output=str(cr_files[0]), 
            error_path=str(cr_files[1]), 
            log_path=str(cr_files[2]), 
            arg_list_string=arg_str,
            accounting_group=self.config['accGroup'],
            user=self.config['user'],
            request_memory=request_memory, 
            request_disk=request_disk, 
            request_cpu=request_cpu,
            use_osg=use_osg, 
            use_osdf=use_osdf,
            image=image
        )
        return sub_file_path

    def _get_execution_kwargs(self, n_seg):
        """Helper to generate common keyword arguments for the executable."""
        extra_stats = "coh2F_det,mean2F,coh2F_det,mean2F_det"
        kwargs = {
            "semi-max-mismatch": self.config['semiMM'],
            "toplist-limit": self.num_top_list,
            "extra-statistics": extra_stats
        }
        
        if n_seg != 1:
            kwargs["coh-max-mismatch"] = self.config['cohMM']
            
        return kwargs

    def search_dag_args(self, freq, stage, params, task_name, n_seg, sft_files, job_index, use_osg, metric):
        """
        Generates the argument string (VARS) for a specific DAG node.
        """
        result_file = self.paths.weave_output_file(freq, task_name, job_index, stage)
        make_dir([result_file])
        
        kwargs = self._get_execution_kwargs(n_seg)
        arg_list = ""
        
        if not use_osg:
            # --- Local Execution Formatting ---
            # Local execution typically passes the entire command line as a single 'argList' variable
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
            # --- OSG Execution Formatting ---
            # OSG execution defines individual VARS that populate the submit file template
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
                # Convert "semi-max-mismatch" -> "SEMIMAXMISMATCH"
                var_name = key.replace('-', '').upper()
                args.append(f'{var_name}="{value}"')        
            
            args.append(f'ALPHA="{params["alpha"]}" DALPHA="{self.target["dalpha"]}"')
            args.append(f'DELTA="{params["delta"]}" DDELTA="{self.target["ddelta"]}"')
            
            for key1, key2 in zip(self.freq_param_names, self.freq_deriv_param_names): 
                args.append(f'{key1.upper()}="{params[key1]}"')
                args.append(f'{key2.upper()}="{params[key2]}"')
            
            arg_list = " ".join(args) + " "
                
        return arg_list
        
    def get_weave_arg_template(self, n_seg): 
        """
        Generates the Submit file argument template using Condor $(VAR) syntax.
        """
        arg_keys = [
            "output-file", "sft-files", "setup-file", 
            "semi-max-mismatch", "toplist-limit", "extra-statistics"
        ]
        
        if n_seg != 1:
            arg_keys.insert(3, "coh-max-mismatch")
            
        # Build template: --key=$({KEY_UPPER})
        template_parts = [
            f"--{key}=$({key.replace('-', '').upper()})" for key in arg_keys
        ]
            
        template_parts.append("--alpha=$(ALPHA)/$(DALPHA)")
        template_parts.append("--delta=$(DELTA)/$(DDELTA)")
        
        for key1, key2 in zip(self.freq_param_names, self.freq_deriv_param_names):
            template_parts.append(f"--{key1}=$({key1.upper()})/$({key2.upper()})")
        
        return " ".join(template_parts) + " "
    
    def make_search_dag(self, task_name, freq, param_chunks, num_top_list, stage, freq_deriv_order, n_seg,
                        sft_files, request_memory='18GB', request_disk='5GB', request_cpu=1, 
                        use_osg=False, use_osdf=False, metric='None',
                        exe=None, image=None):
        """
        Main driver: Creates the DAG and SUB files for a frequency band search.
        
        
        """
        t0 = time.time()
        
        if use_osdf and not use_osg:
            print('Warning: You are reading SFTs from OSDF but not using OSG computing resources.')

        # Initialize state for this run
        self.freq_param_names, self.freq_deriv_param_names = phaseParamName(freq_deriv_order)
        self.num_top_list = num_top_list
            
        dag_file_path = self.paths.dag_file(freq, task_name, stage)
        dag_file_path.parent.mkdir(parents=True, exist_ok=True)
        dag_file_path.unlink(missing_ok=True)
        
        cr_files = self.paths.condor_record_files(freq, task_name, stage)
        makeDir(cr_files)

        # Create SUB file
        arg_template = self.get_weave_arg_template(n_seg)
        sub_file_path = self.write_sub(
            freq, stage, task_name, cr_files, arg_template, 
            request_memory=request_memory, 
            request_disk=request_disk, 
            request_cpu=request_cpu,
            use_osg=use_osg, 
            use_osdf=use_osdf, 
            exe=exe, 
            image=image
        )
        
        # Write DAG entries
        print(f"Generating DAG for {freq} Hz...")
        for job_index, params in tqdm(enumerate(param_chunks, 1), total=param_chunks.size):
            arg_list = self.search_dag_args(
                freq, stage, params, task_name, n_seg, sft_files, job_index, use_osg, metric
            )
            wc.writeSearchDag(
                str(dag_file_path), 
                task_name, 
                str(sub_file_path), 
                job_index, 
                arg_list
            )

        elapsed = time.time() - t0
        print(f'Finished writing {stage} dag files for {freq} Hz')
        print(f'Time used = {elapsed:.2f}s')
        
        return dag_file_path