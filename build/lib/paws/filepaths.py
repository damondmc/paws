# This file contains all file path used in the Weave-based SNR search pipeline
from pathlib import Path

class PathManager:
    """
    Centralized management of file paths for the Weave pipeline.
    Replaces static functions and global 'setup' variables with a class
    that derives paths from the loaded config and target YAML.
    """

    def __init__(self, config, target):
        """
        Initialize with configuration dictionaries.
        
        Args:
            config (dict): The loaded config.yaml
            target (dict): The loaded target.yaml (e.g., GalacticCenter.yaml)
        """
        self.config = config
        self.target = target
        
        # Define Roots as Path objects for easy manipulation
        self.home_dir = Path(config['home_dir'])
        self.osdf_dir = Path(config['osdf_dir'])
        
        # Frequently used attributes
        self.target_name = target['name']
        self.sft_source = config['sft_source']
        self.user = config['user']

    # ---------------------------------------------------------
    # Core Executables
    # ---------------------------------------------------------
    
    @property
    def weave_executable(self):
        # Prefer config definition, fallback to default if missing
        return self.config.get('executables', {}).get('weave', 
            '/cvmfs/software.igwn.org/conda/envs/igwn-py39-20231212/bin/lalpulsar_Weave')
    
    @property
    def estimate_upper_limit_executable(self):
        return self.config.get('executables', {}).get('estimate_uls',
            '/cvmfs/software.igwn.org/conda/envs/igwn-py39-20231212/bin/lalpulsar_ComputeFstatMCUpperLimit')    

    @property
    def follow_up_executable(self):
        return self.home_dir / 'scripts/follow_up.py'

    @property
    def upper_limit_executable(self):
        return self.home_dir / 'scripts/upper_limit.py'
    
    @property
    def analyze_result_executable(self):
        return self.home_dir / 'scripts/analyze.py'

    # ---------------------------------------------------------
    # Input Data (SFTs)
    # ---------------------------------------------------------

    def sft_file_path(self, freq, detector='H1', use_osdf=False):
        """
        Returns the DIRECTORY path containing SFTs for a specific frequency.
        """
        if use_osdf:
            root = self.osdf_dir / 'SFTs' / 'o4ab_data' # Adjusted based on your snippet adding 'o4ab'
        else:
            # /home/user/SFTs/o4ab_data/
            root = Path(f'/home/{self.user}/SFTs/o4ab_data')

        # Construct: root / SFTs / H1 / 100
        return root / 'SFTs' / detector / str(int(freq))

    def sft_ensemble(self, freq, use_osdf=False):
        """
        Returns a list of SFT file paths for H1 and L1 detectors.
        """
        h1_path = Path(self.sft_file_path(freq, detector='H1', use_osdf=use_osdf))
        l1_path = Path(self.sft_file_path(freq, detector='L1', use_osdf=use_osdf))
        
        if not use_osdf:
            sft_list = [str(s) for s in h1_path.glob("*.sft")] + [str(s) for s in l1_path.glob("*.sft")]
        else:
            # Assumes osdf path logic where slicing [5:] removes the local mount prefix
            sft_list = ['osdf://' + str(s)[5:] for s in h1_path.glob("*.sft")] + \
                       ['osdf://' + str(s)[5:] for s in l1_path.glob("*.sft")]     
        return sft_list

    # ---------------------------------------------------------
    # Condor & DAG Management
    # ---------------------------------------------------------

    def dag_group_file(self, f_min, f_max, stage):
        """Path to the text file listing all DAGs for a band."""
        filename = f"{self.target_name}_{stage}_{f_min}-{f_max}Hz_dag.txt"
        return self.home_dir / 'dagJob' / filename

    def dag_file(self, freq, task_name, stage):
        """Path to the specific .dag file."""
        return self.home_dir / 'condorFiles' / stage / self.target_name / str(freq) / f"{task_name}_{freq}Hz.dag"

    def condor_sub_file(self, freq, task_name, stage):
        """Path to the .sub file."""
        return self.home_dir / 'condorFiles' / stage / self.target_name / str(freq) / f"{task_name}_{freq}Hz.sub"
    
    def submit_condor_sub_file(self, freq, stage):
        """Path to the submit-wrapper .sub file."""
        return self.home_dir / 'condorFiles' / stage / self.target_name / str(freq) / 'submit.sub'

    def condor_record_files(self, freq, task_name, stage):
        """
        Returns a list [Output, Error, Log] for Condor logging.
        Note: These contain $(JobID) or similar Condor variables, so they are returned as strings.
        """
        base_dir = self.home_dir / 'results' / stage / self.target_name / self.sft_source / str(freq)
        
        # Ensure parent log directories exist immediately (optional but recommended)
        (base_dir / 'OUT').mkdir(parents=True, exist_ok=True)
        (base_dir / 'ERR').mkdir(parents=True, exist_ok=True)
        (base_dir / 'LOG').mkdir(parents=True, exist_ok=True)

        out = base_dir / 'OUT' / f"{task_name}_{freq}Hz.out.$(JobID)"
        err = base_dir / 'ERR' / f"{task_name}_{freq}Hz.err.$(JobID)"
        log = base_dir / 'LOG' / f"{task_name}_{freq}Hz_log.txt.$(JobID)"
        
        return [str(out), str(err), str(log)]

    # ---------------------------------------------------------
    # Weave Outputs & Analysis
    # ---------------------------------------------------------

    def weave_output_file(self, freq, task_name, job_index, stage):
        """
        Path where Weave writes the resulting FITS file.
        Matches: /osdf/.../o4ab/results/...
        """
        # Logic: OSDFDir + 'o4ab' + results structure
        base = self.osdf_dir / 'o4ab' / 'results' / stage / self.target_name / self.sft_source / str(freq) / 'Result'
        return base / f"{task_name}_{freq}Hz.fts.{job_index}"

    def outlier_file(self, freq, task_name, stage, cluster=False):
        """Path for the analyzed outlier file."""
        base = self.home_dir / 'results' / stage / self.target_name / self.sft_source / str(freq) / 'Outliers'
        
        if cluster:
            filename = f"{task_name}_{freq}Hz_outlier_clustered.fts"
        else:
            filename = f"{task_name}_{freq}Hz_outlier.fts"
            
        return base / filename

    def outlier_from_saturated_file(self, freq, task_name, stage):
        base = self.home_dir / 'results' / stage / self.target_name / self.sft_source / str(freq) / 'Outliers'
        return base / f"{task_name}_{freq}Hz_loudest_outlier_from_saturated.fts"