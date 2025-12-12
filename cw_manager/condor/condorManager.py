from . import writeCondor as wc
from pathlib import Path
from ..io import makeDir
from ..definitions import phaseParamName
from ..filePath import PathManager
from tqdm import tqdm
import time

class condorManager:
    """
    Class to manage condor DAG file creation for Weave searches.
    """
    def __init__(self, target, config):
        """
        Initialize the condorManager with target and configuration.
        Args:
            target (dict): Target object containing target information.
            config (dict): Configuration dictionary with search parameters.
        """
        self.config = config
        self.target = target
        # Initialize PathManager to handle all file paths
        self.paths = PathManager(config, target)
    
    def weaveArgs(self, freq, params, task_name, nSeg, sftFiles, jobIndex, OSG=True, metric='None'):
        """
        Generate the argument string for the Weave executable.
        """
        # Get the output path from PathManager
        resultFile = self.paths.weave_output_file(freq, task_name, jobIndex, self.stage)
        
        # Ensure the directory for the output file exists
        makeDir([resultFile])
        
        extraStats = "coh2F_det,mean2F,coh2F_det,mean2F_det"
        
        # Determine arguments based on segments
        # Note: self.numTopList is set in makeSearchDag
        if nSeg != 1:
            kwargs = {
                "semi-max-mismatch": self.config['semiMM'],
                "coh-max-mismatch": self.config['cohMM'],
                "toplist-limit": self.numTopList,
                "extra-statistics": extraStats
            }
        else:
            kwargs = {
                "semi-max-mismatch": self.config['semiMM'],
                "toplist-limit": self.numTopList,
                "extra-statistics": extraStats
            }
        
        argList = ""
        
        if not OSG:
            # --- Local Execution ---
            argList += "argList= \"--output-file={0} ".format(resultFile)
            argList += "--sft-files={0} ".format(';'.join([str(s) for s in sftFiles]))
            argList += "--setup-file={0} ".format(metric)
            
            for key, value in kwargs.items():  
                argList += "--{0}={1} ".format(key, value)
            
            argList += "--alpha={0}/{1} ".format(params['alpha'], self.target['dalpha'])
            argList += "--delta={0}/{1} ".format(params['delta'], self.target['ddelta'])
        
            for i in range(self.freqDerivOrder + 1):      
                key1, key2 = self.freqParamName[i], self.freqDerivParamName[i]
                argList += "--{0}={1}/{2} ".format(key1, params[key1], params[key2])
            argList += "\""
            
        else: 
            # --- OSG/Grid Execution (File Transfer) ---
            # Condor sees files in the current working directory after transfer
            
            # Use Path(x).name to get just the filename
            argList += "OUTPUTFILE=\"{0}\" ".format(resultFile.name)
            argList += "REMAPOUTPUTFILE=\"{0}\" ".format(resultFile)
            argList += "SETUPFILE=\"{0}\" ".format(Path(metric).name)
            
            # For SFTs in OSG, we usually provide the list of transferred filenames
            sft_names = ';'.join([Path(s).name for s in sftFiles])
            argList += "SFTFILES=\"{0}\" ".format(sft_names)
            
            # Input files to be transferred (Comma separated list)
            # Combine SFT paths and the metric file path
            inputFiles = ', '.join([str(s) for s in sftFiles]) + ', ' + str(metric) 
            argList += "TRANSFERFILES=\"{0}\" ".format(inputFiles)
            
            for key, value in kwargs.items():
                argList += "{0}=\"{1}\" ".format(key.replace('-', '').upper(), value)        
            
            argList += "ALPHA=\"{0}\" DALPHA=\"{1}\" ".format(params['alpha'], self.target['dalpha'])
            argList += "DELTA=\"{0}\" DDELTA=\"{1}\" ".format(params['delta'], self.target['ddelta'])
            
            for i in range(self.freqDerivOrder + 1):
                key1, key2 = self.freqParamName[i], self.freqDerivParamName[i]
                argList += "{0}=\"{1}\" ".format(key1.upper(), params[key1])
                argList += "{0}=\"{1}\" ".format(key2.upper(), params[key2])
                
        return argList
        
    def weaveArgStr(self, nSeg): 
        """
        Generate the argument string template for the Weave executable.
        """ 
        if nSeg != 1:
            argStr = ["output-file", "sft-files", "setup-file", "semi-max-mismatch", "coh-max-mismatch", "toplist-limit", "extra-statistics"]
        else:
            argStr = ["output-file", "sft-files", "setup-file", "semi-max-mismatch", "toplist-limit", "extra-statistics"]
            
        argListString = ""
        for s in argStr:
            # Weave expects arguments like --output-file=$(OUTPUTFILE)
            argListString += "--{0}=$({1}) ".format(s, s.replace('-', '').upper())
            
        argListString += "--alpha=$(ALPHA)/$(DALPHA) --delta=$(DELTA)/$(DDELTA) "
        for i in range(len(self.freqParamName)):
            argListString += "--{0}=$({1})/$({2}) ".format(self.freqParamName[i], self.freqParamName[i].upper(), self.freqDerivParamName[i].upper())
        
        return argListString
    
    def writeSub(self, freq, task_name, crFiles, argStr, request_memory, request_disk, request_cpu, 
                 OSG, OSDF, exe=None, transfer_executable=False, image=None):
        """
        Write the condor .sub file for the Weave executable.
        """
        # Get executable path from PathManager
        if exe is None:
            exe = self.paths.weave_executable
        
        # Get submit file path
        subFileName = self.paths.condor_sub_file(freq, task_name, self.stage)
        
        # Ensure parent dir exists and delete old file
        Path(subFileName).parent.mkdir(parents=True, exist_ok=True)
        Path(subFileName).unlink(missing_ok=True)
        
        # Call writeCondor function to write the .sub file
        wc.writeSearchSub(
            subFileName=str(subFileName), 
            executablePath=str(exe), 
            transfer_executable=transfer_executable, 
            outputPath=str(crFiles[0]), 
            errorPath=str(crFiles[1]), 
            logPath=str(crFiles[2]), 
            argListString=argStr, 
            request_memory=request_memory, 
            request_disk=request_disk, 
            request_cpu=request_cpu,
            OSG=OSG, 
            OSDF=OSDF,
            image=image
        )
        return subFileName
    
    def makeSearchDag(self, task_name, freq, param, numTopList, stage, freqDerivOrder, nSeg,
                      sftFiles, request_memory='18GB', request_disk='5GB', request_cpu=1, 
                      OSG=False, OSDF=False, metric='None',
                      exe=None, image=None):
        """ 
        Make condor DAG file for search stage.
        """
        t0 = time.time()
        
        if OSDF and not OSG:
            print('Warning: You are reading SFTs from OSDF but not using OSG computing resources.')

        # Set instance variables used by weaveArgs
        self.freqParamName, self.freqDerivParamName = phaseParamName(freqDerivOrder)
        self.freqDerivOrder = freqDerivOrder
        self.numTopList = numTopList
        self.stage = stage
                
        # Get DAG file path
        dagFileName = self.paths.dag_file(freq, task_name, self.stage)
        
        # Clean up old DAG
        Path(dagFileName).parent.mkdir(parents=True, exist_ok=True)
        Path(dagFileName).unlink(missing_ok=True)
        
        # Get Log paths [Out, Err, Log]
        crFiles = self.paths.condor_record_files(freq, task_name, self.stage)
        # Note: condor_record_files in updated PathManager already creates directories,
        # but calling makeDir here is safe redundancy.
        makeDir(crFiles)

        # Generate Argument Template
        argStr = self.weaveArgStr(nSeg)
        
        # Write .sub file
        subFileName = self.writeSub(freq, task_name, crFiles, argStr, 
                                    request_memory=request_memory, request_disk=request_disk, request_cpu=request_cpu,
                                    OSG=OSG, OSDF=OSDF, exe=exe, image=image)
        
        # Loop over parameters to write jobs into DAG
        for jobIndex, params in tqdm(enumerate(param, 1), total=param.size):
            # Generate arguments for this specific job
            argList = self.weaveArgs(freq, params, task_name, nSeg, sftFiles, jobIndex, OSG, metric=metric)
            
            # Write job entry to DAG
            wc.writeSearchDag(str(dagFileName), task_name, str(subFileName), jobIndex, argList)

        print('Finish writing {0} dag files for {1} Hz'.format(self.stage, freq))
        print('Time used = {:.2f}s'.format(time.time()-t0))
        
        return dagFileName