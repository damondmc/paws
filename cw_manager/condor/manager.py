from . import writer as wc
from cw_manager.io import makeDir
from cw_manager.definitions import phaseParamName
from cw_manager.filePath import PathManager

from pathlib import Path
from tqdm import tqdm
import time

class CondorManager:
    """
    Class to manage Condor DAG file creation for Weave searches and Upper Limits.
    Handles the interface between the configuration, file paths, and the writer.
    """
    def __init__(self, target, config):
        """
        Initialize the CondorManager.
        
        Parameters:
            target (dict): Target object containing target information.
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.target = target
        self.paths = PathManager(config, target)
    
    def writeSub(self, freq, stage, task_name, crFiles, argStr, request_memory, request_disk, request_cpu, 
                 OSG, OSDF, exe=None, transfer_executable=False, image=None):
        """
        Prepares paths and calls the writer to create the .sub file.
        
        Parameters:
            freq (int): Frequency of the search.
            stage (str): Stage of the search (e.g., 'search', 'followup').
            task_name (str): Name of the task.
            crFiles (list): List of paths [Out, Err, Log] for Condor records.
            argStr (str): Argument string template for the executable.
            request_memory (str): Memory request.
            request_disk (str): Disk request.
            request_cpu (int): Number of CPUs.
            OSG (bool): Whether to use OSG resources.
            OSDF (bool): Whether to use OSDF for SFTs.
            exe (Path, optional): Path to executable.
            transfer_executable (bool): Whether to transfer the executable.
            image (Path, optional): Path to Singularity image.

        Returns:
            subFileName (Path): Path to the written .sub file.
        """
        if exe is None:
            exe = self.paths.weave_executable
        
        subFileName = self.paths.condor_sub_file(freq, task_name, stage)
        
        # Ensure directory exists and clean up old file
        Path(subFileName).parent.mkdir(parents=True, exist_ok=True)
        Path(subFileName).unlink(missing_ok=True)
        
        wc.writeSearchSub(
            subFileName=str(subFileName), 
            executablePath=str(exe), 
            transfer_executable=transfer_executable, 
            outputPath=str(crFiles[0]), 
            errorPath=str(crFiles[1]), 
            logPath=str(crFiles[2]), 
            argListString=argStr,
            accounting_group=self.config['accGroup'],
            user=self.config['user'],
            request_memory=request_memory, 
            request_disk=request_disk, 
            request_cpu=request_cpu,
            OSG=OSG, 
            OSDF=OSDF,
            image=image
        )
        return subFileName

    def searchDagArgs(self, freq, stage, params, task_name, nSeg, sftFiles, jobIndex, OSG=True, metric='None'):
        """
        Generate the specific argument string for a single Weave job (DAG node).

        Parameters:
            freq (int): Frequency of the search.
            stage (str): Stage of the search.
            params (dict): Dictionary of parameters for this specific job instance.
            task_name (str): Name of the task.
            nSeg (int): Number of segments.
            sftFiles (list): List of SFT files.
            jobIndex (int): Index of the job.
            OSG (bool): Whether to use OSG resources.
            metric (str): Path to the metric file.

        Returns:
            argList (str): Argument string for the Weave executable.
        """
        resultFile = self.paths.weave_output_file(freq, task_name, jobIndex, stage)
        makeDir([resultFile])
        
        extraStats = "coh2F_det,mean2F,coh2F_det,mean2F_det"
        
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
        
            for key1, key2 in zip(self.freqParamName, self.freqDerivParamName): 
                argList += "--{0}={1}/{2} ".format(key1, params[key1], params[key2])
            argList += "\""
            
        else: 
            # --- OSG Execution ---
            argList += "OUTPUTFILE=\"{0}\" ".format(resultFile.name)
            argList += "REMAPOUTPUTFILE=\"{0}\" ".format(resultFile)
            argList += "SETUPFILE=\"{0}\" ".format(Path(metric).name)
            
            sft_names = ';'.join([Path(s).name for s in sftFiles])
            argList += "SFTFILES=\"{0}\" ".format(sft_names)
            
            inputFiles = ', '.join([str(s) for s in sftFiles]) + ', ' + str(metric) 
            argList += "TRANSFERFILES=\"{0}\" ".format(inputFiles)
            
            for key, value in kwargs.items():
                argList += "{0}=\"{1}\" ".format(key.replace('-', '').upper(), value)        
            
            argList += "ALPHA=\"{0}\" DALPHA=\"{1}\" ".format(params['alpha'], self.target['dalpha'])
            argList += "DELTA=\"{0}\" DDELTA=\"{1}\" ".format(params['delta'], self.target['ddelta'])
            
            for key1, key2 in zip(self.freqParamName, self.freqDerivParamName): 
                argList += "{0}=\"{1}\" ".format(key1.upper(), params[key1])
                argList += "{0}=\"{1}\" ".format(key2.upper(), params[key2])
                
        return argList
        
    def weaveArgStr(self, nSeg): 
        """
        Generate the argument string template (VARS placeholders) for the Weave executable.

        Parameters:
            nSeg (int): Number of segments in the search.

        Returns:
            argListString (str): Argument string template using Condor $(VAR) syntax.
        """
        if nSeg != 1:
            argStr = ["output-file", 
                      "sft-files", "setup-file", 
                      "semi-max-mismatch", "coh-max-mismatch",
                      "toplist-limit", "extra-statistics"]
        else:
            argStr = ["output-file", 
                      "sft-files", "setup-file", 
                      "semi-max-mismatch", 
                      "toplist-limit", "extra-statistics"]
            
        argListString = ""
        for s in argStr:
            argListString += "--{0}=$({1}) ".format(s, s.replace('-', '').upper())
            
        argListString += "--alpha=$(ALPHA)/$(DALPHA) --delta=$(DELTA)/$(DDELTA) "
        for key1, key2 in zip(self.freqParamName, self.freqDerivParamName):
            argListString += "--{0}=$({1})/$({2}) ".format(key1, key1.upper(), key2.upper())
        
        return argListString
    
    def makeSearchDag(self, task_name, freq, param, numTopList, stage, freqDerivOrder, nSeg,
                      sftFiles, request_memory='18GB', request_disk='5GB', request_cpu=1, 
                      OSG=False, OSDF=False, metric='None',
                      exe=None, image=None):
        """
        Orchestrate the creation of the DAG file and the SUB file for a given frequency band.

        Parameters:
            task_name (str): Unique name for the task.
            freq (int): The current frequency being processed.
            param (numpy.ndarray): Array of parameter space chunks.
            numTopList (int): Number of top list entries to keep.
            stage (str): The pipeline stage.
            freqDerivOrder (int): Order of frequency derivative.
            nSeg (int): Number of segments.   
            sftFiles (list): List of input SFT files.
            request_memory (str): Memory request.
            request_disk (str): Disk request.
            request_cpu (int): CPU request.
            OSG (bool): Whether to use OSG.
            OSDF (bool): Whether to use OSDF.   
            metric (str): Path to the metric file.
            exe (Path, optional): Path to custom executable.
            image (Path, optional): Path to custom image.

        Returns:
            dagFileName (Path): Path to the written DAG file.
        """
        t0 = time.time()
        
        if OSDF and not OSG:
            print('Warning: You are reading SFTs from OSDF but not using OSG computing resources.')

        self.freqParamName, self.freqDerivParamName = phaseParamName(freqDerivOrder)
        self.numTopList = numTopList
            
        dagFileName = self.paths.dag_file(freq, task_name, stage)
        Path(dagFileName).parent.mkdir(parents=True, exist_ok=True)
        Path(dagFileName).unlink(missing_ok=True)
        
        crFiles = self.paths.condor_record_files(freq, task_name, stage)
        makeDir(crFiles)

        argStr = self.weaveArgStr(nSeg)
        subFileName = self.writeSub(freq, stage, task_name, crFiles, argStr, 
                                    request_memory=request_memory, request_disk=request_disk, request_cpu=request_cpu,
                                    OSG=OSG, OSDF=OSDF, exe=exe, image=image)
        
        for jobIndex, params in tqdm(enumerate(param, 1), total=param.size):
            argList = self.searchDagArgs(freq, stage, params, task_name, nSeg, sftFiles, jobIndex, OSG, metric)
            wc.writeSearchDag(str(dagFileName), task_name, str(subFileName), jobIndex, argList)

        print('Finish writing {0} dag files for {1} Hz'.format(stage, freq))
        print('Time used = {:.2f}s'.format(time.time()-t0))
        return dagFileName
    
# ##################################################################################################################################################################


#     def upperLimitArgStr(self, nSeg): 
#         """
#         Generate the argument string template for the Weave executable.
#         Parameters:
#             nSeg (int): Number of segments in the search.
#         Returns:
#             argListString (str): Argument string template.
#         """
#         if nSeg != 1:
#             argStr = ["output-file", 
#                       "sft-files", "setup-file", 
#                       "semi-max-mismatch", "coh-max-mismatch",
#                       "toplist-limit", "extra-statistics"]
#         else:
#             argStr = ["output-file", 
#                       "sft-files", "setup-file", 
#                       "semi-max-mismatch", 
#                       "toplist-limit", "extra-statistics"]
            
#         argListString = ""
#         for s in argStr:
#             argListString += "--{0}=$({1}) ".format(s, s.replace('-', '').upper())
            
#         argListString += "--alpha=$(ALPHA)/$(DALPHA) --delta=$(DELTA)/$(DDELTA) "
#         for key1, key2 in zip(self.freqParamName, self.freqDerivParamName):
#             argListString += "--{0}=$({1})/$({2}) ".format(key1, key1.upper(), key2.upper())
        


#         return argListString


#     def upperLimitArgs(self, freq, stage, mean2F_th, task_name, nSeg, sftFiles, jobIndex, metric, 
#                        sftFiles, nInj, skyUncertainty, h0est, num_cpus, cluster, workInLocalDir, OSDF, OSG):
#         """
#         Internal helper: Generates argument strings for Upper Limit jobs.
#         Parameters:
#             freq (int): Frequency of the search.
#             stage (str): Stage of the search.
#             mean2F_th (float): Mean 2F threshold.
#             task_name (str): Name of the task.
#             nSeg (int): Number of segments in the search.
#             sftFiles (list): List of SFT files used in the search.
#             jobIndex (int): Index of the job.
#             metric (str): Path to the metric file.
#             nInj (int): Number of injections.
#             skyUncertainty (float): Sky uncertainty value.
#             h0est (float): Estimated h0 value.
#             num_cpus (int): Number of CPUs requested.
#             cluster (bool): Whether running on a cluster.
#             workInLocalDir (bool): Whether working in local   
#             OSDF (bool): Whether to use OSDF SFTs.
#             OSG (bool): Whether to use OSG resources.  
#         Returns:
#             pyArgString/osgArgString (str): Argument string for the Upper Limit job.            
#         """
#         # 1. Base Python Arguments (used for Local execution or inside the OSG wrapper)
#         if workInLocalDir:
#             metric_str = Path(metric).name
#         else:
#             metric_str = metric

#         sft_str = ';'.join([Path(s).name for s in sftFiles])
        
#         pyArgString = f'--target {self.target["name"]} --cohDay {cohDay} --freq {freq} --stage {stage} \
#                         --freqDerivOrder {freqDerivOrder} --numTopList {numTopList} --nInj {nInj} --sftFiles {sft_str} \
#                         --num_cpus {num_cpus} --skyUncertainty {skyUncertainty} --h0est {h0est} --metric {metric_str}'
        
#         if cluster:
#             pyArgString += " --cluster"
#         if workInLocalDir:
#             pyArgString += " --workInLocalDir"
#         if OSDF:
#             pyArgString += " --OSDF"

#         # 2. Return based on execution mode
#         if not OSG:
#             # For local execution, the ARG list IS the python command line
#             return pyArgString
#         else:
#             curr_taskName = taskName(self.target['name'], stage, cohDay, freqDerivOrder, freq)
#             outputFile = self.paths.outlier_file(freq, curr_taskName, stage, cluster=cluster)
            
#             # Construct Transfer List
#             exe = self.paths.upper_limit_executable
#             image = self.paths.singularity_image
            
#             # Build comma-separated list of input files
#             # Note: OSG usually requires the full path for transfer
#             inputFiles = [str(exe), str(image), str(searchResultFile), str(metric)]
#             inputFiles.extend([str(s) for s in sftFiles])
#             inputFiles_str = ', '.join(inputFiles)

#             # OSG Argument String (VARS format)
            
#             osgArgString = "OUTPUTFILE=\"{0}\" REMAPOUTPUTFILE=\"{1}\" TRANSFERFILES=\"{2}\" ".format(
#                 Path(outputFile).name, outputFile, inputFiles_str)
    
#             osgArgString += " ARGUMENTS=\"{0}\"".format(pyArgString)
            
#             return osgArgString

#     def makeUpperLimitDag(self, task_name, freq, mean2F_th, numTopList, stage, freqDerivOrder, nSeg,
#                           sftFiles, skyUncertainty=1e-4, h0est=1e-25, nInj=100,
#                           request_memory='4GB', request_disk='4GB', request_cpu=4,
#                           cluster=False, workInLocalDir=False, 
#                           OSG=False, OSDF=False, metric='None',
#                           exe=None, image=None):

#         """
#         Make condor DAG file for Upper Limit stage.
#         """
#         t0 = time.time()
        
#         if OSDF and not OSG:
#             print('Warning: You are reading SFTs from OSDF but not using OSG computing resources.')

#         self.freqParamName, self.freqDerivParamName = phaseParamName(freqDerivOrder)
#         self.numTopList = numTopList
            
#         dagFileName = self.paths.dag_file(freq, task_name, stage)
#         Path(dagFileName).parent.mkdir(parents=True, exist_ok=True)
#         Path(dagFileName).unlink(missing_ok=True)
        
#         crFiles = self.paths.condor_record_files(freq, task_name, stage)
#         makeDir(crFiles)


#         argStr = self.weaveArgStr(nSeg)
#         subFileName = self.writeSub(freq, stage, task_name, crFiles, argStr, 
#                                     request_memory=request_memory, request_disk=request_disk, request_cpu=request_cpu,
#                                     OSG=OSG, OSDF=OSDF, exe=Path(exe).name, image=Path(image).name)
        
#         argList = self.upperLimitArgs(freq, stage, mean2F_th, task_name, nSeg, sftFiles, 1, metric=metric,
#                                       sftFiles=sftFiles, nInj=nInj, skyUncertainty=skyUncertainty, h0est=h0est, num_cpus=request_cpu,
#                                       cluster=cluster, workInLocalDir=workInLocalDir, OSG=OSG, OSDF=OSDF)
        
#         wc.writeSearchDag(str(dagFileName), task_name, str(subFileName), 1, argList)

#         print('Finish writing {0} dag files for {1} Hz'.format(stage, freq))
#         print('Time used = {:.2f}s'.format(time.time()-t0))
#         return dagFileName
