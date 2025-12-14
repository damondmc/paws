import numpy as np
from pathlib import Path
from tqdm import tqdm

from . import writer as wc
from paws.io import make_dir
from paws.filepaths import PathManager

class followupManager:
    def __init__(self, target, config):
        self.target = target
        self.config = config
    
    def followUpArgs(self, h0, cohDay, freq, stage, freqDerivOrder, numTopList, sftFiles, request_cpu, real, inj, cluster, workInLocalDir): 
        argListString = '--target {0} --obsDay {1} --cohDay {2} --freq {3} --stage {4} --freqDerivOrder {5} --numTopList {6} --sftFiles {7} --num_cpus {8} --h0 {9}'.format(
                self.target.name, self.obsDay, cohDay, freq, stage, freqDerivOrder, numTopList, ';'.join([Path(s).name for s in sftFiles]), request_cpu, h0)
        if real:
            argListString += " --real"
        if inj:
            argListString += " --inj"
        if cluster:
            argListString += " --cluster" 
        if workInLocalDir:
            argListString += " --workInLocalDir"
        
        return argListString

    def transferFileArgs(self, exe, configFile, cohDay, freq, freqDerivOrder, stage, sftFiles, old_stage='search', cluster=False, OSG=True, OSDF=False, fromSaturatedBand=False):       
    
        taskname = task_name(self.target, old_stage, cohDay, freqDerivOrder, freq)
        if not fromSaturatedBand:
            searchResultFile = fp.outlierFilePath(self.target, freq, taskname, old_stage, cluster=cluster) 
        else:
            searchResultFile = fp.outlierFromSaturatedFilePath(self.target, freq, taskname, old_stage)
        #exe = fp.followUpExecutableFilePath()
        image = fp.imageFilePath(OSDF)
        inputFiles = "{}, {}, {}".format(exe, image, searchResultFile)
        for sft in sftFiles:
            inputFiles += ", {}".format(sft)
        _, cohTime, nSeg, _, _ = utils.getTimeSetup(self.target.name, self.obsDay, cohDay)
        metric = fp.weaveSetupFilePath(cohTime, nSeg, freqDerivOrder)
        inputFiles += ", {}".format(metric)
        # add initial stage metric
    
        # add follow-up stage metric
        cohDayList, freqDerivOrderList = np.loadtxt(configFile, skiprows=2, dtype=('i4', 'i4')).T 
        for _cohDay, _freqDerivOrder in zip(cohDayList, freqDerivOrderList):
            _, cohTime, nSeg, _, _ = utils.getTimeSetup(self.target.name, self.obsDay, _cohDay)
            metric = fp.weaveSetupFilePath(cohTime, nSeg, _freqDerivOrder)
            inputFiles += ", {}".format(metric)
        # using OSG computing resources (different format for .sub file)
        taskname = task_name(self.target, stage, cohDay, freqDerivOrder, freq)
        outlierFilePath = fp.outlierFilePath(self.target, freq, taskname, stage, cluster=cluster) 
        make_dir([outlierFilePath])
        argList="OUTPUTFILE=\"{0}\" REMAPOUTPUTFILE=\"{1}\" TRANSFERFILES=\"{2}\" ".format(
            Path(outlierFilePath).name, outlierFilePath, inputFiles)
        return argList
   
    def makeFollowUpDag(self, configFile, fmin, fmax, h0, stage='followUp', numTopList=1000, request_cpu=4, request_disk='4GB', old_stage='search', real=True, inj=False, cluster=False, workInLocalDir=False, OSG=False, OSDF=False, fromSaturatedBand=False):
              
        cohDay, freqDerivOrder = np.loadtxt(configFile)[0].T
        if cohDay.is_integer(): 
            cohDay = int(cohDay)
        freqDerivOrder = int(freqDerivOrder)
        
        taskname = task_name(self.target, stage, cohDay, freqDerivOrder, str(fmin)+'-'+str(fmax)) 
        dagFileName = fp.dagFilePath('', self.target, taskname, stage)
        Path(dagFileName).unlink(missing_ok=True)
        
        for jobIndex, freq in tqdm(enumerate(range(fmin, fmax), 1)):
            taskname = task_name(self.target, stage, cohDay, freqDerivOrder, freq)
            exe = fp.followUpExecutableFilePath()
            local_exe = Path(exe).name
            subFileName = fp.condorSubFilePath(self.target, freq, taskname, stage)
            Path(subFileName).unlink(missing_ok=True)
            
            crFiles = fp.condorRecordFilePath(freq, self.target, taskname, stage)
            make_dir(crFiles)
            
            image = fp.imageFilePath(OSDF)
            image = Path(image).name
            sftFiles = utils.sftEnsemble(freq, self.obsDay, OSDF=OSDF)
            
            argList = self.followUpArgs(h0, cohDay, freq, stage, freqDerivOrder, numTopList, sftFiles, request_cpu, real, inj, cluster, workInLocalDir)
            wc.writeSearchSub(subFileName, local_exe, True, crFiles[0], crFiles[1], crFiles[2], argList, request_memory='4GB', request_disk=request_disk, request_cpu=request_cpu, OSG=OSG, OSDF=OSDF, image=image)
            
           # call function to write .sub files for analyze result
            ######################## Argument string use to write to DAG  ########################        
            argList = self.transferFileArgs(exe, configFile, cohDay, freq, freqDerivOrder, stage, sftFiles, old_stage, cluster, OSG, OSDF, fromSaturatedBand)
                             
            # Call function from WriteCondorFiles.py which will write DAG 
            wc.writeSearchDag(dagFileName, taskname, subFileName, jobIndex, argList)
        print('Finish writing follow-up dag from {0} stage for {1}-{2}Hz'.format(stage, fmin, fmax))

