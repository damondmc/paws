from tqdm import tqdm
from pathlib import Path

from .writer import write_search_dagfile, write_search_subfile
from paws.definitions import phase_param_name, task_name
from paws.io import make_dir 

class UpperLimitManager:
    def __init__(self, target, config):
        self.target = target
        self.config = config
    
    def upperLimitArgs(self, metric, cohDay, freq, stage, freqDerivOrder,  numTopList, nInj, skyUncertainty, h0est, num_cpus, sftFiles, cluster, workInLocalDir, OSDF): 
        if workInLocalDir:
            metric = Path(metric).name

        sftFiles = ';'.join([Path(s).name for s in sftFiles])
        #est_sftFiles = ';'.join([Path(s).name for s in est_sftFiles])
        argListString = '--target {0} --obsDay {1} --cohDay {2} --freq {3} --stage {4} --freqDerivOrder {5} --numTopList {6} --nInj {7} --sftFiles {8} --num_cpus {9} --skyUncertainty {10} --h0est {11} --metric {12}'.format(
                self.target.name, self.obsDay, cohDay, freq, stage, freqDerivOrder, numTopList, nInj, sftFiles, num_cpus, skyUncertainty, h0est, metric)
        if cluster:
            argListString += " --cluster"
        
        if workInLocalDir:
            argListString += " --workInLocalDir"
        
        if OSDF:
            argListString += " --OSDF"
        return argListString

    def transferFileArgs(self, cohDay, freq, freqDerivOrder, metric, stage, sftFiles, cluster=False, OSG=True):
    
        taskname = task_name(self.target, 'search', cohDay, freqDerivOrder, freq)
        searchResultFile = fp.outlierFilePath(self.target, freq, taskname, 'search', cluster=cluster) 
        
        exe = fp.upperLimitExecutableFilePath()
        image = fp.imageFilePath()
        inputFiles = "{}, {}, {}".format(exe, image, searchResultFile)
        for sft in sftFiles:
            inputFiles += ", {}".format(sft)
        #for sft in est_sftFiles:
        #    inputFiles += ", {}".format(sft)

        _, cohTime, nSeg, _, _ = utils.getTimeSetup(self.target.name, self.obsDay, cohDay)
        #metric = fp.weaveSetupFilePath(cohTime, nSeg, freqDerivOrder)
        inputFiles += ", {}".format(metric)
        
        # using OSG computing resources (different format for .sub file)
        taskname = task_name(self.target, stage, cohDay, freqDerivOrder, freq)
        outlierFilePath = fp.outlierFilePath(self.target, freq, taskname, stage, cluster=cluster) 
        make_dir([outlierFilePath])
        argList="OUTPUTFILE=\"{0}\" REMAPOUTPUTFILE=\"{1}\" TRANSFERFILES=\"{2}\" ".format(
            Path(outlierFilePath).name, outlierFilePath, inputFiles)
        return argList
    
    def makeUpperLimitDag(self, fmin, fmax, cohDay, freqDerivOrder=2, metric='', stage='upperLimit', skyUncertainty=1e-4, h0est=[1e-25], nInj=100, numTopList=1000, num_cpus=4, request_memory='4GB', request_disk='4GB', cluster=False, workInLocalDir=False, OSG=False, OSDF=False):
              
        taskname = task_name(self.target, stage, cohDay, freqDerivOrder, str(fmin)+'-'+str(fmax)) 
        dagFileName = fp.dagFilePath('', self.target, taskname, stage)
        Path(dagFileName).unlink(missing_ok=True)
        
        for jobIndex, freq in tqdm(enumerate(range(fmin, fmax), 1), total=fmax-fmin):
            taskname = task_name(self.target, stage, cohDay, freqDerivOrder, freq)
            exe = fp.upperLimitExecutableFilePath()
            exe = Path(exe).name
            subFileName = fp.condorSubFilePath(self.target, freq, taskname, stage)
            Path(subFileName).unlink(missing_ok=True)
            
            crFiles = fp.condorRecordFilePath(freq, self.target, taskname, stage)
            make_dir(crFiles)
            
            image = fp.imageFilePath()
            image = Path(image).name
            sftFiles = utils.sftEnsemble(freq, self.obsDay, OSDF=OSDF)
            #est_sftFiles = utils.sftEnsemble(freq, cohDay, OSDF=OSDF)
            argList = self.upperLimitArgs(metric, cohDay, freq, stage, freqDerivOrder, numTopList, nInj, skyUncertainty, h0est[jobIndex-1], num_cpus, sftFiles, cluster, workInLocalDir, OSDF)
            write_search_subfile(subFileName, exe, True, crFiles[0], crFiles[1], crFiles[2], argList, request_memory=request_memory, request_disk=request_disk, OSG=OSG, OSDF=OSDF, image=image)
            
           # call function to write .sub files for analyze result
            ######################## Argument string use to write to DAG  ########################        
            argList = self.transferFileArgs(cohDay, freq, freqDerivOrder, metric, stage, sftFiles, cluster, OSG)
                             
            # Call function from WriteCondorFiles.py which will write DAG 
            write_search_dagfile(dagFileName, taskname, subFileName, jobIndex, argList)
        print('Finish writing upper limit dag from {0} stage for {1}-{2}Hz'.format(stage, fmin, fmax))

