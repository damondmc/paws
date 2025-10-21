from astropy.io import fits
import numpy as np
from ..utils import filePath as fp
from ..utils import setup_parameter as setup
from ..utils import utils as utils
from pathlib import Path
import warnings
import subprocess
import time
from multiprocessing import Pool, cpu_count    
from itertools import islice


def delete_files(resultFileList):
    # delete files to release disk storage
    for f in resultFileList:
        command = 'rm {}'.format(f)
        _ = subprocess.run(command, shell=True, capture_output=True, text=True)
    print('Deleted {} weave result files to release disk storage.\n'.format(len(resultFileList)))
 

# Function to process each search job in parallel
def searchJob(params, sftFiles, metric, semiMM, cohMM, numTopList, extraStats, ra, dec, nc, nf, obsDay):
    
    # Weave main program path
    weave_exe = fp.weaveExecutableFilePath()
    
    resultFile, param = params
    utils.makeDir([resultFile])
    print(resultFile)
    if Path(resultFile).exists():
        print('Exists:{}'.format(resultFile))
        return resultFile
    
    command = '{} --output-file={} --sft-files=\"{}\" --setup-file={} --semi-max-mismatch={} --coh-max-mismatch={} --toplist-limit={} --extra-statistics={} --alpha={} --delta={}'.format(
        weave_exe, resultFile, sftFiles, metric, semiMM, cohMM, numTopList, extraStats, ra, dec)
    if nc == obsDay:
        command = '{} --output-file={} --sft-files=\"{}\" --setup-file={} --semi-max-mismatch={} --toplist-limit={} --extra-statistics={} --alpha={} --delta={}'.format(
            weave_exe, resultFile, sftFiles, metric, semiMM, numTopList, extraStats, ra, dec)

    newFreqParamName, newFreqDerivParamName = utils.phaseParamName(nf)
    for _f, _df in zip(newFreqParamName, newFreqDerivParamName):
        command += ' --{}={}/{}'.format(_f, param[_f], param[_df])

    # Run the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the standard output and errors for debugging
    #print(result.stdout)
    #print(result.stderr)
    return resultFile


# Function to process each injection job in parallel
def injectionJob(params, inj, sftFiles, metric, semiMM, cohMM, numTopList, extraStats, ra, dec, nc, nf, obsDay):

    # Weave main program path
    weave_exe = fp.weaveExecutableFilePath()
    
    resultFile, param = params
    utils.makeDir([resultFile])
    #print(resultFile)
    if Path(resultFile).exists():
        print('Exists:{}'.format(resultFile))
        return resultFile
    
    command = '{} --output-file={} --sft-files=\"{}\" --setup-file={} --semi-max-mismatch={} --coh-max-mismatch={} --toplist-limit={} --extra-statistics={} --alpha={} --delta={}'.format(
        weave_exe, resultFile, sftFiles, metric, semiMM, cohMM, numTopList, extraStats, ra, dec)
    if nc == obsDay:
        command = '{} --output-file={} --sft-files=\"{}\" --setup-file={} --semi-max-mismatch={} --toplist-limit={} --extra-statistics={} --alpha={} --delta={}'.format(
            weave_exe, resultFile, sftFiles, metric, semiMM, numTopList, extraStats, ra, dec)

    newFreqParamName, newFreqDerivParamName = utils.phaseParamName(nf)
    for _f, _df in zip(newFreqParamName, newFreqDerivParamName):
        command += ' --{}={}/{}'.format(_f, param[_f], param[_df])
           
    injection_command = ' --injections=\"{{Alpha={};Delta={};refTime={};aPlus={};aCross={};psi={};Freq={};f1dot={};f2dot={};f3dot={};f4dot={}}}\"'.format(
    inj['Alpha'], inj['Delta'], inj['refTime'], 
    inj['aPlus'], inj['aCross'], inj['psi'], 
    inj['Freq'], inj['f1dot'], inj['f2dot'], 
    inj['f3dot'], inj['f4dot'])
    
    command += injection_command
    print(command)
    # Run the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the standard output and errors for debugging
    #print(result.stdout)
    #print(result.stderr)
    return resultFile

def determineEfficiency(metric, sftFiles, setup, cohDay, obsDay, sp, inj_params, rm, target, taskName, freq, nInj, freqDerivOrder, stage, numTopList, extraStats, num_cpus, cluster, workInLocalDir, saveIntermediate=False, skyUncertainty=1e-5):
    
    #sp, ip = im._genParam(h0=h0est, freq=freq, nInj=nInj, injFreqDerivOrder=4, freqDerivOrder=freqDerivOrder, skyUncertainty=skyUncertainty, workInLocalDir=workInLocalDir, cluster=cluster)

    if workInLocalDir:
        search_params = [(Path(fp.weaveOutputFilePath(target, freq, taskName, jobIndex, stage)).name, params) for jobIndex, params in enumerate(sp[str(freq)].data, 1)]
    else:
        search_params = [(fp.weaveOutputFilePath(target, freq, taskName, jobIndex, stage), params) for jobIndex, params in enumerate(sp[str(freq)].data, 1)]
    #inj_params = ip[str(freq)].data

    #print("Generated params for h0={}, running Weave...".format(h0est)) 
    injResultFileList = [] 
    
    _, cohTime, nSeg, _, _ = utils.getTimeSetup(target.name, obsDay, cohDay)
    #metric = fp.weaveSetupFilePath(cohTime, nSeg, freqDerivOrder)
    #if workInLocalDir:  
    #    metric = Path(metric).name

    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(injectionJob, [(params, inj, sftFiles, metric, setup.semiMM, setup.cohMM, numTopList, extraStats, target.alpha, target.delta, cohDay, freqDerivOrder, obsDay) for params, inj in zip(search_params, inj_params)])
    # Collect the results
    injResultFileList.extend(results)

    taskName = utils.taskName(rm.target, 'search', cohDay, freqDerivOrder, int(freq))
    outlierFilePath = fp.outlierFilePath(rm.target, int(freq), taskName, 'search', cluster=cluster)
    if workInLocalDir:
        outlierFilePath = Path(outlierFilePath).name
    mean2F_th = fits.getheader(outlierFilePath)['HIERARCH mean2F_th']
    
    outlierFilePath = rm.writeInjectionResult(cohDay, freq, mean2F_th, nInj, numTopList=numTopList, stage=stage, freqDerivOrder=freqDerivOrder, cluster=cluster, workInLocalDir=workInLocalDir)

    if not saveIntermediate:
        # delete files to release disk storage
        for _f in injResultFileList:
            command = 'rm {}'.format(_f)
            _ = subprocess.run(command, shell=True, capture_output=True, text=True)
        print('Deleted weave result files to release disk storage.\n')

    nout = fits.getdata(outlierFilePath,1).size
    p = nout / nInj
    print('{}% ({}/{}) above mean2F threshold, saved to {}.'.format(round(p*100,2), nout, nInj, outlierFilePath))
    return p, outlierFilePath

def injectionFollowUp(fm, rm, target, obsDay, freq, sftFiles, 
                      old_cohDay, old_freqDerivOrder, old_stage, new_cohDay, 
                      new_freqDerivOrder, new_stage, nInj, numTopList, extraStats, num_cpus, setup, 
                      cluster, workInLocalDir, saveIntermediate=False):
    print('Doing injection follow-up...')    
    
    sp, ip = fm.genFollowUpParamFromInjection1Hz(old_cohDay, freq, stage=old_stage, oldFreqDerivOrder=old_freqDerivOrder, newFreqDerivOrder=new_freqDerivOrder, cluster=cluster, workInLocalDir=workInLocalDir)

    if sp[str(freq)].data.size == 0:
        print('0 outliers from the previous injection stage: Error!')
        exit()
    else:
        print('{} injections to be carried out.'.format(sp[str(freq)].data.size))

    injResultFileList = []
    taskName = utils.taskName(target, new_stage, new_cohDay, new_freqDerivOrder, freq)
    cohDay, cohTime, nSeg, _, _ = utils.getTimeSetup(target.name, obsDay, new_cohDay)
    metric = fp.weaveSetupFilePath(cohTime, nSeg, new_freqDerivOrder)
    if workInLocalDir:  
        metric = Path(metric).name

    # Prepare job parameters for parallel execution
    if workInLocalDir:
        search_params = [(Path(fp.weaveOutputFilePath(target, freq, taskName, jobIndex, new_stage)).name, params) for jobIndex, params in enumerate(sp[str(freq)].data, 1)]
    else:
        search_params = [(fp.weaveOutputFilePath(target, freq, taskName, jobIndex, new_stage), params) for jobIndex, params in enumerate(sp[str(freq)].data, 1)]

    inj_params = ip[str(freq)].data

    print("Generated params, running Weave...")

    # Use multiprocessing to process the jobs in parallel
    with Pool(processes=num_cpus) as pool:
        results = pool.starmap(injectionJob, [(params, inj, sftFiles, metric, setup.semiMM, setup.cohMM, 1000, extraStats, target.alpha, target.delta, new_cohDay, new_freqDerivOrder, obsDay) for params, inj in zip(search_params, inj_params)])
     
    # Collect the results
    injResultFileList.extend(results)

    print('Analyzing injection result.')
    # analyze the result
    
    # Get old mean2F for the follow-up outlier seleciton
    taskName = utils.taskName(rm.target, old_stage, old_cohDay, old_freqDerivOrder, freq)
    outlierFilePath = fp.outlierFilePath(rm.target, freq, taskName, old_stage, cluster=cluster)
    if workInLocalDir:
        outlierFilePath = Path(outlierFilePath).name
        
    data = fits.getdata(outlierFilePath, 1) 
    old_mean2F = data['mean2F']
    ratio = 0
    outlierFilePath = rm.writeFollowUpResult(old_mean2F, new_cohDay, freq, numTopList=numTopList, 
                                             new_stage=new_stage, new_freqDerivOrder=new_freqDerivOrder, 
                                             ratio=ratio, workInLocalDir=workInLocalDir, inj=True, cluster=cluster)
    
    return outlierFilePath, injResultFileList

# Helper function to split an iterable into chunks
def chunked_iterable(iterable, size):
    """Yield successive chunks from an iterable of given size."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def realFollowUp(rm, sp, target, obsDay, freq, sftFiles, 
                 old_mean2F, mean2F_ratio, 
                 new_cohDay, new_freqDerivOrder, new_stage, 
                 numTopList, extraStats, num_cpus, setup, 
                 cluster, workInLocalDir, saveIntermediate=False):
    print('Doing real follow-up...')
     
    #searchResultFileList = []
    taskName = utils.taskName(target, new_stage, new_cohDay, new_freqDerivOrder, freq)
    cohDay, cohTime, nSeg, _, _ = utils.getTimeSetup(target.name, obsDay, new_cohDay)
    metric = fp.weaveSetupFilePath(cohTime, nSeg, new_freqDerivOrder)
    if workInLocalDir:  
        metric = Path(metric).name

    # Prepare job parameters for parallel execution
    if workInLocalDir:
        search_params = [(Path(fp.weaveOutputFilePath(target, freq, taskName, jobIndex, new_stage)).name, params) for jobIndex, params in enumerate(sp[str(freq)].data, 1)]
    else:
        search_params = [(fp.weaveOutputFilePath(target, freq, taskName, jobIndex, new_stage), params) for jobIndex, params in enumerate(sp[str(freq)].data, 1)]
       
    # divide it into chunks to not to overload the disk storage
    chunk_size = 100  # Define chunk size
    # Calculate the number of chunks before the for loop
    totalJobCounts = len(search_params)
    chunk_count = int(np.ceil(totalJobCounts / chunk_size))

    print("Generated params, running Weave...")
    # Use multiprocessing to process each chunk of jobs in parallel
    with Pool(processes=num_cpus) as pool:
        for chunk_index, chunk in enumerate(chunked_iterable(search_params, chunk_size)):
            print(f"Processing chunk {chunk_index+1} out of {chunk_count}...")
            results = pool.starmap(searchJob, [
                (params, sftFiles, metric, setup.semiMM, setup.cohMM, 1000, extraStats, 
                 target.alpha, target.delta, new_cohDay, new_freqDerivOrder, obsDay)
                for params in chunk
            ])
            
            if chunk_count != 1:
                # Analyze the results immediately after each chunk
                outlierFilePath = rm.writeFollowUpResult(
                    old_mean2F, new_cohDay, freq, numTopList=numTopList, 
                    new_stage=new_stage, new_freqDerivOrder=new_freqDerivOrder, ratio=mean2F_ratio, 
                    workInLocalDir=workInLocalDir, inj=False, cluster=cluster,
                    chunk_count=chunk_count, chunk_index=chunk_index, chunk_size=chunk_size
                )
            else:
                # Analyze the results for the only chunk
                outlierFilePath = rm.writeFollowUpResult(
                    old_mean2F, new_cohDay, freq, numTopList=numTopList, 
                    new_stage=new_stage, new_freqDerivOrder=new_freqDerivOrder, ratio=mean2F_ratio, 
                    workInLocalDir=workInLocalDir, inj=False, cluster=cluster
                )
   
            # Delete the files to free up disk storage
            if not saveIntermediate:
                delete_files(results)
        if chunk_count != 1:
            outlierFilePath = rm.ensembleOutlierChunk(totalJobCounts, chunk_size, chunk_count, new_cohDay, freq, new_stage, new_freqDerivOrder, cluster, workInLocalDir)
    return outlierFilePath

def determineMean2FRatio(percentile, target, freq, 
                         old_cohDay, old_freqDerivOrder, old_stage, 
                         new_cohDay, new_freqDerivOrder, new_stage, 
                         cluster=False, workInLocalDir=False):
    taskName = utils.taskName(target=target, stage=old_stage, cohDay=old_cohDay, order=old_freqDerivOrder, freq=freq)
    filePath = fp.outlierFilePath(target, freq, taskName, old_stage, cluster=cluster)
    if workInLocalDir:
        filePath = Path(filePath).name
    olddata = fits.getdata(filePath, 1)

    taskName = utils.taskName(target=target, stage=new_stage, cohDay=new_cohDay, order=new_freqDerivOrder, freq=freq)
    filePath = fp.outlierFilePath(target, freq, taskName, new_stage, cluster=cluster)
    if workInLocalDir:
        filePath = Path(filePath).name
    
    newdata = fits.getdata(filePath, 1)
    try:
        ratio = np.sort(newdata['mean2F']/olddata['mean2F'])
        r = np.percentile(ratio, percentile) ## 99.5-99.6 percentile
        r = int(r*100.)/100.
        print('Ratio = {} for {}% percentile for {} injections.\n'.format(r, (1-percentile)*100, ratio.size))
    except:
        print('Size not match: Before {}, after {}.\n'.format(olddata.size, newdata.size))
        r=0    
    return r 

