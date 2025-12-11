from pathlib import Path
from astropy.io import fits
from .definitions import phaseParamName 

def makeDir(filenames):
    for name in filenames:
        Path(name).resolve().parent.mkdir(parents=True, exist_ok=True)

def getBinTable(task_name, freq, stage, extname, paths, cluster=False, workInLocalDir=False):
    """
    Reads data from a FITS file and returns it as a numpy array.
    
    Parameters:
        - task_name (str): Name of the task.
        - freq (int): Frequency parameter.
        - stage (str): Stage parameter.
        - extname (str): Extension name in the FITS file.
        - paths (object): Paths object containing the file path.
        - cluster (bool): Boolean indicating if the data is from a cluster.
        - workInLocalDir (bool): Boolean indicating if working in local directory. 
    Returns:
        - Numpy array containing the data from the specified extension in the FITS file.
    """
    dataFilePath = paths.outlier_file(freq, task_name, stage, cluster=cluster)
    
    if workInLocalDir:
        dataFilePath = dataFilePath.name 
        
    return fits.getdata(dataFilePath, extname=extname)

def getHeader(task_name, freq, stage, paths, cluster=False, workInLocalDir=False):
    """
    Reads header information from a FITS file.

    Parameters:
        - task_name (str): Name of the task.
        - freq (int): Frequency parameter.
        - stage (str): Stage parameter.
        - paths (object): Paths object containing the file path.
        - cluster (bool): Boolean indicating if the data is from a cluster.
        - workInLocalDir (bool): Boolean indicating if working in local directory.
    Returns:
        - Header information from the FITS file.
    """

    dataFilePath = paths.outlier_file(freq, task_name, stage, cluster=cluster)
    
    if workInLocalDir:
        dataFilePath = dataFilePath.name
        
    return fits.getheader(dataFilePath)

def readOutlierData(task_name, freq, stage, paths, cluster=False):
    """
    Reads outlier data from a FITS file.

    Parameters:
        - task_name (str): Name of the task.
        - freq (int): Frequency parameter.
        - stage (str): Stage parameter.
        - paths (object): Paths object containing the file path.
        - cluster (bool): Boolean indicating if the data is from a cluster. 
    Returns:
        - Data from the FITS file.  
    """

    outlierFilePath = paths.outlier_file(freq, task_name, stage, cluster)            
    
    with fits.open(outlierFilePath) as hdul:
        data = hdul[1].data
    return data

def getSpacing(dataFilePath, freqDerivOrder):
    """
    Read spacing information from FITS header.
    Parameters:
        - dataFilePath (str): Path to the FITS file.
        - freqDerivOrder (int): Frequency derivative order.
    Returns:
        - Dictionary containing the spacing information.
    """
    metaData = fits.getheader(dataFilePath)

    freqParamName, freqBandWidthName = phaseParamName(freqDerivOrder)
    n = len(freqParamName)

    nTemp_cum = [metaData[f'NSEMITMPL NU{i}DOT'] for i in range(n)]
    nTemp = []
    
    # Calculate templates per dimension
    if n > 0:
        nTemp.append(int(nTemp_cum[0]/nTemp_cum[-1])) # f0
        if n > 1:
            nTemp.append(nTemp_cum[1]) # f1
            for i in range(2, len(freqParamName)):
                nTemp.append(int(nTemp_cum[i]/nTemp_cum[i-1]))

    df = {}
    for i in range(n):
        key = freqParamName[i].upper()
        # Header keys usually have PROGARG prefix
        arg_str = metaData.get(f'PROGARG {key}', None)
        if arg_str:
            a, b = arg_str.split(',')
            bandwidth = float(b) - float(a)
            df[freqBandWidthName[i]] = bandwidth/nTemp[i]
            
    return df