from pathlib import Path
from astropy.io import fits
from .definitions import phaseParamName 
import re
import logging

# Matches integers, floats, and scientific notation (e.g., 1.23e-4)
NUMERIC_PATTERN = re.compile(r'-?\s*\d+\.?\d*(?:[Ee]\s*-?\s*\d+)?')

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

# =============================================================================
# Log Parsing Helpers
# =============================================================================

def _parse_log_value(filePath, trigger_phrase, extract_logic_func):
    """
    Generic helper to parse a specific value from a log file.
    
    Args:
        filePath (str): Path to log file.
        trigger_phrase (str): The string to look for in the line.
        extract_logic_func (callable): Function that takes the line string and returns the extracted value.
    """
    try:
        with open(filePath, 'r') as f:
            for line in f:
                if trigger_phrase in line:
                    return extract_logic_func(line)
    except FileNotFoundError:
        logging.error(f"Log file not found: {filePath}")
    except Exception as e:
        logging.error(f"Error parsing {filePath}: {e}")
    return None

def _extract_number(text):
    """Extracts the first number found in text."""
    matches = NUMERIC_PATTERN.findall(text)
    return float(matches[0]) if matches else None

def readMemoryUsage(filePath):
    """Parses memory usage (GB) from log file."""
    def logic(line):
        parts = line.split(',')
        if len(parts) > 3:
            return _extract_number(parts[3])
        return None
        
    return _parse_log_value(filePath, 'completion-loop', logic)

def readTemplateCount(filePath):
    """Parses number of templates from log file."""
    def logic(line):
        return int(_extract_number(line.split('=')[1]))
        
    return _parse_log_value(filePath, 'Number of semicoherent templates', logic)

def readRunTime(filePath):
    """Parses runtime (seconds) from log file."""
    def logic(line):
        parts = line.split(',')
        if len(parts) > 1:
            return _extract_number(parts[1])
        return None

    return _parse_log_value(filePath, 'completion-loop', logic)

def readEstimatedUpperStrainLimit(filePath):
    """Parses estimated upper limit h0."""
    try:
        with open(filePath, 'r') as f:
            # We need the last few lines, reading whole file might be unavoidable 
            # unless we use seek, but files are usually small enough.
            lines = f.readlines()
            
        if not lines:
            return None

        # Check if finished
        if 'DONE' in lines[-1]:
            target_line = lines[-2]
        else:
            logging.warning(f"Program not finished in {filePath}")
            target_line = lines[-1]
            
        if '=' in target_line:
            return _extract_number(target_line.split('=')[2])
            
    except Exception as e:
        logging.error(f"Error reading upper limit from {filePath}: {e}")
    return None

def isMismatchExist(filePath):
    """Checks if 'Size not match' error exists in the log."""
    try:
        with open(filePath, 'r') as f:
            for line in f:
                if 'Size not match' in line:
                    return True
        return False
    except FileNotFoundError:
        return False