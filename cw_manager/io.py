from pathlib import Path
from astropy.io import fits
from .definitions import phaseParamName 
import re
import logging

# Matches integers, floats, and scientific notation (e.g., 1.23e-4)
NUMERIC_PATTERN = re.compile(r'-?\s*\d+\.?\d*(?:[Ee]\s*-?\s*\d+)?')

def make_dir(filenames):
    for name in filenames:
        Path(name).resolve().parent.mkdir(parents=True, exist_ok=True)

def get_bin_table(task_name, freq, stage, extname, paths, cluster=False, work_in_local_dir=False):
    """
    Reads data from a FITS file and returns it as a numpy array.
    
    Parameters:
        - task_name (str): Name of the task.
        - freq (int): Frequency parameter.
        - stage (str): Stage parameter.
        - extname (str): Extension name in the FITS file.
        - paths (object): Paths object containing the file path.
        - cluster (bool): Boolean indicating if the data is from a cluster.
        - work_in_local_dir (bool): Boolean indicating if working in local directory. 
    Returns:
        - Numpy array containing the data from the specified extension in the FITS file.
    """
    data_file_path = paths.outlier_file(freq, task_name, stage, cluster=cluster)
    
    if work_in_local_dir:
        data_file_path = data_file_path.name 
        
    return fits.getdata(data_file_path, extname=extname)

def get_header(task_name, freq, stage, paths, cluster=False, work_in_local_dir=False):
    """
    Reads header information from a FITS file.

    Parameters:
        - task_name (str): Name of the task.
        - freq (int): Frequency parameter.
        - stage (str): Stage parameter.
        - paths (object): Paths object containing the file path.
        - cluster (bool): Boolean indicating if the data is from a cluster.
        - work_in_local_dir (bool): Boolean indicating if working in local directory.
    Returns:
        - Header information from the FITS file.
    """
    data_file_path = paths.outlier_file(freq, task_name, stage, cluster=cluster)
    
    if work_in_local_dir:
        data_file_path = data_file_path.name
        
    return fits.getheader(data_file_path)

def read_outlier_data(task_name, freq, stage, paths, cluster=False):
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
    outlier_file_path = paths.outlier_file(freq, task_name, stage, cluster)            
    
    with fits.open(outlier_file_path) as hdul:
        data = hdul[1].data
    return data

def get_spacing(data_file_path, freq_deriv_order):
    """
    Read spacing information from FITS header.
    Parameters:
        - data_file_path (str): Path to the FITS file.
        - freq_deriv_order (int): Frequency derivative order.
    Returns:
        - Dictionary containing the spacing information.
    """
    meta_data = fits.getheader(data_file_path)

    freq_param_name, freq_bandwidth_name = phaseParamName(freq_deriv_order)
    n = len(freq_param_name)

    n_temp_cum = [meta_data[f'NSEMITMPL NU{i}DOT'] for i in range(n)]
    n_temp = []
    
    # Calculate templates per dimension
    if n > 0:
        n_temp.append(int(n_temp_cum[0] / n_temp_cum[-1])) # f0
        if n > 1:
            n_temp.append(n_temp_cum[1]) # f1
            for i in range(2, len(freq_param_name)):
                n_temp.append(int(n_temp_cum[i] / n_temp_cum[i-1]))

    df = {}
    for i in range(n):
        key = freq_param_name[i].upper()
        # Header keys usually have PROGARG prefix
        arg_str = meta_data.get(f'PROGARG {key}', None)
        if arg_str:
            a, b = arg_str.split(',')
            bandwidth = float(b) - float(a)
            df[freq_bandwidth_name[i]] = bandwidth / n_temp[i]
            
    return df

# =============================================================================
# Log Parsing Helpers
# =============================================================================

def _parse_log_value(file_path, trigger_phrase, extract_logic_func):
    """
    Generic helper to parse a specific value from a log file.
    
    Args:
        file_path (str): Path to log file.
        trigger_phrase (str): The string to look for in the line.
        extract_logic_func (callable): Function that takes the line string and returns the extracted value.
    """
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if trigger_phrase in line:
                    return extract_logic_func(line)
    except FileNotFoundError:
        logging.error(f"Log file not found: {file_path}")
    except Exception as e:
        logging.error(f"Error parsing {file_path}: {e}")
    return None

def _extract_number(text):
    """Extracts the first number found in text."""
    matches = NUMERIC_PATTERN.findall(text)
    return float(matches[0]) if matches else None

def read_memory_usage(file_path):
    """Parses memory usage (GB) from log file."""
    def logic(line):
        parts = line.split(',')
        if len(parts) > 3:
            return _extract_number(parts[3])
        return None
        
    return _parse_log_value(file_path, 'completion-loop', logic)

def read_template_count(file_path):
    """Parses number of templates from log file."""
    def logic(line):
        return int(_extract_number(line.split('=')[1]))
        
    return _parse_log_value(file_path, 'Number of semicoherent templates', logic)

def read_run_time(file_path):
    """Parses runtime (seconds) from log file."""
    def logic(line):
        parts = line.split(',')
        if len(parts) > 1:
            return _extract_number(parts[1])
        return None

    return _parse_log_value(file_path, 'completion-loop', logic)

def read_estimated_upper_strain_limit(file_path):
    """Parses estimated upper limit h0."""
    try:
        with open(file_path, 'r') as f:
            # We need the last few lines, reading whole file might be unavoidable 
            # unless we use seek, but files are usually small enough.
            lines = f.readlines()
            
        if not lines:
            return None

        # Check if finished
        if 'DONE' in lines[-1]:
            target_line = lines[-2]
        else:
            logging.warning(f"Program not finished in {file_path}")
            target_line = lines[-1]
            
        if '=' in target_line:
            return _extract_number(target_line.split('=')[2])
            
    except Exception as e:
        logging.error(f"Error reading upper limit from {file_path}: {e}")
    return None

def is_mismatch_exist(file_path):
    """Checks if 'Size not match' error exists in the log."""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'Size not match' in line:
                    return True
        return False
    except FileNotFoundError:
        return False