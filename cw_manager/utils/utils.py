import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.table import Table

# =============================================================================
# SFT & Directory Management
# =============================================================================

def sftEnsemble(freq, paths, use_osdf=False):
    """
    Gather SFT files for H1 and L1 for a specific frequency band.

    Args:
        freq (int): The frequency band (e.g., 10, 15).
        paths (PathManager): Instance of PathManager containing path logic.
        use_osdf (bool): Whether to format paths for OSDF (osdf://).

    Returns:
        list: List of file path strings.
    """
    # Use PathManager to get the directory for this freq
    h1_dir = paths.sft_file_path(freq, detector='H1', use_osdf=use_osdf)
    l1_dir = paths.sft_file_path(freq, detector='L1', use_osdf=use_osdf)

    # Find all .sft files
    h1_files = list(h1_dir.glob("*.sft"))
    l1_files = list(l1_dir.glob("*.sft"))

    file_list = []

    if use_osdf:
        # If using OSG, we might need the 'osdf://' protocol string
        # PathManager returns a Path object like: /osdf/igwn/...
        # We need to convert string '/osdf/...' to 'osdf://...'
        for f in h1_files + l1_files:
            s_path = str(f)
            if s_path.startswith('/osdf'):
                # Slice off the first slash to avoid 'osdf:///osdf' if needed, 
                # or just prepend if your mount point expects it.
                # Standard OSDF plugin usually expects: osdf:///igwn/...
                # If s_path is /osdf/igwn/..., we want osdf:///igwn/...
                # So we replace the first /osdf with osdf://
                file_list.append('osdf://' + s_path[5:]) 
            else:
                # Fallback if path doesn't start with /osdf
                file_list.append(s_path)
    else:
        # Standard local paths
        file_list = [str(f) for f in h1_files + l1_files]

    return file_list

def makeDir(filenames):
    """
    Ensure the parent directory for a list of files exists.
    """
    for name in filenames:
        Path(name).resolve().parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Naming Conventions
# =============================================================================

def taskName(target_name, stage, cohDay, order, freq):
    """
    Standardized task naming convention.
    """
    return f'{target_name}_{stage}_TCoh{cohDay}_O{order}_{freq}Hz'

def phaseParamName(order):
    """
    Returns parameter names based on derivative order.
    """
    freqParamName = ["freq", "f1dot", "f2dot", "f3dot", "f4dot"]
    freqDerivParamName = ["df", "df1dot", "df2dot", "df3dot", "df4dot"]        
    return freqParamName[:order+1], freqDerivParamName[:order+1]

def injParamName():
    return ["Alpha", "Delta", "refTime", "aPlus", "aCross", "psi", "Freq"]

# =============================================================================
# Data Analysis & Clustering
# =============================================================================

def clustering(data, freqDerivOrder, cluster_nSpacing=4.0):
    """
    Clusters outliers based on spatial proximity in phase parameter space.
    
    Args:
        data (astropy.table.Table): The outlier data.
        freqDerivOrder (int): Order of f-derivatives.
        cluster_nSpacing (float): Clustering threshold (grid units).
    """
    # Extract phase parameter names
    fn, dfn = phaseParamName(freqDerivOrder)
    
    # Create arrays for coordinates and spacings
    _data = np.column_stack([data[key] for key in fn])
    _spacing = np.column_stack([data[key] for key in dfn])

    # Retrieve loudness
    loudness = data['mean2F']

    # Sort descending by loudness
    sorted_indices = np.argsort(-loudness)
    sorted_coords = _data[sorted_indices]
    sorted_spacing = _spacing[sorted_indices]

    centers_idx = []
    cluster_size = []
    cluster_member = []
    processed_indices = set()

    # Loop over sorted samples
    for i, (center, gridsize) in enumerate(zip(sorted_coords, sorted_spacing)):
        if sorted_indices[i] in processed_indices:
            continue

        within_dim_indices = []

        # Check distance in every dimension
        for dim in range(freqDerivOrder+1):
            r0 = cluster_nSpacing * gridsize[dim]
            distances_dim = np.abs(_data[:, dim] - center[dim])
            within_dim = np.where(distances_dim <= r0)[0]
            within_dim_indices.append(within_dim)

        # Intersection of all dimensions
        within_r0_indices = within_dim_indices[0]
        for dim_indices in within_dim_indices[1:]:
            within_r0_indices = np.intersect1d(within_r0_indices, dim_indices)

        processed_indices.update(within_r0_indices)
        centers_idx.append(sorted_indices[i])
        cluster_size.append(len(within_r0_indices))
        cluster_member.append(within_r0_indices)

    centers_idx = np.array(centers_idx)
    cluster_size = np.array(cluster_size)

    print(f'{len(data)} outliers are grouped to {len(centers_idx)} clusters.')
    return centers_idx, cluster_size, cluster_member

def getBinTable(target, freq, cohDay, freqDerivOrder, stage, extname, paths, cluster=False, workInLocalDir=False):
    """
    Read FITS binary table data.
    Args:
        paths (PathManager): The paths object.
    """
    _taskName = taskName(target['name'], stage, cohDay, freqDerivOrder, freq)
    dataFilePath = paths.outlier_file(freq, _taskName, stage, cluster=cluster)
    
    if workInLocalDir:
        dataFilePath = dataFilePath.name # Just the filename
        
    data = fits.getdata(dataFilePath, extname=extname)
    return data

def getHeader(target, freq, cohDay, freqDerivOrder, stage, paths, cluster=False, workInLocalDir=False):
    """
    Read FITS header.
    Args:
        paths (PathManager): The paths object.
    """
    _taskName = taskName(target['name'], stage, cohDay, freqDerivOrder, freq)
    dataFilePath = paths.outlier_file(freq, _taskName, stage, cluster=cluster)
    
    if workInLocalDir:
        dataFilePath = dataFilePath.name
        
    data = fits.getheader(dataFilePath)
    return data

def readOutlierData(target, freq, cohDay, freqDerivOrder, stage, paths, cluster=False):
    """
    Wrapper to read specific outlier data extension.
    """
    _taskName = taskName(target['name'], stage, cohDay, freqDerivOrder, freq)
    outlierFilePath = paths.outlier_file(freq, _taskName, stage, cluster)            
    
    with fits.open(outlierFilePath) as hdul:
        data = hdul[1].data
        
    return data

def getSpacing(dataFilePath, freqDerivOrder):
    """
    Read spacing information from FITS header.
    """
    with fits.open(dataFilePath) as file:
        metaData = file[0].header

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

# def getTimeSetup(obsDay, cohDay, startTime, secondsInDay=86400):
#     """
#     Calculate observation time segments.
#     Args:
#         obsDay (float): Total observation days.
#         cohDay (float): Coherence time in days.
#         startTime (int): GPS start time (e.g., from config).
#     """
#     cohTime = int(cohDay * secondsInDay)
#     nSeg = int(obsDay / cohDay)
#     obsTime = cohTime * nSeg
#     refTime = int(startTime + obsTime/2)
#     return cohDay, cohTime, nSeg, obsTime, refTime
