import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.stats.distributions import chi2

def detection_stat_threshold(nTemp, nSeg):
    """
    Calculates the detection statistic threshold using Chi-Squared distribution.
    """
    # Inverse survival function (isf) is more precise than ppf(1-x) for small tails
    # But sticking to your logic:
    thresh = chi2.ppf(1.0 - (1.0 / nTemp), 4 * nSeg) / nSeg
    return thresh 

def appendFitsTable(t1, t2):
    """
    Vertically stacks two Astropy tables.
    """
    return vstack([t1, t2])

def appendFitsTableInFile(hdul):
    """
    Combines data from multiple HDUs in an HDUList into a single HDU.
    Assumes all HDUs (from index 1 onwards) have the same columns.
    """
    # Calculate total rows
    nrows = [table.data.shape[0] for table in hdul[1:]]
    total_rows = sum(nrows)
    
    # Create new HDU with combined shape
    new_hdul = fits.HDUList()
    hdu = fits.BinTableHDU.from_columns(hdul[1].columns, nrows=total_rows)

    pos = np.cumsum([0] + nrows) # [0, row1, row1+row2, ...]
    
    for colname in hdul[1].columns.names:
        for i in range(len(nrows)):
            start = pos[i]
            end = pos[i+1]
            hdu.data[colname][start:end] = hdul[i+1].data[colname]
            
    new_hdul.append(hdu)
    return new_hdul

"""
def appendFitsTable(table1, table2):
    nrows1 = table1.data.shape[0]
    nrows2 = table2.data.shape[0]
    nrows = nrows1 + nrows2
    hdu = fits.BinTableHDU.from_columns(table1.columns, nrows=nrows)
    for colname in table1.columns.names:
        hdu.data[colname][nrows1:] = table2.data[colname]
    return hdu
"""

