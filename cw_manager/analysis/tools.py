import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.stats.distributions import chi2


def detection_stat_threshold(nTemp, nSeg):
    thresh =  chi2.ppf(1.0-(1.0/nTemp), 4*nSeg)/nSeg
    return thresh 

def appendFitsTable(t1, t2):
    return vstack([t1,t2])

def appendFitsTableInFile(hdul):
    nrows = [table.data.shape[0] for table in hdul[1:]]
    new_hdul = fits.HDUList()
    hdu = fits.BinTableHDU.from_columns(hdul[1].columns, nrows=sum(nrows))

    pos = np.cumsum(nrows)
    for colname in hdul[1].columns.names:
        for i in range(len(pos)-1):
            hdu.data[colname][pos[i]:pos[i+1]] = hdul[i+2].data[colname]
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


    