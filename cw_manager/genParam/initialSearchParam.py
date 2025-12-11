from ..definitions import phaseParamName
from . import frequencyRange as fr
from tqdm import tqdm
import numpy as np
from astropy.io import fits
from astropy.table import Table

class initSearchParams:    
    """
    Generate initial search parameter for a given age and sky location.
    The class divides the frequency derivative ranges into segments based on specified resolutions
    and constructs a parameter table for each frequency bin.
    """

    def __init__(self, fBand=0.1, freqDerivOrder=2, nc_min=2, nc_max=7):
        """
        Parameters:
        - fBand: float
            Frequency band width for each search segment (default: 0.1 Hz).
        - freqDerivOrder: int
            Order of frequency derivatives to consider (default: 2 for f1dot and f2dot).
        - nc_min: int
            Minimum braking index for frequency derivative calculations (default: 2).
        - nc_max: int
            Maximum braking index for frequency derivative calculations (default: 7).
        """
        self.fBand = fBand
        self.freqParamName, self.freqDerivParamName = phaseParamName(freqDerivOrder)
        self.nc_min = nc_min
        self.nc_max = nc_max
        
    def genParamTable(self, freq, nf1dots, nf2dots):
        """
        Generate a parameter table for a specific frequency bin.
        Parameters:
        - freq: float
            Starting frequency (int) for the parameter table.
        - nf1dots: int
            Number of segments for f1dot range.
        - nf2dots: int
            Number of segments for f2dot range.
        Returns:
        - fits.BinTableHDU
            A FITS binary table HDU containing the generated parameters.    
        """
        n = int(nf1dots*nf2dots/self.fBand)
        data =np.recarray((n,), dtype=[(key, '>f8') for key in (self.freqParamName + self.freqDerivParamName)]) 
        
        for i in range(int(1.0/self.fBand)):
            f0 = freq + i *self.fBand
            f0min, f0max, f0band = fr.f0BroadRange(f0, self.fBand)
            for j in range(nf1dots):
                _f1min, _, _f1Band = fr.f1BroadRange(f0, self.fBand, self.tau, nc_min=self.nc_min, nc_max=self.nc_max)
                f1Band = _f1Band/nf1dots  # divide f1dot into n segment
                f1min = _f1min + j*f1Band
                f1max = f1min + f1Band
                if j == nf1dots - 1:
                    f1max = 0.0             # to mannually set f1dot upper limit to 0 (numerical accuracy/error exits)
                    f1Band = 0.0 - f1min
                        
                for k in range(nf2dots):
                    _f2min, _, _f2Band = fr.f2BroadRange(freq, self.fBand, f1min, f1max, nc_min=self.nc_min, nc_max=self.nc_max)
                    f2Band = _f2Band/nf2dots
                    f2min = _f2min + k*f2Band
                    f2max = f2min + f2Band
                    if j == (nf1dots-1) and k == 0:
                        f2min = 0.0
                        f2Band = f2max    # to mannually set f2dot lower limit to 0 (numerical accuracy/error exits)
                    
                    idx = i * nf1dots * nf2dots + j * nf2dots + k 
                    data[idx]['freq'], data[idx]['df'] = f0min, f0band
                    data[idx]['f1dot'], data[idx]['df1dot'] = f1min, f1Band
                    data[idx]['f2dot'], data[idx]['df2dot'] = f2min, f2Band
                    
        data = Table(data)

        # sky location
        data.add_column(self.alpha*np.ones(n), name='alpha')
        data.add_column(self.dalpha*np.ones(n), name='dalpha')
        data.add_column(self.delta*np.ones(n), name='delta')
        data.add_column(self.ddelta*np.ones(n), name='ddelta')           
        return fits.BinTableHDU(data)
        
    # need to do, at search result stage, append the table.
    def genParam(self, tau, alpha, dalpha, delta, ddelta, fmin, fmax, df1dot=1e-9, df2dot=1e-19):
        """
        Generate initial search parametter for a given age and sky location.
        Parameters:
            - tau: float
                Age of the pulsar (in years).
            - alpha: float
                Right ascension of the target pulsar (in radians).
            - dalpha: float
                Uncertainty in the right ascension (in radians).
            - delta: float
                Declination of the target pulsar (in radians).
            - ddelta: float
                Uncertainty in the declination (in radians).
            - fmin: int
                Minimum starting frequency (in Hz).
            - fmax: int
                Maximum starting frequency (in Hz).
            - df1dot: float
                Desired resolution for f1dot (default: 1e-9 Hz/s).
            - df2dot: float
                Desired resolution for f2dot (default: 1e-19 Hz/s).
        """
        self.tau = tau 
        self.alpha = alpha
        self.dalpha = dalpha
        self.delta = delta
        self.ddelta = ddelta

        params = {}
        for freq in tqdm(range(fmin, fmax)):
            nf1dots = fr.getNf1dot(freq, self.fBand, self.tau, df1dot=df1dot) # number of segment for f1dot range
            nf2dots = fr.getNf2dot(freq, self.fBand, self.tau, df2dot=df2dot) # number of segment for f2dot range
            params[str(freq)] = self.genParamTable(freq, nf1dots, nf2dots)
        return params        
