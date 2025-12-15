from paws.definitions import phase_param_name
from . import models as fr  # Relative import of the sibling file
from tqdm import tqdm
import numpy as np
from astropy.io import fits
from astropy.table import Table

class SearchParamGenerator:    
    """
    Generate initial search parameters for a given age and sky location.
    The class divides the frequency derivative ranges into segments based on specified resolutions
    and constructs a parameter table for each frequency bin.
    """

    def __init__(self, f0_band=0.1, freq_deriv_order=2, nc_min=2, nc_max=7):
        """
        Initialize the generator.

        Parameters:
        - f0_band (float): Frequency band width for each search segment (default: 0.1 Hz).
        - freq_deriv_order (int): Order of frequency derivatives to consider (default: 2 for f1dot and f2dot).
        - nc_min (int): Minimum braking index for frequency derivative calculations (default: 2).
        - nc_max (int): Maximum braking index for frequency derivative calculations (default: 7).
        """
        self.f0_band = f0_band
        # Assuming phaseParamName is defined in definitions.py and returns (names, deriv_names)
        self.freq_param_name, self.freq_deriv_param_name = phase_param_name(freq_deriv_order)
        self.nc_min = nc_min
        self.nc_max = nc_max
        
        # Initialize attributes that are set later
        self.tau = None
        self.alpha = None
        self.dalpha = None
        self.delta = None
        self.ddelta = None
        
    def generate_parameter_table(self, freq, n_f1, n_f2):
        """
        Generate a parameter table for a specific frequency bin.

        Parameters:
        - freq (float): Starting frequency (int) for the parameter table.
        - n_f1 (int): Number of segments for f1dot range.
        - n_f2 (int): Number of segments for f2dot range.

        Returns:
        - fits.BinTableHDU: A FITS binary table HDU containing the generated parameters.    
        """
        # Calculate size of the array
        # n = (number of frequency sub-bands) * (f1 segments) * (f2 segments)
        n = int((1.0 / self.f0_band) * n_f1 * n_f2)
        
        dtype_list = [(key, '>f8') for key in (self.freq_param_name + self.freq_deriv_param_name)]
        data = np.recarray((n,), dtype=dtype_list) 
        
        # Iterate through sub-bands (e.g., 0.1Hz steps within the current 1Hz band)
        steps = int(1.0 / self.f0_band)
        
        for i in range(steps):
            f0 = freq + i * self.f0_band
            f0_min, _, f0_band = fr.f0_broad_range(f0, self.f0_band)
            
            for j in range(n_f1):
                _f1_min, _, _f1_band = fr.f1_broad_range(f0, self.f0_band, self.tau, nc_min=self.nc_min, nc_max=self.nc_max)
                f1_band = _f1_band / n_f1  # divide f1dot into n segments
                f1_min = _f1_min + j * f1_band
                f1_max = f1_min + f1_band
                
                if j == n_f1 - 1:
                    f1_max = 0.0             # manually set f1dot upper limit to 0 (numerical accuracy/error exits)
                    f1_band = 0.0 - f1_min
                        
                for k in range(n_f2):
                    _f2_min, _, _f2_band = fr.f2_broad_range(freq, self.f0_band, f1_min, f1_max, nc_min=self.nc_min, nc_max=self.nc_max)
                    f2_band = _f2_band / n_f2  
                    f2_min = _f2_min + k * f2_band
                    f2_max = f2_min + f2_band
                    
                    if j == (n_f1 - 1) and k == 0:
                        f2_min = 0.0
                        f2_band = f2_max    # manually set f2dot lower limit to 0 (numerical accuracy/error exits)
                    
                    idx = i * n_f1 * n_f2 + j * n_f2 + k 
                    data[idx]['freq'], data[idx]['df'] = f0_min, f0_band
                    data[idx]['f1dot'], data[idx]['df1dot'] = f1_min, f1_band
                    data[idx]['f2dot'], data[idx]['df2dot'] = f2_min, f2_band
                    
        data = Table(data)

        # sky location
        data.add_column(self.alpha * np.ones(n), name='alpha')
        data.add_column(self.dalpha * np.ones(n), name='dalpha')
        data.add_column(self.delta * np.ones(n), name='delta')
        data.add_column(self.ddelta * np.ones(n), name='ddelta')           
        return fits.BinTableHDU(data)
        
    def generate_parameters(self, tau, alpha, dalpha, delta, ddelta, f0_min, f0_max, df1=1e-9, df2=1e-19):
        """
        Generate initial search parameters for a given age and sky location.

        Parameters:
            - tau (float): Age of the pulsar (in years/seconds depending on input).
            - alpha (float): Right ascension of the target pulsar (in radians).
            - dalpha (float): Uncertainty in the right ascension (in radians).
            - delta (float): Declination of the target pulsar (in radians).
            - ddelta (float): Uncertainty in the declination (in radians).
            - f0_min (int): Minimum starting frequency (in Hz).
            - f0_max (int): Maximum starting frequency (in Hz).
            - df1_dot (float): Desired resolution for f1dot (default: 1e-9 Hz/s).
            - df2_dot (float): Desired resolution for f2dot (default: 1e-19 Hz/s).
        
        Returns:
            - dict: A dictionary where keys are frequencies (str) and values are parameter tables.
        """
        self.tau = tau 
        self.alpha = alpha
        self.dalpha = dalpha
        self.delta = delta
        self.ddelta = ddelta

        params = {}
        for freq in tqdm(range(f0_min, f0_max)):
            # number of segment for f1dot range
            n_f1 = fr.get_n_f1_dot(freq, self.f0_band, self.tau, df1=df1) 
            # number of segment for f2dot range
            n_f2 = fr.get_n_f2_dot(freq, self.f0_band, self.tau, df2=df2) 
            params[str(freq)] = self.generate_parameter_table(freq, n_f1, n_f2)
        return params