from paws.definitions import phase_param_name
from .models import PowerLawModel, UniformModel
from tqdm import tqdm
import numpy as np
from astropy.io import fits
from astropy.table import Table

class SearchParamGenerator:    
    """
    Generate initial search parameters for a given sky location and frequency range.
    Uses the provided model (PowerLaw or Uniform) to determine parameter boundaries.
    """

    def __init__(self, model, freq_deriv_order=2):
        """
        Initialize the generator.

        Parameters:
        - model (object): Instance of PowerLawModel or UniformModel (REQUIRED).
        - f0_band (float): Frequency band width for each search segment.
        - freq_deriv_order (int): Order of frequency derivatives (default: 2).
        """
        self.freq_deriv_order = freq_deriv_order
        self.freq_param_name, self.freq_deriv_param_name = phase_param_name(freq_deriv_order)
        self.model = model
        
        # Initialize attributes that are set later via generate_parameters
        self.alpha = None
        self.dalpha = None
        self.delta = None
        self.ddelta = None
        
    def generate_parameter_table(self, freq, df, n_f1, n_f2, n_f3=1, n_f4=1, bandwidth=1.0):
        """
        Generate a parameter table for a specific frequency chunk.
        
        Parameters:
        - bandwidth (float): The frequency range this table covers (default 1.0 Hz).
        """
        # Calculate number of steps (sub-bands) in this file
        # Use round() to avoid floating point truncation errors (e.g., 0.299999 / 0.1)
        n_f0 = int(np.round(bandwidth / df))
        # Total rows = (sub-bands) * (f1 segments) * (f2 segments) ...
        n = int(n_f0 * n_f1 * n_f2 * n_f3 * n_f4)
        
        dtype_list = [(key, '>f8') for key in (self.freq_param_name + self.freq_deriv_param_name)]
        data = np.recarray((n,), dtype=dtype_list) 
        
        for i in range(n_f0):
            f0 = freq + i * df
            
            # Handle f0 range
            f0_min, _, _ = self.model.f0_broad_range(f0, df)


            for j in range(n_f1):
                # 1. Get f1 range
                if isinstance(self.model, PowerLawModel):
                    _f1_min, _, _f1_band = self.model.f1_broad_range(f0, df)
                else:
                    _f1_min, _, _f1_band = self.model.f1_broad_range()
                
                f1_band = _f1_band / n_f1  
                f1_min = _f1_min + j * f1_band
                f1_max = f1_min + f1_band
                
                # Handling numerical edge cases for PowerLawModel
                if j == n_f1 - 1 and isinstance(self.model, PowerLawModel):
                    f1_max = 0.0             
                    f1_band = 0.0 - f1_min
                        
                for k in range(n_f2):
                    # 2. Get f2 range
                    if isinstance(self.model, PowerLawModel):
                        _f2_min, _, _f2_band = self.model.f2_broad_range(f0, df, f1_min, f1_max)
                    else:
                        _f2_min, _, _f2_band = self.model.f2_broad_range()
                    
                    f2_band = _f2_band / n_f2  
                    f2_min = _f2_min + k * f2_band
                    f2_max = f2_min + f2_band
                    
                    if j == (n_f1 - 1) and k == 0 and isinstance(self.model, PowerLawModel):
                        f2_min = 0.0
                        f2_band = f2_max    

                    # --- F3 Loop ---
                    for l in range(n_f3):
                        if self.freq_deriv_order >= 3:
                            if isinstance(self.model, PowerLawModel):
                                _f3_min, _, _f3_band = self.model.f3_broad_range(f0, df, f1_min, f1_max)
                            else:
                                _f3_min, _, _f3_band = self.model.f3_broad_range()
                            
                            f3_band = _f3_band / n_f3
                            f3_min = _f3_min + l * f3_band
                        else:
                            f3_min, f3_band = 0.0, 0.0

                        # --- F4 Loop ---
                        for m in range(n_f4):
                            if self.freq_deriv_order >= 4:
                                if isinstance(self.model, PowerLawModel):
                                    _f4_min, _, _f4_band = self.model.f4_broad_range(f0, df, f1_min, f1_max)
                                else:
                                    _f4_min, _, _f4_band = self.model.f4_broad_range()

                                f4_band = _f4_band / n_f4
                                f4_min = _f4_min + m * f4_band
                            else:
                                f4_min, f4_band = 0.0, 0.0

                            # Calculate Index
                            idx = (i * n_f1 * n_f2 * n_f3 * n_f4) + \
                                  (j * n_f2 * n_f3 * n_f4) + \
                                  (k * n_f3 * n_f4) + \
                                  (l * n_f4) + m

                            # Assign Data
                            data[idx]['freq'], data[idx]['df'] = f0_min, df
                            data[idx]['f1dot'], data[idx]['df1dot'] = f1_min, f1_band
                            data[idx]['f2dot'], data[idx]['df2dot'] = f2_min, f2_band
                            
                            if self.freq_deriv_order >= 3:
                                data[idx]['f3dot'], data[idx]['df3dot'] = f3_min, f3_band
                            if self.freq_deriv_order >= 4:
                                data[idx]['f4dot'], data[idx]['df4dot'] = f4_min, f4_band

        data = Table(data)

        # Sky location
        data.add_column(self.alpha * np.ones(n), name='alpha')
        data.add_column(self.dalpha * np.ones(n), name='dalpha')
        data.add_column(self.delta * np.ones(n), name='delta')
        data.add_column(self.ddelta * np.ones(n), name='ddelta')           
        return fits.BinTableHDU(data)
        
    def generate_parameters(self, alpha, dalpha, delta, ddelta, f0_min, f0_max, 
                            file_bandwidth=1.0, df=0.02, df1=1e-9, df2=1e-19, df3=None, df4=None):
        """
        Generate initial search parameters.
        
        Parameters:
            - f0_min, f0_max (float): Frequency range (e.g., 31.5 to 31.8).
            - file_bandwidth (float): The size of each output file/table in Hz (default 1.0).
            - df (float, optional): Resolution for f0. Defaults to self.f0_band if None.
        """
        self.alpha = alpha
        self.dalpha = dalpha
        self.delta = delta
        self.ddelta = ddelta

        params = {}
        
        # Current start frequency
        current_freq = f0_min
        
        # Use a while loop to handle non-integer ranges
        pbar = tqdm(total=f0_max - f0_min)
        
        while current_freq < f0_max:
                
            next_boundary = int(current_freq + 1)
            
            # The actual stop frequency is the smaller of the boundary or the requested max
            next_freq = min(next_boundary, f0_max)
            
            # Actual bandwidth for this specific table (e.g., 31.5->32.0 is 0.5Hz)
            chunk_bw = next_freq - current_freq
            
            # 2. Stop if chunk is too small (handle float precision residue)
            if chunk_bw < df / 2.0:
                 break

            # 3. Calculate N segments based on current_freq (physics model varies with freq)
            if isinstance(self.model, PowerLawModel):
                n_f1 = self.model.get_n_f1(current_freq, df, df1=df1) 
                n_f2 = self.model.get_n_f2(current_freq, df, df2=df2) 
            else:
                n_f1 = self.model.get_n_f1(df1=df1) 
                n_f2 = self.model.get_n_f2(df2=df2) 
                
            # F3 Logic
            n_f3 = 1
            if self.freq_deriv_order >= 3 and df3 is not None:
                if isinstance(self.model, PowerLawModel):
                    f1_min, f1_max, _ = self.model.f1_broad_range(current_freq, df)
                    _, _, width_f3 = self.model.f3_broad_range(current_freq, df, f1_min, f1_max)
                else:
                    _, _, width_f3 = self.model.f3_broad_range()
                n_f3 = int(np.ceil(width_f3 / df3))

            # F4 Logic
            n_f4 = 1
            if self.freq_deriv_order >= 4 and df4 is not None:
                if isinstance(self.model, PowerLawModel):
                    f1_min, f1_max, _ = self.model.f1_broad_range(current_freq, df)
                    _, _, width_f4 = self.model.f4_broad_range(current_freq, df, f1_min, f1_max)
                else:
                    _, _, width_f4 = self.model.f4_broad_range()
                n_f4 = int(np.ceil(width_f4 / df4))
            
            # 4. Generate table with calculated chunk bandwidth
            table = self.generate_parameter_table(current_freq, df, n_f1, n_f2, n_f3, n_f4, bandwidth=chunk_bw)
            
            if table is not None:
                params[int(current_freq)] = table
            
            pbar.update(chunk_bw)
            current_freq = next_freq
            
        pbar.close()
        return params
