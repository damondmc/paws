from astropy.io import fits
from astropy.table import Table

from . import models as fr         
from paws.definitions import phase_param_name

class FollowUpParamGenerator():    
    def __init__(self, target, nc_min=2, nc_max=7):
        """
        Initialize the FollowUpParams generator.

        Parameters:
        - target (object): Target object containing name and properties.
        - nc_min (int): Minimum braking index for frequency derivative calculations (default: 2).
        - nc_max (int): Maximum braking index for frequency derivative calculations (default: 7).
        """
        self.target = target
        self.nc_min = nc_min
        self.nc_max = nc_max
    
    def make_follow_up_table(self, data, old_freq_deriv_order, new_freq_deriv_order, n_spacing): 
        """
        Constructs the follow-up parameter table by expanding ranges.
        
        Parameters:
        - data (numpy.array): Input outlier data.
        - old_freq_deriv_order (int): The derivative order of the previous search.
        - new_freq_deriv_order (int): The derivative order for the new search.
        - n_spacing (int): The spacing multiplier for expanding ranges.

        Returns:
        - fits.BinTableHDU: The constructed follow-up parameter table.
        """
        data = Table(data)
        
        # Check if there are outliers 
        if len(data) == 0:
            return fits.BinTableHDU(data=data)        
        
        # Get parameter names from utils
        freq_param_names, freq_deriv_param_names = phase_param_name(old_freq_deriv_order)
        new_freq_param_names, new_freq_deriv_param_names = phase_param_name(new_freq_deriv_order)
        
        # Expand existing ranges centered on the outlier
        # Logic: New Start = Old Value - (nSpacing * StepSize)
        #        New Bandwidth = 2 * nSpacing * StepSize
        for freq_name, freq_deriv_name in zip(freq_param_names, freq_deriv_param_names):
            data[freq_name] = data[freq_name] - n_spacing * data[freq_deriv_name] 
            data[freq_deriv_name] = 2 * n_spacing * data[freq_deriv_name] 
            
            # Remove processed params from the "new" list so we know what's left to add
            if freq_name in new_freq_param_names:
                new_freq_param_names.remove(freq_name)
            if freq_deriv_name in new_freq_deriv_param_names:
                new_freq_deriv_param_names.remove(freq_deriv_name)
            
        # Add new higher-order derivatives (f3dot, f4dot) that weren't in the previous stage
        for freq_name, freq_deriv_name in zip(new_freq_param_names, new_freq_deriv_param_names):
            if freq_name == 'f3dot':
                # Calculate broad ranges based on current f1/f2 values
                f3_min, _, f3_band = fr.f3_broad_range(
                    data['freq'], 
                    data['df'], 
                    data['f1dot'], 
                    data['f1dot'] + data['df1dot'],
                    nc_min=self.nc_min,
                    nc_max=self.nc_max
                )
                data.add_column(f3_min, name=freq_name)
                data.add_column(f3_band, name=freq_deriv_name)
                
            elif freq_name == 'f4dot':
                f4_min, _, f4_band = fr.f4_broad_range(
                    data['freq'], 
                    data['df'], 
                    data['f1dot'], 
                    data['f1dot'] + data['df1dot'],
                    nc_min=self.nc_min,
                    nc_max=self.nc_max
                )
                data.add_column(f4_min, name=freq_name)
                data.add_column(f4_band, name=freq_deriv_name)
                
        return fits.BinTableHDU(data=data)
           
    def gen_follow_up_param(self, data, old_freq_deriv_order, new_freq_deriv_order, n_spacing=1): 
        """
        Main entry point to generate parameters.
        """
        if old_freq_deriv_order > 4 or new_freq_deriv_order > 4:
            print('Error: frequency derivative order larger than 4.')
            # You might want to raise an actual Exception here:
            # raise ValueError("Frequency derivative order cannot be larger than 4")
             
        params = self.make_follow_up_table(data, old_freq_deriv_order, new_freq_deriv_order, n_spacing)
        
        # Removed unused args: cluster, workInLocalDir are not used in logic anymore
        print(f"Done generation of {self.target['name']} follow-up parameters\n")
        return params