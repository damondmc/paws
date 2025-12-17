import numpy as np
from astropy.io import fits
from astropy.table import Table

from paws.definitions import ext_param_name, phase_param_name
from .models import f1_broad_range, f2_broad_range, f3_value, f4_value

class InjectionParamGenerator:
    def __init__(self, target, ref_time, f0_band=0.1):
        self.target = target
        self.ref_time = ref_time
        self.f0_band = f0_band
        self.inj_col_names = ext_param_name()

        self.tau = target['age'] * 86400 * 365.25  # Convert to seconds
        self.alpha = target['alpha']
        self.dalpha = target['dalpha']
        self.delta = target['delta']
        self.ddelta = target['ddelta']

    @staticmethod
    def sky_sampler(target_ra, target_dec, sky_uncertainty, n_samples):
        """
        Samples sky locations uniformly within a circular cap around the target.
        """
        # 1. Sample uniformly in cos(theta) and phi
        z_min = np.cos(sky_uncertainty)
        z = np.random.uniform(z_min, 1.0, n_samples)
        theta = np.arccos(z)
        phi = np.random.uniform(0, 2 * np.pi, n_samples)

        # 2. Target vector basis
        sin_d, cos_d = np.sin(target_dec), np.cos(target_dec)
        sin_a, cos_a = np.sin(target_ra), np.cos(target_ra)
        
        # Unit vector of center
        k = np.array([cos_d * cos_a, cos_d * sin_a, sin_d])
        
        # Define arbitrary orthogonal basis (u, v)
        if abs(k[2]) < 0.99:
            u = np.cross([0, 0, 1], k)
        else:
            u = np.cross([0, 1, 0], k)
        u /= np.linalg.norm(u)
        v = np.cross(k, u)

        # 3. Construct sampled vectors
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        x = sin_theta * np.cos(phi)
        y = sin_theta * np.sin(phi)
        
        # Vectorized construction
        vec = (x[None, :] * u[:, None] + 
               y[None, :] * v[:, None] + 
               cos_theta[None, :] * k[:, None])

        # 4. Convert back to RA/Dec
        delta = np.arcsin(vec[2])
        alpha = np.arctan2(vec[1], vec[0]) % (2 * np.pi)

        return alpha, delta

    def get_f0_from_non_sat_bands(self, non_sat_bands, n_inj):
        """Randomly draws f0 values from the available non-saturated bands."""
        if non_sat_bands is None or len(non_sat_bands) == 0:
            raise ValueError("No non-saturated bands provided for injection!")
            
        # Uniform probability across bands
        band_idx = np.random.randint(0, len(non_sat_bands), n_inj)
        selected_bands = non_sat_bands[band_idx]
        
        # Uniform frequency within the specific 0.1Hz (or f0_band) chunk
        f0 = selected_bands + np.random.uniform(0, self.f0_band, n_inj)
        return f0

    def generate_injection_table(self, non_sat_bands, h0, n_inj, inj_freq_deriv_order, sky_uncertainty):
        """Generates the source injection parameters."""
        freq_params, _ = phase_param_name(inj_freq_deriv_order)
        
        # Create column names: Standard Inj names + Phase params (Freq, f1dot...)
        col_names = self.inj_col_names + [p.capitalize() if p=='freq' else p for p in freq_params]
        
        # Pre-allocate
        inj_data = Table(np.zeros((n_inj, len(col_names))), names=col_names)

        # 1. Sky Location
        alpha, delta = self.sky_sampler(self.alpha, self.delta, sky_uncertainty, n_inj)
        inj_data['Alpha'] = alpha
        inj_data['Delta'] = delta

        # 2. Polarization angle & reference time
        inj_data['psi'] = np.random.uniform(-np.pi/4, np.pi/4, n_inj)
        inj_data['refTime'] = self.ref_time
        
        cos_i = np.random.uniform(-1, 1, n_inj)
        inj_data['aPlus'] = h0 * (1. + cos_i**2) / 2.
        inj_data['aCross'] = h0 * cos_i

        # 3. Frequency Evolution
        f0 = self.get_f0_from_non_sat_bands(non_sat_bands, n_inj)
        inj_data['Freq'] = f0
        
        if inj_freq_deriv_order >= 1:
            f1_min, f1_max, _ = f1_broad_range(f0, 0, self.tau)
            f1 = np.random.uniform(f1_min, f1_max)
            inj_data['f1dot'] = f1
            
        if inj_freq_deriv_order >= 2:
            f2_min, f2_max, _ = f2_broad_range(f0, 0, f1, f1)
            f2 = np.random.uniform(f2_min, f2_max)
            inj_data['f2dot'] = f2
        
        if inj_freq_deriv_order >= 3:
            inj_data['f3dot'] = f3_value(f0, f1, f2)

        if inj_freq_deriv_order >= 4:
            inj_data['f4dot'] = f4_value(f0, f1, f2)
            
        return fits.BinTableHDU(inj_data)

    def generate_search_range_table(self, spacing, inj_data, freq_deriv_order, n_spacing=1):
        """
        Generates the small search windows around the injections for Weave.
        
        :param spacing_info: Can be a file path (str/Path) containing header info, or a dict.
        """
        freq_names, deriv_names = phase_param_name(freq_deriv_order)

        # Prepare Table
        n_rows = len(inj_data)
        out_cols = freq_names + deriv_names + ['alpha', 'dalpha', 'delta', 'ddelta']
        search_range = Table(np.zeros((n_rows, len(out_cols))), names=out_cols)

        # 1. Frequency & Spindowns
        # f0
        df = spacing['df']        
        search_range['freq'] = inj_data['Freq'] - n_spacing * df
        search_range['dfreq'] = 2 * n_spacing * df
        
        # f1
        if freq_deriv_order >= 1:
            df1 = spacing['df1dot']
            search_range['f1dot'] = inj_data['f1dot'] - n_spacing * df1
            search_range['df1dot'] = 2 * n_spacing * df1
            
        # f2
        if freq_deriv_order >= 2:
            df2 = spacing['df2dot']
            search_range['f2dot'] = inj_data['f2dot'] - n_spacing * df2
            search_range['df2dot'] = 2 * n_spacing * df2

        # f3
        if freq_deriv_order >= 3:
            df3 = spacing['df3dot'] 
            search_range['f3dot'] = inj_data['f3dot'] - n_spacing * df3
            search_range['df3dot'] = 2 * n_spacing * df3
        
        # f4
        if freq_deriv_order >= 4:
            df4 = spacing['df4dot'] 
            search_range['f4dot'] = inj_data['f4dot'] - n_spacing * df4
            search_range['df4dot'] = 2 * n_spacing * df4

        # 2. Sky
        search_range['alpha'] = self.alpha
        search_range['dalpha'] = self.dalpha
        search_range['delta'] = self.delta
        search_range['ddelta'] = self.ddelta

        return fits.BinTableHDU(search_range)

    def generate_parameters(self, non_sat_bands, spacing, h0, freq, n_inj=1, n_spacing=1,
                            inj_freq_deriv_order=4, freq_deriv_order=2, sky_uncertainty=0):
        """
        High-level wrapper to generate both Injection Parameters and Search Ranges.
        
        :param non_sat_bands: Array of non-saturated band frequencies.
        :param spacing_info: Path to FITS file with spacing info OR a dictionary.
        """

        if freq_deriv_order > 4:
            print('Error: frequency derivative order larger than 4.')
        if inj_freq_deriv_order > 4:
            print('Error: Injection frequency derivative order larger than 4.')
            
        # 1. Generate Injection Table
        ip_hdu = self.generate_injection_table(non_sat_bands, h0, n_inj, inj_freq_deriv_order, sky_uncertainty)
        
        # 2. Generate Search Range Table
        sp_hdu = self.generate_search_range_table(spacing, ip_hdu.data, freq_deriv_order, n_spacing)
        
        injParamDict = {str(freq): ip_hdu}
        searchParamDict = {str(freq): sp_hdu}
       
        return searchParamDict, injParamDict