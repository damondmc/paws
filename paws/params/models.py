import numpy as np
import sys 

class PowerLawModel:
    """
    Defines parameter ranges based on the braking index (Power Law) physics.
    Bounds are dynamically calculated using f0 and characteristic age (tau).
    """
    def __init__(self, nc_min=2, nc_max=7, tau=300*365.25*24*3600):
        self.nc_min = nc_min
        self.nc_max = nc_max
        self.tau = tau

    def get_n_f1(self, freq, f0_band, df1=1.5e-9):
        _, _, bandwidth = self.f1_broad_range(freq, f0_band, self.tau)
        n = bandwidth / df1
        return np.ceil(n).astype(int)

    def get_n_f2(self, freq, f0_band, df2=1.0e-19):
        f1_min, f1_max, _ = self.f1_broad_range(freq, f0_band, self.tau)
        _, _, bandwidth = self.f2_broad_range(freq, f0_band, f1_min, f1_max)
        n = bandwidth / df2
        return np.ceil(n).astype(int)

    def f0_broad_range(self, f0, f0_band):
        f0_min = f0
        f0_max = f0 + f0_band
        return f0_min, f0_max, f0_band

    def f1_broad_range(self, f0, f0_band):
        # f1 is negative for spindown: max is 0 (conservative), min depends on tau/nc
        f1_min = -(f0 + f0_band) / ((self.nc_min - 1.) * self.tau)
        f1_max = np.zeros(np.array(f1_min).shape) 
        return f1_min, f1_max, f1_max - f1_min

    def f2_broad_range(self, f0, f0_band, f1_min, f1_max):
        f2_min = self.nc_min * np.minimum(f1_min**2, f1_max**2) / (f0 + f0_band)
        f2_max = self.nc_max * np.maximum(f1_min**2, f1_max**2) / f0
        return f2_min, f2_max, f2_max - f2_min

    def f3_broad_range(self, f0, f0_band, f1_min, f1_max):
        f3_min = self.nc_max * (2 * self.nc_max - 1) * np.minimum(f1_min**3, f1_max**3) / f0**2
        f3_max = self.nc_min * (2 * self.nc_min - 1) * np.maximum(f1_min**3, f1_max**3) / (f0 + f0_band)**2
        return f3_min, f3_max, f3_max - f3_min

    def f4_broad_range(self, f0, f0_band, f1_min, f1_max):
        f4_min = self.nc_min * (3 * self.nc_min - 1) * (2 * self.nc_min - 2) * np.minimum(f1_min**4, f1_max**4) / (f0 + f0_band)**3
        f4_max = self.nc_max * (3 * self.nc_max - 1) * (2 * self.nc_max - 2) * np.maximum(f1_min**4, f1_max**4) / f0**3
        return f4_min, f4_max, f4_max - f4_min

    def f3_value(self, f0, f1, f2): 
        """Calculates specific f3 based on braking index relationship."""
        with np.errstate(divide='ignore', invalid='ignore'):
            nc = f0 * f2 / f1**2
        nc = np.atleast_1d(nc)
        # We allow checking this even if inputs are arrays; handled by numpy
        # If braking index is physically impossible (<0), we error out.
        if np.any(nc < 0):
            sys.exit('braking index is negative')
        f3 = nc * (2. * nc - 1.) * f1**3 / f0**2  
        return f3

    def f4_value(self, f0, f1, f2): 
        with np.errstate(divide='ignore', invalid='ignore'):
            nc = f0 * f2 / f1**2
        nc = np.atleast_1d(nc)
        if np.any(nc < 0):
            sys.exit('braking index is negative')
        f4 = nc * (3. * nc - 1.) * (2. * nc - 2.) * f1**4 / f0**3  
        return f4


class UniformModel:
    """
    Defines parameter ranges based on fixed rectangular bounds (Uniform).
    Ignores physics inputs like 'tau' but keeps the method signature compatible.
    """
    def __init__(self, f1_lim, f2_lim, f3_lim=None, f4_lim=None):
        """
        :param f1_lim: tuple/list (min, max) for f1
        :param f2_lim: tuple/list (min, max) for f2
        """
        self.f1_lim = f1_lim
        self.f2_lim = f2_lim
        self.f3_lim = f3_lim if f3_lim else (0, 0)
        self.f4_lim = f4_lim if f4_lim else (0, 0)

    def get_n_f1(self, df1):
        # We ignore 'tau' here
        _, _, bandwidth = self.f1_broad_range()
        n = bandwidth / df1
        return np.ceil(n).astype(int)

    def get_n_f2(self, df2):
        # We ignore 'tau' and f1 inputs, using fixed bounds
        _, _, bandwidth = self.f2_broad_range()
        n = bandwidth / df2
        return np.ceil(n).astype(int)

    def f0_broad_range(self, f0, f0_band):
        return f0, f0 + f0_band, f0_band

    def f1_broad_range(self):
        f1_min, f1_max = self.f1_lim
        return f1_min, f1_max, f1_max - f1_min

    def f2_broad_range(self):
        f2_min, f2_max = self.f2_lim
        return f2_min, f2_max, f2_max - f2_min

    def f3_broad_range(self):
        f3_min, f3_max = self.f3_lim
        return f3_min, f3_max, f3_max - f3_min

    def f4_broad_range(self):
        f4_min, f4_max = self.f4_lim
        return f4_min, f4_max, f4_max - f4_min
