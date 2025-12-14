import numpy as np
import sys 

def get_n_f1_dot(freq, f_band, tau, df1_dot=1.5e-9):
    _, _, bandwidth = f1_broad_range(freq, f_band, tau)
    n = bandwidth / df1_dot
    return np.ceil(n).astype(int)

def get_n_f2_dot(freq, f_band, tau, df2_dot=1.0e-19):
    f1_min, f1_max, _ = f1_broad_range(freq, f_band, tau)
    _, _, bandwidth = f2_broad_range(freq, f_band, f1_min, f1_max)
    n = bandwidth / df2_dot
    return np.ceil(n).astype(int)

def f0_broad_range(f0, f_band):
    f0_min = f0
    f0_max = f0 + f_band
    return f0_min, f0_max , f_band

def f1_broad_range(f0, f_band, tau, nc_min=2, nc_max=7):
    f1_min = -(f0 + f_band) / ((nc_min - 1.) * tau)
    # f1_max = -f0/((setup.nc_max-1.)*tau)
    f1_max = np.zeros(np.array(f1_min).shape) # more conservative upper bound f1_max = 0
    return f1_min, f1_max, f1_max - f1_min

def f2_broad_range(f0, f_band, f1_min, f1_max, nc_min=2, nc_max=7):
    f2_min = nc_min * np.minimum(f1_min**2, f1_max**2) / (f0 + f_band)
    f2_max = nc_max * np.maximum(f1_min**2, f1_max**2) / f0
    return f2_min, f2_max, f2_max - f2_min

def f3_value(f, f1, f2): 
    nc = f * f2 / f1**2
    nc = np.atleast_1d(nc)
    if nc[nc < 0].size != 0:
        sys.exit('braking index is negative')
    f3 = nc * (2. * nc - 1.) * f1**3 / f**2  
    return f3

def f4_value(f, f1, f2): 
    nc = f * f2 / f1**2
    nc = np.atleast_1d(nc)
    if nc[nc < 0].size != 0:
        sys.exit('braking index is negative')
    f4 = nc * (3. * nc - 1.) * (2. * nc - 2.) * f1**4 / f**3  
    return f4

def f3_broad_range(f0, f_band, f1_min, f1_max, nc_min=2, nc_max=7):
    # nc_min = f0 * f2_min / np.maximum(f1_min**2, f1_max**2)
    # nc_max = (f0 + f_band) * f2_max / np.minimum(f1_min**2, f1_max**2)
    f3_min = nc_max * (2 * nc_max - 1) * np.minimum(f1_min**3, f1_max**3) / f0**2
    f3_max = nc_min * (2 * nc_min - 1) * np.maximum(f1_min**3, f1_max**3) / (f0 + f_band)**2
    return f3_min, f3_max, f3_max - f3_min

# f4 > 0
def f4_broad_range(f0, f_band, f1_min, f1_max, nc_min=2, nc_max=7):
    # nc_min = f0 * f2_min / np.maximum(f1_min**2, f1_max**2)
    # nc_max = (f0 + f_band) * f2_max / np.minimum(f1_min**2, f1_max**2)
    f4_min = nc_min * (3 * nc_min - 1) * (2 * nc_min - 2) * np.minimum(f1_min**4, f1_max**4) / (f0 + f_band)**3
    f4_max = nc_max * (3 * nc_max - 1) * (2 * nc_max - 2) * np.maximum(f1_min**4, f1_max**4) / f0**3
    return f4_min, f4_max, f4_max - f4_min