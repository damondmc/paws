import numpy as np
import sys 

def getNf1dot(freq, fBand, tau, df1dot=1.5e-9):
    _, _, bandwidth = f1BroadRange(freq, fBand, tau)
    n = bandwidth / df1dot
    return np.ceil(n).astype(int)


def getNf2dot(freq, fBand, tau, df2dot=1.0e-19):
    f1min, f1max, _ = f1BroadRange(freq, fBand, tau)
    _, _, bandwidth = f2BroadRange(freq, fBand, f1min, f1max)
    n = bandwidth / df2dot
    return np.ceil(n).astype(int)

def f0BroadRange(f0, fBand):
    f0min = f0
    f0max = f0 + fBand
    return f0min, f0max , fBand

def f1BroadRange(f0, fBand, tau, nc_min=2, nc_max=7):
    f1min = -(f0+fBand)/((nc_min-1.)*tau)
    #f1max = -f0/((setup.nc_max-1.)*tau)
    f1max = np.zeros(np.array(f1min).shape) # more conservative upper bound f1max = 0
    return f1min, f1max, f1max-f1min

def f2BroadRange(f0, fBand, f1min, f1max, nc_min=2, nc_max=7):
    f2min = nc_min*np.minimum(f1min**2,f1max**2)/(f0+fBand)
    f2max = nc_max*np.maximum(f1min**2,f1max**2)/f0
    return f2min, f2max, f2max-f2min

def f3Value(f, f1, f2): 
    nc = f * f2/ f1**2
    nc = np.atleast_1d(nc)
    if nc[nc<0].size != 0:
        sys.exit('braking index is negatve')
    f3 = nc*(2.*nc-1.)*f1**3/f**2  
    return f3

def f4Value(f, f1, f2): 
    nc = f * f2/ f1**2
    nc = np.atleast_1d(nc)
    if nc[nc<0].size != 0:
        sys.exit('braking index is negatve')
    f4 = nc*(3.*nc-1.)*(2.*nc-2.)*f1**4/f**3  
    return f4

def f3BroadRange(f0, fBand, f1min, f1max, f2min, f2max, nc_min=2, nc_max=7):
    #nc_min = f0 * f2min / np.maximum(f1min**2, f1max**2)
    #nc_max = (f0+fBand) * f2max / np.minimum(f1min**2, f1max**2)
    f3min = nc_max*(2*nc_max-1)*np.minimum(f1min**3, f1max**3)/f0**2
    f3max = nc_min*(2*nc_min-1)*np.maximum(f1min**3, f1max**3)/(f0+fBand)**2
    return f3min, f3max, f3max-f3min

# f4 > 0
def f4BroadRange(f0, fBand, f1min, f1max, f2min, f2max, nc_min=2, nc_max=7):
    #nc_min = f0 * f2min / np.maximum(f1min**2, f1max**2)
    #nc_max = (f0+fBand) * f2max / np.minimum(f1min**2, f1max**2)
    f4min = nc_min*(3*nc_min-1)*(2*nc_min-2)*np.minimum(f1min**4, f1max**4)/(f0+fBand)**3
    f4max = nc_max*(3*nc_max-1)*(2*nc_max-2)*np.maximum(f1min**4, f1max**4)/f0**3
    return f4min, f4max, f4max-f4min
