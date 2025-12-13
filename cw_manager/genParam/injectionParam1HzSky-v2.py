from ..utils import utils as utils
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table
import numpy as np
from ..utils import setup_parameter as setup
from astropy.io import fits
from . import frequency_range as fr
from ..analysis import readFile as rf
from ..utils import filePath as fp
from pathlib import Path    


def skySampler(target_ra, target_dec, skyUncertainty, n_samples):
    """
    Sample sky locations uniformly within a circular region in ICRS coordinates.
    
    Parameters:
    - target_ra (float): Target Right Ascension (RA) in radians.
    - target_dec (float): Target Declination (Dec) in radians.
    - skyUncertainty (float): Angular radius of the circular region in radians.
    - n_samples (int): Number of points to sample.
    
    Returns:
    - alpha (ndarray): Array of sampled RA values in radians [0, 2pi].
    - delta (ndarray): Array of sampled Dec values in radians [-pi/2, pi/2].
    """
    # Sample points uniformly within a circular cap
    z_min = np.cos(skyUncertainty)  # Lower bound for cos(theta)
    z = np.random.uniform(z_min, 1.0, n_samples)  # Sample cos(theta) uniformly
    theta = np.arccos(z)  # Angular separation from the center (in radians)
    phi = np.random.uniform(0, 2 * np.pi, n_samples)  # Azimuthal angle in [0, 2pi]

    # Unit vector of the target position in ICRS coordinates
    z0 = np.sin(target_dec)
    r0 = np.cos(target_dec)
    x0 = r0 * np.cos(target_ra)
    y0 = r0 * np.sin(target_ra)

    # Initialize arrays for sampled ICRS coordinates
    alpha = np.zeros(n_samples)  # Right Ascension (RA)
    delta = np.zeros(n_samples)  # Declination (Dec)

    # Full spherical transformation for each sampled point
    for i in range(n_samples):
        cos_theta = np.cos(theta[i])
        sin_theta = np.sin(theta[i])
        cos_phi = np.cos(phi[i])
        sin_phi = np.sin(phi[i])
        
        # Compute new unit vector (x, y, z) after rotation
        x = cos_theta * x0 + sin_theta * (cos_phi * (-y0) + sin_phi * z0 * x0 / r0)
        y = cos_theta * y0 + sin_theta * (cos_phi * x0 + sin_phi * z0 * y0 / r0)
        z = cos_theta * z0 - sin_theta * sin_phi * r0
        
        # Convert back to ICRS RA (alpha) and Dec (delta)
        delta[i] = np.arcsin(z)
        alpha[i] = np.arctan2(y, x) % (2 * np.pi)  # Ensure alpha is in [0, 2pi]

    return alpha, delta



class injectionParams:    
    def __init__(self, target, obsDay, cohDay, refTime, fBand=0.1):
        self.target = target
        self.cohDay= cohDay
        self.refTime = refTime 
        #_, _, _, _, self.refTime = utils.getTimeSetup(self.target.name, obsDay, cohDay)
        self.fBand = fBand
        self.injParamName = utils.injParamName()
        
    def getF0FromNonSatBands(self, nonSatBands, nInj):
        p = np.ones(nonSatBands.shape)
        p = p/p.sum() # Normalize to sum up to one
        f0 = [np.random.choice([np.random.uniform(freq, freq+self.fBand) for freq in nonSatBands], p=p) for _ in range(nInj)]
        return np.array(f0).reshape(nInj)

    def genInjParamTable(self, nonSatBands, h0, freq, nInj, nAmp, freqDerivOrder, skyUncertainty):
        freqParamName, freqDerivParamName = utils.phaseParamName(freqDerivOrder)
        injData =np.recarray((nInj,), dtype=[(key, '>f8') for key in (self.injParamName+freqParamName[1:])]) 

        for i in range(nInj):
            injData[i]['psi'] = np.random.uniform(-np.pi/4,np.pi/4)
            # alpha is uniformly distributed in [0, 2pi]
            #injData[i]["Alpha"] = np.random.uniform(self.target.alpha-skyUncertainty, self.target.alpha+skyUncertainty)
            # sin(delta) is uniformly distributed in [-1, 1]
            #sinDelta = np.random.uniform(np.sin(self.target.delta-skyUncertainty), np.sin(self.target.delta+skyUncertainty))
            #injData[i]["Delta"] = np.arcsin(sinDelta)
                
            injData[i]["Alpha"], injData[i]["Delta"] = skySampler(self.target.alpha, self.target.delta, skyUncertainty, 1)

            injData[i]["refTime"] = self.refTime
            cosi = np.random.uniform(-1,1)
            _h0 = utils.genh0Points(i, h0, nInj, nAmp) 
            injData[i]["aPlus"] = _h0*(1.+cosi**2)/2.
            injData[i]["aCross"] = _h0*cosi
            
            # draw injection params from defined search range
            f0 = self.getF0FromNonSatBands(nonSatBands, 1) # draw from non saturated bands in 1Hz
            injData[i]['Freq'] = f0
            
            f1min, f1max, _ = fr.f1BroadRange(f0, 0, self.target.tau)
            f1 = np.random.uniform(f1min, f1max, 1)
            injData[i]['f1dot'] = f1
            
            f2min, f2max, _ = fr.f2BroadRange(f0, 0, f1, f1)
            f2 = np.random.uniform(f2min, f2max, 1)
            injData[i]['f2dot'] = f2
            
            if freqDerivOrder >= 3:
                injData[i]['f3dot'] = fr.f3Value(f0, f1, f2)
            
            if freqDerivOrder >= 4:
                injData[i]['f4dot'] = fr.f4Value(f0, f1, f2)
            
        return fits.BinTableHDU(injData)
            
    def genSearchRangeTable(self, spacing, freq, injData, stage, freqDerivOrder):
        freqParamName, freqDerivParamName = utils.phaseParamName(freqDerivOrder)
        nSpacing = setup.followUp_nSpacing
        n = injData.size     

        data =np.recarray((n,), dtype=[(key, '>f8') for key in (freqParamName+freqDerivParamName)]) 
        for i in range(n):            
            # get frequency evolution parameters' spacing 
            taskName = utils.taskName(self.target, stage, self.cohDay, freqDerivOrder, int(freq))

            # avoid the search arange to cross the sub-band bounday and hit the saturated band
            idx1, idx2 = freqParamName[0], freqDerivParamName[0]
            eps = spacing[idx2]
            injf0 =  injData[i][idx1.capitalize()] # Weave use "Freq" not "freq" for injection's f0
            
            data[i][idx1], data[i][idx2] = injf0 - nSpacing*eps, 2*nSpacing*eps
                
            # f1dot
            idx1, idx2 = freqParamName[1], freqDerivParamName[1]
            eps = spacing[idx2]
            injf1 =  injData[i][idx1] 
            data[i][idx1], data[i][idx2] = injf1 - nSpacing*eps, 2*nSpacing*eps            
   
            # f2dot
            idx1, idx2 = freqParamName[2], freqDerivParamName[2]
            eps = spacing[idx2]
            injf2 =  injData[i][idx1] 
            data[i][idx1], data[i][idx2] = injf2 - nSpacing*eps, 2*nSpacing*eps            
                           
            # f3dot
            if freqDerivOrder >= 3:
                idx1, idx2 = freqParamName[2], freqDerivParamName[2]
                eps = spacing[idx2]
                injf3 =  injData[i][idx1] 
                data[i][idx1], data[i][idx2] = injf3 - nSpacing*eps, 2*nSpacing*eps

            # f4dot
            if freqDerivOrder >= 4:
                idx1, idx2 = freqParamName[2], freqDerivParamName[2]
                eps = spacing[idx2]
                injf4 =  injData[i][idx1] # Weave use "Freq" for injection
                data[i][idx1], data[i][idx2] = injf4 - nSpacing*eps, 2*nSpacing*eps

        data = Table(data)
        data.add_column(self.target.alpha*np.ones(n), name='alpha')
        data.add_column(self.target.dalpha*np.ones(n), name='dalpha')
        data.add_column(self.target.delta*np.ones(n), name='delta')
        data.add_column(self.target.ddelta*np.ones(n), name='ddelta')
 
        return fits.BinTableHDU(data)

    def _genParam(self, h0, freq, spacing=None, nonSatBands=None, nInj=1, nAmp=1, injFreqDerivOrder=4, skyUncertainty=0, freqDerivOrder=2, stage='search', cluster=False, workInLocalDir=False):
        if freqDerivOrder > 4:
            print('Error: frequency derivative order larger than 4.')
        if injFreqDerivOrder > 4:
            print('Error: Injection frequency derivative order larger than 4.')
            
        searchParamDict, injParamDict = {}, {}

        ip = self.genInjParamTable(nonSatBands, h0, freq, nInj, nAmp, injFreqDerivOrder, skyUncertainty)
        sp = self.genSearchRangeTable(spacing, freq, ip.data, stage, freqDerivOrder)
        
        injParamDict[str(freq)] = ip
        searchParamDict[str(freq)] = sp
       
        return searchParamDict, injParamDict     



############
        # freqParamName, freqDerivParamName = utils.phaseParamName(freqDerivOrder)
        # nSpacing = setup.followUp_nSpacing
        # n = injData.size

        # _d = fits.getheader(dataFilePath)
        # spacing = {key: _d['HIERARCH ' + key] for key in freqDerivParamName}        


        