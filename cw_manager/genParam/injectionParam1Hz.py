from ..utils import utils as utils
from tqdm import tqdm
from astropy.io import fits
import numpy as np
from ..utils import setup_parameter as setup
from astropy.io import fits
from . import frequency_range as fr
from ..analysis import readFile as rf
from ..utils import filePath as fp
    

class injectionParams:    
    def __init__(self, target, obsDay, cohDay, fBand=0.1):
        self.target = target
        self.cohDay= cohDay
        _, _, _, _, self.refTime = utils.getTimeSetup(self.target.name, obsDay, cohDay)
        self.fBand = fBand
        self.injParamName = utils.injParamName()
        
    def upperStrainLimit(self, freq, fmin, fmax, nBands, nonSatBands, method='ULEstimation'):
        try:
            if method == 'Injection':
                filePath = fp.sensitivityFilePath(self.target, fmin, fmax, stage='injectionUpperLimit-new')
                _freq, h0, dx = np.loadtxt(filePath).T
                _freq  = _freq + 0.5
                idx = np.argmin((_freq-freq)**2)
                return h0[idx] *(1+ 0.75 * dx[idx]) # injection (new version), previously determined h95 is too low so make use a higher h0 for injection
                # 1+0.5 for G347 (93%) 1+0.75 for CassA, VelaJr
            elif method == 'ULEstimation':
                taskName = 'ULEstimation_{0}Days'.format(self.cohDay)
                h0 = []
                for f in nonSatBands:
                    filePath = fp.estimateUpperLimitFilePath(self.target, f, taskName, method)
                    _h0 = rf.readEstimatedUpperStrainLimit(filePath)
                    h0.append(_h0)
                return np.mean(h0)
        except:
            print("No upper strain limit file, return h0 = 1e-25.")
            return 1e-25
        
    def getF0FromNonSatBands(self, nonSatBands, nInj):
        p = np.ones(nonSatBands.shape)
        p = p/p.sum() # Normalize to sum up to one
        f0 = [np.random.choice([np.random.uniform(freq, freq+self.fBand) for freq in nonSatBands], p=p) for _ in range(nInj)]
        return np.array(f0).reshape(nInj)

    def genInjParamTable(self, nonSatBands, h0, freq, nInj, nAmp, freqDerivOrder):
        freqParamName, freqDerivParamName = utils.phaseParamName(freqDerivOrder)
        injData =np.recarray((nInj,), dtype=[(key, '>f8') for key in (self.injParamName+freqParamName[1:])]) 

        for i in range(nInj):
            injData[i]['psi'] = np.random.uniform(0,np.pi/4.0)
            injData[i]["Alpha"], injData[i]["Delta"] = self.target.alpha, self.target.delta
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
            
    def genSearchRangeTable(self, freq, injData, stage, freqDerivOrder):
        freqParamName, freqDerivParamName = utils.phaseParamName(freqDerivOrder)
        nSpacing = setup.followUp_nSpacing
        n = injData.size
        data =np.recarray((n,), dtype=[(key, '>f8') for key in (freqParamName+freqDerivParamName)]) 
        for i in range(n):
            # sky location
            data[i]['alpha'], data[i]['dalpha'] = self.target.alpha, self.target.dalpha
            data[i]['delta'], data[i]['ddelta'] = self.target.delta, self.target.ddelta
            # get spacing 
            taskName = utils.taskName(self.target, stage, self.cohDay, freqDerivOrder, int(freq))
            dataFilePath = fp.weaveOutputFilePath(self.target, int(freq), taskName, 1, stage)
            spacing = utils.getSpacing(dataFilePath, freqDerivOrder)
                
            # avoid the search arange to cross the sub-band bounday and hit the saturated band
            idx1, idx2 = freqParamName[0], freqDerivParamName[0]
            eps = spacing[idx2]
            injf0 =  injData[i][idx1.capitalize()] # Weave use "Freq" not "freq" for injection's f0
            
            f0_round = np.floor(injf0*10)/10
            f0min, f0max, _ = fr.f0BroadRange(f0_round, self.fBand)
            if (injf0-nSpacing*eps)<f0min:
                data[i][idx1], data[i][idx2] = f0min, 2*nSpacing*eps
            elif (injf0+nSpacing*eps)>f0max:
                data[i][idx1], data[i][idx2] = f0max - 2*nSpacing*eps, 2*nSpacing*eps
            else:
                data[i][idx1], data[i][idx2] = injf0 - nSpacing*eps, 2*nSpacing*eps
                
            # f1dot
            f1min, f1max, _ = fr.f1BroadRange(injf0, 0, self.target.tau)
            idx1, idx2 = freqParamName[1], freqDerivParamName[1]
            eps = spacing[idx2]
            injf1 =  injData[i][idx1] 
            if (injf1 - nSpacing * eps) < f1min:
                data[i][idx1], data[i][idx2] = f1min, 2*nSpacing*eps
            elif (injf1 + nSpacing * eps) > f1max:
                data[i][idx1], data[i][idx2] = f1max - 2*nSpacing*eps, 2*nSpacing*eps
            else:
                data[i][idx1], data[i][idx2] = injf1 - nSpacing*eps, 2*nSpacing*eps
                
            # f2dot
            f2min, f2max, _ = fr.f2BroadRange(injf0, 0, injf1, injf1)
            idx1, idx2 = freqParamName[2], freqDerivParamName[2]
            eps = spacing[idx2]
            injf2 =  injData[i][idx1] 
            if (injf2 - nSpacing*eps)<f2min:
                data[i][idx1], data[i][idx2] = f2min, 2*nSpacing*eps
            elif (injf2 + nSpacing*eps)>f2max:
                data[i][idx1], data[i][idx2] = f2max - 2*nSpacing*eps, 2*nSpacing*eps
            else:
                data[i][idx1], data[i][idx2] = injf2 - nSpacing*eps, 2*nSpacing*eps
                                        
            # f3dot
            if freqDerivOrder >= 3:
                f3min, f3max, _ = fr.f3BroadRange(injf0, 0, injf1, injf1, injf2, injf2)
                idx1, idx2 = freqParamName[2], freqDerivParamName[2]
                eps = spacing[idx2]
                injf3 =  injData[i][idx1] 
                if (injf3 - nSpacing*eps)<f3min:
                    data[i][idx1], data[i][idx2] = f3min, 2*nSpacing*eps
                elif (injf3 + nSpacing*eps)>f3max:
                    data[i][idx1], data[i][idx2] = f3max - 2*nSpacing*eps, 2*nSpacing*eps
                else:
                    data[i][idx1], data[i][idx2] = injf3 - nSpacing*eps, 2*nSpacing*eps

            # f4dot
            if freqDerivOrder >= 4:
                f4min, f4max, _ = fr.f4BroadRange(injf0, 0, injf1, injf1, injf2, injf2)
                idx1, idx2 = freqParamName[2], freqDerivParamName[2]
                eps = spacing[idx2]
                injf4 =  injData[i][idx1] # Weave use "Freq" for injection
                if (injf4 - nSpacing*eps)<f4min:
                    data[i][idx1], data[i][idx2] = f4min, 2*nSpacing*eps
                elif (injf4 + nSpacing*eps)>f4max:
                    data[i][idx1], data[i][idx2] = f4max - 2*nSpacing*eps, 2*nSpacing*eps
                else:
                    data[i][idx1], data[i][idx2] = injf4 - nSpacing*eps, 2*nSpacing*eps

        return fits.BinTableHDU(data)
    
    def saveh0Value(self, injDict, fmin=20, fmax=475, nInj=1, nAmp=8):
        filePath = fp.h0_FilePath(self.target, fmin, fmax, stage='injectionUpperLimit')
        freqList = [f for f in injDict.keys()]
        injPerPoint = int(nInj/nAmp)
        utils.makeDir([filePath])
        with open(filePath, 'wt') as file:
            file.write('#{0}'.format('freq'))
            for i in range(nAmp):
                file.write('\t{0}'.format(i))
            file.write('\n')
            
            for freq in freqList:
                file.write('{0}'.format(freq))
                for i in range(nAmp):
                    aplus = injDict[str(freq)].data[i*injPerPoint]['aPlus']
                    across = injDict[str(freq)].data[i*injPerPoint]['aCross']
                    h0 = 0.5*(2.*aplus+2.*np.sqrt(aplus**2-across**2) )
                    file.write('\t{0}'.format(h0))
                file.write('\n')
        return 0

    def genParam(self, freq, nBands=None, nInj=1, nAmp=1, injFreqDerivOrder=4, freqDerivOrder=2, stage='search', fmin=20, fmax=475):
        if freqDerivOrder > 4:
            print('Error: frequency derivative order larger than 4.')
        if injFreqDerivOrder > 4:
            print('Error: Injection frequency derivative order larger than 4.')
            
        searchParamDict, injParamDict = {}, {}
        nonSatBandsList = utils.loadNonSaturatedBand(self.target, fmin, fmax, nBands)
        
        nonSatBands = nonSatBandsList[(nonSatBandsList>=freq)*(nonSatBandsList<(freq+1.0))]
        if nAmp != 1:
            h0 = self.upperStrainLimit(freq, fmin, fmax, nBands, nonSatBands, method='ULEstimation')
        else:
            h0 = self.upperStrainLimit(freq, fmin, fmax, nBands, nonSatBands, method='Injection')

        ip = self.genInjParamTable(nonSatBands, h0, freq, nInj, nAmp, injFreqDerivOrder)
        sp = self.genSearchRangeTable(freq, ip.data, stage, freqDerivOrder)

        injParamDict[str(freq)] = ip
        searchParamDict[str(freq)] = sp
        if nAmp !=1:    
            self.saveh0Value(injParamDict, fmin, fmax, nInj, nAmp)
        
        return searchParamDict, injParamDict        
"""
    def genInjParamTable(self, nonSatBands, h0, freq, nInj, nAmp, freqDerivOrder):
        freqParamName, freqDerivParamName = utils.phaseParamName(freqDerivOrder)
        injData =np.recarray((nInj,), dtype=[(key, '>f8') for key in (self.injParamName+freqParamName[1:])]) 
        
        injData['psi'] = np.random.uniform(0, np.pi/4.0, nInj)
        injData["Alpha"], injData["Delta"] = np.ones(nInj)*self.target.alpha, np.ones(nInj)*self.target.delta
        injData["refTime"] = np.ones(nInj)*self.refTime
        
        cosi = np.random.uniform(-1,1, nInj)
        h0 = utils.genh0Points(range(nInj), h0, nInj, nAmp) 
        injData["aPlus"] = h0*(1.+cosi**2)/2.
        injData["aCross"] = h0*cosi
        
        # draw injection params from defined search range
        f0 = self.getF0FromNonSatBands(nonSatBands, nInj) # draw from non saturated bands in 1Hz
        injData['Freq'] = f0
        
        f1min, f1max, _ = fr.f1BroadRange(f0, 0, self.target.tau)
        f1 = np.random.uniform(f1min, f1max)
        injData['f1dot'] = f1    
        
        f2min, f2max, _ = fr.f2BroadRange(f0, 0, f1, f1)
        f2 = np.random.uniform(f2min, f2max)
        injData['f2dot'] = f2
            
        if freqDerivOrder >= 3:
            injData['f3dot'] = fr.f3Value(f0, f1, f2)
            
        if freqDerivOrder >= 4:
            injData['f4dot'] = fr.f4Value(f0, f1, f2)
                
        return fits.BinTableHDU(injData)

"""
