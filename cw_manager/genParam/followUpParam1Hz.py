from ..utils import utils as utils
from tqdm import tqdm
import numpy as np
from astropy.io import fits
from astropy.table import Table
from . import frequencyRange as fr
from ..utils import setup_parameter as setup
               
class followUpParams():    
    def __init__(self, target, obsDay, fBand=0.1):
        self.obsDay = obsDay
        self.fBand = fBand
        self.target = target
    
    def makeFollowUpTable(self, data, oldFreqDerivOrder, newFreqDerivOrder, cluster=False, workInLocalDir=False): 
        nSpacing = setup.followUp_nSpacing
        data = Table(data)
        # check if there's outlier 
        if len(data) == 0:
            return fits.BinTableHDU(data=data)        
        
        freqParamName, freqDerivParamName = utils.phaseParamName(oldFreqDerivOrder)
        newFreqParamName, newFreqDerivParamName = utils.phaseParamName(newFreqDerivOrder)
        for _f, _df in zip(freqParamName, freqDerivParamName):
            data[_f] = data[_f] - nSpacing*data[_df] # it may change the sign of the parameter (from -ve to +ve or vice versa), due with it in the next step 
            data[_df] = 2*nSpacing*data[_df] 
            newFreqParamName.remove(_f)
            newFreqDerivParamName.remove(_df)
            
        # new added f3dot - f4dot (didnt search over in the previous stage)
        for _f, _df in zip(newFreqParamName, newFreqDerivParamName):
            if _f == 'f3dot':
                f3min, f3max, f3band = fr.f3BroadRange(
                    data['freq'], data['df'], data['f1dot'], data['f1dot']+data['df1dot'], data['f2dot'], data['f2dot']+data['df2dot']
                )
                data.add_column(f3min, name=_f)
                data.add_column(f3band, name=_df)
                
            if _f == 'f4dot':
                f4min, f4max, f4band = fr.f4BroadRange(
                    data['freq'], data['df'], data['f1dot'], data['f1dot']+data['df1dot'], data['f2dot'], data['f2dot']+data['df2dot']
                )
                data.add_column(f4min, name=_f)
                data.add_column(f4band, name=_df)
                
        return fits.BinTableHDU(data=data)
           
    def genFollowUpParam(self, data, oldFreqDerivOrder, newFreqDerivOrder, cluster=False, workInLocalDir=False): 
        if oldFreqDerivOrder > 4 or newFreqDerivOrder > 4:
            print('Error: frequency derivative order larger than 4.')
             
        params = self.makeFollowUpTable(data, oldFreqDerivOrder, newFreqDerivOrder, cluster, workInLocalDir)
        
        print('Done generation of {0} follow-up parameters\n'.format(self.target.name))
        return params
    
    
