from . import readFile as rf
from . import tools as tools
from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np
from ..utils import filePath as fp
from ..utils import setup_parameter as setup
from ..genParam import frequencyRange as fr
from tqdm import tqdm
from ..utils import utils as utils
from pathlib import Path
import warnings
    
class resultManager():
    def __init__(self, target, obsDay):
        self.obsDay = obsDay
        self.setup = setup
        self.target = target
        
    # function to read template count from weave output in each 1Hz band
    def _readTemplateCount(self, cohDay, freq, nJobs, stage='search', freqDerivOrder=2):
        """
        Reads the template count from the weave output for each job in a specified frequency band.

        Parameters:
        - cohDay: int
            The number of coherent observation days for the search.

        - freq: float
            The frequency in Hz for which the template count is being read.

        - nJobs: int
            The number of jobs that were processed.

        - stage: str, optional
            The current stage of the analysis (default is 'search').

        - freqDerivOrder: int, optional
            The order of the frequency derivative used in the analysis (default is 2).

        Returns:
        - templateList: list
            A list of template counts read from the output files for each job.
        """
        templateList = []
        taskName = utils.taskName(self.target, stage, cohDay, freqDerivOrder, freq)
        crfiles = fp.condorRecordFilePath(freq, self.target, taskName, stage)
        for jobIndex in range(1, nJobs+1):
            outFilePath = crfiles[0][:-8] + '{0}'.format(jobIndex)
            templateList.append(rf.readTemplateCount(outFilePath))
        return templateList
        
    def calMean2F_threshold(self, cohDay, freq, nJobs):
        """
        Parameters:
        - cohDay: int
            The number of coherent observation days for the search, which affects the threshold calculation.

        - freq: float
            The frequency for which the threshold is being calculated. This parameter influences the number of templates and possibly the sensitivity.

        - nJobs: int
            Number of jobs over which the work is split for this frequency, affecting the threshold as each job has its own portion of templates to search.
        """
        # Get the number of templates based on coherence time, frequency, and job count
        nTemp = self._readTemplateCount(cohDay, freq, nJobs)
        # Calculate the mean 2F threshold using the total templates and segment count
        mean2F_th = tools.mean2F_threshold(sum(nTemp), self.nSeg)            
        return mean2F_th

    def makeOutlierTable(self, data, spacing, mean2F_th, toplistLimit=1000, freqDerivOrder=2):    
        """
        Parameters:
        - data: table
            BinHDUTable containing data on mean 2F values. The function reads this file to retrieve data on outliers.

        - mean2F_th: float
            The threshold value for the mean 2F statistic, which filters outliers by including only those above this threshold.

        - toplistLimit: int, optional (default=1000)
            Maximum number of top outliers to be returned, limiting the data size to manage memory and processing load.

        - freqDerivOrder: int, optional (default=2)
            Specifies the frequency derivative order, which determines the parameters for spacing and phase calculations used in the output table.
        """
        
        # Read and limit the data to the top entries
        data = data[:toplistLimit]
        # Mask data with mean 2F values greater than the threshold
        mask = data['mean2F'] > mean2F_th
        data = Table(data[mask])       
        data.add_column(mean2F_th*np.ones(len(data)), name='mean2F threshold')
    
        # Get frequency spacing based on the frequency derivative order   
        #spacing = utils.getSpacing(dataFilePath, freqDerivOrder)
        _, name = utils.phaseParamName(freqDerivOrder)
        #print(spacing) 
        # Add spacing parameters as columns in the table
        for i in range(len(name)):
            data.add_column(spacing[name[i]]*np.ones(len(data)), name=name[i]) 
        return data

    # Generates a table of injection data from a FITS file and matches it with search results
    def makeInjectionTable(self, injParam, searchParam, freqDerivOrder):   
        """
        Parameters:
        - injParam: Table
            BinHDUTable containing injection data. The function reads the injection data from this file to analyze and compare it with search results.

        - searchParam: Table
            A table of search results that is used to find matching entries for each injection.

        - freqDerivOrder: int
            The order of frequency derivative to consider (e.g., 1 for first derivative, 2 for second derivative).
            This parameter is used to determine which frequency derivative columns (like df1, df2) are extracted and matched to the injection data.
        """
        injParam = Table(injParam)   
        
        # Calculate h0 from aPlus and aCross, adding it as a new column
        aplus, across = injParam['aPlus'], injParam['aCross']
        h0 = 0.5*(2.*aplus+2.*np.sqrt(aplus**2-across**2) )
        injParam.add_column(h0*np.ones(len(injParam)), name='h0')
        
        # Rename the reference time column for consistency
        injParam.rename_column('refTime_s', 'refTime')   
        
        # Get names for the frequency derivative parameters based on the order
        fn, dfn = utils.phaseParamName(freqDerivOrder)
          
        # Mask for identifying injections that match within a specific frequency range
        mask = np.full(searchParam['freq'].shape, True)
        searchParam = Table(searchParam[mask])[:1] # only follow up the loudest one which covering the injection to save the cost 

        return searchParam, injParam
    
    # Write results from each 1Hz frequency band of the search stage output
    def _writeSearchResult(self, cohDay, freq, mean2F_th, nJobs, numTopListLimit=1000, stage='search', freqDerivOrder=2, cluster=False, workInLocalDir=False):
        """
        Parameters:
        - cohDay: int
            The number of coherent observation days for the search, used in time setup and threshold calculations.

        - freq: int
            The frequency value for the 1Hz band being processed in this function.

        - mean2F_th: float
            The threshold value of the mean 2F statistic, which determines whether an outlier qualifies for follow-up or further analysis.

        - nJobs: int
            Number of jobs to split the work into. Each job handles a portion of the calculations for this frequency band.

        - numTopListLimit: int, optional (default=1000)
            Maximum number of top outliers to be included in the result for each job. This helps manage the computational and memory limits.

        - stage: str, optional (default='search')
            The stage of the analysis, usually 'search' or 'follow-up'. This determines the task naming conventions used when writing output files.

        - freqDerivOrder: int, optional (default=2)
            The order of frequency derivative to be used in the clustering and spacing calculations. It decides which derivatives (like df, df1dot) are considered.

        - cluster: bool, optional (default=False)
            If True, perform clustering on the outliers to consolidate similar results, saving space and computational effort.

        - workInLocalDir: bool, optional (default=False)
            If True, writes output to the local directory rather than the default path. This may be used for testing or troubleshooting.
        """      
        
        # Generate the task name for organizing results
        taskName = utils.taskName(self.target, stage, cohDay, freqDerivOrder, freq)
         
        # Initialize lists to collect outlier tables and data on job completion status
        outlierTableList = []
        info_data = np.recarray((nJobs,), dtype=[(key, '>f8') for key in ['freq', 'jobIndex', 'outliers', 'saturated']]) 

        # Loop over each job to process results
        for i, jobIndex in enumerate(tqdm(range(1, nJobs+1))):
            # Generate file path for each job's result, adjusting if working in local directory
            weaveFilePath = fp.weaveOutputFilePath(self.target, freq, taskName, jobIndex, stage)
            if workInLocalDir:
                weaveFilePath = Path(weaveFilePath).name
                
            weave_data = fits.getdata(weaveFilePath, 1)
            spacing = utils.getSpacing(weaveFilePath, freqDerivOrder)
            # Generate outlier table for the job and assess if it reached the limit
            outlier = self.makeOutlierTable(weave_data, spacing, mean2F_th, numTopListLimit, freqDerivOrder)  

            if len(outlier) == numTopListLimit:
                info_data[i] = freq, jobIndex, 0, 1  # Job saturated if top limit is reached
            else:
                info_data[i] = freq, jobIndex, len(outlier), 0
                outlierTableList.append( outlier )
        
        # Calculate bands that aren't saturated 
        sat = info_data['saturated'].reshape(10, int(nJobs/10)).sum(axis=1)
        idx = np.where(sat == 0)[0] 
        nonSatBand = np.recarray((idx.size,), dtype=[(key, '>f8') for key in ['nonSatBand']])
        nonSatBand['nonSatBand'] = freq + idx * setup.fBand
           
        # Set up a FITS file with outliers, non-saturated bands, and search settings
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th
        primary_hdu.header['HIERARCH cluster_nSpacing'] = ''
        # Write parameter spacing values into header
        for name, value in spacing.items():
            primary_hdu.header['HIERARCH {}'.format(name)] = value
        
        # Create table HDUs for outliers, job information, and non-saturated bands
        outlier_hdu =  fits.BinTableHDU(data=vstack(outlierTableList), name=stage+'_outlier')
        info_hdu =  fits.BinTableHDU(data=info_data, name='info') 
        nsb_hdu =  fits.BinTableHDU(data=nonSatBand, name='nonSatBand')

        # Compile all HDUs into a FITS HDU list and write to a specified file path
        outlier_hdul = fits.HDUList([primary_hdu, outlier_hdu, info_hdu, nsb_hdu])
        outlierFilePath = fp.outlierFilePath(self.target, freq, taskName, stage, cluster=False)
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name
        utils.makeDir([outlierFilePath])
        outlier_hdul.writeto(outlierFilePath, overwrite=True)  
       
        # Perform clustering if requested and save results in another FITS file
        if cluster:
            if outlier_hdu.data.size > 1:
                cluster_hdul = fits.HDUList()
                
                primary_hdu = fits.PrimaryHDU()
                primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th
                primary_hdu.header['HIERARCH cluster_nSpacing'] = setup.cluster_nSpacing
                # Write parameter spacing values into header
                for name, value in spacing.items():
                    primary_hdu.header['HIERARCH {}'.format(name)] = value 
       
                centers_idx, cluster_size, _ = utils.clustering(outlier_hdu.data, freqDerivOrder) 
                cluster_data = outlier_hdu.data[centers_idx]
                cluster_hdu = fits.BinTableHDU(data=cluster_data, name=stage+'_outlier')
     
                info_data = np.recarray((cluster_size.size,), dtype=[(key, '>f8') for key in ['freq', 'clusterIndex', 'noOutliersWithin']]) 
                for i in range(cluster_size.size):
                    info_data[i] = freq, i, cluster_size[i]
                
                info_hdu =  fits.BinTableHDU(data=info_data, name='info')
                cluster_hdul.append(primary_hdu)
                cluster_hdul.append(cluster_hdu)
                cluster_hdul.append(info_hdu)
                cluster_hdul.append(nsb_hdu)
            else:
                cluster_hdul = outlier_hdul
                
            # Write clustered data to a file
            outlierClusterFilePath = fp.outlierFilePath(self.target, freq, taskName, stage, cluster=cluster)
            if workInLocalDir:
                outlierClusterFilePath = Path(outlierClusterFilePath).name
            cluster_hdul.writeto(outlierClusterFilePath, overwrite=True)
            
        if cluster:
            return outlierClusterFilePath
        else:
            return outlierFilePath 
      
    
# # Set up time and frequency parameters for the search based on target and observation day
# self.cohDay, self.cohTime, self.nSeg, self.obsTime, self.refTime = utils.getTimeSetup(self.target.name, self.obsDay, cohDay)    

# # Calculate the number of jobs required based on frequency derivatives and bandwidth
# nf1dots = fr.getNf1dot(freq, self.setup.fBand, self.target.tau, df1dot=df1dot)
# nf2dots = fr.getNf2dot(freq, self.setup.fBand, self.target.tau, df2dot=df2dot)
# nJobs = int(nf1dots*nf2dots/self.setup.fBand)

# # Determine or calculate the mean 2F threshold value

# print('calculating mean2F threshold...')
# mean2F_th = self.calMean2F_threshold(cohDay, freq, nJobs)           
# print('mean2F threshold = ', mean2F_th)

# # Write search results for the specified frequency
# outlierFilePath = self._writeSearchResult(freq, mean2F_th, nJobs, numTopList, stage, freqDerivOrder, cluster, workInLocalDir)
# print('Finish writing search result for {0} Hz'.format(freq))
# return outlierFilePath

   
    # Workflow for writing search results across a frequency range (fmin, fmax)
    def writeSearchResult(self, cohDay, freq, mean2F_th, numTopList=1000, stage='search', freqDerivOrder=2, cluster=False, workInLocalDir=False):
        """
        Parameters:
        - cohDay: int
            The number of coherent observation days for the search, used in time setup and threshold calculations.

        - freq: int
            The frequency value for the 1Hz band being processed.
            
        - mean2F_th: float
            The threshold value of the mean 2F statistic, which determines whether an outlier qualifies for follow-up or further analysis.

        - numTopList: int, optional (default=1000)
            Maximum number of top outliers to keep for each job's results.

        - stage: str, optional (default='search')
            The stage of the analysis. Determines the naming and organizational conventions for output files.

        - freqDerivOrder: int, optional (default=2)
            Specifies the order of frequency derivatives to consider (e.g., df1dot, df2dot) when calculating threshold and creating results.

        - cluster: bool, optional (default=False)
            If True, clusters outliers to consolidate similar results, saving computational costs and storage.

        - workInLocalDir: bool, optional (default=False)
            If True, stores output files in the local directory. This option might be useful for local testing.
        """ 
        
        # Write search results for the specified frequency
        outlierFilePath = self._writeSearchResult(cohDay, freq, mean2F_th, nJobs, numTopList, stage, freqDerivOrder, cluster, workInLocalDir)
        print('Finish writing search result for {0} Hz'.format(freq))
        return outlierFilePath
    
    
    
    # Write results from each 1Hz frequency band of the search stage output
    def _writeSearchResultFromSaturatedBand(self, cohDay, freq, mean2F_th, jobIndex, numTopListLimit=1, stage='search', freqDerivOrder=2, workInLocalDir=False):
        """
        Parameters:
        - cohDay: int
            The number of coherent observation days for the search, used in time setup and threshold calculations.

        - freq: int
            The frequency value for the 1Hz band being processed in this function.

        - mean2F_th: float
            The threshold value of the mean 2F statistic, which determines whether an outlier qualifies for follow-up or further analysis.

        - jobIndex: int array
            Array of indices identifying which jobs within the 1Hz band are saturated, used for tracking and organizing job-specific results.

        - nJobs: int
            Number of jobs to split the work into. Each job handles a portion of the calculations for this frequency band.

        - numTopListLimit: int, optional (default=1000)
            Maximum number of top outliers to be included in the result for each job. This helps manage the computational and memory limits.

        - stage: str, optional (default='search')
            The stage of the analysis, usually 'search' or 'follow-up'. This determines the task naming conventions used when writing output files.

        - freqDerivOrder: int, optional (default=2)
            The order of frequency derivative to be used in the clustering and spacing calculations. It decides which derivatives (like df, df1dot) are considered.

        - cluster: bool, optional (default=False)
            If True, perform clustering on the outliers to consolidate similar results, saving space and computational effort.

        - workInLocalDir: bool, optional (default=False)
            If True, writes output to the local directory rather than the default path. This may be used for testing or troubleshooting.
        """      
        
        # Generate the task name for organizing results
        taskName = utils.taskName(self.target, stage, cohDay, freqDerivOrder, freq)
         
        # Initialize lists to collect outlier tables and data on job completion status
        outlierTableList = []
    
        # Loop over each job to process results
        for idx in tqdm(jobIndex):
            # Generate file path for each job's result, adjusting if working in local directory
            weaveFilePath = fp.weaveOutputFilePath(self.target, freq, taskName, idx, stage)
            if workInLocalDir:
                weaveFilePath = Path(weaveFilePath).name
                
            weave_data = fits.getdata(weaveFilePath, 1)
            spacing = utils.getSpacing(weaveFilePath, freqDerivOrder)
            # Generate outlier table for the job and assess if it reached the limit
            outlier = self.makeOutlierTable(weave_data, spacing, mean2F_th, numTopListLimit, freqDerivOrder)  
            outlierTableList.append( outlier )
           
        # Set up a FITS file with outliers, non-saturated bands, and search settings
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th
        primary_hdu.header['HIERARCH cluster_nSpacing'] = ''
        

        # Create table HDUs for outliers, job information, and non-saturated bands
        if len(outlierTableList) == 0:
            outlier_hdu =  fits.BinTableHDU(name=stage+'SatBand_outlier')
        else:
            outlier_hdu =  fits.BinTableHDU(data=vstack(outlierTableList), name=stage+'SatBand_outlier')
        
        # Compile all HDUs into a FITS HDU list and write to a specified file path
        outlier_hdul = fits.HDUList([primary_hdu, outlier_hdu])
        taskName = utils.taskName(self.target, stage+'SatBand', cohDay, freqDerivOrder, freq)
        outlierFilePath = fp.outlierFilePath(self.target, freq, taskName, stage, cluster=False)
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name
        utils.makeDir([outlierFilePath])
        outlier_hdul.writeto(outlierFilePath, overwrite=True)  
       
        return outlierFilePath 
   
    # Workflow for writing search results across a frequency range (fmin, fmax)
    def writeSearchResultFromSaturatedBand(self, cohDay, freq, mean2F_th, jobIndex, numTopList=1, stage='search', freqDerivOrder=2, workInLocalDir=False):
        """
        Parameters:
        - cohDay: int
            The number of coherent observation days for the search, used in time setup and threshold calculations.

        - freq: int
            The frequency value for the 1Hz band being processed.
        
        - mean2F_th: float
            The threshold value of the mean 2F statistic, which determines whether an outlier qualifies for follow-up or further analysis.
            
        - jobIndex: int array
            Array of indices identifying which jobs within the 1Hz band are saturated, used for tracking and organizing job-specific results.

        - numTopList: int, optional (default=1)
            Maximum number of top outliers to keep for each job's results.

        - stage: str, optional (default='search')
            The stage of the analysis. Determines the naming and organizational conventions for output files.

        - freqDerivOrder: int, optional (default=2)
            Specifies the order of frequency derivatives to consider (e.g., df1dot, df2dot) when calculating threshold and creating results.

        - workInLocalDir: bool, optional (default=False)
            If True, stores output files in the local directory. This option might be useful for local testing.
        """ 
        
        # Write search results for the specified frequency
        outlierFilePath = self._writeSearchResultFromSaturatedBand(cohDay, freq, mean2F_th, jobIndex, numTopList, stage, freqDerivOrder, workInLocalDir)
        print('Finish writing search result for {0} Hz'.format(freq))
        return outlierFilePath

    # function to write result from weave output in each 1Hz band
    def _writeInjectionResult(self, cohDay, freq, mean2F_th, nJobs, numTopListLimit=1000, stage='search', freqDerivOrder=2, workInLocalDir=False, cluster=False):
        """
        Writes the injection results from the weave output for a given frequency.

        Parameters:
        - cohDay: int
            The number of coherent observation days for the search, used in time setup and threshold calculations.

        - freq: int
            The frequency band in Hz for which results are being written.

        - mean2F_th: float
            The mean 2F threshold value used for identifying outliers.

        - nJobs: int
            The number of jobs that were run for this frequency.

        - numTopList: int, optional (default=1000)
            Maximum number of top outliers to keep for each job's results.

        - stage: str, optional (default='search')
            The stage of the analysis. Determines the naming and organizational conventions for output files.

        - freqDerivOrder: int, optional (default=2)
            Specifies the order of frequency derivatives to consider (e.g., df1dot, df2dot) when calculating threshold and creating results.

        - workInLocalDir: bool, optional 
            If True, work with local directory paths. Default is False.

        - cluster: bool, optional
            If True, indicate that the results should be stored for clustering. Default is False.

        Returns:
        - outlierFilePath: str
            The path to the output file containing the results.
        """
        taskName = utils.taskName(self.target, stage, cohDay, freqDerivOrder, freq)
        outlierTableList = []
        injTableList = []
        info_data =np.recarray((nJobs,), dtype=[(key, '>f8') for key in ['freq', 'jobIndex', 'outliers']]) 
  
        weaveFilePath = fp.weaveOutputFilePath(self.target, freq, taskName, 1, stage)
        if workInLocalDir:
                weaveFilePath = Path(weaveFilePath).name
                
        for i, jobIndex in enumerate(range(1, nJobs+1)):
            weaveFilePath = fp.weaveOutputFilePath(self.target, freq, taskName, jobIndex, stage)
            if workInLocalDir:
                weaveFilePath = Path(weaveFilePath).name
            
            weave_data = fits.getdata(weaveFilePath, 1)
            spacing = utils.getSpacing(weaveFilePath, freqDerivOrder)
            outlier = self.makeOutlierTable(weave_data, spacing, mean2F_th, numTopListLimit, freqDerivOrder)  
            injParam = fits.getdata(weaveFilePath, 2)
            outlier, injParam = self.makeInjectionTable(injParam, outlier, freqDerivOrder)
            
            if len(outlier) == 0:
                outlierTableList.append( outlier )
            else:
                outlierTableList.append( outlier )
                injTableList.append( injParam )
            info_data[i] = freq, jobIndex, len(outlier)  

        # append all tables in the file into one
        # Create a PrimaryHDU object
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th
        primary_hdu.header['HIERARCH cluster_nSpacing'] = ''
        
        outlier_hdu =  fits.BinTableHDU(data=vstack(outlierTableList), name=stage+'_outlier')
        info_hdu =  fits.BinTableHDU(data=info_data, name='info') 
      
        if len(injTableList) != 0:
            inj_hdu =  fits.BinTableHDU(data=vstack(injTableList), name='inj')
        else:
            inj_hdu =  fits.BinTableHDU(name='inj')
            print('No outlier.')
        
        outlier_hdul = fits.HDUList()
        outlier_hdul.append(primary_hdu)
        outlier_hdul.append(outlier_hdu)
        outlier_hdul.append(inj_hdu) 
        outlier_hdul.append(info_hdu)
        
        outlierFilePath = fp.outlierFilePath(self.target, freq, taskName, stage, cluster=False)
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name
        utils.makeDir([outlierFilePath])
        outlier_hdul.writeto(outlierFilePath, overwrite=True) 
                
        if cluster:
            if outlier_hdu.data.size > 1:
                cluster_hdul = fits.HDUList()
                
                primary_hdu = fits.PrimaryHDU()
                primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th
                primary_hdu.header['HIERARCH cluster_nSpacing'] = setup.cluster_nSpacing
                
                centers_idx, cluster_size, cluster_member = utils.clustering(outlier_hdu.data, freqDerivOrder) 
                center_idx_for_each_outlier = np.full(outlier_hdu.data.size,-1)
                processed_indices = set()
                for ci, members in zip(centers_idx, cluster_member):
                    idx = np.array([item for item in members if item not in processed_indices])
                    center_idx_for_each_outlier[idx] = ci
                    processed_indices.update(members)

                cluster_data = outlier_hdu.data[center_idx_for_each_outlier]
                cluster_hdu = fits.BinTableHDU(data=cluster_data, name=stage+'_outlier')

                info_data = np.recarray((cluster_size.size,), dtype=[(key, '>f8') for key in ['freq', 'clusterIndex', 'noOutliersWithin']]) 
                for i in range(cluster_size.size):
                    info_data[i] = freq, i, cluster_size[i]
                info_hdu =  fits.BinTableHDU(data=info_data, name='info')

                cluster_hdul = fits.HDUList()
                cluster_hdul.append(primary_hdu)
                cluster_hdul.append(cluster_hdu)
                cluster_hdul.append(inj_hdu)
                cluster_hdul.append(info_hdu)
            else:
                cluster_hdul = outlier_hdu    
            
            outlierFilePath = fp.outlierFilePath(self.target, freq, taskName, stage, cluster=cluster)

            if workInLocalDir:
                outlierFilePath = Path(outlierFilePath).name
            cluster_hdul.writeto(outlierFilePath, overwrite=True)
            
        return outlierFilePath 

    #work flow to write injection-search result in 1Hz band
    def writeInjectionResult(self, cohDay, freq, mean2F_th, nJobs, numTopList=1000, stage='search', freqDerivOrder=2, workInLocalDir=False, cluster=False):
        """
        Writes the injection results for a specified frequency in the injection-search workflow.

        Parameters:
        - cohDay: int
            The number of coherent observation days for the search, used in time setup and threshold calculations.

        - freq: float
            The frequency in Hz for which the injection results are being written.

        - mean2F_th: float
            The threshold of detection statistic for being a outlier.

        - nJobs: int
            The number of jobs in the 1Hz band being processed.

        - numTopList: int, optional
            The limit on the number of top results to return (default is 1000).

        - stage: str, optional
            The current stage of the analysis (default is 'search').

        - freqDerivOrder: int, optional
            The order of the frequency derivative used in the analysis (default is 2).

        - workInLocalDir: bool, optional
            If True, indicates that paths should be treated as local directory paths (default is False).

        - cluster: bool, optional
            If True, indicates that clustering results should be included in the output (default is False).

        Returns:
        - outlierFilePath: str
            The path to the output file containing the injection results.
        """
               
        outlierFilePath = self._writeInjectionResult(cohDay, freq, mean2F_th, nJobs, numTopList, stage, freqDerivOrder, workInLocalDir, cluster)
        print('Finish writing injection result for {0} Hz'.format(freq))
        return outlierFilePath

    def _writeFollowUpResult(self, cohDay, freq, mean2F_th, nJobs, numTopListLimit=1000, stage='search', freqDerivOrder=2, 
                                   workInLocalDir=True, inj=False, cluster=False,
                                   chunk_index=0, chunk_size=1, n_skygrid=1):
        """
        Writes the follow-up results for injections at a given frequency.

        Parameters:
        - cohDay: int
            The coherence day for the analysis.

        - freq: float
            The frequency in Hz for which follow-up results are being written.

        - mean2F_th: numpy.ndarray
            The mean 2F threshold values used for identifying outliers.

        - nJobs: int
            The number of jobs that were run for this frequency.

        - numTopListLimit: int, optional
            The maximum number of top results to consider. Default is 1000.

        - stage: str, optional
            The stage of the analysis (e.g., 'search'). Default is 'search'.

        - freqDerivOrder: int, optional
            The order of frequency derivative used in the analysis. Default is 2.

        - workInLocalDir: bool, optional
            If True, work with local directory paths. Default is True.

        - inj: bool, optional
            If True, includes injections in the follow-up result. Default is False.

        - cluster: bool, optional
            If True, indicates that results should be stored for clustering. Default is False.

        - chunk_index: int, optional
            The index of the current chunk being processed. Default is 0.

        - chunk_size: int, optional
            The size of the chunks for processing. Default is 1.

        Returns:
        - outlierFilePath: str
            The path to the output file containing the follow-up results.
        """
        
        taskName = utils.taskName(self.target, stage, cohDay, freqDerivOrder, freq)
    
        outlierTableList = []
        injTableList = []
        info_data = np.recarray((nJobs,), dtype=[(key, '>f8') for key in ['freq', 'jobIndex', 'outliers']]) 
 
        weaveFilePath = fp.weaveOutputFilePath(self.target, freq, taskName, 1, stage)
        if workInLocalDir:
            weaveFilePath = Path(weaveFilePath).name
            
        # Iterate over each job to gather follow-up results
        for i, jobIndex in enumerate(range(chunk_index*chunk_size+1, chunk_index*chunk_size+nJobs+1)):

            max_mean2F = -float('inf')  # Initialize to negative infinity for comparison
            outlier, injParam = [], []

            for j in range(n_skygrid if inj else 1):  # n_skygrid if inj=True, else 1
                weaveFilePath = fp.weaveOutputFilePath(self.target, freq, taskName, j*nJobs+jobIndex, stage)
                if workInLocalDir:
                    weaveFilePath = Path(weaveFilePath).name 
                # Create outlier table from the weave output
                weave_data = fits.getdata(weaveFilePath, 1)
                spacing = utils.getSpacing(weaveFilePath, freqDerivOrder)
                outlier = self.makeOutlierTable(weave_data, spacing, mean2F_th[i], numTopListLimit, freqDerivOrder)
                # If injections are considered, create an injection table as well
                if inj:
                    _injParam = fits.getdata(weaveFilePath, 2)
                    _outlier, _injParam = self.makeInjectionTable(_injParam, _outlier, freqDerivOrder)

                    if len(_outlier) > 0 and _outlier['mean2F'][0] > max_mean2F:
                        outlier = _outlier
                        injParam = _injParam
                        max_mean2F = _outlier['mean2F'][0]
              
            # Append results to the respective lists
            outlierTableList.append( outlier )
            if inj:
                injTableList.append( injParam )
                    
            info_data[i] = freq, jobIndex, len(outlier)  

        # append all tables in the file into one
        # Create a PrimaryHDU object
        primary_hdu = fits.PrimaryHDU()
               
        if len(outlierTableList) !=0:
            outlier_hdu =  fits.BinTableHDU(data=vstack(outlierTableList), name=stage+'_outlier')
        else:
            outlier_hdu =  fits.BinTableHDU(name=stage+'_outlier')
            print('No outlier.')
        
        # if software injection is included 
        if inj:
            if len(injTableList) != 0:
                inj_hdu =  fits.BinTableHDU(data=vstack(injTableList), name='inj')
            else:
                inj_hdu =  fits.BinTableHDU(name='inj')
        
        # summary information of the outliers
        info_hdu =  fits.BinTableHDU(data=info_data, name='info')
        
        if inj:
            outlierFilePath = fp.outlierFilePath(self.target, freq, taskName, stage, cluster=False)
        else:
            outlierFilePath = fp.outlierFilePath(self.target, freq, taskName, stage, cluster=cluster)
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name
        if chunk_size !=1:
            outlierFilePath = outlierFilePath[:-4] + '_chunk{}.fts'.format(chunk_index)
        utils.makeDir([outlierFilePath])
        
        outlier_hdul = fits.HDUList()
        outlier_hdul.append(primary_hdu)
        outlier_hdul.append(outlier_hdu)
        if inj:
            outlier_hdul.append(inj_hdu)
        outlier_hdul.append(info_hdu)
        outlier_hdul.writeto(outlierFilePath, overwrite=True)     
        
        if cluster:
            if outlier_hdu.data.size > 1:
                cluster_hdul = fits.HDUList()
                centers_idx, cluster_size, cluster_member = utils.clustering(outlier_hdu.data, freqDerivOrder) 
                if inj:
                    center_idx_for_each_outlier = np.full(outlier_hdu.data.size,-1)
                    processed_indices = set()
                    for ci, members in zip(centers_idx, cluster_member):
                        idx = np.array([item for item in members if item not in processed_indices])
                        center_idx_for_each_outlier[idx] = ci
                        processed_indices.update(members)

                    cluster_data = outlier_hdu.data[center_idx_for_each_outlier]
                    cluster_hdu = fits.BinTableHDU(data=cluster_data, name=stage+'_outlier')
                else:
                    cluster_data = outlier_hdu.data[centers_idx]
                    cluster_hdu = fits.BinTableHDU(data=cluster_data, name=stage+'_outlier')

                info_data = np.recarray((cluster_size.size,), dtype=[(key, '>f8') for key in ['freq', 'chunkIndex', 'clusterIndex', 'noOutliersWithin']]) 
                for i in range(cluster_size.size):
                    info_data[i] = freq, chunk_index, i, cluster_size[i]
                info_hdu =  fits.BinTableHDU(data=info_data, name='info_clustered')

                cluster_hdul = fits.HDUList()
                cluster_hdul.append(primary_hdu)
                cluster_hdul.append(cluster_hdu)
                if inj:
                    cluster_hdul.append(inj_hdu)
                cluster_hdul.append(info_hdu)
            else:
                cluster_hdul = outlier_hdu    
            
            outlierFilePath = fp.outlierFilePath(self.target, freq, taskName, stage, cluster=cluster)
            if chunk_size !=1:
                outlierFilePath = outlierFilePath[:-4] + '_chunk{}.fts'.format(chunk_index)
            if workInLocalDir:
                outlierFilePath = Path(outlierFilePath).name
            cluster_hdul.writeto(outlierFilePath, overwrite=True)
        return outlierFilePath 
    
    def writeFollowUpResult(self, new_cohDay, freq, old_mean2F, numTopList=1000, 
                            new_stage='followUp-1', new_freqDerivOrder=2, ratio=0, 
                            workInLocalDir=True, inj=False, cluster=False,
                            chunk_index=0, chunk_size=1, chunk_count=None,
                            n_skygrid=1):
        """
        Writes the follow-up result for a given frequency based on previous analysis.

        Parameters:
        - new_cohDay: int
            The new coherence day for the analysis.

        - freq: float
            The frequency in Hz for which results are being written.
            
        - old_mean2F: numpy.ndarray
            The mean 2F value of the outliers at previous stage (shorter coherence time).

        - numTopList: int, optional
            The maximum number of top results to consider. Default is 1000.

        - new_stage: str, optional
            The stage of the current analysis (e.g., 'followUp-1'). Default is 'followUp-1'.

        - new_freqDerivOrder: int, optional
            The order of frequency derivative used in the new analysis. Default is 2.

        - ratio: float, optional
            The ratio to adjust the mean2F threshold for the follow-up analysis. Default is None.

        - workInLocalDir: bool, optional
            If True, work with local directory paths. Default is True.

        - inj: bool, optional
            If True, includes injections in the follow-up result. Default is False
        """
    
        mean2F_th = old_mean2F * ratio

        print('ratio=',ratio)
        if chunk_count is not None:
            mean2F_th = mean2F_th[chunk_index*chunk_size:(chunk_index+1)*chunk_size]
        nJobs = mean2F_th.size
        outlierFilePath = self._writeFollowUpResult(new_cohDay, freq, mean2F_th, nJobs, numTopList, new_stage, new_freqDerivOrder, 
                                                    workInLocalDir, inj, cluster, chunk_index=chunk_index, chunk_size=chunk_size,
                                                    n_skygrid=n_skygrid)

        print('Finish writing followUp result for {0} Hz'.format(freq))
        return outlierFilePath

    def ensembleOutlierChunk(self, totalJobCounts, chunk_size, chunk_count, cohDay, freq, stage, freqDerivOrder, cluster, workInLocalDir):
        
        """
        Combines outlier results from multiple chunks into a single output file. Notice that injection jobs are not supported by this function.

        Parameters:
        - totalJobCounts: int
            The total number of jobs processed across all chunks.

        - chunk_size: int
            The size of each chunk being processed.

        - chunk_count: int
            The number of chunks to process.

        - cohDay: int
            The number of coherent observation days used in the analysis.

        - freq: float
            The frequency in Hz for which the outlier data is being processed.

        - stage: str
            The current stage of the analysis (e.g., 'search', 'followUp').

        - freqDerivOrder: int
            The order of the frequency derivative used in the analysis.

        - cluster: bool
            If True, indicates that clustering results should be included in the output.

        - workInLocalDir: bool
            If True, indicates that paths should be treated as local directory paths.

        Returns:
        - outlierFilePath: str
            The path to the output file containing the combined outlier results.
        """
        
        # Generate the task name based on the current parameters
        taskName = utils.taskName(self.target, stage, cohDay, freqDerivOrder, freq)
        outlierFilePath = fp.outlierFilePath(self.target, freq, taskName, stage, cluster=cluster) 
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name
            
        outlierTableList = []
        infoTableList = []
        
        # Iterate through each chunk to gather outlier data
        for i in range(chunk_count):
            outlierFilePath = outlierFilePath[:-4] + '_chunk{}.fts'.format(i)
            if workInLocalDir:
                outlierFilePath = Path(outlierFilePath).name

            outlierTableList.append( Table(fits.getdata(outlierFilePath, extname=stage+'_outlier')) )  
            infoTableList.append( fits.getdata(outlierFilePath, extname='info') )

        outlier_hdul = fits.HDUList()
        primary_hdu = fits.getheader(outlierFilePath)
        outlier_hdu =  fits.BinTableHDU(data=vstack(outlierTableList), name=stage+'_outlier')
        outlier_hdul.append(outlier_hdu)
        info_hdu =  fits.BinTableHDU(data=np.hstack(infoTableList), name='info')
        outlier_hdul.append(info_hdu) 
        outlier_hdul.writeto(outlierFilePath, overwrite=True)
        return outlierFilePath

    # Combines follow-up results from multiple outlier files into a single output file
    def ensembleFollowUpResult(self, stage, inj_stage, outlierFilePathList, inj_outlierFilePathList, mean2F_ratio_list, numTopListToFollowUp_list,
                               freq, final_stage, taskName, workInLocalDir=False, cluster=False):
        """
        Parameters:
        - outlierFilePathList: list of str
            List of file paths to outlier files from multiple follow-up stages. Each file contains outlier data for a specific follow-up stage.

        - inj_outlierFilePathList: list of str
            List of file paths to injection outlier files, used to track outliers associated with signal injections across follow-up stages.

        - mean2F_ratio_list: list of float
            List of mean 2F ratio values, one for each follow-up stage. These ratios are stored in the primary header to document each stage's threshold.

        - freq: float
            The frequency of interest for this ensemble follow-up. Helps identify the target frequency range for the data.

        - final_stage: str
            A label or identifier for the final processing stage. This label is included in the file name to indicate the completion stage of the analysis.

        - taskName: str
            Name of the task or job associated with this process, used to label the output file for easy identification.

        - workInLocalDir: bool, optional (default=False)
            Whether to save output files to the local directory. If `True`, saves to the current working directory instead of a central location.

        - cluster: bool, optional (default=False)
            Determines whether clustering is applied to the output data. If `True`, will trigger clustering on the output data to group related outliers.
        """
        
        # Set up stages for file paths in sequential follow-up and injection follow-up stages
        n_injTable = len(inj_outlierFilePathList)
        n_outTable = len(outlierFilePathList)
        # Store mean2F ratios in the header for each follow-up stage
        
        # Initialize primary HDU (header data unit) for storing metadata like mean2F ratios
        primary_hdu = fits.PrimaryHDU()
        outlier_hdul = fits.HDUList()
        mean2F_th = fits.getheader(outlierFilePathList[0])['HIERARCH mean2F_th']
        primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th
        if n_injTable != 0:
            primary_hdu.header['HIERARCH injection_test'] = True
        else:
            primary_hdu.header['HIERARCH injection_test'] = False
            
        for i in range(max(n_injTable-1, n_outTable-1)):
            primary_hdu.header['HIERARCH mean2F_ratio_{}'.format(stage[i+1])] = mean2F_ratio_list[i]
            primary_hdu.header['HIERARCH numTopList_{}'.format(stage[i+1])] = numTopListToFollowUp_list[i]
        outlier_hdul.append(primary_hdu)
                 
        # Append additional injection follow-up stages
        for i in range(n_injTable):           
            data = fits.getdata(inj_outlierFilePathList[i], extname=inj_stage[i]+'_outlier')
            outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_outlier'))
            
            data = fits.getdata(inj_outlierFilePathList[i], extname='inj') 
            outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_inj'))

            data = fits.getdata(inj_outlierFilePathList[i], extname='info')
            outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_info'))

        # Append each follow-up outlier file to the HDU list
        for i in range(n_outTable):
            data = fits.getdata(outlierFilePathList[i], extname=stage[i]+'_outlier')
            outlier_hdul.append(fits.BinTableHDU(data=data, name=stage[i]+'_outlier'))

            data = fits.getdata(outlierFilePathList[i], extname='info')
            outlier_hdul.append(fits.BinTableHDU(data=data, name=stage[i]+'_info'))
        
        # Generate file path for the output and handle local or default directory storage
        outlierFilePath = fp.outlierFilePath(self.target, freq, taskName, final_stage, cluster=cluster)    
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name        
        else:
            utils.makeDir([outlierFilePath])    
        
        # Write the combined HDU list to the output file
        outlier_hdul.writeto(outlierFilePath, overwrite=True)
        return outlierFilePath
