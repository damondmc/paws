from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np
from tqdm import tqdm
from pathlib import Path

from ..filePath import PathManager
from ..definitions import taskName, phaseParamName
from .._io import readTemplateCount, makeDir, getSpacing  
from . import tools
from .clustering import clustering  

class ResultManager:
    """
    Manages the collection, filtering, and storage of search results.
    """
    def __init__(self, target, config):
        """
        Initialize the ResultManager.

        Parameters:
            target (dict): Target object containing target information.
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.target = target
        self.paths = PathManager(config, target)
        
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
        task_name = taskName(self.target['name'], stage, cohDay, freqDerivOrder)
        
        # We need to construct the specific output filenames. 
        # Using the PathManager to get the base pattern.
        crfiles = self.paths.condor_record_files(freq, task_name, stage)
        base_out_path = str(crfiles[0]).rsplit('.', 1)[0] + '.'

        for jobIndex in range(1, nJobs+1):
            outFilePath = f"{base_out_path}{jobIndex}"
            templateList.append(readTemplateCount(outFilePath))
        return templateList
        
    def calMean2F_threshold(self, cohDay, freq, nJobs, nSeg):
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
        if None in nTemp:
            print(f"Warning: Some template counts are None for {freq} Hz.")
            nTemp = [t if t is not None else 0 for t in nTemp]
        total_templates = sum([t for t in nTemp if t is not None])
        mean2F_th = tools.mean2F_threshold(total_templates, nSeg)            
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
        data.add_column(mean2F_th * np.ones(len(data)), name='mean2F threshold')
    
        # Get parameter names (e.g., ['f0', 'f1dot', ...])
        _, deriv_params = phaseParamName(freqDerivOrder)
        
        for param in deriv_params:
            data.add_column(spacing[param] * np.ones(len(data)), name=param) 
        return data

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
        
        # Calculate h0 from aPlus and aCross
        aplus, across = injParam['aPlus'], injParam['aCross']
        h0 = 0.5 * (2.*aplus + 2.*np.sqrt(aplus**2 - across**2))
        injParam.add_column(h0 * np.ones(len(injParam)), name='h0')
        
        # Rename reference time if exists
        if 'refTime_s' in injParam.colnames:
            injParam.rename_column('refTime_s', 'refTime')   

        searchParam = Table(searchParam)[:1] 

        return searchParam, injParam
    
    def _writeSearchResult(self, cohDay, freq, mean2F_th, nJobs, numTopListLimit=1000, stage='search', freqDerivOrder=2, cluster=False, workInLocalDir=False):
        """
        Internal core function to read job outputs, filter outliers, and write the combined FITS file.
        """      
        task_name = taskName(self.target['name'], stage, cohDay, freqDerivOrder)
         
        outlierTableList = []
        # Info table to track stats per job
        info_data = np.recarray((nJobs,), dtype=[(key, '>f8') for key in ['freq', 'jobIndex', 'outliers', 'saturated']]) 

        for i, jobIndex in enumerate(tqdm(range(1, nJobs+1), desc=f"Collecting {freq}Hz")):
            
            # 1. Get Path
            weaveFilePath = self.paths.weave_output_file(freq, task_name, jobIndex, stage)
            if workInLocalDir:
                weaveFilePath = Path(weaveFilePath).name
            
            try:
                weave_data = fits.getdata(weaveFilePath, 1)
                # 3. Get Spacing (Resolution) from FITS header or tools
                spacing = getSpacing(weaveFilePath, freqDerivOrder)
                
                # 4. Filter Outliers
                _outlier = self.makeOutlierTable(weave_data, spacing, mean2F_th, numTopListLimit, freqDerivOrder)  

                # 5. Check Saturation
                if len(_outlier) >= numTopListLimit:
                    info_data[i] = freq, jobIndex, numTopListLimit, 1  # Saturated
                else:
                    info_data[i] = freq, jobIndex, len(_outlier), 0  # OK
                    outlierTableList.append(_outlier)
            except FileNotFoundError:
                print(f"Warning: File not found {weaveFilePath}")
                info_data[i] = freq, jobIndex, 0, 0

        # 6. Identify Non-Saturated Bands
        sat = info_data['saturated'].reshape(int(1/self.config['fBand']), int(nJobs*self.config['fBand']))  

        idx = np.where(sat.sum(axis=1) == 0)[0]
        nonSatBand = np.recarray((len(idx),), dtype=[(key, '>f8') for key in ['nonSatBand']])
        
        if len(idx) > 0:
            nonSatBand['nonSatBand'] = int(freq) + np.array(idx) * self.config['fBand']
           
        # 7. Create HDUs
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th        

        for name, value in spacing.items():
            primary_hdu.header['HIERARCH {}'.format(name)] = value
        
        outlier_hdu = fits.BinTableHDU(data=vstack(outlierTableList), name=stage+'_outlier')
        info_hdu = fits.BinTableHDU(data=info_data, name='info') 
        nsb_hdu = fits.BinTableHDU(data=nonSatBand, name='nonSatBand')

        outlier_hdul = fits.HDUList([primary_hdu, outlier_hdu, info_hdu, nsb_hdu])
        
        # 8. Write Combined File
        outlierFilePath = self.paths.outlier_file(freq, task_name, stage, cluster=False)
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name
            
        makeDir([outlierFilePath])
        outlier_hdul.writeto(outlierFilePath, overwrite=True)  
       
        # 9. Clustering (Optional)
        if cluster:
            if outlier_hdu.data.size > 1:
                cluster_hdul = fits.HDUList()
                
                primary_hdu = fits.PrimaryHDU()
                primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th
                primary_hdu.header['HIERARCH cluster_nSpacing'] = self.config['cluster_nSpacing']
                
                for name, value in spacing.items():
                    primary_hdu.header['HIERARCH {}'.format(name)] = value 
       
                # Call Clustering Tool
                centers_idx, cluster_size, _ = clustering(outlier_hdu.data, freqDerivOrder) 
                
                cluster_data = outlier_hdu.data[centers_idx]
                cluster_hdu = fits.BinTableHDU(data=cluster_data, name=stage+'_outlier')
     
                cluster_info = np.recarray((cluster_size.size,), dtype=[(key, '>f8') for key in ['freq', 'clusterIndex', 'noOutliersWithin']]) 
                for i in range(cluster_size.size):
                    cluster_info[i] = freq, i, cluster_size[i]
                
                cluster_info_hdu = fits.BinTableHDU(data=cluster_info, name='info')
                
                cluster_hdul.append(primary_hdu)
                cluster_hdul.append(cluster_hdu)
                cluster_hdul.append(cluster_info_hdu)
                cluster_hdul.append(nsb_hdu)
            else:
                cluster_hdul = outlier_hdul
                
            outlierClusterFilePath = self.paths.outlier_file(freq, task_name, stage, cluster=True)
            if workInLocalDir:
                outlierClusterFilePath = Path(outlierClusterFilePath).name
            
            cluster_hdul.writeto(outlierClusterFilePath, overwrite=True)
            return outlierClusterFilePath
        else:
            return outlierFilePath 

    def writeSearchResult(self, cohDay, freq, mean2F_th, nJobs, numTopList=1000, stage='search', freqDerivOrder=2, cluster=False, workInLocalDir=False):
        """
        Public wrapper to write search results.
        
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
        
        outlierFilePath = self._writeSearchResult(cohDay, freq, mean2F_th, nJobs, numTopList, stage, freqDerivOrder, cluster, workInLocalDir)
        print('Finish writing search result for {0} Hz'.format(freq))
        return outlierFilePath 
    
    def _writeSearchResultFromSaturatedBand(self, cohDay, freq, mean2F_th, jobIndex, numTopListLimit=1, stage='search', freqDerivOrder=2, workInLocalDir=False):
        """
        Writes results specifically for bands that were saturated in a previous pass.
        """      
        task_name = taskName(self.target['name'], stage, cohDay, freqDerivOrder)
        outlierTableList = []
    
        for idx in tqdm(jobIndex, desc="Processing Sat Bands"):
            weaveFilePath = self.paths.weave_output_file(freq, task_name, idx, stage)
            if workInLocalDir:
                weaveFilePath = Path(weaveFilePath).name
            
            try:
                weave_data = fits.getdata(weaveFilePath, 1)
                spacing = getSpacing(weaveFilePath, freqDerivOrder)
                _outlier = self.makeOutlierTable(weave_data, spacing, mean2F_th, numTopListLimit, freqDerivOrder)  
                outlierTableList.append(_outlier)
            except FileNotFoundError:
                print(f"File missing for sat band job {idx}")
           
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th

        if len(outlierTableList) == 0:
            outlier_hdu = fits.BinTableHDU(name=stage+'SatBand_outlier')
        else:
            outlier_hdu = fits.BinTableHDU(data=vstack(outlierTableList), name=stage+'SatBand_outlier')
        
        outlier_hdul = fits.HDUList([primary_hdu, outlier_hdu])
        
        # Note: changing taskName for filename generation to indicate SatBand
        task_name = taskName(self.target['name'], stage+'SatBand', cohDay, freqDerivOrder)
        outlierFilePath = self.paths.outlier_file(freq, task_name, stage, cluster=False)
        
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name
            
        makeDir([outlierFilePath])
        outlier_hdul.writeto(outlierFilePath, overwrite=True)  
       
        return outlierFilePath 
   
    def writeSearchResultFromSaturatedBand(self, cohDay, freq, mean2F_th, jobIndex, numTopList=1, stage='search', freqDerivOrder=2, workInLocalDir=False):
        """
        Public wrapper for saturated band results.

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

        outlierFilePath = self._writeSearchResultFromSaturatedBand(cohDay, freq, mean2F_th, jobIndex, numTopList, stage, freqDerivOrder, workInLocalDir)
        print('Finish writing search result for {0} Hz'.format(freq))
        return outlierFilePath
    
# --------------------------------------------------------------------------
    # Injection & Follow-up Methods
    # --------------------------------------------------------------------------

    def _writeInjectionResult(self, cohDay, freq, mean2F_th, nJobs, numTopListLimit=1000, stage='search', freqDerivOrder=2, workInLocalDir=False, cluster=False):
        """
        Writes the injection results from the Weave output for a given frequency.
        """
        task_name = taskName(self.target['name'], stage, cohDay, freqDerivOrder)
        outlierTableList = []
        injTableList = []
        info_data = np.recarray((nJobs,), dtype=[(key, '>f8') for key in ['freq', 'jobIndex', 'outliers']]) 
  
        # Iterate over jobs
        for i, jobIndex in enumerate(tqdm(range(1, nJobs+1), desc=f"Inj Collection {freq}Hz")):
            weaveFilePath = self.paths.weave_output_file(freq, task_name, jobIndex, stage)
            if workInLocalDir:
                weaveFilePath = Path(weaveFilePath).name
            
            try:
                # HDU 1: Outliers, HDU 2: Injection Parameters
                weave_data = fits.getdata(weaveFilePath, 1)
                inj_data = fits.getdata(weaveFilePath, 2)
                
                spacing = getSpacing(weaveFilePath, freqDerivOrder)
                
                # Filter outliers
                _outlier = self.makeOutlierTable(weave_data, spacing, mean2F_th, numTopListLimit, freqDerivOrder)  
                
                # Match injections
                _outlier, _injParam = self.makeInjectionTable(inj_data, _outlier, freqDerivOrder)
                
                outlierTableList.append(_outlier)
                if len(_outlier) > 0:
                    injTableList.append(_injParam)
                
                info_data[i] = freq, jobIndex, len(_outlier)

            except FileNotFoundError:
                print(f"Warning: File not found {weaveFilePath}")
                info_data[i] = freq, jobIndex, 0

        # Combine Tables
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th
        primary_hdu.header['HIERARCH cluster_nSpacing'] = ''
        
        if outlierTableList:
            outlier_hdu = fits.BinTableHDU(data=vstack(outlierTableList), name=stage+'_outlier')
        else:
            outlier_hdu = fits.BinTableHDU(Table(), name=stage+'_outlier')
            
        info_hdu = fits.BinTableHDU(data=info_data, name='info') 
      
        if injTableList:
            inj_hdu = fits.BinTableHDU(data=vstack(injTableList), name='inj')
        else:
            inj_hdu = fits.BinTableHDU(name='inj')
            print('No outliers found overlapping with injections.')
        
        outlier_hdul = fits.HDUList([primary_hdu, outlier_hdu, inj_hdu, info_hdu])
        
        outlierFilePath = self.paths.outlier_file(freq, t_name, stage, cluster=False)
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name
            
        makeDir([outlierFilePath])
        outlier_hdul.writeto(outlierFilePath, overwrite=True) 
                
        # Clustering
        if cluster and outlier_hdu.data.size > 1:
            cluster_hdul = fits.HDUList()
            
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th
            primary_hdu.header['HIERARCH cluster_nSpacing'] = self.config.get('cluster_nSpacing', 1)
            
            centers_idx, cluster_size, cluster_member = clustering(outlier_hdu.data, freqDerivOrder) 
            
            # Map every outlier to a cluster center (for injection tracking)
            center_idx_for_each_outlier = np.full(outlier_hdu.data.size, -1)
            processed_indices = set()
            
            for ci, members in zip(centers_idx, cluster_member):
                # Filter members we haven't processed yet to avoid double counting
                idx = np.array([item for item in members if item not in processed_indices])
                if len(idx) > 0:
                    center_idx_for_each_outlier[idx] = ci
                    processed_indices.update(members)

            cluster_data = outlier_hdu.data[center_idx_for_each_outlier]
            cluster_hdu = fits.BinTableHDU(data=cluster_data, name=stage+'_outlier')

            info_data = np.recarray((cluster_size.size,), dtype=[(key, '>f8') for key in ['freq', 'clusterIndex', 'noOutliersWithin']]) 
            for i in range(cluster_size.size):
                info_data[i] = freq, i, cluster_size[i]
            
            info_hdu = fits.BinTableHDU(data=info_data, name='info')

            cluster_hdul = fits.HDUList([primary_hdu, cluster_hdu, inj_hdu, info_hdu])
            
            outlierFilePath = self.paths.outlier_file(freq, t_name, stage, cluster=True)
            if workInLocalDir:
                outlierFilePath = Path(outlierFilePath).name
            
            cluster_hdul.writeto(outlierFilePath, overwrite=True)
            
        return outlierFilePath 

    def writeInjectionResult(self, cohDay, freq, mean2F_th, nJobs, numTopList=1000, stage='search', freqDerivOrder=2, workInLocalDir=False, cluster=False):
        """
        Public wrapper to write injection-search results.
        """
        outlierFilePath = self._writeInjectionResult(cohDay, freq, mean2F_th, nJobs, numTopList, stage, freqDerivOrder, workInLocalDir, cluster)
        print('Finish writing injection result for {0} Hz'.format(freq))
        return outlierFilePath

    def _writeFollowUpResult(self, cohDay, freq, mean2F_th, nJobs, numTopListLimit=1000, stage='search', freqDerivOrder=2, 
                                   workInLocalDir=True, inj=False, cluster=False,
                                   chunk_index=0, chunk_size=1):
        """
        Writes the follow-up results for injections at a given frequency, supporting chunking.
        """
        task_name = taskName(self.target['name'], stage, cohDay, freqDerivOrder)
    
        outlierTableList = []
        injTableList = []
        info_data = np.recarray((nJobs,), dtype=[(key, '>f8') for key in ['freq', 'jobIndex', 'outliers']]) 
 
        # Determine job range based on chunk
        start_job = chunk_index * chunk_size + 1
        end_job = chunk_index * chunk_size + nJobs + 1
        
        # Iterate over each job in the chunk
        for i, jobIndex in enumerate(range(start_job, end_job)):
            weaveFilePath = self.paths.weave_output_file(freq, task_name, jobIndex, stage)
            if workInLocalDir:
                weaveFilePath = Path(weaveFilePath).name
            
            try:
                weave_data = fits.getdata(weaveFilePath, 1)
                spacing = getSpacing(weaveFilePath, freqDerivOrder)
                
                # Note: mean2F_th is an array here, indexed by i
                _outlier = self.makeOutlierTable(weave_data, spacing, mean2F_th[i], numTopListLimit, freqDerivOrder)
                
                if inj:
                    injParam = fits.getdata(weaveFilePath, 2)
                    _outlier, injParam = self.makeInjectionTable(injParam, _outlier, freqDerivOrder)
                    
                if len(_outlier) > 0:
                    outlierTableList.append(_outlier)
                    if inj:
                        injTableList.append(injParam)
                else:
                    outlierTableList.append(_outlier) # Append empty table to keep alignment? Or just skip? Code appended empty.
                        
                info_data[i] = freq, jobIndex, len(_outlier)
            except FileNotFoundError:
                 info_data[i] = freq, jobIndex, 0

        # Combine Tables
        primary_hdu = fits.PrimaryHDU()
        
        if outlierTableList:
            outlier_hdu = fits.BinTableHDU(data=vstack(outlierTableList), name=stage+'_outlier')
        else:
            outlier_hdu = fits.BinTableHDU(name=stage+'_outlier')
            print('No outlier in follow-up chunk.')
        
        info_hdu = fits.BinTableHDU(data=info_data, name='info')
        
        # Construct HDU List
        outlier_hdul = fits.HDUList([primary_hdu, outlier_hdu])
        
        if inj and injTableList:
            inj_hdu = fits.BinTableHDU(data=vstack(injTableList), name='inj')
            outlier_hdul.append(inj_hdu)
        elif inj:
            outlier_hdul.append(fits.BinTableHDU(name='inj'))
            
        outlier_hdul.append(info_hdu)
        
        # Generate Output Path (Handle Chunk Naming)
        outlierFilePath = self.paths.outlier_file(freq, task_name, stage, cluster=(cluster and not inj))
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name
            
        if chunk_size != 1:
            # Inject chunk index into filename. 
            # Ideally PathManager handles this, but for now we append to the string.
            outlierFilePath = outlierFilePath.replace('.fts', f'_chunk{chunk_index}.fts')
            
        makeDir([outlierFilePath])
        outlier_hdul.writeto(outlierFilePath, overwrite=True)     
        
        # Clustering for Follow-Up
        if cluster and outlier_hdu.data.size > 1:
            cluster_hdul = fits.HDUList()
            centers_idx, cluster_size, cluster_member = clustering(outlier_hdu.data, freqDerivOrder) 
            
            if inj:
                # If injection, mapped clustering
                center_idx_for_each_outlier = np.full(outlier_hdu.data.size, -1)
                processed_indices = set()
                for ci, members in zip(centers_idx, cluster_member):
                    idx = np.array([item for item in members if item not in processed_indices])
                    center_idx_for_each_outlier[idx] = ci
                    processed_indices.update(members)
                cluster_data = outlier_hdu.data[center_idx_for_each_outlier]
            else:
                cluster_data = outlier_hdu.data[centers_idx]

            cluster_hdu = fits.BinTableHDU(data=cluster_data, name=stage+'_outlier')

            info_data = np.recarray((cluster_size.size,), dtype=[(key, '>f8') for key in ['freq', 'chunkIndex', 'clusterIndex', 'noOutliersWithin']]) 
            for i in range(cluster_size.size):
                info_data[i] = freq, chunk_index, i, cluster_size[i]
            
            info_clustered_hdu = fits.BinTableHDU(data=info_data, name='info_clustered')

            cluster_hdul = fits.HDUList([primary_hdu, cluster_hdu])
            if inj and len(outlier_hdul) > 2: # Check if inj_hdu exists
                 cluster_hdul.append(outlier_hdul['inj'])
            cluster_hdul.append(info_clustered_hdu)
            
            # Write Clustered File
            outlierFilePath = self.paths.outlier_file(freq, task_name, stage, cluster=cluster)
            if chunk_size != 1:
                outlierFilePath = outlierFilePath.replace('.fts', f'_chunk{chunk_index}.fts')
            if workInLocalDir:
                outlierFilePath = Path(outlierFilePath).name
                
            cluster_hdul.writeto(outlierFilePath, overwrite=True)
            
        return outlierFilePath 
    
    def writeFollowUpResult(self, new_cohDay, freq, old_mean2F, numTopList=1000, 
                            new_stage='followUp-1', new_freqDerivOrder=2, ratio=0, 
                            workInLocalDir=True, inj=False, cluster=False,
                            chunk_index=0, chunk_size=1, chunk_count=None):
        """
        Public wrapper to write follow-up results.
        Calculates the new threshold based on old_mean2F * ratio.

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
        print(f'Follow-up Ratio: {ratio}')
        
        if chunk_count is not None:
            # Slice the threshold array for this specific chunk
            mean2F_th = mean2F_th[chunk_index*chunk_size : (chunk_index+1)*chunk_size]
            
        nJobs = mean2F_th.size
        
        outlierFilePath = self._writeFollowUpResult(new_cohDay, freq, mean2F_th, nJobs, numTopList, new_stage, new_freqDerivOrder, 
                                                    workInLocalDir, inj, cluster, chunk_index=chunk_index, chunk_size=chunk_size)

        print(f'Finish writing followUp result for {freq} Hz')
        return outlierFilePath

    def ensembleOutlierChunk(self, chunk_count, cohDay, freq, stage, freqDerivOrder, cluster, workInLocalDir):
        """
        Combines outlier results from multiple chunks into a single output file.

        Parameters:
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
        task_name = taskName(self.target['name'], stage, cohDay, freqDerivOrder)
        outlierFilePath = self.paths.outlier_file(freq, task_name, stage, cluster=cluster)
        
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name
            
        outlierTableList = []
        infoTableList = []
        
        # Iterate through each chunk
        for i in range(chunk_count):
            # Construct chunk filename manually based on convention
            _outlierFilePath = outlierFilePath.replace('.fts', f'_chunk{i}.fts')
            
            try:
                # Read tables
                outlierTableList.append(Table(fits.getdata(_outlierFilePath, extname=stage+'_outlier')))  
                infoTableList.append(fits.getdata(_outlierFilePath, extname='info'))
            except FileNotFoundError:
                print(f"Warning: Chunk file {_outlierFilePath} missing.")

        # Stack and Write
        outlier_hdul = fits.HDUList()
        # Grab header from first available chunk
        if chunk_count > 0:
             # Re-construct first chunk path
            first_chunk = outlierFilePath.replace('.fts', '_chunk0.fts')
            primary_hdu = fits.PrimaryHDU(header=fits.getheader(first_chunk))
        else:
            primary_hdu = fits.PrimaryHDU()
            
        outlier_hdul.append(primary_hdu)
        
        if outlierTableList:
            outlier_hdu = fits.BinTableHDU(data=vstack(outlierTableList), name=stage+'_outlier')
            outlier_hdul.append(outlier_hdu)
        
        if infoTableList:
            info_hdu = fits.BinTableHDU(data=np.hstack(infoTableList), name='info')
            outlier_hdul.append(info_hdu) 
            
        outlier_hdul.writeto(outlierFilePath, overwrite=True)
        return outlierFilePath

    def ensembleFollowUpResult(self, stage, inj_stage, outlierFilePathList, inj_outlierFilePathList, mean2F_ratio_list, numTopListToFollowUp_list,
                               freq, final_stage, task_name, workInLocalDir=False, cluster=False):
        """
        Combines results from multiple follow-up stages into one summary FITS file.

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
        n_injTable = len(inj_outlierFilePathList)
        n_outTable = len(outlierFilePathList)
        
        primary_hdu = fits.PrimaryHDU()
        outlier_hdul = fits.HDUList()
        
        # Metadata
        try:
            mean2F_th = fits.getheader(outlierFilePathList[0])['HIERARCH mean2F_th']
            primary_hdu.header['HIERARCH mean2F_th'] = mean2F_th
        except (IndexError, KeyError):
            print("Warning: Unable to retrieve mean2F_th from header.")
            pass

        primary_hdu.header['HIERARCH injection_test'] = (n_injTable != 0)
            
        # Record ratios and top lists for every stage in the header
        for i in range(max(n_injTable-1, n_outTable-1)):
            if i < len(mean2F_ratio_list):
                primary_hdu.header[f'HIERARCH mean2F_ratio_{stage[i+1]}'] = mean2F_ratio_list[i]
            if i < len(numTopListToFollowUp_list):
                primary_hdu.header[f'HIERARCH numTopList_{stage[i+1]}'] = numTopListToFollowUp_list[i]
                
        outlier_hdul.append(primary_hdu)
                 
        # 1. Append Injection Follow-up Stages
        for i in range(n_injTable):           
            try:
                # Outliers
                data = fits.getdata(inj_outlierFilePathList[i], extname=inj_stage[i]+'_outlier')
                outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_outlier'))
                
                # Injections
                data = fits.getdata(inj_outlierFilePathList[i], extname='inj') 
                outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_inj'))

                # Info
                data = fits.getdata(inj_outlierFilePathList[i], extname='info')
                outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_info'))
            except FileNotFoundError:
                print(f"Warning: Missing injection file {inj_outlierFilePathList[i]}")

        # 2. Append Search Follow-up Stages
        for i in range(n_outTable):
            try:
                data = fits.getdata(outlierFilePathList[i], extname=stage[i]+'_outlier')
                outlier_hdul.append(fits.BinTableHDU(data=data, name=stage[i]+'_outlier'))

                data = fits.getdata(outlierFilePathList[i], extname='info')
                outlier_hdul.append(fits.BinTableHDU(data=data, name=stage[i]+'_info'))
            except FileNotFoundError:
                print(f"Warning: Missing outlier file {outlierFilePathList[i]}")
        
        # Write Final Ensemble File
        # Using paths manager manually or custom logic because "final_stage" might be arbitrary
        outlierFilePath = self.paths.outlier_file(freq, task_name, final_stage, cluster=cluster)    
        
        if workInLocalDir:
            outlierFilePath = Path(outlierFilePath).name        
        else:
            makeDir([outlierFilePath])    
        
        outlier_hdul.writeto(outlierFilePath, overwrite=True)
        return outlierFilePath