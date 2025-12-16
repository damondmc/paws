from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np
from tqdm import tqdm
from pathlib import Path

from paws.filepaths import PathManager
from paws.definitions import task_name, phase_param_name
from paws.io import read_template_count, make_dir, get_spacing 
from .tools import detection_stat_threshold
from .clustering import clustering  

class ResultAnalysisManager:
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
        
    def read_template_count(self, taskname, freq, n_jobs, stage='search', freq_deriv_order=2):
        """Reads the template count from the weave output for each job in a specified frequency band."""
        template_list = []
        
        # Construct specific output filenames using PathManager
        crfiles = self.paths.condor_record_files(freq, taskname, stage)
        base_out_path = str(crfiles[0]).rsplit('.', 1)[0] + '.'

        for job_index in range(1, n_jobs + 1):
            out_file_path = f"{base_out_path}{job_index}"
            template_list.append(read_template_count(out_file_path))
        return template_list
        
    def get_mean2f_threshold(self, taskname, freq, n_jobs, n_seg):
        """Calculates the Mean 2F threshold."""
        # Get the number of templates based on coherence time, frequency, and job count
        n_temp = self.read_template_count(taskname, freq, n_jobs)
        if None in n_temp:
            print(f"Warning: Some template counts are None for {freq} Hz.")
            n_temp = [t if t is not None else 0 for t in n_temp]
        
        total_templates = sum([t for t in n_temp if t is not None])
        mean2f_th = detection_stat_threshold(total_templates, n_seg)            
        return mean2f_th

    def make_outlier_table(self, data, spacing, mean2f_th, toplist_limit=1000):    
        """Filters data to create an outlier table."""
        # Read and limit the data to the top entries
        data = data[:toplist_limit]
        
        # Mask data with mean 2F values greater than the threshold
        mask = data['mean2F'] > mean2f_th
        data = Table(data[mask])       
        data.add_column(mean2f_th * np.ones(len(data)), name='mean2F threshold')
    
        # Get parameter names (e.g., ['f0', 'f1dot', ...])
        _, deriv_params = phase_param_name(len(spacing)-1)
        
        for param in deriv_params:
            data.add_column(spacing[param] * np.ones(len(data)), name=param) 
        return data

    def make_injection_table(self, inj_param, search_param):   
        """Creates a table comparing injections with search results."""
        inj_param = Table(inj_param)   
        
        # Calculate h0 from aPlus and aCross
        aplus, across = inj_param['aPlus'], inj_param['aCross']
        h0 = 0.5 * (2. * aplus + 2. * np.sqrt(aplus**2 - across**2))
        inj_param.add_column(h0 * np.ones(len(inj_param)), name='h0')
        
        # Rename reference time if exists
        if 'refTime_s' in inj_param.colnames:
            inj_param.rename_column('refTime_s', 'refTime')   

        search_param = Table(search_param)[:1] 

        return search_param, inj_param
    
    def _make_search_outlier(self, taskname, freq, mean2f_th, n_jobs, num_top_list_limit=1000, 
                             stage='search', freq_deriv_order=2, cluster=False, work_in_local_dir=False):
        """Internal core function to read job outputs, filter outliers, and write the combined FITS file."""      
        outlier_table_list = []
        # Info table to track stats per job
        info_data = np.recarray((n_jobs,), dtype=[(key, '>f8') for key in ['freq', 'jobIndex', 'outliers', 'saturated']]) 

        for i, job_index in enumerate(tqdm(range(1, n_jobs + 1), desc=f"Collecting {freq}Hz")):
            
            # 1. Get Path
            weave_file_path = self.paths.weave_output_file(freq, taskname, job_index, stage)
            if work_in_local_dir:
                weave_file_path = Path(weave_file_path).name
            
            try:
                weave_data = fits.getdata(weave_file_path, 1)
                # 3. Get Spacing (Resolution) from FITS header or tools
                spacing = get_spacing(weave_file_path, freq_deriv_order)
                
                # 4. Filter Outliers
                _outlier = self.make_outlier_table(weave_data, spacing, mean2f_th, num_top_list_limit)  

                # 5. Check Saturation
                if len(_outlier) >= num_top_list_limit:
                    info_data[i] = freq, job_index, num_top_list_limit, 1  # Saturated
                else:
                    info_data[i] = freq, job_index, len(_outlier), 0  # OK
                    outlier_table_list.append(_outlier)
            except FileNotFoundError:
                print(f"Warning: File not found {weave_file_path}")
                info_data[i] = freq, job_index, 0, 0

        # 6. Identify Non-Saturated Bands
        sat = info_data['saturated'].reshape(int(1/self.config['f0_band']), int(n_jobs*self.config['f0_band']))  

        idx = np.where(sat.sum(axis=1) == 0)[0]
        non_sat_band = np.recarray((len(idx),), dtype=[(key, '>f8') for key in ['nonSatBand']])
        
        if len(idx) > 0:
            non_sat_band['nonSatBand'] = int(freq) + np.array(idx) * self.config['f0_band']
           
        # 7. Create HDUs
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['HIERARCH mean2F_th'] = mean2f_th        

        for name, value in spacing.items():
            primary_hdu.header['HIERARCH {}'.format(name)] = value
        
        outlier_hdu = fits.BinTableHDU(data=vstack(outlier_table_list), name=stage+'_outlier')
        info_hdu = fits.BinTableHDU(data=info_data, name='info') 
        nsb_hdu = fits.BinTableHDU(data=non_sat_band, name='nonSatBand')

        outlier_hdul = fits.HDUList([primary_hdu, outlier_hdu, info_hdu, nsb_hdu])
        
        # 8. Write Combined File
        outlier_file_path = self.paths.outlier_file(freq, taskname, stage, cluster=False)
        if work_in_local_dir:
            outlier_file_path = Path(outlier_file_path).name
            
        make_dir([outlier_file_path])
        outlier_hdul.writeto(outlier_file_path, overwrite=True)  
       
        # 9. Clustering (Optional)
        if cluster:
            if outlier_hdu.data.size > 1:
                cluster_hdul = fits.HDUList()
                
                primary_hdu = fits.PrimaryHDU()
                primary_hdu.header['HIERARCH mean2F_th'] = mean2f_th
                primary_hdu.header['HIERARCH cluster_n_spacing'] = self.config['cluster_n_spacing']
                
                for name, value in spacing.items():
                    primary_hdu.header['HIERARCH {}'.format(name)] = value 
       
                # Call Clustering Tool
                centers_idx, cluster_size, _ = clustering(outlier_hdu.data, freq_deriv_order) 
                
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
                
            outlier_cluster_file_path = self.paths.outlier_file(freq, taskname, stage, cluster=True)
            if work_in_local_dir:
                outlier_cluster_file_path = Path(outlier_cluster_file_path).name
            
            cluster_hdul.writeto(outlier_cluster_file_path, overwrite=True)
            return outlier_cluster_file_path
        else:
            return outlier_file_path 

    def make_search_outlier(self, taskname, freq, mean2f_th, n_jobs, num_top_list=1000, 
                            stage='search', freq_deriv_order=2, cluster=False, work_in_local_dir=False):
        """
        Public wrapper to write search results.

        Parameters:
        - taskname: str
            The name of the task for the search, used in naming and organizing output files.

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
        
        outlier_file_path = self._make_search_outlier(taskname, freq, mean2f_th, n_jobs, num_top_list, 
                                                      stage, freq_deriv_order, cluster, work_in_local_dir)
        print('Finish writing search result for {0} Hz'.format(freq))
        return outlier_file_path 
    
    def _make_search_outlier_from_saturated_band(self, taskname, freq, mean2f_th, job_index, 
                                                 num_top_list_limit=1, stage='search', freq_deriv_order=2, work_in_local_dir=False):
        """Writes results specifically for bands that were saturated in a previous pass."""      
        outlier_table_list = []
    
        for idx in tqdm(job_index, desc="Processing Sat Bands"):
            weave_file_path = self.paths.weave_output_file(freq, taskname, idx, stage)
            if work_in_local_dir:
                weave_file_path = Path(weave_file_path).name
            
            try:
                weave_data = fits.getdata(weave_file_path, 1)
                spacing = get_spacing(weave_file_path, freq_deriv_order)
                _outlier = self.make_outlier_table(weave_data, spacing, mean2f_th, num_top_list_limit)  
                outlier_table_list.append(_outlier)
            except FileNotFoundError:
                print(f"File missing for sat band job {idx}")
           
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['HIERARCH mean2F_th'] = mean2f_th

        if len(outlier_table_list) == 0:
            outlier_hdu = fits.BinTableHDU(name=stage+'SatBand_outlier')
        else:
            outlier_hdu = fits.BinTableHDU(data=vstack(outlier_table_list), name=stage+'SatBand_outlier')
        
        outlier_hdul = fits.HDUList([primary_hdu, outlier_hdu])
        
        # Note: changing taskName for filename generation to indicate SatBand
        taskname = taskname + '_satband'
        outlier_file_path = self.paths.outlier_file(freq, taskname, stage, cluster=False)
        
        if work_in_local_dir:
            outlier_file_path = Path(outlier_file_path).name
            
        make_dir([outlier_file_path])
        outlier_hdul.writeto(outlier_file_path, overwrite=True)  
       
        return outlier_file_path 
   
    def make_search_outlier_from_saturated_band(self, taskname, freq, mean2f_th, job_index, 
                                                num_top_list=1, stage='search', freq_deriv_order=2, work_in_local_dir=False):
        """
        Public wrapper for saturated band results.

        Parameters:
        - taskname: str
            The name of the task, used for naming and organizing output files.

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

        outlier_file_path = self._make_search_outlier_from_saturated_band(taskname, freq, mean2f_th, job_index, 
                                                                          num_top_list, stage, freq_deriv_order, work_in_local_dir)
        print('Finish writing search result for {0} Hz'.format(freq))
        return outlier_file_path
    
# --------------------------------------------------------------------------
    # Injection & Follow-up Methods
    # --------------------------------------------------------------------------

    def _make_injection_outlier(self, taskname, freq, mean2f_th, n_jobs, num_top_list_limit=1000, 
                                stage='search', freq_deriv_order=2, cluster=False, work_in_local_dir=False):
        """Writes the injection results from the Weave output for a given frequency."""
        outlier_table_list = []
        inj_table_list = []
        info_data = np.recarray((n_jobs,), dtype=[(key, '>f8') for key in ['freq', 'jobIndex', 'outliers']]) 
  
        # Iterate over jobs
        for i, job_index in enumerate(tqdm(range(1, n_jobs + 1), desc=f"Inj Collection {freq}Hz")):
            weave_file_path = self.paths.weave_output_file(freq, taskname, job_index, stage)
            if work_in_local_dir:
                weave_file_path = Path(weave_file_path).name
            
            try:
                # HDU 1: Outliers, HDU 2: Injection Parameters
                weave_data = fits.getdata(weave_file_path, 1)
                inj_data = fits.getdata(weave_file_path, 2)
                
                spacing = get_spacing(weave_file_path, freq_deriv_order)
                
                # Filter outliers
                _outlier = self.make_outlier_table(weave_data, spacing, mean2f_th, num_top_list_limit)  
                
                # Match injections
                _outlier, _inj_param = self.make_injection_table(inj_data, _outlier)
                
                outlier_table_list.append(_outlier)
                if len(_outlier) > 0:
                    inj_table_list.append(_inj_param)
                
                info_data[i] = freq, job_index, len(_outlier)

            except FileNotFoundError:
                print(f"Warning: File not found {weave_file_path}")
                info_data[i] = freq, job_index, 0

        # Combine Tables
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['HIERARCH mean2F_th'] = mean2f_th
        primary_hdu.header['HIERARCH cluster_n_spacing'] = ''
        
        if outlier_table_list:
            outlier_hdu = fits.BinTableHDU(data=vstack(outlier_table_list), name=stage+'_outlier')
        else:
            outlier_hdu = fits.BinTableHDU(Table(), name=stage+'_outlier')
            
        info_hdu = fits.BinTableHDU(data=info_data, name='info') 
      
        if inj_table_list:
            inj_hdu = fits.BinTableHDU(data=vstack(inj_table_list), name='inj')
        else:
            inj_hdu = fits.BinTableHDU(name='inj')
            print('No outliers found overlapping with injections.')
        
        outlier_hdul = fits.HDUList([primary_hdu, outlier_hdu, inj_hdu, info_hdu])
        
        outlier_file_path = self.paths.outlier_file(freq, taskname, stage, cluster=False)
        if work_in_local_dir:
            outlier_file_path = Path(outlier_file_path).name
            
        make_dir([outlier_file_path])
        outlier_hdul.writeto(outlier_file_path, overwrite=True) 
                
        # Clustering
        if cluster and outlier_hdu.data.size > 1:
            cluster_hdul = fits.HDUList()
            
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header['HIERARCH mean2F_th'] = mean2f_th
            primary_hdu.header['HIERARCH cluster_n_spacing'] = self.config.get('cluster_n_spacing', 1)
            
            centers_idx, cluster_size, cluster_member = clustering(outlier_hdu.data, freq_deriv_order) 
            
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
            
            outlier_file_path = self.paths.outlier_file(freq, taskname, stage, cluster=True)
            if work_in_local_dir:
                outlier_file_path = Path(outlier_file_path).name
            
            cluster_hdul.writeto(outlier_file_path, overwrite=True)
            
        return outlier_file_path 

    def make_injection_outlier(self, taskname, freq, mean2f_th, n_jobs, num_top_list=1000, 
                               stage='search', freq_deriv_order=2, cluster=False, work_in_local_dir=False):
        """
        Public wrapper to write injection-search results.

        Parameters:
        - taskname: str
            The name of the task for the search, used in naming and organizing output files.
        - freq: float
            The frequency in Hz for which results are being written.
        - mean2f_th: numpy.ndarray
            The mean 2F threshold value for the injections.
        - n_jobs: int
            The number of jobs processed.
        - num_top_list: int, optional (default=1000)
            Maximum number of top outliers to keep for each job's results.
        - stage: str, optional (default='search')
            The stage of the analysis.
        - freq_deriv_order: int, optional (default=2)
            The order of frequency derivative.
        - cluster: bool, optional (default=False)
            If True, clusters outliers to consolidate similar results.
        - work_in_local_dir: bool, optional (default=False)
            If True, stores output files in the local directory.
        """
        outlier_file_path = self._make_injection_outlier(taskname, freq, mean2f_th, n_jobs, num_top_list, 
                                                         stage, freq_deriv_order, cluster, work_in_local_dir)
        print('Finish writing injection result for {0} Hz'.format(freq))
        return outlier_file_path

    def _make_followup_outlier(self, taskname, freq, mean2f_th, n_jobs, num_top_list_limit=1000, 
                               stage='search', freq_deriv_order=2, 
                               cluster=False, work_in_local_dir=True, inj=False,
                               chunk_index=0, chunk_size=1):
        """Writes the follow-up results for injections at a given frequency, supporting chunking."""
        
        outlier_table_list = []
        inj_table_list = []
        info_data = np.recarray((n_jobs,), dtype=[(key, '>f8') for key in ['freq', 'jobIndex', 'outliers']]) 
 
        # Determine job range based on chunk
        start_job = chunk_index * chunk_size + 1
        end_job = chunk_index * chunk_size + n_jobs + 1
        
        # Iterate over each job in the chunk
        for i, job_index in enumerate(range(start_job, end_job)):
            weave_file_path = self.paths.weave_output_file(freq, taskname, job_index, stage)
            if work_in_local_dir:
                weave_file_path = Path(weave_file_path).name
            
            try:
                weave_data = fits.getdata(weave_file_path, 1)
                spacing = get_spacing(weave_file_path, freq_deriv_order)
                
                # Note: mean2f_th is an array here, indexed by i
                _outlier = self.make_outlier_table(weave_data, spacing, mean2f_th[i], num_top_list_limit)
                
                if inj:
                    inj_param = fits.getdata(weave_file_path, 2)
                    _outlier, inj_param = self.make_injection_table(inj_param, _outlier)
                    
                if len(_outlier) > 0:
                    outlier_table_list.append(_outlier)
                    if inj:
                        inj_table_list.append(inj_param)
                else:
                    outlier_table_list.append(_outlier) 
                        
                info_data[i] = freq, job_index, len(_outlier)
            except FileNotFoundError:
                 info_data[i] = freq, job_index, 0

        # Combine Tables
        primary_hdu = fits.PrimaryHDU()
        
        if outlier_table_list:
            outlier_hdu = fits.BinTableHDU(data=vstack(outlier_table_list), name=stage+'_outlier')
        else:
            outlier_hdu = fits.BinTableHDU(name=stage+'_outlier')
            print('No outlier in follow-up chunk.')
        
        info_hdu = fits.BinTableHDU(data=info_data, name='info')
        
        # Construct HDU List
        outlier_hdul = fits.HDUList([primary_hdu, outlier_hdu])
        
        if inj and inj_table_list:
            inj_hdu = fits.BinTableHDU(data=vstack(inj_table_list), name='inj')
            outlier_hdul.append(inj_hdu)
        elif inj:
            outlier_hdul.append(fits.BinTableHDU(name='inj'))
            
        outlier_hdul.append(info_hdu)
        
        # Generate Output Path (Handle Chunk Naming)
        outlier_file_path = self.paths.outlier_file(freq, taskname, stage, cluster=(cluster and not inj))
        if work_in_local_dir:
            outlier_file_path = Path(outlier_file_path).name
            
        if chunk_size != 1:
            outlier_file_path = outlier_file_path.replace('.fts', f'_chunk{chunk_index}.fts')
            
        make_dir([outlier_file_path])
        outlier_hdul.writeto(outlier_file_path, overwrite=True)     
        
        # Clustering for Follow-Up
        if cluster and outlier_hdu.data.size > 1:
            cluster_hdul = fits.HDUList()
            centers_idx, cluster_size, cluster_member = clustering(outlier_hdu.data, freq_deriv_order) 
            
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
            outlier_file_path = self.paths.outlier_file(freq, taskname, stage, cluster=cluster)
            if chunk_size != 1:
                outlier_file_path = outlier_file_path.replace('.fts', f'_chunk{chunk_index}.fts')
            if work_in_local_dir:
                outlier_file_path = Path(outlier_file_path).name
                
            cluster_hdul.writeto(outlier_file_path, overwrite=True)
            
        return outlier_file_path 
    
    def make_followup_outlier(self, taskname, freq, mean2f_th, num_top_list=1000, 
                              new_stage='followUp-1', new_freq_deriv_order=2, 
                              cluster=False, work_in_local_dir=True, inj=False,
                              chunk_index=0, chunk_size=1, chunk_count=None):
        """
        Public wrapper to write follow-up results.

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
        print(f'Follow-up F-statistic threshold: {mean2f_th}')
        
        if chunk_count is not None:
            # Slice the threshold array for this specific chunk
            mean2f_th = mean2f_th[chunk_index*chunk_size : (chunk_index+1)*chunk_size]
            
        n_jobs = mean2f_th.size
        
        outlier_file_path = self._make_followup_outlier(taskname, freq, mean2f_th, n_jobs, num_top_list, 
                                                        new_stage, new_freq_deriv_order, 
                                                        cluster, work_in_local_dir, inj, 
                                                        chunk_index=chunk_index, chunk_size=chunk_size)

        print(f'Finish writing followUp result for {freq} Hz')
        return outlier_file_path


    def ensemble_outlier_chunk(self, chunk_count, taskname, freq, stage, cluster, work_in_local_dir):
        """
        Combines outlier results from multiple chunks into a single output file.

        Parameters:
        - chunk_count: int
            The number of chunks to process.
        - taskname: str
            The name of the task.
        - freq: float
            The frequency in Hz.
        - stage: str
            The current stage of the analysis.
        - cluster: bool
            If True, indicates that clustering results should be included.
        - work_in_local_dir: bool
            If True, indicates that paths should be treated as local directory paths.

        Returns:
        - outlier_file_path: str
            The path to the output file containing the combined outlier results.
        """
        outlier_file_path = self.paths.outlier_file(freq, taskname, stage, cluster=cluster)
        
        if work_in_local_dir:
            outlier_file_path = Path(outlier_file_path).name
            
        outlier_table_list = []
        info_table_list = []
        
        # Iterate through each chunk
        for i in range(chunk_count):
            # Construct chunk filename manually based on convention
            _outlier_file_path = outlier_file_path.replace('.fts', f'_chunk{i}.fts')
            
            try:
                # Read tables 
                outlier_table_list.append(Table(fits.getdata(_outlier_file_path, extname=stage+'_outlier')))  
                info_table_list.append(fits.getdata(_outlier_file_path, extname='info'))
            except FileNotFoundError:
                print(f"Warning: Chunk file {_outlier_file_path} missing.")

        # Stack and Write
        outlier_hdul = fits.HDUList()
        # Grab header from first available chunk if possible
        if chunk_count > 0:
             # Re-construct first chunk path
            first_chunk = outlier_file_path.replace('.fts', '_chunk0.fts')
            try:
                primary_hdu = fits.PrimaryHDU(header=fits.getheader(first_chunk))
            except FileNotFoundError:
                primary_hdu = fits.PrimaryHDU()
        else:
            primary_hdu = fits.PrimaryHDU()
            
        outlier_hdul.append(primary_hdu)
        
        if outlier_table_list:
            outlier_hdu = fits.BinTableHDU(data=vstack(outlier_table_list), name=stage+'_outlier')
            outlier_hdul.append(outlier_hdu)
        
        if info_table_list:
            info_hdu = fits.BinTableHDU(data=np.hstack(info_table_list), name='info')
            outlier_hdul.append(info_hdu) 
            
        outlier_hdul.writeto(outlier_file_path, overwrite=True)
        return outlier_file_path

    def ensemble_followup_result(self, freq, taskname, stage, inj_stage, outlier_file_path_list, inj_outlier_file_path_list, 
                                 mean2f_ratio_list, num_top_list_to_follow_up_list,
                                 final_stage, cluster=False, work_in_local_dir=False):
        """Combines results from multiple follow-up stages into one summary FITS file."""
        n_inj_table = len(inj_outlier_file_path_list)
        n_out_table = len(outlier_file_path_list)
        
        primary_hdu = fits.PrimaryHDU()
        outlier_hdul = fits.HDUList()
        
        # Metadata
        try:
            # Try to get threshold from the first available file
            source_file = outlier_file_path_list[0] if n_out_table > 0 else (inj_outlier_file_path_list[0] if n_inj_table > 0 else None)
            if source_file:
                mean2f_th = fits.getheader(source_file)['HIERARCH mean2F_th']
                primary_hdu.header['HIERARCH mean2F_th'] = mean2f_th
        except (IndexError, KeyError, FileNotFoundError):
            print("Warning: Unable to retrieve mean2F_th from header.")
            pass

        primary_hdu.header['HIERARCH injection_test'] = (n_inj_table != 0)
            
        # Record ratios and top lists for every stage in the header
        # We iterate up to the max number of stages provided
        max_stages = max(n_inj_table, n_out_table)
        
        # Note: The loop index 'i' corresponds to the follow-up stage index. 
        # Typically stage lists include the initial search, so we might offset by 1 if 'stage' list includes 'search' at index 0.
        for i in range(max_stages):
            # Check bounds for ratio list
            if i < len(mean2f_ratio_list):
                # We use stage[i+1] assuming the lists passed in include the initial search stage name at 0
                stage_name = stage[i+1] if (i+1) < len(stage) else f"stage_{i+1}"
                primary_hdu.header[f'HIERARCH mean2F_ratio_{stage_name}'] = mean2f_ratio_list[i]
            
            # Check bounds for top list
            if i < len(num_top_list_to_follow_up_list):
                stage_name = stage[i+1] if (i+1) < len(stage) else f"stage_{i+1}"
                primary_hdu.header[f'HIERARCH numTopList_{stage_name}'] = num_top_list_to_follow_up_list[i]
                
        outlier_hdul.append(primary_hdu)
                 
        # 1. Append Injection Follow-up Stages
        for i in range(n_inj_table):           
            try:
                # Outliers
                data = fits.getdata(inj_outlier_file_path_list[i], extname=inj_stage[i]+'_outlier')
                outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_outlier'))
                
                # Injections
                data = fits.getdata(inj_outlier_file_path_list[i], extname='inj') 
                outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_inj'))

                # Info
                data = fits.getdata(inj_outlier_file_path_list[i], extname='info')
                outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_info'))
            except FileNotFoundError:
                print(f"Warning: Missing injection file {inj_outlier_file_path_list[i]}")

        # 2. Append Search Follow-up Stages
        for i in range(n_out_table):
            try:
                data = fits.getdata(outlier_file_path_list[i], extname=stage[i]+'_outlier')
                outlier_hdul.append(fits.BinTableHDU(data=data, name=stage[i]+'_outlier'))

                data = fits.getdata(outlier_file_path_list[i], extname='info')
                outlier_hdul.append(fits.BinTableHDU(data=data, name=stage[i]+'_info'))
            except FileNotFoundError:
                print(f"Warning: Missing outlier file {outlier_file_path_list[i]}")
        
        # Write Final Ensemble File
        outlier_file_path = self.paths.outlier_file(freq, taskname, final_stage, cluster=cluster)    
        
        if work_in_local_dir:
            outlier_file_path = Path(outlier_file_path).name        
        else:
            make_dir([outlier_file_path])    
        
        outlier_hdul.writeto(outlier_file_path, overwrite=True)
        return outlier_file_path