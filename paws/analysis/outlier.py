from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from tqdm import tqdm

from paws.definitions import phase_param_name
from paws.filepaths import PathManager
from paws.io import get_spacing, make_dir

from .clustering import clustering


class ResultAnalysisManager:
    """
    Manages the collection, filtering, and storage of search results.
    """

    def __init__(self, config, target):
        """
        Initialize the ResultManager.

        Parameters:
            config (paws.settings.Config)
            target (paws.settings.Target)
        """
        self.config = config
        self.target = target
        self.paths = PathManager(config, target)

    def make_outlier_table(self, data, mean2f_th, num_toplist=1000):
        """Filters data to create an outlier table."""
        # Read and limit the data to the top entries
        data = data[:num_toplist]

        # Mask data with mean 2F values greater than the threshold
        mask = data["mean2F"] >= mean2f_th
        data = Table(data[mask])
        data.add_column(mean2f_th * np.ones(len(data)), name="mean2F threshold")

        return data

    def make_injection_table(self, inj_param, search_param):
        """Creates a table comparing injections with search results."""
        inj_param = Table(inj_param)

        # Calculate h0 from aPlus and aCross
        aplus, across = inj_param["aPlus"], inj_param["aCross"]
        h0 = aplus + np.sqrt(aplus**2 - across**2)
        inj_param.add_column(h0 * np.ones(len(inj_param)), name="h0")

        # Rename reference time if exists
        if "refTime_s" in inj_param.colnames:
            inj_param.rename_column("refTime_s", "refTime")

        search_param = Table(search_param)[:1]

        return search_param, inj_param

    def _collect_outlier_data(
        self,
        taskname,
        freq,
        stage,
        job_indices,
        mean2f_th,
        num_toplist,
        freq_deriv_order,
        n_sky=1,
        work_in_local_dir=False,
        max_workers=32,
        read_inj=False,
        separate_saturated=False,
        desc="Processing",
    ):
        """Central engine for multithreaded FITS reading and outlier filtering."""
        # 1. Handle scalar vs array thresholds
        if np.isscalar(mean2f_th):
            thresholds = [mean2f_th] * len(job_indices)
        else:
            # Follow-up provides an array of thresholds (one per parameter point); expand for sky points
            thresholds = (
                np.repeat(mean2f_th, n_sky).tolist() if n_sky > 1 else list(mean2f_th)
            )

        outlier_table_list = []
        sat_outlier_table_list = []
        inj_table_list = []
        info_list = []  # Stores (freq, job_idx, n_outliers, is_saturated)
        max_spacing = {}

        # 2. Universal Worker Function
        def _worker(args):
            i, job_idx, th, worker_read_inj = args
            file_path = self.paths.weave_output_file(freq, taskname, job_idx, stage)
            if work_in_local_dir:
                file_path = Path(file_path).name

            try:
                weave_data = fits.getdata(file_path, 1)
                spacing = get_spacing(file_path, freq_deriv_order)

                _outlier = self.make_outlier_table(weave_data, th, num_toplist)
                is_sat = int(len(_outlier) >= num_toplist)

                # Handle injections
                _inj_param = None
                if worker_read_inj and _outlier is not None:
                    inj_data = fits.getdata(file_path, 2)
                    _outlier, _inj_param = self.make_injection_table(inj_data, _outlier)

                return (i, job_idx, _outlier, _inj_param, spacing, is_sat)
            except FileNotFoundError:
                return (i, job_idx, None, None, None, 0)

        # 3. Multithreading Queue
        job_args = [
            (i, idx, th, read_inj)
            for i, idx, th in zip(range(len(job_indices)), job_indices, thresholds)
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(_worker, job_args),
                    total=len(job_args),
                    desc=f"{desc} {freq}Hz",
                )
            )

        # 4. Safe Sequential Unpacking
        # Group results: 1 item per group when n_sky=1, n_sky items per parameter point otherwise.
        # Candidates from all sky-point jobs in a group are pooled and trimmed to num_toplist.
        if n_sky == 1:
            groups = [[r] for r in results]
        else:
            n_groups = len(results) // n_sky
            groups = [results[g * n_sky : (g + 1) * n_sky] for g in range(n_groups)]

        missing_files = 0
        for group in groups:
            group_outliers = []
            group_inj = None
            group_spacing = None
            first_job_idx = group[0][1]

            for res in group:
                _, job_idx, _outlier, _inj_param, spacing, is_sat = res
                if _outlier is None:
                    missing_files += 1
                    continue
                if len(_outlier) > 0:
                    group_outliers.append(_outlier)
                if group_inj is None and _inj_param is not None:
                    group_inj = _inj_param
                if group_spacing is None and spacing is not None:
                    group_spacing = spacing

            if group_outliers:
                # Pool all candidates, keep top num_toplist by mean2F
                merged = vstack(group_outliers)
                merged.sort("mean2F")
                merged.reverse()
                merged = merged[:num_toplist]
                is_sat = int(len(merged) >= num_toplist)
                info_list.append((freq, first_job_idx, len(merged), is_sat))

                if separate_saturated and is_sat == 1:
                    sat_outlier_table_list.append(merged[:1])  # Keep ONLY the loudest 1
                else:
                    outlier_table_list.append(merged)

                if group_inj is not None:
                    inj_table_list.append(group_inj)

                if group_spacing is not None:
                    if not max_spacing:
                        max_spacing = group_spacing.copy()
                    else:
                        for k, v in group_spacing.items():
                            max_spacing[k] = max(max_spacing.get(k, 0), v)
            else:
                info_list.append((freq, first_job_idx, 0, 0))

        if missing_files > 0:
            print(f"Warning: {missing_files} files missing for {desc} {freq}Hz")

        return (
            outlier_table_list,
            sat_outlier_table_list,
            inj_table_list,
            info_list,
            max_spacing,
        )

    def _write_clustered_results(
        self,
        freq,
        taskname,
        stage,
        outlier_data,
        freq_deriv_order,
        primary_hdu,
        inj_hdu=None,
        non_sat_hdu=None,
        work_in_local_dir=False,
    ):
        """Central engine for clustering outliers and writing the clustered FITS file."""

        cluster_hdul = fits.HDUList()

        if outlier_data is None or len(outlier_data) == 0:
            # Create an empty outlier table (preserves column schema if outlier_data is an empty recarray)
            if outlier_data is not None:
                cluster_hdu = fits.BinTableHDU(
                    data=outlier_data, name=stage + "_outlier"
                )
            else:
                cluster_hdu = fits.BinTableHDU(name=stage + "_outlier")

            # Create an empty info table
            dtypes = [
                (key, ">f8") for key in ["freq", "clusterIndex", "noOutliersWithin"]
            ]
            info_data = np.recarray((0,), dtype=dtypes)  # 0 rows!
            info_clustered_hdu = fits.BinTableHDU(data=info_data, name="info_clustered")

        else:
            # 1. Run the clustering algorithm
            _, dfn = phase_param_name(freq_deriv_order)
            spacing = {key: primary_hdu.header[f"HIERARCH {key}"] for key in dfn}

            centers_idx, cluster_size, cluster_member = clustering(
                outlier_data, spacing, self.config.cluster_n_spacing
            )

            # 2. Map Data (Handle Injection vs Standard)
            if inj_hdu is not None:
                # Injection mode: map every outlier to a cluster center so we don't lose injection tracking
                center_idx_for_each_outlier = np.full(outlier_data.size, -1)
                processed_indices = set()
                for ci, members in zip(centers_idx, cluster_member):
                    idx = np.array(
                        [item for item in members if item not in processed_indices]
                    )
                    if len(idx) > 0:
                        center_idx_for_each_outlier[idx] = ci
                        processed_indices.update(members)
                cluster_data = outlier_data[center_idx_for_each_outlier]
            else:
                # Standard mode: just grab the centers
                cluster_data = outlier_data[centers_idx]

            cluster_hdu = fits.BinTableHDU(data=cluster_data, name=stage + "_outlier")

            # 3. Build Info Table
            dtypes = [
                (key, ">f8") for key in ["freq", "clusterIndex", "noOutliersWithin"]
            ]
            info_data = np.recarray((len(cluster_size),), dtype=dtypes)
            for i in range(len(cluster_size)):
                info_data[i] = freq, i, cluster_size[i]

            info_clustered_hdu = fits.BinTableHDU(data=info_data, name="info_clustered")

        # 4. Assemble Final HDUList
        cluster_hdul.append(primary_hdu)
        cluster_hdul.append(cluster_hdu)
        if inj_hdu is not None:
            cluster_hdul.append(inj_hdu)
        if non_sat_hdu is not None:
            cluster_hdul.append(non_sat_hdu)
        cluster_hdul.append(info_clustered_hdu)

        # 5. File Path Logic
        outlier_file_path = self.paths.outlier_file(freq, taskname, stage, cluster=True)

        if work_in_local_dir:
            outlier_file_path = Path(outlier_file_path).name

        cluster_hdul.writeto(outlier_file_path, overwrite=True)
        return outlier_file_path

    def make_outlier(
        self,
        taskname,
        freq,
        mean2f_th,
        n_jobs,
        num_toplist=1000,
        stage="search",
        freq_deriv_order=2,
        n_sky=1,
        cluster=False,
        work_in_local_dir=False,
        separate_saturated=False,
        is_injection=False,
        max_workers=32,
        param_indices=None,
    ):
        """
        Unified engine to collect results and write FITS files for Search, Injection, or Follow-up.

        param_indices: optional 0-based indices of the parameter points to
        analyse (None = all n_jobs points). Each point owns n_sky consecutive
        Weave jobs, so only the result files of the selected points are read,
        and an array-valued mean2f_th is subset to match.
        """
        # 1. Saturation Warning
        if "search" in stage.lower() and not separate_saturated:
            print(
                f"Warning: '{stage}' appears to be a search stage, but separate_saturated is False. "
                "Saturated bands will NOT be separated."
            )

        if is_injection and separate_saturated:
            print(
                f"Warning: Injection test running with separate_saturated=True. "
                f"Since num_toplist ({num_toplist}) is often set low for injections to save cost, "
                "this may incorrectly separate valid bands as saturated."
            )

        # Each parameter point maps to n_sky consecutive Weave jobs
        if param_indices is None:
            job_indices = list(range(1, n_jobs * n_sky + 1))
        else:
            param_indices = np.asarray(param_indices, dtype=int)
            job_indices = (
                (param_indices[:, None] * n_sky + np.arange(1, n_sky + 1))
                .ravel()
                .tolist()
            )
            if not np.isscalar(mean2f_th):
                mean2f_th = np.asarray(mean2f_th)[param_indices]
            n_jobs = param_indices.size

        # 2. Collect data using your central engine
        outliers, sat_outliers, inj_data_list, info_list, max_spacing = (
            self._collect_outlier_data(
                taskname,
                freq,
                stage,
                job_indices,
                mean2f_th,
                num_toplist,
                freq_deriv_order,
                n_sky=n_sky,
                work_in_local_dir=work_in_local_dir,
                max_workers=max_workers,
                read_inj=is_injection,
                separate_saturated=separate_saturated,
                desc=stage.capitalize(),
            )
        )

        # 3. Build Primary HDU and Header
        primary_hdu = fits.PrimaryHDU()

        is_scalar_th = np.isscalar(mean2f_th)
        if is_scalar_th:
            primary_hdu.header["HIERARCH mean2F_th"] = mean2f_th

        if max_spacing:
            for key, val in max_spacing.items():
                primary_hdu.header[f"HIERARCH {key}"] = val

        hdus = [primary_hdu]

        # 4. Add Outlier HDU
        if outliers:
            hdus.append(
                fits.BinTableHDU(data=vstack(outliers), name=f"{stage}_outlier")
            )
        else:
            hdus.append(fits.BinTableHDU(name=f"{stage}_outlier"))

        # 5. Add Stage-Specific HDUs
        # A) Injection Data
        if is_injection:
            if inj_data_list:
                hdus.append(
                    fits.BinTableHDU(data=vstack(inj_data_list), name="injection")
                )
            else:
                hdus.append(fits.BinTableHDU(name="injection"))

        # B) Saturated Data
        if separate_saturated:
            if sat_outliers:
                hdus.append(
                    fits.BinTableHDU(
                        data=vstack(sat_outliers), name=f"{stage}_sat_outlier"
                    )
                )
            else:
                hdus.append(fits.BinTableHDU(name=f"{stage}_sat_outlier"))

        # 6. Build and Add Info HDU dynamically based on threshold type
        info_cols = [
            ("freq", ">f8"),
            ("jobIndex", ">f8"),
            ("outliers", ">f8"),
            ("isSaturated", ">f8"),
        ]
        info_data = np.recarray((n_jobs,), dtype=info_cols)

        for i, (f, j, o, s) in enumerate(info_list):
            info_data[i] = (f, j, o, s)

        hdus.append(fits.BinTableHDU(data=info_data, name="info"))

        # 7. Build and Add Non-Saturated Band HDU (Search specific)
        if "search" in stage.lower():
            f0_band = self.config.f0_band
            sat_matrix = info_data["isSaturated"].reshape(int(1.0 / f0_band), -1)
            idx = np.where(~sat_matrix.any(axis=1))[0]
            non_sat_data = np.recarray((len(idx),), dtype=[("non_sat_band", ">f8")])
            non_sat_data["non_sat_band"] = int(freq) + idx * f0_band
            hdus.append(fits.BinTableHDU(data=non_sat_data, name="non_sat_band"))

        # 8. Write Initial File
        outlier_hdul = fits.HDUList(hdus)
        outlier_file_path = self.paths.outlier_file(
            freq, taskname, stage, cluster=False
        )
        if work_in_local_dir:
            outlier_file_path = Path(outlier_file_path).name

        make_dir([outlier_file_path])
        outlier_hdul.writeto(outlier_file_path, overwrite=True)

        # 9. Handle Clustering
        if cluster and hdus[1].data is not None:
            primary_hdu.header["HIERARCH cluster_n_spacing"] = self.config.cluster_n_spacing

            inj_hdu_to_pass = next((h for h in hdus if h.name == "INJECTION"), None)
            non_sat_hdu_to_pass = next(
                (h for h in hdus if h.name == "NON_SAT_BAND"), None
            )

            outlier_file_path = self._write_clustered_results(
                freq,
                taskname,
                stage,
                hdus[1].data,
                freq_deriv_order,
                primary_hdu,
                inj_hdu=inj_hdu_to_pass,
                non_sat_hdu=non_sat_hdu_to_pass,
                work_in_local_dir=work_in_local_dir,
            )

        print(f"Finished writing {stage} result for {freq} Hz")
        return outlier_file_path

    # def ensemble_followup_result(self, freq, taskname, stage, inj_stage, outlier_file_path_list, inj_outlier_file_path_list,
    #                              mean2f_ratio_list, num_toplist_list,
    #                              final_stage, cluster=False, work_in_local_dir=False):
    #     """Combines results from multiple follow-up stages into one summary FITS file."""
    #     n_inj_table = len(inj_outlier_file_path_list)
    #     n_out_table = len(outlier_file_path_list)

    #     primary_hdu = fits.PrimaryHDU()
    #     outlier_hdul = fits.HDUList()

    #     # Metadata
    #     try:
    #         # Try to get threshold from the first available file
    #         source_file = outlier_file_path_list[0] if n_out_table > 0 else (inj_outlier_file_path_list[0] if n_inj_table > 0 else None)
    #         if source_file:
    #             mean2f_th = fits.getheader(source_file)['HIERARCH mean2F_th']
    #             primary_hdu.header['HIERARCH mean2F_th'] = mean2f_th
    #     except (IndexError, KeyError, FileNotFoundError):
    #         print("Warning: Unable to retrieve mean2F_th from header.")
    #         pass

    #     primary_hdu.header['HIERARCH injection_test'] = (n_inj_table != 0)

    #     # Record ratios and top lists for every stage in the header
    #     # We iterate up to the max number of stages provided
    #     max_stages = max(n_inj_table, n_out_table)

    #     # Note: The loop index 'i' corresponds to the follow-up stage index.
    #     # Typically stage lists include the initial search, so we might offset by 1 if 'stage' list includes 'search' at index 0.
    #     for i in range(max_stages):
    #         # Check bounds for ratio list
    #         if i < len(mean2f_ratio_list):
    #             # We use stage[i+1] assuming the lists passed in include the initial search stage name at 0
    #             stage_name = stage[i+1] if (i+1) < len(stage) else f"stage_{i+1}"
    #             primary_hdu.header[f'HIERARCH mean2F_ratio_{stage_name}'] = mean2f_ratio_list[i]

    #         # Check bounds for top list
    #         if i < len(num_toplist_list):
    #             stage_name = stage[i+1] if (i+1) < len(stage) else f"stage_{i+1}"
    #             primary_hdu.header[f'HIERARCH numTopList_{stage_name}'] = num_toplist_list[i]

    #     outlier_hdul.append(primary_hdu)

    #     # 1. Append Injection Follow-up Stages
    #     for i in range(n_inj_table):
    #         try:
    #             # Outliers
    #             data = fits.getdata(inj_outlier_file_path_list[i], extname=inj_stage[i]+'_outlier')
    #             outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_outlier'))

    #             # Injections
    #             data = fits.getdata(inj_outlier_file_path_list[i], extname='inj')
    #             outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_inj'))

    #             # Info
    #             data = fits.getdata(inj_outlier_file_path_list[i], extname='info')
    #             outlier_hdul.append(fits.BinTableHDU(data=data, name=inj_stage[i]+'_info'))
    #         except FileNotFoundError:
    #             print(f"Warning: Missing injection file {inj_outlier_file_path_list[i]}")

    #     # 2. Append Search Follow-up Stages
    #     for i in range(n_out_table):
    #         try:
    #             data = fits.getdata(outlier_file_path_list[i], extname=stage[i]+'_outlier')
    #             outlier_hdul.append(fits.BinTableHDU(data=data, name=stage[i]+'_outlier'))

    #             data = fits.getdata(outlier_file_path_list[i], extname='info')
    #             outlier_hdul.append(fits.BinTableHDU(data=data, name=stage[i]+'_info'))
    #         except FileNotFoundError:
    #             print(f"Warning: Missing outlier file {outlier_file_path_list[i]}")

    #     # Write Final Ensemble File
    #     outlier_file_path = self.paths.outlier_file(freq, taskname, final_stage, cluster=cluster)

    #     if work_in_local_dir:
    #         outlier_file_path = Path(outlier_file_path).name
    #     else:
    #         make_dir([outlier_file_path])

    #     outlier_hdul.writeto(outlier_file_path, overwrite=True)
    #     return outlier_file_path
