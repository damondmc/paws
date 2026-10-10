from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from tqdm import tqdm

from paws.definitions import phase_param_name
from paws.filepaths import PathManager, make_dir
from paws.fits_io import weave_header_spacing, weave_template_spacing

from .clustering import clustering


def read_loudest_row(weave_result_path):
    """First row of a Weave toplist (toplists are sorted by mean2F)."""
    with fits.open(weave_result_path, memmap=True) as hdul:
        toplist = hdul[1].data
        if toplist is None or len(toplist) == 0:
            raise ValueError(f"empty Weave toplist: {weave_result_path}")
        return np.array(toplist[:1])


def loudest_rows(result_files_per_seed, n_threads):
    """Loudest candidate of each seed over its Weave result files (one equal-length file list per seed).
    Reads only row 0 of each file, which gives the same result as make_outlier(num_toplist=1)."""
    n_seeds = len(result_files_per_seed)
    all_files = [path for seed_files in result_files_per_seed for path in seed_files]
    with ThreadPoolExecutor(n_threads) as executor:
        rows = np.concatenate(list(executor.map(read_loudest_row, all_files))).reshape(
            n_seeds, -1
        )
    return rows[np.arange(n_seeds), rows["mean2F"].argmax(axis=1)]


def make_outlier_table(data, mean2f_th, num_toplist):
    """The first num_toplist candidates of a toplist with mean2F >= mean2f_th, plus a "mean2F threshold" column."""
    data = data[:num_toplist]
    mask = data["mean2F"] >= mean2f_th
    data = Table(data[mask])
    data.add_column(mean2f_th * np.ones(len(data)), name="mean2F threshold")
    return data


def make_injection_table(inj_param, search_param):
    """(loudest search candidate, injection parameters with h0 added) of one Weave run with an injection."""
    inj_param = Table(inj_param)
    aplus, across = inj_param["aPlus"], inj_param["aCross"]
    h0 = aplus + np.sqrt(aplus**2 - across**2)
    inj_param.add_column(h0 * np.ones(len(inj_param)), name="h0")
    if "refTime_s" in inj_param.colnames:
        inj_param.rename_column("refTime_s", "refTime")
    search_param = Table(search_param)[:1]
    return search_param, inj_param


def read_job_outliers(job):
    """Outliers of one Weave result file. job: (index, job index, path, mean2F threshold, num_toplist,
    freq_deriv_order, read injections). Returns (index, job index, outlier table, injection table, spacing,
    is saturated); the tables and spacing are None when the file is missing."""
    i, job_idx, file_path, th, num_toplist, freq_deriv_order, read_inj = job
    try:
        # one open per file: toplist, header spacing and injections
        with fits.open(file_path) as hdul:
            spacing = weave_header_spacing(hdul[0].header, freq_deriv_order)
            outliers = make_outlier_table(hdul[1].data, th, num_toplist)
            is_sat = int(len(outliers) >= num_toplist)
            injections = None
            if read_inj:
                outliers, injections = make_injection_table(hdul[2].data, outliers)
        return (i, job_idx, outliers, injections, spacing, is_sat)
    except FileNotFoundError:
        return (i, job_idx, None, None, None, 0)


def read_jobs_with_threads(jobs, n_threads):
    """read_job_outliers of each job, in job order."""
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        return list(executor.map(read_job_outliers, jobs))


def read_jobs(jobs, n_processes, n_threads, desc):
    """read_job_outliers of each job, in job order, over n_processes processes of n_threads threads each."""
    if n_processes == 1:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            return list(
                tqdm(executor.map(read_job_outliers, jobs), total=len(jobs), desc=desc)
            )
    # ~4 chunks per process, for the progress bar
    chunk_size = max(1, -(-len(jobs) // (4 * n_processes)))
    chunks = [
        jobs[start : start + chunk_size] for start in range(0, len(jobs), chunk_size)
    ]
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        chunk_results = executor.map(
            read_jobs_with_threads, chunks, [n_threads] * len(chunks)
        )
        return [
            result
            for results in tqdm(chunk_results, total=len(chunks), desc=desc)
            for result in results
        ]


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
        n_processes=1,
        max_workers=32,
        read_inj=False,
        separate_saturated=False,
        desc="Processing",
    ):
        """Central engine for parallel FITS reading and outlier filtering."""
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

        # 2. Read every result file (n_processes x max_workers in parallel), results in job order
        jobs = []
        for i, (job_idx, th) in enumerate(zip(job_indices, thresholds)):
            file_path = self.paths.weave_output_file(freq, taskname, job_idx, stage)
            if work_in_local_dir:
                file_path = Path(file_path).name
            jobs.append(
                (i, job_idx, file_path, th, num_toplist, freq_deriv_order, read_inj)
            )
        results = read_jobs(jobs, n_processes, max_workers, f"{desc} {freq}Hz")

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
        outlier_file_path = self.paths.outlier_file(
            freq, taskname, stage, cluster=True, location="home"
        )

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
        n_processes,
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

        n_processes: processes reading the result files, each with max_workers threads.
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
                n_processes=n_processes,
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
            freq, taskname, stage, cluster=False, location="home"
        )
        if work_in_local_dir:
            outlier_file_path = Path(outlier_file_path).name

        make_dir([outlier_file_path])
        outlier_hdul.writeto(outlier_file_path, overwrite=True)

        # 9. Handle Clustering
        if cluster and hdus[1].data is not None:
            primary_hdu.header["HIERARCH cluster_n_spacing"] = (
                self.config.cluster_n_spacing
            )

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

    def write_loudest_outliers(
        self,
        taskname,
        freq,
        stage,
        freq_deriv_order,
        n_seeds,
        loudest_per_seed,
        passed,
        mean2f_threshold,
        spacing_files,
    ):
        """
        Unclustered and clustered outlier files of a stage collected as the loudest candidate per seed
        (see loudest_rows); INFO has one row per seed.

        passed: seeds kept as outliers. spacing_files: Weave result files whose template spacings are
        combined (maximum) into the primary header.
        """
        spacing = {}
        for path in spacing_files:
            for key, value in weave_template_spacing(path, freq_deriv_order).items():
                spacing[key] = max(spacing.get(key, 0), value)
        primary_hdu = fits.PrimaryHDU()
        for key, value in spacing.items():
            primary_hdu.header[f"HIERARCH {key}"] = value

        outlier_table = Table(loudest_per_seed[passed])
        outlier_table.add_column(mean2f_threshold[passed], name="mean2F threshold")
        hdu_name = f"{stage}_outlier"
        if passed.any():
            outlier_hdu = fits.BinTableHDU(data=outlier_table, name=hdu_name)
        else:
            outlier_hdu = fits.BinTableHDU(name=hdu_name)

        info_columns = ("freq", "jobIndex", "outliers", "isSaturated")
        info = np.recarray(
            (n_seeds,), dtype=[(column, ">f8") for column in info_columns]
        )
        info["freq"], info["jobIndex"], info["outliers"], info["isSaturated"] = (
            freq,
            np.arange(n_seeds),
            passed,
            0,
        )

        path = self.paths.outlier_file(
            freq, taskname, stage, cluster=False, location="home"
        )
        make_dir([path])
        fits.HDUList(
            [primary_hdu, outlier_hdu, fits.BinTableHDU(data=info, name="info")]
        ).writeto(path, overwrite=True)
        primary_hdu.header["HIERARCH cluster_n_spacing"] = self.config.cluster_n_spacing
        return self._write_clustered_results(
            freq, taskname, stage, outlier_hdu.data, freq_deriv_order, primary_hdu
        )
