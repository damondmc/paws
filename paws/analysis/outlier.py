"""Outlier collection: the loudest candidates of Weave result files, clustered, as outlier FITS files."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from jaxtyping import Bool, Float, Shaped
from tqdm import tqdm

from paws.definitions import phase_param_name
from paws.filepaths import PathManager, make_dir
from paws.fits_io import weave_header_spacing, weave_template_spacing
from paws.settings import FixedSettings

from .clustering import clustering

INFO_COLUMNS = ("freq", "jobIndex", "outliers", "isSaturated")


class ReadJob(NamedTuple):
    """One Weave result file to read."""

    job_index: int
    path: Path
    mean2f_threshold: float
    n_keep: int  # candidates kept from the toplist
    order: int  # spin-down order of the template spacings
    read_injection: bool  # also read the injection parameters (HDU 2)


class ReadResult(NamedTuple):
    """What was read from one Weave result file."""

    job_index: int
    outliers: Table  # the first `n_keep` candidates above the threshold
    injection: Optional[Table]  # the injection parameters, with h0
    spacing: dict[str, float]  # template spacings of the header


class OutlierCollection(NamedTuple):
    """Outliers of a band's Weave result files, pooled per parameter point."""

    outliers: list[Table]  # per point with outliers (not saturated when separated)
    saturated: list[Table]  # loudest candidate of each saturated point, when separated
    injections: list[Table]  # per point with outliers of an injection stage
    info: list[tuple[int, int, int, int]]  # per point: INFO_COLUMNS
    spacing: dict[str, float]  # largest template spacings


def read_loudest_row(weave_result_path: Path) -> Shaped[np.ndarray, "1"]:
    """First row of a Weave toplist (toplists are sorted by mean2F)."""
    with fits.open(weave_result_path, memmap=True) as hdul:
        toplist = hdul[1].data
        if toplist is None or len(toplist) == 0:
            raise ValueError(f"empty Weave toplist: {weave_result_path}")
        return np.array(toplist[:1])


def loudest_rows(
    result_files_per_seed: list[list[Path]], n_threads: int
) -> Shaped[np.ndarray, " n_seed"]:
    """Loudest candidate of each seed over its Weave result files (one equal-length file list per seed).
    Reads only row 0 of each file, which gives the same result as make_outlier(n_keep=1)."""
    n_seeds = len(result_files_per_seed)
    all_files = [path for seed_files in result_files_per_seed for path in seed_files]
    with ThreadPoolExecutor(n_threads) as executor:
        rows = np.concatenate(list(executor.map(read_loudest_row, all_files))).reshape(
            n_seeds, -1
        )
    return rows[np.arange(n_seeds), rows["mean2F"].argmax(axis=1)]


def cached_loudest_rows(
    cache_path: Path, result_files_per_seed: list[list[Path]], n_threads: int
) -> Shaped[np.ndarray, " n_seed"]:
    """loudest_rows of the seeds, computed and saved to cache_path (.npy) the first time, loaded from it after."""
    if not cache_path.exists():
        rows = loudest_rows(result_files_per_seed, n_threads)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(cache_path.stem + ".tmp.npy")
        np.save(temporary_path, rows)
        temporary_path.replace(cache_path)
    return np.load(cache_path)


def make_outlier_table(
    data: Shaped[np.ndarray, " n_row"], mean2f_threshold: float, n_keep: int
) -> Table:
    """The first `n_keep` candidates of a toplist with mean2F >= mean2f_threshold, plus a "mean2F threshold" column."""
    data = data[:n_keep]
    data = Table(data[data["mean2F"] >= mean2f_threshold])
    data.add_column(mean2f_threshold * np.ones(len(data)), name="mean2F threshold")
    return data


def make_injection_table(
    inj_param: Shaped[np.ndarray, " n_row"], search_param: Table
) -> tuple[Table, Table]:
    """(loudest search candidate, injection parameters with h0 added) of one Weave run with an injection."""
    inj_param = Table(inj_param)
    aplus, across = inj_param["aPlus"], inj_param["aCross"]
    h0 = aplus + np.sqrt(aplus**2 - across**2)
    inj_param.add_column(h0 * np.ones(len(inj_param)), name="h0")
    if "refTime_s" in inj_param.colnames:
        inj_param.rename_column("refTime_s", "refTime")
    search_param = Table(search_param)[:1]
    return search_param, inj_param


def read_job_outliers(job: ReadJob) -> ReadResult:
    """Outliers of one Weave result file; a missing file raises FileNotFoundError."""
    with fits.open(job.path) as hdul:
        spacing = weave_header_spacing(hdul[0].header, job.order)
        outliers = make_outlier_table(hdul[1].data, job.mean2f_threshold, job.n_keep)
        injection = None
        if job.read_injection:
            outliers, injection = make_injection_table(hdul[2].data, outliers)
    return ReadResult(job.job_index, outliers, injection, spacing)


def read_jobs_with_threads(jobs: list[ReadJob], n_threads: int) -> list[ReadResult]:
    """read_job_outliers of each job, in job order."""
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        return list(executor.map(read_job_outliers, jobs))


def read_jobs(
    jobs: list[ReadJob], n_processes: int, n_threads: int, desc: str
) -> list[ReadResult]:
    """read_job_outliers of each job, in job order, over n_processes processes of n_threads threads each."""
    if n_processes == 1:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            return list(
                tqdm(
                    executor.map(read_job_outliers, jobs),
                    total=len(jobs),
                    desc=desc,
                    unit="file",
                )
            )
    # ~4 chunks per process; the progress bar counts Weave result files
    chunk_size = max(1, -(-len(jobs) // (4 * n_processes)))
    chunks = [
        jobs[start : start + chunk_size] for start in range(0, len(jobs), chunk_size)
    ]
    outliers = []
    with (
        ProcessPoolExecutor(max_workers=n_processes) as executor,
        tqdm(total=len(jobs), desc=desc, unit="file") as progress,
    ):
        chunk_results = executor.map(
            read_jobs_with_threads, chunks, [n_threads] * len(chunks)
        )
        for results in chunk_results:
            outliers.extend(results)
            progress.update(len(results))
    return outliers


def pool_outliers(
    results: list[ReadResult],
    freq: int,
    n_sky: int,
    n_keep: int,
    separate_saturated: bool,
) -> OutlierCollection:
    """Pools the results of each parameter point's n_sky consecutive jobs, keeping the loudest `n_keep` candidates;
    a point with `n_keep` candidates is saturated."""
    collection = OutlierCollection([], [], [], [], {})
    for first in range(0, len(results), n_sky):
        group = results[first : first + n_sky]
        first_job = group[0].job_index
        tables = [result.outliers for result in group if len(result.outliers) > 0]
        if not tables:
            collection.info.append((freq, first_job, 0, 0))
            continue
        merged = vstack(tables)
        merged.sort("mean2F")
        merged.reverse()
        merged = merged[:n_keep]
        is_saturated = int(len(merged) >= n_keep)
        collection.info.append((freq, first_job, len(merged), is_saturated))
        if separate_saturated and is_saturated:
            collection.saturated.append(merged[:1])
        else:
            collection.outliers.append(merged)
        injection = next((r.injection for r in group if r.injection is not None), None)
        if injection is not None:
            collection.injections.append(injection)
        for key, value in group[0].spacing.items():
            collection.spacing[key] = max(collection.spacing.get(key, 0), value)
    return collection


def table_hdu(tables: list[Table], name: str) -> fits.BinTableHDU:
    """Table HDU of the stacked tables (no rows when there are none)."""
    if not tables:
        return fits.BinTableHDU(name=name)
    return fits.BinTableHDU(data=vstack(tables), name=name)


def info_table(rows: list[tuple[int, int, int, int]]) -> Shaped[np.ndarray, " n_row"]:
    """INFO table rows (INFO_COLUMNS) as a record array."""
    info = np.recarray((len(rows),), dtype=[(column, ">f8") for column in INFO_COLUMNS])
    for i, row in enumerate(rows):
        info[i] = row
    return info


class ResultAnalysisManager:
    """Writes the outlier files of a stage's bands from their Weave result files."""

    def __init__(self, settings: FixedSettings) -> None:
        """settings: paws.settings.FixedSettings (or Settings)."""
        self.settings = settings
        self.paths = PathManager(settings)

    def _write_clustered_results(
        self,
        freq: int,
        taskname: str,
        stage: str,
        outliers: Optional[Shaped[np.ndarray, " n_outlier"]],
        order: int,
        primary_hdu: fits.PrimaryHDU,
        injection_hdu: Optional[fits.BinTableHDU] = None,
        non_sat_hdu: Optional[fits.BinTableHDU] = None,
        work_in_local_dir: bool = False,
    ) -> Path:
        """Clusters the outliers (spacings from primary_hdu) and writes the clustered outlier file: the cluster
        centers (an injection stage: every outlier mapped to its center), the given HDUs and an INFO_CLUSTERED
        table (freq, clusterIndex, noOutliersWithin)."""
        primary_hdu.header["HIERARCH cluster_n_spacing"] = (
            self.settings.search.cluster_n_spacing
        )
        info_dtype = [
            (key, ">f8") for key in ["freq", "clusterIndex", "noOutliersWithin"]
        ]
        if outliers is None or len(outliers) == 0:
            cluster_hdu = fits.BinTableHDU(data=outliers, name=stage + "_outlier")
            info_data = np.recarray((0,), dtype=info_dtype)
        else:
            _, spacing_names = phase_param_name(order)
            spacing = {
                key: primary_hdu.header[f"HIERARCH {key}"] for key in spacing_names
            }
            centers, cluster_size, cluster_members = clustering(
                outliers, spacing, self.settings.search.cluster_n_spacing
            )
            if injection_hdu is not None:
                # every outlier mapped to its cluster center, to keep the injection tracking
                center_of_outlier = np.full(outliers.size, -1)
                processed = set()
                for center, members in zip(centers, cluster_members):
                    new = np.array([m for m in members if m not in processed])
                    if len(new) > 0:
                        center_of_outlier[new] = center
                        processed.update(members)
                cluster_data = outliers[center_of_outlier]
            else:
                cluster_data = outliers[centers]
            cluster_hdu = fits.BinTableHDU(data=cluster_data, name=stage + "_outlier")
            info_data = np.recarray((len(cluster_size),), dtype=info_dtype)
            for i, size in enumerate(cluster_size):
                info_data[i] = freq, i, size

        hdus = [primary_hdu, cluster_hdu]
        if injection_hdu is not None:
            hdus.append(injection_hdu)
        if non_sat_hdu is not None:
            hdus.append(non_sat_hdu)
        hdus.append(fits.BinTableHDU(data=info_data, name="info_clustered"))

        path = self.paths.outlier_file(freq, taskname, stage, cluster=True)
        if work_in_local_dir:
            path = Path(path.name)
        fits.HDUList(hdus).writeto(path, overwrite=True)
        return path

    def make_outlier(
        self,
        taskname: str,
        freq: int,
        stage: str,
        mean2f_threshold: Union[float, Float[np.ndarray, " n_point"]],
        n_points: int,
        n_keep: int,
        order: int,
        n_sky: int,
        cluster: bool,
        separate_saturated: bool,
        is_search: bool,
        is_injection: bool,
        n_processes: int,
        n_threads: int,
        work_in_local_dir: bool = False,
    ) -> Path:
        """Outlier file of a stage's band from the Weave result files of its n_points parameter points (point i
        runs as jobs i * n_sky + 1 to (i + 1) * n_sky, pooled; see pool_outliers), clustered too when `cluster`.
        Returns the path of the last file written.

        mean2f_threshold: one for every job, or one per parameter point.
        is_search: also write the NON_SAT_BAND table (the f0_band sub-bands without a saturated job).
        is_injection: also read and write the injection parameters.
        n_processes: processes reading the result files, each with n_threads threads.
        work_in_local_dir: read and write in the working directory (a Condor job).
        """
        if np.isscalar(mean2f_threshold):
            thresholds = [mean2f_threshold] * (n_points * n_sky)
        else:
            thresholds = np.repeat(mean2f_threshold, n_sky).tolist()
        jobs = []
        for job_index, threshold in enumerate(thresholds, 1):
            path = self.paths.weave_output_file(freq, taskname, job_index, stage)
            if work_in_local_dir:
                path = Path(path.name)
            jobs.append(
                ReadJob(job_index, path, threshold, n_keep, order, is_injection)
            )
        results = read_jobs(
            jobs, n_processes, n_threads, f"{stage.capitalize()} {freq}Hz"
        )
        collection = pool_outliers(results, freq, n_sky, n_keep, separate_saturated)

        primary_hdu = fits.PrimaryHDU()
        if np.isscalar(mean2f_threshold):
            primary_hdu.header["HIERARCH mean2F_th"] = mean2f_threshold
        for key, value in collection.spacing.items():
            primary_hdu.header[f"HIERARCH {key}"] = value
        outlier_hdu = table_hdu(collection.outliers, f"{stage}_outlier")
        hdus = [primary_hdu, outlier_hdu]
        injection_hdu = None
        if is_injection:
            injection_hdu = table_hdu(collection.injections, "injection")
            hdus.append(injection_hdu)
        if separate_saturated:
            hdus.append(table_hdu(collection.saturated, f"{stage}_sat_outlier"))
        info = info_table(collection.info)
        hdus.append(fits.BinTableHDU(data=info, name="info"))
        non_sat_hdu = None
        if is_search:
            # the f0_band sub-bands of the 1 Hz band whose jobs are all unsaturated
            f0_band = self.settings.data.f0_band
            saturated = info["isSaturated"].reshape(int(1.0 / f0_band), -1).any(axis=1)
            sub_bands = np.flatnonzero(~saturated)
            non_sat = np.recarray((len(sub_bands),), dtype=[("non_sat_band", ">f8")])
            non_sat["non_sat_band"] = freq + sub_bands * f0_band
            non_sat_hdu = fits.BinTableHDU(data=non_sat, name="non_sat_band")
            hdus.append(non_sat_hdu)

        path = self.paths.outlier_file(freq, taskname, stage, cluster=False)
        if work_in_local_dir:
            path = Path(path.name)
        make_dir([path])
        fits.HDUList(hdus).writeto(path, overwrite=True)

        if cluster and outlier_hdu.data is not None:
            path = self._write_clustered_results(
                freq,
                taskname,
                stage,
                outlier_hdu.data,
                order,
                primary_hdu,
                injection_hdu=injection_hdu,
                non_sat_hdu=non_sat_hdu,
                work_in_local_dir=work_in_local_dir,
            )
        print(f"Finished writing {stage} result for {freq} Hz")
        return path

    def write_loudest_outliers(
        self,
        taskname: str,
        freq: int,
        stage: str,
        freq_deriv_order: int,
        n_seeds: int,
        loudest_per_seed: Shaped[np.ndarray, " n_seed"],
        passed: Bool[np.ndarray, " n_seed"],
        mean2f_threshold: Float[np.ndarray, " n_seed"],
        history: dict[str, Float[np.ndarray, "n_seed ..."]],
        history_stages: dict[str, str],
        spacing_files: list[Path],
    ) -> Path:
        """
        Unclustered and clustered outlier files of a stage collected as the loudest candidate per seed
        (see loudest_rows); INFO has one row per seed.

        history: {column name: per-seed values} of the earlier stages of the chain, added as columns
        (<statistic>_TCoh<days>); history_stages: {"TCoh<days>": stage name}, written to the primary header.

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
        for suffix, stage_name in history_stages.items():
            primary_hdu.header[f"HIERARCH history {suffix}"] = (
                stage_name,
                "stage of the _" + suffix + " columns",
            )

        outlier_table = Table(loudest_per_seed[passed])
        outlier_table.add_column(mean2f_threshold[passed], name="mean2F threshold")
        for name, values in history.items():
            outlier_table.add_column(np.asarray(values)[passed], name=name)
        hdu_name = f"{stage}_outlier"
        if passed.any():
            outlier_hdu = fits.BinTableHDU(data=outlier_table, name=hdu_name)
        else:
            outlier_hdu = fits.BinTableHDU(name=hdu_name)

        info = np.recarray(
            (n_seeds,), dtype=[(column, ">f8") for column in INFO_COLUMNS]
        )
        info["freq"], info["jobIndex"], info["outliers"], info["isSaturated"] = (
            freq,
            np.arange(n_seeds),
            passed,
            0,
        )

        path = self.paths.outlier_file(freq, taskname, stage, cluster=False)
        make_dir([path])
        fits.HDUList(
            [primary_hdu, outlier_hdu, fits.BinTableHDU(data=info, name="info")]
        ).writeto(path, overwrite=True)
        return self._write_clustered_results(
            freq, taskname, stage, outlier_hdu.data, freq_deriv_order, primary_hdu
        )
