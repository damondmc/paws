import numpy as np
import pytest
from astropy.io import fits
from weave_files import write_weave_result

from paws.analysis.outlier import (
    ReadJob,
    ResultAnalysisManager,
    loudest_rows,
    read_jobs,
    read_loudest_row,
)

TASKNAME = "TestTarget_followup-2_TCoh20_O2_50Hz"


def write_job_results(paths, stage, mean2f_per_job):
    """One Weave result file per job (jobs 1, 2, ...) with the given mean2F rows."""
    for job_index, mean2fs in enumerate(mean2f_per_job, 1):
        write_weave_result(
            paths.weave_output_file(50, TASKNAME, job_index, stage),
            [
                {"freq": 50.0 + 0.001 * i, "mean2F": mean2f}
                for i, mean2f in enumerate(mean2fs)
            ],
        )


def seed_files(tmp_path, seed, mean2f_per_sky_point):
    """One result file per sky point; each toplist has a quieter second row."""
    files = []
    for sky_point, mean2f in enumerate(mean2f_per_sky_point):
        rows = [
            {
                "freq": 50.0 + seed * 0.01,
                "mean2F": mean2f,
                "mean2F_H1": mean2f,
                "mean2F_L1": mean2f,
            },
            {"freq": 50.5, "mean2F": mean2f - 1},
        ]
        files.append(
            write_weave_result(tmp_path / f"seed{seed}_sky{sky_point}.fts", rows)
        )
    return files


def test_read_loudest_row(tmp_path):
    path = write_weave_result(
        tmp_path / "r.fts", [{"mean2F": 6.0}, {"mean2F": 9.0}, {"mean2F": 7.0}]
    )
    assert read_loudest_row(path)["mean2F"][0] == 9.0


def test_read_loudest_row_empty_toplist(tmp_path):
    path = write_weave_result(tmp_path / "r.fts", [])
    with pytest.raises(ValueError):
        read_loudest_row(path)


def test_loudest_rows_over_sky_points(tmp_path):
    files_per_seed = [
        seed_files(tmp_path, 0, [10.0, 30.0, 20.0]),
        seed_files(tmp_path, 1, [8.0, 6.0, 7.0]),
    ]
    loudest = loudest_rows(files_per_seed, n_threads=2)
    assert list(loudest["mean2F"]) == [30.0, 8.0]
    assert loudest["freq"][1] == pytest.approx(50.01)


def test_read_jobs_same_with_processes(tmp_path):
    files = [
        write_weave_result(
            tmp_path / f"r{i}.fts", [{"mean2F": 5.0 + i}, {"mean2F": 4.0}]
        )
        for i in range(9)
    ]
    jobs = [ReadJob(i + 1, path, 4.5, 1000, 2, False) for i, path in enumerate(files)]
    single = read_jobs(jobs, n_processes=1, n_threads=2, desc="test")
    multi = read_jobs(jobs, n_processes=3, n_threads=2, desc="test")
    assert [result.job_index for result in multi] == list(range(1, 10))
    for one, many in zip(single, multi):
        assert one.job_index == many.job_index and one.spacing == many.spacing
        assert list(one.outliers["mean2F"]) == list(many.outliers["mean2F"])
    jobs.insert(4, ReadJob(10, tmp_path / "missing.fts", 4.5, 1000, 2, False))
    for n_processes in (1, 3):
        with pytest.raises(FileNotFoundError):
            read_jobs(jobs, n_processes=n_processes, n_threads=2, desc="test")


def test_make_outlier_search(settings, paths):
    """40 points (1 / f0_band) of one job each: point 3 saturates (n_keep=2), point 7 has one outlier."""
    stage = "search-0"
    mean2f_per_job = [[3.0]] * 40
    mean2f_per_job[3] = [9.0, 8.0, 7.0]
    mean2f_per_job[7] = [6.0, 3.0]
    write_job_results(paths, stage, mean2f_per_job)
    path = ResultAnalysisManager(settings).make_outlier(
        TASKNAME,
        50,
        stage,
        5.0,
        n_points=40,
        n_keep=2,
        order=2,
        n_sky=1,
        cluster=True,
        separate_saturated=True,
        is_search=True,
        is_injection=False,
        n_processes=1,
        n_threads=2,
    )
    assert path == paths.outlier_file(50, TASKNAME, stage, cluster=True)
    with fits.open(paths.outlier_file(50, TASKNAME, stage, cluster=False)) as hdul:
        assert hdul[0].header["HIERARCH mean2F_th"] == 5.0
        assert hdul[0].header["HIERARCH df"] == pytest.approx(0.01)
        outliers = hdul[f"{stage}_outlier"].data
        assert list(outliers["mean2F"]) == [6.0]
        assert list(outliers["mean2F threshold"]) == [5.0]
        assert list(hdul[f"{stage}_sat_outlier"].data["mean2F"]) == [9.0]
        info = hdul["info"].data
        assert list(info["jobIndex"]) == list(range(1, 41))
        assert list(np.flatnonzero(info["outliers"])) == [3, 7]
        assert list(info["outliers"][[3, 7]]) == [2, 1]
        assert list(np.flatnonzero(info["isSaturated"])) == [3]
        non_sat = hdul["non_sat_band"].data["non_sat_band"]
        assert len(non_sat) == 39 and 50.075 not in non_sat
    with fits.open(path) as hdul:
        assert hdul[0].header["HIERARCH cluster_n_spacing"] == (
            settings.search.cluster_n_spacing
        )
        assert list(hdul[f"{stage}_outlier"].data["mean2F"]) == [6.0]
        assert len(hdul["non_sat_band"].data) == 39
        assert list(hdul["info_clustered"].data["noOutliersWithin"]) == [1]


def test_make_outlier_pools_sky_points(settings, paths):
    """Two points of two sky jobs each, one threshold per point: the candidates of a point are pooled."""
    stage = "followup-1"
    write_job_results(paths, stage, [[10.0, 6.0], [8.0, 7.0], [20.0], [30.0]])
    ResultAnalysisManager(settings).make_outlier(
        TASKNAME,
        50,
        stage,
        np.array([5.0, 100.0]),
        n_points=2,
        n_keep=3,
        order=2,
        n_sky=2,
        cluster=False,
        separate_saturated=False,
        is_search=False,
        is_injection=False,
        n_processes=1,
        n_threads=2,
    )
    with fits.open(paths.outlier_file(50, TASKNAME, stage, cluster=False)) as hdul:
        assert "mean2F_th" not in hdul[0].header
        outliers = hdul[f"{stage}_outlier"].data
        assert list(outliers["mean2F"]) == [10.0, 8.0, 7.0]
        assert list(outliers["mean2F threshold"]) == [5.0] * 3
        info = hdul["info"].data
        assert list(info["jobIndex"]) == [1, 3]
        assert list(info["outliers"]) == [3, 0]
        assert list(info["isSaturated"]) == [1, 0]
        assert "non_sat_band" not in hdul


def test_write_loudest_outliers(tmp_path, settings, paths):
    files_per_seed = [
        seed_files(tmp_path, 0, [10.0, 30.0, 20.0]),
        seed_files(tmp_path, 1, [8.0, 6.0, 7.0]),
    ]
    loudest = loudest_rows(files_per_seed, n_threads=2)
    passed = np.array([True, False])
    mean2f_threshold = np.array([25.0, 9.0])
    manager = ResultAnalysisManager(settings)
    manager.write_loudest_outliers(
        TASKNAME,
        50,
        "followup-2",
        2,
        2,
        loudest,
        passed,
        mean2f_threshold,
        {
            "mean2F_TCoh5": np.array([6.5, 7.0]),
            "coh2F_H1_TCoh5": np.array([[1.0, 2.0], [3.0, 4.0]]),
        },
        {"TCoh5": "search-0"},
        [files_per_seed[0][0]],
    )

    unclustered = paths.outlier_file(50, TASKNAME, "followup-2", cluster=False)
    with fits.open(unclustered) as hdul:
        assert hdul[0].header["HIERARCH df"] == pytest.approx(0.01)
        outliers = hdul["followup-2_outlier"].data
        assert list(outliers["mean2F"]) == [30.0] and list(
            outliers["mean2F threshold"]
        ) == [25.0]
        assert list(outliers["mean2F_TCoh5"]) == [6.5]
        assert outliers["coh2F_H1_TCoh5"].tolist() == [[1.0, 2.0]]
        assert hdul[0].header["HIERARCH history TCoh5"] == "search-0"
        info = hdul["info"].data
        assert list(info["jobIndex"]) == [0, 1] and list(info["outliers"]) == [1, 0]
    clustered = paths.outlier_file(50, TASKNAME, "followup-2", cluster=True)
    clustered_rows = fits.getdata(clustered, 1)
    assert len(clustered_rows) == 1 and list(clustered_rows["mean2F_TCoh5"]) == [6.5]
