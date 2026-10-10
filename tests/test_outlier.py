import numpy as np
import pytest
from astropy.io import fits
from weave_files import write_weave_result

from paws.analysis.outlier import (
    ResultAnalysisManager,
    loudest_rows,
    read_jobs,
    read_loudest_row,
)

TASKNAME = "TestTarget_followup-2_TCoh20_O2_50Hz"


def seed_files(tmp_path, seed, mean2f_per_sky_point):
    """One result file per sky point; each toplist has a quieter second row."""
    files = []
    for sky_point, mean2f in enumerate(mean2f_per_sky_point):
        rows = [{"freq": 50.0 + seed * 0.01, "mean2F": mean2f, "mean2F_H1": mean2f, "mean2F_L1": mean2f},
                {"freq": 50.5, "mean2F": mean2f - 1}]
        files.append(write_weave_result(tmp_path / f"seed{seed}_sky{sky_point}.fts", rows))
    return files


def test_read_loudest_row(tmp_path):
    path = write_weave_result(tmp_path / "r.fts", [{"mean2F": 6.0}, {"mean2F": 9.0}, {"mean2F": 7.0}])
    assert read_loudest_row(path)["mean2F"][0] == 9.0


def test_read_loudest_row_empty_toplist(tmp_path):
    path = write_weave_result(tmp_path / "r.fts", [])
    with pytest.raises(ValueError):
        read_loudest_row(path)


def test_loudest_rows_over_sky_points(tmp_path):
    files_per_seed = [seed_files(tmp_path, 0, [10.0, 30.0, 20.0]), seed_files(tmp_path, 1, [8.0, 6.0, 7.0])]
    loudest = loudest_rows(files_per_seed, n_threads=2)
    assert list(loudest["mean2F"]) == [30.0, 8.0]
    assert loudest["freq"][1] == pytest.approx(50.01)


def test_read_jobs_same_with_processes(tmp_path):
    files = [write_weave_result(tmp_path / f"r{i}.fts", [{"mean2F": 5.0 + i}, {"mean2F": 4.0}]) for i in range(9)]
    files.insert(4, tmp_path / "missing.fts")  # a missing result file stays in place
    jobs = [(i, i + 1, path, 4.5, 1000, 2, False) for i, path in enumerate(files)]
    single = read_jobs(jobs, n_processes=1, n_threads=2, desc="test")
    multi = read_jobs(jobs, n_processes=3, n_threads=2, desc="test")
    assert [result[1] for result in multi] == list(range(1, 11))
    for one, many in zip(single, multi):
        assert one[:2] == many[:2] and one[5] == many[5]
        assert (one[2] is None) == (many[2] is None)
        if one[2] is not None:
            assert list(one[2]["mean2F"]) == list(many[2]["mean2F"])


def test_write_loudest_outliers(tmp_path, settings, paths):
    files_per_seed = [seed_files(tmp_path, 0, [10.0, 30.0, 20.0]), seed_files(tmp_path, 1, [8.0, 6.0, 7.0])]
    loudest = loudest_rows(files_per_seed, n_threads=2)
    passed = np.array([True, False])
    mean2f_threshold = np.array([25.0, 9.0])
    manager = ResultAnalysisManager(settings.config, settings.target)
    manager.write_loudest_outliers(TASKNAME, 50, "followup-2", 2, 2, loudest, passed, mean2f_threshold,
                                   [files_per_seed[0][0]])

    unclustered = paths.outlier_file(50, TASKNAME, "followup-2", cluster=False, location="home")
    with fits.open(unclustered) as hdul:
        assert hdul[0].header["HIERARCH df"] == pytest.approx(0.01)
        outliers = hdul["followup-2_outlier"].data
        assert list(outliers["mean2F"]) == [30.0] and list(outliers["mean2F threshold"]) == [25.0]
        info = hdul["info"].data
        assert list(info["jobIndex"]) == [0, 1] and list(info["outliers"]) == [1, 0]
    clustered = paths.outlier_file(50, TASKNAME, "followup-2", cluster=True, location="home")
    assert len(fits.getdata(clustered, 1)) == 1
