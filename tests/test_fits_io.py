import numpy as np
import pytest
from astropy.io import fits
from weave_files import write_weave_result

from paws.fits_io import (
    outlier_file_spacing,
    read_fits_table,
    read_weave_run_stats,
    weave_template_spacing,
)


def test_weave_template_spacing(tmp_path):
    # cumulative templates 1000 / 10 / 100 -> 10 f0 x 10 f1dot x 10 f2dot templates
    path = write_weave_result(tmp_path / "result.fts", [{"mean2F": 5.0}], freq_range=(50.0, 50.1),
                              f1dot_range=(-1e-9, 0.0), f2dot_range=(0.0, 1e-18), templates=(1000, 10, 100))
    spacing = weave_template_spacing(path, 2)
    assert spacing == pytest.approx({"df": 0.01, "df1dot": 1e-10, "df2dot": 1e-19})


def test_read_weave_run_stats(tmp_path):
    header = {"NSEMITPL": 554775, "PEAKMEM": 899.05, "WALL TOTAL": 31.4, "CPU TOTAL": 31.0}
    path = write_weave_result(tmp_path / "result.fts", [{"mean2F": 5.0}], extra_header=header)
    assert read_weave_run_stats(path) == {"templates": 554775, "peak_memory_mb": 899.05, "wall_time_s": 31.4,
                                          "cpu_time_s": 31.0}


def test_outlier_file_spacing(tmp_path):
    primary = fits.PrimaryHDU()
    for name, value in (("df", 1e-6), ("df1dot", 1e-13), ("df2dot", 1e-20), ("df3dot", 1e-27)):
        primary.header[f"HIERARCH {name}"] = value
    path = tmp_path / "outlier.fts"
    primary.writeto(path)
    assert outlier_file_spacing(path, 2) == {"df": 1e-6, "df1dot": 1e-13, "df2dot": 1e-20}


def test_read_fits_table(tmp_path):
    table = np.array([(1.0, 2.0)], dtype=[("freq", ">f8"), ("mean2F", ">f8")])
    path = tmp_path / "table.fts"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(data=table, name="rows"), fits.BinTableHDU(name="empty")]
                 ).writeto(path)
    rows = read_fits_table(path, "rows")
    assert rows["mean2F"][0] == 2.0 and isinstance(rows, np.ndarray)
    assert read_fits_table(path, 1)["freq"][0] == 1.0
    assert len(read_fits_table(path, "empty")) == 0  # empty outlier table of a band without outliers
