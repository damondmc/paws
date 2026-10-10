"""Tiny synthetic Weave result files for the tests."""

import numpy as np
from astropy.io import fits

TOPLIST_COLUMNS = (
    "freq",
    "f1dot",
    "f2dot",
    "alpha",
    "delta",
    "mean2F",
    "mean2F_H1",
    "mean2F_L1",
)


def toplist(rows):
    """Weave toplist rows from dicts (missing columns are 0)."""
    table = np.zeros(len(rows), dtype=[(column, ">f8") for column in TOPLIST_COLUMNS])
    for i, row in enumerate(rows):
        for column, value in row.items():
            table[i][column] = value
    return table


def write_weave_result(
    path,
    rows,
    freq_range=(50.0, 50.1),
    f1dot_range=(-1e-9, 0.0),
    f2dot_range=(0.0, 1e-18),
    templates=(1000, 10, 100),
    extra_header=None,
):
    """A Weave result file: toplist sorted by mean2F (loudest first) plus the header keys paws reads:
    NSEMITMPL NU<k>DOT (cumulative template counts) and PROGARG FREQ / F1DOT / F2DOT (searched ranges)."""
    table = toplist(sorted(rows, key=lambda row: -row["mean2F"]))
    primary = fits.PrimaryHDU()
    for k, n_templates in enumerate(templates):
        primary.header[f"HIERARCH NSEMITMPL NU{k}DOT"] = n_templates
    for name, (start, stop) in (
        ("FREQ", freq_range),
        ("F1DOT", f1dot_range),
        ("F2DOT", f2dot_range),
    ):
        primary.header[f"HIERARCH PROGARG {name}"] = f"{start},{stop}"
    for key, value in (extra_header or {}).items():
        primary.header[f"HIERARCH {key}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, fits.BinTableHDU(data=table)]).writeto(path, overwrite=True)
    return path
