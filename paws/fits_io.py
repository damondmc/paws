"""Reading FITS files: tables, Weave result headers and outlier file headers."""

import numpy as np
from astropy.io import fits

from paws.definitions import phase_param_name


def read_fits_table(path, extension):
    """Table `extension` (index or name) of a FITS file as a numpy array (None when the HDU has no data part;
    an empty table HDU, as written for a band without outliers, gives an empty array)."""
    with fits.open(path) as hdul:
        table = hdul[extension].data
        return np.array(table) if table is not None else None


def weave_template_spacing(weave_result_path, freq_deriv_order):
    """Template spacings (df, df1dot, ...) of a Weave result file (see weave_header_spacing)."""
    return weave_header_spacing(fits.getheader(weave_result_path), freq_deriv_order)


def weave_header_spacing(header, freq_deriv_order):
    """Template spacings (df, df1dot, ...) from a Weave result's primary header: each searched range over its
    number of semicoherent templates, from the PROGARG and NSEMITMPL keys."""
    freq_param_names, spacing_names = phase_param_name(freq_deriv_order)
    n_params = len(freq_param_names)

    cumulative_templates = [header[f"NSEMITMPL NU{i}DOT"] for i in range(n_params)]
    templates_per_param = []

    if n_params > 0:
        templates_per_param.append(int(cumulative_templates[0] / cumulative_templates[-1]))  # f0
        if n_params > 1:
            templates_per_param.append(cumulative_templates[1])  # f1
            for i in range(2, n_params):
                templates_per_param.append(int(cumulative_templates[i] / cumulative_templates[i - 1]))

    spacing = {}
    for i in range(n_params):
        searched_range = header.get(f"PROGARG {freq_param_names[i].upper()}", None)
        if searched_range:
            start, stop = searched_range.split(",")
            spacing[spacing_names[i]] = (float(stop) - float(start)) / templates_per_param[i]
    return spacing


def read_weave_run_stats(weave_result_path):
    """Cost of one Weave run from its result header: semicoherent templates, peak memory (MB), wall and CPU
    time (s)."""
    header = fits.getheader(weave_result_path)
    return {
        "templates": int(header["NSEMITPL"]),
        "peak_memory_mb": float(header["PEAKMEM"]),
        "wall_time_s": float(header["WALL TOTAL"]),
        "cpu_time_s": float(header["CPU TOTAL"]),
    }


def read_injection_outliers(clustered_outlier_path):
    """(outlier rows, injected Freq) of one band of an injection stage; None when the band has no outliers."""
    with fits.open(clustered_outlier_path) as hdul:
        if hdul[1].data is None or len(hdul[1].data) == 0:
            return None
        return np.array(hdul[1].data), np.array(hdul[2].data["Freq"])


def outlier_file_spacing(outlier_path, freq_deriv_order):
    """Grid spacings (df, df1dot, ...) written in an outlier file's primary header."""
    _, spacing_names = phase_param_name(freq_deriv_order)
    return {name: fits.getval(outlier_path, name, 0) for name in spacing_names}
