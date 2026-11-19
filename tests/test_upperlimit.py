from pathlib import Path

import numpy as np
import pytest

from paws.jobs.upperlimit import (
    drawn_injections_hdu,
    injection_sets_hdu,
    parse_args,
    scale_injection,
)
from paws.jobs.weave_runner import EfficiencyRun, loudest_mean2f
from tests.weave_files import write_weave_result

REQUIRED = (
    "--environment_file e.yaml --data_file d.yaml --search_file s.yaml --taskname T "
    "--freq 50 --stage upperlimit --freq_deriv_order 2 --df_grid 1e-6 1e-13 1e-20 "
    "--inj_freq_deriv_order 4 --sft_files 'a.sft;b.sft' --metric_file m.fts --n_inj 200 "
    "--num_toplist 1 --h0_est 6e-26 --mean2f_th 6.5 --non_sat_bands 50.0 50.1 "
    "--sky_radius 1e-4 --n_spacing 3 --n_cpus 8"
).split()


def test_parse_args():
    args = parse_args(REQUIRED + ["--cluster", "--sky_metric_file", "sky.fts"])
    assert args.freq == 50 and args.df_grid == [1e-6, 1e-13, 1e-20]
    assert args.non_sat_bands == [50.0, 50.1] and args.n_cpus == 8
    assert args.cluster and not args.work_in_local_dir
    assert args.sky_metric_file == "sky.fts"


def test_parse_args_has_no_defaults_for_the_run():
    with pytest.raises(SystemExit):
        parse_args([a for a in REQUIRED if a not in ("--h0_est", "6e-26")])


def test_scale_injection():
    rows = np.array(
        [(1.0, 0.5, 7.0)], dtype=[("aPlus", "f8"), ("aCross", "f8"), ("psi", "f8")]
    )
    scaled = scale_injection(rows, 2e-25)
    assert scaled["aPlus"][0] == 2e-25 and scaled["aCross"][0] == 1e-25
    assert scaled["psi"][0] == 7.0 and rows["aPlus"][0] == 1.0


def test_scale_injection_with_one_strain_per_row():
    rows = np.array([(1.0, 0.5), (2.0, 1.0)], dtype=[("aPlus", "f8"), ("aCross", "f8")])
    scaled = scale_injection(rows, np.array([1e-25, 3e-25]))
    assert scaled["aPlus"].tolist() == [1e-25, 6e-25]
    assert scaled["aCross"].tolist() == [0.5e-25, 3e-25]


def test_loudest_mean2f_pools_the_sky_points_of_each_injection(tmp_path):
    """Two injections with two sky points each; the second injection's first toplist is empty."""
    loudest = [
        [{"mean2F": 5.0}, {"mean2F": 3.0}],
        [{"mean2F": 9.0}],
        [],
        [{"mean2F": 4.0}],
    ]
    files = [
        str(write_weave_result(tmp_path / f"r{i}.fts", rows))
        for i, rows in enumerate(loudest)
    ]
    assert loudest_mean2f(files, n_sky=2).tolist() == [9.0, 4.0]


def test_injection_sets_hdu_marks_the_last_set_as_confirmation():
    loudest = np.array([[7.0, 5.0], [9.0, 8.0]])
    detected = loudest >= 6.0
    sets = [
        (1e-25, EfficiencyRun(0.5, Path("a.fts"), loudest[0], detected[0])),
        (2e-25, EfficiencyRun(1.0, Path("b.fts"), loudest[1], detected[1])),
    ]
    data = injection_sets_hdu(sets).data
    assert data["h0"].tolist() == [1e-25, 2e-25]
    assert data["confirmation"].tolist() == [False, True]
    assert data["efficiency"].tolist() == [0.5, 1.0]
    assert np.array_equal(data["loudest_mean2F"], loudest)
    assert np.array_equal(data["detected"], detected)


def test_drawn_injections_hdu_has_one_row_per_injection_of_every_round():
    dtype = [("Freq", "f8"), ("aPlus", "f8")]
    rounds = [
        (np.array([(50.1, 1.0), (50.2, 0.5)], dtype=dtype), np.array([1e-25, 2e-25])),
        (np.array([(50.3, 0.8)], dtype=dtype), np.array([3e-25])),
    ]
    draws = [
        (rows, h0, EfficiencyRun(0.5, Path("r.fts"), 6.0 + h0 * 1e25, h0 > 1.5e-25))
        for rows, h0 in rounds
    ]
    data = drawn_injections_hdu(draws).data
    assert data["Freq"].tolist() == [50.1, 50.2, 50.3]
    assert data["round"].tolist() == [0, 0, 1]
    assert data["h0"].tolist() == [1e-25, 2e-25, 3e-25]
    assert data["detected"].tolist() == [False, True, True]
    assert data["loudest_mean2F"].tolist() == pytest.approx([7.0, 8.0, 9.0])
