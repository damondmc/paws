import numpy as np
import pytest

from paws.jobs.upperlimit import parse_args, scale_injection

REQUIRED = (
    "--environment_file e.yaml --data_file d.yaml --search_file s.yaml --taskname T "
    "--freq 50 --stage upperlimit --freq_deriv_order 2 --df_grid 1e-6 1e-13 1e-20 "
    "--inj_freq_deriv_order 4 --sft_files 'a.sft;b.sft' --metric_file m.fts --n_inj 200 "
    "--num_toplist 1 --h0_est 6e-26 --mean2f_th 6.5 --non_sat_bands 50.0 50.1 "
    "--sky_radius 1e-4 --n_spacing 3 --n_cpus 8"
).split()


def test_parse_args():
    args = parse_args(REQUIRED + ["--cluster", "--spacing_alpha", "2e-4"])
    assert args.freq == 50 and args.df_grid == [1e-6, 1e-13, 1e-20]
    assert args.non_sat_bands == [50.0, 50.1] and args.n_cpus == 8
    assert args.cluster and not args.work_in_local_dir
    assert args.spacing_alpha == 2e-4 and args.spacing_delta is None


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
