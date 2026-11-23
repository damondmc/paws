from pathlib import Path

import numpy as np
import pytest

from paws.analysis.sigmoid import SigmoidFitter
from paws.jobs.upperlimit import (
    BRACKET_FACTORS,
    drawn_injections_hdu,
    find_h95,
    injection_sets_hdu,
    parse_args,
    scale_injection,
)
from paws.jobs.weave_runner import EfficiencyRun, injection_job, loudest_mean2f
from paws.settings import FixedSettings
from tests.test_sigmoid import H95, TARGET, true_efficiency
from tests.weave_files import write_weave_result

DRAW = (0.9, 1.1)

REQUIRED = (
    "--environment_file e.yaml --data_file d.yaml --search_file s.yaml --taskname T "
    "--freq 50 --stage upperlimit --freq_deriv_order 2 --df_grid 1e-6 1e-13 1e-20 "
    "--inj_freq_deriv_order 4 --sft_files 'a.sft;b.sft' --metric_file m.fts --n_inj 200 "
    "--num_toplist 1 --h0_est 6e-26 --mean2f_th 6.5 --non_sat_bands 50.0 50.1 "
    "--sky_radius 1e-4 --n_spacing 3 --n_cpus 8 --draw_window 0.9 1.1 --n_drawn 2000"
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
    """Two injections with two sky points each."""
    loudest = [
        [{"mean2F": 5.0}, {"mean2F": 3.0}],
        [{"mean2F": 9.0}],
        [{"mean2F": 2.0}],
        [{"mean2F": 4.0}],
    ]
    files = [
        str(write_weave_result(tmp_path / f"r{i}.fts", rows))
        for i, rows in enumerate(loudest)
    ]
    assert loudest_mean2f(files, n_sky=2).tolist() == [9.0, 4.0]


def test_loudest_mean2f_rejects_an_empty_toplist(tmp_path):
    files = [
        str(write_weave_result(tmp_path / "r0.fts", [{"mean2F": 5.0}])),
        str(write_weave_result(tmp_path / "r1.fts", [])),
    ]
    with pytest.raises(RuntimeError, match="empty toplist"):
        loudest_mean2f(files, n_sky=1)


def test_injection_job_raises_when_weave_fails(config_dir, tmp_path):
    settings = FixedSettings(
        config_dir / "environment.yaml",
        config_dir / "data.yaml",
        config_dir / "search.yaml",
    )
    names = ("Alpha", "Delta", "refTime", "aPlus", "aCross", "psi", "Freq")
    names += ("f1dot", "f2dot", "f3dot", "f4dot")
    row = np.ones(1, dtype=[(name, "f8") for name in names])[0]
    search_row = np.array(
        [(50.0, 1e-6, 0.0, 1e-13)],
        dtype=[("freq", "f8"), ("df", "f8"), ("f1dot", "f8"), ("df1dot", "f8")],
    )[0]
    with pytest.raises(RuntimeError, match="exit status 1"):
        injection_job(
            settings,
            1,
            1,
            1,
            "a.sft",
            "m.fts",
            "",
            "false",
            (str(tmp_path / "out" / "r.fts"), search_row),
            row,
        )


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


@pytest.fixture(autouse=True)
def seeded_strains():
    """find_h95 draws the strains with np.random, seeded as in the upper-limit job."""
    np.random.seed(0)


def drawn_detection(rng, efficiency=None):
    """detected_at for find_h95: each injection detected with the efficiency at its strain."""
    efficiency = efficiency or true_efficiency
    return lambda h0: rng.random(len(h0)) < np.array([efficiency(h) for h in h0])


def test_find_h95_converges_on_the_sigmoid():
    tried = []

    def efficiency_at(h0):
        tried.append(h0)
        return true_efficiency(h0)

    rng = np.random.default_rng(0)
    limit = find_h95(
        efficiency_at,
        drawn_detection(rng),
        1.5e-25,
        SigmoidFitter(TARGET, n_inj=200),
        DRAW,
        2000,
    )
    assert limit.h95 == pytest.approx(H95, rel=0.06)
    assert abs(limit.dh95 / limit.h95) <= 0.05
    assert list(limit.h0) == tried == [1.5e-25 * f for f in BRACKET_FACTORS]
    assert list(limit.efficiency) == [true_efficiency(h0) for h0 in tried]
    assert len(limit.drawn_h0) == len(limit.drawn_detected) == 2000


def test_find_h95_draws_log_uniform_strains_around_h95():
    """The drawn strains of the accepted round span DRAW around the h95 they were drawn at."""
    rng = np.random.default_rng(1)
    limit = find_h95(
        true_efficiency,
        drawn_detection(rng),
        H95,
        SigmoidFitter(TARGET, n_inj=200),
        DRAW,
        2000,
    )
    log_h0 = np.log(limit.drawn_h0)
    width = np.log(DRAW[1] / DRAW[0])
    assert log_h0.max() - log_h0.min() == pytest.approx(width, rel=0.01)
    assert np.std(log_h0) == pytest.approx(width / np.sqrt(12), rel=0.05)


def test_find_h95_ignores_the_curve_far_from_h95():
    """An efficiency that follows the sigmoid only above 80 %: the starting sets give a wrong first h95, the
    drawn injections around it give H95."""

    def distorted(h0):
        p = true_efficiency(h0)
        return p if p >= 0.8 else 0.5 * p

    rng = np.random.default_rng(2)
    limit = find_h95(
        distorted,
        drawn_detection(rng, distorted),
        1.5e-25,
        SigmoidFitter(TARGET, n_inj=200),
        DRAW,
        2000,
    )
    assert limit.h95 == pytest.approx(H95, rel=0.06)


@pytest.mark.parametrize("estimate", [0.6 * H95, 1.8 * H95])
def test_find_h95_recentres_from_a_poor_first_estimate(estimate):
    rng = np.random.default_rng(3)
    limit = find_h95(
        true_efficiency,
        drawn_detection(rng),
        estimate,
        SigmoidFitter(TARGET, n_inj=200),
        DRAW,
        2000,
    )
    assert limit.h95 == pytest.approx(H95, rel=0.06)


def test_find_h95_with_drawn_injections_is_unbiased_with_matching_errors():
    """Over repeated runs, h95 scatters around H95 by about dh95."""
    rng = np.random.default_rng(4)
    pulls = []
    for _ in range(100):
        limit = find_h95(
            lambda h0: rng.binomial(200, true_efficiency(h0)) / 200,
            drawn_detection(rng),
            1.5e-25,
            SigmoidFitter(TARGET, n_inj=200),
            DRAW,
            2000,
        )
        pulls.append((limit.h95 - H95) / limit.dh95)
    assert abs(np.mean(pulls)) < 0.25
    assert 0.75 < np.std(pulls) < 1.3


def test_find_h95_stops_after_max_rounds():
    """A zero tolerance is never reached: one draw per round, then an error."""
    n_draws = []
    rng = np.random.default_rng(5)
    detect = drawn_detection(rng)

    def detected_at(h0):
        n_draws.append(len(h0))
        return detect(h0)

    with pytest.raises(RuntimeError, match="did not converge"):
        find_h95(
            true_efficiency,
            detected_at,
            1.5e-25,
            SigmoidFitter(TARGET, n_inj=200),
            DRAW,
            500,
            tolerance=0.0,
            max_rounds=2,
        )
    assert n_draws == [500, 500]


def test_find_h95_rejects_an_unconstrained_fit():
    """A flat efficiency gives no h95."""
    with pytest.raises(RuntimeError, match="constrain"):
        find_h95(
            lambda h0: 0.5,
            lambda h0: np.zeros(len(h0), bool),
            1e-25,
            SigmoidFitter(TARGET, n_inj=200),
            DRAW,
            2000,
        )
