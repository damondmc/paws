import numpy as np
import pytest
import yaml
from astropy.io import fits
from pydantic import ValidationError

from paws.pipeline.calibrate import calibrate, write_threshold_file
from paws.pipeline.noise_weights import _write_rows, log10_noise_ratio
from paws.pipeline.thresholds import (
    record_threshold_file,
    stage_thresholds,
    threshold_record_path,
)
from paws.settings import ThresholdBand, ThresholdFile


def band(f_min, f_max, excess_ratio, **optional):
    fields = {
        "excess_ratio_lowest": None,
        "h1_l1_window": None,
        "seed_h1_l1_window": None,
        "growth": None,
        **optional,
    }
    return ThresholdBand(f_min=f_min, f_max=f_max, excess_ratio=excess_ratio, **fields)


def threshold_file(tcoh, seed_tcoh, bands, growth_tcoh=None):
    return ThresholdFile(
        tcoh=tcoh,
        seed_tcoh=seed_tcoh,
        growth_tcoh=growth_tcoh,
        h1_l1_noise_weights=None,
        calibration=None,
        bands=bands,
    )


def write_threshold_yaml(settings, name, file):
    path = settings.threshold_file_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(file.model_dump(mode="json")))
    return path


def test_stage_thresholds_from_file(settings):
    file = threshold_file(
        10,
        5,
        [
            band(
                20, 100, 1.2345, h1_l1_window=(-1.0, 1.0), seed_h1_l1_window=(-3.0, 3.0)
            ),
            band(
                100, 400, 1.5, h1_l1_window=(-2.0, 2.0), seed_h1_l1_window=(-4.0, 4.0)
            ),
        ],
    )
    write_threshold_yaml(settings, "test-t10", file)
    thresholds = stage_thresholds(settings, settings.stage("followup-1"))
    np.testing.assert_array_equal(thresholds.band_edges, [20, 100, 400])
    np.testing.assert_allclose(
        thresholds.in_band("excess_ratio", np.array([25, 150])), [1.2345, 1.5]
    )
    np.testing.assert_allclose(thresholds.in_band("h1_l1_window", 150), [-2.0, 2.0])
    np.testing.assert_allclose(thresholds.in_band("seed_h1_l1_window", 50), [-3.0, 3.0])
    assert thresholds.growth is None and thresholds.excess_ratio_lowest is None
    assert thresholds.h1_l1_noise_weights is None


def test_missing_threshold_file(settings):
    with pytest.raises(FileNotFoundError, match="paws thresholds test-t10"):
        stage_thresholds(settings, settings.stage("followup-1"))


def test_stage_thresholds_check_the_coherence_times(settings):
    # followup-1 has 10-day candidates, 5-day seeds from search-0
    write_threshold_yaml(
        settings, "test-t10", threshold_file(20, 5, [band(20, 400, 1.2)])
    )
    with pytest.raises(ValueError, match="20-day"):
        stage_thresholds(settings, settings.stage("followup-1"))
    # followup-2's chain starts from search-0 (5 days): a growth cut over 10-day outliers does not fit
    write_threshold_yaml(
        settings,
        "test-t20",
        threshold_file(20, 10, [band(20, 400, 1.2, growth=9.0)], growth_tcoh=10),
    )
    with pytest.raises(ValueError, match="growth"):
        stage_thresholds(settings, settings.stage("followup-2"))


def test_threshold_file_fields_in_every_band_or_none():
    with pytest.raises(ValidationError, match="h1_l1_window"):
        threshold_file(
            10, 5, [band(20, 100, 1.2, h1_l1_window=(-1.0, 1.0)), band(100, 400, 1.3)]
        )
    with pytest.raises(ValidationError, match="contiguous"):
        threshold_file(10, 5, [band(20, 100, 1.2), band(200, 400, 1.3)])
    with pytest.raises(ValidationError, match="growth_tcoh"):
        threshold_file(10, 5, [band(20, 400, 1.2, growth=9.0)])


def injection_outlier_file(paths, settings, stage_name, freq, rows):
    """Clustered outlier file of an injection stage: (mean2F, mean2F_H1, mean2F_L1, injected Freq) per injection."""
    stage = settings.stage(stage_name)
    outliers = np.array(
        [row[:3] for row in rows],
        dtype=[("mean2F", ">f8"), ("mean2F_H1", ">f8"), ("mean2F_L1", ">f8")],
    )
    injections = np.array([(row[3],) for row in rows], dtype=[("Freq", ">f8")])
    path = paths.outlier_file(
        freq,
        stage.taskname(settings.target, freq),
        stage.name,
        cluster=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(data=outliers, name=f"{stage.name}_outlier"),
            fits.BinTableHDU(data=injections, name="injection"),
        ]
    ).writeto(path)


def test_calibrate_from_injection_outliers(settings, paths):
    # one 1 Hz band per threshold band; the injections are matched between the stages by their Freq
    for freq in (50, 150, 250, 350):
        # 2F - 4 = 10 and 20 at injections-0; 20 and 60 at injections-1 (in the other order)
        injection_outlier_file(
            paths,
            settings,
            "injections-0",
            freq,
            [(14.0, 14.0, 14.0, freq + 0.1), (24.0, 24.0, 24.0, freq + 0.2)],
        )
        # H1/L1 excess ratios 100 / 10 and 10 / 10
        injection_outlier_file(
            paths,
            settings,
            "injections-1",
            freq,
            [(64.0, 104.0, 14.0, freq + 0.2), (24.0, 14.0, 14.0, freq + 0.1)],
        )
    file = calibrate(settings, "test-t10")
    assert (file.tcoh, file.seed_tcoh, file.growth_tcoh) == (10, 5, None)
    assert file.calibration == settings.followup.calibrations["test-t10"]
    assert [band.f_min for band in file.bands] == [20, 100, 200, 300]
    first = file.bands[0]
    # excess ratios 60 / 20 = 3 and 20 / 10 = 2: 1st percentile 2.01, lowest 2
    assert (first.excess_ratio, first.excess_ratio_lowest) == (2.01, 2.0)
    # log10 H1/L1 excess ratios 1 and 0: the central 99% of the two
    assert first.h1_l1_window == pytest.approx((0.005, 0.995))
    # the seeds (injections-0) have 2F_H1 = 2F_L1
    assert first.seed_h1_l1_window == (0.0, 0.0)

    write_threshold_file(settings, "test-t10")
    thresholds = stage_thresholds(settings, settings.stage("followup-1"))
    assert thresholds.in_band("excess_ratio", 50) == 2.01
    np.testing.assert_allclose(thresholds.in_band("seed_h1_l1_window", 350), [0.0, 0.0])


def test_record_threshold_file(settings):
    path = write_threshold_yaml(
        settings, "test-t10", threshold_file(10, 5, [band(20, 400, 1.2)])
    )
    stage = settings.stage("followup-1")
    record_threshold_file(settings, stage)
    assert threshold_record_path(settings, stage.name).read_text() == path.read_text()


def test_log10_noise_ratio(settings):
    # no noise-weights file: the window is on r_HL itself
    assert list(log10_noise_ratio(settings, None, np.array([50, 51]))) == [0.0, 0.0]
    assert log10_noise_ratio(settings, None, 50) == 0.0
    # with one: log10(W_L1 / W_H1) of each candidate's 1 Hz band
    _write_rows(
        settings.config_dir / "noise.csv", [[50, 1.0, 10.0, 5, 5], [51, 4.0, 1.0, 5, 5]]
    )
    np.testing.assert_allclose(
        log10_noise_ratio(settings, "noise.csv", np.array([51, 50, 50])),
        [np.log10(0.25), 1.0, 1.0],
    )
    assert log10_noise_ratio(settings, "noise.csv", 50) == 1.0
