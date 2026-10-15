from paws.filepaths import make_dir

TASKNAME = "TestTarget_followup-1_TCoh10_O2_50Hz"


def test_outlier_file(paths, settings):
    clustered = paths.outlier_file(50, TASKNAME, "followup-1", cluster=True)
    unclustered = paths.outlier_file(50, TASKNAME, "followup-1", cluster=False)
    outliers_dir = (
        settings.environment.home_dir
        / "results/followup-1/TestTarget/TEST_SFTS/50/Outliers"
    )
    assert clustered == outliers_dir / f"{TASKNAME}_outlier_clustered.fts"
    assert unclustered == outliers_dir / f"{TASKNAME}_outlier.fts"


def test_weave_output_file(paths, settings):
    path = paths.weave_output_file(50, TASKNAME, 7, "followup-1")
    assert path == (
        settings.environment.osdf_dir
        / "o4ab/results/followup-1/TestTarget/TEST_SFTS/50/Result"
        / f"{TASKNAME}.fts.7"
    )


def test_to_osdf_url(paths):
    assert (
        paths.to_osdf_url("/osdf/igwn/cit/staging/user/x.fts")
        == "osdf:///igwn/cit/staging/user/x.fts"
    )


def test_make_dir(tmp_path):
    make_dir([tmp_path / "a/b/file.txt", tmp_path / "c/file.txt"])
    assert (tmp_path / "a/b").is_dir() and (tmp_path / "c").is_dir()
