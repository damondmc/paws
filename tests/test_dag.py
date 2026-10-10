import re

import numpy as np
import pytest

from paws.workflow.manager import WorkflowManager

TASKNAME = "TestTarget_followup-2_TCoh20_O2_50Hz"
D_ALPHA = np.array([-1e-4, 0.0, 1e-4])
D_DELTA = np.array([0.0, 0.0, 5e-5])


def seed_params():
    names = ["alpha", "dalpha", "delta", "ddelta", "freq", "df", "f1dot", "df1dot", "f2dot", "df2dot"]
    params = np.zeros(2, dtype=[(name, float) for name in names])
    params["alpha"], params["delta"] = 4.6, -0.5
    params["freq"], params["df"] = [50.01, 50.02], 1e-5
    params["f1dot"], params["df1dot"] = -1e-10, 1e-12
    params["f2dot"], params["df2dot"] = 1e-20, 1e-19
    return params


def arg(line, name):
    return re.search(rf"--{name}=(\S+)", line).group(1)


@pytest.fixture
def dag_file(settings):
    manager = WorkflowManager(settings.config, settings.target)
    sft_files = ["osdf:///igwn/sfts/H1/a.sft", "osdf:///igwn/sfts/L1/b.sft"]
    return manager.make_search_dag(
        TASKNAME, 50, seed_params(), num_top_list=1000, stage="followup-2", freq_deriv_order=2, n_seg=27,
        sft_files=sft_files, metric_file="osdf:///igwn/metricSetup/test_t20.fts", request_memory="2GB",
        request_disk="4GB", request_cpu=1, use_osg=True, use_osdf=True, tasks_per_job=4,
        sky_offsets=(D_ALPHA, D_DELTA),
    )


def task_lines(dag_file, node):
    return (dag_file.parent / "tasks" / f"{TASKNAME}_task_{node}.txt").read_text().splitlines()


def test_nodes_batch_the_weave_runs(dag_file):
    dag = dag_file.read_text()
    assert dag.count("\nJOB ") + dag.startswith("JOB ") == 2  # 6 Weave runs, 4 per node
    assert len(task_lines(dag_file, 1)) == 4 and len(task_lines(dag_file, 2)) == 2


def test_jobs_tile_each_seed_over_the_sky_grid(dag_file):
    lines = task_lines(dag_file, 1) + task_lines(dag_file, 2)
    assert [arg(line, "output-file") for line in lines] == [f"50Hz_out.fts.{job}" for job in range(1, 7)]
    for job, line in enumerate(lines):
        seed, sky_point = divmod(job, 3)
        alpha, dalpha = arg(line, "alpha").split("/")
        assert float(alpha) == pytest.approx(4.6 + D_ALPHA[sky_point]) and float(dalpha) == 0
        assert float(arg(line, "delta").split("/")[0]) == pytest.approx(-0.5 + D_DELTA[sky_point])
        assert arg(line, "freq") == f"{[50.01, 50.02][seed]}/1e-05"
        assert arg(line, "f1dot") == "-1e-10/1e-12"


def test_weave_arguments(dag_file):
    line = task_lines(dag_file, 1)[0]
    assert '--sft-files="a.sft;b.sft"' in line
    assert arg(line, "setup-file") == "test_t20.fts"
    assert arg(line, "toplist-limit") == "1000"
    assert arg(line, "semi-max-mismatch") == "0.2" and arg(line, "coh-max-mismatch") == "0.1"


def test_outputs_remap_to_osdf_result_files(dag_file, paths):
    second_node = [line for line in dag_file.read_text().splitlines() if line.startswith("VARS ")][1]
    remaps = re.search(r'REMAP_OUTPUT_FILES="([^"]*)"', second_node).group(1).split(";")
    assert [remap.split("=")[0] for remap in remaps] == ["50Hz_out.fts.5", "50Hz_out.fts.6"]
    expected = paths.to_osdf_url(paths.weave_output_file(50, TASKNAME, 6, "followup-2"))
    assert remaps[1].split("=", 1)[1] == expected
    transfers = re.search(r'TRANSFER_FILES="([^"]*)"', second_node).group(1).split(", ")
    assert transfers[:3] == ["osdf:///igwn/sfts/H1/a.sft", "osdf:///igwn/sfts/L1/b.sft",
                             "osdf:///igwn/metricSetup/test_t20.fts"]


def test_sub_file(dag_file, settings):
    sub = dag_file.with_suffix(".sub").read_text()
    assert "request_memory = 2GB" in sub
    assert f"accounting_group = {settings.config.acc_group}" in sub
    assert "arguments = run_weave_batch_followup-2.sh $(CMD_ARGS)" in sub
