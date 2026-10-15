# PAWS
**P**ython **A**utomation for **W**eave **S**earches

PAWS is a comprehensive Python wrapper designed to automate and manage directed continuous gravitational wave searches. It serves as an orchestration layer for the **Weave** search pipeline, handling the generation of HTCondor workflows (DAGs), parameter space partitioning, post-search data analysis, follow-up analysis and upper-limits determinations.

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Container Build

To build the Apptainer image for OSG deployment:

```bash
apptainer build paws.sif paws.def
```

## Usage

A search repo holds a config directory with `environment.yaml`, `data.yaml`, `search.yaml`, `followup.yaml` and the
`spindown.py` model script (see `config/` for a commented example), given as `--config-dir` or `$PAWS_CONFIG_DIR`.
The stages of `search.yaml` and `followup.yaml` are run in order, each one seeded by the outliers of the one before.
`paws <command> -h` lists the options of a command; `--bands` restricts `dag` and `outliers` to selected 1 Hz bands.

```bash
export PAWS_CONFIG_DIR=/path/to/config
```

### Setup

`paws metric <tcoh>` reads the SFT timestamps, writes the segment list of a coherence time in days and checks the
data coverage; with `--metric` it also runs `lalpulsar_WeaveSetup` for the metric file that a stage names in its
`metric` field. Once per coherence time:

```bash
paws metric 5 --metric
```

`paws noise-weights <file>` sums 1 / S_h over the SFTs of each 1 Hz band and detector, for the noise-normalized
H1/L1 window of the thresholds (`h1_l1_noise_weights` in `followup.yaml`):

```bash
paws noise-weights noise_weights.csv --bands $(seq 20 399)
```

### A stage

paws does not run Weave itself: it writes HTCondor DAGs, you submit them, and once the jobs are done it collects
their results. Each stage is the same three steps.

1. `paws dag <stage>` writes the Weave DAG of each band under `condorFiles/<stage>/` and a list of those DAG files
   under `dagFiles/`. For a follow-up it reads the previous stage's outlier files to place the seeds.
2. Submit the list to Condor. The jobs run on the OSG, read the SFTs and write their Weave result files to OSDF.
3. `paws outliers <stage>` reads the Weave results and writes the outlier files of each band under `results/`,
   which seed the next stage.

The initial search:

```bash
paws dag search-0
condor_submit_dag $(cat dagFiles/search-0_<target>_dag20-400Hz.txt)
paws outliers search-0      # once the jobs are done
```

An injection stage (`injection:` in `followup.yaml`) exists to calibrate the follow-up thresholds: it injects
signals at the upper-limit h0 into the non-saturated sub-bands of the search and follows them up at the same
coherence times as the real candidates, so that each cut of the follow-up can be set to lose a known fraction of
real signals. It is run the same way:

```bash
paws dag injections-0
condor_submit_dag $(cat dagFiles/injections-0_<target>_dag20-400Hz.txt)
paws outliers injections-0
```

### Thresholds and the follow-up chain

`paws thresholds <name>` calibrates the cuts of a calibration in `followup.yaml` on its two injection stages and
writes `thresholds/<name>.yaml` into the config directory. A follow-up of the real candidates applies that file with
`thresholds: <name>`: its outliers are the seeds whose loudest candidate passes the cuts.

```bash
paws thresholds t10
paws dag followup-1
condor_submit_dag $(cat dagFiles/followup-1_<target>_dag20-400Hz.txt)
paws outliers followup-1
```
