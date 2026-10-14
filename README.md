# PAWS
**P**ython **A**utomation for **W**eave **S**earches

PAWS is a comprehensive Python wrapper designed to automate and manage directed continuous gravitational wave searches. It serves as an orchestration layer for the **Weave** search pipeline, handling the generation of HTCondor workflows (DAGs), parameter space partitioning, post-search data analysis, follow-up analysis and upper-limits determinations.

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install .
```

## Container Build

To build the Apptainer image for OSG deployment:

```bash
apptainer build paws.sif paws.def
```

## Types

Every function is annotated. Arrays carry their dtype and named shape with
[jaxtyping](https://docs.kidger.site/jaxtyping/), e.g. `Float[np.ndarray, " n_candidate"]` or
`Float[np.ndarray, "n_threshold_band 2"]` (a single dimension name starts with a space so that ruff does not read
it as a Python name). FITS table rows are `Shaped[np.ndarray, " n_seed"]`. The tests check every call against these
annotations: `tests/conftest.py` installs the jaxtyping import hook with beartype (`tests/typecheck.py`), so a wrong
type, dtype or shape fails the test that triggers it.
