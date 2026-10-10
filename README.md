# PAWS
**P**ython **A**utomation for **W**eave **S**earches

PAWS is a comprehensive Python wrapper designed to automate and manage directed continuous gravitational wave searches. It serves as an orchestration layer for the **Weave** search pipeline, handling the generation of HTCondor workflows (DAGs), parameter space partitioning, post-search data analysis, follow-up analysis and upper-limits determinations.

## 📦 Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install .
```

## 🛠️ Container Build

To build the Apptainer image for OSG deployment:

```bash
apptainer build paws.sif paws.def
```
