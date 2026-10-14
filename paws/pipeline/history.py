"""History of a follow-up candidate: its statistics at the earlier stages of its chain (Settings.chain_stages).

Each follow-up of real candidates writes its seed's own statistics as <column>_TCoh<seed stage's tcoh> columns and
copies the ones the seed already has, so an outlier at T_coh = 20 days has the _TCoh5 and _TCoh10 columns.
"""

import numpy as np

HISTORY_COLUMNS = ("mean2F", "mean2F_H1", "mean2F_L1", "coh2F_H1", "coh2F_L1")


def history_column(column, stage):
    return f"{column}_TCoh{stage.tcoh}"


def chain_history(chain, seed_rows):
    """{<column>_TCoh<days>: per-seed values} over the chain of the seeds: the history columns the seeds have from the
    earlier stages, and the seeds' own columns under the coherence time of their stage."""
    *earlier, prev = chain
    history = {}
    for source in earlier:
        for column in HISTORY_COLUMNS:
            name = history_column(column, source)
            if name not in seed_rows.dtype.names:
                raise ValueError(
                    f"the seeds from {prev.name} have no {name} column; recollect its outliers"
                )
            history[name] = np.asarray(seed_rows[name])
    for column in HISTORY_COLUMNS:
        history[history_column(column, prev)] = np.asarray(seed_rows[column])
    return history
