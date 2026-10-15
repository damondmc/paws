"""Names of the phase parameters, the injection parameters and the tasks; the time constants; the plot style."""

from pathlib import Path

SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY

MPLSTYLE = Path(__file__).with_name("paws.mplstyle")  # plt.style.use(MPLSTYLE)


def phase_param_name(order: int) -> tuple[list[str], list[str]]:
    """(frequency parameter names, their spacing names) up to spin-down order `order`."""
    freqParamName = ["freq", "f1dot", "f2dot", "f3dot", "f4dot"]
    freqDerivParamName = ["df", "df1dot", "df2dot", "df3dot", "df4dot"]
    return freqParamName[: order + 1], freqDerivParamName[: order + 1]


def ext_param_name() -> list[str]:
    """Injection parameters besides the phase parameters."""
    return ["Alpha", "Delta", "refTime", "aPlus", "aCross", "psi"]


def task_name(target_name: str, stage: str, cohDay: int, order: int) -> str:
    """Standardized task naming convention."""
    return f"{target_name}_{stage}_TCoh{cohDay}_O{order}"
