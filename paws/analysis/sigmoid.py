"""Sigmoid fit of the detection efficiency against the injected strain, for the h95 upper limit."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Bool, Float, Real
from matplotlib.figure import Figure
from scipy.optimize import minimize

from paws.settings import Target

BRACKET_FACTORS = (
    0.5,
    0.7,
    1.0,
    1.5,
)  # first strains tried, as factors of the h0 estimate
DRAW_WINDOW = (
    0.8,
    1.2,
)  # range of h0 / h95 over which the strains of the drawn injections are log-uniform


class UpperLimit(NamedTuple):
    """h95 with its fit uncertainty, the starting sets (strain, efficiency) and the drawn injections of the
    round that gave h95 (strain, detected)."""

    h95: float
    dh95: float
    h0: Float[np.ndarray, " n_set"]
    efficiency: Float[np.ndarray, " n_set"]
    drawn_h0: Float[np.ndarray, " n_drawn"]
    drawn_detected: Bool[np.ndarray, " n_drawn"]


class SigmoidFitter:
    """
    Handles the sigmoid curve fitting to determine upper limits (h95).
    """

    def __init__(self, target: Target, n_inj: int = 100, n_amp: int = 1) -> None:
        self.target = target
        self.injPerPoint = int(n_inj / n_amp)
        self.popt: Optional[Float[np.ndarray, "2"]] = None  # (steepness, midpoint)
        self.pcov: Optional[Float[np.ndarray, "2 2"]] = None
        self.h0_mean: float = 0
        self.h0_max: float = 1

    @staticmethod
    def sigmoid(
        x: Union[float, Float[np.ndarray, " n_point"]], k: float, x0: float
    ) -> Union[float, Float[np.ndarray, " n_point"]]:
        """Sigmoid function: 1 / (1 + exp(-k*(x-x0)))"""
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    @staticmethod
    def inv_sigmoid(y: float, k: float, x0: float) -> float:
        """Inverse sigmoid function to find x for a given detection probability y."""
        # Clip y to avoid log(0) or log(negative)
        y = np.clip(y, 1e-9, 1.0 - 1e-9)
        return -np.log(1.0 / y - 1.0) / k + x0

    def _rescale_h0(
        self,
        h0: Union[float, Float[np.ndarray, " n_point"]],
        h0_list: Float[np.ndarray, " n_h0"],
    ) -> Union[float, Float[np.ndarray, " n_point"]]:
        """Normalizes h0 values for better fitting stability."""
        self.h0_mean = np.mean(h0_list)
        self.h0_max = np.max(h0_list)
        return (h0 - self.h0_mean) / self.h0_max

    def _inv_rescale_h0(self, x: float) -> float:
        """Converts normalized x back to physical h0."""
        return x * self.h0_max + self.h0_mean

    def binomial_error(
        self, y: Float[np.ndarray, " n_h0"], n: int
    ) -> Float[np.ndarray, " n_h0"]:
        """Binomial standard error of the efficiencies, for the error bars of the plot."""
        err = np.sqrt(y * (1.0 - y) / n)
        # a visible error bar at 0 and 100 % efficiency
        err[err == 0] = 1.0 / n
        return err

    def fit(
        self,
        h0_list: Float[np.ndarray, " n_h0"],
        efficiency_list: Float[np.ndarray, " n_h0"],
    ) -> bool:
        """
        Maximum-likelihood fit of the sigmoid to the efficiencies of injection sets of injPerPoint injections
        each (see fit_counts). Returns False if the fit does not converge to an increasing sigmoid.
        """
        n = self.injPerPoint
        n_detected = np.round(np.asarray(efficiency_list) * n)
        return self.fit_counts(h0_list, n_detected, np.full(len(n_detected), n))

    def fit_counts(
        self,
        h0_list: Float[np.ndarray, " n_h0"],
        n_detected: Real[np.ndarray, " n_h0"],
        n_injected: Real[np.ndarray, " n_h0"],
    ) -> bool:
        """
        Maximum-likelihood fit of the sigmoid to detection counts: n_detected of n_injected injections at each
        h0 is binomial(n_injected, sigmoid(h0)), so single injections (n_injected = 1) at their own h0 can be
        fitted. popt is (steepness, midpoint) in the rescaled h0 and pcov its covariance from the Fisher
        information. Returns False if the fit does not converge to an increasing sigmoid.
        """
        h0_arr = np.asarray(h0_list, dtype=float)
        n = np.asarray(n_injected, dtype=float)
        k = np.asarray(n_detected, dtype=float)
        x_norm = self._rescale_h0(h0_arr, h0_arr)
        # rows (x, 1), so that predictor_matrix @ (a, b) = a * x + b; the negative
        # log-likelihood is convex in (a, b)
        predictor_matrix = np.column_stack([x_norm, np.ones_like(x_norm)])

        def nll(theta: Float[np.ndarray, "2"]) -> float:
            z = predictor_matrix @ theta
            return np.sum(k * np.logaddexp(0, -z) + (n - k) * np.logaddexp(0, z))

        def grad(theta: Float[np.ndarray, "2"]) -> Float[np.ndarray, "2"]:
            return predictor_matrix.T @ (
                n * self.sigmoid(predictor_matrix @ theta, 1.0, 0.0) - k
            )

        def hess(theta: Float[np.ndarray, "2"]) -> Float[np.ndarray, "2 2"]:
            q = self.sigmoid(predictor_matrix @ theta, 1.0, 0.0)
            return predictor_matrix.T @ (predictor_matrix * (n * q * (1 - q))[:, None])

        result = minimize(
            nll, x0=np.array([5.0, 0.0]), jac=grad, hess=hess, method="trust-exact"
        )
        a, b = result.x
        if not result.success or a <= 0:
            logging.warning(
                f"Sigmoid fitting failed: {result.message} (steepness {a:.3g})"
            )
            return False
        # (k, x0) = (a, -b / a)
        jacobian = np.array([[1.0, 0.0], [b / a**2, -1.0 / a]])
        self.popt = np.array([a, -b / a])
        self.pcov = jacobian @ np.linalg.inv(hess(result.x)) @ jacobian.T
        return True

    def get_h_percentile(self, percentile: float = 0.95) -> tuple[float, float]:
        """
        Calculates the h0 value at a specific detection probability (e.g., 95%).
        Returns: (h_value, h_error)
        """
        if self.popt is None:
            raise ValueError("Model not fitted. Run fit() first.")

        x = self.inv_sigmoid(percentile, *self.popt)

        # Propagate errors using covariance matrix
        k, x0 = self.popt
        # Partial derivative wrt k
        dk_da = -1.0 / k**2 * np.log(percentile / (1.0 - percentile))
        dk_db = 1.0  # Partial derivative wrt x0

        # Variance of x
        var_x = (
            self.pcov[0, 0] * dk_da**2
            + self.pcov[1, 1] * dk_db**2
            + 2 * self.pcov[0, 1] * dk_da * dk_db
        )

        dx = np.sqrt(var_x)

        h_val = self._inv_rescale_h0(x)
        h_err = dx * self.h0_max  # Scale error back to physical units

        return h_val, h_err

    def plot(
        self,
        h0_list: Float[np.ndarray, " n_h0"],
        p_list: Float[np.ndarray, " n_h0"],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Generates the Upper Limit plot."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Generate smooth curve
        h99, _ = self.get_h_percentile(0.99)
        h01, _ = self.get_h_percentile(0.01)
        h_smooth = np.linspace(h01 * 0.9, h99 * 1.1, 100)
        x_smooth = self._rescale_h0(h_smooth, np.array(h0_list))
        p_smooth = self.sigmoid(x_smooth, *self.popt)

        ax.plot(h_smooth, p_smooth, label="Sigmoid Fit", color="C0", zorder=1)

        # Plot Data Points
        err = self.binomial_error(np.array(p_list), self.injPerPoint)
        ax.errorbar(
            h0_list,
            p_list,
            yerr=err,
            fmt="o",
            label="Injection Data",
            color="k",
            zorder=2,
        )

        # Highlight h95
        h95, dh95 = self.get_h_percentile(0.95)
        ax.axhline(0.95, color="r", linestyle="--", alpha=0.5)
        ax.axvline(h95, color="r", linestyle="--", alpha=0.5)

        h95_exp = int(np.floor(np.log10(h95)))
        h95_mantissa = h95 / (10**h95_exp)
        label_text = rf"$h_{{95}} = {h95_mantissa:.2f} \times 10^{{{h95_exp}}} \pm {dh95 / h95 * 100:.1f}\%$"
        ax.plot([], [], " ", label=label_text)

        ax.set_xlabel(r"Strain Amplitude $h_0$", fontsize=14)
        ax.set_ylabel(r"Detection Efficiency $p_{\mathrm{det}}$", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig


def find_h95(
    efficiency_at: Callable[[float], float],
    detected_at: Callable[
        [Float[np.ndarray, " n_drawn"]], Bool[np.ndarray, " n_drawn"]
    ],
    h0_estimate: float,
    fitter: SigmoidFitter,
    n_drawn: int = 2000,
    tolerance: float = 0.05,
    max_rounds: int = 3,
) -> UpperLimit:
    """h95 from a sigmoid fit around it. The sets at BRACKET_FACTORS * h0_estimate (efficiency_at: the
    efficiency of a set at one strain), fitted together, give a first h95. Each round then draws n_drawn
    injections, each at its own strain log-uniform in DRAW_WINDOW * h95 (detected_at: which of the injections
    at these strains are detected), and fits them alone. h95 is accepted when its relative uncertainty is
    within tolerance and it lies within DRAW_WINDOW of the previous h95; otherwise the next round is centred
    on it. On return, fitter holds the accepted fit."""
    h0_values = np.array([h0_estimate * factor for factor in BRACKET_FACTORS])
    efficiencies = np.array([efficiency_at(h0) for h0 in h0_values])
    if not fitter.fit(h0_values, efficiencies):
        raise RuntimeError("the sigmoid fit does not constrain h95")
    centre, _ = fitter.get_h_percentile(0.95)
    low, high = DRAW_WINDOW
    for _ in range(max_rounds):
        if not np.isfinite(centre) or centre <= 0:
            raise RuntimeError(
                f"the sigmoid fit does not constrain h95 (h95 = {centre:.3e})"
            )
        drawn_h0 = centre * np.exp(
            np.random.uniform(np.log(low), np.log(high), n_drawn)
        )
        detected = np.asarray(detected_at(drawn_h0), dtype=bool)
        if not fitter.fit_counts(drawn_h0, detected.astype(float), np.ones(n_drawn)):
            raise RuntimeError("the sigmoid fit does not constrain h95")
        h95, dh95 = fitter.get_h_percentile(0.95)
        relative = dh95 / h95
        logging.info(
            f"h95 = {h95:.3e} +- {abs(relative):.2%} from {n_drawn} injections around {centre:.3e} "
            f"({detected.mean():.2%} detected)"
        )
        if abs(relative) <= tolerance and low <= h95 / centre <= high:
            return UpperLimit(h95, dh95, h0_values, efficiencies, drawn_h0, detected)
        centre = h95
    raise RuntimeError(
        f"h95 did not converge to {tolerance:.0%} around h95 in {max_rounds} rounds"
    )
