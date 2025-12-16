import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings

class SigmoidFitter:
    """
    Handles the sigmoid curve fitting to determine upper limits (h95).
    """
    def __init__(self, target, nInj=100, nAmp=1):
        self.target = target
        self.injPerPoint = int(nInj / nAmp)
        self.popt = None
        self.pcov = None
        self.h0_mean = 0
        self.h0_max = 1

    @staticmethod
    def sigmoid(x, k, x0):
        """Sigmoid function: 1 / (1 + exp(-k*(x-x0)))"""
        return 1. / (1. + np.exp(-k * (x - x0)))

    @staticmethod
    def inv_sigmoid(y, k, x0):
        """Inverse sigmoid function to find x for a given detection probability y."""
        # Clip y to avoid log(0) or log(negative)
        y = np.clip(y, 1e-9, 1.0 - 1e-9)
        return -np.log(1. / y - 1.) / k + x0

    def _rescale_h0(self, h0, h0_list):
        """Normalizes h0 values for better fitting stability."""
        self.h0_mean = np.mean(h0_list)
        self.h0_max = np.max(h0_list)
        return (h0 - self.h0_mean) / self.h0_max

    def _inv_rescale_h0(self, x):
        """Converts normalized x back to physical h0."""
        return x * self.h0_max + self.h0_mean

    def binomial_error(self, y, n):
        """Calculates standard error for binomial distribution."""
        err = np.sqrt(y * (1. - y) / n)
        # Avoid zero error for fitting weights
        err[err == 0] = 1. / n
        return err

    def fit(self, h0_list, efficiency_list):
        """
        Fits the sigmoid curve to the h0 vs efficiency data.
        """
        h0_arr = np.array(h0_list)
        p_arr = np.array(efficiency_list)

        if not np.any((p_arr > 0.8) & (p_arr < 0.96)):
            warnings.warn("Warning: Data lacks points in the critical 80%-96% efficiency range.")

        err = self.binomial_error(p_arr, self.injPerPoint)
        x_norm = self._rescale_h0(h0_arr, h0_arr)
        
        # Initial guess: steepness=5, midpoint=0 (normalized)
        try:
            self.popt, self.pcov = curve_fit(self.sigmoid, x_norm, p_arr, p0=[5, 0], sigma=err, absolute_sigma=True)
            return True
        except RuntimeError as e:
            print(f"Sigmoid fitting failed: {e}")
            return False

    def get_h_percentile(self, percentile=0.95):
        """
        Calculates the h0 value at a specific detection probability (e.g., 95%).
        Returns: (h_value, h_error)
        """
        if self.popt is None:
            raise ValueError("Model not fitted. Run fit() first.")

        x = self.inv_sigmoid(percentile, *self.popt)
        
        # Propagate errors using covariance matrix
        k, x0 = self.popt
        dk_da = -1.0 / k**2 * np.log(percentile / (1.0 - percentile)) # Partial derivative wrt k
        dk_db = 1.0                                                   # Partial derivative wrt x0
        
        # Variance of x
        var_x = (self.pcov[0, 0] * dk_da**2 + 
                 self.pcov[1, 1] * dk_db**2 + 
                 2 * self.pcov[0, 1] * dk_da * dk_db)
        
        dx = np.sqrt(var_x)
        
        h_val = self._inv_rescale_h0(x)
        h_err = dx * self.h0_max # Scale error back to physical units
        
        return h_val, h_err 

    def plot(self, h0_list, p_list, save_path=None):
        """Generates the Upper Limit plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate smooth curve
        h99, _ = self.get_h_percentile(0.99)
        h01, _ = self.get_h_percentile(0.01)
        h_smooth = np.linspace(h01 * 0.9, h99 * 1.1, 100)
        x_smooth = self._rescale_h0(h_smooth, np.array(h0_list))
        p_smooth = self.sigmoid(x_smooth, *self.popt)
        
        ax.plot(h_smooth, p_smooth, label='Sigmoid Fit', color='C0', zorder=1)
        
        # Plot Data Points
        err = self.binomial_error(np.array(p_list), self.injPerPoint)
        ax.errorbar(h0_list, p_list, yerr=err, fmt='o', label='Injection Data', color='k', zorder=2)

        # Highlight h95
        h95, h95_rel_err = self.get_h_percentile(0.95)
        ax.axhline(0.95, color='r', linestyle='--', alpha=0.5)
        ax.axvline(h95, color='r', linestyle='--', alpha=0.5)
        
        label_text = r'$h_{95} = ' + f'{h95:.3e}' + r' \pm ' + f'{h95_rel_err*100:.1f}\% $'
        ax.plot([], [], ' ', label=label_text) # Dummy entry for legend

        ax.set_xlabel(r'Strain Amplitude $h_0$', fontsize=14)
        ax.set_ylabel(r'Detection Efficiency $p_{\mathrm{det}}$', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return fig