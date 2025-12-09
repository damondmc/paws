import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20

def normalize_array_with_nan(arr):
    """
    Normalizes a 2D array while ignoring NaN values.
    
    Args:
    arr (np.ndarray): Input 2D array with shape (m, n)
    
    Returns:
    np.ndarray: Normalized array with NaN values preserved
    """
    # Compute mean and standard deviation, ignoring NaN values
    max = np.nanmax(arr)
    min = np.nanmin(arr)
    
    # Normalize the array, keeping NaN values as they are
    normalized_arr = (arr - min)/(max-min)
    
    return normalized_arr

def plot_spectrograms(det, timestamps, frequency, fourier_data, stat=None):
    """
    Plots spectrograms for given detector data.
    Args:
        det (str): Detector name (e.g., 'H1', 'L1')
        timestamps (np.ndarray): Array of timestamps
        frequency (np.ndarray): Array of frequency bins
        fourier_data (np.ndarray): 2D array of Fourier amplitudes
        stat (dict, optional): Dictionary containing statistics for annotations
    Returns:
        fig, axs: Matplotlib figure and axes objects
    """
    fig, axs = plt.subplots(1, 1, figsize=(16, 10))
    axs = [axs]
    for ax in axs:
        ax.set(xlabel="Days", ylabel="Frequency [Hz]")

    time_in_days = (timestamps - timestamps[0]) / 86400

    if stat is None:
        axs[0].set_title("{0} spectrogram".format(det))
    else:
        axs[0].set_title("{0} spectrogram | mean".format(det) +r'$\mathcal{2F}$' +f" (H1L1/H1/L1)={stat['mean2F']:.1f}/{stat['mean2F_H1']:.1f}/{stat['mean2F_L1']:.1f}")
        
    c = axs[0].pcolormesh(
        time_in_days,
        frequency,
        fourier_data,
        norm=colors.CenteredNorm(),
    )
    if stat is not None:
        N = stat['coh2F_H1'].size
        segment_width = (time_in_days[-1] - time_in_days[0]) / N
        for i in range(1, N):
            line_pos = time_in_days[0] + i * segment_width
            axs[0].axvline(x=line_pos, color='black', linestyle='--', linewidth=1.5)
        
        freq_min = frequency[0]
        freq_max = frequency[-1]
        for i in range(N):
            # Center of the i-th segment in time
            segment_center_time = time_in_days[0] + (i+0.5) * segment_width
            # Center of the frequency range
            segment_center_freq = (freq_min + freq_max)/2 + (freq_max - freq_min)/4
            # Annotate with x[i], formatted to 2 decimal places
            axs[0].text(
                segment_center_time,
                segment_center_freq,
                r'$2\mathcal{F}_{H1}$'+f"= {stat['coh2F_H1'][i]:.2f}",
                color='black',
                fontsize=12,
                ha='center',
                va='center'
            )
            
            segment_center_freq = (freq_min + freq_max)/2 + (freq_max - freq_min)/5
            axs[0].text(
                segment_center_time,
                segment_center_freq,
                r'$2\mathcal{F}_{L1}$'+f"= {stat['coh2F_L1'][i]:.2f}",
                color='black',
                fontsize=12,
                ha='center',
                va='center'
            )
        
    #fig.colorbar(c, ax=axs[0], orientation="horizontal", label="Fourier Amplitude")
    return fig, axs
