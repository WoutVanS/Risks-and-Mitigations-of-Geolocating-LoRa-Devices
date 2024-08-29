# File: Lora_Correlation_GUI.py
# Author: Wout Van Steenbergen
# Project: Risks and Mitigations of Geolocating LoRa Devices
# Date: 2024-08-29
#
# Description:
# This script visualize the effects of bandwidth, SNR and multipath on tempural resolution .
# It is used in Chapter, 4 Section 4.6 of the dissertation.
#
# Dependencies:
# - Python 3.9.13
# - numpy
# - matplotlib
# - scipy


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, correlate
from matplotlib.widgets import Slider, Button, CheckButtons

import matplotlib
matplotlib.use('TkAgg')
# Function to add AWGN noise to a signal
def add_awgn_noise(signal, snr_db):
    """
    Add AWGN noise to a signal based on the specified SNR (in dB).

    Parameters:
    signal (numpy array): The original signal
    snr_db (float): Desired signal-to-noise ratio in dB

    Returns:
    numpy array: Signal with added AWGN noise
    """

    noise_real = np.random.normal(0, 1, 2 * len(t) + len(signal))
    noise_imag = np.random.normal(0, 1, 2 * len(t) + len(signal))
    noisy_signal = noise_real + 1j * noise_imag

    noise_power = np.mean(noisy_signal ** 2)
    desired_signal_power = noise_power * (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(desired_signal_power / np.mean(signal ** 2))
    scaled_signal = signal * scaling_factor

    noisy_signal[len(t):len(t) + len(scaled_signal)] += scaled_signal

    return noisy_signal


def channel(signal, snr_db, distance):
    """
    Simulate attenuation additive noise effects on signal.

    Parameters:
    signal (numpy array): The original signal
    snr_db (float): Desired signal-to-noise ratio in dB

    Returns:
    numpy array: Signal with added noise and attenuation
    """

    # Add AWGN noise to the modulated signal
    noisy_signal = add_awgn_noise(signal, snr_db)

    delta_fspl = 20 * np.log10(1 + distance / d1)
    path_loss_linear = 10 ** (delta_fspl / 10)
    attenuated_signal = noisy_signal / np.sqrt(path_loss_linear)
    delayed_signal = attenuated_signal[len(t):-len(t)] #add_awgn_noise(attenuated_signal, snr_db)

    time_delay = distance / 3e8
    delay_samples = time_delay * fs

    if multipath: noisy_signal[int(len(t)) + int(delay_samples) - 1: int(len(t)) + int(delay_samples) - 1 + len(
        delayed_signal)] += delayed_signal

    return noisy_signal


def correlate_preamble(received_signal, preamble):
    """
    Correlate received signal with preamble.

    Parameters:
    received_signal (numpy array): The received signal
    preamble (numpy array): The original preamble

    Returns:
    numpy array: Correlation magnitude values
    int: Index value of the peak
    """

    correlation = correlate(received_signal, preamble, mode='full', method='fft')
    correlation_magnitude = np.abs(correlation)
    peak_index = np.argmax(correlation_magnitude)
    return correlation_magnitude, peak_index


def generate_complex_chirp(t, f0, f1, T_sym):
    """
    Generates a complex chirp signal.

    Parameters:
    t (np.linspace): Time vector
    f0 (float): Starting frequency of the chirp
    f1 (float): Ending frequency of the chirp
    T_sym (float): Symbol duration

    Returns:
    numpy.ndarray: Complex chirp signal
    """
    chirp_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * T_sym)))
    preamble = np.tile(chirp_signal, N)
    return preamble


# Function to update multipath status
def update_multipath(label):
    """
    Update the multipath status based on the checkbox.

    Parameters:
    label (str): The label of the checkbox

    Returns:
    None
    """
    global multipath
    if label == 'Multipath':
        multipath = not multipath

    # Show or hide the distance slider based on the multipath checkbox status
    if multipath:
        distance_slider_ax.set_visible(True)
    else:
        distance_slider_ax.set_visible(False)
    plt.draw()
    update_plot(None)

def update_plot(val):
    global t, BW, SF, N, snr_db, T_sym, preamble, modulated_signal, noisy_signal, demodulated_signal, correlation_magnitude, peak_index, frequencies, times, Sxx,allowed_bw_values

    # Update parameters
    BW = bw_slider.val * 1e3
    SF = int(sf_slider.val)
    N = int(preamble_slider.val)
    snr_db = snr_slider.val
    multipath_distance = distance_slider.val if multipath else 0

    # Update distance slider range based on the new BW value
    distance_slider.valmin = 1
    distance_slider.valmax = 3e8 / BW
    distance_slider.ax.set_xlim(distance_slider.valmin, distance_slider.valmax)
    if distance_slider.val > distance_slider.valmax:
        distance_slider.set_val(distance_slider.valmax)

    # Recalculate dependent parameters
    T_sym = (2 ** SF) / BW
    t = np.linspace(0, T_sym, int(fs * T_sym))
    t_full = np.linspace(0, T_sym * N, int(fs * T_sym * N))

    preamble = generate_complex_chirp(t, f0=0, f1=BW, T_sym=T_sym)
    modulated_signal = preamble
    noisy_signal = channel(modulated_signal, snr_db, multipath_distance)
    demodulated_signal = noisy_signal

    # Correlate and calculate spectrogram
    correlation_magnitude, peak_index = correlate_preamble(demodulated_signal, preamble)
    expected_start_index = len(t) + len(preamble) - 1  # Expected start index for zero delay
    delay_error = peak_index - expected_start_index
    delay_error_time = delay_error / fs

    frequencies, times, Sxx = spectrogram(demodulated_signal.real, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # Update the plots
    start_of_signal = (len(t) - 1) * 1e3 / fs
    end_of_signal = (len(t) + len(preamble) - 1) * 1e3 / fs
    t_noisy_signal = np.linspace(0, len(noisy_signal) / fs, len(noisy_signal))

    axs[0, 1].cla()
    axs[0, 1].plot(t_noisy_signal * 1e3, demodulated_signal)
    axs[0, 1].axvline(start_of_signal, color='g', linestyle='-', label='Start of preamble')
    axs[0, 1].axvline(end_of_signal, color='r', linestyle='-', label='End of preamble')
    axs[0, 1].legend()
    axs[0, 1].set_title('Sinusoidal Representation of Received LoRa Preamble')
    axs[0, 1].set_xlabel('Time (ms)')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].grid(True)

    axs[1, 0].cla()
    pcm = axs[1, 0].pcolormesh(times * 1e3, frequencies / 1e3, 10 * np.log10(Sxx), shading='gouraud')
    axs[1, 0].set_ylabel('Frequency (kHz)')
    axs[1, 0].set_xlabel('Time (ms)')
    axs[1, 0].set_title('Spectrogram')
    axs[1, 0].set_ylim(0, 2 * BW / 1e3)  # Limit to LoRa bandwidth

    axs[1, 1].cla()
    t_correlation = np.linspace(0, len(correlation_magnitude) / fs, len(correlation_magnitude))
    distance_correlation = t_correlation * 3e8

    d_peak_index = peak_index * 3e8 / fs
    axs[1, 1].plot(distance_correlation, correlation_magnitude)
    axs[1, 1].axvline(d_peak_index, color='r', linestyle='--', label=f'Peak error: {delay_error_time * 3e8:.2f} m')
    axs[1, 1].set_title('Correlation between Received Signal and Preamble')
    axs[1, 1].set_ylabel('Magnitude')
    axs[1, 1].set_xlabel('Distance (m)')
    axs[1, 1].legend()
    plt.draw()

# Parameters for LoRa chirps
BW = 125e3  # LoRa Bandwidth in Hz
CR = 868e6  #carrier frequency
fs = 5e6  # Sampling frequency in Hz (Nyquist criterion is not enough)
SF = 7     # LoRa Spreading Factor
T_sym = (2 ** SF) / BW  # Symbol time for LoRa
N = 2 #number of chirps in preamble
snr_db = 20  # Desired SNR in dB

multipath = False
multipath_distance = 0
d1 = 5000    #lign of sight distance

allowed_bw_values = [125, 250, 500]

# Calculate the spectrogram with improved resolution
nperseg = 1024  # Increase FFT size for better frequency resolution
noverlap = 512  # Increase overlap for better time resolution
nfft = 2048  # Zero-padding for smoother frequency representation

# Generate time array for one symbol duration
t = np.linspace(0, T_sym, int(fs * T_sym))
t_full = np.linspace(0, T_sym * N, int(fs * T_sym * N))


preamble = generate_complex_chirp(t, f0=0, f1=BW, T_sym=T_sym)

# Modulate the chirp with the carrier frequency
modulated_signal = preamble

noisy_signal = channel(modulated_signal, snr_db, multipath_distance)

# Demodulate the carrier wave
t_noisy_signal = np.linspace(0, len(noisy_signal)/fs, len(noisy_signal))
demodulated_signal = noisy_signal


# correlate to the preamble
correlation_magnitude, peak_index = correlate_preamble(demodulated_signal, preamble)
expected_start_index = len(t) + len(preamble) - 1  # Expected start index for zero delay
delay_error = peak_index - expected_start_index
delay_error_time = delay_error/fs

print("length of t: ",len(t))
print("length of the noisy signal: ", len(noisy_signal))
print("length of the preamble: ", len(preamble))

# Calculate the spectrogram
frequencies, times, Sxx = spectrogram(demodulated_signal.real, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Define slider axes and sliders directly in ax[0, 0] position
ax_bw = fig.add_axes([0.1, 0.9, 0.3, 0.03], facecolor='lightgoldenrodyellow', transform=axs[0, 0].transAxes)
ax_snr = fig.add_axes([0.1, 0.85, 0.3, 0.03], facecolor='lightgoldenrodyellow', transform=axs[0, 0].transAxes)
ax_preamble = fig.add_axes([0.1, 0.8, 0.3, 0.03], facecolor='lightgoldenrodyellow', transform=axs[0, 0].transAxes)
ax_sf = fig.add_axes([0.1, 0.75, 0.3, 0.03], facecolor='lightgoldenrodyellow', transform=axs[0, 0].transAxes)
button_ax = plt.axes([0.35, 0.65, 0.05, 0.04])  # Position the button under the sliders


bw_slider = Slider(ax_bw, 'Bandwidth (kHz)', 125, 500, valinit=allowed_bw_values[0], valstep=allowed_bw_values)
snr_slider = Slider(ax_snr, 'SNR (dB)', -40, 40, valinit=20)
preamble_slider = Slider(ax_preamble, 'Preambles', 1, 8, valinit=2, valstep=1)
sf_slider = Slider(ax_sf, 'SF', 7, 12, valinit=7, valstep=1)
update_button = Button(button_ax, 'Update')

# Add Checkbox for Multipath
checkbox_ax = plt.axes([0.1, 0.65, 0.05, 0.04])  # Position the checkbox above the update button
checkbox = CheckButtons(checkbox_ax, ['Multipath'], [False])
checkbox.on_clicked(update_multipath)

# Add Slider for Distance when multipath is enabled
distance_slider_ax = fig.add_axes([0.1, 0.7, 0.3, 0.03], facecolor='lightgoldenrodyellow', transform=axs[0, 0].transAxes)
distance_slider = Slider(distance_slider_ax, 'Multipath Distance (m)', 1, 3e8 / BW, valinit=1)
distance_slider_ax.set_visible(False)  # Initially hidden

bw_slider.on_changed(update_plot)
snr_slider.on_changed(update_plot)
preamble_slider.on_changed(update_plot)
sf_slider.on_changed(update_plot)
distance_slider.on_changed(update_plot)
update_button.on_clicked(update_plot)

# Leave the first row first column empty
axs[0, 0].axis('off')

# Mark the start and end of the signal
start_of_signal = (len(t) - 1) * 1e3 / fs
end_of_signal = (len(t) + len(preamble) - 1) * 1e3 / fs

# Plot the noisy signal in the same second quadrant
axs[0, 1].plot(t_noisy_signal * 1e3, demodulated_signal)
axs[0, 1].axvline(start_of_signal, color='g', linestyle='-', label='Start of preamble')
axs[0, 1].axvline(end_of_signal, color='r', linestyle='-', label='End of preamble')
axs[0, 1].legend()
axs[0, 1].set_title('Sinusoidal Representation of Received LoRa Preamble')
axs[0, 1].set_xlabel('Time (ms)')
axs[0, 1].set_ylabel('Amplitude')
axs[0, 1].grid(True)


# Second row, first column plot (Spectrogram)
pcm = axs[1, 0].pcolormesh(times * 1e3, frequencies / 1e3, 10 * np.log10(Sxx), shading='gouraud')
axs[1, 0].set_ylabel('Frequency (kHz)')
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 0].set_title('Spectrogram')
fig.colorbar(pcm, ax=axs[1, 0], label='Intensity (dB)')
axs[1, 0].set_ylim(0, 2 * BW / 1e3)  # Limit to LoRa bandwidth

# Last quadrant plot (Correlation)
t_correlation = np.linspace(0, len(correlation_magnitude) / fs, len(correlation_magnitude))
distance_correlation = t_correlation * 3e8

d_peak_index = peak_index * 3e8 / fs
axs[1, 1].plot(distance_correlation, correlation_magnitude)
axs[1, 1].axvline(d_peak_index, color='r', linestyle='--', label=f'Peak error: {delay_error_time*3e8:.2f} m')
axs[1, 1].set_title('Correlation between Received Signal and Preamble')
axs[1, 1].set_ylabel('Magnitude')
axs[1, 1].set_xlabel('Distance (m)')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
