# File: AoA_simulation.py
# Author: Wout Van Steenbergen
# Project: Risks and Mitigations of Geolocating LoRa Devices
# Date: 2024-08-29
#
# Description:
# This script simulates the average timing error for different bandwidths and SNR value when cross correlating.
# It is used in Chapter 4, Section 4.6 of the dissertation.
#
# Dependencies:
# - Python 3.9.13
# - numpy
# - matplotlib
# - scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram, correlate
import matplotlib
matplotlib.use('TkAgg')

# Function to add AWGN noise to a signal
def add_awgn_noise(signal, snr_db):
    """
    Add AWGN noise to a signal to achieve the desired SNR in dB. The function assumes that the signal has unit power.

    Parameters:
    signal (numpy array): The original signal
    snr_db (float): Desired signal-to-noise ratio in dB

    Returns:
    numpy array: Signal with added noise
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
    Cross-correlate the received signal with the preamble to find the start of the preamble.

    Parameters:
    received_signal (numpy array): The received signal
    preamble (numpy array): The known preamble
    Returns:
    numpy array: The magnitude of the cross-correlation

    """
    correlation = correlate(received_signal, preamble, mode='full', method='fft')
    correlation_magnitude = np.abs(correlation)
    peak_index = np.argmax(correlation_magnitude)
    return correlation_magnitude, peak_index

def generate_complex_chirp(t, f0, f1, T_sym):
    """
    Generate a complex chirp signal with a linear frequency sweep from f0 to f1.

    Parameters:
    t (numpy array): Time vector
    f0 (float): Start frequency in Hz
    f1 (float): End frequency in Hz
    T_sym (float): Symbol time in seconds

    Returns:
    numpy array: Complex chirp signal
    """
    chirp_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * T_sym)))
    preamble = np.tile(chirp_signal, N)
    return preamble

# Parameters for LoRa chirps
fs = 2e6  # Sampling frequency in Hz
SF = 7  # LoRa Spreading Factor
N = 1  # number of chirps in preamble
c = 3e8  # Speed of light in m/s

# Define bandwidths and SNR values
bandwidths = [125e3, 250e3, 500e3]  # Bandwidths in Hz
snr_values = np.arange(-30, 21, 1)  # SNR from -30dB to +20dB

multipath = True
d1 = 5000

# Store results for plotting
results = []

# Run the simulation
iterations = 10  # Number of iterations per bandwidth/SNR combination

for BW in bandwidths:
    T_sym = (2 ** SF) / BW  # Symbol time for LoRa
    t = np.linspace(0, T_sym, int(fs * T_sym))
    t_full = np.linspace(0, T_sym * N, int(fs * T_sym * N))
    preamble = generate_complex_chirp(t, f0=0, f1=BW, T_sym=T_sym)

    average_delay_errors = []

    for snr_db in snr_values:
        delay_errors = []
        delay_errors_time = []

        for i in range(iterations):

            modulated_signal = preamble
            noisy_signal = channel(modulated_signal, snr_db, np.random.uniform(1, (3e8/BW)))
            demodulated_signal = noisy_signal

            correlation_magnitude, peak_index = correlate_preamble(demodulated_signal, preamble)
            expected_start_index = len(t) + len(preamble) - 1  # Expected start index for zero delay
            delay_error = peak_index - expected_start_index
            delay_error_time = delay_error / fs

            # Convert delay error time to meters
            delay_errors_time.append(np.abs(delay_error_time))
            delay_error_meters = delay_error_time * c
            delay_errors.append(np.abs(delay_error_meters))

        print(f"BW: {BW}, SNR: {snr_db}, delay error: {np.mean(delay_errors_time)*1e3} ms")
        average_delay_errors.append(np.mean(delay_errors))

    results.append((BW, snr_values, average_delay_errors))

# Plotting the results
plt.figure(figsize=(12, 8))

for result in results:
    BW, snr_values, average_delay_errors = result
    average_delay_errors_km = [error / 1e3 for error in average_delay_errors]
    plt.plot(snr_values, average_delay_errors, label=f'{BW / 1e3} kHz Bandwidth')

plt.xlabel('SNR (dB)')
plt.ylabel('Average Range Error (m)')
plt.title('Average Range Error vs. SNR for Different Bandwidths with Multipath')
plt.legend()
plt.grid(True)
plt.show()
