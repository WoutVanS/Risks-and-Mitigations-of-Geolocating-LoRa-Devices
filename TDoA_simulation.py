# File: AoA_simulation.py
# Author: Wout Van Steenbergen
# Project: Risks and Mitigations of Geolocating LoRa Devices
# Date: 2024-08-29
#
# Description:
# This script simulates the average estimation error for different number of receivers for TDoA.
# It also helps visualize the workings and estimation spread
# It is used in Chapter, 4 Section 4.3 of the dissertation.
#
# Dependencies:
# - Python 3.9.13
# - numpy
# - matplotlib
# - scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import correlate
import matplotlib
matplotlib.use('TkAgg')

# Define constants
DEBUG = True
min_distance = 4000
max_distance = 5000
distance_between = 2000
numberOfReceivers = 10
numberOfIterations = 100
playfieldX = 10000
playfieldY = 10000

f = 868e6  # Frequency in Hz (868 MHz for LoRa)
c = 3e8  # Speed of light in m/s
BW = 125e3
SNR = 5

fs = 2e6  # Sampling frequency in Hz
SF = 7  # LoRa Spreading Factor
N = 1  # number of chirps in preamble

multipath = True
distance_gateway = 2000

syncErrorMin = 0 #error on sync in nanoseconds (this is the RMS value of MAX-M10S)
syncErrorMax = 100 #error on sync in nanoseconds (this is the 99% value of MAX-M10S)


# Coordinates of receivers (x, y)
receivers = np.array([
    [2000, 2000],
    [2000, 8000],
    [8000, 5000]
])

# True position of the transmitter (for simulation purposes)
transmitter = np.array([5000, 5000])

T_sym = (2 ** SF) / BW  # Symbol time for LoRa
t = np.linspace(0, T_sym, int(fs * T_sym))
t_full = np.linspace(0, T_sym * N, int(fs * T_sym * N))


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


def channel(signal, snr_db, distance, d1):
    global multipath
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

    if multipath:
        noisy_signal[int(len(t)) + int(delay_samples) - 1: int(len(t)) + int(delay_samples) - 1 + len(delayed_signal)] += delayed_signal

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


def plot_hyperbolas(ax, receivers, tdoa, c):
    counter = 0
    for i in range(len(receivers)):
        for j in range(i + 1, len(receivers)):
            xi, yi = receivers[i]
            xj, yj = receivers[j]
            delta_d = tdoa[counter] * 1e-9 * c

            x_vals = np.linspace(0, playfieldX, 400)
            y_vals = np.linspace(0, playfieldY, 400)
            X, Y = np.meshgrid(x_vals, y_vals)
            dist_i = np.sqrt((X - xi)**2 + (Y - yi)**2)
            dist_j = np.sqrt((X - xj)**2 + (Y - yj)**2)
            Z = dist_i - dist_j - delta_d

            # Plot the hyperbola
            contour_set  = ax.contour(X, Y, Z, levels=[0], colors='orange', linewidths=0.5)
            paths = contour_set.allsegs[0]  # allsegs[0] contains the paths for the level=0 contour

            if paths:
                vertices = paths[0]  # Get the first segment of the contour
                label_x = vertices[10, 0]
                label_y = vertices[10, 1]
                label = f"{i + 1}{j + 1}"
                ax.text(label_x, label_y, label, fontsize=8, color='blue', weight='bold', backgroundcolor='white')

            counter = counter + 1


def generate_locations():
    locations = []

    while len(locations) < numberOfReceivers:
        x = np.random.uniform(0, playfieldX)
        y = np.random.uniform(0, playfieldY)
        point = np.array([x, y])

        # Calculate the distance from the transmitter
        distance_from_transmitter = np.linalg.norm(point - transmitter)

        # Check if the point is within the acceptable distance range from the transmitter
        if min_distance <= distance_from_transmitter <= max_distance:
            # Check the distance from all other existing locations
            if all(np.linalg.norm(point - np.array(loc)) >= distance_between for loc in locations):
                locations.append((x, y))

    return np.array(locations)


def generate_correlation_error(preamble, snr_db, distance):
    modulated_signal = preamble
    noisy_signal = channel(modulated_signal, snr_db, np.random.uniform(1, (3e8 / BW)), distance)
    demodulated_signal = noisy_signal

    correlation_magnitude, peak_index = correlate_preamble(demodulated_signal, preamble)
    expected_start_index = len(t) + len(preamble) - 1  # Expected start index for zero delay
    delay_error = peak_index - expected_start_index
    delay_error_time = delay_error / fs
    return delay_error_time


def tdoa_error(x, receivers, tdoa, c):
    # Similar to tdoa_residuals but returns the sum of squares
    estimated_distances = np.linalg.norm(receivers - x, axis=1)
    estimated_arrival_times = (estimated_distances / c) * 1e9

    estimated_tdoa = []
    for i in range(len(estimated_arrival_times)):
        for j in range(i + 1, len(estimated_arrival_times)):
            delta_t = estimated_arrival_times[i] - estimated_arrival_times[j]
            estimated_tdoa.append(delta_t)

    return np.sum((np.array(estimated_tdoa) - np.array(tdoa))**2)


def simulation():
    global numberOfReceivers
    numberOfTimesPerNumberOfReceiver = 10

    # List to store average errors for different numbers of receivers
    num_receivers_list = []
    average_error_list = []

    # List to store errors for specific receiver numbers
    all_errors_3 = []
    all_errors_6 = []
    all_errors_10 = []

    preamble = generate_complex_chirp(t, f0=0, f1=BW, T_sym=T_sym)

    for num_receivers in range(3, 11):

        average_error = []
        numberOfReceivers = num_receivers
        all_errors = []
        for i in range(numberOfTimesPerNumberOfReceiver):
            print(f"{num_receivers}: {i * 10}%")
            estimated_positions = []
            receivers = generate_locations()  # Assuming this function generates `num_receivers` receiver locations

            for _ in range(numberOfIterations):
                # Calculate true distances from transmitter to each receiver
                distances = np.linalg.norm(receivers - transmitter, axis=1)
                arrival_times = (distances/c) * 1e9

                sync_errors = np.random.randint(syncErrorMin, syncErrorMax, size=arrival_times.shape)
                correlation_errors = np.array([generate_correlation_error(preamble, SNR, distance) for distance in distances])
                arrival_times = arrival_times + sync_errors + correlation_errors

                tdoa = [arrival_times[i] - arrival_times[j] for i in range(len(arrival_times)) for j in
                        range(i + 1, len(arrival_times))]

                initial_guess = np.mean(receivers, axis=0)

                # Solve using least squares
                result = minimize(tdoa_error, initial_guess, args=(receivers, tdoa, c))
                # Estimated position of the transmitter
                estimated_positions.append(result.x.astype(float))

            estimated_positions = np.array(estimated_positions)
            errors = np.linalg.norm(estimated_positions - transmitter, axis=1)
            all_errors.extend(errors)
            average_error.append(np.mean(errors))

        # Store errors for specific receiver counts
        if num_receivers == 3:
            all_errors_3 = all_errors
        elif num_receivers == 6:
            all_errors_6 = all_errors
        elif num_receivers == 10:
            all_errors_10 = all_errors

        # Average error for this number of receivers
        average_error = np.array(average_error)

        # Store the number of receivers and the corresponding average error
        num_receivers_list.append(num_receivers)
        average_error_list.append(np.mean(average_error))

    # Plotting the boxplots for 3, 6, and 10 receivers in one figure
    fig, ax = plt.subplots(figsize=(10, 8))
    boxplot_data = ax.boxplot([all_errors_3, all_errors_6, all_errors_10], vert=True, patch_artist=True,
                              labels=['3 Receivers', '6 Receivers', '10 Receivers'])
    ax.set_xlabel('Number of Receivers')
    ax.set_ylabel('Error Distance (m)')
    ax.set_title('Distribution of Position Estimation Errors for TDoA')

    plt.show()

    # Plot the results of average error vs. number of receivers
    plt.figure(figsize=(10, 6))
    plt.plot(num_receivers_list, average_error_list, marker='o')
    for i, txt in enumerate(average_error_list):
        plt.annotate(f'{txt:.2f}', (num_receivers_list[i], average_error_list[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel('Number of receivers')
    plt.ylabel('Average estimation error (m)')
    plt.grid(True)
    plt.show()


def simulation2():
    global numberOfReceivers, BW
    numberOfTimesPerNumberOfReceiver = 10

    # List to store average errors for different numbers of receivers
    num_receivers_list = []
    average_error_with_corr_list = []
    average_error_without_corr_list = []

    # List to store errors for specific receiver numbers (with correlation error)
    all_errors_3_with_corr = []
    all_errors_6_with_corr = []
    all_errors_10_with_corr = []

    # List to store errors for specific receiver numbers (without correlation error)
    all_errors_3_without_corr = []
    all_errors_6_without_corr = []
    all_errors_10_without_corr = []

    preamble = generate_complex_chirp(t, f0=0, f1=BW, T_sym=T_sym)

    for num_receivers in range(3, 11):

        average_error_with_corr = []
        average_error_without_corr = []
        numberOfReceivers = num_receivers
        all_errors_with_corr = []
        all_errors_without_corr = []

        for i in range(numberOfTimesPerNumberOfReceiver):
            print(f"{num_receivers}: {i * 10}%")
            estimated_positions_with_corr = []
            estimated_positions_without_corr = []
            receivers = generate_locations()  # Assuming this function generates `num_receivers` receiver locations

            for _ in range(numberOfIterations):
                # Calculate true distances from transmitter to each receiver
                distances = np.linalg.norm(receivers - transmitter, axis=1)
                arrival_times = (distances / c) * 1e9

                sync_errors = np.random.randint(syncErrorMin, syncErrorMax, size=arrival_times.shape)

                BW = 125e3
                # With correlation error
                correlation_errors_with = np.array(
                    [generate_correlation_error(preamble, SNR, distance)*1e9 for distance in distances])
                arrival_times_with_corr = arrival_times + sync_errors + correlation_errors_with

                BW = 500e3
                # Without correlation error
                correlation_errors_with = np.array(
                    [generate_correlation_error(preamble, SNR, distance) * 1e9 for distance in distances])
                arrival_times_without_corr = arrival_times + sync_errors + correlation_errors_with

                # TDOA calculation with correlation error
                tdoa_with_corr = [arrival_times_with_corr[i] - arrival_times_with_corr[j] for i in
                                  range(len(arrival_times_with_corr)) for j in
                                  range(i + 1, len(arrival_times_with_corr))]

                # TDOA calculation without correlation error
                tdoa_without_corr = [arrival_times_without_corr[i] - arrival_times_without_corr[j] for i in
                                     range(len(arrival_times_without_corr)) for j in
                                     range(i + 1, len(arrival_times_without_corr))]

                initial_guess = np.mean(receivers, axis=0)

                # Solve using least squares (with correlation error)
                result_with_corr = minimize(tdoa_error, initial_guess, args=(receivers, tdoa_with_corr, c))
                estimated_positions_with_corr.append(result_with_corr.x.astype(float))

                # Solve using least squares (without correlation error)
                result_without_corr = minimize(tdoa_error, initial_guess, args=(receivers, tdoa_without_corr, c))
                estimated_positions_without_corr.append(result_without_corr.x.astype(float))

            estimated_positions_with_corr = np.array(estimated_positions_with_corr)
            estimated_positions_without_corr = np.array(estimated_positions_without_corr)

            errors_with_corr = np.linalg.norm(estimated_positions_with_corr - transmitter, axis=1)
            errors_without_corr = np.linalg.norm(estimated_positions_without_corr - transmitter, axis=1)

            all_errors_with_corr.extend(errors_with_corr)
            all_errors_without_corr.extend(errors_without_corr)

            average_error_with_corr.append(np.mean(errors_with_corr))
            average_error_without_corr.append(np.mean(errors_without_corr))

        # Store errors for specific receiver counts
        if num_receivers == 3:
            all_errors_3_with_corr = all_errors_with_corr
            all_errors_3_without_corr = all_errors_without_corr
        elif num_receivers == 6:
            all_errors_6_with_corr = all_errors_with_corr
            all_errors_6_without_corr = all_errors_without_corr
        elif num_receivers == 10:
            all_errors_10_with_corr = all_errors_with_corr
            all_errors_10_without_corr = all_errors_without_corr

        # Average error for this number of receivers
        average_error_with_corr = np.array(average_error_with_corr)
        average_error_without_corr = np.array(average_error_without_corr)

        # Store the number of receivers and the corresponding average errors
        num_receivers_list.append(num_receivers)
        average_error_with_corr_list.append(np.mean(average_error_with_corr))
        average_error_without_corr_list.append(np.mean(average_error_without_corr))

    import matplotlib.pyplot as plt

    # Plot for data with correlation error
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [all_errors_3_with_corr, all_errors_6_with_corr, all_errors_10_with_corr],
        vert=True, patch_artist=True, positions=[1, 2, 3],
        boxprops=dict(facecolor='orange'),
        labels=['3 Receivers', '6 Receivers', '10 Receivers']
    )
    plt.xlabel('Number of Receivers')
    plt.ylabel('Error Distance (m)')
    plt.title('Distribution of Position Estimation Errors for TDoA with Correlation Error')
    plt.show()

    # Plot for data without correlation error
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [all_errors_3_without_corr, all_errors_6_without_corr, all_errors_10_without_corr],
        vert=True, patch_artist=True, positions=[1, 2, 3],
        labels=['3 Receivers', '6 Receivers', '10 Receivers']
    )
    plt.xlabel('Number of Receivers')
    plt.ylabel('Error Distance (m)')
    plt.title('Distribution of Position Estimation Errors for TDoA')
    plt.show()

    # Plot the results of average error vs. number of receivers with and without correlation error
    plt.figure(figsize=(10, 6))
    plt.plot(num_receivers_list, average_error_with_corr_list, marker='o', label='125kHz')
    plt.plot(num_receivers_list, average_error_without_corr_list, marker='o', label='500kHz')

    for i, txt in enumerate(average_error_with_corr_list):
        plt.annotate(f'{txt:.2f}', (num_receivers_list[i], average_error_with_corr_list[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    for i, txt in enumerate(average_error_without_corr_list):
        plt.annotate(f'{txt:.2f}', (num_receivers_list[i], average_error_without_corr_list[i]),
                     textcoords="offset points", xytext=(0, -15), ha='center')

    plt.xlabel('Number of receivers')
    plt.ylabel('Average estimation error (m)')
    plt.legend()
    plt.grid(True)
    plt.show()


def drawingV2():
    estimated_positions = []
    preamble = generate_complex_chirp(t, f0=0, f1=BW, T_sym=T_sym)
    receivers = generate_locations()
    for _ in range(numberOfIterations):
        # Calculate true distances from transmitter to each receiver
        distances = np.linalg.norm(receivers - transmitter, axis=1)
        print(("True distances: ", distances))

        # Calculate RSSI based on FSPL and add noise
        arrival_times = (distances/c) * 1e9
        sync_errors = np.random.randint(syncErrorMin, syncErrorMax, size=arrival_times.shape)
        correlation_errors = np.array([generate_correlation_error(preamble, SNR, distance) for distance in distances])
        arrival_times = arrival_times + sync_errors + correlation_errors
        tdoa = [arrival_times[i] - arrival_times[j] for i in range(len(arrival_times)) for j in
                range(i + 1, len(arrival_times))]

        # Initial guess for the transmitter's position
        initial_guess = np.mean(receivers, axis=0)

        # Solve using least squares
        result = minimize(tdoa_error, initial_guess, args=(receivers, tdoa, c))
        # Estimated position of the transmitter
        estimated_positions.append(result.x.astype(float))

    # Convert to NumPy array
    estimated_positions = np.array(estimated_positions)

    print("True position of transmitter: ", transmitter)
    print("Estimated positions of transmitter (rounded): ", estimated_positions)

    # Create a new figure and axis for the independent plot
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_hyperbolas(ax, receivers, tdoa, c)
    # Plot the receivers, transmitter, and estimated positions
    ax.scatter(receivers[:, 0], receivers[:, 1], color='blue', label='Receivers', s=100)
    ax.scatter(transmitter[0], transmitter[1], color='red', label='True Transmitter', s=100)
    ax.scatter(estimated_positions[:, 0], estimated_positions[:, 1], facecolors='none', edgecolors='green',
               label='Estimated Transmitter', s=100, linewidth=2)

    # Add receiver numbers inside the blue dots
    for index, (x, y) in enumerate(receivers):
        ax.text(x, y, str(index + 1), fontsize=6, ha='center', va='center', color='white', weight='bold')

    # Adding labels
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.set_title(f'Receivers and Transmitter Position for TDoA')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Display the plot
    plt.show()

    # Correct the estimated positions by subtracting the transmitter's location
    corrected_estimated_positions = estimated_positions - transmitter
    average_position = np.mean(corrected_estimated_positions, axis=0)
    average_position_error = np.linalg.norm(average_position - transmitter, axis=0)
    # Create a new figure and axis for the independent plot
    fig, ax = plt.subplots(figsize=(8, 6)) # Adjust the subplot dimensions as needed

    # Scatter plot for transmitter and estimated positions on ax[1][0]
    ax.scatter(corrected_estimated_positions[:, 0], corrected_estimated_positions[:, 1], facecolors='none',
                  edgecolors='green',
                  label='Estimated Transmitter', s=100, linewidth=2)
    ax.scatter(0, 0, color='red', label='True Transmitter (0, 0)', s=100)  # Transmitter at (0, 0)
    ax.scatter(average_position[0], average_position[1], facecolors='purple', edgecolors='purple',
                     label=f'Mean estimated Transmitter', s=100, linewidth=2)

    # Adding labels
    ax.set_xlabel('X error (m)')
    ax.set_ylabel('Y error (m)')
    ax.set_title(f'Transmitter and Estimations Position for TDoA with {numberOfReceivers} Receivers')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Display the plot
    plt.show()

if __name__ == '__main__':
    #simulation()
    simulation2()
    #drawingV2()
