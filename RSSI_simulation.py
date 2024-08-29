# File: AoA_simulation.py
# Author: Wout Van Steenbergen
# Project: Risks and Mitigations of Geolocating LoRa Devices
# Date: 2024-08-29
#
# Description:
# This script simulates the average estimation error for different number of receivers for RSSI.
# It also helps visualize the workings and estimation spread
# It is used in Chapter, 4 Section 4.5 of the dissertation.
#
# Dependencies:
# - Python 3.9.13
# - numpy
# - matplotlib
# - scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib
#matplotlib.use('TkAgg')

# Define constants
DEBUG = True
min_distance = 4000
max_distance = 5000
distance_between = 2000
numberOfReceivers = 3
numberOfIterations = 50
playfieldX = 10000
playfieldY = 10000
P_tx = 14  # Transmitted power in dBm
f = 868e6  # Frequency in Hz (868 MHz for LoRa)
c = 3e8  # Speed of light in m/s
SF = 12
BW = 125e3
R_sensitivity = -137
distance_gateway = 2000
FSPL_gt = 106.78

RSSI_error = 1.1

min_P_tx = R_sensitivity + FSPL_gt


# Coordinates of receivers (x, y)
receivers = np.array([
    [2000, 2000],
    [2000, 8000],
    [8000, 5000]
])

# True position of the transmitter (for simulation purposes)
transmitter = np.array([5000, 5000])

def debug_print(message):
    if DEBUG:
        print(message)


def plot_rssi_circles(ax, receivers, rssi_values, P_tx, f, c=3e8):
    """
    Function to plot RSSI circles around receivers based on the calculated RSSI values.

    Parameters:
    - ax: The axis to plot on.
    - receivers: Array of receiver coordinates.
    - rssi_values: Array of RSSI values (in dB) for each receiver.
    - P_tx: Transmitted power in dBm.
    - f: Frequency in Hz.
    - c: Speed of light in m/s (default is 3e8).
    """
    # Convert RSSI to distances using FSPL formula
    distances = 10 ** ((P_tx - rssi_values - 20 * np.log10(f) - 20 * np.log10(4 * np.pi / c)) / 20)

    for receiver, distance in zip(receivers, distances):
        circle = plt.Circle((receiver[0], receiver[1]), distance, color='blue', fill=False, linestyle='--', linewidth=1)
        ax.add_patch(circle)


def generate_locations():
    """
    Function to generate random locations for the receivers.

    Parameters:

    Returns:
    - numpy array: Array of receiver locations.
    """
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

def fspl_to_distance(P_rx, P_tx, f, c=3e8):
    """
    Function to convert RSSI values to distances using the Free Space Path Loss (FSPL) formula.

    Parameters:
    - P_rx: Received power in dBm.
    - P_tx: Transmitted power in dBm.
    - f: Frequency in Hz.
    - c: Speed of light in m/s (default is 3e8).

    Returns:
    - numpy array: Array of distances calculated from RSSI values.
    """
    # FSPL formula to convert RSSI to distance
    return 10 ** ((P_tx - P_rx - 20 * np.log10(f) - 20 * np.log10(4 * np.pi / c)) / 20)

def distance_error(x, receivers, rssi, P_tx, f):
    """
    Function to calculate the error between the estimated distances and the calculated distances.

    Parameters:
    - x: Estimated position of the transmitter.
    - receivers: Array of receiver coordinates.
    - rssi: Array of RSSI values (in dB) for each receiver.
    - P_tx: Transmitted power in dBm.
    - f: Frequency in Hz.

    Returns:
    - float: Sum of squared errors between estimated and calculated distances.

    """
    estimated_distances = np.linalg.norm(receivers - x, axis=1)
    calculated_distances = fspl_to_distance(rssi, P_tx, f)
    return np.sum((estimated_distances - calculated_distances)**2)


def simulation():
    """
    Function to simulate the average estimation error for different numbers of receivers.

    Parameters:

    Returns:
    - None
    """
    global numberOfReceivers
    numberOfTimesPerNumberOfReceiver = 10

    # List to store average errors for different numbers of receivers
    num_receivers_list = []
    average_error_list = []

    # List to store errors for specific receiver numbers
    all_errors_3 = []
    all_errors_6 = []
    all_errors_10 = []

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
                fspl = 20 * np.log10(distances) + 20 * np.log10(np.random.uniform(867.1e6, 868.8e6, size=distances.shape)) + 20 * np.log10(
                    4 * np.pi / c)
                rssi = P_tx - fspl + np.random.uniform(-RSSI_error, RSSI_error, size=fspl.shape)

                initial_guess = np.mean(receivers, axis=0)

                # Solve using least squares
                result = minimize(distance_error, initial_guess, args=(receivers, rssi, P_tx, f))
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
    ax.set_title('Distribution of Position Estimation Errors for RSSI')

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


def drawing():
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    estimated_positions = []

    receivers = generate_locations()
    for _ in range(numberOfIterations):
        P_tx_random = np.random.uniform(min_P_tx, min_P_tx + 10)
        # Calculate true distances from transmitter to each receiver
        distances = np.linalg.norm(receivers - transmitter, axis=1)
        debug_print(("True distances: ", distances))

        # Calculate RSSI based on FSPL and add noise

        fspl = 20 * np.log10(distances) + 20 * np.log10(f) + 20 * np.log10(4 * np.pi / c)
        rssi = P_tx - fspl + np.random.uniform(-RSSI_error, RSSI_error)
        debug_print(("RSSI values: ", rssi))

        # Initial guess for the transmitter's position
        initial_guess = np.mean(receivers, axis=0)

        # Solve using least squares
        result = minimize(distance_error, initial_guess, args=(receivers, rssi, P_tx, f))
        # Estimated position of the transmitter
        estimated_positions.append(result.x.astype(float))

    # Convert to NumPy array
    estimated_positions = np.array(estimated_positions)

    print("True position of transmitter: ", transmitter)
    print("Estimated positions of transmitter (rounded): ", estimated_positions)

    plot_rssi_circles(ax[0][0], receivers, rssi, P_tx, f, c=3e8)

    ax[0][0].scatter(receivers[:, 0], receivers[:, 1], color='blue', label='Receivers', s=100)
    ax[0][0].scatter(transmitter[0], transmitter[1], color='red', label='True Transmitter', s=100)
    ax[0][0].scatter(estimated_positions[:, 0], estimated_positions[:, 1], facecolors='none', edgecolors='green',
                     label='Estimated Transmitter', s=100, linewidth=2)

    # Add receiver numbers inside the blue dots
    for index, (x, y) in enumerate(receivers):
        ax[0][0].text(x, y, str(index + 1), fontsize=6, ha='center', va='center', color='white', weight='bold')

    # Adding labels
    ax[0][0].set_xlabel('X coordinate (m)')
    ax[0][0].set_ylabel('Y coordinate (m)')
    ax[0][0].set_title('Receivers and Transmitter Position')
    ax[0][0].legend()
    ax[0][0].grid(True)
    ax[0][0].axis('equal')

    average_position = np.mean(estimated_positions, axis=0)
    average_position_error = np.linalg.norm(average_position - transmitter, axis=0)
    print(f"Average position error: {average_position_error:.2f} m.")

    ax[1][0].scatter(transmitter[0], transmitter[1], color='red', label='True Transmitter', s=100)
    ax[1][0].scatter(estimated_positions[:, 0], estimated_positions[:, 1], facecolors='none', edgecolors='green',
                     label='Estimated Transmitter', s=100, linewidth=2)
    ax[1][0].scatter(average_position[0], average_position[1], facecolors='none', edgecolors='purple',
                     label=f'Mean estimated Transmitter\nerror of {average_position_error:.2f} m', s=100, linewidth=2)

    # Adding labels
    ax[1][0].set_xlabel('X coordinate (m)')
    ax[1][0].set_ylabel('Y coordinate (m)')
    ax[1][0].set_title('Transmitter and estimations Position')
    ax[1][0].legend()
    ax[1][0].grid(True)
    ax[1][0].axis('equal')

    # Calculate the Euclidean distance error for each estimation
    errors = np.linalg.norm(estimated_positions - transmitter, axis=1)
    average_error = np.mean(errors)
    print(f"Average Error: {average_error:.2f} m")

    # Plotting the boxplot of errors
    boxplot_data = ax[0][1].boxplot(errors, vert=True)
    ax[0][1].set_xlabel('Error Distance (m)')
    ax[0][1].set_title('Distribution of Position Estimation Errors')
    # Adding the average error as a horizontal line
    x_positions = boxplot_data['medians'][0].get_xdata()  # Get the x-data range of the median line
    ax[0][1].plot(x_positions, [average_error, average_error], color='r', linestyle='--',
                  label=f'Average Error: {average_error:.2f} m')
    ax[0][1].legend()

    # Plotting the crosses for the user-selected receiver positions
    ax[1][1].set_title('Click to place receivers')
    ax[1][1].set_xlim(0, playfieldX)
    ax[1][1].set_ylim(0, playfieldY)
    ax[1][1].grid(True)
    ax[1][1].set_xlabel('X coordinate (m)')
    ax[1][1].set_ylabel('Y coordinate (m)')
    ax[1][1].scatter(receivers[:, 0], receivers[:, 1], color='blue', marker='x')

    plt.tight_layout()
    plt.draw()
    plt.show()


def drawingV2():
    estimated_positions = []
    average_position_errors = []
    receivers = generate_locations()
    for _ in range(numberOfIterations):
        # Calculate true distances from transmitter to each receiver
        distances = np.linalg.norm(receivers - transmitter, axis=1)
        debug_print(("True distances: ", distances))

        # Calculate RSSI based on FSPL and add noise

        fspl = 20 * np.log10(distances) + 20 * np.log10(np.random.uniform(867.1e6, 868.8e6, size=distances.shape)) + 20 * np.log10(4 * np.pi / c)
        rssi = P_tx - fspl + np.random.uniform(-RSSI_error, RSSI_error, size=distances.shape)
        debug_print(("RSSI values: ", rssi))

        # Initial guess for the transmitter's position
        initial_guess = np.mean(receivers, axis=0)

        # Solve using least squares
        result = minimize(distance_error, initial_guess, args=(receivers, rssi, P_tx, f))
        # Estimated position of the transmitter
        estimated_positions.append(result.x.astype(float))

        average_position = np.mean(estimated_positions, axis=0)
        average_position_errors.append(np.linalg.norm(average_position - transmitter, axis=0))

    # Convert to NumPy array
    estimated_positions = np.array(estimated_positions)

    print("True position of transmitter: ", transmitter)
    print("Estimated positions of transmitter (rounded): ", estimated_positions)

    # Assuming 'average_position_errors' is your data list or array
    iterations = range(1, len(average_position_errors) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations,average_position_errors)
    plt.title('Mean-position Error for RSSI with 3 Receivers')
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean-position error (m)')
    plt.grid(True)
    plt.show()

    # Create a new figure and axis for the independent plot
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_rssi_circles(ax, receivers, rssi, P_tx, f, c=3e8)
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
    ax.set_title(f'Receivers and Transmitter Position for RSSI')
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
    ax.set_title(f'Transmitter and Estimations Position for RSSI with {numberOfReceivers} Receivers')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Display the plot
    plt.show()

if __name__ == '__main__':
    #simulation()
    #drawing()
    drawingV2()
