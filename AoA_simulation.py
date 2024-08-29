# File: AoA_simulation.py
# Author: Wout Van Steenbergen
# Project: Risks and Mitigations of Geolocating LoRa Devices
# Date: 2024-08-29
#
# Description:
# This script simulates the average estimation error for different number of receivers for AoA.
# It also helps visualize the workings and estimation spread
# It is used in Chapter, 4 Section 4.4 of the dissertation.
#
# Dependencies:
# - Python 3.9.13
# - numpy
# - matplotlib
# - scipy


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# import matplotlib
# matplotlib.use('TkAgg')

# Define constants
DEBUG = True
min_distance = 4000
max_distance = 5000
distance_between = 2000
numberOfReceivers = 10
numberOfIterations = 100
playfieldX = 10000
playfieldY = 10000


distance_gateway = 2000

AoA_error = 2  # Error in angle measurement (degrees)

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


def generate_locations():
    """
    Function to generate random locations for the receivers.

    Parameters:

    Returns:

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


def angle_error(x, receivers, aoa_values):
    """
    Function to calculate the error in the estimated angles.

    Parameters:
    - x: The estimated position of the transmitter.
    - receivers: Array of receiver coordinates.
    - aoa_values: Array of AoA values (in radians) for each receiver.

    Returns:
    - error: The sum of squared differences between the estimated and true angles.
    """

    estimated_angles = np.arctan2(receivers[:, 1] - x[1], receivers[:, 0] - x[0])
    angle_differences = np.arctan2(np.sin(estimated_angles - aoa_values), np.cos(estimated_angles - aoa_values))
    error = np.sum(angle_differences ** 2)
    return error


def plot_aoa_lines(ax, receivers, aoa_values):
    """
    Function to plot lines representing the direction the receivers think the signal came from.

    Parameters:
    - ax: The axis to plot on.
    - receivers: Array of receiver coordinates.
    - aoa_values: Array of AoA values (in radians) for each receiver.
    """
    for receiver, aoa in zip(receivers, aoa_values):
        # Calculate a point far away in the direction of the AoA
        x_far = receiver[0] - 10000 * np.cos(aoa)
        y_far = receiver[1] - 10000 * np.sin(aoa)
        ax.plot([receiver[0], x_far], [receiver[1], y_far], linestyle='-', color='orange', linewidth=1)

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

    for num_receivers in range(2, 11):

        average_error = []
        numberOfReceivers = num_receivers
        all_errors = []
        for i in range(numberOfTimesPerNumberOfReceiver):
            print(f"{num_receivers}: {i * 10}%")
            estimated_positions = []
            receivers = generate_locations()  # Assuming this function generates `num_receivers` receiver locations

            for _ in range(numberOfIterations):

                true_angles = np.arctan2(receivers[:, 1] - transmitter[1], receivers[:, 0] - transmitter[0])
                aoa_values = true_angles + np.deg2rad(np.random.uniform(-AoA_error, AoA_error, size=true_angles.shape))
                initial_guess = np.mean(receivers, axis=0)
                result = minimize(angle_error, initial_guess, args=(receivers, aoa_values), options={'ftol': 1e-12},
                                  method='Nelder-Mead')
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
    ax.set_title('Distribution of Position Estimation Errors for AoA')

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


# Assuming the following global variables and functions are defined somewhere in your code:
# transmitter, numberOfIterations, AoA_error, generate_locations, angle_error

def simulation(AoA_error):
    global numberOfReceivers
    numberOfTimesPerNumberOfReceiver = 10

    # List to store average errors for different numbers of receivers
    num_receivers_list = []
    average_error_list = []

    # List to store errors for specific receiver numbers
    all_errors_3 = []
    all_errors_6 = []
    all_errors_10 = []

    for num_receivers in range(2, 11):

        average_error = []
        numberOfReceivers = num_receivers
        all_errors = []
        for i in range(numberOfTimesPerNumberOfReceiver):
            print(f"{num_receivers}: {i * 10}%")
            estimated_positions = []
            receivers = generate_locations()  # Assuming this function generates `num_receivers` receiver locations

            for _ in range(numberOfIterations):

                true_angles = np.arctan2(receivers[:, 1] - transmitter[1], receivers[:, 0] - transmitter[0])
                aoa_values = true_angles + np.deg2rad(np.random.uniform(-AoA_error, AoA_error, size=true_angles.shape))
                initial_guess = np.mean(receivers, axis=0)
                result = minimize(angle_error, initial_guess, args=(receivers, aoa_values), options={'ftol': 1e-12},
                                  method='Nelder-Mead')
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

    # Plotting the boxplots for 3, 6, and 10 receivers separately
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.boxplot([all_errors_3, all_errors_6, all_errors_10], vert=True, patch_artist=True,
               labels=['3 Receivers', '6 Receivers', '10 Receivers'])
    ax.set_xlabel('Number of Receivers')
    ax.set_ylabel('Error Distance (m)')
    ax.set_title(f'Distribution of Position Estimation Errors for AoA (AoA_error = {AoA_error})')
    plt.show()

    # Plot the results of average error vs. number of receivers separately
    plt.figure(figsize=(10, 6))
    plt.plot(num_receivers_list, average_error_list, marker='o')
    for i, txt in enumerate(average_error_list):
        plt.annotate(f'{txt:.2f}', (num_receivers_list[i], average_error_list[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel('Number of receivers')
    plt.ylabel('Average estimation error (m)')
    plt.title(f'Average Estimation Error vs. Number of Receivers (AoA_error = {AoA_error})')
    plt.grid(True)
    plt.show()

def simulation2():
    global numberOfReceivers
    numberOfTimesPerNumberOfReceiver = 10
    AoA_errors = [2, 10]  # AoA_error values to iterate over

    # Dictionary to store average errors for different AoA_error values
    results = {}

    for AoA_error in AoA_errors:
        # Lists to store average errors for different numbers of receivers
        num_receivers_list = []
        average_error_list = []

        # Lists to store errors for specific receiver numbers
        all_errors_3 = []
        all_errors_6 = []
        all_errors_10 = []

        for num_receivers in range(2, 11):
            average_error = []
            numberOfReceivers = num_receivers
            all_errors = []

            for i in range(numberOfTimesPerNumberOfReceiver):
                print(f"AoA_error = {AoA_error}, {num_receivers}: {i * 10}%")
                estimated_positions = []
                receivers = generate_locations()  # Assuming this function generates `num_receivers` receiver locations

                for _ in range(numberOfIterations):
                    true_angles = np.arctan2(receivers[:, 1] - transmitter[1], receivers[:, 0] - transmitter[0])
                    aoa_values = true_angles + np.deg2rad(np.random.uniform(-AoA_error, AoA_error, size=true_angles.shape))
                    initial_guess = np.mean(receivers, axis=0)
                    result = minimize(angle_error, initial_guess, args=(receivers, aoa_values), options={'ftol': 1e-12},
                                      method='Nelder-Mead')
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

        # Store the results in the dictionary
        results[AoA_error] = (num_receivers_list, average_error_list, all_errors_3, all_errors_6, all_errors_10)

    # Plotting the results for both AoA_error values on the same plot
    plt.figure(figsize=(10, 6))
    for AoA_error in AoA_errors:
        if(AoA_error == 2): labelname = "LoS"
        else: labelname = "NLoS"
        num_receivers_list, average_error_list, _, _, _ = results[AoA_error]
        plt.plot(num_receivers_list, average_error_list, marker='o', label=f'{labelname}')
        for i, txt in enumerate(average_error_list):
            plt.annotate(f'{txt:.2f}', (num_receivers_list[i], average_error_list[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel('Number of receivers')
    plt.ylabel('Average estimation error (m)')
    plt.title('Average Estimation Error vs. Number of Receivers')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plotting the boxplots for 3, 6, and 10 receivers separately for each AoA_error
    for AoA_error in AoA_errors:
        if (AoA_error == 2):
            labelname = "LoS"
        else:
            labelname = "NLoS"
        _, _, all_errors_3, all_errors_6, all_errors_10 = results[AoA_error]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.boxplot([all_errors_3, all_errors_6, all_errors_10], vert=True, patch_artist=True,
                   labels=['3 Receivers', '6 Receivers', '10 Receivers'])
        ax.set_xlabel('Number of Receivers')
        ax.set_ylabel('Error Distance (m)')
        ax.set_title(f'Distribution of Position Estimation Errors for AoA ({labelname})')
        plt.show()

# Ensure to define or import generate_locations(), angle_error(), transmitter, and numberOfIterations

def drawingAoA():
    estimated_positions = []

    receivers = generate_locations()
    for _ in range(numberOfIterations):
        # Calculate true angles (AoA) from transmitter to each receiver
        true_angles = np.arctan2(receivers[:, 1] - transmitter[1], receivers[:, 0] - transmitter[0])
        debug_print(("True angles: ", true_angles))

        # Add noise to the angles (simulating AoA measurement errors)
        aoa_values = true_angles + np.deg2rad(np.random.uniform(-AoA_error, AoA_error, size=true_angles.shape))
        debug_print(("AoA values: ", aoa_values))

        # Initial guess for the transmitter's position
        initial_guess = np.mean(receivers, axis=0)

        # Solve using least squares
        result = minimize(angle_error, initial_guess, args=(receivers, aoa_values), options={'ftol': 1e-12},  method='Nelder-Mead')
        # Estimated position of the transmitter
        estimated_positions.append(result.x.astype(float))

    # Convert to NumPy array
    estimated_positions = np.array(estimated_positions)

    #print("True position of transmitter: ", transmitter)
    #print("Estimated positions of transmitter (rounded): ", estimated_positions)

    # Create a new figure and axis for the independent plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the receivers, transmitter, and estimated positions
    ax.scatter(receivers[:, 0], receivers[:, 1], color='blue', label='Receivers', s=100)
    ax.scatter(transmitter[0], transmitter[1], color='red', label='True Transmitter', s=100)
    ax.scatter(estimated_positions[:, 0], estimated_positions[:, 1], facecolors='none', edgecolors='green',
               label='Estimated Transmitter', s=100, linewidth=2)

    # Add receiver numbers inside the blue dots
    for index, (x, y) in enumerate(receivers):
        ax.text(x, y, str(index + 1), fontsize=6, ha='center', va='center', color='white', weight='bold')

     # Plot the AoA lines
    plot_aoa_lines(ax, receivers, aoa_values)

    # Adding labels
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.set_title(f'Receivers and Transmitter Position for AoA')
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
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot for transmitter and estimated positions
    ax.scatter(corrected_estimated_positions[:, 0], corrected_estimated_positions[:, 1], facecolors='none',
               edgecolors='green', label='Estimated Transmitter', s=100, linewidth=2)
    ax.scatter(0, 0, color='red', label='True Transmitter (0, 0)', s=100)  # Transmitter at (0, 0)
    ax.scatter(average_position[0], average_position[1], facecolors='purple', edgecolors='purple',
               label=f'Mean estimated Transmitter', s=100, linewidth=2)

    # Adding labels
    ax.set_xlabel('X error (m)')
    ax.set_ylabel('Y error (m)')
    ax.set_title(f'Transmitter and Estimations Position for AoA with {numberOfReceivers} Receivers')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Display the plot
    plt.show()


if __name__ == '__main__':
    #simulation()
    #simulation2()
    drawingAoA()
