# File: AoA_simulation.py
# Author: Wout Van Steenbergen
# Project: Risks and Mitigations of Geolocating LoRa Devices
# Date: 2024-08-29
#
# Description:
# This script visualize the effects of the diffrent geolocating techniques: TDoA, RSSI and AoA. using a GUI.
# It is used in Chapter, 4 Section 4.2 of the dissertation.
#
# Dependencies:
# - Python 3.9.13
# - numpy
# - matplotlib
# - scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from matplotlib.widgets import Slider, Button

# Set the matplotlib backend
import matplotlib
matplotlib.use('TkAgg')
#import LoRa_corrolcation

#################### CONSTANTS ##################################
# Define the debug flag
DEBUG = True

numberOfReceivers = 4
numberOfIterations = 5

playfieldX = 10000
playfieldY = 10000


P_tx = 14  # Transmitted power in dBm
f = 868e6  # Frequency in Hz (868 MHz for LoRa)
c = 3e8  # Speed of light in m/s
SF = 12
BW = 125e3
R_sensitivity = -137

RSSI_error = 1.1  # Error in RSSI measurements in dBm

AoA_error = 2  # Error in angle measurement (degrees)

syncErrorMin = 0 #error on sync in nanoseconds (this is the RMS value of MAX-M10S)
syncErrorMax = 60 #error on sync in nanoseconds (this is the 99% value of MAX-M10S)

# Coordinates of receivers (x, y)
receivers = np.array([
    [0, 200],
    [1000, 200],
    [500, 500]
])

# True position of the transmitter (for simulation purposes)
transmitter = np.array([5000, 5000])

# Define a global variable to store the current simulation type
current_simulation = 'TDoA'

drawing_asist = 0

#################### FUNCTIONS ##################################
# Function to print debug information if debug is enabled
def debug_print(message):
    if DEBUG:
        print(message)

def channel(distances):
    """
    Function to calculate the arrival times of the signal at each receiver based on the distances and the speed of light.
    :param  distances: Array of distances from the transmitter to each receiver.
    :return:        Array of arrival times in seconds.
    """
    stations_arrival_times = distances/c
    return stations_arrival_times



def tdoa_error(x, receivers, tdoa, c):
    """
    Function to calculate the sum of squares of the difference between the estimated TDoA values and the actual TDoA values.

    Parameters:
    - x: Estimated position of the transmitter.
    - receivers: Array of receiver coordinates.
    - tdoa: Array of actual TDoA values.
    - c: Speed of light in m/s.

    Returns:
    - Sum of squares of the difference between the estimated TDoA values and the actual TDoA values.
    """

    # Similar to tdoa_residuals but returns the sum of squares
    estimated_distances = np.linalg.norm(receivers - x, axis=1)
    estimated_arrival_times = (estimated_distances / c) * 1e9

    estimated_tdoa = []
    for i in range(len(estimated_arrival_times)):
        for j in range(i + 1, len(estimated_arrival_times)):
            delta_t = estimated_arrival_times[i] - estimated_arrival_times[j]
            estimated_tdoa.append(delta_t)

    return np.sum((np.array(estimated_tdoa) - np.array(tdoa))**2)


def fspl_to_distance(P_rx, P_tx, f, c=3e8):
    """
    function to convert RSSI to distance using the Free Space Path Loss (FSPL) formula.

    Parameters:
    - P_rx: Received power in dBm.
    - P_tx: Transmitted power in dBm.
    - f: Frequency in Hz.
    - c: Speed of light in m/s (default is 3e8).

    Returns:
    - Distance in meters.
    """

    # FSPL formula to convert RSSI to distance
    return 10 ** ((P_tx - P_rx - 20 * np.log10(f) - 20 * np.log10(4 * np.pi / c)) / 20)


def distance_error(x, receivers, rssi, P_tx, f):
    """
    Function to calculate the sum of squares of the difference between the estimated distances and the calculated distances based on RSSI.

    Parameters:
    - x: Estimated position of the transmitter.
    - receivers: Array of receiver coordinates.
    - rssi: Array of RSSI values (in dBm) for each receiver.
    - P_tx: Transmitted power in dBm.
    - f: Frequency in Hz.

    Returns:
    - Sum of squares of the difference between the estimated distances and the calculated distances.
    """

    estimated_distances = np.linalg.norm(receivers - x, axis=1)
    calculated_distances = fspl_to_distance(rssi, P_tx, f)
    return np.sum((estimated_distances - calculated_distances)**2)


def angle_error(x, receivers, aoa_values):
    """
    Function to calculate the sum of squares of the difference between the estimated angles and the actual angles.

    Parameters:
    - x: Estimated position of the transmitter.
    - receivers: Array of receiver coordinates.
    - aoa_values: Array of actual AoA values (in radians) for each receiver.

    Returns:
    - Sum of squares of the difference between the estimated angles and the actual angles.
    """

    estimated_angles = np.arctan2(receivers[:, 1] - x[1], receivers[:, 0] - x[0])
    angle_differences = np.arctan2(np.sin(estimated_angles - aoa_values), np.cos(estimated_angles - aoa_values))
    error = np.sum(angle_differences ** 2)
    return error


def TDoa():
    """
    Function to estimate the position of the transmitter using the Time Difference of Arrival (TDoA) technique.

    Parameters:

    Returns:
    - result: The result of the optimization containing the estimated position of the transmitter
    """

    global drawing_asist
    # Calculate true distances from transmitter to each receiver
    distances = np.linalg.norm(receivers - transmitter, axis=1)
    debug_print(("distances: ", distances))

    # Calculate arrival times in ns of signal at receivers and add snyc error
    arrival_times = channel(distances) * 1e9
    arrival_times = arrival_times + np.random.randint(syncErrorMin, syncErrorMax, size=arrival_times.shape)

    # for at in arrival_times:
    #     at += LoRa_corrolcation.get_random_correlation_time_delay()

    debug_print(("diffrence time arrival: ", arrival_times))

    # Calculate TDOA (Loop through each pair of receivers)
    tdoa = [arrival_times[i] - arrival_times[j] for i in range(len(arrival_times)) for j in
            range(i + 1, len(arrival_times))]
    debug_print(("tdoa: ", tdoa))

    # Initial guess for the transmitter's position
    initial_guess = np.array([0, 0])

    # Solve using least squares
    # result = least_squares(tdoa_residuals, initial_guess, args=(receivers, tdoa, c), verbose=2)
    result = minimize(tdoa_error, initial_guess, args=(receivers, tdoa, c))
    # Estimated position of the transmitter
    drawing_asist = tdoa
    return result


def RSSI():
    """
    Function to estimate the position of the transmitter using the Received Signal Strength Indicator (RSSI) technique.

    Parameters:

    Returns:
    - result: The result of the optimization containing the estimated position of the transmitter
    """

    global drawing_asist
    # Calculate true distances from transmitter to each receiver
    distances = np.linalg.norm(receivers - transmitter, axis=1)
    debug_print(("True distances: ", distances))

    # Calculate RSSI based on FSPL and add noise

    fspl = 20 * np.log10(distances) + 20 * np.log10(
        np.random.uniform(867.1e6, 868.8e6, size=distances.shape)) + 20 * np.log10(4 * np.pi / c)
    rssi = P_tx - fspl + np.random.uniform(-RSSI_error, RSSI_error, size=distances.shape)
    debug_print(("RSSI values: ", rssi))

    # Initial guess for the transmitter's position
    initial_guess = np.mean(receivers, axis=0)

    # Solve using least squares
    result = minimize(distance_error, initial_guess, args=(receivers, rssi, P_tx, f))
    drawing_asist = rssi
    return result


def AoA():
    """
    Function to estimate the position of the transmitter using the Angle of Arrival (AoA) technique.

    Parameters:

    Returns:
    - result: The result of the optimization containing the estimated position of the transmitter
    """

    global drawing_asist
    true_angles = np.arctan2(receivers[:, 1] - transmitter[1], receivers[:, 0] - transmitter[0])
    aoa_values = true_angles + np.deg2rad(np.random.uniform(-AoA_error, AoA_error, size=true_angles.shape))
    initial_guess = np.mean(receivers, axis=0)
    result = minimize(angle_error, initial_guess, args=(receivers, aoa_values), options={'ftol': 1e-12},
                      method='Nelder-Mead')

    drawing_asist = aoa_values
    return result


def initPlt(fig, ax):
    """
    Function to initialize the plot with the receivers and transmitter.

    Parameters:
    - fig: The figure object.
    - ax: The axis object.
    Returns:

    """
    global receivers

    ax[1][1].set_title('Click to place receivers')
    ax[1][1].set_xlim(0, playfieldX)
    ax[1][1].set_ylim(0, playfieldY)
    ax[1][1].grid(True)
    ax[1][1].set_xlabel('X coordinate (m)')
    ax[1][1].set_ylabel('Y coordinate (m)')
    # Prompt user to place receivers by clicking
    print("Please click on the plot to place 3 receivers.")
    points = plt.ginput(numberOfReceivers, timeout=-1)  # Let user click 3 times
    print(points)

    # Convert the points to numpy array
    receivers = np.array(points)


# Define callback functions for each button
def reset_button_colors():
    tdoa_button.color = 'lightgrey'
    rssi_button.color = 'lightgrey'
    aoa_button.color = 'lightgrey'
    tdoa_button.hovercolor = 'lightgrey'
    rssi_button.hovercolor = 'lightgrey'
    aoa_button.hovercolor = 'lightgrey'

# Define callback functions for each button
def set_tdoa(event):
    global current_simulation
    current_simulation = 'TDoA'
    reset_button_colors()
    tdoa_button.color = 'darkgrey'
    tdoa_button.hovercolor = 'darkgrey'
    update_plot(fig, ax)

def set_rssi(event):
    global current_simulation
    current_simulation = 'RSSI'
    reset_button_colors()
    rssi_button.color = 'darkgrey'
    rssi_button.hovercolor = 'darkgrey'
    update_plot(fig, ax)

def set_aoa(event):
    global current_simulation
    current_simulation = 'AoA'
    reset_button_colors()
    aoa_button.color = 'darkgrey'
    aoa_button.hovercolor = 'darkgrey'
    update_plot(fig, ax)

def clearAxes(fig, ax):
    ax[0][0].clear()
    ax[1][0].clear()
    ax[1][1].clear()
    ax[0][1].clear()

def plot_hyperbolas(ax, receivers, tdoa, c):
    """
    Function to plot hyperbolas based on the TDoA values.

    Parameters:
    - ax: The axis to plot on.
    - receivers: Array of receiver coordinates.
    Returns:

    """
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


def slider_update(val):
    global numberOfIterations
    numberOfIterations = int(slider.val)  # Get current slider value
    update_plot(fig, ax)

def update_plot(fig, ax):
    """
    Function to update the plot based on the current simulation type and the number of iterations.


    Parameters:
    - fig: The figure object.
    - ax: The axis object.
    Returns:

    """

    estimated_positions = []
    clearAxes(fig, ax)

    for x in range(numberOfIterations):

        if current_simulation == 'TDoA':
            result = TDoa()
        elif current_simulation == 'RSSI':
            result = RSSI()
        elif current_simulation == 'AoA':
            result = AoA()
        else:
            raise ValueError(f"Unknown simulation type: {current_simulation}")
        estimated_positions.append(result.x.astype(float))

    # Convert to NumPy array
    estimated_positions = np.array(estimated_positions)

    print("True distance of transmitter: ", transmitter)
    print("Estimated distance of transmitter (rounded): ", estimated_positions)

    clearAxes(fig, ax)

    ax[0][0].scatter(receivers[:, 0], receivers[:, 1], color='blue', label='Receivers', s=100)
    ax[0][0].scatter(transmitter[0], transmitter[1], color='red', label='True Transmitter', s=100)
    ax[0][0].scatter(estimated_positions[:,0], estimated_positions[:,1], facecolors='none', edgecolors='green',
                label='Estimated Transmitter', s=100, linewidth=2)

    # Add receiver numbers inside the blue dots
    for index, (x, y) in enumerate(receivers):
        ax[0][0].text(x, y, str(index + 1), fontsize=6, ha='center', va='center', color='white', weight='bold')

    # Draw hyperbolas based on TDOA
    if current_simulation == 'TDoA':
        plot_hyperbolas(ax[0][0], receivers, drawing_asist, c)
    elif current_simulation == 'RSSI':
        plot_rssi_circles(ax[0][0], receivers, drawing_asist, P_tx, f, c=3e8)
    elif current_simulation == 'AoA':
        plot_aoa_lines(ax[0][0], receivers, drawing_asist)

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

    #plt.tight_layout()
    plt.draw()


def onclick(event):
    """
    Function to handle mouse clicks on the plot. Left click to add a receiver, right click to remove the nearest receiver.

    Parameters:
    - event: The mouse click event.

    Returns:
    """
    if event.inaxes == ax[1][1]:
        ix, iy = event.xdata, event.ydata
        global receivers
        if event.button == 1:  # Left click to add a receiver
            receivers = np.vstack([receivers, [ix, iy]])
        elif event.button == 3:  # Right click to remove the nearest receiver
            distances = np.sqrt((receivers[:, 0] - ix) ** 2 + (receivers[:, 1] - iy) ** 2)
            if len(distances) > 0:
                nearest_index = np.argmin(distances)
                if distances[nearest_index] < 100:  # Only remove if close enough (e.g., 10 units)
                    receivers = np.delete(receivers, nearest_index, axis=0)

        update_plot(fig, ax)


if __name__ == '__main__':
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.95, wspace=0.4, hspace=0.4)

    initPlt(fig, ax)
    update_plot(fig, ax)

    # Create the slider beneath the plots
    slider_ax = plt.axes([0.35, 0.02, 0.45, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(slider_ax, 'Estimations', 1, 50, valinit=numberOfIterations, valstep=1)
    slider.on_changed(slider_update)

    # Connect the click event to the onclick function
    update_plot(fig, ax)

    #copilot: add update button
    update_button_ax = plt.axes([0.85, 0.02, 0.05, 0.03])
    update_button = Button(update_button_ax, 'Update')
    update_button.on_clicked(slider_update)

    # Add buttons for TDoA, RSSI, and AoA simulations
    tdoa_button_ax = plt.axes([0.1, 0.02, 0.05, 0.03])
    tdoa_button = Button(tdoa_button_ax, 'TDoA', color='darkgrey', hovercolor='darkgrey')
    tdoa_button.on_clicked(set_tdoa)

    rssi_button_ax = plt.axes([0.16, 0.02, 0.05, 0.03])
    rssi_button = Button(rssi_button_ax, 'RSSI', color='lightgrey', hovercolor='lightgrey')
    rssi_button.on_clicked(set_rssi)

    aoa_button_ax = plt.axes([0.22, 0.02, 0.05, 0.03])
    aoa_button = Button(aoa_button_ax, 'AoA', color='lightgrey', hovercolor='lightgrey')
    aoa_button.on_clicked(set_aoa)

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()