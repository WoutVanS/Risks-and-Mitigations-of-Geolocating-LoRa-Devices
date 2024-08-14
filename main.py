import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize


# Set the matplotlib backend
import matplotlib
matplotlib.use('TkAgg')

#################### CONSTANTS ##################################
# Define the debug flag
DEBUG = True

numberOfReceivers = 4
numberOfIterations = 100

c = 3e8 #speed of light

syncErrorMin = 0 #error on sync in nanoseconds (this is the RMS value of MAX-M10S)
syncErrorMax = 100 #error on sync in nanoseconds (this is the 99% value of MAX-M10S)

# Coordinates of receivers (x, y)
receivers = np.array([
    [0, 200],
    [1000, 200],
    [500, 500]
])

# True position of the transmitter (for simulation purposes)
transmitter = np.array([400, 300])



#################### FUNCTIONS ##################################
# Function to print debug information if debug is enabled
def debug_print(message):
    if DEBUG:
        print(message)

def channel(distances):
    stations_arrival_times = distances/c
    return stations_arrival_times

# def tdoa_residuals(x, receivers, tdoa, c):
#     # Calculate distances to the estimated position
#     estimated_distances = np.linalg.norm(receivers - x, axis=1)
#     # Calculate arrival times
#     estimated_arrival_times = estimated_distances / c
#     # Calculate residuals based on TDOA
#     estimated_tdoa = estimated_arrival_times[1:] - estimated_arrival_times[0]
#     debug_print((estimated_tdoa-tdoa))
#     return estimated_tdoa - tdoa

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

def initPlt(fig, ax):
    global receivers

    ax[1][1].set_title('Click to place receivers')
    ax[1][1].set_xlim(0, 1000)
    ax[1][1].set_ylim(0, 1000)
    ax[1][1].grid(True)
    ax[1][1].set_xlabel('X coordinate (m)')
    ax[1][1].set_ylabel('Y coordinate (m)')
    plt.tight_layout()
    # Prompt user to place receivers by clicking
    print("Please click on the plot to place 3 receivers.")
    points = plt.ginput(numberOfReceivers, timeout=-1)  # Let user click 3 times
    print(points)

    # Convert the points to numpy array
    receivers = np.array(points)



def clearAxes(fig, ax):
    ax[0][0].clear()
    ax[1][0].clear()
    ax[1][1].clear()
    ax[0][1].clear()

def plot_hyperbolas(ax, receivers, tdoa, c):
    counter = 0
    for i in range(len(receivers)):
        for j in range(i + 1, len(receivers)):
            xi, yi = receivers[i]
            xj, yj = receivers[j]
            delta_d = tdoa[counter] * 1e-9 * c

            x_vals = np.linspace(0, 1000, 400)
            y_vals = np.linspace(0, 1000, 400)
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


def update_plot(fig, ax):
    estimated_positions = []
    clearAxes(fig, ax)

    for x in range(numberOfIterations):
         # Calculate true distances from transmitter to each receiver
         distances = np.linalg.norm(receivers - transmitter, axis=1)
         debug_print(("distances: ", distances))

         # Calculate arrival times in ns of signal at receivers and add snyc error
         arrival_times = channel(distances) * 1e9
         arrival_times = arrival_times + np.random.randint(syncErrorMin, syncErrorMax, size=arrival_times.shape)
         debug_print(("diffrence time arrival: ", arrival_times))

         # Calculate TDOA (Loop through each pair of receivers)
         tdoa = [arrival_times[i] - arrival_times[j] for i in range(len(arrival_times)) for j in range(i + 1, len(arrival_times))]
         debug_print(("tdoa: ", tdoa))

         # Initial guess for the transmitter's position
         initial_guess = np.array([0, 0])

         # Solve using least squares
         #result = least_squares(tdoa_residuals, initial_guess, args=(receivers, tdoa, c), verbose=2)
         result = minimize(tdoa_error, initial_guess, args=(receivers, tdoa, c))
         # Estimated position of the transmitter
         estimated_positions.append(result.x.astype(float))

    # Convert to NumPy array
    estimated_positions = np.array(estimated_positions)

    print("True distance of transmitter: ", transmitter)
    print("Estimated distance of transmitter (rounded): ", estimated_positions)


    ax[0][0].scatter(receivers[:, 0], receivers[:, 1], color='blue', label='Receivers', s=100)
    ax[0][0].scatter(transmitter[0], transmitter[1], color='red', label='True Transmitter', s=100)
    ax[0][0].scatter(estimated_positions[:,0], estimated_positions[:,1], facecolors='none', edgecolors='green',
                label='Estimated Transmitter', s=100, linewidth=2)

    # Add receiver numbers inside the blue dots
    for index, (x, y) in enumerate(receivers):
        ax[0][0].text(x, y, str(index + 1), fontsize=6, ha='center', va='center', color='white', weight='bold')

    # Draw hyperbolas based on TDOA
    plot_hyperbolas(ax[0][0], receivers, tdoa, c)

    # Adding labels
    ax[0][0].set_xlabel('X coordinate (m)')
    ax[0][0].set_ylabel('Y coordinate (m)')
    ax[0][0].set_title('Receivers and Transmitter Position')
    ax[0][0].legend()
    ax[0][0].grid(True)
    ax[0][0].axis('equal')

    ax[1][0].scatter(transmitter[0], transmitter[1], color='red', label='True Transmitter', s=100)
    ax[1][0].scatter(estimated_positions[:, 0], estimated_positions[:, 1], facecolors='none', edgecolors='green',
                     label='Estimated Transmitter', s=100, linewidth=2)

    # Adding labels
    ax[0][0].set_xlabel('X coordinate (m)')
    ax[0][0].set_ylabel('Y coordinate (m)')
    ax[0][0].set_title('Transmitter and estimations Position')
    ax[0][0].legend()
    ax[0][0].grid(True)
    ax[0][0].axis('equal')

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
    ax[1][1].set_xlim(0, 1000)
    ax[1][1].set_ylim(0, 1000)
    ax[1][1].grid(True)
    ax[1][1].set_xlabel('X coordinate (m)')
    ax[1][1].set_ylabel('Y coordinate (m)')
    ax[1][1].scatter(receivers[:, 0], receivers[:, 1], color='blue', marker='x')

    plt.tight_layout()
    plt.draw()


def onclick(event):
    if event.inaxes == ax[1][1]:
        ix, iy = event.xdata, event.ydata
        global receivers
        if event.button == 1:  # Left click to add a receiver
            receivers = np.vstack([receivers, [ix, iy]])
        elif event.button == 3:  # Right click to remove the nearest receiver
            distances = np.sqrt((receivers[:, 0] - ix) ** 2 + (receivers[:, 1] - iy) ** 2)
            if len(distances) > 0:
                nearest_index = np.argmin(distances)
                if distances[nearest_index] < 10:  # Only remove if close enough (e.g., 10 units)
                    receivers = np.delete(receivers, nearest_index, axis=0)

        update_plot(fig, ax)


if __name__ == '__main__':
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    initPlt(fig, ax)
    update_plot(fig, ax)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()