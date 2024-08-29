# Dissertation Code: Risks and Mitigations of Geolocating LoRa Devices

## Overview

This repository contains the code used for my dissertation, titled **"Risks and Mitigations of Geolocating LoRa Devices."** The research focuses on evaluating the risks associated with geolocating LoRa devices and proposes mitigation strategies. The project includes simulation and visualization GUI's for different geolocation techniques, including Time Difference of Arrival (TDoA), Angle of Arrival (AoA), and Received Signal Strength Indicator (RSSI).

:
### 1. Simulation Files (`xx_simulation.py`)

These files simulate the average estimation error for each geolocation technique (TDoA, AoA, RSSI). The simulations are used to evaluate the accuracy and effectiveness of each method under various conditions.

- **`TDoA_simulation.py`**: Simulates the average estimation error using the TDoA technique.
- **`AoA_simulation.py`**: Simulates the average estimation error using the AoA technique.
- **`RSSI_simulation.py`**: Simulates the average estimation error using the RSSI technique.

### 2. GUI Files (`xx_GUI.py`)

The GUI files are designed to help visualize the workings of the geolocation techniques and time resolution concepts. 

- **`geolocation_GUI.py`**: Visualizes the  geolocation techniques.
- **`LoRa_correlation_GUI.py`**: Visualizes the concept of time resolution in geolocation.

## Requirements

To run the code, you need the following Dependencies:
- `Python 3.9.13`
- `numpy`
- `matplotlib`
- `scipy`

You can install the required packages using the following command:

```bash
pip install numpy matplotlib scipy
