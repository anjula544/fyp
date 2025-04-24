import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.optimize import fsolve
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

os.environ["LOKY_MAX_CPU_COUNT"] = "6" 

# Global variables
x = 0
y = 0
def scatter_plot(y_values):
    x_values = [0] * len(y_values)  # X values are all zero
    plt.scatter(x_values, y_values, color='blue', label='Y Values')
    
    plt.xlabel("X (Fixed at 0)")
    plt.ylabel("Y Values")
    plt.title("Scatter Plot with X=0")
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

def plot_epc_data(data, csv_file):
    # Load CSV file
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(6, 6))

    # Plot Grad points
    plt.scatter(data['x_grad'], data['y_grad'], label='Grad', marker='o', color='b')

    # Plot Clus points
    #plt.scatter(data['x_clus'], data['y_clus'], label='Clus', marker='x', color='r')

    # Plot CSV points
    for i, row in df.iterrows():
        plt.scatter(row['X'], row['Y'], label=f'Actual location {row["EPC"]}', marker='s', color='g')
        plt.text(row['X'], row['Y'], f' {row["EPC"]}', fontsize=10, verticalalignment='bottom', color='g')

    # Annotate Grad and Clus points with EPC values
    for i, epc in enumerate(data['EPC']):
        plt.text(data['x_grad'][i], data['y_grad'][i], f' {epc}', fontsize=10, verticalalignment='bottom', color='b')
       # plt.text(data['x_clus'][i], data['y_clus'][i], f' {epc}', fontsize=10, verticalalignment='top', color='r')

    # Set axis limits
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatter Plot of EPC Data")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def equation(Y, Xfixed, x_i, x_ip1, delta_d_i):
    term1 = np.sqrt((Xfixed - x_i)**2 + Y**2)
    term2 = np.sqrt((Xfixed - x_ip1)**2 + Y**2)
    return np.abs(term1 - term2) - delta_d_i

def find_optimum_intersection(x_array, y_array, delta_d_array):
    X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 400), np.linspace(-2.5, 2.5, 400))
    error_grid = np.zeros_like(X)
    
    for i in range(len(delta_d_array)):
        hyperbola = np.abs(np.sqrt((X - x_array[i])**2 + Y**2) - np.sqrt((X - x_array[i+1])**2 + Y**2)) - delta_d_array[i]
        error_grid += np.abs(hyperbola)  # Accumulate absolute errors
    
    # Find the points with the minimum error
    min_indices = np.where(error_grid == np.min(error_grid))
    optimum_points = [(X[min_indices][i], Y[min_indices][i]) for i in range(len(min_indices[0]))]
    
    # Select the point with the highest Y value (only +Y side)
    optimum_point = max(optimum_points, key=lambda p: p[1])
    
    return optimum_point, error_grid

def plot_hyperbolas_and_optimum(x_array, y_array, delta_d_array,point_x,epc):
    optimum_point, error_grid = find_optimum_intersection(x_array, y_array, delta_d_array)
    
    plt.figure(figsize=(8, 6))
    colors = ['b', 'r', 'g', 'm', 'c', 'y']  # Add more colors if needed

    intersections = []
    X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 400), np.linspace(-2.5, 2.5, 400))
    for i in range(len(delta_d_array)):
        hyperbola = np.abs(np.sqrt((X - x_array[i])**2 + Y**2) - np.sqrt((X - x_array[i+1])**2 + Y**2)) - delta_d_array[i]
        plt.contour(X, Y, hyperbola, levels=[0], colors=colors[i % len(colors)])
        Y_initial_guess = 1.0
        Y_solution = fsolve(equation, Y_initial_guess, args=(point_x, x_array[i], x_array[i+1], delta_d_array[i]))
        intersections.append(round(float(Y_solution[0]), 4))
        
       # plt.scatter(point_x, Y_solution, color='k', marker='x', label='Optimum Intersection (+Y)')
   
    intersections = [x for x in intersections if x >= 0]
    print(intersections)
    
    data = np.array(intersections).reshape(-1, 1)
    dbscan = DBSCAN(eps=0.5, min_samples=3)  # Adjust eps and min_samples as needed
    labels = dbscan.fit_predict(data)
    unique_labels = set(labels)
    max_label = max(unique_labels, key=lambda label: np.sum(labels == label))
    cluster_points = np.array([data[i][0] for i in range(len(labels)) if labels[i] == max_label])
    centroid_y = np.mean(cluster_points) - 0.125 * (max(cluster_points) - min(cluster_points))
    '''
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    smallest_cluster_idx = np.argmin(centroids)
    smallest_cluster_points = data[labels == smallest_cluster_idx]
    centroid_smallest_cluster = smallest_cluster_points.mean()
    '''
    optimum_point_2 = np.array([point_x,centroid_y]) 
    
    # Plot the optimum intersection point (only +Y side)
    plt.scatter(optimum_point[0], optimum_point[1], color='k', marker='o', label='Optimum Intersection (+Y)')
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"Hyperbolic Method {epc}")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
    scatter_plot(intersections)
    return optimum_point,optimum_point_2

def unwrap_phase_angles(phase_angles):
    """Unwrap phase angles and return negative values."""
    unwrapped = []
    prev_angle = phase_angles[0]
    offset = 0
    for angle in phase_angles:
        diff = angle - prev_angle
        if diff > np.pi:
            offset -= 2 * np.pi
        elif diff < -np.pi:
            offset += 2 * np.pi
        unwrapped.append(-(angle + offset))  
        prev_angle = angle
    return unwrapped



def filter_csv(input_file, output_dir, frequency=914.75):  
    global x, y

    df = pd.read_csv(input_file)

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    print("Columns in CSV:", df.columns.tolist())

    # Ensure required columns exist
    required_columns = ['Timestamp', 'EPC', 'Frequency', 'PhaseAngle','RSSI']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing columns in CSV: {missing_columns}")
        return
    
    df_filtered = df[required_columns].copy()

    # Convert Timestamp to seconds
    df_filtered['Timestamp'] = pd.to_datetime(df_filtered['Timestamp'])
    start_time = df_filtered['Timestamp'].min()
    df_filtered['Timestamp'] = (df_filtered['Timestamp'] - start_time).dt.total_seconds()

    # Filter based on frequency
    df_filtered = df_filtered[df_filtered['Frequency'] == frequency]

    # Initialize new columns with NaN instead of None
    df_filtered['UnwrappedPhaseAngle'] = np.nan
    df_filtered['MinZero'] = np.nan
    df_filtered['RegressedUnwrappedPhase'] = np.nan
    df_filtered['RegressedShiftedPhase'] = np.nan

    plt.figure(figsize=(10, 6))

    for epc, epc_df in df_filtered.groupby('EPC'):
        unwrapped_angles = unwrap_phase_angles(epc_df['PhaseAngle'].values)

        # Store negative of original phase angles
        df_filtered.loc[epc_df.index, 'PhaseAngle'] = -epc_df['PhaseAngle'].values
        df_filtered.loc[epc_df.index, 'UnwrappedPhaseAngle'] = unwrapped_angles

        # Shift unwrapped phase angles to start at zero
        min_unwrapped = np.min(unwrapped_angles)
        shifted_unwrapped = unwrapped_angles - min_unwrapped
        df_filtered.loc[epc_df.index, 'MinZero'] = shifted_unwrapped

        # Polynomial regression (degree 3)
        timestamps = epc_df['Timestamp'].values
        poly_coeffs = np.polyfit(timestamps, unwrapped_angles, deg=3)
        regressed_angles = np.polyval(poly_coeffs, timestamps)

        df_filtered.loc[epc_df.index, 'RegressedUnwrappedPhase'] = regressed_angles

        # Shift regressed values to start at zero
        min_regressed = np.min(regressed_angles)
        shifted_regressed = regressed_angles - min_regressed
        df_filtered.loc[epc_df.index, 'RegressedShiftedPhase'] = shifted_regressed

        plt.plot(epc_df['Timestamp'], shifted_regressed, marker='o', linestyle='-', label=f'EPC {epc}')


    plt.xlabel('Timestamp (seconds)')
    plt.ylabel('Regressed (shifted phase angles)')
    plt.title(f'Timestamp vs RegressedShiftedPhase for All EPCs (Frequency {frequency})')
    plt.grid(True)
    plt.legend()

    # Save combined plot
    plot_file = os.path.join(output_dir, f"combined_plot_freq_{frequency}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Combined plot saved as {plot_file}")

    unique_epcs = df_filtered['EPC'].unique()
    
    # Initialize pos_dict to hold EPC and calculated positions
    pos_dict = {
        "EPC": [],
        "x_grad": [],
        "y_grad": [],
        "x_clus": [],
        "y_clus": []
    }
    
    for epc in unique_epcs:
        epc_df = df_filtered[df_filtered['EPC'] == epc].copy()
        epc_df['Timestamp'] = pd.to_numeric(epc_df['Timestamp'])
        pos_dict["EPC"].append(str(epc))

        # Ensure meaningful time gaps
        epc_df_filtered = epc_df[epc_df['Timestamp'].diff() > 1]

        if epc_df_filtered.empty:
            print(f"No valid data for EPC {epc}, skipping...")
            continue
        
        # Find minimum value index for regressed phase
        min_zero_idx = epc_df_filtered['RegressedShiftedPhase'].idxmin()
        min_zero_time = epc_df_filtered.loc[min_zero_idx, 'Timestamp']
        #print(f"EPC {epc}: Minimum MinZero value occurs at {min_zero_time:.6f} seconds")

        x = min_zero_time * 0.496
        #print(f"EPC {epc}: X = {x:.6f} cm")

        diff_phase = epc_df_filtered['RegressedShiftedPhase'].values
        x_array = 0.00496 * epc_df_filtered['Timestamp'].values

        del_d = []

        for i in range(len(diff_phase) - 1):
            del_d_temp = 0.332318 / (4 * np.pi) * (diff_phase[i] - diff_phase[i + 1])
            del_d.append(abs(del_d_temp))  # **Absolute value of Δd**

        y_array = [0] * len(x_array)

        optimum_point_1, optimum_point_2 = plot_hyperbolas_and_optimum(x_array, y_array, del_d, x / 100, epc)
        optimum_x_1 = float(optimum_point_1[0])
        pos_dict["x_grad"].append(optimum_x_1)
        optimum_y_1 = abs(float(optimum_point_1[1]))
        pos_dict["y_grad"].append(optimum_y_1)
        print(f"Optimum Intersection Point by gradient descent {epc}: {optimum_x_1:.6f}, {abs(optimum_y_1):.6f}")

        optimum_x_2 = float(optimum_point_2[0])
        pos_dict["x_clus"].append(optimum_x_2)
        optimum_y_2 = abs(float(optimum_point_2[1]))
        pos_dict["y_clus"].append(optimum_y_2)
        print(f"Optimum Intersection Point by clustering {epc}: {optimum_x_2:.6f}, {abs(optimum_y_2):.6f}")

        est_tag = [optimum_x_1,optimum_y_1]

        generate_sar_heatmap(epc,x_array, diff_phase, est_tag)

        # Save filtered dataset for EPC
        output_file = os.path.join(output_dir, f"dataset_{epc}_freq_{frequency}.csv")
        epc_df_filtered.to_csv(output_file, index=False, float_format='%.6f')

    print(pos_dict)
    return pos_dict

def generate_sar_heatmap(epc,slider_positions, simulated_phase, true_tag=None, frequency=902.75e6, grid_size=(100, 100), xlim=(0, 1.1), ylim=(0.05, 1.2)):
    """
    Generate and plot SAR heatmap from phase data.

    Parameters:
    - slider_positions: List or np.array of X positions of the antenna (1D array)
    - simulated_phase: List or np.array of unwrapped phase values (1D array, same length as slider_positions)
    - true_tag: Tuple (x, y) of true tag position (optional, shown as cyan 'x')
    - frequency: RFID frequency in Hz (default 902.75 MHz)
    - grid_size: Tuple (num_x, num_y) resolution of heatmap
    - xlim: Tuple (min_x, max_x) range of X axis
    - ylim: Tuple (min_y, max_y) range of Y axis
    """

    c = 3e8  # Speed of light
    wavelength = c / frequency

    slider_positions = np.array(slider_positions)
    simulated_phase = np.unwrap(np.array(simulated_phase))
    amplitude = np.ones_like(slider_positions)

    x_grid = np.linspace(xlim[0], xlim[1], grid_size[0])
    y_grid = np.linspace(ylim[0], ylim[1], grid_size[1])
    X, Y = np.meshgrid(x_grid, y_grid)
    intensity = np.zeros_like(X, dtype=np.complex128)

    for i, x_pos in enumerate(slider_positions):
        R = np.sqrt((X - x_pos)**2 + Y**2)
        expected_phase = (4 * np.pi * R) / wavelength
        signal = np.exp(1j * (simulated_phase[i] - expected_phase))
        intensity += signal

    intensity_magnitude = np.abs(intensity)
    intensity_magnitude /= np.max(intensity_magnitude)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(x_grid, y_grid, intensity_magnitude, levels=100, cmap="inferno")
    plt.colorbar(label="Normalized SAR Intensity")

    if true_tag:
        plt.scatter([true_tag[0]], [true_tag[1]], color='cyan', label="Estimate by Grad (Tag Position)", marker='x')

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Distance from Antenna (m)")
    plt.title(f"SAR Backprojection {epc}")
    if true_tag:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# File paths
input_csv = "E:\FYP\Fyp_code\SAR_method\datasets\calc_14\dataset.csv"
output_directory = "E:\FYP\Fyp_code\SAR_method\datasets\calc_14"
coordinatefile = "E:\FYP\Fyp_code\SAR_method\datasets\calc_14\coordinates.csv"

# Process frequency
frequency_to_process = 902.75
pos_dict = filter_csv(input_csv, output_directory, frequency=frequency_to_process)
plot_epc_data(pos_dict, coordinatefile)
