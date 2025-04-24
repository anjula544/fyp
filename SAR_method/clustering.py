import numpy as np
import matplotlib.pyplot as plt

# Simulated slider positions (56 positions over 1.1m)
slider_positions = d = [0.061620947, 0.111987684, 0.162398802, 0.212804113, 0.263313003, 0.313700438, 0.364213628, 0.414801303, 0.464935302, 0.515589634, 0.565872567, 0.616357763, 0.666815471, 0.717279379, 0.767690716, 0.818013983, 0.868629464, 0.918881808, 0.969305456, 1.019756854, 1.07015718]
slider_positions = np.array(slider_positions)

# Simulated tag position (ground truth)
true_tag_x = 0.65  # meters
true_tag_y = 0.9   # meters

# Parameters
frequency = 902.75e6  # 920 MHz typical RFID freq
c = 3e8  # speed of light
wavelength = c / frequency

# Simulate phase response from tag at (true_tag_x, true_tag_y)
simulated_phase = [6.208949, 5.136964, 4.166814, 3.299207, 2.532177, 1.86871, 1.305182, 0.84239, 0.483651, 0.221815, 0.061556, 0.000165, 0.038058, 0.17482, 0.409775, 0.74197, 1.17413, 1.700076, 2.324469, 3.045726, 3.862275]
simulated_phase = np.unwrap(np.array(simulated_phase))

# Constant amplitude for simplicity
amplitude = np.ones_like(simulated_phase)

# Create 2D grid for SAR image
x_grid = np.linspace(0, 1.1, 100)
y_grid = np.linspace(0.05, 1.2, 100)
X, Y = np.meshgrid(x_grid, y_grid)
intensity = np.zeros_like(X, dtype=np.complex128)

# SAR Backprojection
for i, x_pos in enumerate(slider_positions):
    R = np.sqrt((X - x_pos)**2 + Y**2)
    expected_phase = (4 * np.pi * R) / wavelength
    signal =  np.exp(1j * (simulated_phase[i] - expected_phase))
    intensity += signal

# Final image intensity (magnitude)
intensity_magnitude = np.abs(intensity)
intensity_magnitude /= np.max(intensity_magnitude)

# Find maximum intensity point (estimated tag location)
max_idx = np.unravel_index(np.argmax(intensity_magnitude), intensity_magnitude.shape)
estimated_tag_x = x_grid[max_idx[1]]
estimated_tag_y = y_grid[max_idx[0]]

# Plot SAR image
plt.figure(figsize=(10, 6))
plt.contourf(x_grid, y_grid, intensity_magnitude, levels=100, cmap="inferno")
plt.colorbar(label="Normalized SAR Intensity")
plt.scatter([true_tag_x], [true_tag_y], color='cyan', label="True Tag Position", marker='x')
plt.scatter([estimated_tag_x], [estimated_tag_y], color='lime', label="Estimated Tag Position", marker='o')
plt.xlabel("X Position (m)")
plt.ylabel("Y Distance from Antenna (m)")
plt.title("Corrected 2D RFID Localization Using 1D SAR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Estimated Tag Position: x = {estimated_tag_x:.3f} m, y = {estimated_tag_y:.3f} m")
