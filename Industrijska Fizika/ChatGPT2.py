import numpy as np
import matplotlib.pyplot as plt

# Load phantom data from a .dat file
file_path = 'path/to/your/phantom.dat'


import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\'
print(dir_path)

file_phantom = dir_path+'phantom.dat'
file_sg1 = dir_path+'sg1.dat'
file_sg2 = dir_path+'sg2.dat'

phantom_data = np.loadtxt(file_phantom)

# Define parameters
phantom_size = int(np.sqrt(phantom_data.size))
theta = np.linspace(0., 180., 180, endpoint=False)

# Reshape the phantom data to a 2D array
phantom = phantom_data.reshape((phantom_size, phantom_size))


# Simulate projections
projections = np.zeros((len(theta), phantom_size))

for i, angle in enumerate(theta):
    # Perform simple sum along each line for back projection
    projections[i, :] = np.sum(np.roll(phantom, int(angle), axis=1), axis=0)


# Filter the projections
filtered_projections = np.fft.fftshift(projections, axes=0)

# Apply the filter to each projection individually
for i in range(phantom_size):
    filtered_projections[:, i] *= np.linspace(0, 1, len(theta))

filtered_projections = np.fft.ifftshift(filtered_projections, axes=0)

"""
# Back-project the filtered projections
reconstruction = np.zeros_like(phantom)

for i, angle in enumerate(theta):
    # Perform simple back projection
    reconstruction += np.roll(filtered_projections[:, i], -int(angle), axis=0)

# Normalize the reconstruction to account for the number of projections
reconstruction /= len(theta)
"""
"""
def back_project(projection, angle, size):
    # Create a 2D grid for the back-projection
    x = np.arange(-size // 2, size // 2)
    y = np.arange(-size // 2, size // 2)
    X, Y = np.meshgrid(x, y)

    # Compute the coordinates along the projection direction
    t = X * np.cos(np.deg2rad(angle)) + Y * np.sin(np.deg2rad(angle))

    # Interpolate the projection values onto the grid
    from scipy.interpolate import interp1d
    interp_func = interp1d(np.arange(-size // 2, size // 2), projection, bounds_error=False, fill_value=0)
    return interp_func(t)
"""
"""
def back_project(projection, angle, size):
    # Create a 2D grid for the back-projection
    projection_length = len(projection)
    x = np.linspace(-size // 2, size // 2, projection_length)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)

    # Compute the coordinates along the projection direction
    t = X * np.cos(np.deg2rad(angle)) + Y * np.sin(np.deg2rad(angle))

    # Interpolate the projection values onto the grid
    from scipy.interpolate import interp1d
    interp_func = interp1d(x, projection, bounds_error=False, fill_value=0)
    return interp_func(t)
"""
def back_project(projection, angle, size):
    # Create a 2D grid for the back-projection
    x = np.arange(-size / 2, size / 2)
    y = np.arange(-size / 2, size / 2)
    X, Y = np.meshgrid(x, y)

    # Compute the coordinates along the projection direction
    t = X * np.cos(np.deg2rad(angle)) + Y * np.sin(np.deg2rad(angle))

    # Interpolate the projection values onto the grid
    from scipy.interpolate import interp1d
    projection_length = len(projection)
    t_points = np.linspace(-projection_length / 2, projection_length / 2, projection_length)
    interp_func = interp1d(t_points, projection, bounds_error=False, fill_value=0)
    return interp_func(t)

# Back-project the filtered projections
reconstruction = np.zeros_like(phantom)
"""
for i, angle in enumerate(theta):
    projection = filtered_projections[:, i]
    reconstruction += back_project(projection, angle, phantom_size)
"""
# Assuming filtered_projections is shaped (180, 100)
for i, angle in enumerate(theta):
    projection = filtered_projections[i, :]  # Get the ith projection
    reconstruction += back_project(projection, angle, phantom_size)

# Normalize the reconstruction to account for the number of projections
reconstruction /= len(theta)


# Display results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(phantom, cmap='gray')
plt.title('Original Phantom')

plt.subplot(132)
plt.imshow(projections, cmap='gray', extent=(0, 180, 0, phantom_size), aspect='auto')
plt.title('Sinogram')

plt.subplot(133)
plt.imshow(reconstruction, cmap='gray')
plt.title('Filtered Back Projection Reconstruction')

plt.show()
