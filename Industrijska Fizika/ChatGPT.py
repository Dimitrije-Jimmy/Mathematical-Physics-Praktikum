import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
#from skimage import *


# Generate a simple phantom image (you can replace this with your own data)
phantom_size = 256
theta = np.linspace(0., 180., 180, endpoint=False)
phantom = np.zeros((phantom_size, phantom_size))
phantom[50:200, 50:200] = 1

# Simulate projections
projections = radon(phantom, theta=theta, circle=True)

import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\'
print(dir_path)

file_phantom = dir_path+'phantom.dat'
file_sg1 = dir_path+'sg1.dat'
file_sg2 = dir_path+'sg2.dat'

#phantom = np.loadtxt(file_phantom).T
# Simulate projections
#projections = radon(phantom, theta=theta, circle=True)

projections = np.loadtxt(file_phantom).T
projections = np.loadtxt(file_sg1).T
projections = np.loadtxt(file_sg2).T
theta =  np.linspace(0., 180., len(projections), endpoint=False)
#phantom_size = 100
## Filter the projections
#filtered_projections = np.fft.fftshift(projections, axes=0)
#filtered_projections *= np.linspace(0, 1, phantom_size)[:, None]
#filtered_projections = np.fft.ifftshift(filtered_projections, axes=0)
#
## Back-project the filtered projections
#reconstruction = iradon(filtered_projections, theta=theta, circle=True)
reconstruction = iradon(projections, theta=theta, circle=True)

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