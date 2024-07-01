import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

from main5_final import *

import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\'
print(dir_path)

# Grizzly and tit pic ______________________________________________________
    
from skimage import io, color, transform

# Load the image
file_Grizzly = dir_path+'Grizzly.jpg'
file_Grizzly2 = dir_path+'Grizzly2.jpg'
file_GrizzlyTit = dir_path+'GrizzlyTit.jpg'
original_image = io.imread(file_GrizzlyTit)
gray_image = color.rgb2gray(original_image)

# Pad the image with zeros to ensure it is zero outside the reconstruction circle
#padded_image = np.pad(gray_image, ((0, 0), (0, gray_image.shape[1]//2)), mode='constant')
padded_image = gray_image

# Perform Radon transform to create sinogram
theta = np.linspace(0., 180., max(padded_image.shape), endpoint=False)
sinogram = transform.radon(padded_image, theta=theta)#, circle=True)

# Perform filtered back projection to reconstruct the image
reconstructed_image = transform.iradon(sinogram, theta=theta)#, circle=True)

# Manual with FFT
filtered_sinogram = filtersinc(sinogram)
reconstructed_manual = backproject3(filtered_sinogram, theta)

# Plotting the results
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Plot the original image
axes[0].imshow(padded_image, cmap='gray')#, aspect='auto', origin='lower')
axes[0].set_title('Originalna slika')

# Plot the sinogram
axes[1].imshow(sinogram, cmap='gray', extent=[0, 180, 0, sinogram.shape[0]], aspect='auto')#, origin='lower')
axes[1].set_title('Sinogram')

# Plot the reconstructed image FFT
axes[2].imshow(reconstructed_manual, cmap='gray')#, aspect='auto', origin='lower')
axes[2].set_title('Rekonstruirana slika z FFT')

# Plot the reconstructed image
axes[3].imshow(reconstructed_image, cmap='gray')#, aspect='auto', origin='lower')
axes[3].set_title('Rekonstruirana slika z Iradon')

# Set a main title for the entire figure
fig.suptitle('Grizzly pozira', fontsize=16)

plt.tight_layout()
plt.show()
