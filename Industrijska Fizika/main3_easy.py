import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\'
print(dir_path)

file_phantom = dir_path+'phantom.dat'
file_sg1 = dir_path+'sg1.dat'
file_sg2 = dir_path+'sg2.dat'

sinogram = np.loadtxt(file_phantom).T
#sinogram = np.loadtxt(file_sg1).T
#sinogram = np.loadtxt(file_sg2).T
theta =  np.linspace(0., 180., len(sinogram), endpoint=False)

reconstruction = iradon(sinogram, theta=theta, circle=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
axes[0].imshow(sinogram, cmap='gray', extent=[0, np.pi, -1, 1], aspect='auto', origin='lower')
axes[0].set_xlabel(r'$\phi$')
axes[0].set_ylabel(r'$s$')
axes[0].set_title('Sinogram')

axes[1].imshow(reconstruction, cmap='gray', extent=[-1, 1, -1, 1])
axes[1].set_xlabel(r'$x$')
axes[1].set_ylabel(r'$y$')
axes[1].set_title('Rekonstruirana slika')

plt.tight_layout()
plt.show()