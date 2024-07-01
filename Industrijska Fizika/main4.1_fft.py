import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def read_sinogram(file_path):
    return np.loadtxt(file_path)#.T

def ram_lak_filter(size):
    # Generate a 1D Ram-Lak filter
    filter_1d = np.abs(np.linspace(-1, 1, size))
    return np.outer(filter_1d, filter_1d)

def filter_sinogram(sinogram):
    # Apply a filter to the sinogram (e.g., Ram-Lak filter)
    
    # attempts 1-5
    filtered_sinogram = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(sinogram, axes=0), axis=0), axes=0).real
    #filtered_sinogram = np.fft.ifft(np.fft.ifftshift(sinogram, axes=0), axis=0).real
    #filtered_sinogram = np.fft.ifft(sinogram, axis=0).real
    #filtered_sinogram = np.fft.fftshift(np.fft.ifft(sinogram, axis=0), axes=0).real
    #filtered_sinogram = sinogram
    
    # attempt 6
    #filter_2d = ram_lak_filter(sinogram.shape[0])
    #filtered_sinogram = convolve2d(sinogram, filter_2d, mode='same', boundary='symm')

    # attempt 7
    #num_projections, num_detectors = sinogram.shape
    #num_projections = sinogram.shape[0]
    #filter_2d = ram_lak_filter(num_projections)
    #
    #sinogram_ft = np.fft.fftshift(np.fft.fft(sinogram, axis=0), axes=0)
    #filtered_sinogram_ft = sinogram_ft * filter_2d[:, None]
    #filtered_sinogram = np.fft.ifft(np.fft.ifftshift(filtered_sinogram_ft, axes=0), axis=0).real
    
    return filtered_sinogram

def filter_sinogram3(sinogram):
    # Apply a filter to the sinogram (e.g., Ram-Lak filter)

    num_projections = sinogram.shape[0]
    filter_2d = ram_lak_filter(num_projections)

    sinogram_ft = np.fft.fftshift(np.fft.fft(sinogram, axis=0), axes=0)
    filtered_sinogram_ft = sinogram_ft * filter_2d[:, None]
    filtered_sinogram = np.fft.ifft(np.fft.ifftshift(filtered_sinogram_ft, axes=0), axis=0).real

    return filtered_sinogram

def filter_sinogram(sinogram):
    Length, Count = sinogram.shape
    #w = np.linspace(-pi:(2*pi)/Length, np.pi - (2*np.pi)/Length)
    
    

def back_projection(filtered_sinogram, theta):
    # Back projection
    reconstruction = np.zeros_like(filtered_sinogram)
    num_projections, num_detectors = filtered_sinogram.shape
    #num_projections = filtered_sinogram.shape[0]

    for i in range(num_projections):
        angle = theta[i]
        projection = filtered_sinogram[i, :]
        reconstruction += np.roll(projection, int(angle), axis=0)

    return reconstruction

def main(sinogram_file):
    # Read sinogram data
    sinogram = read_sinogram(sinogram_file)

    # Define the angles for each projection
    theta = np.linspace(0, np.pi, sinogram.shape[0], endpoint=False)

    # Filter the sinogram
    filtered_sinogram = filter_sinogram(sinogram)

    # Back projection
    reconstruction = back_projection(filtered_sinogram, theta)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(sinogram, cmap='gray', extent=[0, 180, 0, sinogram.shape[1]], aspect='auto')
    plt.title('Sinogram')

    plt.subplot(1, 3, 2)
    plt.imshow(filtered_sinogram, cmap='gray', extent=[0, 180, 0, sinogram.shape[1]], aspect='auto')
    plt.title('Filtered Sinogram')

    plt.subplot(1, 3, 3)
    plt.imshow(reconstruction, cmap='gray', extent=[-1, 1, -1, 1], aspect='auto')
    plt.title('Reconstruction')

    plt.show()


if __name__ == "__main__":
    
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))+'\\'
    print(dir_path)

    file_phantom = dir_path+'phantom.dat'
    file_sg1 = dir_path+'sg1.dat'
    file_sg2 = dir_path+'sg2.dat'
    
    main(file_phantom)