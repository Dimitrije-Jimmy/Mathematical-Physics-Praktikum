import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift

def read_sinogram(file_path):
    return np.loadtxt(file_path).T

def heaviside_func(x, k0):
    return np.heaviside(k0 - np.abs(x), 0.5)

def ramp_filter(k, k0):
    return np.abs(k) * heaviside_func(k0, np.abs(k))

def filter_sinogram(sinogram):
    num_projections, num_detectors = sinogram.shape
    ds = 2 / num_detectors  # assuming the distance range is [-1, 1]
    df = np.pi / num_projections
    kmax = (num_detectors * np.pi) / 2
    k0 = kmax / 2

    # Apply Fourier transform to get into frequency space
    sinogram_ft = fftshift(fft(sinogram, axis=0), axes=0)
    
    # Apply the ramp filter in frequency space
    k_values = np.fft.fftfreq(num_detectors, ds)
    filter_function = ramp_filter(k_values, k0)
    filtered_sinogram_ft = sinogram_ft * filter_function[:, None]
    
    # Apply inverse Fourier transform to get back into spatial space
    filtered_sinogram = ifft(ifftshift(filtered_sinogram_ft, axes=0), axis=0).real

    return filtered_sinogram


def filtersinc(PR):
    # Parameter "a" that varies the filter magnitude response
    a = 1
    
    Length, Count = PR.shape
    w = np.linspace(-np.pi, np.pi - 2*np.pi/Length, Length)

    rn1 = np.abs(2/a * np.sin(a * w/2))
    rn2 = np.sin(a * w/2)
    rd = (a * w)/2
    r = rn1 * (rn2/rd)**2

    f = np.fft.fftshift(r)
    
    g = np.zeros_like(PR, dtype=float)
    for i in range(Count):
        IMG = np.fft.fft(PR[:, i])
        fimg = IMG * f
        g[:, i] = np.fft.ifft(fimg).real

    return g

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
    #filtered_sinogram = filter_sinogram(sinogram)
    filtered_sinogram = filtersinc(sinogram)
    
    # Back projection
    reconstruction = back_projection(filtered_sinogram, theta)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(sinogram, cmap='gray', extent=[0, np.pi, -1, 1], aspect='auto')
    plt.title('Original Sinogram')

    plt.subplot(1, 3, 2)
    plt.imshow(filtered_sinogram, cmap='gray', extent=[0, np.pi, -1, 1], aspect='auto')
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
