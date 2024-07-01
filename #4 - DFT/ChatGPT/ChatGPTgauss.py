import numpy as np
import matplotlib.pyplot as plt

# Create a Gaussian function
t = np.linspace(-10, 10, 1000)
gaussian = np.exp(-t**2 / 2)

# Shift the data to center the Gaussian function
gaussian_centered = np.roll(gaussian, len(gaussian)//2)

# Perform the Fourier transform
gaussian_fft = np.fft.fft(gaussian_centered)
frequency = np.fft.fftfreq(len(gaussian_centered), d=t[1] - t[0])

# Plot the Gaussian function and its Fourier transform
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Gaussian Function")
plt.plot(t, gaussian_centered)
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.title("Fourier Transform of Gaussian Function")
plt.plot(frequency, np.abs(gaussian_fft))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.xlim(-2, 2)  # Zoom in to see the central peak

plt.tight_layout()
plt.show()