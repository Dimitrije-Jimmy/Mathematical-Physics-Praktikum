import numpy as np
import matplotlib.pyplot as plt
import time

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = 0
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

sample_points = [100, 200, 400, 800, 1600]  # Varying number of points

custom_dft_times = []
numpy_fft_times = []

for N in sample_points:
    print(N)
    t = np.linspace(0, 400, N, endpoint=False)  # Keep the same range, vary the number of points
    s_t = np.sin(2 * np.pi / 100 * t) + np.cos(2 * np.pi / 0.08 * t)
    
    start_time = time.time()
    S_f = dft(s_t)
    custom_dft_time = time.time() - start_time
    custom_dft_times.append(custom_dft_time)

    start_time = time.time()
    S_f_np = np.fft.fft(s_t)
    numpy_fft_time = time.time() - start_time
    numpy_fft_times.append(numpy_fft_time)

# Plot the execution times vs. the number of sample points
plt.plot(sample_points, custom_dft_times, label="Custom DFT")
plt.plot(sample_points, numpy_fft_times, label="NumPy FFT")
plt.xlabel("Number of Sample Points")
plt.ylabel("Execution Time (seconds)")
plt.legend()
plt.title("Execution Time vs. Number of Sample Points")
plt.show()


# Deviation ____________________________________________________________________
# Create the original signal with a 50Hz component
t = np.linspace(0, 1, 1000, endpoint=False)
original_signal = np.sin(2 * np.pi * 50 * t)

# Define different sampling frequencies ranging from 100Hz down to 20Hz
sample_frequencies = np.linspace(100, 20, 10)

# Create a colormap for plotting
cmap = plt.get_cmap("viridis")
colors = [cmap(i) for i in np.linspace(0, 1, len(sample_frequencies))]

# Initialize subplots for the deviation and inverse Fourier transform
plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.title("Deviation from Correct Result")
plt.plot(t, original_signal, label="Original Signal", color='blue')

# Initialize arrays to store deviations and reconstructed signals
deviations = []
reconstructed_signals = []

# Sample and plot each signal with different sampling frequencies
for i, sample_freq in enumerate(sample_frequencies):
    sampled_t = np.linspace(0, 1, int(100 / sample_freq), endpoint=False)
    sampled_signal = np.sin(2 * np.pi * 50 * sampled_t)  # Sample at 50Hz

    # Calculate deviation from the correct result
    deviation = original_signal[:len(sampled_signal)] - sampled_signal
    deviations.append(deviation)

    # Reconstruct the signal using the Inverse Fourier Transform
    reconstructed_signal = np.fft.ifft(np.fft.fft(sampled_signal, len(original_signal)))
    reconstructed_signals.append(reconstructed_signal)

    plt.plot(sampled_t, sampled_signal, label=f"Sampled at {sample_freq}Hz", color=colors[i])

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

# Plot the deviation
plt.subplot(2, 1, 2)
plt.title("Reconstructed Signals (Inverse Fourier Transform)")

for i, sample_freq in enumerate(sample_frequencies):
    plt.plot(t, np.real(reconstructed_signals[i]), label=f"Sampled at {sample_freq}Hz", color=colors[i])

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

