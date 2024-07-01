import numpy as np
import matplotlib.pyplot as plt

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

    # Apply the Hann window to the sampled signal
    hann_window = np.hanning(len(sampled_signal))
    sampled_signal_windowed = sampled_signal * hann_window

    # Calculate deviation from the correct result
    deviation = original_signal[:len(sampled_signal)] - sampled_signal
    deviations.append(deviation)

    # Reconstruct the signal using the Inverse Fourier Transform
    reconstructed_signal = np.fft.ifft(np.fft.fft(sampled_signal_windowed, len(original_signal)))
    reconstructed_signals.append(reconstructed_signal)

    plt.plot(sampled_t, sampled_signal_windowed, label=f"Sampled at {sample_freq}Hz (Windowed)", color=colors[i])

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

# Plot the deviation
plt.subplot(2, 1, 2)
plt.title("Reconstructed Signals (Inverse Fourier Transform)")

for i, sample_freq in enumerate(sample_frequencies):
    plt.plot(t, np.real(reconstructed_signals[i]), label=f"Sampled at {sample_freq}Hz (Windowed)", color=colors[i])

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()