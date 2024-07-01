import numpy as np
import matplotlib.pyplot as plt

# Create a signal with two frequencies
t = np.linspace(0, 1, 1000, endpoint=False)  # Time vector
high_freq = 50  # High-frequency component
low_freq = 10  # Low-frequency component
signal = np.sin(2 * np.pi * high_freq * t) + np.sin(2 * np.pi * low_freq * t)

# Sample the signal at a lower rate
sample_rate = 300
sampled_t = np.linspace(0, 1, sample_rate, endpoint=False)
sampled_signal = np.sin(2 * np.pi * high_freq * sampled_t) + np.sin(2 * np.pi * low_freq * sampled_t)

# Perform DFT on the sampled signal
DFT_sampled_signal = np.fft.fft(sampled_signal)

# Plot the original signal and sampled signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Original Signal")
plt.plot(t, signal)
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.title("Sampled Signal (Aliased)")
plt.plot(sampled_t, sampled_signal)
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Plot the magnitude of the DFT
plt.figure(figsize=(8, 4))
plt.title("Magnitude of DFT of Sampled Signal (Aliased)")
plt.stem(np.abs(DFT_sampled_signal))
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")
plt.show()


# Dodatno ______________________________________________________________________

# Create the original signal with a 50Hz component
t = np.linspace(0, 1, 1000, endpoint=False)
original_signal = np.sin(2 * np.pi * 50 * t)
original_signal = np.sin(2 * np.pi * high_freq * t  ) + np.sin(2 * np.pi * low_freq * t   )


# Define different sampling frequencies ranging from 100Hz down to 20Hz
sample_frequencies = np.linspace(100, 20, 10)

# Create a colormap for plotting
cmap = plt.get_cmap("viridis")
colors = [cmap(i) for i in np.linspace(0, 1, len(sample_frequencies))]

# Initialize a subplot for the original signal
plt.figure(figsize=(12, 6))
plt.subplot(1, 1, 1)
plt.title("Original Signal (50Hz)")
plt.plot(t, original_signal, label="Original Signal", color='blue')

# Sample and plot each signal with different sampling frequencies
for i, sample_freq in enumerate(sample_frequencies):
    sampled_t = np.linspace(0, 1, int(100 / sample_freq), endpoint=False)
    sampled_signal = np.sin(2 * np.pi * 50 * sampled_t)  # Sample at 50Hz
    plt.plot(sampled_t, sampled_signal, label=f"Sampled at {sample_freq}Hz", color=colors[i])

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
