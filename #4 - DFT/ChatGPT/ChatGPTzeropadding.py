import numpy as np
import matplotlib.pyplot as plt

# Create the original signal with a 50Hz component
t = np.linspace(0, 1, 1000, endpoint=False)
original_signal = np.sin(2 * np.pi * 50 * t)

# Define different amounts of zero-padding
zero_padding_factors = [1, 2, 4]

# Initialize a subplot for the original signal
plt.figure(figsize=(12, 6))
plt.subplot(1, 1, 1)
plt.title("Original Signal (50Hz)")
plt.plot(t, original_signal, label="Original Signal", color='blue')

# Initialize arrays to store zero-padded signals and their Fourier transforms
zero_padded_signals = []
zero_padded_frequencies = []

# Zero-pad the original signal and calculate Fourier transform
for factor in zero_padding_factors:
    # Calculate the new length after zero-padding
    new_length = len(original_signal) * factor
    
    # Create the zero-padded signal
    zero_padded_signal = np.pad(original_signal, (0, new_length - len(original_signal)), 'constant')
    
    # Perform the Fourier transform
    zero_padded_frequency = np.fft.fft(zero_padded_signal)
    
    zero_padded_signals.append(zero_padded_signal)
    zero_padded_frequencies.append(zero_padded_frequency)

    plt.plot(np.linspace(0, 1, new_length, endpoint=False), zero_padded_signal, label=f"Zero-Padded by {factor}", alpha=0.7)

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()