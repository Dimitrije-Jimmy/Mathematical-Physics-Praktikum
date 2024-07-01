import numpy as np
import matplotlib.pyplot as plt
import time
import psutil


# Function to calculate autocorrelation manually
def manual_autocorrelation(signal):
    N = len(signal)
    autocorr = np.zeros(2 * N - 1)
    for k in range(2 * N - 1):
        for n in range(N):
            if k - n >= 0 and k - n < N:
                autocorr[k] += signal[n] * signal[k - n]
        autocorr[k] /= N  # Normalize by the number of samples
    return autocorr

def correlate(g, h):  # Simple and quick version
    N = g.shape[0]
    g = np.concatenate((g, np.zeros(N)))
    h = np.concatenate((h, np.zeros(N)))
    n = np.arange(0, 2*N)

    G = np.fft.fft(g)
    H = np.fft.fft(h)
    p = G * np.conj(H)
    pi = 1/N*np.fft.ifft(p)
    return n, np.real(np.fft.ifftshift(pi))

# Parameters
A = 1.0          # Amplitude of the sinusoid
f0 = 0.1        # Frequency of the sinusoid
N = 200         # Number of samples
#N = 500         # Number of samples
noise_stddev = 0.2  # Standard deviation of the noise

# Generate the noisy sinusoidal signal
n = np.arange(N)
n = np.linspace(-10, 10, N)
signal = A * np.sin(2 * np.pi * f0 * n) + np.random.normal(0, noise_stddev, N)
signal2 = A * np.sin(2 * np.pi * f0 * n) - 10
signal4 = A*np.sin(2 * np.pi * f0 * n -np.pi*n) 

# Calculate and plot the autocorrelation
lags = np.arange(-N + 1, N, 2)
signal3 = A * np.sin(2 * np.pi * f0 * lags) + np.random.normal(0, noise_stddev, N)

autocorrelation = np.correlate(signal, signal, mode='full') 
autocorrelation /= (np.max(autocorrelation) + 1)
autocorrelation = manual_autocorrelation(signal)
xdata, autocorrelation = correlate(signal, signal)

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(n, signal)
plt.title("Noisy Sinusoidal Signal")

# Plot the autocorrelation
lags = np.arange(-N + 1, N)
plt.subplot(3, 1, 2)
#plt.plot(lags, autocorrelation[:-1])
plt.plot(xdata, autocorrelation)
plt.title("Autocorrelation")
plt.xlabel("Lag (k)")

xdata, autocorrelation = correlate(autocorrelation, autocorrelation)
xdata, autocorrelation = correlate(autocorrelation, autocorrelation)
xdata, autocorrelation = correlate(autocorrelation, autocorrelation)
xdata, autocorrelation = correlate(autocorrelation, autocorrelation)
xdata, autocorrelation = correlate(autocorrelation, autocorrelation)
xdata, autocorrelation = correlate(autocorrelation, autocorrelation)
xdata, autocorrelation = correlate(autocorrelation, autocorrelation)
plt.subplot(3, 1, 3)
#plt.plot(lags, autocorrelation[:-1])
plt.plot(xdata, autocorrelation)
plt.title("Autocorrelation multiple times (8)")
plt.xlabel("Lag (k)")

plt.tight_layout()
plt.show()


# Measure the execution time and memory usage
start_time_builtin = time.time()
autocorrelation_builtin = np.correlate(signal, signal, mode='full')
autocorrelation_builtin /= np.max(autocorrelation_builtin)
end_time_builtin = time.time()
cpu_usage_builtin = psutil.cpu_percent(interval=0.1)
ram_usage_builtin = psutil.virtual_memory().percent

start_time_manual = time.time()
autocorrelation_manual = manual_autocorrelation(signal)
end_time_manual = time.time()
cpu_usage_manual = psutil.cpu_percent(interval=0.1)
ram_usage_manual = psutil.virtual_memory().percent

# Calculate the difference between the two methods
difference = autocorrelation_manual - autocorrelation_builtin


import sys
sys.exit()
# Plot results
plt.figure(figsize=(12, 12))

# Plot the original signal
plt.subplot(4, 1, 1)
plt.plot(n, signal)
plt.title("Noisy Sinusoidal Signal")

# Plot the difference between manual and built-in calculations
plt.subplot(4, 1, 2)
plt.plot(difference)
plt.title("Difference between Manual and Built-in Autocorrelation")

# Plot the execution time comparison
plt.subplot(4, 1, 3)
plt.plot(["Manual", "Built-in"], [end_time_manual - start_time_manual, end_time_builtin - start_time_builtin], marker='o')
plt.title("Execution Time Comparison")

# Plot the CPU and RAM usage comparison
plt.subplot(4, 1, 4)
plt.plot(["Manual", "Built-in"], [cpu_usage_manual, cpu_usage_builtin], label='CPU Usage', marker='o', alpha=0.6)
plt.plot(["Manual", "Built-in"], [ram_usage_manual, ram_usage_builtin], label='RAM Usage', marker='o', alpha=0.6)
plt.title("CPU and RAM Usage Comparison")
plt.legend()

plt.tight_layout()
plt.show()
