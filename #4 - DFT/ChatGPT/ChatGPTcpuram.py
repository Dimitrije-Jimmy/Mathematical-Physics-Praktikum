import numpy as np
import matplotlib.pyplot as plt
import time
import psutil

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = 0
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

sample_points = [100, 200, 400, 800, 1600]  # Varying number of points
sample_points = np.arange(100, 1600, 50)  # Varying number of points

custom_dft_cpu = []
custom_dft_ram = []

numpy_fft_cpu = []
numpy_fft_ram = []

for N in sample_points:
    print(N)
    t = np.linspace(0, 400, N, endpoint=False)  # Keep the same range, vary the number of points
    s_t = np.sin(2 * np.pi / 100 * t) + np.cos(2 * np.pi / 0.08 * t)
    
    # Monitor CPU and RAM usage before the operation
    cpu_before = psutil.cpu_percent()
    ram_before = psutil.virtual_memory().used
    
    start_time = time.time()
    S_f = dft(s_t)
    custom_dft_time = time.time() - start_time
    
    # Monitor CPU and RAM usage after the operation
    cpu_after = psutil.cpu_percent()
    ram_after = psutil.virtual_memory().used
    
    custom_dft_cpu.append(cpu_after - cpu_before)
    custom_dft_ram.append(ram_after - ram_before)

    # Repeat the same process for NumPy FFT
    cpu_before = psutil.cpu_percent()
    ram_before = psutil.virtual_memory().used

    start_time = time.time()
    S_f_np = np.fft.fft(s_t)
    numpy_fft_time = time.time() - start_time
    
    cpu_after = psutil.cpu_percent()
    ram_after = psutil.virtual_memory().used
    
    numpy_fft_cpu.append(cpu_after - cpu_before)
    numpy_fft_ram.append(ram_after - ram_before)

# Plot the CPU and RAM usage vs. the number of sample points
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(sample_points, custom_dft_cpu, label="Custom DFT CPU Usage")
plt.plot(sample_points, numpy_fft_cpu, label="NumPy FFT CPU Usage")
plt.xlabel("Number of Sample Points")
plt.ylabel("CPU Usage (%)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sample_points, custom_dft_ram, label="Custom DFT RAM Usage")
plt.plot(sample_points, numpy_fft_ram, label="NumPy FFT RAM Usage")
plt.xlabel("Number of Sample Points")
plt.ylabel("RAM Usage (bytes)")
plt.legend()

plt.tight_layout()
plt.show()
