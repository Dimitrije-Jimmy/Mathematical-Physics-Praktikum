import numpy as np
import matplotlib.pyplot as plt
import time

# Step 1: Generate time values
t = np.linspace(0, 400, 400)  # 400 samples from 0 to 400

# Step 2: Calculate s(t)
s_t = np.sin(2 * np.pi / 100 * t) + np.cos(2 * np.pi / 0.08 * t)

# Step 3: Compute the DFT
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = 0
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

S_f = dft(s_t)

# Step 4: Compute the Inverse DFT to reconstruct the original signal
def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N) / N
    return x

reconstructed_s = idft(S_f)

# Step 5: Plot the results
plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
plt.title("Original Function s(t)")
plt.plot(t, s_t)

plt.subplot(5, 1, 2)
plt.title("Real Part of DFT")
plt.plot(np.real(S_f))

plt.subplot(5, 1, 3)
plt.title("Imaginary Part of DFT")
plt.plot(np.imag(S_f))

plt.subplot(5, 1, 4)
plt.title("Energy/Parseval Values")
energy = np.abs(S_f) ** 2 / len(S_f)
plt.plot(energy)

plt.subplot(5, 1, 5)
plt.title("Reconstructed Signal from IDFT")
plt.plot(t, np.real(reconstructed_s))

plt.tight_layout()
plt.show()


# Time _________________________________________________________________________
sample_points = [100, 200, 400, 800, 1600]  # Varying number of sample points

custom_dft_times = []
numpy_fft_times = []

for N in sample_points:
    t = np.linspace(0, N, N)
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

