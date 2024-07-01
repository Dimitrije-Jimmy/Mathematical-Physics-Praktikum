import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import librosa
import os
import time
import psutil

directory = os.path.dirname(__file__)+'\\Sove\\'

# Function to calculate autocorrelation manually
def manual_autocorrelation(signal):
    N = len(signal)
    n = np.arange(0, 2*N)
    autocorr = np.zeros(2 * N - 1)
    for k in range(2 * N - 1):
        for n in range(N):
            if k - n >= 0 and k - n < N:
                autocorr[k] += signal[n] * signal[k - n]
        autocorr[k] /= N  # Normalize by the number of samples
    return n, autocorr

def correlate(g, h):  # Simple and quick version
    N = g.shape[0]
    g = np.concatenate((g, np.zeros(N))) # Zero-Padding
    h = np.concatenate((h, np.zeros(N)))
    n = np.arange(0, 2*N)

    G = np.fft.fft(g)
    H = np.fft.fft(h)
    p = G * np.conj(H)
    pi = 1/N*np.fft.ifft(p)
    #pi = 1/(N-n)*np.fft.ifft(p)
    return n, np.real(np.fft.ifftshift(pi))


# np.correlate