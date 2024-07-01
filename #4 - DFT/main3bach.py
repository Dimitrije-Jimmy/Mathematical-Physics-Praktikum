import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

import librosa

import os
directory = os.path.dirname(__file__)+'\\Bach\\'

# Functions ____________________________________________________________________
def DFT(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1)) # in essence row->column transformation, k*n is then the dot product..
    #print("Vector",k)
    #print("Matrix",k*n)
    M = np.exp(-2j * np.pi * k * n / N)
    #print("Matrix",M)
    return np.dot(M, x)


def prepare_time(n, T):
    t_min = 0
    t_max = T
    return np.linspace(t_min, t_max, n, endpoint=False)


def prepare_freq(n, T):
    """
    Return DFT frequencies.
    n: number of samples (window size?)
    T: sample spacing (1/sample_rate)
    """
    freq = np.empty(n)
    scale = 1/(n * T)

    if n % 2 == 0:  # Even
        N = int(n/2 + 1)
        freq[:N] = np.arange(0, N)
        freq[N:] = np.arange(-(n/2) + 1, 0)
    else:
        N = int((n-1)/2)
        print(N)
        freq[:N] = np.arange(0, N)
        freq[N:] = np.arange(-N - 1, 0)

    return freq*scale


def load_mp3(filename, preparetime=False):
    signal, fs_rate = librosa.load(filename, sr=None)
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1)/2  # Povpreci oba channela
    n = signal.shape[0]  # Number of samples
    secs = n / fs_rate
    T = 1/fs_rate
    print("Sampling frequency: {}\nSample period: {}\nSamples: {}\nSecs: {}\n".format(fs_rate, T, n, secs))
    if preparetime:
        # t = np.linspace(0, secs, n)
        t = np.arange(0, secs, T)
        return signal, t
    else:
        return signal

def analyze_mp3(filename, onesided=False):
    signal, fs_rate = librosa.load(filename, sr=None)
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1)/2  # Povpreci oba channela
    n = signal.shape[0]  # Number of samples
    secs = n / fs_rate
    T = 1/fs_rate
    print("Sampling frequency: {}\nSample period: {}\nSamples: {}\nSecs: {}\n".format(fs_rate, T, n, secs))
    # ft = dft(signal) # waaaaaaay slow
    ft = np.fft.fft(signal)
    # t = np.linspace(0, secs, n)
    t = np.arange(0, secs, T)
    print(secs, t)
    if onesided:
        return t, signal, prepare_freq(np.array(ft).size//2, t[1]-t[0]), ft[:n//2]
    return t, signal, prepare_freq(np.array(ft).size, t[1]-t[0]), ft, n


# Plotting _____________________________________________________________________

#data, t = load_mp3('sine_440.mp3', preparetime=True)
#dft = np.fft.fft(data)
#freqs = np.fft.fftfreq(data.size, t[1]-t[0])

filename = 'Bach.44100.mp3'

t, signal, freq, ft_signal, n = analyze_mp3(directory+filename)

#print(ft_signal)
freq = np.delete(freq, np.s_[int(n//2):])
ft_signal = np.delete(ft_signal, np.s_[int(n//2):])
#print(ft_signal)


plt.plot(freq, np.abs(ft_signal)**2, color="#772D8B")
plt.title("Spekter datoteke {}".format(filename))
plt.xlabel(r"$\nu$")
plt.ylabel(r"Amplituda")
#plt.xscale("log")
plt.yscale("log")
plt.xlim(0, 25000)

plt.title("Degredacija kvalitete Bachove partite")
plt.show()