import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os
import time
import psutil
import librosa

directory = os.path.dirname(__file__)+'\\Sove\\'


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

def correlate2(g, h):  # Simple and quick version
    N = g.shape[0]
    g = np.concatenate((g, np.zeros(N)))
    h = np.concatenate((h, np.zeros(N)))
    n = np.arange(0, 2*N)

    G = np.fft.fft(g)
    H = np.fft.fft(h)
    p = G * np.conj(H)
    pi = 1/(N-n)*np.fft.ifft(p)
    return n, np.real(np.fft.ifftshift(pi))

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

# Function to calculate autocorrelation manually
def konvolucija_perhaps(signal1, signal2):
    # Dejansko sam konvolucija 
    N = len(signal1)
    n = np.arange(0, 2*N)
    autocorr = np.zeros(2 * N - 1)
    for k in range(2 * N - 1):
        for n in range(N):
            if k - n >= 0 and k - n < N:
                autocorr[k] += signal1[n] * signal2[k - n]
        autocorr[k] /= N  # Normalize by the number of samples
    return n, autocorr

def numpy_correlation(signal1, signal2):
    N = len(signal1)
    autocorr = np.correlate(signal1, signal2, mode='full')
    #autocorr /= np.max(autocorr)
    autocorr /= N
    return autocorr

def scipy_correlation(signal1, signal2):
    N = len(signal1)
    scipy_data = scipy.signal.correlate(signal1, signal2, method="fft")
    #scipy_data = scipy_data/np.max(scipy_data)
    scipy_data /= N
    return scipy_data

    
# Autocorrelation/correlation test plot
x = np.linspace(-3, 3, 100)
y1 = np.append(np.zeros(40), np.append(np.repeat(1, 20), np.zeros(40)))  # Box signal
y2 = np.append(np.zeros(40), np.append(np.arange(20, 0, -1)/20, np.zeros(40)))  # Sawtooth signal


xdata, data = correlate(y2, y1)
#xdata, data = manual_autocorrelation(y1)
#xdata, data = np.corrcoef(y1, y1)  # DO NOT USE!

# Plotting Wiki ________________________________________________________________
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(x, y1, c="#3344FF")
plt.title("Kvadratni signal")
plt.xlabel("n")
plt.ylabel("Normirana amplituda")

plt.subplot(2, 1, 2)
plt.plot(x, y2, c="#FF5733")
plt.xlabel("n")
plt.title("Trikotni signal")
plt.ylabel("Normirana amplituda")

plt.tight_layout()
plt.show()
plt.clf()
plt.close()

# _____________________________________________
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(data, c="#3344FF")
plt.xlabel("Vzorci")
plt.title("Samostojna implementacija korelacije")
plt.ylabel("Normirana amplituda")


scipy_data = scipy_correlation(y1, y2)

plt.subplot(2, 1, 2)
plt.plot(scipy_data, c="#FF5733")
plt.title(r"$scipy.signal.correlate$")
plt.xlabel("Vzorci")
plt.ylabel("Normirana amplituda")

plt.tight_layout()
plt.show()
plt.clf()
plt.close()


# 1/N or 1/N-n _________________________________________________________________

signal_bubomono, fs_rate_bubomono = librosa.load(directory+'bubomono.wav', sr=None)
signal_mix2, fs_rate_mix2 = librosa.load(directory+'mix2.wav', sr=None)

_, corr1 = correlate(signal_mix2, signal_bubomono)
_, corr2 = correlate2(signal_mix2, signal_bubomono)
n1 = np.arange(0, len(signal_mix2))/44100  # Delim z sampling frequencyjem, da dobim x-os v sekundha
n2 = np.arange(0, len(signal_bubomono))/44100 

diff = np.abs(corr2 - corr1)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(n1, signal_mix2, c="red", label='zašumljen signal')
plt.plot(n2, signal_bubomono, alpha=0.7, lw=0.7, c="orange", label='Tom :)')
plt.xlabel("Čas [s]")
plt.ylabel("Normirana amplituda")
plt.title("Vhodni signal")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(corr1, c="blue", alpha=0.7, lw=0.8, label=r'$\frac{1}{N}$')
plt.plot(corr2, c="cyan", alpha=0.7, lw=0.8, label=r'$\frac{1}{N-n}$')
plt.xlabel("n")
plt.title("Korelacijska funkcija obeh signalov")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(diff, c="#3344FF")
plt.yscale('log')
plt.xlabel("n")
plt.ylabel("Absolutna napaka")
plt.title(r"Absolutna napaka med normiranjem z $\frac{1}{N}$ in $\frac{1}{N-n}$")

plt.tight_layout()
plt.show()
plt.clf()
plt.close()

# Error ________________________________________________________________________
x = np.linspace(-3, 3, 100)
y1 = np.sin(2*np.pi*x) + 3*np.random.random(100)
y2 = np.cos(2*np.pi*x) + 3*np.random.random(100)

scipy_data = scipy_correlation(y1, y2)
numpy_data = numpy_correlation(y1, y2)
abs_error1 = np.abs(numpy_data - scipy_data)
n, corr1 = correlate(y1, y2)
abs_error2 = np.abs(corr1[:-1]- scipy_data)
_, corr2= correlate2(y1, y2)
abs_error3 = np.abs(corr2[:-1]- scipy_data)
_, diy = konvolucija_perhaps(y1, y2)
abs_error4 = np.abs(diy - scipy_data)


plt.figure(figsize=(12, 6))

#plt.scatter(n[:-1], abs_error1, c='red'  , marker='D',  s=50, alpha=0.7, lw=0.7, label=r'numpy.correlate')
#plt.scatter(n[:-1], abs_error1, c='blue' , marker='o',  s=50, alpha=0.7, lw=0.7, label=r'DIY z numpy $1/N$')
##plt.scatter(n[:-1], abs_error1, c='purple', marker='s', s=50, alpha=0.7, lw=0.7, label=r'DIY z numpy $1/(N-n)$')
#plt.scatter(n[:-1], abs_error1, c='green', marker='x',  s=50, alpha=0.7, lw=0.7, label=r'DIY brez numpy')

plt.subplot(3, 1, 1)
plt.axhline(y=1e-15, color='black', linestyle='--')
plt.plot(n[:-1], abs_error1, c='red'  ,  alpha=1, lw=1, label=r'numpy.correlate')
plt.ylabel(r"Absolutna napaka")
plt.xlabel("Vzorci")
plt.yscale("log")
plt.legend()
plt.title(r"Absolutna napaka v odvisnosti od $scipy.signal.correlate$")



plt.subplot(3, 1, 2)
plt.axhline(y=1e-15, color='black', linestyle='--')
plt.plot(n[:-1], abs_error1, c='blue' ,  alpha=1, lw=1, label=r'DIY z numpy')# $1/N$')
#plt.plot(n[:-1], abs_error1, c='purple', alpha=0.7, lw=0.7, label=r'DIY z numpy $1/(N-n)$')
plt.ylabel(r"Absolutna napaka")
plt.xlabel("Vzorci")
plt.yscale("log")
plt.legend()


plt.subplot(3, 1, 3)
plt.axhline(y=1e-15, color='black', linestyle='--')
plt.plot(n[:-1], abs_error1, c='green',  alpha=1, lw=1, label=r'DIY brez numpy')
plt.ylabel(r"Absolutna napaka")
plt.xlabel("Vzorci")
plt.yscale("log")
plt.legend()

plt.tight_layout()
plt.show()


