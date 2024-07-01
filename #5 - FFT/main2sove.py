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

# Convert the audio to a NumPy array
signal_bubomono, fs_rate_bubomono = librosa.load(directory+'bubomono.wav', sr=None)
signal_bubo2mono, fs_rate_bubo2mono = librosa.load(directory+'bubo2mono.wav', sr=None)
signal_mix, fs_rate_mix = librosa.load(directory+'mix.wav', sr=None)
signal_mix1, fs_rate_mix1 = librosa.load(directory+'mix1.wav', sr=None)
signal_mix2, fs_rate_mix2 = librosa.load(directory+'mix2.wav', sr=None)
signal_mix22, fs_rate_mix22 = librosa.load(directory+'mix22.wav', sr=None)

print(fs_rate_bubomono)


y1 = signal_mix2[:-1]#[:-1]
y2 = signal_bubomono[:-1]
y3 = signal_bubo2mono
xdata, data = correlate(y1, y2)
#x1 = np.linspace(0, len(y1), fs_rate_mix2)
#x2 = np.linspace(0, len(y2), fs_rate_bubomono)
print(len(y1))
x1 = np.arange(0, len(y1))/44100  # Delim z sampling frequencyjem, da dobim x-os v sekundha
x2 = np.arange(0, len(y3))/44100 

#FF5733 (Coral)
#33FF57 (Lime Green)
#3344FF (Royal Blue)

# Tom - bubomono
# Jerry - bubo2mono

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(x1, y1, c="#3344FF")
plt.xlabel("Čas [s]")
plt.ylabel("Normirana amplituda")
plt.title("Signal mix")


plt.subplot(3, 1, 2)
plt.plot(x2, y2, c="#FF5733")
plt.xlabel("Čas [s]")
plt.ylabel("Normirana amplituda")
plt.title("Jerry")


plt.subplot(3, 1, 3)
#x3 = np.linspace(0, len(x1)/44100, len(data))
plt.plot(data, c="#33FF57")
plt.xlabel("Število vzorcev")
plt.ylabel("Normirana amplituda")
plt.title("Korelacijska funkcija obeh signalov")


plt.tight_layout()
plt.show()


import sys
sys.exit()
# Spekter frekvenc _____________________________________________________________

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.title(f"Frekvenčni spekter")
plt.plot(y1)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"t")
plt.ylabel("signal")

plt.subplot(2, 1, 2)
plt.title(f"Frekvenčni spekter")
plt.plot(y2)
plt.plot(y3)
plt.xlabel(r"t")
plt.ylabel("signal")

plt.tight_layout()
plt.show()
#plt.close()
plt.clf()

signal1 = y1
signal1 = np.roll(signal1, int(len(signal1)/2))
fft_result1 = np.fft.fft(signal1)
fft_result1 = np.roll(fft_result1, -int(len(fft_result1)/2))
# Calculate the magnitude spectrum
spectrum_real1 = np.real(fft_result1)
spectrum_imag1 = np.imag(fft_result1)
spectrum_powr1 = np.absolute(fft_result1)**2

signal2 = y2
signal2 = np.roll(signal2, int(len(signal2)/2))
fft_result2 = np.fft.fft(signal2)
fft_result2 = np.roll(fft_result2, -int(len(fft_result2)/2))
# Calculate the magnitude spectrum
spectrum_real2 = np.real(fft_result2)
spectrum_imag2 = np.imag(fft_result2)
spectrum_powr2 = np.absolute(fft_result2)**2

signal3 = y3
signal3 = np.roll(signal3, int(len(signal3)/2))
fft_result3 = np.fft.fft(signal3)
fft_result3 = np.roll(fft_result3, -int(len(fft_result3)/2))
# Calculate the magnitude spectrum
spectrum_real3 = np.real(fft_result3)
spectrum_imag3 = np.imag(fft_result3)
spectrum_powr3 = np.absolute(fft_result3)**2

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title(f"Frekvenčni spekter")
plt.plot(spectrum_real1)
plt.plot(spectrum_imag1)
plt.plot(spectrum_powr1, alpha=0.3, lw=0.3)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"t")
plt.ylabel("signal")

plt.subplot(3, 1, 2)
plt.title(f"Frekvenčni spekter")
plt.plot(spectrum_real2)
plt.plot(spectrum_imag2)
plt.plot(spectrum_powr2, alpha=0.3, lw=0.3)
plt.xlabel(r"t")
plt.ylabel("signal")

plt.subplot(3, 1, 3)
plt.title(f"Frekvenčni spekter")
plt.plot(spectrum_real3)
plt.plot(spectrum_imag3)
plt.plot(spectrum_powr3, alpha=0.3, lw=0.3)
plt.xlabel(r"t")
plt.ylabel("signal")

plt.tight_layout()
plt.show()
plt.close()
plt.clf()