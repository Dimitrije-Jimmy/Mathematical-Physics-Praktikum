import numpy as np
import matplotlib.pyplot as plt

import librosa

import os

directory_vmes = os.path.dirname(__file__)
directory = os.path.dirname(directory_vmes)+'\\Bach\\'
filename = 'Bach.44100.mp3'
file_path = directory + filename

# Convert the audio to a NumPy array
signal, fs_rate = librosa.load(file_path, sr=None)

# Apply the Fourier transform
fft_result = np.fft.fft(signal)

# Calculate the magnitude spectrum
spectrum_real = np.real(fft_result)
spectrum_imag = np.imag(fft_result)
spectrum_powr = np.absolute(fft_result)**2

# Plot the spectrum ____________________________________________________________
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.title("Spectrum of the Audio File")
plt.plot(signal)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel("Frequency")
plt.ylabel("Magnitude")


plt.subplot(3, 1, 2)
plt.title("Spectrum of the Audio File")
plt.plot(spectrum_real)
plt.plot(spectrum_imag)
plt.xscale('log')
#plt.yscale('log')
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.subplot(3, 1, 3)
plt.title("Spectrum of the Audio File")
plt.plot(spectrum_powr)
plt.xscale('log')
#plt.yscale('log')
plt.xlabel("Frequency")
plt.ylabel("Magnitude")


plt.tight_layout()
plt.show()