import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

#frekvenca = 44100
#frekvenca = 11025
#frekvenca = 5512
#frekvenca = 2756
#frekvenca = 1378
frekvenca = 882
directory = os.path.dirname(__file__)+'\\Bach\\'
filename = f'Bach.{frekvenca}.mp3'
file_path = directory + filename


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

# Convert the audio to a NumPy array
signal, fs_rate = librosa.load(file_path, sr=None)
signal2 = np.roll(signal, int(len(signal)/2))

# Apply the Fourier transform
fft_result1 = np.fft.fft(signal)
#fft_result = DFT(signal)
fft_result = np.roll(fft_result1, -int(len(fft_result1)/2))

# Calculate the magnitude spectrum
spectrum_real = np.real(fft_result)
spectrum_imag = np.imag(fft_result)
spectrum_powr = np.absolute(fft_result)**2

# Plot the spectrum ____________________________________________________________
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.title(f"Spekter frekvenci vzorčenja {frekvenca}Hz")
plt.plot(signal)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r"t")
plt.ylabel("signal")

plt.subplot(3, 1, 2)
plt.title("Realni in imaginarni del")
plt.plot(spectrum_real, 'b-', label='real')
plt.plot(spectrum_imag, 'c-', label='imag')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'$\nu$')
plt.ylabel(r'$Re[H_k]$')
plt.legend()

plt.subplot(3, 1, 3)
plt.title("Moč vzorcev")
plt.plot(spectrum_powr)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(r'$\nu$')
plt.ylabel(r'$\vert H_k \vert ^2$')


plt.tight_layout()
plt.show()
plt.close()
plt.clf()


import sys
sys.exit()
# Vse skup _____________________________________________________________________

seznam_frekvenc = [44100, 11025, 5512, 2756, 1378, 882]
barve = plt.get_cmap('viridis')(np.linspace(0, 1, len(seznam_frekvenc)))

spectrum_sign = []
spectrum_real = []
spectrum_imag = []
spectrum_powr = []
for i, frekvenca in enumerate(seznam_frekvenc):
    # Convert the audio to a NumPy array
    signal, fs_rate = librosa.load(directory+f'Bach.{frekvenca}.mp3', sr=None)
    #signal2 = np.roll(signal, int(len(signal)/2))

    # Apply the Fourier transform
    fft_result1 = np.fft.fft(signal)
    #fft_result = DFT(signal)
    fft_result = np.roll(fft_result1, int(len(fft_result1)/2))

    # Calculate the magnitude spectrum
    spectrum_sign.append(signal)
    spectrum_real.append(np.real(fft_result))
    spectrum_imag.append(np.imag(fft_result))
    spectrum_powr.append(np.absolute(fft_result)**2)


for i, frekvenca in enumerate(seznam_frekvenc):
    plt.plot(spectrum_sign[i], color=barve[i])
plt.title(f"Spekter frekvenc vzorčenja")
plt.xlabel(r"t")
plt.ylabel("signal")
plt.tight_layout()
plt.savefig(directory+'signal_all.png', dpi=300)
plt.show()


for i, frekvenca in enumerate(seznam_frekvenc):
    plt.plot(spectrum_real[i], color=barve[i])
plt.title(f"Realni deli")
plt.xlabel(r'$\nu$')
plt.ylabel(r'$Re[H_k]$')
plt.tight_layout()
plt.savefig(directory+'real_all.png', dpi=300)
plt.show()


for i, frekvenca in enumerate(seznam_frekvenc):
    plt.plot(spectrum_powr[i], color=barve[i])
plt.title(f"Moč vzorcev")
plt.xlabel(r'$\nu$')
plt.ylabel(r'$\vert H_k \vert ^2$')
plt.tight_layout()
plt.savefig(directory+'powr_all.png', dpi=300)
plt.show()


