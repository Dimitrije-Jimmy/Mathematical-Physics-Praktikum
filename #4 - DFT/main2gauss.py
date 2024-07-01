import numpy as np
import matplotlib.pyplot as plt
import os

#Hk = np.fft.fft(hk)
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

def iDFT(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1)) # in essence row->column transformation, k*n is then the dot product..
    #print("Vector",k)
    #print("Matrix",k*n)
    M = np.exp(2j * np.pi * k * n / N)
    #print("Matrix",M)
    return np.dot(M, x)/N

def Gauss(x, mu, sig):
    return 1/(np.sqrt(2*np.pi)*sig) * np.exp(-(x - mu)**2 / (2*sig**2))


# Plotting _____________________________________________________________________

n = 10000
t = np.linspace(-50, 50, n, endpoint=False)
#ht = np.exp(-t**2 / 2)
hk = Gauss(t, 0, 0.68)
ht = np.roll(hk, int(n/2))

# plot the functions
plt.plot(t, hk,'b-', label='neperiodična')
plt.plot(t, ht,'r-', label='periodična')
plt.xlabel("t")
plt.ylabel("h(t)")
#plt.title(r'h(t)=\sin($\omega_1$t)+2cos($\omega_2$t)')
#plt.title(r'h(t) = sin($\omega_1$t)+2cos($\omega_2$t)')
plt.title(r'Gaussova porazdelitev, $\sigma = 0.68$')

plt.legend()
plt.show()
#input( "Press Enter to continue... " )


# Frequencies
#nu = np.linspace(-nuc, nuc, n, endpoint=False)


#ht = np.roll(ht, int(n/2))
dft = DFT(ht)


#Hk=np.roll(dft,int(n/2))/n

Hk2 = np.fft.fft(hk)
Hk = dft
nu = np.fft.fftfreq(len(Hk), d=t[1] - t[0])
#fk=dft

plt.figure(figsize=(10, 8))

# Plot Cosine terms
plt.subplot(3, 1, 1)
plt.plot(nu, np.real(Hk2), color='g', label='nepopravljen signal')
plt.plot(nu, np.real(Hk), color='b', label='vnaprej popravljen signal')
plt.ylabel(r'$Re[H_k]$')#, size = 'x-large')
plt.legend()
plt.title("Fourierova transformacija Gaussove funkcije")

# Plot Sine terms
plt.subplot(3, 1, 2)
plt.plot(nu, np.imag(Hk2), color='c')
plt.plot(nu, np.imag(Hk), color='r')
plt.ylabel(r'$Im[H_k]$')#, size = 'x-large')

# Plot spectral power
plt.subplot(3, 1, 3)
plt.plot(nu, np.absolute(Hk)**2, color='y')
plt.ylabel(r'$\vert H_k \vert ^2$')#, size = 'x-large')
plt.xlabel(r'$\nu$')#, size = 'x-large')
#plt.legend()

plt.tight_layout()
plt.show()


"""
# Spenko
f = np.fromfile(fname, dtype = float)
n = len(f)
delta = n/fsampling
fc = fsampling/2.

x = np.linspace(0, delta - (delta/n), n)
freq = np.linspace(-fc, fc, n)

fft2 = np.fft.ifftshift(np.fft.fft(f))
plt.figure()
plt.title("Realna komponenta DFT")
plt.ylabel("Re|F($\omega$)|")
plt.xlabel("$\omega$")
plt.plot(freq, (fft2).real)
plt.figure()
plt.title("Imaginarna komponenta DFT")
plt.ylabel("Im|F($\omega$)|")
plt.xlabel("$\omega$")
plt.plot(freq, (fft2).imag)
plt.figure()
plt.title("Absolutna vrednost DFT")
plt.ylabel("|F($\omega$)|")
plt.xlabel("$\omega$")
plt.plot(freq, np.absolute(fft2))
plt.figure()
plt.title("Absolutna vrednost DFT (0, fc)")
plt.ylabel("|F($\omega$)|")
plt.xlabel("$\omega$")
plt.plot(freq[math.ceil(n/2):], np.absolute(fft2)[math.ceil(n/2):])

#spekter nazaj shiftamo pred ifft
ifft1 = np.fft.ifft(np.fft.fftshift(fft2))
plt.figure()
plt.title(fname + " po IDFT")
plt.plot(x, (ifft1).real)
#plt.figure()
#plt.plot(x, (ifft1).imag)

plt.show()
"""