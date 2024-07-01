from numpy import fft
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

n = 100 # Number of data points
dx = 5.0 # Sampling period (in meters)
x = dx*np.arange(0,n) # x coordinates
w1 = 100.0 # sine wavelength (meters) 
fx = np.sin(2*np.pi*x/w1)*np.sin(2*np.pi*x/w1+25) # signal
Fk = fft.fft(fx)/n # Fourier coefficients (divided by n)
nu = fft.fftfreq(n,dx) # Natural frequencies
Fk = fft.fftshift(Fk) # Shift zero freq to center
nu = fft.fftshift(nu) # Shift zero freq to center

plt.plot(x,fx.real,'r-o')

plt.show()
#input( "Press Enter to continue... " )


#plt.cla()
f, ax = plt.subplots(3,1,sharex=True)
# Plot Cosine terms
ax[0].plot(nu, np.real(Fk),color='b')
ax[0].set_ylabel(r'$Re[F_k]$', size = 'x-large')
# Plot Sine terms
ax[1].plot(nu, np.imag(Fk),color='r')
ax[1].set_ylabel(r'$Im[F_k]$', size = 'x-large')
# Plot spectral power
ax[2].plot(nu, np.absolute(Fk)**2,color='y')
ax[2].set_ylabel(r'$\vert F_k \vert ^2$', size = 'x-large')
ax[2].set_xlabel(r'$\widetilde{\nu}$', size = 'x-large')
f.suptitle("FFT")
plt.show()
input( "Press Enter to continue... " )
