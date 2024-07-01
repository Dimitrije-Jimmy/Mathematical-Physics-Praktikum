import numpy as np
import matplotlib.pyplot as plt
import os


directory = os.path.dirname(__file__)+'\\Images\\'

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
    """Compute the inverse discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1)) # in essence row->column transformation, k*n is then the dot product..
    #print("Vector",k)
    #print("Matrix",k*n)
    M = 1/N*np.exp(2j * np.pi * k * n / N)
    #print("Matrix",M)
    return np.dot(M, x)

def dft_inverse(x):
    a = np.copy(x)
    output = []
    N = a.shape[0]
    for k in range(N):
        print(k)
        f_k = 0
        for n in range(N):
            f_k += 1/N*(a[n] * np.exp(2j * np.pi * k * n / N))
        output.append(f_k)

    return output

def Funkcija(x):
    t01 = 100.0 # sine t0 # spremeni na 110, da vidis ne-periodicnost!
    t02 = 10 # cosine t0 # aliasing n=200,T=2000, t01=100,t02=10 
    print("nu1 (sin)=", 1./t01)
    print("nu2 (cos)=", 1./t02)

    return (np.sin(2*np.pi*x/t01) + 2*np.cos(2*np.pi*x/t02)) # signal
    #return np.sin(2*np.pi*x/250) + np.cos(2*np.pi*x/0.08)


# Calculations _________________________________________________________________
#sampling parameters 200,100 = default
n = 200 # Number of data points
T = 100. # Sampling period
dt = T/n
tmin = 0.
tmax = dt*n
print("sampling freq:",1./dt)
nuc = 0.5/dt
print("critical freq:",nuc)


# Sampling time
#t = dt*np.arange(0,n) # x coordinates - OK
#t = np.linspace(0, T, n, endpoint=False) # OK (same)
#t=np.linspace(tmin,tmax,n) # non-periodic endpoint
#print(t)

t_time = np.linspace(-T, T, 1000, endpoint=False)
t = np.linspace(-T*2/3, T*2/3, n, endpoint=True)

nu = np.linspace(-nuc,nuc,n,endpoint=False)

ht = Funkcija(t)
ht2 = Funkcija(t_time)
dft = DFT(ht)
Hk=np.roll(dft,int(n/2))/n

#ht3 = iDFT(Hk)/n
ht3 = iDFT(dft)
ht3 = dft_inverse(dft)

# Plotting _____________________________________________________________________

# Začetno stanje in sampling
# plot the functions
plt.plot(t_time, ht2, 'b-', lw=0.5, label='Originalna funkcija')
plt.scatter(t, ht, c='red', label='Vzorčen delež funkcije')
plt.xlabel("t")
plt.ylabel("h(t)")
#plt.title(r'h(t) = sin($\omega_1$t)+2cos($\omega_2$t)')
plt.title(r'h(t) = sin($\omega_1$t)+cos($\omega_3$t)')
plt.legend()
plt.savefig(directory+f'start4_{n}.{T}.png', dpi=300)
plt.show()


# Realni in Imaginarni deli
plt.figure(figsize=(10, 8))

# Plot Cosine terms
plt.subplot(3, 1, 1)
plt.plot(nu, np.real(Hk), color='g', label='Realni del')
plt.xlabel(r'$\nu$')
plt.ylabel(r'$Re[H_k]$')
plt.legend()
plt.title("Fourierova transformacija funkcije")

# Plot Sine terms
plt.subplot(3, 1, 2)
plt.plot(nu, np.imag(Hk), color='c', label='Imaginarni del')
plt.xlabel(r'$\nu$')
plt.ylabel(r'$Im[H_k]$')
plt.legend()

# Plot spectral power
plt.subplot(3, 1, 3)
plt.plot(nu, np.absolute(Hk)**2, color='y', label='Moč')
plt.xlabel(r'$\nu$')
plt.ylabel(r'$\vert H_k \vert ^2$')
plt.legend()

plt.tight_layout()
plt.savefig(directory+f'pwr4_{n}.{T}.png', dpi=300)
plt.show()


# Inverse TDF
plt.plot(t_time, ht2, 'b-', lw=0.5, label='Originalna funkcija')
plt.plot(t, ht3, 'r--', label='Inverzna funkcija')
plt.xlabel("t")
plt.ylabel("h(t)")
plt.title(r'Inverz Fourierove transformacije')
plt.legend()

plt.savefig(directory+f'inverse4_{n}.{T}.png', dpi=300)
plt.show()