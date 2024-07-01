#isort
#manim sideview
#pylance
#python
import numpy as np
import matplotlib.pyplot as plt

# importing datetime module for now()
import datetime
import os
 
# using now() to get current time
current_time = str(datetime.datetime.now())
print(current_time)

current_time_fixed = current_time.split('.')[0]
current_time_fixed = current_time_fixed.replace('-', '_').replace(':', '.')


directory = os.path.dirname(__file__)+'\\Images\\'
file_path = directory + f'{current_time_fixed}.png'
print(file_path)

#plt.ion()

# Functions ____________________________________________________________________
def DFT_matrix(x):
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

def DFT_simplest(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    F=[]
    for k in range(N):
            fk=0.
            for n in range(N):
                Mkn=np.exp(-2j * np.pi * k * n / N)
                fk += Mkn*x[n]
            F.append(fk)
            #print("k,F[k]",k,fk)
    return np.asarray(F)


def funkcija(t):
    return np.sin(2 * np.pi / 100 * t) + np.cos(2 * np.pi / 0.08 * t)


# Execution ____________________________________________________________________
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

t = np.linspace(-T/2, T/2, n, endpoint=False)

t01 = 100.0 # sine t0 # spremeni na 110, da vidis ne-periodicnost!
t02 = 10 # cosine t0 # aliasing n=200,T=2000, t01=100,t02=10 
print("nu1 (sin)=", 1./t01)
print("nu2 (cos)=", 1./t02)

ht = np.sin(2*np.pi*t/t01) + 2*np.cos(2*np.pi*t/t02) # signal

#ht = np.sin(2*np.pi*t/t01)
#ht = np.cos(2*np.pi*x/t02)

#ht = Gauss(t, 0, 1)


# plot the functions
plt.plot(t,ht,'r.-')
plt.xlabel("t")
plt.ylabel("h(t)")
#plt.title(r'h(t)=\sin($\omega_1$t)+2cos($\omega_2$t)')
plt.title(r'h(t) = sin($\omega_1$t)+2cos($\omega_2$t)')

#plt.show()
plt.clf()
#input( "Press Enter to continue... " )

# Frequencies
nu = np.linspace(-nuc,nuc,n,endpoint=False)

dft=DFT_matrix(ht)
#dft=DFT_simplest(fx)

Hk=np.roll(dft,int(n/2))/n
#fk=dft

#plt.cla()
f, ax = plt.subplots(3,1,sharex=True)
# Plot Cosine terms
ax[0].plot(nu, np.real(Hk),color='b')
ax[0].set_ylabel(r'$Re[H_k]$', size = 'x-large')
# Plot Sine terms
ax[1].plot(nu, np.imag(Hk),color='r')
ax[1].set_ylabel(r'$Im[H_k]$', size = 'x-large')
# Plot spectral power
ax[2].plot(nu, np.absolute(Hk)**2,color='y')
ax[2].set_ylabel(r'$\vert H_k \vert ^2$', size = 'x-large')
ax[2].set_xlabel(r'$\nu$', size = 'x-large')
f.suptitle("FT[cos(t)]")
#plt.show()
plt.clf()


# Time, CPU and RAM ____________________________________________________________
import time
import psutil

sample_points = [800, 2200, 11000, 22000, 35000, 44100]  # Varying number of points
#sample_points = np.arange(100, 44100, 4000)  # Varying number of points
#sample_points = np.arange(100, 2000, 100)  # Varying number of points


dft_simplest_time = []
dft_simplest_cpu = []
dft_simplest_ram = []
dft_matrix_time = []
dft_matrix_cpu = []
dft_matrix_ram = []
numpy_fft_time = []
numpy_fft_cpu = []
numpy_fft_ram = []
for N in sample_points:
    print(N)
    t = np.linspace(0, 400, N, endpoint=False)  # Keep the same range, vary the number of points
    #s_t = np.sin(2 * np.pi / 100 * t) + np.cos(2 * np.pi / 0.08 * t)
    s_t = funkcija(t)
    
    # DFT Simplest _____________________________________________________________
    # Monitor CPU and RAM usage before the operation
    cpu_before = psutil.cpu_percent()
    ram_before = psutil.virtual_memory().used
    
    start_time = time.time()
    S_f = DFT_simplest(s_t)
    dft__time = time.time() - start_time
    
    # Monitor CPU and RAM usage after the operation
    cpu_after = psutil.cpu_percent()
    ram_after = psutil.virtual_memory().used
    

    dft_simplest_time.append(dft__time)
    dft_simplest_cpu.append(cpu_after - cpu_before)
    dft_simplest_ram.append(ram_after - ram_before)


    # DFT Matrix _______________________________________________________________
    # Monitor CPU and RAM usage before the operation
    cpu_before = psutil.cpu_percent()
    ram_before = psutil.virtual_memory().used
    
    start_time = time.time()
    S_f = DFT_matrix(s_t)
    dft__time = time.time() - start_time
    
    # Monitor CPU and RAM usage after the operation
    cpu_after = psutil.cpu_percent()
    ram_after = psutil.virtual_memory().used
    

    dft_matrix_time.append(dft__time)
    dft_matrix_cpu.append(cpu_after - cpu_before)
    dft_matrix_ram.append(ram_after - ram_before)


    # Numpy FFT ________________________________________________________________
    cpu_before = psutil.cpu_percent()
    ram_before = psutil.virtual_memory().used

    start_time = time.time()
    S_f_np = np.fft.fft(s_t)
    fft__time = time.time() - start_time
    
    cpu_after = psutil.cpu_percent()
    ram_after = psutil.virtual_memory().used
    
    numpy_fft_time.append(fft__time)
    numpy_fft_cpu.append(cpu_after - cpu_before)
    numpy_fft_ram.append(ram_after - ram_before)


# Plot the CPU and RAM usage vs. the number of sample points
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.scatter(sample_points, dft_simplest_time, c='blue', label="DFT simplest")
plt.plot(sample_points, dft_simplest_time, c='blue', alpha=0.7, lw=0.5)
plt.scatter(sample_points, dft_matrix_time, c='green', label="DFT matrix")
plt.plot(sample_points, dft_matrix_time, c='green', alpha=0.7, lw=0.5)
plt.scatter(sample_points, numpy_fft_time, c='red', label="NumPy FFT")
plt.plot(sample_points, numpy_fft_time, c='red', alpha=0.7, lw=0.5)
plt.xlabel("Number of Sample Points")
plt.ylabel(r"time $t$ [s]")
plt.legend()

plt.subplot(3, 1, 2)
plt.scatter(sample_points, dft_simplest_cpu, c='blue', label="DFT simplest")
plt.plot(sample_points, dft_simplest_cpu, c='blue', alpha=0.7, lw=0.5)
plt.scatter(sample_points, dft_matrix_cpu, c='green', label="DFT matrix")
plt.plot(sample_points, dft_matrix_cpu, c='green', alpha=0.7, lw=0.5)
plt.scatter(sample_points, numpy_fft_cpu, c='red', label="NumPy FFT")
plt.plot(sample_points, numpy_fft_cpu, c='red', alpha=0.7, lw=0.5)
plt.xlabel("Number of Sample Points")
plt.ylabel("CPU Usage [%]")
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(sample_points, dft_simplest_ram, c='blue', label="DFT simplest")
plt.plot(sample_points, dft_simplest_ram, c='blue', alpha=0.7, lw=0.5)
plt.scatter(sample_points, dft_matrix_ram, c='green', label="DFT matrix")
plt.plot(sample_points, dft_matrix_ram, c='green', alpha=0.7, lw=0.5)
plt.scatter(sample_points, numpy_fft_ram, c='red', label="NumPy FFT")
plt.plot(sample_points, numpy_fft_ram, c='red', alpha=0.7, lw=0.5)
plt.xlabel("Number of Sample Points")
plt.ylabel("RAM Usage [bytes]")
plt.legend()


plt.tight_layout()
plt.savefig(file_path, dpi=300)
plt.show()
