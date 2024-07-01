import numpy as np
import matplotlib.pyplot as plt

import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\'
print(dir_path)

file_phantom = dir_path+'phantom.dat'
file_sg1 = dir_path+'sg1.dat'
file_sg2 = dir_path+'sg2.dat'


# Importing data into matrix
def zac_matrika(filename):
    matrika = np.loadtxt(filename).T
    return np.flipud(matrika)

# Function Rk0
"""
\tilde{R_k0}(k) = |k|\theta(k0 - |k|)
\theta = Heavyside

-> FFT or analytic
R_k0(s) = {1\over2\pi} [2k0/s \sin(k0*s) - 4/s^2 \sin^2(k0s/2)]

preveri za k0 = 5
"""
def Rfft(ssez, ksez, k0):
    theta = np.heaviside(k0 - np.abs(ksez), 0.5)
    R_kspace = np.abs(ksez)*theta
    """
    #R_sspace = np.fft.fftshift(R_kspace, axes=0)
    R_sspace = np.fft.fft(R_kspace)
    #R_sspace /= np.max(R_sspace)
    R_sspace = np.roll(R_sspace, int(len(R_sspace)/2))
    R_sspace = np.imag(R_sspace)
    #R_sspace = np.fft.fftshift(R_kspace, axes=0)
    """
    R_sspace = np.fft.ifft(R_kspace)
    R_sspace = np.fft.fftshift(R_sspace)

    # Normalize the result based on the amplitude of the analytical solution
    normalization_factor = np.max(np.abs(R_sspace))
    #R_sspace /= normalization_factor
    
    R_sspace = np.real(R_sspace)
    return R_sspace, R_kspace


def Ranalyt(ssez, k0):
    factor1 = 2*k0/ssez * np.sin(k0*ssez)
    factor2 = (2/ssez * np.sin(k0*ssez/2))**2
    return (factor1 - factor2) / (2*np.pi)
    

def Rmatrix(pmat, x, y):
    ns = len(pmat[0])
    nf = len(pmat)
    ds = 2/ns
    df = np.pi/nf
    
    #x = np.arange(-1, 1, ds)
    #y = np.arange(-1, 1, ds)
    #i = np.linspace(0, nf, nf)
    #s = x*
    
    
    fi = np.arange(0, 1, df)
    kmax = (ns*np.pi)/2
    k0 = kmax/2
    
    R = np.zeros_like(pmat)
    i = np.linspace(0, np.pi, nf)
    j = np.linspace(0, 2, ns)
    s = x*np.cos(i) + y*np.sin(i)
    R[:, :] = Ranalyt(s + 1 - j, k0)
    
    #for i in range(len(R)):
    #    s = x*np.cos(i*df) + y*np.sin(i*df)
    #    R[i, :] = Ranalyt(s + 1 - np.arange(0, 2, ds), k0)
        #for j, ps in enumerate(pf):
        #    R[i, j] = Ranalyt(s + 1 - j*ds, k0)
            
    return R
            
def fxy(pmat, rmat):
    ns = len(pmat[0])
    nf = len(pmat)
    print(ns, nf)
    #R = Rmatrix(pmat)
    R = rmat
    fval = np.matmul(pmat, R)
    fval = np.nan_to_num(fval, nan=0.0)
    print(fval)
    fval = np.trace(fval) / (ns*nf)
    print(fval)
    print('fval', fval.shape)
    return fval    


def fxy_manual(pmat, xsez, ysez):
    ns = len(pmat[0])
    nf = len(pmat)
    ds = 2/ns
    df = np.pi/nf
    kmax = (ns*np.pi)/2
    k0 = kmax/2
    
    fmat = np.zeros((len(xsez), len(ysez)))
    for m, x in enumerate(xsez):
        for n, y in enumerate(ysez):
            print(m, n)
            fval = 0
            for i in range(nf):
                s = x*np.cos(i*df) + y*np.sin(i*df)
                for j in range(ns):
                    R = Ranalyt(s + 1 - j*ds, k0)
                    fval += pmat[i, j]*R

            fmat[m, n] = fval

"""    
def fxy_vectorized(pmat, xsez, ysez):
    ns, nf = pmat.shape
    ds = 2 / ns
    df = np.pi / nf
    kmax = (ns * np.pi) / 2
    k0 = kmax / 2

    x, y = np.meshgrid(xsez, ysez, indexing='ij')
    df_values = np.arange(nf) * df
    ds_values = np.arange(ns) * ds

    s_values = x * np.cos(df_values) + y * np.sin(df_values)
    R_values = Ranalyt(s_values[:, :, None] + 1 - ds_values, k0)

    fmat = np.sum(pmat[:, :, None] * R_values, axis=(0, 1)) / (ns * nf)

    return fmat

def fxy_vectorized2(pmat, xsez, ysez):
    ns = len(pmat[0])
    nf = len(pmat)
    ds = 2 / ns
    df = np.pi / nf
    kmax = (ns * np.pi) / 2
    k0 = kmax / 2

    x, y = np.meshgrid(xsez, ysez, indexing='ij')
    df_values = np.arange(nf) * df
    ds_values = np.arange(ns) * ds

    s_values = x * np.cos(df_values) + y * np.sin(df_values)
    R_values = Ranalyt(s_values[:, :, None] + 1 - ds_values, k0)

    fmat = np.sum(pmat[:, :, None, None] * R_values, axis=(0, 1)) / (ns * nf)

    return fmat 


def calculate_R_values(x, y, df_values, ds_values, k0):
    s_values = x * np.cos(df_values) + y * np.sin(df_values)
    R_values = Ranalyt(s_values[:, :, None] + 1 - ds_values, k0)
    return R_values


def fxy_vectorized(pmat, x, y):
    ns, nf = pmat.shape
    ds = 2 / ns
    df = np.pi / nf
    kmax = (ns * np.pi) / 2
    k0 = kmax / 2

    #x, y = np.meshgrid(xsez, ysez, indexing='ij')
    df_values = np.arange(nf) * df
    ds_values = np.arange(ns) * ds

    #R_values = calculate_R_values(x, y, df_values, ds_values, k0)
    s_values = x * np.cos(df_values) + y * np.sin(df_values)
    R_values = Ranalyt(s_values + 1 - ds_values, k0)
    print(R_values.shape)
    tval = np.matmul(pmat, R_values)
    print(tval.shape)
    return np.trace(tval) / (ns*nf)
    
    
    
pmat = zac_matrika(file_phantom)  
ns = len(pmat[0])
nf = len(pmat)
ds = 2/ns
df = np.pi/nf
xsez = np.arange(-1, 1, ds)
ysez = np.arange(-1, 1, ds) 
fmat = np.zeros((len(xsez), len(ysez)))
for i, x in enumerate(xsez):
    for j, y in enumerate(xsez):
        print(i, j)
        fmat[i, j] = fxy_vectorized(pmat, x, y) 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
ax1.imshow(pmat, cmap='gray', extent=[0, np.pi, -1, 1])#, aspect='auto', origin='lower')
ax2.imshow(fmat, cmap='gray')#, extent=[-1, 1, -1, 1])
"""
#import sys
#sys.exit()

if __name__ == "__main__":
    
    k0 = 5.
    n = 1000
    ksez = np.linspace(-10., 10., n)
    ssez = np.linspace(-10., 10., n)
    
    Values_fft_sspace, Values_fft_kspace = Rfft(ssez, ksez, k0)
    Values_analyt = Ranalyt(ssez, k0)
    
    #print(np.abs(Values_fft_sspace - Values_analyt))
    
    # dejanska implementacija filtra:
    a = k0
    w = ksez

    rn1 = np.abs(2/a * np.sin(a * w/2))
    rn2 = np.sin(a * w/2)
    rd = (a * w)/2
    r = rn1 * (rn2/rd)**2

    implemented_filter = np.fft.fftshift(r)
    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(ksez, Values_fft_kspace)
    ax1.set_xlabel(r'$k$')
    ax1.set_ylabel(r'$\tilde{R}_{k_0}(k)$')
    ax1.set_title(r'filter v $k$ - frekvenčnem prostoru')
    
    ax2.plot(ssez, Values_fft_sspace)
    ax2.set_xlabel(r'$s$')
    ax2.set_ylabel(r'$R_{k_0}(s)$')
    ax2.set_title(r'IFFT filtra')
    
    ax3.plot(ssez, Values_analyt)
    ax3.set_xlabel(r'$s$')
    ax3.set_ylabel(r'$R_{k_0 \;, analytic}(s)$')
    ax3.set_title(r'Analitična rešitev filtra')
    
    ax4.plot(ssez, implemented_filter)
    ax4.set_xlabel(r'$s$')
    ax4.set_ylabel(r'SINC')
    ax4.set_title(r'Implementiran SINC filter')
    
    #fig.suptitle('')
    plt.tight_layout()
    
    plt.show()
    plt.clf()
    plt.close()
    
    
    import sys
    sys.exit()
    # Calculation ______________________________________________________
    pmat = zac_matrika(file_phantom)
    ns = len(pmat[0])
    nf = len(pmat)
    ds = 2/ns
    df = np.pi/nf
    xsez = np.arange(-1, 1, ds)
    ysez = np.arange(-1, 1, ds)
    
    #Rmatrika = Rmatrix(pmat, xsez, ysez)
    #projekcija = fxy(pmat, Rmatrika)
    projekcija = fxy_manual(pmat, xsez, ysez)
    #projekcija = fxy_vectorized2(pmat, xsez, ysez)
    
    print(pmat.shape)
    #print(Rmatrika.shape)
    print(projekcija.shape)
    
    
    import sys
    #sys.exit()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    ax1.imshow(pmat, cmap='gray', extent=[0, np.pi, -1, 1])#, aspect='auto', origin='lower')
    #ax2.imshow(Rmatrika, cmap='gray')
    ax3.imshow(projekcija, cmap='gray')#, extent=[-1, 1, -1, 1])
    
    
    plt.show()