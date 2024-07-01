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

def Ranalyt(ssez, k0):
    factor1 = 2*k0/ssez * np.sin(k0*ssez)
    factor2 = (2/ssez * np.sin(k0*ssez/2))**2
    return (factor1 - factor2) / (2*np.pi)


def Rmatrix(pmat, x, y):
    ns, nf = pmat.shape
    ds = 2/ns
    df = np.pi/nf    
    
    #fi = np.arange(0, 1, df)
    kmax = (ns*np.pi)/2
    k0 = kmax/2
    
    R = np.zeros_like(pmat)
    i = np.linspace(0, np.pi, nf)
    j = np.linspace(0, 2, ns)
    s = x*np.cos(i) + y*np.sin(i)
    R[:, :] = Ranalyt(s + 1 - j, k0)
    
    return R
    

# Calculations _________________________________________________________
pmat = zac_matrika(file_phantom)  
ns = len(pmat[0])
nf = len(pmat)
ds = 2/ns
df = np.pi/nf
xsez = np.arange(-1, 1, ds)
ysez = np.arange(-1, 1, ds) 


def Fmatrix(pmat, xsez, ysez):
    fmat = np.zeros((len(xsez), len(ysez)))
    for i, x in enumerate(xsez):
        for j, y in enumerate(ysez):
            print(i, j)
            R = Rmatrix(pmat, x, y)
            val = np.matmul(pmat, R.T)
            val = np.trace(val) / (ns * nf)
            fmat[i, j] = val#/np.max(val)
    
    return fmat

def Fmatrix2(pmat, xsez, ysez):
    ns, nf = pmat.shape
    
    # Vectorized calculation of Rmat
    i, j = np.meshgrid(np.linspace(0, np.pi, nf), np.linspace(0, 2, ns), indexing='ij')
    #s_values = np.outer(xsez, np.cos(i)) + np.outer(ysez, np.sin(i))
    s_values = np.outer(xsez, np.cos(i))[:, :, None] + np.outer(ysez, np.sin(i))[:, None, :]
    Rmat = Ranalyt(s_values + 1 - j, (ns * np.pi) / 4)

    # Calculate fmat using vectorized operations
    fmat = np.einsum('ij,klj->ik', pmat, Rmat) / (ns * nf)

    return fmat

fmat = Fmatrix1(pmat, xsez, ysez)            
            

# Plotting _____________________________________________________________
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
ax1.imshow(pmat, cmap='gray', extent=[0, np.pi, -1, 1])#, aspect='auto', origin='lower')
ax2.imshow(fmat, cmap='gray')#, extent=[-1, 1, -1, 1])

plt.show()