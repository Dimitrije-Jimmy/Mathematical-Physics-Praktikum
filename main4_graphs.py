import numpy as np
import matplotlib.pyplot as plt

from main1_functions import *

import os
dir_path = os.path.dirname(os.path.realpath(__file__))+'\\Images\\'
print(dir_path)

# Functions ____________________________________________________________________

# Plotanje testnih funkcij X Y in R PHI ________________________________
def singleZ_plot(m, n):
    global R, PHI, X, Y
    
    Z = testne(m, n, R, PHI)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    q1 = ax1.contourf(X, Y, Z, cmap='jet')
    fig.colorbar(q1, ax=ax1)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_title(f"$m$ = {m}, $n$ = {n}")
    
    plt.tight_layout()
    
    q2 = ax2.contourf(R, PHI, Z, cmap='jet')
    fig.colorbar(q2, ax=ax2)
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$\phi$")
    ax2.set_title(f"$m$ = {m}, $n$ = {n}")
    
    plt.tight_layout()
    
    plt.savefig(dir_path+f'testne_m{m}n{n}.png', dpi=300)
    """
    plt.show()
    plt.clf()
    plt.close()
    """

# Plotanje testnih z pcolormesh in imshow ______________________________
def singleZ_plot2(m, n):
    global R, PHI, X, Y
    
    Z = testne(m, n, R, PHI)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    q1 = ax1.pcolormesh(X, Y, Z, cmap='jet')#, shading='auto')
    fig.colorbar(q1, ax=ax1)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_title(f"$m$ = {m}, $n$ = {n}")
    
    plt.tight_layout()
    
    q2 = ax2.imshow(Z, cmap='jet', extent=(R.min(), R.max(), PHI.min(), PHI.max()), origin='lower')
    #q2 = ax2.pcolormesh(X, Y, Z, cmap='jet', shading='auto')
    fig.colorbar(q2, ax=ax2)
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$\phi$")
    ax2.set_title(f"$m$ = {m}, $n$ = {n}")
    
    plt.tight_layout()
    
    #plt.savefig(dir_path+f'testne_m{m}n{n}.png', dpi=300)
    """
    plt.show()
    plt.clf()
    plt.close()
    """
    
# plot 3D testnih ______________________________________________________
def singleZ_3Dplot(m, n):
    global R, PHI, X, Y
    
    Z = testne(m, n, R, PHI)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot the surface
    surf1 = ax1.plot_surface(X, Y, Z, cmap='jet')#, edgecolor='k')

    # Customize the plot
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_zlabel(r"testna funkcija")
    ax1.set_title(f"$m$ = {m}, $n$ = {n}")

    # Add colorbar for reference
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=15)
    
    
    # Plot the surface
    surf2 = ax2.plot_surface(R, PHI, Z, cmap='jet')#, edgecolor='k')

    # Customize the plot
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$\phi$")
    ax2.set_zlabel(r"testna funkcija")
    ax2.set_title(f"$m$ = {m}, $n$ = {n}")

    # Add colorbar for reference
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=15)
    
    # Show the plot
    plt.tight_layout()
    #plt.show()
    

# Plot single Galerkin _________________________________________________
def singleGalerkin_plot(M_max, N_max, N):
    global R, PHI, X, Y

    C, a = galerkin(M_max, N_max)
    
    Z = np.zeros((N, N))
    for m in range(M_max + 1):
        for n in range(1, N_max+1):
            print(f"$m$ = {m}, $n$ = {n}")
            #Z += a[m*N_max + n]*testne(m, n+1, R, PHI)
            Z += a[m*N_max + n-1]*testne(m, n, R, PHI)
            
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    q1 = ax1.contourf(X, Y, Z, cmap='jet')
    fig.colorbar(q1, ax=ax1)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_title(f"$m$ = {m}, $n$ = {n}")
    
    plt.tight_layout()
    
    q2 = ax2.contourf(R, PHI, Z, cmap='jet')
    fig.colorbar(q2, ax=ax2)
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$\phi$")
    ax2.set_title(f"$m$ = {m}, $n$ = {n}")
    
    plt.tight_layout()
    
    plt.savefig(dir_path+f'Galerkin_m{m}n{n}.png', dpi=300)
    """
    plt.show()
    plt.clf()
    plt.close()
    """


# Parameters ___________________________________________________________________

N = 1000
M_max = 3
N_max = 3

r = np.linspace(0, 1, N)
phi = np.linspace(0, np.pi, N)
R, PHI = np.meshgrid(r, phi) #za plt.countourf
X = R * np.cos(PHI)
Y = R * np.sin(PHI)


#C, a = galerkin(3, 3)
#Z = izracunZ(M_max, N_max)



# Plotting _____________________________________________________________________

# Plotanje testnih funkcij X Y in R PHI ________________________________
singleZ_plot(2, 2)
#singleZ_plot2(2, 2)
plt.show()

def plot_testnih(M_max, N_max):
    for m in range(M_max + 1):
        for n in range(1, N_max + 1):
            singleZ_plot(m, n)

    plt.show()
    plt.clf()
    plt.close()
    
#plot_testnih(M_max, N_max)

# plot 3D testnih ______________________________________________________
#singleZ_plot2(2, 2)

#singleZ_3Dplot(0, 1)

# Plot single Galerkin _________________________________________________
#singleGalerkin_plot(3, 3, N)
#singleGalerkin_plot(100, 100, N)
#plt.show()


# Napake in Time _______________________________________________________________
import csv
import ast
from io import StringIO
def read_results_from_csv(filename):
    # Set the CSV field size limit to a higher value
    csv.field_size_limit(2**31-1)
    
    results = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert string back to array using np.loadtxt
            #array_data = np.fromstring(row['a'], sep=',')
            
            # Use StringIO to read the array from a string buffer
            buffer = StringIO(row['a'])
            array_data = np.loadtxt(buffer)
            
            results.append((
                int(row['M']),
                int(row['N']),
                float(row['C']),
                #np.array(ast.literal_eval(row['a'])),  # Use ast.literal_eval to convert string to array
                array_data,
                float(row['err_C']),
                float(row['execution_time'])
            ))
    return results


def plotErrTime(Msez, Nnum):
    
    results = read_results_from_csv(dir_path+'Results\\'+'C100a100.csv')[0]
    
    print(results)
    C100 = results[2]
    a100 = results[3]
    
    
    #print(a100[3])
    Csez = np.zeros((len(Msez), Nnum))
    Cerr = np.zeros((len(Msez), Nnum))
    ExecTime = np.zeros((len(Msez), Nnum))
    for m, Mnum in enumerate(Msez):
        print(f'm = {Mnum}')
        for n in range(Nnum):
            start_time = time.process_time()
            C = galerkin(Mnum, n+1)[0]
            execution_time = time.process_time() - start_time
            #print(C)
            Csez[m, n] = C
            err = np.abs(C - C100)
            Cerr[m, n] = err
            ExecTime[m, n] = execution_time
    

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(Msez)))
    #for m, sezm in enumerate(Cerr):
    for m in range(len(Msez)):
        x = np.arange(Nnum)
        ax1.plot(x, Cerr[m], lw=2, alpha=0.9, c=colors[m], label=f'$m = {Msez[m]}$')
        #ax1.plot(x, Cerr[m], lw=2, alpha=0.9, label=f'$m = {Msez[m]}$')
        
        ax2.plot(x, ExecTime[m], lw=2, alpha=0.9, c=colors[m], label=f'$m = {Msez[m]}$')

    # abs error ________________________________________________________
    ax1.set_yscale('log')
    ax1.set_xlabel(r"$n_{max}$")
    ax1.set_ylabel(r"$|C_{100, 100} - C_{m_{max},n_{max}}|$")
    ax1.set_title(f"Absolutna napaka")
    ax1.legend()

    
    plt.tight_layout()
    
    # exec time ________________________________________________________
    ax2.set_xlabel(r"$n_{max}$")
    ax2.set_ylabel(r"$t$")
    ax2.set_title(f"Čas izvajanja")
    ax2.legend()
    
    plt.tight_layout()
    
    plt.savefig(dir_path+f'ErrTime_m{Msez[-1]}n{Nnum}.png', dpi=300)

Msez = np.array([0,1,2,5,10,20,30,50,75,100])
#Nsez = np.array([0,1,2,5,10,20,30,50,75,100])
#Nsez = (50 * np.ones(len(Msez))).astype(int)
#Msez = [1, 2, 3]
Nnum = 50
#plotErrTime(Msez, Nnum)
#plt.show()


# Plot Galerkin m100n100 _______________________________________________________

def GalerkinFromFile_plot(filename):
    global R, PHI, X, Y

    results = read_results_from_csv(dir_path+'Results\\'+filename)[0]
    
    print(results)
    M_max = results[0]
    N_max = results[1]
    Cfromfile = results[2]
    afromfile = results[3]
    #"""
    Z = np.zeros((N, N))
    for m in range(M_max + 1):
        for n in range(1, N_max+1):
            print(f"$m$ = {m}, $n$ = {n}")
            #Z += afromfile[m*N_max + n]*testne(m, n+1, R, PHI)
            #Z += afromfile[m*N_max + n-1]*testne(m, n, R, PHI)
            Z += afromfile[m*N_max + n-1]*testne(m, n, R, PHI)
    """   
    # Assuming afromfile is a NumPy array and has the correct shape
    # Adjust the shape if needed
    afromfile_reshaped = afromfile.reshape((M_max + 1, N_max))

    # Create meshgrid for m and n
    m, n = np.meshgrid(np.arange(M_max + 1), np.arange(1, N_max + 1), indexing='ij')

    # Compute Z using vectorized operations
    Z_vectorized = np.sum(afromfile_reshaped[m, n-1] * testne_reshaped(m, n, R, PHI), axis=(0, 1))
    Z = Z_vectorized
    # Z_vectorized now contains the result without explicit loops     
    """      
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    q1 = ax1.contourf(X, Y, Z, cmap='jet')
    fig.colorbar(q1, ax=ax1)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_title(f"$m$ = {m}, $n$ = {n}")
    
    plt.tight_layout()
    
    q2 = ax2.contourf(R, PHI, Z, cmap='jet')
    fig.colorbar(q2, ax=ax2)
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$\phi$")
    ax2.set_title(f"$m$ = {m}, $n$ = {n}")
    
    plt.tight_layout()
    
    filename_for_pic = filename.rsplit('.', 1)[0]
    plt.savefig(dir_path+f'Galerkin_{filename_for_pic}.png', dpi=300)
    """
    plt.show()
    plt.clf()
    plt.close()
    """

#GalerkinFromFile_plot('C3a3_threaded.csv')
#GalerkinFromFile_plot('C5a5_threaded.csv')
#GalerkinFromFile_plot('C100a100.csv')
#plt.show()


def singleGalerkin_3Dplot(filename):
    global R, PHI, X, Y
    
    #Z = testne(m, n, R, PHI)

    results = read_results_from_csv(dir_path+'Results\\'+filename)[0]
    
    print(results)
    M_max = results[0]
    N_max = results[1]
    Cfromfile = results[2]
    afromfile = results[3]
    #"""
    Z = np.zeros((N, N))
    for m in range(M_max + 1):
        for n in range(1, N_max+1):
            print(f"$m$ = {m}, $n$ = {n}")
            #Z += afromfile[m*N_max + n]*testne(m, n+1, R, PHI)
            #Z += afromfile[m*N_max + n-1]*testne(m, n, R, PHI)
            Z += afromfile[m*N_max + n-1]*testne(m, n, R, PHI)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot the surface
    surf1 = ax1.plot_surface(X, Y, Z, cmap='jet', edgecolor='k')

    # Customize the plot
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_zlabel(r"Galerkin rešitev")
    ax1.set_title(f"$m$ = {m}, $n$ = {n}")

    # Add colorbar for reference
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=15)
    
    
    # Plot the surface
    surf2 = ax2.plot_surface(R, PHI, Z, cmap='jet', edgecolor='k')

    # Customize the plot
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$\phi$")
    ax2.set_zlabel(r"Galerkin rešitev")
    ax2.set_title(f"$m$ = {m}, $n$ = {n}")

    # Add colorbar for reference
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=15)
    
    # Show the plot
    plt.tight_layout()
    #plt.show()
    
singleGalerkin_3Dplot('C3a3_threaded.csv')
plt.show()