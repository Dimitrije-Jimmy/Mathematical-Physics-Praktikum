import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import time

from main1_functions import *


# Initialization _______________________________________________________________

# prvi del _____________________________________________________________
k = 1.
w = k**0.5

w = 0.2
k = w**2.
lamb = 10.
alpha = k**(1/4)

a = -40.
b = 40.
t0 = 0.
tnih = 2*np.pi / w
tf = 314.
tf = 1.*tnih


N = 300
M = 1000

dx = (b - a)/(N-1)
dt = (tf - t0)/(M-1)

x = np.linspace(a, b, N)
t = np.linspace(t0, tf, M)
#t = np.arange(t0, tf, dt)


# drugi del ____________________________________________________________
sigma0 = 1./20.
k0 = 50.*np.pi
lamb2 = 0.25

a2 = -0.5
b2 = 1.5
t02 = 0.
tf2 = 1.
tf2 = .005

dx2 = (b2 - a2)/(N-1)
dt = 2.*dx**2
x2 = np.arange(a2, b2+dx2, dx2)
t2 = np.arange(t02, tf2, dt)

x2 = np.linspace(a2, b2, N)
t2 = np.linspace(t02, tf2, M)


# Calculations _________________________________________________________________

# prvi del
phi1_anal = analytic_koherent2(x, t)
zp1 = valovna_prvi(x)
phi1 = C_N(x, t, zp1)

napaka1 = np.abs(phi1 - phi1_anal)
napaka1sq = np.abs(np.abs(phi1**2) - np.abs(phi1_anal**2))

# drugi del
phi2_anal = analytic_empty2(x2, t2)
zp2 = valovna_drugi(x2)
phi2 = C_N(x2, t2, zp2)

napaka2 = np.abs(phi2 - phi2_anal)
napaka2sq = np.abs(np.abs(phi2**2) - np.abs(phi2_anal**2))

# Plotting _____________________________________________________________________

# spodnje grafe sem že shranil
import sys
#sys.exit()

# Začetni pogoj ________________________________________________________

#plt.figure(figsize=(8, 6))                                          
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))                   # PLOT 1

ax1.set_title('Začetni pogoj: Prvi paket')
ax1.plot(x, np.real(zp1), 'b')#, label='Analytical')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$Re(\Psi)$')
#ax1.set_legend()
ax1.grid(True)

ax2.set_title('Začetni pogoj: Drugi paket')
ax2.plot(x, np.real(zp2), 'r')#, label='Analytical')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$Re(\Psi)$')
#ax2.set_legend()
ax2.grid(True)


# Analytical ___________________________________________________________________

# prvi paket ___________________________________________________________
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))              # PLOT 2

q1 = ax1.imshow(np.real(phi1_anal), cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
ax1.set_title('Real part')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')

q2 = ax2.imshow(np.imag(phi1_anal), cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
ax2.set_title('Imaginary part')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$t$')

phisq = np.abs(phi1_anal**2)
q3 = ax3.imshow(phisq, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
ax3.set_title(r'$|\Psi|^2$')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$t$')

plt.colorbar(q1, ax=ax1, label='Ampl.')
plt.colorbar(q2, ax=ax2, label='Ampl.')
cbar3 = plt.colorbar(q3, ax=ax3, label=r'$|\Psi|^2$')
cbar3.set_label('Ampl.', rotation=270, labelpad=15)

plt.suptitle('Analitična rešitev: Prvi paket')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))              # PLOT 3

colors = plt.cm.jet(np.linspace(1, 0, M))
for i in range(M):
    ax1.plot(x, np.real(phi1_anal[i]), color=colors[i])
ax1.set_title('Real part')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\mathrm{Re}(\Psi)$')

for i in range(M):
    ax2.plot(x, np.imag(phi1_anal[i]), color=colors[i])
ax2.set_title('Imaginary part')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$\mathrm{Im}(\Psi)$')

for i in range(M):
    phisq = np.abs(phi1_anal[i]**2)
    ax3.plot(x, phisq, color=colors[i])
ax3.set_title(r'$|\Psi|^2$')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$|\Psi|^2$')

# Create a ScalarMappable
sm = cm.ScalarMappable(cmap=cm.jet_r, norm=mcolors.Normalize(vmin=t[0], vmax=t[-1]))
sm.set_array([])  # An array with no data to normalize

cbar = plt.colorbar(sm, ax=ax3, label='Time')
cbar.set_label('Time', rotation=270, labelpad=15)

plt.suptitle('Analitična rešitev: Prvi paket')
plt.tight_layout()


# drugi paket __________________________________________________________
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))              # PLOT 4

q1 = ax1.imshow(np.real(phi2_anal), cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x2[0], x2[-1], t2[0], t2[-1]])
ax1.set_title('Real part')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')

q2 = ax2.imshow(np.imag(phi2_anal), cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x2[0], x2[-1], t2[0], t2[-1]])
ax2.set_title('Imaginary part')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$t$')

phisq = np.abs(phi2_anal**2)
q3 = ax3.imshow(phisq, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x2[0], x2[-1], t2[0], t2[-1]])
ax3.set_title(r'$|\Psi|^2$')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$t$')

plt.colorbar(q1, ax=ax1, label='Ampl.')
plt.colorbar(q2, ax=ax2, label='Ampl.')
cbar3 = plt.colorbar(q3, ax=ax3, label=r'$|\Psi|^2$')
cbar3.set_label('Ampl.', rotation=270, labelpad=15)

plt.suptitle('Analitična rešitev: Drugi paket')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))              # PLOT 5

colors = plt.cm.jet(np.linspace(1, 0, M))
for i in range(M):
    ax1.plot(x2, np.real(phi2_anal[i]), color=colors[i])
ax1.set_title('Real part')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\mathrm{Re}(\Psi)$')

for i in range(M):
    ax2.plot(x2, np.imag(phi2_anal[i]), color=colors[i])
ax2.set_title('Imaginary part')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$\mathrm{Im}(\Psi)$')

for i in range(M):
    phisq = np.abs(phi2_anal[i]**2)
    ax3.plot(x2, phisq, color=colors[i])
ax3.set_title(r'$|\Psi|^2$')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$|\Psi|^2$')

# Create a ScalarMappable
sm = cm.ScalarMappable(cmap=cm.jet_r, norm=mcolors.Normalize(vmin=t2[0], vmax=t2[-1]))
sm.set_array([])  # An array with no data to normalize

cbar = plt.colorbar(sm, ax=ax3, label='Time')
cbar.set_label('Time', rotation=270, labelpad=15)

plt.suptitle('Analitična rešitev: Drugi paket')
plt.tight_layout()


# Crank-Nicolson _______________________________________________________________

# prvi paket ___________________________________________________________
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))              # PLOT 6

q1 = ax1.imshow(np.real(phi1), cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
ax1.set_title('Real part')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')

q2 = ax2.imshow(np.imag(phi1), cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
ax2.set_title('Imaginary part')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$t$')

phisq = np.abs(phi1**2)
q3 = ax3.imshow(phisq, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
ax3.set_title(r'$|\Psi|^2$')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$t$')

plt.colorbar(q1, ax=ax1, label='Ampl.')
plt.colorbar(q2, ax=ax2, label='Ampl.')
cbar3 = plt.colorbar(q3, ax=ax3, label=r'$|\Psi|^2$')
cbar3.set_label('Ampl.', rotation=270, labelpad=15)

plt.suptitle('Numerična rešitev: Prvi paket')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))              # PLOT 7

colors = plt.cm.jet(np.linspace(1, 0, M))
for i in range(M):
    ax1.plot(x, np.real(phi1[i]), color=colors[i])
ax1.set_title('Real part')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\mathrm{Re}(\Psi)$')

for i in range(M):
    ax2.plot(x, np.imag(phi1[i]), color=colors[i])
ax2.set_title('Imaginary part')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$\mathrm{Im}(\Psi)$')

for i in range(M):
    phisq = np.abs(phi1[i]**2)
    ax3.plot(x, phisq, color=colors[i])
ax3.set_title(r'$|\Psi|^2$')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$|\Psi|^2$')

# Create a ScalarMappable
sm = cm.ScalarMappable(cmap=cm.jet_r, norm=mcolors.Normalize(vmin=t[0], vmax=t[-1]))
sm.set_array([])  # An array with no data to normalize

cbar = plt.colorbar(sm, ax=ax3, label='Time')
cbar.set_label('Time', rotation=270, labelpad=15)

plt.suptitle('Numerična rešitev: Prvi paket')
plt.tight_layout()


# drugi paket __________________________________________________________
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))              # PLOT 8

q1 = ax1.imshow(np.real(phi2), cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x2[0], x2[-1], t2[0], t2[-1]])
ax1.set_title('Real part')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')

q2 = ax2.imshow(np.imag(phi2), cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x2[0], x2[-1], t2[0], t2[-1]])
ax2.set_title('Imaginary part')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$t$')

phisq = np.abs(phi2**2)
q3 = ax3.imshow(phisq, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x2[0], x2[-1], t2[0], t2[-1]])
ax3.set_title(r'$|\Psi|^2$')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$t$')

plt.colorbar(q1, ax=ax1, label='Ampl.')
plt.colorbar(q2, ax=ax2, label='Ampl.')
cbar3 = plt.colorbar(q3, ax=ax3, label=r'$|\Psi|^2$')
cbar3.set_label('Ampl.', rotation=270, labelpad=15)

plt.suptitle('Numerična rešitev: Drugi paket')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))              # PLOT 9

colors = plt.cm.jet(np.linspace(1, 0, M))
for i in range(M):
    ax1.plot(x2, np.real(phi2[i]), color=colors[i])
ax1.set_title('Real part')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\mathrm{Re}(\Psi)$')

for i in range(M):
    ax2.plot(x2, np.imag(phi2[i]), color=colors[i])
ax2.set_title('Imaginary part')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$\mathrm{Im}(\Psi)$')

for i in range(M):
    phisq = np.abs(phi2[i]**2)
    ax3.plot(x2, phisq, color=colors[i])
ax3.set_title(r'$|\Psi|^2$')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$|\Psi|^2$')

# Create a ScalarMappable
sm = cm.ScalarMappable(cmap=cm.jet_r, norm=mcolors.Normalize(vmin=t2[0], vmax=t2[-1]))
sm.set_array([])  # An array with no data to normalize

cbar = plt.colorbar(sm, ax=ax3, label='Time')
cbar.set_label('Time', rotation=270, labelpad=15)

plt.suptitle('Numerična rešitev: Drugi paket')
plt.tight_layout()


plt.show()

