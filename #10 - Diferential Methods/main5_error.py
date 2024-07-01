import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import time

from main1_functions import *

dir_path = os.path.dirname(os.path.realpath(__file__))+'\\Results\\'
print(dir_path)

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
tf = 10.*tnih

# drugi del ____________________________________________________________
sigma0 = 1./20.
k0 = 50.*np.pi
lamb2 = 0.25

a2 = -0.5
b2 = 1.5
t02 = 0.
tf2 = 1.
tf2 = .005


# Calculations _________________________________________________________________
a = -20.
b = 20.
t0 = 0.
tf = 5.*tnih

N = 300
N1 = 300
N2 = 1000
N3 = 2500
N4 = 5000
N5 = 700
M = 300
M1 = 300
M2 = 1000
M3 = 2500
M4 = 5000
M5 = 700

x = np.linspace(a, b, N)
t = np.linspace(t0, tf, M)
x1 = np.linspace(a, b, N1)
t1 = np.linspace(t0, tf, M1)
x2 = np.linspace(a, b, N2)
t2 = np.linspace(t0, tf, M2)
x3 = np.linspace(a, b, N3)
t3 = np.linspace(t0, tf, M3)
x4 = np.linspace(a, b, N4)
t4 = np.linspace(t0, tf, M4)
x5 = np.linspace(a, b, N5)
t5 = np.linspace(t0, tf, M5)


print("Beggining calculations... Patience please.")
# prvi del
phi1_anal = analytic_koherent2(x, t)
start_time1 = time.process_time()
zp1 = valovna_prvi(x1)
phi1 = C_N(x1, t1, zp1)
napaka1 = np.abs(phi1 - phi1_anal)
napaka1sq = np.abs(np.abs(phi1**2) - np.abs(phi1_anal**2))
elapsed_time1 = time.process_time() - start_time1

phi2_anal = analytic_koherent2(x2, t2)
start_time2 = time.process_time()
zp2 = valovna_prvi(x2)
phi2 = C_N(x2, t2, zp2)
napaka2 = np.abs(phi2 - phi2_anal)
napaka2sq = np.abs(np.abs(phi2**2) - np.abs(phi2_anal**2))
elapsed_time2 = time.process_time() - start_time2



phi3_anal = analytic_koherent2(x3, t3)
start_time3 = time.process_time()
zp3 = valovna_prvi(x3)
phi3 = C_N(x3, t3, zp3)
napaka3 = np.abs(phi3 - phi3_anal)
napaka3sq = np.abs(np.abs(phi3**2) - np.abs(phi3_anal**2))
elapsed_time3 = time.process_time() - start_time3


phi4_anal = analytic_koherent2(x4, t4)
start_time4 = time.process_time()
zp4 = valovna_prvi(x4)
phi4 = C_N(x4, t4, zp4)
napaka4 = np.abs(phi4 - phi4_anal)
napaka4sq = np.abs(np.abs(phi4**2) - np.abs(phi4_anal**2))
elapsed_time4 = time.process_time() - start_time4

phi5_anal = analytic_koherent2(x5, t5)
start_time5 = time.process_time()
zp5 = valovna_prvi(x5)
phi5 = C_N(x5, t5, zp5)
napaka5 = np.abs(phi5 - phi5_anal)
napaka5sq = np.abs(np.abs(phi5**2) - np.abs(phi5_anal**2))
elapsed_time5 = time.process_time() - start_time5

print("Calculations done!")

# Plotting _____________________________________________________________________
"""
# Napaka time
fig, (ax1) = plt.subplots(1, 1, figsize=(9, 6))

#povpr1 = [np.mean(sez) for sez in napaka1sq]
povpr1 = np.mean(napaka1sq, axis=1)
ax1.plot(t1, povpr1, c='b', ls='-', lw=0.8, alpha=0.9, label=f'$N$ = {M1}')
ax1.scatter(t1, povpr1, c='b', marker='o', s=0.9)
povpr2 = np.mean(napaka2sq, axis=1)
ax1.plot(t2, povpr2, c='g', ls='-', lw=0.8, alpha=0.9, label=f'$N$ = {M2}')
ax1.scatter(t2, povpr2, c='g', marker='o', s=0.9)

povpr3 = np.mean(napaka3sq, axis=1)
ax1.plot(t3, povpr3, c='r', ls='-', lw=0.8, alpha=0.9, label=f'$N$ = {M3}')
ax1.scatter(t3, povpr3, c='r', marker='o', s=0.9)
povpr4 = np.mean(napaka4sq, axis=1)
ax1.plot(t4, povpr4, c='c', ls='-', lw=0.8, alpha=0.9, label=f'$N$ = {M4}')
ax1.scatter(t4, povpr4, c='c', marker='o', s=0.9)
povpr5 = np.mean(napaka5sq, axis=1)
ax1.plot(t5, povpr5, c='y', ls='-', lw=0.8, alpha=0.9, label=f'$N$ = {M5}')
ax1.scatter(t5, povpr5, c='y', marker='o', s=0.9)

#ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$||\Psi|^2 - |\Psi_{analytical}|^2|$')
ax1.legend()

plt.title(f'Absolutna napaka za različne $N$')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
"""

# Assuming you have these variables defined
dimensionsM = np.array([M1, M5, M2, M3, M4])
dimensionsN = np.array([N1, N5, N2, N3, N4])
execution_time = np.array([elapsed_time1, elapsed_time5, elapsed_time2, elapsed_time3, elapsed_time4])

execution_time3 = np.array([[elapsed_time1, elapsed_time5, elapsed_time2, elapsed_time3, elapsed_time4],
                            [elapsed_time1, elapsed_time5, elapsed_time2, elapsed_time3, elapsed_time4],
                            [elapsed_time1, elapsed_time5, elapsed_time2, elapsed_time3, elapsed_time4],
                            [elapsed_time1, elapsed_time5, elapsed_time2, elapsed_time3, elapsed_time4],
                            [elapsed_time1, elapsed_time5, elapsed_time2, elapsed_time3, elapsed_time4]])

fig, ax = plt.subplots(figsize=(10, 6))

# Create a meshgrid for time and integration values
X, Y = np.meshgrid(dimensionsN, dimensionsM)

# Scatter plot for discrete data points
#sc = ax.scatter(X.flatten(), Y.flatten(), c=execution_time.flatten(), cmap='viridis', s=50, marker='s')

# Contour plot for blocky regions
heatmap = ax.pcolormesh(X, Y, execution_time3, cmap='viridis')

# Customize the plot
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$M$')
ax.set_title(r'Čas izvajanja v odvisnosti od dimenzije $N\times M$')

# Add a colorbar
cbar = plt.colorbar(heatmap, label=r'Čas izvajanja - $t$')

# Show the plot
plt.show()

import sys
sys.exit()
dimensionsM = np.array([M1, M5, M2, M3, M4])
dimensionsN = np.array([N1, N5, N2, N3, N4])
execution_time = np.array([elapsed_time1, elapsed_time5, elapsed_time2, elapsed_time3, elapsed_time4])
execution_time2 = np.array([[elapsed_time1, elapsed_time5, elapsed_time2, elapsed_time3, elapsed_time4],
                            [elapsed_time1, elapsed_time5, elapsed_time2, elapsed_time3, elapsed_time4]])

#execution_time_2d = execution_time.reshape((len(dimensionsM), len(dimensionsN)))


plt.plot(dimensionsM, execution_time, c='g', ls='-')#, lw=0.8, alpha=0.9)
plt.scatter(dimensionsM, execution_time, c='b', marker='o')#, s=0.9)

plt.xlabel(r"Dimenzija - $N$")
plt.ylabel(r"Čas izvajanja - $t$")

plt.title(f'Čas izvajanja za različne $N$')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()



fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create a meshgrid for time and integration values
X, Y = np.meshgrid(dimensionsN, dimensionsM[:-1])
#X, Y = np.meshgrid(dimensionsM, [1])

execution_time2 = execution_time2.T.reshape((5, 2))

cmap = plt.get_cmap('viridis')
#heatmap = ax.pcolormesh(X, Y, execution_time.reshape(X.shape), cmap=cmap)
#heatmap = ax.contourf(X, Y, execution_time, cmap=cmap)
heatmap = ax.pcolormesh(X, Y, execution_time2, cmap=cmap, shading='flat')

# Customize the plot
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$M$')
ax.set_title(r'Čas izvajanja v odvisnosti od dimenzije $N\times M$')

# Add a colorbar
cbar = plt.colorbar(heatmap, label=r'Čas izvajanja - $t$')

# Show the plot
plt.show()


import sys
sys.exit()
# Napaka side to side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

q1 = ax1.imshow(napaka2sq, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x2[0], x2[-1], t2[0], t2[-1]])
#ax1.set_title(f'$N$ = {len(x1)}, $M$ = {len(t1)}')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')
plt.colorbar(q1, ax=ax1, label=r'Ampl.')

colors = plt.cm.jet(np.linspace(0, 1, M2))
for i in range(M2):
    ax2.plot(x2, napaka2sq[i], color=colors[i])
#ax2.set_title(r'Absolutna napaka')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$||\Psi|^2 - |\Psi_{analytical}|^2|$')

# Create a ScalarMappable
sm = cm.ScalarMappable(cmap=cm.jet, norm=mcolors.Normalize(vmin=t2[0], vmax=t2[-1]))
sm.set_array([])  # An array with no data to normalize

cbar = plt.colorbar(sm, ax=ax2, label='Time')
cbar.set_label('Time', labelpad=15)

plt.suptitle(f'Absolutna napaka za $N \\times N = {N2}\\times{M2}$ in $t = {tf/tnih}$ nihajev')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.show()


import sys
sys.exit()

# Divergenca NxN
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

phisq1 = np.abs(phi1**2)
print(phisq1.shape)
q1 = ax1.imshow(phisq1, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x1[0], x1[-1], t1[0], t1[-1]])
ax1.set_title(f'$N$ = {len(x1)}, $M$ = {len(t1)}')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')

phisq2 = np.abs(phi2**2)
q2 = ax2.imshow(phisq2, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x2[0], x2[-1], t2[0], t2[-1]])
ax2.set_title(f'$N$ = {len(x2)}, $M$ = {len(t2)}')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$t$')

phisq3 = np.abs(phi3**2)
q3 = ax3.imshow(phisq3, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x3[0], x3[-1], t3[0], t3[-1]])
ax3.set_title(f'$N$ = {len(x3)}, $M$ = {len(t3)}')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$t$')

phisq_anal = np.abs(phi1_anal**2)
q4 = ax4.imshow(phisq_anal, cmap='jet', interpolation='nearest', aspect='auto', origin='lower', extent=[x[0], x[-1], t[0], t[-1]])
ax4.set_title(f'Analitična rešitev $N$ = {len(x)}, $M$ = {len(t)}')
ax4.set_xlabel(r'$x$')
ax4.set_ylabel(r'$t$')


plt.colorbar(q1, ax=ax1, label='Ampl.')
plt.colorbar(q2, ax=ax2, label='Ampl.')
plt.colorbar(q3, ax=ax3, label='Ampl.')
cbar3 = plt.colorbar(q3, ax=ax4, label='Ampl.$')
#cbar3.set_label('Ampl.')#, rotation=270, labelpad=15)

#plt.suptitle('Analitična rešitev: Prvi paket')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()