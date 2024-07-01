import numpy as np
import matplotlib.pyplot as plt

import threading
import os
from queue import Queue

from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation


# Seed value
# Apparently you may use different seed values at each stage
seed_value= 10001

# importing datetime module for now()
import datetime
 
# using now() to get current time
current_time = datetime.datetime.now()
print(current_time)
seed_value = current_time.second

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#sampling
N = 10000      # število korakov
num = N/10    # število sprehodov
#num = 5
num = N/2
num = N/10

N = 5000
#N = 10000
num = 500

mu = 5.0
nu = 1.0
nu = mu
vkonst = 5.
tkonst = 1

directory = os.path.dirname(__file__)+'\\Simulations\\'
filename = f"walk_{mu}.csv"
file_path = directory + '\\' + filename


file = open(file_path, 'w')


def linear(x, k, n):
    return k*x + n

def distance(x, y):
    return (x**2 + y**2)**0.5



queue = Queue()
def fill_queue(n):
    for primer in range(1, int(n)+1):
        queue.put(primer)


def walk2(N, mu, nu):
    global vkonst, tkonst

    x_sez = np.zeros(N)
    y_sez = np.zeros(N)
    time1 = np.zeros(N)     # the time associated with walk
    time2 = np.zeros(N)     # the time associated trapping
    #x_sez[0], y_sez[0] = 0., 0.

    theta = np.random.random(N)*2*np.pi
    #pareto distribution, pazi na (a+1) potenco!
    #a, m = 25., 2.  # shape and mode
    m = 2
    l = (np.random.pareto(mu-1, N) + 1) * m

    
    time2 = (np.random.pareto(nu-1, N) + 1) * m

    for i in range(1, N):
        x_sez[i] = x_sez[i-1] + l[i]*np.cos(theta[i])
        y_sez[i] = y_sez[i-1] + l[i]*np.sin(theta[i])
        time1[i] = time1[i-1] + l[i]/vkonst
        
        
    razdalja1 = distance(x_sez, y_sez)

    return x_sez, y_sez, time1, time2, theta, l, razdalja1



matrix = []
#matrix = np.zeros(int(num))
#matrix = [0 for _ in range(int(num))]
def worker():
    # The main function
    global N, num, mu, nu
    
    while not queue.empty():
        subject = queue.get()
        print(subject)

        x_sez, y_sez, time1, time2, theta, l, razdalja1 = walk2(N, mu, nu)
        #print(x_sez)

        #razdalja2 = 

        file.write(f"{subject},\t {x_sez},\t {y_sez},\t {time1},\t {time2},\t {theta},\t {l},\t {razdalja1}\n")


        file.flush()

        matrix.append([x_sez, y_sez, time1, time2, theta, l, razdalja1])#, razdalja2])
        #matrix[subject-1] = np.array([x_sez, y_sez, time1, time2, theta, l, razdalja1])
        #matrix[subject-1] = [x_sez, y_sez, time1, time2, theta, l, razdalja1]
        

def run_threads(threads, n):
    global N, mu, nu

    file.write(f"N = {N}, ,\t mu = {mu},\t nu = {nu}\n")

    print("running")

    fill_queue(n)

    thread_list = []

    for t in range(threads):
        thread = threading.Thread(target=worker)
        thread_list.append(thread)

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()

    file.close()
    print("done")


run_threads(500, num)



# Calculation __________________________________________________________________

#matrix[
# [x_sez, y_sez, time1, time2, theta, l, razdalja1],
# [],
# ...
#       ]

matrix = np.array(matrix)

x_coord = matrix[:, 0]
y_coord = matrix[:, 1]
time1 = matrix[:, 2]
time2 = matrix[:, 3]
razdalje = matrix[:, 6]

time0 = np.linspace(0, tkonst*N, N) # the time associated with flight


#MAD_flight = [median_abs_deviation([r[i] for r in razdalje]) for i in range(len(razdalje[0]))]
MAD_flight = [np.sqrt(median_abs_deviation([x[i] for x in x_coord])**2 + median_abs_deviation([y[i] for y in y_coord])**2) for i in range(len(x_coord[0]))]


tmax = N
#t = np.linspace(0, tkonst*N, N)
#t_interp = [np.linspace(0, time1[i, -1], N*2) for i in range(len(time1))]
#razdalje2 = [np.interp(t_interp[i], time1[i], razdalje[i]) for i in range(len(razdalje))]

t_interp = np.linspace(0, time1[0, -1], N*2)
x_interp = np.array([np.interp(t_interp, time1[i], x_coord[i]) for i in range(len(razdalje))])
y_interp = np.array([np.interp(t_interp, time1[i], y_coord[i]) for i in range(len(razdalje))])
razdalje2 = distance(x_interp, y_interp)


#MAD_walk = [median_abs_deviation([r[i] for r in razdalje2]) for i in range(len(razdalje2[0]))]
MAD_walk = [np.sqrt(median_abs_deviation([x[i] for x in x_interp])**2 + median_abs_deviation([y[i] for y in y_interp])**2) for i in range(len(x_interp[0]))]


print(np.where(np.isnan(MAD_walk)), np.where(np.isinf(MAD_walk)))
#(array([], dtype=int64),) (array([], dtype=int64),)

# Plotting _____________________________________________________________________
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

r1 = np.linspace(np.amin(np.log(time0[1:])), np.amax(np.log(time0[1:])))
par1, cov1 = curve_fit(linear, np.log(time0[1:]), 2*np.log(MAD_flight[1:]))
perr1 = np.sqrt(np.diag(cov1))
print (par1)
print (perr1)

print("_________________", time0[-1])
plt.scatter(np.log(time0[1:]), 2*np.log(MAD_flight[1:]))
plt.plot(r1, linear(r1, par1[0], par1[1]), 'r-', label=f'$\gamma = {np.around(par1[0], 4)} \pm {np.around(perr1[0], 4)}$')

plt.xlabel(r'$\ln(t)$')
plt.ylabel(r'$2\,\ln(MAD(\; r(t) \;)$')
plt.legend()
plt.title(f'Polet za $\mu = {mu}$')


plt.subplot(1, 2, 2)

r2 = np.linspace(np.amin(np.log(t_interp[1:])), np.amax(np.log(t_interp[1:])))
par2, cov2 = curve_fit(linear, np.log(t_interp[2:]), 2*np.log(MAD_walk[2:]))
perr2 = np.sqrt(np.diag(cov2))
print (par2)
print (perr2)


plt.plot(r2, linear(r2, par2[0], par2[1]), 'r-', label=f'$\gamma = {np.around(par2[0], 4)} \pm {np.around(perr2[0], 4)}$')

plt.scatter(np.log(t_interp[2:]), 2*np.log(MAD_walk[2:]))


plt.xlabel(r'$\ln(t)$')
plt.ylabel(r'$2\,\ln(MAD(\; r(t) \;)$')
plt.legend()
plt.title(f'Sprehod za $\mu = {mu}$')


plt.tight_layout()
plt.show()
plt.clf()
plt.close()



#Interpolation _________________________________________________________________

t_interp = np.linspace(0, time1[0, -1], N*2)
x_interp = np.array([np.interp(t_interp, time1[i], x_coord[i]) for i in range(len(razdalje))])
y_interp = np.array([np.interp(t_interp, time1[i], y_coord[i]) for i in range(len(razdalje))])
razdalje2 = distance(x_interp, y_interp)


plt.figure(figsize=(12, 8))


#plt.plot(time1[0, 1:], razdalje[0, 1:], 'r-', lw=0.5, label='Sprehod')
#plt.scatter(time1[0], razdalje[0], 'b')
#plt.plot(t_interp[1:], razdalje2[0, 1:], 'k-', lw=0.25, label='Interpolacija')
#plt.scatter(t_interp, razdalje2[0], 'k')


plt.plot(x_coord[0, :25], y_coord[0, :25], 'k-', lw=0.5, label='Sprehod')
plt.scatter(x_coord[0, :25], y_coord[0, :25], c='k')
plt.plot(x_interp[0, :50], y_interp[0, :50], 'g-', lw=0.25, label='Interpolacija')
plt.scatter(x_interp[0, :50], y_interp[0, :50], c='g')

plt.scatter(x_coord[0, 0], y_coord[0, 0], c='b', label=f'Start $t=0$')
#plt.scatter(x_coord[0, -1], y_coord[0, -1], c='r', label=f'End $t={N}$')
plt.scatter(x_coord[0, 24], y_coord[0, 24], c='r', label=f'End $t={100}$')


plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Interpolacija sprehoda')

plt.tight_layout()
#plt.show()
plt.clf()
plt.close()

# ______________________________________________________________________________