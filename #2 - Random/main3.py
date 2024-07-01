import numpy as np
import matplotlib.pyplot as plt

import threading
import os
from queue import Queue

from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation


# Seed value
# Apparently you may use different seed values at each stage
seed_value= 35

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

vkonst = 5.
tkonst = 1

def linear(x, k, n):
    return k*x + n

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def model_flight(x):
    return 2/(x - 1)

def model_flight2(x, a, b):
    return a/(x - b)

def model_walk(x):
    return 4 - x

def model_walk2(x, a):
    return a - x

def distance(x, y):
    return (x**2 + y**2)**0.5



queue = Queue()
def fill_queue(num):
    for primer in range(1, int(num)+1):
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




#matrix = np.zeros(int(num))
#matrix = [0 for _ in range(int(num))]
def worker(N, num, mu, nu, matrix):
    # The main function
    #global N, num, mu, nu

    while not queue.empty():
        subject = queue.get()
        print(subject)

        x_sez, y_sez, time1, time2, theta, l, razdalja1 = walk2(N, mu, nu)
        #print(x_sez)

        #razdalja2 = 

        #file.write(f"{subject},\t {x_sez},\t {y_sez},\t {time1},\t {time2},\t {theta},\t {l},\t {razdalja1}\n")


        #file.flush()

        matrix.append([x_sez, y_sez, time1, time2, theta, l, razdalja1])#, razdalja2])
        #matrix[subject-1] = np.array([x_sez, y_sez, time1, time2, theta, l, razdalja1])
        #matrix[subject-1] = [x_sez, y_sez, time1, time2, theta, l, razdalja1]
    


def run_threads(threads, N, num, mu, nu):

    #file.write(f"N = {N}, ,\t mu = {mu},\t nu = {nu}\n")

    print("running")

    fill_queue(num)

    thread_list = []

    matrix = []
    for t in range(threads):
        thread = threading.Thread(target=worker, args=[N, num, mu, nu, matrix])
        thread_list.append(thread)

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()

    
    #file.close()
    print("done")

    return matrix



# Calculation __________________________________________________________________

#matrix[
# [x_sez, y_sez, time1, time2, theta, l, razdalja1],
# [],
# ...
#       ]

def velik(N, num, mu_sez):
    gamma_flight_sez = []
    gamma_flight_err_sez = []
    gamma_walk_sez = []
    gamma_walk_err_sez = []

    for mu in mu_sez:

        matrix = run_threads(500, N, num, mu, mu)
        matrix = np.array(matrix)

        #print(matrix)

        #x_coord = matrix[:, 0]
        #y_coord = matrix[:, 1]
        time1 = matrix[:, 2]
        #time2 = matrix[:, 3]
        razdalje = matrix[:, 6]

        t_flight = np.linspace(0, tkonst*N, N) # the time associated with flight

        MAD_flight = [median_abs_deviation([r[i] for r in razdalje]) for i in range(len(razdalje[0]))]

        t_walk = np.linspace(0, time1[0, -1], N*2)
        #x_interp = np.array([np.interp(t_walk, time1[i], x_coord[i]) for i in range(len(razdalje))])
        #y_interp = np.array([np.interp(t_walk, time1[i], y_coord[i]) for i in range(len(razdalje))])
        #razdalje2 = distance(x_interp, y_interp)
        razdalje2 = np.array([np.interp(t_walk, time1[i], razdalje[i]) for i in range(len(razdalje))])

        MAD_walk = [median_abs_deviation([r[i] for r in razdalje2]) for i in range(len(razdalje2[0]))]

        par_flight, cov_flight = curve_fit(linear, np.log(t_flight[4:]), 2*np.log(MAD_flight[4:]))
        par_walk, cov_walk = curve_fit(linear, np.log(t_walk[4:]), 2*np.log(MAD_walk[4:]))

        gamma_flight_sez.append(par_flight)
        gamma_flight_err_sez.append(np.sqrt(cov_flight))
        gamma_walk_sez.append(par_walk)
        gamma_walk_err_sez.append(np.sqrt(cov_walk))

    return gamma_flight_sez, gamma_walk_sez


N = 1000
num = 500

mu_sez = [1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
mu_sez = np.linspace(1.3, 5.0, 20)
gamma_flight, gamma_walk = velik(N, num, mu_sez)
gamma_flight, gamma_walk = np.array(gamma_flight), np.array(gamma_walk)

print(gamma_flight)
print("_"*20)
print(gamma_walk)

# Plotting _____________________________________________________________________

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

r1 = np.linspace(np.amin(mu_sez), np.amax(mu_sez))
#par1, cov1 = curve_fit(linear, mu_sez, gamma_flight[:, 0])
par1, cov1 = curve_fit(model_flight2, mu_sez, gamma_flight[:, 0])
perr1 = np.sqrt(np.diag(cov1))
print (par1, perr1)

r2 = np.linspace(np.amin(mu_sez), np.amax(mu_sez))
par2, cov2 = curve_fit(linear, mu_sez[int(len(mu_sez)/2):], gamma_walk[int(len(mu_sez)/2):, 0])
perr2 = np.sqrt(np.diag(cov2))
print (par2, perr2)

plt.plot(r2, linear(r2, par2[0], par2[1]), 'k--', label=f'$\gamma = {np.around(par2[0], 3)}\cdot\mu + {np.around(par2[1], 3)}$')
plt.plot(r1, model_flight2(r1, par1[0], par1[1]), 'r-', label=f'$\gamma = {np.around(par1[0], 3)}/(\mu - {np.around(par1[1], 3)})$')
plt.scatter(mu_sez, gamma_flight[:, 0], label=f'Podatki')


plt.xlabel(r'$\mu$')
plt.ylabel(r'$\gamma$')
#plt.yscale('log')
plt.legend()
plt.title(f'Poleti')


plt.subplot(1, 2, 2)

r3 = np.linspace(np.amin(mu_sez), np.amax(mu_sez))
par3, cov3 = curve_fit(linear, mu_sez[int(len(mu_sez)/2):], gamma_walk[int(len(mu_sez)/2):, 0])
perr3 = np.sqrt(np.diag(cov3))
print (par3, perr3)

r4 = np.linspace(np.amin(mu_sez), np.amax(mu_sez))
par4, cov4 = curve_fit(model_walk2, mu_sez[:int(len(mu_sez)/2)], gamma_walk[:int(len(mu_sez)/2), 0])
perr4 = np.sqrt(np.diag(cov4))
print (par4, perr4)


plt.plot(r3, linear(r3, par3[0], par3[1]), 'k--', label=f'$\gamma = {np.around(par3[0], 3)}\cdot\mu + {np.around(par3[1], 3)}$')
plt.plot(r4[:16], model_walk2(r4[:16], par4[0]), 'r--', label=f'$\gamma = {np.around(par4[0], 3)} - \mu$')

plt.scatter(mu_sez, gamma_walk[:, 0], label='Podatki')


plt.xlabel(r'$\mu$')
plt.ylabel(r'$\gamma$')
#plt.yscale('log')
plt.legend()
plt.title(f'Sprehodi')


plt.tight_layout()
plt.show()
plt.clf()
plt.close()



