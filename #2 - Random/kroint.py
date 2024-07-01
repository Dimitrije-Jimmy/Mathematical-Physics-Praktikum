import numpy as np
import matplotlib.pyplot as plt

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 10001

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#sampling
N=100

#gauss
r1=np.random.random_sample(N)
r2=np.random.random_sample(N)

x=r1*2 -1
y=r2*2 -1

#plot  histogram!
plt.figure(figsize=(4,4))

#theta = np.arange(0, np.pi / 2, 0.01)
theta = np.arange(0, 2*np.pi, 0.01)
plt.plot(np.cos(theta),np.sin(theta))
r2 = x*x + y*y
all_points=np.ones(N)
#python trik risanja z maskiranjem
znotraj = np.ma.masked_where(r2 < 1., all_points)
zunaj = np.ma.masked_where(r2 >= 1., all_points)
#izracunaj
uspeh=np.sum(znotraj.mask.astype(int))
xres=uspeh/N*4.
print("ocena {}, pi = {}, rel. napaka {}".format(xres,np.pi,1.-xres/np.pi))

plt.scatter(x, y, s=znotraj, marker='^', c='blue')
plt.scatter(x, y, s=zunaj, marker='o', c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('MC integracija kroga')
plt.show()
#plt.savefig('dist.png')