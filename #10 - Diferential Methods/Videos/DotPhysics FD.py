# Finite Difference with overrelaxation

"""
Code is not Python but Python cast into Javascript to render in browser
"""


GlowScript 3.0 VPython

g1 = graph(xtitle = "x", ytitle="y", width=400, height=200)
fa = gcurve(color=color.red)
fd = gcurve(color=color.blue, markers=True)

def f(xx):
  C = (11-81/12)/3
  return(xx**4/12 -xx**2+C*xx)

x = 0
dx = 0.01
while x<3:
  fa.plot(x,f(x))
  x = x + dx
  
h = 0.2

xp = []
for i in range(0,3+h,h):
  xp = xp + [i]

#print(xp)

m = 2/3
fp = []
fp2 = []
for i in range(len(xp)):
  fp = fp + [m*xp[i]]
  fp2 = fp2 + [m*xp[i]]
  #fd.plot(xp[i],fp[i])

n = 0
N = 200
while n< N:
  rate(50)
  fdata=[]
  for i in range(1,len(xp)-1):
    fp2[i] = 0.5*(fp[i+1]+fp[i-1]-h**2*(xp[i]**2-2))
  
  for i in range(len(xp)):
    fp[i]=fp2[i]
    fdata=fdata + [[xp[i],fp[i]]]
  fd.data=fdata
  n = n + 1


