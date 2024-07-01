import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import timeit
import cProfile as cprf
import pstats
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def H_0(N):
    h_0 = np.zeros((N, N))
    for i in range(N):
        h_0[i, i] = (i + 0.5)

    return h_0


def Q(N):
    q = np.zeros((N, N))
    q[0, 1] = 0.5 * 2. ** 0.5
    q[N - 1, N - 2] = 0.5 * (N + N - 2) ** 0.5
    for i in range(1, N - 1):
        for j in range(i - 1, i + 2, 2):
            q[i, j] = 0.5 * (i + j + 1.) ** 0.5

    return q


def Q4_1(N):
    q4_1 = np.matmul(Q(N), Q(N))
    for i in range(2):
        q4_1 = np.matmul(q4_1, Q(N))
    return q4_1


def Q2(N):
    q2 = np.zeros((N, N))
    q2[0, 0] = 0.5
    q2[0, 2] = 0.5 * 2. ** 0.5
    q2[1, 1] = 0.5 * 3.
    q2[1, 3] = 0.5 * 6. ** 0.5
    q2[N - 2, N - 4] = 0.5 * ((N - 3.) * (N - 2.)) ** 0.5
    q2[N - 2, N - 2] = 0.5 * (2. * (N - 2.) + 1)
    q2[N - 1, N - 3] = 0.5 * ((N - 2.) * (N - 1.)) ** 0.5
    q2[N - 1, N - 1] = 0.5 * (2. * (N - 1.) + 1)
    for i in range(2, N - 2):
        for j in range(i - 2, i + 3, 2):
            q2[i, j] = 0.5 * ((j * (j - 1.)) ** 0.5 * (i - j) * (-0.5) *
                              (i - (j + 2.)) * (-0.25) + (2. * j + 1.) *
                              (i - (j + 2.)) * (-0.5) * (i - (j - 2.)) * 0.5 +
                              ((j + 1.) * (j + 2.)) ** 0.5 * (i - j) * 0.5 * (i - (j - 2.)) * 0.25)

    return q2


def Q4_2(N):
    q4_2 = np.matmul(Q2(N), Q2(N))
    return q4_2


def Q4_4(N):
    q4 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (j + 4 == i):
                q4[i, j] = (0.5) ** 4. * (2. ** 4. * i * (i - 1.) * (i - 2.) * (i - 3.))**0.5
            if (j + 2 == i):
                q4[i, j] = (0.5) ** 4. * (2. ** i * i * (i - 1.) / (2. ** j)) ** 0.5 *\
                           4. * (2. * j + 3.)
            if (j == i):
                q4[i, j] = (0.5) ** 4. * 12. * (2. * j ** 2 + 2. * j + 1.)
            if (j - 2 == i):
                q4[i, j] = (0.5) ** 4. * (2. ** i / (2. ** j * j * (j - 1.))) ** 0.5 *\
                           16. * j * (2. * j ** 2. - 3. * j + 1.)
            if (j - 4 == i):
                q4[i, j] = (0.5) ** 4. * (1. / (2. ** 4. * j * (j - 1.) *
                                                (j - 2.) * (j - 3.))) ** 0.5 *\
                           16. * j * (j ** 3. - 6. * j ** 2. + 11. * j - 6.)

    return q4


# Jakobijeva metoda z vsoto S
def jacobi(a, tol=1e-8):
    su = 0.
    for q0 in range(len(a)):
        for q1 in range(len(a)):
            if q0 != q1:
                su = su + np.abs(a[q0, q1]) ** 2.

    def maxElem(a):
        n = len(a)
        aMax = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if np.abs(a[i, j]) >= aMax:
                    aMax = np.abs(a[i, j])
                    k = i
                    l = j

        return aMax, k, l

    def rotate(a, p, k, l):  # rotacija
        n = len(a)
        aDiff = a[l, l] - a[k, k]
        if np.abs(a[k, l]) < np.abs(aDiff) * 1e-36:
            t = a[k, l] / aDiff
        else:
            phi = aDiff / (2.0 * a[k, l])
            t = 1.0 / (np.abs(phi) + np.sqrt(phi ** 2 + 1.0))
            if phi < 0.0:
                t = -t
        c = 1.0 / np.sqrt(t ** 2. + 1.0)
        s = t * c
        tau = s / (1.0 + c)
        temp = a[k, l]
        a[k, l] = 0.0
        a[k, k] = a[k, k] - t * temp
        a[l, l] = a[l, l] + t * temp
        for i in range(k):  # za i < k
            temp = a[i, k]
            a[i, k] = temp - s * (a[i, l] + tau * temp)
            a[i, l] = a[i, l] + s * (temp - tau * a[i, l])
        for i in range(k + 1, l):  # za k < i < l
            temp = a[k, i]
            a[k, i] = temp - s * (a[i, l] + tau * a[k, i])
            a[i, l] = a[i, l] + s * (temp - tau * a[i, l])
        for i in range(l + 1, n):  # za i > l
            temp = a[k, i]
            a[k, i] = temp - s * (a[l, i] + tau * temp)
            a[l, i] = a[l, i] + s * (temp - tau * a[l, i])
        for i in range(n):
            temp = p[i, k]
            p[i, k] = temp - s * (p[i, l] + tau * p[i, k])
            p[i, l] = p[i, l] + s * (temp - tau * p[i, l])

    n = len(a)
    maxRot = 5 * (n**2)  # meja za  stevilo rotacij
    p = np.identity(n) * 1.0
    for i in range(maxRot):
        aMax, k, l = maxElem(a)
        if su <= tol:
            return np.diagonal(a), p, i
        rotate(a, p, k, l)
        su = su - 2. * (aMax) ** 2.
    'Jakobijeva metoda ne konvergira'


# Householderjeva redukcija do tridiagonalne matrike
def housetrid(ain):

    def sign(aa,bb):
        if bb < 0:
            return -abs(aa)
        else:
            return abs(aa)

    n=len(ain)
    d=np.zeros(n)
    e=np.zeros(n)
    b=np.copy(ain)
    a=np.copy(ain)  # za shranjevanje lastnih vektorjev
    eia=np.eye(n)
    for i in range(n-1,0,-1):
        l=i-1
        h=0.
        scale=0.
        if l>0:
            u=np.copy(b[i,:l+1])
            for el in u:
                scale += abs(el)
            #print "\n ere",i,"scale=",scale
            if scale == 0.:
                e[i] = b[i,l]
            else:
                h=np.dot(u,u)
                f=a[i,l]
                g=-sign(np.sqrt(h),f)
                e[i]=g
                a[i,l]=f-g # u eigenvec store
                u[l] = u[l]+np.sign(u[l])*np.linalg.norm(u)
                h=np.linalg.norm(u)
                u.resize(n)
                w=u/h
                #u=np.append(u,0.)
                #print i,"u",u
                # print i,"u=",u,"dot=",h,"w=",w,"dot=",np.dot(w,w)
                pp=np.identity(n)-2*np.outer(w,w)
                eia=np.dot(pp,eia)
                # print "PP\n",pp,"\n"
                # print np.dot(np.dot(pp,b),pp),"\n"
                #print np.dot(pp,np.dot(b,pp))
                f=0
                a[:,i]=w # eigenvec store
                p=np.dot(b,w)
                # print "p\n",p
                k=np.dot(w,p)
                # print "k=",k,"\n"
                q=p-k*w
                # print "q\n",q
                a -= 2*(np.outer(q,w)+np.outer(w,q))
                #print "a\n",a
                # print "outer\n",2*(np.outer(q,w)+np.outer(w,q)),"\n"
                b -= (2*np.outer(q,w)+2*np.outer(w,q))
                # print "a\n",b
        else:
            #print "aaa",b[i,l]
            e[i]=b[i,l]
        d[i]=h
    d[0]=0.
    e[0]=0.
    for i in range(0,n):
        l=i-1
        if d[i] !=0.:
            for j in range(0,l):
                g=0.
                for k in range(0,l):
                    g +=a[i,k]*a[k,j]
                for k in range(0,l):
                    a[k,j] -= g*a[k,i]
        d[i]=b[i,i]
        a[i,i]=1.0
        for j in range(0,l):
            a[i,j]=0
            a[j,i]=0
    return d,e,eia.T


# QL algoritem z menjavami
def qlnr(d, e, z, tol=1.0e-8):
    n=len(d)
    e=np.roll(e,-1) #reorder
    itmax=1000
    for l in range(n):
        for iter in range(itmax):
            m=n-1
            for mm in range(l,n-1):
                dd=abs(d[mm])+abs(d[mm+1])
                if abs(e[mm])+dd == dd:
                    m=mm
                    break
                if abs(e[mm]) < tol:
                    m=mm
                    break
            if iter==itmax-1:
                print "too many iterations",iter
                exit(0)
            if m!=l:
                g=(d[l+1]-d[l])/(2.*e[l])
                r=np.sqrt(g*g+1.)
                g=d[m]-d[l]+e[l]/(g+np.sign(g)*r)
                s=1.
                c=1.
                p=0.
                for i in range(m-1,l-1,-1):
                    f=s*e[i]
                    b=c*e[i]
                    if abs(f) > abs(g):
                        c=g/f
                        r=np.sqrt(c*c+1.)
                        e[i+1]=f*r
                        s=1./r
                        c *= s
                    else:
                        s=f/g
                        r=np.sqrt(s*s+1.)
                        e[i+1]=g*r
                        c=1./r
                        s *= c
                    g=d[i+1]-p
                    r=(d[i]-g)*s+2.*c*b
                    p=s*r
                    d[i+1]=g+p
                    g=c*r-b
                    for k in range(n):
                        f=z[k,i+1]
                        z[k,i+1]=s*z[k,i]+c*f
                        z[k,i]=c*z[k,i]-s*f
                d[l] -= p
                e[l]=g
                e[m]=0.
            else:
                break
    return d,z


# lambda
lambd = 0.3
N = 100
H = H_0(N) + 0.1 * Q4_4(N) - 1.5 * Q2(N)
print H, '\n'
w, v = np.linalg.eigh(H)
wdtrid, whtrid, vtrid = housetrid(H)
wql, vql = qlnr(wdtrid, whtrid, vtrid)
wql = np.sort(wql)
# wj, vj, st = jacobi(H)
# wj = np.sort(wj)
print wql[:10],'\n', w[:10]
# print w[0:10]
# print st
# print Q4_1, '\n', Q4_2, '\n', Q4_4


# tq = timeit.timeit('Q4_1(1000)', setup='from __main__ import Q4_1, Q', number=1)
# tq2 = timeit.timeit('Q4_2(1000)', setup='from __main__ import Q4_2, Q2', number=1)
# tq4 = timeit.timeit('Q4_4(1000)', setup='from __main__ import Q4_4', number=1)
# print tq, tq2, tq4
# casovne zahtevnosti
# cprf.run('np.linalg.eigh(H)')
# cprf.run('housetrid(H)')
# cprf.run('qlnr(wdtrid, whtrid, vtrid)')
# cprf.run('jacobi(H)')
# t_trid = timeit.timeit('housetrid(H)',
#                   setup='from __main__ import housetrid, H', number=1)
# t_qlnr = timeit.timeit('qlnr(wdtrid, whtrid, vtrid)',
#                   setup='from __main__ import qlnr, wdtrid, whtrid, vtrid', number=1)
# t_jac = timeit.timeit('jacobi(H)',
#                   setup='from __main__ import jacobi, H', number=1)
# print t_vgrajen


# print 'lastne vrednosti:\n', w, '\nvektorji:\n', v,# '\nproba:\n', \
    #(np.dot(H, v[:, 0])) / w[0]
# print '\nHouseholder trid.', '\nlastne vrednosti:\n', wql, \
#     '\nvektorji:\n', vql
# print '\nJakobi najvecji element\n', 'lastne vrednosti:\n', wj, '\nlastni vektorji\n', vj


# p = -1,1 l.vetktorji
# def lastne_funkcije(N, p, n, x):
#     value = 0
#     for i in range(N):
#         h = special.hermite(i)
#         value = value + (2. ** i * np.math.factorial(i) * np.pi ** 0.5) ** (-0.5) * \
#                 np.exp(-x ** 2. * 0.5) * h(x) * v[i, n] * p
#
#     return value


# lastne energije v odvisnosti od dimenzij matrike N
# E = np.zeros((60, 5))
# Dm = np.zeros(60)
# for i in range(60):
#     Hd = H_0(i + 5) + lambd * Q4_4(i + 5)
#     a, b, c = housetrid(Hd)
#     e, f = qlnr(a, b, c)
#     av, bv = np.linalg.eigh(Hd)
#     aj, bj, stj = jacobi(Hd)
#     e = np.sort(e)
#     E[i, :] = e[0:5]
#     Dm[i] = (i + 5)
#     print av[0:5], e[0:5]

e_n = np.zeros((5,5))
gi = np.linspace(0., 1., 5)
# print gi
for i in range(5):
    H = H_0(N) + gi[i] * Q4_4(N)
    w, v = np.linalg.eigh(H)
    e_n[i, :] = w[0:5]
print e_n[0,:]

# plt.title('Prvih nekaj lastnih funkcij')
# plt.xlabel('$x$')
# plt.ylabel('$|n\!>(q)\, , U(q)$')
# x = np.linspace(-5., 5., 1000)
# plt.xlim(-5., 5.)
# plt.ylim(-10.5, 5.)
# plt.plot(x, -2. * x ** 2 + 0.1 * x ** 4, 'k-.', alpha=0.4, label='$U(q)$')
# plt.plot(x, lastne_funkcije(N, 1, 0, x) + -3, '-', label='$n = 0$')
# plt.plot(x, lastne_funkcije(N, 1, 1, x) + -1, '-', label='$n = 1$')
# plt.plot(x, lastne_funkcije(N, 1, 2, x) + 1, '-', label='$n = 2$')
# plt.plot(x, lastne_funkcije(N, 1, 3, x) + 3, '-', label='$n = 3$')
# plt.plot(x, lastne_funkcije(N, -1, 0, x) + -3, 'k--', alpha=0.2)  # , label='$n_- = 0$')
# plt.plot(x, lastne_funkcije(N, -1, 1, x) + -1, 'k--', alpha=0.2)  # , label='$n_- = 1$')
# plt.plot(x, lastne_funkcije(N, -1, 2, x) + 1, 'k--', alpha=0.2)  # , label='$n_- = 2$')
# plt.plot(x, lastne_funkcije(N, -1, 3, x) + 3, 'k--', alpha=0.2)  # , label='$n_- = 3$')
# plt.hlines([-3, -1, 1, 3], -5., 5., color='k', linewidth=0.5, linestyle=':')
# plt.axhline(y=0, color='k', linewidth=0.5, linestyle=':')
# plt.axvline(x=0, color='k', linewidth=0.5, linestyle=':')
# plt.legend(loc='lower left')
# plt.grid(linestyle=':')

# plt.plot(Dm, E[:, 0], '-', label='$E_0$')
# plt.plot(Dm, E[:, 1], '-', label='$E_1$')
# plt.plot(Dm, E[:, 2], '-', label='$E_2$')
# plt.plot(Dm, E[:, 3], '-', label='$E_3$')
# plt.plot(Dm, E[:, 4], '-', label='$E_4$')

# plt.matshow(H)

# fig, axs = plt.subplots(nrows=3)
# fig.subplots_adjust(hspace=0.5)
#
# axs[0].set_title('Jacobijeva metoda metoda, matrika $q_{ij}^{(4)}$')
# axs[0].set_xlabel('$N$')
# axs[0].set_ylabel('$max|E_{nv} - E|$')
# axs[1].set_xlabel('$N$')
# axs[1].set_ylabel('$max|E_{nv} - En|$')
# axs[2].set_xlabel('$N$')
# axs[2].set_ylabel('$t$')
#
# axs[0].semilogy()
# Dim, Nap = np.meshgrid(dim[:i+1], nap[:i+1])
# Casz = np.diag(casz[:i+1])
# cm = plt.cm.get_cmap('Oranges')
# im0 = axs[0].pcolormesh(Dim, Nap, Casz)
# cbar = fig.colorbar(im0, ax=axs[0])
# cbar.set_label('$t$')
#
# axs[1].semilogy()
# im1 = axs[1].plot(dim[0:i+1], nap[0:i+1], '-')
# axs[1].grid(linestyle=':')
#
# im2 = axs[2].plot(dim[0:i+1], casz[0:i+1], 'r-')
# axs[2].grid(linestyle=':')

plt.title('$E_n$ v odvisnosti od $\lambda$.')
plt.xlabel('$\lambda$')
plt.ylabel('$E_n$')
plt.plot(gi, e_n[:, 0], '.-', label='$n = 0$')
plt.plot(gi, e_n[:, 1], '.-', label='$n = 1$')
plt.plot(gi, e_n[:, 2], '.-', label='$n = 2$')
plt.plot(gi, e_n[:, 3], '.-', label='$n = 3$')
plt.plot(gi, e_n[:, 4], '.-', label='$n = 4$')
plt.legend(loc='upper left')

plt.savefig('lve_lambda.pdf')
plt.show()