# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate

L = 64

#f = np.array([(l-L/2)*math.e**(-(l-L/2)**2/L) for l in range(L)])
#f = np.array([math.e**(-(l-L/2)**2) for l in range(L)])
#f = np.array([math.sin(l/5) for l in range(L)])
f = np.array([l/10-l**2/150+l**3/3000-l**4/1000000-math.e**(-l/17) for l in range(L)])
g = np.array([math.e**(-(l)**6/(L/4)) for l in range(L)])

def f_fun(l):
    return f[l % L]

def g_fun(l):
    return g[l % L]

def gmn(l, m, n):
    return math.e**(-2*math.pi*1j*m*l/L)*g_fun(l-n)

def c(m, n):
    s = 0
    for l in range(L):
        s += f_fun(n)*gmn(l, m, n)
    return s

# m = modulation, n = translation
m = 4
n = 11

xs = []
ys1 = []
ys2 = []
for i in range(L):
    xs.append(i)
    ys1.append(f_fun(i))
    ys2.append(gmn(i, m, n))

plt.plot(xs, ys1, label="f")
plt.plot(xs, ys2, label="gmn")

plt.legend(loc="upper left")

plt.show()


"""
REAL
"""

col = []
for j in range(L):
    col.append(np.real(c(i,j)))
a = np.array(col)

for i in range(1, L):
    col = []
    for j in range(L):
        col.append(np.real(c(i,j)))
    col = np.array(col)
    a = np.column_stack((a, col))

print(a)

zvals = a
img = plt.imshow(zvals)

plt.show()


"""
IMAG
"""

col = []
for j in range(L):
    col.append(np.imag(c(i,j)))
a = np.array(col)

for i in range(1, L):
    col = []
    for j in range(L):
        col.append(np.imag(c(i,j)))
    col = np.array(col)
    a = np.column_stack((a, col))

print(a)

zvals = a
img = plt.imshow(zvals)

plt.show()


"""
ABSOLUTE
"""
col = []
for j in range(L):
    col.append(np.absolute(c(i,j)))
a = np.array(col)

for i in range(1, L):
    col = []
    for j in range(L):
        col.append(np.absolute(c(i,j)))
    col = np.array(col)
    a = np.column_stack((a, col))

print(a)

zvals = a
img = plt.imshow(zvals)

plt.show()





"""
def Sf(j, k):
    s = 0
    for n in range(L):
        for m in range(L):
            s += math.e**(-2*math.pi*1j*m*j/L)*g(j+n)*np.conj(math.e**(-2*math.pi*1j*m*j/L)*g(k+n))
    return s


S = np.array([[Sf(j,k) for k  in range(L)] for j in range(L)])
Si = np.linalg.inv(S)

xs = []
ys = []


rest_f = np.zeros(L)

for l in range(L):
    xs.append(l)
    s = 0
    for n in range(L):
        for m in range(L):
            g_vec = np.array([gmn(l, m, n) for l in range(L)])
            rest_f += c(m, n)*np.real(Si.dot(g_vec))
            


plt.plot(xs, f)
plt.plot(xs, rest_f)
"""















"""
L = 128

def g(l):
    w = 1
    s = 0
    ct = L/2
    for k in range(2):
        s += math.e**(-math.pi*(l+ct+(k-1)*L)**2/(w*L))
    return s * (w*L/2)**(-1/4)

def gmn(l, m, n):
    return math.e**(2*math.pi*1j*n/L)*g(l-m)


f = [math.e**(-(l-L/4)**2/(L/2)) + math.e**(-(l-L/2)**2/(L/2)) + math.e**(-(l-3*L/4)**2/(L/2)) for l in range(L)]


c = []
for m in range(L):
    inner = []
    for n in range(L):
        s = 0
        for l in range(L):
            s += math.e**(2*math.pi*1j*l*m/L)*f[l]*np.conj(g(l-n))
        inner.append(s)
    c.append(inner)


def reconst(l):
    s = 0
    for m in range(L):
        for n in range(L):
            s += c[m][n]*gmn(l, m, n)
            
    return s

xs = []
ys = []

for l in range(L):
    xs.append(l)
    ys.append(abs(reconst(l)))
    
plt.plot(xs, ys)"""