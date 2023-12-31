# -*- coding: utf-8 -*-

import scipy.integrate as integrate
import numpy as np
from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.integrate as integrate

mpl.rcParams["figure.dpi"] = 300

def freal(x):
    real = integrate.quad(lambda w : (fhat(w)*e**(2*pi*1j*w*x)).real, 0, 4)[0]    
    return real

def fimag(x):
    imag = integrate.quad(lambda w : (fhat(w)*e**(2*pi*1j*w*x)).imag, 0, 4)[0]
    return imag

def f2(x):
    real = integrate.quad(lambda w : (fhat(w)*e**(2*pi*1j*w*x)).real, 0, 4)[0]
    imag = integrate.quad(lambda w : (fhat(w)*e**(2*pi*1j*w*x)).imag, 0, 4)[0]
    
    return real**2 + imag**2

def fhat(w):
    return sqrt(2)*2*w**2*e**(-w**2/2)/(sqrt(3*sqrt(pi)))*np.heaviside(w, 0)

def fhat2(w):
    return abs(sqrt(2)*2*w**2*e**(-w**2/2)/(sqrt(3*sqrt(pi)))*np.heaviside(w, 0))**2
    
def ftilde2(s):
    return abs(e**(-s/2)*fhat(e**(-s)))**2
    
N = 4

e1 = integrate.quad(lambda x : x*f2(x), -N, N)[0]
v1 = integrate.quad(lambda x : (x-e1)**2*f2(x), -N, N)[0]

e2 = integrate.quad(lambda w : w*fhat2(w), -N, N)[0]
v2 = integrate.quad(lambda w : (w-e2)**2*fhat2(w), -N, N)[0]

e3 = integrate.quad(lambda s : s*ftilde2(s), -N, N)[0]
v3 = integrate.quad(lambda s : (s-e3)**2*ftilde2(s), -N, N)[0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False)

#ax1.set_aspect('equal')

fig.set_size_inches(13, 2.8)


x1 = np.arange(-1.3, 3, 0.01)
x2 = np.arange(-1.3, 3, 0.01)
x3 = np.arange(-1.3, 3, 0.01)


y1 = []
for i in x1:
    y1.append(f2(i))

y2 = fhat2(x2)
y3 = ftilde2(x3)


ax1.plot(x1, y1, color='black')
ax1.fill_between(x1, 0, y1, where = abs(x1-e1) < sqrt(v1), label=r'$|f(t)|^2$')
ax1.set_title("Time")
ax1.legend()

ax2.plot(x2, y2, color='black')
ax2.fill_between(x2, 0, y2, where = abs(x2-e2) < sqrt(v2), label=r'$|\hat{f}(\omega)|^2$')
ax2.set_title("Frequency")
ax2.legend()

ax3.plot(x3, y3, color='black')
ax3.fill_between(x3, 0, y3, where = abs(x3-e3) < sqrt(v3), label=r'$|\tilde{f}(\sigma)|^2$')
ax3.set_title("Scale")
ax3.legend()






"""
xs = np.linspace(-5, 5, 400)
ys = []
for x in xs:
    ys.append(f(x)**2)

plt.plot(xs, ys)

fig = plt.figure(figsize=plt.figaspect(0.8)*1.0)

plt.savefig('filename.png', dpi=300)"""
