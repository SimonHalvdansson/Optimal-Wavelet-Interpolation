# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 08:59:47 2020

@author: Simon Halvdansson
"""
import numpy as np
from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pywt
from scipy.fft import fft, ifft, fftfreq, fftshift

alpha = 0
beta = 0
#Difficulties normalizing if we set this too large
mu = 300

def fhat(omega, C):
    return C*e**(omega*(1j*alpha + mu*(beta+1-log(omega))))

def L2(f):
    return integrate.quad(lambda x: abs(f(x))**2, 0, 10)[0]

def bump(x):
    if x < 0.001:
        return 0
    if x > .99:
        return 0
    return e**(1/((2*x-1)**2-1))

def fhat_n(omega, n):
    return 1/sqrt(n)*bump((omega-n**2)/n)
    
fhat_norm = L2(lambda x : fhat(x,1))

fhat_normalized = lambda x : fhat(x, 1/sqrt(fhat_norm))

fig, ax = plt.subplots()
mpl.rcParams["figure.dpi"] = 400

xs = np.linspace(0.0001, 14, 256)
ys = []
ys1 = []
ys2 = []
ys3 = []

for x in xs:
    ys.append(fhat_normalized(x))
    ys1.append(fhat_normalized(x).real)
    ys2.append(fhat_normalized(x).imag)
    ys3.append(abs(fhat_normalized(x)))
    
ax.plot(xs, ys3, label = "|f^|")
ax.plot(xs, ys1, label = "Re(f^)")
ax.plot(xs, ys2, label = "Im(f^)")


ax.legend()

plt.show()

#Let's get the inverse fourier transform of fhat_normalized manually to get
#around domain issues
a,b = -4, 4

os = np.linspace(a, b, 400)
ys_real = []
ys_imag = []
ys = []
for omega in os:
    real = integrate.quad(lambda x : (fhat_normalized(x)*e**(2*pi*1j*x*omega)).real, 0.0001, 14)[0]
    imag = integrate.quad(lambda x : (fhat_normalized(x)*e**(2*pi*1j*x*omega)).imag, 0.0001, 14)[0]
    ys_real.append(real)
    ys_imag.append(imag)
    ys.append(abs(real+imag*1j))
    
fig, ax = plt.subplots()
    
ax.plot(os, ys, label = "|f|")
ax.plot(os, ys_real, label = "Re(f)")
ax.plot(os, ys_imag, label = "Im(f)")

ax.legend()

plt.show()
"""

os = np.linspace(0.1, 115, 600)
ys1 = []
ys3 = []
ys5 = []
ys10 = []

for omega in os:
   ys1.append(fhat_n(omega, 1))
   ys3.append(fhat_n(omega, 3)) 
   ys5.append(fhat_n(omega, 5))
   ys10.append(fhat_n(omega, 10))
   
fig, ax = plt.subplots()

ax.plot(os, ys1, label = "n=1")
ax.plot(os, ys3, label = "n=3")
ax.plot(os, ys5, label = "n=5")
ax.plot(os, ys10, label = "n=10")

ax.legend()
   
#Let's get the inverse fourier transform of fhat_n manually to get
#around domain issues
a,b = -1, 1

os = np.linspace(a, b, 600)
ys_real5 = []
ys_imag5 = []
ys5 = []

ys_real10 = []
ys_imag10 = []
ys10 = []
for omega in os:
    real5 = integrate.quad(lambda x : (fhat_n(x, 5)*e**(2*pi*1j*x*omega)).real, 20, 35)[0]
    imag5 = integrate.quad(lambda x : (fhat_n(x, 5)*e**(2*pi*1j*x*omega)).imag, 20, 35)[0]
    ys_real5.append(real5)
    ys_imag5.append(imag5)
    ys5.append(abs(real5+imag5*1j))
    
    real10 = integrate.quad(lambda x : (fhat_n(x, 10)*e**(2*pi*1j*x*omega)).real, 95, 120)[0]
    imag10 = integrate.quad(lambda x : (fhat_n(x, 10)*e**(2*pi*1j*x*omega)).imag, 95, 120)[0]
    ys_real10.append(real10)
    ys_imag10.append(imag10)
    ys10.append(abs(real10+imag10*1j))
    
fig, ax = plt.subplots()
    
ax.plot(os, ys5, label = "|f_5|")
#ax.plot(os, ys10, label = "|f_10|")
ax.plot(os, ys_real5, label = "Re(f_5)")
ax.plot(os, ys_imag5, label = "Im(f_5)")

ax.legend()

plt.show()




"""









