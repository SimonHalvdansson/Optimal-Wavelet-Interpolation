# -*- coding: utf-8 -*-

import scipy.integrate as integrate
import numpy as np
from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt

N = 128

mpl.rcParams["figure.dpi"] = 300


def g1(x):
    b = 0.4
    
    return pi**(-1/4)*b**(-1/4)*e**(-x**2/(2*b**2))
    if abs(x) < 0.5:
        return 1
    else:
        return 0
    
def g2(x):
    return (1-x**2)*e**(-x**2/2)

def g3(x):
    if x < 0:
        return 0
    elif x < 0.5:
        return 1
    elif x < 1:
        return -1
    else:
        return 0

def f1(t):
    if 5 < t and t < 15:
        return sin(2*pi*t)
    elif t < 20:
        return sin(2*pi*t/2)*0.5
    else:
        return sin(2*pi*t/2)
    
def f2(x):
    t = x/2
    if t < 20:
        return sin(t*abs(t)**0.7/3)
    elif t < 40:
        return e**(-(t-25)**2/2) + 2*e**(-(t-30)**2/0.2) + e**(-(t-32)**2/0.2) + 2*e**(-(t-34)**2/0.2) + e**(-(t-36)**2/0.2)
    elif t < 46:
        return sin((t-40)**2)
    else:
        return g2(t-56)
    #if t < 13 or t > 19:
    #    return 0
    #return 1
    #return sin(3*t*abs(t)**1)
    
    #if t < 16:
    #    return sin(t)
    #return sin(0.3*t)
    #if t < 16:
    #    return 0
    #return 1
    #return e**(-(t-16)**2/50)#*sin(t)

def f3(t):
    return sin(2*t)*e**(-abs(t)/5)

def f4(t):
    if t < 0:
        return 0
    return sin(abs(t)**1.2)
    

def stft(x, omega):
    r = integrate.quad(lambda t: (f1(t)*g1(t-x)*e**(-2*pi*1j*t*omega)).real, x-7, x+7)[0]
    i = integrate.quad(lambda t: (f1(t)*g1(t-x)*e**(-2*pi*1j*t*omega)).imag, x-7, x+7)[0]
    return r+1j*i

def wavelet_transform(x, s):
    return e**(-s/2)*integrate.quad(lambda t: f4(t)*g1((t-x)/e**s), x-30, x+30)[0]

def plot_spec():
    heatmap = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            heatmap[j,i] = abs(stft(i/4, (j/N*2)))**2
    
    plt.imshow(heatmap, cmap='plasma', interpolation='nearest', origin='lower')
    plt.axis('off')
    
    plt.show()
    
def plot_spec_ideal():
    heatmap = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i < 5:
                if (j == 1):
                    heatmap[j*6,i] = 1/2
                    
            elif i < 15:
                if (j == 2):
                    heatmap[j*6,i] = 1
            elif i < 20:
                if (j == 1):
                    heatmap[j*6,i] = 1/2
            else:
                if (j == 1):
                    heatmap[j*6,i] = 1
            
            
    plt.imshow(heatmap, cmap='plasma', interpolation='nearest', origin='lower')
    plt.axis('off')
    
    plt.show()
    

def plot_wt():
    heatmap = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            heatmap[j,i] = abs(wavelet_transform(i, (j/N)*7-1.7))**2
    
    plt.imshow(heatmap, cmap='plasma', interpolation='nearest', origin='lower')
    plt.axis('off')
    
    plt.show()

def plot_func(f):
    fig = plt.figure(figsize=plt.figaspect(0.8)*1.5)
    
    xs = np.linspace(0, N, 19700)
    ys = []
    
    for x in xs:
        ys.append(f(x))
        
    plt.axis('off')
        
    plt.plot(xs, ys)
    #plt.savefig('filename.png', dpi=300)
    plt.show()

plot_wt()
#plot_spec()
plot_func(f4)



