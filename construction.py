# -*- coding: utf-8 -*-

import numpy as np
from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random
from timeit import default_timer as timer

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams['savefig.dpi'] = 300

h = 0.6
b = 1.5
eps = 0.0000001

def interpolate(f, x):
    n = floor(x/h)
    
    dx = x - n*h
    old_val = f(n*h + eps)
    next_val = f((n+1)*h + eps)
    
    return old_val * (1-dx/h) + next_val*dx/h

def exp_scale(f):
    return integrate.quad(lambda x : -log(x)*abs(f(x))**2, 0, 20)[0]

def norm(f):
    return integrate.quad(lambda x : abs(f(x))**2, 0, 20)[0]

def smooth_cutoff(f, x):
    if x < b:
        return f(x)
    else:
        c = f(b)
        if x < b+c:
            return c - x + b
        else:
            return 0
        
def pl(f, title=None):
    xs = np.linspace(0, 4, 600)
    ys = []
    for x in xs:
        ys.append(f(x))
        
    plt.plot(xs, ys)
    plt.title(title)
    plt.ylim([0, 1.5])
    plt.show()

f1 = lambda x: x*e**(-(x-1)**2)
n = norm(f1)
f2 = lambda x : f1(x)/sqrt(n)
e2 = exp_scale(f2)
f0 = lambda x : e**(-e2/2)*f2(x*e**(-e2))
pl(f0, r'$f_0$')

fb = lambda x : smooth_cutoff(f0, x)
pl(fb, r'$f_b$')

p0 = lambda x : interpolate(fb, x)
pl(p0, r'$p_0$')

alpha = -0.11695
k = lambda x : p0(x) + alpha*interpolate(lambda t : log(t)*p0(t), x)
pl(k, r'$k$')

n = norm(k)
kn = lambda x : k(x)/sqrt(n)
pl(kn, r'$k/\Vert k \Vert$')




"""
N = norm(g3)
g4 = lambda x : g3(x)/sqrt(N)
pl(g4, "g4")
print(norm(g4))


e2 = exp_scale(g4)
g5 = lambda x : e**(-e2/2)*g4(x*e**(-e2))
pl(g5, "g5")
print(exp_scale(g5))
"""

