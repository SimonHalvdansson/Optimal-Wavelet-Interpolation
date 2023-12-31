# -*- coding: utf-8 -*-

import scipy.integrate as integrate
import numpy as np
from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def f(x):
    return 2/(sqrt(3*sqrt(pi)))*(1-x**2)*e**(-x**2/2)





def f(n, x):
    if abs(x) > n:
        return 0
    
    return sqrt(3/2)*(1/sqrt(n)-abs(x)/(n*sqrt(n)))

def plot_func(f, lab = None):
    #fig = plt.figure(figsize=plt.figaspect(0.8)*1.5)
    
    xs = np.linspace(-11, 11, 700)
    ys = []
    
    for x in xs:
        ys.append(f(x))
        
    #plt.axis('off')
        
    plt.plot(xs, ys, label=lab)
    #plt.show()


fig = plt.figure(figsize=plt.figaspect(0.8)*1.0)


plot_func(lambda x: f(2,x), "$f_2$")
plot_func(lambda x: f(5,x), "$f_5$")
plot_func(lambda x: f(10,x), "$f_{10}$")

ax = plt.gca()

ax.legend()


plt.savefig('filename.png', dpi=300)
