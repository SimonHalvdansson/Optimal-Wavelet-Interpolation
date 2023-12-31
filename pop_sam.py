# -*- coding: utf-8 -*-

import scipy.integrate as integrate
import numpy as np
from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.integrate as integrate

mpl.rcParams["figure.dpi"] = 300

def f(x):
    return (1-x**2)*e**(-x**2/2)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
fig.set_size_inches(10, 2.5)

N = 5

x = np.arange(-N, N, 0.01)

y1 = f(0.8*x)
y2 = -f(3*x-4)*0.7
y3 = y1 + y2


ax1.plot(x, y1, color='black')
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)

ax2.plot(x, y2, color='black')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
#ax2.get_yaxis().set_text("+")

ax3.plot(x, y3, color='black')
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

fig, ax = plt.subplots(3, 3, sharex=False, squeeze=False, sharey=True)

plt.subplots_adjust(wspace=0.05, hspace=0.05)
fig.set_size_inches(10, 10)

N = 8

x = np.arange(-N, N, 0.01)

y1 = f(0.8*x)
y2 = -f(3*x-4)*0.7
y3 = y1 + y2

i = -1
j = -1
for cols in ax:
    j = -1
    for a in cols:
        a.plot(x, 2**(i/2)*f((x-j*2.5)*2**i), color='black')
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        j += 1
    i += 1






"""
xs = np.linspace(-5, 5, 400)
ys = []
for x in xs:
    ys.append(f(x)**2)

plt.plot(xs, ys)

fig = plt.figure(figsize=plt.figaspect(0.8)*1.0)

plt.savefig('filename.png', dpi=300)"""
