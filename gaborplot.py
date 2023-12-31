# -*- coding: utf-8 -*-
import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

"""
N = 512
threshold = 0.002
modulation_steps = 5
ang_rot = 110
ang_pitch = 50
"""

N = 512
threshold = 0.002
modulation_steps = 6
ang_rot = 120
ang_pitch = 65


xs = []
ysr = []

for x in range(N):
    xs.append(x)

for p in range(modulation_steps):
    pr = []
    for t in range(4):
        tr = []
        for x in range(N):
            v = math.e**(-(x-t*N/4-N/8)**2/(N/2))
            avr = np.real(v*math.e**(2*math.pi*1j*x*p/N*9))
            if np.abs(v) > threshold:
                tr.append(avr)
            else:
                tr.append(np.nan)
            
        pr.append(tr)
    
    ysr.append(pr)
    
    
with plt.style.context('seaborn-notebook'):
    fig = plt.figure(figsize=plt.figaspect(0.5)*2.0)
    ax = fig.gca(projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    mpl.rcParams["figure.dpi"] = 300
    
    colors = ['k', 'b', 'r', 'g', 'm', 'k', 'k', 'k', 'k', 'k', 'k', 'k']
    #colors = ['#0000d6', '#1c00db', '#3d00e0', '#5300e8', '#6002ee', '#7e3ff2', '#9965f4', '#b794f6']
    
    for p in range(modulation_steps):
        for t in range(0, 4):
            ax.plot(np.zeros(N) + p, xs, ysr[p][t], colors[p])
            

    # make labels
    ax.view_init(ang_pitch, ang_rot)
            
    plt.show()
