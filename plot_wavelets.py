# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:51:39 2021

@author: Simon Halvdansson
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pywt

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams['savefig.dpi'] = 300

w = pywt.ContinuousWavelet('fbsp')

(phi, psi) = w.wavefun(level=20)

plt.axis('off')
#plt.ylim([-3, 3])  
plt.plot(phi)
