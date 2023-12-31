# -*- coding: utf-8 -*-

import numpy as np
from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pywt
from scipy.fft import fft, ifft, fftfreq, fftshift
from prettytable import PrettyTable

"""
Define all wavelets of interest
"""
#Induces ONB https://www.win.tue.nl/casa/meetings/seminar/previous/_index_files/wavelets1.pdf
def haar(x):
    if x < 0:
        return 0
    if x < 1/2:
        return 1
    if x < 1:
        return -1
    return 0

haar_wavelet = [haar, "Haar wavelet", 0, 1, -60, 60, True]

# NOT ONB https://www2.mps.mpg.de/solar-system-school/lectures/wavelets/wavelets.pdf
def mexhat(x):
    return 2/sqrt(3)*pi**(-1/4)*(x**2-1)*e**(-x**2/2)

mexhat_wavelet = [mexhat, "Mexican hat wavelet", -10, 10, -3, 3, True]

#ONB https://www.win.tue.nl/casa/meetings/seminar/previous/_index_files/wavelets1.pdf
def meyer(x):
    t = x-1/2
    if (t == 0.75):
        return meyer(x+0.00001)
    psi1 = (4/3/pi*t*cos(2*pi/3*t) - 1/pi*sin(4*pi/3*t))/(t-16/9*t**3)
    psi2 = (8/3/pi*t*cos(8*pi/3*t) + 1/pi*sin(4*pi/3*t))/(t-64/9*t**3)
            
    return psi1 + psi2
    
meyer_wavelet = [meyer, "Meyer wavelet", -10, 10, -2, 2, True]

#PROB. ONB https://www.researchgate.net/publication/312023250_Shannon_Wavelet_Analysis_with_Applications_A_Survey
def shannon(x):
    if (x == 0.0):
        return shannon(x+0.00001)
    return 2*sin(2*pi*x)/2/x/pi - sin(pi*x)/pi/x
    
shannon_wavelet = [shannon, "Shannon wavelet", -50, 50, -1.5, 1.5, True]

"""
#NOT ONB http://www.maths.lu.se/fileadmin/maths/personal_staff/mariasandsten/TFkompver4.pdf
def morlet(x):
    sigma = 0.1
    c = 1/sqrt(1+e**(-sigma**2)-2*e**(-3*sigma**2/4))
    k = e**(-sigma**2/2)
    return c*pi**(-1/4)*e**(-x**2/2)*(e**(1j*sigma*x)-k)

morlet_wavelet = [morlet, "Morlet wavelet", -20, 20, -3, 3, False]
"""

def hat(func, omega, a, b):
    real = integrate.quad(lambda x : (func(x)*e**(-2*pi*1j*x*omega)).real, a, b, limit = 300)[0]
    imag = integrate.quad(lambda x : (func(x)*e**(-2*pi*1j*x*omega)).imag, a, b, limit = 300)[0]
    
    return real + imag*1j

def sigma1(wavelet_info):
    func = wavelet_info[0]
    t0 = wavelet_info[2]
    t1 = wavelet_info[3]
    
    e1 = e_1(wavelet_info)
    
    return integrate.quad(lambda x : (x-e1)**2*abs(func(x))**2, t0, t1, limit = 300)[0]

def sigma2(wavelet_info):
    t0 = wavelet_info[2]
    t1 = wavelet_info[3]
    f0 = wavelet_info[4]
    f1 = wavelet_info[5]
    
    e2 = e_2(wavelet_info)
    
    return integrate.quad(lambda w : (w-e2)**2*abs(hat(wavelet_info[0], w, t0, t1))**2, f0, f1, limit = 300)[0]

def Sigma1(wavelet_info):
    return sigma1(wavelet_info)*e**(-2*e_2(wavelet_info))

def Sigma2(wavelet_info):
    return sigma2(wavelet_info)

def e_1(wavelet_info):
    func = wavelet_info[0]
    t0 = wavelet_info[2]
    t1 = wavelet_info[3]
    return integrate.quad(lambda x : x*abs(func(x))**2, t0, t1, limit = 100)[0]

def e_2(wavelet_info):
    t0 = wavelet_info[2]
    t1 = wavelet_info[3]
    
    return integrate.quad(lambda w : -log(w)*abs(hat(wavelet_info[0], w, t0, t1))**2, 0.000001, wavelet_info[5], limit=300)[0]

def calc_uncertainty(wavelet_info):
    t0 = wavelet_info[2]
    t1 = wavelet_info[3]
    
    Cf = 0
    
    if t0 < 0:
        Cf = Cf = integrate.quad(lambda w : abs(hat(wavelet_info[0], w, t0, -0.0001))**2/abs(w), 0, wavelet_info[5], limit=300)[0] + integrate.quad(lambda w : abs(hat(wavelet_info[0], w, 0.0001, t1))**2/abs(w), 0, wavelet_info[5], limit=300)[0]
    else:
        Cf = integrate.quad(lambda w : abs(hat(wavelet_info[0], w, t0, t1))**2/abs(w), 0, wavelet_info[5], limit=300)[0]
        
    
    ef = e_2(wavelet_info)
    
    s1 = '\N{GREEK SMALL LETTER SIGMA}\N{SUBSCRIPT ONE}(f)'
    s2 = '\N{GREEK SMALL LETTER SIGMA}\N{SUBSCRIPT TWO}(f)'
    e1 = 'e\N{SUBSCRIPT ONE}(f)'
    e2 = 'e\N{SUBSCRIPT TWO}(f)'
    S1 = '\N{GREEK CAPITAL LETTER SIGMA}\N{SUBSCRIPT ONE}(f)'
    S2 = '\N{GREEK CAPITAL LETTER SIGMA}\N{SUBSCRIPT TWO}(f)'
    CF = 'Cf'
    
    Sig1 = Sigma1(wavelet_info)
    Sig2 = Sigma2(wavelet_info)
    
    t = PrettyTable(['Name', e1, e2, s1, s2, S1, S2, 'S(f)', 'Cf', 'Lower bound'])#, s1, s1, s1])
    t.add_row([wavelet_info[1],
               round(e_1(wavelet_info), 4),
               round(e_2(wavelet_info), 4),
               round(sigma1(wavelet_info), 4),
               round(sigma2(wavelet_info), 4),
               round(Sig1, 4),
               round(Sig2, 4),
               round(Sig1*Sig2, 4),
               round(Cf, 4),
               round(Cf*e**(-2*ef), 4)])
    print(t)


def plot_wavelet(wavelet_info):
    fig, ax = plt.subplots()
    mpl.rcParams["figure.dpi"] = 400
    
    xs = np.linspace(wavelet_info[2], wavelet_info[3], 400)
    os = np.linspace(wavelet_info[4], wavelet_info[5], 400)
    
    if wavelet_info[6]:
        ys = []
        
        for x in xs:
            ys.append(wavelet_info[0](x))
            
        ax.plot(xs, ys)
    else:
        ys1 = []
        ys2 = []
        ys3 = []
        
        for x in xs:
            ys1.append(wavelet_info[0](x).real)
            ys2.append(wavelet_info[0](x).imag)
            ys3.append(abs(wavelet_info[0](x)))
            
        ax.plot(xs, ys1, label = "real")
        ax.plot(xs, ys2, label = "imag")
        ax.plot(xs, ys3, label = "abs")
        ax.legend()

    plt.title(wavelet_info[1])
    plt.show()
    
    fig, ax = plt.subplots()
    
    ys = []
    
    for o in os:
        ys.append(abs(hat(wavelet_info[0], o, wavelet_info[2], wavelet_info[3])))
        
    ax.plot(os, ys)
        
    plt.title(wavelet_info[1] + " transformed")

    plt.show()
    

plot_wavelet(haar_wavelet)
plot_wavelet(mexhat_wavelet)
plot_wavelet(meyer_wavelet)
plot_wavelet(shannon_wavelet)

 

calc_uncertainty(haar_wavelet)
calc_uncertainty(mexhat_wavelet)
calc_uncertainty(meyer_wavelet)
calc_uncertainty(shannon_wavelet)


