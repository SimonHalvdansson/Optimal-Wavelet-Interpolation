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

N = 80
S = 3.5

xs = []
for n in range(N+1):
    xs.append(n*S/N)

def get_random_function_in_fourier_domain():
    phi = [0]
    for n in range(1,N):
        if (random.random() < 0.5):
            phi.append((np.random.normal())*0.01)
        else:    
            phi.append(np.random.normal())
            
    phi.append(0)
    phi = np.array(phi)

    return phi/sqrt(l2_norm(phi))

def func_to_phi(f):
    phi = [0]
    for n in range(1,N):
        phi.append(f(xs[n]))

    phi.append(0)
    
    phi = np.array(phi)
    
    return phi/sqrt(l2_norm(phi))

def get_fhat(phi):
    def func(omega):
        for i in range(N):
            if xs[i] <= omega and xs[i+1] > omega:
                deltaXMax = xs[i+1] - xs[i]
                deltaY = phi[i+1] - phi[i]
                deltaX = omega - xs[i]
                return phi[i] + deltaY * deltaX/deltaXMax
    return func

def inverse_fourier_fast(phi, omega):
    #https://www.wolframalpha.com/input/?i=int+from+a+to+b+of+%28d+%2B+cx%29*e%5E%28i*2*pi*x*omega%29dx
    s = 0
    for i in range(N):
        #int from a to b of (cx+d)*e^(i 2 pi x omega) dx = BIG thing, lets sort out a,b,c,d
        a = xs[i]
        b = xs[i+1]
        c = (phi[i+1] - phi[i])/(b-a)
        d = phi[i] - c*a
        
        s += (1j*e**(1j*2*pi*a*omega)*(2*pi*omega*(a*c+d) + 1j*c) + e**(1j*2*pi*b*omega)*(c-2*1j*pi*omega*(b*c+d)))/(4*pi**2*omega**2)
    
    return s

def inverse_fourier(phi):
    return lambda omega : inverse_fourier_fast(phi, omega)

def l2_norm(phi):
    #https://www.wolframalpha.com/input/?i=int+from+a+to+b+of+%28d+%2B+cx%29%5E2, real and imag separately
    L2Norm = 0
    for i in range(N):
        a = xs[i]
        b = xs[i+1]
        c = (phi[i+1] - phi[i])/(b-a)
        d = phi[i] - c*a
        
        dr = d.real
        di = d.imag
        cr = c.real
        ci = c.imag
        
        real2 = 0
        imag2 = 0
        
        if (cr != 0):
            real2 += ((b*cr+dr)**3 - (a*cr+dr)**3)/(3*cr)
        
        if (ci != 0):
            imag2 += ((b*ci+di)**3 - (a*ci+di)**3)/(3*ci)
        
        L2Norm += real2 + imag2
    return L2Norm
 
def loss(phi, w1 = 1, w2 = 1, w3 = 1):
    e1 = 0   
    e2 = 0
    
    sigma1 = 0
    sigma2 = 0
    sigma3 = 0
    
    for i in range(N):
        a = xs[i]+0.00000001
        b = xs[i+1]+0.00000001
        c = (phi[i+1] - phi[i])/(b-a)
        d = phi[i] - c*a
        
        dr = d
        cr = c
        
        e2 -= (b*(6*log(b)*(b**2*cr**2+3*b*cr*dr+3*dr**2)-2*b**2*cr**2-9*b*cr*dr-18*dr**2) - a*(6*log(a)*(a**2*cr**2+3*a*cr*dr+3*dr**2)-2*a**2*cr**2-9*a*cr*dr-18*dr**2))/18

    A = e2

    for i in range(N):
        a = xs[i]+0.00000001
        b = xs[i+1]+0.00000001
        c = (phi[i+1] - phi[i])/(b-a)
        d = phi[i] - c*a
        
        dr = d
        cr = c

        sigma1 += c**2*(b-a)

        sigma2 += (-2*(2 - 6*A + 9*A**2)*(a**3 - b**3)*cr**2 - 27*(1 + 2*(A**2-A))*(a**2 - b**2)*cr*dr - 54*(2 + (A**2-2*A))*(a - b)*dr**2 - 6*a*log(a)*(2*a**2*(3*A-1)*cr**2 + 9*a*(2*A-1)*cr*dr + 18*(A-1)*dr**2 + 3*(a**2*cr**2 + 3*a*cr*dr + 3*dr**2)*log(a)) + 6*b*(2*(3*A-1)*b**2*cr**2 + 9*(2*A-1)*b*cr*dr + 18*(A-1)*dr**2)*log(b) + 18*b*(b**2*cr**2 + 3*b*cr*dr + 3*dr**2)*log(b)**2)/54
        
        sigma3 += -0.5*c**2*(a**2 - b**2)
        sigma3 += 2*c*d*(b-a)
        sigma3 += d**2*log(b/a)
    
    return w1*e**(-2*e2)*sigma1 + w2*sigma2 + w3*e**(-e2)*sigma3

def plot_freq(f, current_loss, show = True, save = False, name = ""):
    xs = np.linspace(0.0000001, S-0.00001, 200)
        
    ys = []
    
    for x in xs:
        ys.append(f(x))

    plt.plot(xs, ys)
    #plt.title("Frequency")
    plt.ylim([-1, 2])  
    plt.axis('off')
    
    if save:
        plt.savefig(name)
    if show:
        plt.show()
    
def plot_time(f, current_loss, show = True, save = False, name = ""):
    xs = np.linspace(-3.0001, 3, 400)
        
    ys_r = []
    ys_i = []
    ys_a = []
    
    for x in xs:
        val = f(x)
        ys_r.append(val.real)
        ys_a.append(abs(val))

    plt.plot(xs, ys_r)
    #plt.title("Time")
    plt.ylim([-1, 2])    
    plt.axis('off')
    
    if save:
        plt.savefig(name)
    if show:
        plt.show()

def get_normalized_alteration(old_phi, scaling_factor = 1):
    new_phi = old_phi*0
    for i in range(1,N):
        if (random.random() > 0.5):
            new_phi[i] = np.random.normal()*np.random.normal()*scaling_factor*0.01
            
    new_phi = new_phi + old_phi
    return new_phi / sqrt(l2_norm(new_phi))

def bench():
    start = timer()
    for i in range(1000):
        loss(get_random_function_in_fourier_domain())
    end = timer()
    
    return end-start


def run(max_steps, name_time = None, name_freq = None, w1 = 1, w2 = 1, w3 = 1, save = False):
    smallest_loss = 1000
    best_phi = []
            
    f = lambda x : e**(-(x-1)**2/0.14)
    best_phi = func_to_phi(f)
    smallest_loss = loss(best_phi, w1, w2, w3)
    
    i = 0
    last_improvement = 0
    while i < max_steps and i < last_improvement + 4000:
        new_phi = get_normalized_alteration(best_phi, 0.1)
        new_loss = loss(new_phi, w1, w2, w3)
        if new_loss < smallest_loss:
            smallest_loss = new_loss
            best_phi = new_phi
            last_improvement = i
            print("{}%: At step {}, new lowest loss: {}".format(i*100/max_steps, i, smallest_loss))
            
        i += 1
        
    e2 = 0
    
    for i in range(N):
        a = xs[i]+0.00000001
        b = xs[i+1]+0.00000001
        c = (best_phi[i+1] - best_phi[i])/(b-a)
        d = best_phi[i] - c*a
        
        dr = d
        cr = c
        
        e2 -= (b*(6*log(b)*(b**2*cr**2+3*b*cr*dr+3*dr**2)-2*b**2*cr**2-9*b*cr*dr-18*dr**2) - a*(6*log(a)*(a**2*cr**2+3*a*cr*dr+3*dr**2)-2*a**2*cr**2-9*a*cr*dr-18*dr**2))/18

    time_off_scale = inverse_fourier(best_phi)
    time_on_scale = lambda x : e**(e2/2)*time_off_scale(x*e**(e2))
    
    freq_off_scale = get_fhat(best_phi)
    freq_on_scale = lambda x : e**(-e2/2)*freq_off_scale(x*e**(-e2))
    
    plot_freq(freq_on_scale, smallest_loss, True, save, name_freq)
    plot_time(time_on_scale, smallest_loss, True, save, name_time)

i = 1
frames = 1000

for _ in range(frames + 1):
    a = 0.5*(erf(4*i/frames-2) + 1)
    
    run(5*10**5, "video/{}time.png".format(i), "video/{}freq.png".format(i), a, 1 - a, 1, True)
    i += 1

#run(10, 1, 0.01, 1, True)