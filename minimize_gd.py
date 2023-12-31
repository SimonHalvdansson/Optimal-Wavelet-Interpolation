# -*- coding: utf-8 -*-

import numpy as np
from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pywt
from scipy.fft import fft, ifft, fftfreq, fftshift

N = 10
S = 4

xs = []
for n in range(N+1):
    xs.append(n*S/N)

def func_to_phi(f):
    phi = [0+0j]
    for n in range(1,N):
        phi.append(f(xs[n]))

    phi.append(0+0j)
    
    phi = np.array(phi)
    
    return phi/sqrt(l2_norm(phi))

def l2_norm(phi):
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

def inverse_fourier_old(f):
     real = lambda x : integrate.quad(lambda w : (f(w)*e**(2*pi*1j*x*w)).real, 0, S, limit = 200)[0]
     imag = lambda x : integrate.quad(lambda w : (f(w)*e**(2*pi*1j*x*w)).imag, 0, S, limit = 200)[0]
     return lambda x : real(x) + 1j*imag(x)
 
def inverse_fourier_fast(phi, omega):
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
    
def get_fhat(phi):
    def func(omega):
        for i in range(N):
            if xs[i] <= omega and xs[i+1] > omega:
                deltaXMax = xs[i+1] - xs[i]
                deltaY = phi[i+1] - phi[i]
                deltaX = omega - xs[i]
                return phi[i] + deltaY * deltaX/deltaXMax
    return func
    

def loss(phi, useNew = True):
    fhat = get_fhat(phi)
    f = inverse_fourier(phi)
    
    e2 = 0
    #https://www.wolframalpha.com/input/?i=f%28x%29+%3D+%28x+%28-18+d%5E2+-+9+c+d+x+-+2+c%5E2+x%5E2+%2B+6+%283+d%5E2+%2B+3+c+d+x+%2B+c%5E2+x%5E2%29+Log%5Bx%5D%29%29%2F18%2C+what+is+f%28b%29-f%28a%29
    for i in range(N):
        a = xs[i]+0.00000001
        b = xs[i+1]+0.00000001
        c = (phi[i+1] - phi[i])/(b-a)
        d = phi[i] - c*a
        
        dr = d.real
        di = d.imag
        cr = c.real
        ci = c.imag
        
        real2 = 0
        imag2 = 0
        
        real2 += (b*(6*log(b)*(b**2*cr**2+3*b*cr*dr+3*dr**2)-2*b**2*cr**2-9*b*cr*dr-18*dr**2) - a*(6*log(a)*(a**2*cr**2+3*a*cr*dr+3*dr**2)-2*a**2*cr**2-9*a*cr*dr-18*dr**2))/18
        imag2 += (b*(6*log(b)*(b**2*ci**2+3*b*ci*di+3*di**2)-2*b**2*ci**2-9*b*ci*di-18*di**2) - a*(6*log(a)*(a**2*ci**2+3*a*ci*di+3*di**2)-2*a**2*ci**2-9*a*ci*di-18*di**2))/18
        e2 -= (real2 + imag2)
    
    sigma1 = integrate.quad(lambda x : abs(x)**2*abs(f(x))**2, -np.inf, np.inf, limit = 300, epsabs=0)[0]
    
    sigma2 = 0
    
    for i in range(N):
        a = xs[i]+0.00000001
        b = xs[i+1]+0.00000001
        c = (phi[i+1] - phi[i])/(b-a)
        d = phi[i] - c*a
        
        dr = d.real
        di = d.imag
        cr = c.real
        ci = c.imag
        
        A = e2
    
        #numerical value is correct wrt mathematica
        real2 = (-2*(2 - 6*A + 9*A**2)*(a**3 - b**3)*cr**2 - 27*(1 + 2*(A**2-A))*(a**2 - b**2)*cr*dr - 54*(2 + (A**2-2*A))*(a - b)*dr**2 - 6*a*log(a)*(2*a**2*(3*A-1)*cr**2 + 9*a*(2*A-1)*cr*dr + 18*(A-1)*dr**2 + 3*(a**2*cr**2 + 3*a*cr*dr + 3*dr**2)*log(a)) + 6*b*(2*(3*A-1)*b**2*cr**2 + 9*(2*A-1)*b*cr*dr + 18*(A-1)*dr**2)*log(b) + 18*b*(b**2*cr**2 + 3*b*cr*dr + 3*dr**2)*log(b)**2)/54
        imag2 = (-2*(2 - 6*A + 9*A**2)*(a**3 - b**3)*ci**2 - 27*(1 + 2*(A**2-A))*(a**2 - b**2)*ci*di - 54*(2 + (A**2-2*A))*(a - b)*di**2 - 6*a*log(a)*(2*a**2*(3*A-1)*ci**2 + 9*a*(2*A-1)*ci*di + 18*(A-1)*di**2 + 3*(a**2*ci**2 + 3*a*ci*di + 3*di**2)*log(a)) + 6*b*(2*(3*A-1)*b**2*ci**2 + 9*(2*A-1)*b*ci*di + 18*(A-1)*di**2)*log(b) + 18*b*(b**2*ci**2 + 3*b*ci*di + 3*di**2)*log(b)**2)/54
        sigma2 += (real2 + imag2)
        

    sigma3 = integrate.quad(lambda w : (abs(fhat(w))**2)/2, 0.00001, S, limit = 200, epsabs=0)[0]
        
    return e**(-2*e2)*sigma1 + sigma2 + e**(-2*e2)*sigma3

def plot_freq(f, current_loss):
    xs = np.linspace(0.0000001, S-0.00001, 200)
        
    ys_r = []
    ys_i = []
    ys_a = []
    
    for x in xs:
        val = f(x)
        ys_r.append(val.real)
        ys_i.append(val.imag)
        ys_a.append(abs(val))

    plt.plot(xs, ys_r)
    plt.plot(xs, ys_i)
    plt.plot(xs, ys_a)
    
    plt.title("Frequency domain, loss: " + str(current_loss))
    
    plt.show()
    
def plot_time(f, current_loss):
    xs = np.linspace(-5.0001, 5, 400)
        
    ys_r = []
    ys_i = []
    ys_a = []
    
    for x in xs:
        val = f(x)
        ys_r.append(val.real)
        ys_i.append(val.imag)
        ys_a.append(abs(val))

    plt.plot(xs, ys_r)
    plt.plot(xs, ys_i)
    plt.plot(xs, ys_a)
    
    plt.title("Time domain, loss: " + str(current_loss))
    
    plt.show()

def grad_des(phi):
    phi = np.array(phi)
    #we compute the gradient first
    h = 0.00001
    step = h/10
    grad = [0]
    
    base_loss = loss(phi)
    
    for i in range(1, N):
        #the derivative in the i:th element
        phi[i] += h
        norm = l2_norm(phi)
        factor = 1/sqrt(l2_norm(phi))
        phi *= factor
        
        dy = loss(phi) - base_loss
        grad.append(dy/h)
    
        phi /= factor
        phi[i] -= h
        
    grad.append(0)
    
    print(phi)
    print(np.array(grad)*step)
        
    #grad is now computed, move in opposite direction
    phi -= np.array(grad)*step
    return phi/sqrt(l2_norm(phi))


def how_many_iterations_per_hour():
    return 0


best_phi = []

#initialize f
f = lambda x : e**(-(x-1)**2/0.2)
best_phi = func_to_phi(f)

smallest_loss = loss(best_phi)


for i in range(10):
    print("---------------------------")
    print("      Iteration: " + str(i))
    print("---------------------------")
    print("Old loss: " + str(smallest_loss))
    new_phi = grad_des(best_phi)
    new_loss = loss(new_phi)
    if (new_loss < smallest_loss):
        print("Reduced loss by: " + str((smallest_loss - new_loss)/smallest_loss*100) + "%")
        print("New lowest loss: " + str(new_loss))
        smallest_loss = new_loss
        plot_freq(get_fhat(best_phi), smallest_loss)

    
             
plot_freq(get_fhat(best_phi), smallest_loss)
plot_time(inverse_fourier(best_phi), smallest_loss)
