import numpy as np
import scipy as sp
import cmath
import math

from scipy.integrate import quad

import warnings
warnings.filterwarnings("ignore")

#defaults from https://arxiv.org/pdf/2012.12894.pdf (Tc0 = T0, delta_T, ml, c, d)
default_Tc0         = 154. #2+1 Nf Tc0
#default_Tc0         = 170. #2 Nf Tc0 from https://arxiv.org/pdf/hep-lat/0501030.pdf
default_delta_T     = 26.
default_ml          = 5.5
#default_c           = 350.0
#default_d           = 3.2
#defaults from me, based on a fit to https://arxiv.org/abs/hep-ph/9911474
default_c           = 14.
default_d           = -0.08
#defaults from https://arxiv.org/pdf/1812.08235.pdf (kappa)
default_kappa       = 0.012
#defaults from https://arxiv.org/pdf/hep-ph/0506234.pdf (Gs, T0, a0, a1, a2, a3, b3, b4)
default_Gs          = 10.08e-6
default_T0          = 175. #this one results in U(T) in agreement with https://arxiv.org/pdf/2012.12894.pdf
#default_T0          = 270.
default_a0          = 6.75
default_a1          = -1.95
default_a2          = 2.625
default_a3          = -7.44
default_b3          = 0.75
default_b4          = 7.5
#defaults from me (M0, Nf, Nc)
default_M0          = 400.
default_Nf          = 2.0
default_Nc          = 3.0
#defaults from me (Lambda)
default_Lambda      = 900.0

#default cluster masses
default_B           = 100
default_MM          = 2 * (default_M0 + default_ml) - default_B
default_MD          = 2 * (default_M0 + default_ml) - default_B
default_MN          = 3 * (default_M0 + default_ml) - 2 * default_B
default_MT          = 4 * (default_M0 + default_ml) - 3 * default_B
default_MF          = 4 * (default_M0 + default_ml) - 3 * default_B
default_MP          = 5 * (default_M0 + default_ml) - 4 * default_B
default_MQ          = 5 * (default_M0 + default_ml) - 4 * default_B
default_MH          = 6 * (default_M0 + default_ml) - 5 * default_B

def Tc(mu : float, **kwargs) -> float:

    options ={'kappa' : default_kappa, 'Tc0' : default_Tc0}
    options.update(kwargs)

    Tc0 = options['Tc0']
    kappa = options['kappa']

    return Tc0 * (1.0 - kappa * (( mu / Tc0) ** 2))
def Delta_ls(T: float, mu : float, **kwargs) -> float:

    options = {'delta_T' : default_delta_T}
    options.update(kwargs)

    delta_T = options['delta_T']

    return 0.5 * (1.0 - math.tanh((T - Tc(mu, **kwargs)) / delta_T))
def M(T : float, mu : float, **kwargs) -> float:

    options = {'M0' : default_M0, 'ml' : default_ml}
    options.update(kwargs)

    M0 = options['M0']
    ml = options['ml']

    return M0 * Delta_ls(T, mu, **kwargs) + ml
def dMdmu(T : float, mu : float, **kwargs) -> float:

    options = {'M0' : default_M0, 'kappa' : default_kappa, 'Tc0' : default_Tc0, 'delta_T' : default_delta_T}
    options.update(kwargs)

    M0 = options['M0']
    kappa = options['kappa']
    Tc0 = options['Tc0']
    delta_T = options['delta_T']

    return -((M0 * kappa * mu) / (Tc0 * delta_T)) * ((1.0 / math.cosh((T - Tc(mu, **kwargs)) / delta_T)) ** 2)
def dMdT(T : float, mu : float, **kwargs) -> float:
    
    options = {'M0' : default_M0, 'delta_T' : default_delta_T}
    options.update(kwargs)

    M0 = options['M0']
    delta_T = options['delta_T']

    return -(M0 / (2.0 * delta_T)) * ((1.0 / math.cosh((T - Tc(mu, **kwargs)) / delta_T)) ** 2)
def En(p : float, mass : float, **kwargs) -> float:
    #
    return math.sqrt((p ** 2) + (mass ** 2))
def dEn_dmu(p : float, T : float, mu : float, mass : float, deriv_M_mu : float, **kwargs) -> float:
    #
    return (mass / En(p, mass, **kwargs)) * deriv_M_mu
def dEn_dT(p : float, T : float, mu : float, mass : float, deriv_M_T : float, **kwargs) -> float:
    #
    return (mass / En(p, mass, **kwargs)) * deriv_M_T
def V(T : float, mu : float, **kwargs) -> float:
    
    options = {'M0' : default_M0, 'Gs' : default_Gs}
    options.update(kwargs)

    M0 = options['M0']
    Gs = options['Gs']

    return ((M0 ** 2) * (Delta_ls(T, mu, **kwargs) ** 2)) / (4.0 * Gs)
def b2(T : float, **kwargs) -> float:
    
    options = {'T0' : default_T0, 'a0' : default_a0, 'a1' : default_a1, 'a2' : default_a2, 'a3' : default_a3}
    options.update(kwargs)

    T0 = options['T0']
    a0 = options['a0']
    a1 = options['a1']
    a2 = options['a2']
    a3 = options['a3']

    return a0 + a1 * (T0 / T) + a2 * ((T0 / T) ** 2) + a3 * ((T0 / T) ** 3)
def U(T : float, Phi : complex, Phib : complex, **kwargs) -> complex:
    
    options = {'b3' : default_b3, 'b4' : default_b4}
    options.update(kwargs)

    b3 = options['b3']
    b4 = options['b4']

    return -(T ** 4) * ((b2(T, **kwargs) / 2.0) * Phi * Phib + (b3 / 6.0) * ((Phi ** 3) + (Phib ** 3)) - (b4 / 4.0) * ((Phi * Phib) ** 2))
def y_plus(p : float, T : float, mu : float, mass : float, a : float, **kwargs) -> float:
    #
    return math.exp(-(En(p, mass, **kwargs) - a * mu) / T)
def y_minus(p : float, T : float, mu : float, mass : float, a : float, **kwargs) -> float:
    #
    return math.exp(-(En(p, mass, **kwargs) + a * mu) / T)
def z_plus(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, a : int, **kwargs) -> complex:
    #positive energy, color charge
    ex = 0.0
    ex2 = 0.0
    ex3 = 0.0
    try:
        ex = y_plus(p, T, mu, mass, float(a), **kwargs)
        ex2 = ex ** 2
        ex3 = ex ** 3
    except OverflowError :
        return complex(-3.0 * (En(p, mass, **kwargs) - float(a) * mu) * np.heaviside(-(En(p, mass, **kwargs) - float(a) * mu), 0.5), 0.0)
    el = complex(0.0, 0.0)
    if a == 0:
        return 3.0 * T * cmath.log(1.0 - ex)
    elif (a % 3 == 0) and (not a % 2 == 0):
        return 3.0 * T * cmath.log(1.0 + ex)
    elif (not a % 3 == 0) and (a % 2 == 0):
        el = 1.0 - 3.0 * Phib * ex + 3.0 * Phi *ex2 - ex3
        return T * cmath.log(el)
    elif (a % 3 == 0) and (a % 2 == 0):
        return 3.0 * T * cmath.log(1.0 - ex)
    else:
        el = 1.0 + 3.0 * Phib * ex + 3.0 * Phi *ex2 + ex3
        return T * cmath.log(el)
def z_minus(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, a : int, **kwargs) -> complex:
    #negative energy, anticolor charge
    ex = 0.0
    ex2 = 0.0
    ex3 = 0.0
    try:
        ex = y_minus(p, T, mu, mass, float(a), **kwargs)
        ex2 = y_minus(p, T, mu, mass, float(a), **kwargs) ** 2
        ex3 = y_minus(p, T, mu, mass, float(a), **kwargs) ** 3
    except OverflowError :
        return complex(-3.0 * (En(p, mass, **kwargs) + float(a) * mu) * np.heaviside(-(En(p, mass, **kwargs) + float(a) * mu), 0.5), 0.0)
    el = complex(0.0, 0.0)
    if a == 0:
        return 3.0 * T * cmath.log(1.0 - ex)
    elif (a % 3 == 0) and (not a % 2 == 0):
        return 3.0 * T * cmath.log(1.0 + ex)
    elif (not a % 3 == 0) and (a % 2 == 0):
        el = 1.0 - 3.0 * Phib * ex + 3.0 * Phi *ex2 - ex3
        return T * cmath.log(el)
    elif (a % 3 == 0) and (a % 2 == 0):
        return 3.0 * T * cmath.log(1.0 - ex)
    else:
        el = 1.0 + 3.0 * Phib * ex + 3.0 * Phi *ex2 + ex3
        return T * cmath.log(el)
def f_plus(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, a : int, **kwargs) -> complex:
    #positive energy, color charge
    ex = 0.0
    ex2 = 0.0
    ex3 = 0.0
    try:
        ex = y_plus(p, T, mu, mass, float(a), **kwargs)
        ex2 = y_plus(p, T, mu, mass, float(a), **kwargs) ** 2
        ex3 = y_plus(p, T, mu, mass, float(a), **kwargs) ** 3
    except OverflowError :
        return complex(3.0, 0.0)
    el = complex(0.0, 0.0)
    if a == 0:
        el = 3.0 * complex(ex / (1.0 - ex), 0.0)
    elif (a % 3 == 0) and (not a % 2 == 0):
        el = 3.0 * complex(ex / (1.0 + ex), 0.0)
    elif (not a % 3 == 0) and (a % 2 == 0):
        el = (6.0 * Phi * ex2 - 3.0 * Phib * ex - 3.0 * ex3) / (1.0 - 3.0 * Phib * ex + 3.0 * Phi * ex2 - ex3)
    elif (a % 3 == 0) and (a % 2 == 0):
        el = 3.0 * complex(ex / (1.0 - ex), 0.0)
    else:
        el = (6.0 * Phi * ex2 + 3.0 * Phib * ex + 3.0 * ex3) / (1.0 + 3.0 * Phib * ex + 3.0 * Phi * ex2 + ex3)
    return el
def f_minus(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, a : int, **kwargs) -> complex:
    #negative energy, anticolor charge
    ex = 0.0
    ex2 = 0.0
    ex3 = 0.0
    try:
        ex = y_minus(p, T, mu, mass, float(a), **kwargs)
        ex2 = y_minus(p, T, mu, mass, float(a), **kwargs) ** 2
        ex3 = y_minus(p, T, mu, mass, float(a), **kwargs) ** 3
    except OverflowError :
        return complex(3.0, 0.0)
    el = complex(0.0, 0.0)
    if a == 0:
        el = 3.0 * complex(ex / (1.0 - ex), 0.0)
    elif (a % 3 == 0) and (not a % 2 == 0):
        el = 3.0 * complex(ex / (1.0 + ex), 0.0)
    elif (not a % 3 == 0) and (a % 2 == 0):
        el = (6.0 * Phib * ex2 - 3.0 * Phi * ex - 3.0 * ex3) / (1.0 - 3.0 * Phi * ex + 3.0 * Phib * ex2 - ex3)
    elif (a % 3 == 0) and (a % 2 == 0):
        el = 3.0 * complex(ex / (1.0 - ex), 0.0)
    else:
        el = (6.0 * Phib * ex2 + 3.0 * Phi * ex + 3.0 * ex3) / (1.0 + 3.0 * Phi * ex + 3.0 * Phib * ex2 + ex3)
    return el
def df_plus_dmu(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, deriv_M_mu : float, a : int, **kwargs) -> complex:
    #positive energy, color charge
    ex = 0.0
    ex2 = 0.0
    ex3 = 0.0
    try:
        ex = y_plus(p, T, mu, mass, float(a), **kwargs)
        ex2 = y_plus(p, T, mu, mass, float(a), **kwargs) ** 2
        ex3 = y_plus(p, T, mu, mass, float(a), **kwargs) ** 3
    except OverflowError :
        return complex(0.0, 0.0)
    if a == 0:
        fpa = f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        return ((deriv_M_mu * mass) / (En(p, mass, **kwargs) * T)) * ( ((fpa ** 2) / 3.0) - fpa )
    elif (a % 3 == 0) and (not a % 2 == 0):
        fpa = f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        return (( deriv_M_mu * (mass / En(p, mass, **kwargs)) - float(a) ) / T) * ( ((fpa ** 2) / 3.0) - fpa )
    elif (not a % 3 == 0) and (a % 2 == 0):
        numerator = 12.0 * Phi * ex2 - 3.0 * Phib * ex - 9.0 * ex3
        denominator = 1.0 - 3.0 * Phib * ex + 3.0 * Phi * ex2 - ex3
        fpa = f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        return (( deriv_M_mu * (mass / En(p, mass, **kwargs)) - float(a) ) / T) * ((fpa ** 2) - (numerator / denominator))
    elif (a % 3 == 0) and (a % 2 == 0):
        fpa = f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        return (( deriv_M_mu * (mass / En(p, mass, **kwargs)) - float(a) ) / T) * ( ((fpa ** 2) / 3.0) - fpa )
    else:
        numerator = 12.0 * Phi * ex2 + 3.0 * Phib * ex + 9.0 * ex3
        denominator = 1.0 + 3.0 * Phib * ex + 3.0 * Phi * ex2 + ex3
        fpa = f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        return (( deriv_M_mu * (mass / En(p, mass, **kwargs)) - float(a) ) / T) * ((fpa ** 2) - (numerator / denominator))
def df_plus_dT(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, deriv_M_T : float, a : int, **kwargs) -> complex:
    #positive energy, color charge
    ex = 0.0
    ex2 = 0.0
    ex3 = 0.0
    try:
        ex = y_plus(p, T, mu, mass, float(a), **kwargs)
        ex2 = y_plus(p, T, mu, mass, float(a), **kwargs) ** 2
        ex3 = y_plus(p, T, mu, mass, float(a), **kwargs) ** 3
    except OverflowError :
        return complex(0.0, 0.0)
    if a == 0:
        fpa = f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        energy = En(p, mass, **kwargs)
        first = ((mass * deriv_M_T / energy) - (energy - float(a) * mu) / T) / T
        return first * ( ((fpa ** 2) / 3.0) - fpa )
    elif (a % 3 == 0) and (not a % 2 == 0):
        fpa = f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        energy = En(p, mass, **kwargs)
        first = ((mass * deriv_M_T / energy) - (energy - float(a) * mu) / T) / T
        return first * ( ((fpa ** 2) / 3.0) - fpa )
    elif (not a % 3 == 0) and (a % 2 == 0):
        numerator = 12.0 * Phi * ex2 - 3.0 * Phib * ex - 9.0 * ex3
        denominator = 1.0 - 3.0 * Phib * ex + 3.0 * Phi * ex2 - ex3
        fpa = f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        energy = En(p, mass, **kwargs)
        first = ((mass * deriv_M_T / energy) - (energy - float(a) * mu) / T) / T
        return first * ((fpa ** 2) - (numerator / denominator))
    elif (a % 3 == 0) and (a % 2 == 0):
        fpa = f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        energy = En(p, mass, **kwargs)
        first = ((mass * deriv_M_T / energy) - (energy - float(a) * mu) / T) / T
        return first * ( ((fpa ** 2) / 3.0) - fpa )
    else:
        numerator = 12.0 * Phi * ex2 + 3.0 * Phib * ex + 9.0 * ex3
        denominator = 1.0 + 3.0 * Phib * ex + 3.0 * Phi * ex2 + ex3
        fpa = f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        energy = En(p, mass, **kwargs)
        first = ((mass * deriv_M_T / energy) - (energy - float(a) * mu) / T) / T
        return first * ((fpa ** 2) - (numerator / denominator))
def df_minus_dmu(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, deriv_M_mu : float, a : int, **kwargs) -> complex:
    #negative energy, anticolor charge
    ex = 0.0
    ex2 = 0.0
    ex3 = 0.0
    try:
        ex = y_minus(p, T, mu, mass, float(a), **kwargs)
        ex2 = y_minus(p, T, mu, mass, float(a), **kwargs) ** 2
        ex3 = y_minus(p, T, mu, mass, float(a), **kwargs) ** 3
    except OverflowError :
        return complex(0.0, 0.0)
    if a == 0:
        fpa = f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        return ((deriv_M_mu * mass) / (En(p, mass, **kwargs) * T)) * ( ((fpa ** 2) / 3.0) - fpa )
    elif (a % 3 == 0) and (not a % 2 == 0):
        fpa = f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        return (( deriv_M_mu * (mass / En(p, mass, **kwargs)) + float(a) ) / T) * ( ((fpa ** 2) / 3.0) - fpa )
    elif (not a % 3 == 0) and (a % 2 == 0):
        numerator = 12.0 * Phib * ex2 - 3.0 * Phi * ex - 9.0 * ex3
        denominator = 1.0 - 3.0 * Phi * ex + 3.0 * Phib * ex2 - ex3
        fpa = f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        return (( deriv_M_mu * (mass / En(p, mass, **kwargs)) + float(a) ) / T) * ((fpa ** 2) - (numerator / denominator))
    elif (a % 3 == 0) and (a % 2 == 0):
        fpa = f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        return (( deriv_M_mu * (mass / En(p, mass, **kwargs)) + float(a) ) / T) * ( ((fpa ** 2) / 3.0) - fpa )
    else:
        numerator = 12.0 * Phib * ex2 + 3.0 * Phi * ex + 9.0 * ex3
        denominator = 1.0 + 3.0 * Phi * ex + 3.0 * Phib * ex2 + ex3
        fpa = f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        return (( deriv_M_mu * (mass / En(p, mass, **kwargs)) + float(a) ) / T) * ((fpa ** 2) - (numerator / denominator))
def df_minus_dT(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, deriv_M_T : float, a : int, **kwargs) -> complex:
    #negative energy, anticolor charge
    ex = 0.0
    ex2 = 0.0
    ex3 = 0.0
    try:
        ex = y_minus(p, T, mu, mass, float(a), **kwargs)
        ex2 = y_minus(p, T, mu, mass, float(a), **kwargs) ** 2
        ex3 = y_minus(p, T, mu, mass, float(a), **kwargs) ** 3
    except OverflowError :
        return complex(0.0, 0.0)
    if a == 0:
        fpa = f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        energy = En(p, mass, **kwargs)
        first = ((mass * deriv_M_T / energy) - (energy + float(a) * mu) / T) / T
        return first * ( ((fpa ** 2) / 3.0) - fpa )
    elif (a % 3 == 0) and (not a % 2 == 0):
        fpa = f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        energy = En(p, mass, **kwargs)
        first = ((mass * deriv_M_T / energy) - (energy + float(a) * mu) / T) / T
        return first * ( ((fpa ** 2) / 3.0) - fpa )
    elif (not a % 3 == 0) and (a % 2 == 0):
        numerator = 12.0 * Phib * ex2 - 3.0 * Phi * ex - 9.0 * ex3
        denominator = 1.0 - 3.0 * Phi * ex + 3.0 * Phib * ex2 - ex3
        fpa = f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        energy = En(p, mass, **kwargs)
        first = ((mass * deriv_M_T / energy) - (energy + float(a) * mu) / T) / T
        return first * ((fpa ** 2) - (numerator / denominator))
    elif (a % 3 == 0) and (a % 2 == 0):
        fpa = f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        energy = En(p, mass, **kwargs)
        first = ((mass * deriv_M_T / energy) - (energy + float(a) * mu) / T) / T
        return first * ( ((fpa ** 2) / 3.0) - fpa )
    else:
        numerator = 12.0 * Phib * ex2 + 3.0 * Phi * ex + 9.0 * ex3
        denominator = 1.0 + 3.0 * Phi * ex + 3.0 * Phib * ex2 + ex3
        fpa = f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs)
        energy = En(p, mass, **kwargs)
        first = ((mass * deriv_M_T / energy) - (energy + float(a) * mu) / T) / T
        return first * ((fpa ** 2) - (numerator / denominator))
def I_plus_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'I_plus_real_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['I_plus_real_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        fpe = f_plus(x * _T, _T, _mu, _Phi, _Phib, 0.0, 1, **key)
        return (x / 3.0) * fpe.real

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in I_plus_real did not succeed!")

    return integral
def I_plus_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'I_plus_imag_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['I_plus_imag_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        fpe = f_plus(x * _T, _T, _mu, _Phi, _Phib, 0.0, 1, **key)
        return (x / 3.0) * fpe.imag

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in I_plus_imag did not succeed!")

    return integral
def I_minus_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'I_minus_real_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['I_minus_real_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        fpe = f_minus(x * _T, _T, _mu, _Phi, _Phib, 0.0, 1, **key)
        return (x / 3.0) * fpe.real

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))
    
    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in I_plus_real did not succeed!")
    
    return integral
def I_minus_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'I_minus_imag_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['I_minus_imag_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        fpe = f_minus(x * _T, _T, _mu, _Phi, _Phib, 0.0, 1, **key)
        return (x / 3.0) * fpe.imag

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in I_plus_real did not succeed!")

    return integral
def J_plus_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'J_plus_real_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['J_plus_real_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) : 
        return (x / 3.0) * df_plus_dmu(x * _T, _T, _mu, _Phi, _Phib, 0.0, 0.0, 1, **key).real

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in J_plus_real did not succeed!")

    fpe = f_plus(M(T, mu, **kwargs), T, mu, Phi, Phib, 0.0, 1, **kwargs)

    return integral - (M(T, mu, **kwargs) * fpe.real * dMdmu(T, mu, **kwargs)) / (3.0 * (T ** 2))
def J_plus_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'J_plus_imag_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['J_plus_imag_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) : 
        return (x / 3.0) * df_plus_dmu(x * _T, _T, _mu, _Phi, _Phib, 0.0, 0.0, 1, **key).imag

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in J_plus_imag did not succeed!")

    fpe = f_plus(M(T, mu, **kwargs), T, mu, Phi, Phib, 0.0, 1, **kwargs)

    return integral - (M(T, mu, **kwargs) * fpe.imag * dMdmu(T, mu, **kwargs)) / (3.0 * (T ** 2))
def J_minus_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'J_minus_real_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['J_minus_real_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) : 
        return (x / 3.0) * df_minus_dmu(x * _T, _T, _mu, _Phi, _Phib, 0.0, 0.0, 1, **key).real

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in J_minus_real did not succeed!")
    
    fme = f_minus(M(T, mu, **kwargs), T, mu, Phi, Phib, 0.0, 1, **kwargs)
    
    return integral - (M(T, mu, **kwargs) * fme.real * dMdmu(T, mu, **kwargs)) / (3.0 * (T ** 2))
def J_minus_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'J_minus_imag_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['J_minus_imag_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) : 
        return (x / 3.0) * df_minus_dmu(x * _T, _T, _mu, _Phi, _Phib, 0.0, 0.0, 1, **key).imag

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in J_minus_imag did not succeed!")

    fme = f_minus(M(T, mu, **kwargs), T, mu, Phi, Phib, 0.0, 1, **kwargs)

    return integral - (M(T, mu, **kwargs) * fme.imag * dMdmu(T, mu, **kwargs)) / (3.0 * (T ** 2))
def K_plus_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'K_plus_real_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['K_plus_real_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) : 
        return (x / 3.0) * df_plus_dT(x * _T, _T, _mu, _Phi, _Phib, 0.0, 0.0, 1, **key).real

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in K_plus_real did not succeed!")

    fpe = f_plus(M(T, mu, **kwargs), T, mu, Phi, Phib, 0.0, 1, **kwargs)

    return integral - ( (M(T, mu, **kwargs) / (3.0 * T)) * (fpe.real / T) * ( dMdT(T, mu, **kwargs) - M(T, mu, **kwargs) / T ))
def K_plus_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'K_plus_imag_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['K_plus_imag_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) : 
        return (x / 3.0) * df_plus_dT(x * _T, _T, _mu, _Phi, _Phib, 0.0, 0.0, 1, **key).imag

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in K_plus_imag did not succeed!")

    fpe = f_plus(M(T, mu, **kwargs), T, mu, Phi, Phib, 0.0, 1, **kwargs)
    
    return integral - ( (M(T, mu, **kwargs) / (3.0 * T)) * (fpe.imag / T) * ( dMdT(T, mu, **kwargs) - M(T, mu, **kwargs) / T ))
def K_minus_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'K_minus_real_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['K_minus_real_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) : 
        return (x / 3.0) * df_minus_dT(x * _T, _T, _mu, _Phi, _Phib, 0.0, 0.0, 1, **key).real

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in K_plus_real did not succeed!")
    
    fme = f_minus(M(T, mu, **kwargs), T, mu, Phi, Phib, 0.0, 1, **kwargs)

    return integral - ( (M(T, mu, **kwargs) / (3.0 * T)) * (fme.real / T) * ( dMdT(T, mu, **kwargs) - M(T, mu, **kwargs) / T ))
def K_minus_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'K_minus_imag_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['K_minus_imag_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) : 
        return (x / 3.0) * df_minus_dT(x * _T, _T, _mu, _Phi, _Phib, 0.0, 0.0, 1, **key).imag

    integral, error = quad(integrand, M(T, mu, **kwargs) / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in K_plus_imag did not succeed!")

    fme = f_minus(M(T, mu, **kwargs), T, mu, Phi, Phib, 0.0, 1, **kwargs)

    return integral - ( (M(T, mu, **kwargs) / (3.0 * T)) * (fme.imag / T) * ( dMdT(T, mu, **kwargs) - M(T, mu, **kwargs) / T ))
def alpha_s(T : float, mu : float, **kwargs) -> float:
    
    options = {'Nf' : default_Nf, 'Nc' : default_Nc, 'c' : default_c, 'd' : default_d}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    c = options['c']
    d = options['d']

    r = d * T
    return ((12.0 * np.pi) / (11 * Nc - 2 * Nf)) * ( (1.0 / (math.log((r ** 2) / (c ** 2)))) - ((c ** 2) / ((r ** 2) - (c ** 2))) )
def dalpha_s_dT(T : float, mu : float, **kwargs) -> float:
    
    options = {'Nf' : default_Nf, 'Nc' : default_Nc, 'c' : default_c, 'd' : default_d}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    c = options['c']
    d = options['d']

    return ((12.0 * np.pi) / (11 * Nc - 2 * Nf)) * ((2.0 * (c ** 2) * (d ** 2) * T) / (((d ** 2) * (T ** 2) - (c ** 2)) ** 2) - 2.0 / (T * (( math.log(d ** 2) + math.log((T ** 2) / (c ** 2))) ** 2)))
def sigma_plus_real(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, deriv_M_T : float, a : int, **kwargs) -> float:
    #
    return z_plus(p, T, mu, Phi, Phib, mass, a, **kwargs).real + f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs).real * (En(p, mass, **kwargs) - float(a) * mu - T * dEn_dT(p, T, mu, mass, deriv_M_T, **kwargs))
def sigma_plus_imag(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, deriv_M_T : float, a : int, **kwargs) -> float:
    #
    return z_plus(p, T, mu, Phi, Phib, mass, a, **kwargs).imag + f_plus(p, T, mu, Phi, Phib, mass, a, **kwargs).imag * (En(p, mass, **kwargs) - float(a) * mu - T * dEn_dT(p, T, mu, mass, deriv_M_T, **kwargs))
def sigma_minus_real(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, deriv_M_T : float, a : int, **kwargs) -> float:
    #
    return z_minus(p, T, mu, Phi, Phib, mass, a, **kwargs).real + f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs).real * (En(p, mass, **kwargs) + float(a) * mu - T * dEn_dT(p, T, mu, mass, deriv_M_T, **kwargs))
def sigma_minus_imag(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, deriv_M_T : float, a : int, **kwargs) -> float:
    #
    return z_minus(p, T, mu, Phi, Phib, mass, a, **kwargs).imag + f_minus(p, T, mu, Phi, Phib, mass, a, **kwargs).imag * (En(p, mass, **kwargs) + float(a) * mu - T * dEn_dT(p, T, mu, mass, deriv_M_T, **kwargs))

def Omega_Delta(T : float, mu : float, **kwargs) -> float:
    
    options = {'M0' : default_M0, 'Nf' : default_Nf, 'Nc' : default_Nc, 'Lambda' : default_Lambda, 'Omega_Delta_debug_flag' : False}
    options.update(kwargs)

    M0 = options['M0']
    Nf = options['Nf']
    Nc = options['Nc']
    Lambda = options['Lambda']
    debug_flag = options['Omega_Delta_debug_flag']

    integrand = lambda p, _T, _mu, key : (p ** 2) * (En(p, M(_T, _mu, **key), **key) - En(p, M(0.0, 0.0, **key), **key))

    integral, error = quad(integrand, 0.0, Lambda, args = (T, mu, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in Omega_Delta did not succeed!")

    return V(T, mu, **kwargs) - V(0.0, 0.0, **kwargs) - (Nf / (math.pi ** 2)) * (Nc / 3.0) * integral
def Omega_Q_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : default_Nf, 'Nc' : default_Nc, 'Omega_Q_real_debug_flag' : False}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    debug_flag = options['Omega_Q_real_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, key): 
        return ((p ** 2) / 3.0) * (z_plus(p, _T, _mu, _Phi, _Phib, M(_T, _mu, **key), 1, **key).real + z_minus(p, _T, _mu, _Phi, _Phib, M(_T, _mu, **key), 1, **key).real)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in Omega_Q_real did not succeed!")

    return -(Nf / (math.pi ** 2)) * Nc * integral
def Omega_Q_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : default_Nf, 'Nc' : default_Nc, 'Omega_Q_imag_debug_flag' : False}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    debug_flag = options['Omega_Q_imag_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, key): 
        return ((p ** 2) / 3.0) * (z_plus(p, _T, _mu, _Phi, _Phib, M(_T, _mu, **key), 1, **key).imag + z_minus(p, _T, _mu, _Phi, _Phib, M(_T, _mu, **key), 1, **key).imag)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in Omega_Q_imag did not succeed!")

    return -(Nf / (math.pi ** 2)) * Nc * integral
def Omega_g_real(T : float, Phi : complex, Phib : complex, **kwargs) -> float:
    #
    return U(T, Phi, Phib, **kwargs).real
def Omega_g_imag(T : float, Phi : complex, Phib : complex, **kwargs) -> float:
    #
    return U(T, Phi, Phib, **kwargs).imag
def Omega_pert_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : default_Nf}
    options.update(kwargs)

    Nf = options['Nf']

    Ip_full = complex(I_plus_real(T, mu, Phi, Phib, **kwargs), I_plus_imag(T, mu, Phi, Phib, **kwargs))
    Im_full = complex(I_minus_real(T, mu, Phi, Phib, **kwargs), I_minus_imag(T, mu, Phi, Phib, **kwargs))
    spart = (Ip_full + Im_full) ** 2

    return ((4.0 * Nf) / (3.0 * np.pi)) * alpha_s(T, mu, **kwargs) * (T ** 4) * (Ip_full.real + Im_full.real + (3.0 / (2.0 * (np.pi ** 2))) * spart.real)
def Omega_pert_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : default_Nf}
    options.update(kwargs)

    Nf = options['Nf']

    Ip_full = complex(I_plus_real(T, mu, Phi, Phib, **kwargs), I_plus_imag(T, mu, Phi, Phib, **kwargs))
    Im_full = complex(I_minus_real(T, mu, Phi, Phib, **kwargs), I_minus_imag(T, mu, Phi, Phib, **kwargs))
    spart = (Ip_full + Im_full) ** 2

    return ((4.0 * Nf)/ (3.0 * np.pi)) * alpha_s(T, mu, **kwargs) * (T ** 4) * (Ip_full.imag + Im_full.imag + (3.0 / (2.0 * (np.pi ** 2))) * spart.imag)
def Omega_cluster_real(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, **kwargs) -> float:
    
    options = {'Omega_cluster_real_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['Omega_cluster_real_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, _a, key):
        bound = z_plus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).real + z_minus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).real
        scattering = z_plus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).real + z_minus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).real
        return ((p ** 2) / 3.0) * (bound - scattering) * np.heaviside(_Mth - _M, 0.5)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, bmass, thmass, a, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in Omega_N_real did not succeed!")

    if a == 0:
        return ((-1.0) ** a) * (1.0 / (4.0 * (math.pi ** 2))) * integral
    else:
        return ((-1.0) ** a) * (1.0 / (2.0 * (math.pi ** 2))) * integral
def Omega_cluster_imag(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, **kwargs) -> float:
    
    options = {'Omega_cluster_imag_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['Omega_cluster_imag_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, _a, key):
        bound = z_plus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).imag + z_minus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).imag
        scattering = z_plus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).imag + z_minus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).imag
        return ((p ** 2) / 3.0) * (bound - scattering) * np.heaviside(_Mth - _M, 0.5)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, bmass, thmass, a, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in Omega_N_real did not succeed!")

    if a == 0:
        return ((-1.0) ** a) * (1.0 / (4.0 * (math.pi ** 2))) * integral
    else:
        return ((-1.0) ** a) * (1.0 / (2.0 * (math.pi ** 2))) * integral

def Pressure_Q(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    #
    return -Omega_Q_real(T, mu, Phi, Phib, **kwargs)
def Pressure_g(T : float, Phi : complex, Phib : complex, **kwargs) -> float:
    #
    return -Omega_g_real(T, Phi, Phib, **kwargs)
def Pressure_pert(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    #
    return -Omega_pert_real(T, mu, Phi, Phib, **kwargs)
def Pressure_cluster(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, dx : int, **kwargs) -> float:
    #
    return -float(dx) * Omega_cluster_real(T, mu, Phi, Phib, bmass, thmass, a, **kwargs)

def BDensity_Q(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:

    options = {'Nf' : default_Nf, 'Nc' : default_Nc, 'BDensity_Q_debug_flag' : False}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    debug_flag = options['BDensity_Q_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _M, _dM, key):
        bound = f_plus(p, _T, _mu, _Phi, _Phib, _M, 1, **key).real * (1.0 - dEn_dmu(p, _T, _mu, _M, _dM, **key)) - f_minus(p, _T, _mu, _Phi, _Phib, _M, 1, **key).real * (1.0 + dEn_dmu(p, _T, _mu, _M, _dM, **key))
        return ((p ** 2) / 3.0) * bound

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, M(T, mu, **kwargs), dMdmu(T, mu, **kwargs), kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in BDensity_Q did not succeed!")

    return ((2.0 * Nf * Nc) / (2.0 * (math.pi ** 2))) * integral
def BDensity_pert(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    I_plus_complex = complex(I_plus_real(T, mu, Phi, Phib, **kwargs), I_plus_imag(T, mu, Phi, Phib, **kwargs))
    I_minus_complex = complex(I_minus_real(T, mu, Phi, Phib, **kwargs), I_minus_imag(T, mu, Phi, Phib, **kwargs))
    J_plus_complex = complex(J_plus_real(T, mu, Phi, Phib, **kwargs), J_plus_imag(T, mu, Phi, Phib, **kwargs))
    J_minus_complex = complex(J_minus_real(T, mu, Phi, Phib, **kwargs), J_minus_imag(T, mu, Phi, Phib, **kwargs))
    par = (I_plus_complex + I_minus_complex) * (J_plus_complex + J_minus_complex)
    return -(4.0 / math.pi) * alpha_s(T, mu, **kwargs) * (T ** 4) * (J_plus_complex.real + J_minus_complex.real + (3.0 / (math.pi ** 2)) * par.real)
def BDensity_cluster(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d_bmass : float, d_thmass : float, a : int, dx : int, **kwargs) -> float:
    if a > 0:
        options = {'BDensity_cluster_debug_flag' : False}
        options.update(kwargs)

        debug_flag = options['BDensity_cluster_debug_flag']

        def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, _dM, _dMth, _a, key):
            bound = f_plus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).real * (_a - dEn_dmu(p, _T, _mu, _M, _dM, **key)) - f_minus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).real * (_a + dEn_dmu(p, _T, _mu, _M, _dM, **key))
            scattering_positive = f_plus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).real * (_a - dEn_dmu(p, _T, _mu, _Mth, _dMth, **key))
            scattering_negative = f_minus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).real * (_a + dEn_dmu(p, _T, _mu, _Mth, _dMth, **key))
            return ((p ** 2) / 3.0) * (bound - (scattering_positive - scattering_negative)) * np.heaviside(_Mth - _M, 0.5)

        integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, bmass, thmass, d_bmass, d_thmass, a, kwargs))
        
        if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
            print("The integration in BDensity_cluster did not succeed!")

        if a % 2 == 0 and not a % 3 == 0:
            return -(float(dx) / (2.0 * (math.pi ** 2))) * integral
        else:
            return (float(dx) / (2.0 * (math.pi ** 2))) * integral
    else:
        return 0.0

def SDensity_Q(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:

    options = {'Nf' : default_Nf, 'Nc' : default_Nc, 'SDensity_Q_debug_flag' : False}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    debug_flag = options['SDensity_Q_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, key):
        plus = sigma_plus_real(p, _T, _mu, _Phi, _Phib, M(_T, _mu, **key), dMdT(_T, _mu, **key), 1, **key)
        minus = sigma_minus_real(p, _T, _mu, _Phi, _Phib, M(_T, _mu, **key), dMdT(_T, _mu, **key), 1, **key)
        return ((p ** 2) / 3.0) * (plus + minus)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, kwargs))
    
    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in SDensity_Q did not succeed!")

    return ((2.0 * Nf * Nc) / (2.0 * (math.pi ** 2) * T)) * integral
def SDensity_g(T : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'T0' : default_T0, 'a1' : default_a1, 'a2' : default_a2, 'a3' : default_a3}
    options.update(kwargs)

    T0 = options['T0']
    a1 = options['a1']
    a2 = options['a2']
    a3 = options['a3']

    first = -(4.0 / T) * U(T, Phi, Phib, **kwargs)
    second = ((T ** 4) / 2) * Phib * Phi * (a1 * (T0 / (T ** 2)) + 2.0 * a2 * ((T0 ** 2) / (T ** 3)) + 3.0 * a3 * ((T0 ** 3) / (T ** 4)))
    return first.real - second.real
def SDensity_pert(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    I_plus_complex = complex(I_plus_real(T, mu, Phi, Phib, **kwargs), I_plus_imag(T, mu, Phi, Phib, **kwargs))
    I_minus_complex = complex(I_minus_real(T, mu, Phi, Phib, **kwargs), I_minus_imag(T, mu, Phi, Phib, **kwargs))
    K_plus_complex = complex(K_plus_real(T, mu, Phi, Phib, **kwargs), K_plus_imag(T, mu, Phi, Phib, **kwargs))
    K_minus_complex = complex(K_minus_real(T, mu, Phi, Phib, **kwargs), K_minus_imag(T, mu, Phi, Phib, **kwargs))
    par1 = (I_plus_complex + I_minus_complex) ** 2
    par2 = (I_plus_complex + I_minus_complex) * (K_plus_complex + K_minus_complex)
    first = -(16.0 / math.pi) * alpha_s(T, mu, **kwargs) * (T ** 3) * (I_plus_complex.real + I_minus_complex.real + (3.0 / (2.0 * (math.pi ** 2))) * par1.real)
    second = -(4.0 / math.pi) * dalpha_s_dT(T, mu, **kwargs) * (T ** 4) * (I_plus_complex.real + I_minus_complex.real + (3.0 / (2.0 * (math.pi ** 2))) * par1.real)
    third = -(4.0 / math.pi) * alpha_s(T, mu, **kwargs) * (T ** 4) * (K_plus_complex.real + K_minus_complex.real + (3.0 / (math.pi ** 2)) * par2.real)
    return first + second + third
def SDensity_cluster(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d_bmass : float, d_thmass : float, a : int, dx : int, **kwargs) -> float:
    options = {'SDensity_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['SDensity_cluster_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, _dM, _dMth, _a, key):
        bound = sigma_plus_real(p, _T, _mu, _Phi, _Phib, _M, _dM, _a, **key) + sigma_minus_real(p, _T, _mu, _Phi, _Phib, _M, _dM, _a, **key)
        scatter = sigma_plus_real(p, _T, _mu, _Phi, _Phib, _Mth, _dMth, _a, **key) + sigma_minus_real(p, _T, _mu, _Phi, _Phib, _Mth, _dMth, _a, **key)
        return ((p ** 2) / 3.0) * (bound - scatter) * np.heaviside(_Mth - _M, 0.5)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, bmass, thmass, d_bmass, d_thmass, a, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in SDensity_cluster did not succeed!")

    if a == 0:
        return (float(dx) / (4.0 * (math.pi ** 2) * T)) * integral
    elif a % 2 == 0 and not a % 3 == 0:
        return -(float(dx) / (4.0 * (math.pi ** 2) * T)) * integral
    else:
        return (float(dx) / (2.0 * (math.pi ** 2) * T)) * integral

def alt_z_plus(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, a : int, **kwargs) -> complex:
    #positive energy, color charge
    if a == 0:
        try:
            ex = math.exp(En(p, mass, **kwargs) / T)
        except OverflowError:
            return complex(0.0, 0.0)
        return complex(((p ** 4) / En(p, mass, **kwargs)) * (1.0 / (ex - 1.0)), 0.0)
    elif a == 3:
        try:
            ex = math.exp((En(p, mass, **kwargs) - float(a) * mu) / T)
        except OverflowError:
            return complex(0.0, 0.0)
        return complex(((p ** 4) / En(p, mass, **kwargs)) * (1.0 / (ex + 1.0)), 0.0)
    elif a == 6:
        try:
            ex = math.exp((En(p, mass, **kwargs) - float(a) * mu) / T)
        except OverflowError:
            return complex(0.0, 0.0)
        return complex(((p ** 4) / En(p, mass, **kwargs)) * (1.0 / (ex - 1.0)), 0.0)
    else:
        if a % 2 != 0:
            part1 = 0.0
            try:
                ex1 = math.exp((En(p, mass, **kwargs) - float(a) * mu) / T)
                exm1 = math.exp(-(En(p, mass, **kwargs) - float(a) * mu) / T)
                exm2 = math.exp(-2.0 * (En(p, mass, **kwargs) - float(a) * mu) / T)
                part1 = Phib / (ex1 + 3.0 * Phi * exm1 + exm2 + 3.0 * Phib)
            except OverflowError:
                part1 = 0.0
            part2 = 0.0
            try:
                ex2 = math.exp(2.0 * (En(p, mass, **kwargs) - float(a) * mu) / T)
                ex1 = math.exp((En(p, mass, **kwargs) - float(a) * mu) / T)
                exm1 = math.exp(-(En(p, mass, **kwargs) - float(a) * mu) / T)
                part2 = 2.0 * Phi / (ex2 + 3.0 * Phib * ex1 + exm1 + 3.0 * Phi)
            except OverflowError:
                part2 = 0.0
            part3 = 0.0
            try:
                ex3 = math.exp(3.0 * (En(p, mass, **kwargs) - float(a) * mu) / T)
                ex2 = math.exp(2.0 * (En(p, mass, **kwargs) - float(a) * mu) / T)
                ex1 = math.exp((En(p, mass, **kwargs) - float(a) * mu) / T)
                part3 = 1.0 / (ex3 + 3.0 * Phib * ex2 + 3.0 * Phi * ex1 + 1.0)
            except OverflowError:
                part3 = 0.0
            return ((p ** 4) / En(p, mass, **kwargs)) * (part1 + part2 + part3)
        else:
            part1 = 0.0
            try:
                ex1 = math.exp((En(p, mass, **kwargs) - float(a) * mu) / T)
                exm1 = math.exp(-(En(p, mass, **kwargs) - float(a) * mu) / T)
                exm2 = math.exp(-2.0 * (En(p, mass, **kwargs) - float(a) * mu) / T)
                part1 = Phi / (ex1 + 3.0 * Phib * exm1 - exm2 - 3.0 * Phi)
            except OverflowError:
                part1 = 0.0
            part2 = 0.0
            try:
                ex2 = math.exp(2.0 * (En(p, mass, **kwargs) - float(a) * mu) / T)
                ex1 = math.exp((En(p, mass, **kwargs) - float(a) * mu) / T)
                exm1 = math.exp(-(En(p, mass, **kwargs) - float(a) * mu) / T)
                part2 = -2.0 * Phib / (ex2 - 3.0 * Phi * ex1 - exm1 + 3.0 * Phib)
            except OverflowError:
                part2 = 0.0
            part3 = 0.0
            try:
                ex3 = math.exp(3.0 * (En(p, mass, **kwargs) - float(a) * mu) / T)
                ex2 = math.exp(2.0 * (En(p, mass, **kwargs) - float(a) * mu) / T)
                ex1 = math.exp((En(p, mass, **kwargs) - float(a) * mu) / T)
                part3 = 1.0 / (ex3 - 3.0 * Phi * ex2 + 3.0 * Phib * ex1 - 1.0)
            except OverflowError:
                part3 = 0.0
            return ((p ** 4) / En(p, mass, **kwargs)) * (part1 + part2 + part3)
def alt_z_minus(p : float, T : float, mu : float, Phi : complex, Phib : complex, mass : float, a : int, **kwargs) -> complex:
    #negative energy, color anticharge
    if a == 0:
        try:
            ex = math.exp(En(p, mass, **kwargs) / T)
        except OverflowError:
            return complex(0.0, 0.0)
        return complex(((p ** 4) / En(p, mass, **kwargs)) * (1.0 / (ex - 1.0)), 0.0)
    elif a == 3:
        try:
            ex = math.exp((En(p, mass, **kwargs) + float(a) * mu) / T)
        except OverflowError:
            return complex(0.0, 0.0)
        return complex(((p ** 4) / En(p, mass, **kwargs)) * (1.0 / (ex + 1.0)), 0.0)
    elif a == 6:
        try:
            ex = math.exp((En(p, mass, **kwargs) + float(a) * mu) / T)
        except OverflowError:
            return complex(0.0, 0.0)
        return complex(((p ** 4) / En(p, mass, **kwargs)) * (1.0 / (ex - 1.0)), 0.0)
    else:
        if a % 2 != 0:
            part1 = 0.0
            try:
                ex1 = math.exp((En(p, mass, **kwargs) + float(a) * mu) / T)
                exm1 = math.exp(-(En(p, mass, **kwargs) + float(a) * mu) / T)
                exm2 = math.exp(-2.0 * (En(p, mass, **kwargs) + float(a) * mu) / T)
                part1 = Phi / (ex1 + 3.0 * Phib * exm1 + exm2 + 3.0 * Phi)
            except OverflowError:
                part1 = 0.0
            part2 = 0.0
            try:
                ex2 = math.exp(2.0 * (En(p, mass, **kwargs) + float(a) * mu) / T)
                ex1 = math.exp((En(p, mass, **kwargs) + float(a) * mu) / T)
                exm1 = math.exp(-(En(p, mass, **kwargs) + float(a) * mu) / T)
                part2 = 2.0 * Phib / (ex2 + 3.0 * Phi * ex1 + exm1 + 3.0 * Phib)
            except OverflowError:
                part2 = 0.0
            part3 = 0.0
            try:
                ex3 = math.exp(3.0 * (En(p, mass, **kwargs) + float(a) * mu) / T)
                ex2 = math.exp(2.0 * (En(p, mass, **kwargs) + float(a) * mu) / T)
                ex1 = math.exp((En(p, mass, **kwargs) + float(a) * mu) / T)
                part3 = 1.0 / (ex3 + 3.0 * Phi * ex2 + 3.0 * Phib * ex1 + 1.0)
            except OverflowError:
                part3 = 0.0
            return ((p ** 4) / En(p, mass, **kwargs)) * (part1 + part2 + part3)
        else:
            part1 = 0.0
            try:
                ex1 = math.exp((En(p, mass, **kwargs) + float(a) * mu) / T)
                exm1 = math.exp(-(En(p, mass, **kwargs) + float(a) * mu) / T)
                exm2 = math.exp(-2.0 * (En(p, mass, **kwargs) + float(a) * mu) / T)
                part1 = Phib / (ex1 + 3.0 * Phi * exm1 - exm2 - 3.0 * Phib)
            except OverflowError:
                part1 = 0.0
            part2 = 0.0
            try:
                ex2 = math.exp(2.0 * (En(p, mass, **kwargs) + float(a) * mu) / T)
                ex1 = math.exp((En(p, mass, **kwargs) + float(a) * mu) / T)
                exm1 = math.exp(-(En(p, mass, **kwargs) + float(a) * mu) / T)
                part2 = -2.0 * Phi / (ex2 - 3.0 * Phib * ex1 - exm1 + 3.0 * Phi)
            except OverflowError:
                part2 = 0.0
            part3 = 0.0
            try:
                ex3 = math.exp(3.0 * (En(p, mass, **kwargs) + float(a) * mu) / T)
                ex2 = math.exp(2.0 * (En(p, mass, **kwargs) + float(a) * mu) / T)
                ex1 = math.exp((En(p, mass, **kwargs) + float(a) * mu) / T)
                part3 = 1.0 / (ex3 - 3.0 * Phib * ex2 + 3.0 * Phi * ex1 - 1.0)
            except OverflowError:
                part3 = 0.0
            return ((p ** 4) / En(p, mass, **kwargs)) * (part1 + part2 + part3)
def alt_Omega_Q_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : default_Nf, 'Nc' : default_Nc, 'alt_Omega_Q_real_debug_flag' : False}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    debug_flag = options['alt_Omega_Q_real_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, key): 
        return (alt_z_plus(p, _T, _mu, _Phi, _Phib, M(_T, _mu, **key), 1, **key).real + alt_z_minus(p, _T, _mu, _Phi, _Phib, M(_T, _mu, **key), 1, **key).real)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in alt_Omega_Q_real did not succeed!")

    return -2.0 * Nf * Nc * (1.0 / (3.0 * (math.pi ** 2))) * integral / 2.0 #the last part is taken from thin air, need to figure out if I missed any coefficients in the analytics...
def alt_Omega_Q_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : default_Nf, 'Nc' : default_Nc, 'alt_Omega_Q_imag_debug_flag' : False}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    debug_flag = options['alt_Omega_Q_imag_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, key): 
        return (alt_z_plus(p, _T, _mu, _Phi, _Phib, M(_T, _mu, **key), 1, **key).imag + alt_z_minus(p, _T, _mu, _Phi, _Phib, M(_T, _mu, **key), 1, **key).imag)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in alt_Omega_Q_imag did not succeed!")

    return -2.0 * Nf * Nc * (1.0 / (3.0 * (math.pi ** 2))) * integral / 2.0 #the last part is taken from thin air, need to figure out if I missed any coefficients in the analytics...
def alt_Omega_cluster_real(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, **kwargs) -> float:
    
    options = {'alt_Omega_cluster_real_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['alt_Omega_cluster_real_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, _a, key):
        bound = alt_z_plus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).real + alt_z_minus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).real
        scattering = alt_z_plus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).real + alt_z_minus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).real
        return (bound - scattering) * np.heaviside(_Mth - _M, 0.5)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, bmass, thmass, a, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in alt_Omega_cluster_real did not succeed!")

    if a == 0:
        return ((-1.0) ** (a + 1)) * (1.0 / (3.0 * (math.pi ** 2))) * integral / 4.0 #the last part is taken from thin air, need to figure out if I missed any coefficients in the analytics...
    else:
        return ((-1.0) ** (a + 1)) * (1.0 / (3.0 * (math.pi ** 2))) * integral / 2.0 #the last part is taken from thin air, need to figure out if I missed any coefficients in the analytics...
def alt_Omega_cluster_imag(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, **kwargs) -> float:
    
    options = {'alt_Omega_cluster_imag_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['alt_Omega_cluster_imag_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, _a, key):
        bound = alt_z_plus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).imag + alt_z_minus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).imag
        scattering = alt_z_plus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).imag + alt_z_minus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).imag
        return (bound - scattering) * np.heaviside(_Mth - _M, 0.5)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, bmass, thmass, a, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in alt_Omega_cluster_imag did not succeed!")

    if a == 0:
        return ((-1.0) ** (a + 1)) * (1.0 / (3.0 * (math.pi ** 2))) * integral / 4.0 #the last part is taken from thin air, need to figure out if I missed any coefficients in the analytics...
    else:
        return ((-1.0) ** (a + 1)) * (1.0 / (3.0 * (math.pi ** 2))) * integral / 2.0 #the last part is taken from thin air, need to figure out if I missed any coefficients in the analytics...

def alt_Pressure_Q(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    #
    return -alt_Omega_Q_real(T, mu, Phi, Phib, **kwargs)
def alt_Pressure_cluster(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, dx : int, **kwargs) -> float:
    #
    return -float(dx) * alt_Omega_cluster_real(T, mu, Phi, Phib, bmass, thmass, a, **kwargs)