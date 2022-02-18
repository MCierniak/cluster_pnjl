import math

import pnjl.defaults

#General

def En(p : float, mass : float, **kwargs) -> float:
    #
    return math.sqrt((p ** 2) + (mass ** 2))
def dEn_dmu(p : float, T : float, mu : float, mass : float, deriv_M_mu : float, **kwargs) -> float:
    #
    return (mass / En(p, mass, **kwargs)) * deriv_M_mu
def dEn_dT(p : float, T : float, mu : float, mass : float, deriv_M_T : float, **kwargs) -> float:
    #
    return (mass / En(p, mass, **kwargs)) * deriv_M_T
def y_plus(p : float, T : float, mu : float, mass : float, a : float, b : float, **kwargs) -> (float, int):
    body = b * (En(p, mass, **kwargs) - a * mu) / T
    if body > 709.7827:
        return (math.inf, 0)
    elif body < -709.7827:
        return (0.0, 1)
    elif body == 0.0:
        return (1.0, 2)
    else:
        return (math.exp(body), 3)
def y_minus(p : float, T : float, mu : float, mass : float, a : float, b : float, **kwargs) -> (float, int):
    body = b * (En(p, mass, **kwargs) + a * mu) / T
    if body > 709.7827:
        return (math.inf, 0)
    elif body < -709.7827:
        return (0.0, 1)
    elif body == 0.0:
        return (1.0, 2)
    else:
        return (math.exp(body), 3)

#Polynomial approximation of the Polyakov-loop potential from https://arxiv.org/pdf/hep-ph/0506234.pdf

def b2(T : float, **kwargs) -> float:
    
    options = {'T0' : pnjl.defaults.default_T0, 'a0' : pnjl.defaults.default_a0, 'a1' : pnjl.defaults.default_a1, 'a2' : pnjl.defaults.default_a2, 'a3' : pnjl.defaults.default_a3}
    options.update(kwargs)

    T0 = options['T0']
    a0 = options['a0']
    a1 = options['a1']
    a2 = options['a2']
    a3 = options['a3']

    return a0 + a1 * (T0 / T) + a2 * ((T0 / T) ** 2) + a3 * ((T0 / T) ** 3)
def U(T : float, Phi : complex, Phib : complex, **kwargs) -> complex:
    
    options = {'b3' : pnjl.defaults.default_b3, 'b4' : pnjl.defaults.default_b4}
    options.update(kwargs)

    b3 = options['b3']
    b4 = options['b4']

    return -(T ** 4) * ((b2(T, **kwargs) / 2.0) * Phi * Phib + (b3 / 6.0) * ((Phi ** 3) + (Phib ** 3)) - (b4 / 4.0) * ((Phi * Phib) ** 2))

#Running coupling of the perturbative contribution to the PNJL thermodynamic potential from https://arxiv.org/pdf/2012.12894.pdf

def alpha_s(T : float, mu : float, **kwargs) -> float:
    
    options = {'Nf' : pnjl.defaults.default_Nf, 'Nc' : pnjl.defaults.default_Nc, 'c' : pnjl.defaults.default_c, 'd' : pnjl.defaults.default_d}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    c = options['c']
    d = options['d']

    r = d * T
    return ((12.0 * np.pi) / (11 * Nc - 2 * Nf)) * ( (1.0 / (math.log((r ** 2) / (c ** 2)))) - ((c ** 2) / ((r ** 2) - (c ** 2))) )
def dalpha_s_dT(T : float, mu : float, **kwargs) -> float:
    
    options = {'Nf' : pnjl.defaults.default_Nf, 'Nc' : pnjl.defaults.default_Nc, 'c' : pnjl.defaults.default_c, 'd' : pnjl.defaults.default_d}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    c = options['c']
    d = options['d']

    return ((12.0 * np.pi) / (11 * Nc - 2 * Nf)) * ((2.0 * (c ** 2) * (d ** 2) * T) / (((d ** 2) * (T ** 2) - (c ** 2)) ** 2) - 2.0 / (T * (( math.log(d ** 2) + math.log((T ** 2) / (c ** 2))) ** 2)))