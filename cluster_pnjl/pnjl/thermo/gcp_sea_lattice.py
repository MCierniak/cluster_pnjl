import scipy.integrate
import math

import pnjl.aux_functions
import pnjl.defaults

#Mass fit to lattice from https://arxiv.org/pdf/2012.12894.pdf

def Tc(mu : float, **kwargs) -> float:

    options ={'kappa' : pnjl.defaults.default_kappa, 'Tc0' : pnjl.defaults.default_Tc0}
    options.update(kwargs)

    Tc0 = options['Tc0']
    kappa = options['kappa']

    return Tc0 * (1.0 - kappa * (( mu / Tc0) ** 2))
def Delta_ls(T: float, mu : float, **kwargs) -> float:

    options = {'delta_T' : pnjl.defaults.default_delta_T}
    options.update(kwargs)

    delta_T = options['delta_T']

    return 0.5 * (1.0 - math.tanh((T - Tc(mu, **kwargs)) / delta_T))
def M(T : float, mu : float, **kwargs) -> float:

    options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
    options.update(kwargs)

    M0 = options['M0']
    ml = options['ml']

    return M0 * Delta_ls(T, mu, **kwargs) + ml
def dMdmu(T : float, mu : float, **kwargs) -> float:

    options = {'M0' : pnjl.defaults.default_M0, 'kappa' : pnjl.defaults.default_kappa, 'Tc0' : pnjl.defaults.default_Tc0, 'delta_T' : pnjl.defaults.default_delta_T}
    options.update(kwargs)

    M0 = options['M0']
    kappa = options['kappa']
    Tc0 = options['Tc0']
    delta_T = options['delta_T']

    return -((M0 * kappa * mu) / (Tc0 * delta_T)) * ((1.0 / math.cosh((T - Tc(mu, **kwargs)) / delta_T)) ** 2)
def dMdT(T : float, mu : float, **kwargs) -> float:
    
    options = {'M0' : pnjl.defaults.default_M0, 'delta_T' : pnjl.defaults.default_delta_T}
    options.update(kwargs)

    M0 = options['M0']
    delta_T = options['delta_T']

    return -(M0 / (2.0 * delta_T)) * ((1.0 / math.cosh((T - Tc(mu, **kwargs)) / delta_T)) ** 2)
def V(T : float, mu : float, **kwargs) -> float:
    
    options = {'M0' : pnjl.defaults.default_M0, 'Gs' : pnjl.defaults.default_Gs}
    options.update(kwargs)

    M0 = options['M0']
    Gs = options['Gs']

    return ((M0 ** 2) * (Delta_ls(T, mu, **kwargs) ** 2)) / (4.0 * Gs)

#Grandcanonical potential (Fermi sea part) with hard cutoff regularization

def gcp_real(T : float, mu : float, **kwargs) -> float:
    
    options = {'M0' : pnjl.defaults.default_M0, 'Nf' : pnjl.defaults.default_Nf, 'Nc' : pnjl.defaults.default_Nc, 'Lambda' : pnjl.defaults.default_Lambda, 'gcp_sea_lattice_debug_flag' : False}
    options.update(kwargs)

    M0 = options['M0']
    Nf = options['Nf']
    Nc = options['Nc']
    Lambda = options['Lambda']
    debug_flag = options['gcp_sea_lattice_debug_flag']

    integrand = lambda p, _T, _mu, key : (p ** 2) * (pnjl.aux_functions.En(p, M(_T, _mu, **key), **key) - pnjl.aux_functions.En(p, M(0.0, 0.0, **key), **key))

    sigma_contrib = V(T, mu, **kwargs) - V(0.0, 0.0, **kwargs)

    integral, error = scipy.integrate.quad(integrand, 0.0, Lambda, args = (T, mu, kwargs))

    return sigma_contrib - (Nf / (math.pi ** 2)) * (Nc / 3.0) * integral