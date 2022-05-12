import scipy.integrate
import numpy
import math

import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.distributions
import pnjl.aux_functions
import pnjl.defaults

#Grandcanonical potential (PNJL quark part)

def gcp_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:

    options = {'Nf' : pnjl.defaults.default_Nf, 'Nc' : pnjl.defaults.default_Nc, 'gcp_quark_debug_flag' : False}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    debug_flag = options['gcp_quark_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _mass, key):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 1.0, **key)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 2.0, **key)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 3.0, **key)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 1.0, **key)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 2.0, **key)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 3.0, **key)
        fp = pnjl.thermo.distributions.f_fermion_triplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        fm = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi, _Phib, **ym1, **ym2, **ym3)
        return ((p ** 4) / pnjl.aux_functions.En(p, _mass)) * (fp.real + fm.real)

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)

    integral, error = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, mu, Phi, Phib, mass, kwargs))

    return -(Nf / (math.pi ** 2)) * (Nc / 3.0) * integral / 3.0 #the 1/3 factor is arbitrarily added to match https://arxiv.org/pdf/2012.12894.pdf
def gcp_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : pnjl.defaults.default_Nf, 'Nc' : pnjl.defaults.default_Nc, 'gcp_quark_debug_flag' : False}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    debug_flag = options['gcp_quark_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _mass, key):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 1.0, **key)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 2.0, **key)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 3.0, **key)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 1.0, **key)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 2.0, **key)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 3.0, **key)
        fp = pnjl.thermo.distributions.f_fermion_triplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        fm = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi, _Phib, **ym1, **ym2, **ym3)
        return ((p ** 4) / pnjl.aux_functions.En(p, _mass)) * (fp.imag + fm.imag)

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, mu, Phi, Phib, mass, kwargs))

    return -(Nf / (math.pi ** 2)) * (Nc / 3.0) * integral / 3.0 #the 1/3 factor is arbitrarily added to match https://arxiv.org/pdf/2012.12894.pdf

#Extensive thermodynamic properties

def pressure(T : float, mu : float, Phi : complex, Phib : complex, **kwargs):
    #
    return -gcp_real(T, mu, Phi, Phib, **kwargs)

#correct form of the mu vector would be mu_vec = [mu + 2 * h, mu + h, mu - h, mu - 2 * h] with some interval h
#or if mu = 0 then mu_vec = [h, 0.0]
#Phi/Phib vectors should correspond to Phi/Phib at the appropriate values of mu!
def bdensity(T : float, mu_vec : list, Phi_vec : list, Phib_vec : list, **kwargs):
    
    if len(mu_vec) == len(Phi_vec) and len(mu_vec) == len(Phib_vec):
        if len(mu_vec) == 4 and numpy.all(mu_vec[i] > mu_vec[i + 1] for i, el in enumerate(mu_vec[:-1])):
            h = mu_vec[0] - mu_vec[1]
            p_vec = [pressure(T, mu_el, Phi_el, Phib_el, **kwargs) for mu_el, Phi_el, Phib_el in zip(mu_vec, Phi_vec, Phib_vec)]
            return (1.0 / 3.0) * (p_vec[3] - 8.0 * p_vec[2] + 8.0 * p_vec[1] - p_vec[0]) / (12.0 * h)
        elif len(mu_vec) == 2 and mu_vec[0] > mu_vec[1]:
            h = mu_vec[0]
            p_vec = [pressure(T, mu_el, Phi_el, Phib_el, **kwargs) for mu_el, Phi_el, Phib_el in zip(mu_vec, Phi_vec, Phib_vec)]
            return (1.0 / 3.0) * (p_vec[0] - p_vec[1]) / h
        else:
            raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
    else:
        raise RuntimeError("Value vectors don't match!")

#correct form of the T vector would be T_vec = [T + 2 * h, T + h, T - h, T - 2 * h] with some interval h
#Phi/Phib vectors should correspond to Phi/Phib at the appropriate values of T!
def sdensity(T_vec : list, mu : float, Phi_vec : list, Phib_vec : list, **kwargs):
    
    if len(T_vec) == len(Phi_vec) and len(T_vec) == len(Phib_vec):
        if len(T_vec) == 4 and numpy.all(T_vec[i] > T_vec[i + 1] for i, el in enumerate(T_vec[:-1])):
            h = T_vec[0] - T_vec[1]
            p_vec = [pressure(T_el, mu, Phi_el, Phib_el, **kwargs) for T_el, Phi_el, Phib_el in zip(T_vec, Phi_vec, Phib_vec)]
            return (1.0 / 3.0) * (p_vec[3] - 8.0 * p_vec[2] + 8.0 * p_vec[1] - p_vec[0]) / (12.0 * h)
        elif len(T_vec) == 2 and T_vec[0] > T_vec[1]:
            h = T_vec[0]
            p_vec = [pressure(T_el, mu, Phi_el, Phib_el, **kwargs) for T_el, Phi_el, Phib_el in zip(T_vec, Phi_vec, Phib_vec)]
            return (1.0 / 3.0) * (p_vec[0] - p_vec[1]) / h
        else:
            raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
    else:
        raise RuntimeError("Value vectors don't match!")
