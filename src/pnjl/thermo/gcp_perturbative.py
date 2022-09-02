import scipy.integrate
import numpy
import math

import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.distributions
import pnjl.defaults

#Perturbative contribution to the GCP from https://arxiv.org/pdf/2012.12894.pdf

def alpha_s(T : float, mu : float, **kwargs) -> float:
    
    options = {'Nf' : pnjl.defaults.default_Nf, 'Nc' : pnjl.defaults.default_Nc, 'c' : pnjl.defaults.default_c, 'd' : pnjl.defaults.default_d}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    c = options['c']
    d = options['d']

    r = d * T
    return ((12.0 * math.pi) / (11 * Nc - 2 * Nf)) * ( (1.0 / (math.log((r ** 2) / (c ** 2)))) - ((c ** 2) / ((r ** 2) - (c ** 2))) )
def dalpha_s_dT(T : float, mu : float, **kwargs) -> float:
    
    options = {'Nf' : pnjl.defaults.default_Nf, 'Nc' : pnjl.defaults.default_Nc, 'c' : pnjl.defaults.default_c, 'd' : pnjl.defaults.default_d}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    c = options['c']
    d = options['d']

    return ((12.0 * np.pi) / (11 * Nc - 2 * Nf)) * ((2.0 * (c ** 2) * (d ** 2) * T) / (((d ** 2) * (T ** 2) - (c ** 2)) ** 2) - 2.0 / (T * (( math.log(d ** 2) + math.log((T ** 2) / (c ** 2))) ** 2)))
def I_plus_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'gcp_perturbative_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_perturbative_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        yp1 = {}
        yp2 = {}
        yp3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 1.0, **key)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 2.0, **key)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 3.0, **key)
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        return (x / 3.0) * fpe.real

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = scipy.integrate.quad(integrand, mass / T, math.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in pnj.thermo.gcp_perturbative.I_plus_real did not succeed!")

    return integral
def I_plus_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'gcp_perturbative_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_perturbative_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        yp1 = {}
        yp2 = {}
        yp3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 1.0, **key)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 2.0, **key)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 3.0, **key)
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        return (x / 3.0) * fpe.imag

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = scipy.integrate.quad(integrand, mass / T, math.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in pnj.thermo.gcp_perturbative.I_plus_imag did not succeed!")

    return integral
def I_minus_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'gcp_perturbative_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_perturbative_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        yp1 = {}
        yp2 = {}
        yp3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 1.0, **key)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 2.0, **key)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 3.0, **key)
        fpe = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        return (x / 3.0) * fpe.real

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = scipy.integrate.quad(integrand, mass / T, math.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in pnj.thermo.gcp_perturbative.I_minus_real did not succeed!")

    return integral
def I_minus_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'gcp_perturbative_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_perturbative_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        yp1 = {}
        yp2 = {}
        yp3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 1.0, **key)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 2.0, **key)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 3.0, **key)
        fpe = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        return (x / 3.0) * fpe.imag

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = scipy.integrate.quad(integrand, mass / T, math.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in pnj.thermo.gcp_perturbative.I_minus_imag did not succeed!")

    return integral

#Grandcanonical potential (perturbative part)

def gcp_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : pnjl.defaults.default_Nf}
    options.update(kwargs)

    Nf = options['Nf']

    Ip_full = complex(I_plus_real(T, mu, Phi, Phib, **kwargs), I_plus_imag(T, mu, Phi, Phib, **kwargs))
    Im_full = complex(I_minus_real(T, mu, Phi, Phib, **kwargs), I_minus_imag(T, mu, Phi, Phib, **kwargs))
    spart = (Ip_full + Im_full) ** 2

    ms_kwargs = {}
    for key in kwargs:
        ms_kwargs[key] = kwargs[key]
    ms_kwargs['ml'] = 100.0

    s_Ip_full = complex(I_plus_real(T, mu, Phi, Phib, **ms_kwargs), I_plus_imag(T, mu, Phi, Phib, **ms_kwargs))
    s_Im_full = complex(I_minus_real(T, mu, Phi, Phib, **ms_kwargs), I_minus_imag(T, mu, Phi, Phib, **ms_kwargs))
    s_spart = (Ip_full + Im_full) ** 2

    l_part = (8.0 / (3.0 * math.pi)) * (Ip_full.real + Im_full.real + (3.0 / (2.0 * (math.pi ** 2))) * spart.real)
    s_part = (4.0 / (3.0 * math.pi)) * (s_Ip_full.real + s_Im_full.real + (3.0 / (2.0 * (math.pi ** 2))) * s_spart.real)

    return alpha_s(T, mu, **kwargs) * (T ** 4) * (l_part + s_part)
def gcp_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : pnjl.defaults.default_Nf}
    options.update(kwargs)

    Nf = options['Nf']

    Ip_full = complex(I_plus_real(T, mu, Phi, Phib, **kwargs), I_plus_imag(T, mu, Phi, Phib, **kwargs))
    Im_full = complex(I_minus_real(T, mu, Phi, Phib, **kwargs), I_minus_imag(T, mu, Phi, Phib, **kwargs))
    spart = (Ip_full + Im_full) ** 2

    ms_kwargs = {}
    for key in kwargs:
        ms_kwargs[key] = kwargs[key]
    ms_kwargs['ml'] = 100.0

    s_Ip_full = complex(I_plus_real(T, mu, Phi, Phib, **ms_kwargs), I_plus_imag(T, mu, Phi, Phib, **ms_kwargs))
    s_Im_full = complex(I_minus_real(T, mu, Phi, Phib, **ms_kwargs), I_minus_imag(T, mu, Phi, Phib, **ms_kwargs))
    s_spart = (Ip_full + Im_full) ** 2

    l_part = (8.0 / (3.0 * math.pi)) * (Ip_full.imag + Im_full.imag + (3.0 / (2.0 * (math.pi ** 2))) * spart.imag)
    s_part = (4.0 / (3.0 * math.pi)) * (s_Ip_full.imag + s_Im_full.imag + (3.0 / (2.0 * (math.pi ** 2))) * s_spart.imag)

    return alpha_s(T, mu, **kwargs) * (T ** 4) * (l_part + s_part)

#Extensive thermodynamic properties

def pressure(T : float, mu : float, Phi : complex, Phib : complex, **kwargs):
    #
    return -gcp_real(T, mu, Phi, Phib, **kwargs)

def bdensity(T : float, mu : float, Phi : complex, Phib : complex, **kwargs):
    
    h = 1e-2
    mu_vec = []
    Phi_vec = []
    Phib_vec = []
    if mu - 2 * h > 0.0:
        mu_vec = [mu + 2 * h, mu + h, mu - h, mu - 2 * h]
    else:
        mu_vec = [mu + h, mu]
        if numpy.any([el < 0.0 for el in mu_vec]):
            return bdensity(T, mu + h, Phi, Phib, **kwargs)
    Phi_vec = [Phi for el in mu_vec]
    Phib_vec = [Phib for el in mu_vec]

    if len(mu_vec) == len(Phi_vec) and len(mu_vec) == len(Phib_vec):
        if len(mu_vec) == 4 and numpy.all(mu_vec[i] > mu_vec[i + 1] for i, el in enumerate(mu_vec[:-1])):
            p_vec = [pressure(T, mu_el, Phi_el, Phib_el, **kwargs) for mu_el, Phi_el, Phib_el in zip(mu_vec, Phi_vec, Phib_vec)]
            return (1.0 / 3.0) * (p_vec[3] - 8.0 * p_vec[2] + 8.0 * p_vec[1] - p_vec[0]) / (12.0 * h)
        elif len(mu_vec) == 2 and mu_vec[0] > mu_vec[1]:
            p_vec = [pressure(T, mu_el, Phi_el, Phib_el, **kwargs) for mu_el, Phi_el, Phib_el in zip(mu_vec, Phi_vec, Phib_vec)]
            return (1.0 / 3.0) * (p_vec[0] - p_vec[1]) / h
        else:
            raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
    else:
        raise RuntimeError("Value vectors don't match!")

def bnumber_cumulant(rank : int, T : float, mu : float, Phi : complex, Phib : complex, **kwargs):

    if rank == 1:
        #return bdensity(T, mu, Phi, Phib, **kwargs) / (T ** 4)
        return bdensity(T, mu, Phi, Phib, **kwargs)
    elif rank < 1:
        raise RuntimeError("Cumulant rank lower than 1!")
    else:
        h = 1e-2
        mu_vec = []
        Phi_vec = []
        Phib_vec = []
        if mu - 2 * h > 0.0:
            mu_vec = [mu + 2 * h, mu + h, mu - h, mu - 2 * h]
        else:
            mu_vec = [mu + h, mu]
            if numpy.any([el < 0.0 for el in mu_vec]):
                return bnumber_cumulant(rank, T, mu + h, Phi, Phib, **kwargs)
        Phi_vec = [Phi for el in mu_vec]
        Phib_vec = [Phib for el in mu_vec]

        if len(mu_vec) == len(Phi_vec) and len(mu_vec) == len(Phib_vec):
            if len(mu_vec) == 4 and numpy.all(mu_vec[i] > mu_vec[i + 1] for i, el in enumerate(mu_vec[:-1])):
                out_vec = [bnumber_cumulant(rank - 1, T, mu_el, Phi_el, Phib_el, **kwargs) for mu_el, Phi_el, Phib_el in zip(mu_vec, Phi_vec, Phib_vec)]
                #return (T / 3.0) * (out_vec[3] - 8.0 * out_vec[2] + 8.0 * out_vec[1] - out_vec[0]) / (12.0 * h)
                return (1.0 / 3.0) * (out_vec[3] - 8.0 * out_vec[2] + 8.0 * out_vec[1] - out_vec[0]) / (12.0 * h)
            elif len(mu_vec) == 2 and mu_vec[0] > mu_vec[1]:
                out_vec = [bnumber_cumulant(rank - 1, T, mu_el, Phi_el, Phib_el, **kwargs) for mu_el, Phi_el, Phib_el in zip(mu_vec, Phi_vec, Phib_vec)]
                #return (T / 3.0) * (out_vec[0] - out_vec[1]) / h
                return (1.0 / 3.0) * (out_vec[0] - out_vec[1]) / h
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