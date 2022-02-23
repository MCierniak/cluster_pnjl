import scipy.integrate

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
    return ((12.0 * np.pi) / (11 * Nc - 2 * Nf)) * ( (1.0 / (math.log((r ** 2) / (c ** 2)))) - ((c ** 2) / ((r ** 2) - (c ** 2))) )
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
        yp1 = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 1.0, **key)
        yp2 = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 2.0, **key)
        yp3 = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 3.0, **key)
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        return (x / 3.0) * fpe.real

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = scipy.integrate.quad(integrand, mass / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in pnj.thermo.gcp_perturbative.I_plus_real did not succeed!")

    return integral
def I_plus_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'gcp_perturbative_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_perturbative_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        yp1 = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 1.0, **key)
        yp2 = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 2.0, **key)
        yp3 = pnjl.aux_functions.y_plus(x * _T, _T, _mu, 0.0, 1.0, 3.0, **key)
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        return (x / 3.0) * fpe.imag

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = scipy.integrate.quad(integrand, mass / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in pnj.thermo.gcp_perturbative.I_plus_imag did not succeed!")

    return integral
def I_minus_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'gcp_perturbative_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_perturbative_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        yp1 = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 1.0, **key)
        yp2 = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 2.0, **key)
        yp3 = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 3.0, **key)
        fpe = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        return (x / 3.0) * fpe.real

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = scipy.integrate.quad(integrand, mass / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in pnj.thermo.gcp_perturbative.I_minus_real did not succeed!")

    return integral
def I_minus_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'gcp_perturbative_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_perturbative_debug_flag']

    def integrand(x, _mu, _T, _Phi, _Phib, key) :
        yp1 = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 1.0, **key)
        yp2 = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 2.0, **key)
        yp3 = pnjl.aux_functions.y_minus(x * _T, _T, _mu, 0.0, 1.0, 3.0, **key)
        fpe = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        return (x / 3.0) * fpe.imag

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = scipy.integrate.quad(integrand, mass / T, np.inf, args = (mu, T, Phi, Phib, kwargs))

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

    return ((4.0 * Nf) / (3.0 * np.pi)) * alpha_s(T, mu, **kwargs) * (T ** 4) * (Ip_full.real + Im_full.real + (3.0 / (2.0 * (np.pi ** 2))) * spart.real)
def gcp_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : pnjl.defaults.default_Nf}
    options.update(kwargs)

    Nf = options['Nf']

    Ip_full = complex(I_plus_real(T, mu, Phi, Phib, **kwargs), I_plus_imag(T, mu, Phi, Phib, **kwargs))
    Im_full = complex(I_minus_real(T, mu, Phi, Phib, **kwargs), I_minus_imag(T, mu, Phi, Phib, **kwargs))
    spart = (Ip_full + Im_full) ** 2

    return ((4.0 * Nf)/ (3.0 * np.pi)) * alpha_s(T, mu, **kwargs) * (T ** 4) * (Ip_full.imag + Im_full.imag + (3.0 / (2.0 * (np.pi ** 2))) * spart.imag)

#Extensive thermodynamic properties

def pressure(T : float, mu : float, Phi : complex, Phib : complex, **kwargs):
    #
    return -gcp_real(T, mu, Phi, Phib, **kwargs)