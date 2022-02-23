#LEGACY CODE, DO NOT USE

import numpy as np
import scipy as sp
import cmath
import math

from scipy.integrate import quad

import warnings
warnings.filterwarnings("ignore")

#to be replaced/removed
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
##### end of replace/remove ########