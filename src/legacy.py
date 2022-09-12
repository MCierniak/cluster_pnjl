"""#LEGACY CODE, DO NOT USE

#pnjl_functions.py

import numpy as np
import scipy as sp
import cmath
import math

from scipy.integrate import quad

import warnings
warnings.filterwarnings("ignore")

#import matplotlib.pyplot as plt
#import numpy as np
#import scipy as sp
#import time
#import math
#import csv
#
#import papers.epja_2022
#
#from scipy.interpolate import UnivariateSpline, Akima1DInterpolator
#from scipy.optimize import dual_annealing, basinhopping
#from scipy.special import binom
#
#from scipy.signal import find_peaks
#
#from matplotlib.patches import Polygon, FancyArrowPatch
#
#from random import uniform, randint
#
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
#
#from joblib import Parallel, delayed

#from pnjl_functions import alt_Omega_Q_real, alt_Omega_Q_imag
#from pnjl_functions import Omega_pert_real, Omega_pert_imag
#from pnjl_functions import Omega_g_real, Omega_g_imag
#from pnjl_functions import alt_Omega_cluster_real, alt_Omega_cluster_imag
#from pnjl_functions import Omega_Delta

#from pnjl_functions import M, dMdmu, dMdT, Delta_ls, Tc
#from pnjl_functions import alt_Pressure_Q, BDensity_Q, SDensity_Q 
#from pnjl_functions import Pressure_g, SDensity_g
#from pnjl_functions import Pressure_pert, BDensity_pert, SDensity_pert
#from pnjl_functions import alt_Pressure_cluster, BDensity_cluster, SDensity_cluster

#from pnjl_functions import default_MN, default_MM, default_MD, default_MF, default_MT, default_MP, default_MQ, default_MH
#from pnjl_functions import default_M0

#from utils import data_collect

#np.seterr(all = 'raise')

#continuum_lambda_basis = math.sqrt(2) * M(0, 0) * 0.0001

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

#cluster_pnjl.py

#TODO:
#1. make BDensity and SDensity routines.
#2. add hexa-diquark cluster
#3. consider d phi / d (mu/T).
#4. consider a mu dependent PL potential of https://arxiv.org/abs/1207.4890
#5. return to sigma/phi derivation using a momentum-dependent quark mass (check Omega_Delta!).
#6. change the pl potential to this one https://arxiv.org/pdf/1307.5958.pdf

import matplotlib

import matplotlib.patches
import matplotlib.pyplot
import scipy.integrate
import scipy.optimize
import numpy
import time
import glob
import math
import tqdm
import csv
import os

from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pnjl.thermo.gcp_cluster.bound_step_continuum_arccos_cos as cluster
import pnjl.thermo.gcp_cluster.bound_step_continuum_step
import pnjl.thermo.gcp_pl_polynomial
import pnjl.thermo.gcp_perturbative
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_quark
import pnjl.defaults
import utils

import papers.epja_2022

import warnings
warnings.filterwarnings("ignore")

def PNJL_thermodynamics_mu_test():

    phi_re_mu0          = []
    phi_im_mu0          = []
    phi_re_mu200        = []
    phi_im_mu200        = []
    phi_re_mu300        = []
    phi_im_mu300        = []
    Pres_Q_mu0          = []
    Pres_g_mu0          = []
    Pres_pert_mu0       = []
    Pres_Q_mu200        = []
    Pres_g_mu200        = []
    Pres_pert_mu200     = []
    Pres_Q_mu300        = []
    Pres_g_mu300        = []
    Pres_pert_mu300     = []
    Pres_pi_mu0         = []
    Pres_rho_mu0        = []
    Pres_omega_mu0      = []
    Pres_D_mu0          = []
    Pres_N_mu0          = []
    Pres_T_mu0          = []
    Pres_F_mu0          = []
    Pres_P_mu0          = []
    Pres_Q5_mu0         = []
    Pres_H_mu0          = []
    Pres_pi_mu200       = []
    Pres_rho_mu200      = []
    Pres_omega_mu200    = []
    Pres_D_mu200        = []
    Pres_N_mu200        = []
    Pres_T_mu200        = []
    Pres_F_mu200        = []
    Pres_P_mu200        = []
    Pres_Q5_mu200       = []
    Pres_H_mu200        = []
    Pres_pi_mu300       = []
    Pres_rho_mu300      = []
    Pres_omega_mu300    = []
    Pres_D_mu300        = []
    Pres_N_mu300        = []
    Pres_T_mu300        = []
    Pres_F_mu300        = []
    Pres_P_mu300        = []
    Pres_Q5_mu300       = []
    Pres_H_mu300        = []
    
    T = numpy.linspace(1.0, 450.0, 200)
    mu0   = [0.0 for el in T]
    mu200 = [200.0 / 3.0 for el in T]
    mu300 = [300.0 / 3.0 for el in T]
    
    recalc_pl_mu0         = False
    recalc_pl_mu200       = False
    recalc_pl_mu300       = False
    recalc_pressure_mu0   = False
    recalc_pressure_mu200 = False
    recalc_pressure_mu300 = False

    cluster_backreaction  = False
    pl_turned_off         = False

    pl_mu0_file   = "D:/EoS/BDK/mu_test/pl_mu0.dat"
    pl_mu200_file = "D:/EoS/BDK/mu_test/pl_mu200.dat"
    pl_mu300_file = "D:/EoS/BDK/mu_test/pl_mu300.dat"
    pressure_mu0_file   = "D:/EoS/BDK/mu_test/pressure_c_mu0.dat"
    pressure_mu200_file = "D:/EoS/BDK/mu_test/pressure_c_mu200.dat"
    pressure_mu300_file = "D:/EoS/BDK/mu_test/pressure_c_mu300.dat"

    if recalc_pl_mu0:
        phi_re_mu0.append(1e-15)
        phi_im_mu0.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm.tqdm(zip(T, mu0), desc = "Traced Polyakov loop (mu = 0)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu0[-1], phi_im_mu0[-1], with_clusters = cluster_backreaction)
            phi_re_mu0.append(temp_phi_re)
            phi_im_mu0.append(temp_phi_im)
        phi_re_mu0 = phi_re_mu0[1:]
        phi_im_mu0 = phi_im_mu0[1:]
        with open(pl_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu0, phi_im_mu0)])
    else:
        T, phi_re_mu0 = utils.data_collect(0, 1, pl_mu0_file)
        phi_im_mu0, _ = utils.data_collect(2, 2, pl_mu0_file)

    if recalc_pl_mu200:
        phi_re_mu200.append(1e-15)
        phi_im_mu200.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm.tqdm(zip(T, mu200), desc = "Traced Polyakov loop (mu = 200)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu200[-1], phi_im_mu200[-1], with_clusters = cluster_backreaction)
            phi_re_mu200.append(temp_phi_re)
            phi_im_mu200.append(temp_phi_im)
        phi_re_mu200 = phi_re_mu200[1:]
        phi_im_mu200 = phi_im_mu200[1:]
        with open(pl_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu200, phi_im_mu200)])
    else:
        T, phi_re_mu200 = utils.data_collect(0, 1, pl_mu200_file)
        phi_im_mu200, _ = utils.data_collect(2, 2, pl_mu200_file)
    
    if recalc_pl_mu300:
        phi_re_mu300.append(1e-15)
        phi_im_mu300.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm.tqdm(zip(T, mu300), desc = "Traced Polyakov loop (mu = 300)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu300[-1], phi_im_mu300[-1], with_clusters = cluster_backreaction)
            phi_re_mu300.append(temp_phi_re)
            phi_im_mu300.append(temp_phi_im)
        phi_re_mu300 = phi_re_mu300[1:]
        phi_im_mu300 = phi_im_mu300[1:]
        with open(pl_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu300, phi_im_mu300)])
    else:
        T, phi_re_mu300 = utils.data_collect(0, 1, pl_mu300_file)
        phi_im_mu300, _ = utils.data_collect(2, 2, pl_mu300_file)

    if recalc_pressure_mu0:
        Pres_g_mu0 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T, phi_re_mu0, phi_im_mu0), 
                desc = "Gluon pressure (mu = 0)", 
                total = len(T), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu0 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu0, phi_re_mu0, phi_im_mu0), 
                    desc = "Quark pressure (mu = 0)", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert_mu0 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu0, phi_re_mu0, phi_im_mu0), 
                    desc = "Perturbative pressure (mu = 0)", 
                    total = len(T), 
                    ascii = True
                    )]        
            (
                (Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu0, phi_re_mu0, phi_im_mu0)
        else:
            Pres_Q_mu0 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu0, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 0)", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert_mu0 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu0, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 0)", 
                    total = len(T), 
                    ascii = True
                    )]        
            (
                (Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu0, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu0, Pres_g_mu0, Pres_pert_mu0, Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0)])
    else:
        Pres_Q_mu0, Pres_g_mu0       = utils.data_collect(0, 1, pressure_mu0_file)
        Pres_pert_mu0, Pres_pi_mu0   = utils.data_collect(2, 3, pressure_mu0_file)
        Pres_rho_mu0, Pres_omega_mu0 = utils.data_collect(4, 5, pressure_mu0_file)
        Pres_D_mu0, Pres_N_mu0       = utils.data_collect(6, 7, pressure_mu0_file)
        Pres_T_mu0, Pres_F_mu0       = utils.data_collect(8, 9, pressure_mu0_file)
        Pres_P_mu0, Pres_Q5_mu0      = utils.data_collect(10, 11, pressure_mu0_file)
        Pres_H_mu0, _                = utils.data_collect(12, 12, pressure_mu0_file)

    if recalc_pressure_mu200:
        Pres_g_mu200 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T, phi_re_mu200, phi_im_mu200), 
                desc = "Gluon pressure (mu = 200)", 
                total = len(T), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu200 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu200, phi_re_mu200, phi_im_mu200), 
                    desc = "Quark pressure (mu = 200)", 
                    total = len(T), 
                    ascii = True
                    )]        
            Pres_pert_mu200 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu200, phi_re_mu200, phi_im_mu200), 
                    desc = "Perturbative pressure (mu = 200)", 
                    total = len(T), 
                    ascii = True
                    )]        
            (
                (Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu200, phi_re_mu200, phi_im_mu200)
        else:
            Pres_Q_mu200 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu200, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 200)", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert_mu200 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu200, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 200)", 
                    total = len(T), 
                    ascii = True
                    )]        
            (
                (Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu200, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu200, Pres_g_mu200, Pres_pert_mu200, Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200)])
    else:
        Pres_Q_mu200, Pres_g_mu200       = utils.data_collect(0, 1, pressure_mu200_file)
        Pres_pert_mu200, Pres_pi_mu200   = utils.data_collect(2, 3, pressure_mu200_file)
        Pres_rho_mu200, Pres_omega_mu200 = utils.data_collect(4, 5, pressure_mu200_file)
        Pres_D_mu200, Pres_N_mu200       = utils.data_collect(6, 7, pressure_mu200_file)
        Pres_T_mu200, Pres_F_mu200       = utils.data_collect(8, 9, pressure_mu200_file)
        Pres_P_mu200, Pres_Q5_mu200      = utils.data_collect(10, 11, pressure_mu200_file)
        Pres_H_mu200, _                  = utils.data_collect(12, 12, pressure_mu200_file)
    
    if recalc_pressure_mu300:
        Pres_g_mu300 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T, phi_re_mu300, phi_im_mu300), 
                desc = "Gluon pressure (mu = 300)", 
                total = len(T), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu300 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu300, phi_re_mu300, phi_im_mu300), 
                    desc = "Quark pressure (mu = 300)", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert_mu300 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu300, phi_re_mu300, phi_im_mu300), 
                    desc = "Perturbative pressure (mu = 300)", 
                    total = len(T), 
                    ascii = True
                    )]        
            (
                (Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu300, phi_re_mu300, phi_im_mu300)
        else:
            Pres_Q_mu300 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el in 
                tqdm.tqdm(
                    zip(T, mu300, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 300)", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert_mu300 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu300, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 300)", 
                    total = len(T), 
                    ascii = True
                    )]
            (
                (Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu300, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu300, Pres_g_mu300, Pres_pert_mu300, Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300)])
    else:
        Pres_Q_mu300, Pres_g_mu300       = utils.data_collect(0, 1, pressure_mu300_file)
        Pres_pert_mu300, Pres_pi_mu300   = utils.data_collect(2, 3, pressure_mu300_file)
        Pres_rho_mu300, Pres_omega_mu300 = utils.data_collect(4, 5, pressure_mu300_file)
        Pres_D_mu300, Pres_N_mu300       = utils.data_collect(6, 7, pressure_mu300_file)
        Pres_T_mu300, Pres_F_mu300       = utils.data_collect(8, 9, pressure_mu300_file)
        Pres_P_mu300, Pres_Q5_mu300      = utils.data_collect(10, 11, pressure_mu300_file)
        Pres_H_mu300, _                  = utils.data_collect(12, 12, pressure_mu300_file)

    contrib_q_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q_mu0   )]
    contrib_g_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_g_mu0   )]
    contrib_pert_mu0                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pert_mu0)]
    contrib_qgp_mu0                 = [sum(el) for el in zip(contrib_q_mu0, contrib_g_mu0, contrib_pert_mu0)]
    contrib_q_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q_mu200   )]
    contrib_g_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_g_mu200   )]
    contrib_pert_mu200              = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pert_mu200)]
    contrib_qgp_mu200               = [sum(el) for el in zip(contrib_q_mu200, contrib_g_mu200, contrib_pert_mu200)]
    contrib_q_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q_mu300   )]
    contrib_g_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_g_mu300   )]
    contrib_pert_mu300              = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pert_mu300)]
    contrib_qgp_mu300               = [sum(el) for el in zip(contrib_q_mu300, contrib_g_mu300, contrib_pert_mu300)]
    contrib_pi_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi_mu0)]
    contrib_rho_mu0                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho_mu0)]
    contrib_omega_mu0               = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega_mu0)]
    contrib_D_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D_mu0)]
    contrib_N_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N_mu0)]
    contrib_T_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T_mu0)]
    contrib_F_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F_mu0)]
    contrib_P_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P_mu0)]
    contrib_Q5_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5_mu0)]
    contrib_H_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H_mu0)]
    contrib_cluster_mu0             = [sum(el) for el in zip(contrib_pi_mu0, contrib_rho_mu0, contrib_omega_mu0, contrib_D_mu0, contrib_N_mu0, contrib_T_mu0, contrib_F_mu0, contrib_P_mu0, contrib_Q5_mu0, contrib_H_mu0)]
    contrib_cluster_singlet_mu0     = [sum(el) for el in zip(contrib_pi_mu0, contrib_rho_mu0, contrib_omega_mu0, contrib_N_mu0, contrib_T_mu0, contrib_P_mu0, contrib_H_mu0)]
    contrib_cluster_color_mu0       = [sum(el) for el in zip(contrib_D_mu0, contrib_F_mu0, contrib_Q5_mu0)]
    contrib_pi_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi_mu200)]
    contrib_rho_mu200               = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho_mu200)]
    contrib_omega_mu200             = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega_mu200)]
    contrib_D_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D_mu200)]
    contrib_N_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N_mu200)]
    contrib_T_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T_mu200)]
    contrib_F_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F_mu200)]
    contrib_P_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P_mu200)]
    contrib_Q5_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5_mu200)]
    contrib_H_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H_mu200)]
    contrib_cluster_mu200           = [sum(el) for el in zip(contrib_pi_mu200, contrib_rho_mu200, contrib_omega_mu200, contrib_D_mu200, contrib_N_mu200, contrib_T_mu200, contrib_F_mu200, contrib_P_mu200, contrib_Q5_mu200, contrib_H_mu200)]
    contrib_cluster_singlet_mu200   = [sum(el) for el in zip(contrib_pi_mu200, contrib_rho_mu200, contrib_omega_mu200, contrib_N_mu200, contrib_T_mu200, contrib_P_mu200, contrib_H_mu200)]
    contrib_cluster_color_mu200     = [sum(el) for el in zip(contrib_D_mu200, contrib_F_mu200, contrib_Q5_mu200)]
    contrib_pi_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi_mu300)]
    contrib_rho_mu300               = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho_mu300)]
    contrib_omega_mu300             = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega_mu300)]
    contrib_D_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D_mu300)]
    contrib_N_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N_mu300)]
    contrib_T_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T_mu300)]
    contrib_F_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F_mu300)]
    contrib_P_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P_mu300)]
    contrib_Q5_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5_mu300)]
    contrib_H_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H_mu300)]
    contrib_cluster_mu300           = [sum(el) for el in zip(contrib_pi_mu300, contrib_rho_mu300, contrib_omega_mu300, contrib_D_mu300, contrib_N_mu300, contrib_T_mu300, contrib_F_mu300, contrib_P_mu300, contrib_Q5_mu300, contrib_H_mu300)]
    contrib_cluster_singlet_mu300   = [sum(el) for el in zip(contrib_pi_mu300, contrib_rho_mu300, contrib_omega_mu300, contrib_N_mu300, contrib_T_mu300, contrib_P_mu300, contrib_H_mu300)]
    contrib_cluster_color_mu300     = [sum(el) for el in zip(contrib_D_mu300, contrib_F_mu300, contrib_Q5_mu300)]

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y)       = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)     = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_low.dat")
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_high.dat")

    borsanyi_1204_6710v2_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = numpy.array(borsanyi_1204_6710v2_mu0)
    borsanyi_1204_6710v2_mu200 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu200_x[::-1], low_1204_6710v2_mu200_y[::-1]):
        borsanyi_1204_6710v2_mu200.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu200 = numpy.array(borsanyi_1204_6710v2_mu200)
    borsanyi_1204_6710v2_mu300 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu300_x[::-1], low_1204_6710v2_mu300_y[::-1]):
        borsanyi_1204_6710v2_mu300.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu300 = numpy.array(borsanyi_1204_6710v2_mu300)

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([0., 450., 0., 4.0])
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=0}$'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'red', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=200}$ MeV'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu300, closed = True, fill = True, color = 'yellow', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=300}$ MeV'))
    #ax1.plot(T, contrib_q_mu0, '-', c = 'blue', label = r'$\mathrm{P_{Q,0}}$')
    #ax1.plot(T, contrib_g_mu0, '-', c = 'red', label = r'$\mathrm{P_{g,0}}$')
    #ax1.plot(T, contrib_pert_mu0, '-', c = 'pink', label = r'$\mathrm{P_{pert,0}}$')
    ax1.plot(T, contrib_qgp_mu0, '-', c = 'black', label = r'$\mathrm{P_{QGP,0}}$')
    ax1.plot(T, contrib_cluster_mu0, '-', c = 'blue', label = r'$\mathrm{P_{cluster,0}}$')
    ax1.plot(T, contrib_cluster_singlet_mu0, '-', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,0}}$')
    ax1.plot(T, contrib_cluster_color_mu0, '-', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,0}}$')
    #ax1.plot(T, contrib_q_mu200, '--', c = 'blue', label = r'$\mathrm{P_{Q,200}}$')
    #ax1.plot(T, contrib_g_mu200, '--', c = 'red', label = r'$\mathrm{P_{g,200}}$')
    #ax1.plot(T, contrib_pert_mu200, '--', c = 'pink', label = r'$\mathrm{P_{pert,200}}$')
    ax1.plot(T, contrib_qgp_mu200, '--', c = 'black', label = r'$\mathrm{P_{QGP,200}}$')
    ax1.plot(T, contrib_cluster_mu200, '--', c = 'blue', label = r'$\mathrm{P_{cluster,200}}$')
    ax1.plot(T, contrib_cluster_singlet_mu200, '--', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,200}}$')
    ax1.plot(T, contrib_cluster_color_mu200, '--', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,200}}$')
    #ax1.plot(T, contrib_q_mu300, '-.', c = 'blue', label = r'$\mathrm{P_{Q,300}}$')
    #ax1.plot(T, contrib_g_mu300, '-.', c = 'red', label = r'$\mathrm{P_{g,300}}$')
    #ax1.plot(T, contrib_pert_mu300, '-.', c = 'pink', label = r'$\mathrm{P_{pert,300}}$')
    ax1.plot(T, contrib_qgp_mu300, '-.', c = 'black', label = r'$\mathrm{P_{QGP,300}}$')
    ax1.plot(T, contrib_cluster_mu300, '-.', c = 'blue', label = r'$\mathrm{P_{cluster,300}}$')
    ax1.plot(T, contrib_cluster_singlet_mu300, '-.', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,300}}$')
    ax1.plot(T, contrib_cluster_color_mu300, '-.', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,300}}$')
    #ax1.legend(loc = 2)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig2 = matplotlib.pyplot.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.axis([0., 450., 0., 1.2])
    ax2.plot(T, [pnjl.thermo.gcp_sea_lattice.M(el, 0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T], '-', c = 'green')
    ax2.plot(T, [pnjl.thermo.gcp_sea_lattice.M(el, 200.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T], '--', c = 'green')
    ax2.plot(T, [pnjl.thermo.gcp_sea_lattice.M(el, 300.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T], '-.', c = 'green')
    ax2.plot(T, phi_re_mu0, '-', c = 'blue')
    ax2.plot(T, phi_re_mu200, '--', c = 'blue')
    ax2.plot(T, phi_re_mu300, '-.', c = 'blue')
    ax2.plot(T, phi_im_mu0, '-', c = 'red')
    ax2.plot(T, phi_im_mu200, '--', c = 'red')
    ax2.plot(T, phi_im_mu300, '-.', c = 'red')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{\Phi}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    fig2.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def PNJL_thermodynamics_continuum_test():

    phi_re_mu0, phi_re_mu200, phi_re_mu300, phi_re_mu400, phi_re_mu500 = [], [], [], [], []
    phi_im_mu0, phi_im_mu200, phi_im_mu300, phi_im_mu400, phi_im_mu500 = [], [], [], [], []

    Pres_Q_mu0, Pres_Q_mu200, Pres_Q_mu300, Pres_Q_mu400, Pres_Q_mu500 = [], [], [], [], []
    Pres_g_mu0, Pres_g_mu200, Pres_g_mu300, Pres_g_mu400, Pres_g_mu500 = [], [], [], [], []
    Pres_pert_mu0, Pres_pert_mu200, Pres_pert_mu300, Pres_pert_mu400, Pres_pert_mu500 = [], [], [], [], []

    Pres_pi_step_mu0, Pres_pi_step_mu200, Pres_pi_step_mu300, Pres_pi_step_mu400, Pres_pi_step_mu500 = [], [], [], [], []
    Pres_K_step_mu0, Pres_K_step_mu200, Pres_K_step_mu300, Pres_K_step_mu400, Pres_K_step_mu500 = [], [], [], [], []
    Pres_rho_step_mu0, Pres_rho_step_mu200, Pres_rho_step_mu300, Pres_rho_step_mu400, Pres_rho_step_mu500 = [], [], [], [], []
    Pres_omega_step_mu0, Pres_omega_step_mu200, Pres_omega_step_mu300, Pres_omega_step_mu400, Pres_omega_step_mu500 = [], [], [], [], []
    Pres_D_step_mu0, Pres_D_step_mu200, Pres_D_step_mu300, Pres_D_step_mu400, Pres_D_step_mu500 = [], [], [], [], []
    Pres_N_step_mu0, Pres_N_step_mu200, Pres_N_step_mu300, Pres_N_step_mu400, Pres_N_step_mu500 = [], [], [], [], []
    Pres_T_step_mu0, Pres_T_step_mu200, Pres_T_step_mu300, Pres_T_step_mu400, Pres_T_step_mu500 = [], [], [], [], []
    Pres_F_step_mu0, Pres_F_step_mu200, Pres_F_step_mu300, Pres_F_step_mu400, Pres_F_step_mu500 = [], [], [], [], []
    Pres_P_step_mu0, Pres_P_step_mu200, Pres_P_step_mu300, Pres_P_step_mu400, Pres_P_step_mu500 = [], [], [], [], []
    Pres_Q5_step_mu0, Pres_Q5_step_mu200, Pres_Q5_step_mu300, Pres_Q5_step_mu400, Pres_Q5_step_mu500 = [], [], [], [], []
    Pres_H_step_mu0, Pres_H_step_mu200, Pres_H_step_mu300, Pres_H_step_mu400, Pres_H_step_mu500 = [], [], [], [], []

    Pres_pi_c_mu0, Pres_pi_c_mu200, Pres_pi_c_mu300, Pres_pi_c_mu400, Pres_pi_c_mu500 = [], [], [], [], []
    Pres_K_c_mu0, Pres_K_c_mu200, Pres_K_c_mu300, Pres_K_c_mu400, Pres_K_c_mu500 = [], [], [], [], []
    Pres_rho_c_mu0, Pres_rho_c_mu200, Pres_rho_c_mu300, Pres_rho_c_mu400, Pres_rho_c_mu500 = [], [], [], [], []
    Pres_omega_c_mu0, Pres_omega_c_mu200, Pres_omega_c_mu300, Pres_omega_c_mu400, Pres_omega_c_mu500 = [], [], [], [], []
    Pres_D_c_mu0, Pres_D_c_mu200, Pres_D_c_mu300, Pres_D_c_mu400, Pres_D_c_mu500 = [], [], [], [], []
    Pres_N_c_mu0, Pres_N_c_mu200, Pres_N_c_mu300, Pres_N_c_mu400, Pres_N_c_mu500 = [], [], [], [], []
    Pres_T_c_mu0, Pres_T_c_mu200, Pres_T_c_mu300, Pres_T_c_mu400, Pres_T_c_mu500 = [], [], [], [], []
    Pres_F_c_mu0, Pres_F_c_mu200, Pres_F_c_mu300, Pres_F_c_mu400, Pres_F_c_mu500 = [], [], [], [], []
    Pres_P_c_mu0, Pres_P_c_mu200, Pres_P_c_mu300, Pres_P_c_mu400, Pres_P_c_mu500 = [], [], [], [], []
    Pres_Q5_c_mu0, Pres_Q5_c_mu200, Pres_Q5_c_mu300, Pres_Q5_c_mu400, Pres_Q5_c_mu500 = [], [], [], [], []
    Pres_H_c_mu0, Pres_H_c_mu200, Pres_H_c_mu300, Pres_H_c_mu400, Pres_H_c_mu500 = [], [], [], [], []
    
    #T = numpy.linspace(1.0, 450.0, 200)
    T_mu0 = numpy.linspace(1.0, 2000.0, 200)
    T_mu200 = numpy.linspace(1.0, 2000.0, 200)
    T_mu300 = numpy.linspace(1.0, 2000.0, 200)
    T_mu400 = numpy.linspace(1.0, 2000.0, 200)
    T_mu500 = numpy.linspace(1.0, 2000.0, 200)
    mu0   = [0.0 for el in T_mu0]
    mu200 = [200.0 / 3.0 for el in T_mu200]
    mu300 = [300.0 / 3.0 for el in T_mu300]
    mu400 = [400.0 / 3.0 for el in T_mu400]
    mu500 = [500.0 / 3.0 for el in T_mu500]
    
    recalc_pl_mu0           = False
    recalc_pl_mu200         = False
    recalc_pl_mu300         = False
    recalc_pl_mu400         = False
    recalc_pl_mu500         = False

    recalc_pressure_mu0     = False
    recalc_pressure_mu200   = False
    recalc_pressure_mu300   = False
    recalc_pressure_mu400   = False
    recalc_pressure_mu500   = False

    recalc_pressure_mu0_c   = False
    recalc_pressure_mu200_c = False
    recalc_pressure_mu300_c = False
    recalc_pressure_mu400_c = False
    recalc_pressure_mu500_c = False

    cluster_backreaction    = False
    pl_turned_off           = False

    pl_mu0_file   = "D:/EoS/BDK/continuum_test/pl_mu0.dat"
    pl_mu200_file = "D:/EoS/BDK/continuum_test/pl_mu200.dat"
    pl_mu300_file = "D:/EoS/BDK/continuum_test/pl_mu300.dat"
    pl_mu400_file = "D:/EoS/BDK/continuum_test/pl_mu400.dat"
    pl_mu500_file = "D:/EoS/BDK/continuum_test/pl_mu500.dat"
    pressure_mu0_file   = "D:/EoS/BDK/continuum_test/pressure_mu0.dat"
    pressure_mu200_file = "D:/EoS/BDK/continuum_test/pressure_mu200.dat"
    pressure_mu300_file = "D:/EoS/BDK/continuum_test/pressure_mu300.dat"
    pressure_mu400_file = "D:/EoS/BDK/continuum_test/pressure_mu400.dat"
    pressure_mu500_file = "D:/EoS/BDK/continuum_test/pressure_mu500.dat"
    pressure_mu0_c_file   = "D:/EoS/BDK/continuum_test/pressure_c_mu0.dat"
    pressure_mu200_c_file = "D:/EoS/BDK/continuum_test/pressure_c_mu200.dat"
    pressure_mu300_c_file = "D:/EoS/BDK/continuum_test/pressure_c_mu300.dat"
    pressure_mu400_c_file = "D:/EoS/BDK/continuum_test/pressure_c_mu400.dat"
    pressure_mu500_c_file = "D:/EoS/BDK/continuum_test/pressure_c_mu500.dat"

    if recalc_pl_mu0:
        phi_re_mu0.append(1e-15)
        phi_im_mu0.append(2e-15)
        lT = len(T_mu0)
        for T_el, mu_el in tqdm.tqdm(zip(T_mu0, mu0), desc = "Traced Polyakov loop (mu = 0)", total = len(T_mu0), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_mu0[-1], phi_im_mu0[-1], with_clusters = cluster_backreaction)
            phi_re_mu0.append(temp_phi_re)
            phi_im_mu0.append(temp_phi_im)
        phi_re_mu0 = phi_re_mu0[1:]
        phi_im_mu0 = phi_im_mu0[1:]
        with open(pl_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T_mu0, phi_re_mu0, phi_im_mu0)])
    else:
        T_mu0, phi_re_mu0 = utils.data_collect(0, 1, pl_mu0_file)
        phi_im_mu0, _ = utils.data_collect(2, 2, pl_mu0_file)

    if recalc_pl_mu200:
        phi_re_mu200.append(1e-15)
        phi_im_mu200.append(2e-15)
        lT = len(T_mu200)
        for T_el, mu_el in tqdm.tqdm(zip(T_mu200, mu200), desc = "Traced Polyakov loop (mu = 200)", total = len(T_mu200), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_mu200[-1], phi_im_mu200[-1], with_clusters = cluster_backreaction)
            phi_re_mu200.append(temp_phi_re)
            phi_im_mu200.append(temp_phi_im)
        phi_re_mu200 = phi_re_mu200[1:]
        phi_im_mu200 = phi_im_mu200[1:]
        with open(pl_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T_mu200, phi_re_mu200, phi_im_mu200)])
    else:
        T_mu200, phi_re_mu200 = utils.data_collect(0, 1, pl_mu200_file)
        phi_im_mu200, _ = utils.data_collect(2, 2, pl_mu200_file)
    
    if recalc_pl_mu300:
        phi_re_mu300.append(1e-15)
        phi_im_mu300.append(2e-15)
        lT = len(T_mu300)
        for T_el, mu_el in tqdm.tqdm(zip(T_mu300, mu300), desc = "Traced Polyakov loop (mu = 300)", total = len(T_mu300), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_mu300[-1], phi_im_mu300[-1], with_clusters = cluster_backreaction)
            phi_re_mu300.append(temp_phi_re)
            phi_im_mu300.append(temp_phi_im)
        phi_re_mu300 = phi_re_mu300[1:]
        phi_im_mu300 = phi_im_mu300[1:]
        with open(pl_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T_mu300, phi_re_mu300, phi_im_mu300)])
    else:
        T_mu300, phi_re_mu300 = utils.data_collect(0, 1, pl_mu300_file)
        phi_im_mu300, _ = utils.data_collect(2, 2, pl_mu300_file)

    if recalc_pl_mu400:
        phi_re_mu400.append(1e-15)
        phi_im_mu400.append(2e-15)
        lT = len(T_mu400)
        for T_el, mu_el in tqdm.tqdm(zip(T_mu400, mu400), desc = "Traced Polyakov loop (mu = 400)", total = len(T_mu400), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_mu400[-1], phi_im_mu400[-1], with_clusters = cluster_backreaction)
            phi_re_mu400.append(temp_phi_re)
            phi_im_mu400.append(temp_phi_im)
        phi_re_mu400 = phi_re_mu400[1:]
        phi_im_mu400 = phi_im_mu400[1:]
        with open(pl_mu400_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T_mu400, phi_re_mu400, phi_im_mu400)])
    else:
        T_mu400, phi_re_mu400 = utils.data_collect(0, 1, pl_mu400_file)
        phi_im_mu400, _ = utils.data_collect(2, 2, pl_mu400_file)

    if recalc_pl_mu500:
        phi_re_mu500.append(1e-15)
        phi_im_mu500.append(2e-15)
        lT = len(T_mu500)
        for T_el, mu_el in tqdm.tqdm(zip(T_mu500, mu500), desc = "Traced Polyakov loop (mu = 500)", total = len(T_mu500), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_mu500[-1], phi_im_mu500[-1], with_clusters = cluster_backreaction)
            phi_re_mu500.append(temp_phi_re)
            phi_im_mu500.append(temp_phi_im)
        phi_re_mu500 = phi_re_mu500[1:]
        phi_im_mu500 = phi_im_mu500[1:]
        with open(pl_mu500_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T_mu500, phi_re_mu500, phi_im_mu500)])
    else:
        T_mu500, phi_re_mu500 = utils.data_collect(0, 1, pl_mu500_file)
        phi_im_mu500, _ = utils.data_collect(2, 2, pl_mu500_file)

    if recalc_pressure_mu0:
        Pres_g_mu0 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_mu0, phi_re_mu0, phi_im_mu0), 
                desc = "Gluon pressure (mu = 0)", 
                total = len(T_mu0),
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu0 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu0, mu0, phi_re_mu0, phi_im_mu0), 
                    desc = "Quark pressure (mu = 0)", 
                    total = len(T_mu0), 
                    ascii = True
                    )]
            Pres_pert_mu0 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu0, mu0, phi_re_mu0, phi_im_mu0), 
                    desc = "Perturbative pressure (mu = 0)", 
                    total = len(T_mu0), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu0, Pres_rho_step_mu0, Pres_omega_step_mu0, Pres_K_step_mu0, Pres_D_step_mu0, Pres_N_step_mu0, Pres_T_step_mu0, Pres_F_step_mu0, Pres_P_step_mu0, Pres_Q5_step_mu0, Pres_H_step_mu0),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu0, mu0, phi_re_mu0, phi_im_mu0)
        else:
            Pres_Q_mu0 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu0, mu0, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 0)", 
                    total = len(T_mu0), 
                    ascii = True
                    )]
            Pres_pert_mu0 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu0, mu0, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 0)", 
                    total = len(T_mu0), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu0, Pres_rho_step_mu0, Pres_omega_step_mu0, Pres_K_step_mu0, Pres_D_step_mu0, Pres_N_step_mu0, Pres_T_step_mu0, Pres_F_step_mu0, Pres_P_step_mu0, Pres_Q5_step_mu0, Pres_H_step_mu0),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu0, mu0, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu0, Pres_g_mu0, Pres_pert_mu0, Pres_pi_step_mu0, Pres_K_step_mu0, Pres_rho_step_mu0, Pres_omega_step_mu0, Pres_D_step_mu0, Pres_N_step_mu0, Pres_T_step_mu0, Pres_F_step_mu0, Pres_P_step_mu0, Pres_Q5_step_mu0, Pres_H_step_mu0)])
    else:
        Pres_Q_mu0, Pres_g_mu0       = utils.data_collect(0, 1, pressure_mu0_file)
        Pres_pert_mu0, Pres_pi_step_mu0   = utils.data_collect(2, 3, pressure_mu0_file)
        Pres_K_step_mu0, _ = utils.data_collect(4, 4, pressure_mu0_file)
        Pres_rho_step_mu0, Pres_omega_step_mu0 = utils.data_collect(5, 6, pressure_mu0_file)
        Pres_D_step_mu0, Pres_N_step_mu0       = utils.data_collect(7, 8, pressure_mu0_file)
        Pres_T_step_mu0, Pres_F_step_mu0       = utils.data_collect(9, 10, pressure_mu0_file)
        Pres_P_step_mu0, Pres_Q5_step_mu0      = utils.data_collect(11, 12, pressure_mu0_file)
        Pres_H_step_mu0, _                = utils.data_collect(13, 13, pressure_mu0_file)

    if recalc_pressure_mu0_c:
        if not pl_turned_off:
            (
                (Pres_pi_c_mu0, Pres_rho_c_mu0, Pres_omega_c_mu0, Pres_K_c_mu0, Pres_D_c_mu0, Pres_N_c_mu0, Pres_T_c_mu0, Pres_F_c_mu0, Pres_P_c_mu0, Pres_Q5_c_mu0, Pres_H_c_mu0),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu0, mu0, phi_re_mu0, phi_im_mu0)
        else:
            (
                (Pres_pi_c_mu0, Pres_rho_c_mu0, Pres_omega_c_mu0, Pres_K_c_mu0, Pres_D_c_mu0, Pres_N_c_mu0, Pres_T_c_mu0, Pres_F_c_mu0, Pres_P_c_mu0, Pres_Q5_c_mu0, Pres_H_c_mu0),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu0, mu0, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu0_c_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu0, Pres_g_mu0, Pres_pert_mu0, Pres_pi_c_mu0, Pres_K_c_mu0, Pres_rho_c_mu0, Pres_omega_c_mu0, Pres_D_c_mu0, Pres_N_c_mu0, Pres_T_c_mu0, Pres_F_c_mu0, Pres_P_c_mu0, Pres_Q5_c_mu0, Pres_H_c_mu0)])
    else:
        Pres_Q_mu0, Pres_g_mu0            = utils.data_collect(0, 1, pressure_mu0_c_file)
        Pres_pert_mu0, Pres_pi_c_mu0      = utils.data_collect(2, 3, pressure_mu0_c_file)
        Pres_K_c_mu0, _ = utils.data_collect(4, 4, pressure_mu0_c_file)
        Pres_rho_c_mu0, Pres_omega_c_mu0  = utils.data_collect(5, 6, pressure_mu0_c_file)
        Pres_D_c_mu0, Pres_N_c_mu0        = utils.data_collect(7, 8, pressure_mu0_c_file)
        Pres_T_c_mu0, Pres_F_c_mu0        = utils.data_collect(9, 10, pressure_mu0_c_file)
        Pres_P_c_mu0, Pres_Q5_c_mu0       = utils.data_collect(11, 12, pressure_mu0_c_file)
        Pres_H_c_mu0, _                   = utils.data_collect(13, 13, pressure_mu0_c_file)

    if recalc_pressure_mu200:
        Pres_g_mu200 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_mu200, phi_re_mu200, phi_im_mu200), 
                desc = "Gluon pressure (mu = 200)", 
                total = len(T_mu200), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu200 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu200, mu200, phi_re_mu200, phi_im_mu200), 
                    desc = "Quark pressure (mu = 200)", 
                    total = len(T_mu200), 
                    ascii = True
                    )]        
            Pres_pert_mu200 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu200, mu200, phi_re_mu200, phi_im_mu200), 
                    desc = "Perturbative pressure (mu = 200)", 
                    total = len(T_mu200), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu200, Pres_rho_step_mu200, Pres_omega_step_mu200, Pres_K_step_mu200, Pres_D_step_mu200, Pres_N_step_mu200, Pres_T_step_mu200, Pres_F_step_mu200, Pres_P_step_mu200, Pres_Q5_step_mu200, Pres_H_step_mu200),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu200, mu200, phi_re_mu200, phi_im_mu200)
        else:
            Pres_Q_mu200 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu200, mu200, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 200)", 
                    total = len(T_mu200), 
                    ascii = True
                    )]
            Pres_pert_mu200 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu200, mu200, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 200)", 
                    total = len(T_mu200), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu200, Pres_rho_step_mu200, Pres_omega_step_mu200, Pres_K_step_mu200, Pres_D_step_mu200, Pres_N_step_mu200, Pres_T_step_mu200, Pres_F_step_mu200, Pres_P_step_mu200, Pres_Q5_step_mu200, Pres_H_step_mu200),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu200, mu200, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu200, Pres_g_mu200, Pres_pert_mu200, Pres_pi_step_mu200, Pres_K_step_mu200, Pres_rho_step_mu200, Pres_omega_step_mu200, Pres_D_step_mu200, Pres_N_step_mu200, Pres_T_step_mu200, Pres_F_step_mu200, Pres_P_step_mu200, Pres_Q5_step_mu200, Pres_H_step_mu200)])
    else:
        Pres_Q_mu200, Pres_g_mu200       = utils.data_collect(0, 1, pressure_mu200_file)
        Pres_pert_mu200, Pres_pi_step_mu200   = utils.data_collect(2, 3, pressure_mu200_file)
        Pres_K_step_mu200, _ = utils.data_collect(4, 4, pressure_mu200_file)
        Pres_rho_step_mu200, Pres_omega_step_mu200 = utils.data_collect(5, 6, pressure_mu200_file)
        Pres_D_step_mu200, Pres_N_step_mu200       = utils.data_collect(7, 8, pressure_mu200_file)
        Pres_T_step_mu200, Pres_F_step_mu200       = utils.data_collect(9, 10, pressure_mu200_file)
        Pres_P_step_mu200, Pres_Q5_step_mu200      = utils.data_collect(11, 12, pressure_mu200_file)
        Pres_H_step_mu200, _                  = utils.data_collect(13, 13, pressure_mu200_file)
    
    if recalc_pressure_mu200_c:
        if not pl_turned_off:
            (
                (Pres_pi_c_mu200, Pres_rho_c_mu200, Pres_omega_c_mu200, Pres_K_c_mu200, Pres_D_c_mu200, Pres_N_c_mu200, Pres_T_c_mu200, Pres_F_c_mu200, Pres_P_c_mu200, Pres_Q5_c_mu200, Pres_H_c_mu200),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu200, mu200, phi_re_mu200, phi_im_mu200)
        else:
            (
                (Pres_pi_c_mu200, Pres_rho_c_mu200, Pres_omega_c_mu200, Pres_K_c_mu200, Pres_D_c_mu200, Pres_N_c_mu200, Pres_T_c_mu200, Pres_F_c_mu200, Pres_P_c_mu200, Pres_Q5_c_mu200, Pres_H_c_mu200),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu200, mu200, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu200_c_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu200, Pres_g_mu200, Pres_pert_mu200, Pres_pi_c_mu200, Pres_K_c_mu200, Pres_rho_c_mu200, Pres_omega_c_mu200, Pres_D_c_mu200, Pres_N_c_mu200, Pres_T_c_mu200, Pres_F_c_mu200, Pres_P_c_mu200, Pres_Q5_c_mu200, Pres_H_c_mu200)])
    else:
        Pres_Q_mu200, Pres_g_mu200       = utils.data_collect(0, 1, pressure_mu200_c_file)
        Pres_pert_mu200, Pres_pi_c_mu200   = utils.data_collect(2, 3, pressure_mu200_c_file)
        Pres_K_c_mu200, _ = utils.data_collect(4, 4, pressure_mu200_c_file)
        Pres_rho_c_mu200, Pres_omega_c_mu200 = utils.data_collect(5, 6, pressure_mu200_c_file)
        Pres_D_c_mu200, Pres_N_c_mu200       = utils.data_collect(7, 8, pressure_mu200_c_file)
        Pres_T_c_mu200, Pres_F_c_mu200       = utils.data_collect(9, 10, pressure_mu200_c_file)
        Pres_P_c_mu200, Pres_Q5_c_mu200      = utils.data_collect(11, 12, pressure_mu200_c_file)
        Pres_H_c_mu200, _                  = utils.data_collect(13, 13, pressure_mu200_c_file)

    if recalc_pressure_mu300:
        Pres_g_mu300 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_mu300, phi_re_mu300, phi_im_mu300), 
                desc = "Gluon pressure (mu = 300)", 
                total = len(T_mu300), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu300 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu300, mu300, phi_re_mu300, phi_im_mu300), 
                    desc = "Quark pressure (mu = 300)", 
                    total = len(T_mu300), 
                    ascii = True
                    )]
            Pres_pert_mu300 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu300, mu300, phi_re_mu300, phi_im_mu300), 
                    desc = "Perturbative pressure (mu = 300)", 
                    total = len(T_mu300), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu300, Pres_rho_step_mu300, Pres_omega_step_mu300, Pres_K_step_mu300, Pres_D_step_mu300, Pres_N_step_mu300, Pres_T_step_mu300, Pres_F_step_mu300, Pres_P_step_mu300, Pres_Q5_step_mu300, Pres_H_step_mu300),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu300, mu300, phi_re_mu300, phi_im_mu300)
        else:
            Pres_Q_mu300 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el in 
                tqdm.tqdm(
                    zip(T_mu300, mu300, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 300)", 
                    total = len(T_mu300), 
                    ascii = True
                    )]
            Pres_pert_mu300 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu300, mu300, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 300)", 
                    total = len(T_mu300), 
                    ascii = True
                    )]
            (
                (Pres_pi_step_mu300, Pres_rho_step_mu300, Pres_omega_step_mu300, Pres_K_step_mu300, Pres_D_step_mu300, Pres_N_step_mu300, Pres_T_step_mu300, Pres_F_step_mu300, Pres_P_step_mu300, Pres_Q5_step_mu300, Pres_H_step_mu300),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu300, mu300, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu300, Pres_g_mu300, Pres_pert_mu300, Pres_pi_step_mu300, Pres_K_step_mu300, Pres_rho_step_mu300, Pres_omega_step_mu300, Pres_D_step_mu300, Pres_N_step_mu300, Pres_T_step_mu300, Pres_F_step_mu300, Pres_P_step_mu300, Pres_Q5_step_mu300, Pres_H_step_mu300)])
    else:
        Pres_Q_mu300, Pres_g_mu300       = utils.data_collect(0, 1, pressure_mu300_file)
        Pres_pert_mu300, Pres_pi_step_mu300   = utils.data_collect(2, 3, pressure_mu300_file)
        Pres_K_step_mu300, _ = utils.data_collect(4, 4, pressure_mu300_file)
        Pres_rho_step_mu300, Pres_omega_step_mu300 = utils.data_collect(5, 6, pressure_mu300_file)
        Pres_D_step_mu300, Pres_N_step_mu300       = utils.data_collect(7, 8, pressure_mu300_file)
        Pres_T_step_mu300, Pres_F_step_mu300       = utils.data_collect(9, 10, pressure_mu300_file)
        Pres_P_step_mu300, Pres_Q5_step_mu300      = utils.data_collect(11, 12, pressure_mu300_file)
        Pres_H_step_mu300, _                  = utils.data_collect(13, 13, pressure_mu300_file)

    if recalc_pressure_mu300_c:
        if not pl_turned_off:
            (
                (Pres_pi_c_mu300, Pres_rho_c_mu300, Pres_omega_c_mu300, Pres_K_c_mu300, Pres_D_c_mu300, Pres_N_c_mu300, Pres_T_c_mu300, Pres_F_c_mu300, Pres_P_c_mu300, Pres_Q5_c_mu300, Pres_H_c_mu300),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu300, mu300, phi_re_mu300, phi_im_mu300)
        else:
            (
                (Pres_pi_c_mu300, Pres_rho_c_mu300, Pres_omega_c_mu300, Pres_K_c_mu300, Pres_D_c_mu300, Pres_N_c_mu300, Pres_T_c_mu300, Pres_F_c_mu300, Pres_P_c_mu300, Pres_Q5_c_mu300, Pres_H_c_mu300),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu300, mu300, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu300_c_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu300, Pres_g_mu300, Pres_pert_mu300, Pres_pi_c_mu300, Pres_K_c_mu300, Pres_rho_c_mu300, Pres_omega_c_mu300, Pres_D_c_mu300, Pres_N_c_mu300, Pres_T_c_mu300, Pres_F_c_mu300, Pres_P_c_mu300, Pres_Q5_c_mu300, Pres_H_c_mu300)])
    else:
        Pres_Q_mu300, Pres_g_mu300       = utils.data_collect(0, 1, pressure_mu300_c_file)
        Pres_pert_mu300, Pres_pi_c_mu300   = utils.data_collect(2, 3, pressure_mu300_c_file)
        Pres_K_c_mu300, _ = utils.data_collect(4, 4, pressure_mu300_c_file)
        Pres_rho_c_mu300, Pres_omega_c_mu300 = utils.data_collect(5, 6, pressure_mu300_c_file)
        Pres_D_c_mu300, Pres_N_c_mu300       = utils.data_collect(7, 8, pressure_mu300_c_file)
        Pres_T_c_mu300, Pres_F_c_mu300       = utils.data_collect(9, 10, pressure_mu300_c_file)
        Pres_P_c_mu300, Pres_Q5_c_mu300      = utils.data_collect(11, 12, pressure_mu300_c_file)
        Pres_H_c_mu300, _                  = utils.data_collect(13, 13, pressure_mu300_c_file)

    if recalc_pressure_mu400:
        Pres_g_mu400 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_mu400, phi_re_mu400, phi_im_mu400), 
                desc = "Gluon pressure (mu = 400)", 
                total = len(T_mu400), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu400 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu400, mu400, phi_re_mu400, phi_im_mu400), 
                    desc = "Quark pressure (mu = 400)", 
                    total = len(T_mu400), 
                    ascii = True
                    )]
            Pres_pert_mu400 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu400, mu400, phi_re_mu400, phi_im_mu400), 
                    desc = "Perturbative pressure (mu = 400)", 
                    total = len(T_mu400), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu400, Pres_rho_step_mu400, Pres_omega_step_mu400, Pres_K_step_mu400, Pres_D_step_mu400, Pres_N_step_mu400, Pres_T_step_mu400, Pres_F_step_mu400, Pres_P_step_mu400, Pres_Q5_step_mu400, Pres_H_step_mu400),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu400, mu400, phi_re_mu400, phi_im_mu400)
        else:
            Pres_Q_mu400 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el in 
                tqdm.tqdm(
                    zip(T_mu400, mu400, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 400)", 
                    total = len(T_mu400), 
                    ascii = True
                    )]
            Pres_pert_mu400 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu400, mu400, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 400)", 
                    total = len(T_mu400), 
                    ascii = True
                    )]
            (
                (Pres_pi_step_mu400, Pres_rho_step_mu400, Pres_omega_step_mu400, Pres_K_step_mu400, Pres_D_step_mu400, Pres_N_step_mu400, Pres_T_step_mu400, Pres_F_step_mu400, Pres_P_step_mu400, Pres_Q5_step_mu400, Pres_H_step_mu400),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu400, mu400, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu400_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu400, Pres_g_mu400, Pres_pert_mu400, Pres_pi_step_mu400, Pres_K_step_mu400, Pres_rho_step_mu400, Pres_omega_step_mu400, Pres_D_step_mu400, Pres_N_step_mu400, Pres_T_step_mu400, Pres_F_step_mu400, Pres_P_step_mu400, Pres_Q5_step_mu400, Pres_H_step_mu400)])
    else:
        Pres_Q_mu400, Pres_g_mu400                 = utils.data_collect(0, 1, pressure_mu400_file)
        Pres_pert_mu400, Pres_pi_step_mu400        = utils.data_collect(2, 3, pressure_mu400_file)
        Pres_pi_step_mu400, _ = utils.data_collect(4, 4, pressure_mu400_file)
        Pres_rho_step_mu400, Pres_omega_step_mu400 = utils.data_collect(5, 6, pressure_mu400_file)
        Pres_D_step_mu400, Pres_N_step_mu400       = utils.data_collect(7, 8, pressure_mu400_file)
        Pres_T_step_mu400, Pres_F_step_mu400       = utils.data_collect(9, 10, pressure_mu400_file)
        Pres_P_step_mu400, Pres_Q5_step_mu400      = utils.data_collect(11, 12, pressure_mu400_file)
        Pres_H_step_mu400, _                       = utils.data_collect(13, 13, pressure_mu400_file)

    if recalc_pressure_mu400_c:
        if not pl_turned_off:
            (
                (Pres_pi_c_mu400, Pres_rho_c_mu400, Pres_omega_c_mu400, Pres_K_c_mu400, Pres_D_c_mu400, Pres_N_c_mu400, Pres_T_c_mu400, Pres_F_c_mu400, Pres_P_c_mu400, Pres_Q5_c_mu400, Pres_H_c_mu400),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu400, mu400, phi_re_mu400, phi_im_mu400)
        else:
            (
                (Pres_pi_c_mu400, Pres_rho_c_mu400, Pres_omega_c_mu400, Pres_K_c_mu400, Pres_D_c_mu400, Pres_N_c_mu400, Pres_T_c_mu400, Pres_F_c_mu400, Pres_P_c_mu400, Pres_Q5_c_mu400, Pres_H_c_mu400),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu400, mu400, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu400_c_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu400, Pres_g_mu400, Pres_pert_mu400, Pres_pi_c_mu400, Pres_K_c_mu400, Pres_rho_c_mu400, Pres_omega_c_mu400, Pres_D_c_mu400, Pres_N_c_mu400, Pres_T_c_mu400, Pres_F_c_mu400, Pres_P_c_mu400, Pres_Q5_c_mu400, Pres_H_c_mu400)])
    else:
        Pres_Q_mu400, Pres_g_mu400       = utils.data_collect(0, 1, pressure_mu400_c_file)
        Pres_pert_mu400, Pres_pi_c_mu400   = utils.data_collect(2, 3, pressure_mu400_c_file)
        Pres_pi_c_mu400, _ = utils.data_collect(4, 4, pressure_mu400_c_file)
        Pres_rho_c_mu400, Pres_omega_c_mu400 = utils.data_collect(5, 6, pressure_mu400_c_file)
        Pres_D_c_mu400, Pres_N_c_mu400       = utils.data_collect(7, 8, pressure_mu400_c_file)
        Pres_T_c_mu400, Pres_F_c_mu400       = utils.data_collect(9, 10, pressure_mu400_c_file)
        Pres_P_c_mu400, Pres_Q5_c_mu400      = utils.data_collect(11, 12, pressure_mu400_c_file)
        Pres_H_c_mu400, _                  = utils.data_collect(13, 13, pressure_mu400_c_file)

    if recalc_pressure_mu500:
        Pres_g_mu500 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_mu500, phi_re_mu500, phi_im_mu500), 
                desc = "Gluon pressure (mu = 500)", 
                total = len(T_mu500), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu500 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu500, mu500, phi_re_mu500, phi_im_mu500), 
                    desc = "Quark pressure (mu = 500)", 
                    total = len(T_mu500), 
                    ascii = True
                    )]
            Pres_pert_mu500 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu500, mu500, phi_re_mu500, phi_im_mu500), 
                    desc = "Perturbative pressure (mu = 500)", 
                    total = len(T_mu500), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu500, Pres_rho_step_mu500, Pres_omega_step_mu500, Pres_K_step_mu500, Pres_D_step_mu500, Pres_N_step_mu500, Pres_T_step_mu500, Pres_F_step_mu500, Pres_P_step_mu500, Pres_Q5_step_mu500, Pres_H_step_mu500),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu500, mu500, phi_re_mu500, phi_im_mu500)
        else:
            Pres_Q_mu500 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el in 
                tqdm.tqdm(
                    zip(T_mu500, mu500, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 500)", 
                    total = len(T_mu500), 
                    ascii = True
                    )]
            Pres_pert_mu500 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu500, mu500, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 500)", 
                    total = len(T_mu500), 
                    ascii = True
                    )]
            (
                (Pres_pi_step_mu500, Pres_rho_step_mu500, Pres_omega_step_mu500, Pres_K_step_mu500, Pres_D_step_mu500, Pres_N_step_mu500, Pres_T_step_mu500, Pres_F_step_mu500, Pres_P_step_mu500, Pres_Q5_step_mu500, Pres_H_step_mu500),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu500, mu500, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu500_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu500, Pres_g_mu500, Pres_pert_mu500, Pres_pi_step_mu500, Pres_K_step_mu500, Pres_rho_step_mu500, Pres_omega_step_mu500, Pres_D_step_mu500, Pres_N_step_mu500, Pres_T_step_mu500, Pres_F_step_mu500, Pres_P_step_mu500, Pres_Q5_step_mu500, Pres_H_step_mu500)])
    else:
        Pres_Q_mu500, Pres_g_mu500                 = utils.data_collect(0, 1, pressure_mu500_file)
        Pres_pert_mu500, Pres_pi_step_mu500        = utils.data_collect(2, 3, pressure_mu500_file)
        Pres_K_step_mu500, _ = utils.data_collect(4, 4, pressure_mu500_file)
        Pres_rho_step_mu500, Pres_omega_step_mu500 = utils.data_collect(5, 6, pressure_mu500_file)
        Pres_D_step_mu500, Pres_N_step_mu500       = utils.data_collect(7, 8, pressure_mu500_file)
        Pres_T_step_mu500, Pres_F_step_mu500       = utils.data_collect(9, 10, pressure_mu500_file)
        Pres_P_step_mu500, Pres_Q5_step_mu500      = utils.data_collect(11, 12, pressure_mu500_file)
        Pres_H_step_mu500, _                       = utils.data_collect(13, 13, pressure_mu500_file)

    if recalc_pressure_mu500_c:
        if not pl_turned_off:
            (
                (Pres_pi_c_mu500, Pres_rho_c_mu500, Pres_omega_c_mu500, Pres_K_c_mu500, Pres_D_c_mu500, Pres_N_c_mu500, Pres_T_c_mu500, Pres_F_c_mu500, Pres_P_c_mu500, Pres_Q5_c_mu500, Pres_H_c_mu500),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu500, mu500, phi_re_mu500, phi_im_mu500)
        else:
            (
                (Pres_pi_c_mu500, Pres_rho_c_mu500, Pres_omega_c_mu500, Pres_K_c_mu500, Pres_D_c_mu500, Pres_N_c_mu500, Pres_T_c_mu500, Pres_F_c_mu500, Pres_P_c_mu500, Pres_Q5_c_mu500, Pres_H_c_mu500),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu500, mu500, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu500_c_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu500, Pres_g_mu500, Pres_pert_mu500, Pres_pi_c_mu500, Pres_K_c_mu500, Pres_rho_c_mu500, Pres_omega_c_mu500, Pres_D_c_mu500, Pres_N_c_mu500, Pres_T_c_mu500, Pres_F_c_mu500, Pres_P_c_mu500, Pres_Q5_c_mu500, Pres_H_c_mu500)])
    else:
        Pres_Q_mu500, Pres_g_mu500       = utils.data_collect(0, 1, pressure_mu500_c_file)
        Pres_pert_mu500, Pres_pi_c_mu500   = utils.data_collect(2, 3, pressure_mu500_c_file)
        Pres_pi_c_mu500, _ = utils.data_collect(4, 4, pressure_mu500_c_file)
        Pres_rho_c_mu500, Pres_omega_c_mu500 = utils.data_collect(5, 6, pressure_mu500_c_file)
        Pres_D_c_mu500, Pres_N_c_mu500       = utils.data_collect(7, 8, pressure_mu500_c_file)
        Pres_T_c_mu500, Pres_F_c_mu500       = utils.data_collect(9, 10, pressure_mu500_c_file)
        Pres_P_c_mu500, Pres_Q5_c_mu500      = utils.data_collect(11, 12, pressure_mu500_c_file)
        Pres_H_c_mu500, _                  = utils.data_collect(13, 13, pressure_mu500_c_file)

    contrib_q_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_Q_mu0   )]
    contrib_g_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_g_mu0   )]
    contrib_pert_mu0                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_pert_mu0)]
    contrib_qgp_mu0                 = [sum(el) for el in zip(contrib_q_mu0, contrib_g_mu0, contrib_pert_mu0)]
    contrib_q_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_Q_mu200   )]
    contrib_g_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_g_mu200   )]
    contrib_pert_mu200              = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_pert_mu200)]
    contrib_qgp_mu200               = [sum(el) for el in zip(contrib_q_mu200, contrib_g_mu200, contrib_pert_mu200)]
    contrib_q_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_Q_mu300   )]
    contrib_g_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_g_mu300   )]
    contrib_pert_mu300              = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_pert_mu300)]
    contrib_qgp_mu300               = [sum(el) for el in zip(contrib_q_mu300, contrib_g_mu300, contrib_pert_mu300)]
    contrib_q_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_Q_mu400   )]
    contrib_g_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_g_mu400   )]
    contrib_pert_mu400              = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_pert_mu400)]
    contrib_qgp_mu400               = [sum(el) for el in zip(contrib_q_mu400, contrib_g_mu400, contrib_pert_mu400)]
    contrib_q_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_Q_mu500   )]
    contrib_g_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_g_mu500   )]
    contrib_pert_mu500              = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_pert_mu500)]
    contrib_qgp_mu500               = [sum(el) for el in zip(contrib_q_mu500, contrib_g_mu500, contrib_pert_mu500)]

    contrib_pi_step_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_pi_step_mu0)]
    contrib_rho_step_mu0                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_rho_step_mu0)]
    contrib_omega_step_mu0               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_omega_step_mu0)]
    contrib_K_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_K_step_mu0)]
    contrib_D_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_D_step_mu0)]
    contrib_N_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_N_step_mu0)]
    contrib_T_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_T_step_mu0)]
    contrib_F_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_F_step_mu0)]
    contrib_P_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_P_step_mu0)]
    contrib_Q5_step_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_Q5_step_mu0)]
    contrib_H_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_H_step_mu0)]
    contrib_cluster_step_mu0             = [sum(el) for el in zip(contrib_pi_step_mu0, contrib_rho_step_mu0, contrib_omega_step_mu0, contrib_K_step_mu0, contrib_D_step_mu0, contrib_N_step_mu0, contrib_T_step_mu0, contrib_F_step_mu0, contrib_P_step_mu0, contrib_Q5_step_mu0, contrib_H_step_mu0)]
    contrib_cluster_singlet_step_mu0     = [sum(el) for el in zip(contrib_pi_step_mu0, contrib_rho_step_mu0, contrib_omega_step_mu0, contrib_K_step_mu0, contrib_N_step_mu0, contrib_T_step_mu0, contrib_P_step_mu0, contrib_H_step_mu0)]
    contrib_cluster_color_step_mu0       = [sum(el) for el in zip(contrib_D_step_mu0, contrib_F_step_mu0, contrib_Q5_step_mu0)]
    contrib_pi_step_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_pi_step_mu200)]
    contrib_rho_step_mu200               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_rho_step_mu200)]
    contrib_omega_step_mu200             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_omega_step_mu200)]
    contrib_K_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_K_step_mu200)]
    contrib_D_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_D_step_mu200)]
    contrib_N_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_N_step_mu200)]
    contrib_T_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_T_step_mu200)]
    contrib_F_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_F_step_mu200)]
    contrib_P_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_P_step_mu200)]
    contrib_Q5_step_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_Q5_step_mu200)]
    contrib_H_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_H_step_mu200)]
    contrib_cluster_step_mu200           = [sum(el) for el in zip(contrib_pi_step_mu200, contrib_rho_step_mu200, contrib_omega_step_mu200, contrib_K_step_mu200, contrib_D_step_mu200, contrib_N_step_mu200, contrib_T_step_mu200, contrib_F_step_mu200, contrib_P_step_mu200, contrib_Q5_step_mu200, contrib_H_step_mu200)]
    contrib_cluster_singlet_step_mu200   = [sum(el) for el in zip(contrib_pi_step_mu200, contrib_rho_step_mu200, contrib_omega_step_mu200, contrib_K_step_mu200, contrib_N_step_mu200, contrib_T_step_mu200, contrib_P_step_mu200, contrib_H_step_mu200)]
    contrib_cluster_color_step_mu200     = [sum(el) for el in zip(contrib_D_step_mu200, contrib_F_step_mu200, contrib_Q5_step_mu200)]
    contrib_pi_step_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_pi_step_mu300)]
    contrib_rho_step_mu300               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_rho_step_mu300)]
    contrib_omega_step_mu300             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_omega_step_mu300)]
    contrib_K_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_K_step_mu300)]
    contrib_D_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_D_step_mu300)]
    contrib_N_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_N_step_mu300)]
    contrib_T_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_T_step_mu300)]
    contrib_F_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_F_step_mu300)]
    contrib_P_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_P_step_mu300)]
    contrib_Q5_step_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_Q5_step_mu300)]
    contrib_H_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_H_step_mu300)]
    contrib_cluster_step_mu300           = [sum(el) for el in zip(contrib_pi_step_mu300, contrib_rho_step_mu300, contrib_omega_step_mu300, contrib_K_step_mu300, contrib_D_step_mu300, contrib_N_step_mu300, contrib_T_step_mu300, contrib_F_step_mu300, contrib_P_step_mu300, contrib_Q5_step_mu300, contrib_H_step_mu300)]
    contrib_cluster_singlet_step_mu300   = [sum(el) for el in zip(contrib_pi_step_mu300, contrib_rho_step_mu300, contrib_omega_step_mu300, contrib_K_step_mu300, contrib_N_step_mu300, contrib_T_step_mu300, contrib_P_step_mu300, contrib_H_step_mu300)]
    contrib_cluster_color_step_mu300     = [sum(el) for el in zip(contrib_D_step_mu300, contrib_F_step_mu300, contrib_Q5_step_mu300)]
    contrib_pi_step_mu400                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_pi_step_mu400)]
    contrib_rho_step_mu400               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_rho_step_mu400)]
    contrib_omega_step_mu400             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_omega_step_mu400)]
    contrib_K_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_K_step_mu400)]
    contrib_D_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_D_step_mu400)]
    contrib_N_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_N_step_mu400)]
    contrib_T_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_T_step_mu400)]
    contrib_F_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_F_step_mu400)]
    contrib_P_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_P_step_mu400)]
    contrib_Q5_step_mu400                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_Q5_step_mu400)]
    contrib_H_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_H_step_mu400)]
    contrib_cluster_step_mu400           = [sum(el) for el in zip(contrib_pi_step_mu400, contrib_rho_step_mu400, contrib_omega_step_mu400, contrib_K_step_mu400, contrib_D_step_mu400, contrib_N_step_mu400, contrib_T_step_mu400, contrib_F_step_mu400, contrib_P_step_mu400, contrib_Q5_step_mu400, contrib_H_step_mu400)]
    contrib_cluster_singlet_step_mu400   = [sum(el) for el in zip(contrib_pi_step_mu400, contrib_rho_step_mu400, contrib_omega_step_mu400, contrib_K_step_mu400, contrib_N_step_mu400, contrib_T_step_mu400, contrib_P_step_mu400, contrib_H_step_mu400)]
    contrib_cluster_color_step_mu400     = [sum(el) for el in zip(contrib_D_step_mu400, contrib_F_step_mu400, contrib_Q5_step_mu400)]
    contrib_pi_step_mu500                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_pi_step_mu500)]
    contrib_rho_step_mu500               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_rho_step_mu500)]
    contrib_omega_step_mu500             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_omega_step_mu500)]
    contrib_K_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_K_step_mu500)]
    contrib_D_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_D_step_mu500)]
    contrib_N_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_N_step_mu500)]
    contrib_T_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_T_step_mu500)]
    contrib_F_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_F_step_mu500)]
    contrib_P_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_P_step_mu500)]
    contrib_Q5_step_mu500                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_Q5_step_mu500)]
    contrib_H_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_H_step_mu500)]
    contrib_cluster_step_mu500           = [sum(el) for el in zip(contrib_pi_step_mu500, contrib_rho_step_mu500, contrib_omega_step_mu500, contrib_K_step_mu500, contrib_D_step_mu500, contrib_N_step_mu500, contrib_T_step_mu500, contrib_F_step_mu500, contrib_P_step_mu500, contrib_Q5_step_mu500, contrib_H_step_mu500)]
    contrib_cluster_singlet_step_mu500   = [sum(el) for el in zip(contrib_pi_step_mu500, contrib_rho_step_mu500, contrib_omega_step_mu500, contrib_K_step_mu500, contrib_N_step_mu500, contrib_T_step_mu500, contrib_P_step_mu500, contrib_H_step_mu500)]
    contrib_cluster_color_step_mu500     = [sum(el) for el in zip(contrib_D_step_mu500, contrib_F_step_mu500, contrib_Q5_step_mu500)]

    contrib_pi_c_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_pi_c_mu0)]
    contrib_rho_c_mu0                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_rho_c_mu0)]
    contrib_omega_c_mu0               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_omega_c_mu0)]
    contrib_K_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_K_c_mu0)]
    contrib_D_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_D_c_mu0)]
    contrib_N_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_N_c_mu0)]
    contrib_T_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_T_c_mu0)]
    contrib_F_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_F_c_mu0)]
    contrib_P_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_P_c_mu0)]
    contrib_Q5_c_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_Q5_c_mu0)]
    contrib_H_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_H_c_mu0)]
    contrib_cluster_c_mu0             = [sum(el) for el in zip(contrib_pi_c_mu0, contrib_rho_c_mu0, contrib_omega_c_mu0, contrib_K_c_mu0, contrib_D_c_mu0, contrib_N_c_mu0, contrib_T_c_mu0, contrib_F_c_mu0, contrib_P_c_mu0, contrib_Q5_c_mu0, contrib_H_c_mu0)]
    contrib_cluster_singlet_c_mu0     = [sum(el) for el in zip(contrib_pi_c_mu0, contrib_rho_c_mu0, contrib_omega_c_mu0, contrib_K_c_mu0, contrib_N_c_mu0, contrib_T_c_mu0, contrib_P_c_mu0, contrib_H_c_mu0)]
    contrib_cluster_color_c_mu0       = [sum(el) for el in zip(contrib_D_c_mu0, contrib_F_c_mu0, contrib_Q5_c_mu0)]
    contrib_pi_c_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_pi_c_mu200)]
    contrib_rho_c_mu200               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_rho_c_mu200)]
    contrib_omega_c_mu200             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_omega_c_mu200)]
    contrib_K_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_K_c_mu200)]
    contrib_D_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_D_c_mu200)]
    contrib_N_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_N_c_mu200)]
    contrib_T_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_T_c_mu200)]
    contrib_F_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_F_c_mu200)]
    contrib_P_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_P_c_mu200)]
    contrib_Q5_c_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_Q5_c_mu200)]
    contrib_H_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_H_c_mu200)]
    contrib_cluster_c_mu200           = [sum(el) for el in zip(contrib_pi_c_mu200, contrib_rho_c_mu200, contrib_omega_c_mu200, contrib_K_c_mu200, contrib_D_c_mu200, contrib_N_c_mu200, contrib_T_c_mu200, contrib_F_c_mu200, contrib_P_c_mu200, contrib_Q5_c_mu200, contrib_H_c_mu200)]
    contrib_cluster_singlet_c_mu200   = [sum(el) for el in zip(contrib_pi_c_mu200, contrib_rho_c_mu200, contrib_omega_c_mu200, contrib_K_c_mu200, contrib_N_c_mu200, contrib_T_c_mu200, contrib_P_c_mu200, contrib_H_c_mu200)]
    contrib_cluster_color_c_mu200     = [sum(el) for el in zip(contrib_D_c_mu200, contrib_F_c_mu200, contrib_Q5_c_mu200)]
    contrib_pi_c_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_pi_c_mu300)]
    contrib_rho_c_mu300               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_rho_c_mu300)]
    contrib_omega_c_mu300             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_omega_c_mu300)]
    contrib_K_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_K_c_mu300)]
    contrib_D_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_D_c_mu300)]
    contrib_N_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_N_c_mu300)]
    contrib_T_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_T_c_mu300)]
    contrib_F_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_F_c_mu300)]
    contrib_P_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_P_c_mu300)]
    contrib_Q5_c_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_Q5_c_mu300)]
    contrib_H_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_H_c_mu300)]
    contrib_cluster_c_mu300           = [sum(el) for el in zip(contrib_pi_c_mu300, contrib_rho_c_mu300, contrib_omega_c_mu300, contrib_K_c_mu300, contrib_D_c_mu300, contrib_N_c_mu300, contrib_T_c_mu300, contrib_F_c_mu300, contrib_P_c_mu300, contrib_Q5_c_mu300, contrib_H_c_mu300)]
    contrib_cluster_singlet_c_mu300   = [sum(el) for el in zip(contrib_pi_c_mu300, contrib_rho_c_mu300, contrib_omega_c_mu300, contrib_K_c_mu300, contrib_N_c_mu300, contrib_T_c_mu300, contrib_P_c_mu300, contrib_H_c_mu300)]
    contrib_cluster_color_c_mu300     = [sum(el) for el in zip(contrib_D_c_mu300, contrib_F_c_mu300, contrib_Q5_c_mu300)]
    contrib_pi_c_mu400                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_pi_c_mu400)]
    contrib_rho_c_mu400               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_rho_c_mu400)]
    contrib_omega_c_mu400             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_omega_c_mu400)]
    contrib_K_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_K_c_mu400)]
    contrib_D_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_D_c_mu400)]
    contrib_N_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_N_c_mu400)]
    contrib_T_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_T_c_mu400)]
    contrib_F_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_F_c_mu400)]
    contrib_P_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_P_c_mu400)]
    contrib_Q5_c_mu400                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_Q5_c_mu400)]
    contrib_H_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_H_c_mu400)]
    contrib_cluster_c_mu400           = [sum(el) for el in zip(contrib_pi_c_mu400, contrib_rho_c_mu400, contrib_omega_c_mu400, contrib_K_c_mu400, contrib_D_c_mu400, contrib_N_c_mu400, contrib_T_c_mu400, contrib_F_c_mu400, contrib_P_c_mu400, contrib_Q5_c_mu400, contrib_H_c_mu400)]
    contrib_cluster_singlet_c_mu400   = [sum(el) for el in zip(contrib_pi_c_mu400, contrib_rho_c_mu400, contrib_omega_c_mu400, contrib_K_c_mu400, contrib_N_c_mu400, contrib_T_c_mu400, contrib_P_c_mu400, contrib_H_c_mu400)]
    contrib_cluster_color_c_mu400     = [sum(el) for el in zip(contrib_D_c_mu400, contrib_F_c_mu400, contrib_Q5_c_mu400)]
    contrib_pi_c_mu500                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_pi_c_mu500)]
    contrib_rho_c_mu500               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_rho_c_mu500)]
    contrib_omega_c_mu500             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_omega_c_mu500)]
    contrib_K_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_K_c_mu500)]
    contrib_D_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_D_c_mu500)]
    contrib_N_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_N_c_mu500)]
    contrib_T_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_T_c_mu500)]
    contrib_F_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_F_c_mu500)]
    contrib_P_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_P_c_mu500)]
    contrib_Q5_c_mu500                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_Q5_c_mu500)]
    contrib_H_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_H_c_mu500)]
    contrib_cluster_c_mu500           = [sum(el) for el in zip(contrib_pi_c_mu500, contrib_rho_c_mu500, contrib_omega_c_mu500, contrib_K_c_mu500, contrib_D_c_mu500, contrib_N_c_mu500, contrib_T_c_mu500, contrib_F_c_mu500, contrib_P_c_mu500, contrib_Q5_c_mu500, contrib_H_c_mu500)]
    contrib_cluster_singlet_c_mu500   = [sum(el) for el in zip(contrib_pi_c_mu500, contrib_rho_c_mu500, contrib_omega_c_mu500, contrib_K_c_mu500, contrib_N_c_mu500, contrib_T_c_mu500, contrib_P_c_mu500, contrib_H_c_mu500)]
    contrib_cluster_color_c_mu500     = [sum(el) for el in zip(contrib_D_c_mu500, contrib_F_c_mu500, contrib_Q5_c_mu500)]

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y)       = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)     = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_low.dat")
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_high.dat")
    (low_1204_6710v2_mu400_x, low_1204_6710v2_mu400_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu400_low.dat")
    (high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu400_high.dat")
    (high_1407_6387_mu0_x, high_1407_6387_mu0_y)         = utils.data_collect(0, 3, "D:/EoS/archive/BDK/lattice_data/const_mu/1407_6387_table1_pressure_mu0.dat")
    (low_1407_6387_mu0_x, low_1407_6387_mu0_y)           = utils.data_collect(0, 2, "D:/EoS/archive/BDK/lattice_data/const_mu/1407_6387_table1_pressure_mu0.dat")
    (high_1309_5258_mu0_x, high_1309_5258_mu0_y)         = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1309_5258_figure6_pressure_mu0_high.dat")
    (low_1309_5258_mu0_x, low_1309_5258_mu0_y)           = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1309_5258_figure6_pressure_mu0_low.dat")
    (high_1710_05024_mu0_x, high_1710_05024_mu0_y)       = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1710_05024_figure8_pressure_mu0_high.dat")
    (low_1710_05024_mu0_x, low_1710_05024_mu0_y)         = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1710_05024_figure8_pressure_mu0_low.dat")

    borsanyi_1204_6710v2_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = numpy.array(borsanyi_1204_6710v2_mu0)
    borsanyi_1204_6710v2_mu200 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu200_x[::-1], low_1204_6710v2_mu200_y[::-1]):
        borsanyi_1204_6710v2_mu200.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu200 = numpy.array(borsanyi_1204_6710v2_mu200)
    borsanyi_1204_6710v2_mu300 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu300_x[::-1], low_1204_6710v2_mu300_y[::-1]):
        borsanyi_1204_6710v2_mu300.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu300 = numpy.array(borsanyi_1204_6710v2_mu300)
    borsanyi_1204_6710v2_mu400 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu400_x[::-1], low_1204_6710v2_mu400_y[::-1]):
        borsanyi_1204_6710v2_mu400.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu400 = numpy.array(borsanyi_1204_6710v2_mu400)
    bazavov_1407_6387_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1407_6387_mu0_x, high_1407_6387_mu0_y)]
    for x_el, y_el in zip(low_1407_6387_mu0_x[::-1], low_1407_6387_mu0_y[::-1]):
        bazavov_1407_6387_mu0.append(numpy.array([x_el, y_el]))
    bazavov_1407_6387_mu0 = numpy.array(bazavov_1407_6387_mu0)
    borsanyi_1309_5258_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1309_5258_mu0_x, high_1309_5258_mu0_y)]
    for x_el, y_el in zip(low_1309_5258_mu0_x[::-1], low_1309_5258_mu0_y[::-1]):
        borsanyi_1309_5258_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1309_5258_mu0 = numpy.array(borsanyi_1309_5258_mu0)
    bazavov_1710_05024_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1710_05024_mu0_x, high_1710_05024_mu0_y)]
    for x_el, y_el in zip(low_1710_05024_mu0_x[::-1], low_1710_05024_mu0_y[::-1]):
        bazavov_1710_05024_mu0.append(numpy.array([x_el, y_el]))
    bazavov_1710_05024_mu0 = numpy.array(bazavov_1710_05024_mu0)

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    #ax1.axis([0., 450., 0., 4.0])
    ax1.axis([50., 300., 0., 0.7])
    ax1.add_patch(matplotlib.patches.Polygon(bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Bazavov et al. (2014), $\mathrm{\mu=0}$'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2014), $\mathrm{\mu=0}$'))
    ax1.add_patch(matplotlib.patches.Polygon(bazavov_1710_05024_mu0, closed = True, fill = True, color = 'magenta', alpha = 0.3, label = r'Bazavov et al. (2018), $\mathrm{\mu=0}$'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=0}$'))
    #ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'red', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=200}$ MeV'))
    #ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu300, closed = True, fill = True, color = 'yellow', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=300}$ MeV'))
    #ax1.plot(T_mu0, contrib_q_mu0, '-', c = 'blue', label = r'$\mathrm{P_{Q,0}}$')
    #ax1.plot(T_mu0, contrib_g_mu0, '-', c = 'red', label = r'$\mathrm{P_{g,0}}$')
    #ax1.plot(T_mu0, contrib_pert_mu0, '-', c = 'pink', label = r'$\mathrm{P_{pert,0}}$')
    #ax1.plot(T_mu0, contrib_qgp_mu0, '-', c = 'black', label = r'$\mathrm{P_{QGP,0}}$')
    ax1.plot(T_mu0, contrib_pi_c_mu0, '-', c = 'blue', label = r'$\mathrm{P_{\pi}}$')
    ax1.plot(T_mu0, contrib_K_c_mu0, '-', c = 'red', label = r'$\mathrm{P_{K}}$')
    ax1.plot(T_mu0, contrib_pi_step_mu0, '-.', c = 'blue', label = r'$\mathrm{P_{\pi,step}}$')
    ax1.plot(T_mu0, contrib_K_step_mu0, '-.', c = 'red', label = r'$\mathrm{P_{K,step}}$')
    #ax1.plot(T_mu0, contrib_cluster_step_mu0, '-', c = 'blue', label = r'$\mathrm{P_{cluster,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_singlet_step_mu0, '-', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_color_step_mu0, '-', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_c_mu0, '-.', c = 'blue', label = r'$\mathrm{P_{cluster,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_singlet_c_mu0, '-.', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_color_c_mu0, '-.', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,0}}$')
    #ax1.legend(loc = 2)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig3 = matplotlib.pyplot.figure(num = 3, figsize = (5.9, 5))
    ax3 = fig3.add_subplot(1, 1, 1)
    #ax3.axis([0., 2000., 0., 5.])
    ax3.axis([100., 200., 0., 2.])
    ax3.add_patch(matplotlib.patches.Polygon(bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Bazavov et al. (2014), $\mathrm{\mu=0}$'))
    ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2014), $\mathrm{\mu=0}$'))
    ax3.add_patch(matplotlib.patches.Polygon(bazavov_1710_05024_mu0, closed = True, fill = True, color = 'magenta', alpha = 0.3, label = r'Bazavov et al. (2018), $\mathrm{\mu=0}$'))
    ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=0}$'))
    #ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=200}$ MeV'))
    #ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu300, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=300}$ MeV'))
    #ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu400, closed = True, fill = True, color = 'cyan', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=400}$ MeV'))
    ax3.plot(T_mu0, [sum(el) for el in zip(contrib_qgp_mu0, contrib_cluster_step_mu0)], '-.', c = 'blue', label = r'step--up step--down $\mathrm{\mu=0}$')
    ax3.plot(T_mu0, [sum(el) for el in zip(contrib_qgp_mu0, contrib_cluster_c_mu0)], '-', c = 'blue', label = r'extended continuum $\mathrm{\mu=0}$')
    #ax3.plot(T_mu200, [sum(el) for el in zip(contrib_qgp_mu200, contrib_cluster_step_mu200)], '-.', c = 'red', label = r'step--up step--down $\mathrm{\mu=200}$')
    #ax3.plot(T_mu200, [sum(el) for el in zip(contrib_qgp_mu200, contrib_cluster_c_mu200)], '-', c = 'red', label = r'extended continuum $\mathrm{\mu=200}$')
    #ax3.plot(T_mu300, [sum(el) for el in zip(contrib_qgp_mu300, contrib_cluster_step_mu300)], '-.', c = 'green', label = r'step--up step--down $\mathrm{\mu=300}$')
    #ax3.plot(T_mu300, [sum(el) for el in zip(contrib_qgp_mu300, contrib_cluster_c_mu300)], '-', c = 'green', label = r'extended continuum $\mathrm{\mu=300}$')
    #ax3.plot(T_mu400, [sum(el) for el in zip(contrib_qgp_mu400, contrib_cluster_step_mu400)], '-.', c = 'cyan', label = r'step--up step--down $\mathrm{\mu=400}$')
    #ax3.plot(T_mu400, [sum(el) for el in zip(contrib_qgp_mu400, contrib_cluster_c_mu400)], '-', c = 'cyan', label = r'extended continuum $\mathrm{\mu=400}$')
    #ax3.plot(T_mu500, [sum(el) for el in zip(contrib_qgp_mu500, contrib_cluster_step_mu500)], '-.', c = 'magenta', label = r'step--up step--down $\mathrm{\mu=500}$')
    #ax3.plot(T_mu500, [sum(el) for el in zip(contrib_qgp_mu500, contrib_cluster_c_mu500)], '-', c = 'magenta', label = r'extended continuum $\mathrm{\mu=500}$')
    #ax3.legend()
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig2 = matplotlib.pyplot.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.axis([0., 450., 0., 1.2])
    ax2.plot(T_mu0, [pnjl.thermo.gcp_sea_lattice.M(el, 0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T_mu0], '-', c = 'green')
    ax2.plot(T_mu200, [pnjl.thermo.gcp_sea_lattice.M(el, 200.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T_mu200], '--', c = 'green')
    ax2.plot(T_mu300, [pnjl.thermo.gcp_sea_lattice.M(el, 300.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T_mu300], '-.', c = 'green')
    ax2.plot(T_mu400, [pnjl.thermo.gcp_sea_lattice.M(el, 400.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T_mu400], ':', c = 'green')
    ax2.scatter(T_mu500, [pnjl.thermo.gcp_sea_lattice.M(el, 500.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T_mu500], marker = 'o', c = 'green')
    ax2.plot(T_mu0, phi_re_mu0, '-', c = 'blue')
    ax2.plot(T_mu200, phi_re_mu200, '--', c = 'blue')
    ax2.plot(T_mu300, phi_re_mu300, '-.', c = 'blue')
    ax2.plot(T_mu400, phi_re_mu400, ':', c = 'blue')
    ax2.scatter(T_mu500, phi_re_mu500, marker = 'o', c = 'blue')
    ax2.plot(T_mu0, phi_im_mu0, '-', c = 'red')
    ax2.plot(T_mu200, phi_im_mu200, '--', c = 'red')
    ax2.plot(T_mu300, phi_im_mu300, '-.', c = 'red')
    ax2.plot(T_mu400, phi_im_mu400, ':', c = 'red')
    ax2.scatter(T_mu500, phi_im_mu500, marker = 'o', c = 'red')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{\Phi}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    fig2.tight_layout(pad = 0.1)
    fig3.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def arccos_cos_phase_factor():

    def phase(M, Mi, Mthi, Ni, Lambda_i):

        hz = 0.5
        nlambda = Ni * Lambda_i

        frac1 = M / nlambda
        frac2 = Mthi / nlambda

        heavi2 = numpy.heaviside((M ** 2) - (Mi ** 2), hz)
        heavi3 = numpy.heaviside((M ** 2) - (Mthi ** 2), hz)
        heavi4 = numpy.heaviside(Mthi + nlambda - M, hz)
        heavi5 = numpy.heaviside(Mthi + nlambda - Mi, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        first = 0.0
        second = 0.0
        third = 0.0

        if (Mthi ** 2) >= (Mi **2):
            #arccos_el = math.acos(arccos_in) if heavi3 > 0 and heavi4 > 0 else 0.0
            if M >= Mi and M <= Mthi:
                first = heavi2 - heavi3
            if M >= Mthi and M <= Mthi + nlambda:
                arccos_el = math.acos(arccos_in)
                second = heavi3 * heavi4 * arccos_el / math.pi
        else:
            if M >= Mthi and M <= Mthi + nlambda:
                #arccos_el2 = math.acos(arccos_in) if heavi5 > 0 and heavi4 > 0 and heavi7 > 0 else 0.0
                arccos_el2 = math.acos(arccos_in)
                third = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((Mthi + nlambda - Mi) / nlambda)

        return first + second + third

    Mi_val = pnjl.defaults.default_MN
    Ni_val = 3.0
    Lam_val = pnjl.defaults.default_L

    Mthi_val1 = 1.4 * Mi_val
    Mthi_val2 = 1.2 * Mi_val
    Mthi_val3 = 1.0 * Mi_val
    Mthi_val4 = 0.8 * Mi_val
    Mthi_val5 = 0.45 * Mi_val

    M_vec = numpy.linspace(0.0, 1.2 * (Mthi_val1 + Ni_val * Lam_val), 2000)
    phase_vec1 = [
        phase(el, Mi_val, Mthi_val1, Ni_val, Lam_val) 
        for el in M_vec]
    phase_vec2 = [
        phase(el, Mi_val, Mthi_val2, Ni_val, Lam_val) 
        for el in M_vec]
    phase_vec3 = [
        phase(el, Mi_val, Mthi_val3, Ni_val, Lam_val) 
        for el in M_vec]
    phase_vec4 = [
        phase(el, Mi_val, Mthi_val4, Ni_val, Lam_val) 
        for el in M_vec]
    phase_vec5 = [
        phase(el, Mi_val, Mthi_val5, Ni_val, Lam_val) 
        for el in M_vec]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(M_vec, phase_vec1, '-', c = 'blue', label = r'$M_{th,i}\gg M_i$')
    ax1.plot(M_vec, phase_vec2, '-', c = 'red', label = r'$M_{th,i}>M_i$')
    ax1.plot(M_vec, phase_vec3, '-', c = 'green', label = r'$M_{th,i}=M_i$')
    ax1.plot(M_vec, phase_vec4, '-', c = 'magenta', label = r'$M_{th,i}<M_i$')
    ax1.plot(M_vec, phase_vec5, '-', c = 'black', label = r'$M_{th,i}\ll M_i$')
    ax1.legend()
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'M [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\delta_i$', fontsize = 16)
    fig1.tight_layout(pad = 0.1)

    T = numpy.linspace(0.0, 500.0, 200)
    delta = [(pnjl.thermo.gcp_sea_lattice.M(el, 0.0) - pnjl.defaults.default_ml) for el in T]
    delta2 = [(pnjl.thermo.gcp_sea_lattice.M(el, 0.0, ml = pnjl.defaults.default_ms) - pnjl.defaults.default_ms) for el in T]

    fig2 = matplotlib.pyplot.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(T, delta, '-', c = 'blue')
    ax2.plot(T, delta2, '-', c = 'red')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    fig2.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def PNJL_perturbative_fit():

    phi_re_1, phi_im_1               = [], []
    phi_re_2, phi_im_2               = [], []
    phi_re_3, phi_im_3               = [], []
    phi_re_4, phi_im_4               = [], []
    phi_re_5, phi_im_5               = [], []
    phi_re_6, phi_im_6               = [], []

    Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_sea_1  = [], [], [], []
    Pres_Q_2, Pres_g_2, Pres_pert_2, Pres_sea_2  = [], [], [], []
    Pres_Q_3, Pres_g_3, Pres_pert_3, Pres_sea_3  = [], [], [], []
    Pres_Q_4, Pres_g_4, Pres_pert_4, Pres_sea_4  = [], [], [], []
    Pres_Q_5, Pres_g_5, Pres_pert_5, Pres_sea_5  = [], [], [], []
    Pres_Q_6, Pres_g_6, Pres_pert_6, Pres_sea_6  = [], [], [], []

    Pres_pi_1, Pres_K_1, Pres_rho_1  = [], [], []
    Pres_omega_1, Pres_D_1, Pres_N_1 = [], [], []
    Pres_T_1, Pres_F_1, Pres_P_1     = [], [], []
    Pres_Q5_1, Pres_H_1              = [], []

    Pres_pi_2, Pres_K_2, Pres_rho_2  = [], [], [] 
    Pres_omega_2, Pres_D_2, Pres_N_2 = [], [], []
    Pres_T_2, Pres_F_2, Pres_P_2     = [], [], []
    Pres_Q5_2, Pres_H_2              = [], []

    Pres_pi_3, Pres_K_3, Pres_rho_3  = [], [], [] 
    Pres_omega_3, Pres_D_3, Pres_N_3 = [], [], []
    Pres_T_3, Pres_F_3, Pres_P_3     = [], [], []
    Pres_Q5_3, Pres_H_3              = [], []

    Pres_pi_4, Pres_K_4, Pres_rho_4  = [], [], [] 
    Pres_omega_4, Pres_D_4, Pres_N_4 = [], [], []
    Pres_T_4, Pres_F_4, Pres_P_4     = [], [], []
    Pres_Q5_4, Pres_H_4              = [], []

    Pres_pi_5, Pres_K_5, Pres_rho_5  = [], [], [] 
    Pres_omega_5, Pres_D_5, Pres_N_5 = [], [], []
    Pres_T_5, Pres_F_5, Pres_P_5     = [], [], []
    Pres_Q5_5, Pres_H_5              = [], []

    Pres_pi_6, Pres_K_6, Pres_rho_6  = [], [], [] 
    Pres_omega_6, Pres_D_6, Pres_N_6 = [], [], []
    Pres_T_6, Pres_F_6, Pres_P_6     = [], [], []
    Pres_Q5_6, Pres_H_6              = [], []

    T_1 = numpy.linspace(1.0, 2000.0, 200)
    T_2 = numpy.linspace(1.0, 2000.0, 200)
    T_3 = numpy.linspace(1.0, 2000.0, 200)
    T_4 = numpy.linspace(1.0, 2000.0, 200)
    T_5 = numpy.linspace(1.0, 2000.0, 200)
    T_6 = numpy.linspace(1.0, 2000.0, 200)

    mu_1   = [0.0 / 3.0 for el in T_1]
    mu_2   = [0.0 / 3.0 for el in T_2]
    mu_3   = [400.0 / 3.0 for el in T_3]
    mu_4   = [100.0 / 3.0 for el in T_3]
    mu_5   = [200.0 / 3.0 for el in T_3]
    mu_6   = [300.0 / 3.0 for el in T_3]

    calc_1                  = False
    calc_2                  = False
    calc_3                  = False
    calc_4                  = False
    calc_5                  = False
    calc_6                  = False

    cluster_backreaction    = False
    pl_turned_off           = False

    pl_1_file       = "D:/EoS/BDK/perturbative_test/pl_1.dat"
    pl_2_file       = "D:/EoS/BDK/perturbative_test/pl_2.dat"
    pl_3_file       = "D:/EoS/BDK/perturbative_test/pl_3.dat"
    pl_4_file       = "D:/EoS/BDK/perturbative_test/pl_4.dat"
    pl_5_file       = "D:/EoS/BDK/perturbative_test/pl_5.dat"
    pl_6_file       = "D:/EoS/BDK/perturbative_test/pl_6.dat"
    pressure_1_file = "D:/EoS/BDK/perturbative_test/pressure_1.dat"
    pressure_2_file = "D:/EoS/BDK/perturbative_test/pressure_2.dat"
    pressure_3_file = "D:/EoS/BDK/perturbative_test/pressure_3.dat"
    pressure_4_file = "D:/EoS/BDK/perturbative_test/pressure_4.dat"
    pressure_5_file = "D:/EoS/BDK/perturbative_test/pressure_5.dat"
    pressure_6_file = "D:/EoS/BDK/perturbative_test/pressure_6.dat"

    if calc_1:
        phi_re_1.append(1e-15)
        phi_im_1.append(2e-15)
        lT = len(T_1)
        for T_el, mu_el in tqdm.tqdm(zip(T_1, mu_1), desc = "Traced Polyakov loop (calc #1)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_1[-1], phi_im_1[-1], with_clusters = cluster_backreaction)
            phi_re_1.append(temp_phi_re)
            phi_im_1.append(temp_phi_im)
        phi_re_1 = phi_re_1[1:]
        phi_im_1 = phi_im_1[1:]
        with open(pl_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_1, mu_1, phi_re_1, phi_im_1)])
    else:
        T_1, mu_1 = utils.data_collect(0, 1, pl_1_file)
        phi_re_1, phi_im_1 = utils.data_collect(2, 3, pl_1_file)

    if calc_2:
        phi_re_2.append(1e-15)
        phi_im_2.append(2e-15)
        lT = len(T_2)
        for T_el, mu_el in tqdm.tqdm(zip(T_2, mu_2), desc = "Traced Polyakov loop (calc #2)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_2[-1], phi_im_2[-1], with_clusters = cluster_backreaction)
            phi_re_2.append(temp_phi_re)
            phi_im_2.append(temp_phi_im)
        phi_re_2 = phi_re_2[1:]
        phi_im_2 = phi_im_2[1:]
        with open(pl_2_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_2, mu_2, phi_re_2, phi_im_2)])
    else:
        T_2, mu_2 = utils.data_collect(0, 1, pl_2_file)
        phi_re_2, phi_im_2 = utils.data_collect(2, 3, pl_2_file)

    if calc_3:
        phi_re_3.append(1e-15)
        phi_im_3.append(2e-15)
        lT = len(T_3)
        for T_el, mu_el in tqdm.tqdm(zip(T_3, mu_3), desc = "Traced Polyakov loop (calc #3)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_3[-1], phi_im_3[-1], with_clusters = cluster_backreaction)
            phi_re_3.append(temp_phi_re)
            phi_im_3.append(temp_phi_im)
        phi_re_3 = phi_re_3[1:]
        phi_im_3 = phi_im_3[1:]
        with open(pl_3_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_3, mu_3, phi_re_3, phi_im_3)])
    else:
        T_3, mu_3 = utils.data_collect(0, 1, pl_3_file)
        phi_re_3, phi_im_3 = utils.data_collect(2, 3, pl_3_file)

    if calc_4:
        phi_re_4.append(1e-15)
        phi_im_4.append(2e-15)
        lT = len(T_4)
        for T_el, mu_el in tqdm.tqdm(zip(T_4, mu_4), desc = "Traced Polyakov loop (calc #4)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_4[-1], phi_im_4[-1], with_clusters = cluster_backreaction)
            phi_re_4.append(temp_phi_re)
            phi_im_4.append(temp_phi_im)
        phi_re_4 = phi_re_4[1:]
        phi_im_4 = phi_im_4[1:]
        with open(pl_4_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_4, mu_4, phi_re_4, phi_im_4)])
    else:
        T_4, mu_4 = utils.data_collect(0, 1, pl_4_file)
        phi_re_4, phi_im_4 = utils.data_collect(2, 3, pl_4_file)

    if calc_5:
        phi_re_5.append(1e-15)
        phi_im_5.append(2e-15)
        lT = len(T_5)
        for T_el, mu_el in tqdm.tqdm(zip(T_5, mu_5), desc = "Traced Polyakov loop (calc #5)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_5[-1], phi_im_5[-1], with_clusters = cluster_backreaction)
            phi_re_5.append(temp_phi_re)
            phi_im_5.append(temp_phi_im)
        phi_re_5 = phi_re_5[1:]
        phi_im_5 = phi_im_5[1:]
        with open(pl_5_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_5, mu_5, phi_re_5, phi_im_5)])
    else:
        T_5, mu_5 = utils.data_collect(0, 1, pl_5_file)
        phi_re_5, phi_im_5 = utils.data_collect(2, 3, pl_5_file)

    if calc_6:
        phi_re_6.append(1e-15)
        phi_im_6.append(2e-15)
        lT = len(T_6)
        for T_el, mu_el in tqdm.tqdm(zip(T_6, mu_6), desc = "Traced Polyakov loop (calc #6)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_6[-1], phi_im_6[-1], with_clusters = cluster_backreaction)
            phi_re_6.append(temp_phi_re)
            phi_im_6.append(temp_phi_im)
        phi_re_6 = phi_re_6[1:]
        phi_im_6 = phi_im_6[1:]
        with open(pl_6_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_6, mu_6, phi_re_6, phi_im_6)])
    else:
        T_6, mu_6 = utils.data_collect(0, 1, pl_6_file)
        phi_re_6, phi_im_6 = utils.data_collect(2, 3, pl_6_file)

    if calc_1:
        Pres_g_1 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_1, phi_re_1, phi_im_1), 
                desc = "Gluon pressure (calc #1)", 
                total = len(T_1),
                ascii = True
                )]
        Pres_sea_1 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(
                zip(T_1, mu_1), 
                desc = "Sigma mf pressure (calc #1)", 
                total = len(T_1),
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_1 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Quark pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True
                    )]
            Pres_pert_1 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Perturbative pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True
                    )]        
            (
                (Pres_pi_1, Pres_rho_1, Pres_omega_1, Pres_K_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, 
                 Pres_Q5_1, Pres_H_1),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_1, mu_1, phi_re_1, phi_im_1)
        else:
            Pres_Q_1 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_1, mu_1, [1.0 for el in T_1], [0.0 for el in T_1]), 
                    desc = "Quark pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True
                    )]
            Pres_pert_1 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_1, mu_1, [1.0 for el in T_1], [0.0 for el in T_1]), 
                    desc = "Perturbative pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True
                    )]        
            (
                (Pres_pi_1, Pres_rho_1, Pres_omega_1, Pres_K_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, 
                 Pres_Q5_1, Pres_H_1),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_1, mu_1, [1.0 for el in T_1], [0.0 for el in T_1])
        with open(pressure_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_pi_1, Pres_K_1, Pres_rho_1, Pres_omega_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, Pres_Q5_1, Pres_H_1, Pres_sea_1)])
    else:
        Pres_Q_1, Pres_g_1       = utils.data_collect(0, 1, pressure_1_file)
        Pres_pert_1, Pres_pi_1   = utils.data_collect(2, 3, pressure_1_file)
        Pres_K_1, _              = utils.data_collect(4, 4, pressure_1_file)
        Pres_rho_1, Pres_omega_1 = utils.data_collect(5, 6, pressure_1_file)
        Pres_D_1, Pres_N_1       = utils.data_collect(7, 8, pressure_1_file)
        Pres_T_1, Pres_F_1       = utils.data_collect(9, 10, pressure_1_file)
        Pres_P_1, Pres_Q5_1      = utils.data_collect(11, 12, pressure_1_file)
        Pres_H_1, Pres_sea_1     = utils.data_collect(13, 14, pressure_1_file)

    if calc_2:
        Pres_g_2 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_2, phi_re_2, phi_im_2), 
                desc = "Gluon pressure (calc #2)", 
                total = len(T_2),
                ascii = True
                )]
        Pres_sea_2 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(
                zip(T_2, mu_2), 
                desc = "Sigma mf pressure (calc #2)", 
                total = len(T_2),
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_2 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_2, mu_2, phi_re_2, phi_im_2), 
                    desc = "Quark pressure (calc #2)", 
                    total = len(T_2), 
                    ascii = True
                    )]
            Pres_pert_2 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_2, mu_2, phi_re_2, phi_im_2), 
                    desc = "Perturbative pressure (calc #2)", 
                    total = len(T_2), 
                    ascii = True
                    )]        
            (
                (Pres_pi_2, Pres_rho_2, Pres_omega_2, Pres_K_2, Pres_D_2, Pres_N_2, Pres_T_2, Pres_F_2, Pres_P_2, 
                 Pres_Q5_2, Pres_H_2),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_2, mu_2, phi_re_2, phi_im_2)
        else:
            Pres_Q_2 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_2, mu_2, [1.0 for el in T_2], [0.0 for el in T_2]), 
                    desc = "Quark pressure (calc #2)", 
                    total = len(T_2), 
                    ascii = True
                    )]
            Pres_pert_2 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_2, mu_2, [1.0 for el in T_2], [0.0 for el in T_2]), 
                    desc = "Perturbative pressure (calc #2)", 
                    total = len(T_2), 
                    ascii = True
                    )]        
            (
                (Pres_pi_2, Pres_rho_2, Pres_omega_2, Pres_K_2, Pres_D_2, Pres_N_2, Pres_T_2, Pres_F_2, Pres_P_2, 
                 Pres_Q5_2, Pres_H_2),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_2, mu_2, [1.0 for el in T_2], [0.0 for el in T_2])
        with open(pressure_2_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_2, Pres_g_2, Pres_pert_2, Pres_pi_2, Pres_K_2, Pres_rho_2, Pres_omega_2, Pres_D_2, Pres_N_2, Pres_T_2, Pres_F_2, Pres_P_2, Pres_Q5_2, Pres_H_2, Pres_sea_2)])
    else:
        Pres_Q_2, Pres_g_2       = utils.data_collect(0, 1, pressure_2_file)
        Pres_pert_2, Pres_pi_2   = utils.data_collect(2, 3, pressure_2_file)
        Pres_K_2, _              = utils.data_collect(4, 4, pressure_2_file)
        Pres_rho_2, Pres_omega_2 = utils.data_collect(5, 6, pressure_2_file)
        Pres_D_2, Pres_N_2       = utils.data_collect(7, 8, pressure_2_file)
        Pres_T_2, Pres_F_2       = utils.data_collect(9, 10, pressure_2_file)
        Pres_P_2, Pres_Q5_2      = utils.data_collect(11, 12, pressure_2_file)
        Pres_H_2, Pres_sea_2     = utils.data_collect(13, 14, pressure_2_file)

    if calc_3:
        Pres_g_3 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_3, phi_re_3, phi_im_3), 
                desc = "Gluon pressure (calc #3)", 
                total = len(T_3),
                ascii = True
                )]
        Pres_sea_3 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(
                zip(T_3, mu_3), 
                desc = "Sigma mf pressure (calc #3)", 
                total = len(T_3),
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_3 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_3, mu_3, phi_re_3, phi_im_3), 
                    desc = "Quark pressure (calc #3)", 
                    total = len(T_3), 
                    ascii = True
                    )]
            Pres_pert_3 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_3, mu_3, phi_re_3, phi_im_3), 
                    desc = "Perturbative pressure (calc #3)", 
                    total = len(T_3), 
                    ascii = True
                    )]        
            (
                (Pres_pi_3, Pres_rho_3, Pres_omega_3, Pres_K_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, 
                 Pres_Q5_3, Pres_H_3),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_3, mu_3, phi_re_3, phi_im_3)
        else:
            Pres_Q_3 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3]), 
                    desc = "Quark pressure (calc #3)", 
                    total = len(T_3), 
                    ascii = True
                    )]
            Pres_pert_3 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3]), 
                    desc = "Perturbative pressure (calc #3)", 
                    total = len(T_3), 
                    ascii = True
                    )]        
            (
                (Pres_pi_3, Pres_rho_3, Pres_omega_3, Pres_K_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, 
                 Pres_Q5_3, Pres_H_3),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3])
        with open(pressure_3_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_3, Pres_g_3, Pres_pert_3, Pres_pi_3, Pres_K_3, Pres_rho_3, Pres_omega_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, Pres_Q5_3, Pres_H_3, Pres_sea_3)])
    else:
        Pres_Q_3, Pres_g_3       = utils.data_collect(0, 1, pressure_3_file)
        Pres_pert_3, Pres_pi_3   = utils.data_collect(2, 3, pressure_3_file)
        Pres_K_3, _              = utils.data_collect(4, 4, pressure_3_file)
        Pres_rho_3, Pres_omega_3 = utils.data_collect(5, 6, pressure_3_file)
        Pres_D_3, Pres_N_3       = utils.data_collect(7, 8, pressure_3_file)
        Pres_T_3, Pres_F_3       = utils.data_collect(9, 10, pressure_3_file)
        Pres_P_3, Pres_Q5_3      = utils.data_collect(11, 12, pressure_3_file)
        Pres_H_3, Pres_sea_3     = utils.data_collect(13, 14, pressure_3_file)

    if calc_4:
        Pres_g_4 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_4, phi_re_4, phi_im_4), 
                desc = "Gluon pressure (calc #4)", 
                total = len(T_4),
                ascii = True
                )]
        Pres_sea_4 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(
                zip(T_4, mu_4), 
                desc = "Sigma mf pressure (calc #4)", 
                total = len(T_4),
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_4 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_4, mu_4, phi_re_4, phi_im_4), 
                    desc = "Quark pressure (calc #4)", 
                    total = len(T_4), 
                    ascii = True
                    )]
            Pres_pert_4 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_4, mu_4, phi_re_4, phi_im_4), 
                    desc = "Perturbative pressure (calc #4)", 
                    total = len(T_3), 
                    ascii = True
                    )]        
            (
                (Pres_pi_4, Pres_rho_4, Pres_omega_4, Pres_K_4, Pres_D_4, Pres_N_4, Pres_T_4, Pres_F_4, Pres_P_4, 
                 Pres_Q5_4, Pres_H_4),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_4, mu_4, phi_re_4, phi_im_4)
        else:
            Pres_Q_4 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_4, mu_4, [1.0 for el in T_4], [0.0 for el in T_4]), 
                    desc = "Quark pressure (calc #4)", 
                    total = len(T_4), 
                    ascii = True
                    )]
            Pres_pert_4 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_4, mu_4, [1.0 for el in T_4], [0.0 for el in T_4]), 
                    desc = "Perturbative pressure (calc #4)", 
                    total = len(T_4), 
                    ascii = True
                    )]        
            (
                (Pres_pi_4, Pres_rho_4, Pres_omega_4, Pres_K_4, Pres_D_4, Pres_N_4, Pres_T_4, Pres_F_4, Pres_P_4, 
                 Pres_Q5_4, Pres_H_4),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_4, mu_4, [1.0 for el in T_4], [0.0 for el in T_4])
        with open(pressure_4_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_4, Pres_g_4, Pres_pert_4, Pres_pi_4, Pres_K_4, Pres_rho_4, Pres_omega_4, Pres_D_4, Pres_N_4, Pres_T_4, Pres_F_4, Pres_P_4, Pres_Q5_4, Pres_H_4, Pres_sea_4)])
    else:
        Pres_Q_4, Pres_g_4       = utils.data_collect(0, 1, pressure_4_file)
        Pres_pert_4, Pres_pi_4   = utils.data_collect(2, 3, pressure_4_file)
        Pres_K_4, _              = utils.data_collect(4, 4, pressure_4_file)
        Pres_rho_4, Pres_omega_4 = utils.data_collect(5, 6, pressure_4_file)
        Pres_D_4, Pres_N_4       = utils.data_collect(7, 8, pressure_4_file)
        Pres_T_4, Pres_F_4       = utils.data_collect(9, 10, pressure_4_file)
        Pres_P_4, Pres_Q5_4      = utils.data_collect(11, 12, pressure_4_file)
        Pres_H_4, Pres_sea_4     = utils.data_collect(13, 14, pressure_4_file)

    if calc_5:
        Pres_g_5 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_5, phi_re_5, phi_im_5), 
                desc = "Gluon pressure (calc #5)", 
                total = len(T_5),
                ascii = True
                )]
        Pres_sea_5 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(
                zip(T_5, mu_5), 
                desc = "Sigma mf pressure (calc #5)", 
                total = len(T_5),
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_5 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_5, mu_5, phi_re_5, phi_im_5), 
                    desc = "Quark pressure (calc #5)", 
                    total = len(T_5), 
                    ascii = True
                    )]
            Pres_pert_5 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_5, mu_5, phi_re_5, phi_im_5), 
                    desc = "Perturbative pressure (calc #5)", 
                    total = len(T_5), 
                    ascii = True
                    )]        
            (
                (Pres_pi_5, Pres_rho_5, Pres_omega_5, Pres_K_5, Pres_D_5, Pres_N_5, Pres_T_5, Pres_F_5, Pres_P_5, 
                 Pres_Q5_5, Pres_H_5),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_5, mu_5, phi_re_5, phi_im_5)
        else:
            Pres_Q_5 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_5, mu_5, [1.0 for el in T_5], [0.0 for el in T_5]), 
                    desc = "Quark pressure (calc #5)", 
                    total = len(T_5), 
                    ascii = True
                    )]
            Pres_pert_5 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_5, mu_5, [1.0 for el in T_5], [0.0 for el in T_5]), 
                    desc = "Perturbative pressure (calc #5)", 
                    total = len(T_5), 
                    ascii = True
                    )]        
            (
                (Pres_pi_5, Pres_rho_5, Pres_omega_5, Pres_K_5, Pres_D_5, Pres_N_5, Pres_T_5, Pres_F_5, Pres_P_5, 
                 Pres_Q5_5, Pres_H_5),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_5, mu_5, [1.0 for el in T_5], [0.0 for el in T_5])
        with open(pressure_5_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_5, Pres_g_5, Pres_pert_5, Pres_pi_5, Pres_K_5, Pres_rho_5, Pres_omega_5, Pres_D_5, Pres_N_5, Pres_T_5, Pres_F_5, Pres_P_5, Pres_Q5_5, Pres_H_5, Pres_sea_5)])
    else:
        Pres_Q_5, Pres_g_5       = utils.data_collect(0, 1, pressure_5_file)
        Pres_pert_5, Pres_pi_5   = utils.data_collect(2, 3, pressure_5_file)
        Pres_K_5, _              = utils.data_collect(4, 4, pressure_5_file)
        Pres_rho_5, Pres_omega_5 = utils.data_collect(5, 6, pressure_5_file)
        Pres_D_5, Pres_N_5       = utils.data_collect(7, 8, pressure_5_file)
        Pres_T_5, Pres_F_5       = utils.data_collect(9, 10, pressure_5_file)
        Pres_P_5, Pres_Q5_5      = utils.data_collect(11, 12, pressure_5_file)
        Pres_H_5, Pres_sea_5     = utils.data_collect(13, 14, pressure_5_file)

    if calc_6:
        Pres_g_6 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_6, phi_re_6, phi_im_6), 
                desc = "Gluon pressure (calc #6)", 
                total = len(T_6),
                ascii = True
                )]
        Pres_sea_6 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(
                zip(T_6, mu_6), 
                desc = "Sigma mf pressure (calc #6)", 
                total = len(T_6),
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_6 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_6, mu_6, phi_re_6, phi_im_6), 
                    desc = "Quark pressure (calc #6)", 
                    total = len(T_6), 
                    ascii = True
                    )]
            Pres_pert_6 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_6, mu_6, phi_re_6, phi_im_6), 
                    desc = "Perturbative pressure (calc #6)", 
                    total = len(T_6), 
                    ascii = True
                    )]        
            (
                (Pres_pi_6, Pres_rho_6, Pres_omega_6, Pres_K_6, Pres_D_6, Pres_N_6, Pres_T_6, Pres_F_6, Pres_P_6, 
                 Pres_Q5_6, Pres_H_6),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_6, mu_6, phi_re_6, phi_im_6)
        else:
            Pres_Q_6 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_6, mu_6, [1.0 for el in T_6], [0.0 for el in T_6]), 
                    desc = "Quark pressure (calc #6)", 
                    total = len(T_6), 
                    ascii = True
                    )]
            Pres_pert_6 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_6, mu_6, [1.0 for el in T_6], [0.0 for el in T_6]), 
                    desc = "Perturbative pressure (calc #6)", 
                    total = len(T_6), 
                    ascii = True
                    )]        
            (
                (Pres_pi_6, Pres_rho_6, Pres_omega_6, Pres_K_6, Pres_D_6, Pres_N_6, Pres_T_6, Pres_F_6, Pres_P_6, 
                 Pres_Q5_6, Pres_H_6),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_6, mu_6, [1.0 for el in T_6], [0.0 for el in T_6])
        with open(pressure_6_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_6, Pres_g_6, Pres_pert_6, Pres_pi_6, Pres_K_6, Pres_rho_6, Pres_omega_6, Pres_D_6, Pres_N_6, Pres_T_6, Pres_F_6, Pres_P_6, Pres_Q5_6, Pres_H_6, Pres_sea_6)])
    else:
        Pres_Q_6, Pres_g_6       = utils.data_collect(0, 1, pressure_6_file)
        Pres_pert_6, Pres_pi_6   = utils.data_collect(2, 3, pressure_6_file)
        Pres_K_6, _              = utils.data_collect(4, 4, pressure_6_file)
        Pres_rho_6, Pres_omega_6 = utils.data_collect(5, 6, pressure_6_file)
        Pres_D_6, Pres_N_6       = utils.data_collect(7, 8, pressure_6_file)
        Pres_T_6, Pres_F_6       = utils.data_collect(9, 10, pressure_6_file)
        Pres_P_6, Pres_Q5_6      = utils.data_collect(11, 12, pressure_6_file)
        Pres_H_6, Pres_sea_6     = utils.data_collect(13, 14, pressure_6_file)

    contrib_q_1    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_Q_1)]
    contrib_g_1    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_g_1)]
    contrib_pert_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_pert_1)]
    contrib_sea_1  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_1)]
    contrib_qgp_1  = [sum(el) for el in zip(contrib_q_1, contrib_g_1, contrib_pert_1, contrib_sea_1)]

    contrib_q_2    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_Q_2)]
    contrib_g_2    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_g_2)]
    contrib_pert_2 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_pert_2)]
    contrib_sea_2  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_2)]
    contrib_qgp_2  = [sum(el) for el in zip(contrib_q_2, contrib_g_2, contrib_pert_2, contrib_sea_2)]

    contrib_q_3    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_Q_3)]
    contrib_g_3    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_g_3)]
    contrib_pert_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_pert_3)]
    contrib_sea_3  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_3)]
    contrib_qgp_3  = [sum(el) for el in zip(contrib_q_3, contrib_g_3, contrib_pert_3, contrib_sea_3)]

    contrib_q_4    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_Q_4)]
    contrib_g_4    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_g_4)]
    contrib_pert_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_pert_4)]
    contrib_sea_4  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_4)]
    contrib_qgp_4  = [sum(el) for el in zip(contrib_q_4, contrib_g_4, contrib_pert_4, contrib_sea_4)]

    contrib_q_5    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_Q_5)]
    contrib_g_5    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_g_5)]
    contrib_pert_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_pert_5)]
    contrib_sea_5  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_5)]
    contrib_qgp_5  = [sum(el) for el in zip(contrib_q_5, contrib_g_5, contrib_pert_5, contrib_sea_5)]

    contrib_q_6    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_Q_6)]
    contrib_g_6    = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_g_6)]
    contrib_pert_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_pert_6)]
    contrib_sea_6  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_6)]
    contrib_qgp_6  = [sum(el) for el in zip(contrib_q_6, contrib_g_6, contrib_pert_6, contrib_sea_6)]

    contrib_pi_1                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_pi_1)]
    contrib_rho_1                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_rho_1)]
    contrib_omega_1               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_omega_1)]
    contrib_K_1                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_K_1)]
    contrib_D_1                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_D_1)]
    contrib_N_1                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_N_1)]
    contrib_T_1                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_T_1)]
    contrib_F_1                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_F_1)]
    contrib_P_1                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_P_1)]
    contrib_Q5_1                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_Q5_1)]
    contrib_H_1                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_H_1)]
    contrib_cluster_1             = [sum(el) for el in zip(contrib_pi_1, contrib_rho_1, contrib_omega_1, contrib_K_1, contrib_D_1, contrib_N_1, contrib_T_1, contrib_F_1, contrib_P_1, contrib_Q5_1, contrib_H_1)]
    contrib_cluster_singlet_1     = [sum(el) for el in zip(contrib_pi_1, contrib_rho_1, contrib_omega_1, contrib_K_1, contrib_N_1, contrib_T_1, contrib_P_1, contrib_H_1)]
    contrib_cluster_color_1       = [sum(el) for el in zip(contrib_D_1, contrib_F_1, contrib_Q5_1)]
    contrib_total_1               = [sum(el) for el in zip(contrib_cluster_1, contrib_qgp_1)]

    contrib_pi_2                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_pi_2)]
    contrib_rho_2                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_rho_2)]
    contrib_omega_2               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_omega_2)]
    contrib_K_2                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_K_2)]
    contrib_D_2                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_D_2)]
    contrib_N_2                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_N_2)]
    contrib_T_2                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_T_2)]
    contrib_F_2                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_F_2)]
    contrib_P_2                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_P_2)]
    contrib_Q5_2                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_Q5_2)]
    contrib_H_2                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_2, Pres_H_2)]
    contrib_cluster_2             = [sum(el) for el in zip(contrib_pi_2, contrib_rho_2, contrib_omega_2, contrib_K_2, contrib_D_2, contrib_N_2, contrib_T_2, contrib_F_2, contrib_P_2, contrib_Q5_2, contrib_H_2)]
    contrib_cluster_singlet_2     = [sum(el) for el in zip(contrib_pi_2, contrib_rho_2, contrib_omega_2, contrib_K_2, contrib_N_2, contrib_T_2, contrib_P_2, contrib_H_2)]
    contrib_cluster_color_2       = [sum(el) for el in zip(contrib_D_2, contrib_F_2, contrib_Q5_2)]
    contrib_total_2               = [sum(el) for el in zip(contrib_cluster_2, contrib_qgp_2)]

    contrib_pi_3                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_pi_3)]
    contrib_rho_3                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_rho_3)]
    contrib_omega_3               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_omega_3)]
    contrib_K_3                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_K_3)]
    contrib_D_3                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_D_3)]
    contrib_N_3                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_N_3)]
    contrib_T_3                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_T_3)]
    contrib_F_3                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_F_3)]
    contrib_P_3                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_P_3)]
    contrib_Q5_3                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_Q5_3)]
    contrib_H_3                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_H_3)]
    contrib_cluster_3             = [sum(el) for el in zip(contrib_pi_3, contrib_rho_3, contrib_omega_3, contrib_K_3, contrib_D_3, contrib_N_3, contrib_T_3, contrib_F_3, contrib_P_3, contrib_Q5_3, contrib_H_3)]
    contrib_cluster_singlet_3     = [sum(el) for el in zip(contrib_pi_3, contrib_rho_3, contrib_omega_3, contrib_K_3, contrib_N_3, contrib_T_3, contrib_P_3, contrib_H_3)]
    contrib_cluster_color_3       = [sum(el) for el in zip(contrib_D_3, contrib_F_3, contrib_Q5_3)]
    contrib_total_3               = [sum(el) for el in zip(contrib_cluster_3, contrib_qgp_3)]

    contrib_pi_4                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_pi_4)]
    contrib_rho_4                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_rho_4)]
    contrib_omega_4               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_omega_4)]
    contrib_K_4                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_K_4)]
    contrib_D_4                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_D_4)]
    contrib_N_4                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_N_4)]
    contrib_T_4                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_T_4)]
    contrib_F_4                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_F_4)]
    contrib_P_4                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_P_4)]
    contrib_Q5_4                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_Q5_4)]
    contrib_H_4                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_H_4)]
    contrib_cluster_4             = [sum(el) for el in zip(contrib_pi_4, contrib_rho_4, contrib_omega_4, contrib_K_4, contrib_D_4, contrib_N_4, contrib_T_4, contrib_F_4, contrib_P_4, contrib_Q5_4, contrib_H_4)]
    contrib_cluster_singlet_4     = [sum(el) for el in zip(contrib_pi_4, contrib_rho_4, contrib_omega_4, contrib_K_4, contrib_N_4, contrib_T_4, contrib_P_4, contrib_H_4)]
    contrib_cluster_color_4       = [sum(el) for el in zip(contrib_D_4, contrib_F_4, contrib_Q5_4)]
    contrib_total_4               = [sum(el) for el in zip(contrib_cluster_4, contrib_qgp_4)]

    contrib_pi_5                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_pi_5)]
    contrib_rho_5                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_rho_5)]
    contrib_omega_5               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_omega_5)]
    contrib_K_5                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_K_5)]
    contrib_D_5                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_D_5)]
    contrib_N_5                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_N_5)]
    contrib_T_5                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_T_5)]
    contrib_F_5                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_F_5)]
    contrib_P_5                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_P_5)]
    contrib_Q5_5                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_Q5_5)]
    contrib_H_5                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_H_5)]
    contrib_cluster_5             = [sum(el) for el in zip(contrib_pi_5, contrib_rho_5, contrib_omega_5, contrib_K_5, contrib_D_5, contrib_N_5, contrib_T_5, contrib_F_5, contrib_P_5, contrib_Q5_5, contrib_H_5)]
    contrib_cluster_singlet_5     = [sum(el) for el in zip(contrib_pi_5, contrib_rho_5, contrib_omega_5, contrib_K_5, contrib_N_5, contrib_T_5, contrib_P_5, contrib_H_5)]
    contrib_cluster_color_3       = [sum(el) for el in zip(contrib_D_5, contrib_F_5, contrib_Q5_5)]
    contrib_total_5               = [sum(el) for el in zip(contrib_cluster_5, contrib_qgp_5)]

    contrib_pi_6                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_pi_6)]
    contrib_rho_6                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_rho_6)]
    contrib_omega_6               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_omega_6)]
    contrib_K_6                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_K_6)]
    contrib_D_6                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_D_6)]
    contrib_N_6                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_N_6)]
    contrib_T_6                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_T_6)]
    contrib_F_6                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_F_6)]
    contrib_P_6                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_P_6)]
    contrib_Q5_6                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_Q5_6)]
    contrib_H_6                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_H_6)]
    contrib_cluster_6             = [sum(el) for el in zip(contrib_pi_6, contrib_rho_6, contrib_omega_6, contrib_K_6, contrib_D_6, contrib_N_6, contrib_T_6, contrib_F_6, contrib_P_6, contrib_Q5_6, contrib_H_6)]
    contrib_cluster_singlet_6     = [sum(el) for el in zip(contrib_pi_6, contrib_rho_6, contrib_omega_6, contrib_K_6, contrib_N_6, contrib_T_6, contrib_P_6, contrib_H_6)]
    contrib_cluster_color_6       = [sum(el) for el in zip(contrib_D_6, contrib_F_6, contrib_Q5_6)]
    contrib_total_6               = [sum(el) for el in zip(contrib_cluster_6, contrib_qgp_6)]

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y)       = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)     = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")
    (low_1204_6710v2_mu100_x, low_1204_6710v2_mu100_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu100_low.dat")
    (high_1204_6710v2_mu100_x, high_1204_6710v2_mu100_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu100_high.dat")
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_low.dat")
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_high.dat")
    (low_1204_6710v2_mu400_x, low_1204_6710v2_mu400_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu400_low.dat")
    (high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu400_high.dat")
    (high_1407_6387_mu0_x, high_1407_6387_mu0_y)         = utils.data_collect(0, 3, "D:/EoS/archive/BDK/lattice_data/const_mu/1407_6387_table1_pressure_mu0.dat")
    (low_1407_6387_mu0_x, low_1407_6387_mu0_y)           = utils.data_collect(0, 2, "D:/EoS/archive/BDK/lattice_data/const_mu/1407_6387_table1_pressure_mu0.dat")
    (high_1309_5258_mu0_x, high_1309_5258_mu0_y)         = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1309_5258_figure6_pressure_mu0_high.dat")
    (low_1309_5258_mu0_x, low_1309_5258_mu0_y)           = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1309_5258_figure6_pressure_mu0_low.dat")
    (high_1710_05024_mu0_x, high_1710_05024_mu0_y)       = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1710_05024_figure8_pressure_mu0_high.dat")
    (low_1710_05024_mu0_x, low_1710_05024_mu0_y)         = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1710_05024_figure8_pressure_mu0_low.dat")

    borsanyi_1204_6710v2_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = numpy.array(borsanyi_1204_6710v2_mu0)
    borsanyi_1204_6710v2_mu100 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu100_x, high_1204_6710v2_mu100_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu100_x[::-1], low_1204_6710v2_mu100_y[::-1]):
        borsanyi_1204_6710v2_mu100.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu100 = numpy.array(borsanyi_1204_6710v2_mu100)
    borsanyi_1204_6710v2_mu200 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu200_x[::-1], low_1204_6710v2_mu200_y[::-1]):
        borsanyi_1204_6710v2_mu200.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu200 = numpy.array(borsanyi_1204_6710v2_mu200)
    borsanyi_1204_6710v2_mu300 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu300_x[::-1], low_1204_6710v2_mu300_y[::-1]):
        borsanyi_1204_6710v2_mu300.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu300 = numpy.array(borsanyi_1204_6710v2_mu300)
    borsanyi_1204_6710v2_mu400 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu400_x[::-1], low_1204_6710v2_mu400_y[::-1]):
        borsanyi_1204_6710v2_mu400.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu400 = numpy.array(borsanyi_1204_6710v2_mu400)
    bazavov_1407_6387_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1407_6387_mu0_x, high_1407_6387_mu0_y)]
    for x_el, y_el in zip(low_1407_6387_mu0_x[::-1], low_1407_6387_mu0_y[::-1]):
        bazavov_1407_6387_mu0.append(numpy.array([x_el, y_el]))
    bazavov_1407_6387_mu0 = numpy.array(bazavov_1407_6387_mu0)
    borsanyi_1309_5258_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1309_5258_mu0_x, high_1309_5258_mu0_y)]
    for x_el, y_el in zip(low_1309_5258_mu0_x[::-1], low_1309_5258_mu0_y[::-1]):
        borsanyi_1309_5258_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1309_5258_mu0 = numpy.array(borsanyi_1309_5258_mu0)
    bazavov_1710_05024_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1710_05024_mu0_x, high_1710_05024_mu0_y)]
    for x_el, y_el in zip(low_1710_05024_mu0_x[::-1], low_1710_05024_mu0_y[::-1]):
        bazavov_1710_05024_mu0.append(numpy.array([x_el, y_el]))
    bazavov_1710_05024_mu0 = numpy.array(bazavov_1710_05024_mu0)

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([50., 300., 0., 0.7])
    #ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=0}$'))
    #ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu100, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=100}$ MeV'))
    #ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=200}$ MeV'))
    #ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu300, closed = True, fill = True, color = 'magenta', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=300}$ MeV'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu400, closed = True, fill = True, color = 'cyan', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=400}$ MeV'))
    #ax1.plot(T_2, contrib_total_2, '-', c = 'blue', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    #ax1.plot(T_2, contrib_total_4, '-', c = 'red', label = r'$\mathrm{P_{PNJL+BU},~\mu=100}$ MeV')
    #ax1.plot(T_2, contrib_total_5, '-', c = 'green', label = r'$\mathrm{P_{PNJL+BU},~\mu=200}$ MeV')
    #ax1.plot(T_2, contrib_total_6, '-', c = 'magenta', label = r'$\mathrm{P_{PNJL+BU},~\mu=300}$ MeV')
    ax1.plot(T_2, contrib_total_3, '-', c = 'cyan', label = r'$\mathrm{P_{PNJL+BU},~\mu=400}$ MeV')
    ax1.legend(loc = 2)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig2 = matplotlib.pyplot.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.axis([50., 300., 0., 0.7])
    ax2.add_patch(matplotlib.patches.Polygon(bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Bazavov et al. (2014)'))
    ax2.add_patch(matplotlib.patches.Polygon(borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2014)'))
    ax2.add_patch(matplotlib.patches.Polygon(bazavov_1710_05024_mu0, closed = True, fill = True, color = 'magenta', alpha = 0.3, label = r'Bazavov et al. (2018)'))
    ax2.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012)'))
    ax2.plot(T_2, contrib_total_1, '-.', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    ax2.plot(T_2, contrib_total_2, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    ax2.plot(T_2, contrib_g_2, '-', c = 'magenta')
    ax2.plot(T_2, contrib_g_1, '-.', c = 'magenta')
    ax2.plot(T_2, contrib_sea_2, '-', c = 'blue')
    ax2.plot(T_2, contrib_sea_1, '-.', c = 'blue')
    ax2.legend(loc = 2)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    fig2.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def deriv_test_bdensity():
    T = 1.0
    mu = numpy.linspace(0.0, 2000.0, num = 200)
    phi_re = []
    phi_im = []
    p_vec = []
    b_vec_alt = []

    calc = False

    pl_file       = "D:/EoS/BDK/deriv_test/pl_mu.dat"

    lmu = len(mu)
    if calc:
        phi_re.append(1e-15)
        phi_im.append(2e-15)
        for mu_el in tqdm.tqdm(mu, desc = "Traced Polyakov loop (calc f(mu))", total = lmu, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T, mu_el, phi_re[-1], phi_im[-1], with_clusters = False)
            phi_re.append(temp_phi_re)
            phi_im.append(temp_phi_im)
        phi_re = phi_re[1:]
        phi_im = phi_im[1:]
        with open(pl_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[mu_el, phi_re_el, phi_im_el] for mu_el, phi_re_el, phi_im_el in zip(mu, phi_re, phi_im)])
    else:
        _, mu           = utils.data_collect(0, 0, pl_file)
        phi_re, phi_im  = utils.data_collect(1, 2, pl_file)

    p_vec = [pnjl.thermo.gcp_cluster.bound_step_continuum_arccos_cos.pressure(
        T, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el),
        pnjl.defaults.default_Mpi, 2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, mu_el),
        0, 0, 3.0, 2.0, pnjl.defaults.default_L) for mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(mu, phi_re, phi_im), desc = "Pressure (calc f(mu))", total = lmu, ascii = True)]
    for mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(mu, phi_re, phi_im), desc = "Baryon density without phi (calc f(mu))", total = lmu, ascii = True):
        h = 1e-2
        mu_vec = []
        Phi_vec = []
        Phib_vec = []
        if mu_el > 0.0:
            mu_vec = [mu_el + 2 * h, mu_el + h, mu_el - h, mu_el - 2 * h]
        else:
            mu_vec = [h, 0.0]
        Phi_vec = [complex(phi_re_el, phi_im_el) for el in mu_vec]
        Phib_vec = [complex(phi_re_el, -phi_im_el) for el in mu_vec]
        Mth_vec = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, el) for el in mu_vec]
        b_vec_alt.append(pnjl.thermo.gcp_cluster.bound_step_continuum_arccos_cos.bdensity(
            T, mu_vec, Phi_vec, Phib_vec,
            pnjl.defaults.default_Mpi, Mth_vec, 0, 0, 3.0, 2.0, pnjl.defaults.default_L))

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([0.0, 2000., min(p_vec), max(p_vec)])

    ax1.plot(mu, p_vec, '-', c = 'blue', label = r'pressure')
    ax1.plot(mu, b_vec_alt, '-', c = 'red', label = r'baryon density numeric')

    ax1.legend()
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'$\mathrm{\mu}$ [MeV]', fontsize = 16)
    ax1.set_ylabel(r'p or $\mathrm{n_B}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def deriv_test_sdensity():
    T = numpy.linspace(1.0, 2000.0, num = 200)
    mu = 0.0
    phi_re = []
    phi_im = []
    p_vec = []
    s_vec_alt = []

    calc = False

    pl_file       = "D:/EoS/BDK/deriv_test/pl_T.dat"

    lT = len(T)
    if calc:
        phi_re.append(1e-15)
        phi_im.append(2e-15)
        for T_el in tqdm.tqdm(T, desc = "Traced Polyakov loop (calc f(T))", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu, phi_re[-1], phi_im[-1], with_clusters = False)
            phi_re.append(temp_phi_re)
            phi_im.append(temp_phi_im)
        phi_re = phi_re[1:]
        phi_im = phi_im[1:]
        with open(pl_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re, phi_im)])
    else:
        _, T           = utils.data_collect(0, 0, pl_file)
        phi_re, phi_im  = utils.data_collect(1, 2, pl_file)

    p_vec = [pnjl.thermo.gcp_cluster.bound_step_continuum_arccos_cos.pressure(
        T_el, mu, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el),
        pnjl.defaults.default_Mpi, 2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu),
        0, 0, 3.0, 2.0, pnjl.defaults.default_L) for T_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, phi_re, phi_im), desc = "Pressure (calc f(mu))", total = lT, ascii = True)]
    for T_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, phi_re, phi_im), desc = "Entropy density without phi (calc f(T))", total = lT, ascii = True):
        h = 1e-2
        T_vec = []
        Phi_vec = []
        Phib_vec = []
        if T_el > 0.0:
            T_vec = [T_el + 2 * h, T_el + h, T_el - h, T_el - 2 * h]
        else:
            T_vec = [h, 0.0]
        Phi_vec = [complex(phi_re_el, phi_im_el) for el in T_vec]
        Phib_vec = [complex(phi_re_el, -phi_im_el) for el in T_vec]
        Mth_vec = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(el, mu) for el in T_vec]
        s_vec_alt.append(pnjl.thermo.gcp_cluster.bound_step_continuum_arccos_cos.sdensity(
            T_vec, mu, Phi_vec, Phib_vec,
            pnjl.defaults.default_Mpi, Mth_vec, 0, 0, 3.0, 2.0, pnjl.defaults.default_L))

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([0.0, 2000., min(p_vec), max(p_vec)])

    ax1.plot(T, p_vec, '-', c = 'blue', label = r'pressure')
    ax1.plot(T, s_vec_alt, '-', c = 'red', label = r'entropy density numeric')

    ax1.legend()
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'p or s', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def PNJL_mu_over_T():

    phi_re, phi_im               = [], []

    Pres_Q, Pres_g, Pres_pert, Pres_sea  = [], [], [], []
    BDen_Q, BDen_g, BDen_pert, BDen_sea  = [], [], [], []
    SDen_Q, SDen_g, SDen_pert, SDen_sea  = [], [], [], []

    Pres_pi, Pres_K, Pres_rho  = [], [], []
    Pres_omega, Pres_D, Pres_N = [], [], []
    Pres_T, Pres_F, Pres_P     = [], [], []
    Pres_Q5, Pres_H            = [], []

    BDen_pi, BDen_K, BDen_rho  = [], [], []
    BDen_omega, BDen_D, BDen_N = [], [], []
    BDen_T, BDen_F, BDen_P     = [], [], []
    BDen_Q5, BDen_H            = [], []

    SDen_pi, SDen_K, SDen_rho  = [], [], []
    SDen_omega, SDen_D, SDen_N = [], [], []
    SDen_T, SDen_F, SDen_P     = [], [], []
    SDen_Q5, SDen_H            = [], []

    Pres_tot, BDen_tot, SDen_tot = [], [], []

    mu_over_T = 1.0
    T = numpy.linspace(1.0, 2000.0, num = 200)
    mu = [(mu_over_T * el) / 3.0 for el in T]

    calc_pl     = False
    calc_thermo = False

    cluster_backreaction    = False
    pl_turned_off           = False

    pl_file     = "D:/EoS/BDK/mu_over_t/pl_1.dat"
    pressure_file = "D:/EoS/BDK/mu_over_t/pressure_1.dat"
    baryon_file = "D:/EoS/BDK/mu_over_t/bdensity_1.dat"
    entropy_file = "D:/EoS/BDK/mu_over_t/entropy_1.dat"

    if calc_pl:
        phi_re.append(1e-15)
        phi_im.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm.tqdm(zip(T, mu), desc = "Traced Polyakov loop", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re[-1], phi_im[-1], with_clusters = cluster_backreaction)
            phi_re.append(temp_phi_re)
            phi_im.append(temp_phi_im)
        phi_re = phi_re[1:]
        phi_im = phi_im[1:]
        with open(pl_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T, mu, phi_re, phi_im)])
    else:
        T, mu = utils.data_collect(0, 1, pl_file)
        phi_re, phi_im = utils.data_collect(2, 3, pl_file)

    if calc_thermo:
        Pres_g = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T, phi_re, phi_im), 
                desc = "Gluon pressure", 
                total = len(T),
                ascii = True
                )]
        BDen_g = [0.0 for T_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, phi_re, phi_im), desc = "Gluon baryon density", total = len(T), ascii = True)]
        for T_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, phi_re, phi_im), desc = "Gluon entropy density", total = len(T), ascii = True):
            h = 1e-2
            T_vec = []
            Phi_vec = []
            Phib_vec = []
            if T_el > 0.0:
                T_vec = [T_el + 2 * h, T_el + h, T_el - h, T_el - 2 * h]
            else:
                T_vec = [h, 0.0]
            Phi_vec = [complex(phi_re_el, phi_im_el) for el in T_vec]
            Phib_vec = [complex(phi_re_el, -phi_im_el) for el in T_vec]
            SDen_g.append(pnjl.thermo.gcp_pl_polynomial.sdensity(T_vec, Phi_vec, Phib_vec))
        Pres_sea = [0.0 for T_el, mu_el in tqdm.tqdm(zip(T, mu), desc = "Sigma mf pressure", total = len(T), ascii = True)]
        BDen_sea = [0.0 for T_el, mu_el in tqdm.tqdm(zip(T, mu), desc = "Sigma mf baryon density", total = len(T), ascii = True)]
        SDen_sea = [0.0 for T_el, mu_el in tqdm.tqdm(zip(T, mu), desc = "Sigma mf entropy density", total = len(T), ascii = True)]
        if not pl_turned_off:
            Pres_Q = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu, phi_re, phi_im), 
                    desc = "Quark pressure", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu, phi_re, phi_im), 
                    desc = "Perturbative pressure", 
                    total = len(T), 
                    ascii = True
                    )]
            for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im), desc = "Quark and perturbative baryon density", total = len(T), ascii = True):
                h = 1e-2
                mu_vec = []
                Phi_vec = []
                Phib_vec = []
                if mu_el > 0.0:
                    mu_vec = [mu_el + 2 * h, mu_el + h, mu_el - h, mu_el - 2 * h]
                else:
                    mu_vec = [h, 0.0]
                Phi_vec = [complex(phi_re_el, phi_im_el) for el in mu_vec]
                Phib_vec = [complex(phi_re_el, -phi_im_el) for el in mu_vec]
                BDen_Q.append(pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec) + pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, ml = pnjl.defaults.default_ms))
                BDen_pert.append(pnjl.thermo.gcp_perturbative.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, Nf = 3.0))
            for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im), desc = "Quark and perturbative entropy density", total = len(T), ascii = True):
                h = 1e-2
                T_vec = []
                Phi_vec = []
                Phib_vec = []
                if T_el > 0.0:
                    T_vec = [T_el + 2 * h, T_el + h, T_el - h, T_el - 2 * h]
                else:
                    T_vec = [h, 0.0]
                Phi_vec = [complex(phi_re_el, phi_im_el) for el in T_vec]
                Phib_vec = [complex(phi_re_el, -phi_im_el) for el in T_vec]
                SDen_Q.append(pnjl.thermo.gcp_quark.sdensity(T_vec, mu_el, Phi_vec, Phib_vec) + pnjl.thermo.gcp_quark.sdensity(T_vec, mu_el, Phi_vec, Phib_vec, ml = pnjl.defaults.default_ms))
                SDen_pert.append(pnjl.thermo.gcp_perturbative.sdensity(T_vec, mu_el, Phi_vec, Phib_vec, Nf = 3.0))
            (
                (Pres_pi, Pres_rho, Pres_omega, Pres_K, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, 
                 Pres_Q5, Pres_H),
                (BDen_pi, BDen_rho, BDen_omega, BDen_K, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, 
                 BDen_Q5, BDen_H),
                (SDen_pi, SDen_rho, SDen_omega, SDen_K, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, 
                 SDen_Q5, SDen_H)
            ) = clusters_c(T, mu, phi_re, phi_im)
        else:
            Pres_Q = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure", 
                    total = len(T), 
                    ascii = True
                    )]
            for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, mu, [1.0 for el in T], [0.0 for el in T]), desc = "Quark and perturbative baryon density", total = len(T), ascii = True):
                h = 1e-2
                mu_vec = []
                Phi_vec = []
                Phib_vec = []
                if mu_el > 0.0:
                    mu_vec = [mu_el + 2 * h, mu_el + h, mu_el - h, mu_el - 2 * h]
                else:
                    mu_vec = [h, 0.0]
                Phi_vec = [complex(phi_re_el, phi_im_el) for el in mu_vec]
                Phib_vec = [complex(phi_re_el, -phi_im_el) for el in mu_vec]
                BDen_Q.append(pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec) + pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, ml = pnjl.defaults.default_ms))
                BDen_pert.append(pnjl.thermo.gcp_perturbative.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, Nf = 3.0))
            for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, mu, [1.0 for el in T], [0.0 for el in T]), desc = "Quark and perturbative entropy density", total = len(T), ascii = True):
                h = 1e-2
                T_vec = []
                Phi_vec = []
                Phib_vec = []
                if T_el > 0.0:
                    T_vec = [T_el + 2 * h, T_el + h, T_el - h, T_el - 2 * h]
                else:
                    T_vec = [h, 0.0]
                Phi_vec = [complex(phi_re_el, phi_im_el) for el in T_vec]
                Phib_vec = [complex(phi_re_el, -phi_im_el) for el in T_vec]
                SDen_Q.append(pnjl.thermo.gcp_quark.sdensity(T_vec, mu_el, Phi_vec, Phib_vec) + pnjl.thermo.gcp_quark.sdensity(T_vec, mu_el, Phi_vec, Phib_vec, ml = pnjl.defaults.default_ms))
                SDen_pert.append(pnjl.thermo.gcp_perturbative.sdensity(T_vec, mu_el, Phi_vec, Phib_vec, Nf = 3.0))
            (
                (Pres_pi, Pres_rho, Pres_omega, Pres_K, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, 
                 Pres_Q5, Pres_H),
                (BDen_pi, BDen_rho, BDen_omega, BDen_K, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, 
                 BDen_Q5, BDen_H),
                (SDen_pi, SDen_rho, SDen_omega, SDen_K, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, 
                 SDen_Q5, SDen_H)
            ) = clusters_c(T, mu, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(T, mu, Pres_Q, Pres_g, Pres_pert, Pres_pi, Pres_K, Pres_rho, Pres_omega, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, Pres_Q5, Pres_H, Pres_sea)])
        with open(baryon_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(T, mu, BDen_Q, BDen_g, BDen_pert, BDen_pi, BDen_K, BDen_rho, BDen_omega, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, BDen_Q5, BDen_H, BDen_sea)])
        with open(entropy_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(T, mu, SDen_Q, SDen_g, SDen_pert, SDen_pi, SDen_K, SDen_rho, SDen_omega, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, SDen_Q5, SDen_H, SDen_sea)])
    else:
        T, mu              = utils.data_collect(0, 1, pressure_file)
        Pres_Q, Pres_g     = utils.data_collect(2, 3, pressure_file)
        Pres_pert, Pres_pi = utils.data_collect(4, 5, pressure_file)
        Pres_K, Pres_rho   = utils.data_collect(6, 7, pressure_file)
        Pres_omega, Pres_D = utils.data_collect(8, 9, pressure_file)
        Pres_N, Pres_T     = utils.data_collect(10, 11, pressure_file)
        Pres_F, Pres_P     = utils.data_collect(12, 13, pressure_file)
        Pres_Q5, Pres_H    = utils.data_collect(14, 15, pressure_file)
        Pres_sea, _        = utils.data_collect(16, 16, pressure_file)
        BDen_Q, BDen_g     = utils.data_collect(2, 3, baryon_file)
        BDen_pert, BDen_pi = utils.data_collect(4, 5, baryon_file)
        BDen_K, BDen_rho   = utils.data_collect(6, 7, baryon_file)
        BDen_omega, BDen_D = utils.data_collect(8, 9, baryon_file)
        BDen_N, BDen_T     = utils.data_collect(10, 11, baryon_file)
        BDen_F, BDen_P     = utils.data_collect(12, 13, baryon_file)
        BDen_Q5, BDen_H    = utils.data_collect(14, 15, baryon_file)
        BDen_sea, _        = utils.data_collect(16, 16, baryon_file)
        SDen_Q, SDen_g     = utils.data_collect(2, 3, entropy_file)
        SDen_pert, SDen_pi = utils.data_collect(4, 5, entropy_file)
        SDen_K, SDen_rho   = utils.data_collect(6, 7, entropy_file)
        SDen_omega, SDen_D = utils.data_collect(8, 9, entropy_file)
        SDen_N, SDen_T     = utils.data_collect(10, 11, entropy_file)
        SDen_F, SDen_P     = utils.data_collect(12, 13, entropy_file)
        SDen_Q5, SDen_H    = utils.data_collect(14, 15, entropy_file)
        SDen_sea, _        = utils.data_collect(16, 16, entropy_file)

    total_s = [sum(el) for el in zip(SDen_Q, SDen_g, SDen_pert, SDen_pi, SDen_K, SDen_rho, SDen_omega, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, SDen_Q5, SDen_H, SDen_sea)]
    total_b = [sum(el) for el in zip(BDen_Q, BDen_g, BDen_pert, BDen_pi, BDen_K, BDen_rho, BDen_omega, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, BDen_Q5, BDen_H, BDen_sea)]

    entropy_per_baryon = [el1 / el2 for el1, el2 in zip(total_s, total_b)]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([0., 2000., min(entropy_per_baryon[1:]), max(entropy_per_baryon[1:])])

    ax1.plot(T[1:], entropy_per_baryon[1:], '-', c = 'blue')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'entropy per baryon', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def phase_shift_test():

    matplotlib.use('Agg')

    T = numpy.linspace(1.0, 400.0, 200)

    for T_el in T:

        try:
            fig = matplotlib.pyplot.figure(num = 1, figsize = (11.0, 5))
            ax = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            fig.subplots_adjust(wspace = 0.3)

            paths = glob.glob("D:\\EoS\\epja\\bulkdat\\cont_T_" + str(T_el).replace('.', 'p') + "_*.dat")
            colormap = matplotlib.pyplot.cm.rainbow(numpy.linspace(0, 1, len(paths)))

            min_M = 1e300
            max_M = 0.0

            min_integ = 1e300
            max_integ = 0.0

            for path, color in zip(paths, colormap):
                p, M                = utils.data_collect(2, 3, path)
                delta_i, integ_i    = utils.data_collect(4, 5, path)
                flag, _             = utils.data_collect(6, 6, path)
                
                for M_el in M:
                    if M_el < min_M:
                        min_M = M_el
                    if M_el > max_M:
                        max_M = M_el

                for iel in integ_i:
                    if iel < min_integ:
                        min_integ = iel
                    if iel > max_integ:
                        max_integ = iel

                M, delta_i, integ_i = zip(*sorted(zip(M, delta_i, integ_i)))

                lab = "p=" + str(p[0])
                ax.plot(M, delta_i, c = color, label = lab)
                ax2.plot(M, integ_i, c = color, label = lab)

            ax.axis([min_M, max_M, -0.1, 1.1])
            ax2.axis([min_M, max_M, min_integ, max_integ])
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(12)
            ax.set_xlabel(r'M [MeV]', fontsize = 16)
            ax.set_ylabel(r'$\mathrm{\delta_i}$', fontsize = 16)
            #ax.legend()

            for tick in ax2.xaxis.get_major_ticks():
                tick.label.set_fontsize(12) 
            for tick in ax2.yaxis.get_major_ticks():
                tick.label.set_fontsize(12)
            ax2.set_xlabel(r'M [MeV]', fontsize = 16)
            ax2.set_ylabel(r'integ', fontsize = 16)
            #ax2.legend()
    
            #fig.tight_layout()
            #matplotlib.pyplot.show()
            fig.savefig("D:\\EoS\\epja\\testfig\\cont_T_" + str(T_el).replace('.', 'p') + ".png")
            matplotlib.pyplot.close(fig)
            time.sleep(5)
        except RuntimeError as e:
            print("Jest błąd!")
            print(e)
            print("Jak żyć?")
            input()
    #end"""

""" def main():

    import utils, pickle

    @utils.cached
    def func1(a, b):
        print("I used func1!")
        return a+b

    @utils.cached
    def func2(a, b):
        print("I used func2!")
        return a-b

    print(func1(1,1))
    print(func1(1,1))
    print(func2(1,1))
    print(func2(1,1))

    with (
        open(".cache/__main__.func1.cache", "rb") as file1,
        open(".cache/__main__.func2.cache", "rb") as file2
    ):
        print(pickle.load(file1))
        print(pickle.load(file2))

    import timeit, utils, pickle

    #utils.flush_cache()

    print("Tc time", timeit.timeit("Tc(2.0)", "from pnjl.thermo.gcp_sea_lattice import Tc", number=1000000))
    print("Delta_ls time", timeit.timeit("Delta_ls(1.0, 2.0)", "from pnjl.thermo.gcp_sea_lattice import Delta_ls", number=1000000))
    print("Pressure time", timeit.timeit("pressure(2.0, 2.0, 'l', no_sea = False)", "from pnjl.thermo.gcp_sea_lattice import pressure", number=100000))

    #main()

    with (
        open(".cache/pnjl.thermo.distributions.f_boson_triplet.cache", "rb") as file1,
        open(".cache/pnjl.thermo.distributions.f_fermion_triplet.cache", "rb") as file2,
        open(".cache/pnjl.thermo.gcp_sea_lattice.pressure.cache", "rb") as file3
    ):
        print(pickle.load(file1))
        print(pickle.load(file2))
        print(pickle.load(file3))

    import gc, utils;
    gc.collect()
    print(len([obj for obj in gc.get_objects() if isinstance(obj, utils.cached)])) """


import numpy
import pickle

import utils


def pickle_lattice_data():

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y) = utils.data_load(
        "D:/EoS/epja/lattice_data_raw/1204_6710v2_table4_pressure_mu0_low.dat", 0, 1
    )
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y) = utils.data_load(
        "D:/EoS/epja/lattice_data_raw/1204_6710v2_table4_pressure_mu0_high.dat", 0, 1
    )
    (high_1407_6387_mu0_x, high_1407_6387_mu0_y) = utils.data_load(
        "D:/EoS/epja/lattice_data_raw/1407_6387_table1_pressure_mu0.dat", 0, 3
    )
    (low_1407_6387_mu0_x, low_1407_6387_mu0_y) = utils.data_load(
        "D:/EoS/epja/lattice_data_raw/1407_6387_table1_pressure_mu0.dat", 0, 2
    )
    (high_1309_5258_mu0_x, high_1309_5258_mu0_y) = utils.data_load(
        "D:/EoS/epja/lattice_data_raw/1309_5258_figure6_pressure_mu0_high.dat", 0, 1
    )
    (low_1309_5258_mu0_x, low_1309_5258_mu0_y) = utils.data_load(
        "D:/EoS/epja/lattice_data_raw/1309_5258_figure6_pressure_mu0_low.dat", 0, 1
    )
    (high_1710_05024_mu0_x, high_1710_05024_mu0_y) = utils.data_load(
        "D:/EoS/epja/lattice_data_raw/1710_05024_figure8_pressure_mu0_high.dat", 0, 1
    )
    (low_1710_05024_mu0_x, low_1710_05024_mu0_y) = utils.data_load(
        "D:/EoS/epja/lattice_data_raw/1710_05024_figure8_pressure_mu0_low.dat", 0, 1
    )

    (low_1204_6710v2_mu100_x, low_1204_6710v2_mu100_y) = utils.data_load(
        "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu100_low.dat", 0, 1
    )
    (high_1204_6710v2_mu100_x, high_1204_6710v2_mu100_y) = utils.data_load(
        "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu100_high.dat", 0, 1
    )
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y) = utils.data_load(
        "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu200_low.dat", 0, 1
    )
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = utils.data_load(
        "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu200_high.dat", 0, 1
    )
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y) = utils.data_load(
        "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu300_low.dat", 0, 1
    )
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = utils.data_load(
        "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu300_high.dat", 0, 1
    )
    (low_1204_6710v2_mu400_x, low_1204_6710v2_mu400_y) = utils.data_load(
        "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu400_low.dat", 0, 1
    )
    (high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y) = utils.data_load(
        "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu400_high.dat", 0, 1
    )

    borsanyi_1204_6710v2_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = numpy.array(borsanyi_1204_6710v2_mu0)
    bazavov_1407_6387_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1407_6387_mu0_x, high_1407_6387_mu0_y)]
    for x_el, y_el in zip(low_1407_6387_mu0_x[::-1], low_1407_6387_mu0_y[::-1]):
        bazavov_1407_6387_mu0.append(numpy.array([x_el, y_el]))
    bazavov_1407_6387_mu0 = numpy.array(bazavov_1407_6387_mu0)
    borsanyi_1309_5258_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1309_5258_mu0_x, high_1309_5258_mu0_y)]
    for x_el, y_el in zip(low_1309_5258_mu0_x[::-1], low_1309_5258_mu0_y[::-1]):
        borsanyi_1309_5258_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1309_5258_mu0 = numpy.array(borsanyi_1309_5258_mu0)
    bazavov_1710_05024_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1710_05024_mu0_x, high_1710_05024_mu0_y)]
    for x_el, y_el in zip(low_1710_05024_mu0_x[::-1], low_1710_05024_mu0_y[::-1]):
        bazavov_1710_05024_mu0.append(numpy.array([x_el, y_el]))
    bazavov_1710_05024_mu0 = numpy.array(bazavov_1710_05024_mu0)
    borsanyi_1204_6710v2_mu100 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu100_x, high_1204_6710v2_mu100_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu100_x[::-1], low_1204_6710v2_mu100_y[::-1]):
        borsanyi_1204_6710v2_mu100.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu100 = numpy.array(borsanyi_1204_6710v2_mu100)
    borsanyi_1204_6710v2_mu200 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu200_x[::-1], low_1204_6710v2_mu200_y[::-1]):
        borsanyi_1204_6710v2_mu200.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu200 = numpy.array(borsanyi_1204_6710v2_mu200)
    borsanyi_1204_6710v2_mu300 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu300_x[::-1], low_1204_6710v2_mu300_y[::-1]):
        borsanyi_1204_6710v2_mu300.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu300 = numpy.array(borsanyi_1204_6710v2_mu300)
    borsanyi_1204_6710v2_mu400 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu400_x[::-1], low_1204_6710v2_mu400_y[::-1]):
        borsanyi_1204_6710v2_mu400.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu400 = numpy.array(borsanyi_1204_6710v2_mu400)

    with open("D:/EoS/epja/lattice_data_pickled/bazavov_1407_6387_mu0.pickle", "wb") as file:
        pickle.dump(bazavov_1407_6387_mu0, file)
    with open("D:/EoS/epja/lattice_data_pickled/borsanyi_1309_5258_mu0.pickle", "wb") as file:
        pickle.dump(borsanyi_1309_5258_mu0, file)
    with open("D:/EoS/epja/lattice_data_pickled/bazavov_1710_05024_mu0.pickle", "wb") as file:
        pickle.dump(bazavov_1710_05024_mu0, file)
    with open("D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu0.pickle", "wb") as file:
        pickle.dump(borsanyi_1204_6710v2_mu0, file)
    with open("D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu100.pickle", "wb") as file:
        pickle.dump(borsanyi_1204_6710v2_mu100, file)
    with open("D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu200.pickle", "wb") as file:
        pickle.dump(borsanyi_1204_6710v2_mu200, file)
    with open("D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu300.pickle", "wb") as file:
        pickle.dump(borsanyi_1204_6710v2_mu300, file)
    with open("D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu400.pickle", "wb") as file:
        pickle.dump(borsanyi_1204_6710v2_mu400, file)