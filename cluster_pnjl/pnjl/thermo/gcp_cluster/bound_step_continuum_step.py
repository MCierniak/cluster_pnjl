import scipy.integrate
import math

import pnjl.thermo.distributions
import pnjl.aux_functions

#Mott-hadron resonance gas contribution to the GCP from https://arxiv.org/pdf/2012.12894.pdf

def gcp_real_a0(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, key2):
        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, 0.0, _m, 0.0, 1.0, **key2)
        return (_m / pnjl.aux_functions.En(_p, _m)) * pnjl.thermo.distributions.f_baryon_singlet(**yp)
    def integrand(p, _T, _M, _Mth, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand, _M, _Mth, args = (p, _T, key))
        return (p ** 2) * inner_int

    integral = 0.0
    if thmass > bmass:
        integral, _ = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, bmass, thmass, kwargs))

    return -(d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a0(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):
    #
    return 0.0
def gcp_real_a2(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, _mu2, _Phi2, _Phib2, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme)
    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand, _M, _Mth, args = (p, _T, _mu, _Phi, _Phib, key))
        return (p ** 2) * inner_int

    integral = 0.0
    if thmass > bmass:
        integral, _ = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, kwargs))

    return -(d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a2(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, _mu2, _Phi2, _Phib2, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme)
    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand, _M, _Mth, args = (p, _T, _mu, _Phi, _Phib, key))
        return (p ** 2) * inner_int

    integral = 0.0
    if thmass > bmass:
        integral, _ = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, bmass, thmass, kwargs))

    return -(d / (2.0 * (math.pi ** 2))) * integral
def gcp_real_a3(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, _mu2, key2):
        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 3.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 3.0, 1.0, **key2)
        fpe = pnjl.thermo.distributions.f_fermion_singlet(**yp)
        fme = pnjl.thermo.distributions.f_fermion_singlet(**ym)
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme)
    def integrand(p, _T, _mu, _M, _Mth, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand, _M, _Mth, args = (p, _T, _mu, key))
        return (p ** 2) * inner_int

    integral = 0.0
    if thmass > bmass:
        integral, _ = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, mu, bmass, thmass, kwargs))

    return (d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a3(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):
    #
    return 0.0
def gcp_real_a4(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, _mu2, _Phi2, _Phib2, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme)
    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand, _M, _Mth, args = (p, _T, _mu, _Phi, _Phib, key))
        return (p ** 2) * inner_int

    integral = 0.0
    if thmass > bmass:
        integral, _ = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, kwargs))

    return -(d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a4(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, _mu2, _Phi2, _Phib2, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)
        ym3 = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme)
    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand, _M, _Mth, args = (p, _T, _mu, _Phi, _Phib, key))
        return (p ** 2) * inner_int

    integral = 0.0
    if thmass > bmass:
        integral, _ = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, kwargs))

    return -(d / (2.0 * (math.pi ** 2))) * integral
def gcp_real_a5(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, _mu2, _Phi2, _Phib2, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"]  = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme)
    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand, _M, _Mth, args = (p, _T, _mu, _Phi, _Phib, key))
        return (p ** 2) * inner_int

    integral = 0.0
    if thmass > bmass:
        integral, _ = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, kwargs))

    return (d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a5(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, _mu2, _Phi2, _Phib2, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme)
    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand, _M, _Mth, args = (p, _T, _mu, _Phi, _Phib, key))
        return (p ** 2) * inner_int

    integral = 0.0
    if thmass > bmass:
        integral, _ = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, kwargs))

    return (d / (2.0 * (math.pi ** 2))) * integral
def gcp_real_a6(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, _mu2, key2):
        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)
        fpe = pnjl.thermo.distributions.f_baryon_singlet(**yp)
        fme = pnjl.thermo.distributions.f_baryon_singlet(**ym)
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme)
    def integrand(p, _T, _mu, _M, _Mth, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand, _M, _Mth, args = (p, _T, _mu, key))
        return (p ** 2) * inner_int

    integral = 0.0
    if thmass > bmass:
        integral, _ = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, mu, bmass, thmass, kwargs))

    return -(d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a6(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):
    #
    return 0.0

#Grandcanonical potential (MHRG Beth-Uhlenbeck part)

def gcp_real(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, d : float, **kwargs) -> float:
    #
    return {0 : gcp_real_a0, 2 : gcp_real_a2, 3 : gcp_real_a3, 4 : gcp_real_a4, 5 : gcp_real_a5, 6 : gcp_real_a6}[a](T, mu, Phi, Phib, bmass, thmass, d)
def gcp_imag(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, d : float, **kwargs) -> float:
    #
    return {0 : gcp_imag_a0, 2 : gcp_imag_a2, 3 : gcp_imag_a3, 4 : gcp_imag_a4, 5 : gcp_imag_a5, 6 : gcp_imag_a6}[a](T, mu, Phi, Phib, bmass, thmass, d)

#Extensive thermodynamic properties

def pressure(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, d : float, **kwargs):
    #
    return -gcp_real(T, mu, Phi, Phib, bmass, thmass, a, d, **kwargs)