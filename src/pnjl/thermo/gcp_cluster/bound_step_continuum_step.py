"""### Description
Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential and 
associated functions based on https://arxiv.org/pdf/2012.12894.pdf .

### Functions
"""


import math

import scipy.integrate

import utils
import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sigma_lattice


utils.verify_checksum()


@utils.cached
def gcp_boson_singlet_inner_integrand(
    M: float, p: float, T: float, mu: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, a, '+')
    fm = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, a, '-')

    return (M/En)*math.fsum([fp, fm])


@utils.cached
def gcp_fermion_singlet_inner_integrand(
    M: float, p: float, T: float, mu: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, a, '+')
    fm = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, a, '-')

    return (M/En)*math.fsum([fp, fm])


@utils.cached
def gcp_boson_singlet_integrand(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_singlet_inner_integrand, M_I, M_TH_I, args = (p, T, mu, a)
    )

    return (p**2)*integral


@utils.cached
def gcp_fermion_singlet_integrand(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_singlet_inner_integrand, M_I, M_TH_I, args = (p, T, mu, a)
    )

    return (p**2)*integral


@utils.cached
def gcp_boson_triplet_integrand(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_triplet_inner_integrand, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@utils.cached
def gcp_boson_antitriplet_integrand(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_antitriplet_inner_integrand, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@utils.cached
def gcp_fermion_triplet_integrand(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_triplet_inner_integrand, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@utils.cached
def gcp_fermion_antitriplet_integrand(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_antitriplet_inner_integrand, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral








@utils.cached
def gcp_a0(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:

    M_I = pnjl.defaults.MI[cluster]
    N_I = pnjl.defaults.NI[cluster]
    S_I = pnjl.defaults.S[cluster]
    M_th_i = math.fsum([
        math.fsum([N_I,-S_I])*pnjl.thermo.gcp_sigma_lattice.Ml(T, mu),
        S_I*pnjl.thermo.gcp_sigma_lattice.Ms(T, mu)
    ])
    D_I = pnjl.defaults.DI[cluster]

    integral = 0.0
    if M_th_i > M_I:
        integral, error = scipy.integrate.quad(
            gcp_a0_integrand, 0.0, math.inf, args = (T, mu, M_I, M_th_i)
        )

    return -((D_I*3.0)/(2.0*(math.pi**2)))*integral


def gcp_real_a1_bm1(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):
    
    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, key2):
        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, 0.0, _m, 1.0, 1.0, **key2)
        return (_m / pnjl.aux_functions.En(_p, _m)) * pnjl.thermo.distributions.f_boson_singlet(**yp)
    def integrand(p, _T, _M, _Mth, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand, _M, _Mth, args = (p, _T, key))
        return (p ** 2) * inner_int

    integral = 0.0
    if thmass > bmass:
        integral, _ = scipy.integrate.quad(integrand, 0.0, math.inf, args = (T, bmass, thmass, kwargs))

    return -(d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a1_bm1(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):
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
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
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
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
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

    return -(d / (2.0 * (math.pi ** 2))) * integral
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
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
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
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
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

    return -(d / (2.0 * (math.pi ** 2))) * integral
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

    return -(d / (2.0 * (math.pi ** 2))) * integral
def gcp_real_a6(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand(_m, _p, _T2, _mu2, key2):
        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)
        fpe = pnjl.thermo.distributions.f_boson_singlet(**yp)
        fme = pnjl.thermo.distributions.f_boson_singlet(**ym)
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

def gcp_real(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, **kwargs) -> float:
    #
    return {(0, 0) : gcp_real_a0, (2, 0) : gcp_real_a2, (3, 0) : gcp_real_a3, (4, 0) : gcp_real_a4, (5, 0) : gcp_real_a5, (6, 0) : gcp_real_a6, (1, -1) : gcp_real_a1_bm1}[a, b](T, mu, Phi, Phib, bmass, thmass, d)
def gcp_imag(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, **kwargs) -> float:
    #
    return {(0, 0) : gcp_imag_a0, (2, 0) : gcp_imag_a2, (3, 0) : gcp_imag_a3, (4, 0) : gcp_imag_a4, (5, 0) : gcp_imag_a5, (6, 0) : gcp_imag_a6, (1, -1) : gcp_imag_a1_bm1}[a, b](T, mu, Phi, Phib, bmass, thmass, d)

#Extensive thermodynamic properties

def pressure(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, **kwargs):
    #
    return -gcp_real(T, mu, Phi, Phib, bmass, thmass, a, b, d, **kwargs)