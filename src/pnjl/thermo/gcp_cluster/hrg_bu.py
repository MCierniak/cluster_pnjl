"""### Description
(needs updating)

### Functions
(needs updating)
"""


import math
import functools

import scipy.optimize
import scipy.integrate

import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_sigma_lattice
import pnjl.thermo.gcp_pl_polynomial


MI_OVERRIDE = {
    'pi': 140.0,
    'K': 500.0,
    'eta': 550.0,
    'rho': 775.0,
    'omega': 783.0,
    'Kstar': 895,
    'N': 940.0,
    'etastar': 960.0,
    'a0': 980.0,
    'f0': 980.0,
    'D': math.fsum([2.0*pnjl.defaults.M0, 2.0*pnjl.defaults.ML, -pnjl.defaults.B]),
    'T': math.fsum([4.0*pnjl.defaults.M0, 4.0*pnjl.defaults.ML, -3.0*pnjl.defaults.B]),
    'F': math.fsum([4.0*pnjl.defaults.M0, 4.0*pnjl.defaults.ML, -3.0*pnjl.defaults.B]),
    'P': math.fsum([5.0*pnjl.defaults.M0, 5.0*pnjl.defaults.ML, -4.0*pnjl.defaults.B]),
    'Q': math.fsum([5.0*pnjl.defaults.M0, 5.0*pnjl.defaults.ML, -4.0*pnjl.defaults.B]),
    'H': math.fsum([6.0*pnjl.defaults.M0, 6.0*pnjl.defaults.ML, -5.0*pnjl.defaults.B]),
}

DI_OVERRIDE = {
    'pi': 3.0,
    'K': 4.0,
    'eta': 1.0,
    'rho': 9.0,
    'omega': 3.0,
    'Kstar': 12.0,
    'N': 4.0,
    'etastar': 1.0,
    'a0': 3.0,
    'f0': 1.0,
    'D': 3.0,
    'T': 1.0,
    'F': 3.0,
    'P': 4.0,
    'Q': 6.0,
    'H': 3.0
}

NET_QL_OVERRIDE = {
    'pi': 0,
    'K': 1,
    'eta': 0,
    'rho': 0,
    'omega': 0,
    'Kstar': 1,
    'N': 3,
    'etastar': 1,
    'a0': -1,
    'f0': 0,
    'D': 2,
    'T': 0,
    'F': 4,
    'P': 3,
    'Q': 5,
    'H': 6
}

NET_QS_OVERRIDE = {
    'pi': 0,
    'K': -1,
    'eta': 0,
    'rho': 0,
    'omega': 0,
    'Kstar': -1,
    'N': 0,
    'etastar': -1,
    'a0': 1,
    'f0': 0,
    'D': 0,
    'T': 0,
    'F': 0,
    'P': 0,
    'Q': 0,
    'H': 0
}

NI_OVERRIDE = {
    'pi': 2.0,
    'K': 2.0,
    'eta': 2.0,
    'rho': 2.0,
    'omega': 2.0,
    'Kstar': 2.0,
    'N': 3.0,
    'etastar': 2.0,
    'a0': 4.0,
    'f0': 4.0,
    'D': 2.0,
    'T': 4.0,
    'F': 4.0,
    'P': 5.0,
    'Q': 5.0,
    'H': 6.0
}

S_OVERRIDE = {
    'pi': 0,
    'K': 1,
    'eta': 0,
    'rho': 0,
    'omega': 0,
    'Kstar': 1,
    'N': 0,
    'etastar': 1,
    'a0': 1,
    'f0': 0,
    'D': 0,
    'T': 0,
    'F': 0,
    'P': 0,
    'Q': 0,
    'H': 0
}


@functools.lru_cache(maxsize=1024)
def bu_s_boson_singlet_integrand(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:
    sigma_p = 0.0
    fp = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, a, '+')
    dfp = pnjl.thermo.distributions.dfdM_boson_singlet(p, T, mu, M, a, '+')
    if not (fp == 0.0 or dfp == 0.0):
        sigma_p = math.fsum([math.log(fp), -math.log1p(fp)])*dfp
    sigma_m = 0.0
    fm = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, a, '-')
    dfm = pnjl.thermo.distributions.dfdM_boson_singlet(p, T, mu, M, a, '-')
    if not (fm == 0.0 or dfm == 0.0):
        sigma_m = math.fsum([math.log(fm), -math.log1p(fm)])*dfm
    return -math.fsum([sigma_p, sigma_m])
    

@functools.lru_cache(maxsize=1024)
def bu_s_fermion_singlet_integrand(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:
    sigma_p = 0.0
    fp = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, a, '+')
    dfp = pnjl.thermo.distributions.dfdM_fermion_singlet(p, T, mu, M, a, '+')
    if not (fp == 0.0 or dfp == 0.0):
        sigma_p = math.fsum([math.log(fp), -math.log(math.fsum([1.0, -fp]))])*dfp
    sigma_m = 0.0
    fm = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, a, '-')
    dfm = pnjl.thermo.distributions.dfdM_fermion_singlet(p, T, mu, M, a, '-')
    if not (fm == 0.0 or dfm == 0.0):
        sigma_m = math.fsum([math.log(fm), -math.log(math.fsum([1.0, -fm]))])*dfm
    return -math.fsum([sigma_p, sigma_m])
    

@functools.lru_cache(maxsize=1024)
def bu_s_boson_triplet_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:
    energy = pnjl.thermo.distributions.En(p, M)
    dfp = pnjl.thermo.distributions.dfdM_boson_triplet(p, T, mu, phi_re, phi_im, M, a, '+').real
    sigma_p = ((energy-a*mu)/T)*dfp
    dfm = pnjl.thermo.distributions.dfdM_boson_antitriplet(p, T, mu, phi_re, phi_im, M, a, '-').real
    sigma_m = ((energy+a*mu)/T)*dfm
    return -math.fsum([sigma_p, sigma_m])
    

@functools.lru_cache(maxsize=1024)
def bu_s_boson_antitriplet_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:
    energy = pnjl.thermo.distributions.En(p, M)
    dfp = pnjl.thermo.distributions.dfdM_boson_antitriplet(p, T, mu, phi_re, phi_im, M, a, '+').real
    sigma_p = ((energy-a*mu)/T)*dfp
    dfm = pnjl.thermo.distributions.dfdM_boson_triplet(p, T, mu, phi_re, phi_im, M, a, '-').real
    sigma_m = ((energy+a*mu)/T)*dfm
    return -math.fsum([sigma_p, sigma_m])
    

@functools.lru_cache(maxsize=1024)
def bu_s_fermion_triplet_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:
    energy = pnjl.thermo.distributions.En(p, M)
    dfp = pnjl.thermo.distributions.dfdM_fermion_triplet(p, T, mu, phi_re, phi_im, M, a, '+').real
    sigma_p = ((energy-a*mu)/T)*dfp
    dfm = pnjl.thermo.distributions.dfdM_fermion_antitriplet(p, T, mu, phi_re, phi_im, M, a, '-').real
    sigma_m = ((energy+a*mu)/T)*dfm
    return -math.fsum([sigma_p, sigma_m])
    

@functools.lru_cache(maxsize=1024)
def bu_s_fermion_antitriplet_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:
    energy = pnjl.thermo.distributions.En(p, M)
    dfp = pnjl.thermo.distributions.dfdM_fermion_antitriplet(p, T, mu, phi_re, phi_im, M, a, '+').real
    sigma_p = ((energy-a*mu)/T)*dfp
    dfm = pnjl.thermo.distributions.dfdM_fermion_triplet(p, T, mu, phi_re, phi_im, M, a, '-').real
    sigma_m = ((energy+a*mu)/T)*dfm
    return -math.fsum([sigma_p, sigma_m])


bu_s_integral_hash = {
    'pi': bu_s_boson_singlet_integrand,
    'K': bu_s_boson_singlet_integrand,
    'eta': bu_s_boson_singlet_integrand,
    'rho': bu_s_boson_singlet_integrand,
    'omega': bu_s_boson_singlet_integrand,
    'Kstar': bu_s_boson_singlet_integrand,
    'N': bu_s_fermion_singlet_integrand,
    'etastar': bu_s_boson_singlet_integrand,
    'a0': bu_s_boson_singlet_integrand,
    'f0': bu_s_boson_singlet_integrand,
    'D': bu_s_boson_antitriplet_integrand_real,
    'T': bu_s_boson_singlet_integrand,
    'F': bu_s_boson_triplet_integrand_real,
    'P': bu_s_fermion_singlet_integrand,
    'Q': bu_s_fermion_antitriplet_integrand_real,
    'H': bu_s_boson_singlet_integrand
}


@functools.lru_cache(maxsize=1024)
def sdensity_bu_integral(
    p: float, T: float, mu: float, phi_re: float, phi_im: float, 
    mass: float, a: float, cluster: str
) -> float:
    
    integral, error = scipy.integrate.quad(
        bu_s_integral_hash[cluster], mass, math.inf,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def sdensity_bu(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:

    D_I = DI_OVERRIDE[cluster]
    A_I = math.fsum([
        NET_QL_OVERRIDE[cluster],
        NET_QS_OVERRIDE[cluster]
    ])
    M_I = MI_OVERRIDE[cluster]

    integral, _ = scipy.integrate.quad(
        sdensity_bu_integral, 0.0, math.inf,
        args = (T, mu, phi_re, phi_im, M_I, A_I, cluster)
    )

    return (D_I/(2.0*(math.pi**2)))*0.5*integral