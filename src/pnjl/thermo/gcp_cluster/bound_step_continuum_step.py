"""### Description
Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential and 
associated functions based on https://arxiv.org/pdf/2012.12894.pdf .

### Functions
gcp_real
    Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential 
    (real part).
gcp_imag
    Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential 
    (imaginary part).
pressure
    Generalized Beth-Uhlenbeck cluster pressure.
bdensity
    Generalized Beth-Uhlenbeck cluster baryon density.
qnumber_cumulant
    Generalized Beth-Uhlenbeck cluster quark number cumulant chi_q. Based on 
    Eq.29 of https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline 
    definition.
sdensity
    Generalized Beth-Uhlenbeck cluster entropy density.
Polyakov_loop
"""


import math
import typing
import functools

import scipy.optimize
import scipy.integrate

import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_sigma_lattice
import pnjl.thermo.gcp_pl.polynomial


MI = {
    'pi0': 135.0, 'pi': 140.0, 'K': 494.0, 'K0': 498.0, 'eta': 548.0,
    'rho': 775.0, 'omega': 783.0, 'K*(892)': 892.0, 'K*0(892)': 896.0, 'p': 938.0,
    'n': 940.0, 'etaPrime': 958.0, 'a0': 980.0, 'f0': 990.0, 'phi': 1019.0,
    'Lambda': 1116.0, 'h1': 1170.0, 'Sigma+': 1189.0, 'Sigma0': 1193.0,
    'Sigma-': 1197.0, 'b1': 1230.0, 'a1': 1230.0, 'Delta': 1232.0,
    'D1': 700.0, 'D2': 850.0, '4q1': 1300.0, '4q2': 1450.0, '4q3': 1600.0,
    '4q4': 1750.0, 'P1': 1540.0, 'P2': 1860.0, '5q1': 1600.0, '5q2': 1750.0,
    '5q3': 1900.0, 'd': 1880.0
}

DI = {
    'pi0': 1.0/2.0, 'pi': 2.0/2.0, 'K': 2.0/2.0, 'K0': 2.0/2.0, 'eta': 1.0/2.0,
    'rho': 9.0/2.0, 'omega': 3.0/2.0, 'K*(892)': 6.0/2.0, 'K*0(892)': 6.0/2.0, 'p': 2.0,
    'n': 2.0, 'etaPrime': 1.0/2.0, 'a0': 3.0/2.0, 'f0': 1.0/2.0, 'phi': 3.0/2.0,
    'Lambda': 2.0, 'h1': 3.0/2.0, 'Sigma+': 2.0, 'Sigma0': 2.0,
    'Sigma-': 2.0, 'b1': 9.0/2.0, 'a1': 9.0/2.0, 'Delta': 16.0,
    'D1': 3.0, 'D2': 6.0, '4q1': 3.0, '4q2': 6.0, '4q3': 3.0,
    '4q4': 6.0, 'P1': 16.0, 'P2': 16.0, '5q1': 12.0, '5q2': 24.0,
    '5q3': 36.0, 'd': 3.0
}

BI = {
    'pi0': 0.0, 'pi': 0.0, 'K': 0.0, 'K0': 0.0, 'eta': 0.0,
    'rho': 0.0, 'omega': 0.0, 'K*(892)': 0.0, 'K*0(892)': 0.0, 'p': 1.0,
    'n': 1.0, 'etaPrime': 0.0, 'a0': 0.0, 'f0': 0.0, 'phi': 0.0,
    'Lambda': 1.0, 'h1': 0.0, 'Sigma+': 1.0, 'Sigma0': 1.0,
    'Sigma-': 1.0, 'b1': 0.0, 'a1': 0.0, 'Delta': 1.0,
    'D1': 2.0/3.0, 'D2': 2.0/3.0, '4q1': 4.0/3.0, '4q2': 4.0/3.0, '4q3': 4.0/3.0,
    '4q4': 4.0/3.0, 'P1': 1.0, 'P2': 1.0, '5q1': 5.0/3.0, '5q2': 5.0/3.0,
    '5q3': 5.0/3.0, 'd': 2.0
}

NI = {
    'pi0': 2.0, 'pi': 2.0, 'K': 2.0, 'K0': 2.0, 'eta': 2.0,
    'rho': 2.0, 'omega': 2.0, 'K*(892)': 2.0, 'K*0(892)': 2.0, 'p': 3.0,
    'n': 3.0, 'etaPrime': 2.0, 'a0': 4.0, 'f0': 4.0, 'phi': 2.0,
    'Lambda': 3.0, 'h1': 2.0, 'Sigma+': 3.0, 'Sigma0': 3.0,
    'Sigma-': 3.0, 'b1': 4.0, 'a1': 4.0, 'Delta': 3.0,
    'D1': 2.0, 'D2': 2.0, '4q1': 4.0, '4q2': 4.0, '4q3': 4.0,
    '4q4': 4.0, 'P1': 5.0, 'P2': 5.0, '5q1': 5.0, '5q2': 5.0,
    '5q3': 5.0, 'd': 6.0
}
    
SI = {
    'pi0': 0.0, 'pi': 0.0, 'K': 1.0, 'K0': 1.0, 'eta': 1.0,
    'rho': 0.0, 'omega': 0.0, 'K*(892)': 1.0, 'K*0(892)': 1.0, 'p': 0.0,
    'n': 0.0, 'etaPrime': 1.0, 'a0': 1.0, 'f0': 0.0, 'phi': 2.0,
    'Lambda': 1.0, 'h1': 2.0, 'Sigma+': 1.0, 'Sigma0': 1.0,
    'Sigma-': 1.0, 'b1': 0.0, 'a1': 0.0, 'Delta': 0.0,
    'D1': 0.0, 'D2': 1.0, '4q1': 0.0, '4q2': 1.0, '4q3': 2.0,
    '4q4': 3.0, 'P1': 1.0, 'P2': 2.0, '5q1': 0.0, '5q2': 1.0,
    '5q3': 2.0, 'd': 0.0
}

SQRT2 = math.sqrt(2.0)


@functools.lru_cache(maxsize=1024)
def M_th(T: float, muB: float, hadron: str) -> float:

    N_I = NI[hadron]
    S_I = SI[hadron]
    M_th_i = SQRT2*math.fsum([
        math.fsum([N_I,-S_I])*pnjl.thermo.gcp_sigma_lattice.Ml(T, muB),
        S_I*pnjl.thermo.gcp_sigma_lattice.Ms(T, muB)
    ])

    return M_th_i


@functools.lru_cache(maxsize=1024)
def b_boson_singlet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    fp_i = pnjl.thermo.distributions.f_boson_singlet(p, T, muB, M_I, a, '+')
    fm_i = pnjl.thermo.distributions.f_boson_singlet(p, T, muB, M_I, a, '-')
    integ_i = math.fsum([fp_i, -fm_i])

    fp_th = pnjl.thermo.distributions.f_boson_singlet(p, T, muB, M_TH_I, a, '+')
    fm_th = pnjl.thermo.distributions.f_boson_singlet(p, T, muB, M_TH_I, a, '-')
    integ_th = math.fsum([fp_th, -fm_th])

    return (p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def b_fermion_singlet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    fp_i = pnjl.thermo.distributions.f_fermion_singlet(p, T, muB, M_I, a, '+')
    fm_i = pnjl.thermo.distributions.f_fermion_singlet(p, T, muB, M_I, a, '-')
    integ_i = math.fsum([fp_i, -fm_i])

    fp_th = pnjl.thermo.distributions.f_fermion_singlet(p, T, muB, M_TH_I, a, '+')
    fm_th = pnjl.thermo.distributions.f_fermion_singlet(p, T, muB, M_TH_I, a, '-')
    integ_th = math.fsum([fp_th, -fm_th])

    return (p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def b_boson_triplet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    fp_i = pnjl.thermo.distributions.f_boson_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    fm_i = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    integ_i = math.fsum([fp_i, -fm_i])

    fp_th = pnjl.thermo.distributions.f_boson_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    fm_th = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    integ_th = math.fsum([fp_th, -fm_th])

    return (p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def b_boson_antitriplet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    fp_i = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    fm_i = pnjl.thermo.distributions.f_boson_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    integ_i = math.fsum([fp_i, -fm_i])

    fp_th = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    fm_th = pnjl.thermo.distributions.f_boson_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    integ_th = math.fsum([fp_th, -fm_th])

    return (p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def b_fermion_triplet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    fp_i = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    fm_i = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    integ_i = math.fsum([fp_i, -fm_i])

    fp_th = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    fm_th = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    integ_th = math.fsum([fp_th, -fm_th])

    return (p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def b_fermion_antitriplet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    fp_i = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    fm_i = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    integ_i = math.fsum([fp_i, -fm_i])

    fp_th = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    fm_th = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    integ_th = math.fsum([fp_th, -fm_th])

    return (p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def s_boson_singlet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    fp_i = pnjl.thermo.distributions.f_boson_singlet(p, T, muB, M_I, a, '+')
    fm_i = pnjl.thermo.distributions.f_boson_singlet(p, T, muB, M_I, a, '-')
    sigma_p_i = 0.0
    if fp_i != 0.0:
        sigma_p_i = math.fsum([fp_i*math.log(fp_i), -math.fsum([1.0, fp_i])*math.log1p(fp_i)])
    sigma_m_i = 0.0
    if fm_i != 0.0:
        sigma_m_i = math.fsum([fm_i*math.log(fm_i), -math.fsum([1.0, fm_i])*math.log1p(fm_i)])
    integ_i = math.fsum([sigma_p_i, sigma_m_i])

    fp_th = pnjl.thermo.distributions.f_boson_singlet(p, T, muB, M_TH_I, a, '+')
    fm_th = pnjl.thermo.distributions.f_boson_singlet(p, T, muB, M_TH_I, a, '-')
    sigma_p_th = 0.0
    if fp_th != 0.0:
        sigma_p_th = math.fsum([fp_th*math.log(fp_th), -math.fsum([1.0, fp_th])*math.log1p(fp_th)])
    sigma_m_th = 0.0
    if fm_th != 0.0:
        sigma_m_th = math.fsum([fm_th*math.log(fm_th), -math.fsum([1.0, fm_th])*math.log1p(fm_th)])
    integ_th = math.fsum([sigma_p_th, sigma_m_th])

    return (p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def s_fermion_singlet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    fp_i = pnjl.thermo.distributions.f_fermion_singlet(p, T, muB, M_I, a, '+')
    fm_i = pnjl.thermo.distributions.f_fermion_singlet(p, T, muB, M_I, a, '-')
    sigma_p_i = 0.0
    if fp_i != 0.0:
        sigma_p_i = math.fsum([fp_i*math.log(fp_i), math.fsum([1.0, -fp_i])*math.log1p(-fp_i)])
    sigma_m_i = 0.0
    if fm_i != 0.0:
        sigma_m_i = math.fsum([fm_i*math.log(fm_i), math.fsum([1.0, -fm_i])*math.log1p(-fm_i)])
    integ_i = math.fsum([sigma_p_i, sigma_m_i])

    fp_th = pnjl.thermo.distributions.f_fermion_singlet(p, T, muB, M_TH_I, a, '+')
    fm_th = pnjl.thermo.distributions.f_fermion_singlet(p, T, muB, M_TH_I, a, '-')
    sigma_p_th = 0.0
    if fp_th != 0.0:
        sigma_p_th = math.fsum([fp_th*math.log(fp_th), math.fsum([1.0, -fp_th])*math.log1p(-fp_th)])
    sigma_m_th = 0.0
    if fm_th != 0.0:
        sigma_m_th = math.fsum([fm_th*math.log(fm_th), math.fsum([1.0, -fm_th])*math.log1p(-fm_th)])
    integ_th = math.fsum([sigma_p_th, sigma_m_th])

    return (p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def s_boson_triplet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    En_i = pnjl.thermo.distributions.En(p, M_I)
    fp_i = pnjl.thermo.distributions.f_boson_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    fm_i = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    zp_i = pnjl.thermo.distributions.z_boson_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    zm_i = pnjl.thermo.distributions.z_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    sigma_p_i = ((En_i-a*muB)/T)*fp_i - zp_i/3.0
    sigma_m_i = ((En_i+a*muB)/T)*fm_i - zm_i/3.0
    integ_i = math.fsum([sigma_p_i, sigma_m_i])

    En_th = pnjl.thermo.distributions.En(p, M_TH_I)
    fp_th = pnjl.thermo.distributions.f_boson_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    fm_th = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    zp_th = pnjl.thermo.distributions.z_boson_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    zm_th = pnjl.thermo.distributions.z_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    sigma_p_th = ((En_th-a*muB)/T)*fp_th - zp_th/3.0
    sigma_m_th = ((En_th+a*muB)/T)*fm_th - zm_th/3.0
    integ_th = math.fsum([sigma_p_th, sigma_m_th])

    return -(p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def s_boson_antitriplet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    En_i = pnjl.thermo.distributions.En(p, M_I)
    fp_i = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    fm_i = pnjl.thermo.distributions.f_boson_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    zp_i = pnjl.thermo.distributions.z_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    zm_i = pnjl.thermo.distributions.z_boson_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    sigma_p_i = ((En_i-a*muB)/T)*fp_i - zp_i/3.0
    sigma_m_i = ((En_i+a*muB)/T)*fm_i - zm_i/3.0
    integ_i = math.fsum([sigma_p_i, sigma_m_i])

    En_th = pnjl.thermo.distributions.En(p, M_TH_I)
    fp_th = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    fm_th = pnjl.thermo.distributions.f_boson_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    zp_th = pnjl.thermo.distributions.z_boson_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    zm_th = pnjl.thermo.distributions.z_boson_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    sigma_p_th = ((En_th-a*muB)/T)*fp_th - zp_th/3.0
    sigma_m_th = ((En_th+a*muB)/T)*fm_th - zm_th/3.0
    integ_th = math.fsum([sigma_p_th, sigma_m_th])

    return -(p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def s_fermion_triplet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    En_i = pnjl.thermo.distributions.En(p, M_I)
    fp_i = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    fm_i = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    zp_i = pnjl.thermo.distributions.z_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    zm_i = pnjl.thermo.distributions.z_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    sigma_p_i = ((En_i-a*muB)/T)*fp_i + zp_i/3.0
    sigma_m_i = ((En_i+a*muB)/T)*fm_i + zm_i/3.0
    integ_i = math.fsum([sigma_p_i, sigma_m_i])

    En_th = pnjl.thermo.distributions.En(p, M_TH_I)
    fp_th = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    fm_th = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    zp_th = pnjl.thermo.distributions.z_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    zm_th = pnjl.thermo.distributions.z_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    sigma_p_th = ((En_th-a*muB)/T)*fp_th + zp_th/3.0
    sigma_m_th = ((En_th+a*muB)/T)*fm_th + zm_th/3.0
    integ_th = math.fsum([sigma_p_th, sigma_m_th])

    return -(p**2)*math.fsum([integ_i, -integ_th])


@functools.lru_cache(maxsize=1024)
def s_fermion_antitriplet_integrand(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: float
) -> float:

    En_i = pnjl.thermo.distributions.En(p, M_I)
    fp_i = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    fm_i = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    zp_i = pnjl.thermo.distributions.z_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_I, a, '+'
    ).real
    zm_i = pnjl.thermo.distributions.z_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_I, a, '-'
    ).real
    sigma_p_i = ((En_i-a*muB)/T)*fp_i + zp_i/3.0
    sigma_m_i = ((En_i+a*muB)/T)*fm_i + zm_i/3.0
    integ_i = math.fsum([sigma_p_i, sigma_m_i])

    En_th = pnjl.thermo.distributions.En(p, M_TH_I)
    fp_th = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    fm_th = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    zp_th = pnjl.thermo.distributions.z_fermion_antitriplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '+'
    ).real
    zm_th = pnjl.thermo.distributions.z_fermion_triplet(
        p, T, muB, phi_re, phi_im, M_TH_I, a, '-'
    ).real
    sigma_p_th = ((En_th-a*muB)/T)*fp_th + zp_th/3.0
    sigma_m_th = ((En_th+a*muB)/T)*fm_th + zm_th/3.0
    integ_th = math.fsum([sigma_p_th, sigma_m_th])

    return -(p**2)*math.fsum([integ_i, -integ_th])


b_hash = {
    'pi0': b_boson_singlet_integrand, 'pi': b_boson_singlet_integrand,
    'K': b_boson_singlet_integrand, 'K0': b_boson_singlet_integrand,
    'eta': b_boson_singlet_integrand, 'rho': b_boson_singlet_integrand,
    'omega': b_boson_singlet_integrand, 'K*(892)': b_boson_singlet_integrand,
    'K*0(892)': b_boson_singlet_integrand, 'p': b_fermion_singlet_integrand,
    'n': b_fermion_singlet_integrand, 'etaPrime': b_boson_singlet_integrand,
    'a0': b_boson_singlet_integrand, 'f0': b_boson_singlet_integrand,
    'phi': b_boson_singlet_integrand, 'Lambda': b_fermion_singlet_integrand,
    'h1': b_boson_singlet_integrand, 'Sigma+': b_fermion_singlet_integrand,
    'Sigma0': b_fermion_singlet_integrand, 'Sigma-': b_fermion_singlet_integrand,
    'b1': b_boson_singlet_integrand, 'a1': b_boson_singlet_integrand,
    'Delta': b_fermion_singlet_integrand, 'D1': b_boson_antitriplet_integrand,
    'D2': b_boson_antitriplet_integrand, '4q1': b_boson_triplet_integrand,
    '4q2': b_boson_triplet_integrand, '4q3': b_boson_triplet_integrand,
    '4q4': b_boson_triplet_integrand, 'P1': b_fermion_singlet_integrand,
    'P2': b_fermion_singlet_integrand, '5q1': b_fermion_antitriplet_integrand,
    '5q2': b_fermion_antitriplet_integrand, '5q3': b_fermion_antitriplet_integrand,
    'd': b_boson_singlet_integrand
}

s_hash = {
    'pi0': s_boson_singlet_integrand, 'pi': s_boson_singlet_integrand,
    'K': s_boson_singlet_integrand, 'K0': s_boson_singlet_integrand,
    'eta': s_boson_singlet_integrand, 'rho': s_boson_singlet_integrand,
    'omega': s_boson_singlet_integrand, 'K*(892)': s_boson_singlet_integrand,
    'K*0(892)': s_boson_singlet_integrand, 'p': s_fermion_singlet_integrand,
    'n': s_fermion_singlet_integrand, 'etaPrime': s_boson_singlet_integrand,
    'a0': s_boson_singlet_integrand, 'f0': s_boson_singlet_integrand,
    'phi': s_boson_singlet_integrand, 'Lambda': s_fermion_singlet_integrand,
    'h1': s_boson_singlet_integrand, 'Sigma+': s_fermion_singlet_integrand,
    'Sigma0': s_fermion_singlet_integrand, 'Sigma-': s_fermion_singlet_integrand,
    'b1': s_boson_singlet_integrand, 'a1': s_boson_singlet_integrand,
    'Delta': s_fermion_singlet_integrand, 'D1': s_boson_antitriplet_integrand,
    'D2': s_boson_antitriplet_integrand, '4q1': s_boson_triplet_integrand,
    '4q2': s_boson_triplet_integrand, '4q3': s_boson_triplet_integrand,
    '4q4': s_boson_triplet_integrand, 'P1': s_fermion_singlet_integrand,
    'P2': s_fermion_singlet_integrand, '5q1': s_fermion_antitriplet_integrand,
    '5q2': s_fermion_antitriplet_integrand, '5q3': s_fermion_antitriplet_integrand,
    'd': s_boson_singlet_integrand
}


@functools.lru_cache(maxsize=1024)
def bdensity(
    T: float, muB: float, phi_re: float, phi_im: float, hadron: str
) -> float:
    M_I = MI[hadron]
    D_I = DI[hadron]
    A_I = BI[hadron]
    if A_I == 0.0:
        return 0.0
    else:
        M_th_i = math.sqrt(2.0)*M_th(T, muB, hadron)
        integral = 0.0
        if M_th_i > M_I:
            integral, error = scipy.integrate.quad(
                b_hash[hadron], 0.0, math.inf,
                args = (T, muB, phi_re, phi_im, M_I, M_th_i, A_I)
            )
        return (A_I*D_I/(2.0*(math.pi**2)))*integral


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant(
    rank: int, T: float, muB: float, phi_re: float, phi_im: float, hadron: str
) -> float:
    h = 1e-2
    M_I = MI[hadron]
    M_th_i_min = M_th(T, math.fsum([muB, 2*h]), hadron)
    M_th_i_max = M_th(T, math.fsum([muB, -2*h]), hadron)
    if M_th_i_min < M_I and M_th_i_max < M_I:
        return 0.0
    else:
        if rank == 1:
            return 3.0 * bdensity(T, muB, phi_re, phi_im, hadron)
        else:
            if math.fsum([muB, -2*h]) > 0.0:
                muB_vec = [
                    math.fsum([muB, 2*h]), math.fsum([muB, h]),
                    math.fsum([muB, -h]), math.fsum([muB, -2*h])
                ]
                deriv_coef = [
                    -1.0/(12.0*h), 8.0/(12.0*h),
                    -8.0/(12.0*h), 1.0/(12.0*h)
                ]
                phi_vec = [
                    tuple([phi_re, phi_im])
                    for _ in muB_vec
                ]
                out_vec = [
                    coef*qnumber_cumulant(
                        rank-1, T, muB_el, phi_el[0], phi_el[1], hadron)
                    for muB_el, coef, phi_el in zip(muB_vec, deriv_coef, phi_vec)
                ]
                return math.fsum(out_vec)
            else:
                new_muB = math.fsum([muB, h])
                new_phi_re, new_phi_im = phi_re, phi_im
                return qnumber_cumulant(
                    rank, T, new_muB, new_phi_re, new_phi_im, hadron
                )


@functools.lru_cache(maxsize=1024)
def sdensity(
    T: float, muB: float, phi_re: float, phi_im: float, hadron: str
) -> float:
    M_I = MI[hadron]
    D_I = DI[hadron]
    A_I = BI[hadron]
    M_th_i = math.sqrt(2.0)*M_th(T, muB, hadron)
    integral = 0.0
    if M_th_i > M_I:
        integral, error = scipy.integrate.quad(
            s_hash[hadron], 0.0, math.inf,
            args = (T, muB, phi_re, phi_im, M_I, M_th_i, A_I)
        )
    return -(D_I/(2.0*(math.pi**2)))*integral


@functools.lru_cache(maxsize=1024)
def sdensity_multi(
    T: float, muB: float, phi_re: float, phi_im: float, hadrons="all"
):
    partial = list()
    if hadrons == "all":
        for hadron in MI:
            partial.append(sdensity(T, muB, phi_re, phi_im, hadron))
    else:
        for hadron in hadrons:
            partial.append(sdensity(T, muB, phi_re, phi_im, hadron))
    return math.fsum(partial), (*partial,)


@functools.lru_cache(maxsize=1024)
def bdensity_multi(
    T: float, muB: float, phi_re: float, phi_im: float, hadrons="all"
):
    partial = list()
    if hadrons == "all":
        for hadron in MI:
            partial.append(bdensity(T, muB, phi_re, phi_im, hadron))
    else:
        for hadron in hadrons:
            partial.append(bdensity(T, muB, phi_re, phi_im, hadron))
    return math.fsum(partial), (*partial,)