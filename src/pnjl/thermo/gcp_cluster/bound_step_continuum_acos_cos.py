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
import numpy
import functools

import scipy.optimize
import scipy.integrate

import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_sigma_lattice
import pnjl.thermo.gcp_pl.polynomial
import pnjl.thermo.gcp_cluster.bound_step_continuum_step


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

TC02 = pnjl.defaults.TC0**2

TC0DT = pnjl.defaults.TC0*pnjl.defaults.DELTA_T

SQRT2 = math.sqrt(2.0)

MSC_SLOPE = 15.0

LAMBDA = 150.0

CONT_POW_M = 2


@functools.lru_cache(maxsize=1024)
def M_th(T: float, muB: float, cluster: str) -> float:
    N_I = NI[cluster]
    S_I = SI[cluster]
    M_th_i = SQRT2*math.fsum([
        math.fsum([N_I,-S_I])*pnjl.thermo.gcp_sigma_lattice.Ml(T, muB/3.0),
        S_I*pnjl.thermo.gcp_sigma_lattice.Ms(T, muB/3.0)
    ])
    return M_th_i
    

@functools.lru_cache(maxsize=1024)
def T_Mott(muB: float, cluster: str):
    kappa = pnjl.defaults.KAPPA
    dT = pnjl.defaults.DELTA_T
    Tc0 = pnjl.defaults.TC0
    M0 = pnjl.defaults.M0
    ml = pnjl.defaults.ML
    ms = pnjl.defaults.MS
    N_I = NI[cluster]
    S_I = SI[cluster]
    M_I = MI[cluster]
    atanh_inner = math.fsum([
        -2.0*M_I, SQRT2*M0*N_I, SQRT2*2.0*ml*N_I, -SQRT2*2.0*ml*S_I, SQRT2*2.0*ms*S_I
    ])/(M0*N_I*SQRT2)
    if math.fabs(atanh_inner) > 1.0: #|atanh_inner| is > 1 when a cluster has no bound state
        return pnjl.thermo.gcp_sigma_lattice.Tc(muB/3.0)
    else:
        return math.fsum([Tc0, -kappa*(muB**2)/Tc0, dT*math.atanh(atanh_inner)])


@functools.lru_cache(maxsize=1024)
def T_Mott2(muB: float, cluster: str):
    kappa = pnjl.defaults.KAPPA
    dT = pnjl.defaults.DELTA_T
    Tc0 = pnjl.defaults.TC0
    M0 = pnjl.defaults.M0
    ml = pnjl.defaults.ML
    ms = pnjl.defaults.MS
    N_I = NI[cluster]
    S_I = SI[cluster]
    M_I = MI[cluster]
    TM = T_Mott(muB, cluster)
    suma = math.fsum(
        [
            -2.0*M_I, 2.0*LAMBDA*N_I, M0*N_I*SQRT2, 2.0*ml*N_I*SQRT2, -2.0*ml*S_I*SQRT2,
            2.0*ms*S_I*SQRT2, 2.0*MSC_SLOPE*TM, M0*N_I*SQRT2*math.tanh(Tc0/dT)
        ]
    )
    return suma/(2.0*MSC_SLOPE)


@functools.lru_cache(maxsize=1024)
def bound_factor(M: float, T: float, muB: float, cluster: str) -> float:

    H_ZERO = pnjl.defaults.HEAVISIDE_ZERO

    Mi = MI[cluster]
    M_th_i = M_th(T, muB, cluster)

    if M_th_i >= Mi and False:
        return  numpy.heaviside(math.fsum([M**2 , -Mi**2]), H_ZERO) \
                -numpy.heaviside(math.fsum([M**2 , -M_th_i**2]), H_ZERO)
    else:
        return 0.0
    

@functools.lru_cache(maxsize=1024)
def bound_factor2(M: float, T: float, muB: float, cluster: str) -> float:

    H_ZERO = pnjl.defaults.HEAVISIDE_ZERO

    Mi = MI[cluster]
    M_th_i = M_th(T, muB, cluster)

    if M_th_i >= Mi:
        return  numpy.heaviside(math.fsum([M**2 , -Mi**2]), H_ZERO) \
                -numpy.heaviside(math.fsum([M**2 , -M_th_i**2]), H_ZERO)
    else:
        return 0.0
    

@functools.lru_cache(maxsize=1024)
def continuum_factor1(M: float, T: float, muB: float, cluster: str) -> float:
    Ni = NI[cluster]
    Mi = MI[cluster]
    M_th_0 = M_th(0, 0, cluster)
    M_th_i = M_th(T, muB, cluster)
    nlambda = Ni*LAMBDA
    M_thL = math.fsum([M_th_0, nlambda])
    if M_th_i >= Mi:
        if M >= M_th_i and M <= M_thL:
            a = -1.0/math.fsum([-M_th_i, M_thL])
            b = -M_thL/math.fsum([M_th_i, -M_thL])
            return (a*M+b)**CONT_POW_M
        else:
            return 0.0
    else:
        return 0.0
    

@functools.lru_cache(maxsize=1024)
def continuum_factor2(M: float, T: float, muB: float, TM: float, TM2: float, cluster: str) -> float:
    Ni = NI[cluster]
    Mi = MI[cluster]
    M_th_0 = M_th(0, 0, cluster)
    M_th_i = M_th(T, muB, cluster)
    nlambda = Ni*LAMBDA
    M_thL = math.fsum([M_th_0, nlambda])
    M_sc = MSC_SLOPE*(T - TM) + Mi
    if M_th_i < Mi and M_thL > M_sc and M_sc > Mi:
        if M >= Mi and M < M_sc:
            a = 1.0/math.fsum([-Mi, M_sc])
            b = Mi/math.fsum([Mi, -M_sc])
            a2 = -1.0/math.fsum([TM2,-TM])
            b2 = -TM2/math.fsum([TM, -TM2])
            return ((a*M+b)**CONT_POW_M)*(a2*T+b2)
        elif M >= M_sc and M < M_thL:
            a = -1.0/math.fsum([-M_sc, M_thL])
            b = -M_thL/math.fsum([M_sc, -M_thL])
            a2 = -1.0/math.fsum([TM2,-TM])
            b2 = -TM2/math.fsum([TM, -TM2])
            return ((a*M+b)**CONT_POW_M)*(a2*T+b2)
        else:
            return 0.0
    else:
        return 0.0
    

@functools.lru_cache(maxsize=1024)
def phase_factor(M: float, T: float, muB: float, TM: float, TM2: float, cluster: str) -> float:
    delta_i = bound_factor(M, T, muB, cluster)+continuum_factor1(M, T, muB, cluster)+continuum_factor2(M, T, muB, TM, TM2, cluster)  
    return (delta_i-(1.0/(2.0*math.pi))*math.sin(2.0*math.pi*delta_i))


@functools.lru_cache(maxsize=1024)
def b_boson_singlet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        fp = pnjl.thermo.distributions.dfdM_boson_singlet(p, T, muB, M, a, '+')
        fm = pnjl.thermo.distributions.dfdM_boson_singlet(p, T, muB, M, a, '-')
        return math.fsum([fp, -fm])*delta_i


@functools.lru_cache(maxsize=1024)
def b_fermion_singlet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        fp = pnjl.thermo.distributions.dfdM_fermion_singlet(p, T, muB, M, a, '+')
        fm = pnjl.thermo.distributions.dfdM_fermion_singlet(p, T, muB, M, a, '-')
        return math.fsum([fp, -fm])*delta_i


@functools.lru_cache(maxsize=1024)
def b_boson_triplet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        fp = pnjl.thermo.distributions.dfdM_boson_triplet(p, T, muB, phi_re, phi_im, M, a, '+').real
        fm = pnjl.thermo.distributions.dfdM_boson_antitriplet(p, T, muB, phi_re, phi_im, M, a, '-').real
        return -math.fsum([fp, -fm])*delta_i


@functools.lru_cache(maxsize=1024)
def b_boson_antitriplet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        fp = pnjl.thermo.distributions.dfdM_boson_antitriplet(p, T, muB, phi_re, phi_im, M, a, '+').real
        fm = pnjl.thermo.distributions.dfdM_boson_triplet(p, T, muB, phi_re, phi_im, M, a, '-').real
        return -math.fsum([fp, -fm])*delta_i


@functools.lru_cache(maxsize=1024)
def b_fermion_triplet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        fp = pnjl.thermo.distributions.dfdM_fermion_triplet(p, T, muB, phi_re, phi_im, M, a, '+').real
        fm = pnjl.thermo.distributions.dfdM_fermion_antitriplet(p, T, muB, phi_re, phi_im, M, a, '-').real
        return -math.fsum([fp, -fm])*delta_i


@functools.lru_cache(maxsize=1024)
def b_fermion_antitriplet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        fp = pnjl.thermo.distributions.dfdM_fermion_antitriplet(p, T, muB, phi_re, phi_im, M, a, '+').real
        fm = pnjl.thermo.distributions.dfdM_fermion_triplet(p, T, muB, phi_re, phi_im, M, a, '-').real
        return -math.fsum([fp, -fm])*delta_i
    

b_integral_hash = {
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


@functools.lru_cache(maxsize=1024)
def bdensity_integral(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_min: float, M_max: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    integral, _ = scipy.integrate.quad(
        b_integral_hash[cluster], M_min, M_max,
        args = (p, T, muB, phi_re, phi_im, a, TM, TM2, cluster)
    )
    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def bdensity(
    T: float, muB: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    D_I = DI[cluster]
    N_I = NI[cluster]
    A_I = BI[cluster]
    if A_I == 0.0:
        return 0.0
    else:
        TM = T_Mott(muB, cluster)
        TM2 = T_Mott2(muB, cluster)
        M_th_0 = M_th(0.0, 0.0, cluster)
        M_min = M_th(T, muB, cluster)
        M_max = math.fsum([M_th_0, LAMBDA*N_I])
        b_susd = pnjl.thermo.gcp_cluster.bound_step_continuum_step.bdensity(
            T, muB, phi_re, phi_im, cluster
        )
        integral, _ = scipy.integrate.quad(
            bdensity_integral, 0.0, math.inf,
            args = (T, muB, phi_re, phi_im, M_min, M_max, A_I, TM, TM2, cluster)
        )
        return b_susd + (A_I*D_I/(2.0*(math.pi**2)))*integral


@functools.lru_cache(maxsize=1024)
def s_boson_singlet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        sigma_p = 0.0
        logyp = pnjl.thermo.distributions.log_y(p, T, muB, M, a, 1, '+')
        if logyp != 0.0:
            dfp = pnjl.thermo.distributions.dfdM_boson_singlet(p, T, muB, M, a, '+')
            sigma_p = logyp*dfp
        sigma_m = 0.0
        logym = pnjl.thermo.distributions.log_y(p, T, muB, M, a, 1, '-')
        if logym != 0.0:
            dfm = pnjl.thermo.distributions.dfdM_boson_singlet(p, T, muB, M, a, '-')
            sigma_m = logym*dfm
        return math.fsum([sigma_p, sigma_m])*delta_i
    

@functools.lru_cache(maxsize=1024)
def s_fermion_singlet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        sigma_p = 0.0
        fp = pnjl.thermo.distributions.f_fermion_singlet(p, T, muB, M, a, '+')
        dfp = pnjl.thermo.distributions.dfdM_fermion_singlet(p, T, muB, M, a, '+')
        if not (fp == 0.0 or dfp == 0.0):
            sigma_p = math.fsum([math.log(fp), -math.log(math.fsum([1.0, -fp]))])*dfp
        sigma_m = 0.0
        fm = pnjl.thermo.distributions.f_fermion_singlet(p, T, muB, M, a, '-')
        dfm = pnjl.thermo.distributions.dfdM_fermion_singlet(p, T, muB, M, a, '-')
        if not (fm == 0.0 or dfm == 0.0):
            sigma_m = math.fsum([math.log(fm), -math.log(math.fsum([1.0, -fm]))])*dfm
        return -math.fsum([sigma_p, sigma_m])*delta_i
    

@functools.lru_cache(maxsize=1024)
def s_boson_triplet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        energy = pnjl.thermo.distributions.En(p, M)
        dfp = pnjl.thermo.distributions.dfdM_boson_triplet(p, T, muB, phi_re, phi_im, M, a, '+').real
        sigma_p = ((energy-a*muB)/T)*dfp
        dfm = pnjl.thermo.distributions.dfdM_boson_antitriplet(p, T, muB, phi_re, phi_im, M, a, '-').real
        sigma_m = ((energy+a*muB)/T)*dfm
        return -math.fsum([sigma_p, sigma_m])*delta_i
    

@functools.lru_cache(maxsize=1024)
def s_boson_antitriplet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        energy = pnjl.thermo.distributions.En(p, M)
        dfp = pnjl.thermo.distributions.dfdM_boson_antitriplet(p, T, muB, phi_re, phi_im, M, a, '+').real
        sigma_p = ((energy-a*muB)/T)*dfp
        dfm = pnjl.thermo.distributions.dfdM_boson_triplet(p, T, muB, phi_re, phi_im, M, a, '-').real
        sigma_m = ((energy+a*muB)/T)*dfm
        return -math.fsum([sigma_p, sigma_m])*delta_i
    

@functools.lru_cache(maxsize=1024)
def s_fermion_triplet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        energy = pnjl.thermo.distributions.En(p, M)
        dfp = pnjl.thermo.distributions.dfdM_fermion_triplet(p, T, muB, phi_re, phi_im, M, a, '+').real
        sigma_p = ((energy-a*muB)/T)*dfp
        dfm = pnjl.thermo.distributions.dfdM_fermion_antitriplet(p, T, muB, phi_re, phi_im, M, a, '-').real
        sigma_m = ((energy+a*muB)/T)*dfm
        return -math.fsum([sigma_p, sigma_m])*delta_i
    

@functools.lru_cache(maxsize=1024)
def s_fermion_antitriplet_integrand(
    M: float, p: float, T: float, muB: float,
    phi_re: float, phi_im: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    delta_i = phase_factor(M, T, muB, TM, TM2, cluster)
    if delta_i == 0.0:
        return 0.0
    else:
        energy = pnjl.thermo.distributions.En(p, M)
        dfp = pnjl.thermo.distributions.dfdM_fermion_antitriplet(p, T, muB, phi_re, phi_im, M, a, '+').real
        sigma_p = ((energy-a*muB)/T)*dfp
        dfm = pnjl.thermo.distributions.dfdM_fermion_triplet(p, T, muB, phi_re, phi_im, M, a, '-').real
        sigma_m = ((energy+a*muB)/T)*dfm
        return -math.fsum([sigma_p, sigma_m])*delta_i
    

s_integral_hash = {
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
def sdensity_integral(
    p: float, T: float, muB: float, phi_re: float, phi_im: float,
    M_min: float, M_max: float, a: float, TM: float, TM2: float, cluster: str
) -> float:
    integral, _ = scipy.integrate.quad(
        s_integral_hash[cluster], M_min, M_max,
        args = (p, T, muB, phi_re, phi_im, a, TM, TM2, cluster)
    )
    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def sdensity(
    T: float, muB: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    D_I = DI[cluster]
    N_I = NI[cluster]
    A_I = BI[cluster]
    if A_I == 0.0 and muB != 0.0:
        return sdensity(T, 0.0, phi_re, phi_im, cluster)
    else:
        M_th_0 = M_th(0.0, 0.0, cluster)
        M_max = math.fsum([M_th_0, LAMBDA*N_I])
        M_min = M_th(T, muB, cluster)
        TM = T_Mott(muB, cluster)
        TM2 = T_Mott2(muB, cluster)
        s_susd = pnjl.thermo.gcp_cluster.bound_step_continuum_step.sdensity(
            T, muB, phi_re, phi_im, cluster
        )
        integral, _ = scipy.integrate.quad(
            sdensity_integral, 0.0, math.inf,
            args = (T, muB, phi_re, phi_im, M_min, M_max, A_I, TM, TM2, cluster)
        )
        return s_susd + (D_I/(2.0*(math.pi**2)))*integral


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant(
    rank: int, T: float, muB: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    if rank == 1:
        return 3.0 * bdensity(T, muB, phi_re, phi_im, cluster)
    else:
        h = 1e-2
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
                    rank-1, T, muB_el, phi_el[0], phi_el[1], cluster
                )
                for muB_el, coef, phi_el in zip(muB_vec, deriv_coef, phi_vec)
            ]
            return math.fsum(out_vec)
        else:
            new_muB = math.fsum([muB, h])
            new_phi_re, new_phi_im = phi_re, phi_im
            return qnumber_cumulant(
                rank, T, new_muB, new_phi_re, new_phi_im, cluster
            )


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


#to trzeba przerobic...
@functools.lru_cache(maxsize=1024)
def pressure_ib_buns(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    
    integral, error = scipy.integrate.quad(
        lambda _mu, _T, _phi_re, _phi_im, _cluster: bdensity(_T, _mu, _phi_re, _phi_im, _cluster),
        0.0, mu, args = (T, phi_re, phi_im, cluster)
    )

    return integral


#to też
@functools.lru_cache(maxsize=1024)
def pressure_is_buns(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    
    integral, error = scipy.integrate.quad(
        sdensity,
        0.0, T, args = (mu, phi_re, phi_im, cluster)
    )

    return integral