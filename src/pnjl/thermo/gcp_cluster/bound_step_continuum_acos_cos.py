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
import typing
import functools

import scipy.optimize
import scipy.integrate

import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_sigma_lattice
import pnjl.thermo.gcp_pl_polynomial
import pnjl.thermo.gcp_cluster.bound_step_continuum_step


step_up_step_down_max = {
    'pi': 0.0,
    'K': 0.0,
    'rho': 0.0,
    'omega': 0.0,
    'D': 0.0,
    'N': 0.0,
    'T': 0.0,
    'F': 0.0,
    'P': 0.0,
    'Q': 0.0,
    'H': 0.0
}


@functools.lru_cache(maxsize=1024)
def M_th(T: float, mu: float, cluster: str) -> float:

    N_I = pnjl.defaults.NI[cluster]
    S_I = pnjl.defaults.S[cluster]
    M_th_i = math.sqrt(2)*math.fsum([
        math.fsum([N_I,-S_I])*pnjl.thermo.gcp_sigma_lattice.Ml(T, mu),
        S_I*pnjl.thermo.gcp_sigma_lattice.Ms(T, mu)
    ])

    return M_th_i


@functools.lru_cache(maxsize=1024)
def continuum_factor1(M: float, T: float, mu: float, cluster: str) -> float:

    L = pnjl.defaults.L
    NI = pnjl.defaults.NI[cluster]
    H_ZERO = pnjl.defaults.HEAVISIDE_ZERO

    M_th_0 = M_th(0, 0, cluster)
    M_th_i = M_th(T, mu, cluster)

    nlambda = NI*L

    if M >= M_th_i and M <= math.fsum([M_th_0, nlambda]):
        
        arccos_in_a = -2.0
        heavi1_i = math.fsum([M**2 , -M_th_i**2])
        heavi2_i = math.fsum([M_th_0, nlambda, -M])
        arccos_in_b = math.fsum([M_th_i, M_th_0, nlambda])
        arccos_in_denom = math.fsum([M_th_i, -M_th_0, -nlambda])

        arccos_in = (arccos_in_a*M + arccos_in_b)/arccos_in_denom

        arccos_el = math.acos(arccos_in)
        heavi1 = numpy.heaviside(heavi1_i, H_ZERO)
        heavi2 = numpy.heaviside(heavi2_i, H_ZERO)

        return heavi1*heavi2*arccos_el/math.pi

    else:
        return 0.0


@functools.lru_cache(maxsize=1024)
def continuum_factor2(M: float, T: float, mu: float, cluster: str) -> float:

    L = pnjl.defaults.L
    NI = pnjl.defaults.NI[cluster]
    H_ZERO = pnjl.defaults.HEAVISIDE_ZERO
    MI = pnjl.defaults.MI[cluster]

    M_th_0 = M_th(0, 0, cluster)
    M_th_i = M_th(T, mu, cluster)

    nlambda = NI*L

    if M >= M_th_i and M <= math.fsum([M_th_0, nlambda]):
        
        arccos_in_a = -2.0
        heavi1_i = math.fsum([M**2 , -M_th_i**2])
        heavi2_i = math.fsum([M_th_0, nlambda, -M])
        arccos_in_b = math.fsum([M_th_i, M_th_0, nlambda])
        arccos_in_denom = math.fsum([M_th_i, -M_th_0, -nlambda])

        arccos_in = math.fsum([arccos_in_a*M, arccos_in_b])/arccos_in_denom
        cos_in = math.fsum(
            [-math.pi*M, 0.5*(math.pi*arccos_in_b)]
        )/arccos_in_denom

        cos_el = math.cos(cos_in)
        arccos_el = math.acos(arccos_in)
        heavi1 = numpy.heaviside(heavi1_i, H_ZERO)
        heavi2 = numpy.heaviside(heavi2_i, H_ZERO)

        damp = 1.8*nlambda

        dampening_factor = math.fsum(
            [-math.fsum([MI, -damp]), M_th_i]
        )/damp

        if dampening_factor <= 0.0:
            return 0.0
        else:
            return heavi1*heavi2*dampening_factor*math.fsum(
                [arccos_el*dampening_factor, cos_el*(1.0-dampening_factor)]
            )/math.pi

    else:
        return 0.0
    

@functools.lru_cache(maxsize=1024)
def gcp_boson_singlet_inner_integrand1(
    M: float, p: float, T: float, mu: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, A_I, '+')
    fm = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, A_I, '-')

    cf = continuum_factor1(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_boson_singlet_inner_integrand2(
    M: float, p: float, T: float, mu: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, A_I, '+')
    fm = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, A_I, '-')

    cf = continuum_factor2(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_fermion_singlet_inner_integrand1(
    M: float, p: float, T: float, mu: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, A_I, '+')
    fm = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, A_I, '-')

    cf = continuum_factor1(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_fermion_singlet_inner_integrand2(
    M: float, p: float, T: float, mu: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, A_I, '+')
    fm = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, A_I, '-')

    cf = continuum_factor2(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_inner_integrand_real1(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).real
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).real

    cf = continuum_factor1(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_inner_integrand_real2(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).real
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).real

    cf = continuum_factor2(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_inner_integrand_real1(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).real
    fm = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).real

    cf = continuum_factor1(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_inner_integrand_real2(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).real
    fm = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).real

    cf = continuum_factor2(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_inner_integrand_imag1(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).imag

    cf = continuum_factor1(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_inner_integrand_imag2(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).imag

    cf = continuum_factor2(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_inner_integrand_imag1(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).imag

    cf = continuum_factor1(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_inner_integrand_imag2(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).imag

    cf = continuum_factor2(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_inner_integrand_real1(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).real
    fm = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).real

    cf = continuum_factor1(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_inner_integrand_real2(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).real
    fm = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).real

    cf = continuum_factor2(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_inner_integrand_real1(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).real
    fm = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).real

    cf = continuum_factor1(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_inner_integrand_real2(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).real
    fm = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).real

    cf = continuum_factor2(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_inner_integrand_imag1(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).imag

    cf = continuum_factor1(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_inner_integrand_imag2(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).imag

    cf = continuum_factor2(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_inner_integrand_imag1(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).imag

    cf = continuum_factor1(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_inner_integrand_imag2(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, cluster: str
) -> float:

    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, A_I, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, A_I, '-'
    ).imag

    cf = continuum_factor2(M, T, mu, cluster)

    return (M/En)*math.fsum([fp, fm])*cf


@functools.lru_cache(maxsize=1024)
def gcp_boson_singlet_integrand1(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_singlet_inner_integrand1,
        M_TH_I, M_max, args = (p, T, mu, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_singlet_integrand2(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_singlet_inner_integrand2,
        M_TH_I, M_max, args = (p, T, mu, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_singlet_integrand1(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_singlet_inner_integrand1,
        M_TH_I, M_max, args = (p, T, mu, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_singlet_integrand2(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_singlet_inner_integrand2,
        M_TH_I, M_max, args = (p, T, mu, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_integrand_real1(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_triplet_inner_integrand_real1, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_integrand_real2(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_triplet_inner_integrand_real2, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_integrand_real1(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_antitriplet_inner_integrand_real1, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_integrand_real2(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_antitriplet_inner_integrand_real2, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_integrand_real1(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_triplet_inner_integrand_real1, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_integrand_real2(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_triplet_inner_integrand_real2, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_integrand_real1(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_antitriplet_inner_integrand_real1, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_integrand_real2(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_antitriplet_inner_integrand_real2, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_integrand_imag1(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_triplet_inner_integrand_imag1, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_integrand_imag2(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_triplet_inner_integrand_imag2, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_integrand_imag1(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_antitriplet_inner_integrand_imag1, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_integrand_imag2(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_antitriplet_inner_integrand_imag2, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_integrand_imag1(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_triplet_inner_integrand_imag1, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_integrand_imag2(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_triplet_inner_integrand_imag2, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_integrand_imag1(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_antitriplet_inner_integrand_imag1, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_integrand_imag2(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_TH_I: float, M_max: float, cluster: str
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_antitriplet_inner_integrand_imag2, M_TH_I, M_max,
        args = (p, T, mu, phi_re, phi_im, cluster)
    )

    return (p**2)*integral


gcp_real_hash = {
    'pi': (
        gcp_boson_singlet_integrand1,
        gcp_boson_singlet_integrand2
    ),
    'K': (
        gcp_boson_singlet_integrand1,
        gcp_boson_singlet_integrand2
    ),
    'rho': (
        gcp_boson_singlet_integrand1,
        gcp_boson_singlet_integrand2
    ),
    'omega': (
        gcp_boson_singlet_integrand1,
        gcp_boson_singlet_integrand2
    ),
    'D': (
        gcp_boson_antitriplet_integrand_real1,
        gcp_boson_antitriplet_integrand_real2
    ),
    'N': (
        gcp_fermion_singlet_integrand1,
        gcp_fermion_singlet_integrand2
    ),
    'T': (
        gcp_boson_singlet_integrand1,
        gcp_boson_singlet_integrand2
    ),
    'F': (
        gcp_boson_triplet_integrand_real1,
        gcp_boson_triplet_integrand_real2
    ),
    'P': (
        gcp_fermion_singlet_integrand1,
        gcp_fermion_singlet_integrand2
    ),
    'Q': (
        gcp_fermion_antitriplet_integrand_real1,
        gcp_fermion_antitriplet_integrand_real2
    ),
    'H': (
        gcp_boson_singlet_integrand1,
        gcp_boson_singlet_integrand2
    )
}


gcp_imag_hash = {
    'D': (
        gcp_boson_antitriplet_integrand_imag1,
        gcp_boson_antitriplet_integrand_imag2
    ),
    'F': (
        gcp_boson_triplet_integrand_imag1,
        gcp_boson_triplet_integrand_imag2
    ),
    'Q': (
        gcp_fermion_antitriplet_integrand_imag1,
        gcp_fermion_antitriplet_integrand_imag2
    )
}


@functools.lru_cache(maxsize=1024)
def gcp_real(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential 
    (real part).
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    gcp_real : float
        Value of the thermodynamic potential in MeV^4.
    """

    L = pnjl.defaults.L
    M_I = pnjl.defaults.MI[cluster]
    N_I = pnjl.defaults.NI[cluster]
    M_th_i = M_th(T, mu, cluster)
    M_th_0 = M_th(0.0, 0.0, cluster)
    D_I = pnjl.defaults.DI[cluster]

    M_max = math.fsum([M_th_0, L*N_I])

    integral = 0.0
    if M_th_i > M_I:
        integral, error = scipy.integrate.quad(
            gcp_real_hash[cluster][0], 0.0, math.inf,
            args = (T, mu, phi_re, phi_im, M_th_i, M_max, cluster)
        )
    else:
        integral, error = scipy.integrate.quad(
            gcp_real_hash[cluster][1], 0.0, math.inf,
            args = (T, mu, phi_re, phi_im, M_th_i, M_max, cluster)
        )

    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real(
        T, mu, phi_re, phi_im, cluster
    )

    return math.fsum([step, -((D_I*3.0)/(2.0*(math.pi**2)))*integral])


@functools.lru_cache(maxsize=1024)
def gcp_imag(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential 
    (imaginary part).
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    gcp_imag : float
        Value of the thermodynamic potential in MeV^4.
    """

    L = pnjl.defaults.L
    M_I = pnjl.defaults.MI[cluster]
    N_I = pnjl.defaults.NI[cluster]
    M_th_i = M_th(T, mu, cluster)
    M_th_0 = M_th(0.0, 0.0, cluster)
    D_I = pnjl.defaults.DI[cluster]

    M_max = math.fsum([M_th_0, L*N_I])

    if cluster in gcp_imag_hash:
        integral = 0.0
        if M_th_i > M_I:
            integral, error = scipy.integrate.quad(
                gcp_imag_hash[cluster][0], 0.0, math.inf,
                args = (T, mu, phi_re, phi_im, M_th_i, M_max, cluster)
            )
        else:
            integral, error = scipy.integrate.quad(
                gcp_imag_hash[cluster][1], 0.0, math.inf,
                args = (T, mu, phi_re, phi_im, M_th_i, M_max, cluster)
            )

        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_imag(
            T, mu, phi_re, phi_im, cluster
        )

        return math.fsum([step, -((D_I*3.0)/(2.0*(math.pi**2)))*integral])
    else:
        return 0.0


@functools.lru_cache(maxsize=1024)
def pressure(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster pressure.
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    pressure : float
        Value of the thermodynamic pressure in MeV^4.
    """

    return -gcp_real(T, mu, phi_re, phi_im, cluster)


@functools.lru_cache(maxsize=1024)
def bdensity(
    T: float, mu: float, phi_re: float, phi_im: float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster baryon density.
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    phi_solver : Callable
        Function calculating the traced Polyakov-loop for given
        T and mu. Must be of the form
            (T: float, mu: float,
            phi_re0: float, phi_im0: float) -> Tuple[float, float],
            where
                T : temperature in MeV
                mu : quark chemical potential in MeV
                phi_re0 : initial guess for phi_re
                phi_im0 : initial guess for phi_im
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    bdensity : float
        Value of the thermodynamic baryon density in MeV^3.
    """

    h = 1e-2

    if math.fsum([mu, -2*h]) > 0.0:

        mu_vec = [
            math.fsum([mu, 2*h]), math.fsum([mu, h]),
            math.fsum([mu, -h]), math.fsum([mu, -2*h])
        ]
        deriv_coef = [
            -1.0/(12.0*h), 8.0/(12.0*h),
            -8.0/(12.0*h), 1.0/(12.0*h)
        ]
        phi_vec = []
        if pnjl.defaults.D_PHI_D_MU_0:
            phi_vec = [
                tuple([phi_re, phi_im])
                for _ in mu_vec
            ]
        else:
            phi_vec = [
                phi_solver(T, mu_el, phi_re, phi_im)
                for mu_el in mu_vec
            ]

        p_vec = [
            coef*pressure(T, mu_el, phi_el[0], phi_el[1], cluster)/3.0
            for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
        ]

        return math.fsum(p_vec)

    else:

        new_mu = math.fsum([mu, h])
        new_phi_re, new_phi_im = phi_re, phi_im
            
        if not pnjl.defaults.D_PHI_D_MU_0:
            new_phi_re, new_phi_im = phi_solver(T, new_mu, phi_re, phi_im)

        return bdensity(
            T, new_mu, new_phi_re, new_phi_im, 
            phi_solver, cluster
        )


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant(
    rank: int, T: float, mu: float, phi_re: float, phi_im: float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster quark number cumulant chi_q. Based on 
    Eq.29 of https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline 
    definition.
    
    ### Prameters
    rank : int
        Cumulant rank. Rank 1 equals to 3 times the baryon density.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    phi_solver : Callable
        Function calculating the traced Polyakov-loop for given
        T and mu. Must be of the form
            (T: float, mu: float,
            phi_re0: float, phi_im0: float) -> Tuple[float, float],
            where
                T : temperature in MeV
                mu : quark chemical potential in MeV
                phi_re0 : initial guess for phi_re
                phi_im0 : initial guess for phi_im
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    qnumber_cumulant : float
        Value of the thermodynamic quark number cumulant in MeV^3.
    """

    if rank == 1:

        return 3.0 * bdensity(T, mu, phi_re, phi_im,  phi_solver, cluster)

    else:

        h = 1e-2

        if math.fsum([mu, -2*h]) > 0.0:

            mu_vec = [
                math.fsum([mu, 2*h]), math.fsum([mu, h]),
                math.fsum([mu, -h]), math.fsum([mu, -2*h])
            ]
            deriv_coef = [
                -1.0/(12.0*h), 8.0/(12.0*h),
                -8.0/(12.0*h), 1.0/(12.0*h)
            ]
            phi_vec = []
            if pnjl.defaults.D_PHI_D_MU_0:
                phi_vec = [
                    tuple([phi_re, phi_im])
                    for _ in mu_vec
                ]
            else:
                phi_vec = [
                    phi_solver(T, mu_el, phi_re, phi_im)
                    for mu_el in mu_vec
                ]

            out_vec = [
                coef*qnumber_cumulant(
                    rank-1, T, mu_el, phi_el[0], phi_el[1], 
                    phi_solver, cluster)
                for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
            ]

            return math.fsum(out_vec)

        else:

            new_mu = math.fsum([mu, h])
            new_phi_re, new_phi_im = phi_re, phi_im
            
            if not pnjl.defaults.D_PHI_D_MU_0:
                new_phi_re, new_phi_im = phi_solver(T, new_mu, phi_re, phi_im)

            return qnumber_cumulant(
                rank, T, new_mu, new_phi_re, new_phi_im, 
                phi_solver, cluster
            )


@functools.lru_cache(maxsize=1024)
def sdensity(
    T: float, mu: float, phi_re : float, phi_im : float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster entropy density.
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    phi_solver : Callable
        Function calculating the traced Polyakov-loop for given
        T and mu. Must be of the form
            (T: float, mu: float,
            phi_re0: float, phi_im0: float) -> Tuple[float, float],
            where
                T : temperature in MeV
                mu : quark chemical potential in MeV
                phi_re0 : initial guess for phi_re
                phi_im0 : initial guess for phi_im
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    sdensity : float
        Value of the thermodynamic entropy density in MeV^3.
    """

    h = 1e-2

    if math.fsum([T, -2*h]) > 0.0:

        T_vec = [
            math.fsum([T, 2*h]), math.fsum([T, h]),
            math.fsum([T, -h]), math.fsum([T, -2*h])
        ]
        deriv_coef = [
            -1.0/(12.0*h), 8.0/(12.0*h),
            -8.0/(12.0*h), 1.0/(12.0*h)
        ]
        phi_vec = []
        if pnjl.defaults.D_PHI_D_T_0:
            phi_vec = [
                tuple([phi_re, phi_im])
                for _ in T_vec
            ]
        else:
            phi_vec = [
                phi_solver(T_el, mu, phi_re, phi_im)
                for T_el in T_vec
            ]

        p_vec = [
            coef*pressure(T_el, mu, phi_el[0], phi_el[1], cluster)
            for T_el, coef, phi_el in zip(T_vec, deriv_coef, phi_vec)
        ]

        return math.fsum(p_vec)

    else:

        new_T = math.fsum([T, h])
        new_phi_re, new_phi_im = phi_re, phi_im
            
        if not pnjl.defaults.D_PHI_D_T_0:
            new_phi_re, new_phi_im = phi_solver(new_T, mu, phi_re, phi_im)

        return sdensity(
            new_T, mu, new_phi_re, new_phi_im, 
            phi_solver, cluster
        )