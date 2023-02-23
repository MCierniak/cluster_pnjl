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

import utils
import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_sigma_lattice
import pnjl.thermo.gcp_pl_polynomial
import pnjl.thermo.gcp_cluster.bound_step_continuum_step


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
def bound_factor(M: float, T: float, mu: float, cluster: str) -> float:

    H_ZERO = pnjl.defaults.HEAVISIDE_ZERO

    Mi = pnjl.defaults.MI[cluster]
    M_th_i = M_th(T, mu, cluster)

    if M_th_i >= Mi:
        return  numpy.heaviside(math.fsum([M**2 , -Mi**2]), H_ZERO) \
                -numpy.heaviside(math.fsum([M**2 , -M_th_i**2]), H_ZERO)
    else:
        return 0.0


@functools.lru_cache(maxsize=1024)
def continuum_factor1(M: float, T: float, mu: float, cluster: str) -> float:

    L = pnjl.defaults.L
    NI = pnjl.defaults.NI[cluster]
    H_ZERO = pnjl.defaults.HEAVISIDE_ZERO
    MI = pnjl.defaults.MI[cluster]

    M_th_0 = M_th(0, 0, cluster)
    M_th_i = M_th(T, mu, cluster)

    nlambda = NI*L

    if M_th_i >= MI:
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

    if MI > M_th_i:
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
    else:
        return 0.0
    

@functools.lru_cache(maxsize=1024)
def phase_factor(M: float, T: float, mu: float, cluster: str) -> float:

        return bound_factor(M, T, mu, cluster)+continuum_factor1(M, T, mu, cluster)+continuum_factor2(M, T, mu, cluster)  


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

    if cluster in ["pi", "K", "rho", "omega", "T"]:
        return 0.0
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
def buns_b_boson_singlet_integrand(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int, cluster: str
) -> float:
    fp = pnjl.thermo.distributions.dfdM_boson_singlet(p, T, mu, M, a, '+')
    fm = pnjl.thermo.distributions.dfdM_boson_singlet(p, T, mu, M, a, '-')
    return math.fsum([fp, -fm])
    # log_p = pnjl.thermo.distributions.log_y(p, T, mu, M, a, 1, '+')
    # log_m = pnjl.thermo.distributions.log_y(p, T, mu, M, a, 1, '-')
    # if log_p >= utils.EXP_LIMIT or log_m >= utils.EXP_LIMIT:
    #     return 0.0
    # else:
    #     energy = pnjl.thermo.distributions.En(p, M)
    #     ex_p = math.exp(log_p)
    #     ex_m = math.exp(log_m)
    #     fp_i = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, a, '+')**2
    #     fm_i = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, a, '-')**2
    #     delta_i = phase_factor(M, T, mu, cluster)
    #     return math.fsum([ex_p*fp_i, -ex_m*fm_i])*(M/(energy*T))*delta_i


@functools.lru_cache(maxsize=1024)
def buns_b_fermion_singlet_integrand(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int, cluster: str
) -> float:
    fp = pnjl.thermo.distributions.dfdM_fermion_singlet(p, T, mu, M, a, '+')
    fm = pnjl.thermo.distributions.dfdM_fermion_singlet(p, T, mu, M, a, '-')
    return math.fsum([fp, -fm])
    # log_p = pnjl.thermo.distributions.log_y(p, T, mu, M, a, 1, '+')
    # log_m = pnjl.thermo.distributions.log_y(p, T, mu, M, a, 1, '-')
    # if log_p >= utils.EXP_LIMIT or log_m >= utils.EXP_LIMIT:
    #     return 0.0
    # else:
    #     energy = pnjl.thermo.distributions.En(p, M)
    #     ex_p = math.exp(log_p)
    #     ex_m = math.exp(log_m)
    #     fp_i = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, a, '+')**2
    #     fm_i = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, a, '-')**2
    #     delta_i = phase_factor(M, T, mu, cluster)
    #     return math.fsum([ex_p*fp_i, -ex_m*fm_i])*(M/(energy*T))*delta_i


@functools.lru_cache(maxsize=1024)
def buns_b_aux_fermion(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int, typ: str
) -> complex:
    log_1 = pnjl.thermo.distributions.log_y(p, T, mu, M, a, 1, typ)
    log_2 = pnjl.thermo.distributions.log_y(p, T, mu, M, a, 2, typ)
    if log_1 >= utils.EXP_LIMIT:
        return complex(0.0, 0.0)
    elif log_1 < utils.EXP_LIMIT and log_2 >= utils.EXP_LIMIT:
        num_1_re = 2.0*math.fsum([math.exp(-log_1)*phi_re, phi_re])
        num_1_im = 2.0*math.fsum([math.exp(-log_1)*phi_im, -phi_im])
        den_1_re = math.fsum([math.exp(log_1), math.exp(-log_1)*phi_re, 2.0*phi_re])
        den_1_im = math.fsum([math.exp(-log_1)*phi_im, -2.0*phi_im])
        num_2_re = 2.0*math.fsum([math.exp(-log_1)*1.0, phi_re])
        num_2_im = 2.0*phi_im
        den_2_re = math.fsum([math.exp(-log_1), 2.0*phi_re, math.exp(log_1)*phi_re])
        den_2_im = math.fsum([2.0*phi_im, -math.exp(log_1)*phi_im])
        return  complex(1.0, 0.0) \
                -(complex(num_1_re, num_1_im)/complex(den_1_re, den_1_im)) \
                +(complex(num_2_re, num_2_im)/complex(den_2_re, den_2_im))
    else:
        num_1_re = 2.0*math.fsum([phi_re, math.exp(log_1)*phi_re])
        num_1_im = 2.0*math.fsum([phi_im, -math.exp(log_1)*phi_im])
        den_1_re = math.fsum([math.exp(log_2), phi_re, 2.0*phi_re*math.exp(log_1)])
        den_1_im = math.fsum([phi_im, -2.0*phi_im*math.exp(log_1)])
        num_2_re = 2.0*math.fsum([1.0, math.exp(log_1)*phi_re])
        num_2_im = 2.0*math.exp(log_1)*phi_im
        den_2_re = math.fsum([1.0, 2.0*math.exp(log_1)*phi_re, math.exp(log_2)*phi_re])
        den_2_im = math.fsum([2.0*math.exp(log_1)*phi_im, -math.exp(log_2)*phi_im])
        return  complex(1.0, 0.0) \
                -(complex(num_1_re, num_1_im)/complex(den_1_re, den_1_im)) \
                +(complex(num_2_re, num_2_im)/complex(den_2_re, den_2_im))


@functools.lru_cache(maxsize=1024)
def buns_b_aux_boson(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int, typ: str
) -> complex:
    log_1 = pnjl.thermo.distributions.log_y(p, T, mu, M, a, 1, typ)
    log_2 = pnjl.thermo.distributions.log_y(p, T, mu, M, a, 2, typ)
    if log_1 >= utils.EXP_LIMIT:
        return complex(0.0, 0.0)
    elif log_1 < utils.EXP_LIMIT and log_2 >= utils.EXP_LIMIT:
        num_1_re = 2.0*math.fsum([math.exp(-log_1)*phi_re, -phi_re])
        num_1_im = 2.0*math.fsum([math.exp(-log_1)*phi_im, phi_im])
        den_1_re = math.fsum([math.exp(log_1), math.exp(-log_1)*phi_re, -2.0*phi_re])
        den_1_im = math.fsum([math.exp(-log_1)*phi_im, 2.0*phi_im])
        num_2_re = 2.0*math.fsum([-math.exp(-log_1)*1.0, phi_re])
        num_2_im = 2.0*phi_im
        den_2_re = math.fsum([math.exp(-log_1), -2.0*phi_re, math.exp(log_1)*phi_re])
        den_2_im = math.fsum([-2.0*phi_im, -math.exp(log_1)*phi_im])
        return  -complex(1.0, 0.0) \
                +(complex(num_1_re, num_1_im)/complex(den_1_re, den_1_im)) \
                +(complex(num_2_re, num_2_im)/complex(den_2_re, den_2_im))
    else:
        num_1_re = 2.0*math.fsum([phi_re, -math.exp(log_1)*phi_re])
        num_1_im = 2.0*math.fsum([phi_im, math.exp(log_1)*phi_im])
        den_1_re = math.fsum([math.exp(log_2), phi_re, -2.0*phi_re*math.exp(log_1)])
        den_1_im = math.fsum([phi_im, 2.0*phi_im*math.exp(log_1)])
        num_2_re = 2.0*math.fsum([-1.0, math.exp(log_1)*phi_re])
        num_2_im = 2.0*math.exp(log_1)*phi_im
        den_2_re = math.fsum([1.0, -2.0*math.exp(log_1)*phi_re, math.exp(log_2)*phi_re])
        den_2_im = math.fsum([-2.0*math.exp(log_1)*phi_im, -math.exp(log_2)*phi_im])
        return  -complex(1.0, 0.0) \
                +(complex(num_1_re, num_1_im)/complex(den_1_re, den_1_im)) \
                +(complex(num_2_re, num_2_im)/complex(den_2_re, den_2_im))


@functools.lru_cache(maxsize=1024)
def buns_b_boson_triplet_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int, cluster: str
) -> float:
    energy = pnjl.thermo.distributions.En(p, M)
    fp_i = pnjl.thermo.distributions.f_boson_triplet(p, T, mu, phi_re, phi_im, M, a, '+')
    fp2_i = fp_i**2
    aux_p = buns_b_aux_boson(M, p, T, mu, phi_re, phi_im, a, '+')
    fm_i = pnjl.thermo.distributions.f_boson_antitriplet(p, T, mu, phi_re, phi_im, M, a, '-')
    fm2_i = fm_i**2
    aux_m = buns_b_aux_boson(M, p, T, mu, phi_re, phi_im, a, '-')
    delta_i = phase_factor(M, T, mu, cluster)
    el_p = ((fp2_i+fp_i)*aux_p).real
    el_m = ((fm2_i+fm_i)*aux_m).real
    return -math.fsum([el_p, -el_m])*(M/(energy*T))*delta_i


@functools.lru_cache(maxsize=1024)
def buns_b_boson_antitriplet_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int, cluster: str
) -> float:
    energy = pnjl.thermo.distributions.En(p, M)
    fp_i = pnjl.thermo.distributions.f_boson_antitriplet(p, T, mu, phi_re, phi_im, M, a, '+')
    fp2_i = fp_i**2
    aux_p = buns_b_aux_boson(M, p, T, mu, phi_re, -phi_im, a, '+')
    fm_i = pnjl.thermo.distributions.f_boson_triplet(p, T, mu, phi_re, phi_im, M, a, '-')
    fm2_i = fm_i**2
    aux_m = buns_b_aux_boson(M, p, T, mu, phi_re, -phi_im, a, '-')
    delta_i = phase_factor(M, T, mu, cluster)
    el_p = ((fp2_i+fp_i)*aux_p).real
    el_m = ((fm2_i+fm_i)*aux_m).real
    return -math.fsum([el_p, -el_m])*(M/(energy*T))*delta_i


@functools.lru_cache(maxsize=1024)
def buns_b_fermion_triplet_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int, cluster: str
) -> float:
    energy = pnjl.thermo.distributions.En(p, M)
    fp_i = pnjl.thermo.distributions.f_fermion_triplet(p, T, mu, phi_re, phi_im, M, a, '+')
    fp2_i = fp_i**2
    aux_p = buns_b_aux_fermion(M, p, T, mu, phi_re, phi_im, a, '+')
    fm_i = pnjl.thermo.distributions.f_fermion_antitriplet(p, T, mu, phi_re, phi_im, M, a, '-')
    fm2_i = fm_i**2
    aux_m = buns_b_aux_fermion(M, p, T, mu, phi_re, phi_im, a, '-')
    delta_i = phase_factor(M, T, mu, cluster)
    el_p = ((fp2_i-fp_i)*aux_p).real
    el_m = ((fm2_i-fm_i)*aux_m).real
    return -math.fsum([el_p, -el_m])*(M/(energy*T))*delta_i


@functools.lru_cache(maxsize=1024)
def buns_b_fermion_antitriplet_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int, cluster: str
) -> float:
    energy = pnjl.thermo.distributions.En(p, M)
    fp_i = pnjl.thermo.distributions.f_fermion_antitriplet(p, T, mu, phi_re, phi_im, M, a, '+')
    fp2_i = fp_i**2
    aux_p = buns_b_aux_fermion(M, p, T, mu, phi_re, -phi_im, a, '+')
    fm_i = pnjl.thermo.distributions.f_fermion_triplet(p, T, mu, phi_re, phi_im, M, a, '-')
    fm2_i = fm_i**2
    aux_m = buns_b_aux_fermion(M, p, T, mu, phi_re, -phi_im, a, '-')
    delta_i = phase_factor(M, T, mu, cluster)
    el_p = ((fp2_i-fp_i)*aux_p).real
    el_m = ((fm2_i-fm_i)*aux_m).real
    return -math.fsum([el_p, -el_m])*(M/(energy*T))*delta_i


buns_b_integral_hash = {
    'pi': buns_b_boson_singlet_integrand,
    'K': buns_b_boson_singlet_integrand,
    'rho': buns_b_boson_singlet_integrand,
    'omega': buns_b_boson_singlet_integrand,
    'D': buns_b_boson_antitriplet_integrand_real,
    'N': buns_b_fermion_singlet_integrand,
    'T': buns_b_boson_singlet_integrand,
    'F': buns_b_boson_triplet_integrand_real,
    'P': buns_b_fermion_singlet_integrand,
    'Q': buns_b_fermion_antitriplet_integrand_real,
    'H': buns_b_boson_singlet_integrand
}


@functools.lru_cache(maxsize=1024)
def bdensity_buns_integral(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_max: float, cluster: str
) -> float:
    
    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    integral, error = scipy.integrate.quad(
        buns_b_integral_hash[cluster], 0.0, M_max,
        args = (p, T, mu, phi_re, phi_im, A_I, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def bdensity_buns(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:

    L = pnjl.defaults.L
    D_I = pnjl.defaults.DI[cluster]
    N_I = pnjl.defaults.NI[cluster]
    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    M_th_0 = M_th(0.0, 0.0, cluster)
    M_max = math.fsum([M_th_0, L*N_I])

    integral, error = scipy.integrate.quad(
        bdensity_buns_integral, 0.0, math.inf,
        args = (T, mu, phi_re, phi_im, M_max, cluster)
    )

    return (A_I*D_I/(2.0*(math.pi**2)))*integral


buns_s_integral_hash = {
    'pi': buns_s_boson_singlet_integrand,
    'K': buns_s_boson_singlet_integrand,
    'rho': buns_s_boson_singlet_integrand,
    'omega': buns_s_boson_singlet_integrand,
    'D': buns_s_boson_antitriplet_integrand_real,
    'N': buns_s_fermion_singlet_integrand,
    'T': buns_s_boson_singlet_integrand,
    'F': buns_s_boson_triplet_integrand_real,
    'P': buns_s_fermion_singlet_integrand,
    'Q': buns_s_fermion_antitriplet_integrand_real,
    'H': buns_s_boson_singlet_integrand
}


@functools.lru_cache(maxsize=1024)
def sdensity_buns_integral(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_max: float, cluster: str
) -> float:
    
    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    integral, error = scipy.integrate.quad(
        buns_s_integral_hash[cluster], 0.0, M_max,
        args = (p, T, mu, phi_re, phi_im, A_I, cluster)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def sdensity_buns(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:

    L = pnjl.defaults.L
    D_I = pnjl.defaults.DI[cluster]
    N_I = pnjl.defaults.NI[cluster]
    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    M_th_0 = M_th(0.0, 0.0, cluster)
    M_max = math.fsum([M_th_0, L*N_I])

    integral, error = scipy.integrate.quad(
        bdensity_buns_integral, 0.0, math.inf,
        args = (T, mu, phi_re, phi_im, M_max, cluster)
    )

    return (D_I/(2.0*(math.pi**2)))*integral


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