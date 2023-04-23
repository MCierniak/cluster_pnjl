"""### Description
Pertrubative correction of the PNJL grandcanonical potential.
Finite-mu extension of formulas in https://arxiv.org/pdf/2012.12894.pdf .
Based on https://inspirehep.net/files/901191eb2f4d03c023787042343325d2 .

### Functions
alpha_s
    QCD running coupling.
I_fermion
    Integral of the fermionic correction to the PNJL thermodynamic potential.
I_boson
    Integral of the bosonic correction to the PNJL thermodynamic potential.
gcp_fermion_l_real
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single light quark flavor (real part).
gcp_fermion_l_imag
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single light quark flavor (imaginary part).
gcp_fermion_s_real
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (real part).
gcp_fermion_s_imag
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (imaginary part).
gcp_boson_real
    Perturbative bosonic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (real part).
gcp_boson_imag
    Perturbative bosonic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (imag part).
pressure
    Pertrubative correction to the pressure of a single quark flavor.
bdensity
    Pertrubative correction to the baryon density of a single quark flavor.
qnumber_cumulant
    Pertrubative correction to the quark number cumulant chi_q of a single 
    quark flavor. Based on Eq.29 of https://arxiv.org/pdf/2012.12894.pdf and 
    the subsequent inline definition.
sdensity
    Pertrubative correction to the entropy density of a single quark flavor.
"""


import math
import functools

import scipy.integrate

import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sigma_lattice


mass_hash = {
    'l': pnjl.thermo.gcp_sigma_lattice.Ml,
    's': pnjl.thermo.gcp_sigma_lattice.Ms,
}


@functools.lru_cache(maxsize=1024)
def alpha_s(T : float, mu : float) -> float:
    """### Description
    QCD running coupling.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    alpha_s : float
        Value of the running coupling.
    """

    NF = 3.0
    NC = pnjl.defaults.NC
    C = pnjl.defaults.C
    D = pnjl.defaults.D

    den1 = math.fsum([11.0 * NC, -2.0*NF])
    den2 = math.fsum([
        2.0*math.log(D),
        2.0*math.log(T),
        -2.0*math.log(C)
    ])
    den3 = math.fsum([((D*T)**2), -(C**2)])

    par = math.fsum([1.0/den2, -(C**2)/den3])

    return ((12.0*math.pi)/den1)*par


@functools.lru_cache(maxsize=1024)
def I_integrand_real(
    p: float, T: float, mu: float, phi_re: float, phi_im: float
) -> float:
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, 0.0, 1, "+"
    ).real
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, 0.0, 1, "-"
    ).real
    return p*(fp + fm)


@functools.lru_cache(maxsize=1024)
def I_integrand_imag(
    p: float, T: float, mu: float, phi_re: float, phi_im: float
) -> float:
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, 0.0, 1, "+"
    ).imag
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, 0.0, 1, "-"
    ).imag
    return p*(fp + fm)


@functools.lru_cache(maxsize=1024)
def I(
    T: float, mu: float, phi_re: float, phi_im: float, typ: str
) -> complex:
    integral_real, _ = scipy.integrate.quad(
        I_integrand_real, mass_hash[typ](T, mu), math.inf,
        args = (T, mu, phi_re, phi_im)
    )
    integral_imag, _ = scipy.integrate.quad(
        I_integrand_imag, mass_hash[typ](T, mu), math.inf,
        args = (T, mu, phi_re, phi_im)
    )
    return complex(integral_real, integral_imag)/(T**2)


@functools.lru_cache(maxsize=1024)
def gcp(
    T: float, mu: float, phi_re: float, phi_im: float, typ: str
) -> float:
    I_val = I(T, mu, phi_re, phi_im, typ)
    alpha = alpha_s(T, mu)
    I_val2 = I_val**2
    par_real = math.fsum([I_val.real/6.0, (1.0/(4.0*(math.pi**2)))*I_val2.real])
    return (8.0/math.pi)*alpha*(T**4)*par_real


@functools.lru_cache(maxsize=1024)
def pressure(
    T: float, mu: float, phi_re: float, phi_im: float, typ: str
) -> float:
    return -gcp(T, mu, phi_re, phi_im, typ)


@functools.lru_cache(maxsize=1024)
def bdensity(
    T: float, mu: float, phi_re: float, phi_im: float, typ: str
) -> float:
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
        phi_vec = [
            tuple([phi_re, phi_im])
            for _ in mu_vec
        ]
        p_vec = [
            coef*pressure(T, mu_el, phi_el[0], phi_el[1], typ)/3.0
            for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
        ]
        return math.fsum(p_vec)
    else:
        new_mu = math.fsum([mu, h])
        new_phi_re, new_phi_im = phi_re, phi_im
        return bdensity(T, new_mu, new_phi_re, new_phi_im, typ)


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant(
    rank: int, T: float, mu: float, phi_re: float, phi_im: float, typ: str
) -> float:
    if rank == 1:
        return 3.0 * bdensity(T, mu, phi_re, phi_im, typ)
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
            phi_vec = [
                tuple([phi_re, phi_im])
                for _ in mu_vec
            ]
            out_vec = [
                coef*qnumber_cumulant(rank-1, T, mu_el, phi_el[0], phi_el[1], typ)
                for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
            ]
            return math.fsum(out_vec)
        else:
            new_mu = math.fsum([mu, h])
            new_phi_re, new_phi_im = phi_re, phi_im
            return qnumber_cumulant(rank, T, new_mu, new_phi_re, new_phi_im, typ)


@functools.lru_cache(maxsize=1024)
def sdensity(
    T: float, mu: float, phi_re : float, phi_im : float, typ: str
) -> float:
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
        phi_vec = [
            tuple([phi_re, phi_im])
            for _ in T_vec
        ]
        p_vec = [
            coef*pressure(T_el, mu, phi_el[0], phi_el[1], typ)
            for T_el, coef, phi_el in zip(T_vec, deriv_coef, phi_vec)
        ]
        return math.fsum(p_vec)
    else:
        new_T = math.fsum([T, h])
        new_phi_re, new_phi_im = phi_re, phi_im
        return sdensity(new_T, mu, new_phi_re, new_phi_im, typ)