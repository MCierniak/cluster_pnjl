import math
import functools

import scipy.integrate

import pnjl.thermo.distributions


NC = 3.0

M_L_VAC = 400.0
M_S_VAC = 550.0

LAMBDA = 300.0

TC0 = 154.
KAPPA = 0.012
DELTA_T = 26.

M0 = 395.0
ML = 5.0
MS = 155.0

GS = 10.08e-6

T_MOTT_0 = 160.0


@functools.lru_cache(maxsize=1024)
def T_Mott(mu: float) -> float:
    return math.fsum([T_MOTT_0, -T_MOTT_0 * KAPPA * (((3.0 * mu) / T_MOTT_0) ** 2)])


@functools.lru_cache(maxsize=1024)
def Tc(mu: float) -> float:
    return math.fsum([TC0, -TC0 * KAPPA * (((3.0 * mu) / TC0) ** 2)])


@functools.lru_cache(maxsize=1024)
def Delta_ls(T: float, mu: float) -> float:
    tanh_internal = math.fsum([T/DELTA_T, -Tc(mu)/DELTA_T])
    return math.fsum([0.5, -0.5*math.tanh(tanh_internal)])


@functools.lru_cache(maxsize=1024)
def Ml(T: float, mu: float) -> float:
    if T < T_Mott(mu):
        return M_L_VAC
    else:
        return math.fsum([M0*Delta_ls(T, mu), ML])


@functools.lru_cache(maxsize=1024)
def Ms(T: float, mu: float) -> float:
    if T < T_Mott(mu):
        return M_S_VAC
    else:
        return math.fsum([M0*Delta_ls(T, mu), MS])
    

@functools.lru_cache(maxsize=1024)
def En(p: float, mass: float) -> float:
    body = math.fsum([p**2, mass**2])
    return math.sqrt(body)


@functools.lru_cache(maxsize=1024)
def gcp_sea_integrand(p: float, mass: float) -> float:
    en = En(p, mass)
    inner_sum = math.fsum([mass**2, 2.0*(p**2)])
    log_sum = math.fsum([en, -p])
    return math.fsum([p*en*inner_sum, (mass**4)*math.log(log_sum)])/8.0


@functools.lru_cache(maxsize=1024)
def gcp_sea_l(T: float, mu: float) -> float:
    mass = Ml(T, mu)
    mass0 = Ml(0.0, 0.0)
    one = gcp_sea_integrand(LAMBDA, mass)
    two = gcp_sea_integrand(0.0, mass)
    three = gcp_sea_integrand(LAMBDA, mass0)
    four = gcp_sea_integrand(0.0, mass0)
    return (1.0/(math.pi**2))*(NC/3.0)*math.fsum([one, -two, -three, four])


@functools.lru_cache(maxsize=1024)
def gcp_sea_s(T: float, mu: float) -> float:
    mass = Ms(T, mu)
    mass0 = Ms(0.0, 0.0)
    one = gcp_sea_integrand(LAMBDA, mass)
    two = gcp_sea_integrand(0.0, mass)
    three = gcp_sea_integrand(LAMBDA, mass0)
    four = gcp_sea_integrand(0.0, mass0)
    return (1.0/(math.pi**2))*(NC/3.0)*math.fsum([one, -two, -three, four])


sea_hash = {
    'l' : gcp_sea_l,
    's' : gcp_sea_s
}


@functools.lru_cache(maxsize=1024)
def pressure_sea(T: float, mu: float, typ: str) -> float:
    # return -sea_hash[typ](T, mu)
    return 0.0


@functools.lru_cache(maxsize=1024)
def bdensity_sea(T: float, mu: float, typ: str) -> float:
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
        p_vec = [
            coef*pressure_sea(T, mu_el, typ)/3.0
            for mu_el, coef in zip(mu_vec, deriv_coef)
        ]
        return math.fsum(p_vec)
    else:
        return bdensity_sea(T, math.fsum([mu, h]), typ)


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant_sea(rank: int, T: float, mu: float, typ: str) -> float:
    if rank == 1:
        return 3.0 * bdensity_sea(T, mu, typ)
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
            out_vec = [
                coef*qnumber_cumulant_sea(rank-1, T, mu_el, typ)
                for mu_el, coef in zip(mu_vec, deriv_coef)
            ]
            return math.fsum(out_vec)
        else:
            return qnumber_cumulant_sea(rank, T, math.fsum([mu, h]), typ)


@functools.lru_cache(maxsize=1024)
def sdensity_sea(T: float, mu: float, typ: str) -> float:
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
        p_vec = [
            coef*pressure_sea(T_el, mu, typ)
            for T_el, coef in zip(T_vec, deriv_coef)
        ]
        return math.fsum(p_vec)
    else:
        return sdensity_sea(math.fsum([T, h]), mu, typ)
    

@functools.lru_cache(maxsize=1024)
def gcp_q_l_integrand_real(p: float, T: float, mu: float, phi_re: float, phi_im: float) -> float:
    mass = Ml(T, mu)
    energy = En(p, mass)
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, mass, 1, '+'
    )
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, mass, 1, '-'
    )
    return ((p**4)/energy)*math.fsum([fp.real, fm.real])


@functools.lru_cache(maxsize=1024)
def gcp_q_l_integrand_imag(p: float, T: float, mu: float, phi_re: float, phi_im: float) -> float:
    mass = Ml(T, mu)
    energy = En(p, mass)
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, mass, 1, '+'
    )
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, mass, 1, '-'
    )
    return ((p**4)/energy)*math.fsum([fp.imag, fm.imag])


@functools.lru_cache(maxsize=1024)
def gcp_q_s_integrand_real(p: float, T: float, mu: float, phi_re: float, phi_im: float) -> float:
    mass = Ms(T, mu)
    energy = En(p, mass)
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, mass, 1, '+'
    )
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, mass, 1, '-'
    )
    return ((p**4)/energy)*math.fsum([fp.real, fm.real])


@functools.lru_cache(maxsize=1024)
def gcp_q_s_integrand_imag(p: float, T: float, mu: float, phi_re: float, phi_im: float) -> float:
    mass = Ms(T, mu)
    energy = En(p, mass)
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, mass, 1, '+'
    )
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, mass, 1, '-'
    )
    return ((p**4)/energy)*math.fsum([fp.imag, fm.imag])


@functools.lru_cache(maxsize=1024)
def gcp_q_l_real(T : float, mu : float, phi_re : float, phi_im : float) -> float:
    integral, _ = scipy.integrate.quad(
        gcp_q_l_integrand_real, 0.0, math.inf, args=(T, mu, phi_re, phi_im)
    )
    return -(NC/3.0)*(1.0/(math.pi**2))*integral


@functools.lru_cache(maxsize=1024)
def gcp_q_l_imag(T : float, mu : float, phi_re : float, phi_im : float) -> float:
    integral, _ = scipy.integrate.quad(
        gcp_q_l_integrand_imag, 0.0, math.inf, args=(T, mu, phi_re, phi_im)
    )
    return -(NC/3.0)*(1.0/(math.pi**2))*integral


@functools.lru_cache(maxsize=1024)
def gcp_q_s_real(T : float, mu : float, phi_re : float, phi_im : float) -> float:
    integral, _ = scipy.integrate.quad(
        gcp_q_s_integrand_real, 0.0, math.inf, args=(T, mu, phi_re, phi_im)
    )
    return -(NC/3.0)*(1.0/(math.pi**2))*integral


@functools.lru_cache(maxsize=1024)
def gcp_q_s_imag(T : float, mu : float, phi_re : float, phi_im : float) -> float:
    integral, error = scipy.integrate.quad(
        gcp_q_s_integrand_imag, 0.0, math.inf, args=(T, mu, phi_re, phi_im)
    )
    return -(NC/3.0)*(1.0/(math.pi**2))*integral


q_hash = {
    'l': gcp_q_l_real,
    's': gcp_q_s_real
}


@functools.lru_cache(maxsize=1024)
def pressure_q(T: float, mu: float, phi_re: float, phi_im: float, typ: str) -> float:
    return -q_hash[typ](T, mu, phi_re, phi_im)


@functools.lru_cache(maxsize=1024)
def bdensity_q(T: float, mu: float, phi_re: float, phi_im: float, typ: str) -> float:
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
            coef*pressure_q(T, mu_el, phi_el[0], phi_el[1], typ)/3.0
            for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
        ]
        return math.fsum(p_vec)
    else:
        new_mu = math.fsum([mu, h])
        new_phi_re, new_phi_im = phi_re, phi_im
        return bdensity_q(T, new_mu, new_phi_re, new_phi_im, typ)


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant_q(rank: int, T: float, mu: float, phi_re: float, phi_im: float, typ: str) -> float:
    if rank == 1:
        return 3.0 * bdensity_q(T, mu, phi_re, phi_im, typ)
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
                coef*qnumber_cumulant_q(rank-1, T, mu_el, phi_el[0], phi_el[1], typ)
                for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
            ]
            return math.fsum(out_vec)
        else:
            new_mu = math.fsum([mu, h])
            new_phi_re, new_phi_im = phi_re, phi_im
            return qnumber_cumulant_q(rank, T, new_mu, new_phi_re, new_phi_im, typ)


@functools.lru_cache(maxsize=1024)
def sdensity_q(T: float, mu: float, phi_re : float, phi_im : float, typ: str) -> float:
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
            coef*pressure_q(T_el, mu, phi_el[0], phi_el[1], typ)
            for T_el, coef, phi_el in zip(T_vec, deriv_coef, phi_vec)
        ]
        return math.fsum(p_vec)
    else:
        new_T = math.fsum([T, h])
        new_phi_re, new_phi_im = phi_re, phi_im
        return sdensity_q(new_T, mu, new_phi_re, new_phi_im, typ)
    

@functools.lru_cache(maxsize=1024)
def gcp_field(T: float, mu: float) -> float:
    sigma = Ml(T, mu) - ML
    sigma0 = Ml(0.0, 0.0) - ML
    return -math.fsum([(sigma**2)/(4.0*GS), -(sigma0**2)/(4.0*GS)])


@functools.lru_cache(maxsize=1024)
def pressure_field(T: float, mu: float) -> float:
    # return -gcp_field(T, mu)
    return 0.0


@functools.lru_cache(maxsize=1024)
def bdensity_field(T: float, mu: float) -> float:
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
        p_vec = [
            coef*pressure_field(T, mu_el)/3.0
            for mu_el, coef in zip(mu_vec, deriv_coef)
        ]
        return math.fsum(p_vec)
    else:
        return bdensity_field(T, math.fsum([mu, h]))


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant_field(rank: int, T: float, mu: float) -> float:
    if rank == 1:
        return 3.0 * bdensity_field(T, mu)
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
            out_vec = [
                coef*qnumber_cumulant_field(rank-1, T, mu_el)
                for mu_el, coef in zip(mu_vec, deriv_coef)
            ]
            return math.fsum(out_vec)
        else:
            return qnumber_cumulant_field(rank, T, math.fsum([mu, h]))


@functools.lru_cache(maxsize=1024)
def sdensity_field(T: float, mu: float) -> float:
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
        p_vec = [
            coef*pressure_field(T_el, mu)
            for T_el, coef in zip(T_vec, deriv_coef)
        ]
        return math.fsum(p_vec)
    else:
        return sdensity_field(math.fsum([T, h]), mu)