"""Model and A0, C0 values from Sasaki_2_PhysRevD_86_(2012)
Mg from Braaten_2_PhysRevD_45_(1992)
"""

import math

import scipy.integrate

import pnjl.thermo.gcp_perturbative.const


N = 3.0
NF = 3.0


A0 = (0.197*1000.0)**3
C0 = -((0.180*1000.0)**4)


def Mg(T: float, muB: float):
    g2 = 4.0*math.pi*pnjl.thermo.gcp_perturbative.const.alpha_s(T, muB)
    return math.fsum(
        [
            N*(T**2)*g2/9.0, NF*(T**2)*g2/18.0,
            NF*g2*(3.0/(math.pi**2))*(muB**2)/18.0
        ]
    )
    # return 0.0


def En(p: float, M: float) -> float:
    return math.sqrt((p**2) + (M**2))


def C1(phi_re: float, phi_im: float) -> float:
    return math.fsum([1.0, -9.0*(phi_re**2), -9.0*(phi_im**2)])


def C2(phi_re: float, phi_im: float) -> float:
    return math.fsum(
        [
            1.0, -27.0*(phi_re**2), 54.0*(phi_re**3),
            -27.0*(phi_im**2), -162.0*phi_re*(phi_im**2)
        ]
    )


def C3(phi_re: float, phi_im: float) -> float:
    return math.fsum(
        [
            -2.0, 27.0*(phi_re**2), -81.0*(phi_re**4), 27.0*(phi_im**2),
            -162.0*(phi_re**2)*(phi_im**2), -81.0*(phi_im**4)
        ]
    )


def C4(phi_re: float, phi_im: float) -> float:
    return math.fsum(
        [
            -2.0, 18.0*(phi_re**2), -108.0*(phi_re**3), 162.0*(phi_re**4),
            18.0*(phi_im**2), 324.0*phi_re*(phi_im**2),
            324.0*(phi_re**2)*(phi_im**2), 162.0*(phi_im**4)
        ]
    )


def C5(phi_re: float, phi_im: float) -> float:
    return C3(phi_re, phi_im)


def C6(phi_re: float, phi_im: float) -> float:
    return C2(phi_re, phi_im)


def C7(phi_re: float, phi_im: float) -> float:
    return C1(phi_re, phi_im)


def C8(phi_re: float, phi_im: float) -> float:
    return 1.0


Cn_hash = {
    1: C1,
    2: C2,
    3: C3,
    4: C4,
    5: C5,
    6: C6,
    7: C7,
    8: C8
}


def Cn(n: int, phi_re: float, phi_im: float) -> float:
    return Cn_hash[n](phi_re, phi_im)


def M_H(phi_re: float, phi_im: float) -> float:
    phi_re2 = phi_re**2
    phi_re3 = phi_re**3
    phi_re4 = phi_re**4
    phi_im2 = phi_im**2
    phi_im4 = phi_im**4
    return math.fsum(
        [
            1.0, -6.0*phi_im2, -3.0*phi_im4, -24.0*phi_im2*phi_re,
            -6.0*phi_re2, -6.0*phi_im2*phi_re2, 8.0*phi_re3, -3.0*phi_re4
        ]
    )


def gcp_phi(T: float, phi_re: float, phi_im: float) -> float:
    mh = M_H(phi_re, phi_im)
    if mh <= 0.0:
        return math.inf
    else:
        return -A0*T*math.log(mh)
    

def gcp_g_inner(p: float, T: float, muB: float, phi_re: float, phi_im: float) -> float:
    logterm = 0.0
    for i in range(8):
        n = i + 1
        en = En(p, Mg(T, muB))
        logterm += Cn(n, phi_re, phi_im)*math.exp(-float(n)*en/T)
    if logterm <= -1.0:
        return math.inf
    else:
        return (p**2)*math.log1p(logterm)


def gcp_g(T: float, muB: float, phi_re: float, phi_im: float) -> float:
    integral, _ = scipy.integrate.quad(
        gcp_g_inner, 0.0, math.inf,
        args = (T, muB, phi_re, phi_im)
    )
    return (T/(math.pi**2))*integral


def gcp(T: float, muB: float, phi_re: float, phi_im: float) -> float:
    omega_g = gcp_g(T, muB, phi_re, phi_im)
    omega_phi = gcp_phi(T, phi_re, phi_im)
    return math.fsum([omega_g, omega_phi, C0])


def pressure(T: float, muB: float, phi_re: float, phi_im: float) -> float:
    """### Description
    Polyakov-loop pressure.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.

    ### Returns
    pressure : float
        Value of the thermodynamic pressure in MeV^4.
    """
    return -gcp(T, muB, phi_re, phi_im)


def bdensity(T: float, muB: float, phi_re : float, phi_im : float) -> float:
    """### Description
    Polyakov-loop baryon density.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.

    ### Returns
    bdensity : float
        Value of the thermodynamic baryon density in MeV^3.
    """
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
        p_vec = [
            coef*pressure(T, muB_el, phi_el[0], phi_el[1])
            for muB_el, coef, phi_el in zip(muB_vec, deriv_coef, phi_vec)
        ]
        return math.fsum(p_vec)
    else:

        new_muB = math.fsum([muB, h])
        new_phi_re, new_phi_im = phi_re, phi_im
        return bdensity(T, new_muB, new_phi_re, new_phi_im)


def qnumber_cumulant(rank: int, T: float, muB: float, phi_re : float, phi_im : float) -> float:
    """### Description
    Polyakov-loop quark number cumulant chi_q. Based on Eq.29 of
    https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline definition.

    ### Parameters
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

    ### Returns
    qnumber_cumulant : float
        Value of the thermodynamic quark number cumulant in MeV^3.
    """
    if rank == 1:
        return bdensity(T, muB, phi_re, phi_im)
    else:
        h = 1e-2
        if math.fsum([muB, -2*h]) > 0.0:
            mu_vec = [
                math.fsum([muB, 2*h]), math.fsum([muB, h]),
                math.fsum([muB, -h]), math.fsum([muB, -2*h])
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
                coef*qnumber_cumulant(rank-1, T, mu_el, phi_el[0], phi_el[1])
                for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
            ]
            return math.fsum(out_vec)
        else:
            new_mu = math.fsum([muB, h])
            new_phi_re, new_phi_im = phi_re, phi_im
            return qnumber_cumulant(rank, T, new_mu, new_phi_re, new_phi_im)


def sdensity(T: float, mu: float, phi_re : float, phi_im : float) -> float:
    """### Description
    Polyakov-loop entropy density.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.

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
        phi_vec = [
            tuple([phi_re, phi_im])
            for _ in T_vec
        ]
        p_vec = [
            coef*pressure(T_el, mu, phi_el[0], phi_el[1])
            for T_el, coef, phi_el in zip(T_vec, deriv_coef, phi_vec)
        ]
        return math.fsum(p_vec)
    else:
        new_T = math.fsum([T, h])
        new_phi_re, new_phi_im = phi_re, phi_im
        return sdensity(new_T, mu, new_phi_re, new_phi_im)