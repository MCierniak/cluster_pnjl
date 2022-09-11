"""### Description
Sigma mean-field grandcanonical thermodynamic potential and associated functions.
Lattice-fit version from https://arxiv.org/pdf/2012.12894.pdf .

### Functions
Tc
    Pseudo-critical temperature ansatz.
Delta_ls
    LQCD-fit ansatz for the 2+1 Nf normalized chiral condensate.
Ml
    Mass of up / down quarks (ansatz).
Ms
    Mass of the strange quarks (ansatz).
gcp
    Sigma mean-field grandcanonical thermodynamic potential.
pressure
    Sigma mean-field pressure of a single quark flavor.
bdensity
    Sigma mean-field baryon density of a single quark flavor.
qnumber_cumulant
    Sigma mean-field quark number cumulant chi_q of a single quark flavor.
bnumber_cumulant
    (Not implemented yet)
sdensity
    Sigma mean-field entropy density of a single quark flavor.
"""


import math
import functools

import pnjl.defaults


@functools.lru_cache(maxsize=1024)
def Tc(mu: float) -> float:
    """### Description
    Pseudo-critical temperature ansatz.

    ### Parameters
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    Tc : float
        Value of the pseudocritical temperature in MeV for a given mu.
    """

    TC0 = pnjl.defaults.TC0
    KAPPA = pnjl.defaults.KAPPA

    return math.fsum([TC0, -TC0*KAPPA*((mu/TC0)**2)])


@functools.lru_cache(maxsize=1024)
def Delta_ls(T: float, mu: float) -> float:
    """### Description
    LQCD-fit ansatz for the 2+1 Nf normalized chiral condensate.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    Delta_ls : float
        Value of the normalized chiral condensate for a given T and mu.
    """

    DELTA_T = pnjl.defaults.DELTA_T

    tanh_internal = math.fsum([T/DELTA_T, -Tc(mu)/DELTA_T])

    return math.fsum([0.5, -0.5*math.tanh(tanh_internal)])


@functools.lru_cache(maxsize=1024)
def Ml(T: float, mu: float) -> float:
    """### Description
    Mass of up / down quarks (ansatz).

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    Ml : float
        Quark mass in MeV.
    """
    
    M0 = pnjl.defaults.M0
    ML = pnjl.defaults.ML
    
    return math.fsum([M0*Delta_ls(T, mu), ML])


@functools.lru_cache(maxsize=1024)
def Ms(T: float, mu: float) -> float:
    """### Description
    Mass of the strange quarks (ansatz).

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    Ms : float
        Quark mass in MeV.
    """

    M0 = pnjl.defaults.M0
    MS = pnjl.defaults.MS
    
    return math.fsum([M0*Delta_ls(T, mu), MS])


@functools.lru_cache(maxsize=1024)
def gcp(T: float, mu: float) -> float:
    """### Description
    Sigma mean-field grandcanonical thermodynamic potential.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    V : float
        Mean-field value in MeV^4.
    """

    M0 = pnjl.defaults.M0
    GS = pnjl.defaults.GS

    if pnjl.defaults.NO_SIGMA:
        return 0.0
    else:
        return math.fsum([
            ((Delta_ls(T, mu)**2)*(M0**2))/(4.0*GS),
            -((Delta_ls(0.0, 0.0)**2)*(M0**2))/(4.0*GS)
        ])


@functools.lru_cache(maxsize=1024)
def pressure(T: float, mu: float) -> float:
    """### Description
    Sigma mean-field pressure.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    pressure : float
        Value of the thermodynamic pressure in MeV^4.
    """

    return -gcp(T, mu)


@functools.lru_cache(maxsize=1024)
def bdensity(T: float, mu: float) -> float:
    """### Description
    Sigma mean-field baryon density.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    bdensity : float
        Value of the thermodynamic baryon density in MeV^3.
    """

    h = 1e-2

    if math.fsum([mu, -2*h]) > 0.0:

        mu_vec = [
            math.fsum(mu, 2*h), math.fsum(mu, h),
            math.fsum(mu, -h), math.fsum(mu, -2*h)
        ]
        deriv_coef = [
            -1.0/(12.0*h), 8.0/(12.0*h),
            -8.0/(12.0*h), 1.0/(12.0*h)
        ]

        p_vec = [
            coef*pressure(T, mu_el)/3.0
            for mu_el, coef in zip(mu_vec, deriv_coef)
        ]

        return math.fsum(p_vec)

    else:
        return bdensity(T, math.fsum([mu, h]))


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant(rank: int, T: float, mu: float) -> float:
    """### Description
    Sigma mean-field quark number cumulant chi_q. Based on Eq.29 of 
    https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline definition.

    ### Parameters
    rank : int
        Cumulant rank. Rank 1 equals to 3 times the baryon density.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    qnumber_cumulant : float
        Value of the thermodynamic quark number cumulant in MeV^3.
    """

    if rank == 1:
        return 3.0 * bdensity(T, mu)
    else:

        h = 1e-2

        if math.fsum([mu, -2*h]) > 0.0:

            mu_vec = [
                math.fsum(mu, 2*h), math.fsum(mu, h),
                math.fsum(mu, -h), math.fsum(mu, -2*h)
            ]
            deriv_coef = [
                -1.0/(12.0*h), 8.0/(12.0*h),
                -8.0/(12.0*h), 1.0/(12.0*h)
            ]

            out_vec = [
                coef*qnumber_cumulant(rank-1, T, mu_el)
                for mu_el, coef in zip(mu_vec, deriv_coef)
            ]

            return math.fsum(out_vec)

        else:
            return qnumber_cumulant(rank, T, math.fsum([mu, h]))


@functools.lru_cache(maxsize=1024)
def sdensity(T: float, mu: float) -> float:
    """### Description
    Sigma mean-field entropy density.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    sdensity : float
        Value of the thermodynamic entropy density in MeV^3.
    """

    h = 1e-2

    if math.fsum([T, -2*h]) > 0.0:

        T_vec = [
            math.fsum(T, 2*h), math.fsum(T, h),
            math.fsum(T, -h), math.fsum(T, -2*h)
        ]
        deriv_coef = [
            -1.0/(12.0*h), 8.0/(12.0*h),
            -8.0/(12.0*h), 1.0/(12.0*h)
        ]

        p_vec = [
            coef*pressure(T_el, mu)
            for T_el, coef in zip(T_vec, deriv_coef)
        ]

        return math.fsum(p_vec)

    else:
        return sdensity(math.fsum([T, h]), mu)