"""### Description
Fermi-sea grandcanonical thermodynamic potential and associated functions.
Lattice-fit version from https://arxiv.org/pdf/2012.12894.pdf .

### Functions
gcp_real_l
    Fermi sea grandcanonical thermodynamic potential of a single light quark flavor.
gcp_real_s
    Fermi sea grandcanonical thermodynamic potential of a single strange quark flavor.
pressure
    Fermi sea pressure of a single quark flavor.
bdensity
    Fermi sea baryon density of a single quark flavor.
qnumber_cumulant
    Fermi sea quark number cumulant chi_q of a single quark flavor.
bnumber_cumulant
    (Not implemented yet)
sdensity
    Fermi sea entropy density of a single quark flavor.
"""


import math
import functools

import scipy.integrate

import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sigma_lattice


@functools.lru_cache(maxsize=1024)
def gcp_l_integrand(p: float, mass: float) -> float:   

    M_L_VAC = pnjl.defaults.M_L_VAC

    energy = pnjl.thermo.distributions.En(p, mass)
    energy0 = pnjl.thermo.distributions.En(p, M_L_VAC)

    energy_norm = math.fsum([energy, -energy0])

    return (p**2)*energy_norm


@functools.lru_cache(maxsize=1024)
def gcp_l(T: float, mu: float) -> float:
    """### Description
    Fermi sea grandcanonical thermodynamic potential of a single light quark 
    flavor.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    gcp : float
        Value of the thermodynamic potential in MeV^4.
    """

    NC = pnjl.defaults.NC
    LAMBDA = pnjl.defaults.LAMBDA

    mass = pnjl.thermo.gcp_sigma_lattice.Ml(T, mu)
    
    integral, error = scipy.integrate.quad(gcp_l_integrand, 0.0, LAMBDA, args=(mass,))
    
    return (1.0/(math.pi**2))*(NC/3.0)*integral


@functools.lru_cache(maxsize=1024)
def gcp_s_integrand(p: float, mass: float) -> float:   

    M_S_VAC = pnjl.defaults.M_S_VAC

    energy = pnjl.thermo.distributions.En(p, mass)
    energy0 = pnjl.thermo.distributions.En(p, M_S_VAC)

    energy_norm = math.fsum([energy, -energy0])

    return (p**2)*energy_norm


@functools.lru_cache(maxsize=1024)
def gcp_s(T: float, mu: float) -> float:
    """### Description
    Fermi sea grandcanonical thermodynamic potential of a single strange quark 
    flavor.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    gcp : float
        Value of the thermodynamic potential in MeV^4.
    """

    NC = pnjl.defaults.NC
    LAMBDA = pnjl.defaults.LAMBDA

    mass = pnjl.thermo.gcp_sigma_lattice.Ms(T, mu)
    
    integral, error = scipy.integrate.quad(gcp_s_integrand, 0.0, LAMBDA, args=(mass,))
    
    return (1.0/(math.pi**2))*(NC/3.0)*integral


gcp_hash = {
    'l' : gcp_l,
    's' : gcp_s
}


@functools.lru_cache(maxsize=1024)
def pressure(T: float, mu: float, typ: str) -> float:
    """### Description
    Fermi sea pressure of a single quark flavor.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    typ : string 
        Type of quark
            'l' : up / down quark
            's' : strange quark

    ### Returns
    pressure : float
        Value of the thermodynamic pressure in MeV^4.
    """

    if pnjl.defaults.NO_SEA:
        return 0.0
    else:
        return -gcp_hash[typ](T, mu)


@functools.lru_cache(maxsize=1024)
def bdensity(T: float, mu: float, typ: str) -> float:
    """### Description
    Fermi sea baryon density of a single quark flavor.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    typ : string 
        Type of quark
            'l' : up / down quark
            's' : strange quark

    ### Returns
    bdensity : float
        Value of the thermodynamic baryon density in MeV^3.
    """

    if pnjl.defaults.NO_SEA:
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

            p_vec = [
                coef*pressure(T, mu_el, typ)/3.0
                for mu_el, coef in zip(mu_vec, deriv_coef)
            ]

            return math.fsum(p_vec)

        else:
            return bdensity(T, math.fsum([mu, h]), typ)


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant(rank: int, T: float, mu: float, typ: str) -> float:
    """### Description
    Fermi sea quark number cumulant chi_q of a single quark flavor. Based on Eq.29 of
    https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline definition.

    ### Parameters
    rank : int
        Cumulant rank. Rank 1 equals to 3 times the baryon density.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    typ : string 
        Type of quark
            'l' : up / down quark
            's' : strange quark

    ### Returns
    qnumber_cumulant : float
        Value of the thermodynamic quark number cumulant in MeV^3.
    """

    if pnjl.defaults.NO_SEA:

        return 0.0
        
    else:

        if rank == 1:
            return 3.0 * bdensity(T, mu, typ)
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
                    coef*qnumber_cumulant(rank-1, T, mu_el, typ)
                    for mu_el, coef in zip(mu_vec, deriv_coef)
                ]

                return math.fsum(out_vec)

            else:
                return qnumber_cumulant(rank, T, math.fsum([mu, h]), typ)


@functools.lru_cache(maxsize=1024)
def sdensity(T: float, mu: float, typ: str) -> float:
    """### Description
    Fermi sea entropy density of a single quark flavor.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    typ : string 
        Type of quark
            'l' : up / down quark
            's' : strange quark

    ### Returns
    sdensity : float
        Value of the thermodynamic entropy density in MeV^3.
    """

    if pnjl.defaults.NO_SEA:

        return 0.0

    else:

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
                coef*pressure(T_el, mu, typ)
                for T_el, coef in zip(T_vec, deriv_coef)
            ]

            return math.fsum(p_vec)

        else:
            return sdensity(math.fsum([T, h]), mu, typ)