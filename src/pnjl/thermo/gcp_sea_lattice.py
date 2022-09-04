"""Fermi-sea grandcanonical thermodynamic potential and associated functions.
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
V
    Sigma mean-field grandcanonical thermodynamic potential.
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

import scipy.integrate

import pnjl.thermo.distributions
import pnjl.defaults


def Tc(mu: float) -> float:
    """Pseudo-critical temperature ansatz.

    ### Parameters
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    Tc : float
        Value of the pseudocritical temperature in MeV for a given mu.
    """

    Tc0 = pnjl.defaults.default_Tc0
    kappa = pnjl.defaults.default_kappa

    return math.fsum([Tc0, -Tc0*kappa*((mu/Tc0)**2)])


def Delta_ls(T: float, mu: float) -> float:
    """LQCD-fit ansatz for the 2+1 Nf normalized chiral condensate.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    Delta_ls : float
        Value of the normalized chiral condensate for a given T and mu.
    """

    delta_T = pnjl.defaults.default_delta_T

    tanh_internal = math.fsum([T/delta_T, -Tc(mu)/delta_T])

    return math.fsum([0.5, -0.5*math.tanh(tanh_internal)])


def Ml(T: float, mu: float) -> float:
    """Mass of up / down quarks (ansatz).

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    Ml : float
        Quark mass in MeV.
    """
    
    M0 = pnjl.defaults.default_M0
    ml = pnjl.defaults.default_ml
    
    return math.fsum([M0*Delta_ls(T, mu), ml])


def Ms(T: float, mu: float) -> float:
    """Mass of the strange quarks (ansatz).

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    Ms : float
        Quark mass in MeV.
    """

    M0 = pnjl.defaults.default_M0
    ms = pnjl.defaults.default_ms
    
    return math.fsum([M0*Delta_ls(T, mu), ms])


def V(T: float, mu: float) -> float:
    """Sigma mean-field grandcanonical thermodynamic potential.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    V : float
        Mean-field value in MeV^4.
    """

    M0 = pnjl.defaults.default_M0
    Gs = pnjl.defaults.default_Gs

    return math.fsum([
        ((Delta_ls(T, mu)**2) * (M0**2)) / (4.0*Gs),
        -((Delta_ls(0.0, 0.0)**2) * (M0**2)) / (4.0*Gs)
    ])


def gcp_real_l(T: float, mu: float) -> float:
    """Fermi sea grandcanonical thermodynamic potential of a single light quark flavor.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    gcp : float
        Value of the thermodynamic potential in MeV^4.
    """

    Nc = pnjl.defaults.default_Nc
    Lambda = pnjl.defaults.default_Lambda

    def integrand(p):   

        mass = Ml(T, mu)
        mass0 = Ml(0.0, 0.0)

        energy = pnjl.thermo.distributions.En(p, mass)
        energy0 = pnjl.thermo.distributions.En(p, mass0)

        energy_norm = math.fsum([energy, -energy0])

        return (p**2)*energy_norm
    
    integral, error = scipy.integrate.quad(integrand, 0.0, Lambda)
    
    return (1.0/(math.pi**2))*(Nc/3.0)*integral


def gcp_real_s(T: float, mu: float) -> float:
    """Fermi sea grandcanonical thermodynamic potential of a single strange quark flavor.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    gcp : float
        Value of the thermodynamic potential in MeV^4.
    """

    Nc = pnjl.defaults.default_Nc
    Lambda = pnjl.defaults.default_Lambda

    def integrand(p):

        mass = Ms(T, mu)
        mass0 = Ms(0.0, 0.0)

        energy = pnjl.thermo.distributions.En(p, mass)
        energy0 = pnjl.thermo.distributions.En(p, mass0)

        energy_norm = math.fsum([energy, -energy0])

        return (p**2)*energy_norm
    
    integral, error = scipy.integrate.quad(integrand, 0.0, Lambda)
    
    return (1.0/(math.pi**2))*(Nc/3.0)*integral


gcp_hash = {
    'l' : gcp_real_l,
    's' : gcp_real_s,
    'sigma' : V
}


def pressure(T: float, mu: float, typ: str, no_sea: bool = True) -> float:
    """Fermi sea pressure of a single quark flavor.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    typ : string 
        Type of quark
            'l' : up / down quark
            's' : strange quark
            'sigma' : sigma mean-field
    no_sea : bool, optional
        No-sea approximation flag.

    ### Returns
    pressure : float
        Value of the thermodynamic pressure in MeV^4.
    """

    if no_sea:
        return 0.0
    else:
        return -gcp_hash[typ](T, mu)


def bdensity(T: float, mu: float, typ: str, no_sea: bool = True) -> float:
    """Fermi sea baryon density of a single quark flavor.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    typ : string 
        Type of quark
            'l' : up / down quark
            's' : strange quark
            'sigma' : sigma mean-field
    no_sea : bool, optional
        No-sea approximation flag.

    ### Returns
    bdensity : float
        Value of the thermodynamic baryon density in MeV^3.
    """

    if no_sea:
        return 0.0
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

            p_vec = [
                coef*pressure(T, mu_el, typ, no_sea=no_sea)/3.0
                for mu_el, coef in zip(mu_vec, deriv_coef)
            ]

            return math.fsum(p_vec)

        else:
            return bdensity(T, math.fsum([mu, h]), typ, no_sea=no_sea)


def qnumber_cumulant(rank: int, T: float, mu: float, typ: str, no_sea: bool = True) -> float:
    """Fermi sea quark number cumulant chi_q of a single quark flavor. Based on Eq.29 of
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
            'sigma' : sigma mean-field
    no_sea : bool, optional
        No-sea approximation flag.

    ### Returns
    qnumber_cumulant : float
        Value of the thermodynamic quark number cumulant in MeV^3.
    """

    if no_sea:
        return 0.0
    else:

        if rank == 1:
            return 3.0 * bdensity(T, mu, typ, no_sea=no_sea)
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
                    coef*qnumber_cumulant(rank-1, T, mu_el, typ, no_sea=no_sea)
                    for mu_el, coef in zip(mu_vec, deriv_coef)
                ]

                return math.fsum(out_vec)

            else:
                return qnumber_cumulant(rank, T, math.fsum([mu, h]), typ, no_sea=no_sea)


def sdensity(T: float, mu: float, typ: str, no_sea: bool = True):
    """Fermi sea entropy density of a single quark flavor.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    typ : string 
        Type of quark
            'l' : up / down quark
            's' : strange quark
            'sigma' : sigma mean-field
    no_sea : bool, optional
        No-sea approximation flag.

    ### Returns
    sdensity : float
        Value of the thermodynamic entropy density in MeV^3.
    """

    if no_sea:
        return 0.0
    else:

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
                coef*pressure(T_el, mu, typ, no_sea=no_sea)
                for T_el, coef in zip(T_vec, deriv_coef)
            ]

            return math.fsum(p_vec)

        else:
            return sdensity(math.fsum([T, h]), mu, typ, no_sea=no_sea)

