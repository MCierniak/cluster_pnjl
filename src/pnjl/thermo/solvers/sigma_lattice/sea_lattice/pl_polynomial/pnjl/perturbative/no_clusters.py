"""### Description
Polyakov-loop solver

## Functions
"""


import math
import typing
import functools

import scipy.optimize

import pnjl.defaults
import pnjl.thermo.gcp_pnjl
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_perturbative
import pnjl.thermo.gcp_sigma_lattice
import pnjl.thermo.gcp_pl.polynomial


def Polyakov_loop_inner(phi, T, mu):
    sigma = pnjl.thermo.gcp_sigma_lattice.gcp(T, mu)
    gluon = pnjl.thermo.gcp_pl.polynomial.U(T, phi[0], phi[1])
    sea_l = 2.0*pnjl.thermo.gcp_sea_lattice.gcp_l(T, mu)
    sea_s = pnjl.thermo.gcp_sea_lattice.gcp_s(T, mu)
    perturbative_l = 2.0*pnjl.thermo.gcp_perturbative.gcp(
        T, mu, phi[0], phi[1]
    )
    perturbative_s = pnjl.thermo.gcp_perturbative.gcp(
        T, mu, phi[0], phi[1]
    )
    pnjl_l = 2.0*pnjl.thermo.gcp_pnjl.gcp_l_real(T, mu, phi[0], phi[1])
    pnjl_s = pnjl.thermo.gcp_pnjl.gcp_s_real(T, mu, phi[0], phi[1])
    return math.fsum([
        sigma, gluon,
        sea_l, pnjl_l, perturbative_l,
        sea_s, pnjl_s, perturbative_s
    ])


@functools.lru_cache(maxsize=1024)
def Polyakov_loop(
    T: float, mu: float, phi_re0: float, phi_im0: float
) -> typing.Tuple[float, float]:
    """### Description
    _summary_
    
    ### Prameters
    T : float
        _description_
    mu : float
        _description_
    phi_re0 : float
        _description_
    phi_im0 : float
        _description_
    
    ### Returns
    calc_PL : Tuple[float, float]
        _description_
    """

    omega_result = scipy.optimize.dual_annealing(
        Polyakov_loop_inner,
        bounds=[[0.0, 3.0], [-3.0, 3.0]],
        args=(T, mu),
        x0=[phi_re0, phi_im0],
        maxiter=20,
        seed=1234
    )

    return tuple([omega_result.x[0], omega_result.x[1]])