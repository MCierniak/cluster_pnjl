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
import pnjl.thermo.gcp_pl_lo
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_perturbative
import pnjl.thermo.gcp_sigma_lattice


def phi_im(phi_re, phi_im_ratio):
    if phi_re == 1.0:
        return 0.0
    else:
        inner_sqrt = 2.0*math.sqrt(math.fsum([1.0, 2.0*phi_re])**3)/math.sqrt(3.0)
        inner_sum = math.fsum([4.0, phi_re])
        range = math.sqrt(math.fsum([-1.0, -phi_re*inner_sum, inner_sqrt]))
        return (phi_im_ratio*2.0 - 1.0)*range


def inv_phi_im(phi_re, phi_im):
    inner_sqrt = 2.0*math.sqrt(math.fsum([1.0, 2.0*phi_re])**3)/math.sqrt(3.0)
    inner_sum = math.fsum([4.0, phi_re])
    square_root = math.sqrt(math.fsum([-1.0, -phi_re*inner_sum, inner_sqrt]))
    return -math.fsum([-phi_im, -square_root])/(2.0*square_root)


def Polyakov_loop_inner(phi, T, mu):
    phiim = phi_im(phi[0], phi[1])
    MH = pnjl.thermo.gcp_pl_lo.M_H(phi[0], phiim)
    if MH < 0.0:
        return math.inf
    else:
        sigma = pnjl.thermo.gcp_sigma_lattice.gcp(T, mu)
        gluon = pnjl.thermo.gcp_pl_lo.U(T, phi[0], phiim)
        sea_l = 2.0*pnjl.thermo.gcp_sea_lattice.gcp_l(T, mu)
        sea_s = pnjl.thermo.gcp_sea_lattice.gcp_s(T, mu)
        perturbative_l = 2.0*pnjl.thermo.gcp_perturbative.gcp(
            T, mu, phi[0], phiim
        )
        perturbative_s = pnjl.thermo.gcp_perturbative.gcp(
            T, mu, phi[0], phiim
        )
        pnjl_l = 2.0*pnjl.thermo.gcp_pnjl.gcp_l_real(T, mu, phi[0], phiim)
        pnjl_s = pnjl.thermo.gcp_pnjl.gcp_s_real(T, mu, phi[0], phiim)
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
    bnds = ((0.0, 1.0), (0.0, 1.0))
    omega_result = scipy.optimize.dual_annealing(
        Polyakov_loop_inner,
        bounds=bnds,
        args=(T, mu),
        x0=[phi_re0, inv_phi_im(phi_re0, phi_im0)],
        maxiter=20,
        seed=2020202020,
        minimizer_kwargs={
            "method": "Nelder-Mead",
            "bounds": bnds
        }
    )
    return tuple([omega_result.x[0], phi_im(omega_result.x[0], omega_result.x[1])])