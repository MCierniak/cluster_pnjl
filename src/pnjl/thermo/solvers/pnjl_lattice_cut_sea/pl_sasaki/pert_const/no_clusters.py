"""### Description
Polyakov-loop solver

## Functions
"""


import tqdm
import math
import typing

import scipy.optimize

import pnjl.defaults
import pnjl.thermo.gcp_pl.sasaki
import pnjl.thermo.gcp_perturbative.const
import pnjl.thermo.gcp_pnjl.lattice_cut_sea


def phi_im(phi_re, phi_im_ratio):
    if phi_re >= 0.9999:
        return 0.0
    else:
        inner_sqrt = 2.0*math.sqrt(math.fsum([1.0, 2.0*phi_re])**3)/math.sqrt(3.0)
        inner_sum = math.fsum([4.0, phi_re])
        range = math.sqrt(math.fsum([-1.0, -phi_re*inner_sum, inner_sqrt]))
        return (phi_im_ratio*2.0 - 1.0)*range


def inv_phi_im(phi_re, phi_im):
    if phi_im == 0.0:
        return 0.0
    else:
        inner_sqrt = 2.0*math.sqrt(math.fsum([1.0, 2.0*phi_re])**3)/math.sqrt(3.0)
        inner_sum = math.fsum([4.0, phi_re])
        square_root = math.sqrt(math.fsum([-1.0, -phi_re*inner_sum, inner_sqrt]))
        return -math.fsum([-phi_im, -square_root])/(2.0*square_root)


def Polyakov_loop_inner(phi, T, mu):
    phiim = phi_im(phi[0], phi[1])
    MH = pnjl.thermo.gcp_pl.sasaki.M_H(phi[0], phiim)
    if MH < 0.0:
        return math.inf
    else:
        sigma = pnjl.thermo.gcp_pnjl.lattice_cut_sea.gcp_field(T, mu)
        gluon = pnjl.thermo.gcp_pl.sasaki.gcp(T, 3.0*mu, phi[0], phiim)
        sea_l = 2.0*pnjl.thermo.gcp_pnjl.lattice_cut_sea.gcp_sea_l(T, mu)
        sea_s = pnjl.thermo.gcp_pnjl.lattice_cut_sea.gcp_sea_s(T, mu)
        perturbative_l = 2.0*pnjl.thermo.gcp_perturbative.const.gcp(
            T, mu, phi[0], phiim, 'l'
        )
        perturbative_s = pnjl.thermo.gcp_perturbative.const.gcp(
            T, mu, phi[0], phiim, 's'
        )
        pnjl_l = 2.0*pnjl.thermo.gcp_pnjl.lattice_cut_sea.gcp_q_l_real(T, mu, phi[0], phiim)
        pnjl_s = pnjl.thermo.gcp_pnjl.lattice_cut_sea.gcp_q_s_real(T, mu, phi[0], phiim)
        return math.fsum([
            sigma, gluon,
            sea_l, pnjl_l, perturbative_l,
            sea_s, pnjl_s, perturbative_s
        ])


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


def pressure_single(T: float, muB: float, phi_re_0=1e-5, phi_im_0=2e-5, calc_phi=True):
    partial = list()
    phi_result = (phi_re_0, phi_im_0)
    if calc_phi:
        phi_result = Polyakov_loop(T, muB/3.0, phi_re_0, phi_im_0)
    pars = (T, muB/3.0, phi_result[0], phi_result[1])
    #Sigma pressure
    partial.append(
        pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_field(T, muB/3.0)/(T**4)
    )
    #Sea pressure
    lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_sea(T, muB/3.0, 'l')/(T**4)
    sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_sea(T, muB/3.0, 's')/(T**4)
    partial.append(lq_temp)
    partial.append(lq_temp)
    partial.append(sq_temp)
    #Gluon pressure
    partial.append(pnjl.thermo.gcp_pl.sasaki.pressure(*pars)/(T**4))
    #PNJL pressure
    lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_q(*pars, 'l')/(T**4)
    sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_q(*pars, 's')/(T**4)
    partial.append(lq_temp)
    partial.append(lq_temp)
    partial.append(sq_temp)
    #Perturbative pressure
    lq_temp = pnjl.thermo.gcp_perturbative.const.pressure(*pars, 'l')/(T**4)
    sq_temp = pnjl.thermo.gcp_perturbative.const.pressure(*pars, 's')/(T**4)
    partial.append(lq_temp)
    partial.append(lq_temp)
    partial.append(sq_temp)
    return phi_result[0], phi_result[1], math.fsum(partial), (*partial,)
    

def pressure_all(T, muB, phi_re=None, phi_im=None, label="Pressure"):
    if phi_re is None and phi_im is None:
        phi_re, phi_im = list(), list()
        pressure_full, pressure_partials = list(), list()
        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print(label)
        for T_el, muB_el in tqdm.tqdm(
            zip(T, muB), total=len(T), ncols=100
        ):
            phi_re_0, phi_im_0, temp_full, temp_partials = pressure_single(
                T_el, muB_el, phi_re_0, phi_im_0
            )
            phi_re.append(phi_re_0)
            phi_im.append(phi_im_0)
            pressure_full.append(temp_full)
            pressure_partials.append(temp_partials)
        return phi_re, phi_im, pressure_full, pressure_partials
    elif phi_re is not None and phi_im is not None:
        pressure_full, pressure_partials = list(), list()
        print(label)
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T, muB, phi_re, phi_im), total=len(T), ncols=100
        ):
            _, _, temp_full, temp_partials = pressure_single(
                T_el, muB_el, phi_re_el, phi_im_el, calc_phi=False
            )
            pressure_full.append(temp_full)
            pressure_partials.append(temp_partials)
        return pressure_full, pressure_partials
    else:
        raise ValueError("Coś nie pykło.")
    

def bdensity_single(T: float, muB: float, phi_re_0=1e-5, phi_im_0=2e-5, calc_phi=True):
    partial = list()
    phi_result = (phi_re_0, phi_im_0)
    if calc_phi:
        phi_result = Polyakov_loop(T, muB/3.0, phi_re_0, phi_im_0)
    pars = (T, muB/3.0, phi_result[0], phi_result[1])
    #Sigma bdensity
    partial.append(
        pnjl.thermo.gcp_pnjl.lattice_cut_sea.bdensity_field(T, muB/3.0)/(T**3)
    )
    #Sea bdensity
    lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.bdensity_sea(T, muB/3.0, 'l')/(T**3)
    sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.bdensity_sea(T, muB/3.0, 's')/(T**3)
    partial.append(lq_temp)
    partial.append(lq_temp)
    partial.append(sq_temp)
    #Gluon bdensity
    partial.append(pnjl.thermo.gcp_pl.sasaki.bdensity(*pars)/(T**3))
    #PNJL bdensity
    lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.bdensity_q(*pars, 'l')/(T**3)
    sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.bdensity_q(*pars, 's')/(T**3)
    partial.append(lq_temp)
    partial.append(lq_temp)
    partial.append(sq_temp)
    #Perturbative bdensity
    lq_temp = pnjl.thermo.gcp_perturbative.const.bdensity(*pars, 'l')/(T**3)
    sq_temp = pnjl.thermo.gcp_perturbative.const.bdensity(*pars, 's')/(T**3)
    partial.append(lq_temp)
    partial.append(lq_temp)
    partial.append(sq_temp)
    return phi_result[0], phi_result[1], math.fsum(partial), (*partial,)
    

def bdensity_all(T, muB, phi_re=None, phi_im=None, label="BDensity"):
    if phi_re is None and phi_im is None:
        phi_re, phi_im = list(), list()
        bdensity_full, bdensity_partials = list(), list()
        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print(label)
        for T_el, muB_el in tqdm.tqdm(
            zip(T, muB), total=len(T), ncols=100
        ):
            phi_re_0, phi_im_0, temp_full, temp_partials = bdensity_single(
                T_el, muB_el, phi_re_0, phi_im_0
            )
            phi_re.append(phi_re_0)
            phi_im.append(phi_im_0)
            bdensity_full.append(temp_full)
            bdensity_partials.append(temp_partials)
        return phi_re, phi_im, bdensity_full, bdensity_partials
    elif phi_re is not None and phi_im is not None:
        bdensity_full, bdensity_partials = list(), list()
        print(label)
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T, muB, phi_re, phi_im), total=len(T), ncols=100
        ):
            _, _, temp_full, temp_partials = bdensity_single(
                T_el, muB_el, phi_re_el, phi_im_el, calc_phi=False
            )
            bdensity_full.append(temp_full)
            bdensity_partials.append(temp_partials)
        return bdensity_full, bdensity_partials
    else:
        raise ValueError("Coś nie pykło.")


def sdensity_single(T: float, muB: float, phi_re_0=1e-5, phi_im_0=2e-5, calc_phi=True):
    partial = list()
    phi_result = (phi_re_0, phi_im_0)
    if calc_phi:
        phi_result = Polyakov_loop(T, muB/3.0, phi_re_0, phi_im_0)
    pars = (T, muB/3.0, phi_result[0], phi_result[1])
    #Sigma sdensity
    partial.append(
        pnjl.thermo.gcp_pnjl.lattice_cut_sea.sdensity_field(T, muB/3.0)/(T**3)
    )
    #Sea sdensity
    lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.sdensity_sea(T, muB/3.0, 'l')/(T**3)
    sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.sdensity_sea(T, muB/3.0, 's')/(T**3)
    partial.append(lq_temp)
    partial.append(lq_temp)
    partial.append(sq_temp)
    #Gluon sdensity
    partial.append(pnjl.thermo.gcp_pl.sasaki.sdensity(*pars)/(T**3))
    #PNJL sdensity
    lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.sdensity_q(*pars, 'l')/(T**3)
    sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.sdensity_q(*pars, 's')/(T**3)
    partial.append(lq_temp)
    partial.append(lq_temp)
    partial.append(sq_temp)
    #Perturbative sdensity
    lq_temp = pnjl.thermo.gcp_perturbative.const.sdensity(*pars, 'l')/(T**3)
    sq_temp = pnjl.thermo.gcp_perturbative.const.sdensity(*pars, 's')/(T**3)
    partial.append(lq_temp)
    partial.append(lq_temp)
    partial.append(sq_temp)
    return phi_result[0], phi_result[1], math.fsum(partial), (*partial,)


def sdensity_all(T, muB, phi_re=None, phi_im=None, label="SDensity"):
    if phi_re is None and phi_im is None:
        phi_re, phi_im = list(), list()
        sdensity_full, sdensity_partials = list(), list()
        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print(label)
        for T_el, muB_el in tqdm.tqdm(
            zip(T, muB), total=len(T), ncols=100
        ):
            phi_re_0, phi_im_0, temp_full, temp_partials = sdensity_single(
                T_el, muB_el, phi_re_0, phi_im_0
            )
            phi_re.append(phi_re_0)
            phi_im.append(phi_im_0)
            sdensity_full.append(temp_full)
            sdensity_partials.append(temp_partials)
        return phi_re, phi_im, sdensity_full, sdensity_partials
    elif phi_re is not None and phi_im is not None:
        sdensity_full, sdensity_partials = list(), list()
        print(label)
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T, muB, phi_re, phi_im), total=len(T), ncols=100
        ):
            _, _, temp_full, temp_partials = sdensity_single(
                T_el, muB_el, phi_re_el, phi_im_el, calc_phi=False
            )
            sdensity_full.append(temp_full)
            sdensity_partials.append(temp_partials)
        return sdensity_full, sdensity_partials
    else:
        raise ValueError("Coś nie pykło.")