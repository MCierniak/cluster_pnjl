#TODO:
#1. make BDensity and SDensity routines.
#2. add hexa-diquark cluster
#3. consider d phi / d (mu/T).
#4. consider a mu dependent PL potential of https://arxiv.org/abs/1207.4890
#5. return to sigma/phi derivation using a momentum-dependent quark mass (check Omega_Delta!).
#6. change the pl potential to this one https://arxiv.org/pdf/1307.5958.pdf

import matplotlib.patches
import matplotlib.pyplot
import scipy.optimize
import numpy
import math
import tqdm
import csv

import pnjl.thermo.gcp_cluster.bound_step_continuum_arccos_cos as cluster
import pnjl.thermo.gcp_cluster.bound_step_continuum_step
import pnjl.thermo.gcp_pl_polynomial
import pnjl.thermo.gcp_perturbative
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_quark
import pnjl.defaults
import utils

import warnings
warnings.filterwarnings("ignore")

def cluster_thermo(T, mu, phi_re, phi_im, M, Mth, dMdmu, dMthdmu, dMdT, dMthdT, a, dx):
    Pres = [
        pnjl.thermo.gcp_cluster.bound_step_continuum_step.pressure(
            T_el,                               #temperature
            mu_el,                              #baryochemical potential
            complex(phi_re_el, phi_im_el),      #traced PL
            complex(phi_re_el, -phi_im_el),     #traced PL c.c.
            M_el,                               #bound state mass
            Mth_el,                             #threshold mass
            a,                                  #nr of valence quarks - nr of valence antiquarks
            dx                                  #degeneracy factor
            ) 
        for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im, M, Mth), desc = "Pres", total = len(T), ascii = True)]
    BDen = [0.0 for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el, dM_el, dMth_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im, M, Mth, dMdmu, dMthdmu), desc = "BDen", total = len(T), ascii = True)]
    SDen = [0.0 for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el, dM_el, dMth_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im, M, Mth, dMdT, dMthdT), desc = "SDen", total = len(T), ascii = True)]
    return Pres, BDen, SDen

def cluster_c_thermo(T, mu, phi_re, phi_im, M, Mth, dMdmu, dMthdmu, dMdT, dMthdT, a, dx, Ni, L):
    Pres = [
        cluster.pressure(
            T_el,                               #temperature
            mu_el,                              #baryochemical potential
            complex(phi_re_el, phi_im_el),      #traced PL
            complex(phi_re_el, -phi_im_el),     #traced PL c.c.
            M_el,                               #bound state mass
            Mth_el,                             #threshold mass
            a,                                  #nr of valence quarks - nr of valence antiquarks
            dx,                                 #degeneracy factor
            Ni,                                 #total number of valence d.o.f.'s
            L                                   #continuum energy scale
            ) 
        for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im, M, Mth), desc = "Pres", total = len(T), ascii = True)]
    BDen = [0.0 for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el, dM_el, dMth_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im, M, Mth, dMdmu, dMthdmu), desc = "BDen", total = len(T), ascii = True)]
    SDen = [0.0 for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el, dM_el, dMth_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im, M, Mth, dMdT, dMthdT), desc = "SDen", total = len(T), ascii = True)]
    return Pres, BDen, SDen

def calc_PL(
    T : float, mu : float, phi_re0 : float, phi_im0 : float,
    light_kwargs = None, strange_kwargs = None, gluon_kwargs = None, perturbative_kwargs = None,
    with_clusters : bool = True) -> (float, float):
    
    if light_kwargs is None:
        light_kwargs = {}
    if strange_kwargs is None:
        strange_kwargs = {'Nf' : 1.0, 'ml' : 100.0}
    if gluon_kwargs is None:
        gluon_kwargs = {}
    if perturbative_kwargs is None:
        perturbative_kwargs = {'Nf' : 3.0}

    sd = 1234
    attempt = 1
    max_attempt = 100
    bnds = ((0.0, 3.0), (-3.0, 3.0),)

    def thermodynamic_potential_with_clusters(
        x, 
        _T, _mu, 
        _diquark_bmass, _diquark_thmass, _d_diquark, 
        _fquark_bmass, _fquark_thmass, _d_fquark, 
        _qquark_bmass, _qquark_thmass, _d_qquark, 
        _s_kwargs, _q_kwargs, _pert_kwargs, _g_kwargs):
        sq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_s_kwargs)
        lq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_q_kwargs)
        per = pnjl.thermo.gcp_perturbative.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_pert_kwargs)
        glue = pnjl.thermo.gcp_pl_polynomial.gcp_real(_T, complex(x[0], x[1]), complex(x[0], -x[1]), **_g_kwargs)
        diquark = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _diquark_bmass, _diquark_thmass, 2, _d_diquark)
        fquark = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _fquark_bmass, _fquark_thmass, 4, _d_fquark)
        qquark = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _qquark_bmass, _qquark_thmass, 5, _d_qquark)

        return sq + lq + per + glue + diquark + fquark + qquark

    def thermodynamic_potential(x, _T, _mu, _s_kwargs, _q_kwargs, _pert_kwargs, _g_kwargs):
        sq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_s_kwargs)
        lq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_q_kwargs)
        per = pnjl.thermo.gcp_perturbative.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_pert_kwargs)
        glue = pnjl.thermo.gcp_pl_polynomial.gcp_real(_T, complex(x[0], x[1]), complex(x[0], -x[1]), **_g_kwargs)

        return sq + lq + per + glue

    omega_result = None
    if with_clusters:

        diquark_bmass = pnjl.defaults.default_MD
        diquark_thmass = 2.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, mu)
        d_diquark = 1.0 * 1.0 * 3.0
        fquark_bmass = pnjl.defaults.default_MF
        fquark_thmass = 4.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, mu)
        d_fquark = 1.0 * 1.0 * 3.0
        qquark_bmass = pnjl.defaults.default_MQ
        qquark_thmass = 5.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, mu)
        d_qquark = 2.0 * 2.0 * 3.0

        omega_result = scipy.optimize.dual_annealing(
                thermodynamic_potential_with_clusters,
                bounds = bnds,
                args = (T, mu, 
                        diquark_bmass, diquark_thmass, d_diquark, 
                        fquark_bmass, fquark_thmass, d_fquark,
                        qquark_bmass, qquark_thmass, d_qquark,
                        strange_kwargs, light_kwargs, perturbative_kwargs, gluon_kwargs),
                x0 = [phi_re0, phi_im0],
                maxiter = 20,
                seed = sd
                )
    else:
        omega_result = scipy.optimize.dual_annealing(
                thermodynamic_potential,
                bounds = bnds,
                args = (T, mu, strange_kwargs, light_kwargs, perturbative_kwargs, gluon_kwargs),
                x0 = [phi_re0, phi_im0],
                maxiter = 20,
                seed = sd
                )

    return (omega_result.x[0], omega_result.x[1])

def calc_PL_c(
    T : float, mu : float, phi_re0 : float, phi_im0 : float,
    light_kwargs = None, strange_kwargs = None, gluon_kwargs = None, perturbative_kwargs = None,
    with_clusters : bool = True) -> (float, float):
    
    if light_kwargs is None:
        light_kwargs = {}
    if strange_kwargs is None:
        strange_kwargs = {'Nf' : 1.0, 'ml' : 100.0}
    if gluon_kwargs is None:
        gluon_kwargs = {}
    if perturbative_kwargs is None:
        perturbative_kwargs = {'Nf' : 3.0}

    sd = 1234
    attempt = 1
    max_attempt = 100
    bnds = ((0.0, 3.0), (-3.0, 3.0),)

    def thermodynamic_potential_with_clusters(
        x, 
        _T, _mu, 
        _diquark_bmass, _diquark_thmass, _d_diquark, _ni_diquark, _l_diquark,
        _fquark_bmass, _fquark_thmass, _d_fquark, _ni_fquark, _l_fquark,
        _qquark_bmass, _qquark_thmass, _d_qquark, _ni_qquark, _l_qquark,
        _s_kwargs, _q_kwargs, _pert_kwargs, _g_kwargs):
        sq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_s_kwargs)
        lq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_q_kwargs)
        per = pnjl.thermo.gcp_perturbative.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_pert_kwargs)
        glue = pnjl.thermo.gcp_pl_polynomial.gcp_real(_T, complex(x[0], x[1]), complex(x[0], -x[1]), **_g_kwargs)

        diquark = cluster.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _diquark_bmass, _diquark_thmass, 2, _d_diquark, _ni_diquark, _l_diquark)
        fquark  = cluster.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _fquark_bmass , _fquark_thmass , 4, _d_fquark , _ni_fquark , _l_fquark)
        qquark  = cluster.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _qquark_bmass , _qquark_thmass , 5, _d_qquark , _ni_qquark , _l_qquark)

        return sq + lq + per + glue + diquark + fquark + qquark

    def thermodynamic_potential(x, _T, _mu, _s_kwargs, _q_kwargs, _pert_kwargs, _g_kwargs):
        sq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_s_kwargs)
        lq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_q_kwargs)
        per = pnjl.thermo.gcp_perturbative.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_pert_kwargs)
        glue = pnjl.thermo.gcp_pl_polynomial.gcp_real(_T, complex(x[0], x[1]), complex(x[0], -x[1]), **_g_kwargs)

        return sq + lq + per + glue

    omega_result = None
    if with_clusters:

        diquark_bmass = pnjl.defaults.default_MD
        diquark_thmass = 2.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, mu)
        d_diquark = 1.0 * 1.0 * 3.0
        ni_diquark = 2.0
        l_diquark = pnjl.defaults.default_L
        fquark_bmass = pnjl.defaults.default_MF
        fquark_thmass = 4.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, mu)
        d_fquark = 1.0 * 1.0 * 3.0
        ni_fquark = 4.0
        l_fquark = pnjl.defaults.default_L
        qquark_bmass = pnjl.defaults.default_MQ
        qquark_thmass = 5.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, mu)
        d_qquark = 2.0 * 2.0 * 3.0
        ni_qquark = 5.0
        l_qquark = pnjl.defaults.default_L

        omega_result = scipy.optimize.dual_annealing(
                thermodynamic_potential_with_clusters,
                bounds = bnds,
                args = (T, mu, 
                        diquark_bmass, diquark_thmass, d_diquark, ni_diquark, l_diquark, 
                        fquark_bmass, fquark_thmass, d_fquark, ni_fquark, l_fquark,
                        qquark_bmass, qquark_thmass, d_qquark, ni_qquark, l_qquark,
                        strange_kwargs, light_kwargs, perturbative_kwargs, gluon_kwargs),
                x0 = [phi_re0, phi_im0],
                maxiter = 20,
                seed = sd
                )
    else:
        omega_result = scipy.optimize.dual_annealing(
                thermodynamic_potential,
                bounds = bnds,
                args = (T, mu, strange_kwargs, light_kwargs, perturbative_kwargs, gluon_kwargs),
                x0 = [phi_re0, phi_im0],
                maxiter = 20,
                seed = sd
                )

    return (omega_result.x[0], omega_result.x[1])

def clusters(T, mu, phi_re, phi_im):
    print("Calculating nucleon thermo..")
    M_N         = [pnjl.defaults.default_MN                                           for T_el, mu_el in zip(T, mu)]
    dM_N_dmu    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_N_dT     = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_N       = [3. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_N_dmu  = [3. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_N_dT   = [3. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #(N(Dq): spin * isospin * color)
    dN = (2.0 * 2.0 * 1.0)
    Pres_N, BDen_N, SDen_N = cluster_thermo(T, mu, phi_re, phi_im, M_N, Mth_N, dM_N_dmu, dMth_N_dmu, dM_N_dT, dMth_N_dT, 3, dN)

    print("Calculating pentaquark thermo..")
    M_P         = [pnjl.defaults.default_MP                                            for T_el, mu_el in zip(T, mu)]
    dM_P_dmu    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_P_dT     = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_P       = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_P_dmu  = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_P_dT   = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #P(NM) + P(NM)
    dP = (4.0 * 2.0 * 1.0) + (2.0 * 4.0 * 1.0)
    Pres_P, BDen_P, SDen_P = cluster_thermo(T, mu, phi_re, phi_im, M_P, Mth_P, dM_P_dmu, dMth_P_dmu, dM_P_dT, dMth_P_dT, 3, dP)

    print("Calculating hexaquark thermo..")
    M_H         = [pnjl.defaults.default_MH                                           for T_el, mu_el in zip(T, mu)]
    dM_H_dmu    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_H_dT     = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_H       = [6. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_H_dmu  = [6. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_H_dT   = [6. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #H(Qq) / H(FD) / H(NN) + H(Qq) / H(NN)
    dH = (1.0 * 3.0 * 1.0) + (3.0 * 1.0 * 1.0)
    Pres_H, BDen_H, SDen_H = cluster_thermo(T, mu, phi_re, phi_im, M_H, Mth_H, dM_H_dmu, dMth_H_dmu, dM_H_dT, dMth_H_dT, 6, dH)

    print("Calculating pi meson thermo..")
    M_pi         = [140.                                                               for T_el, mu_el in zip(T, mu)]
    dM_pi_dmu    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_pi_dT     = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_pi       = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_pi_dmu  = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_pi_dT   = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #pi(q aq)
    dpi = (1.0 * 3.0 * 1.0)
    Pres_pi, BDen_pi, SDen_pi = cluster_thermo(T, mu, phi_re, phi_im, M_pi, Mth_pi, dM_pi_dmu, dMth_pi_dmu, dM_pi_dT, dMth_pi_dT, 0, dpi)

    print("Calculating rho meson thermo..")
    M_rho        = [pnjl.defaults.default_MM                                           for T_el, mu_el in zip(T, mu)]
    dM_rho_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_rho_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_rho      = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_rho_dmu = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_rho_dT  = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #rho(q aq)
    drho = (3.0 * 3.0 * 1.0)
    Pres_rho, BDen_rho, SDen_rho = cluster_thermo(T, mu, phi_re, phi_im, M_rho, Mth_rho, dM_rho_dmu, dMth_rho_dmu, dM_rho_dT, dMth_rho_dT, 0, drho)

    print("Calculating omega meson thermo..")
    M_omega        = [pnjl.defaults.default_MM                                           for T_el, mu_el in zip(T, mu)]
    dM_omega_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_omega_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_omega      = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_omega_dmu = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_omega_dT  = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #omega(q aq)
    domega = (3.0 * 1.0 * 1.0) #should this have the 1/2 factor?
    Pres_omega, BDen_omega, SDen_omega = cluster_thermo(T, mu, phi_re, phi_im, M_omega, Mth_omega, dM_omega_dmu, dMth_omega_dmu, dM_omega_dT, dMth_omega_dT, 0, domega)

    print("Calculating tetraquark thermo..")
    M_T        = [pnjl.defaults.default_MT                                           for T_el, mu_el in zip(T, mu)]
    dM_T_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_T_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_T      = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_T_dmu = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_T_dT  = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #T(MM) + T(MM) + T(MM)
    dT = (1.0 * 5.0 * 1.0) + (5.0 * 1.0 * 1.0) + (3.0 * 3.0 * 1.0)
    Pres_T, BDen_T, SDen_T = cluster_thermo(T, mu, phi_re, phi_im, M_T, Mth_T, dM_T_dmu, dMth_T_dmu, dM_T_dT, dMth_T_dT, 0, dT)

    print("Calculating diquark thermo..")
    M_D        = [pnjl.defaults.default_MD                                           for T_el, mu_el in zip(T, mu)]
    dM_D_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_D_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_D      = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_D_dmu = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_D_dT  = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #D(qq)
    dD = (1.0 * 1.0 * 3.0)
    Pres_D, BDen_D, SDen_D = cluster_thermo(T, mu, phi_re, [-el for el in phi_im], M_D, Mth_D, dM_D_dmu, dMth_D_dmu, dM_D_dT, dMth_D_dT, 2, dD)

    print("Calculating 4-quark thermo..")
    M_F        = [pnjl.defaults.default_MF                                           for T_el, mu_el in zip(T, mu)]
    dM_F_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_F_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_F      = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_F_dmu = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_F_dT  = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #F(Nq)
    dF = (1.0 * 1.0 * 3.0)
    Pres_F, BDen_F, SDen_F = cluster_thermo(T, mu, phi_re, phi_im, M_F, Mth_F, dM_F_dmu, dMth_F_dmu, dM_F_dT, dMth_F_dT, 4, dF)

    print("Calculating 5-quark thermo..")
    M_Q5        = [pnjl.defaults.default_MQ                                           for T_el, mu_el in zip(T, mu)]
    dM_Q5_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_Q5_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_Q5      = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_Q5_dmu = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_Q5_dT  = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #Q5(F(Nq)q) / Q5(F(DD)q) / Q5(ND)
    dQ5 = (2.0 * 2.0 * 3.0)
    Pres_Q5, BDen_Q5, SDen_Q5 = cluster_thermo(T, mu, phi_re, [-el for el in phi_im], M_Q5, Mth_Q5, dM_Q5_dmu, dMth_Q5_dmu, dM_Q5_dT, dMth_Q5_dT, 5, dQ5)

    return (
        (Pres_pi, Pres_rho, Pres_omega, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, Pres_Q5, Pres_H),
        (BDen_pi, BDen_rho, BDen_omega, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, BDen_Q5, BDen_H),
        (SDen_pi, SDen_rho, SDen_omega, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, SDen_Q5, SDen_H)
        )

def clusters_c(T, mu, phi_re, phi_im):
    print("Calculating nucleon thermo..")
    M_N         = [pnjl.defaults.default_MN                                           for T_el, mu_el in zip(T, mu)]
    dM_N_dmu    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_N_dT     = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_N       = [3. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_N_dmu  = [3. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_N_dT   = [3. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #(N(Dq): spin * isospin * color)
    dN = (2.0 * 2.0 * 1.0)
    NN = 3.0
    LN = pnjl.defaults.default_L
    Pres_N, BDen_N, SDen_N = cluster_c_thermo(T, mu, phi_re, phi_im, M_N, Mth_N, dM_N_dmu, dMth_N_dmu, dM_N_dT, dMth_N_dT, 3, dN, NN, LN)

    print("Calculating pentaquark thermo..")
    M_P         = [pnjl.defaults.default_MP                                            for T_el, mu_el in zip(T, mu)]
    dM_P_dmu    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_P_dT     = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_P       = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_P_dmu  = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_P_dT   = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #P(NM) + P(NM)
    dP = (4.0 * 2.0 * 1.0) + (2.0 * 4.0 * 1.0)
    NP = 5.0
    LP = pnjl.defaults.default_L
    Pres_P, BDen_P, SDen_P = cluster_c_thermo(T, mu, phi_re, phi_im, M_P, Mth_P, dM_P_dmu, dMth_P_dmu, dM_P_dT, dMth_P_dT, 3, dP, NP, LP)

    print("Calculating hexaquark thermo..")
    M_H         = [pnjl.defaults.default_MH                                           for T_el, mu_el in zip(T, mu)]
    dM_H_dmu    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_H_dT     = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_H       = [6. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_H_dmu  = [6. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_H_dT   = [6. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #H(Qq) / H(FD) / H(NN) + H(Qq) / H(NN)
    dH = (1.0 * 3.0 * 1.0) + (3.0 * 1.0 * 1.0)
    NH = 6.0
    LH = pnjl.defaults.default_L
    Pres_H, BDen_H, SDen_H = cluster_c_thermo(T, mu, phi_re, phi_im, M_H, Mth_H, dM_H_dmu, dMth_H_dmu, dM_H_dT, dMth_H_dT, 6, dH, NH, LH)

    print("Calculating pi meson thermo..")
    M_pi         = [140.                                                               for T_el, mu_el in zip(T, mu)]
    dM_pi_dmu    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_pi_dT     = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_pi       = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_pi_dmu  = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_pi_dT   = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #pi(q aq)
    dpi = (1.0 * 3.0 * 1.0)
    Npi = 2.0
    Lpi = pnjl.defaults.default_L
    Pres_pi, BDen_pi, SDen_pi = cluster_c_thermo(T, mu, phi_re, phi_im, M_pi, Mth_pi, dM_pi_dmu, dMth_pi_dmu, dM_pi_dT, dMth_pi_dT, 0, dpi, Npi, Lpi)

    print("Calculating rho meson thermo..")
    M_rho        = [pnjl.defaults.default_MM                                           for T_el, mu_el in zip(T, mu)]
    dM_rho_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_rho_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_rho      = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_rho_dmu = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_rho_dT  = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #rho(q aq)
    drho = (3.0 * 3.0 * 1.0)
    Nrho = 2.0
    Lrho = pnjl.defaults.default_L
    Pres_rho, BDen_rho, SDen_rho = cluster_c_thermo(T, mu, phi_re, phi_im, M_rho, Mth_rho, dM_rho_dmu, dMth_rho_dmu, dM_rho_dT, dMth_rho_dT, 0, drho, Nrho, Lrho)

    print("Calculating omega meson thermo..")
    M_omega        = [pnjl.defaults.default_MM                                           for T_el, mu_el in zip(T, mu)]
    dM_omega_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_omega_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_omega      = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_omega_dmu = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_omega_dT  = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #omega(q aq)
    domega = (3.0 * 1.0 * 1.0)
    Nomega = 2.0
    Lomega = pnjl.defaults.default_L
    Pres_omega, BDen_omega, SDen_omega = cluster_c_thermo(T, mu, phi_re, phi_im, M_omega, Mth_omega, dM_omega_dmu, dMth_omega_dmu, dM_omega_dT, dMth_omega_dT, 0, domega, Nomega, Lomega)

    print("Calculating tetraquark thermo..")
    M_T        = [pnjl.defaults.default_MT                                           for T_el, mu_el in zip(T, mu)]
    dM_T_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_T_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_T      = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_T_dmu = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_T_dT  = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #T(MM) + T(MM) + T(MM)
    dT = (1.0 * 5.0 * 1.0) + (5.0 * 1.0 * 1.0) + (3.0 * 3.0 * 1.0)
    NT = 4.0
    LT = pnjl.defaults.default_L
    Pres_T, BDen_T, SDen_T = cluster_c_thermo(T, mu, phi_re, phi_im, M_T, Mth_T, dM_T_dmu, dMth_T_dmu, dM_T_dT, dMth_T_dT, 0, dT, NT, LT)

    print("Calculating diquark thermo..")
    M_D        = [pnjl.defaults.default_MD                                           for T_el, mu_el in zip(T, mu)]
    dM_D_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_D_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_D      = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_D_dmu = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_D_dT  = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #D(qq)
    dD = (1.0 * 1.0 * 3.0)
    ND = 2.0
    LD = pnjl.defaults.default_L
    Pres_D, BDen_D, SDen_D = cluster_c_thermo(T, mu, phi_re, [-el for el in phi_im], M_D, Mth_D, dM_D_dmu, dMth_D_dmu, dM_D_dT, dMth_D_dT, 2, dD, ND, LD)

    print("Calculating 4-quark thermo..")
    M_F        = [pnjl.defaults.default_MF                                           for T_el, mu_el in zip(T, mu)]
    dM_F_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_F_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_F      = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_F_dmu = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_F_dT  = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #F(Nq)
    dF = (1.0 * 1.0 * 3.0)
    NF = 4.0
    LF = pnjl.defaults.default_L
    Pres_F, BDen_F, SDen_F = cluster_c_thermo(T, mu, phi_re, phi_im, M_F, Mth_F, dM_F_dmu, dMth_F_dmu, dM_F_dT, dMth_F_dT, 4, dF, NF, LF)

    print("Calculating 5-quark thermo..")
    M_Q5        = [pnjl.defaults.default_MQ                                           for T_el, mu_el in zip(T, mu)]
    dM_Q5_dmu   = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_Q5_dT    = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_Q5      = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_Q5_dmu = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_Q5_dT  = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #Q5(F(Nq)q) / Q5(F(DD)q) / Q5(ND)
    dQ5 = (2.0 * 2.0 * 3.0)
    NQ5 = 5.0
    LQ5 = pnjl.defaults.default_L
    Pres_Q5, BDen_Q5, SDen_Q5 = cluster_c_thermo(T, mu, phi_re, [-el for el in phi_im], M_Q5, Mth_Q5, dM_Q5_dmu, dMth_Q5_dmu, dM_Q5_dT, dMth_Q5_dT, 5, dQ5, NQ5, LQ5)

    return (
        (Pres_pi, Pres_rho, Pres_omega, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, Pres_Q5, Pres_H),
        (BDen_pi, BDen_rho, BDen_omega, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, BDen_Q5, BDen_H),
        (SDen_pi, SDen_rho, SDen_omega, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, SDen_Q5, SDen_H)
        )

def PNJL_thermodynamics_mu_test():

    phi_re_mu0          = []
    phi_im_mu0          = []
    phi_re_mu200        = []
    phi_im_mu200        = []
    phi_re_mu300        = []
    phi_im_mu300        = []
    Pres_Q_mu0          = []
    Pres_g_mu0          = []
    Pres_pert_mu0       = []
    Pres_Q_mu200        = []
    Pres_g_mu200        = []
    Pres_pert_mu200     = []
    Pres_Q_mu300        = []
    Pres_g_mu300        = []
    Pres_pert_mu300     = []
    Pres_pi_mu0         = []
    Pres_rho_mu0        = []
    Pres_omega_mu0      = []
    Pres_D_mu0          = []
    Pres_N_mu0          = []
    Pres_T_mu0          = []
    Pres_F_mu0          = []
    Pres_P_mu0          = []
    Pres_Q5_mu0         = []
    Pres_H_mu0          = []
    Pres_pi_mu200       = []
    Pres_rho_mu200      = []
    Pres_omega_mu200    = []
    Pres_D_mu200        = []
    Pres_N_mu200        = []
    Pres_T_mu200        = []
    Pres_F_mu200        = []
    Pres_P_mu200        = []
    Pres_Q5_mu200       = []
    Pres_H_mu200        = []
    Pres_pi_mu300       = []
    Pres_rho_mu300      = []
    Pres_omega_mu300    = []
    Pres_D_mu300        = []
    Pres_N_mu300        = []
    Pres_T_mu300        = []
    Pres_F_mu300        = []
    Pres_P_mu300        = []
    Pres_Q5_mu300       = []
    Pres_H_mu300        = []
    
    T = numpy.linspace(1.0, 450.0, 200)
    mu0   = [0.0 for el in T]
    mu200 = [200.0 / 3.0 for el in T]
    mu300 = [300.0 / 3.0 for el in T]
    
    recalc_pl_mu0         = False
    recalc_pl_mu200       = False
    recalc_pl_mu300       = False
    recalc_pressure_mu0   = False
    recalc_pressure_mu200 = False
    recalc_pressure_mu300 = False

    cluster_backreaction  = False
    pl_turned_off         = False

    pl_mu0_file   = "D:/EoS/BDK/mu_test/pl_mu0.dat"
    pl_mu200_file = "D:/EoS/BDK/mu_test/pl_mu200.dat"
    pl_mu300_file = "D:/EoS/BDK/mu_test/pl_mu300.dat"
    pressure_mu0_file   = "D:/EoS/BDK/mu_test/pressure_c_mu0.dat"
    pressure_mu200_file = "D:/EoS/BDK/mu_test/pressure_c_mu200.dat"
    pressure_mu300_file = "D:/EoS/BDK/mu_test/pressure_c_mu300.dat"

    if recalc_pl_mu0:
        phi_re_mu0.append(1e-15)
        phi_im_mu0.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm.tqdm(zip(T, mu0), desc = "Traced Polyakov loop (mu = 0)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu0[-1], phi_im_mu0[-1], with_clusters = cluster_backreaction)
            phi_re_mu0.append(temp_phi_re)
            phi_im_mu0.append(temp_phi_im)
        phi_re_mu0 = phi_re_mu0[1:]
        phi_im_mu0 = phi_im_mu0[1:]
        with open(pl_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu0, phi_im_mu0)])
    else:
        T, phi_re_mu0 = utils.data_collect(0, 1, pl_mu0_file)
        phi_im_mu0, _ = utils.data_collect(2, 2, pl_mu0_file)

    if recalc_pl_mu200:
        phi_re_mu200.append(1e-15)
        phi_im_mu200.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm.tqdm(zip(T, mu200), desc = "Traced Polyakov loop (mu = 200)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu200[-1], phi_im_mu200[-1], with_clusters = cluster_backreaction)
            phi_re_mu200.append(temp_phi_re)
            phi_im_mu200.append(temp_phi_im)
        phi_re_mu200 = phi_re_mu200[1:]
        phi_im_mu200 = phi_im_mu200[1:]
        with open(pl_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu200, phi_im_mu200)])
    else:
        T, phi_re_mu200 = utils.data_collect(0, 1, pl_mu200_file)
        phi_im_mu200, _ = utils.data_collect(2, 2, pl_mu200_file)
    
    if recalc_pl_mu300:
        phi_re_mu300.append(1e-15)
        phi_im_mu300.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm.tqdm(zip(T, mu300), desc = "Traced Polyakov loop (mu = 300)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu300[-1], phi_im_mu300[-1], with_clusters = cluster_backreaction)
            phi_re_mu300.append(temp_phi_re)
            phi_im_mu300.append(temp_phi_im)
        phi_re_mu300 = phi_re_mu300[1:]
        phi_im_mu300 = phi_im_mu300[1:]
        with open(pl_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu300, phi_im_mu300)])
    else:
        T, phi_re_mu300 = utils.data_collect(0, 1, pl_mu300_file)
        phi_im_mu300, _ = utils.data_collect(2, 2, pl_mu300_file)

    if recalc_pressure_mu0:
        Pres_g_mu0 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T, phi_re_mu0, phi_im_mu0), 
                desc = "Gluon pressure (mu = 0)", 
                total = len(T), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu0 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu0, phi_re_mu0, phi_im_mu0), 
                    desc = "Quark pressure (mu = 0)", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert_mu0 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu0, phi_re_mu0, phi_im_mu0), 
                    desc = "Perturbative pressure (mu = 0)", 
                    total = len(T), 
                    ascii = True
                    )]        
            (
                (Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu0, phi_re_mu0, phi_im_mu0)
        else:
            Pres_Q_mu0 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu0, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 0)", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert_mu0 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu0, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 0)", 
                    total = len(T), 
                    ascii = True
                    )]        
            (
                (Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu0, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu0, Pres_g_mu0, Pres_pert_mu0, Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0)])
    else:
        Pres_Q_mu0, Pres_g_mu0       = utils.data_collect(0, 1, pressure_mu0_file)
        Pres_pert_mu0, Pres_pi_mu0   = utils.data_collect(2, 3, pressure_mu0_file)
        Pres_rho_mu0, Pres_omega_mu0 = utils.data_collect(4, 5, pressure_mu0_file)
        Pres_D_mu0, Pres_N_mu0       = utils.data_collect(6, 7, pressure_mu0_file)
        Pres_T_mu0, Pres_F_mu0       = utils.data_collect(8, 9, pressure_mu0_file)
        Pres_P_mu0, Pres_Q5_mu0      = utils.data_collect(10, 11, pressure_mu0_file)
        Pres_H_mu0, _                = utils.data_collect(12, 12, pressure_mu0_file)

    if recalc_pressure_mu200:
        Pres_g_mu200 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T, phi_re_mu200, phi_im_mu200), 
                desc = "Gluon pressure (mu = 200)", 
                total = len(T), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu200 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu200, phi_re_mu200, phi_im_mu200), 
                    desc = "Quark pressure (mu = 200)", 
                    total = len(T), 
                    ascii = True
                    )]        
            Pres_pert_mu200 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu200, phi_re_mu200, phi_im_mu200), 
                    desc = "Perturbative pressure (mu = 200)", 
                    total = len(T), 
                    ascii = True
                    )]        
            (
                (Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu200, phi_re_mu200, phi_im_mu200)
        else:
            Pres_Q_mu200 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu200, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 200)", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert_mu200 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu200, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 200)", 
                    total = len(T), 
                    ascii = True
                    )]        
            (
                (Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu200, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu200, Pres_g_mu200, Pres_pert_mu200, Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200)])
    else:
        Pres_Q_mu200, Pres_g_mu200       = utils.data_collect(0, 1, pressure_mu200_file)
        Pres_pert_mu200, Pres_pi_mu200   = utils.data_collect(2, 3, pressure_mu200_file)
        Pres_rho_mu200, Pres_omega_mu200 = utils.data_collect(4, 5, pressure_mu200_file)
        Pres_D_mu200, Pres_N_mu200       = utils.data_collect(6, 7, pressure_mu200_file)
        Pres_T_mu200, Pres_F_mu200       = utils.data_collect(8, 9, pressure_mu200_file)
        Pres_P_mu200, Pres_Q5_mu200      = utils.data_collect(10, 11, pressure_mu200_file)
        Pres_H_mu200, _                  = utils.data_collect(12, 12, pressure_mu200_file)
    
    if recalc_pressure_mu300:
        Pres_g_mu300 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T, phi_re_mu300, phi_im_mu300), 
                desc = "Gluon pressure (mu = 300)", 
                total = len(T), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu300 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu300, phi_re_mu300, phi_im_mu300), 
                    desc = "Quark pressure (mu = 300)", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert_mu300 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu300, phi_re_mu300, phi_im_mu300), 
                    desc = "Perturbative pressure (mu = 300)", 
                    total = len(T), 
                    ascii = True
                    )]        
            (
                (Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu300, phi_re_mu300, phi_im_mu300)
        else:
            Pres_Q_mu300 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el in 
                tqdm.tqdm(
                    zip(T, mu300, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 300)", 
                    total = len(T), 
                    ascii = True
                    )]
            Pres_pert_mu300 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T, mu300, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 300)", 
                    total = len(T), 
                    ascii = True
                    )]
            (
                (Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu300, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu300, Pres_g_mu300, Pres_pert_mu300, Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300)])
    else:
        Pres_Q_mu300, Pres_g_mu300       = utils.data_collect(0, 1, pressure_mu300_file)
        Pres_pert_mu300, Pres_pi_mu300   = utils.data_collect(2, 3, pressure_mu300_file)
        Pres_rho_mu300, Pres_omega_mu300 = utils.data_collect(4, 5, pressure_mu300_file)
        Pres_D_mu300, Pres_N_mu300       = utils.data_collect(6, 7, pressure_mu300_file)
        Pres_T_mu300, Pres_F_mu300       = utils.data_collect(8, 9, pressure_mu300_file)
        Pres_P_mu300, Pres_Q5_mu300      = utils.data_collect(10, 11, pressure_mu300_file)
        Pres_H_mu300, _                  = utils.data_collect(12, 12, pressure_mu300_file)

    contrib_q_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q_mu0   )]
    contrib_g_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_g_mu0   )]
    contrib_pert_mu0                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pert_mu0)]
    contrib_qgp_mu0                 = [sum(el) for el in zip(contrib_q_mu0, contrib_g_mu0, contrib_pert_mu0)]
    contrib_q_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q_mu200   )]
    contrib_g_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_g_mu200   )]
    contrib_pert_mu200              = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pert_mu200)]
    contrib_qgp_mu200               = [sum(el) for el in zip(contrib_q_mu200, contrib_g_mu200, contrib_pert_mu200)]
    contrib_q_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q_mu300   )]
    contrib_g_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_g_mu300   )]
    contrib_pert_mu300              = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pert_mu300)]
    contrib_qgp_mu300               = [sum(el) for el in zip(contrib_q_mu300, contrib_g_mu300, contrib_pert_mu300)]
    contrib_pi_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi_mu0)]
    contrib_rho_mu0                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho_mu0)]
    contrib_omega_mu0               = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega_mu0)]
    contrib_D_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D_mu0)]
    contrib_N_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N_mu0)]
    contrib_T_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T_mu0)]
    contrib_F_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F_mu0)]
    contrib_P_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P_mu0)]
    contrib_Q5_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5_mu0)]
    contrib_H_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H_mu0)]
    contrib_cluster_mu0             = [sum(el) for el in zip(contrib_pi_mu0, contrib_rho_mu0, contrib_omega_mu0, contrib_D_mu0, contrib_N_mu0, contrib_T_mu0, contrib_F_mu0, contrib_P_mu0, contrib_Q5_mu0, contrib_H_mu0)]
    contrib_cluster_singlet_mu0     = [sum(el) for el in zip(contrib_pi_mu0, contrib_rho_mu0, contrib_omega_mu0, contrib_N_mu0, contrib_T_mu0, contrib_P_mu0, contrib_H_mu0)]
    contrib_cluster_color_mu0       = [sum(el) for el in zip(contrib_D_mu0, contrib_F_mu0, contrib_Q5_mu0)]
    contrib_pi_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi_mu200)]
    contrib_rho_mu200               = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho_mu200)]
    contrib_omega_mu200             = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega_mu200)]
    contrib_D_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D_mu200)]
    contrib_N_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N_mu200)]
    contrib_T_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T_mu200)]
    contrib_F_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F_mu200)]
    contrib_P_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P_mu200)]
    contrib_Q5_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5_mu200)]
    contrib_H_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H_mu200)]
    contrib_cluster_mu200           = [sum(el) for el in zip(contrib_pi_mu200, contrib_rho_mu200, contrib_omega_mu200, contrib_D_mu200, contrib_N_mu200, contrib_T_mu200, contrib_F_mu200, contrib_P_mu200, contrib_Q5_mu200, contrib_H_mu200)]
    contrib_cluster_singlet_mu200   = [sum(el) for el in zip(contrib_pi_mu200, contrib_rho_mu200, contrib_omega_mu200, contrib_N_mu200, contrib_T_mu200, contrib_P_mu200, contrib_H_mu200)]
    contrib_cluster_color_mu200     = [sum(el) for el in zip(contrib_D_mu200, contrib_F_mu200, contrib_Q5_mu200)]
    contrib_pi_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi_mu300)]
    contrib_rho_mu300               = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho_mu300)]
    contrib_omega_mu300             = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega_mu300)]
    contrib_D_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D_mu300)]
    contrib_N_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N_mu300)]
    contrib_T_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T_mu300)]
    contrib_F_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F_mu300)]
    contrib_P_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P_mu300)]
    contrib_Q5_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5_mu300)]
    contrib_H_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H_mu300)]
    contrib_cluster_mu300           = [sum(el) for el in zip(contrib_pi_mu300, contrib_rho_mu300, contrib_omega_mu300, contrib_D_mu300, contrib_N_mu300, contrib_T_mu300, contrib_F_mu300, contrib_P_mu300, contrib_Q5_mu300, contrib_H_mu300)]
    contrib_cluster_singlet_mu300   = [sum(el) for el in zip(contrib_pi_mu300, contrib_rho_mu300, contrib_omega_mu300, contrib_N_mu300, contrib_T_mu300, contrib_P_mu300, contrib_H_mu300)]
    contrib_cluster_color_mu300     = [sum(el) for el in zip(contrib_D_mu300, contrib_F_mu300, contrib_Q5_mu300)]

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y)       = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)     = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_low.dat")
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_high.dat")

    borsanyi_1204_6710v2_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = numpy.array(borsanyi_1204_6710v2_mu0)
    borsanyi_1204_6710v2_mu200 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu200_x[::-1], low_1204_6710v2_mu200_y[::-1]):
        borsanyi_1204_6710v2_mu200.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu200 = numpy.array(borsanyi_1204_6710v2_mu200)
    borsanyi_1204_6710v2_mu300 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu300_x[::-1], low_1204_6710v2_mu300_y[::-1]):
        borsanyi_1204_6710v2_mu300.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu300 = numpy.array(borsanyi_1204_6710v2_mu300)

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([0., 450., 0., 4.0])
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=0}$'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'red', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=200}$ MeV'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu300, closed = True, fill = True, color = 'yellow', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=300}$ MeV'))
    #ax1.plot(T, contrib_q_mu0, '-', c = 'blue', label = r'$\mathrm{P_{Q,0}}$')
    #ax1.plot(T, contrib_g_mu0, '-', c = 'red', label = r'$\mathrm{P_{g,0}}$')
    #ax1.plot(T, contrib_pert_mu0, '-', c = 'pink', label = r'$\mathrm{P_{pert,0}}$')
    ax1.plot(T, contrib_qgp_mu0, '-', c = 'black', label = r'$\mathrm{P_{QGP,0}}$')
    ax1.plot(T, contrib_cluster_mu0, '-', c = 'blue', label = r'$\mathrm{P_{cluster,0}}$')
    ax1.plot(T, contrib_cluster_singlet_mu0, '-', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,0}}$')
    ax1.plot(T, contrib_cluster_color_mu0, '-', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,0}}$')
    #ax1.plot(T, contrib_q_mu200, '--', c = 'blue', label = r'$\mathrm{P_{Q,200}}$')
    #ax1.plot(T, contrib_g_mu200, '--', c = 'red', label = r'$\mathrm{P_{g,200}}$')
    #ax1.plot(T, contrib_pert_mu200, '--', c = 'pink', label = r'$\mathrm{P_{pert,200}}$')
    ax1.plot(T, contrib_qgp_mu200, '--', c = 'black', label = r'$\mathrm{P_{QGP,200}}$')
    ax1.plot(T, contrib_cluster_mu200, '--', c = 'blue', label = r'$\mathrm{P_{cluster,200}}$')
    ax1.plot(T, contrib_cluster_singlet_mu200, '--', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,200}}$')
    ax1.plot(T, contrib_cluster_color_mu200, '--', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,200}}$')
    #ax1.plot(T, contrib_q_mu300, '-.', c = 'blue', label = r'$\mathrm{P_{Q,300}}$')
    #ax1.plot(T, contrib_g_mu300, '-.', c = 'red', label = r'$\mathrm{P_{g,300}}$')
    #ax1.plot(T, contrib_pert_mu300, '-.', c = 'pink', label = r'$\mathrm{P_{pert,300}}$')
    ax1.plot(T, contrib_qgp_mu300, '-.', c = 'black', label = r'$\mathrm{P_{QGP,300}}$')
    ax1.plot(T, contrib_cluster_mu300, '-.', c = 'blue', label = r'$\mathrm{P_{cluster,300}}$')
    ax1.plot(T, contrib_cluster_singlet_mu300, '-.', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,300}}$')
    ax1.plot(T, contrib_cluster_color_mu300, '-.', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,300}}$')
    #ax1.legend(loc = 2)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig2 = matplotlib.pyplot.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.axis([0., 450., 0., 1.2])
    ax2.plot(T, [pnjl.thermo.gcp_sea_lattice.M(el, 0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T], '-', c = 'green')
    ax2.plot(T, [pnjl.thermo.gcp_sea_lattice.M(el, 200.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T], '--', c = 'green')
    ax2.plot(T, [pnjl.thermo.gcp_sea_lattice.M(el, 300.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T], '-.', c = 'green')
    ax2.plot(T, phi_re_mu0, '-', c = 'blue')
    ax2.plot(T, phi_re_mu200, '--', c = 'blue')
    ax2.plot(T, phi_re_mu300, '-.', c = 'blue')
    ax2.plot(T, phi_im_mu0, '-', c = 'red')
    ax2.plot(T, phi_im_mu200, '--', c = 'red')
    ax2.plot(T, phi_im_mu300, '-.', c = 'red')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{\Phi}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    fig2.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def PNJL_thermodynamics_continuum_test():

    phi_re_mu0, phi_re_mu200, phi_re_mu300, phi_re_mu400, phi_re_mu500 = [], [], [], [], []
    phi_im_mu0, phi_im_mu200, phi_im_mu300, phi_im_mu400, phi_im_mu500 = [], [], [], [], []

    Pres_Q_mu0, Pres_Q_mu200, Pres_Q_mu300, Pres_Q_mu400, Pres_Q_mu500 = [], [], [], [], []
    Pres_g_mu0, Pres_g_mu200, Pres_g_mu300, Pres_g_mu400, Pres_g_mu500 = [], [], [], [], []
    Pres_pert_mu0, Pres_pert_mu200, Pres_pert_mu300, Pres_pert_mu400, Pres_pert_mu500 = [], [], [], [], []

    Pres_pi_step_mu0, Pres_pi_step_mu200, Pres_pi_step_mu300, Pres_pi_step_mu400, Pres_pi_step_mu500 = [], [], [], [], []
    Pres_rho_step_mu0, Pres_rho_step_mu200, Pres_rho_step_mu300, Pres_rho_step_mu400, Pres_rho_step_mu500 = [], [], [], [], []
    Pres_omega_step_mu0, Pres_omega_step_mu200, Pres_omega_step_mu300, Pres_omega_step_mu400, Pres_omega_step_mu500 = [], [], [], [], []
    Pres_D_step_mu0, Pres_D_step_mu200, Pres_D_step_mu300, Pres_D_step_mu400, Pres_D_step_mu500 = [], [], [], [], []
    Pres_N_step_mu0, Pres_N_step_mu200, Pres_N_step_mu300, Pres_N_step_mu400, Pres_N_step_mu500 = [], [], [], [], []
    Pres_T_step_mu0, Pres_T_step_mu200, Pres_T_step_mu300, Pres_T_step_mu400, Pres_T_step_mu500 = [], [], [], [], []
    Pres_F_step_mu0, Pres_F_step_mu200, Pres_F_step_mu300, Pres_F_step_mu400, Pres_F_step_mu500 = [], [], [], [], []
    Pres_P_step_mu0, Pres_P_step_mu200, Pres_P_step_mu300, Pres_P_step_mu400, Pres_P_step_mu500 = [], [], [], [], []
    Pres_Q5_step_mu0, Pres_Q5_step_mu200, Pres_Q5_step_mu300, Pres_Q5_step_mu400, Pres_Q5_step_mu500 = [], [], [], [], []
    Pres_H_step_mu0, Pres_H_step_mu200, Pres_H_step_mu300, Pres_H_step_mu400, Pres_H_step_mu500 = [], [], [], [], []

    Pres_pi_c_mu0, Pres_pi_c_mu200, Pres_pi_c_mu300, Pres_pi_c_mu400, Pres_pi_c_mu500 = [], [], [], [], []
    Pres_rho_c_mu0, Pres_rho_c_mu200, Pres_rho_c_mu300, Pres_rho_c_mu400, Pres_rho_c_mu500 = [], [], [], [], []
    Pres_omega_c_mu0, Pres_omega_c_mu200, Pres_omega_c_mu300, Pres_omega_c_mu400, Pres_omega_c_mu500 = [], [], [], [], []
    Pres_D_c_mu0, Pres_D_c_mu200, Pres_D_c_mu300, Pres_D_c_mu400, Pres_D_c_mu500 = [], [], [], [], []
    Pres_N_c_mu0, Pres_N_c_mu200, Pres_N_c_mu300, Pres_N_c_mu400, Pres_N_c_mu500 = [], [], [], [], []
    Pres_T_c_mu0, Pres_T_c_mu200, Pres_T_c_mu300, Pres_T_c_mu400, Pres_T_c_mu500 = [], [], [], [], []
    Pres_F_c_mu0, Pres_F_c_mu200, Pres_F_c_mu300, Pres_F_c_mu400, Pres_F_c_mu500 = [], [], [], [], []
    Pres_P_c_mu0, Pres_P_c_mu200, Pres_P_c_mu300, Pres_P_c_mu400, Pres_P_c_mu500 = [], [], [], [], []
    Pres_Q5_c_mu0, Pres_Q5_c_mu200, Pres_Q5_c_mu300, Pres_Q5_c_mu400, Pres_Q5_c_mu500 = [], [], [], [], []
    Pres_H_c_mu0, Pres_H_c_mu200, Pres_H_c_mu300, Pres_H_c_mu400, Pres_H_c_mu500 = [], [], [], [], []
    
    #T = numpy.linspace(1.0, 450.0, 200)
    T_mu0 = numpy.linspace(1.0, 2000.0, 200)
    T_mu200 = numpy.linspace(1.0, 2000.0, 200)
    T_mu300 = numpy.linspace(1.0, 2000.0, 200)
    T_mu400 = numpy.linspace(1.0, 2000.0, 200)
    T_mu500 = numpy.linspace(1.0, 2000.0, 200)
    mu0   = [0.0 for el in T_mu0]
    mu200 = [200.0 / 3.0 for el in T_mu200]
    mu300 = [300.0 / 3.0 for el in T_mu300]
    mu400 = [400.0 / 3.0 for el in T_mu400]
    mu500 = [500.0 / 3.0 for el in T_mu500]
    
    recalc_pl_mu0           = False
    recalc_pl_mu200         = False
    recalc_pl_mu300         = False
    recalc_pl_mu400         = False
    recalc_pl_mu500         = False

    recalc_pressure_mu0     = False
    recalc_pressure_mu200   = False
    recalc_pressure_mu300   = False
    recalc_pressure_mu400   = False
    recalc_pressure_mu500   = False

    recalc_pressure_mu0_c   = False
    recalc_pressure_mu200_c = False
    recalc_pressure_mu300_c = False
    recalc_pressure_mu400_c = False
    recalc_pressure_mu500_c = False

    cluster_backreaction    = False
    pl_turned_off           = False

    pl_mu0_file   = "D:/EoS/BDK/continuum_test/pl_mu0.dat"
    pl_mu200_file = "D:/EoS/BDK/continuum_test/pl_mu200.dat"
    pl_mu300_file = "D:/EoS/BDK/continuum_test/pl_mu300.dat"
    pl_mu400_file = "D:/EoS/BDK/continuum_test/pl_mu400.dat"
    pl_mu500_file = "D:/EoS/BDK/continuum_test/pl_mu500.dat"
    pressure_mu0_file   = "D:/EoS/BDK/continuum_test/pressure_mu0.dat"
    pressure_mu200_file = "D:/EoS/BDK/continuum_test/pressure_mu200.dat"
    pressure_mu300_file = "D:/EoS/BDK/continuum_test/pressure_mu300.dat"
    pressure_mu400_file = "D:/EoS/BDK/continuum_test/pressure_mu400.dat"
    pressure_mu500_file = "D:/EoS/BDK/continuum_test/pressure_mu500.dat"
    pressure_mu0_c_file   = "D:/EoS/BDK/continuum_test/pressure_c_mu0.dat"
    pressure_mu200_c_file = "D:/EoS/BDK/continuum_test/pressure_c_mu200.dat"
    pressure_mu300_c_file = "D:/EoS/BDK/continuum_test/pressure_c_mu300.dat"
    pressure_mu400_c_file = "D:/EoS/BDK/continuum_test/pressure_c_mu400.dat"
    pressure_mu500_c_file = "D:/EoS/BDK/continuum_test/pressure_c_mu500.dat"

    if recalc_pl_mu0:
        phi_re_mu0.append(1e-15)
        phi_im_mu0.append(2e-15)
        lT = len(T_mu0)
        for T_el, mu_el in tqdm.tqdm(zip(T_mu0, mu0), desc = "Traced Polyakov loop (mu = 0)", total = len(T_mu0), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_mu0[-1], phi_im_mu0[-1], with_clusters = cluster_backreaction)
            phi_re_mu0.append(temp_phi_re)
            phi_im_mu0.append(temp_phi_im)
        phi_re_mu0 = phi_re_mu0[1:]
        phi_im_mu0 = phi_im_mu0[1:]
        with open(pl_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T_mu0, phi_re_mu0, phi_im_mu0)])
    else:
        T_mu0, phi_re_mu0 = utils.data_collect(0, 1, pl_mu0_file)
        phi_im_mu0, _ = utils.data_collect(2, 2, pl_mu0_file)

    if recalc_pl_mu200:
        phi_re_mu200.append(1e-15)
        phi_im_mu200.append(2e-15)
        lT = len(T_mu200)
        for T_el, mu_el in tqdm.tqdm(zip(T_mu200, mu200), desc = "Traced Polyakov loop (mu = 200)", total = len(T_mu200), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_mu200[-1], phi_im_mu200[-1], with_clusters = cluster_backreaction)
            phi_re_mu200.append(temp_phi_re)
            phi_im_mu200.append(temp_phi_im)
        phi_re_mu200 = phi_re_mu200[1:]
        phi_im_mu200 = phi_im_mu200[1:]
        with open(pl_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T_mu200, phi_re_mu200, phi_im_mu200)])
    else:
        T_mu200, phi_re_mu200 = utils.data_collect(0, 1, pl_mu200_file)
        phi_im_mu200, _ = utils.data_collect(2, 2, pl_mu200_file)
    
    if recalc_pl_mu300:
        phi_re_mu300.append(1e-15)
        phi_im_mu300.append(2e-15)
        lT = len(T_mu300)
        for T_el, mu_el in tqdm.tqdm(zip(T_mu300, mu300), desc = "Traced Polyakov loop (mu = 300)", total = len(T_mu300), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_mu300[-1], phi_im_mu300[-1], with_clusters = cluster_backreaction)
            phi_re_mu300.append(temp_phi_re)
            phi_im_mu300.append(temp_phi_im)
        phi_re_mu300 = phi_re_mu300[1:]
        phi_im_mu300 = phi_im_mu300[1:]
        with open(pl_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T_mu300, phi_re_mu300, phi_im_mu300)])
    else:
        T_mu300, phi_re_mu300 = utils.data_collect(0, 1, pl_mu300_file)
        phi_im_mu300, _ = utils.data_collect(2, 2, pl_mu300_file)

    if recalc_pl_mu400:
        phi_re_mu400.append(1e-15)
        phi_im_mu400.append(2e-15)
        lT = len(T_mu400)
        for T_el, mu_el in tqdm.tqdm(zip(T_mu400, mu400), desc = "Traced Polyakov loop (mu = 400)", total = len(T_mu400), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_mu400[-1], phi_im_mu400[-1], with_clusters = cluster_backreaction)
            phi_re_mu400.append(temp_phi_re)
            phi_im_mu400.append(temp_phi_im)
        phi_re_mu400 = phi_re_mu400[1:]
        phi_im_mu400 = phi_im_mu400[1:]
        with open(pl_mu400_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T_mu400, phi_re_mu400, phi_im_mu400)])
    else:
        T_mu400, phi_re_mu400 = utils.data_collect(0, 1, pl_mu400_file)
        phi_im_mu400, _ = utils.data_collect(2, 2, pl_mu400_file)

    if recalc_pl_mu500:
        phi_re_mu500.append(1e-15)
        phi_im_mu500.append(2e-15)
        lT = len(T_mu500)
        for T_el, mu_el in tqdm.tqdm(zip(T_mu500, mu500), desc = "Traced Polyakov loop (mu = 500)", total = len(T_mu500), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_mu500[-1], phi_im_mu500[-1], with_clusters = cluster_backreaction)
            phi_re_mu500.append(temp_phi_re)
            phi_im_mu500.append(temp_phi_im)
        phi_re_mu500 = phi_re_mu500[1:]
        phi_im_mu500 = phi_im_mu500[1:]
        with open(pl_mu500_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T_mu500, phi_re_mu500, phi_im_mu500)])
    else:
        T_mu500, phi_re_mu500 = utils.data_collect(0, 1, pl_mu500_file)
        phi_im_mu500, _ = utils.data_collect(2, 2, pl_mu500_file)

    if recalc_pressure_mu0:
        Pres_g_mu0 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_mu0, phi_re_mu0, phi_im_mu0), 
                desc = "Gluon pressure (mu = 0)", 
                total = len(T_mu0),
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu0 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu0, mu0, phi_re_mu0, phi_im_mu0), 
                    desc = "Quark pressure (mu = 0)", 
                    total = len(T_mu0), 
                    ascii = True
                    )]
            Pres_pert_mu0 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu0, mu0, phi_re_mu0, phi_im_mu0), 
                    desc = "Perturbative pressure (mu = 0)", 
                    total = len(T_mu0), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu0, Pres_rho_step_mu0, Pres_omega_step_mu0, Pres_D_step_mu0, Pres_N_step_mu0, Pres_T_step_mu0, Pres_F_step_mu0, Pres_P_step_mu0, Pres_Q5_step_mu0, Pres_H_step_mu0),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu0, mu0, phi_re_mu0, phi_im_mu0)
        else:
            Pres_Q_mu0 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu0, mu0, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 0)", 
                    total = len(T_mu0), 
                    ascii = True
                    )]
            Pres_pert_mu0 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu0, mu0, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 0)", 
                    total = len(T_mu0), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu0, Pres_rho_step_mu0, Pres_omega_step_mu0, Pres_D_step_mu0, Pres_N_step_mu0, Pres_T_step_mu0, Pres_F_step_mu0, Pres_P_step_mu0, Pres_Q5_step_mu0, Pres_H_step_mu0),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu0, mu0, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu0, Pres_g_mu0, Pres_pert_mu0, Pres_pi_step_mu0, Pres_rho_step_mu0, Pres_omega_step_mu0, Pres_D_step_mu0, Pres_N_step_mu0, Pres_T_step_mu0, Pres_F_step_mu0, Pres_P_step_mu0, Pres_Q5_step_mu0, Pres_H_step_mu0)])
    else:
        Pres_Q_mu0, Pres_g_mu0       = utils.data_collect(0, 1, pressure_mu0_file)
        Pres_pert_mu0, Pres_pi_step_mu0   = utils.data_collect(2, 3, pressure_mu0_file)
        Pres_rho_step_mu0, Pres_omega_step_mu0 = utils.data_collect(4, 5, pressure_mu0_file)
        Pres_D_step_mu0, Pres_N_step_mu0       = utils.data_collect(6, 7, pressure_mu0_file)
        Pres_T_step_mu0, Pres_F_step_mu0       = utils.data_collect(8, 9, pressure_mu0_file)
        Pres_P_step_mu0, Pres_Q5_step_mu0      = utils.data_collect(10, 11, pressure_mu0_file)
        Pres_H_step_mu0, _                = utils.data_collect(12, 12, pressure_mu0_file)

    if recalc_pressure_mu0_c:
        if not pl_turned_off:
            (
                (Pres_pi_c_mu0, Pres_rho_c_mu0, Pres_omega_c_mu0, Pres_D_c_mu0, Pres_N_c_mu0, Pres_T_c_mu0, Pres_F_c_mu0, Pres_P_c_mu0, Pres_Q5_c_mu0, Pres_H_c_mu0),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu0, mu0, phi_re_mu0, phi_im_mu0)
        else:
            (
                (Pres_pi_c_mu0, Pres_rho_c_mu0, Pres_omega_c_mu0, Pres_D_c_mu0, Pres_N_c_mu0, Pres_T_c_mu0, Pres_F_c_mu0, Pres_P_c_mu0, Pres_Q5_c_mu0, Pres_H_c_mu0),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu0, mu0, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu0_c_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu0, Pres_g_mu0, Pres_pert_mu0, Pres_pi_c_mu0, Pres_rho_c_mu0, Pres_omega_c_mu0, Pres_D_c_mu0, Pres_N_c_mu0, Pres_T_c_mu0, Pres_F_c_mu0, Pres_P_c_mu0, Pres_Q5_c_mu0, Pres_H_c_mu0)])
    else:
        Pres_Q_mu0, Pres_g_mu0            = utils.data_collect(0, 1, pressure_mu0_c_file)
        Pres_pert_mu0, Pres_pi_c_mu0      = utils.data_collect(2, 3, pressure_mu0_c_file)
        Pres_rho_c_mu0, Pres_omega_c_mu0  = utils.data_collect(4, 5, pressure_mu0_c_file)
        Pres_D_c_mu0, Pres_N_c_mu0        = utils.data_collect(6, 7, pressure_mu0_c_file)
        Pres_T_c_mu0, Pres_F_c_mu0        = utils.data_collect(8, 9, pressure_mu0_c_file)
        Pres_P_c_mu0, Pres_Q5_c_mu0       = utils.data_collect(10, 11, pressure_mu0_c_file)
        Pres_H_c_mu0, _                   = utils.data_collect(12, 12, pressure_mu0_c_file)

    if recalc_pressure_mu200:
        Pres_g_mu200 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_mu200, phi_re_mu200, phi_im_mu200), 
                desc = "Gluon pressure (mu = 200)", 
                total = len(T_mu200), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu200 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu200, mu200, phi_re_mu200, phi_im_mu200), 
                    desc = "Quark pressure (mu = 200)", 
                    total = len(T_mu200), 
                    ascii = True
                    )]        
            Pres_pert_mu200 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu200, mu200, phi_re_mu200, phi_im_mu200), 
                    desc = "Perturbative pressure (mu = 200)", 
                    total = len(T_mu200), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu200, Pres_rho_step_mu200, Pres_omega_step_mu200, Pres_D_step_mu200, Pres_N_step_mu200, Pres_T_step_mu200, Pres_F_step_mu200, Pres_P_step_mu200, Pres_Q5_step_mu200, Pres_H_step_mu200),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu200, mu200, phi_re_mu200, phi_im_mu200)
        else:
            Pres_Q_mu200 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu200, mu200, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 200)", 
                    total = len(T_mu200), 
                    ascii = True
                    )]
            Pres_pert_mu200 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu200, mu200, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 200)", 
                    total = len(T_mu200), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu200, Pres_rho_step_mu200, Pres_omega_step_mu200, Pres_D_step_mu200, Pres_N_step_mu200, Pres_T_step_mu200, Pres_F_step_mu200, Pres_P_step_mu200, Pres_Q5_step_mu200, Pres_H_step_mu200),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu200, mu200, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu200, Pres_g_mu200, Pres_pert_mu200, Pres_pi_step_mu200, Pres_rho_step_mu200, Pres_omega_step_mu200, Pres_D_step_mu200, Pres_N_step_mu200, Pres_T_step_mu200, Pres_F_step_mu200, Pres_P_step_mu200, Pres_Q5_step_mu200, Pres_H_step_mu200)])
    else:
        Pres_Q_mu200, Pres_g_mu200       = utils.data_collect(0, 1, pressure_mu200_file)
        Pres_pert_mu200, Pres_pi_step_mu200   = utils.data_collect(2, 3, pressure_mu200_file)
        Pres_rho_step_mu200, Pres_omega_step_mu200 = utils.data_collect(4, 5, pressure_mu200_file)
        Pres_D_step_mu200, Pres_N_step_mu200       = utils.data_collect(6, 7, pressure_mu200_file)
        Pres_T_step_mu200, Pres_F_step_mu200       = utils.data_collect(8, 9, pressure_mu200_file)
        Pres_P_step_mu200, Pres_Q5_step_mu200      = utils.data_collect(10, 11, pressure_mu200_file)
        Pres_H_step_mu200, _                  = utils.data_collect(12, 12, pressure_mu200_file)
    
    if recalc_pressure_mu200_c:
        if not pl_turned_off:
            (
                (Pres_pi_c_mu200, Pres_rho_c_mu200, Pres_omega_c_mu200, Pres_D_c_mu200, Pres_N_c_mu200, Pres_T_c_mu200, Pres_F_c_mu200, Pres_P_c_mu200, Pres_Q5_c_mu200, Pres_H_c_mu200),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu200, mu200, phi_re_mu200, phi_im_mu200)
        else:
            (
                (Pres_pi_c_mu200, Pres_rho_c_mu200, Pres_omega_c_mu200, Pres_D_c_mu200, Pres_N_c_mu200, Pres_T_c_mu200, Pres_F_c_mu200, Pres_P_c_mu200, Pres_Q5_c_mu200, Pres_H_c_mu200),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu200, mu200, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu200_c_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu200, Pres_g_mu200, Pres_pert_mu200, Pres_pi_c_mu200, Pres_rho_c_mu200, Pres_omega_c_mu200, Pres_D_c_mu200, Pres_N_c_mu200, Pres_T_c_mu200, Pres_F_c_mu200, Pres_P_c_mu200, Pres_Q5_c_mu200, Pres_H_c_mu200)])
    else:
        Pres_Q_mu200, Pres_g_mu200       = utils.data_collect(0, 1, pressure_mu200_c_file)
        Pres_pert_mu200, Pres_pi_c_mu200   = utils.data_collect(2, 3, pressure_mu200_c_file)
        Pres_rho_c_mu200, Pres_omega_c_mu200 = utils.data_collect(4, 5, pressure_mu200_c_file)
        Pres_D_c_mu200, Pres_N_c_mu200       = utils.data_collect(6, 7, pressure_mu200_c_file)
        Pres_T_c_mu200, Pres_F_c_mu200       = utils.data_collect(8, 9, pressure_mu200_c_file)
        Pres_P_c_mu200, Pres_Q5_c_mu200      = utils.data_collect(10, 11, pressure_mu200_c_file)
        Pres_H_c_mu200, _                  = utils.data_collect(12, 12, pressure_mu200_c_file)

    if recalc_pressure_mu300:
        Pres_g_mu300 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_mu300, phi_re_mu300, phi_im_mu300), 
                desc = "Gluon pressure (mu = 300)", 
                total = len(T_mu300), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu300 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu300, mu300, phi_re_mu300, phi_im_mu300), 
                    desc = "Quark pressure (mu = 300)", 
                    total = len(T_mu300), 
                    ascii = True
                    )]
            Pres_pert_mu300 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu300, mu300, phi_re_mu300, phi_im_mu300), 
                    desc = "Perturbative pressure (mu = 300)", 
                    total = len(T_mu300), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu300, Pres_rho_step_mu300, Pres_omega_step_mu300, Pres_D_step_mu300, Pres_N_step_mu300, Pres_T_step_mu300, Pres_F_step_mu300, Pres_P_step_mu300, Pres_Q5_step_mu300, Pres_H_step_mu300),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu300, mu300, phi_re_mu300, phi_im_mu300)
        else:
            Pres_Q_mu300 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el in 
                tqdm.tqdm(
                    zip(T_mu300, mu300, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 300)", 
                    total = len(T_mu300), 
                    ascii = True
                    )]
            Pres_pert_mu300 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu300, mu300, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 300)", 
                    total = len(T_mu300), 
                    ascii = True
                    )]
            (
                (Pres_pi_step_mu300, Pres_rho_step_mu300, Pres_omega_step_mu300, Pres_D_step_mu300, Pres_N_step_mu300, Pres_T_step_mu300, Pres_F_step_mu300, Pres_P_step_mu300, Pres_Q5_step_mu300, Pres_H_step_mu300),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu300, mu300, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu300, Pres_g_mu300, Pres_pert_mu300, Pres_pi_step_mu300, Pres_rho_step_mu300, Pres_omega_step_mu300, Pres_D_step_mu300, Pres_N_step_mu300, Pres_T_step_mu300, Pres_F_step_mu300, Pres_P_step_mu300, Pres_Q5_step_mu300, Pres_H_step_mu300)])
    else:
        Pres_Q_mu300, Pres_g_mu300       = utils.data_collect(0, 1, pressure_mu300_file)
        Pres_pert_mu300, Pres_pi_step_mu300   = utils.data_collect(2, 3, pressure_mu300_file)
        Pres_rho_step_mu300, Pres_omega_step_mu300 = utils.data_collect(4, 5, pressure_mu300_file)
        Pres_D_step_mu300, Pres_N_step_mu300       = utils.data_collect(6, 7, pressure_mu300_file)
        Pres_T_step_mu300, Pres_F_step_mu300       = utils.data_collect(8, 9, pressure_mu300_file)
        Pres_P_step_mu300, Pres_Q5_step_mu300      = utils.data_collect(10, 11, pressure_mu300_file)
        Pres_H_step_mu300, _                  = utils.data_collect(12, 12, pressure_mu300_file)

    if recalc_pressure_mu300_c:
        if not pl_turned_off:
            (
                (Pres_pi_c_mu300, Pres_rho_c_mu300, Pres_omega_c_mu300, Pres_D_c_mu300, Pres_N_c_mu300, Pres_T_c_mu300, Pres_F_c_mu300, Pres_P_c_mu300, Pres_Q5_c_mu300, Pres_H_c_mu300),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu300, mu300, phi_re_mu300, phi_im_mu300)
        else:
            (
                (Pres_pi_c_mu300, Pres_rho_c_mu300, Pres_omega_c_mu300, Pres_D_c_mu300, Pres_N_c_mu300, Pres_T_c_mu300, Pres_F_c_mu300, Pres_P_c_mu300, Pres_Q5_c_mu300, Pres_H_c_mu300),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu300, mu300, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu300_c_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu300, Pres_g_mu300, Pres_pert_mu300, Pres_pi_c_mu300, Pres_rho_c_mu300, Pres_omega_c_mu300, Pres_D_c_mu300, Pres_N_c_mu300, Pres_T_c_mu300, Pres_F_c_mu300, Pres_P_c_mu300, Pres_Q5_c_mu300, Pres_H_c_mu300)])
    else:
        Pres_Q_mu300, Pres_g_mu300       = utils.data_collect(0, 1, pressure_mu300_c_file)
        Pres_pert_mu300, Pres_pi_c_mu300   = utils.data_collect(2, 3, pressure_mu300_c_file)
        Pres_rho_c_mu300, Pres_omega_c_mu300 = utils.data_collect(4, 5, pressure_mu300_c_file)
        Pres_D_c_mu300, Pres_N_c_mu300       = utils.data_collect(6, 7, pressure_mu300_c_file)
        Pres_T_c_mu300, Pres_F_c_mu300       = utils.data_collect(8, 9, pressure_mu300_c_file)
        Pres_P_c_mu300, Pres_Q5_c_mu300      = utils.data_collect(10, 11, pressure_mu300_c_file)
        Pres_H_c_mu300, _                  = utils.data_collect(12, 12, pressure_mu300_c_file)

    if recalc_pressure_mu400:
        Pres_g_mu400 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_mu400, phi_re_mu400, phi_im_mu400), 
                desc = "Gluon pressure (mu = 400)", 
                total = len(T_mu400), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu400 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu400, mu400, phi_re_mu400, phi_im_mu400), 
                    desc = "Quark pressure (mu = 400)", 
                    total = len(T_mu400), 
                    ascii = True
                    )]
            Pres_pert_mu400 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu400, mu400, phi_re_mu400, phi_im_mu400), 
                    desc = "Perturbative pressure (mu = 400)", 
                    total = len(T_mu400), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu400, Pres_rho_step_mu400, Pres_omega_step_mu400, Pres_D_step_mu400, Pres_N_step_mu400, Pres_T_step_mu400, Pres_F_step_mu400, Pres_P_step_mu400, Pres_Q5_step_mu400, Pres_H_step_mu400),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu400, mu400, phi_re_mu400, phi_im_mu400)
        else:
            Pres_Q_mu400 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el in 
                tqdm.tqdm(
                    zip(T_mu400, mu400, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 400)", 
                    total = len(T_mu400), 
                    ascii = True
                    )]
            Pres_pert_mu400 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu400, mu400, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 400)", 
                    total = len(T_mu400), 
                    ascii = True
                    )]
            (
                (Pres_pi_step_mu400, Pres_rho_step_mu400, Pres_omega_step_mu400, Pres_D_step_mu400, Pres_N_step_mu400, Pres_T_step_mu400, Pres_F_step_mu400, Pres_P_step_mu400, Pres_Q5_step_mu400, Pres_H_step_mu400),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu400, mu400, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu400_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu400, Pres_g_mu400, Pres_pert_mu400, Pres_pi_step_mu400, Pres_rho_step_mu400, Pres_omega_step_mu400, Pres_D_step_mu400, Pres_N_step_mu400, Pres_T_step_mu400, Pres_F_step_mu400, Pres_P_step_mu400, Pres_Q5_step_mu400, Pres_H_step_mu400)])
    else:
        Pres_Q_mu400, Pres_g_mu400                 = utils.data_collect(0, 1, pressure_mu400_file)
        Pres_pert_mu400, Pres_pi_step_mu400        = utils.data_collect(2, 3, pressure_mu400_file)
        Pres_rho_step_mu400, Pres_omega_step_mu400 = utils.data_collect(4, 5, pressure_mu400_file)
        Pres_D_step_mu400, Pres_N_step_mu400       = utils.data_collect(6, 7, pressure_mu400_file)
        Pres_T_step_mu400, Pres_F_step_mu400       = utils.data_collect(8, 9, pressure_mu400_file)
        Pres_P_step_mu400, Pres_Q5_step_mu400      = utils.data_collect(10, 11, pressure_mu400_file)
        Pres_H_step_mu400, _                       = utils.data_collect(12, 12, pressure_mu400_file)

    if recalc_pressure_mu400_c:
        if not pl_turned_off:
            (
                (Pres_pi_c_mu400, Pres_rho_c_mu400, Pres_omega_c_mu400, Pres_D_c_mu400, Pres_N_c_mu400, Pres_T_c_mu400, Pres_F_c_mu400, Pres_P_c_mu400, Pres_Q5_c_mu400, Pres_H_c_mu400),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu400, mu400, phi_re_mu400, phi_im_mu400)
        else:
            (
                (Pres_pi_c_mu400, Pres_rho_c_mu400, Pres_omega_c_mu400, Pres_D_c_mu400, Pres_N_c_mu400, Pres_T_c_mu400, Pres_F_c_mu400, Pres_P_c_mu400, Pres_Q5_c_mu400, Pres_H_c_mu400),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu400, mu400, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu400_c_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu400, Pres_g_mu400, Pres_pert_mu400, Pres_pi_c_mu400, Pres_rho_c_mu400, Pres_omega_c_mu400, Pres_D_c_mu400, Pres_N_c_mu400, Pres_T_c_mu400, Pres_F_c_mu400, Pres_P_c_mu400, Pres_Q5_c_mu400, Pres_H_c_mu400)])
    else:
        Pres_Q_mu400, Pres_g_mu400       = utils.data_collect(0, 1, pressure_mu400_c_file)
        Pres_pert_mu400, Pres_pi_c_mu400   = utils.data_collect(2, 3, pressure_mu400_c_file)
        Pres_rho_c_mu400, Pres_omega_c_mu400 = utils.data_collect(4, 5, pressure_mu400_c_file)
        Pres_D_c_mu400, Pres_N_c_mu400       = utils.data_collect(6, 7, pressure_mu400_c_file)
        Pres_T_c_mu400, Pres_F_c_mu400       = utils.data_collect(8, 9, pressure_mu400_c_file)
        Pres_P_c_mu400, Pres_Q5_c_mu400      = utils.data_collect(10, 11, pressure_mu400_c_file)
        Pres_H_c_mu400, _                  = utils.data_collect(12, 12, pressure_mu400_c_file)

    if recalc_pressure_mu500:
        Pres_g_mu500 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(
                zip(T_mu500, phi_re_mu500, phi_im_mu500), 
                desc = "Gluon pressure (mu = 500)", 
                total = len(T_mu500), 
                ascii = True
                )]
        if not pl_turned_off:
            Pres_Q_mu500 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu500, mu500, phi_re_mu500, phi_im_mu500), 
                    desc = "Quark pressure (mu = 500)", 
                    total = len(T_mu500), 
                    ascii = True
                    )]
            Pres_pert_mu500 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu500, mu500, phi_re_mu500, phi_im_mu500), 
                    desc = "Perturbative pressure (mu = 500)", 
                    total = len(T_mu500), 
                    ascii = True
                    )]        
            (
                (Pres_pi_step_mu500, Pres_rho_step_mu500, Pres_omega_step_mu500, Pres_D_step_mu500, Pres_N_step_mu500, Pres_T_step_mu500, Pres_F_step_mu500, Pres_P_step_mu500, Pres_Q5_step_mu500, Pres_H_step_mu500),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu500, mu500, phi_re_mu500, phi_im_mu500)
        else:
            Pres_Q_mu500 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) 
                for T_el, mu_el, phi_re_el, phi_im_el in 
                tqdm.tqdm(
                    zip(T_mu500, mu500, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Quark pressure (mu = 500)", 
                    total = len(T_mu500), 
                    ascii = True
                    )]
            Pres_pert_mu500 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(
                    zip(T_mu500, mu500, [1.0 for el in T], [0.0 for el in T]), 
                    desc = "Perturbative pressure (mu = 500)", 
                    total = len(T_mu500), 
                    ascii = True
                    )]
            (
                (Pres_pi_step_mu500, Pres_rho_step_mu500, Pres_omega_step_mu500, Pres_D_step_mu500, Pres_N_step_mu500, Pres_T_step_mu500, Pres_F_step_mu500, Pres_P_step_mu500, Pres_Q5_step_mu500, Pres_H_step_mu500),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T_mu500, mu500, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu500_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu500, Pres_g_mu500, Pres_pert_mu500, Pres_pi_step_mu500, Pres_rho_step_mu500, Pres_omega_step_mu500, Pres_D_step_mu500, Pres_N_step_mu500, Pres_T_step_mu500, Pres_F_step_mu500, Pres_P_step_mu500, Pres_Q5_step_mu500, Pres_H_step_mu500)])
    else:
        Pres_Q_mu500, Pres_g_mu500                 = utils.data_collect(0, 1, pressure_mu500_file)
        Pres_pert_mu500, Pres_pi_step_mu500        = utils.data_collect(2, 3, pressure_mu500_file)
        Pres_rho_step_mu500, Pres_omega_step_mu500 = utils.data_collect(4, 5, pressure_mu500_file)
        Pres_D_step_mu500, Pres_N_step_mu500       = utils.data_collect(6, 7, pressure_mu500_file)
        Pres_T_step_mu500, Pres_F_step_mu500       = utils.data_collect(8, 9, pressure_mu500_file)
        Pres_P_step_mu500, Pres_Q5_step_mu500      = utils.data_collect(10, 11, pressure_mu500_file)
        Pres_H_step_mu500, _                       = utils.data_collect(12, 12, pressure_mu500_file)

    if recalc_pressure_mu500_c:
        if not pl_turned_off:
            (
                (Pres_pi_c_mu500, Pres_rho_c_mu500, Pres_omega_c_mu500, Pres_D_c_mu500, Pres_N_c_mu500, Pres_T_c_mu500, Pres_F_c_mu500, Pres_P_c_mu500, Pres_Q5_c_mu500, Pres_H_c_mu500),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu500, mu500, phi_re_mu500, phi_im_mu500)
        else:
            (
                (Pres_pi_c_mu500, Pres_rho_c_mu500, Pres_omega_c_mu500, Pres_D_c_mu500, Pres_N_c_mu500, Pres_T_c_mu500, Pres_F_c_mu500, Pres_P_c_mu500, Pres_Q5_c_mu500, Pres_H_c_mu500),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters_c(T_mu500, mu500, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu500_c_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu500, Pres_g_mu500, Pres_pert_mu500, Pres_pi_c_mu500, Pres_rho_c_mu500, Pres_omega_c_mu500, Pres_D_c_mu500, Pres_N_c_mu500, Pres_T_c_mu500, Pres_F_c_mu500, Pres_P_c_mu500, Pres_Q5_c_mu500, Pres_H_c_mu500)])
    else:
        Pres_Q_mu500, Pres_g_mu500       = utils.data_collect(0, 1, pressure_mu500_c_file)
        Pres_pert_mu500, Pres_pi_c_mu500   = utils.data_collect(2, 3, pressure_mu500_c_file)
        Pres_rho_c_mu500, Pres_omega_c_mu500 = utils.data_collect(4, 5, pressure_mu500_c_file)
        Pres_D_c_mu500, Pres_N_c_mu500       = utils.data_collect(6, 7, pressure_mu500_c_file)
        Pres_T_c_mu500, Pres_F_c_mu500       = utils.data_collect(8, 9, pressure_mu500_c_file)
        Pres_P_c_mu500, Pres_Q5_c_mu500      = utils.data_collect(10, 11, pressure_mu500_c_file)
        Pres_H_c_mu500, _                  = utils.data_collect(12, 12, pressure_mu500_c_file)

    contrib_q_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_Q_mu0   )]
    contrib_g_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_g_mu0   )]
    #contrib_pert_mu0                = [0.0 for T_el, p_el in zip(T_mu0, Pres_pert_mu0)]
    contrib_pert_mu0                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_pert_mu0)]
    contrib_qgp_mu0                 = [sum(el) for el in zip(contrib_q_mu0, contrib_g_mu0, contrib_pert_mu0)]
    contrib_q_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_Q_mu200   )]
    contrib_g_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_g_mu200   )]
    #contrib_pert_mu200              = [0.0 for T_el, p_el in zip(T_mu200, Pres_pert_mu0)]
    contrib_pert_mu200              = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_pert_mu200)]
    contrib_qgp_mu200               = [sum(el) for el in zip(contrib_q_mu200, contrib_g_mu200, contrib_pert_mu200)]
    contrib_q_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_Q_mu300   )]
    contrib_g_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_g_mu300   )]
    #contrib_pert_mu300              = [0.0 for T_el, p_el in zip(T_mu300, Pres_pert_mu0)]
    contrib_pert_mu300              = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_pert_mu300)]
    contrib_qgp_mu300               = [sum(el) for el in zip(contrib_q_mu300, contrib_g_mu300, contrib_pert_mu300)]
    contrib_q_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_Q_mu400   )]
    contrib_g_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_g_mu400   )]
    #contrib_pert_mu400              = [0.0 for T_el, p_el in zip(T_mu400, Pres_pert_mu0)]
    contrib_pert_mu400              = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_pert_mu400)]
    contrib_qgp_mu400               = [sum(el) for el in zip(contrib_q_mu400, contrib_g_mu400, contrib_pert_mu400)]
    contrib_q_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_Q_mu500   )]
    contrib_g_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_g_mu500   )]
    #contrib_pert_mu500              = [0.0 for T_el, p_el in zip(T_mu500, Pres_pert_mu0)]
    contrib_pert_mu500              = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_pert_mu500)]
    contrib_qgp_mu500               = [sum(el) for el in zip(contrib_q_mu500, contrib_g_mu500, contrib_pert_mu500)]

    contrib_pi_step_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_pi_step_mu0)]
    contrib_rho_step_mu0                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_rho_step_mu0)]
    contrib_omega_step_mu0               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_omega_step_mu0)]
    contrib_D_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_D_step_mu0)]
    contrib_N_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_N_step_mu0)]
    contrib_T_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_T_step_mu0)]
    contrib_F_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_F_step_mu0)]
    contrib_P_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_P_step_mu0)]
    contrib_Q5_step_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_Q5_step_mu0)]
    contrib_H_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_H_step_mu0)]
    contrib_cluster_step_mu0             = [sum(el) for el in zip(contrib_pi_step_mu0, contrib_rho_step_mu0, contrib_omega_step_mu0, contrib_D_step_mu0, contrib_N_step_mu0, contrib_T_step_mu0, contrib_F_step_mu0, contrib_P_step_mu0, contrib_Q5_step_mu0, contrib_H_step_mu0)]
    contrib_cluster_singlet_step_mu0     = [sum(el) for el in zip(contrib_pi_step_mu0, contrib_rho_step_mu0, contrib_omega_step_mu0, contrib_N_step_mu0, contrib_T_step_mu0, contrib_P_step_mu0, contrib_H_step_mu0)]
    contrib_cluster_color_step_mu0       = [sum(el) for el in zip(contrib_D_step_mu0, contrib_F_step_mu0, contrib_Q5_step_mu0)]
    contrib_pi_step_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_pi_step_mu200)]
    contrib_rho_step_mu200               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_rho_step_mu200)]
    contrib_omega_step_mu200             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_omega_step_mu200)]
    contrib_D_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_D_step_mu200)]
    contrib_N_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_N_step_mu200)]
    contrib_T_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_T_step_mu200)]
    contrib_F_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_F_step_mu200)]
    contrib_P_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_P_step_mu200)]
    contrib_Q5_step_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_Q5_step_mu200)]
    contrib_H_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_H_step_mu200)]
    contrib_cluster_step_mu200           = [sum(el) for el in zip(contrib_pi_step_mu200, contrib_rho_step_mu200, contrib_omega_step_mu200, contrib_D_step_mu200, contrib_N_step_mu200, contrib_T_step_mu200, contrib_F_step_mu200, contrib_P_step_mu200, contrib_Q5_step_mu200, contrib_H_step_mu200)]
    contrib_cluster_singlet_step_mu200   = [sum(el) for el in zip(contrib_pi_step_mu200, contrib_rho_step_mu200, contrib_omega_step_mu200, contrib_N_step_mu200, contrib_T_step_mu200, contrib_P_step_mu200, contrib_H_step_mu200)]
    contrib_cluster_color_step_mu200     = [sum(el) for el in zip(contrib_D_step_mu200, contrib_F_step_mu200, contrib_Q5_step_mu200)]
    contrib_pi_step_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_pi_step_mu300)]
    contrib_rho_step_mu300               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_rho_step_mu300)]
    contrib_omega_step_mu300             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_omega_step_mu300)]
    contrib_D_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_D_step_mu300)]
    contrib_N_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_N_step_mu300)]
    contrib_T_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_T_step_mu300)]
    contrib_F_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_F_step_mu300)]
    contrib_P_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_P_step_mu300)]
    contrib_Q5_step_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_Q5_step_mu300)]
    contrib_H_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_H_step_mu300)]
    contrib_cluster_step_mu300           = [sum(el) for el in zip(contrib_pi_step_mu300, contrib_rho_step_mu300, contrib_omega_step_mu300, contrib_D_step_mu300, contrib_N_step_mu300, contrib_T_step_mu300, contrib_F_step_mu300, contrib_P_step_mu300, contrib_Q5_step_mu300, contrib_H_step_mu300)]
    contrib_cluster_singlet_step_mu300   = [sum(el) for el in zip(contrib_pi_step_mu300, contrib_rho_step_mu300, contrib_omega_step_mu300, contrib_N_step_mu300, contrib_T_step_mu300, contrib_P_step_mu300, contrib_H_step_mu300)]
    contrib_cluster_color_step_mu300     = [sum(el) for el in zip(contrib_D_step_mu300, contrib_F_step_mu300, contrib_Q5_step_mu300)]
    contrib_pi_step_mu400                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_pi_step_mu400)]
    contrib_rho_step_mu400               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_rho_step_mu400)]
    contrib_omega_step_mu400             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_omega_step_mu400)]
    contrib_D_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_D_step_mu400)]
    contrib_N_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_N_step_mu400)]
    contrib_T_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_T_step_mu400)]
    contrib_F_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_F_step_mu400)]
    contrib_P_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_P_step_mu400)]
    contrib_Q5_step_mu400                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_Q5_step_mu400)]
    contrib_H_step_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_H_step_mu400)]
    contrib_cluster_step_mu400           = [sum(el) for el in zip(contrib_pi_step_mu400, contrib_rho_step_mu400, contrib_omega_step_mu400, contrib_D_step_mu400, contrib_N_step_mu400, contrib_T_step_mu400, contrib_F_step_mu400, contrib_P_step_mu400, contrib_Q5_step_mu400, contrib_H_step_mu400)]
    contrib_cluster_singlet_step_mu400   = [sum(el) for el in zip(contrib_pi_step_mu400, contrib_rho_step_mu400, contrib_omega_step_mu400, contrib_N_step_mu400, contrib_T_step_mu400, contrib_P_step_mu400, contrib_H_step_mu400)]
    contrib_cluster_color_step_mu400     = [sum(el) for el in zip(contrib_D_step_mu400, contrib_F_step_mu400, contrib_Q5_step_mu400)]
    contrib_pi_step_mu500                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_pi_step_mu500)]
    contrib_rho_step_mu500               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_rho_step_mu500)]
    contrib_omega_step_mu500             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_omega_step_mu500)]
    contrib_D_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_D_step_mu500)]
    contrib_N_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_N_step_mu500)]
    contrib_T_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_T_step_mu500)]
    contrib_F_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_F_step_mu500)]
    contrib_P_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_P_step_mu500)]
    contrib_Q5_step_mu500                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_Q5_step_mu500)]
    contrib_H_step_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_H_step_mu500)]
    contrib_cluster_step_mu500           = [sum(el) for el in zip(contrib_pi_step_mu500, contrib_rho_step_mu500, contrib_omega_step_mu500, contrib_D_step_mu500, contrib_N_step_mu500, contrib_T_step_mu500, contrib_F_step_mu500, contrib_P_step_mu500, contrib_Q5_step_mu500, contrib_H_step_mu500)]
    contrib_cluster_singlet_step_mu500   = [sum(el) for el in zip(contrib_pi_step_mu500, contrib_rho_step_mu500, contrib_omega_step_mu500, contrib_N_step_mu500, contrib_T_step_mu500, contrib_P_step_mu500, contrib_H_step_mu500)]
    contrib_cluster_color_step_mu500     = [sum(el) for el in zip(contrib_D_step_mu500, contrib_F_step_mu500, contrib_Q5_step_mu500)]

    contrib_pi_c_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_pi_c_mu0)]
    contrib_rho_c_mu0                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_rho_c_mu0)]
    contrib_omega_c_mu0               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_omega_c_mu0)]
    contrib_D_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_D_c_mu0)]
    contrib_N_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_N_c_mu0)]
    contrib_T_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_T_c_mu0)]
    contrib_F_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_F_c_mu0)]
    contrib_P_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_P_c_mu0)]
    contrib_Q5_c_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_Q5_c_mu0)]
    contrib_H_c_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu0, Pres_H_c_mu0)]
    contrib_cluster_c_mu0             = [sum(el) for el in zip(contrib_pi_c_mu0, contrib_rho_c_mu0, contrib_omega_c_mu0, contrib_D_c_mu0, contrib_N_c_mu0, contrib_T_c_mu0, contrib_F_c_mu0, contrib_P_c_mu0, contrib_Q5_c_mu0, contrib_H_c_mu0)]
    contrib_cluster_singlet_c_mu0     = [sum(el) for el in zip(contrib_pi_c_mu0, contrib_rho_c_mu0, contrib_omega_c_mu0, contrib_N_c_mu0, contrib_T_c_mu0, contrib_P_c_mu0, contrib_H_c_mu0)]
    contrib_cluster_color_c_mu0       = [sum(el) for el in zip(contrib_D_c_mu0, contrib_F_c_mu0, contrib_Q5_c_mu0)]
    contrib_pi_c_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_pi_c_mu200)]
    contrib_rho_c_mu200               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_rho_c_mu200)]
    contrib_omega_c_mu200             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_omega_c_mu200)]
    contrib_D_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_D_c_mu200)]
    contrib_N_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_N_c_mu200)]
    contrib_T_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_T_c_mu200)]
    contrib_F_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_F_c_mu200)]
    contrib_P_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_P_c_mu200)]
    contrib_Q5_c_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_Q5_c_mu200)]
    contrib_H_c_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu200, Pres_H_c_mu200)]
    contrib_cluster_c_mu200           = [sum(el) for el in zip(contrib_pi_c_mu200, contrib_rho_c_mu200, contrib_omega_c_mu200, contrib_D_c_mu200, contrib_N_c_mu200, contrib_T_c_mu200, contrib_F_c_mu200, contrib_P_c_mu200, contrib_Q5_c_mu200, contrib_H_c_mu200)]
    contrib_cluster_singlet_c_mu200   = [sum(el) for el in zip(contrib_pi_c_mu200, contrib_rho_c_mu200, contrib_omega_c_mu200, contrib_N_c_mu200, contrib_T_c_mu200, contrib_P_c_mu200, contrib_H_c_mu200)]
    contrib_cluster_color_c_mu200     = [sum(el) for el in zip(contrib_D_c_mu200, contrib_F_c_mu200, contrib_Q5_c_mu200)]
    contrib_pi_c_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_pi_c_mu300)]
    contrib_rho_c_mu300               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_rho_c_mu300)]
    contrib_omega_c_mu300             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_omega_c_mu300)]
    contrib_D_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_D_c_mu300)]
    contrib_N_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_N_c_mu300)]
    contrib_T_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_T_c_mu300)]
    contrib_F_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_F_c_mu300)]
    contrib_P_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_P_c_mu300)]
    contrib_Q5_c_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_Q5_c_mu300)]
    contrib_H_c_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu300, Pres_H_c_mu300)]
    contrib_cluster_c_mu300           = [sum(el) for el in zip(contrib_pi_c_mu300, contrib_rho_c_mu300, contrib_omega_c_mu300, contrib_D_c_mu300, contrib_N_c_mu300, contrib_T_c_mu300, contrib_F_c_mu300, contrib_P_c_mu300, contrib_Q5_c_mu300, contrib_H_c_mu300)]
    contrib_cluster_singlet_c_mu300   = [sum(el) for el in zip(contrib_pi_c_mu300, contrib_rho_c_mu300, contrib_omega_c_mu300, contrib_N_c_mu300, contrib_T_c_mu300, contrib_P_c_mu300, contrib_H_c_mu300)]
    contrib_cluster_color_c_mu300     = [sum(el) for el in zip(contrib_D_c_mu300, contrib_F_c_mu300, contrib_Q5_c_mu300)]
    contrib_pi_c_mu400                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_pi_c_mu400)]
    contrib_rho_c_mu400               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_rho_c_mu400)]
    contrib_omega_c_mu400             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_omega_c_mu400)]
    contrib_D_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_D_c_mu400)]
    contrib_N_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_N_c_mu400)]
    contrib_T_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_T_c_mu400)]
    contrib_F_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_F_c_mu400)]
    contrib_P_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_P_c_mu400)]
    contrib_Q5_c_mu400                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_Q5_c_mu400)]
    contrib_H_c_mu400                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu400, Pres_H_c_mu400)]
    contrib_cluster_c_mu400           = [sum(el) for el in zip(contrib_pi_c_mu400, contrib_rho_c_mu400, contrib_omega_c_mu400, contrib_D_c_mu400, contrib_N_c_mu400, contrib_T_c_mu400, contrib_F_c_mu400, contrib_P_c_mu400, contrib_Q5_c_mu400, contrib_H_c_mu400)]
    contrib_cluster_singlet_c_mu400   = [sum(el) for el in zip(contrib_pi_c_mu400, contrib_rho_c_mu400, contrib_omega_c_mu400, contrib_N_c_mu400, contrib_T_c_mu400, contrib_P_c_mu400, contrib_H_c_mu400)]
    contrib_cluster_color_c_mu400     = [sum(el) for el in zip(contrib_D_c_mu400, contrib_F_c_mu400, contrib_Q5_c_mu400)]
    contrib_pi_c_mu500                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_pi_c_mu500)]
    contrib_rho_c_mu500               = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_rho_c_mu500)]
    contrib_omega_c_mu500             = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_omega_c_mu500)]
    contrib_D_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_D_c_mu500)]
    contrib_N_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_N_c_mu500)]
    contrib_T_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_T_c_mu500)]
    contrib_F_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_F_c_mu500)]
    contrib_P_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_P_c_mu500)]
    contrib_Q5_c_mu500                = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_Q5_c_mu500)]
    contrib_H_c_mu500                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_mu500, Pres_H_c_mu500)]
    contrib_cluster_c_mu500           = [sum(el) for el in zip(contrib_pi_c_mu500, contrib_rho_c_mu500, contrib_omega_c_mu500, contrib_D_c_mu500, contrib_N_c_mu500, contrib_T_c_mu500, contrib_F_c_mu500, contrib_P_c_mu500, contrib_Q5_c_mu500, contrib_H_c_mu500)]
    contrib_cluster_singlet_c_mu500   = [sum(el) for el in zip(contrib_pi_c_mu500, contrib_rho_c_mu500, contrib_omega_c_mu500, contrib_N_c_mu500, contrib_T_c_mu500, contrib_P_c_mu500, contrib_H_c_mu500)]
    contrib_cluster_color_c_mu500     = [sum(el) for el in zip(contrib_D_c_mu500, contrib_F_c_mu500, contrib_Q5_c_mu500)]

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y)       = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)     = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_low.dat")
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_high.dat")
    (low_1204_6710v2_mu400_x, low_1204_6710v2_mu400_y)   = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu400_low.dat")
    (high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y) = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu400_high.dat")
    (high_1407_6387_mu0_x, high_1407_6387_mu0_y)         = utils.data_collect(0, 3, "D:/EoS/archive/BDK/lattice_data/const_mu/1407_6387_table1_pressure_mu0.dat")
    (low_1407_6387_mu0_x, low_1407_6387_mu0_y)           = utils.data_collect(0, 2, "D:/EoS/archive/BDK/lattice_data/const_mu/1407_6387_table1_pressure_mu0.dat")
    (high_1309_5258_mu0_x, high_1309_5258_mu0_y)         = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1309_5258_figure6_pressure_mu0_high.dat")
    (low_1309_5258_mu0_x, low_1309_5258_mu0_y)           = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1309_5258_figure6_pressure_mu0_low.dat")
    (high_1710_05024_mu0_x, high_1710_05024_mu0_y)       = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1710_05024_figure8_pressure_mu0_high.dat")
    (low_1710_05024_mu0_x, low_1710_05024_mu0_y)         = utils.data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1710_05024_figure8_pressure_mu0_low.dat")

    borsanyi_1204_6710v2_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = numpy.array(borsanyi_1204_6710v2_mu0)
    borsanyi_1204_6710v2_mu200 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu200_x[::-1], low_1204_6710v2_mu200_y[::-1]):
        borsanyi_1204_6710v2_mu200.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu200 = numpy.array(borsanyi_1204_6710v2_mu200)
    borsanyi_1204_6710v2_mu300 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu300_x[::-1], low_1204_6710v2_mu300_y[::-1]):
        borsanyi_1204_6710v2_mu300.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu300 = numpy.array(borsanyi_1204_6710v2_mu300)
    borsanyi_1204_6710v2_mu400 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu400_x[::-1], low_1204_6710v2_mu400_y[::-1]):
        borsanyi_1204_6710v2_mu400.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu400 = numpy.array(borsanyi_1204_6710v2_mu400)
    bazavov_1407_6387_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1407_6387_mu0_x, high_1407_6387_mu0_y)]
    for x_el, y_el in zip(low_1407_6387_mu0_x[::-1], low_1407_6387_mu0_y[::-1]):
        bazavov_1407_6387_mu0.append(numpy.array([x_el, y_el]))
    bazavov_1407_6387_mu0 = numpy.array(bazavov_1407_6387_mu0)
    borsanyi_1309_5258_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1309_5258_mu0_x, high_1309_5258_mu0_y)]
    for x_el, y_el in zip(low_1309_5258_mu0_x[::-1], low_1309_5258_mu0_y[::-1]):
        borsanyi_1309_5258_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1309_5258_mu0 = numpy.array(borsanyi_1309_5258_mu0)
    bazavov_1710_05024_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1710_05024_mu0_x, high_1710_05024_mu0_y)]
    for x_el, y_el in zip(low_1710_05024_mu0_x[::-1], low_1710_05024_mu0_y[::-1]):
        bazavov_1710_05024_mu0.append(numpy.array([x_el, y_el]))
    bazavov_1710_05024_mu0 = numpy.array(bazavov_1710_05024_mu0)

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    #ax1.axis([0., 450., 0., 4.0])
    ax1.axis([50., 300., 0., 0.7])
    ax1.add_patch(matplotlib.patches.Polygon(bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.5, label = r'Bazavov et al. (2014), $\mathrm{\mu=0}$'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2014), $\mathrm{\mu=0}$'))
    #ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=0}$'))
    #ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'red', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=200}$ MeV'))
    #ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu300, closed = True, fill = True, color = 'yellow', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=300}$ MeV'))
    ax1.plot(T_mu0, contrib_q_mu0, '-', c = 'blue', label = r'$\mathrm{P_{Q,0}}$')
    ax1.plot(T_mu0, contrib_g_mu0, '-', c = 'red', label = r'$\mathrm{P_{g,0}}$')
    ax1.plot(T_mu0, contrib_pert_mu0, '-', c = 'pink', label = r'$\mathrm{P_{pert,0}}$')
    ax1.plot(T_mu0, contrib_qgp_mu0, '-', c = 'black', label = r'$\mathrm{P_{QGP,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_step_mu0, '-', c = 'blue', label = r'$\mathrm{P_{cluster,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_singlet_step_mu0, '-', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_color_step_mu0, '-', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_c_mu0, '-.', c = 'blue', label = r'$\mathrm{P_{cluster,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_singlet_c_mu0, '-.', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,0}}$')
    #ax1.plot(T_mu0, contrib_cluster_color_c_mu0, '-.', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,0}}$')
    #ax1.legend(loc = 2)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig3 = matplotlib.pyplot.figure(num = 3, figsize = (5.9, 5))
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.axis([0., 2000., 0., 5.])
    #ax3.axis([100., 200., 0., 2.])
    ax3.add_patch(matplotlib.patches.Polygon(bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Bazavov et al. (2014), $\mathrm{\mu=0}$'))
    ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2014), $\mathrm{\mu=0}$'))
    ax3.add_patch(matplotlib.patches.Polygon(bazavov_1710_05024_mu0, closed = True, fill = True, color = 'magenta', alpha = 0.3, label = r'Bazavov et al. (2018), $\mathrm{\mu=0}$'))
    ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=0}$'))
    #ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=200}$ MeV'))
    #ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu300, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=300}$ MeV'))
    #ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu400, closed = True, fill = True, color = 'cyan', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=400}$ MeV'))
    ax3.plot(T_mu0, [sum(el) for el in zip(contrib_qgp_mu0, contrib_cluster_step_mu0)], '-.', c = 'blue', label = r'step--up step--down $\mathrm{\mu=0}$')
    ax3.plot(T_mu0, [sum(el) for el in zip(contrib_qgp_mu0, contrib_cluster_c_mu0)], '-', c = 'blue', label = r'extended continuum $\mathrm{\mu=0}$')
    #ax3.plot(T_mu200, [sum(el) for el in zip(contrib_qgp_mu200, contrib_cluster_step_mu200)], '-.', c = 'red', label = r'step--up step--down $\mathrm{\mu=200}$')
    #ax3.plot(T_mu200, [sum(el) for el in zip(contrib_qgp_mu200, contrib_cluster_c_mu200)], '-', c = 'red', label = r'extended continuum $\mathrm{\mu=200}$')
    #ax3.plot(T_mu300, [sum(el) for el in zip(contrib_qgp_mu300, contrib_cluster_step_mu300)], '-.', c = 'green', label = r'step--up step--down $\mathrm{\mu=300}$')
    #ax3.plot(T_mu300, [sum(el) for el in zip(contrib_qgp_mu300, contrib_cluster_c_mu300)], '-', c = 'green', label = r'extended continuum $\mathrm{\mu=300}$')
    #ax3.plot(T_mu400, [sum(el) for el in zip(contrib_qgp_mu400, contrib_cluster_step_mu400)], '-.', c = 'cyan', label = r'step--up step--down $\mathrm{\mu=400}$')
    #ax3.plot(T_mu400, [sum(el) for el in zip(contrib_qgp_mu400, contrib_cluster_c_mu400)], '-', c = 'cyan', label = r'extended continuum $\mathrm{\mu=400}$')
    #ax3.plot(T_mu500, [sum(el) for el in zip(contrib_qgp_mu500, contrib_cluster_step_mu500)], '-.', c = 'magenta', label = r'step--up step--down $\mathrm{\mu=500}$')
    #ax3.plot(T_mu500, [sum(el) for el in zip(contrib_qgp_mu500, contrib_cluster_c_mu500)], '-', c = 'magenta', label = r'extended continuum $\mathrm{\mu=500}$')
    #ax3.legend()
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig2 = matplotlib.pyplot.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.axis([0., 450., 0., 1.2])
    ax2.plot(T_mu0, [pnjl.thermo.gcp_sea_lattice.M(el, 0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T_mu0], '-', c = 'green')
    ax2.plot(T_mu200, [pnjl.thermo.gcp_sea_lattice.M(el, 200.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T_mu200], '--', c = 'green')
    ax2.plot(T_mu300, [pnjl.thermo.gcp_sea_lattice.M(el, 300.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T_mu300], '-.', c = 'green')
    ax2.plot(T_mu400, [pnjl.thermo.gcp_sea_lattice.M(el, 400.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T_mu400], ':', c = 'green')
    ax2.scatter(T_mu500, [pnjl.thermo.gcp_sea_lattice.M(el, 500.0 / 3.0) / pnjl.thermo.gcp_sea_lattice.M(0, 0) for el in T_mu500], marker = 'o', c = 'green')
    ax2.plot(T_mu0, phi_re_mu0, '-', c = 'blue')
    ax2.plot(T_mu200, phi_re_mu200, '--', c = 'blue')
    ax2.plot(T_mu300, phi_re_mu300, '-.', c = 'blue')
    ax2.plot(T_mu400, phi_re_mu400, ':', c = 'blue')
    ax2.scatter(T_mu500, phi_re_mu500, marker = 'o', c = 'blue')
    ax2.plot(T_mu0, phi_im_mu0, '-', c = 'red')
    ax2.plot(T_mu200, phi_im_mu200, '--', c = 'red')
    ax2.plot(T_mu300, phi_im_mu300, '-.', c = 'red')
    ax2.plot(T_mu400, phi_im_mu400, ':', c = 'red')
    ax2.scatter(T_mu500, phi_im_mu500, marker = 'o', c = 'red')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{\Phi}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    fig2.tight_layout(pad = 0.1)
    fig3.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def arccos_cos_phase_factor():

    def phase(M, Mi, Mthi, Ni, Lambda_i):

        hz = 0.5
        nlambda = Ni * Lambda_i

        frac1 = M / nlambda
        frac2 = Mthi / nlambda

        heavi2 = numpy.heaviside((M ** 2) - (Mi ** 2), hz)
        heavi3 = numpy.heaviside((M ** 2) - (Mthi ** 2), hz)
        heavi4 = numpy.heaviside(Mthi + nlambda - M, hz)
        heavi5 = numpy.heaviside(Mthi + nlambda - Mi, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        first = 0.0
        second = 0.0
        third = 0.0

        if (Mthi ** 2) >= (Mi **2):
            #arccos_el = math.acos(arccos_in) if heavi3 > 0 and heavi4 > 0 else 0.0
            if M >= Mi and M <= Mthi:
                first = heavi2 - heavi3
            if M >= Mthi and M <= Mthi + nlambda:
                arccos_el = math.acos(arccos_in)
                second = heavi3 * heavi4 * arccos_el / math.pi
        else:
            if M >= Mthi and M <= Mthi + nlambda:
                #arccos_el2 = math.acos(arccos_in) if heavi5 > 0 and heavi4 > 0 and heavi7 > 0 else 0.0
                arccos_el2 = math.acos(arccos_in)
                third = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((Mthi + nlambda - Mi) / nlambda)

        return first + second + third

    Mi_val = pnjl.defaults.default_MN
    Ni_val = 3.0
    Lam_val = pnjl.defaults.default_L

    Mthi_val1 = 1.4 * Mi_val
    Mthi_val2 = 1.2 * Mi_val
    Mthi_val3 = 1.0 * Mi_val
    Mthi_val4 = 0.8 * Mi_val
    Mthi_val5 = 0.45 * Mi_val

    M_vec = numpy.linspace(0.0, 1.2 * (Mthi_val1 + Ni_val * Lam_val), 2000)
    phase_vec1 = [
        phase(el, Mi_val, Mthi_val1, Ni_val, Lam_val) 
        for el in M_vec]
    phase_vec2 = [
        phase(el, Mi_val, Mthi_val2, Ni_val, Lam_val) 
        for el in M_vec]
    phase_vec3 = [
        phase(el, Mi_val, Mthi_val3, Ni_val, Lam_val) 
        for el in M_vec]
    phase_vec4 = [
        phase(el, Mi_val, Mthi_val4, Ni_val, Lam_val) 
        for el in M_vec]
    phase_vec5 = [
        phase(el, Mi_val, Mthi_val5, Ni_val, Lam_val) 
        for el in M_vec]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(M_vec, phase_vec1, '-', c = 'blue', label = r'$M_{th,i}\gg M_i$')
    ax1.plot(M_vec, phase_vec2, '-', c = 'red', label = r'$M_{th,i}>M_i$')
    ax1.plot(M_vec, phase_vec3, '-', c = 'green', label = r'$M_{th,i}=M_i$')
    ax1.plot(M_vec, phase_vec4, '-', c = 'magenta', label = r'$M_{th,i}<M_i$')
    ax1.plot(M_vec, phase_vec5, '-', c = 'black', label = r'$M_{th,i}\ll M_i$')
    ax1.legend()
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'M [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\delta_i$', fontsize = 16)
    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

if __name__ == '__main__':

    PNJL_thermodynamics_continuum_test()

    #print("END")