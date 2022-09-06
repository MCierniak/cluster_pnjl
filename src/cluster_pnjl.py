"""import math
import matplotlib

import matplotlib.patches
import matplotlib.pyplot
import scipy.integrate
import scipy.optimize
import numpy
import time
import glob
import math
import tqdm
import csv
import os

from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pnjl.thermo.gcp_cluster.bound_step_continuum_arccos_cos as cluster
import pnjl.thermo.gcp_cluster.bound_step_continuum_step
import pnjl.thermo.gcp_pl_polynomial
import pnjl.thermo.gcp_perturbative
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_quark
import pnjl.defaults
import utils

import papers.epja_2022

import warnings
warnings.filterwarnings("ignore")


def cluster_thermo(T, mu, phi_re, phi_im, M, Mth, dMdmu, dMthdmu, dMdT, dMthdT, a, b, dx):
    #temperature, baryochemical potential, traced PL, traced PL c.c., bound
    #state mass, threshold mass, nr of valence quarks - nr of valence,
    #antiquarks, nr of valence s quarks - nr of valence anti, s quarks,
    #degeneracy factor
    Pres = [
        pnjl.thermo.gcp_cluster.bound_step_continuum_step.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), M_el, Mth_el, a, b, dx)
        for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im, M, Mth), desc = "Pres", total = len(T), ascii = True)]
    BDen = [0.0 for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el, dM_el, dMth_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im, M, Mth, dMdmu, dMthdmu), desc = "BDen", total = len(T), ascii = True)]
    SDen = [0.0 for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el, dM_el, dMth_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im, M, Mth, dMdT, dMthdT), desc = "SDen", total = len(T), ascii = True)]
    return Pres, BDen, SDen

def calc_PL(T: float, mu: float, phi_re0: float, phi_im0: float,
    light_kwargs=None, strange_kwargs=None, gluon_kwargs=None, perturbative_kwargs=None,
    with_clusters: bool=True) -> (float, float):
    
    if light_kwargs is None:
        light_kwargs = {}
    if strange_kwargs is None:
        strange_kwargs = {'Nf' : 1.0, 'ml' : pnjl.defaults.default_ms}
    if gluon_kwargs is None:
        gluon_kwargs = {}
    if perturbative_kwargs is None:
        perturbative_kwargs = {'Nf' : 3.0}

    sd = 1234
    attempt = 1
    max_attempt = 100
    bnds = ((0.0, 3.0), (-3.0, 3.0),)

    def thermodynamic_potential_with_clusters(x, 
        _T, _mu, 
        _diquark_bmass, _diquark_thmass, _d_diquark, 
        _fquark_bmass, _fquark_thmass, _d_fquark, 
        _qquark_bmass, _qquark_thmass, _d_qquark, 
        _s_kwargs, _q_kwargs, _pert_kwargs, _g_kwargs):
        sea = pnjl.thermo.gcp_sea_lattice.gcp_real(_T, _mu)
        sq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_s_kwargs)
        lq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_q_kwargs)
        per = pnjl.thermo.gcp_perturbative.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_pert_kwargs)
        glue = pnjl.thermo.gcp_pl_polynomial.gcp_real(_T, complex(x[0], x[1]), complex(x[0], -x[1]), **_g_kwargs)
        diquark = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _diquark_bmass, _diquark_thmass, 2, 0, _d_diquark)
        fquark = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _fquark_bmass, _fquark_thmass, 4, 0, _d_fquark)
        qquark = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _qquark_bmass, _qquark_thmass, 5, 0, _d_qquark)

        return sea + sq + lq + per + glue + diquark + fquark + qquark

    def thermodynamic_potential(x, _T, _mu, _s_kwargs, _q_kwargs, _pert_kwargs, _g_kwargs):
        sea = pnjl.thermo.gcp_sea_lattice.gcp_real(_T, _mu)
        sq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_s_kwargs)
        lq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_q_kwargs)
        per = pnjl.thermo.gcp_perturbative.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_pert_kwargs)
        glue = pnjl.thermo.gcp_pl_polynomial.gcp_real(_T, complex(x[0], x[1]), complex(x[0], -x[1]), **_g_kwargs)

        return sea + sq + lq + per + glue

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

        omega_result = scipy.optimize.dual_annealing(thermodynamic_potential_with_clusters,
                bounds = bnds,
                args = (T, mu, 
                        diquark_bmass, diquark_thmass, d_diquark, 
                        fquark_bmass, fquark_thmass, d_fquark,
                        qquark_bmass, qquark_thmass, d_qquark,
                        strange_kwargs, light_kwargs, perturbative_kwargs, gluon_kwargs),
                x0 = [phi_re0, phi_im0],
                maxiter = 20,
                seed = sd)
    else:
        omega_result = scipy.optimize.dual_annealing(thermodynamic_potential,
                bounds = bnds,
                args = (T, mu, strange_kwargs, light_kwargs, perturbative_kwargs, gluon_kwargs),
                x0 = [phi_re0, phi_im0],
                maxiter = 20,
                seed = sd)

    return (omega_result.x[0], omega_result.x[1])

def clusters(T, mu, phi_re, phi_im):
    print("Calculating nucleon thermo..")
    M_N = [pnjl.defaults.default_MN                                           for T_el, mu_el in zip(T, mu)]
    dM_N_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_N_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_N = [3. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_N_dmu = [3. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_N_dT = [3. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #(N(Dq): spin * isospin * color)
    dN = (2.0 * 2.0 * 1.0)
    Pres_N, BDen_N, SDen_N = cluster_thermo(T, mu, phi_re, phi_im, M_N, Mth_N, dM_N_dmu, dMth_N_dmu, dM_N_dT, dMth_N_dT, 3, 0, dN)

    print("Calculating pentaquark thermo..")
    M_P = [pnjl.defaults.default_MP                                            for T_el, mu_el in zip(T, mu)]
    dM_P_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_P_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_P = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_P_dmu = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_P_dT = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #P(NM) + P(NM)
    dP = (4.0 * 2.0 * 1.0) + (2.0 * 4.0 * 1.0)
    Pres_P, BDen_P, SDen_P = cluster_thermo(T, mu, phi_re, phi_im, M_P, Mth_P, dM_P_dmu, dMth_P_dmu, dM_P_dT, dMth_P_dT, 3, 0, dP)

    print("Calculating hexaquark thermo..")
    M_H = [pnjl.defaults.default_MH                                           for T_el, mu_el in zip(T, mu)]
    dM_H_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_H_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_H = [6. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_H_dmu = [6. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_H_dT = [6. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #H(Qq) / H(FD) / H(NN) + H(Qq) / H(NN)
    dH = (1.0 * 3.0 * 1.0) + (3.0 * 1.0 * 1.0)
    Pres_H, BDen_H, SDen_H = cluster_thermo(T, mu, phi_re, phi_im, M_H, Mth_H, dM_H_dmu, dMth_H_dmu, dM_H_dT, dMth_H_dT, 6, 0, dH)

    print("Calculating pi meson thermo..")
    M_pi = [pnjl.defaults.default_Mpi                                          for T_el, mu_el in zip(T, mu)]
    dM_pi_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_pi_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_pi = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_pi_dmu = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_pi_dT = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #pi(q aq)
    dpi = (1.0 * 3.0 * 1.0)
    Pres_pi, BDen_pi, SDen_pi = cluster_thermo(T, mu, phi_re, phi_im, M_pi, Mth_pi, dM_pi_dmu, dMth_pi_dmu, dM_pi_dT, dMth_pi_dT, 0, 0, dpi)

    print("Calculating K meson thermo..")
    M_K = [pnjl.defaults.default_MK                                           for T_el, mu_el in zip(T, mu)]
    dM_K_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_K_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_K = [math.sqrt(2) * (pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el) + pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el, ml = pnjl.defaults.default_ms))
                                                                                      for T_el, mu_el in zip(T, mu)]
    dMth_K_dmu = [math.sqrt(2) * (pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) + pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el, ml = pnjl.defaults.default_ms))
                                                                                      for T_el, mu_el in zip(T, mu)]
    dMth_K_dT = [math.sqrt(2) * (pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el) + pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el, ml = pnjl.defaults.default_ms))
                                                                                      for T_el, mu_el in zip(T, mu)]
    #K(q aq)
    dK = (1.0 * 4.0 * 1.0)
    Pres_K, BDen_K, SDen_K = cluster_thermo(T, mu, phi_re, phi_im, M_K, Mth_K, dM_K_dmu, dMth_K_dmu, dM_K_dT, dMth_K_dT, 0, 0, dK)

    print("Calculating rho meson thermo..")
    M_rho = [pnjl.defaults.default_MM                                           for T_el, mu_el in zip(T, mu)]
    dM_rho_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_rho_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_rho = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_rho_dmu = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_rho_dT = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #rho(q aq)
    drho = (3.0 * 3.0 * 1.0)
    Pres_rho, BDen_rho, SDen_rho = cluster_thermo(T, mu, phi_re, phi_im, M_rho, Mth_rho, dM_rho_dmu, dMth_rho_dmu, dM_rho_dT, dMth_rho_dT, 0, 0, drho)

    print("Calculating omega meson thermo..")
    M_omega = [pnjl.defaults.default_MM                                           for T_el, mu_el in zip(T, mu)]
    dM_omega_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_omega_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_omega = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_omega_dmu = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_omega_dT = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #omega(q aq)
    domega = (3.0 * 1.0 * 1.0) #should this have the 1/2 factor?
    Pres_omega, BDen_omega, SDen_omega = cluster_thermo(T, mu, phi_re, phi_im, M_omega, Mth_omega, dM_omega_dmu, dMth_omega_dmu, dM_omega_dT, dMth_omega_dT, 0, 0, domega)

    print("Calculating tetraquark thermo..")
    M_T = [pnjl.defaults.default_MT                                           for T_el, mu_el in zip(T, mu)]
    dM_T_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_T_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_T = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_T_dmu = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_T_dT = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #T(MM) + T(MM) + T(MM)
    dT = (1.0 * 5.0 * 1.0) + (5.0 * 1.0 * 1.0) + (3.0 * 3.0 * 1.0)
    Pres_T, BDen_T, SDen_T = cluster_thermo(T, mu, phi_re, phi_im, M_T, Mth_T, dM_T_dmu, dMth_T_dmu, dM_T_dT, dMth_T_dT, 0, 0, dT)

    print("Calculating diquark thermo..")
    M_D = [pnjl.defaults.default_MD                                           for T_el, mu_el in zip(T, mu)]
    dM_D_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_D_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_D = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_D_dmu = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_D_dT = [2. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #D(qq)
    dD = (1.0 * 1.0 * 3.0)
    Pres_D, BDen_D, SDen_D = cluster_thermo(T, mu, phi_re, [-el for el in phi_im], M_D, Mth_D, dM_D_dmu, dMth_D_dmu, dM_D_dT, dMth_D_dT, 2, 0, dD)

    print("Calculating 4-quark thermo..")
    M_F = [pnjl.defaults.default_MF                                           for T_el, mu_el in zip(T, mu)]
    dM_F_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_F_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_F = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_F_dmu = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_F_dT = [4. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #F(Nq)
    dF = (1.0 * 1.0 * 3.0)
    Pres_F, BDen_F, SDen_F = cluster_thermo(T, mu, phi_re, phi_im, M_F, Mth_F, dM_F_dmu, dMth_F_dmu, dM_F_dT, dMth_F_dT, 4, 0, dF)

    print("Calculating 5-quark thermo..")
    M_Q5 = [pnjl.defaults.default_MQ                                           for T_el, mu_el in zip(T, mu)]
    dM_Q5_dmu = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    dM_Q5_dT = [0.                                                                 for T_el, mu_el in zip(T, mu)]
    Mth_Q5 = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T_el, mu_el)     for T_el, mu_el in zip(T, mu)]
    dMth_Q5_dmu = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdmu(T_el, mu_el) for T_el, mu_el in zip(T, mu)]
    dMth_Q5_dT = [5. * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.dMdT(T_el, mu_el)  for T_el, mu_el in zip(T, mu)]
    #Q5(F(Nq)q) / Q5(F(DD)q) / Q5(ND)
    dQ5 = (2.0 * 2.0 * 3.0)
    Pres_Q5, BDen_Q5, SDen_Q5 = cluster_thermo(T, mu, phi_re, [-el for el in phi_im], M_Q5, Mth_Q5, dM_Q5_dmu, dMth_Q5_dmu, dM_Q5_dT, dMth_Q5_dT, 5, 0, dQ5)

    return ((Pres_pi, Pres_rho, Pres_omega, Pres_K, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, Pres_Q5, Pres_H),
        (BDen_pi, BDen_rho, BDen_omega, BDen_K, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, BDen_Q5, BDen_H),
        (SDen_pi, SDen_rho, SDen_omega, SDen_K, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, SDen_Q5, SDen_H))

def clusters_c(name, T, mu, phi_re, phi_im,
               regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2, 
               regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2,
               regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52):

    print("Calculating nucleon thermo..")
    Pres_N, BDen_N, SDen_N = [], [], []
    if os.path.exists(name + "_N.dat"):
        Pres_N, BDen_N = utils.data_collect(0, 1, name + "_N.dat")
        SDen_N, _ = utils.data_collect(2, 2, name + "_N.dat")
    else:
        M_N = pnjl.defaults.default_MN
        #(N(Dq): spin * isospin * color)
        dN = (2.0 * 2.0 * 1.0) / 2.0
        NN = 3.0
        LN = pnjl.defaults.default_L
        Pres_N, BDen_N, SDen_N = cluster_c_thermo(T, mu, phi_re, phi_im, M_N, 3, 0, dN, NN, LN, 0, regulator_N, regulator_N2)
        with open(name + "_N.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_N, BDen_N, SDen_N)])

    print("Calculating pentaquark thermo..")
    Pres_P, BDen_P, SDen_P = [], [], []
    if os.path.exists(name + "_P.dat"):
        Pres_P, BDen_P = utils.data_collect(0, 1, name + "_P.dat")
        SDen_P, _ = utils.data_collect(2, 2, name + "_P.dat")
    else:
        M_P = pnjl.defaults.default_MP
        #P(NM) + P(NM)
        dP = ((4.0 * 2.0 * 1.0) + (2.0 * 4.0 * 1.0)) / 2.0
        NP = 5.0
        LP = pnjl.defaults.default_L
        Pres_P, BDen_P, SDen_P = cluster_c_thermo(T, mu, phi_re, phi_im, M_P, 3, 0, dP, NP, LP, 0, regulator_P, regulator_P2)
        with open(name + "_P.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_P, BDen_P, SDen_P)])

    print("Calculating hexaquark thermo..")
    Pres_H, BDen_H, SDen_H = [], [], []
    if os.path.exists(name + "_H.dat"):
        Pres_H, BDen_H = utils.data_collect(0, 1, name + "_H.dat")
        SDen_H, _ = utils.data_collect(2, 2, name + "_H.dat")
    else:
        M_H = pnjl.defaults.default_MH
        #H(Qq) / H(FD) / H(NN) + H(Qq) / H(NN)
        dH = ((1.0 * 3.0 * 1.0) + (3.0 * 1.0 * 1.0)) / 2.0
        NH = 6.0
        LH = pnjl.defaults.default_L
        Pres_H, BDen_H, SDen_H = cluster_c_thermo(T, mu, phi_re, phi_im, M_H, 6, 0, dH, NH, LH, 0, regulator_H, regulator_H2)
        with open(name + "_H.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_H, BDen_H, SDen_H)])

    print("Calculating pi meson thermo..")
    Pres_pi, BDen_pi, SDen_pi = [], [], []
    if os.path.exists(name + "_pi.dat"):
        Pres_pi, BDen_pi = utils.data_collect(0, 1, name + "_pi.dat")
        SDen_pi, _ = utils.data_collect(2, 2, name + "_pi.dat")
    else:
        M_pi = pnjl.defaults.default_Mpi
        #pi(q aq)
        #the 3.0 isospin factor spans pi^+/pi^-/pi^0...  all other particles
        #should have a factor of 1/2 to account for antiparticles
        dpi = (1.0 * 3.0 * 1.0)
        Npi = 2.0
        Lpi = pnjl.defaults.default_L
        Pres_pi, BDen_pi, SDen_pi = cluster_c_thermo(T, mu, phi_re, phi_im, M_pi, 0, 0, dpi, Npi, Lpi, 0, regulator_pi, regulator_pi2)
        with open(name + "_pi.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_pi, BDen_pi, SDen_pi)])

    print("Calculating K meson thermo..")
    Pres_K, BDen_K, SDen_K = [], [], []
    if os.path.exists(name + "_K.dat"):
        Pres_K, BDen_K = utils.data_collect(0, 1, name + "_K.dat")
        SDen_K, _ = utils.data_collect(2, 2, name + "_K.dat")
    else:
        M_K = pnjl.defaults.default_MK
        #K(q aq)
        #the 6.0 isospin factor spans K^+/K^-/K^0/Kbar^0/K^0_S/K^0_L...  all
        #other particles should have a factor of 1/2 to account for
        #antiparticles
        dK = (1.0 * 6.0 * 1.0)
        NK = 2.0
        LK = pnjl.defaults.default_L
        Pres_K, BDen_K, SDen_K = cluster_c_thermo(T, mu, phi_re, phi_im, M_K, 1, -1, dK, NK, LK, 1, regulator_K, regulator_K2)
        with open(name + "_K.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_K, BDen_K, SDen_K)])

    print("Calculating rho meson thermo..")
    Pres_rho, BDen_rho, SDen_rho = [], [], []
    if os.path.exists(name + "_rho.dat"):
        Pres_rho, BDen_rho = utils.data_collect(0, 1, name + "_rho.dat")
        SDen_rho, _ = utils.data_collect(2, 2, name + "_rho.dat")
    else:
        M_rho = pnjl.defaults.default_MM
        #rho(q aq)
        #the 3.0 isospin factor spans rho^+/rho^-/rho^0...  all other particles
        #should have a factor of 1/2 to account for antiparticles
        drho = (3.0 * 3.0 * 1.0)
        Nrho = 2.0
        Lrho = pnjl.defaults.default_L
        Pres_rho, BDen_rho, SDen_rho = cluster_c_thermo(T, mu, phi_re, phi_im, M_rho, 0, 0, drho, Nrho, Lrho, 0, regulator_rho, regulator_rho2)
        with open(name + "_rho.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_rho, BDen_rho, SDen_rho)])

    print("Calculating omega meson thermo..")
    Pres_omega, BDen_omega, SDen_omega = [], [], []
    if os.path.exists(name + "_omega.dat"):
        Pres_omega, BDen_omega = utils.data_collect(0, 1, name + "_omega.dat")
        SDen_omega, _ = utils.data_collect(2, 2, name + "_omega.dat")
    else:
        M_omega = pnjl.defaults.default_MM
        #omega(q aq)
        #the 1.0 isospin factor spans the omega and its self-antiparticle...
        #all other particles should have a factor of 1/2 to account for
        #antiparticles
        domega = (3.0 * 1.0 * 1.0)
        Nomega = 2.0
        Lomega = pnjl.defaults.default_L
        Pres_omega, BDen_omega, SDen_omega = cluster_c_thermo(T, mu, phi_re, phi_im, M_omega, 0, 0, domega, Nomega, Lomega, 0, regulator_omega, regulator_omega2)
        with open(name + "_omega.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_omega, BDen_omega, SDen_omega)])

    print("Calculating tetraquark thermo..")
    Pres_T, BDen_T, SDen_T = [], [], []
    if os.path.exists(name + "_T.dat"):
        Pres_T, BDen_T = utils.data_collect(0, 1, name + "_T.dat")
        SDen_T, _ = utils.data_collect(2, 2, name + "_T.dat")
    else:
        M_T = pnjl.defaults.default_MT
        #T(MM) + T(MM) + T(MM)
        dT = ((1.0 * 5.0 * 1.0) + (5.0 * 1.0 * 1.0) + (3.0 * 3.0 * 1.0)) / 2.0
        NT = 4.0
        LT = pnjl.defaults.default_L
        Pres_T, BDen_T, SDen_T = cluster_c_thermo(T, mu, phi_re, phi_im, M_T, 0, 0, dT, NT, LT, 0, regulator_T, regulator_T2)
        with open(name + "_T.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_T, BDen_T, SDen_T)])

    print("Calculating diquark thermo..")
    Pres_D, BDen_D, SDen_D = [], [], []
    if os.path.exists(name + "_D.dat"):
        Pres_D, BDen_D = utils.data_collect(0, 1, name + "_D.dat")
        SDen_D, _ = utils.data_collect(2, 2, name + "_D.dat")
    else:
        M_D = pnjl.defaults.default_MD
        #D(qq)
        dD = (1.0 * 1.0 * 3.0) / 2.0
        ND = 2.0
        LD = pnjl.defaults.default_L
        Pres_D, BDen_D, SDen_D = cluster_c_thermo(T, mu, phi_re, [-el for el in phi_im], M_D, 2, 0, dD, ND, LD, 0, regulator_D, regulator_D2)
        with open(name + "_D.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_D, BDen_D, SDen_D)])

    print("Calculating 4-quark thermo..")
    Pres_F, BDen_F, SDen_F = [], [], []
    if os.path.exists(name + "_F.dat"):
        Pres_F, BDen_F = utils.data_collect(0, 1, name + "_F.dat")
        SDen_F, _ = utils.data_collect(2, 2, name + "_F.dat")
    else:
        M_F = pnjl.defaults.default_MF
        #F(Nq)
        dF = (1.0 * 1.0 * 3.0) / 2.0
        NF = 4.0
        LF = pnjl.defaults.default_L
        Pres_F, BDen_F, SDen_F = cluster_c_thermo(T, mu, phi_re, phi_im, M_F, 4, 0, dF, NF, LF, 0, regulator_F, regulator_F2)
        with open(name + "_F.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_F, BDen_F, SDen_F)])

    print("Calculating 5-quark thermo..")
    Pres_Q5, BDen_Q5, SDen_Q5 = [], [], []
    if os.path.exists(name + "_Q5.dat"):
        Pres_Q5, BDen_Q5 = utils.data_collect(0, 1, name + "_Q5.dat")
        SDen_Q5, _ = utils.data_collect(2, 2, name + "_Q5.dat")
    else:
        M_Q5 = pnjl.defaults.default_MQ
        #Q5(F(Nq)q) / Q5(F(DD)q) / Q5(ND)
        dQ5 = (2.0 * 1.0 * 3.0) / 2.0
        NQ5 = 5.0
        LQ5 = pnjl.defaults.default_L
        Pres_Q5, BDen_Q5, SDen_Q5 = cluster_c_thermo(T, mu, phi_re, [-el for el in phi_im], M_Q5, 5, 0, dQ5, NQ5, LQ5, 0, regulator_Q5, regulator_Q52)
        with open(name + "_Q5.dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[p_el, b_el, s_el] for p_el, b_el, s_el in zip(Pres_Q5, BDen_Q5, SDen_Q5)])

    return ((Pres_pi, Pres_rho, Pres_omega, Pres_K, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, Pres_Q5, Pres_H),
        (BDen_pi, BDen_rho, BDen_omega, BDen_K, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, BDen_Q5, BDen_H),
        (SDen_pi, SDen_rho, SDen_omega, SDen_K, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, SDen_Q5, SDen_H))

def cumulants_c(name, rank, T, mu, phi_re, phi_im,
               regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2, 
               regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2,
               regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52):

    print("Calculating nucleon thermo..")
    Chi_N = []
    if os.path.exists(name + "_N" + str(rank) + ".dat"):
        Chi_N, _ = utils.data_collect(0, 0, name + "_N" + str(rank) + ".dat")
    else:
        M_N = pnjl.defaults.default_MN
        #(N(Dq): spin * isospin * color)
        dN = (2.0 * 2.0 * 1.0) / 2.0
        NN = 3.0
        LN = pnjl.defaults.default_L
        Chi_N = cumulants_c_thermo(rank, T, mu, phi_re, phi_im, M_N, 3, 0, dN, NN, LN, 0, regulator_N, regulator_N2)
        with open(name + "_N" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_N])

    print("Calculating pentaquark thermo..")
    Chi_P = []
    if os.path.exists(name + "_P" + str(rank) + ".dat"):
        Chi_P, _ = utils.data_collect(0, 0, name + "_P" + str(rank) + ".dat")
    else:
        M_P = pnjl.defaults.default_MP
        #P(NM) + P(NM)
        dP = ((4.0 * 2.0 * 1.0) + (2.0 * 4.0 * 1.0)) / 2.0
        NP = 5.0
        LP = pnjl.defaults.default_L
        Chi_P = cumulants_c_thermo(rank, T, mu, phi_re, phi_im, M_P, 3, 0, dP, NP, LP, 0, regulator_P, regulator_P2)
        with open(name + "_P" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_P])

    print("Calculating hexaquark thermo..")
    Chi_H = []
    if os.path.exists(name + "_H" + str(rank) + ".dat"):
        Chi_H, _ = utils.data_collect(0, 0, name + "_H" + str(rank) + ".dat")
    else:
        M_H = pnjl.defaults.default_MH
        #H(Qq) / H(FD) / H(NN) + H(Qq) / H(NN)
        dH = ((1.0 * 3.0 * 1.0) + (3.0 * 1.0 * 1.0)) / 2.0
        NH = 6.0
        LH = pnjl.defaults.default_L
        Chi_H = cumulants_c_thermo(rank, T, mu, phi_re, phi_im, M_H, 6, 0, dH, NH, LH, 0, regulator_H, regulator_H2)
        with open(name + "_H" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_H])

    print("Calculating pi meson thermo..")
    Chi_pi = []
    if os.path.exists(name + "_pi" + str(rank) + ".dat"):
        Chi_pi, _ = utils.data_collect(0, 0, name + "_pi" + str(rank) + ".dat")
    else:
        M_pi = pnjl.defaults.default_Mpi
        #pi(q aq)
        #the 3.0 isospin factor spans pi^+/pi^-/pi^0...  all other particles
        #should have a factor of 1/2 to account for antiparticles
        dpi = (1.0 * 3.0 * 1.0)
        Npi = 2.0
        Lpi = pnjl.defaults.default_L
        Chi_pi = cumulants_c_thermo(rank, T, mu, phi_re, phi_im, M_pi, 0, 0, dpi, Npi, Lpi, 0, regulator_pi, regulator_pi2)
        with open(name + "_pi" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_pi])

    print("Calculating K meson thermo..")
    Chi_K = []
    if os.path.exists(name + "_K" + str(rank) + ".dat"):
        Chi_K, _ = utils.data_collect(0, 0, name + "_K" + str(rank) + ".dat")
    else:
        M_K = pnjl.defaults.default_MK
        #K(q aq)
        #the 6.0 isospin factor spans K^+/K^-/K^0/Kbar^0/K^0_S/K^0_L...  all
        #other particles should have a factor of 1/2 to account for
        #antiparticles
        dK = (1.0 * 6.0 * 1.0)
        NK = 2.0
        LK = pnjl.defaults.default_L
        Chi_K = cumulants_c_thermo(rank, T, mu, phi_re, phi_im, M_K, 1, -1, dK, NK, LK, 1, regulator_K, regulator_K2)
        with open(name + "_K" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_K])

    print("Calculating rho meson thermo..")
    Chi_rho = []
    if os.path.exists(name + "_rho" + str(rank) + ".dat"):
        Chi_rho, _ = utils.data_collect(0, 0, name + "_rho" + str(rank) + ".dat")
    else:
        M_rho = pnjl.defaults.default_MM
        #rho(q aq)
        #the 3.0 isospin factor spans rho^+/rho^-/rho^0...  all other particles
        #should have a factor of 1/2 to account for antiparticles
        drho = (3.0 * 3.0 * 1.0)
        Nrho = 2.0
        Lrho = pnjl.defaults.default_L
        Chi_rho = cumulants_c_thermo(rank, T, mu, phi_re, phi_im, M_rho, 0, 0, drho, Nrho, Lrho, 0, regulator_rho, regulator_rho2)
        with open(name + "_rho" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_rho])

    print("Calculating omega meson thermo..")
    Chi_omega = []
    if os.path.exists(name + "_omega" + str(rank) + ".dat"):
        Chi_omega, _ = utils.data_collect(0, 0, name + "_omega" + str(rank) + ".dat")
    else:
        M_omega = pnjl.defaults.default_MM
        #omega(q aq)
        #the 1.0 isospin factor spans the omega and its self-antiparticle...
        #all other particles should have a factor of 1/2 to account for
        #antiparticles
        domega = (3.0 * 1.0 * 1.0)
        Nomega = 2.0
        Lomega = pnjl.defaults.default_L
        Chi_omega = cumulants_c_thermo(rank, T, mu, phi_re, phi_im, M_omega, 0, 0, domega, Nomega, Lomega, 0, regulator_omega, regulator_omega2)
        with open(name + "_omega" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_omega])

    print("Calculating tetraquark thermo..")
    Chi_T = []
    if os.path.exists(name + "_T" + str(rank) + ".dat"):
        Chi_T, _ = utils.data_collect(0, 0, name + "_T" + str(rank) + ".dat")
    else:
        M_T = pnjl.defaults.default_MT
        #T(MM) + T(MM) + T(MM)
        dT = ((1.0 * 5.0 * 1.0) + (5.0 * 1.0 * 1.0) + (3.0 * 3.0 * 1.0)) / 2.0
        NT = 4.0
        LT = pnjl.defaults.default_L
        Chi_T = cumulants_c_thermo(rank, T, mu, phi_re, phi_im, M_T, 0, 0, dT, NT, LT, 0, regulator_T, regulator_T2)
        with open(name + "_T" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_T])

    print("Calculating diquark thermo..")
    Chi_D = []
    if os.path.exists(name + "_D" + str(rank) + ".dat"):
        Chi_D, _ = utils.data_collect(0, 0, name + "_D" + str(rank) + ".dat")
    else:
        M_D = pnjl.defaults.default_MD
        #D(qq)
        dD = (1.0 * 1.0 * 3.0) / 2.0
        ND = 2.0
        LD = pnjl.defaults.default_L
        Chi_D = cumulants_c_thermo(rank, T, mu, phi_re, [-el for el in phi_im], M_D, 2, 0, dD, ND, LD, 0, regulator_D, regulator_D2)
        with open(name + "_D" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_D])

    print("Calculating 4-quark thermo..")
    Chi_F = []
    if os.path.exists(name + "_F" + str(rank) + ".dat"):
        Chi_F, _ = utils.data_collect(0, 0, name + "_F" + str(rank) + ".dat")
    else:
        M_F = pnjl.defaults.default_MF
        #F(Nq)
        dF = (1.0 * 1.0 * 3.0) / 2.0
        NF = 4.0
        LF = pnjl.defaults.default_L
        Chi_F = cumulants_c_thermo(rank, T, mu, phi_re, phi_im, M_F, 4, 0, dF, NF, LF, 0, regulator_F, regulator_F2)
        with open(name + "_F" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_F])

    print("Calculating 5-quark thermo..")
    Chi_Q5 = []
    if os.path.exists(name + "_Q5" + str(rank) + ".dat"):
        Chi_Q5, _ = utils.data_collect(0, 0, name + "_Q5" + str(rank) + ".dat")
    else:
        M_Q5 = pnjl.defaults.default_MQ
        #Q5(F(Nq)q) / Q5(F(DD)q) / Q5(ND)
        dQ5 = (2.0 * 1.0 * 3.0) / 2.0
        NQ5 = 5.0
        LQ5 = pnjl.defaults.default_L
        Chi_Q5 = cumulants_c_thermo(rank, T, mu, phi_re, [-el for el in phi_im], M_Q5, 5, 0, dQ5, NQ5, LQ5, 0, regulator_Q5, regulator_Q52)
        with open(name + "_Q5" + str(rank) + ".dat", 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[el] for el in Chi_Q5])

    return (Chi_pi, Chi_rho, Chi_omega, Chi_K, Chi_D, Chi_N, Chi_T, Chi_F, Chi_P, Chi_Q5, Chi_H)

def PNJL_mu_T_plane_calc():

    calc_pl = True
    calc_thermo = True

    cluster_backreaction = False
    pl_turned_off = False

    for mu_trgt in numpy.linspace(1.0, 1000.0, num = 200)[94:95]:

        mu_trgt_round = math.floor(mu_trgt * 10) / 10.0

        phi_re, phi_im = [], []

        Pres_Q, Pres_g, Pres_pert, Pres_sea = [], [], [], []
        BDen_Q, BDen_g, BDen_pert, BDen_sea = [], [], [], []
        SDen_Q, SDen_g, SDen_pert, SDen_sea = [], [], [], []

        Pres_pi, Pres_K, Pres_rho = [], [], []
        Pres_omega, Pres_D, Pres_N = [], [], []
        Pres_T, Pres_F, Pres_P = [], [], []
        Pres_Q5, Pres_H = [], []

        BDen_pi, BDen_K, BDen_rho = [], [], []
        BDen_omega, BDen_D, BDen_N = [], [], []
        BDen_T, BDen_F, BDen_P = [], [], []
        BDen_Q5, BDen_H = [], []

        SDen_pi, SDen_K, SDen_rho = [], [], []
        SDen_omega, SDen_D, SDen_N = [], [], []
        SDen_T, SDen_F, SDen_P = [], [], []
        SDen_Q5, SDen_H = [], []

        Pres_tot, BDen_tot, SDen_tot = [], [], []

        T = numpy.linspace(1.0, 1000.0, num = 200)
        mu = [mu_trgt_round / 3.0 for el in T]

        pl_file = "D:/EoS/BDK/mu_T_plane/pl_" + str(mu_trgt_round).replace(".", "p") + ".dat"
        pressure_file = "D:/EoS/BDK/mu_T_plane/pressure_" + str(mu_trgt_round).replace(".", "p") + ".dat"
        baryon_file = "D:/EoS/BDK/mu_T_plane/bdensity_" + str(mu_trgt_round).replace(".", "p") + ".dat"
        entropy_file = "D:/EoS/BDK/mu_T_plane/entropy_" + str(mu_trgt_round).replace(".", "p") + ".dat"

        if calc_pl:
            phi_re.append(1e-15)
            phi_im.append(2e-15)
            lT = len(T)
            for T_el, mu_el in tqdm.tqdm(zip(T, mu), desc = "Traced Polyakov loop, mu = " + str(mu_trgt_round), total = lT, ascii = True):
                temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re[-1], phi_im[-1], with_clusters = cluster_backreaction)
                phi_re.append(temp_phi_re)
                phi_im.append(temp_phi_im)
            phi_re = phi_re[1:]
            phi_im = phi_im[1:]
            with open(pl_file, 'w', newline = '') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T, mu, phi_re, phi_im)])
        else:
            T, mu = utils.data_collect(0, 1, pl_file)
            phi_re, phi_im = utils.data_collect(2, 3, pl_file)
        
        if calc_thermo:
            Pres_g = [
                pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                for T_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T, phi_re, phi_im), 
                    desc = "Gluon pressure, mu = " + str(mu_trgt_round), 
                    total = len(T),
                    ascii = True)]
            BDen_g = [0.0 for T_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, phi_re, phi_im), desc = "Gluon baryon density, mu = " + str(mu_trgt_round), total = len(T), ascii = True)]
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, phi_re, phi_im), desc = "Gluon entropy density, mu = " + str(mu_trgt_round), total = len(T), ascii = True):
                h = 1e-2
                T_vec = []
                Phi_vec = []
                Phib_vec = []
                if T_el > 0.0:
                    T_vec = [T_el + 2 * h, T_el + h, T_el - h, T_el - 2 * h]
                else:
                    T_vec = [h, 0.0]
                Phi_vec = [complex(phi_re_el, phi_im_el) for el in T_vec]
                Phib_vec = [complex(phi_re_el, -phi_im_el) for el in T_vec]
                SDen_g.append(pnjl.thermo.gcp_pl_polynomial.sdensity(T_vec, Phi_vec, Phib_vec))
            Pres_sea = [0.0 for T_el, mu_el in tqdm.tqdm(zip(T, mu), desc = "Sigma mf pressure, mu = " + str(mu_trgt_round), total = len(T), ascii = True)]
            BDen_sea = [0.0 for T_el, mu_el in tqdm.tqdm(zip(T, mu), desc = "Sigma mf baryon density, mu = " + str(mu_trgt_round), total = len(T), ascii = True)]
            SDen_sea = [0.0 for T_el, mu_el in tqdm.tqdm(zip(T, mu), desc = "Sigma mf entropy density, mu = " + str(mu_trgt_round), total = len(T), ascii = True)]
            if not pl_turned_off:
                Pres_Q = [
                    pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                    for T_el, mu_el, phi_re_el, phi_im_el 
                    in tqdm.tqdm(zip(T, mu, phi_re, phi_im), 
                        desc = "Quark pressure, mu = " + str(mu_trgt_round), 
                        total = len(T), 
                        ascii = True)]
                Pres_pert = [
                    pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                    for T_el, mu_el, phi_re_el, phi_im_el 
                    in tqdm.tqdm(zip(T, mu, phi_re, phi_im), 
                        desc = "Perturbative pressure, mu = " + str(mu_trgt_round), 
                        total = len(T), 
                        ascii = True)]
                for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im), desc = "Quark and perturbative baryon density, mu = " + str(mu_trgt_round), total = len(T), ascii = True):
                    h = 1e-2
                    mu_vec = []
                    Phi_vec = []
                    Phib_vec = []
                    if mu_el > 0.0:
                        mu_vec = [mu_el + 2 * h, mu_el + h, mu_el - h, mu_el - 2 * h]
                    else:
                        mu_vec = [h, 0.0]
                    Phi_vec = [complex(phi_re_el, phi_im_el) for el in mu_vec]
                    Phib_vec = [complex(phi_re_el, -phi_im_el) for el in mu_vec]
                    BDen_Q.append(pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec) + pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, ml = pnjl.defaults.default_ms))
                    BDen_pert.append(pnjl.thermo.gcp_perturbative.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, Nf = 3.0))
                for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, mu, phi_re, phi_im), desc = "Quark and perturbative entropy density, mu = " + str(mu_trgt_round), total = len(T), ascii = True):
                    h = 1e-2
                    T_vec = []
                    Phi_vec = []
                    Phib_vec = []
                    if T_el > 0.0:
                        T_vec = [T_el + 2 * h, T_el + h, T_el - h, T_el - 2 * h]
                    else:
                        T_vec = [h, 0.0]
                    Phi_vec = [complex(phi_re_el, phi_im_el) for el in T_vec]
                    Phib_vec = [complex(phi_re_el, -phi_im_el) for el in T_vec]
                    SDen_Q.append(pnjl.thermo.gcp_quark.sdensity(T_vec, mu_el, Phi_vec, Phib_vec) + pnjl.thermo.gcp_quark.sdensity(T_vec, mu_el, Phi_vec, Phib_vec, ml = pnjl.defaults.default_ms))
                    SDen_pert.append(pnjl.thermo.gcp_perturbative.sdensity(T_vec, mu_el, Phi_vec, Phib_vec, Nf = 3.0))
                ((Pres_pi, Pres_rho, Pres_omega, Pres_K, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, 
                     Pres_Q5, Pres_H),
                    (BDen_pi, BDen_rho, BDen_omega, BDen_K, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, 
                     BDen_Q5, BDen_H),
                    (SDen_pi, SDen_rho, SDen_omega, SDen_K, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, 
                     SDen_Q5, SDen_H)) = clusters_c(T, mu, phi_re, phi_im)
            else:
                Pres_Q = [
                    pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                    for T_el, mu_el, phi_re_el, phi_im_el 
                    in tqdm.tqdm(zip(T, mu, [1.0 for el in T], [0.0 for el in T]), 
                        desc = "Quark pressure, mu = " + str(mu_trgt_round), 
                        total = len(T), 
                        ascii = True)]
                Pres_pert = [
                    pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                    for T_el, mu_el, phi_re_el, phi_im_el 
                    in tqdm.tqdm(zip(T, mu, [1.0 for el in T], [0.0 for el in T]), 
                        desc = "Perturbative pressure, mu = " + str(mu_trgt_round), 
                        total = len(T), 
                        ascii = True)]
                for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, mu, [1.0 for el in T], [0.0 for el in T]), desc = "Quark and perturbative baryon density, mu = " + str(mu_trgt_round), total = len(T), ascii = True):
                    h = 1e-2
                    mu_vec = []
                    Phi_vec = []
                    Phib_vec = []
                    if mu_el > 0.0:
                        mu_vec = [mu_el + 2 * h, mu_el + h, mu_el - h, mu_el - 2 * h]
                    else:
                        mu_vec = [h, 0.0]
                    Phi_vec = [complex(phi_re_el, phi_im_el) for el in mu_vec]
                    Phib_vec = [complex(phi_re_el, -phi_im_el) for el in mu_vec]
                    BDen_Q.append(pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec) + pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, ml = pnjl.defaults.default_ms))
                    BDen_pert.append(pnjl.thermo.gcp_perturbative.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, Nf = 3.0))
                for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T, mu, [1.0 for el in T], [0.0 for el in T]), desc = "Quark and perturbative entropy density, mu = " + str(mu_trgt_round), total = len(T), ascii = True):
                    h = 1e-2
                    T_vec = []
                    Phi_vec = []
                    Phib_vec = []
                    if T_el > 0.0:
                        T_vec = [T_el + 2 * h, T_el + h, T_el - h, T_el - 2 * h]
                    else:
                        T_vec = [h, 0.0]
                    Phi_vec = [complex(phi_re_el, phi_im_el) for el in T_vec]
                    Phib_vec = [complex(phi_re_el, -phi_im_el) for el in T_vec]
                    SDen_Q.append(pnjl.thermo.gcp_quark.sdensity(T_vec, mu_el, Phi_vec, Phib_vec) + pnjl.thermo.gcp_quark.sdensity(T_vec, mu_el, Phi_vec, Phib_vec, ml = pnjl.defaults.default_ms))
                    SDen_pert.append(pnjl.thermo.gcp_perturbative.sdensity(T_vec, mu_el, Phi_vec, Phib_vec, Nf = 3.0))
                ((Pres_pi, Pres_rho, Pres_omega, Pres_K, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, 
                     Pres_Q5, Pres_H),
                    (BDen_pi, BDen_rho, BDen_omega, BDen_K, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, 
                     BDen_Q5, BDen_H),
                    (SDen_pi, SDen_rho, SDen_omega, SDen_K, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, 
                     SDen_Q5, SDen_H)) = clusters_c(T, mu, [1.0 for el in T], [0.0 for el in T])
            with open(pressure_file, 'w', newline = '') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerows([[temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(T, mu, Pres_Q, Pres_g, Pres_pert, Pres_pi, Pres_K, Pres_rho, Pres_omega, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, Pres_Q5, Pres_H, Pres_sea)])
            with open(baryon_file, 'w', newline = '') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerows([[temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(T, mu, BDen_Q, BDen_g, BDen_pert, BDen_pi, BDen_K, BDen_rho, BDen_omega, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, BDen_Q5, BDen_H, BDen_sea)])
            with open(entropy_file, 'w', newline = '') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerows([[temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for temp_el, mu_el, q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(T, mu, SDen_Q, SDen_g, SDen_pert, SDen_pi, SDen_K, SDen_rho, SDen_omega, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, SDen_Q5, SDen_H, SDen_sea)])
        else:
            T, mu = utils.data_collect(0, 1, pressure_file)
            Pres_Q, Pres_g = utils.data_collect(2, 3, pressure_file)
            Pres_pert, Pres_pi = utils.data_collect(4, 5, pressure_file)
            Pres_K, Pres_rho = utils.data_collect(6, 7, pressure_file)
            Pres_omega, Pres_D = utils.data_collect(8, 9, pressure_file)
            Pres_N, Pres_T = utils.data_collect(10, 11, pressure_file)
            Pres_F, Pres_P = utils.data_collect(12, 13, pressure_file)
            Pres_Q5, Pres_H = utils.data_collect(14, 15, pressure_file)
            Pres_sea, _ = utils.data_collect(16, 16, pressure_file)
            BDen_Q, BDen_g = utils.data_collect(2, 3, baryon_file)
            BDen_pert, BDen_pi = utils.data_collect(4, 5, baryon_file)
            BDen_K, BDen_rho = utils.data_collect(6, 7, baryon_file)
            BDen_omega, BDen_D = utils.data_collect(8, 9, baryon_file)
            BDen_N, BDen_T = utils.data_collect(10, 11, baryon_file)
            BDen_F, BDen_P = utils.data_collect(12, 13, baryon_file)
            BDen_Q5, BDen_H = utils.data_collect(14, 15, baryon_file)
            BDen_sea, _ = utils.data_collect(16, 16, baryon_file)
            SDen_Q, SDen_g = utils.data_collect(2, 3, entropy_file)
            SDen_pert, SDen_pi = utils.data_collect(4, 5, entropy_file)
            SDen_K, SDen_rho = utils.data_collect(6, 7, entropy_file)
            SDen_omega, SDen_D = utils.data_collect(8, 9, entropy_file)
            SDen_N, SDen_T = utils.data_collect(10, 11, entropy_file)
            SDen_F, SDen_P = utils.data_collect(12, 13, entropy_file)
            SDen_Q5, SDen_H = utils.data_collect(14, 15, entropy_file)
            SDen_sea, _ = utils.data_collect(16, 16, entropy_file)

def epja_figure3():

    T0 = numpy.linspace(1.0, 250.0, num = 200)
    T300 = numpy.linspace(1.0, 250.0, num = 200)
    T600 = numpy.linspace(1.0, 250.0, num = 200)
    T900 = numpy.linspace(1.0, 250.0, num = 200)
    Talt0 = numpy.linspace(1.0, 250.0, num = 200)
    Talt300 = numpy.linspace(1.0, 250.0, num = 200)
    Talt600 = numpy.linspace(1.0, 250.0, num = 200)
    Talt900 = numpy.linspace(1.0, 250.0, num = 200)

    Tstep0 = numpy.linspace(1.0, 250.0, num = 200)
    Tstep600 = numpy.linspace(1.0, 250.0, num = 200)

    phi_re_0, phi_im_0 = [], []
    phi_re_300, phi_im_300 = [], []
    phi_re_600, phi_im_600 = [], []
    phi_re_900, phi_im_900 = [], []
    phi_re_alt_0, phi_im_alt_0 = [], []
    phi_re_alt_300, phi_im_alt_300 = [], []
    phi_re_alt_600, phi_im_alt_600 = [], []
    phi_re_alt_900, phi_im_alt_900 = [], []

    phi_re_step_0, phi_im_step_0 = [], []
    phi_re_step_600, phi_im_step_600 = [], []

    pl0_file = "D:/EoS/epja/figure2/pl_0.dat"
    pl300_file = "D:/EoS/epja/figure2/pl_300.dat"
    pl600_file = "D:/EoS/epja/figure2/pl_600.dat"
    pl900_file = "D:/EoS/epja/figure2/pl_900.dat"
    plalt0_file = "D:/EoS/epja/figure2/pl0_0.dat"
    plalt300_file = "D:/EoS/epja/figure2/pl0_300.dat"
    plalt600_file = "D:/EoS/epja/figure2/pl0_600.dat"
    plalt900_file = "D:/EoS/epja/figure2/pl0_900.dat"

    plstep0_file = "D:/EoS/epja/figure2/pl_step_0.dat"
    plstep600_file = "D:/EoS/epja/figure2/pl_step_600.dat"

    pl0_calc = False
    pl300_calc = False
    pl600_calc = False
    pl900_calc = False
    plalt0_calc = False
    plalt300_calc = False
    plalt600_calc = False
    plalt900_calc = False

    plstep0_calc = False
    plstep600_calc = False

    sigma_0 = [pnjl.thermo.gcp_sea_lattice.Delta_ls(el, 0.0)   for el in T0]
    sigma_300 = [pnjl.thermo.gcp_sea_lattice.Delta_ls(el, 100.0) for el in T300]
    sigma_600 = [pnjl.thermo.gcp_sea_lattice.Delta_ls(el, 200.0) for el in T600]
    sigma_900 = [pnjl.thermo.gcp_sea_lattice.Delta_ls(el, 300.0) for el in T900]

    if pl900_calc:
        with open(pl900_file, 'a', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            phi_re_900.append(1e-15)
            phi_im_900.append(2e-15)
            lT = len(T900)
            for T_el in tqdm.tqdm(T900, desc = "Traced Polyakov loop, mu = 900 MeV", total = lT, ascii = True):
                temp_phi_re, temp_phi_im = calc_PL_c(T_el, 300.0, phi_re_900[-1], phi_im_900[-1], with_clusters = True)
                phi_re_900.append(temp_phi_re)
                phi_im_900.append(temp_phi_im)
                writer.writerow([T_el, temp_phi_re, temp_phi_im])
                file.flush()
            phi_re_900 = phi_re_900[1:]
            phi_im_900 = phi_im_900[1:]
    else:
        T900, phi_re_900 = utils.data_collect(0, 1, pl900_file)
        phi_im_900, _ = utils.data_collect(2, 2, pl900_file)

    if pl600_calc:
        phi_re_600.append(1e-15)
        phi_im_600.append(2e-15)
        lT = len(T600)
        for T_el in tqdm.tqdm(T600, desc = "Traced Polyakov loop, mu = 600 MeV", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, 200.0, phi_re_600[-1], phi_im_600[-1], with_clusters = True)
            phi_re_600.append(temp_phi_re)
            phi_im_600.append(temp_phi_im)
        phi_re_600 = phi_re_600[1:]
        phi_im_600 = phi_im_600[1:]
        with open(pl600_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T600, phi_re_600, phi_im_600)])
    else:
        T600, phi_re_600 = utils.data_collect(0, 1, pl600_file)
        phi_im_600, _ = utils.data_collect(2, 2, pl600_file)

    if pl300_calc:
        phi_re_300.append(1e-15)
        phi_im_300.append(2e-15)
        lT = len(T300)
        for T_el in tqdm.tqdm(T300, desc = "Traced Polyakov loop, mu = 300 MeV", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, 100.0, phi_re_300[-1], phi_im_300[-1], with_clusters = True)
            phi_re_300.append(temp_phi_re)
            phi_im_300.append(temp_phi_im)
        phi_re_300 = phi_re_300[1:]
        phi_im_300 = phi_im_300[1:]
        with open(pl300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T300, phi_re_300, phi_im_300)])
    else:
        T300, phi_re_300 = utils.data_collect(0, 1, pl300_file)
        phi_im_300, _ = utils.data_collect(2, 2, pl300_file)

    if pl0_calc:
        phi_re_0.append(1e-15)
        phi_im_0.append(2e-15)
        lT = len(T0)
        for T_el in tqdm.tqdm(T0, desc = "Traced Polyakov loop, mu = 0", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, 0.0, phi_re_0[-1], phi_im_0[-1], with_clusters = True)
            phi_re_0.append(temp_phi_re)
            phi_im_0.append(temp_phi_im)
        phi_re_0 = phi_re_0[1:]
        phi_im_0 = phi_im_0[1:]
        with open(pl0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T0, phi_re_0, phi_im_0)])
    else:
        T0, phi_re_0 = utils.data_collect(0, 1, pl0_file)
        phi_im_0, _ = utils.data_collect(2, 2, pl0_file)

    if plalt0_calc:
        phi_re_alt_0.append(1e-15)
        phi_im_alt_0.append(2e-15)
        lT = len(Talt0)
        for T_el in tqdm.tqdm(Talt0, desc = "Traced Polyakov loop alt, mu = 0", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, 0.0, phi_re_alt_0[-1], phi_im_alt_0[-1], with_clusters = False)
            phi_re_alt_0.append(temp_phi_re)
            phi_im_alt_0.append(temp_phi_im)
        phi_re_alt_0 = phi_re_alt_0[1:]
        phi_im_alt_0 = phi_im_alt_0[1:]
        with open(plalt0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(Talt0, phi_re_alt_0, phi_im_alt_0)])
    else:
        Talt0, phi_re_alt_0 = utils.data_collect(0, 1, plalt0_file)
        phi_im_alt_0, _ = utils.data_collect(2, 2, plalt0_file)

    if plalt300_calc:
        phi_re_alt_300.append(1e-15)
        phi_im_alt_300.append(2e-15)
        lT = len(Talt300)
        for T_el in tqdm.tqdm(Talt300, desc = "Traced Polyakov loop alt, mu = 300 MeV", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, 100.0, phi_re_alt_300[-1], phi_im_alt_300[-1], with_clusters = False)
            phi_re_alt_300.append(temp_phi_re)
            phi_im_alt_300.append(temp_phi_im)
        phi_re_alt_300 = phi_re_alt_300[1:]
        phi_im_alt_300 = phi_im_alt_300[1:]
        with open(plalt300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(Talt300, phi_re_alt_300, phi_im_alt_300)])
    else:
        Talt300, phi_re_alt_300 = utils.data_collect(0, 1, plalt300_file)
        phi_im_alt_300, _ = utils.data_collect(2, 2, plalt300_file)

    if plalt600_calc:
        phi_re_alt_600.append(1e-15)
        phi_im_alt_600.append(2e-15)
        lT = len(Talt600)
        for T_el in tqdm.tqdm(Talt600, desc = "Traced Polyakov loop alt, mu = 600 MeV", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, 200.0, phi_re_alt_600[-1], phi_im_alt_600[-1], with_clusters = False)
            phi_re_alt_600.append(temp_phi_re)
            phi_im_alt_600.append(temp_phi_im)
        phi_re_alt_600 = phi_re_alt_600[1:]
        phi_im_alt_600 = phi_im_alt_600[1:]
        with open(plalt600_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(Talt600, phi_re_alt_600, phi_im_alt_600)])
    else:
        Talt600, phi_re_alt_600 = utils.data_collect(0, 1, plalt600_file)
        phi_im_alt_600, _ = utils.data_collect(2, 2, plalt600_file)

    if plalt900_calc:
        phi_re_alt_900.append(1e-15)
        phi_im_alt_900.append(2e-15)
        lT = len(Talt900)
        for T_el in tqdm.tqdm(Talt900, desc = "Traced Polyakov loop alt, mu = 900 MeV", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, 300.0, phi_re_alt_900[-1], phi_im_alt_900[-1], with_clusters = False)
            phi_re_alt_900.append(temp_phi_re)
            phi_im_alt_900.append(temp_phi_im)
        phi_re_alt_900 = phi_re_alt_900[1:]
        phi_im_alt_900 = phi_im_alt_900[1:]
        with open(plalt900_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(Talt900, phi_re_alt_900, phi_im_alt_900)])
    else:
        Talt900, phi_re_alt_900 = utils.data_collect(0, 1, plalt900_file)
        phi_im_alt_900, _ = utils.data_collect(2, 2, plalt900_file)

    if plstep0_calc:
        phi_re_step_0.append(1e-15)
        phi_im_step_0.append(2e-15)
        lT = len(Tstep0)
        for T_el in tqdm.tqdm(Tstep0, desc = "Traced Polyakov loop (step), mu = 0", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, 0.0, phi_re_step_0[-1], phi_im_step_0[-1], with_clusters = True)
            phi_re_step_0.append(temp_phi_re)
            phi_im_step_0.append(temp_phi_im)
        phi_re_step_0 = phi_re_step_0[1:]
        phi_im_step_0 = phi_im_step_0[1:]
        with open(plstep0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(Tstep0, phi_re_step_0, phi_im_step_0)])
    else:
        Tstep0, phi_re_step_0 = utils.data_collect(0, 1, plstep0_file)
        phi_im_step_0, _ = utils.data_collect(2, 2, plstep0_file)

    if plstep600_calc:
        phi_re_step_600.append(1e-15)
        phi_im_step_600.append(2e-15)
        lT = len(Tstep600)
        for T_el in tqdm.tqdm(Tstep600, desc = "Traced Polyakov loop (step), mu = 600 MeV", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, 200.0, phi_re_step_600[-1], phi_im_step_600[-1], with_clusters = True)
            phi_re_step_600.append(temp_phi_re)
            phi_im_step_600.append(temp_phi_im)
        phi_re_step_600 = phi_re_step_600[1:]
        phi_im_step_600 = phi_im_step_600[1:]
        with open(plstep600_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(Tstep600, phi_re_step_600, phi_im_step_600)])
    else:
        Tstep600, phi_re_step_600 = utils.data_collect(0, 1, plstep600_file)
        phi_im_step_600, _ = utils.data_collect(2, 2, plstep600_file)

    fig = matplotlib.pyplot.figure(num = 1, figsize = (11.0, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.axis([min(T0), max(T0), min(phi_re_0), 1.1])
    ax.plot(T0, phi_re_0, c = 'blue')
    ax.plot(Tstep0, phi_re_step_0, '--', c = 'blue')
    ax.plot(Talt0, phi_re_alt_0, '--', c = 'red')
    ax.plot(T0, sigma_0, c = 'green')
    ax.text(10., 1.03, r'$\mathrm{M_q}$ / $\mathrm{M_{q,vac}}$', fontsize = 14)
    ax.text(175., 1.03, r'$\mathrm{\mu_B=0}$', fontsize = 14)
    ax.text(135., 0.25, r'$\mathrm{\Phi}$', fontsize = 14, color = 'blue')
    ax.text(150., 0.25, r'$\mathrm{\Phi_0}$', fontsize = 14, color = 'red')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax.set_xlabel(r'T [MeV]', fontsize = 16)
    ax.set_ylabel(r'$\mathrm{\Phi}$', fontsize = 16)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis([min(T600), max(T600), min(phi_re_0), 1.1])
    ax2.plot([el for i, el in enumerate(T600) if not i == 115], [el for i, el in enumerate(phi_re_600) if not i == 115], c = 'blue', label = r'$\mathrm{\Phi}$', zorder = 4)
    ax2.plot(Talt600, phi_re_alt_600, '--', c = 'red', zorder = 5)
    ax2.plot(T600, sigma_600, c = 'green')
    ax2.fill_between(Tstep600, phi_re_step_600, y2 = phi_re_alt_600, color = 'blue', edgecolor = 'blue', alpha = 0.3)
    ax2.fill_between(Tstep600, phi_re_600, y2 = phi_re_step_600, color = 'salmon', edgecolor = 'none', alpha = 1.0, zorder = 3)
    ax2.plot([135.0, 172.0], [0.34, 0.44], '-', c = 'black', lw = 1, zorder = 5)
    ax2.plot([149.0, 172.0], [0.54, 0.65], '-', c = 'black', lw = 1, zorder = 5)
    ax2.text(173., 0.50, r'bound state', fontsize = 14)
    ax2.text(173., 0.45, r'influence', fontsize = 14)
    ax2.text(173., 0.71, r'continuum', fontsize = 14)
    ax2.text(173., 0.66, r'influence', fontsize = 14)
    ax2.text(10., 1.03, r'$\mathrm{M_q}$ / $\mathrm{M_{q,vac}}$', fontsize = 14)
    ax2.text(145., 1.03, r'$\mathrm{\mu_B=600}$ MeV', fontsize = 14)
    ax2.text(110., 0.25, r'$\mathrm{\Phi}$', fontsize = 14, color = 'blue')
    ax2.text(140., 0.25, r'$\mathrm{\Phi_0}$', fontsize = 14, color = 'red')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{\Phi}$', fontsize = 16)

    fig.tight_layout()
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def epja_figure4():

    phi_re_1, phi_im_1 = [], []

    Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_sea_1 = [], [], [], []

    T_1 = numpy.linspace(1.0, 2000.0, 200)

    mu_1 = [0.0 / 3.0 for el in T_1]

    calc_1 = False

    cluster_backreaction = False
    pl_turned_off = False

    pl_1_file = "D:/EoS/epja/figure3/pl_1.dat"

    pressure_1_file = "D:/EoS/epja/figure3/pressure_1.dat"

    if calc_1:
        phi_re_1.append(1e-15)
        phi_im_1.append(2e-15)
        lT = len(T_1)
        for T_el, mu_el in tqdm.tqdm(zip(T_1, mu_1), desc = "Traced Polyakov loop (calc #1)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_1[-1], phi_im_1[-1], with_clusters = cluster_backreaction)
            phi_re_1.append(temp_phi_re)
            phi_im_1.append(temp_phi_im)
        phi_re_1 = phi_re_1[1:]
        phi_im_1 = phi_im_1[1:]
        with open(pl_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_1, mu_1, phi_re_1, phi_im_1)])
    else:
        T_1, mu_1 = utils.data_collect(0, 1, pl_1_file)
        phi_re_1, phi_im_1 = utils.data_collect(2, 3, pl_1_file)

    if calc_1:
        Pres_g_1 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_1, phi_re_1, phi_im_1), 
                desc = "Gluon pressure (calc #1)", 
                total = len(T_1),
                ascii = True)]
        Pres_sea_1 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(zip(T_1, mu_1), 
                desc = "Sigma mf pressure (calc #1)", 
                total = len(T_1),
                ascii = True)]
        if not pl_turned_off:
            Pres_Q_1 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Quark pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            Pres_pert_1 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Perturbative pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]        
        else:
            Pres_Q_1 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, [1.0 for el in T_1], [0.0 for el in T_1]), 
                    desc = "Quark pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            Pres_pert_1 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, [1.0 for el in T_1], [0.0 for el in T_1]), 
                    desc = "Perturbative pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]        
        with open(pressure_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sea_el] for q_el, g_el, pert_el, sea_el in zip(Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_sea_1)])
    else:
        Pres_Q_1, Pres_g_1 = utils.data_collect(0, 1, pressure_1_file)
        Pres_pert_1, Pres_sea_1 = utils.data_collect(2, 14, pressure_1_file)

    contrib_q_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_Q_1)]
    contrib_g_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_g_1)]
    contrib_pert_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_pert_1)]
    contrib_sea_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_1)]
    contrib_qgp_1 = [sum(el) for el in zip(contrib_q_1, contrib_g_1, contrib_pert_1, contrib_sea_1)]

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure3/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure3/1204_6710v2_table4_pressure_mu0_high.dat")
    (high_1407_6387_mu0_x, high_1407_6387_mu0_y) = utils.data_collect(0, 3, "D:/EoS/epja/figure3/1407_6387_table1_pressure_mu0.dat")
    (low_1407_6387_mu0_x, low_1407_6387_mu0_y) = utils.data_collect(0, 2, "D:/EoS/epja/figure3/1407_6387_table1_pressure_mu0.dat")
    (high_1309_5258_mu0_x, high_1309_5258_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure3/1309_5258_figure6_pressure_mu0_high.dat")
    (low_1309_5258_mu0_x, low_1309_5258_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure3/1309_5258_figure6_pressure_mu0_low.dat")
    (high_1710_05024_mu0_x, high_1710_05024_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure3/1710_05024_figure8_pressure_mu0_high.dat")
    (low_1710_05024_mu0_x, low_1710_05024_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure3/1710_05024_figure8_pressure_mu0_low.dat")

    borsanyi_1204_6710v2_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = numpy.array(borsanyi_1204_6710v2_mu0)
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

    fig2 = matplotlib.pyplot.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.axis([10., 2000., -1.5, 5.1])
    ax2.add_patch(matplotlib.patches.Polygon(bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Bazavov et al. (2014)'))
    ax2.add_patch(matplotlib.patches.Polygon(borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2014)'))
    ax2.add_patch(matplotlib.patches.Polygon(bazavov_1710_05024_mu0, closed = True, fill = True, color = 'magenta', alpha = 0.3, label = r'Bazavov et al. (2018)'))
    ax2.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012)'))
    ax2.plot(T_1, contrib_qgp_1, '-', c = 'black')
    ax2.plot(T_1, contrib_g_1, '--', c = 'red')
    ax2.plot(T_1, contrib_q_1, '--', c = 'blue')
    ax2.plot(T_1, contrib_pert_1, '--', c = 'pink')
    ax2.text(960.0, 1.15, r'Polyakov--loop potential', color = 'red', fontsize = 14)
    ax2.text(960.0, -0.5, r'Perturbative correction', color = 'red', alpha = 0.6, fontsize = 14)
    ax2.text(1230.0, 3.5, r'PNJL quarks', color = 'blue', fontsize = 14)
    ax2.text(1230.0, 4.3, r'total pressure', color = 'black', fontsize = 14)
    ax2.text(45.0, 4.8, r'$\mathrm{\mu_B=0}$', color = 'black', fontsize = 14)
    ax2.text(430.0, 3.1, r'Borsanyi et al. (2012)', color = 'blue', alpha = 0.7, fontsize = 14)
    ax2.text(580.0, 3.88, r'Borsanyi et al. (2014)', color = 'green', alpha = 0.7, fontsize = 14)
    ax2.text(345.0, 2.5, r'Bazavov et al. (2014)', color = 'red', alpha = 0.7, fontsize = 14)
    ax2.text(1000.0, 4.8, r'Bazavov et al. (2018)', color = 'magenta', alpha = 0.7, fontsize = 14)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig2.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def epja_figure5():

    phi_re_1, phi_im_1 = [], []
    phi_re_3, phi_im_3 = [], []
    phi_re_4, phi_im_4 = [], []
    phi_re_5, phi_im_5 = [], []
    phi_re_6, phi_im_6 = [], []

    Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_sea_1 = [], [], [], []
    Pres_Q_3, Pres_g_3, Pres_pert_3, Pres_sea_3 = [], [], [], []
    Pres_Q_4, Pres_g_4, Pres_pert_4, Pres_sea_4 = [], [], [], []
    Pres_Q_5, Pres_g_5, Pres_pert_5, Pres_sea_5 = [], [], [], []
    Pres_Q_6, Pres_g_6, Pres_pert_6, Pres_sea_6 = [], [], [], []

    Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_sea_1 = [], [], [], []

    Pres_pi_1, Pres_K_1, Pres_rho_1 = [], [], []
    Pres_omega_1, Pres_D_1, Pres_N_1 = [], [], []
    Pres_T_1, Pres_F_1, Pres_P_1 = [], [], []
    Pres_Q5_1, Pres_H_1 = [], []

    Pres_pi_3, Pres_K_3, Pres_rho_3 = [], [], [] 
    Pres_omega_3, Pres_D_3, Pres_N_3 = [], [], []
    Pres_T_3, Pres_F_3, Pres_P_3 = [], [], []
    Pres_Q5_3, Pres_H_3 = [], []

    Pres_pi_4, Pres_K_4, Pres_rho_4 = [], [], [] 
    Pres_omega_4, Pres_D_4, Pres_N_4 = [], [], []
    Pres_T_4, Pres_F_4, Pres_P_4 = [], [], []
    Pres_Q5_4, Pres_H_4 = [], []

    Pres_pi_5, Pres_K_5, Pres_rho_5 = [], [], [] 
    Pres_omega_5, Pres_D_5, Pres_N_5 = [], [], []
    Pres_T_5, Pres_F_5, Pres_P_5 = [], [], []
    Pres_Q5_5, Pres_H_5 = [], []

    Pres_pi_6, Pres_K_6, Pres_rho_6 = [], [], [] 
    Pres_omega_6, Pres_D_6, Pres_N_6 = [], [], []
    Pres_T_6, Pres_F_6, Pres_P_6 = [], [], []
    Pres_Q5_6, Pres_H_6 = [], []

    T_1 = numpy.linspace(1.0, 2000.0, 200)
    T_3 = numpy.linspace(1.0, 2000.0, 200)
    T_4 = numpy.linspace(1.0, 2000.0, 200)
    T_5 = numpy.linspace(1.0, 2000.0, 200)
    T_6 = numpy.linspace(1.0, 2000.0, 200)

    mu_1 = [0.0 / 3.0 for el in T_1]
    mu_3 = [400.0 / 3.0 for el in T_3]
    mu_4 = [100.0 / 3.0 for el in T_3]
    mu_5 = [200.0 / 3.0 for el in T_3]
    mu_6 = [300.0 / 3.0 for el in T_3]

    calc_1 = False
    calc_3 = False
    calc_4 = False
    calc_5 = False
    calc_6 = False

    cluster_backreaction = False
    pl_turned_off = False

    pl_1_file = "D:/EoS/epja/figure4/pl_1.dat"
    pl_3_file = "D:/EoS/epja/figure4/pl_3.dat"
    pl_4_file = "D:/EoS/epja/figure4/pl_4.dat"
    pl_5_file = "D:/EoS/epja/figure4/pl_5.dat"
    pl_6_file = "D:/EoS/epja/figure4/pl_6.dat"
    pressure_1_file = "D:/EoS/epja/figure4/pressure_1.dat"
    pressure_3_file = "D:/EoS/epja/figure4/pressure_3.dat"
    pressure_4_file = "D:/EoS/epja/figure4/pressure_4.dat"
    pressure_5_file = "D:/EoS/epja/figure4/pressure_5.dat"
    pressure_6_file = "D:/EoS/epja/figure4/pressure_6.dat"

    if calc_1:
        phi_re_1.append(1e-15)
        phi_im_1.append(2e-15)
        lT = len(T_1)
        for T_el, mu_el in tqdm.tqdm(zip(T_1, mu_1), desc = "Traced Polyakov loop (calc #1)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_1[-1], phi_im_1[-1], with_clusters = cluster_backreaction)
            phi_re_1.append(temp_phi_re)
            phi_im_1.append(temp_phi_im)
        phi_re_1 = phi_re_1[1:]
        phi_im_1 = phi_im_1[1:]
        with open(pl_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_1, mu_1, phi_re_1, phi_im_1)])
    else:
        T_1, mu_1 = utils.data_collect(0, 1, pl_1_file)
        phi_re_1, phi_im_1 = utils.data_collect(2, 3, pl_1_file)

    if calc_3:
        phi_re_3.append(1e-15)
        phi_im_3.append(2e-15)
        lT = len(T_3)
        for T_el, mu_el in tqdm.tqdm(zip(T_3, mu_3), desc = "Traced Polyakov loop (calc #3)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_3[-1], phi_im_3[-1], with_clusters = cluster_backreaction)
            phi_re_3.append(temp_phi_re)
            phi_im_3.append(temp_phi_im)
        phi_re_3 = phi_re_3[1:]
        phi_im_3 = phi_im_3[1:]
        with open(pl_3_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_3, mu_3, phi_re_3, phi_im_3)])
    else:
        T_3, mu_3 = utils.data_collect(0, 1, pl_3_file)
        phi_re_3, phi_im_3 = utils.data_collect(2, 3, pl_3_file)

    if calc_4:
        phi_re_4.append(1e-15)
        phi_im_4.append(2e-15)
        lT = len(T_4)
        for T_el, mu_el in tqdm.tqdm(zip(T_4, mu_4), desc = "Traced Polyakov loop (calc #4)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_4[-1], phi_im_4[-1], with_clusters = cluster_backreaction)
            phi_re_4.append(temp_phi_re)
            phi_im_4.append(temp_phi_im)
        phi_re_4 = phi_re_4[1:]
        phi_im_4 = phi_im_4[1:]
        with open(pl_4_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_4, mu_4, phi_re_4, phi_im_4)])
    else:
        T_4, mu_4 = utils.data_collect(0, 1, pl_4_file)
        phi_re_4, phi_im_4 = utils.data_collect(2, 3, pl_4_file)

    if calc_5:
        phi_re_5.append(1e-15)
        phi_im_5.append(2e-15)
        lT = len(T_5)
        for T_el, mu_el in tqdm.tqdm(zip(T_5, mu_5), desc = "Traced Polyakov loop (calc #5)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_5[-1], phi_im_5[-1], with_clusters = cluster_backreaction)
            phi_re_5.append(temp_phi_re)
            phi_im_5.append(temp_phi_im)
        phi_re_5 = phi_re_5[1:]
        phi_im_5 = phi_im_5[1:]
        with open(pl_5_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_5, mu_5, phi_re_5, phi_im_5)])
    else:
        T_5, mu_5 = utils.data_collect(0, 1, pl_5_file)
        phi_re_5, phi_im_5 = utils.data_collect(2, 3, pl_5_file)

    if calc_6:
        phi_re_6.append(1e-15)
        phi_im_6.append(2e-15)
        lT = len(T_6)
        for T_el, mu_el in tqdm.tqdm(zip(T_6, mu_6), desc = "Traced Polyakov loop (calc #6)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_6[-1], phi_im_6[-1], with_clusters = cluster_backreaction)
            phi_re_6.append(temp_phi_re)
            phi_im_6.append(temp_phi_im)
        phi_re_6 = phi_re_6[1:]
        phi_im_6 = phi_im_6[1:]
        with open(pl_6_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_6, mu_6, phi_re_6, phi_im_6)])
    else:
        T_6, mu_6 = utils.data_collect(0, 1, pl_6_file)
        phi_re_6, phi_im_6 = utils.data_collect(2, 3, pl_6_file)

    if calc_1:
        Pres_g_1 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_1, phi_re_1, phi_im_1), 
                desc = "Gluon pressure (calc #1)", 
                total = len(T_1),
                ascii = True)]
        Pres_sea_1 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(zip(T_1, mu_1), 
                desc = "Sigma mf pressure (calc #1)", 
                total = len(T_1),
                ascii = True)]
        if not pl_turned_off:
            Pres_Q_1 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Quark pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            Pres_pert_1 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Perturbative pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]        
            ((Pres_pi_1, Pres_rho_1, Pres_omega_1, Pres_K_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, 
                 Pres_Q5_1, Pres_H_1),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_1, mu_1, phi_re_1, phi_im_1)
        else:
            Pres_Q_1 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, [1.0 for el in T_1], [0.0 for el in T_1]), 
                    desc = "Quark pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            Pres_pert_1 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, [1.0 for el in T_1], [0.0 for el in T_1]), 
                    desc = "Perturbative pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]        
            ((Pres_pi_1, Pres_rho_1, Pres_omega_1, Pres_K_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, 
                 Pres_Q5_1, Pres_H_1),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_1, mu_1, [1.0 for el in T_1], [0.0 for el in T_1])
        with open(pressure_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_pi_1, Pres_K_1, Pres_rho_1, Pres_omega_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, Pres_Q5_1, Pres_H_1, Pres_sea_1)])
    else:
        Pres_Q_1, Pres_g_1 = utils.data_collect(0, 1, pressure_1_file)
        Pres_pert_1, Pres_pi_1 = utils.data_collect(2, 3, pressure_1_file)
        Pres_K_1, _ = utils.data_collect(4, 4, pressure_1_file)
        Pres_rho_1, Pres_omega_1 = utils.data_collect(5, 6, pressure_1_file)
        Pres_D_1, Pres_N_1 = utils.data_collect(7, 8, pressure_1_file)
        Pres_T_1, Pres_F_1 = utils.data_collect(9, 10, pressure_1_file)
        Pres_P_1, Pres_Q5_1 = utils.data_collect(11, 12, pressure_1_file)
        Pres_H_1, Pres_sea_1 = utils.data_collect(13, 14, pressure_1_file)

    if calc_3:
        Pres_g_3 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_3, phi_re_3, phi_im_3), 
                desc = "Gluon pressure (calc #3)", 
                total = len(T_3),
                ascii = True)]
        Pres_sea_3 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(zip(T_3, mu_3), 
                desc = "Sigma mf pressure (calc #3)", 
                total = len(T_3),
                ascii = True)]
        if not pl_turned_off:
            Pres_Q_3 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_3, mu_3, phi_re_3, phi_im_3), 
                    desc = "Quark pressure (calc #3)", 
                    total = len(T_3), 
                    ascii = True)]
            Pres_pert_3 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_3, mu_3, phi_re_3, phi_im_3), 
                    desc = "Perturbative pressure (calc #3)", 
                    total = len(T_3), 
                    ascii = True)]        
            ((Pres_pi_3, Pres_rho_3, Pres_omega_3, Pres_K_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, 
                 Pres_Q5_3, Pres_H_3),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_3, mu_3, phi_re_3, phi_im_3)
        else:
            Pres_Q_3 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3]), 
                    desc = "Quark pressure (calc #3)", 
                    total = len(T_3), 
                    ascii = True)]
            Pres_pert_3 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3]), 
                    desc = "Perturbative pressure (calc #3)", 
                    total = len(T_3), 
                    ascii = True)]        
            ((Pres_pi_3, Pres_rho_3, Pres_omega_3, Pres_K_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, 
                 Pres_Q5_3, Pres_H_3),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3])
        with open(pressure_3_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_3, Pres_g_3, Pres_pert_3, Pres_pi_3, Pres_K_3, Pres_rho_3, Pres_omega_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, Pres_Q5_3, Pres_H_3, Pres_sea_3)])
    else:
        Pres_Q_3, Pres_g_3 = utils.data_collect(0, 1, pressure_3_file)
        Pres_pert_3, Pres_pi_3 = utils.data_collect(2, 3, pressure_3_file)
        Pres_K_3, _ = utils.data_collect(4, 4, pressure_3_file)
        Pres_rho_3, Pres_omega_3 = utils.data_collect(5, 6, pressure_3_file)
        Pres_D_3, Pres_N_3 = utils.data_collect(7, 8, pressure_3_file)
        Pres_T_3, Pres_F_3 = utils.data_collect(9, 10, pressure_3_file)
        Pres_P_3, Pres_Q5_3 = utils.data_collect(11, 12, pressure_3_file)
        Pres_H_3, Pres_sea_3 = utils.data_collect(13, 14, pressure_3_file)

    if calc_4:
        Pres_g_4 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_4, phi_re_4, phi_im_4), 
                desc = "Gluon pressure (calc #4)", 
                total = len(T_4),
                ascii = True)]
        Pres_sea_4 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(zip(T_4, mu_4), 
                desc = "Sigma mf pressure (calc #4)", 
                total = len(T_4),
                ascii = True)]
        if not pl_turned_off:
            Pres_Q_4 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_4, mu_4, phi_re_4, phi_im_4), 
                    desc = "Quark pressure (calc #4)", 
                    total = len(T_4), 
                    ascii = True)]
            Pres_pert_4 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_4, mu_4, phi_re_4, phi_im_4), 
                    desc = "Perturbative pressure (calc #4)", 
                    total = len(T_3), 
                    ascii = True)]        
            ((Pres_pi_4, Pres_rho_4, Pres_omega_4, Pres_K_4, Pres_D_4, Pres_N_4, Pres_T_4, Pres_F_4, Pres_P_4, 
                 Pres_Q5_4, Pres_H_4),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_4, mu_4, phi_re_4, phi_im_4)
        else:
            Pres_Q_4 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_4, mu_4, [1.0 for el in T_4], [0.0 for el in T_4]), 
                    desc = "Quark pressure (calc #4)", 
                    total = len(T_4), 
                    ascii = True)]
            Pres_pert_4 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_4, mu_4, [1.0 for el in T_4], [0.0 for el in T_4]), 
                    desc = "Perturbative pressure (calc #4)", 
                    total = len(T_4), 
                    ascii = True)]        
            ((Pres_pi_4, Pres_rho_4, Pres_omega_4, Pres_K_4, Pres_D_4, Pres_N_4, Pres_T_4, Pres_F_4, Pres_P_4, 
                 Pres_Q5_4, Pres_H_4),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_4, mu_4, [1.0 for el in T_4], [0.0 for el in T_4])
        with open(pressure_4_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_4, Pres_g_4, Pres_pert_4, Pres_pi_4, Pres_K_4, Pres_rho_4, Pres_omega_4, Pres_D_4, Pres_N_4, Pres_T_4, Pres_F_4, Pres_P_4, Pres_Q5_4, Pres_H_4, Pres_sea_4)])
    else:
        Pres_Q_4, Pres_g_4 = utils.data_collect(0, 1, pressure_4_file)
        Pres_pert_4, Pres_pi_4 = utils.data_collect(2, 3, pressure_4_file)
        Pres_K_4, _ = utils.data_collect(4, 4, pressure_4_file)
        Pres_rho_4, Pres_omega_4 = utils.data_collect(5, 6, pressure_4_file)
        Pres_D_4, Pres_N_4 = utils.data_collect(7, 8, pressure_4_file)
        Pres_T_4, Pres_F_4 = utils.data_collect(9, 10, pressure_4_file)
        Pres_P_4, Pres_Q5_4 = utils.data_collect(11, 12, pressure_4_file)
        Pres_H_4, Pres_sea_4 = utils.data_collect(13, 14, pressure_4_file)

    if calc_5:
        Pres_g_5 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_5, phi_re_5, phi_im_5), 
                desc = "Gluon pressure (calc #5)", 
                total = len(T_5),
                ascii = True)]
        Pres_sea_5 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(zip(T_5, mu_5), 
                desc = "Sigma mf pressure (calc #5)", 
                total = len(T_5),
                ascii = True)]
        if not pl_turned_off:
            Pres_Q_5 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_5, mu_5, phi_re_5, phi_im_5), 
                    desc = "Quark pressure (calc #5)", 
                    total = len(T_5), 
                    ascii = True)]
            Pres_pert_5 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_5, mu_5, phi_re_5, phi_im_5), 
                    desc = "Perturbative pressure (calc #5)", 
                    total = len(T_5), 
                    ascii = True)]        
            ((Pres_pi_5, Pres_rho_5, Pres_omega_5, Pres_K_5, Pres_D_5, Pres_N_5, Pres_T_5, Pres_F_5, Pres_P_5, 
                 Pres_Q5_5, Pres_H_5),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_5, mu_5, phi_re_5, phi_im_5)
        else:
            Pres_Q_5 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_5, mu_5, [1.0 for el in T_5], [0.0 for el in T_5]), 
                    desc = "Quark pressure (calc #5)", 
                    total = len(T_5), 
                    ascii = True)]
            Pres_pert_5 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_5, mu_5, [1.0 for el in T_5], [0.0 for el in T_5]), 
                    desc = "Perturbative pressure (calc #5)", 
                    total = len(T_5), 
                    ascii = True)]        
            ((Pres_pi_5, Pres_rho_5, Pres_omega_5, Pres_K_5, Pres_D_5, Pres_N_5, Pres_T_5, Pres_F_5, Pres_P_5, 
                 Pres_Q5_5, Pres_H_5),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_5, mu_5, [1.0 for el in T_5], [0.0 for el in T_5])
        with open(pressure_5_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_5, Pres_g_5, Pres_pert_5, Pres_pi_5, Pres_K_5, Pres_rho_5, Pres_omega_5, Pres_D_5, Pres_N_5, Pres_T_5, Pres_F_5, Pres_P_5, Pres_Q5_5, Pres_H_5, Pres_sea_5)])
    else:
        Pres_Q_5, Pres_g_5 = utils.data_collect(0, 1, pressure_5_file)
        Pres_pert_5, Pres_pi_5 = utils.data_collect(2, 3, pressure_5_file)
        Pres_K_5, _ = utils.data_collect(4, 4, pressure_5_file)
        Pres_rho_5, Pres_omega_5 = utils.data_collect(5, 6, pressure_5_file)
        Pres_D_5, Pres_N_5 = utils.data_collect(7, 8, pressure_5_file)
        Pres_T_5, Pres_F_5 = utils.data_collect(9, 10, pressure_5_file)
        Pres_P_5, Pres_Q5_5 = utils.data_collect(11, 12, pressure_5_file)
        Pres_H_5, Pres_sea_5 = utils.data_collect(13, 14, pressure_5_file)

    if calc_6:
        Pres_g_6 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_6, phi_re_6, phi_im_6), 
                desc = "Gluon pressure (calc #6)", 
                total = len(T_6),
                ascii = True)]
        Pres_sea_6 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(zip(T_6, mu_6), 
                desc = "Sigma mf pressure (calc #6)", 
                total = len(T_6),
                ascii = True)]
        if not pl_turned_off:
            Pres_Q_6 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_6, mu_6, phi_re_6, phi_im_6), 
                    desc = "Quark pressure (calc #6)", 
                    total = len(T_6), 
                    ascii = True)]
            Pres_pert_6 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_6, mu_6, phi_re_6, phi_im_6), 
                    desc = "Perturbative pressure (calc #6)", 
                    total = len(T_6), 
                    ascii = True)]        
            ((Pres_pi_6, Pres_rho_6, Pres_omega_6, Pres_K_6, Pres_D_6, Pres_N_6, Pres_T_6, Pres_F_6, Pres_P_6, 
                 Pres_Q5_6, Pres_H_6),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_6, mu_6, phi_re_6, phi_im_6)
        else:
            Pres_Q_6 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_6, mu_6, [1.0 for el in T_6], [0.0 for el in T_6]), 
                    desc = "Quark pressure (calc #6)", 
                    total = len(T_6), 
                    ascii = True)]
            Pres_pert_6 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_6, mu_6, [1.0 for el in T_6], [0.0 for el in T_6]), 
                    desc = "Perturbative pressure (calc #6)", 
                    total = len(T_6), 
                    ascii = True)]        
            ((Pres_pi_6, Pres_rho_6, Pres_omega_6, Pres_K_6, Pres_D_6, Pres_N_6, Pres_T_6, Pres_F_6, Pres_P_6, 
                 Pres_Q5_6, Pres_H_6),
                (_, _, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_6, mu_6, [1.0 for el in T_6], [0.0 for el in T_6])
        with open(pressure_6_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_6, Pres_g_6, Pres_pert_6, Pres_pi_6, Pres_K_6, Pres_rho_6, Pres_omega_6, Pres_D_6, Pres_N_6, Pres_T_6, Pres_F_6, Pres_P_6, Pres_Q5_6, Pres_H_6, Pres_sea_6)])
    else:
        Pres_Q_6, Pres_g_6 = utils.data_collect(0, 1, pressure_6_file)
        Pres_pert_6, Pres_pi_6 = utils.data_collect(2, 3, pressure_6_file)
        Pres_K_6, _ = utils.data_collect(4, 4, pressure_6_file)
        Pres_rho_6, Pres_omega_6 = utils.data_collect(5, 6, pressure_6_file)
        Pres_D_6, Pres_N_6 = utils.data_collect(7, 8, pressure_6_file)
        Pres_T_6, Pres_F_6 = utils.data_collect(9, 10, pressure_6_file)
        Pres_P_6, Pres_Q5_6 = utils.data_collect(11, 12, pressure_6_file)
        Pres_H_6, Pres_sea_6 = utils.data_collect(13, 14, pressure_6_file)

    contrib_q_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_Q_1)]
    contrib_g_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_g_1)]
    contrib_pert_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_pert_1)]
    contrib_sea_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_1)]
    contrib_qgp_1 = [sum(el) for el in zip(contrib_q_1, contrib_g_1, contrib_pert_1, contrib_sea_1)]

    contrib_q_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_Q_3)]
    contrib_g_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_g_3)]
    contrib_pert_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_pert_3)]
    contrib_sea_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_3)]
    contrib_qgp_3 = [sum(el) for el in zip(contrib_q_3, contrib_g_3, contrib_pert_3, contrib_sea_3)]

    contrib_q_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_Q_4)]
    contrib_g_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_g_4)]
    contrib_pert_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_pert_4)]
    contrib_sea_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_4)]
    contrib_qgp_4 = [sum(el) for el in zip(contrib_q_4, contrib_g_4, contrib_pert_4, contrib_sea_4)]

    contrib_q_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_Q_5)]
    contrib_g_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_g_5)]
    contrib_pert_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_pert_5)]
    contrib_sea_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_5)]
    contrib_qgp_5 = [sum(el) for el in zip(contrib_q_5, contrib_g_5, contrib_pert_5, contrib_sea_5)]

    contrib_q_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_Q_6)]
    contrib_g_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_g_6)]
    contrib_pert_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_pert_6)]
    contrib_sea_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_6)]
    contrib_qgp_6 = [sum(el) for el in zip(contrib_q_6, contrib_g_6, contrib_pert_6, contrib_sea_6)]

    contrib_pi_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_pi_1)]
    contrib_rho_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_rho_1)]
    contrib_omega_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_omega_1)]
    contrib_K_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_K_1)]
    contrib_D_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_D_1)]
    contrib_N_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_N_1)]
    contrib_T_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_T_1)]
    contrib_F_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_F_1)]
    contrib_P_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_P_1)]
    contrib_Q5_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_Q5_1)]
    contrib_H_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_H_1)]
    contrib_cluster_1 = [sum(el) for el in zip(contrib_pi_1, contrib_rho_1, contrib_omega_1, contrib_K_1, contrib_D_1, contrib_N_1, contrib_T_1, contrib_F_1, contrib_P_1, contrib_Q5_1, contrib_H_1)]
    contrib_cluster_singlet_1 = [sum(el) for el in zip(contrib_pi_1, contrib_rho_1, contrib_omega_1, contrib_K_1, contrib_N_1, contrib_T_1, contrib_P_1, contrib_H_1)]
    contrib_cluster_color_1 = [sum(el) for el in zip(contrib_D_1, contrib_F_1, contrib_Q5_1)]
    contrib_total_1 = [sum(el) for el in zip(contrib_cluster_1, contrib_qgp_1)]

    contrib_pi_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_pi_3)]
    contrib_rho_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_rho_3)]
    contrib_omega_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_omega_3)]
    contrib_K_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_K_3)]
    contrib_D_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_D_3)]
    contrib_N_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_N_3)]
    contrib_T_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_T_3)]
    contrib_F_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_F_3)]
    contrib_P_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_P_3)]
    contrib_Q5_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_Q5_3)]
    contrib_H_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_H_3)]
    contrib_cluster_3 = [sum(el) for el in zip(contrib_pi_3, contrib_rho_3, contrib_omega_3, contrib_K_3, contrib_D_3, contrib_N_3, contrib_T_3, contrib_F_3, contrib_P_3, contrib_Q5_3, contrib_H_3)]
    contrib_cluster_singlet_3 = [sum(el) for el in zip(contrib_pi_3, contrib_rho_3, contrib_omega_3, contrib_K_3, contrib_N_3, contrib_T_3, contrib_P_3, contrib_H_3)]
    contrib_cluster_color_3 = [sum(el) for el in zip(contrib_D_3, contrib_F_3, contrib_Q5_3)]
    contrib_total_3 = [sum(el) for el in zip(contrib_cluster_3, contrib_qgp_3)]

    contrib_pi_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_pi_4)]
    contrib_rho_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_rho_4)]
    contrib_omega_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_omega_4)]
    contrib_K_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_K_4)]
    contrib_D_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_D_4)]
    contrib_N_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_N_4)]
    contrib_T_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_T_4)]
    contrib_F_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_F_4)]
    contrib_P_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_P_4)]
    contrib_Q5_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_Q5_4)]
    contrib_H_4 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_4, Pres_H_4)]
    contrib_cluster_4 = [sum(el) for el in zip(contrib_pi_4, contrib_rho_4, contrib_omega_4, contrib_K_4, contrib_D_4, contrib_N_4, contrib_T_4, contrib_F_4, contrib_P_4, contrib_Q5_4, contrib_H_4)]
    contrib_cluster_singlet_4 = [sum(el) for el in zip(contrib_pi_4, contrib_rho_4, contrib_omega_4, contrib_K_4, contrib_N_4, contrib_T_4, contrib_P_4, contrib_H_4)]
    contrib_cluster_color_4 = [sum(el) for el in zip(contrib_D_4, contrib_F_4, contrib_Q5_4)]
    contrib_total_4 = [sum(el) for el in zip(contrib_cluster_4, contrib_qgp_4)]

    contrib_pi_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_pi_5)]
    contrib_rho_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_rho_5)]
    contrib_omega_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_omega_5)]
    contrib_K_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_K_5)]
    contrib_D_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_D_5)]
    contrib_N_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_N_5)]
    contrib_T_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_T_5)]
    contrib_F_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_F_5)]
    contrib_P_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_P_5)]
    contrib_Q5_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_Q5_5)]
    contrib_H_5 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_5, Pres_H_5)]
    contrib_cluster_5 = [sum(el) for el in zip(contrib_pi_5, contrib_rho_5, contrib_omega_5, contrib_K_5, contrib_D_5, contrib_N_5, contrib_T_5, contrib_F_5, contrib_P_5, contrib_Q5_5, contrib_H_5)]
    contrib_cluster_singlet_5 = [sum(el) for el in zip(contrib_pi_5, contrib_rho_5, contrib_omega_5, contrib_K_5, contrib_N_5, contrib_T_5, contrib_P_5, contrib_H_5)]
    contrib_cluster_color_3 = [sum(el) for el in zip(contrib_D_5, contrib_F_5, contrib_Q5_5)]
    contrib_total_5 = [sum(el) for el in zip(contrib_cluster_5, contrib_qgp_5)]

    contrib_pi_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_pi_6)]
    contrib_rho_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_rho_6)]
    contrib_omega_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_omega_6)]
    contrib_K_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_K_6)]
    contrib_D_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_D_6)]
    contrib_N_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_N_6)]
    contrib_T_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_T_6)]
    contrib_F_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_F_6)]
    contrib_P_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_P_6)]
    contrib_Q5_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_Q5_6)]
    contrib_H_6 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_6, Pres_H_6)]
    contrib_cluster_6 = [sum(el) for el in zip(contrib_pi_6, contrib_rho_6, contrib_omega_6, contrib_K_6, contrib_D_6, contrib_N_6, contrib_T_6, contrib_F_6, contrib_P_6, contrib_Q5_6, contrib_H_6)]
    contrib_cluster_singlet_6 = [sum(el) for el in zip(contrib_pi_6, contrib_rho_6, contrib_omega_6, contrib_K_6, contrib_N_6, contrib_T_6, contrib_P_6, contrib_H_6)]
    contrib_cluster_color_6 = [sum(el) for el in zip(contrib_D_6, contrib_F_6, contrib_Q5_6)]
    contrib_total_6 = [sum(el) for el in zip(contrib_cluster_6, contrib_qgp_6)]

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1204_6710v2_table4_pressure_mu0_high.dat")
    (low_1204_6710v2_mu100_x, low_1204_6710v2_mu100_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1204_6710v2_table4_pressure_mu100_low.dat")
    (high_1204_6710v2_mu100_x, high_1204_6710v2_mu100_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1204_6710v2_table4_pressure_mu100_high.dat")
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1204_6710v2_table4_pressure_mu200_high.dat")
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1204_6710v2_table4_pressure_mu300_low.dat")
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1204_6710v2_table4_pressure_mu300_high.dat")
    (low_1204_6710v2_mu400_x, low_1204_6710v2_mu400_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1204_6710v2_table4_pressure_mu400_low.dat")
    (high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1204_6710v2_table4_pressure_mu400_high.dat")
    (high_1407_6387_mu0_x, high_1407_6387_mu0_y) = utils.data_collect(0, 3, "D:/EoS/epja/figure4/1407_6387_table1_pressure_mu0.dat")
    (low_1407_6387_mu0_x, low_1407_6387_mu0_y) = utils.data_collect(0, 2, "D:/EoS/epja/figure4/1407_6387_table1_pressure_mu0.dat")
    (high_1309_5258_mu0_x, high_1309_5258_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1309_5258_figure6_pressure_mu0_high.dat")
    (low_1309_5258_mu0_x, low_1309_5258_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1309_5258_figure6_pressure_mu0_low.dat")
    (high_1710_05024_mu0_x, high_1710_05024_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1710_05024_figure8_pressure_mu0_high.dat")
    (low_1710_05024_mu0_x, low_1710_05024_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure4/1710_05024_figure8_pressure_mu0_low.dat")

    borsanyi_1204_6710v2_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = numpy.array(borsanyi_1204_6710v2_mu0)
    borsanyi_1204_6710v2_mu100 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu100_x, high_1204_6710v2_mu100_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu100_x[::-1], low_1204_6710v2_mu100_y[::-1]):
        borsanyi_1204_6710v2_mu100.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu100 = numpy.array(borsanyi_1204_6710v2_mu100)
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

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (10.5, 10))
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.axis([10., 540., 0., 4.2])
    ax1.add_patch(matplotlib.patches.Polygon(bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Bazavov et al. (2014)'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2014)'))
    ax1.add_patch(matplotlib.patches.Polygon(bazavov_1710_05024_mu0, closed = True, fill = True, color = 'magenta', alpha = 0.3, label = r'Bazavov et al. (2018)'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012)'))
    ax1.plot(T_1, contrib_total_1, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    ax1.plot(T_1, contrib_cluster_1, '--', c = 'red', label = r'BU')
    ax1.plot(T_1, contrib_qgp_1, '--', c = 'blue', label = r'PNJL')
    ax1.text(180.0, 0.1, r'Clusters', color = 'red', fontsize = 14)
    ax1.text(170.0, 0.58, r'PNJL', color = 'blue', fontsize = 14)
    ax1.text(195.0, 1.16, r'total pressure', color = 'black', fontsize = 14)
    ax1.text(21.0, 3.9, r'$\mathrm{\mu_B=0}$', color = 'black', fontsize = 14)
    ax1.text(228.0, 1.8, r'Borsanyi et al. (2012)', color = 'blue', alpha = 0.7, fontsize = 14)
    ax1.text(175.0, 3.95, r'Borsanyi et al. (2014)', color = 'green', alpha = 0.7, fontsize = 14)
    ax1.text(30.0, 3.4, r'Bazavov et al. (2014)', color = 'red', alpha = 0.7, fontsize = 14)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.axis([10., 2000., 0., 5.0])
    ax2.add_patch(matplotlib.patches.Polygon(bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Bazavov et al. (2014)'))
    ax2.add_patch(matplotlib.patches.Polygon(borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2014)'))
    ax2.add_patch(matplotlib.patches.Polygon(bazavov_1710_05024_mu0, closed = True, fill = True, color = 'magenta', alpha = 0.3, label = r'Bazavov et al. (2018)'))
    ax2.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012)'))
    ax2.plot(T_1, contrib_total_1, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    ax2.plot(T_1, contrib_cluster_1, '--', c = 'red', label = r'BU')
    ax2.plot(T_1, contrib_qgp_1, '--', c = 'blue', label = r'PNJL')
    ax2.text(220.0, 0.1, r'Clusters', color = 'red', fontsize = 14)
    ax2.text(210.0, 0.58, r'PNJL', color = 'blue', fontsize = 14)
    ax2.text(220.0, 1.16, r'total pressure', color = 'black', fontsize = 14)
    ax2.text(65.0, 4.65, r'$\mathrm{\mu_B=0}$', color = 'black', fontsize = 14)
    ax2.text(260.0, 2.0, r'Borsanyi et al. (2012)', color = 'blue', alpha = 0.7, fontsize = 14)
    ax2.text(435.0, 3.28, r'Borsanyi et al. (2014)', color = 'green', alpha = 0.7, fontsize = 14)
    ax2.text(350.0, 2.66, r'Bazavov et al. (2014)', color = 'red', alpha = 0.7, fontsize = 14)
    ax2.text(850.0, 4.15, r'Bazavov et al. (2018)', color = 'magenta', alpha = 0.7, fontsize = 14)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    ax4 = fig1.add_subplot(2, 2, 3)
    ax4.axis([10., 540., 0., 4.2])
    ax4.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=200}$ MeV'))
    ax4.plot(T_5, contrib_total_5, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=200}$')
    ax4.plot(T_5, contrib_cluster_5, '--', c = 'red', label = r'BU')
    ax4.plot(T_5, contrib_qgp_5, '--', c = 'blue', label = r'PNJL')
    ax4.text(180.0, 0.1, r'Clusters', color = 'red', fontsize = 14)
    ax4.text(170.0, 0.58, r'PNJL', color = 'blue', fontsize = 14)
    ax4.text(195.0, 1.16, r'total pressure', color = 'black', fontsize = 14)
    ax4.text(21.0, 3.9, r'$\mathrm{\mu_B=200}$ MeV', color = 'black', fontsize = 14)
    ax4.text(228.0, 1.8, r'Borsanyi et al. (2012)', color = 'blue', alpha = 0.7, fontsize = 14)
    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax4.set_xlabel(r'T [MeV]', fontsize = 16)
    ax4.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    ax6 = fig1.add_subplot(2, 2, 4)
    ax6.axis([10., 540., 0., 4.2])
    ax6.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu400, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012), $\mathrm{\mu=400}$ MeV'))
    ax6.plot(T_3, contrib_total_3, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=400}$')
    ax6.plot(T_3, contrib_cluster_3, '--', c = 'red', label = r'BU')
    ax6.plot(T_3, contrib_qgp_3, '--', c = 'blue', label = r'PNJL')
    ax6.text(180.0, 0.1, r'Clusters', color = 'red', fontsize = 14)
    ax6.text(170.0, 0.58, r'PNJL', color = 'blue', fontsize = 14)
    ax6.text(195.0, 1.16, r'total pressure', color = 'black', fontsize = 14)
    ax6.text(21.0, 3.9, r'$\mathrm{\mu_B=400}$ MeV', color = 'black', fontsize = 14)
    ax6.text(205.0, 1.8, r'Borsanyi et al. (2012)', color = 'blue', alpha = 0.7, fontsize = 14)
    for tick in ax6.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax6.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax6.set_xlabel(r'T [MeV]', fontsize = 16)
    ax6.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def epja_figure6():
    col_n = '#DEA54B'
    col_pi = '#653239'
    col_rho = '#858AE3'
    col_omega = '#FF37A6'
    col_p = '#78BC61'
    col_t = '#23CE6B'
    col_h = '#A846A0'
    col_d = '#4CB944'
    col_f = '#DB222A'
    col_q5 = '#55DBCB'
    col_qgp = 'blue'
    col_pert = 'red'
    col_gluon = 'pink'
    col_pnjl = 'magenta'
    col_total = 'black'

    const_mu = 0.0

    T = numpy.linspace(1.0, 200.0, num = 200)

    fig = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([min(T), max(T), 0.0, 1.05 * max([6.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(el, const_mu) for el in T])])

    ax.plot(T, [pnjl.defaults.default_MN for el in T]                            , '-'   , c = col_n     , label = r'nucleon')
    ax.add_patch(matplotlib.patches.FancyArrowPatch((30., 1750.), (30., 1000.), arrowstyle='<->', mutation_scale=20, color = col_n))
    ax.plot(T, [3.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(el, const_mu) for el in T]  , '--'  , c = col_n)
    ax.text(35., 1320., r'nucleon', fontsize = 14)

    ax.plot(T, [140 for el in T]                                   , '-'   , c = col_pi    , label = r'$\mathrm{\pi}$')
    ax.add_patch(matplotlib.patches.FancyArrowPatch((50., 1150.), (50., 135.), arrowstyle='<->', mutation_scale=20, color = col_pi))
    ax.plot(T, [2.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(el, const_mu) for el in T]  , '--'  , c = col_pi)
    ax.text(55., 620., r'pion', fontsize = 14)

    ax.plot(T, [pnjl.defaults.default_MH for el in T]                            , '-'   , c = col_h     , label = r'hexaquark')
    ax.add_patch(matplotlib.patches.FancyArrowPatch((10., 3450.), (10., 1900.), arrowstyle='<->', mutation_scale=20, color = col_h))
    ax.plot(T, [6.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(el, const_mu) for el in T]  , '--'  , c = col_h)
    ax.text(15., 2650., r'hexaquark', fontsize = 14)

    ax.plot(T, [pnjl.thermo.gcp_sea_lattice.M(el, const_mu) for el in T], '-', c = col_qgp, label = r'u/d quark')
    ax.text(5., 480., r'u/d quark', fontsize = 10)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax.set_xlabel(r'T [MeV]', fontsize = 16)
    ax.set_ylabel(r'mass [MeV]', fontsize = 16)

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def epja_figure7():

    T0 = numpy.linspace(1.0, 250.0, num = 200)

    sigma_0 = [pnjl.thermo.gcp_sea_lattice.Delta_ls(el, 0.0) for el in T0]

    (low_1005_3508_x, low_1005_3508_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure6/1005_3508_table3_delta.dat")
    (high_1005_3508_x, high_1005_3508_y) = utils.data_collect(0, 2, "D:/EoS/epja/figure6/1005_3508_table3_delta.dat")

    borsanyi_1005_3508 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1005_3508_x, high_1005_3508_y)]
    for x_el, y_el in zip(low_1005_3508_x[::-1], low_1005_3508_y[::-1]):
        borsanyi_1005_3508.append(numpy.array([x_el, y_el]))
    borsanyi_1005_3508 = numpy.array(borsanyi_1005_3508)

    fig = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([100, 220, 0.0, 1.1])
    ax.add_patch(matplotlib.patches.Polygon(borsanyi_1005_3508, closed = True, fill = True, color = 'blue', alpha = 0.3))
    ax.plot(T0, sigma_0, c = 'green')
    ax.text(200, 1, r'$\mathrm{\mu=0}$', fontsize = 14)
    ax.text(155, 0.55, r'Borsanyi et al. (2010)', color = 'blue', alpha = 0.7, fontsize = 14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax.set_xlabel(r'T [MeV]', fontsize = 16)
    ax.set_ylabel(r'$\mathrm{\Delta_l(T,\mu)}$', fontsize = 16)

    fig.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def epja_figure8():

    def phase(M, T, Mi, Mthi, Ni, Lambda_i, Lambda2_i, **kwargs):

        options = {'kappa' : pnjl.defaults.default_kappa, 'Tc0' : pnjl.defaults.default_Tc0, 'delta_T' : pnjl.defaults.default_delta_T, 'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(kwargs)

        Tc0 = options['Tc0']
        kappa = options['kappa']
        delta_T = options['delta_T']
        M0 = options['M0']
        ml = options['ml']

        mu = 0.0

        hz = 0.5
        nlambda = Ni * Lambda_i
        nlambda2 = Ni * Lambda2_i
        
        M_gap = (Ni * M0 + Ni * ml) - Mi

        T_slope = 10.0

        T_trans = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * Ni - 2.0 * Mi) / (M0 * Ni))) / Tc0

        heavi2 = numpy.heaviside((M ** 2) - (Mi ** 2), hz)
        heavi3 = numpy.heaviside((M ** 2) - (Mthi ** 2), hz)
        heavi4 = numpy.heaviside((Ni * M0 + Ni * ml) + nlambda - M, hz)
        arccos_in = -2.0 * (M / (Mthi - M0 * Ni - ml * Ni - nlambda)) + ((Mthi + M0 * Ni + ml * Ni + nlambda) / (Mthi - M0 * Ni - ml * Ni - nlambda))

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (T - T_trans), hz)
        heavi7 = numpy.heaviside(math.pi * (M - Mthi), hz)
        heavi8 = numpy.heaviside(math.pi * (((Mi + T_slope * (T - T_trans) + M_gap) / nlambda) + 1.0 - (M / nlambda)), hz)
        arccos_in2 = (2.0 * M / (Mi - Mthi + nlambda + T_slope * (T - T_trans) + M_gap)) + ((Mi + Mthi + nlambda + T_slope * (T - T_trans) + M_gap) / (Mthi - Mi - nlambda - T_slope * (T - T_trans) - M_gap))
        cos_in = math.pi * arccos_in2 / 2.0
        factor = 1.0 * ((nlambda2 - T_slope * (T - T_trans)) / nlambda2)
        factor2 = 1.0 * ((nlambda - T_slope * (T - T_trans)) / nlambda)

        first = 0.0
        second = 0.0
        third = 0.0

        if Mthi >= Mi:
            if M >= Mi and M <= Mthi:
                first = (heavi2 - heavi3)
            if M >= Mthi and M <= (Ni * M0 + Ni * ml) + nlambda:
                arccos_el = math.acos(arccos_in)
                second = heavi3 * heavi4 * arccos_el / math.pi
        else:
            if M >= Mthi and M <= Mi + T_slope * (T - T_trans) + nlambda + M_gap:
                arccos_el2 = math.acos(arccos_in2)
                if factor2 > 0.0:
                    third = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
                else:
                    third = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        
        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(3.0 * T / 2.0, T, mu, M, Ni, 1.0)

        return (first + second + third) #* pnjl.thermo.distributions.f_boson_singlet(**yp)

    Mi_val = pnjl.defaults.default_MH
    Ni_val = 6.0
    Lam_val = pnjl.defaults.default_L
    Lam2_val = 5.8 * pnjl.defaults.default_L

    T_list = numpy.linspace(120, 220, num = 17)

    T_val1, T_val2, T_val3, T_val4, T_val5, T_val6, T_val7, T_val8, T_val9, T_val10, T_val11, T_val12, T_val13, T_val14, T_val15, T_val16, T_val17 = T_list

    Mthi_val1 = pnjl.thermo.gcp_sea_lattice.M(T_val1, 0.0) * Ni_val
    Mthi_val2 = pnjl.thermo.gcp_sea_lattice.M(T_val2, 0.0) * Ni_val
    Mthi_val3 = pnjl.thermo.gcp_sea_lattice.M(T_val3, 0.0) * Ni_val
    Mthi_val4 = pnjl.thermo.gcp_sea_lattice.M(T_val4, 0.0) * Ni_val
    Mthi_val5 = pnjl.thermo.gcp_sea_lattice.M(T_val5, 0.0) * Ni_val
    Mthi_val6 = pnjl.thermo.gcp_sea_lattice.M(T_val6, 0.0) * Ni_val
    Mthi_val7 = pnjl.thermo.gcp_sea_lattice.M(T_val7, 0.0) * Ni_val
    Mthi_val8 = pnjl.thermo.gcp_sea_lattice.M(T_val8, 0.0) * Ni_val
    Mthi_val9 = pnjl.thermo.gcp_sea_lattice.M(T_val9, 0.0) * Ni_val
    Mthi_val10 = pnjl.thermo.gcp_sea_lattice.M(T_val10, 0.0) * Ni_val
    Mthi_val11 = pnjl.thermo.gcp_sea_lattice.M(T_val11, 0.0) * Ni_val
    Mthi_val12 = pnjl.thermo.gcp_sea_lattice.M(T_val12, 0.0) * Ni_val
    Mthi_val13 = pnjl.thermo.gcp_sea_lattice.M(T_val13, 0.0) * Ni_val
    Mthi_val14 = pnjl.thermo.gcp_sea_lattice.M(T_val14, 0.0) * Ni_val
    Mthi_val15 = pnjl.thermo.gcp_sea_lattice.M(T_val15, 0.0) * Ni_val
    Mthi_val16 = pnjl.thermo.gcp_sea_lattice.M(T_val16, 0.0) * Ni_val
    Mthi_val17 = pnjl.thermo.gcp_sea_lattice.M(T_val17, 0.0) * Ni_val

    Mthi_vec = [Mthi_val1, Mthi_val2, Mthi_val3, Mthi_val4, Mthi_val5, Mthi_val6, Mthi_val7, Mthi_val8, Mthi_val9, Mthi_val10, Mthi_val11, Mthi_val12, Mthi_val13, Mthi_val14, Mthi_val15, Mthi_val16, Mthi_val17]

    M_gap = (Ni_val * pnjl.defaults.default_M0 + Ni_val * pnjl.defaults.default_ml) - Mi_val
    T_critical = ((pnjl.defaults.default_Tc0 ** 2) - pnjl.defaults.default_kappa * (0.0 ** 2) + pnjl.defaults.default_Tc0 * pnjl.defaults.default_delta_T * math.atanh(((pnjl.defaults.default_M0 + 2.0 * pnjl.defaults.default_ml) * Ni_val - 2.0 * Mi_val) / (pnjl.defaults.default_M0 * Ni_val))) / pnjl.defaults.default_Tc0
    redline = [Mi_val + M_gap + Ni_val * Lam_val if Mthi_el > Mi_val else Mi_val + M_gap + Ni_val * Lam_val + 10.0 * (T_el - T_critical) for Mthi_el, T_el in zip(Mthi_vec, T_list)]

    M_vec_min = 0.0
    M_vec_max = 5500.0
    M_vec = numpy.linspace(M_vec_min, M_vec_max, 2000)
    phase_vec1 = [
        phase(el, T_val1, Mi_val, Mthi_val1, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec2 = [
        phase(el, T_val2, Mi_val, Mthi_val2, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec3 = [
        phase(el, T_val3, Mi_val, Mthi_val3, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec4 = [
        phase(el, T_val4, Mi_val, Mthi_val4, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec5 = [
        phase(el, T_val5, Mi_val, Mthi_val5, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec6 = [
        phase(el, T_val6, Mi_val, Mthi_val6, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec7 = [
        phase(el, T_val7, Mi_val, Mthi_val7, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec8 = [
        phase(el, T_val8, Mi_val, Mthi_val8, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec9 = [
        phase(el, T_val9, Mi_val, Mthi_val9, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec10 = [
        phase(el, T_val10, Mi_val, Mthi_val10, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec11 = [
        phase(el, T_val11, Mi_val, Mthi_val11, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec12 = [
        phase(el, T_val12, Mi_val, Mthi_val12, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec13 = [
        phase(el, T_val13, Mi_val, Mthi_val13, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec14 = [
        phase(el, T_val14, Mi_val, Mthi_val14, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec15 = [
        phase(el, T_val15, Mi_val, Mthi_val15, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec16 = [
        phase(el, T_val16, Mi_val, Mthi_val16, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]
    phase_vec17 = [
        phase(el, T_val17, Mi_val, Mthi_val17, Ni_val, Lam_val, Lam2_val)
        for el in M_vec]

    if numpy.any([el > 1.0 for el in phase_vec1]):
        print("phase_vec1 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec2]):
        print("phase_vec2 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec3]):
        print("phase_vec3 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec4]):
        print("phase_vec4 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec5]):
        print("phase_vec5 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec6]):
        print("phase_vec6 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec7]):
        print("phase_vec7 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec8]):
        print("phase_vec8 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec9]):
        print("phase_vec9 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec10]):
        print("phase_vec10 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec11]):
        print("phase_vec11 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec12]):
        print("phase_vec12 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec13]):
        print("phase_vec13 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec14]):
        print("phase_vec14 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec15]):
        print("phase_vec15 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec16]):
        print("phase_vec16 exceeds 1!")
    if numpy.any([el > 1.0 for el in phase_vec17]):
        print("phase_vec17 exceeds 1!")

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(111, projection='3d')
    fig1.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
    ax1.set_ylim3d(T_val17, T_val1 - 1.0)
    ax1.set_xlim3d(M_vec_min, M_vec_max)
    #ax1.set_zlim3d(0, 1)
    ax1.plot3D(M_vec, [T_val1 for el in M_vec], phase_vec1, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val2 for el in M_vec], phase_vec2, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val3 for el in M_vec], phase_vec3, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val4 for el in M_vec], phase_vec4, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val5 for el in M_vec], phase_vec5, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val6 for el in M_vec], phase_vec6, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val7 for el in M_vec], phase_vec7, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val8 for el in M_vec], phase_vec8, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val9 for el in M_vec], phase_vec9, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val10 for el in M_vec], phase_vec10, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val11 for el in M_vec], phase_vec11, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val12 for el in M_vec], phase_vec12, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val13 for el in M_vec], phase_vec13, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val14 for el in M_vec], phase_vec14, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val15 for el in M_vec], phase_vec15, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val16 for el in M_vec], phase_vec16, '-', c = 'black')
    ax1.plot3D(M_vec, [T_val17 for el in M_vec], phase_vec17, '-', c = 'black')
    ax1.plot3D([Mi_val if Mi_val < pnjl.thermo.gcp_sea_lattice.M(T_el, 0.0) * Ni_val else Mi_val + 5.0 * T_el - 5.0 * ((pnjl.defaults.default_Tc0 ** 2) - pnjl.defaults.default_kappa * (0.0 ** 2) + pnjl.defaults.default_Tc0 * pnjl.defaults.default_delta_T * math.atanh(((pnjl.defaults.default_M0 + 2.0 * pnjl.defaults.default_ml) * Ni_val - 2.0 * Mi_val) / (pnjl.defaults.default_M0 * Ni_val))) / pnjl.defaults.default_Tc0 for M_el, T_el in zip(M_vec, numpy.linspace(T_val17, T_val1, num = len(M_vec)))], numpy.linspace(T_val17, T_val1, num = len(M_vec)), [0.0 for el in M_vec], '--', c = 'blue')
    ax1.plot3D([Mthi_val1, Mthi_val2, Mthi_val3, Mthi_val4, Mthi_val5, Mthi_val6, Mthi_val7, Mthi_val8, Mthi_val9, Mthi_val10, Mthi_val11, Mthi_val12, Mthi_val13, Mthi_val14, Mthi_val15, Mthi_val16, Mthi_val17], [T_val1, T_val2, T_val3, T_val4, T_val5, T_val6, T_val7, T_val8, T_val9, T_val10, T_val11, T_val12, T_val13, T_val14, T_val15, T_val16, T_val17], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], '--', c = 'green')
    #ax1.plot3D([Mthi_val1, Mthi_val1], [T_val1, T_val1], [0, 1], '--', c =
    #'green')
    #ax1.plot3D([Mthi_val2, Mthi_val2], [T_val2, T_val2], [0, 1], '--', c =
    #'green')
    #ax1.plot3D([Mthi_val3, Mthi_val3], [T_val3, T_val3], [0, 1], '--', c =
    #'green')
    #ax1.plot3D([Mi_val, Mi_val], [T_val1, T_val1], [0, 1], '--', c = 'blue')
    #ax1.plot3D([Mi_val, Mi_val], [T_val2, T_val2], [0, 1], '--', c = 'blue')
    #ax1.plot3D([Mi_val, Mi_val], [T_val3, T_val3], [0, 1], '--', c = 'blue')
    ax1.plot3D(redline, T_list, [0.0 for el in T_list], '--', c = 'red')
    ax1.text(400, 300, 0.065, r'$\mathrm{M_{thr,i}}$', color = 'green', fontsize = 16, bbox = dict(color = 'white', boxstyle = 'square, pad=-0.5'))
    ax1.text(1500, 300, 0.08, r'$\mathrm{M_i}$', color = 'blue', fontsize = 16, bbox = dict(color = 'white', boxstyle = 'square, pad=0.0'))
    ax1.text(2200, 300, 0.08, r'$\mathrm{M_{c,i}}$', color = 'red', fontsize = 16, bbox = dict(color = 'white', boxstyle = 'square, pad=0.0'))
    ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.set_xticklabels([0, 0.5, 1, 1.5, 2, 2.5])
    #ax1.set_yticks([125, 175, 225, 275])
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax1.zaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.tick_params(axis='x', which='major', pad=-3)
    ax1.tick_params(axis='y', which='major', pad=-2)
    ax1.set_xlabel(r'M [GeV]', fontsize = 16)
    ax1.set_ylabel(r'T [MeV]', fontsize = 16)
    ax1.set_zlabel(r'$\delta_i$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    #fig2.tight_layout(pad = 0.1)
    #fig3.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def epja_figure9():

    phi_re_1, phi_im_1 = [], []
    phi_re_3, phi_im_3 = [], []

    Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_sea_1 = [], [], [], []
    Pres_Q_3, Pres_g_3, Pres_pert_3, Pres_sea_3 = [], [], [], []

    Pres_pi_1, Pres_K_1, Pres_rho_1 = [], [], []
    Pres_omega_1, Pres_D_1, Pres_N_1 = [], [], []
    Pres_T_1, Pres_F_1, Pres_P_1 = [], [], []
    Pres_Q5_1, Pres_H_1 = [], []

    Pres_pi_3, Pres_K_3, Pres_rho_3 = [], [], [] 
    Pres_omega_3, Pres_D_3, Pres_N_3 = [], [], []
    Pres_T_3, Pres_F_3, Pres_P_3 = [], [], []
    Pres_Q5_3, Pres_H_3 = [], []

    T_1 = numpy.linspace(1.0, 400.0, 200)
    T_3 = numpy.linspace(1.0, 400.0, 200)

    mu_1 = [0.0 / 3.0 for el in T_1]
    mu_3 = [0.0 / 3.0 for el in T_3]

    calc_1 = False
    calc_3 = False

    cluster_backreaction = False

    pl_1_file = "D:/EoS/epja/figure8/pl_1.dat"
    pl_3_file = "D:/EoS/epja/figure8/pl_3.dat"
    pressure_1_file = "D:/EoS/epja/figure8/pressure_1.dat"
    pressure_3_file = "D:/EoS/epja/figure8/pressure_3.dat"

    if False:
        phi_re_1.append(1e-15)
        phi_im_1.append(2e-15)
        lT = len(T_1)
        for T_el, mu_el in tqdm.tqdm(zip(T_1, mu_1), desc = "Traced Polyakov loop (calc #1)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_1[-1], phi_im_1[-1], with_clusters = cluster_backreaction)
            phi_re_1.append(temp_phi_re)
            phi_im_1.append(temp_phi_im)
        phi_re_1 = phi_re_1[1:]
        phi_im_1 = phi_im_1[1:]
        with open(pl_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_1, mu_1, phi_re_1, phi_im_1)])
    else:
        T_1, mu_1 = utils.data_collect(0, 1, pl_1_file)
        phi_re_1, phi_im_1 = utils.data_collect(2, 3, pl_1_file)

    if False:
        phi_re_3.append(1e-15)
        phi_im_3.append(2e-15)
        lT = len(T_3)
        for T_el, mu_el in tqdm.tqdm(zip(T_3, mu_3), desc = "Traced Polyakov loop (calc #3)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_3[-1], phi_im_3[-1], with_clusters = cluster_backreaction)
            phi_re_3.append(temp_phi_re)
            phi_im_3.append(temp_phi_im)
        phi_re_3 = phi_re_3[1:]
        phi_im_3 = phi_im_3[1:]
        with open(pl_3_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_3, mu_3, phi_re_3, phi_im_3)])
    else:
        T_3, mu_3 = utils.data_collect(0, 1, pl_3_file)
        phi_re_3, phi_im_3 = utils.data_collect(2, 3, pl_3_file)

    if calc_1:
        Pres_g_1 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_1, phi_re_1, phi_im_1), 
                desc = "Gluon pressure (calc #1)", 
                total = len(T_1),
                ascii = True)]
        Pres_sea_1 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(zip(T_1, mu_1), 
                desc = "Sigma mf pressure (calc #1)", 
                total = len(T_1),
                ascii = True)]
        Pres_Q_1 = [
            pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
            for T_el, mu_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                desc = "Quark pressure (calc #1)", 
                total = len(T_1), 
                ascii = True)]
        Pres_pert_1 = [
            pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
            for T_el, mu_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                desc = "Perturbative pressure (calc #1)", 
                total = len(T_1), 
                ascii = True)]
        (regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2,
         regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2,
         regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52) = cluster_regulator_calc(mu_1[0])
        ((Pres_pi_1, Pres_rho_1, Pres_omega_1, Pres_K_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, 
             Pres_Q5_1, Pres_H_1),
            (_, _, _, _, _, _, _, _, _, _, _),
            (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_1, mu_1, phi_re_1, phi_im_1, regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2, regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2, regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52)
        with open(pressure_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_pi_1, Pres_K_1, Pres_rho_1, Pres_omega_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, Pres_Q5_1, Pres_H_1, Pres_sea_1)])
    else:
        Pres_Q_1, Pres_g_1 = utils.data_collect(0, 1, pressure_1_file)
        Pres_pert_1, Pres_pi_1 = utils.data_collect(2, 3, pressure_1_file)
        Pres_K_1, _ = utils.data_collect(4, 4, pressure_1_file)
        Pres_rho_1, Pres_omega_1 = utils.data_collect(5, 6, pressure_1_file)
        Pres_D_1, Pres_N_1 = utils.data_collect(7, 8, pressure_1_file)
        Pres_T_1, Pres_F_1 = utils.data_collect(9, 10, pressure_1_file)
        Pres_P_1, Pres_Q5_1 = utils.data_collect(11, 12, pressure_1_file)
        Pres_H_1, Pres_sea_1 = utils.data_collect(13, 14, pressure_1_file)

    if calc_3:
        Pres_g_3 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_3, phi_re_3, phi_im_3), 
                desc = "Gluon pressure (calc #3)", 
                total = len(T_3),
                ascii = True)]
        Pres_sea_3 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(zip(T_3, mu_3), 
                desc = "Sigma mf pressure (calc #3)", 
                total = len(T_3),
                ascii = True)]
        Pres_Q_3 = [
            pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
            for T_el, mu_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3]), 
                desc = "Quark pressure (calc #3)", 
                total = len(T_3), 
                ascii = True)]
        Pres_pert_3 = [
            pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
            for T_el, mu_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3]), 
                desc = "Perturbative pressure (calc #3)", 
                total = len(T_3), 
                ascii = True)]
        (regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2,
         regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2,
         regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52) = cluster_regulator_calc_nophi(mu_1[0])
        ((Pres_pi_3, Pres_rho_3, Pres_omega_3, Pres_K_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, 
             Pres_Q5_3, Pres_H_3),
            (_, _, _, _, _, _, _, _, _, _, _),
            (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3], regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2, regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2, regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52)
        with open(pressure_3_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_3, Pres_g_3, Pres_pert_3, Pres_pi_3, Pres_K_3, Pres_rho_3, Pres_omega_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, Pres_Q5_3, Pres_H_3, Pres_sea_3)])
    else:
        Pres_Q_3, Pres_g_3 = utils.data_collect(0, 1, pressure_3_file)
        Pres_pert_3, Pres_pi_3 = utils.data_collect(2, 3, pressure_3_file)
        Pres_K_3, _ = utils.data_collect(4, 4, pressure_3_file)
        Pres_rho_3, Pres_omega_3 = utils.data_collect(5, 6, pressure_3_file)
        Pres_D_3, Pres_N_3 = utils.data_collect(7, 8, pressure_3_file)
        Pres_T_3, Pres_F_3 = utils.data_collect(9, 10, pressure_3_file)
        Pres_P_3, Pres_Q5_3 = utils.data_collect(11, 12, pressure_3_file)
        Pres_H_3, Pres_sea_3 = utils.data_collect(13, 14, pressure_3_file)

    contrib_q_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_Q_1)]
    contrib_g_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_g_1)]
    contrib_pert_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_pert_1)]
    contrib_sea_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_1)]
    contrib_qgp_1 = [sum(el) for el in zip(contrib_q_1, contrib_g_1, contrib_pert_1, contrib_sea_1)]

    contrib_q_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_Q_3)]
    contrib_g_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_g_3)]
    contrib_pert_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_pert_3)]
    contrib_sea_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_3)]
    contrib_qgp_3 = [sum(el) for el in zip(contrib_q_3, contrib_g_3, contrib_pert_3, contrib_sea_3)]

    contrib_pi_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_pi_1)]
    contrib_rho_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_rho_1)]
    contrib_omega_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_omega_1)]
    contrib_K_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_K_1)]
    contrib_D_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_D_1)]
    contrib_N_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_N_1)]
    contrib_T_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_T_1)]
    contrib_F_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_F_1)]
    contrib_P_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_P_1)]
    #contrib_Q5_1 = [0.0 for T_el, p_el in zip(T_1, Pres_Q5_1)]
    contrib_Q5_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_Q5_1)]
    contrib_H_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_H_1)]
    contrib_cluster_1 = [sum(el) for el in zip(contrib_pi_1, contrib_rho_1, contrib_omega_1, contrib_K_1, contrib_D_1, contrib_N_1, contrib_T_1, contrib_F_1, contrib_P_1, contrib_Q5_1, contrib_H_1)]
    contrib_cluster_singlet_1 = [sum(el) for el in zip(contrib_pi_1, contrib_rho_1, contrib_omega_1, contrib_K_1, contrib_N_1, contrib_T_1, contrib_P_1, contrib_H_1)]
    contrib_cluster_color_1 = [sum(el) for el in zip(contrib_D_1, contrib_F_1, contrib_Q5_1)]
    contrib_total_1 = [sum(el) for el in zip(contrib_cluster_1, contrib_qgp_1)]

    contrib_pi_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_pi_3)]
    contrib_rho_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_rho_3)]
    contrib_omega_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_omega_3)]
    contrib_K_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_K_3)]
    contrib_D_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_D_3)]
    contrib_N_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_N_3)]
    contrib_T_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_T_3)]
    contrib_F_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_F_3)]
    contrib_P_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_P_3)]
    contrib_Q5_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_Q5_3)]
    contrib_H_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_H_3)]
    contrib_cluster_3 = [sum(el) for el in zip(contrib_pi_3, contrib_rho_3, contrib_omega_3, contrib_K_3, contrib_D_3, contrib_N_3, contrib_T_3, contrib_F_3, contrib_P_3, contrib_Q5_3, contrib_H_3)]
    contrib_cluster_singlet_3 = [sum(el) for el in zip(contrib_pi_3, contrib_rho_3, contrib_omega_3, contrib_K_3, contrib_N_3, contrib_T_3, contrib_P_3, contrib_H_3)]
    contrib_cluster_color_3 = [sum(el) for el in zip(contrib_D_3, contrib_F_3, contrib_Q5_3)]
    contrib_total_3 = [sum(el) for el in zip(contrib_cluster_3, contrib_qgp_3)]

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu0_high.dat")
    (low_1204_6710v2_mu100_x, low_1204_6710v2_mu100_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu100_low.dat")
    (high_1204_6710v2_mu100_x, high_1204_6710v2_mu100_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu100_high.dat")
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu200_high.dat")
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu300_low.dat")
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu300_high.dat")
    (low_1204_6710v2_mu400_x, low_1204_6710v2_mu400_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu400_low.dat")
    (high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1204_6710v2_table4_pressure_mu400_high.dat")
    (high_1407_6387_mu0_x, high_1407_6387_mu0_y) = utils.data_collect(0, 3, "D:/EoS/epja/figure8/1407_6387_table1_pressure_mu0.dat")
    (low_1407_6387_mu0_x, low_1407_6387_mu0_y) = utils.data_collect(0, 2, "D:/EoS/epja/figure8/1407_6387_table1_pressure_mu0.dat")
    (high_1309_5258_mu0_x, high_1309_5258_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1309_5258_figure6_pressure_mu0_high.dat")
    (low_1309_5258_mu0_x, low_1309_5258_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1309_5258_figure6_pressure_mu0_low.dat")
    (high_1710_05024_mu0_x, high_1710_05024_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1710_05024_figure8_pressure_mu0_high.dat")
    (low_1710_05024_mu0_x, low_1710_05024_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure8/1710_05024_figure8_pressure_mu0_low.dat")

    borsanyi_1204_6710v2_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = numpy.array(borsanyi_1204_6710v2_mu0)
    borsanyi_1204_6710v2_mu100 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu100_x, high_1204_6710v2_mu100_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu100_x[::-1], low_1204_6710v2_mu100_y[::-1]):
        borsanyi_1204_6710v2_mu100.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu100 = numpy.array(borsanyi_1204_6710v2_mu100)
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

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12.0, 10))
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.axis([10., 400., 0., 4.2])
    ax1.add_patch(matplotlib.patches.Polygon(bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Bazavov et al. (2014)'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2014)'))
    ax1.add_patch(matplotlib.patches.Polygon(bazavov_1710_05024_mu0, closed = True, fill = True, color = 'magenta', alpha = 0.3, label = r'Bazavov et al. (2018)'))
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012)'))
    ax1.fill_between(T_1, contrib_cluster_1, color = 'green')
    ax1.fill_between(T_1, contrib_cluster_color_1, color = 'red')
    ax1.plot(T_1, contrib_total_1, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    ax1.plot(T_1, contrib_qgp_1, '--', c = 'blue', label = r'PNJL')
    ax1.text(180.0, 0.1, r'Color triplet/antitriplet', color = 'red', fontsize = 14)
    ax1.text(165.0, 0.34, r'Color singlet', color = 'green', fontsize = 14)
    ax1.text(170.0, 0.58, r'PNJL', color = 'blue', fontsize = 14)
    ax1.text(195.0, 1.16, r'total pressure', color = 'black', fontsize = 14)
    ax1.text(21.0, 3.9, r'$\mathrm{\mu_B=0}$', color = 'black', fontsize = 14)
    ax1.text(22.0, 2.2, r'Borsanyi et al. (2012)', color = 'blue', alpha = 0.7, fontsize = 14)
    ax1.text(175.0, 3.7, r'Borsanyi et al. (2014)', color = 'green', alpha = 0.7, fontsize = 14)
    ax1.text(75.0, 3.0, r'Bazavov et al. (2014)', color = 'red', alpha = 0.7, fontsize = 14)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    contrib_qgp_1 = [el / norm for el, norm in zip(contrib_qgp_1, contrib_total_1)]
    contrib_pi_1 = [el / norm for el, norm in zip(contrib_pi_1, contrib_total_1)]
    contrib_rho_1 = [el / norm for el, norm in zip(contrib_rho_1, contrib_total_1)]
    contrib_omega_1 = [el / norm for el, norm in zip(contrib_omega_1, contrib_total_1)]
    contrib_K_1 = [el / norm for el, norm in zip(contrib_K_1, contrib_total_1)]
    contrib_D_1 = [el / norm for el, norm in zip(contrib_D_1, contrib_total_1)]
    contrib_N_1 = [el / norm for el, norm in zip(contrib_N_1, contrib_total_1)]
    contrib_T_1 = [el / norm for el, norm in zip(contrib_T_1, contrib_total_1)]
    contrib_F_1 = [el / norm for el, norm in zip(contrib_F_1, contrib_total_1)]
    contrib_P_1 = [el / norm for el, norm in zip(contrib_P_1, contrib_total_1)]
    contrib_Q5_1 = [el / norm for el, norm in zip(contrib_Q5_1, contrib_total_1)]
    contrib_H_1 = [el / norm for el, norm in zip(contrib_H_1, contrib_total_1)]

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.axis([20., 265., 1e-6, 1e1])
    ax2.plot(T_1, [1.0 for el in T_1], '-', c = 'black')
    ax2.plot(T_1, contrib_qgp_1, '-', c = 'blue')
    ax2.plot(T_1, contrib_pi_1, '-', c = '#653239')
    ax2.plot(T_1, contrib_rho_1, '-', c = '#858AE3')
    ax2.plot(T_1, contrib_omega_1, '-', c = '#FF37A6')
    ax2.plot(T_1, contrib_K_1, '-', c = 'red')
    ax2.plot(T_1, contrib_D_1, '--', c = '#4CB944')
    ax2.plot(T_1, contrib_N_1, '-', c = '#DEA54B')
    ax2.plot(T_1, contrib_T_1, '-', c = '#23CE6B')
    ax2.plot(T_1, contrib_F_1, '--', c = '#DB222A')
    ax2.plot(T_1, contrib_P_1, '-', c = '#78BC61')
    ax2.plot(T_1, contrib_Q5_1, '--', c = '#55DBCB')
    ax2.plot(T_1, contrib_H_1, '-', c = '#A846A0')
    ax2.text(221.5, 3e-6, r'$\mathrm{\pi}$', color = '#653239', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(54.0, 0.008, r'K', color = 'red', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(135.0, 6.0e-5, r'H', color = '#A846A0', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(70.0, 0.002, r'$\mathrm{\omega}$', color = '#FF37A6', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(66.0, 1.4e-5, r'D', color = '#4CB944', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.001', fc ='white', ec ='none'))
    ax2.text(104.0, 0.0035, r'N', color = '#DEA54B', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.05', fc ='white', ec ='none'))
    ax2.text(128.0, 0.003, r'T', color = '#23CE6B', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(95.0, 2.3e-6, r'F', color = '#DB222A', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(113.0, 0.0002, r'P', color = '#78BC61', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(119.0, 4.38e-5, r'Q', color = '#55DBCB', fontsize = 16, bbox = dict(boxstyle = 'square,pad=-0.1', fc ='white', ec ='none'))
    ax2.text(80.0, 0.015, r'$\mathrm{\rho}$', color = '#858AE3', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(22., 1.5, r'total pressure', color = 'black', fontsize = 14)
    ax2.text(225., 1.5, r'PNJL', color = 'blue', fontsize = 14)
    ax2.set_yscale('log')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{\log~p}$', fontsize = 16)

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.axis([10., 400., 0., 4.2])
    ax3.add_patch(matplotlib.patches.Polygon(bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.3, label = r'Bazavov et al. (2014)'))
    ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.3, label = r'Borsanyi et al. (2014)'))
    ax3.add_patch(matplotlib.patches.Polygon(bazavov_1710_05024_mu0, closed = True, fill = True, color = 'magenta', alpha = 0.3, label = r'Bazavov et al. (2018)'))
    ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012)'))
    ax3.fill_between(T_3, contrib_cluster_3, color = 'green')
    ax3.fill_between(T_3, contrib_cluster_color_3, color = 'red')
    ax3.plot(T_3, contrib_total_3, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    ax3.plot(T_3, contrib_qgp_3, '--', c = 'blue', label = r'PNJL')
    ax3.text(180.0, 0.1, r'Color triplet/antitriplet', color = 'red', fontsize = 14)
    ax3.text(165.0, 0.34, r'Color singlet', color = 'green', fontsize = 14)
    ax3.text(170.0, 0.58, r'PNJL', color = 'blue', fontsize = 14)
    ax3.text(20.0, 1.5, r'total pressure', color = 'black', fontsize = 14)
    ax3.text(21.0, 3.9, r'$\mathrm{\mu_B=0}$', color = 'black', fontsize = 14)
    ax3.text(18.0, 2.2, r'Borsanyi et al. (2012)', color = 'blue', alpha = 0.7, fontsize = 14)
    ax3.text(175.0, 3.7, r'Borsanyi et al. (2014)', color = 'green', alpha = 0.7, fontsize = 14)
    ax3.text(75.0, 3.0, r'Bazavov et al. (2014)', color = 'red', alpha = 0.7, fontsize = 14)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    contrib_qgp_3 = [el / norm for el, norm in zip(contrib_qgp_3, contrib_total_3)]
    contrib_pi_3 = [el / norm for el, norm in zip(contrib_pi_3, contrib_total_3)]
    contrib_rho_3 = [el / norm for el, norm in zip(contrib_rho_3, contrib_total_3)]
    contrib_omega_3 = [el / norm for el, norm in zip(contrib_omega_3, contrib_total_3)]
    contrib_K_3 = [el / norm for el, norm in zip(contrib_K_3, contrib_total_3)]
    contrib_D_3 = [el / norm for el, norm in zip(contrib_D_3, contrib_total_3)]
    contrib_N_3 = [el / norm for el, norm in zip(contrib_N_3, contrib_total_3)]
    contrib_T_3 = [el / norm for el, norm in zip(contrib_T_3, contrib_total_3)]
    contrib_F_3 = [el / norm for el, norm in zip(contrib_F_3, contrib_total_3)]
    contrib_P_3 = [el / norm for el, norm in zip(contrib_P_3, contrib_total_3)]
    contrib_Q5_3 = [el / norm for el, norm in zip(contrib_Q5_3, contrib_total_3)]
    contrib_H_3 = [el / norm for el, norm in zip(contrib_H_3, contrib_total_3)]

    ax4 = fig1.add_subplot(2, 2, 4)
    ax4.axis([20., 265., 1e-6, 1e1])
    ax4.plot(T_3, [1.0 for el in T_3], '-', c = 'black')
    ax4.plot(T_3, contrib_qgp_3, '-', c = 'blue')
    ax4.plot(T_3, contrib_pi_3, '-', c = '#653239')
    ax4.plot(T_3, contrib_rho_3, '-', c = '#858AE3')
    ax4.plot(T_3, contrib_omega_3, '-', c = '#FF37A6')
    ax4.plot(T_3, contrib_K_3, '-', c = 'red')
    ax4.plot(T_3, contrib_D_3, '--', c = '#4CB944')
    ax4.plot(T_3, contrib_N_3, '-', c = '#DEA54B')
    ax4.plot(T_3, contrib_T_3, '-', c = '#23CE6B')
    ax4.plot(T_3, contrib_F_3, '--', c = '#DB222A')
    ax4.plot(T_3, contrib_P_3, '-', c = '#78BC61')
    ax4.plot(T_3, contrib_Q5_3, '--', c = '#55DBCB')
    ax4.plot(T_3, contrib_H_3, '-', c = '#A846A0')
    ax4.text(221.5, 3e-6, r'$\mathrm{\pi}$', color = '#653239', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(54.0, 0.008, r'K', color = 'red', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(128.0, 1.5e-5, r'H', color = '#A846A0', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(70.0, 0.0013, r'$\mathrm{\omega}$', color = '#FF37A6', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(65.5, 0.0055, r'D', color = '#4CB944', fontsize = 16, bbox = dict(boxstyle = 'square,pad=-0.08', fc ='white', ec ='none'))
    ax4.text(104.0, 0.0015, r'N', color = '#DEA54B', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.05', fc ='white', ec ='none'))
    ax4.text(92.7, 8.0e-5, r'T', color = '#23CE6B', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(67.5, 2.3e-6, r'F', color = '#DB222A', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(113.0, 6.5e-5, r'P', color = '#78BC61', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(126.0, 0.0006, r'Q', color = '#55DBCB', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0', fc ='white', ec ='none'))
    ax4.text(80.0, 0.007, r'$\mathrm{\rho}$', color = '#858AE3', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(22., 1.5, r'total pressure', color = 'black', fontsize = 14)
    ax4.text(225., 1.5, r'PNJL', color = 'blue', fontsize = 14)
    ax4.set_yscale('log')
    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax4.set_xlabel(r'T [MeV]', fontsize = 16)
    ax4.set_ylabel(r'$\mathrm{\log~p}$', fontsize = 16)

    fig2 = matplotlib.pyplot.figure(num = 2, figsize = (6, 5))
    ax5 = fig2.add_subplot(1, 1, 1)
    ax5.axis([20., 250., 0., 0.5])
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_pi_1)], '-', c = '#653239')
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_rho_1)], '-', c = '#858AE3')
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_omega_1)], '-', c = '#FF37A6')
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_K_1)], '-', c = 'red')
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_D_1)], '--', c = '#4CB944')
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_N_1)], '-', c = '#DEA54B')
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_T_1)], '-', c = '#23CE6B')
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_F_1)], '--', c = '#DB222A')
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_P_1)], '-', c = '#78BC61')
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_Q5_1)], '--', c = '#55DBCB')
    ax5.plot(T_1, [ el_p / (el_T ** 4) for el_T, el_p in zip(T_1, Pres_H_1)], '-', c = '#A846A0')
    for tick in ax5.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax5.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax5.set_xlabel(r'T [MeV]', fontsize = 16)
    ax5.set_ylabel(r'pressure', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    fig2.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def epja_figure10():

    phi_re_1, phi_im_1 = [], []
    phi_re_3, phi_im_3 = [], []

    Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_sea_1 = [], [], [], []
    Pres_Q_3, Pres_g_3, Pres_pert_3, Pres_sea_3 = [], [], [], []

    Pres_pi_1, Pres_K_1, Pres_rho_1 = [], [], []
    Pres_omega_1, Pres_D_1, Pres_N_1 = [], [], []
    Pres_T_1, Pres_F_1, Pres_P_1 = [], [], []
    Pres_Q5_1, Pres_H_1 = [], []

    Pres_pi_3, Pres_K_3, Pres_rho_3 = [], [], [] 
    Pres_omega_3, Pres_D_3, Pres_N_3 = [], [], []
    Pres_T_3, Pres_F_3, Pres_P_3 = [], [], []
    Pres_Q5_3, Pres_H_3 = [], []

    T_1 = numpy.linspace(1.0, 400.0, 200)
    T_3 = numpy.linspace(1.0, 400.0, 200)

    mu_1 = [200.0 / 3.0 for el in T_1]
    mu_3 = [200.0 / 3.0 for el in T_3]

    calc_1 = False
    calc_3 = False

    cluster_backreaction = False

    pl_1_file = "D:/EoS/epja/figure9/pl_1.dat"
    pl_3_file = "D:/EoS/epja/figure9/pl_3.dat"
    pressure_1_file = "D:/EoS/epja/figure9/pressure_1.dat"
    pressure_3_file = "D:/EoS/epja/figure9/pressure_3.dat"

    if calc_1:
        phi_re_1.append(1e-15)
        phi_im_1.append(2e-15)
        lT = len(T_1)
        for T_el, mu_el in tqdm.tqdm(zip(T_1, mu_1), desc = "Traced Polyakov loop (calc #1)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_1[-1], phi_im_1[-1], with_clusters = cluster_backreaction)
            phi_re_1.append(temp_phi_re)
            phi_im_1.append(temp_phi_im)
        phi_re_1 = phi_re_1[1:]
        phi_im_1 = phi_im_1[1:]
        with open(pl_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_1, mu_1, phi_re_1, phi_im_1)])
    else:
        T_1, mu_1 = utils.data_collect(0, 1, pl_1_file)
        phi_re_1, phi_im_1 = utils.data_collect(2, 3, pl_1_file)

    if calc_3:
        phi_re_3.append(1e-15)
        phi_im_3.append(2e-15)
        lT = len(T_3)
        for T_el, mu_el in tqdm.tqdm(zip(T_3, mu_3), desc = "Traced Polyakov loop (calc #3)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_3[-1], phi_im_3[-1], with_clusters = cluster_backreaction)
            phi_re_3.append(temp_phi_re)
            phi_im_3.append(temp_phi_im)
        phi_re_3 = phi_re_3[1:]
        phi_im_3 = phi_im_3[1:]
        with open(pl_3_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_3, mu_3, phi_re_3, phi_im_3)])
    else:
        T_3, mu_3 = utils.data_collect(0, 1, pl_3_file)
        phi_re_3, phi_im_3 = utils.data_collect(2, 3, pl_3_file)

    if calc_1:
        Pres_g_1 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_1, phi_re_1, phi_im_1), 
                desc = "Gluon pressure (calc #1)", 
                total = len(T_1),
                ascii = True)]
        Pres_sea_1 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(zip(T_1, mu_1), 
                desc = "Sigma mf pressure (calc #1)", 
                total = len(T_1),
                ascii = True)]
        Pres_Q_1 = [
            pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
            for T_el, mu_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                desc = "Quark pressure (calc #1)", 
                total = len(T_1), 
                ascii = True)]
        Pres_pert_1 = [
            pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
            for T_el, mu_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                desc = "Perturbative pressure (calc #1)", 
                total = len(T_1), 
                ascii = True)]        
        ((Pres_pi_1, Pres_rho_1, Pres_omega_1, Pres_K_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, 
             Pres_Q5_1, Pres_H_1),
            (_, _, _, _, _, _, _, _, _, _, _),
            (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_1, mu_1, phi_re_1, phi_im_1)
        with open(pressure_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_pi_1, Pres_K_1, Pres_rho_1, Pres_omega_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, Pres_Q5_1, Pres_H_1, Pres_sea_1)])
    else:
        Pres_Q_1, Pres_g_1 = utils.data_collect(0, 1, pressure_1_file)
        Pres_pert_1, Pres_pi_1 = utils.data_collect(2, 3, pressure_1_file)
        Pres_K_1, _ = utils.data_collect(4, 4, pressure_1_file)
        Pres_rho_1, Pres_omega_1 = utils.data_collect(5, 6, pressure_1_file)
        Pres_D_1, Pres_N_1 = utils.data_collect(7, 8, pressure_1_file)
        Pres_T_1, Pres_F_1 = utils.data_collect(9, 10, pressure_1_file)
        Pres_P_1, Pres_Q5_1 = utils.data_collect(11, 12, pressure_1_file)
        Pres_H_1, Pres_sea_1 = utils.data_collect(13, 14, pressure_1_file)

    if calc_3:
        Pres_g_3 = [
            pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_3, phi_re_3, phi_im_3), 
                desc = "Gluon pressure (calc #3)", 
                total = len(T_3),
                ascii = True)]
        Pres_sea_3 = [
            pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el) 
            for T_el, mu_el
            in tqdm.tqdm(zip(T_3, mu_3), 
                desc = "Sigma mf pressure (calc #3)", 
                total = len(T_3),
                ascii = True)]
        Pres_Q_3 = [
            pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms) 
            for T_el, mu_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3]), 
                desc = "Quark pressure (calc #3)", 
                total = len(T_3), 
                ascii = True)]
        Pres_pert_3 = [
            pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0) 
            for T_el, mu_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3]), 
                desc = "Perturbative pressure (calc #3)", 
                total = len(T_3), 
                ascii = True)]        
        ((Pres_pi_3, Pres_rho_3, Pres_omega_3, Pres_K_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, 
             Pres_Q5_3, Pres_H_3),
            (_, _, _, _, _, _, _, _, _, _, _),
            (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3])
        with open(pressure_3_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_3, Pres_g_3, Pres_pert_3, Pres_pi_3, Pres_K_3, Pres_rho_3, Pres_omega_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, Pres_Q5_3, Pres_H_3, Pres_sea_3)])
    else:
        Pres_Q_3, Pres_g_3 = utils.data_collect(0, 1, pressure_3_file)
        Pres_pert_3, Pres_pi_3 = utils.data_collect(2, 3, pressure_3_file)
        Pres_K_3, _ = utils.data_collect(4, 4, pressure_3_file)
        Pres_rho_3, Pres_omega_3 = utils.data_collect(5, 6, pressure_3_file)
        Pres_D_3, Pres_N_3 = utils.data_collect(7, 8, pressure_3_file)
        Pres_T_3, Pres_F_3 = utils.data_collect(9, 10, pressure_3_file)
        Pres_P_3, Pres_Q5_3 = utils.data_collect(11, 12, pressure_3_file)
        Pres_H_3, Pres_sea_3 = utils.data_collect(13, 14, pressure_3_file)

    contrib_q_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_Q_1)]
    contrib_g_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_g_1)]
    contrib_pert_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_pert_1)]
    contrib_sea_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_1)]
    contrib_qgp_1 = [sum(el) for el in zip(contrib_q_1, contrib_g_1, contrib_pert_1, contrib_sea_1)]

    contrib_q_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_Q_3)]
    contrib_g_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_g_3)]
    contrib_pert_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_pert_3)]
    contrib_sea_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_sea_3)]
    contrib_qgp_3 = [sum(el) for el in zip(contrib_q_3, contrib_g_3, contrib_pert_3, contrib_sea_3)]

    contrib_pi_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_pi_1)]
    contrib_rho_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_rho_1)]
    contrib_omega_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_omega_1)]
    contrib_K_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_K_1)]
    contrib_D_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_D_1)]
    contrib_N_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_N_1)]
    contrib_T_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_T_1)]
    contrib_F_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_F_1)]
    contrib_P_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_P_1)]
    contrib_Q5_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_Q5_1)]
    contrib_H_1 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_1, Pres_H_1)]
    contrib_cluster_1 = [sum(el) for el in zip(contrib_pi_1, contrib_rho_1, contrib_omega_1, contrib_K_1, contrib_D_1, contrib_N_1, contrib_T_1, contrib_F_1, contrib_P_1, contrib_Q5_1, contrib_H_1)]
    contrib_cluster_singlet_1 = [sum(el) for el in zip(contrib_pi_1, contrib_rho_1, contrib_omega_1, contrib_K_1, contrib_N_1, contrib_T_1, contrib_P_1, contrib_H_1)]
    contrib_cluster_color_1 = [sum(el) for el in zip(contrib_D_1, contrib_F_1, contrib_Q5_1)]
    contrib_total_1 = [sum(el) for el in zip(contrib_cluster_1, contrib_qgp_1)]

    contrib_pi_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_pi_3)]
    contrib_rho_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_rho_3)]
    contrib_omega_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_omega_3)]
    contrib_K_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_K_3)]
    contrib_D_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_D_3)]
    contrib_N_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_N_3)]
    contrib_T_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_T_3)]
    contrib_F_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_F_3)]
    contrib_P_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_P_3)]
    contrib_Q5_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_Q5_3)]
    contrib_H_3 = [p_el / (T_el ** 4) for T_el, p_el in zip(T_3, Pres_H_3)]
    contrib_cluster_3 = [sum(el) for el in zip(contrib_pi_3, contrib_rho_3, contrib_omega_3, contrib_K_3, contrib_D_3, contrib_N_3, contrib_T_3, contrib_F_3, contrib_P_3, contrib_Q5_3, contrib_H_3)]
    contrib_cluster_singlet_3 = [sum(el) for el in zip(contrib_pi_3, contrib_rho_3, contrib_omega_3, contrib_K_3, contrib_N_3, contrib_T_3, contrib_P_3, contrib_H_3)]
    contrib_cluster_color_3 = [sum(el) for el in zip(contrib_D_3, contrib_F_3, contrib_Q5_3)]
    contrib_total_3 = [sum(el) for el in zip(contrib_cluster_3, contrib_qgp_3)]

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1204_6710v2_table4_pressure_mu0_high.dat")
    (low_1204_6710v2_mu100_x, low_1204_6710v2_mu100_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1204_6710v2_table4_pressure_mu100_low.dat")
    (high_1204_6710v2_mu100_x, high_1204_6710v2_mu100_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1204_6710v2_table4_pressure_mu100_high.dat")
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1204_6710v2_table4_pressure_mu200_high.dat")
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1204_6710v2_table4_pressure_mu300_low.dat")
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1204_6710v2_table4_pressure_mu300_high.dat")
    (low_1204_6710v2_mu400_x, low_1204_6710v2_mu400_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1204_6710v2_table4_pressure_mu400_low.dat")
    (high_1204_6710v2_mu400_x, high_1204_6710v2_mu400_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1204_6710v2_table4_pressure_mu400_high.dat")
    (high_1407_6387_mu0_x, high_1407_6387_mu0_y) = utils.data_collect(0, 3, "D:/EoS/epja/figure9/1407_6387_table1_pressure_mu0.dat")
    (low_1407_6387_mu0_x, low_1407_6387_mu0_y) = utils.data_collect(0, 2, "D:/EoS/epja/figure9/1407_6387_table1_pressure_mu0.dat")
    (high_1309_5258_mu0_x, high_1309_5258_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1309_5258_figure6_pressure_mu0_high.dat")
    (low_1309_5258_mu0_x, low_1309_5258_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1309_5258_figure6_pressure_mu0_low.dat")
    (high_1710_05024_mu0_x, high_1710_05024_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1710_05024_figure8_pressure_mu0_high.dat")
    (low_1710_05024_mu0_x, low_1710_05024_mu0_y) = utils.data_collect(0, 1, "D:/EoS/epja/figure9/1710_05024_figure8_pressure_mu0_low.dat")

    borsanyi_1204_6710v2_mu0 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = numpy.array(borsanyi_1204_6710v2_mu0)
    borsanyi_1204_6710v2_mu100 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu100_x, high_1204_6710v2_mu100_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu100_x[::-1], low_1204_6710v2_mu100_y[::-1]):
        borsanyi_1204_6710v2_mu100.append(numpy.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu100 = numpy.array(borsanyi_1204_6710v2_mu100)
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

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12.0, 10.0))
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.axis([10., 400., 0., 4.2])
    ax1.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012)'))
    ax1.fill_between(T_1, contrib_cluster_1, color = 'green')
    ax1.fill_between(T_1, contrib_cluster_color_1, color = 'red')
    ax1.plot(T_1, contrib_total_1, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    ax1.plot(T_1, contrib_qgp_1, '--', c = 'blue', label = r'PNJL')
    ax1.text(180.0, 0.1, r'Color triplet/antitriplet', color = 'red', fontsize = 14)
    ax1.text(165.0, 0.34, r'Color singlet', color = 'green', fontsize = 14)
    ax1.text(170.0, 0.58, r'PNJL', color = 'blue', fontsize = 14)
    ax1.text(195.0, 1.16, r'total pressure', color = 'black', fontsize = 14)
    ax1.text(21.0, 3.9, r'$\mathrm{\mu_B=200}$ MeV', color = 'black', fontsize = 14)
    ax1.text(22.0, 2.2, r'Borsanyi et al. (2012)', color = 'blue', alpha = 0.7, fontsize = 14)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    contrib_qgp_1 = [el / norm for el, norm in zip(contrib_qgp_1, contrib_total_1)]
    contrib_pi_1 = [el / norm for el, norm in zip(contrib_pi_1, contrib_total_1)]
    contrib_rho_1 = [el / norm for el, norm in zip(contrib_rho_1, contrib_total_1)]
    contrib_omega_1 = [el / norm for el, norm in zip(contrib_omega_1, contrib_total_1)]
    contrib_K_1 = [el / norm for el, norm in zip(contrib_K_1, contrib_total_1)]
    contrib_D_1 = [el / norm for el, norm in zip(contrib_D_1, contrib_total_1)]
    contrib_N_1 = [el / norm for el, norm in zip(contrib_N_1, contrib_total_1)]
    contrib_T_1 = [el / norm for el, norm in zip(contrib_T_1, contrib_total_1)]
    contrib_F_1 = [el / norm for el, norm in zip(contrib_F_1, contrib_total_1)]
    contrib_P_1 = [el / norm for el, norm in zip(contrib_P_1, contrib_total_1)]
    contrib_Q5_1 = [el / norm for el, norm in zip(contrib_Q5_1, contrib_total_1)]
    contrib_H_1 = [el / norm for el, norm in zip(contrib_H_1, contrib_total_1)]

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.axis([20., 250., 1e-6, 1e1])
    ax2.plot(T_1, [1.0 for el in T_1], '-', c = 'black')
    ax2.plot(T_1, contrib_qgp_1, '-', c = 'blue')
    ax2.plot(T_1, contrib_pi_1, '-', c = '#653239')
    ax2.plot(T_1, contrib_rho_1, '-', c = '#858AE3')
    ax2.plot(T_1, contrib_omega_1, '-', c = '#FF37A6')
    ax2.plot(T_1, contrib_K_1, '-', c = 'red')
    ax2.plot(T_1, contrib_D_1, '--', c = '#4CB944')
    ax2.plot(T_1, contrib_N_1, '-', c = '#DEA54B')
    ax2.plot(T_1, contrib_T_1, '-', c = '#23CE6B')
    ax2.plot(T_1, contrib_F_1, '--', c = '#DB222A')
    ax2.plot(T_1, contrib_P_1, '-', c = '#78BC61')
    ax2.plot(T_1, contrib_Q5_1, '--', c = '#55DBCB')
    ax2.plot(T_1, contrib_H_1, '-', c = '#A846A0')
    ax2.text(221.5, 3e-6, r'$\mathrm{\pi}$', color = '#653239', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(54.0, 0.008, r'K', color = 'red', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(135.0, 0.0004, r'H', color = '#A846A0', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(69.0, 0.0025, r'$\mathrm{\omega}$', color = '#FF37A6', fontsize = 16, bbox = dict(boxstyle = 'square,pad=-0.05', fc ='white', ec ='none'))
    ax2.text(66.0, 2.5e-5, r'D', color = '#4CB944', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.001', fc ='white', ec ='none'))
    ax2.text(60.0, 0.0003, r'N', color = '#DEA54B', fontsize = 16, bbox = dict(boxstyle = 'square,pad=-0.1', fc ='white', ec ='none'))
    ax2.text(128.0, 0.0035, r'T', color = '#23CE6B', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.01', fc ='white', ec ='none'))
    ax2.text(80.0, 2.3e-6, r'F', color = '#DB222A', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(102.0, 0.0002, r'P', color = '#78BC61', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(119.0, 0.0005, r'Q', color = '#55DBCB', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0', fc ='white', ec ='none'))
    ax2.text(80.0, 0.015, r'$\mathrm{\rho}$', color = '#858AE3', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(22., 1.5, r'total pressure', color = 'black', fontsize = 14)
    ax2.text(225., 1.5, r'PNJL', color = 'blue', fontsize = 14)
    ax2.set_yscale('log')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{\log~p}$', fontsize = 16)

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.axis([10., 400., 0., 4.2])
    ax3.add_patch(matplotlib.patches.Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'blue', alpha = 0.3, label = r'Borsanyi et al. (2012)'))
    ax3.fill_between(T_3, contrib_cluster_3, color = 'green')
    ax3.fill_between(T_3, contrib_cluster_color_3, color = 'red')
    ax3.plot(T_3, contrib_total_3, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    ax3.plot(T_3, contrib_qgp_3, '--', c = 'blue', label = r'PNJL')
    ax3.text(180.0, 0.1, r'Color triplet/antitriplet', color = 'red', fontsize = 14)
    ax3.text(165.0, 0.34, r'Color singlet', color = 'green', fontsize = 14)
    ax3.text(170.0, 0.58, r'PNJL', color = 'blue', fontsize = 14)
    ax3.text(20.0, 1.6, r'total pressure', color = 'black', fontsize = 14)
    ax3.text(21.0, 3.9, r'$\mathrm{\mu_B=200}$ MeV', color = 'black', fontsize = 14)
    ax3.text(18.0, 2.3, r'Borsanyi et al. (2012)', color = 'blue', alpha = 0.7, fontsize = 14)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    contrib_qgp_3 = [el / norm for el, norm in zip(contrib_qgp_3, contrib_total_3)]
    contrib_pi_3 = [el / norm for el, norm in zip(contrib_pi_3, contrib_total_3)]
    contrib_rho_3 = [el / norm for el, norm in zip(contrib_rho_3, contrib_total_3)]
    contrib_omega_3 = [el / norm for el, norm in zip(contrib_omega_3, contrib_total_3)]
    contrib_K_3 = [el / norm for el, norm in zip(contrib_K_3, contrib_total_3)]
    contrib_D_3 = [el / norm for el, norm in zip(contrib_D_3, contrib_total_3)]
    contrib_N_3 = [el / norm for el, norm in zip(contrib_N_3, contrib_total_3)]
    contrib_T_3 = [el / norm for el, norm in zip(contrib_T_3, contrib_total_3)]
    contrib_F_3 = [el / norm for el, norm in zip(contrib_F_3, contrib_total_3)]
    contrib_P_3 = [el / norm for el, norm in zip(contrib_P_3, contrib_total_3)]
    contrib_Q5_3 = [el / norm for el, norm in zip(contrib_Q5_3, contrib_total_3)]
    contrib_H_3 = [el / norm for el, norm in zip(contrib_H_3, contrib_total_3)]

    ax4 = fig1.add_subplot(2, 2, 4)
    ax4.axis([20., 250., 1e-6, 1e1])
    ax4.plot(T_3, [1.0 for el in T_3], '-', c = 'black')
    ax4.plot(T_3, contrib_qgp_3, '-', c = 'blue')
    ax4.plot(T_3, contrib_pi_3, '-', c = '#653239')
    ax4.plot(T_3, contrib_rho_3, '-', c = '#858AE3')
    ax4.plot(T_3, contrib_omega_3, '-', c = '#FF37A6')
    ax4.plot(T_3, contrib_K_3, '-', c = 'red')
    ax4.plot(T_3, contrib_D_3, '--', c = '#4CB944')
    ax4.plot(T_3, contrib_N_3, '-', c = '#DEA54B')
    ax4.plot(T_3, contrib_T_3, '-', c = '#23CE6B')
    ax4.plot(T_3, contrib_F_3, '--', c = '#DB222A')
    ax4.plot(T_3, contrib_P_3, '-', c = '#78BC61')
    ax4.plot(T_3, contrib_Q5_3, '--', c = '#55DBCB')
    ax4.plot(T_3, contrib_H_3, '-', c = '#A846A0')
    ax4.text(221.5, 3e-6, r'$\mathrm{\pi}$', color = '#653239', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(166.0, 3e-6, r'K', color = 'red', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(135.0, 0.00018, r'H', color = '#A846A0', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(68.0, 0.0012, r'$\mathrm{\omega}$', color = '#FF37A6', fontsize = 16, bbox = dict(boxstyle = 'square,pad=-0.05', fc ='white', ec ='none'))
    ax4.text(112.5, 0.065, r'D', color = '#4CB944', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.001', fc ='white', ec ='none'))
    ax4.text(60.0, 0.00014, r'N', color = '#DEA54B', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0', fc ='white', ec ='none'))
    ax4.text(120.0, 0.0008, r'T', color = '#23CE6B', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.01', fc ='white', ec ='none'))
    ax4.text(54.0, 2.3e-6, r'F', color = '#DB222A', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(102.0, 0.00008, r'P', color = '#78BC61', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(97.0, 0.0005, r'Q', color = '#55DBCB', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0', fc ='white', ec ='none'))
    ax4.text(80.0, 0.006, r'$\mathrm{\rho}$', color = '#858AE3', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(22., 1.5, r'total pressure', color = 'black', fontsize = 14)
    ax4.text(225., 1.5, r'PNJL', color = 'blue', fontsize = 14)
    ax4.set_yscale('log')
    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax4.set_xlabel(r'T [MeV]', fontsize = 16)
    ax4.set_ylabel(r'$\mathrm{\log~p}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def epja_figure11():

    phi_re_1, phi_im_1 = [], []
    phi_re_3, phi_im_3 = [], []

    Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_sea_1 = [], [], [], []
    Pres_Q_3, Pres_g_3, Pres_pert_3, Pres_sea_3 = [], [], [], []

    Pres_pi_1, Pres_K_1, Pres_rho_1 = [], [], []
    Pres_omega_1, Pres_D_1, Pres_N_1 = [], [], []
    Pres_T_1, Pres_F_1, Pres_P_1 = [], [], []
    Pres_Q5_1, Pres_H_1 = [], []

    Pres_pi_3, Pres_K_3, Pres_rho_3 = [], [], [] 
    Pres_omega_3, Pres_D_3, Pres_N_3 = [], [], []
    Pres_T_3, Pres_F_3, Pres_P_3 = [], [], []
    Pres_Q5_3, Pres_H_3 = [], []

    T_1 = numpy.linspace(1.0, 400.0, 200)
    T_3 = numpy.linspace(1.0, 400.0, 200)

    mu_1 = [200.0 / 3.0 for el in T_1]
    mu_3 = [200.0 / 3.0 for el in T_3]

    calc_1 = False
    calc_3 = False

    cluster_backreaction = False

    pl_1_file = "D:/EoS/epja/figure10/pl_1.dat"
    pl_3_file = "D:/EoS/epja/figure10/pl_3.dat"
    pressure_1_file = "D:/EoS/epja/figure10/pressure_1.dat"
    pressure_3_file = "D:/EoS/epja/figure10/pressure_3.dat"

    if calc_1:
        phi_re_1.append(1e-15)
        phi_im_1.append(2e-15)
        lT = len(T_1)
        for T_el, mu_el in tqdm.tqdm(zip(T_1, mu_1), desc = "Traced Polyakov loop (calc #1)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_1[-1], phi_im_1[-1], with_clusters = cluster_backreaction)
            phi_re_1.append(temp_phi_re)
            phi_im_1.append(temp_phi_im)
        phi_re_1 = phi_re_1[1:]
        phi_im_1 = phi_im_1[1:]
        with open(pl_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_1, mu_1, phi_re_1, phi_im_1)])
    else:
        T_1, mu_1 = utils.data_collect(0, 1, pl_1_file)
        phi_re_1, phi_im_1 = utils.data_collect(2, 3, pl_1_file)

    if calc_3:
        phi_re_3.append(1e-15)
        phi_im_3.append(2e-15)
        lT = len(T_3)
        for T_el, mu_el in tqdm.tqdm(zip(T_3, mu_3), desc = "Traced Polyakov loop (calc #3)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_3[-1], phi_im_3[-1], with_clusters = cluster_backreaction)
            phi_re_3.append(temp_phi_re)
            phi_im_3.append(temp_phi_im)
        phi_re_3 = phi_re_3[1:]
        phi_im_3 = phi_im_3[1:]
        with open(pl_3_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_3, mu_3, phi_re_3, phi_im_3)])
    else:
        T_3, mu_3 = utils.data_collect(0, 1, pl_3_file)
        phi_re_3, phi_im_3 = utils.data_collect(2, 3, pl_3_file)

    if calc_1:
        Pres_g_1 = [
            0.0
            for T_el, mu_el
            in tqdm.tqdm(zip(T_1, mu_1), 
                desc = "Gluon baryon density (calc #1)", 
                total = len(T_1),
                ascii = True)]
        Pres_sea_1 = [
            0.0
            for T_el, mu_el
            in tqdm.tqdm(zip(T_1, mu_1), 
                desc = "Sigma mf baryon density (calc #1)", 
                total = len(T_1),
                ascii = True)]
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), desc = "Quark and perturbative baryon density (calc #1)", total = len(T_1), ascii = True):
            h = 1e-2
            mu_vec = [mu_el + 2 * h, mu_el + h, mu_el - h, mu_el - 2 * h]
            Phi_vec = [complex(phi_re_el, phi_im_el) for el in mu_vec]
            Phib_vec = [complex(phi_re_el, -phi_im_el) for el in mu_vec]
            Pres_Q_1.append(pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec) + pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, Nf = 1.0, ml = pnjl.defaults.default_ms))
            Pres_pert_1.append(pnjl.thermo.gcp_perturbative.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, Nf = 3.0))
        ((_, _, _, _, _, _, _, _, _, _, _),
            (Pres_pi_1, Pres_rho_1, Pres_omega_1, Pres_K_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, 
             Pres_Q5_1, Pres_H_1),
            (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_1, mu_1, phi_re_1, phi_im_1)
        with open(pressure_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_1, Pres_g_1, Pres_pert_1, Pres_pi_1, Pres_K_1, Pres_rho_1, Pres_omega_1, Pres_D_1, Pres_N_1, Pres_T_1, Pres_F_1, Pres_P_1, Pres_Q5_1, Pres_H_1, Pres_sea_1)])
    else:
        Pres_Q_1, Pres_g_1 = utils.data_collect(0, 1, pressure_1_file)
        Pres_pert_1, Pres_pi_1 = utils.data_collect(2, 3, pressure_1_file)
        Pres_K_1, _ = utils.data_collect(4, 4, pressure_1_file)
        Pres_rho_1, Pres_omega_1 = utils.data_collect(5, 6, pressure_1_file)
        Pres_D_1, Pres_N_1 = utils.data_collect(7, 8, pressure_1_file)
        Pres_T_1, Pres_F_1 = utils.data_collect(9, 10, pressure_1_file)
        Pres_P_1, Pres_Q5_1 = utils.data_collect(11, 12, pressure_1_file)
        Pres_H_1, Pres_sea_1 = utils.data_collect(13, 14, pressure_1_file)

    if calc_3:
        Pres_g_3 = [
            0.0
            for T_el, phi_re_el, phi_im_el 
            in tqdm.tqdm(zip(T_3, phi_re_3, phi_im_3), 
                desc = "Gluon pressure (calc #3)", 
                total = len(T_3),
                ascii = True)]
        Pres_sea_3 = [
            0.0
            for T_el, mu_el
            in tqdm.tqdm(zip(T_3, mu_3), 
                desc = "Sigma mf pressure (calc #3)", 
                total = len(T_3),
                ascii = True)]
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3]), desc = "Quark and perturbative baryon density (calc #3)", total = len(T_3), ascii = True):
            h = 1e-2
            mu_vec = [mu_el + 2 * h, mu_el + h, mu_el - h, mu_el - 2 * h]
            Phi_vec = [complex(phi_re_el, phi_im_el) for el in mu_vec]
            Phib_vec = [complex(phi_re_el, -phi_im_el) for el in mu_vec]
            Pres_Q_3.append(pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec) + pnjl.thermo.gcp_quark.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, Nf = 1.0, ml = pnjl.defaults.default_ms))
            Pres_pert_3.append(pnjl.thermo.gcp_perturbative.bdensity(T_el, mu_vec, Phi_vec, Phib_vec, Nf = 3.0))
        ((_, _, _, _, _, _, _, _, _, _, _),
            (Pres_pi_3, Pres_rho_3, Pres_omega_3, Pres_K_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, 
             Pres_Q5_3, Pres_H_3),
            (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c(T_3, mu_3, [1.0 for el in T_3], [0.0 for el in T_3])
        with open(pressure_3_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el] for q_el, g_el, pert_el, pi_el, k_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el, sea_el in zip(Pres_Q_3, Pres_g_3, Pres_pert_3, Pres_pi_3, Pres_K_3, Pres_rho_3, Pres_omega_3, Pres_D_3, Pres_N_3, Pres_T_3, Pres_F_3, Pres_P_3, Pres_Q5_3, Pres_H_3, Pres_sea_3)])
    else:
        Pres_Q_3, Pres_g_3 = utils.data_collect(0, 1, pressure_3_file)
        Pres_pert_3, Pres_pi_3 = utils.data_collect(2, 3, pressure_3_file)
        Pres_K_3, _ = utils.data_collect(4, 4, pressure_3_file)
        Pres_rho_3, Pres_omega_3 = utils.data_collect(5, 6, pressure_3_file)
        Pres_D_3, Pres_N_3 = utils.data_collect(7, 8, pressure_3_file)
        Pres_T_3, Pres_F_3 = utils.data_collect(9, 10, pressure_3_file)
        Pres_P_3, Pres_Q5_3 = utils.data_collect(11, 12, pressure_3_file)
        Pres_H_3, Pres_sea_3 = utils.data_collect(13, 14, pressure_3_file)

    contrib_q_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_Q_1)]
    contrib_g_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_g_1)]
    contrib_pert_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_pert_1)]
    contrib_sea_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_sea_1)]
    contrib_qgp_1 = [sum(el) for el in zip(contrib_q_1, contrib_pert_1, contrib_sea_1)]

    contrib_q_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_3, Pres_Q_3)]
    contrib_g_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_3, Pres_g_3)]
    contrib_pert_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_3, Pres_pert_3)]
    contrib_sea_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_sea_3)]
    contrib_qgp_3 = [sum(el) for el in zip(contrib_q_3, contrib_pert_3, contrib_sea_3)]

    contrib_D_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_D_1)]
    contrib_N_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_N_1)]
    contrib_T_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_T_1)]
    contrib_F_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_F_1)]
    contrib_P_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_P_1)]
    contrib_Q5_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_Q5_1)]
    contrib_H_1 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_1, Pres_H_1)]
    contrib_cluster_1 = [sum(el) for el in zip(contrib_D_1, contrib_N_1, contrib_T_1, contrib_F_1, contrib_P_1, contrib_Q5_1, contrib_H_1)]
    contrib_cluster_singlet_1 = [sum(el) for el in zip(contrib_N_1, contrib_T_1, contrib_P_1, contrib_H_1)]
    contrib_cluster_color_1 = [sum(el) for el in zip(contrib_D_1, contrib_F_1, contrib_Q5_1)]
    contrib_total_1 = [sum(el) for el in zip(contrib_cluster_1, contrib_qgp_1)]

    contrib_D_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_3, Pres_D_3)]
    contrib_N_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_3, Pres_N_3)]
    contrib_T_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_3, Pres_T_3)]
    contrib_F_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_3, Pres_F_3)]
    contrib_P_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_3, Pres_P_3)]
    contrib_Q5_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_3, Pres_Q5_3)]
    contrib_H_3 = [p_el / (T_el ** 3) for T_el, p_el in zip(T_3, Pres_H_3)]
    contrib_cluster_3 = [sum(el) for el in zip(contrib_D_3, contrib_N_3, contrib_T_3, contrib_F_3, contrib_P_3, contrib_Q5_3, contrib_H_3)]
    contrib_cluster_singlet_3 = [sum(el) for el in zip(contrib_N_3, contrib_T_3, contrib_P_3, contrib_H_3)]
    contrib_cluster_color_3 = [sum(el) for el in zip(contrib_D_3, contrib_F_3, contrib_Q5_3)]
    contrib_total_3 = [sum(el) for el in zip(contrib_cluster_3, contrib_qgp_3)]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12.0, 10.0))
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.axis([10., 400., 0., 0.3])
    ax1.fill_between(T_1, contrib_cluster_1, color = 'green')
    ax1.fill_between(T_1, contrib_cluster_color_1, color = 'red')
    ax1.plot(T_1, contrib_total_1, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    ax1.plot(T_1, contrib_qgp_1, '--', c = 'blue', label = r'PNJL')
    ax1.text(160.0, 0.0042, r'Color triplet/antitriplet', color = 'red', fontsize = 14)
    ax1.text(156.2, 0.021, r'Color singlet', color = 'green', fontsize = 14)
    ax1.text(156.0, 0.04, r'PNJL', color = 'blue', fontsize = 14)
    ax1.text(18.0, 0.2, r'total baryon density', color = 'black', fontsize = 14)
    ax1.text(21.0, 0.28, r'$\mathrm{\mu_B=200}$ MeV', color = 'black', fontsize = 14)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    contrib_qgp_1 = [el / norm for el, norm in zip(contrib_qgp_1, contrib_total_1)]
    contrib_D_1 = [el / norm for el, norm in zip(contrib_D_1, contrib_total_1)]
    contrib_N_1 = [el / norm for el, norm in zip(contrib_N_1, contrib_total_1)]
    contrib_T_1 = [el / norm for el, norm in zip(contrib_T_1, contrib_total_1)]
    contrib_F_1 = [el / norm for el, norm in zip(contrib_F_1, contrib_total_1)]
    contrib_P_1 = [el / norm for el, norm in zip(contrib_P_1, contrib_total_1)]
    contrib_Q5_1 = [el / norm for el, norm in zip(contrib_Q5_1, contrib_total_1)]
    contrib_H_1 = [el / norm for el, norm in zip(contrib_H_1, contrib_total_1)]

    #fig2 = matplotlib.pyplot.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.axis([20., 250., 1e-6, 1e1])
    ax2.plot(T_1, [1.0 for el in T_1], '-', c = 'black')
    ax2.plot(T_1, contrib_qgp_1, '-', c = 'blue')
    ax2.plot(T_1, contrib_D_1, '--', c = '#4CB944')
    ax2.plot(T_1, contrib_N_1, '-', c = '#DEA54B')
    ax2.plot(T_1, contrib_F_1, '--', c = '#DB222A')
    ax2.plot(T_1, contrib_P_1, '-', c = '#78BC61')
    ax2.plot(T_1, contrib_Q5_1, '--', c = '#55DBCB')
    ax2.plot(T_1, contrib_H_1, '-', c = '#A846A0')
    ax2.text(130.0, 0.0088, r'H', color = '#A846A0', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(112.5, 0.085, r'D', color = '#4CB944', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.001', fc ='white', ec ='none'))
    ax2.text(110.0, 0.4, r'N', color = '#DEA54B', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0', fc ='white', ec ='none'))
    ax2.text(47.2, 2.3e-6, r'F', color = '#DB222A', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(102.0, 0.01, r'P', color = '#78BC61', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax2.text(80.0, 0.0002, r'Q', color = '#55DBCB', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0', fc ='white', ec ='none'))
    ax2.text(22., 1.5, r'total baryon density', color = 'black', fontsize = 14)
    ax2.text(225., 1.5, r'PNJL', color = 'blue', fontsize = 14)
    ax2.set_yscale('log')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{\log~n_B}$', fontsize = 16)

    #fig3 = matplotlib.pyplot.figure(num = 3, figsize = (5.9, 5))
    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.axis([10., 400., 0., 0.3])
    ax3.fill_between(T_3, contrib_cluster_3, color = 'green')
    ax3.fill_between(T_3, contrib_cluster_color_3, color = 'red')
    ax3.plot(T_3, contrib_total_3, '-', c = 'black', label = r'$\mathrm{P_{PNJL+BU},~\mu=0}$')
    ax3.plot(T_3, contrib_qgp_3, '--', c = 'blue', label = r'PNJL')
    ax3.text(160.0, 0.0042, r'Color triplet/antitriplet', color = 'red', fontsize = 14)
    ax3.text(153.8, 0.07, r'Color singlet', color = 'green', fontsize = 14)
    ax3.text(145.7, 0.14, r'PNJL', color = 'blue', fontsize = 14)
    ax3.text(160.0, 0.25, r'total baryon density', color = 'black', fontsize = 14)
    ax3.text(21.0, 0.28, r'$\mathrm{\mu_B=200}$ MeV', color = 'black', fontsize = 14)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    contrib_qgp_3 = [el / norm for el, norm in zip(contrib_qgp_3, contrib_total_3)]
    contrib_D_3 = [el / norm for el, norm in zip(contrib_D_3, contrib_total_3)]
    contrib_N_3 = [el / norm for el, norm in zip(contrib_N_3, contrib_total_3)]
    contrib_T_3 = [el / norm for el, norm in zip(contrib_T_3, contrib_total_3)]
    contrib_F_3 = [el / norm for el, norm in zip(contrib_F_3, contrib_total_3)]
    contrib_P_3 = [el / norm for el, norm in zip(contrib_P_3, contrib_total_3)]
    contrib_Q5_3 = [el / norm for el, norm in zip(contrib_Q5_3, contrib_total_3)]
    contrib_H_3 = [el / norm for el, norm in zip(contrib_H_3, contrib_total_3)]

    ax4 = fig1.add_subplot(2, 2, 4)
    ax4.axis([20., 250., 1e-6, 1e1])
    ax4.plot(T_3, [1.0 for el in T_3], '-', c = 'black')
    ax4.plot(T_3, contrib_qgp_3, '-', c = 'blue')
    ax4.plot(T_3, contrib_D_3, '--', c = '#4CB944')
    ax4.plot(T_3, contrib_N_3, '-', c = '#DEA54B')
    ax4.plot(T_3, contrib_T_3, '-', c = '#23CE6B')
    ax4.plot(T_3, contrib_F_3, '--', c = '#DB222A')
    ax4.plot(T_3, contrib_P_3, '-', c = '#78BC61')
    ax4.plot(T_3, contrib_Q5_3, '--', c = '#55DBCB')
    ax4.plot(T_3, contrib_H_3, '-', c = '#A846A0')
    ax4.text(130.0, 0.0014, r'H', color = '#A846A0', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(112.5, 0.22, r'D', color = '#4CB944', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.001', fc ='white', ec ='none'))
    ax4.text(64.4, 0.0025, r'N', color = '#DEA54B', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0', fc ='white', ec ='none'))
    ax4.text(42.4, 2.3e-6, r'F', color = '#DB222A', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(102.0, 0.0006, r'P', color = '#78BC61', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0.1', fc ='white', ec ='none'))
    ax4.text(70.0, 0.0002, r'Q', color = '#55DBCB', fontsize = 16, bbox = dict(boxstyle = 'square,pad=0', fc ='white', ec ='none'))
    ax4.text(22., 1.5, r'total baryon density', color = 'black', fontsize = 14)
    ax4.text(225., 1.5, r'PNJL', color = 'blue', fontsize = 14)
    ax4.set_yscale('log')
    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax4.set_xlabel(r'T [MeV]', fontsize = 16)
    ax4.set_ylabel(r'$\mathrm{\log~n_B}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def epja_figure12():

    def find_first(val, T, entropy_per_baryon):
        found = False
        output = 0.0
        for ii, el in enumerate(entropy_per_baryon[::-1][:-1]):
            next = entropy_per_baryon[::-1][ii + 1]
            if el == val:
                found = True
                output = el
                break
            elif next == val:
                found = True
                output = next
                break
            elif (el < val and next > val and ii < 198) or (el > val and next < val and ii < 198):
                found = True
                f = interp1d([next, el], [T[::-1][:-1][ii + 1], T[::-1][:-1][ii]])
                trgt = f(val)
                output = trgt
                break
        if not found:
            output = float("NaN")
        return output

    def find_second(val, T, entropy_per_baryon):
        found = False
        first = True
        output = 0.0
        for ii, el in enumerate(entropy_per_baryon[::-1][:-1]):
            next = entropy_per_baryon[::-1][ii + 1]
            if first:
                if el == val:
                    first = False
                    continue
                elif next == val:
                    first = False
                    continue
                elif (el < val and next > val and ii < 198) or (el > val and next < val and ii < 198):
                    first = False
                    continue
            if not first:
                if el == val:
                    found = True
                    output = el
                    break
                elif next == val:
                    found = True
                    output = next
                    break
                elif (el < val and next > val and ii < 198) or (el > val and next < val and ii < 198):
                    found = True
                    f = interp1d([next, el], [T[::-1][:-1][ii + 1], T[::-1][:-1][ii]])
                    trgt = f(val)
                    output = trgt
                    break
        if not found:
            output = float("NaN")
        return output

    def find_third(val, T, entropy_per_baryon):
        found = False
        first = True
        second = False
        output = 0.0
        for ii, el in enumerate(entropy_per_baryon[::-1][:-1]):
            next = entropy_per_baryon[::-1][ii + 1]
            if first:
                if el == val:
                    first = False
                    second = True
                    continue
                elif next == val:
                    first = False
                    second = True
                    continue
                elif (el < val and next > val and ii < 198) or (el > val and next < val and ii < 198):
                    first = False
                    second = True
                    continue
            if second:
                if el == val:
                    second = False
                    continue
                elif next == val:
                    second = False
                    continue
                elif (el < val and next > val and ii < 198) or (el > val and next < val and ii < 198):
                    second = False
                    continue
            if not first and not second:
                if el == val:
                    found = True
                    output = el
                    break
                elif next == val:
                    found = True
                    output = next
                    break
                elif (el < val and next > val and ii < 198) or (el > val and next < val and ii < 198):
                    found = True
                    f = interp1d([next, el], [T[::-1][:-1][ii + 1], T[::-1][:-1][ii]])
                    trgt = f(val)
                    output = trgt
                    break
        if not found:
            output = float("NaN")
        return output

    T, mu = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))
    mu_t = []
    phi_re, phi_im = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))

    Pres_Q, Pres_g, Pres_pert, Pres_sea = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))
    BDen_Q, BDen_g, BDen_pert, BDen_sea = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))
    SDen_Q, SDen_g, SDen_pert, SDen_sea = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))

    Pres_pi, Pres_K, Pres_rho, Pres_omega = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))
    Pres_D, Pres_N, Pres_T, Pres_F = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))
    Pres_P, Pres_Q5, Pres_H = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))

    BDen_pi, BDen_K, BDen_rho, BDen_omega = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))
    BDen_D, BDen_N, BDen_T, BDen_F = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))
    BDen_P, BDen_Q5, BDen_H = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))

    SDen_pi, SDen_K, SDen_rho, SDen_omega = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))
    SDen_D, SDen_N, SDen_T, SDen_F = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))
    SDen_P, SDen_Q5, SDen_H = numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200)), numpy.zeros(shape = (200,200))

    total_nb = numpy.zeros(shape = (200,200))
    total_s = numpy.zeros(shape = (200,200))
    entropy_per_baryon = numpy.zeros(shape = (200,200))

    epb_50, epb_50_alt, epb_50_alt2 = [], [], []
    epb_25, epb_25_alt, epb_25_alt2 = [], [], []
    epb_10, epb_10_alt, epb_10_alt2 = [], [], []
    epb_5, epb_5_alt, epb_5_alt2 = [], [], []

    epb_pnjl_50, epb_pnjl_50_alt, epb_pnjl_50_alt2 = [], [], []
    epb_pnjl_25, epb_pnjl_25_alt, epb_pnjl_25_alt2 = [], [], []
    epb_pnjl_10, epb_pnjl_10_alt, epb_pnjl_10_alt2 = [], [], []
    epb_pnjl_5, epb_pnjl_5_alt, epb_pnjl_5_alt2 = [], [], []

    const_lines_file = "D:/EoS/epja/figure11/lines.dat"

    calc = False

    if calc:
        for j, mu_trgt in enumerate(numpy.linspace(1.0, 1000.0, num = 200)):

            mu_trgt_round = math.floor(mu_trgt * 10) / 10.0

            pl_file = "D:/EoS/BDK/mu_T_plane/pl_" + str(mu_trgt_round).replace(".", "p") + ".dat"
            pressure_file = "D:/EoS/BDK/mu_T_plane/pressure_" + str(mu_trgt_round).replace(".", "p") + ".dat"
            baryon_file = "D:/EoS/BDK/mu_T_plane/bdensity_" + str(mu_trgt_round).replace(".", "p") + ".dat"
            entropy_file = "D:/EoS/BDK/mu_T_plane/entropy_" + str(mu_trgt_round).replace(".", "p") + ".dat"

            T_temp, mu_temp = utils.data_collect(0, 1, pl_file)
            phi_re_temp, phi_im_temp = utils.data_collect(2, 3, pl_file)

            mu_temp = [el * 3.0 for el in mu_temp]

            Pres_Q_temp, Pres_g_temp = utils.data_collect(2, 3, pressure_file)
            Pres_pert_temp, Pres_pi_temp = utils.data_collect(4, 5, pressure_file)
            Pres_K_temp, Pres_rho_temp = utils.data_collect(6, 7, pressure_file)
            Pres_omega_temp, Pres_D_temp = utils.data_collect(8, 9, pressure_file)
            Pres_N_temp, Pres_T_temp = utils.data_collect(10, 11, pressure_file)
            Pres_F_temp, Pres_P_temp = utils.data_collect(12, 13, pressure_file)
            Pres_Q5_temp, Pres_H_temp = utils.data_collect(14, 15, pressure_file)
            Pres_sea_temp, _ = utils.data_collect(16, 16, pressure_file)

            BDen_Q_temp, BDen_g_temp = utils.data_collect(2, 3, baryon_file)
            BDen_pert_temp, BDen_pi_temp = utils.data_collect(4, 5, baryon_file)
            BDen_K_temp, BDen_rho_temp = utils.data_collect(6, 7, baryon_file)
            BDen_omega_temp, BDen_D_temp = utils.data_collect(8, 9, baryon_file)
            BDen_N_temp, BDen_T_temp = utils.data_collect(10, 11, baryon_file)
            BDen_F_temp, BDen_P_temp = utils.data_collect(12, 13, baryon_file)
            BDen_Q5_temp, BDen_H_temp = utils.data_collect(14, 15, baryon_file)
            BDen_sea_temp, _ = utils.data_collect(16, 16, baryon_file)

            SDen_Q_temp, SDen_g_temp = utils.data_collect(2, 3, entropy_file)
            SDen_pert_temp, SDen_pi_temp = utils.data_collect(4, 5, entropy_file)
            SDen_K_temp, SDen_rho_temp = utils.data_collect(6, 7, entropy_file)
            SDen_omega_temp, SDen_D_temp = utils.data_collect(8, 9, entropy_file)
            SDen_N_temp, SDen_T_temp = utils.data_collect(10, 11, entropy_file)
            SDen_F_temp, SDen_P_temp = utils.data_collect(12, 13, entropy_file)
            SDen_Q5_temp, SDen_H_temp = utils.data_collect(14, 15, entropy_file)
            SDen_sea_temp, _ = utils.data_collect(16, 16, entropy_file)

            T[:,j] = T_temp
            mu[:,j] = mu_temp

            phi_re[:,j] = phi_re_temp
            phi_im[:,j] = phi_im_temp
            Pres_Q[:,j] = Pres_Q_temp
            Pres_g[:,j] = Pres_g_temp
            Pres_pert[:,j] = Pres_pert_temp
            Pres_sea[:,j] = Pres_sea_temp
            BDen_Q[:,j] = BDen_Q_temp
            BDen_g[:,j] = BDen_g_temp
            BDen_pert[:,j] = BDen_pert_temp
            BDen_sea[:,j] = BDen_sea_temp
            SDen_Q[:,j] = SDen_Q_temp
            SDen_g[:,j] = SDen_g_temp
            SDen_pert[:,j] = SDen_pert_temp
            SDen_sea[:,j] = SDen_sea_temp
            Pres_pi[:,j] = Pres_pi_temp
            Pres_K[:,j] = Pres_K_temp
            Pres_rho[:,j] = Pres_rho_temp
            Pres_omega[:,j] = Pres_omega_temp
            Pres_D[:,j] = Pres_D_temp
            Pres_N[:,j] = Pres_N_temp
            Pres_T[:,j] = Pres_T_temp
            Pres_F[:,j] = Pres_F_temp
            Pres_P[:,j] = Pres_P_temp
            Pres_Q5[:,j] = Pres_Q5_temp
            Pres_H[:,j] = Pres_H_temp
            BDen_pi[:,j] = BDen_pi_temp
            BDen_K[:,j] = BDen_K_temp
            BDen_rho[:,j] = BDen_rho_temp
            BDen_omega[:,j] = BDen_omega_temp
            BDen_D[:,j] = BDen_D_temp
            BDen_N[:,j] = BDen_N_temp
            BDen_T[:,j] = BDen_T_temp
            BDen_F[:,j] = BDen_F_temp
            BDen_P[:,j] = BDen_P_temp
            BDen_Q5[:,j] = BDen_Q5_temp
            BDen_H[:,j] = BDen_H_temp
            SDen_pi[:,j] = SDen_pi_temp
            SDen_K[:,j] = SDen_K_temp
            SDen_rho[:,j] = SDen_rho_temp
            SDen_omega[:,j] = SDen_omega_temp
            SDen_D[:,j] = SDen_D_temp
            SDen_N[:,j] = SDen_N_temp
            SDen_T[:,j] = SDen_T_temp
            SDen_F[:,j] = SDen_F_temp
            SDen_P[:,j] = SDen_P_temp
            SDen_Q5[:,j] = SDen_Q5_temp
            SDen_H[:,j] = SDen_H_temp

            total_nb_temp = [sum(el) for el in zip(BDen_Q_temp, BDen_g_temp, BDen_pert_temp, BDen_sea_temp, BDen_pi_temp, BDen_K_temp, BDen_rho_temp, BDen_omega_temp, BDen_D_temp, BDen_N_temp, BDen_T_temp, BDen_F_temp, BDen_P_temp, BDen_Q5_temp, BDen_H_temp)]
            total_s_temp = [sum(el) for el in zip(SDen_Q_temp, SDen_g_temp, SDen_pert_temp, SDen_sea_temp, SDen_pi_temp, SDen_K_temp, SDen_rho_temp, SDen_omega_temp, SDen_D_temp, SDen_N_temp, SDen_T_temp, SDen_F_temp, SDen_P_temp, SDen_Q5_temp, SDen_H_temp)]
            entropy_per_baryon_temp = [el1 / el2 for el1, el2 in zip(total_s_temp, total_nb_temp)]

            pnjl_nb_temp = [sum(el) for el in zip(BDen_Q_temp, BDen_g_temp, BDen_pert_temp, BDen_sea_temp)]
            pnjl_s_temp = [sum(el) for el in zip(SDen_Q_temp, SDen_g_temp, SDen_pert_temp, SDen_sea_temp)]
            pnjl_ent_per_baryon_temp = [el1 / el2 for el1, el2 in zip(pnjl_s_temp, pnjl_nb_temp)]

            #epb_25.append(find_first(25.0, T_temp, entropy_per_baryon_temp))
            #epb_25_alt.append(find_second(25.0, T_temp,
            #entropy_per_baryon_temp))
            #epb_25_alt2.append(find_third(25.0, T_temp,
            #entropy_per_baryon_temp))
            #epb_10.append(find_first(10.0, T_temp, entropy_per_baryon_temp))
            #epb_10_alt.append(find_second(10.0, T_temp,
            #entropy_per_baryon_temp))
            #epb_10_alt2.append(find_third(10.0, T_temp,
            #entropy_per_baryon_temp))
            epb_5.append(find_first(5.0, T_temp, entropy_per_baryon_temp))
            epb_5_alt.append(find_second(5.0, T_temp, entropy_per_baryon_temp))
            epb_5_alt2.append(find_third(5.0, T_temp, entropy_per_baryon_temp))
            #epb_50.append(find_first(50.0, T_temp, entropy_per_baryon_temp))
            #epb_50_alt.append(find_second(50.0, T_temp,
            #entropy_per_baryon_temp))
            #epb_50_alt2.append(find_third(50.0, T_temp,
            #entropy_per_baryon_temp))

            epb_50.append(find_first(100.0, T_temp, entropy_per_baryon_temp))
            epb_50_alt.append(find_second(100.0, T_temp, entropy_per_baryon_temp))
            epb_50_alt2.append(find_third(100.0, T_temp, entropy_per_baryon_temp))
            epb_25.append(find_first(15.0, T_temp, entropy_per_baryon_temp))
            epb_25_alt.append(find_second(15.0, T_temp, entropy_per_baryon_temp))
            epb_25_alt2.append(find_third(15.0, T_temp, entropy_per_baryon_temp))
            epb_10.append(find_first(10.0, T_temp, entropy_per_baryon_temp))
            epb_10_alt.append(find_second(10.0, T_temp, entropy_per_baryon_temp))
            epb_10_alt2.append(find_third(10.0, T_temp, entropy_per_baryon_temp))

            #epb_pnjl_50.append(find_first(50.0, T_temp,
            #pnjl_ent_per_baryon_temp))
            #epb_pnjl_50_alt.append(find_second(50.0, T_temp,
            #pnjl_ent_per_baryon_temp))
            #epb_pnjl_50_alt2.append(find_third(50.0, T_temp,
            #pnjl_ent_per_baryon_temp))
            epb_pnjl_50.append(find_first(100.0, T_temp, pnjl_ent_per_baryon_temp))
            epb_pnjl_50_alt.append(find_second(100.0, T_temp, pnjl_ent_per_baryon_temp))
            epb_pnjl_50_alt2.append(find_third(100.0, T_temp, pnjl_ent_per_baryon_temp))

            #epb_pnjl_25.append(find_first(25.0, T_temp,
            #pnjl_ent_per_baryon_temp))
            #epb_pnjl_25_alt.append(find_second(25.0, T_temp,
            #pnjl_ent_per_baryon_temp))
            #epb_pnjl_25_alt2.append(find_third(25.0, T_temp,
            #pnjl_ent_per_baryon_temp))
            epb_pnjl_25.append(find_first(15.0, T_temp, pnjl_ent_per_baryon_temp))
            epb_pnjl_25_alt.append(find_second(15.0, T_temp, pnjl_ent_per_baryon_temp))
            epb_pnjl_25_alt2.append(find_third(15.0, T_temp, pnjl_ent_per_baryon_temp))

            #epb_pnjl_10.append(find_first(10.0, T_temp,
            #pnjl_ent_per_baryon_temp))
            #epb_pnjl_10_alt.append(find_second(10.0, T_temp,
            #pnjl_ent_per_baryon_temp))
            #epb_pnjl_10_alt2.append(find_third(10.0, T_temp,
            #pnjl_ent_per_baryon_temp))
            epb_pnjl_10.append(find_first(10.0, T_temp, pnjl_ent_per_baryon_temp))
            epb_pnjl_10_alt.append(find_second(10.0, T_temp, pnjl_ent_per_baryon_temp))
            epb_pnjl_10_alt2.append(find_third(10.0, T_temp, pnjl_ent_per_baryon_temp))

            epb_pnjl_5.append(find_first(5.0, T_temp, pnjl_ent_per_baryon_temp))
            epb_pnjl_5_alt.append(find_second(5.0, T_temp, pnjl_ent_per_baryon_temp))
            epb_pnjl_5_alt2.append(find_third(5.0, T_temp, pnjl_ent_per_baryon_temp))

            total_nb[:,j] = total_nb_temp
            total_s[:,j] = total_s_temp
            entropy_per_baryon[:,j] = entropy_per_baryon_temp

        mu_t = mu[0,:]
        with open(const_lines_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[mu_el, e50_el, e50a_el, e50a2_el, e25_el, e25a_el, e25a2_el, e10_el, e10a_el, e10a2_el, e5_el, e5a_el, e5a2_el, e50_pnjl_el, e50a_pnjl_el, e50a2_pnjl_el, e25_pnjl_el, e25a_pnjl_el, e25a2_pnjl_el, e10_pnjl_el, e10a_pnjl_el, e10a2_pnjl_el, e5_pnjl_el, e5a_pnjl_el, e5a2_pnjl_el] for mu_el, e50_el, e50a_el, e50a2_el, e25_el, e25a_el, e25a2_el, e10_el, e10a_el, e10a2_el, e5_el, e5a_el, e5a2_el, e50_pnjl_el, e50a_pnjl_el, e50a2_pnjl_el, e25_pnjl_el, e25a_pnjl_el, e25a2_pnjl_el, e10_pnjl_el, e10a_pnjl_el, e10a2_pnjl_el, e5_pnjl_el, e5a_pnjl_el, e5a2_pnjl_el in zip(mu[0,:], epb_50, epb_50_alt, epb_50_alt2, epb_25, epb_25_alt, epb_25_alt2, epb_10, epb_10_alt, epb_10_alt2, epb_5, epb_5_alt, epb_5_alt2, epb_pnjl_50, epb_pnjl_50_alt, epb_pnjl_50_alt2, epb_pnjl_25, epb_pnjl_25_alt, epb_pnjl_25_alt2, epb_pnjl_10, epb_pnjl_10_alt, epb_pnjl_10_alt2, epb_pnjl_5, epb_pnjl_5_alt, epb_pnjl_5_alt2)])
    else:
        mu_t, epb_50 = utils.data_collect(0, 1, const_lines_file)
        epb_50_alt, epb_50_alt2 = utils.data_collect(2, 3, const_lines_file)
        epb_25, epb_25_alt = utils.data_collect(4, 5, const_lines_file)
        epb_25_alt2, epb_10 = utils.data_collect(6, 7, const_lines_file)
        epb_10_alt, epb_10_alt2 = utils.data_collect(8, 9, const_lines_file)
        epb_5, epb_5_alt = utils.data_collect(10, 11, const_lines_file)
        epb_5_alt2, epb_pnjl_50 = utils.data_collect(12, 13, const_lines_file)
        epb_pnjl_50_alt, epb_pnjl_50_alt2 = utils.data_collect(14, 15, const_lines_file)
        epb_pnjl_25, epb_pnjl_25_alt = utils.data_collect(16, 17, const_lines_file)
        epb_pnjl_25_alt2, epb_pnjl_10 = utils.data_collect(18, 19, const_lines_file)
        epb_pnjl_10_alt, epb_pnjl_10_alt2 = utils.data_collect(20, 21, const_lines_file)
        epb_pnjl_5, epb_pnjl_5_alt = utils.data_collect(22, 23, const_lines_file)
        epb_pnjl_5_alt2, _ = utils.data_collect(24, 24, const_lines_file)

    (Ratti_PoS_2019_figure5_sn48_low_x, Ratti_PoS_2019_figure5_sn48_low_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Ratti_PoS_2019_figure5_sn48_low.dat")
    (Ratti_PoS_2019_figure5_sn48_high_x, Ratti_PoS_2019_figure5_sn48_high_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Ratti_PoS_2019_figure5_sn48_high.dat")
    (Ratti_PoS_2019_figure5_sn68_low_x, Ratti_PoS_2019_figure5_sn68_low_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Ratti_PoS_2019_figure5_sn68_low.dat")
    (Ratti_PoS_2019_figure5_sn68_high_x, Ratti_PoS_2019_figure5_sn68_high_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Ratti_PoS_2019_figure5_sn68_high.dat")
    (Ratti_PoS_2019_figure5_sn94_low_x, Ratti_PoS_2019_figure5_sn94_low_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Ratti_PoS_2019_figure5_sn94_low.dat")
    (Ratti_PoS_2019_figure5_sn94_high_x, Ratti_PoS_2019_figure5_sn94_high_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Ratti_PoS_2019_figure5_sn94_high.dat")
    (Ratti_PoS_2019_figure5_sn144_low_x, Ratti_PoS_2019_figure5_sn144_low_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Ratti_PoS_2019_figure5_sn144_low.dat")
    (Ratti_PoS_2019_figure5_sn144_high_x, Ratti_PoS_2019_figure5_sn144_high_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Ratti_PoS_2019_figure5_sn144_high.dat")
    (Ratti_PoS_2019_figure5_sn420_low_x, Ratti_PoS_2019_figure5_sn420_low_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Ratti_PoS_2019_figure5_sn420_low.dat")
    (Ratti_PoS_2019_figure5_sn420_high_x, Ratti_PoS_2019_figure5_sn420_high_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Ratti_PoS_2019_figure5_sn420_high.dat")
    (Schmidt_EPJC_2009_figure6_sn30_N4_low_x, Schmidt_EPJC_2009_figure6_sn30_N4_low_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Schmidt_EPJC_2009_figure6_sn30_N6_low.dat")
    (Schmidt_EPJC_2009_figure6_sn30_N4_high_x, Schmidt_EPJC_2009_figure6_sn30_N4_high_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Schmidt_EPJC_2009_figure6_sn30_N6_high.dat")
    (Schmidt_EPJC_2009_figure6_sn45_N4_low_x, Schmidt_EPJC_2009_figure6_sn45_N4_low_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Schmidt_EPJC_2009_figure6_sn45_N6_low.dat")
    (Schmidt_EPJC_2009_figure6_sn45_N4_high_x, Schmidt_EPJC_2009_figure6_sn45_N4_high_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Schmidt_EPJC_2009_figure6_sn45_N6_high.dat")
    (Schmidt_EPJC_2009_figure6_sn300_N4_low_x, Schmidt_EPJC_2009_figure6_sn300_N4_low_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Schmidt_EPJC_2009_figure6_sn300_N6_low.dat")
    (Schmidt_EPJC_2009_figure6_sn300_N4_high_x, Schmidt_EPJC_2009_figure6_sn300_N4_high_y) = utils.data_collect(0, 1, "D:/EoS/lattice_data/const_s_over_n/Schmidt_EPJC_2009_figure6_sn300_N6_high.dat")

    Ratti_PoS_2019_figure5_sn48 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(Ratti_PoS_2019_figure5_sn48_high_x, Ratti_PoS_2019_figure5_sn48_high_y)]
    for x_el, y_el in zip(Ratti_PoS_2019_figure5_sn48_low_x[::-1], Ratti_PoS_2019_figure5_sn48_low_y[::-1]):
        Ratti_PoS_2019_figure5_sn48.append(numpy.array([x_el, y_el]))
    Ratti_PoS_2019_figure5_sn48 = numpy.array(Ratti_PoS_2019_figure5_sn48)
    Ratti_PoS_2019_figure5_sn68 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(Ratti_PoS_2019_figure5_sn68_high_x, Ratti_PoS_2019_figure5_sn68_high_y)]
    for x_el, y_el in zip(Ratti_PoS_2019_figure5_sn68_low_x[::-1], Ratti_PoS_2019_figure5_sn68_low_y[::-1]):
        Ratti_PoS_2019_figure5_sn68.append(numpy.array([x_el, y_el]))
    Ratti_PoS_2019_figure5_sn68 = numpy.array(Ratti_PoS_2019_figure5_sn68)
    Ratti_PoS_2019_figure5_sn94 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(Ratti_PoS_2019_figure5_sn94_high_x, Ratti_PoS_2019_figure5_sn94_high_y)]
    for x_el, y_el in zip(Ratti_PoS_2019_figure5_sn94_low_x[::-1], Ratti_PoS_2019_figure5_sn94_low_y[::-1]):
        Ratti_PoS_2019_figure5_sn94.append(numpy.array([x_el, y_el]))
    Ratti_PoS_2019_figure5_sn94 = numpy.array(Ratti_PoS_2019_figure5_sn94)
    Ratti_PoS_2019_figure5_sn144 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(Ratti_PoS_2019_figure5_sn144_high_x, Ratti_PoS_2019_figure5_sn144_high_y)]
    for x_el, y_el in zip(Ratti_PoS_2019_figure5_sn144_low_x[::-1], Ratti_PoS_2019_figure5_sn144_low_y[::-1]):
        Ratti_PoS_2019_figure5_sn144.append(numpy.array([x_el, y_el]))
    Ratti_PoS_2019_figure5_sn144 = numpy.array(Ratti_PoS_2019_figure5_sn144)
    Ratti_PoS_2019_figure5_sn420 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(Ratti_PoS_2019_figure5_sn420_high_x, Ratti_PoS_2019_figure5_sn420_high_y)]
    for x_el, y_el in zip(Ratti_PoS_2019_figure5_sn420_low_x[::-1], Ratti_PoS_2019_figure5_sn420_low_y[::-1]):
        Ratti_PoS_2019_figure5_sn420.append(numpy.array([x_el, y_el]))
    Ratti_PoS_2019_figure5_sn420 = numpy.array(Ratti_PoS_2019_figure5_sn420)
    Schmidt_EPJC_2009_figure6_sn30_N4 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(Schmidt_EPJC_2009_figure6_sn30_N4_high_x, Schmidt_EPJC_2009_figure6_sn30_N4_high_y)]
    for x_el, y_el in zip(Schmidt_EPJC_2009_figure6_sn30_N4_low_x[::-1], Schmidt_EPJC_2009_figure6_sn30_N4_low_y[::-1]):
        Schmidt_EPJC_2009_figure6_sn30_N4.append(numpy.array([x_el, y_el]))
    Schmidt_EPJC_2009_figure6_sn30_N4 = numpy.array(Schmidt_EPJC_2009_figure6_sn30_N4)
    Schmidt_EPJC_2009_figure6_sn45_N4 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(Schmidt_EPJC_2009_figure6_sn45_N4_high_x, Schmidt_EPJC_2009_figure6_sn45_N4_high_y)]
    for x_el, y_el in zip(Schmidt_EPJC_2009_figure6_sn45_N4_low_x[::-1], Schmidt_EPJC_2009_figure6_sn45_N4_low_y[::-1]):
        Schmidt_EPJC_2009_figure6_sn45_N4.append(numpy.array([x_el, y_el]))
    Schmidt_EPJC_2009_figure6_sn45_N4 = numpy.array(Schmidt_EPJC_2009_figure6_sn45_N4)
    Schmidt_EPJC_2009_figure6_sn300_N4 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(Schmidt_EPJC_2009_figure6_sn300_N4_high_x, Schmidt_EPJC_2009_figure6_sn300_N4_high_y)]
    for x_el, y_el in zip(Schmidt_EPJC_2009_figure6_sn300_N4_low_x[::-1], Schmidt_EPJC_2009_figure6_sn300_N4_low_y[::-1]):
        Schmidt_EPJC_2009_figure6_sn300_N4.append(numpy.array([x_el, y_el]))
    Schmidt_EPJC_2009_figure6_sn300_N4 = numpy.array(Schmidt_EPJC_2009_figure6_sn300_N4)

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([0., 750., 20., 500.])

    #ax1.add_patch(
    #    matplotlib.patches.Polygon(
    #        Ratti_PoS_2019_figure5_sn48,
    #        closed = True, fill = True, color = 'blue', alpha = 0.3
    #    )
    #)
    #ax1.add_patch(
    #    matplotlib.patches.Polygon(
    #        Ratti_PoS_2019_figure5_sn68,
    #        closed = True, fill = True, color = 'blue', alpha = 0.3
    #    )
    #)
    #ax1.add_patch(
    #    matplotlib.patches.Polygon(
    #        Ratti_PoS_2019_figure5_sn94,
    #        closed = True, fill = True, color = 'blue', alpha = 0.3
    #    )
    #)
    #ax1.add_patch(
    #    matplotlib.patches.Polygon(
    #        Ratti_PoS_2019_figure5_sn144,
    #        closed = True, fill = True, color = 'blue', alpha = 0.3
    #    )
    #)
    #ax1.add_patch(
    #    matplotlib.patches.Polygon(
    #        Ratti_PoS_2019_figure5_sn420,
    #        closed = True, fill = True, color = 'blue', alpha = 0.3
    #    )
    #)
    ax1.add_patch(matplotlib.patches.Polygon(Schmidt_EPJC_2009_figure6_sn30_N4, 
            closed = True, fill = True, color = 'red', alpha = 0.3))
    ax1.add_patch(matplotlib.patches.Polygon(Schmidt_EPJC_2009_figure6_sn45_N4, 
            closed = True, fill = True, color = 'red', alpha = 0.3))
    ax1.add_patch(matplotlib.patches.Polygon(Schmidt_EPJC_2009_figure6_sn300_N4, 
            closed = True, fill = True, color = 'red', alpha = 0.3))

    epj_50_colleced_x = numpy.append(mu_t[152:][::-1], numpy.append(mu_t[144:151][::-1], numpy.append(mu_t[38:142][::-1], numpy.append(mu_t[9:38][::-1], numpy.append(mu_t[7:9], numpy.append(mu_t[7:9][::-1], mu_t[7:38]))))))
    epj_50_collected_y = numpy.append(epb_50[152:][::-1], numpy.append(epb_50[144:151][::-1], numpy.append(epb_50[38:142][::-1], numpy.append(epb_50_alt[9:38][::-1], numpy.append(epb_50_alt2[7:9], numpy.append(epb_50_alt[7:9][::-1], epb_50[7:38]))))))
    
    epj_25_collected_x = numpy.append(mu_t[152:172][::-1], numpy.append(mu_t[147:151][::-1], numpy.append(mu_t[79:142][::-1], numpy.append(mu_t[58:78][::-1], numpy.append(mu_t[40:46][::-1], numpy.append(mu_t[40:46], numpy.append(mu_t[46:58], numpy.append(mu_t[46:58][::-1], mu_t[46:]))))))))
    epj_25_collected_y = numpy.append(epb_25_alt[152:172][::-1], numpy.append(epb_25_alt[147:151][::-1], numpy.append(epb_25_alt[79:142][::-1], numpy.append(epb_25_alt[58:78][::-1], numpy.append(epb_25_alt[40:46][::-1], numpy.append(epb_25[40:46], numpy.append(epb_25_alt2[46:58], numpy.append(epb_25_alt[46:58][::-1], epb_25[46:]))))))))
    epj_25_collected_x = [el for el, test in zip(epj_25_collected_x, epj_25_collected_y) if not numpy.isnan(test)]
    epj_25_collected_y = [el for el in epj_25_collected_y if not numpy.isnan(el)]

    epj_10_collected_x = numpy.append(mu_t[152:175][::-1], numpy.append(mu_t[147:151][::-1], numpy.append(mu_t[91:142][::-1], numpy.append(mu_t[55:70][::-1], numpy.append(mu_t[55:70], numpy.append(mu_t[70:91], numpy.append(mu_t[70:91][::-1], mu_t[70:])))))))
    epj_10_collected_y = numpy.append(epb_10_alt[152:175][::-1], numpy.append(epb_10_alt[147:151][::-1], numpy.append(epb_10_alt[91:142][::-1], numpy.append(epb_10_alt[55:70][::-1], numpy.append(epb_10[55:70], numpy.append(epb_10_alt2[70:91], numpy.append(epb_10_alt[70:91][::-1], epb_10[70:])))))))

    epj_pnjl_50_collected_x = numpy.append(mu_t[2:7], numpy.append(mu_t[7:10], numpy.append(mu_t[7:10][::-1], mu_t[7:])))
    epj_pnjl_50_collected_y = numpy.append(epb_pnjl_50[2:7], numpy.append(epb_pnjl_50_alt2[7:10], numpy.append(epb_pnjl_50_alt[7:10][::-1], epb_pnjl_50[7:])))
    epj_pnjl_50_collected_x = [el for el, test in zip(epj_pnjl_50_collected_x, epj_pnjl_50_collected_y) if not numpy.isnan(test)]
    epj_pnjl_50_collected_y = [el for el in epj_pnjl_50_collected_y if not numpy.isnan(el)]

    epj_pnjl_25_collected_x = numpy.append([1000.0], numpy.append(mu_t[60:108][::-1], numpy.append(mu_t[25:48][::-1], numpy.append(mu_t[25:48], numpy.append(mu_t[48:60], numpy.append(mu_t[48:60][::-1], mu_t[48:]))))))
    epj_pnjl_25_collected_y = numpy.append([5.0], numpy.append(epb_pnjl_25_alt[60:108][::-1], numpy.append(epb_pnjl_25_alt[25:48][::-1], numpy.append(epb_pnjl_25[25:48], numpy.append(epb_pnjl_25_alt2[48:60], numpy.append(epb_pnjl_25_alt[48:60][::-1], epb_pnjl_25[48:]))))))

    epj_pnjl_10_collected_x = numpy.append(mu_t[98:][::-1], numpy.append(mu_t[44:73][::-1], numpy.append(mu_t[44:73], numpy.append(mu_t[73:98], numpy.append(mu_t[73:98][::-1], mu_t[73:])))))
    epj_pnjl_10_collected_y = numpy.append(epb_pnjl_10_alt[98:][::-1], numpy.append(epb_pnjl_10_alt[44:73][::-1], numpy.append(epb_pnjl_10[44:73], numpy.append(epb_pnjl_10_alt2[73:98], numpy.append(epb_pnjl_10_alt[73:98][::-1], epb_pnjl_10[73:])))))

    ax1.plot(epj_pnjl_50_collected_x[:10], epj_pnjl_50_collected_y[:10], '--', c = 'black')
    ax1.plot(epj_50_colleced_x[:202], epj_50_collected_y[:202], '-',c = 'black')
    ax1.plot(epj_50_colleced_x, epj_50_collected_y, ':',c = 'black')
    ax1.plot(epj_pnjl_25_collected_x[:120], epj_pnjl_25_collected_y[:120], '--',c = 'blue')
    ax1.plot(epj_25_collected_x[:202], epj_25_collected_y[:202], '-',c = 'blue')
    ax1.plot(epj_25_collected_x, epj_25_collected_y, ':',c = 'blue')
    ax1.plot(epj_pnjl_10_collected_x[:220], epj_pnjl_10_collected_y[:220], '--',c = 'green')
    ax1.plot(epj_10_collected_x[:202], epj_10_collected_y[:202], '-',c = 'green')
    ax1.plot(epj_10_collected_x, epj_10_collected_y, ':',c = 'green')
    #ax1.plot(numpy.linspace(0, 1700, num = 300), [0.52 * el for el in
    #numpy.linspace(0, 1700, num = 300)], '-', c = 'magenta')

    (sn331_x, sn331_y) = utils.data_collect(0, 1, "D:/EoS/sn331.dat")
    sn331_x = [el * 3.0 for el in sn331_x]
    sn331_y, sn331_x = [list(tuple) for tuple in zip(*sorted(zip(sn331_y, sn331_x)))]
    (sn277_x, sn277_y) = utils.data_collect(0, 1, "D:/EoS/sn277.dat")
    sn277_x = [el * 3.0 for el in sn277_x]
    sn2771_y, sn277_x = [list(tuple) for tuple in zip(*sorted(zip(sn277_y, sn277_x)))]
    (sn123_x, sn123_y) = utils.data_collect(0, 1, "D:/EoS/sn123.dat")
    sn123_x = [el * 3.0 for el in sn123_x]
    sn123_y, sn123_x = [list(tuple) for tuple in zip(*sorted(zip(sn123_y, sn123_x)))]
    (sn84_x, sn84_y) = utils.data_collect(0, 1, "D:/EoS/sn84.dat")
    sn84_x = [el * 3.0 for el in sn84_x]
    sn84_y, sn84_x = [list(tuple) for tuple in zip(*sorted(zip(sn84_y, sn84_x)))]
    (sn56_x, sn56_y) = utils.data_collect(0, 1, "D:/EoS/sn56.dat")
    sn56_x = [el * 3.0 for el in sn56_x]
    sn56_y, sn56_x = [list(tuple) for tuple in zip(*sorted(zip(sn56_y, sn56_x)))]
    (sn45_x, sn45_y) = utils.data_collect(0, 1, "D:/EoS/sn45.dat")
    sn45_x = [el * 3.0 for el in sn45_x]
    sn45_y, sn45_x = [list(tuple) for tuple in zip(*sorted(zip(sn45_y, sn45_x)))]
    (sn26_x, sn26_y) = utils.data_collect(0, 1, "D:/EoS/sn26.dat")
    sn26_x = [el * 3.0 for el in sn26_x]
    sn26_y, sn26_x = [list(tuple) for tuple in zip(*sorted(zip(sn26_y, sn26_x)))]

    ax1.plot(mu_t, [pnjl.thermo.gcp_sea_lattice.Tc(el / 3.0) for el in mu_t], ':', c = 'black')

    ax1.text(110.0, 465.0, r'T/${\rm \mu}$=5.3', c = 'black', fontsize = 14)
    ax1.text(583.0, 431.0, r'T/${\rm \mu}$=0.79', c = 'blue', fontsize = 14)
    ax1.text(610.0, 300.0, r'T/${\rm \mu}$=0.52', c = 'green', fontsize = 14)
    ax1.text(65.0, 270.0, r's/n=300', c = 'black', fontsize = 14)
    ax1.text(360.0, 270.0, r's/n=45', c = 'blue', fontsize = 14)
    ax1.text(512.0, 242.0, r's/n=30', c = 'green', fontsize = 14)
    ax1.text(95, 370, r'Schmidt et al. (2009)', c = 'red', alpha = 0.5, fontsize = 14)
    ax1.text(675, 121, r'$\mathrm{T_c(\mu)}$', c = 'black', fontsize = 14)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'$\mathrm{\mu_B}$ [MeV]', fontsize = 16)
    ax1.set_ylabel(r'T [MeV]', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def epja_figure13():

    phi_re_1, phi_im_1 = [], []
    phi_re_2, phi_im_2 = [], []

    Chi4_Q_1, Chi4_pert_1 = [], []
    Chi4_Q_2, Chi4_pert_2 = [], []

    Chi4_pi_1, Chi4_K_1, Chi4_rho_1 = [], [], []
    Chi4_omega_1, Chi4_D_1, Chi4_N_1 = [], [], []
    Chi4_T_1, Chi4_F_1, Chi4_P_1 = [], [], []
    Chi4_Q5_1, Chi4_H_1 = [], []

    Chi4_pi_2, Chi4_K_2, Chi4_rho_2 = [], [], [] 
    Chi4_omega_2, Chi4_D_2, Chi4_N_2 = [], [], []
    Chi4_T_2, Chi4_F_2, Chi4_P_2 = [], [], []
    Chi4_Q5_2, Chi4_H_2 = [], []

    Chi2_Q_1, Chi2_pert_1 = [], []
    Chi2_Q_2, Chi2_pert_2 = [], []

    Chi2_pi_1, Chi2_K_1, Chi2_rho_1 = [], [], []
    Chi2_omega_1, Chi2_D_1, Chi2_N_1 = [], [], []
    Chi2_T_1, Chi2_F_1, Chi2_P_1 = [], [], []
    Chi2_Q5_1, Chi2_H_1 = [], []

    Chi2_pi_2, Chi2_K_2, Chi2_rho_2 = [], [], [] 
    Chi2_omega_2, Chi2_D_2, Chi2_N_2 = [], [], []
    Chi2_T_2, Chi2_F_2, Chi2_P_2 = [], [], []
    Chi2_Q5_2, Chi2_H_2 = [], []

    T_1 = numpy.linspace(1.0, 300.0, 200)
    T_2 = numpy.linspace(1.0, 300.0, 200)

    mu_1 = [1.0 for el in T_1]
    mu_2 = [0.0 for el in T_2]

    calc_1 = False
    calc_2 = False

    cluster_backreaction = False

    pl_1_file = "D:/EoS/epja/figure13/pl_1.dat"
    pl_2_file = "D:/EoS/epja/figure13/pl_2.dat"
    chi_1_file = "D:/EoS/epja/figure13/chi_1.dat"
    chi_2_file = "D:/EoS/epja/figure13/chi_2.dat"

    if False:
        phi_re_1.append(1e-15)
        phi_im_1.append(2e-15)
        lT = len(T_1)
        for T_el, mu_el in tqdm.tqdm(zip(T_1, mu_1), desc = "Traced Polyakov loop (calc #1)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_1[-1], phi_im_1[-1], with_clusters = cluster_backreaction)
            phi_re_1.append(temp_phi_re)
            phi_im_1.append(temp_phi_im)
        phi_re_1 = phi_re_1[1:]
        phi_im_1 = phi_im_1[1:]
        with open(pl_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_1, mu_1, phi_re_1, phi_im_1)])
    else:
        T_1, mu_1 = utils.data_collect(0, 1, pl_1_file)
        phi_re_1, phi_im_1 = utils.data_collect(2, 3, pl_1_file)

    if False:
        phi_re_2.append(1e-15)
        phi_im_2.append(2e-15)
        lT = len(T_2)
        for T_el, mu_el in tqdm.tqdm(zip(T_2, mu_2), desc = "Traced Polyakov loop (calc #2)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_2[-1], phi_im_2[-1], with_clusters = cluster_backreaction)
            phi_re_2.append(temp_phi_re)
            phi_im_2.append(temp_phi_im)
        phi_re_2 = phi_re_2[1:]
        phi_im_2 = phi_im_2[1:]
        with open(pl_2_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_2, mu_2, phi_re_2, phi_im_2)])
    else:
        T_2, mu_2 = utils.data_collect(0, 1, pl_2_file)
        phi_re_2, phi_im_2 = utils.data_collect(2, 3, pl_2_file)

    if calc_1:
        if os.path.exists("D:/EoS/epja/figure13/chi_1_Q.dat"):
            Chi2_Q_1, Chi4_Q_1 = utils.data_collect(0, 1, "D:/EoS/epja/figure13/chi_1_Q.dat")
        else:
            Chi2_Q_1 = [
                pnjl.thermo.gcp_quark.bnumber_cumulant(2, T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.bnumber_cumulant(2, T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms)
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Quark chiB2 (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            Chi4_Q_1 = [
                pnjl.thermo.gcp_quark.bnumber_cumulant(4, T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.bnumber_cumulant(4, T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms)
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Quark chiB4 (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            with open("D:/EoS/epja/figure13/chi_1_Q.dat", 'w', newline = '') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerows([[el2, el4] for el2, el4 in zip(Chi2_Q_1, Chi4_Q_1)])
        if os.path.exists("D:/EoS/epja/figure13/chi_1_pert.dat"):
            Chi2_pert_1, Chi4_pert_1 = utils.data_collect(0, 1, "D:/EoS/epja/figure13/chi_1_pert.dat")
        else:
            Chi2_pert_1 = [
                pnjl.thermo.gcp_perturbative.bnumber_cumulant(2, T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0)
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Perturbative chiB2 (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            Chi4_pert_1 = [
                pnjl.thermo.gcp_perturbative.bnumber_cumulant(4, T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0)
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Perturbative chiB4 (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            with open("D:/EoS/epja/figure13/chi_1_pert.dat", 'w', newline = '') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerows([[el2, el4] for el2, el4 in zip(Chi2_pert_1, Chi4_pert_1)])
        (regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2,
         regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2,
         regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52) = cluster_regulator_calc(mu_1[0])
        (Chi2_pi_1, Chi2_rho_1, Chi2_omega_1, Chi2_K_1, Chi2_D_1, Chi2_N_1, Chi2_T_1, Chi2_F_1, 
         Chi2_P_1, Chi2_Q5_1, Chi2_H_1) = cumulants_c("D:/EoS/epja/figure13/chi_1", 2, T_1, mu_1, phi_re_1, phi_im_1, regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2, regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2, regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52)
        (Chi4_pi_1, Chi4_rho_1, Chi4_omega_1, Chi4_K_1, Chi4_D_1, Chi4_N_1, Chi4_T_1, Chi4_F_1, 
         Chi4_P_1, Chi4_Q5_1, Chi4_H_1) = cumulants_c("D:/EoS/epja/figure13/chi_1", 4, T_1, mu_1, phi_re_1, phi_im_1, regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2, regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2, regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52)
        for el in glob.glob("D:/EoS/epja/figure13/chi_1_*.dat"): os.remove(el)
        with open(chi_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q2_el, q4_el, pert2_el, pert4_el, pi2_el, pi4_el, k2_el, k4_el, rho2_el, rho4_el, omega2_el, omega4_el, D2_el, D4_el, N2_el, N4_el, T2_el, T4_el, F2_el, F4_el, P2_el, P4_el, Q52_el, Q54_el, H2_el, H4_el] for q2_el, pert2_el, pi2_el, k2_el, rho2_el, omega2_el, D2_el, N2_el, T2_el, F2_el, P2_el, Q52_el, H2_el, q4_el, pert4_el, pi4_el, k4_el, rho4_el, omega4_el, D4_el, N4_el, T4_el, F4_el, P4_el, Q54_el, H4_el in zip(Chi2_Q_1, Chi2_pert_1, Chi2_pi_1, Chi2_K_1, Chi2_rho_1, Chi2_omega_1, Chi2_D_1, Chi2_N_1, Chi2_T_1, Chi2_F_1, Chi2_P_1, Chi2_Q5_1, Chi2_H_1, Chi4_Q_1, Chi4_pert_1, Chi4_pi_1, Chi4_K_1, Chi4_rho_1, Chi4_omega_1, Chi4_D_1, Chi4_N_1, Chi4_T_1, Chi4_F_1, Chi4_P_1, Chi4_Q5_1, Chi4_H_1)])
    else:
        Chi2_Q_1, Chi4_Q_1 = utils.data_collect(0, 1, chi_1_file)
        Chi2_pert_1, Chi4_pert_1 = utils.data_collect(2, 3, chi_1_file)
        Chi2_pi_1, Chi4_pi_1 = utils.data_collect(4, 5, chi_1_file)
        Chi2_K_1, Chi4_K_1 = utils.data_collect(6, 7, chi_1_file)
        Chi2_rho_1, Chi4_rho_1 = utils.data_collect(8, 9, chi_1_file)
        Chi2_omega_1, Chi4_omega_1 = utils.data_collect(10, 11, chi_1_file)
        Chi2_D_1, Chi4_D_1 = utils.data_collect(12, 13, chi_1_file)
        Chi2_N_1, Chi4_N_1 = utils.data_collect(14, 15, chi_1_file)
        Chi2_T_1, Chi4_T_1 = utils.data_collect(16, 17, chi_1_file)
        Chi2_F_1, Chi4_F_1 = utils.data_collect(18, 19, chi_1_file)
        Chi2_P_1, Chi4_P_1 = utils.data_collect(20, 21, chi_1_file)
        Chi2_Q5_1, Chi4_Q5_1 = utils.data_collect(22, 23, chi_1_file)
        Chi2_H_1, Chi4_H_1 = utils.data_collect(24, 25, chi_1_file)

    #if calc_2:
    #    if os.path.exists("D:/EoS/epja/figure13/chi_2_Q.dat"):
    #        Chi2_Q_2, Chi4_Q_2 = utils.data_collect(0, 1,
    #        "D:/EoS/epja/figure13/chi_2_Q.dat")
    #    else:
    #        Chi2_Q_2 = [
    #            pnjl.thermo.gcp_quark.bnumber_cumulant(2, T_el, mu_el,
    #            complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el))
    #            + pnjl.thermo.gcp_quark.bnumber_cumulant(2, T_el, mu_el,
    #            complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el),
    #            Nf = 1.0, ml = pnjl.defaults.default_ms)
    #            for T_el, mu_el, phi_re_el, phi_im_el
    #            in tqdm.tqdm(
    #                zip(T_2, mu_2, phi_re_2, phi_im_2),
    #                desc = "Quark chiB2 (calc #2)",
    #                total = len(T_2),
    #                ascii = True
    #                )]
    #        Chi4_Q_2 = [
    #            pnjl.thermo.gcp_quark.bnumber_cumulant(4, T_el, mu_el,
    #            complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el))
    #            + pnjl.thermo.gcp_quark.bnumber_cumulant(4, T_el, mu_el,
    #            complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el),
    #            Nf = 1.0, ml = pnjl.defaults.default_ms)
    #            for T_el, mu_el, phi_re_el, phi_im_el
    #            in tqdm.tqdm(
    #                zip(T_2, mu_2, phi_re_2, phi_im_2),
    #                desc = "Quark chiB4 (calc #2)",
    #                total = len(T_2),
    #                ascii = True
    #                )]
    #        with open("D:/EoS/epja/figure13/chi_2_Q.dat", 'w', newline = '')
    #        as file:
    #            writer = csv.writer(file, delimiter = '\t')
    #            writer.writerows([[el2, el4] for el2, el4 in zip(Chi2_Q_2,
    #            Chi4_Q_2)])
    #    if os.path.exists("D:/EoS/epja/figure13/chi_2_pert.dat"):
    #        Chi2_pert_2, Chi4_pert_2 = utils.data_collect(0, 1,
    #        "D:/EoS/epja/figure13/chi_2_pert.dat")
    #    else:
    #        Chi2_pert_2 = [
    #            pnjl.thermo.gcp_perturbative.bnumber_cumulant(2, T_el, mu_el,
    #            complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el),
    #            Nf = 3.0)
    #            for T_el, mu_el, phi_re_el, phi_im_el
    #            in tqdm.tqdm(
    #                zip(T_2, mu_2, phi_re_2, phi_im_2),
    #                desc = "Perturbative chiB2 (calc #2)",
    #                total = len(T_2),
    #                ascii = True
    #                )]
    #        Chi4_pert_2 = [
    #            pnjl.thermo.gcp_perturbative.bnumber_cumulant(4, T_el, mu_el,
    #            complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el),
    #            Nf = 3.0)
    #            for T_el, mu_el, phi_re_el, phi_im_el
    #            in tqdm.tqdm(
    #                zip(T_2, mu_2, phi_re_2, phi_im_2),
    #                desc = "Perturbative chiB4 (calc #2)",
    #                total = len(T_2),
    #                ascii = True
    #                )]
    #        with open("D:/EoS/epja/figure13/chi_2_pert.dat", 'w', newline =
    #        '') as file:
    #            writer = csv.writer(file, delimiter = '\t')
    #            writer.writerows([[el2, el4] for el2, el4 in zip(Chi2_pert_2,
    #            Chi4_pert_2)])
    #    (regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H,
    #    regulator_H2, regulator_pi, regulator_pi2,
    #     regulator_K, regulator_K2, regulator_rho, regulator_rho2,
    #     regulator_omega, regulator_omega2,
    #     regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F,
    #     regulator_F2, regulator_Q5, regulator_Q52) =
    #     cluster_regulator_calc(mu_1[0])
    #    (Chi2_pi_2, Chi2_rho_2, Chi2_omega_2, Chi2_K_2, Chi2_D_2, Chi2_N_2,
    #    Chi2_T_2, Chi2_F_2,
    #     Chi2_P_2, Chi2_Q5_2, Chi2_H_2) =
    #     cumulants_c("D:/EoS/epja/figure13/chi_2", 2, T_2, mu_2, phi_re_2,
    #     phi_im_2, regulator_N, regulator_N2, regulator_P, regulator_P2,
    #     regulator_H, regulator_H2, regulator_pi, regulator_pi2, regulator_K,
    #     regulator_K2, regulator_rho, regulator_rho2, regulator_omega,
    #     regulator_omega2, regulator_T, regulator_T2, regulator_D,
    #     regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52)
    #    (Chi4_pi_2, Chi4_rho_2, Chi4_omega_2, Chi4_K_2, Chi4_D_2, Chi4_N_2,
    #    Chi4_T_2, Chi4_F_2,
    #     Chi4_P_2, Chi4_Q5_2, Chi4_H_2) =
    #     cumulants_c("D:/EoS/epja/figure13/chi_2", 4, T_2, mu_2, phi_re_2,
    #     phi_im_2, regulator_N, regulator_N2, regulator_P, regulator_P2,
    #     regulator_H, regulator_H2, regulator_pi, regulator_pi2, regulator_K,
    #     regulator_K2, regulator_rho, regulator_rho2, regulator_omega,
    #     regulator_omega2, regulator_T, regulator_T2, regulator_D,
    #     regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52)
    #    for el in glob.glob("D:/EoS/epja/figure13/chi_2_*.dat"): os.remove(el)
    #    with open(chi_2_file, 'w', newline = '') as file:
    #        writer = csv.writer(file, delimiter = '\t')
    #        writer.writerows([[q2_el, q4_el, pert2_el, pert4_el, pi2_el,
    #        pi4_el, k2_el, k4_el, rho2_el, rho4_el, omega2_el, omega4_el,
    #        D2_el, D4_el, N2_el, N4_el, T2_el, T4_el, F2_el, F4_el, P2_el,
    #        P4_el, Q52_el, Q54_el, H2_el, H4_el] for q2_el, pert2_el, pi2_el,
    #        k2_el, rho2_el, omega2_el, D2_el, N2_el, T2_el, F2_el, P2_el,
    #        Q52_el, H2_el, q4_el, pert4_el, pi4_el, k4_el, rho4_el, omega4_el,
    #        D4_el, N4_el, T4_el, F4_el, P4_el, Q54_el, H4_el in zip(Chi2_Q_2,
    #        Chi2_pert_2, Chi2_pi_2, Chi2_K_2, Chi2_rho_2, Chi2_omega_2,
    #        Chi2_D_2, Chi2_N_2, Chi2_T_2, Chi2_F_2, Chi2_P_2, Chi2_Q5_2,
    #        Chi2_H_2, Chi4_Q_2, Chi4_pert_2, Chi4_pi_2, Chi4_K_2, Chi4_rho_2,
    #        Chi4_omega_2, Chi4_D_2, Chi4_N_2, Chi4_T_2, Chi4_F_2, Chi4_P_2,
    #        Chi4_Q5_2, Chi4_H_2)])
    #else:
    #    Chi2_Q_2, Chi4_Q_2 = utils.data_collect(0, 1, chi_2_file)
    #    Chi2_pert_2, Chi4_pert_2 = utils.data_collect(2, 3, chi_2_file)
    #    Chi2_pi_2, Chi4_pi_2 = utils.data_collect(4, 5, chi_2_file)
    #    Chi2_K_2, Chi4_K_2 = utils.data_collect(6, 7, chi_2_file)
    #    Chi2_rho_2, Chi4_rho_2 = utils.data_collect(8, 9, chi_2_file)
    #    Chi2_omega_2, Chi4_omega_2 = utils.data_collect(10, 11, chi_2_file)
    #    Chi2_D_2, Chi4_D_2 = utils.data_collect(12, 13, chi_2_file)
    #    Chi2_N_2, Chi4_N_2 = utils.data_collect(14, 15, chi_2_file)
    #    Chi2_T_2, Chi4_T_2 = utils.data_collect(16, 17, chi_2_file)
    #    Chi2_F_2, Chi4_F_2 = utils.data_collect(18, 19, chi_2_file)
    #    Chi2_P_2, Chi4_P_2 = utils.data_collect(20, 21, chi_2_file)
    #    Chi2_Q5_2, Chi4_Q5_2 = utils.data_collect(22, 23, chi_2_file)
    #    Chi2_H_2, Chi4_H_2 = utils.data_collect(24, 25, chi_2_file)

    total_chi4_1 = [sum(el) for el in zip(Chi4_Q_1, Chi4_pert_1, Chi4_pi_1, Chi4_K_1, Chi4_rho_1, Chi4_omega_1, Chi4_D_1, Chi4_N_1, Chi4_T_1, Chi4_F_1, Chi4_P_1, Chi4_Q5_1, Chi4_H_1)]
    total_chi2_1 = [sum(el) for el in zip(Chi2_Q_1, Chi2_pert_1, Chi2_pi_1, Chi2_K_1, Chi2_rho_1, Chi2_omega_1, Chi2_D_1, Chi2_N_1, Chi2_T_1, Chi2_F_1, Chi2_P_1, Chi2_Q5_1, Chi2_H_1)]
    total_42ratio_1 = [el1 / el2 for el1, el2 in zip(total_chi4_1, total_chi2_1)]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([min(T_1), max(T_1), min(total_42ratio_1), max(total_42ratio_1)])
    ax1.plot(T_1, total_42ratio_1, '-', c = 'black')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{\chi^B_4 / \chi^B_2}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def cumulant_test():

    phi_re_1, phi_im_1 = [], []

    Chi2_Q_1, Chi2_pert_1 = [], []

    Chi2_pi_1, Chi2_K_1, Chi2_rho_1 = [], [], []
    Chi2_omega_1, Chi2_D_1, Chi2_N_1 = [], [], []
    Chi2_T_1, Chi2_F_1, Chi2_P_1 = [], [], []
    Chi2_Q5_1, Chi2_H_1 = [], []

    Pres_Q_1, Pres_pert_1, Pres_g_1 = [], [], []

    Pres_pi_1, Pres_K_1, Pres_rho_1 = [], [], []
    Pres_omega_1, Pres_D_1, Pres_N_1 = [], [], []
    Pres_T_1, Pres_F_1, Pres_P_1 = [], [], []
    Pres_Q5_1, Pres_H_1 = [], []

    BDen_Q_1, BDen_pert_1 = [], []

    BDen_pi_1, BDen_K_1, BDen_rho_1 = [], [], []
    BDen_omega_1, BDen_D_1, BDen_N_1 = [], [], []
    BDen_T_1, BDen_F_1, BDen_P_1 = [], [], []
    BDen_Q5_1, BDen_H_1 = [], []

    T_1 = numpy.linspace(1.0, 320.0, 200)

    mu_1 = [0.4 * el for el in T_1]

    calc_1 = False

    cluster_backreaction = False

    pl_1_file = "D:/EoS/epja/cumulant_test/pl_1.dat"
    thermo_1_file = "D:/EoS/epja/cumulant_test/thermo_1.dat"

    if False:
        phi_re_1.append(1e-15)
        phi_im_1.append(2e-15)
        lT = len(T_1)
        for T_el, mu_el in tqdm.tqdm(zip(T_1, mu_1), desc = "Traced Polyakov loop (calc #1)", total = lT, ascii = True):
            temp_phi_re, temp_phi_im = calc_PL_c(T_el, mu_el, phi_re_1[-1], phi_im_1[-1], with_clusters = cluster_backreaction)
            phi_re_1.append(temp_phi_re)
            phi_im_1.append(temp_phi_im)
        phi_re_1 = phi_re_1[1:]
        phi_im_1 = phi_im_1[1:]
        with open(pl_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, mu_el, phi_re_el, phi_im_el] for T_el, mu_el, phi_re_el, phi_im_el in zip(T_1, mu_1, phi_re_1, phi_im_1)])
    else:
        T_1, mu_1 = utils.data_collect(0, 1, pl_1_file)
        phi_re_1, phi_im_1 = utils.data_collect(2, 3, pl_1_file)

    if calc_1:
        if os.path.exists("D:/EoS/epja/cumulant_test/temp_Q_1.dat"):
            Pres_Q_1, BDen_Q_1 = utils.data_collect(0, 1, "D:/EoS/epja/cumulant_test/temp_Q_1.dat")
            Chi2_Q_1, _ = utils.data_collect(2, 2, "D:/EoS/epja/cumulant_test/temp_Q_1.dat")
        else:
            Chi2_Q_1 = [
                pnjl.thermo.gcp_quark.bnumber_cumulant(2, T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.bnumber_cumulant(2, T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms)
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Quark chiB2 (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            Pres_Q_1 = [
                pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms)
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Quark pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            BDen_Q_1 = [
                pnjl.thermo.gcp_quark.bdensity(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + pnjl.thermo.gcp_quark.bdensity(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = pnjl.defaults.default_ms)
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Quark baryon density (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            with open("D:/EoS/epja/cumulant_test/temp_Q_1.dat", 'w', newline = '') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerows([[p_el, bd_el, chi_el] for p_el, bd_el, chi_el in zip(Pres_Q_1, BDen_Q_1, Chi2_Q_1)])
        if os.path.exists("D:/EoS/epja/cumulant_test/temp_g_1.dat"):
            Pres_g_1, _ = utils.data_collect(0, 0, "D:/EoS/epja/cumulant_test/temp_g_1.dat")
        else:
            Pres_g_1 = [
                pnjl.thermo.gcp_pl_polynomial.pressure(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) 
                for T_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, phi_re_1, phi_im_1), 
                    desc = "Gluon pressure (calc #1)", 
                    total = len(T_1),
                    ascii = True)]
            with open("D:/EoS/epja/cumulant_test/temp_g_1.dat", 'w', newline = '') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerows([[p_el] for p_el in Pres_g_1])
        if os.path.exists("D:/EoS/epja/cumulant_test/temp_pert_1.dat"):
            Pres_pert_1, BDen_pert_1 = utils.data_collect(0, 1, "D:/EoS/epja/cumulant_test/temp_pert_1.dat")
            Chi2_pert_1, _ = utils.data_collect(2, 2, "D:/EoS/epja/cumulant_test/temp_pert_1.dat")
        else:
            Chi2_pert_1 = [
                pnjl.thermo.gcp_perturbative.bnumber_cumulant(2, T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0)
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Perturbative chiB2 (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            Pres_pert_1 = [
                pnjl.thermo.gcp_perturbative.pressure(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0)
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Perturbative pressure (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            BDen_pert_1 = [
                pnjl.thermo.gcp_perturbative.bdensity(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 3.0)
                for T_el, mu_el, phi_re_el, phi_im_el 
                in tqdm.tqdm(zip(T_1, mu_1, phi_re_1, phi_im_1), 
                    desc = "Perturbative baryon density (calc #1)", 
                    total = len(T_1), 
                    ascii = True)]
            with open("D:/EoS/epja/cumulant_test/temp_pert_1.dat", 'w', newline = '') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerows([[p_el, bd_el, chi_el] for p_el, bd_el, chi_el in zip(Pres_pert_1, BDen_pert_1, Chi2_pert_1)])

        for i, (T_el, mu_el, phi_re_el, phi_im_el) in enumerate(zip(T_1, mu_1, phi_re_1, phi_im_1)):
            print(i + 1, "/", len(T_1))
            (regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2,
             regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2,
             regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52) = cluster_regulator_calc(mu_el)
            (Chi2_pi_1_tmp, Chi2_rho_1_tmp, Chi2_omega_1_tmp, Chi2_K_1_tmp, Chi2_D_1_tmp, Chi2_N_1_tmp, Chi2_T_1_tmp, Chi2_F_1_tmp, 
             Chi2_P_1_tmp, Chi2_Q5_1_tmp, Chi2_H_1_tmp) = cumulants_c("D:/EoS/epja/cumulant_test/temp_chi2_1", 2, [T_el], [mu_el], [phi_re_el], [phi_im_el], regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2, regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2, regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52)
            ((Pres_pi_1_tmp, Pres_rho_1_tmp, Pres_omega_1_tmp, Pres_K_1_tmp, Pres_D_1_tmp, Pres_N_1_tmp, Pres_T_1_tmp, Pres_F_1_tmp, Pres_P_1_tmp, 
                 Pres_Q5_1_tmp, Pres_H_1_tmp),
                (BDen_pi_1_tmp, BDen_rho_1_tmp, BDen_omega_1_tmp, BDen_K_1_tmp, BDen_D_1_tmp, BDen_N_1_tmp, BDen_T_1_tmp, BDen_F_1_tmp, BDen_P_1_tmp, 
                 BDen_Q5_1_tmp, BDen_H_1_tmp),
                (_, _, _, _, _, _, _, _, _, _, _)) = clusters_c("D:/EoS/epja/cumulant_test/temp_thermo_1", [T_el], [mu_el], [phi_re_el], [phi_im_el], regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2, regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2, regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52)
            
            Chi2_pi_1.append(Chi2_pi_1_tmp[0])
            Chi2_rho_1.append(Chi2_rho_1_tmp[0])
            Chi2_omega_1.append(Chi2_omega_1_tmp[0])
            Chi2_K_1.append(Chi2_K_1_tmp[0])
            Chi2_D_1.append(Chi2_D_1_tmp[0])
            Chi2_N_1.append(Chi2_N_1_tmp[0])
            Chi2_T_1.append(Chi2_T_1_tmp[0])
            Chi2_F_1.append(Chi2_F_1_tmp[0])
            Chi2_P_1.append(Chi2_P_1_tmp[0])
            Chi2_Q5_1.append(Chi2_Q5_1_tmp[0])
            Chi2_H_1.append(Chi2_H_1_tmp[0])

            Pres_pi_1.append(Pres_pi_1_tmp[0])
            Pres_rho_1.append(Pres_rho_1_tmp[0])
            Pres_omega_1.append(Pres_omega_1_tmp[0])
            Pres_K_1.append(Pres_K_1_tmp[0])
            Pres_D_1.append(Pres_D_1_tmp[0])
            Pres_N_1.append(Pres_N_1_tmp[0])
            Pres_T_1.append(Pres_T_1_tmp[0])
            Pres_F_1.append(Pres_F_1_tmp[0])
            Pres_P_1.append(Pres_P_1_tmp[0])
            Pres_Q5_1.append(Pres_Q5_1_tmp[0])
            Pres_H_1.append(Pres_H_1_tmp[0])

            BDen_pi_1.append(BDen_pi_1_tmp[0])
            BDen_rho_1.append(BDen_rho_1_tmp[0])
            BDen_omega_1.append(BDen_omega_1_tmp[0])
            BDen_K_1.append(BDen_K_1_tmp[0])
            BDen_D_1.append(BDen_D_1_tmp[0])
            BDen_N_1.append(BDen_N_1_tmp[0])
            BDen_T_1.append(BDen_T_1_tmp[0])
            BDen_F_1.append(BDen_F_1_tmp[0])
            BDen_P_1.append(BDen_P_1_tmp[0])
            BDen_Q5_1.append(BDen_Q5_1_tmp[0])
            BDen_H_1.append(BDen_H_1_tmp[0])

            for el in glob.glob("D:/EoS/epja/cumulant_test/temp_*.dat"): os.remove(el)
        with open(thermo_1_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([
                    [p_g_1,
                        p_Q_1, b_Q_1, c2_Q_1,
                        p_pert_1, b_pert_1, c2_pert_1,
                        p_pi_1, b_pi_1, c2_pi_1,
                        p_rho_1, b_rho_1, c2_rho_1,
                        p_omega_1, b_omega_1, c2_omega_1,
                        p_K_1, b_K_1, c2_K_1,
                        p_D_1, b_D_1, c2_D_1,
                        p_N_1, b_N_1, c2_N_1,
                        p_T_1, b_T_1, c2_T_1,
                        p_F_1, b_F_1, c2_F_1,
                        p_P_1, b_P_1, c2_P_1,
                        p_Q5_1, b_Q5_1, c2_Q5_1,
                        p_H_1, b_H_1, c2_H_1] 
                    for p_g_1,
                        p_Q_1, b_Q_1, c2_Q_1,
                        p_pert_1, b_pert_1, c2_pert_1,
                        p_pi_1, b_pi_1, c2_pi_1,
                        p_rho_1, b_rho_1, c2_rho_1,
                        p_omega_1, b_omega_1, c2_omega_1,
                        p_K_1, b_K_1, c2_K_1,
                        p_D_1, b_D_1, c2_D_1,
                        p_N_1, b_N_1, c2_N_1,
                        p_T_1, b_T_1, c2_T_1,
                        p_F_1, b_F_1, c2_F_1,
                        p_P_1, b_P_1, c2_P_1,
                        p_Q5_1, b_Q5_1, c2_Q5_1,
                        p_H_1, b_H_1, c2_H_1
                    in zip(Pres_g_1,
                        Pres_Q_1, BDen_Q_1, Chi2_Q_1,
                        Pres_pert_1, BDen_pert_1, Chi2_pert_1,
                        Pres_pi_1, BDen_pi_1, Chi2_pi_1,
                        Pres_rho_1, BDen_rho_1, Chi2_rho_1,
                        Pres_omega_1, BDen_omega_1, Chi2_omega_1,
                        Pres_K_1, BDen_K_1, Chi2_K_1,
                        Pres_D_1, BDen_D_1, Chi2_D_1,
                        Pres_N_1, BDen_N_1, Chi2_N_1,
                        Pres_T_1, BDen_T_1, Chi2_T_1,
                        Pres_F_1, BDen_F_1, Chi2_F_1,
                        Pres_P_1, BDen_P_1, Chi2_P_1,
                        Pres_Q5_1, BDen_Q5_1, Chi2_Q5_1,
                        Pres_H_1, BDen_H_1, Chi2_H_1)
                    ])
    else:
        Pres_g_1, _ = utils.data_collect(0, 0, thermo_1_file)

        Pres_Q_1, BDen_Q_1 = utils.data_collect(1, 2, thermo_1_file)
        Chi2_Q_1, _ = utils.data_collect(3, 3, thermo_1_file)

        Pres_pert_1, BDen_pert_1 = utils.data_collect(4, 5, thermo_1_file)
        Chi2_pert_1, _ = utils.data_collect(6, 6, thermo_1_file)

        Pres_pi_1, BDen_pi_1 = utils.data_collect(7, 8, thermo_1_file)
        Chi2_pi_1, _ = utils.data_collect(9, 9, thermo_1_file)

        Pres_rho_1, BDen_rho_1 = utils.data_collect(10, 11, thermo_1_file)
        Chi2_rho_1, _ = utils.data_collect(12, 12, thermo_1_file)

        Pres_omega_1, BDen_omega_1 = utils.data_collect(13, 14, thermo_1_file)
        Chi2_omega_1, _ = utils.data_collect(15, 15, thermo_1_file)

        Pres_K_1, BDen_K_1 = utils.data_collect(16, 17, thermo_1_file)
        Chi2_K_1, _ = utils.data_collect(18, 18, thermo_1_file)

        Pres_D_1, BDen_D_1 = utils.data_collect(19, 20, thermo_1_file)
        Chi2_D_1, _ = utils.data_collect(21, 21, thermo_1_file)

        Pres_N_1, BDen_N_1 = utils.data_collect(22, 23, thermo_1_file)
        Chi2_N_1, _ = utils.data_collect(24, 24, thermo_1_file)

        Pres_T_1, BDen_T_1 = utils.data_collect(25, 26, thermo_1_file)
        Chi2_T_1, _ = utils.data_collect(27, 27, thermo_1_file)

        Pres_F_1, BDen_F_1 = utils.data_collect(28, 29, thermo_1_file)
        Chi2_F_1, _ = utils.data_collect(30, 30, thermo_1_file)

        Pres_P_1, BDen_P_1 = utils.data_collect(31, 32, thermo_1_file)
        Chi2_P_1, _ = utils.data_collect(33, 33, thermo_1_file)

        Pres_Q5_1, BDen_Q5_1 = utils.data_collect(34, 35, thermo_1_file)
        Chi2_Q5_1, _ = utils.data_collect(36, 36, thermo_1_file)

        Pres_H_1, BDen_H_1 = utils.data_collect(37, 38, thermo_1_file)
        Chi2_H_1, _ = utils.data_collect(39, 39, thermo_1_file)

    #for i, el in enumerate(Pres_H_1):
    #    print("Checking Pres_H", i + 1, "/", 200)
    #    if el == math.inf or el == -math.inf or numpy.isnan(el):
    #        print("Found inf or nan!  Correcting...")
    #        j = 1
    #        k = 1
    #        while Pres_H_1[i - j] == math.inf or Pres_H_1[i - j] == -math.inf
    #        or numpy.isnan(Pres_H_1[i - j]): j+= 1
    #        while Pres_H_1[i + k] == math.inf or Pres_H_1[i + k] == -math.inf
    #        or numpy.isnan(Pres_H_1[i + k]): k+= 1
    #        x0 = T_1[i - j]
    #        x = T_1[i]
    #        x1 = T_1[i + k]
    #        y0 = Pres_H_1[i - j]
    #        y1 = Pres_H_1[i + k]
    #        Pres_H_1[i] = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
    #
    #for i, el in enumerate(BDen_H_1):
    #    print("Checking BDen_H", i + 1, "/", 200)
    #    if el == math.inf or el == -math.inf or numpy.isnan(el):
    #        print("Found inf or nan!  Correcting...")
    #        j = 1
    #        k = 1
    #        while BDen_H_1[i - j] == math.inf or BDen_H_1[i - j] == -math.inf
    #        or numpy.isnan(BDen_H_1[i - j]): j+= 1
    #        while BDen_H_1[i + k] == math.inf or BDen_H_1[i + k] == -math.inf
    #        or numpy.isnan(BDen_H_1[i + k]): k+= 1
    #        x0 = T_1[i - j]
    #        x = T_1[i]
    #        x1 = T_1[i + k]
    #        y0 = BDen_H_1[i - j]
    #        y1 = BDen_H_1[i + k]
    #        BDen_H_1[i] = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
    #
    #for i, el in enumerate(Chi2_H_1):
    #    print("Checking Chi2_H", i + 1, "/", 200)
    #    if el == math.inf or el == -math.inf or numpy.isnan(el):
    #        print("Found inf or nan!  Correcting...")
    #        j = 1
    #        k = 1
    #        while Chi2_H_1[i - j] == math.inf or Chi2_H_1[i - j] == -math.inf
    #        or numpy.isnan(Chi2_H_1[i - j]): j+= 1
    #        while Chi2_H_1[i + k] == math.inf or Chi2_H_1[i + k] == -math.inf
    #        or numpy.isnan(Chi2_H_1[i + k]): k+= 1
    #        x0 = T_1[i - j]
    #        x = T_1[i]
    #        x1 = T_1[i + k]
    #        y0 = Chi2_H_1[i - j]
    #        y1 = Chi2_H_1[i + k]
    #        Chi2_H_1[i] = (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

    T_R12_pred, R12_pred = utils.data_collect(0, 1, "D:/EoS/epja/cumulant_test/R12_prediction.dat")
    T_lQCD_R12_0p4_high, lQCD_R12_0p4_high = utils.data_collect(0, 1, "D:/EoS/epja/cumulant_test/lQCD_0p4_high.dat")
    T_lQCD_R12_0p4_low, lQCD_R12_0p4_low = utils.data_collect(0, 1, "D:/EoS/epja/cumulant_test/lQCD_0p4_low.dat")

    lQCD_0p4 = [numpy.array([x_el, y_el]) for x_el, y_el in zip(T_lQCD_R12_0p4_high, lQCD_R12_0p4_high)]
    for x_el, y_el in zip(T_lQCD_R12_0p4_low[::-1], lQCD_R12_0p4_low[::-1]):
        lQCD_0p4.append(numpy.array([x_el, y_el]))
    lQCD_0p4 = numpy.array(lQCD_0p4)

    Nf = 3.0
    Nc = 3.0
    pT4_SB = [((Nf * Nc) / 6.0) * (((7.0 * (math.pi ** 2)) / 30.0) + ((mu_el / T_el) ** 2) + (1.0 / (2.0 * (math.pi ** 2))) * ((mu_el / T_el) ** 4)) + ((Nc ** 2) - 1.0) * ((math.pi ** 2) / 45.0) for T_el, mu_el in zip(T_1, mu_1)]
    bdenT3_SB = [((Nf * Nc) / 3.0) * ((mu_el / T_el) + (1.0 / (math.pi ** 2)) * ((mu_el / T_el) ** 3)) for T_el, mu_el in zip(T_1, mu_1)]
    Chi2_SB = [((Nf * Nc) / 3.0) * (1.0 + (3.0 / (math.pi ** 2)) * ((mu_el / T_el) ** 2)) for T_el, mu_el in zip(T_1, mu_1)]

    Chi2_tot = [sum(el) for el in zip(Chi2_Q_1, Chi2_pert_1, Chi2_pi_1, Chi2_rho_1, Chi2_omega_1, Chi2_K_1, Chi2_D_1, Chi2_N_1, Chi2_T_1, Chi2_F_1, Chi2_P_1, Chi2_Q5_1)]
    
    BDen_total_1 = [sum(el) for el in zip(BDen_Q_1, BDen_pert_1, BDen_pi_1, BDen_rho_1, BDen_omega_1, BDen_K_1, BDen_D_1, BDen_N_1, BDen_T_1, BDen_F_1, BDen_P_1, BDen_Q5_1)]
    R12dev = [(1.0 / 3.0) * (bden_sum / (mu_el * chi2_sum)) for chi2_sum, mu_el, bden_sum in zip(Chi2_tot, mu_1, BDen_total_1)]

    pT4_Q_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_Q_1, T_1)]
    pT4_g_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_g_1, T_1)]
    pT4_pert_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_pert_1, T_1)]
    pT4_pi_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_pi_1, T_1)]
    pT4_rho_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_rho_1, T_1)]
    pT4_omega_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_omega_1, T_1)]
    pT4_K_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_K_1, T_1)]
    pT4_D_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_D_1, T_1)]
    pT4_N_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_N_1, T_1)]
    pT4_T_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_T_1, T_1)]
    pT4_F_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_F_1, T_1)]
    pT4_P_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_P_1, T_1)]
    pT4_Q5_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_Q5_1, T_1)]
    #pT4_H_1 = [el / (T_el ** 4) for el, T_el in zip(Pres_H_1, T_1)]
    pT4_tot_1 = [sum(el) for el in zip(pT4_Q_1, pT4_g_1, pT4_pert_1, pT4_pi_1, pT4_rho_1, pT4_omega_1, pT4_K_1, pT4_D_1, pT4_N_1, pT4_T_1, pT4_F_1, pT4_P_1, pT4_Q5_1)]

    bdenT3_Q_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_Q_1, T_1)]
    bdenT3_pert_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_pert_1, T_1)]
    bdenT3_pi_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_pi_1, T_1)]
    bdenT3_rho_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_rho_1, T_1)]
    bdenT3_omega_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_omega_1, T_1)]
    bdenT3_K_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_K_1, T_1)]
    bdenT3_D_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_D_1, T_1)]
    bdenT3_N_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_N_1, T_1)]
    bdenT3_T_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_T_1, T_1)]
    bdenT3_F_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_F_1, T_1)]
    bdenT3_P_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_P_1, T_1)]
    bdenT3_Q5_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_Q5_1, T_1)]
    #bdenT3_H_1 = [el / (T_el ** 3) for el, T_el in zip(BDen_H_1, T_1)]
    bdenT3_tot_1 = [sum(el) for el in zip(bdenT3_Q_1, bdenT3_pert_1, bdenT3_pi_1, bdenT3_rho_1, bdenT3_omega_1, bdenT3_K_1, bdenT3_D_1, bdenT3_N_1, bdenT3_T_1, bdenT3_F_1, bdenT3_P_1, bdenT3_Q5_1)]

    color_pi = '#653239'
    color_rho = '#858AE3'
    color_omega = '#FF37A6'
    color_K = 'red'
    color_D = '#4CB944'
    color_N = '#DEA54B'
    color_T = '#23CE6B'
    color_F = '#DB222A'
    color_P = '#78BC61'
    color_Q5 = '#55DBCB'
    color_H = '#A846A0'

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([min(T_1), max(T_1), min(pT4_tot_1), max(pT4_tot_1)])
    ax1.plot(T_1, pT4_Q_1, '-', c = 'blue')
    ax1.plot(T_1, pT4_g_1, '-', c = 'pink')
    ax1.plot(T_1, pT4_pert_1, '-', c = 'cyan')
    ax1.plot(T_1, pT4_pi_1, '-', c = color_pi)
    ax1.plot(T_1, pT4_rho_1, '-', c = color_rho)
    ax1.plot(T_1, pT4_omega_1, '-', c = color_omega)
    ax1.plot(T_1, pT4_K_1, '-', c = color_K)
    ax1.plot(T_1, pT4_D_1, '-', c = color_D)
    ax1.plot(T_1, pT4_N_1, '-', c = color_N)
    ax1.plot(T_1, pT4_T_1, '-', c = color_T)
    ax1.plot(T_1, pT4_F_1, '-', c = color_F)
    ax1.plot(T_1, pT4_P_1, '-', c = color_P)
    ax1.plot(T_1, pT4_Q5_1, '-', c = color_Q5)
    #ax1.plot(T_1, pT4_H_1, '-', c = color_H)
    ax1.plot(T_1, pT4_tot_1, '-', c = 'black')
    ax1.plot(T_1, pT4_SB, '--', c = 'black')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'P [MeV/fm3]', fontsize = 16)

    fig2 = matplotlib.pyplot.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.axis([min(T_1), max(T_1), min(bdenT3_tot_1), max(bdenT3_tot_1)])
    ax2.plot(T_1, bdenT3_Q_1, '-', c = 'blue')
    ax2.plot(T_1, bdenT3_pert_1, '-', c = 'cyan')
    ax2.plot(T_1, bdenT3_pi_1, '-', c = color_pi)
    ax2.plot(T_1, bdenT3_rho_1, '-', c = color_rho)
    ax2.plot(T_1, bdenT3_omega_1, '-', c = color_omega)
    ax2.plot(T_1, bdenT3_K_1, '-', c = color_K)
    ax2.plot(T_1, bdenT3_D_1, '-', c = color_D)
    ax2.plot(T_1, bdenT3_N_1, '-', c = color_N)
    ax2.plot(T_1, bdenT3_T_1, '-', c = color_T)
    ax2.plot(T_1, bdenT3_F_1, '-', c = color_F)
    ax2.plot(T_1, bdenT3_P_1, '-', c = color_P)
    ax2.plot(T_1, bdenT3_Q5_1, '-', c = color_Q5)
    #ax2.plot(T_1, bdenT3_H_1, '-', c = color_H)
    ax2.plot(T_1, bdenT3_tot_1, '-', c = 'black')
    ax2.plot(T_1, bdenT3_SB, '--', c = 'black')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{n_B}$ [1/fm3]', fontsize = 16)

    fig3 = matplotlib.pyplot.figure(num = 3, figsize = (5.9, 5))
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.axis([min(T_1), max(T_1), min(Chi2_tot), max(Chi2_tot)])
    ax3.plot(T_1, Chi2_Q_1, '-', c = 'blue')
    ax3.plot(T_1, Chi2_pert_1, '-', c = 'cyan')
    ax3.plot(T_1, Chi2_pi_1, '-', c = color_pi)
    ax3.plot(T_1, Chi2_rho_1, '-', c = color_rho)
    ax3.plot(T_1, Chi2_omega_1, '-', c = color_omega)
    ax3.plot(T_1, Chi2_K_1, '-', c = color_K)
    ax3.plot(T_1, Chi2_D_1, '-', c = color_D)
    ax3.plot(T_1, Chi2_N_1, '-', c = color_N)
    ax3.plot(T_1, Chi2_T_1, '-', c = color_T)
    ax3.plot(T_1, Chi2_F_1, '-', c = color_F)
    ax3.plot(T_1, Chi2_P_1, '-', c = color_P)
    ax3.plot(T_1, Chi2_Q5_1, '-', c = color_Q5)
    ax3.plot(T_1, Chi2_H_1, '-', c = color_H)
    ax3.plot(T_1, Chi2_tot, '-', c = 'black')
    ax3.plot(T_1, Chi2_SB, '--', c = 'black')
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{chi^B_2}$', fontsize = 16)

    fig4 = matplotlib.pyplot.figure(num = 4, figsize = (5.9, 5))
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.axis([min(T_1), max(T_1), min(R12dev), max(R12dev)])
    ax4.add_patch(matplotlib.patches.Polygon(lQCD_0p4, 
            closed = True, fill = True, color = 'red', alpha = 0.3))
    ax4.plot(T_1, R12dev, '-', c = 'red')
    ax4.plot(T_R12_pred, R12_pred, '--', c = 'red')
    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax4.set_xlabel(r'T [MeV]', fontsize = 16)
    ax4.set_ylabel(r'$\mathrm{R_{12}}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    fig2.tight_layout(pad = 0.1)
    fig3.tight_layout(pad = 0.1)
    fig4.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def pnjl_with_sigma_test():

    def Mq(p, m0, sigma, L):
        #
        return m0 + math.exp(-((p ** 2) / (L ** 2))) * sigma
    def dMq_dp(p, m0, sigma, L):
        #
        return -2.0 * p * sigma * math.exp(-((p ** 2) / (L ** 2))) / (L ** 2)
    def En(p, M):
        #
        return math.sqrt((p ** 2) + (M ** 2))
    def dEn_dp(p, M, dMdp):
        #
        return (p + M * dMdp) / math.sqrt((p ** 2) + (M ** 2))
    def f_plus_phi(T, mu, energy, Phi, Phib):
        exp_term = -(energy - mu) / T
        exp_term2 = 2.0 * exp_term
        exp_term3 = 3.0 * exp_term
        if exp_term > 700.0 or exp_term2 > 700.0 or exp_term3 > 700.0:
            return complex(1.0, 0.0)
        else:
            exp1 = math.exp(exp_term)
            exp2 = math.exp(exp_term2)
            exp3 = math.exp(exp_term3)
            num = Phib * exp1 + 2.0 * Phi * exp2 + exp3
            den = 1.0 + 3.0 * Phib * exp1 + 3.0 * Phi * exp2 + exp3
            return num / den
    def f_minus_phi(T, mu, energy, Phi, Phib):
        exp_term = -(energy + mu) / T
        exp_term2 = 2.0 * exp_term
        exp_term3 = 3.0 * exp_term
        if exp_term > 700.0 or exp_term2 > 700.0 or exp_term3 > 700.0:
            return complex(1.0, 0.0)
        else:
            exp1 = math.exp(exp_term)
            exp2 = math.exp(exp_term2)
            exp3 = math.exp(exp_term3)
            num = Phi * exp1 + 2.0 * Phib * exp2 + exp3
            den = 1.0 + 3.0 * Phi * exp1 + 3.0 * Phib * exp2 + exp3
            return num / den

    def Omega_T0_Nf1(mu, sigma, sigma0, m=5.0, Gs=10.08e-6, L=891.0):

        def integrand(p, _mu, _sigma, _m, _L, _sigma0):
            mass = Mq(p, _m, _sigma, _L)
            mass0 = Mq(p, _m, _sigma0, _L)
            energy = En(p, mass)
            energy0 = En(p, mass0)
            return (p ** 2) * (energy - energy0 - numpy.heaviside(_mu - energy, 0.5) * 3.0 * (energy - _mu))

        integ, err = scipy.integrate.quad(integrand, 0.0, numpy.inf, args = (mu, sigma, m, L, sigma0))

        return (((sigma ** 2) - (sigma0 ** 2)) / (4.0 * Gs)) - (1.0 / (numpy.pi ** 2)) * integ

    def Omega_Nf1(T, mu, sigma, phi, phib, sigma0, m=5.0, Gs=10.08e-6, L=891.0):

        if T < 1.0:
            return complex(Omega_T0_Nf1(mu, sigma, sigma0, m = m, Gs = Gs, L = L), 0.0)
        else:

            def integrand(p, _T, _mu, _sigma, _phi, _phib, _m, _L, _sigma0):
                mass = Mq(p, _m, _sigma, _L)
                dmass_dp = dMq_dp(p, _m, _sigma, _L)
                mass0 = Mq(p, _m, _sigma0, _L)
                energy = En(p, mass)
                denergy_dp = dEn_dp(p, mass, dmass_dp)
                energy0 = En(p, mass0)
                return (p ** 2) * (energy - energy0) + (p ** 3) * denergy_dp * (f_plus_phi(_T, _mu, energy, _phi, _phib) + f_minus_phi(_T, _mu, energy, _phi, _phib))

            integ, err = scipy.integrate.quad(integrand, 0.0, numpy.inf, args = (T, mu, sigma, phi, phib, m, L, sigma0))

            return (((sigma ** 2) - (sigma0 ** 2)) / (4.0 * Gs)) - (1.0 / (numpy.pi ** 2)) * integ

    def find_L(sigma0, m=5.0, Gs=10.08e-6):
        print("Looking for L for sigma0 =", sigma0, ", m =", m, ", Gs =", Gs)
        def rootfunc(y, _sigma0, _m, _Gs):
            min_s = scipy.optimize.minimize(lambda x, _s0, _l, _mass, _g: Omega_T0_Nf1(0.0, x, _s0, L = _l, m = _mass, Gs = _g), 
                [2.0 * _sigma0], 
                args = (_sigma0, y, _m, _Gs))
            return min_s.x[0] - _sigma0
        found = scipy.optimize.root(rootfunc, [2e3], args = (sigma0, m, Gs))
        return found.x[0]

    sigma_l_0 = 400.0
    sigma_s_0 = 400.0

    L_l = find_L(sigma_l_0)
    L_s = find_L(sigma_s_0, m = 150.0)
"""


if __name__ == '__main__':

    #cumulant_test()

    class cached:
        def __init__(self, func):
            print("Wrapper initialized!")
            self.func = func
        def __call__(self, *args, **kwds):
            print("Wrapper called!")
            print("Function name:", self.func.__name__)
            print("Function args:", args)
            print("Function kwds:", kwds)
            print("Function return:\n")
            return self.func(*args, **kwds)

    @cached
    def testfunc(a, b, c, d="test"):
        print(a, b, c, d)

    testfunc(1, 2, 3)
    testfunc(3, 2, 1)
    testfunc(3, 2, 1, d="last test")

    print("END")