#TODO:
#1. make alt_BDensity and alt_SDensity routines.
#2. add hexa-diquark cluster
#3. consider d phi / d (mu/T).
#4. consider a mu dependent chemical potential of https://arxiv.org/abs/1207.4890
#5. return to sigma/phi derivation using a momentum-dependent quark mass (check Omega_Delta!).
#6. change the pl potential to this one https://arxiv.org/pdf/1307.5958.pdf

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time
import math
import csv

from scipy.interpolate import UnivariateSpline, Akima1DInterpolator
from scipy.optimize import dual_annealing, basinhopping
from scipy.special import binom

from scipy.signal import find_peaks

from matplotlib.patches import Polygon, FancyArrowPatch

from random import uniform, randint

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from joblib import Parallel, delayed

from tqdm import tqdm

from pnjl_functions import alt_Omega_Q_real, alt_Omega_Q_imag
from pnjl_functions import Omega_pert_real, Omega_pert_imag
from pnjl_functions import Omega_g_real, Omega_g_imag
from pnjl_functions import alt_Omega_cluster_real, alt_Omega_cluster_imag
from pnjl_functions import Omega_Delta

from pnjl_functions import M, dMdmu, dMdT, Delta_ls, Tc
from pnjl_functions import alt_Pressure_Q, BDensity_Q, SDensity_Q 
from pnjl_functions import Pressure_g, SDensity_g
from pnjl_functions import Pressure_pert, BDensity_pert, SDensity_pert
from pnjl_functions import alt_Pressure_cluster, BDensity_cluster, SDensity_cluster

from pnjl_functions import default_MN, default_MM, default_MD, default_MF, default_MT, default_MP, default_MQ, default_MH
from pnjl_functions import default_M0

from utils import data_collect

#np.seterr(all = 'raise')

continuum_lambda_basis = math.sqrt(2) * M(0, 0) * 0.0001

def cluster_thermo(T, mu, phi_re, phi_im, M, Mth, dMdmu, dMthdmu, dMdT, dMthdT, a, Ni, Lambda, dx):
    Pres = [
        alt_Pressure_cluster(
            T_el,                               #temperature
            mu_el,                              #baryochemical potential
            complex(phi_re_el, phi_im_el),      #traced PL
            complex(phi_re_el, -phi_im_el),     #traced PL c.c.
            M_el,                               #bound state mass
            Mth_el,                             #threshold mass
            a,                                  #nr of valence quarks - nr of valence antiquarks
            dx,                                 #degeneracy factor
            Ni,                                 #sum of valence quarks and antiquarks
            Lambda                              #continuum mass range
            ) 
        for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el in tqdm(zip(T, mu, phi_re, phi_im, M, Mth), desc = "Pres", total = len(T), ascii = True)]
    #BDen = [
    #    BDensity_cluster(
    #        T_el,                                   #temperature
    #        mu_el,                                  #baryochemical potential
    #        complex(phi_re_el, phi_im_el),          #traced PL
    #        complex(phi_re_el, -phi_im_el),         #traced PL c.c.
    #        M_el,                                   #bound state mass
    #        Mth_el,                                 #threshold mass
    #        dM_el,                                  #derivative of bound state mass w.r.t. baryochemical potential
    #        dMth_el,                                #derivative of threshold state mass w.r.t. baryochemical potential
    #        a,                                      #nr of valence quarks - nr of valence antiquarks
    #        dx                                      #degeneracy factor
    #        ) 
    #    for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el, dM_el, dMth_el in tqdm(zip(T, mu, phi_re, phi_im, M, Mth, dMdmu, dMthdmu), desc = "BDen", total = len(T), ascii = True)]
    #SDen = [
    #    SDensity_cluster(
    #        T_el,                                   #temperature
    #        mu_el,                                     #baryochemical potential
    #        complex(phi_re_el, phi_im_el),          #traced PL
    #        complex(phi_re_el, -phi_im_el),         #traced PL c.c.
    #        M_el,                                   #bound state mass
    #        Mth_el,                                 #threshold mass
    #        dM_el,                                  #derivative of bound state mass w.r.t. temperature
    #        dMth_el,                                #derivative of threshold state mass w.r.t. temperature
    #        a,                                      #nr of valence quarks - nr of valence antiquarks
    #        dx                                      #degeneracy factor
    #        ) 
    #    for T_el, mu_el, phi_re_el, phi_im_el, M_el, Mth_el, dM_el, dMth_el in tqdm(zip(T, mu, phi_re, phi_im, M, Mth, dMdT, dMthdT), desc = "SDen", total = len(T), ascii = True)]
    BDen = 0.0
    SDen = 0.0
    return Pres, BDen, SDen

def calc_PL(
    T : float, mu : float, phi_re0 : float, phi_im0 : float,
    light_kwargs = {}, strange_kwargs = {'Nf' : 1.0, 'ml' : 100.0}, gluon_kwargs = {}, perturbative_kwargs = {'Nf' : 2.0},
    with_clusters : bool = True) -> (float, float):
    
    sd = 1234
    attempt = 1
    max_attempt = 100
    bnds = ((0.0, 3.0), (-3.0, 3.0),)

    def thermodynamic_potential(x, _T, _mu, _with_clusters, s_kwargs, q_kwargs, pert_kwargs, g_kwargs):
        sq   = alt_Omega_Q_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **s_kwargs)
        lq   = alt_Omega_Q_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **q_kwargs)
        per  = Omega_pert_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **pert_kwargs)
        glue = Omega_g_real(_T, complex(x[0], x[1]), complex(x[0], -x[1]), **g_kwargs)
        diquark = 1.0 * alt_Omega_cluster_real(_T, _mu, complex(x[0], -x[1]), complex(x[0],  x[1]), default_MD, 2.0 * math.sqrt(2) * M(_T, _mu), 2, 2, 2. * continuum_lambda_basis) if _with_clusters else 0.0
        fquark  = 3.0 * alt_Omega_cluster_real(_T, _mu, complex(x[0],  x[1]), complex(x[0], -x[1]), default_MF, 4.0 * math.sqrt(2) * M(_T, _mu), 4, 4, 4. * continuum_lambda_basis) if _with_clusters else 0.0
        qquark  = 4.0 * alt_Omega_cluster_real(_T, _mu, complex(x[0], -x[1]), complex(x[0],  x[1]), default_MQ, 5.0 * math.sqrt(2) * M(_T, _mu), 5, 5, 5. * continuum_lambda_basis) if _with_clusters else 0.0
        return sq + lq + per + glue + diquark + fquark + qquark

    omega_result = dual_annealing(
            thermodynamic_potential,
            bounds = bnds,
            args = (T, mu, with_clusters, strange_kwargs, light_kwargs, perturbative_kwargs, gluon_kwargs),
            x0 = [phi_re0, phi_im0],
            maxiter = 20,
            seed = sd
            )

    return (omega_result.x[0], omega_result.x[1])

def clusters(T, mu, phi_re, phi_im):
    print("Calculating nucleon thermo..")
    M_N         = [default_MN                               for T_el, mu_el in zip(T, mu)]
    dM_N_dmu    = [0.                                       for T_el, mu_el in zip(T, mu)]
    dM_N_dT     = [0.                                       for T_el, mu_el in zip(T, mu)]
    Mth_N       = [3. * math.sqrt(2) * M(T_el, mu_el)       for T_el, mu_el in zip(T, mu)]
    dMth_N_dmu  = [3. * math.sqrt(2) * dMdmu(T_el, mu_el)   for T_el, mu_el in zip(T, mu)]
    dMth_N_dT   = [3. * math.sqrt(2) * dMdT(T_el, mu_el)    for T_el, mu_el in zip(T, mu)]
    #(N(Dq): spin * isospin * color)
    dN = (2.0 * 2.0 * (1.0 / 3.0))
    NN = 3
    LambdaN = 3. * continuum_lambda_basis
    Pres_N, BDen_N, SDen_N = cluster_thermo(T, mu, phi_re, phi_im, M_N, Mth_N, dM_N_dmu, dMth_N_dmu, dM_N_dT, dMth_N_dT, 3, dN, NN, LambdaN)

    print("Calculating pentaquark thermo..")
    M_P         = [default_MP                               for T_el, mu_el in zip(T, mu)]
    dM_P_dmu    = [0.                                       for T_el, mu_el in zip(T, mu)]
    dM_P_dT     = [0.                                       for T_el, mu_el in zip(T, mu)]
    Mth_P       = [5. * math.sqrt(2) * M(T_el, mu_el)       for T_el, mu_el in zip(T, mu)]
    dMth_P_dmu  = [5. * math.sqrt(2) * dMdmu(T_el, mu_el)   for T_el, mu_el in zip(T, mu)]
    dMth_P_dT   = [5. * math.sqrt(2) * dMdT(T_el, mu_el)    for T_el, mu_el in zip(T, mu)]
    #P(NM) + P(NM)
    dP = (4.0 * 2.0 * (1.0 / 3.0)) + (2.0 * 4.0 * (1.0 / 3.0))
    NP = 5
    LambdaP = 5. * continuum_lambda_basis
    Pres_P, BDen_P, SDen_P = cluster_thermo(T, mu, phi_re, phi_im, M_P, Mth_P, dM_P_dmu, dMth_P_dmu, dM_P_dT, dMth_P_dT, 3, dP, NP, LambdaP)

    print("Calculating hexaquark thermo..")
    M_H         = [default_MH                               for T_el, mu_el in zip(T, mu)]
    dM_H_dmu    = [0.                                       for T_el, mu_el in zip(T, mu)]
    dM_H_dT     = [0.                                       for T_el, mu_el in zip(T, mu)]
    Mth_H       = [6. * math.sqrt(2) * M(T_el, mu_el)       for T_el, mu_el in zip(T, mu)]
    dMth_H_dmu  = [6. * math.sqrt(2) * dMdmu(T_el, mu_el)   for T_el, mu_el in zip(T, mu)]
    dMth_H_dT   = [6. * math.sqrt(2) * dMdT(T_el, mu_el)    for T_el, mu_el in zip(T, mu)]
    #H(Qq) / H(FD) / H(NN) + H(Qq) / H(NN)
    dH = (1.0 * 3.0 * (1.0 / 3.0)) + (3.0 * 1.0 * (1.0 / 3.0))
    NH = 6
    LambdaH = 6. * continuum_lambda_basis
    Pres_H, BDen_H, SDen_H = cluster_thermo(T, mu, phi_re, phi_im, M_H, Mth_H, dM_H_dmu, dMth_H_dmu, dM_H_dT, dMth_H_dT, 6, dH, NH, LambdaH)

    print("Calculating pi meson thermo..")
    M_pi         = [140.                                     for T_el, mu_el in zip(T, mu)]
    dM_pi_dmu    = [0.                                       for T_el, mu_el in zip(T, mu)]
    dM_pi_dT     = [0.                                       for T_el, mu_el in zip(T, mu)]
    Mth_pi       = [2. * math.sqrt(2) * M(T_el, mu_el)       for T_el, mu_el in zip(T, mu)]
    dMth_pi_dmu  = [2. * math.sqrt(2) * dMdmu(T_el, mu_el)   for T_el, mu_el in zip(T, mu)]
    dMth_pi_dT   = [2. * math.sqrt(2) * dMdT(T_el, mu_el)    for T_el, mu_el in zip(T, mu)]
    #pi(q aq)
    dpi = ((1.0 / 2.0) * 3.0 * (1.0 / 3.0))
    Npi = 2
    Lambdapi = 2. * continuum_lambda_basis
    Pres_pi, BDen_pi, SDen_pi = cluster_thermo(T, mu, phi_re, phi_im, M_pi, Mth_pi, dM_pi_dmu, dMth_pi_dmu, dM_pi_dT, dMth_pi_dT, 0, dpi, Npi, Lambdapi)

    print("Calculating rho meson thermo..")
    M_rho        = [default_MM                               for T_el, mu_el in zip(T, mu)]
    dM_rho_dmu   = [0.                                       for T_el, mu_el in zip(T, mu)]
    dM_rho_dT    = [0.                                       for T_el, mu_el in zip(T, mu)]
    Mth_rho      = [2. * math.sqrt(2) * M(T_el, mu_el)       for T_el, mu_el in zip(T, mu)]
    dMth_rho_dmu = [2. * math.sqrt(2) * dMdmu(T_el, mu_el)   for T_el, mu_el in zip(T, mu)]
    dMth_rho_dT  = [2. * math.sqrt(2) * dMdT(T_el, mu_el)    for T_el, mu_el in zip(T, mu)]
    #rho(q aq)
    drho = ((3.0 / 2.0) * 3.0 * (1.0 / 3.0)) #should this have the 1/2 factor?
    Nrho = 2
    Lambdarho = 2. * continuum_lambda_basis
    Pres_rho, BDen_rho, SDen_rho = cluster_thermo(T, mu, phi_re, phi_im, M_rho, Mth_rho, dM_rho_dmu, dMth_rho_dmu, dM_rho_dT, dMth_rho_dT, 0, drho, Nrho, Lambdarho)

    print("Calculating omega meson thermo..")
    M_omega        = [default_MM                               for T_el, mu_el in zip(T, mu)]
    dM_omega_dmu   = [0.                                       for T_el, mu_el in zip(T, mu)]
    dM_omega_dT    = [0.                                       for T_el, mu_el in zip(T, mu)]
    Mth_omega      = [2. * math.sqrt(2) * M(T_el, mu_el)       for T_el, mu_el in zip(T, mu)]
    dMth_omega_dmu = [2. * math.sqrt(2) * dMdmu(T_el, mu_el)   for T_el, mu_el in zip(T, mu)]
    dMth_omega_dT  = [2. * math.sqrt(2) * dMdT(T_el, mu_el)    for T_el, mu_el in zip(T, mu)]
    #omega(q aq)
    domega = ((3.0 / 2.0) * 1.0 * (1.0 / 3.0)) #should this have the 1/2 factor?
    Nomega = 2
    Lambdaomega = 2. * continuum_lambda_basis
    Pres_omega, BDen_omega, SDen_omega = cluster_thermo(T, mu, phi_re, phi_im, M_omega, Mth_omega, dM_omega_dmu, dMth_omega_dmu, dM_omega_dT, dMth_omega_dT, 0, domega, Nomega, Lambdaomega)

    print("Calculating tetraquark thermo..")
    M_T        = [default_MT                               for T_el, mu_el in zip(T, mu)]
    dM_T_dmu   = [0.                                       for T_el, mu_el in zip(T, mu)]
    dM_T_dT    = [0.                                       for T_el, mu_el in zip(T, mu)]
    Mth_T      = [4. * math.sqrt(2) * M(T_el, mu_el)       for T_el, mu_el in zip(T, mu)]
    dMth_T_dmu = [4. * math.sqrt(2) * dMdmu(T_el, mu_el)   for T_el, mu_el in zip(T, mu)]
    dMth_T_dT  = [4. * math.sqrt(2) * dMdT(T_el, mu_el)    for T_el, mu_el in zip(T, mu)]
    #T(MM) + T(MM) + T(MM)
    dT = ((1.0 / 2.0) * 5.0 * (1.0 / 3.0)) + ((5.0 / 2.0) * 1.0 * (1.0 / 3.0)) + ((3.0 / 2.0) * 3.0 * (1.0 / 3.0))
    NT = 4
    LambdaT = 4. * continuum_lambda_basis
    Pres_T, BDen_T, SDen_T = cluster_thermo(T, mu, phi_re, phi_im, M_T, Mth_T, dM_T_dmu, dMth_T_dmu, dM_T_dT, dMth_T_dT, 0, dT, NT, LambdaT)

    print("Calculating diquark thermo..")
    M_D        = [default_MD                               for T_el, mu_el in zip(T, mu)]
    dM_D_dmu   = [0.                                       for T_el, mu_el in zip(T, mu)]
    dM_D_dT    = [0.                                       for T_el, mu_el in zip(T, mu)]
    Mth_D      = [2. * math.sqrt(2) * M(T_el, mu_el)       for T_el, mu_el in zip(T, mu)]
    dMth_D_dmu = [2. * math.sqrt(2) * dMdmu(T_el, mu_el)   for T_el, mu_el in zip(T, mu)]
    dMth_D_dT  = [2. * math.sqrt(2) * dMdT(T_el, mu_el)    for T_el, mu_el in zip(T, mu)]
    #D(qq)
    dD = (1.0 * 1.0 * 3.0)
    ND = 2
    LambdaD = 2. * continuum_lambda_basis
    Pres_D, BDen_D, SDen_D = cluster_thermo(T, mu, phi_re, [-el for el in phi_im], M_D, Mth_D, dM_D_dmu, dMth_D_dmu, dM_D_dT, dMth_D_dT, 2, dD, ND, LambdaD)

    print("Calculating 4-quark thermo..")
    M_F        = [default_MF                               for T_el, mu_el in zip(T, mu)]
    dM_F_dmu   = [0.                                       for T_el, mu_el in zip(T, mu)]
    dM_F_dT    = [0.                                       for T_el, mu_el in zip(T, mu)]
    Mth_F      = [4. * math.sqrt(2) * M(T_el, mu_el)       for T_el, mu_el in zip(T, mu)]
    dMth_F_dmu = [4. * math.sqrt(2) * dMdmu(T_el, mu_el)   for T_el, mu_el in zip(T, mu)]
    dMth_F_dT  = [4. * math.sqrt(2) * dMdT(T_el, mu_el)    for T_el, mu_el in zip(T, mu)]
    #F(Nq)
    dF = (1.0 * 1.0 * 3.0)
    NF = 4
    LambdaF = 4. * continuum_lambda_basis
    Pres_F, BDen_F, SDen_F = cluster_thermo(T, mu, phi_re, phi_im, M_F, Mth_F, dM_F_dmu, dMth_F_dmu, dM_F_dT, dMth_F_dT, 4, dF, NF, LambdaF)

    print("Calculating 5-quark thermo..")
    M_Q5        = [default_MQ                               for T_el, mu_el in zip(T, mu)]
    dM_Q5_dmu   = [0.                                       for T_el, mu_el in zip(T, mu)]
    dM_Q5_dT    = [0.                                       for T_el, mu_el in zip(T, mu)]
    Mth_Q5      = [5. * math.sqrt(2) * M(T_el, mu_el)       for T_el, mu_el in zip(T, mu)]
    dMth_Q5_dmu = [5. * math.sqrt(2) * dMdmu(T_el, mu_el)   for T_el, mu_el in zip(T, mu)]
    dMth_Q5_dT  = [5. * math.sqrt(2) * dMdT(T_el, mu_el)    for T_el, mu_el in zip(T, mu)]
    #Q5(F(Nq)q) / Q5(F(DD)q) / Q5(ND)
    dQ5 = (2.0 * 2.0 * 3.0)
    NQ5 = 5
    LambdaQ5 = 5. * continuum_lambda_basis
    Pres_Q5, BDen_Q5, SDen_Q5 = cluster_thermo(T, mu, phi_re, [-el for el in phi_im], M_Q5, Mth_Q5, dM_Q5_dmu, dMth_Q5_dmu, dM_Q5_dT, dMth_Q5_dT, 5, dQ5, NQ5, LambdaQ5)

    return (
        (Pres_pi, Pres_rho, Pres_omega, Pres_D, Pres_N, Pres_T, Pres_F, Pres_P, Pres_Q5, Pres_H),
        (BDen_pi, BDen_rho, BDen_omega, BDen_D, BDen_N, BDen_T, BDen_F, BDen_P, BDen_Q5, BDen_H),
        (SDen_pi, SDen_rho, SDen_omega, SDen_D, SDen_N, SDen_T, SDen_F, SDen_P, SDen_Q5, SDen_H)
        )

def epja_figure2():

    (T, phi_re_0) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(0.0)) + "p0_cluster_dense.dat")
    (T, phi_re_100) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(100.0)) + "p0_cluster_dense.dat")
    (T, phi_re_200) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(200.0)) + "p0_cluster_dense.dat")
    (T, phi_re_300) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(300.0)) + "p0_cluster_dense.dat")
    (T, phi_re_alt_0) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(0.0)) + "p0_dev_dense.dat")
    (T, phi_re_alt_100) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(100.0)) + "p0_dev_dense.dat")
    (T, phi_re_alt_200) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(200.0)) + "p0_dev_dense.dat")
    (T, phi_re_alt_300) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(300.0)) + "p0_dev_dense.dat")

    sigma_0 = [(400. * Delta_ls(el, 0.0) + 5.5) / 405.5 for el in T]
    sigma_100 = [(400. * Delta_ls(el, 100.0) + 5.5) / 405.5 for el in T]
    sigma_200 = [(400. * Delta_ls(el, 200.0) + 5.5) / 405.5 for el in T]
    sigma_300 = [(400. * Delta_ls(el, 300.0) + 5.5) / 405.5 for el in T]

    fig = plt.figure(num = 1, figsize = (5.9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([min(T), max(T), min(phi_re_0), 1.1])
    ax.plot(T, phi_re_0, c = 'blue', label = r'$\mathrm{\Phi}$')
    ax.plot(T, phi_re_alt_0, '--', c = 'red', label = r'$\mathrm{\Phi_0}$')
    ax.plot(T, sigma_0, c = 'green', label = r'$\mathrm{M_q}$ / $\mathrm{M_{q,vac}}$')
    ax.text(10., 1.03, r'$\mathrm{M_q}$ / $\mathrm{M_{q,vac}}$', fontsize = 14)
    ax.text(175., 1.03, r'$\mathrm{\mu=0}$', fontsize = 14)
    ax.text(121., 0.25, r'$\mathrm{\Phi=\Phi_0}$', fontsize = 14)
    #ax.legend(loc = 'center left')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax.set_xlabel(r'T [MeV]', fontsize = 16)
    ax.set_ylabel(r'$\mathrm{\Phi}$', fontsize = 16)

    fig2 = plt.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.axis([min(T), max(T), min(phi_re_0), 1.1])
    ax2.plot(T, phi_re_300, c = 'blue', label = r'$\mathrm{\Phi}$')
    ax2.plot(T, phi_re_alt_300, '--', c = 'red', label = r'$\mathrm{\Phi_0}$')
    ax2.plot(T, sigma_300, c = 'green', label = r'$\mathrm{M_q}$ / $\mathrm{M_{q,vac}}$')
    ax2.text(10., 1.03, r'$\mathrm{M_q}$ / $\mathrm{M_{q,vac}}$', fontsize = 14)
    ax2.text(145., 1.03, r'$\mathrm{\mu=300}$ MeV', fontsize = 14)
    ax2.text(85., 0.25, r'$\mathrm{\Phi}$', fontsize = 14)
    ax2.text(110., 0.25, r'$\mathrm{\Phi_0}$', fontsize = 14)
    #ax2.legend(loc = 'center left')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{\Phi}$', fontsize = 16)

    fig.tight_layout()
    fig2.tight_layout()
    plt.show()
    plt.close()

def epja_figure3():

    #druga wersja napisu "Borsanyi et al. (2012)", w poziomie ze strzalka

    col_qgp = 'blue'
    col_pert = 'red'
    col_gluon = 'pink'
    col_pnjl = 'magenta'

    (T, phi_re_0) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(0.0)) + "p0_cluster_dense.dat")
    Pres_Q, BDen_Q              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(0.0)) + "p0.dat")
    SDen_Q, _                   = data_collect(4, 4, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(0.0)) + "p0.dat")
    Pres_g, SDen_g              = data_collect(2, 4, "D:/EoS/BDK/thermo_mu_const/pl_mu_" + str(int(0.0)) + "p0.dat")
    Pres_pert, BDen_pert        = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(0.0)) + "p0.dat")
    SDen_pert, _                = data_collect(4, 4, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(0.0)) + "p0.dat")

    (low_1204_6710v2_x, low_1204_6710v2_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_x, high_1204_6710v2_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")

    borsanyi_1204_6710v2 = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_x, high_1204_6710v2_y)]
    for x_el, y_el in zip(low_1204_6710v2_x[::-1], low_1204_6710v2_y[::-1]):
        borsanyi_1204_6710v2.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2 = np.array(borsanyi_1204_6710v2)

    fig = plt.figure(num = 1, figsize = (5.9, 5))
    ax = fig.add_subplot(1, 1, 1)

    contrib_pert = [(ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_g = [(pg_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_quark = [(pq_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_qgp = [(pq_el + pg_el + ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]

    ax.axis([95., 200., -1.0, 1.5])
    ax.plot(T, [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pert)]    , ':', c = col_pert     , label = r'$\mathrm{P_{pert}=-\Omega_{pert}}$'    )
    ax.plot(T, [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q)]       , '-.', c = col_pnjl     , label = r'$\mathrm{P_{Q}=-\Omega_Q}$'    )
    ax.plot(T, [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_g)]       , '--', c = col_gluon    , label = r'$\mathrm{P_{gluon}=-U}$'   )
    ax.plot(T, contrib_qgp                                                 , '-', c = col_qgp      , label = r'$\mathrm{P_{QGP}=P_{pert}+P_{Q}+P_{gluon}}$')
    ax.add_patch(Polygon(borsanyi_1204_6710v2, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2012)'))
    ax.text(180., -0.88, r'$\mathrm{P_{pert}}$', fontsize = 14)
    ax.text(180., -0.28, r'$\mathrm{P_{gluon}}$', fontsize = 14)
    ax.text(167., 0.46, r'$\mathrm{P_{total}=P_{QGP}}$', fontsize = 14)
    ax.text(180., -0.28, r'$\mathrm{P_{gluon}}$', fontsize = 14)
    ax.text(165., 1.2, r'$\mathrm{P_{Q}}$', fontsize = 14)
    ax.text(120., 0.75, r'Borsanyi et al. (2012)', fontsize = 10, rotation = 23)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax.set_xlabel(r'T [MeV]', fontsize = 16)
    ax.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    plt.tight_layout()
    plt.show()
    plt.close()

def epja_figure4():

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

    (T, phi_re_0) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(const_mu)) + "p0_cluster_dense.dat")

    fig = plt.figure(num = 1, figsize = (5.9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([min(T), max(T), 0.0, 1.05 * max([6.0 * math.sqrt(2) * M(el, const_mu) for el in T])])

    ax.plot(T, [default_MN for el in T]                            , '-'   , c = col_n     , label = r'nucleon'            )
    #ax.fill_between(T[T < 150.], [default_MN for el in T if el < 150.], [3.0 * math.sqrt(2) * M(el, const_mu) for el in T if el < 150.], color = col_n, alpha = 0.5)
    ax.add_patch(FancyArrowPatch((30., 1750.), (30., 1000.), arrowstyle='<->', mutation_scale=20, color = col_n))
    ax.plot(T, [3.0 * math.sqrt(2) * M(el, const_mu) for el in T]  , '--'  , c = col_n)
    ax.text(35., 1320., r'nucleon', fontsize = 14)

    ax.plot(T, [140 for el in T]                                   , '-'   , c = col_pi    , label = r'$\mathrm{\pi}$'               )
    #ax.fill_between(T[T < 180.], [140 for el in T if el < 180.], [2.0 * math.sqrt(2) * M(el, const_mu) for el in T if el < 180.], color = col_pi, alpha = 0.5)
    ax.add_patch(FancyArrowPatch((50., 1150.), (50., 135.), arrowstyle='<->', mutation_scale=20, color = col_pi))
    ax.plot(T, [2.0 * math.sqrt(2) * M(el, const_mu) for el in T]  , '--'  , c = col_pi)
    ax.text(55., 620., r'pion', fontsize = 14)

    ax.plot(T, [default_MH for el in T]                            , '-'   , c = col_h     , label = r'hexaquark'          )
    #ax.fill_between(T[T < 151.], [default_MH for el in T if el < 151.], [6.0 * math.sqrt(2) * M(el, const_mu) for el in T if el < 151.], color = col_h, alpha = 0.5)
    ax.add_patch(FancyArrowPatch((10., 3450.), (10., 1900.), arrowstyle='<->', mutation_scale=20, color = col_h))
    ax.plot(T, [6.0 * math.sqrt(2) * M(el, const_mu) for el in T]  , '--'  , c = col_h)
    ax.text(15., 2650., r'hexaquark', fontsize = 14)

    ax.plot(T, [M(el, const_mu) for el in T], '-', c = col_qgp, label = r'u/d quark')
    ax.text(5., 480., r'u/d quark', fontsize = 10)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax.set_xlabel(r'T [MeV]', fontsize = 16)
    ax.set_ylabel(r'mass [MeV]', fontsize = 16)

    plt.tight_layout()
    plt.show()
    plt.close()

def epja_figure5():

    const_mu = 0.0
    (T, phi_re) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(const_mu)) + "p0_cluster_dense.dat")

    Pres_Q, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_g, _              = data_collect(2, 4, "D:/EoS/BDK/thermo_mu_const/pl_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_pert, _        = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_pi, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pi_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_rho, _          = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/rho_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_omega, _      = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/omega_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_D, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_N, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_T, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/T_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_F, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_P, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_Q5, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_H, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0.dat")

    Pres_Q0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_g0, _            = data_collect(2, 4, "D:/EoS/BDK/thermo_mu_const/pl_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_pert0, _      = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_pi0, _          = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pi_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_rho0, _        = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/rho_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_omega0, _    = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/omega_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_D0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_N0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_T0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/T_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_F0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_P0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_Q50, _          = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_H0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0_no_phi.dat")

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

    (borsanyi_upper_x, borsanyi_upper_y) = data_collect(0, 1, "D:/EoS/archive/BDK/borsanyi_upper.dat")
    (borsanyi_lower_x, borsanyi_lower_y) = data_collect(0, 1, "D:/EoS/archive/BDK/borsanyi_lower.dat")
    (bazavov_upper_x, bazavov_upper_y) = data_collect(0, 1, "D:/EoS/archive/BDK/bazavov_upper.dat")
    (bazavov_lower_x, bazavov_lower_y) = data_collect(0, 1, "D:/EoS/archive/BDK/bazavov_lower.dat")
    (low_1204_6710v2_x, low_1204_6710v2_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_x, high_1204_6710v2_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")

    bazavov = [np.array([x_el, y_el]) for x_el, y_el in zip(bazavov_upper_x, bazavov_upper_y)]
    for x_el, y_el in zip(bazavov_lower_x[::-1], bazavov_lower_y[::-1]):
        bazavov.append(np.array([x_el, y_el]))
    borsanyi = [np.array([x_el, y_el]) for x_el, y_el in zip(borsanyi_upper_x, borsanyi_upper_y)]
    for x_el, y_el in zip(borsanyi_lower_x[::-1], borsanyi_lower_y[::-1]):
        borsanyi.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2 = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_x, high_1204_6710v2_y)]
    for x_el, y_el in zip(low_1204_6710v2_x[::-1], low_1204_6710v2_y[::-1]):
        borsanyi_1204_6710v2.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2 = np.array(borsanyi_1204_6710v2)
    bazavov = np.array(bazavov)
    borsanyi = np.array(borsanyi)

    fig1 = plt.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    contrib_n = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N)]
    contrib_pi = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi)]
    contrib_rho = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho)]
    contrib_omega = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega)]
    contrib_t = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T)]
    contrib_p = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P)]
    contrib_h = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H)]
    contrib_d = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D)]
    contrib_f = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F)]
    contrib_q = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5)]
    contrib_qgp = [(pq_el + pg_el + ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_total = [sum(el) for el in zip(contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]
    ax1.axis([0., 200., 0., 2.0])
    ax1.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'red', label = 'color non-singlet clusters')
    ax1.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q, contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h)], y2 = [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'blue', label = 'color singlet clusters')
    ax1.plot(T, contrib_qgp, ':', c = 'black', label = r'$\mathrm{P_{QGP}}$')
    ax1.plot(T, contrib_total, '-', c = 'black', label = r'$\mathrm{P_{cluster}}$')
    ax1.plot(T, [el1 + el2 for el1, el2 in zip(contrib_qgp, contrib_total)], '--', c = 'black', label = r'$\mathrm{P_{total}}$')
    ax1.add_patch(Polygon(borsanyi_1204_6710v2, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2012)'))
    ax1.legend()
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig2 = plt.figure(num = 2, figsize = (7.4, 5))
    ax2 = fig2.add_subplot(1, 1, 1)

    contrib_n = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N)]
    contrib_pi = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi)]
    contrib_rho = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho)]
    contrib_omega = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega)]
    contrib_t = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T)]
    contrib_p = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P)]
    contrib_h = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H)]
    contrib_d = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D)]
    contrib_f = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F)]
    contrib_q = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5)]
    contrib_pert = [(ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_g = [(pg_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_quark = [(pq_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_qgp = [(pq_el + pg_el + ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_total = [sum(el) for el in zip(contrib_qgp, contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]

    contrib_n = [el/norm for el, norm in zip(contrib_n, contrib_total)]
    contrib_pi = [el/norm for el, norm in zip(contrib_pi, contrib_total)]
    contrib_rho = [el/norm for el, norm in zip(contrib_rho, contrib_total)]
    contrib_omega = [el/norm for el, norm in zip(contrib_omega, contrib_total)]
    contrib_t = [el/norm for el, norm in zip(contrib_t, contrib_total)]
    contrib_p = [el/norm for el, norm in zip(contrib_p, contrib_total)]
    contrib_h = [el/norm for el, norm in zip(contrib_h, contrib_total)]
    contrib_d = [el/norm for el, norm in zip(contrib_d, contrib_total)]
    contrib_f = [el/norm for el, norm in zip(contrib_f, contrib_total)]
    contrib_q = [el/norm for el, norm in zip(contrib_q, contrib_total)]
    contrib_qgp = [el/norm for el, norm in zip(contrib_qgp, contrib_total)]
    contrib_total = [el/norm for el, norm in zip(contrib_total, contrib_total)]

    ax2.axis([10., 200., 1e-6, 10.])
    ax2.plot(T, contrib_n       , '-'   , c = col_n     , label = r'nucleon'            )
    ax2.plot(T, contrib_pi      , '-'   , c = col_pi    , label = r'$\mathrm{\pi}$'     )
    ax2.plot(T, contrib_rho     , '-'   , c = col_rho   , label = r'$\mathrm{\rho}$'    )
    ax2.plot(T, contrib_omega   , '-'   , c = col_omega , label = r'$\mathrm{\omega}$'  )
    ax2.plot(T, contrib_t       , '-'   , c = col_t     , label = r'tetraquark'         )
    ax2.plot(T, contrib_p       , '-'   , c = col_p     , label = r'pentaquark'         )
    ax2.plot(T, contrib_h       , '-'   , c = col_h     , label = r'hexaquark'          )
    ax2.plot(T, contrib_d       , '--'  , c = col_d     , label = r'diquark'            )
    ax2.plot(T, contrib_f       , '--'  , c = col_f     , label = r'4--quark'           )
    ax2.plot(T, contrib_q       , '--'  , c = col_q5    , label = r'5--quark'           )
    ax2.plot(T, contrib_qgp     , '-'   , c = col_qgp   , label = r'QGP'                )
    ax2.plot(T, contrib_total   , '-'   , c = col_total , label = r'total'              )
    ax2.plot([0.0, 300.0], [1.0, 1.0], c = 'black')
    ax2.legend(bbox_to_anchor=(1.01, 0.85))
    ax2.set_yscale('log')

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'p fraction', fontsize = 16)

    fig3 = plt.figure(num = 3, figsize = (5.9, 5))
    ax3 = fig3.add_subplot(1, 1, 1)

    contrib_n = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N0)]
    contrib_pi = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi0)]
    contrib_rho = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho0)]
    contrib_omega = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega0)]
    contrib_t = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T0)]
    contrib_p = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P0)]
    contrib_h = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H0)]
    contrib_d = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D0)]
    contrib_f = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F0)]
    contrib_q = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q50)]
    contrib_qgp = [(pq_el + pg_el + ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q0, Pres_g0, Pres_pert0)]
    contrib_total = [sum(el) for el in zip(contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]

    ax3.axis([0., 200., 0., 2.0])
    ax3.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'red', label = 'color non-singlet clusters')
    ax3.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q, contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h)], y2 = [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'blue', label = 'color singlet clusters')
    ax3.plot(T, contrib_qgp, ':', c = 'black', label = r'$\mathrm{P_{QGP}}$')
    ax3.plot(T, contrib_total, '-', c = 'black', label = r'$\mathrm{P_{cluster}}$')
    ax3.plot(T, [el1 + el2 for el1, el2 in zip(contrib_qgp, contrib_total)], '--', c = 'black', label = r'$\mathrm{P_{total}}$')
    ax3.add_patch(Polygon(borsanyi_1204_6710v2, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2012)'))
    ax3.legend()
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig4 = plt.figure(num = 4, figsize = (7.4, 5))
    ax4 = fig4.add_subplot(1, 1, 1)

    contrib_n = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N0)]
    contrib_pi = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi0)]
    contrib_rho = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho0)]
    contrib_omega = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega0)]
    contrib_t = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T0)]
    contrib_p = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P0)]
    contrib_h = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H0)]
    contrib_d = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D0)]
    contrib_f = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F0)]
    contrib_q = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q50)]
    contrib_pert = [(ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q0, Pres_g0, Pres_pert0)]
    contrib_g = [(pg_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q0, Pres_g0, Pres_pert0)]
    contrib_quark = [(pq_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q0, Pres_g0, Pres_pert0)]
    contrib_qgp = [(pq_el + pg_el + ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q0, Pres_g0, Pres_pert0)]
    contrib_total = [sum(el) for el in zip(contrib_qgp, contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]

    contrib_n = [el/norm for el, norm in zip(contrib_n, contrib_total)]
    contrib_pi = [el/norm for el, norm in zip(contrib_pi, contrib_total)]
    contrib_rho = [el/norm for el, norm in zip(contrib_rho, contrib_total)]
    contrib_omega = [el/norm for el, norm in zip(contrib_omega, contrib_total)]
    contrib_t = [el/norm for el, norm in zip(contrib_t, contrib_total)]
    contrib_p = [el/norm for el, norm in zip(contrib_p, contrib_total)]
    contrib_h = [el/norm for el, norm in zip(contrib_h, contrib_total)]
    contrib_d = [el/norm for el, norm in zip(contrib_d, contrib_total)]
    contrib_f = [el/norm for el, norm in zip(contrib_f, contrib_total)]
    contrib_q = [el/norm for el, norm in zip(contrib_q, contrib_total)]
    contrib_qgp = [el/norm for el, norm in zip(contrib_qgp, contrib_total)]
    contrib_total = [el/norm for el, norm in zip(contrib_total, contrib_total)]

    ax4.axis([10., 200., 1e-6, 10.])
    ax4.plot(T, contrib_n       , '-'   , c = col_n     , label = r'nucleon'            )
    ax4.plot(T, contrib_pi      , '-'   , c = col_pi    , label = r'$\mathrm{\pi}$'     )
    ax4.plot(T, contrib_rho     , '-'   , c = col_rho   , label = r'$\mathrm{\rho}$'    )
    ax4.plot(T, contrib_omega   , '-'   , c = col_omega , label = r'$\mathrm{\omega}$'  )
    ax4.plot(T, contrib_t       , '-'   , c = col_t     , label = r'tetraquark'         )
    ax4.plot(T, contrib_p       , '-'   , c = col_p     , label = r'pentaquark'         )
    ax4.plot(T, contrib_h       , '-'   , c = col_h     , label = r'hexaquark'          )
    ax4.plot(T, contrib_d       , '--'  , c = col_d     , label = r'diquark'            )
    ax4.plot(T, contrib_f       , '--'  , c = col_f     , label = r'4--quark'           )
    ax4.plot(T, contrib_q       , '--'  , c = col_q5    , label = r'5--quark'           )
    ax4.plot(T, contrib_qgp     , '-'   , c = col_qgp   , label = r'QGP'                )
    ax4.plot(T, contrib_total   , '-'   , c = col_total , label = r'total'              )
    ax4.plot([0.0, 300.0], [1.0, 1.0], c = 'black')
    ax4.legend(bbox_to_anchor=(1.01, 0.85))
    ax4.set_yscale('log')

    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    ax4.set_xlabel(r'T [MeV]', fontsize = 16)
    ax4.set_ylabel(r'p fraction', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    fig2.tight_layout(pad = 0.1)
    fig3.tight_layout(pad = 0.1)
    fig4.tight_layout(pad = 0.1)
    plt.show()
    plt.close()

def epja_figure6():

    #przerobic ten plot zeby porownac zmiany mu=0..200 dla lattice i dla mojego modelu

    const_mu = 200.0
    (T, phi_re) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(const_mu)) + "p0_cluster_dense.dat")

    Pres_Q, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_g, _              = data_collect(2, 4, "D:/EoS/BDK/thermo_mu_const/pl_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_pert, _        = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_pi, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pi_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_rho, _          = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/rho_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_omega, _      = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/omega_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_D, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_N, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_T, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/T_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_F, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_P, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_Q5, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_H, _              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0.dat")

    Pres_Q0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_g0, _            = data_collect(2, 4, "D:/EoS/BDK/thermo_mu_const/pl_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_pert0, _      = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_pi0, _          = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pi_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_rho0, _        = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/rho_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_omega0, _    = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/omega_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_D0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_N0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_T0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/T_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_F0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_P0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_Q50, _          = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_H0, _            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0_no_phi.dat")

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

    (borsanyi_upper_x, borsanyi_upper_y) = data_collect(0, 1, "D:/EoS/archive/BDK/borsanyi_upper.dat")
    (borsanyi_lower_x, borsanyi_lower_y) = data_collect(0, 1, "D:/EoS/archive/BDK/borsanyi_lower.dat")
    (bazavov_upper_x, bazavov_upper_y) = data_collect(0, 1, "D:/EoS/archive/BDK/bazavov_upper.dat")
    (bazavov_lower_x, bazavov_lower_y) = data_collect(0, 1, "D:/EoS/archive/BDK/bazavov_lower.dat")
    (low_1204_6710v2_x, low_1204_6710v2_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_x, high_1204_6710v2_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")

    (low_1204_6710v2_x_alt, low_1204_6710v2_y_alt) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_x_alt, high_1204_6710v2_y_alt) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")

    bazavov = [np.array([x_el, y_el]) for x_el, y_el in zip(bazavov_upper_x, bazavov_upper_y)]
    for x_el, y_el in zip(bazavov_lower_x[::-1], bazavov_lower_y[::-1]):
        bazavov.append(np.array([x_el, y_el]))
    borsanyi = [np.array([x_el, y_el]) for x_el, y_el in zip(borsanyi_upper_x, borsanyi_upper_y)]
    for x_el, y_el in zip(borsanyi_lower_x[::-1], borsanyi_lower_y[::-1]):
        borsanyi.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2 = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_x, high_1204_6710v2_y)]
    for x_el, y_el in zip(low_1204_6710v2_x[::-1], low_1204_6710v2_y[::-1]):
        borsanyi_1204_6710v2.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2 = np.array(borsanyi_1204_6710v2)
    borsanyi_1204_6710v2_alt = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_x_alt, high_1204_6710v2_y_alt)]
    for x_el, y_el in zip(low_1204_6710v2_x_alt[::-1], low_1204_6710v2_y_alt[::-1]):
        borsanyi_1204_6710v2_alt.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2_alt = np.array(borsanyi_1204_6710v2_alt)
    bazavov = np.array(bazavov)
    borsanyi = np.array(borsanyi)

    fig1 = plt.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    contrib_n = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N)]
    contrib_pi = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi)]
    contrib_rho = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho)]
    contrib_omega = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega)]
    contrib_t = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T)]
    contrib_p = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P)]
    contrib_h = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H)]
    contrib_d = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D)]
    contrib_f = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F)]
    contrib_q = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5)]
    contrib_pert = [ppert_el / T_el ** 4 for T_el, ppert_el in zip(T, Pres_pert)]
    contrib_gluon = [ppert_el / T_el ** 4 for T_el, ppert_el in zip(T, Pres_g)]
    contrib_qgp = [(pq_el + pg_el + ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_total = [sum(el) for el in zip(contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]
    ax1.axis([0., 200., 0., 2.0])
    ax1.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'red', label = 'color non-singlet clusters')
    ax1.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q, contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h)], y2 = [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'blue', label = 'color singlet clusters')
    ax1.plot(T, contrib_qgp, ':', c = 'black', label = r'$\mathrm{P_{QGP}}$')
    ax1.plot(T, contrib_pert, ':', c = 'red', label = r'$\mathrm{P_{pert}}$')
    ax1.plot(T, contrib_gluon, ':', c = 'blue', label = r'$\mathrm{P_{pert}}$')
    ax1.plot(T, contrib_total, '-', c = 'black', label = r'$\mathrm{P_{cluster}}$')
    ax1.add_patch(Polygon(borsanyi_1204_6710v2, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2012)'))
    ax1.add_patch(Polygon(borsanyi_1204_6710v2_alt, closed = True, fill = True, color = 'red', alpha = 0.5, label = r'Borsanyi et al. (2012)'))
    ax1.plot(T, [el1 + el2 for el1, el2 in zip(contrib_qgp, contrib_total)], '--', c = 'black', label = r'$\mathrm{P_{total}}$')
    ax1.legend()
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig2 = plt.figure(num = 2, figsize = (7.4, 5))
    ax2 = fig2.add_subplot(1, 1, 1)

    contrib_n = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N)]
    contrib_pi = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi)]
    contrib_rho = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho)]
    contrib_omega = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega)]
    contrib_t = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T)]
    contrib_p = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P)]
    contrib_h = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H)]
    contrib_d = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D)]
    contrib_f = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F)]
    contrib_q = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5)]
    contrib_pert = [(ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_g = [(pg_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_quark = [(pq_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_qgp = [(pq_el + pg_el + ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q, Pres_g, Pres_pert)]
    contrib_total = [sum(el) for el in zip(contrib_qgp, contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]

    contrib_n = [el/norm for el, norm in zip(contrib_n, contrib_total)]
    contrib_pi = [el/norm for el, norm in zip(contrib_pi, contrib_total)]
    contrib_rho = [el/norm for el, norm in zip(contrib_rho, contrib_total)]
    contrib_omega = [el/norm for el, norm in zip(contrib_omega, contrib_total)]
    contrib_t = [el/norm for el, norm in zip(contrib_t, contrib_total)]
    contrib_p = [el/norm for el, norm in zip(contrib_p, contrib_total)]
    contrib_h = [el/norm for el, norm in zip(contrib_h, contrib_total)]
    contrib_d = [el/norm for el, norm in zip(contrib_d, contrib_total)]
    contrib_f = [el/norm for el, norm in zip(contrib_f, contrib_total)]
    contrib_q = [el/norm for el, norm in zip(contrib_q, contrib_total)]
    contrib_qgp = [el/norm for el, norm in zip(contrib_qgp, contrib_total)]
    contrib_total = [el/norm for el, norm in zip(contrib_total, contrib_total)]

    ax2.axis([10., 200., 1e-6, 10.])
    ax2.plot(T, contrib_n       , '-'   , c = col_n     , label = r'nucleon'            )
    ax2.plot(T, contrib_pi      , '-'   , c = col_pi    , label = r'$\mathrm{\pi}$'     )
    ax2.plot(T, contrib_rho     , '-'   , c = col_rho   , label = r'$\mathrm{\rho}$'    )
    ax2.plot(T, contrib_omega   , '-'   , c = col_omega , label = r'$\mathrm{\omega}$'  )
    ax2.plot(T, contrib_t       , '-'   , c = col_t     , label = r'tetraquark'         )
    ax2.plot(T, contrib_p       , '-'   , c = col_p     , label = r'pentaquark'         )
    ax2.plot(T, contrib_h       , '-'   , c = col_h     , label = r'hexaquark'          )
    ax2.plot(T, contrib_d       , '--'  , c = col_d     , label = r'diquark'            )
    ax2.plot(T, contrib_f       , '--'  , c = col_f     , label = r'4--quark'           )
    ax2.plot(T, contrib_q       , '--'  , c = col_q5    , label = r'5--quark'           )
    ax2.plot(T, contrib_qgp     , '-'   , c = col_qgp   , label = r'QGP'                )
    ax2.plot(T, contrib_total   , '-'   , c = col_total , label = r'total'              )
    ax2.plot([0.0, 300.0], [1.0, 1.0], c = 'black')
    ax2.legend(bbox_to_anchor=(1.01, 0.85))
    ax2.set_yscale('log')

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'p fraction', fontsize = 16)

    fig3 = plt.figure(num = 3, figsize = (5.9, 5))
    ax3 = fig3.add_subplot(1, 1, 1)

    contrib_n = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N0)]
    contrib_pi = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi0)]
    contrib_rho = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho0)]
    contrib_omega = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega0)]
    contrib_t = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T0)]
    contrib_p = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P0)]
    contrib_h = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H0)]
    contrib_d = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D0)]
    contrib_f = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F0)]
    contrib_q = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q50)]
    contrib_qgp = [(pq_el + pg_el + ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q0, Pres_g0, Pres_pert0)]
    contrib_total = [sum(el) for el in zip(contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]

    ax3.axis([0., 200., 0., 2.0])
    ax3.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'red', label = 'color non-singlet clusters')
    ax3.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q, contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h)], y2 = [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'blue', label = 'color singlet clusters')
    ax3.plot(T, contrib_qgp, ':', c = 'black', label = r'$\mathrm{P_{QGP}}$')
    ax3.plot(T, contrib_total, '-', c = 'black', label = r'$\mathrm{P_{cluster}}$')
    ax3.plot(T, [el1 + el2 for el1, el2 in zip(contrib_qgp, contrib_total)], '--', c = 'black', label = r'$\mathrm{P_{total}}$')
    ax3.add_patch(Polygon(borsanyi_1204_6710v2, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2012)'))
    ax3.legend()
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig4 = plt.figure(num = 4, figsize = (7.4, 5))
    ax4 = fig4.add_subplot(1, 1, 1)

    contrib_n = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N0)]
    contrib_pi = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi0)]
    contrib_rho = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho0)]
    contrib_omega = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega0)]
    contrib_t = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T0)]
    contrib_p = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P0)]
    contrib_h = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H0)]
    contrib_d = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D0)]
    contrib_f = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F0)]
    contrib_q = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q50)]
    contrib_pert = [(ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q0, Pres_g0, Pres_pert0)]
    contrib_g = [(pg_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q0, Pres_g0, Pres_pert0)]
    contrib_quark = [(pq_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q0, Pres_g0, Pres_pert0)]
    contrib_qgp = [(pq_el + pg_el + ppert_el) / (T_el ** 4) for T_el, pq_el, pg_el, ppert_el in zip(T, Pres_Q0, Pres_g0, Pres_pert0)]
    contrib_total = [sum(el) for el in zip(contrib_qgp, contrib_n, contrib_pi, contrib_rho, contrib_omega, contrib_t, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]

    contrib_n = [el/norm for el, norm in zip(contrib_n, contrib_total)]
    contrib_pi = [el/norm for el, norm in zip(contrib_pi, contrib_total)]
    contrib_rho = [el/norm for el, norm in zip(contrib_rho, contrib_total)]
    contrib_omega = [el/norm for el, norm in zip(contrib_omega, contrib_total)]
    contrib_t = [el/norm for el, norm in zip(contrib_t, contrib_total)]
    contrib_p = [el/norm for el, norm in zip(contrib_p, contrib_total)]
    contrib_h = [el/norm for el, norm in zip(contrib_h, contrib_total)]
    contrib_d = [el/norm for el, norm in zip(contrib_d, contrib_total)]
    contrib_f = [el/norm for el, norm in zip(contrib_f, contrib_total)]
    contrib_q = [el/norm for el, norm in zip(contrib_q, contrib_total)]
    contrib_qgp = [el/norm for el, norm in zip(contrib_qgp, contrib_total)]
    contrib_total = [el/norm for el, norm in zip(contrib_total, contrib_total)]

    ax4.axis([4.5, 200., 1e-6, 10.])
    ax4.plot(T, contrib_n       , '-'   , c = col_n     , label = r'nucleon'            )
    ax4.plot(T, contrib_pi      , '-'   , c = col_pi    , label = r'$\mathrm{\pi}$'     )
    ax4.plot(T, contrib_rho     , '-'   , c = col_rho   , label = r'$\mathrm{\rho}$'    )
    ax4.plot(T, contrib_omega   , '-'   , c = col_omega , label = r'$\mathrm{\omega}$'  )
    ax4.plot(T, contrib_t       , '-'   , c = col_t     , label = r'tetraquark'         )
    ax4.plot(T, contrib_p       , '-'   , c = col_p     , label = r'pentaquark'         )
    ax4.plot(T, contrib_h       , '-'   , c = col_h     , label = r'hexaquark'          )
    ax4.plot(T, contrib_d       , '--'  , c = col_d     , label = r'diquark'            )
    ax4.plot(T, contrib_f       , '--'  , c = col_f     , label = r'4--quark'           )
    ax4.plot(T, contrib_q       , '--'  , c = col_q5    , label = r'5--quark'           )
    ax4.plot(T, contrib_qgp     , '-'   , c = col_qgp   , label = r'QGP'                )
    ax4.plot(T, contrib_total   , '-'   , c = col_total , label = r'total'              )
    ax4.plot([0.0, 300.0], [1.0, 1.0], c = 'black')
    ax4.legend(bbox_to_anchor=(1.01, 0.85))
    ax4.set_yscale('log')

    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    ax4.set_xlabel(r'T [MeV]', fontsize = 16)
    ax4.set_ylabel(r'p fraction', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    fig2.tight_layout(pad = 0.1)
    fig3.tight_layout(pad = 0.1)
    fig4.tight_layout(pad = 0.1)
    plt.show()
    plt.close()

def epja_figure7():

    const_mu = 100.0
    (T, phi_re) = data_collect(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(const_mu)) + "p0_cluster_dense.dat")

    _ , BDen_Q              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_pert        = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_D              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_N              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_F              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_P              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_Q5            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_H              = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0.dat")

    _ , BDen_Q0            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_pert0      = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_D0            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_N0            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_F0            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_P0            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_Q50          = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_H0            = data_collect(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0_no_phi.dat")

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

    (borsanyi_upper_x, borsanyi_upper_y) = data_collect(0, 1, "D:/EoS/archive/BDK/borsanyi_upper.dat")
    (borsanyi_lower_x, borsanyi_lower_y) = data_collect(0, 1, "D:/EoS/archive/BDK/borsanyi_lower.dat")
    (bazavov_upper_x, bazavov_upper_y) = data_collect(0, 1, "D:/EoS/archive/BDK/bazavov_upper.dat")
    (bazavov_lower_x, bazavov_lower_y) = data_collect(0, 1, "D:/EoS/archive/BDK/bazavov_lower.dat")
    (low_1204_6710v2_x, low_1204_6710v2_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_x, high_1204_6710v2_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")

    bazavov = [np.array([x_el, y_el]) for x_el, y_el in zip(bazavov_upper_x, bazavov_upper_y)]
    for x_el, y_el in zip(bazavov_lower_x[::-1], bazavov_lower_y[::-1]):
        bazavov.append(np.array([x_el, y_el]))
    borsanyi = [np.array([x_el, y_el]) for x_el, y_el in zip(borsanyi_upper_x, borsanyi_upper_y)]
    for x_el, y_el in zip(borsanyi_lower_x[::-1], borsanyi_lower_y[::-1]):
        borsanyi.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2 = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_x, high_1204_6710v2_y)]
    for x_el, y_el in zip(low_1204_6710v2_x[::-1], low_1204_6710v2_y[::-1]):
        borsanyi_1204_6710v2.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2 = np.array(borsanyi_1204_6710v2)
    bazavov = np.array(bazavov)
    borsanyi = np.array(borsanyi)

    fig1 = plt.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    contrib_n = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_N)]
    contrib_p = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_P)]
    contrib_h = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_H)]
    contrib_d = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_D)]
    contrib_f = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_F)]
    contrib_q = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_Q5)]
    contrib_qgp = [(pq_el + ppert_el) / (T_el ** 3) for T_el, pq_el, ppert_el in zip(T, BDen_Q, BDen_pert)]
    contrib_total = [sum(el) for el in zip(contrib_n, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]
    ax1.axis([0., 200., 0., 1.0])
    ax1.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'red', label = 'color non-singlet clusters')
    ax1.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q, contrib_n, contrib_p, contrib_h)], y2 = [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'blue', label = 'color singlet clusters')
    ax1.plot(T, contrib_qgp, ':', c = 'black', label = r'$\mathrm{n_{QGP}}$')
    ax1.plot(T, contrib_total, '-', c = 'black', label = r'$\mathrm{n_{cluster}}$')
    ax1.plot(T, [el1 + el2 for el1, el2 in zip(contrib_qgp, contrib_total)], '--', c = 'black', label = r'$\mathrm{n_{total}}$')
    ax1.legend()
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    fig2 = plt.figure(num = 2, figsize = (7.4, 5))
    ax2 = fig2.add_subplot(1, 1, 1)

    contrib_n = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_N)]
    contrib_p = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_P)]
    contrib_h = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_H)]
    contrib_d = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_D)]
    contrib_f = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_F)]
    contrib_q = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_Q5)]
    contrib_pert = [(ppert_el) / (T_el ** 3) for T_el, pq_el, ppert_el in zip(T, BDen_Q, BDen_pert)]
    contrib_quark = [(pq_el) / (T_el ** 3) for T_el, pq_el, ppert_el in zip(T, BDen_Q, BDen_pert)]
    contrib_qgp = [(pq_el + ppert_el) / (T_el ** 3) for T_el, pq_el, ppert_el in zip(T, BDen_Q, BDen_pert)]
    contrib_total = [sum(el) for el in zip(contrib_qgp, contrib_n, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]

    contrib_n = [el/norm for el, norm in zip(contrib_n, contrib_total)]
    contrib_p = [el/norm for el, norm in zip(contrib_p, contrib_total)]
    contrib_h = [el/norm for el, norm in zip(contrib_h, contrib_total)]
    contrib_d = [el/norm for el, norm in zip(contrib_d, contrib_total)]
    contrib_f = [el/norm for el, norm in zip(contrib_f, contrib_total)]
    contrib_q = [el/norm for el, norm in zip(contrib_q, contrib_total)]
    contrib_qgp = [el/norm for el, norm in zip(contrib_qgp, contrib_total)]
    contrib_total = [el/norm for el, norm in zip(contrib_total, contrib_total)]

    ax2.axis([10., 200., 1e-6, 10.])
    ax2.plot(T[100:], contrib_n[100:]       , '-'   , c = col_n     , label = r'nucleon'            )
    ax2.plot(T, contrib_p       , '-'   , c = col_p     , label = r'pentaquark'         )
    ax2.plot(T, contrib_h       , '-'   , c = col_h     , label = r'hexaquark'          )
    ax2.plot(T, contrib_d       , '--'  , c = col_d     , label = r'diquark'            )
    ax2.plot(T, contrib_f       , '--'  , c = col_f     , label = r'4--quark'           )
    ax2.plot(T, contrib_q       , '--'  , c = col_q5    , label = r'5--quark'           )
    ax2.plot(T[100:], contrib_qgp[100:], '-'   , c = col_qgp   , label = r'QGP'                )
    #ax2.scatter(T[103], contrib_qgp[103], s = 20, c = 'blue')
    #ax2.scatter(T[200], contrib_qgp[200], s = 20, c = 'blue')
    ax2.plot(T, contrib_total   , '-'   , c = col_total , label = r'total'              )
    ax2.plot([0.0, 300.0], [1.0, 1.0], c = 'black')
    ax2.legend(bbox_to_anchor=(1.3, 0.74))
    ax2.set_yscale('log')

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{n_B}$ fraction', fontsize = 16)

    fig3 = plt.figure(num = 3, figsize = (5.9, 5))
    ax3 = fig3.add_subplot(1, 1, 1)

    contrib_n = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_N0)]
    contrib_p = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_P0)]
    contrib_h = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_H0)]
    contrib_d = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_D0)]
    contrib_f = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_F0)]
    contrib_q = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_Q50)]
    contrib_qgp = [(pq_el + ppert_el) / (T_el ** 3) for T_el, pq_el, ppert_el in zip(T, BDen_Q0, BDen_pert0)]
    contrib_total = [sum(el) for el in zip(contrib_n, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]

    ax3.axis([0., 200., 0., 1.0])
    ax3.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'red', label = 'color non-singlet clusters')
    ax3.fill_between(T, [sum(el) for el in zip(contrib_d, contrib_f, contrib_q, contrib_n, contrib_p, contrib_h)], y2 = [sum(el) for el in zip(contrib_d, contrib_f, contrib_q)], color = 'blue', label = 'color singlet clusters')
    ax3.plot(T, contrib_qgp, ':', c = 'black', label = r'$\mathrm{n_{QGP}}$')
    ax3.plot(T, contrib_total, '-', c = 'black', label = r'$\mathrm{n_{cluster}}$')
    ax3.plot(T, [el1 + el2 for el1, el2 in zip(contrib_qgp, contrib_total)], '--', c = 'black', label = r'$\mathrm{n_{total}}$')
    ax3.legend()
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    fig4 = plt.figure(num = 4, figsize = (7.4, 5))
    ax4 = fig4.add_subplot(1, 1, 1)

    contrib_n = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_N0)]
    contrib_p = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_P0)]
    contrib_h = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_H0)]
    contrib_d = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_D0)]
    contrib_f = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_F0)]
    contrib_q = [p_el / (T_el ** 3) for T_el, p_el in zip(T, BDen_Q50)]
    contrib_pert = [(ppert_el) / (T_el ** 3) for T_el, pq_el, ppert_el in zip(T, BDen_Q0, BDen_pert0)]
    contrib_quark = [(pq_el) / (T_el ** 3) for T_el, pq_el, ppert_el in zip(T, BDen_Q0, BDen_pert0)]
    contrib_qgp = [(pq_el + ppert_el) / (T_el ** 3) for T_el, pq_el, ppert_el in zip(T, BDen_Q0, BDen_pert0)]
    contrib_total = [sum(el) for el in zip(contrib_qgp, contrib_n, contrib_p, contrib_h, contrib_d, contrib_f, contrib_q)]

    contrib_n = [el/norm for el, norm in zip(contrib_n, contrib_total)]
    contrib_p = [el/norm for el, norm in zip(contrib_p, contrib_total)]
    contrib_h = [el/norm for el, norm in zip(contrib_h, contrib_total)]
    contrib_d = [el/norm for el, norm in zip(contrib_d, contrib_total)]
    contrib_f = [el/norm for el, norm in zip(contrib_f, contrib_total)]
    contrib_q = [el/norm for el, norm in zip(contrib_q, contrib_total)]
    contrib_qgp = [el/norm for el, norm in zip(contrib_qgp, contrib_total)]
    contrib_total = [el/norm for el, norm in zip(contrib_total, contrib_total)]

    ax4.axis([4.5, 200., 1e-6, 10.])
    ax4.plot(T, contrib_n       , '-'   , c = col_n     , label = r'nucleon'            )
    ax4.plot(T, contrib_p       , '-'   , c = col_p     , label = r'pentaquark'         )
    ax4.plot(T, contrib_h       , '-'   , c = col_h     , label = r'hexaquark'          )
    ax4.plot(T, contrib_d       , '--'  , c = col_d     , label = r'diquark'            )
    ax4.plot(T, contrib_f       , '--'  , c = col_f     , label = r'4--quark'           )
    ax4.plot(T, contrib_q       , '--'  , c = col_q5    , label = r'5--quark'           )
    ax4.plot(T, contrib_qgp     , '-'   , c = col_qgp   , label = r'QGP'                )
    ax4.plot(T, contrib_total   , '-'   , c = col_total , label = r'total'              )
    ax4.plot([0.0, 300.0], [1.0, 1.0], c = 'black')
    ax4.legend(bbox_to_anchor=(1.3, 0.74))
    ax4.set_yscale('log')

    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    ax4.set_xlabel(r'T [MeV]', fontsize = 16)
    ax4.set_ylabel(r'$\mathrm{n_B}$ fraction', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    #fig2.tight_layout()
    fig2.tight_layout(pad = 0.1)
    fig3.tight_layout(pad = 0.1)
    #fig4.tight_layout()
    fig4.tight_layout(pad = 0.1)
    plt.show()
    plt.close()

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
    
    T = np.linspace(1.0, 450.0, 200)
    mu0   = [0.0 for el in T]
    mu200 = [200.0 / 3.0 for el in T]
    mu300 = [300.0 / 3.0 for el in T]
    
    recalc_pl_mu0         = False
    recalc_pl_mu200       = False
    recalc_pl_mu300       = False
    recalc_pressure_mu0   = True
    recalc_pressure_mu200 = True
    recalc_pressure_mu300 = True

    pl_turned_off = False

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
        for T_el, mu_el in tqdm(zip(T, mu0), desc = "Traced Polyakov loop (mu = 0)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu0[-1], phi_im_mu0[-1], with_clusters = True, perturbative_kwargs = {'Nf' : 2.0})
            phi_re_mu0.append(temp_phi_re)
            phi_im_mu0.append(temp_phi_im)
        phi_re_mu0 = phi_re_mu0[1:]
        phi_im_mu0 = phi_im_mu0[1:]
        with open(pl_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu0, phi_im_mu0)])
    else:
        T, phi_re_mu0 = data_collect(0, 1, pl_mu0_file)
        phi_im_mu0, _ = data_collect(2, 2, pl_mu0_file)

    if recalc_pl_mu200:
        phi_re_mu200.append(1e-15)
        phi_im_mu200.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm(zip(T, mu200), desc = "Traced Polyakov loop (mu = 200)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu200[-1], phi_im_mu200[-1], with_clusters = True, perturbative_kwargs = {'Nf' : 2.0})
            phi_re_mu200.append(temp_phi_re)
            phi_im_mu200.append(temp_phi_im)
        phi_re_mu200 = phi_re_mu200[1:]
        phi_im_mu200 = phi_im_mu200[1:]
        with open(pl_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu200, phi_im_mu200)])
    else:
        T, phi_re_mu200 = data_collect(0, 1, pl_mu200_file)
        phi_im_mu200, _ = data_collect(2, 2, pl_mu200_file)
    
    if recalc_pl_mu300:
        phi_re_mu300.append(1e-15)
        phi_im_mu300.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm(zip(T, mu300), desc = "Traced Polyakov loop (mu = 300)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu300[-1], phi_im_mu300[-1], with_clusters = True, perturbative_kwargs = {'Nf' : 2.0})
            phi_re_mu300.append(temp_phi_re)
            phi_im_mu300.append(temp_phi_im)
        phi_re_mu300 = phi_re_mu300[1:]
        phi_im_mu300 = phi_im_mu300[1:]
        with open(pl_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu300, phi_im_mu300)])
    else:
        T, phi_re_mu300 = data_collect(0, 1, pl_mu300_file)
        phi_im_mu300, _ = data_collect(2, 2, pl_mu300_file)

    if recalc_pressure_mu0:
        Pres_g_mu0 = [Pressure_g(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) for T_el, phi_re_el, phi_im_el in tqdm(zip(T, phi_re_mu0, phi_im_mu0), desc = "Gluon pressure (mu = 0)", total = len(T), ascii = True)]
        if not pl_turned_off:
            Pres_Q_mu0 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu0, phi_re_mu0, phi_im_mu0), desc = "Quark pressure (mu = 0)", total = len(T), ascii = True)]
            Pres_pert_mu0 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu0, phi_re_mu0, phi_im_mu0), desc = "Perturbative pressure (mu = 0)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu0, phi_re_mu0, phi_im_mu0)
        else:
            Pres_Q_mu0 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu0, [1.0 for el in T], [0.0 for el in T]), desc = "Quark pressure (mu = 0)", total = len(T), ascii = True)]
            Pres_pert_mu0 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu0, [1.0 for el in T], [0.0 for el in T]), desc = "Perturbative pressure (mu = 0)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu0, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu0, Pres_g_mu0, Pres_pert_mu0, Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0)])
    else:
        Pres_Q_mu0, Pres_g_mu0 = data_collect(0, 1, pressure_mu0_file)
        Pres_pert_mu0, Pres_pi_mu0 = data_collect(2, 3, pressure_mu0_file)
        Pres_rho_mu0, Pres_omega_mu0 = data_collect(4, 5, pressure_mu0_file)
        Pres_D_mu0, Pres_N_mu0 = data_collect(6, 7, pressure_mu0_file)
        Pres_T_mu0, Pres_F_mu0 = data_collect(8, 9, pressure_mu0_file)
        Pres_P_mu0, Pres_Q5_mu0 = data_collect(10, 11, pressure_mu0_file)
        Pres_H_mu0, _ = data_collect(12, 12, pressure_mu0_file)

    if recalc_pressure_mu200:
        Pres_g_mu200 = [Pressure_g(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) for T_el, phi_re_el, phi_im_el in tqdm(zip(T, phi_re_mu200, phi_im_mu200), desc = "Gluon pressure (mu = 200)", total = len(T), ascii = True)]
        if not pl_turned_off:
            Pres_Q_mu200 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu200, phi_re_mu200, phi_im_mu200), desc = "Quark pressure (mu = 200)", total = len(T), ascii = True)]        
            Pres_pert_mu200 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu200, phi_re_mu200, phi_im_mu200), desc = "Perturbative pressure (mu = 200)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu200, phi_re_mu200, phi_im_mu200)
        else:
            Pres_Q_mu200 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu200, [1.0 for el in T], [0.0 for el in T]), desc = "Quark pressure (mu = 200)", total = len(T), ascii = True)]        
            Pres_pert_mu200 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu200, [1.0 for el in T], [0.0 for el in T]), desc = "Perturbative pressure (mu = 200)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu200, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu200, Pres_g_mu200, Pres_pert_mu200, Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200)])
    else:
        Pres_Q_mu200, Pres_g_mu200 = data_collect(0, 1, pressure_mu200_file)
        Pres_pert_mu200, Pres_pi_mu200 = data_collect(2, 3, pressure_mu200_file)
        Pres_rho_mu200, Pres_omega_mu200 = data_collect(4, 5, pressure_mu200_file)
        Pres_D_mu200, Pres_N_mu200 = data_collect(6, 7, pressure_mu200_file)
        Pres_T_mu200, Pres_F_mu200 = data_collect(8, 9, pressure_mu200_file)
        Pres_P_mu200, Pres_Q5_mu200 = data_collect(10, 11, pressure_mu200_file)
        Pres_H_mu200, _ = data_collect(12, 12, pressure_mu200_file)
    
    if recalc_pressure_mu300:
        Pres_g_mu300 = [Pressure_g(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) for T_el, phi_re_el, phi_im_el in tqdm(zip(T, phi_re_mu300, phi_im_mu300), desc = "Gluon pressure (mu = 300)", total = len(T), ascii = True)]
        if not pl_turned_off:
            Pres_Q_mu300 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu300, phi_re_mu300, phi_im_mu300), desc = "Quark pressure (mu = 300)", total = len(T), ascii = True)]        
            Pres_pert_mu300 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu300, phi_re_mu300, phi_im_mu300), desc = "Perturbative pressure (mu = 300)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu300, phi_re_mu300, phi_im_mu300)
        else:
            Pres_Q_mu300 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu300, [1.0 for el in T], [0.0 for el in T]), desc = "Quark pressure (mu = 300)", total = len(T), ascii = True)]        
            Pres_pert_mu300 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu300, [1.0 for el in T], [0.0 for el in T]), desc = "Perturbative pressure (mu = 300)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu300, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu300, Pres_g_mu300, Pres_pert_mu300, Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300)])
    else:
        Pres_Q_mu300, Pres_g_mu300 = data_collect(0, 1, pressure_mu300_file)
        Pres_pert_mu300, Pres_pi_mu300 = data_collect(2, 3, pressure_mu300_file)
        Pres_rho_mu300, Pres_omega_mu300 = data_collect(4, 5, pressure_mu300_file)
        Pres_D_mu300, Pres_N_mu300 = data_collect(6, 7, pressure_mu300_file)
        Pres_T_mu300, Pres_F_mu300 = data_collect(8, 9, pressure_mu300_file)
        Pres_P_mu300, Pres_Q5_mu300 = data_collect(10, 11, pressure_mu300_file)
        Pres_H_mu300, _ = data_collect(12, 12, pressure_mu300_file)

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

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_low.dat")
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_high.dat")

    borsanyi_1204_6710v2_mu0 = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = np.array(borsanyi_1204_6710v2_mu0)
    borsanyi_1204_6710v2_mu200 = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu200_x[::-1], low_1204_6710v2_mu200_y[::-1]):
        borsanyi_1204_6710v2_mu200.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu200 = np.array(borsanyi_1204_6710v2_mu200)
    borsanyi_1204_6710v2_mu300 = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu300_x[::-1], low_1204_6710v2_mu300_y[::-1]):
        borsanyi_1204_6710v2_mu300.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu300 = np.array(borsanyi_1204_6710v2_mu300)

    fig1 = plt.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([0., 450., 0., 4.0])
    ax1.add_patch(Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=0}$'))
    ax1.add_patch(Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'red', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=200}$ MeV'))
    ax1.add_patch(Polygon(borsanyi_1204_6710v2_mu300, closed = True, fill = True, color = 'yellow', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=300}$ MeV'))
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

    fig2 = plt.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.axis([0., 450., 0., 1.2])
    ax2.plot(T, [M(el, 0) / M(0, 0) for el in T], '-', c = 'green')
    ax2.plot(T, [M(el, 200.0 / 3.0) / M(0, 0) for el in T], '--', c = 'green')
    ax2.plot(T, [M(el, 300.0 / 3.0) / M(0, 0) for el in T], '-.', c = 'green')
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

    plt.show()
    plt.close()

def PNJL_thermodynamics_continuum():

    print("continuum_lambda_basis =", continuum_lambda_basis, "MeV")
    input()

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
    
    Pres_pi_step_mu0         = []
    Pres_rho_step_mu0        = []
    Pres_omega_step_mu0      = []
    Pres_D_step_mu0          = []
    Pres_N_step_mu0          = []
    Pres_T_step_mu0          = []
    Pres_F_step_mu0          = []
    Pres_P_step_mu0          = []
    Pres_Q5_step_mu0         = []
    Pres_H_step_mu0          = []
    Pres_pi_step_mu200       = []
    Pres_rho_step_mu200      = []
    Pres_omega_step_mu200    = []
    Pres_D_step_mu200        = []
    Pres_N_step_mu200        = []
    Pres_T_step_mu200        = []
    Pres_F_step_mu200        = []
    Pres_P_step_mu200        = []
    Pres_Q5_step_mu200       = []
    Pres_H_step_mu200        = []
    Pres_pi_step_mu300       = []
    Pres_rho_step_mu300      = []
    Pres_omega_step_mu300    = []
    Pres_D_step_mu300        = []
    Pres_N_step_mu300        = []
    Pres_T_step_mu300        = []
    Pres_F_step_mu300        = []
    Pres_P_step_mu300        = []
    Pres_Q5_step_mu300       = []
    Pres_H_step_mu300        = []

    T = np.linspace(1.0, 450.0, 200)
    mu0   = [0.0 for el in T]
    mu200 = [200.0 / 3.0 for el in T]
    mu300 = [300.0 / 3.0 for el in T]
    
    recalc_pl_mu0         = False
    recalc_pl_mu200       = False
    recalc_pl_mu300       = False
    recalc_pressure_mu0   = False
    recalc_pressure_mu200 = False
    recalc_pressure_mu300 = False

    pl_turned_off = False

    pl_mu0_file   = "D:/EoS/BDK/mu_test/pl_mu0.dat"
    pl_mu200_file = "D:/EoS/BDK/mu_test/pl_mu200.dat"
    pl_mu300_file = "D:/EoS/BDK/mu_test/pl_mu300.dat"
    pressure_mu0_step_file   = "D:/EoS/BDK/mu_test/pressure_mu0.dat"
    pressure_mu200_step_file = "D:/EoS/BDK/mu_test/pressure_mu200.dat"
    pressure_mu300_step_file = "D:/EoS/BDK/mu_test/pressure_mu300.dat"
    pressure_mu0_file   = "D:/EoS/BDK/mu_test/pressure_c_mu0.dat"
    pressure_mu200_file = "D:/EoS/BDK/mu_test/pressure_c_mu200.dat"
    pressure_mu300_file = "D:/EoS/BDK/mu_test/pressure_c_mu300.dat"

    if recalc_pl_mu0:
        phi_re_mu0.append(1e-15)
        phi_im_mu0.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm(zip(T, mu0), desc = "Traced Polyakov loop (mu = 0)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu0[-1], phi_im_mu0[-1], with_clusters = True, perturbative_kwargs = {'Nf' : 2.0})
            phi_re_mu0.append(temp_phi_re)
            phi_im_mu0.append(temp_phi_im)
        phi_re_mu0 = phi_re_mu0[1:]
        phi_im_mu0 = phi_im_mu0[1:]
        with open(pl_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu0, phi_im_mu0)])
    else:
        T, phi_re_mu0 = data_collect(0, 1, pl_mu0_file)
        phi_im_mu0, _ = data_collect(2, 2, pl_mu0_file)

    if recalc_pl_mu200:
        phi_re_mu200.append(1e-15)
        phi_im_mu200.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm(zip(T, mu200), desc = "Traced Polyakov loop (mu = 200)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu200[-1], phi_im_mu200[-1], with_clusters = True, perturbative_kwargs = {'Nf' : 2.0})
            phi_re_mu200.append(temp_phi_re)
            phi_im_mu200.append(temp_phi_im)
        phi_re_mu200 = phi_re_mu200[1:]
        phi_im_mu200 = phi_im_mu200[1:]
        with open(pl_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu200, phi_im_mu200)])
    else:
        T, phi_re_mu200 = data_collect(0, 1, pl_mu200_file)
        phi_im_mu200, _ = data_collect(2, 2, pl_mu200_file)
    
    if recalc_pl_mu300:
        phi_re_mu300.append(1e-15)
        phi_im_mu300.append(2e-15)
        lT = len(T)
        for T_el, mu_el in tqdm(zip(T, mu300), desc = "Traced Polyakov loop (mu = 300)", total = len(T), ascii = True):
            temp_phi_re, temp_phi_im = calc_PL(T_el, mu_el, phi_re_mu300[-1], phi_im_mu300[-1], with_clusters = True, perturbative_kwargs = {'Nf' : 2.0})
            phi_re_mu300.append(temp_phi_re)
            phi_im_mu300.append(temp_phi_im)
        phi_re_mu300 = phi_re_mu300[1:]
        phi_im_mu300 = phi_im_mu300[1:]
        with open(pl_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[T_el, phi_re_el, phi_im_el] for T_el, phi_re_el, phi_im_el in zip(T, phi_re_mu300, phi_im_mu300)])
    else:
        T, phi_re_mu300 = data_collect(0, 1, pl_mu300_file)
        phi_im_mu300, _ = data_collect(2, 2, pl_mu300_file)

    if recalc_pressure_mu0:
        Pres_g_mu0 = [Pressure_g(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) for T_el, phi_re_el, phi_im_el in tqdm(zip(T, phi_re_mu0, phi_im_mu0), desc = "Gluon pressure (mu = 0)", total = len(T), ascii = True)]
        if not pl_turned_off:
            Pres_Q_mu0 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu0, phi_re_mu0, phi_im_mu0), desc = "Quark pressure (mu = 0)", total = len(T), ascii = True)]
            Pres_pert_mu0 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu0, phi_re_mu0, phi_im_mu0), desc = "Perturbative pressure (mu = 0)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu0, phi_re_mu0, phi_im_mu0)
        else:
            Pres_Q_mu0 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu0, [1.0 for el in T], [0.0 for el in T]), desc = "Quark pressure (mu = 0)", total = len(T), ascii = True)]
            Pres_pert_mu0 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu0, [1.0 for el in T], [0.0 for el in T]), desc = "Perturbative pressure (mu = 0)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu0, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu0_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu0, Pres_g_mu0, Pres_pert_mu0, Pres_pi_mu0, Pres_rho_mu0, Pres_omega_mu0, Pres_D_mu0, Pres_N_mu0, Pres_T_mu0, Pres_F_mu0, Pres_P_mu0, Pres_Q5_mu0, Pres_H_mu0)])
    else:
        Pres_Q_mu0, Pres_g_mu0 = data_collect(0, 1, pressure_mu0_file)
        Pres_pert_mu0, Pres_pi_mu0 = data_collect(2, 3, pressure_mu0_file)
        Pres_rho_mu0, Pres_omega_mu0 = data_collect(4, 5, pressure_mu0_file)
        Pres_D_mu0, Pres_N_mu0 = data_collect(6, 7, pressure_mu0_file)
        Pres_T_mu0, Pres_F_mu0 = data_collect(8, 9, pressure_mu0_file)
        Pres_P_mu0, Pres_Q5_mu0 = data_collect(10, 11, pressure_mu0_file)
        Pres_H_mu0, _ = data_collect(12, 12, pressure_mu0_file)

    Pres_H_step_mu0, Pres_pi_step_mu0 = data_collect(12, 3, pressure_mu0_step_file)
    Pres_rho_step_mu0, Pres_omega_step_mu0 = data_collect(4, 5, pressure_mu0_step_file)
    Pres_D_step_mu0, Pres_N_step_mu0 = data_collect(6, 7, pressure_mu0_step_file)
    Pres_T_step_mu0, Pres_F_step_mu0 = data_collect(8, 9, pressure_mu0_step_file)
    Pres_P_step_mu0, Pres_Q5_step_mu0 = data_collect(10, 11, pressure_mu0_step_file)

    if recalc_pressure_mu200:
        Pres_g_mu200 = [Pressure_g(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) for T_el, phi_re_el, phi_im_el in tqdm(zip(T, phi_re_mu200, phi_im_mu200), desc = "Gluon pressure (mu = 200)", total = len(T), ascii = True)]
        if not pl_turned_off:
            Pres_Q_mu200 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu200, phi_re_mu200, phi_im_mu200), desc = "Quark pressure (mu = 200)", total = len(T), ascii = True)]        
            Pres_pert_mu200 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu200, phi_re_mu200, phi_im_mu200), desc = "Perturbative pressure (mu = 200)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu200, phi_re_mu200, phi_im_mu200)
        else:
            Pres_Q_mu200 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu200, [1.0 for el in T], [0.0 for el in T]), desc = "Quark pressure (mu = 200)", total = len(T), ascii = True)]        
            Pres_pert_mu200 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu200, [1.0 for el in T], [0.0 for el in T]), desc = "Perturbative pressure (mu = 200)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu200, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu200_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu200, Pres_g_mu200, Pres_pert_mu200, Pres_pi_mu200, Pres_rho_mu200, Pres_omega_mu200, Pres_D_mu200, Pres_N_mu200, Pres_T_mu200, Pres_F_mu200, Pres_P_mu200, Pres_Q5_mu200, Pres_H_mu200)])
    else:
        Pres_Q_mu200, Pres_g_mu200 = data_collect(0, 1, pressure_mu200_file)
        Pres_pert_mu200, Pres_pi_mu200 = data_collect(2, 3, pressure_mu200_file)
        Pres_rho_mu200, Pres_omega_mu200 = data_collect(4, 5, pressure_mu200_file)
        Pres_D_mu200, Pres_N_mu200 = data_collect(6, 7, pressure_mu200_file)
        Pres_T_mu200, Pres_F_mu200 = data_collect(8, 9, pressure_mu200_file)
        Pres_P_mu200, Pres_Q5_mu200 = data_collect(10, 11, pressure_mu200_file)
        Pres_H_mu200, _ = data_collect(12, 12, pressure_mu200_file)
    
    Pres_H_step_mu200, Pres_pi_step_mu200 = data_collect(12, 3, pressure_mu200_step_file)
    Pres_rho_step_mu200, Pres_omega_step_mu200 = data_collect(4, 5, pressure_mu200_step_file)
    Pres_D_step_mu200, Pres_N_step_mu200 = data_collect(6, 7, pressure_mu200_step_file)
    Pres_T_step_mu200, Pres_F_step_mu200 = data_collect(8, 9, pressure_mu200_step_file)
    Pres_P_step_mu200, Pres_Q5_step_mu200 = data_collect(10, 11, pressure_mu200_step_file)

    if recalc_pressure_mu300:
        Pres_g_mu300 = [Pressure_g(T_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) for T_el, phi_re_el, phi_im_el in tqdm(zip(T, phi_re_mu300, phi_im_mu300), desc = "Gluon pressure (mu = 300)", total = len(T), ascii = True)]
        if not pl_turned_off:
            Pres_Q_mu300 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu300, phi_re_mu300, phi_im_mu300), desc = "Quark pressure (mu = 300)", total = len(T), ascii = True)]        
            Pres_pert_mu300 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu300, phi_re_mu300, phi_im_mu300), desc = "Perturbative pressure (mu = 300)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu300, phi_re_mu300, phi_im_mu300)
        else:
            Pres_Q_mu300 = [alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el)) + alt_Pressure_Q(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 1.0, ml = 100.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu300, [1.0 for el in T], [0.0 for el in T]), desc = "Quark pressure (mu = 300)", total = len(T), ascii = True)]        
            Pres_pert_mu300 = [Pressure_pert(T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el), Nf = 2.0) for T_el, mu_el, phi_re_el, phi_im_el in tqdm(zip(T, mu300, [1.0 for el in T], [0.0 for el in T]), desc = "Perturbative pressure (mu = 300)", total = len(T), ascii = True)]        
            (
                (Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300),
                (_, _, _, _, _, _, _, _, _, _),
                (_, _, _, _, _, _, _, _, _, _)
            ) = clusters(T, mu300, [1.0 for el in T], [0.0 for el in T])
        with open(pressure_mu300_file, 'w', newline = '') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerows([[q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el] for q_el, g_el, pert_el, pi_el, rho_el, omega_el, D_el, N_el, T_el, F_el, P_el, Q5_el, H_el in zip(Pres_Q_mu300, Pres_g_mu300, Pres_pert_mu300, Pres_pi_mu300, Pres_rho_mu300, Pres_omega_mu300, Pres_D_mu300, Pres_N_mu300, Pres_T_mu300, Pres_F_mu300, Pres_P_mu300, Pres_Q5_mu300, Pres_H_mu300)])
    else:
        Pres_Q_mu300, Pres_g_mu300 = data_collect(0, 1, pressure_mu300_file)
        Pres_pert_mu300, Pres_pi_mu300 = data_collect(2, 3, pressure_mu300_file)
        Pres_rho_mu300, Pres_omega_mu300 = data_collect(4, 5, pressure_mu300_file)
        Pres_D_mu300, Pres_N_mu300 = data_collect(6, 7, pressure_mu300_file)
        Pres_T_mu300, Pres_F_mu300 = data_collect(8, 9, pressure_mu300_file)
        Pres_P_mu300, Pres_Q5_mu300 = data_collect(10, 11, pressure_mu300_file)
        Pres_H_mu300, _ = data_collect(12, 12, pressure_mu300_file)

    Pres_H_step_mu300, Pres_pi_step_mu300 = data_collect(12, 3, pressure_mu300_step_file)
    Pres_rho_step_mu300, Pres_omega_step_mu300 = data_collect(4, 5, pressure_mu300_step_file)
    Pres_D_step_mu300, Pres_N_step_mu300 = data_collect(6, 7, pressure_mu300_step_file)
    Pres_T_step_mu300, Pres_F_step_mu300 = data_collect(8, 9, pressure_mu300_step_file)
    Pres_P_step_mu300, Pres_Q5_step_mu300 = data_collect(10, 11, pressure_mu300_step_file)

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

    contrib_pi_step_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi_step_mu0)]
    contrib_rho_step_mu0                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho_step_mu0)]
    contrib_omega_step_mu0               = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega_step_mu0)]
    contrib_D_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D_step_mu0)]
    contrib_N_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N_step_mu0)]
    contrib_T_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T_step_mu0)]
    contrib_F_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F_step_mu0)]
    contrib_P_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P_step_mu0)]
    contrib_Q5_step_mu0                  = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5_step_mu0)]
    contrib_H_step_mu0                   = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H_step_mu0)]
    contrib_cluster_step_mu0             = [sum(el) for el in zip(contrib_pi_step_mu0, contrib_rho_step_mu0, contrib_omega_step_mu0, contrib_D_step_mu0, contrib_N_step_mu0, contrib_T_step_mu0, contrib_F_step_mu0, contrib_P_step_mu0, contrib_Q5_step_mu0, contrib_H_step_mu0)]
    contrib_cluster_singlet_step_mu0     = [sum(el) for el in zip(contrib_pi_step_mu0, contrib_rho_step_mu0, contrib_omega_step_mu0, contrib_N_step_mu0, contrib_T_step_mu0, contrib_P_step_mu0, contrib_H_step_mu0)]
    contrib_cluster_color_step_mu0       = [sum(el) for el in zip(contrib_D_step_mu0, contrib_F_step_mu0, contrib_Q5_step_mu0)]
    contrib_pi_step_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi_step_mu200)]
    contrib_rho_step_mu200               = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho_step_mu200)]
    contrib_omega_step_mu200             = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega_step_mu200)]
    contrib_D_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D_step_mu200)]
    contrib_N_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N_step_mu200)]
    contrib_T_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T_step_mu200)]
    contrib_F_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F_step_mu200)]
    contrib_P_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P_step_mu200)]
    contrib_Q5_step_mu200                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5_step_mu200)]
    contrib_H_step_mu200                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H_step_mu200)]
    contrib_cluster_step_mu200           = [sum(el) for el in zip(contrib_pi_step_mu200, contrib_rho_step_mu200, contrib_omega_step_mu200, contrib_D_step_mu200, contrib_N_step_mu200, contrib_T_step_mu200, contrib_F_step_mu200, contrib_P_step_mu200, contrib_Q5_step_mu200, contrib_H_step_mu200)]
    contrib_cluster_singlet_step_mu200   = [sum(el) for el in zip(contrib_pi_step_mu200, contrib_rho_step_mu200, contrib_omega_step_mu200, contrib_N_step_mu200, contrib_T_step_mu200, contrib_P_step_mu200, contrib_H_step_mu200)]
    contrib_cluster_color_step_mu200     = [sum(el) for el in zip(contrib_D_step_mu200, contrib_F_step_mu200, contrib_Q5_step_mu200)]
    contrib_pi_step_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_pi_step_mu300)]
    contrib_rho_step_mu300               = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_rho_step_mu300)]
    contrib_omega_step_mu300             = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_omega_step_mu300)]
    contrib_D_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_D_step_mu300)]
    contrib_N_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_N_step_mu300)]
    contrib_T_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_T_step_mu300)]
    contrib_F_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_F_step_mu300)]
    contrib_P_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_P_step_mu300)]
    contrib_Q5_step_mu300                = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_Q5_step_mu300)]
    contrib_H_step_mu300                 = [p_el / (T_el ** 4) for T_el, p_el in zip(T, Pres_H_step_mu300)]
    contrib_cluster_step_mu300           = [sum(el) for el in zip(contrib_pi_step_mu300, contrib_rho_step_mu300, contrib_omega_step_mu300, contrib_D_step_mu300, contrib_N_step_mu300, contrib_T_step_mu300, contrib_F_step_mu300, contrib_P_step_mu300, contrib_Q5_step_mu300, contrib_H_step_mu300)]
    contrib_cluster_singlet_step_mu300   = [sum(el) for el in zip(contrib_pi_step_mu300, contrib_rho_step_mu300, contrib_omega_step_mu300, contrib_N_step_mu300, contrib_T_step_mu300, contrib_P_step_mu300, contrib_H_step_mu300)]
    contrib_cluster_color_step_mu300     = [sum(el) for el in zip(contrib_D_step_mu300, contrib_F_step_mu300, contrib_Q5_step_mu300)]

    (low_1204_6710v2_mu0_x, low_1204_6710v2_mu0_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")
    (low_1204_6710v2_mu200_x, low_1204_6710v2_mu200_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")
    (low_1204_6710v2_mu300_x, low_1204_6710v2_mu300_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_low.dat")
    (high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y) = data_collect(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu300_high.dat")

    borsanyi_1204_6710v2_mu0 = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu0_x, high_1204_6710v2_mu0_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu0_x[::-1], low_1204_6710v2_mu0_y[::-1]):
        borsanyi_1204_6710v2_mu0.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu0 = np.array(borsanyi_1204_6710v2_mu0)
    borsanyi_1204_6710v2_mu200 = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu200_x, high_1204_6710v2_mu200_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu200_x[::-1], low_1204_6710v2_mu200_y[::-1]):
        borsanyi_1204_6710v2_mu200.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu200 = np.array(borsanyi_1204_6710v2_mu200)
    borsanyi_1204_6710v2_mu300 = [np.array([x_el, y_el]) for x_el, y_el in zip(high_1204_6710v2_mu300_x, high_1204_6710v2_mu300_y)]
    for x_el, y_el in zip(low_1204_6710v2_mu300_x[::-1], low_1204_6710v2_mu300_y[::-1]):
        borsanyi_1204_6710v2_mu300.append(np.array([x_el, y_el]))
    borsanyi_1204_6710v2_mu300 = np.array(borsanyi_1204_6710v2_mu300)

    fig1 = plt.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([0., 450., 0., 4.0])
    ax1.add_patch(Polygon(borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'green', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=0}$'))
    ax1.add_patch(Polygon(borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'red', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=200}$ MeV'))
    ax1.add_patch(Polygon(borsanyi_1204_6710v2_mu300, closed = True, fill = True, color = 'yellow', alpha = 0.5, label = r'Borsanyi et al. (2012), $\mathrm{\mu=300}$ MeV'))
    #ax1.plot(T, contrib_q_mu0, '-', c = 'blue', label = r'$\mathrm{P_{Q,0}}$')
    #ax1.plot(T, contrib_g_mu0, '-', c = 'red', label = r'$\mathrm{P_{g,0}}$')
    #ax1.plot(T, contrib_pert_mu0, '-', c = 'pink', label = r'$\mathrm{P_{pert,0}}$')
    ax1.plot(T, contrib_qgp_mu0, '-', c = 'black', label = r'$\mathrm{P_{QGP,0}}$')
    #ax1.plot(T, contrib_cluster_mu0, '-', c = 'blue', label = r'$\mathrm{P_{cluster,0}}$')
    ax1.plot(T, contrib_cluster_singlet_mu0, '-', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,0}}$')
    ax1.plot(T, contrib_cluster_color_mu0, '-', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,0}}$')
    ax1.plot(T, contrib_cluster_singlet_step_mu0, '-.', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,0}}$')
    ax1.plot(T, contrib_cluster_color_step_mu0, '-.', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,0}}$')
    #ax1.plot(T, contrib_q_mu200, '--', c = 'blue', label = r'$\mathrm{P_{Q,200}}$')
    #ax1.plot(T, contrib_g_mu200, '--', c = 'red', label = r'$\mathrm{P_{g,200}}$')
    #ax1.plot(T, contrib_pert_mu200, '--', c = 'pink', label = r'$\mathrm{P_{pert,200}}$')
    #ax1.plot(T, contrib_qgp_mu200, '--', c = 'black', label = r'$\mathrm{P_{QGP,200}}$')
    #ax1.plot(T, contrib_cluster_mu200, '--', c = 'blue', label = r'$\mathrm{P_{cluster,200}}$')
    #ax1.plot(T, contrib_cluster_singlet_mu200, '--', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,200}}$')
    #ax1.plot(T, contrib_cluster_color_mu200, '--', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,200}}$')
    #ax1.plot(T, contrib_q_mu300, '-.', c = 'blue', label = r'$\mathrm{P_{Q,300}}$')
    #ax1.plot(T, contrib_g_mu300, '-.', c = 'red', label = r'$\mathrm{P_{g,300}}$')
    #ax1.plot(T, contrib_pert_mu300, '-.', c = 'pink', label = r'$\mathrm{P_{pert,300}}$')
    #ax1.plot(T, contrib_qgp_mu300, '-.', c = 'black', label = r'$\mathrm{P_{QGP,300}}$')
    #ax1.plot(T, contrib_cluster_mu300, '-.', c = 'blue', label = r'$\mathrm{P_{cluster,300}}$')
    #ax1.plot(T, contrib_cluster_singlet_mu300, '-.', c = 'green', label = r'$\mathrm{P^{(1)}_{cluster,300}}$')
    #ax1.plot(T, contrib_cluster_color_mu300, '-.', c = 'red', label = r'$\mathrm{P^{(3/\bar{3})}_{cluster,300}}$')
    #ax1.legend(loc = 2)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig2 = plt.figure(num = 2, figsize = (5.9, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.axis([0., 450., 0., 1.2])
    ax2.plot(T, [M(el, 0) / M(0, 0) for el in T], '-', c = 'green')
    ax2.plot(T, [M(el, 200.0 / 3.0) / M(0, 0) for el in T], '--', c = 'green')
    ax2.plot(T, [M(el, 300.0 / 3.0) / M(0, 0) for el in T], '-.', c = 'green')
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

    plt.show()
    plt.close()

if __name__ == '__main__':

    PNJL_thermodynamics_continuum()

    #print("END")