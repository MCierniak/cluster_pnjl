import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.patches import Polygon, FancyArrowPatch

import pnjl.defaults
import utils

def figure2():

    (T, phi_re_0)       = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(0.0)) + "p0_cluster_dense.dat")
    (T, phi_re_100)     = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(100.0)) + "p0_cluster_dense.dat")
    (T, phi_re_200)     = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(200.0)) + "p0_cluster_dense.dat")
    (T, phi_re_300)     = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(300.0)) + "p0_cluster_dense.dat")
    (T, phi_re_alt_0)   = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(0.0)) + "p0_dev_dense.dat")
    (T, phi_re_alt_100) = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(100.0)) + "p0_dev_dense.dat")
    (T, phi_re_alt_200) = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(200.0)) + "p0_dev_dense.dat")
    (T, phi_re_alt_300) = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(300.0)) + "p0_dev_dense.dat")

    sigma_0 = [(400. * pnjl.aux_functions.Delta_ls(el, 0.0) + 5.5) / 405.5 for el in T]
    sigma_100 = [(400. * pnjl.aux_functions.Delta_ls(el, 100.0) + 5.5) / 405.5 for el in T]
    sigma_200 = [(400. * pnjl.aux_functions.Delta_ls(el, 200.0) + 5.5) / 405.5 for el in T]
    sigma_300 = [(400. * pnjl.aux_functions.Delta_ls(el, 300.0) + 5.5) / 405.5 for el in T]

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

def figure3():

    #druga wersja napisu "Borsanyi et al. (2012)", w poziomie ze strzalka

    col_qgp = 'blue'
    col_pert = 'red'
    col_gluon = 'pink'
    col_pnjl = 'magenta'

    (T, phi_re_0)               = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(0.0)) + "p0_cluster_dense.dat")
    Pres_Q, BDen_Q              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(0.0)) + "p0.dat")
    SDen_Q, _                   = utils.data_load(4, 4, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(0.0)) + "p0.dat")
    Pres_g, SDen_g              = utils.data_load(2, 4, "D:/EoS/BDK/thermo_mu_const/pl_mu_" + str(int(0.0)) + "p0.dat")
    Pres_pert, BDen_pert        = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(0.0)) + "p0.dat")
    SDen_pert, _                = utils.data_load(4, 4, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(0.0)) + "p0.dat")

    (low_1204_6710v2_x, low_1204_6710v2_y)      = utils.data_load(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_x, high_1204_6710v2_y)    = utils.data_load(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")

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

def figure4():

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

    (T, phi_re_0) = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(const_mu)) + "p0_cluster_dense.dat")

    fig = plt.figure(num = 1, figsize = (5.9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([min(T), max(T), 0.0, 1.05 * max([6.0 * math.sqrt(2) * pnjl.aux_functions.M(el, const_mu) for el in T])])

    ax.plot(T, [pnjl.defaults.default_MN for el in T]                            , '-'   , c = col_n     , label = r'nucleon'            )
    #ax.fill_between(T[T < 150.], [pnjl.defaults.default_MN for el in T if el < 150.], [3.0 * math.sqrt(2) * pnjl.aux_functions.M(el, const_mu) for el in T if el < 150.], color = col_n, alpha = 0.5)
    ax.add_patch(FancyArrowPatch((30., 1750.), (30., 1000.), arrowstyle='<->', mutation_scale=20, color = col_n))
    ax.plot(T, [3.0 * math.sqrt(2) * pnjl.aux_functions.M(el, const_mu) for el in T]  , '--'  , c = col_n)
    ax.text(35., 1320., r'nucleon', fontsize = 14)

    ax.plot(T, [140 for el in T]                                   , '-'   , c = col_pi    , label = r'$\mathrm{\pi}$'               )
    #ax.fill_between(T[T < 180.], [140 for el in T if el < 180.], [2.0 * math.sqrt(2) * pnjl.aux_functions.M(el, const_mu) for el in T if el < 180.], color = col_pi, alpha = 0.5)
    ax.add_patch(FancyArrowPatch((50., 1150.), (50., 135.), arrowstyle='<->', mutation_scale=20, color = col_pi))
    ax.plot(T, [2.0 * math.sqrt(2) * pnjl.aux_functions.M(el, const_mu) for el in T]  , '--'  , c = col_pi)
    ax.text(55., 620., r'pion', fontsize = 14)

    ax.plot(T, [pnjl.defaults.default_MH for el in T]                            , '-'   , c = col_h     , label = r'hexaquark'          )
    #ax.fill_between(T[T < 151.], [pnjl.defaults.default_MH for el in T if el < 151.], [6.0 * math.sqrt(2) * pnjl.aux_functions.M(el, const_mu) for el in T if el < 151.], color = col_h, alpha = 0.5)
    ax.add_patch(FancyArrowPatch((10., 3450.), (10., 1900.), arrowstyle='<->', mutation_scale=20, color = col_h))
    ax.plot(T, [6.0 * math.sqrt(2) * pnjl.aux_functions.M(el, const_mu) for el in T]  , '--'  , c = col_h)
    ax.text(15., 2650., r'hexaquark', fontsize = 14)

    ax.plot(T, [pnjl.aux_functions.M(el, const_mu) for el in T], '-', c = col_qgp, label = r'u/d quark')
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

def figure5():

    const_mu = 0.0
    (T, phi_re) = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(const_mu)) + "p0_cluster_dense.dat")

    Pres_Q, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_g, _              = utils.data_load(2, 4, "D:/EoS/BDK/thermo_mu_const/pl_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_pert, _           = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_pi, _             = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pi_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_rho, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/rho_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_omega, _          = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/omega_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_D, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_N, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_T, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/T_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_F, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_P, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_Q5, _             = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_H, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0.dat")

    Pres_Q0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_g0, _            = utils.data_load(2, 4, "D:/EoS/BDK/thermo_mu_const/pl_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_pert0, _         = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_pi0, _           = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pi_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_rho0, _          = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/rho_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_omega0, _        = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/omega_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_D0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_N0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_T0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/T_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_F0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_P0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_Q50, _           = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_H0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0_no_phi.dat")

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

    (borsanyi_upper_x, borsanyi_upper_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/borsanyi_upper.dat")
    (borsanyi_lower_x, borsanyi_lower_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/borsanyi_lower.dat")
    (bazavov_upper_x, bazavov_upper_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/bazavov_upper.dat")
    (bazavov_lower_x, bazavov_lower_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/bazavov_lower.dat")
    (low_1204_6710v2_x, low_1204_6710v2_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_x, high_1204_6710v2_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")

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

def figure6():

    #przerobic ten plot zeby porownac zmiany mu=0..200 dla lattice i dla mojego modelu

    const_mu = 200.0
    (T, phi_re) = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(const_mu)) + "p0_cluster_dense.dat")

    Pres_Q, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_g, _              = utils.data_load(2, 4, "D:/EoS/BDK/thermo_mu_const/pl_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_pert, _           = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_pi, _             = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pi_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_rho, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/rho_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_omega, _          = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/omega_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_D, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_N, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_T, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/T_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_F, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_P, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_Q5, _             = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0.dat")
    Pres_H, _              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0.dat")

    Pres_Q0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_g0, _            = utils.data_load(2, 4, "D:/EoS/BDK/thermo_mu_const/pl_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_pert0, _         = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_pi0, _           = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pi_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_rho0, _          = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/rho_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_omega0, _        = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/omega_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_D0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_N0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_T0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/T_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_F0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_P0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_Q50, _           = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    Pres_H0, _            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0_no_phi.dat")

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

    (borsanyi_upper_x, borsanyi_upper_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/borsanyi_upper.dat")
    (borsanyi_lower_x, borsanyi_lower_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/borsanyi_lower.dat")
    (bazavov_upper_x, bazavov_upper_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/bazavov_upper.dat")
    (bazavov_lower_x, bazavov_lower_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/bazavov_lower.dat")
    (low_1204_6710v2_x, low_1204_6710v2_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_x, high_1204_6710v2_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")

    (low_1204_6710v2_x_alt, low_1204_6710v2_y_alt) = utils.data_load(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_low.dat")
    (high_1204_6710v2_x_alt, high_1204_6710v2_y_alt) = utils.data_load(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu0_high.dat")

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

def figure7():

    const_mu = 100.0
    (T, phi_re) = utils.data_load(0, 1, "D:/EoS/BDK/gap_sol_mu_const/gap_mu_" + str(int(const_mu)) + "p0_cluster_dense.dat")

    _ , BDen_Q              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_pert           = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_D              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_N              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_F              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_P              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_Q5             = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0.dat")
    _ , BDen_H              = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0.dat")

    _ , BDen_Q0            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pnjl_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_pert0         = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/pert_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_D0            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/D_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_N0            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/N_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_F0            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/F_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_P0            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/P_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_Q50           = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/Q5_mu_" + str(int(const_mu)) + "p0_no_phi.dat")
    _ , BDen_H0            = utils.data_load(2, 3, "D:/EoS/BDK/thermo_mu_const/H_mu_" + str(int(const_mu)) + "p0_no_phi.dat")

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

    (borsanyi_upper_x, borsanyi_upper_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/borsanyi_upper.dat")
    (borsanyi_lower_x, borsanyi_lower_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/borsanyi_lower.dat")
    (bazavov_upper_x, bazavov_upper_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/bazavov_upper.dat")
    (bazavov_lower_x, bazavov_lower_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/bazavov_lower.dat")
    (low_1204_6710v2_x, low_1204_6710v2_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_low.dat")
    (high_1204_6710v2_x, high_1204_6710v2_y) = utils.data_load(0, 1, "D:/EoS/archive/BDK/lattice_data/const_mu/1204_6710v2_table4_pressure_mu200_high.dat")

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
