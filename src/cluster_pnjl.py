

def epja_figure1():

    import numpy
    import platform

    import matplotlib.pyplot
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    import pnjl.defaults

    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_susd
    import pnjl.thermo.gcp_cluster.breit_wigner \
        as cluster_bw

    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    hadron = 'Delta'

    T_Min = 155.0
    T_Max = 175.0

    T_list = numpy.linspace(T_Min, T_Max, num=17)
    M_list = numpy.linspace(0.5, 3.5, num=1000)

    muB_T = 0.0

    def phase(M, T, muB, hadron):
        delta_i = cluster_bw.bound_factor2(M, T, muB, hadron) \
            + cluster_bw.continuum_factor1(M, T, muB, hadron) \
            + cluster_bw.continuum_factor2(M, T, muB, hadron)
        return delta_i
        
    def phase2(M, T, muB, hadron):
        delta_i = cluster_bw.bound_factor2(M, T, muB, hadron)
        return delta_i

    phase_list = [
        [phase(M_el*1000.0, T_el, muB_T*T_el, hadron) for M_el in M_list]
        for T_el in T_list
    ]
    phase2_list = [
        [phase2(M_el*1000.0, T_el, muB_T*T_el, hadron) for M_el in M_list]
        for T_el in T_list
    ]

    M_I = cluster_bw.MI[hadron]
    nLambda = cluster_bw.NI[hadron]*cluster_bw.LAMBDA

    Mthi_vec = [
        cluster_bw.M_th(T_el, muB_T*T_el, hadron)/1000.0 for T_el in T_list
    ]

    Mthi0_vec = [
        (cluster_bw.M_th(0.0, 0.0, hadron)+nLambda)/1000.0 for _ in T_list
    ]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12, 5))
    ax1 = fig1.add_subplot(122, projection='3d')
    ax2 = fig1.add_subplot(121, projection='3d')
    fig1.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)

    ax1.set_ylim3d(max(T_list), min(T_list))
    ax1.set_xlim3d(min(M_list), max(M_list))
    ax1.set_zlim3d(0, 1)

    ax1.plot3D(
        M_list, [T_list[0] for el in M_list], phase_list[0], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[1] for el in M_list], phase_list[1], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[2] for el in M_list], phase_list[2], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[3] for el in M_list], phase_list[3], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[4] for el in M_list], phase_list[4], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[5] for el in M_list], phase_list[5], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[6] for el in M_list], phase_list[6], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[7] for el in M_list], phase_list[7], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[8] for el in M_list], phase_list[8], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[9] for el in M_list], phase_list[9], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[10] for el in M_list], phase_list[10], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[11] for el in M_list], phase_list[11], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[12] for el in M_list], phase_list[12], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[13] for el in M_list], phase_list[13], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[14] for el in M_list], phase_list[14], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[15] for el in M_list], phase_list[15], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[16] for el in M_list], phase_list[16], '-',
        c='black'
    )
    
    # ax1.plot3D(
    #     [
    #         (cluster.MSC_SLOPE*(el-cluster.T_Mott(muB_T*el, hadron)) + cluster.MI[hadron])/1000.0 if el > cluster.T_Mott(muB_T*el, hadron)-1.0 else float('NaN') for el in T_list
    #     ], 
    #     T_list, [0.0 for el in T_list], '--', c='green'
    # )
    ax1.plot3D(Mthi0_vec, T_list, [0.0 for el in T_list], '--', c='red')
    ax1.plot3D(Mthi_vec, T_list, [0.0 for el in T_list], '--', c='green')
    ax1.plot3D(
        [cluster_bw.M_i(T_el, 0.0, hadron)/1000.0 for T_el in T_list], T_list, [0.0 for el in T_list], '--',
        c='blue'
    )
    ax1.text(
        0.36, 155, 0.0, r'$\mathrm{M_{thr,i}}$', 'y', color='green',
        fontsize=16, bbox=dict(color='white', boxstyle='square, pad=0')
    )
    ax1.text(
        1.2, 155, 0.0, r'$\mathrm{M_i}$', 'y', color='blue',
        fontsize=16, bbox=dict(color='white', boxstyle='square, pad=-0.1')
    )
    ax1.text(
        2.1, 159, 0.0, r'$\mathrm{M_{thr,i,0}+N_i\Lambda}$', 'y',
        color='red', fontsize=16,
        bbox=dict(color='white', boxstyle='square, pad=0.0')
    )
    
    ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.set_xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
    ax1.set_xticklabels([0.5, '', 1, '', 1.5, '', 2, '', 2.5])
    # ax1.set_yticks([135, 140, 145, 150, 155, 160])
    ax1.set_yticks([150, 155, 160])
    # ax1.set_yticklabels([135, 140, 145, 150, 155, ''])
    
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

    ax2.set_ylim3d(max(T_list), min(T_list))
    ax2.set_xlim3d(min(M_list), max(M_list))
    ax2.set_zlim3d(0, 1)

    ax2.plot3D(
        M_list, [T_list[0] for el in M_list], phase2_list[0], '-',
        c = 'black'
    )
    ax2.plot3D(
        M_list, [T_list[2] for el in M_list], phase2_list[2], '-',
        c = 'black'
    )
    ax2.plot3D(
        M_list, [T_list[4] for el in M_list], phase2_list[4], '-',
        c = 'black'
    )
    ax2.plot3D(
        M_list, [T_list[6] for el in M_list], phase2_list[6], '-',
        c = 'black'
    )
    ax2.plot3D(
        M_list, [T_list[8] for el in M_list], phase2_list[8], '-',
        c = 'black'
    )
    ax2.plot3D(
        M_list, [T_list[10] for el in M_list], phase2_list[10], '-',
        c = 'black'
    )
    ax2.plot3D(
        M_list, [T_list[12] for el in M_list], phase2_list[12], '-',
        c = 'black'
    )
    ax2.plot3D(
        M_list, [T_list[14] for el in M_list], phase2_list[14], '-',
        c = 'black'
    )
    ax2.plot3D(
        M_list, [T_list[16] for el in M_list], phase2_list[16], '-',
        c='black'
    )
    
    ax2.plot3D(Mthi0_vec, T_list, [0.0 for el in T_list], '--', c='red')
    ax2.plot3D(Mthi_vec, T_list, [0.0 for el in T_list], '--', c='green')
    ax2.plot3D(
        [M_I/1000.0 for _ in T_list], T_list, [0.0 for el in T_list], '--',
        c='blue'
    )
    ax2.text(
        0.36, 155, 0.0, r'$\mathrm{M_{thr,i}}$', 'y', color='green',
        fontsize=16, bbox=dict(color='white', boxstyle='square, pad=0')
    )
    ax2.text(
        1.2, 155, 0.0, r'$\mathrm{M_i}$', 'y', color='blue',
        fontsize=16, bbox=dict(color='white', boxstyle='square, pad=-0.1')
    )
    ax2.text(
        2.1, 159, 0.0, r'$\mathrm{M_{thr,i,0}+N_i\Lambda}$', 'y',
        color='red', fontsize=16,
        bbox=dict(color='white', boxstyle='square, pad=0.0')
    )
    
    ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax2.set_xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
    ax2.set_xticklabels([0.5, '', 1, '', 1.5, '', 2, '', 2.5])
    ax2.set_yticks([135, 140, 145, 150, 155, 160])
    ax2.set_yticklabels([135, 140, 145, 150, 155, ''])
    
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax2.zaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    ax2.tick_params(axis='x', which='major', pad=-3)
    ax2.tick_params(axis='y', which='major', pad=-2)
    ax2.set_xlabel(r'M [GeV]', fontsize = 16)
    ax2.set_ylabel(r'T [MeV]', fontsize = 16)
    ax2.set_zlabel(r'$\delta_i$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure2():

    import numpy
    import platform

    import matplotlib.pyplot
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")
    import matplotlib.patches

    import pnjl.defaults
    import pnjl.thermo.gcp_pnjl.lattice_cut_sea

    import pnjl.thermo.gcp_cluster.breit_wigner \
        as cluster

    col_n = '#DEA54B'
    col_pi = '#653239'
    col_h = '#A846A0'
    col_qgp = 'blue'

    T = numpy.linspace(1.0, 200.0, num = 200)

    fig = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis(
        [
            min(T), max(T),
            0.0, 1.05 * max([cluster.M_th(el, 0.0, 'd') for el in T])
        ]
    )

    ax.plot(
        T, [cluster.MI['p'] for el in T], '-', c=col_n,
        label = r'nucleon'
    )
    ax.add_patch(
        matplotlib.patches.FancyArrowPatch(
            (30., 1700.), (30., 940.), arrowstyle='<->',
            mutation_scale=20, color=col_n
        )
    )
    ax.plot(T, [cluster.M_th(el, 0.0, 'p') for el in T], '--', c=col_n)
    ax.text(35., 1320., r'nucleon', fontsize = 14)

    ax.plot(
        T, [cluster.MI['pi0'] for el in T], '-', c=col_pi,
        label=r'$\mathrm{\pi}$'
    )
    ax.add_patch(
        matplotlib.patches.FancyArrowPatch(
            (50., 1125.), (50., 125.), arrowstyle='<->',
            mutation_scale=20, color=col_pi
        )
    )
    ax.plot(T, [cluster.M_th(el, 0.0, 'pi0') for el in T], '--', c=col_pi)
    ax.text(55., 620., r'pion', fontsize = 14)

    ax.plot(
        T, [cluster.MI['d'] for el in T], '-', c=col_h,
        label = r'hexaquark'
    )
    ax.add_patch(
        matplotlib.patches.FancyArrowPatch(
            (10., 3375.), (10., 1875.), arrowstyle='<->',
            mutation_scale=20, color=col_h
        )
    )
    ax.plot(T, [cluster.M_th(el, 0.0, 'd') for el in T], '--', c=col_h)
    ax.text(15., 2650., r'hexaquark', fontsize = 14)

    ax.plot(
        T, [pnjl.thermo.gcp_pnjl.lattice_cut_sea.Ml(el, 0.0) for el in T],
        '-', c = col_qgp, label = r'u/d quark'
    )
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


def epja_figure3():

    import tqdm
    import numpy
    import pickle
    import platform
    import warnings

    import matplotlib.pyplot

    import pnjl.thermo.gcp_pl.sasaki
    import pnjl.thermo.gcp_pnjl.lattice_cut_sea

    import pnjl.thermo.solvers.\
        pnjl_lattice_cut_sea.\
        pl_sasaki.\
        pert_const.\
        no_clusters \
    as solver

    warnings.filterwarnings("ignore")
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    files = "D:/EoS/epja/figure3/"
    lattice = "D:/EoS/epja/lattice_data_pickled/"
    if platform.system() == "Linux":
        lattice = "/home/mcierniak/Data/2023_epja/lattice_data_pickled/"
        files = "/home/mcierniak/Data/2023_epja/figure3/"

    calc_1 = False

    T = numpy.linspace(1.0, 250.0, num = 200)

    sigma = [pnjl.thermo.gcp_pnjl.lattice_cut_sea.Delta_ls(el, 0.0) for el in T]
    mg = [pnjl.thermo.gcp_pl.sasaki.Mg(el, 0.0) for el in T]

    phi_re_v_1 = list()
    phi_im_v_1 = list()

    if calc_1:
        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("QGP PL")
        for T_el in tqdm.tqdm(T, total=len(T), ncols=100):
            phi_re_0, phi_im_0 = solver.Polyakov_loop(T_el, 0.0, phi_re_0, phi_im_0)
            phi_re_v_1.append(phi_re_0)
            phi_im_v_1.append(phi_im_0)
        with open(files+"phi_re_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)
    else:
        with open(files+"phi_re_v_0p0.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_0p0.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
    
    with open(lattice + "1005_3508_table3_delta.pickle", "rb") as file:
        borsanyi_1005_3508 = pickle.load(file)

    fig = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([100, 220, 0.0, 1.1])

    ax.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1005_3508, closed = True, fill = True, color = 'blue',
            alpha = 0.3
        )
    )
    
    ax.plot(T, sigma, c = 'green')
    ax.plot(T, mg, c = 'magenta')
    ax.plot(T, phi_re_v_1, c = 'blue')
    ax.plot(T, phi_im_v_1, c = 'red')
    ax.text(200, 1, r'$\mathrm{\mu_B=0}$', fontsize = 14)
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


def epja_experimental_annals_phys_comparison():

    import math
    import numpy
    import platform

    import matplotlib.pyplot

    import pnjl.thermo.distributions
    import pnjl.thermo.gcp_cluster.bound_step_continuum_quad \
        as cluster

    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    hadron = 'p'

    def phase(M, T, mu):
        MI_N = cluster.MI[hadron]
        Mth_N = cluster.M_th(T, mu, hadron)
        if Mth_N > MI_N:
            if M < MI_N:
                return 0.0
            if M >= MI_N and M < Mth_N:
                return 1.0
            else:
                return cluster.continuum_factor1(M, T, mu, hadron)
        else:
            TM = cluster.T_Mott(mu, hadron)
            TM2 = cluster.T_Mott2(mu, hadron)
            return cluster.continuum_factor2(M, T, mu, TM, TM2, hadron)

    T_1 = 10.0
    T_2 = 100.0
    T_3 = 125.0
    T_4 = 150.0

    # muB_1 = 0.0
    # muB_2 = 0.0
    # muB_3 = 0.0
    # muB_4 = 0.0

    muB_1 = 2.5*T_1
    muB_2 = 2.5*T_2
    muB_3 = 2.5*T_3
    muB_4 = 2.5*T_4

    TM_1 = cluster.T_Mott(muB_1, hadron)
    TM_2 = cluster.T_Mott(muB_2, hadron)
    TM_3 = cluster.T_Mott(muB_3, hadron)
    TM_4 = cluster.T_Mott(muB_4, hadron)
    
    TM2_1 = cluster.T_Mott2(muB_1, hadron)
    TM2_2 = cluster.T_Mott2(muB_2, hadron)
    TM2_3 = cluster.T_Mott2(muB_3, hadron)
    TM2_4 = cluster.T_Mott2(muB_4, hadron)

    s_v = numpy.linspace(0.004, 13.0, num=10000)

    delta_v_1 = [
            cluster.bound_factor2(math.sqrt(s_el)*1000.0, T_1, muB_1, hadron) \
        +   cluster.continuum_factor1(math.sqrt(s_el)*1000.0, T_1, muB_1, hadron) \
        +   cluster.continuum_factor2(math.sqrt(s_el)*1000.0, T_1, muB_1, TM_1, TM2_1, hadron)
        for s_el in s_v
    ]
    delta_v_2 = [
            cluster.bound_factor2(math.sqrt(s_el)*1000.0, T_2, muB_2, hadron) \
        +   cluster.continuum_factor1(math.sqrt(s_el)*1000.0, T_2, muB_2, hadron) \
        +   cluster.continuum_factor2(math.sqrt(s_el)*1000.0, T_2, muB_2, TM_2, TM2_2, hadron)
        for s_el in s_v
    ]
    delta_v_3 = [
            cluster.bound_factor2(math.sqrt(s_el)*1000.0, T_3, muB_3, hadron) \
        +   cluster.continuum_factor1(math.sqrt(s_el)*1000.0, T_3, muB_3, hadron) \
        +   cluster.continuum_factor2(math.sqrt(s_el)*1000.0, T_3, muB_3, TM_3, TM2_3, hadron)
        for s_el in s_v
    ]
    delta_v_4 = [
            cluster.bound_factor2(math.sqrt(s_el)*1000.0, T_4, muB_4, hadron) \
        +   cluster.continuum_factor1(math.sqrt(s_el)*1000.0, T_4, muB_4, hadron) \
        +   cluster.continuum_factor2(math.sqrt(s_el)*1000.0, T_4, muB_4, TM_4, TM2_4, hadron)
        for s_el in s_v
    ]

    f_v_1 = [
        pnjl.thermo.distributions.f_fermion_singlet(1000.0, T_1, muB_1, math.sqrt(s_el)*1000.0, 1, '+')*d_el
        if d_el > 0.0 else 0.0
        for s_el, d_el in zip(s_v, delta_v_1)
    ]
    f_v_1_norm = [el/max(f_v_1) for el in f_v_1]
    f_v_2 = [
        pnjl.thermo.distributions.f_fermion_singlet(1000.0, T_2, muB_2, math.sqrt(s_el)*1000.0, 1, '+')*d_el
        if d_el > 0.0 else 0.0
        for s_el, d_el in zip(s_v, delta_v_2)
    ]
    f_v_2_norm = [el/max(f_v_2) for el in f_v_2]
    f_v_3 = [
        pnjl.thermo.distributions.f_fermion_singlet(1000.0, T_3, muB_3, math.sqrt(s_el)*1000.0, 1, '+')*d_el
        if d_el > 0.0 else 0.0
        for s_el, d_el in zip(s_v, delta_v_3)
    ]
    f_v_3_norm = [el/max(f_v_3) for el in f_v_3]
    f_v_4 = [
        pnjl.thermo.distributions.f_fermion_singlet(1000.0, T_4, muB_4, math.sqrt(s_el)*1000.0, 1, '+')*d_el
        if d_el > 0.0 else 0.0
        for s_el, d_el in zip(s_v, delta_v_4)
    ]
    f_v_4_norm = [el/max(f_v_4) for el in f_v_4]

    T_v_1 = numpy.linspace(0.0, 0.3, num=200)
    T_v_2 = numpy.linspace(0.0, 0.3, num=200)
    T_v_3 = numpy.linspace(0.0, 0.3, num=200)
    T_v_4 = numpy.linspace(0.0, 0.3, num=200)

    muB_1 = 0.0
    muB_2 = 3.0*150.0
    muB_3 = 3.0*250.0
    muB_4 = 3.0*300.0

    TM_1 = cluster.T_Mott(muB_1, hadron)
    TM_2 = cluster.T_Mott(muB_2, hadron)
    TM_3 = cluster.T_Mott(muB_3, hadron)
    TM_4 = cluster.T_Mott(muB_4, hadron)

    Mi_v_1 = [
        (cluster.MSC_SLOPE*(T_el*1000.0 - TM_1) + cluster.MI[hadron])/1000.0
        if T_el*1000.0 > TM_1 else cluster.MI[hadron]/1000.0 for T_el in T_v_1
    ]
    Mi_v_2 = [
        (cluster.MSC_SLOPE*(T_el*1000.0 - TM_2) + cluster.MI[hadron])/1000.0
        if T_el*1000.0 > TM_2 else cluster.MI[hadron]/1000.0 for T_el in T_v_2
    ]
    Mi_v_3 = [
        (cluster.MSC_SLOPE*(T_el*1000.0 - TM_3) + cluster.MI[hadron])/1000.0
        if T_el*1000.0 > TM_3 else cluster.MI[hadron]/1000.0 for T_el in T_v_3
    ]
    Mi_v_4 = [
        (cluster.MSC_SLOPE*(T_el*1000.0 - TM_4) + cluster.MI[hadron])/1000.0
        if T_el*1000.0 > TM_4 else cluster.MI[hadron]/1000.0 for T_el in T_v_4
    ]

    Mth_v_1 = [
        cluster.M_th(T_el*1000.0, muB_1, hadron)/1000.0
        for T_el in T_v_1
    ]
    Mth_v_2 = [
        cluster.M_th(T_el*1000.0, muB_2, hadron)/1000.0
        for T_el in T_v_2
    ]
    Mth_v_3 = [
        cluster.M_th(T_el*1000.0, muB_3, hadron)/1000.0
        for T_el in T_v_3
    ]
    Mth_v_4 = [
        cluster.M_th(T_el*1000.0, muB_4, hadron)/1000.0
        for T_el in T_v_4
    ]

    muB_v_1 = numpy.linspace(0.0, 3.0*0.45, num=200)
    muB_v_2 = numpy.linspace(0.0, 3.0*0.45, num=200)
    muB_v_3 = numpy.linspace(0.0, 3.0*0.45, num=200)
    muB_v_4 = numpy.linspace(0.0, 3.0*0.45, num=200)

    T_1 = 0.0
    T_2 = 50.0
    T_3 = 75.0
    T_4 = 100.0

    Mi_v2_1 = [
        (cluster.MSC_SLOPE*(T_1 - cluster.T_Mott(muB_el*1000.0, hadron)) + cluster.MI[hadron])/1000.0
        if T_1 > cluster.T_Mott(muB_el*1000.0, hadron) else cluster.MI[hadron]/1000.0 for muB_el in muB_v_1
    ]
    Mi_v2_2 = [
        (cluster.MSC_SLOPE*(T_2 - cluster.T_Mott(muB_el*1000.0, hadron)) + cluster.MI[hadron])/1000.0
        if T_2 > cluster.T_Mott(muB_el*1000.0, hadron) else cluster.MI[hadron]/1000.0 for muB_el in muB_v_2
    ]
    Mi_v2_3 = [
        (cluster.MSC_SLOPE*(T_3 - cluster.T_Mott(muB_el*1000.0, hadron)) + cluster.MI[hadron])/1000.0
        if T_3 > cluster.T_Mott(muB_el*1000.0, hadron) else cluster.MI[hadron]/1000.0 for muB_el in muB_v_3
    ]
    Mi_v2_4 = [
        (cluster.MSC_SLOPE*(T_4 - cluster.T_Mott(muB_el*1000.0, hadron)) + cluster.MI[hadron])/1000.0
        if T_4 > cluster.T_Mott(muB_el*1000.0, hadron) else cluster.MI[hadron]/1000.0 for muB_el in muB_v_4
    ]

    Mth_v2_1 = [
        cluster.M_th(T_1, muB_el*1000.0, hadron)/1000.0
        for muB_el in muB_v_1
    ]
    Mth_v2_2 = [
        cluster.M_th(T_2, muB_el*1000.0, hadron)/1000.0
        for muB_el in muB_v_2
    ]
    Mth_v2_3 = [
        cluster.M_th(T_3, muB_el*1000.0, hadron)/1000.0
        for muB_el in muB_v_3
    ]
    Mth_v2_4 = [
        cluster.M_th(T_4, muB_el*1000.0, hadron)/1000.0
        for muB_el in muB_v_4
    ]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (18, 5))
    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.axis([0.0, 0.275, 0.0, 0.7])

    ax1.plot(T_v_1, Mi_v_1, '-', c="red")
    ax1.plot(T_v_1, Mth_v_1, ':', c="red")
    ax1.plot(T_v_2, Mi_v_2, '-', c="purple")
    ax1.plot(T_v_2, Mth_v_2, ':', c="purple")
    ax1.plot(T_v_3, Mi_v_3, '-', c="green")
    ax1.plot(T_v_3, Mth_v_3, ':', c="green")
    ax1.plot(T_v_4, Mi_v_4, '-', c="yellow")
    ax1.plot(T_v_4, Mth_v_4, ':', c="yellow")

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    ax1.set_xlabel(r'T [GeV]', fontsize = 16)
    ax1.set_ylabel(r'masses [GeV]', fontsize = 16)

    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.axis([0.0, 0.45, 0.0, 0.7])

    ax2.plot([el/3.0 for el in muB_v_1], Mi_v2_1, '-', c="red")
    ax2.plot([el/3.0 for el in muB_v_1], Mth_v2_1, ':', c="red")
    ax2.plot([el/3.0 for el in muB_v_2], Mi_v2_2, '-', c="purple")
    ax2.plot([el/3.0 for el in muB_v_2], Mth_v2_2, ':', c="purple")
    ax2.plot([el/3.0 for el in muB_v_3], Mi_v2_3, '-', c="green")
    ax2.plot([el/3.0 for el in muB_v_3], Mth_v2_3, ':', c="green")
    ax2.plot([el/3.0 for el in muB_v_4], Mi_v2_4, '-', c="yellow")
    ax2.plot([el/3.0 for el in muB_v_4], Mth_v2_4, ':', c="yellow")

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    ax2.set_xlabel(r'$\mathrm{\mu}$ [GeV]', fontsize = 16)
    ax2.set_ylabel(r'masses [GeV]', fontsize = 16)

    ax3 = fig1.add_subplot(1, 3, 3)
    ax3.axis([0.004, 13., -0.25, 1.5])
    ax3.set_xscale("log")

    ax3.plot(s_v, delta_v_1, '-', c="blue")
    ax3.plot(s_v, delta_v_2, ':', c="blue")
    ax3.plot(s_v, delta_v_3, '--', c="blue")
    ax3.plot(s_v, delta_v_4, '-.', c="blue")
    
    ax3.plot(s_v, f_v_1, '-', c="red")
    ax3.plot(s_v, f_v_2, ':', c="red")
    ax3.plot(s_v, f_v_3, '--', c="red")
    ax3.plot(s_v, f_v_4, '-.', c="red")

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    ax3.set_xlabel(r's [$\mathrm{GeV^2}$]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{\delta_\pi}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_experimental_pressure_int():

    import tqdm
    import numpy
    import pickle
    import warnings
    import platform

    import matplotlib.pyplot
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    import pnjl.thermo.gcp_pl.polynomial
    import pnjl.thermo.gcp_perturbative.const
    import pnjl.thermo.gcp_pnjl.lattice_cut_sea
    import pnjl.thermo.gcp_cluster.bound_step_continuum_quad \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_quad \
        as cluster

    import pnjl.thermo.solvers.\
        pnjl_lattice_cut_sea.\
        pl_polynomial.\
        pert_const.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")
    
    calc_1 = False
    calc_2 = False

    files = "D:/EoS/epja/experimental/pressure_int/"
    lattice = "D:/EoS/epja/lattice_data_pickled/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/experimental/pressure_int/"
        lattice = "/home/mcierniak/Data/2023_epja/lattice_data_pickled/"

    T_1 = numpy.linspace(1.0, 400.0, 200)
    T_2 = numpy.linspace(1.0, 400.0, 200)

    mu_1 = [0.0 / 3.0 for el in T_1]
    mu_2 = [200.0 / 3.0 for el in T_2]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()
    phi_re_v_2, phi_im_v_2 = \
        list(), list()

    sigma_v_1, gluon_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1 = \
        list(), list(), list(), list(), list()
    perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
        list(), list(), list()
    perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = \
        list(), list(), list(), list()

    sigma_v_2, gluon_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2 = \
        list(), list(), list(), list(), list()
    perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2 = \
        list(), list(), list()
    perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2 = \
        list(), list(), list(), list()

    pi_v_1, K_v_1, rho_v_1, omega_v_1, D_v_1, N_v_1 = \
        list(), list(), list(), list(), list(), list()
    T_v_1, F_v_1, P_v_1, Q_v_1, H_v_1 = \
        list(), list(), list(), list(), list()

    pi_v_1s, K_v_1s, rho_v_1s, omega_v_1s, D_v_1s, N_v_1s = \
        list(), list(), list(), list(), list(), list()
    T_v_1s, F_v_1s, P_v_1s, Q_v_1s, H_v_1s = \
        list(), list(), list(), list(), list()
    
    pi_v_2, K_v_2, rho_v_2, omega_v_2, D_v_2, N_v_2 = \
        list(), list(), list(), list(), list(), list()
    T_v_2, F_v_2, P_v_2, Q_v_2, H_v_2 = \
        list(), list(), list(), list(), list()

    pi_v_2s, K_v_2s, rho_v_2s, omega_v_2s, D_v_2s, N_v_2s = \
        list(), list(), list(), list(), list(), list()
    T_v_2s, F_v_2s, P_v_2s, Q_v_2s, H_v_2s = \
        list(), list(), list(), list(), list()

    if calc_1 and False:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_1, mu_1), total=len(T_1), ncols=100
        ):
            phi_result = solver_1.Polyakov_loop(
                T_el, mu_el, phi_re_0, phi_im_0
            )
            phi_re_v_1.append(phi_result[0])
            phi_im_v_1.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_1.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_1.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)

        print("Sigma pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            sigma_v_1.append(
                pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_field(
                    T_el, mu_el
                )/(T_el**4)
            )
        with open(files+"sigma_v_1.pickle", "wb") as file:
            pickle.dump(sigma_v_1, file)

        print("Gluon pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            gluon_v_1.append(
                pnjl.thermo.gcp_pl_polynomial.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el
                )/(T_el**4)
            )
        with open(files+"gluon_v_1.pickle", "wb") as file:
            pickle.dump(gluon_v_1, file)

        print("Sea pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_sea(
                T_el, mu_el, 'l'
            )
            sea_u_v_1.append(lq_temp/(T_el**4))
            sea_d_v_1.append(lq_temp/(T_el**4))
            sea_s_v_1.append(
                pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_sea(
                    T_el, mu_el, 's'
                )/(T_el**4)
            )
        with open(files+"sea_u_v_1.pickle", "wb") as file:
            pickle.dump(sea_u_v_1, file)
        with open(files+"sea_d_v_1.pickle", "wb") as file:
            pickle.dump(sea_d_v_1, file)
        with open(files+"sea_s_v_1.pickle", "wb") as file:
            pickle.dump(sea_s_v_1, file)

        print("Perturbative pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            perturbative_u_v_1.append(lq_temp/(T_el**4))
            perturbative_d_v_1.append(lq_temp/(T_el**4))
            perturbative_s_v_1.append(
                pnjl.thermo.gcp_perturbative.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 's'
                )/(T_el**4)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                perturbative_gluon_v_1.append(
                    -pnjl.thermo.gcp_perturbative.gcp_boson_real(
                        T_el, mu_el, phi_re_el, phi_im_el
                    )/(T_el**4)
                )
            else:
                perturbative_gluon_v_1.append(0.0)
        with open(files+"perturbative_u_v_1.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_1, file)
        with open(files+"perturbative_d_v_1.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_1, file)
        with open(files+"perturbative_s_v_1.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_1, file)
        with open(files+"perturbative_gluon_v_1.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_1, file)

        print("PNJL pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_q(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_q(
                T_el, mu_el, phi_re_el, phi_im_el, 's'
            )
            pnjl_u_v_1.append(lq_temp/(T_el**4))
            pnjl_d_v_1.append(lq_temp/(T_el**4))
            pnjl_s_v_1.append(sq_temp/(T_el**4))
        with open(files+"pnjl_u_v_1.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_1, file)
        with open(files+"pnjl_d_v_1.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_1, file)
        with open(files+"pnjl_s_v_1.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_1, file)
    else:
        with open(files+"phi_re_v_1.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_1.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
        with open(files+"sigma_v_1.pickle", "rb") as file:
            sigma_v_1 = pickle.load(file)
        with open(files+"gluon_v_1.pickle", "rb") as file:
            gluon_v_1 = pickle.load(file)
        with open(files+"sea_u_v_1.pickle", "rb") as file:
            sea_u_v_1 = pickle.load(file)
        with open(files+"sea_d_v_1.pickle", "rb") as file:
            sea_d_v_1 = pickle.load(file)
        with open(files+"sea_s_v_1.pickle", "rb") as file:
            sea_s_v_1 = pickle.load(file)
        with open(files+"perturbative_u_v_1.pickle", "rb") as file:
            perturbative_u_v_1 = pickle.load(file)
        with open(files+"perturbative_d_v_1.pickle", "rb") as file:
            perturbative_d_v_1 = pickle.load(file)
        with open(files+"perturbative_s_v_1.pickle", "rb") as file:
            perturbative_s_v_1 = pickle.load(file)
        with open(files+"perturbative_gluon_v_1.pickle", "rb") as file:
            perturbative_gluon_v_1 = pickle.load(file)
        with open(files+"pnjl_u_v_1.pickle", "rb") as file:
            pnjl_u_v_1 = pickle.load(file)
        with open(files+"pnjl_d_v_1.pickle", "rb") as file:
            pnjl_d_v_1 = pickle.load(file)
        with open(files+"pnjl_s_v_1.pickle", "rb") as file:
            pnjl_s_v_1 = pickle.load(file)

    if calc_2 and False:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_2, mu_2), total=len(T_2), ncols=100
        ):
            phi_result = solver_1.Polyakov_loop(
                T_el, mu_el, phi_re_0, phi_im_0
            )
            phi_re_v_2.append(phi_result[0])
            phi_im_v_2.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_2.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)

        print("Sigma pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            sigma_v_2.append(
                pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_field(
                    T_el, mu_el
                )/(T_el**4)
            )
        with open(files+"sigma_v_2.pickle", "wb") as file:
            pickle.dump(sigma_v_2, file)

        print("Gluon pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_1, phi_im_v_1),
            total=len(T_2), ncols=100
        ):
            gluon_v_2.append(
                pnjl.thermo.gcp_pl_polynomial.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el
                )/(T_el**4)
            )
        with open(files+"gluon_v_2.pickle", "wb") as file:
            pickle.dump(gluon_v_2, file)

        print("Sea pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_sea(
                T_el, mu_el, 'l'
            )
            sea_u_v_2.append(lq_temp/(T_el**4))
            sea_d_v_2.append(lq_temp/(T_el**4))
            sea_s_v_2.append(
                pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_sea(
                    T_el, mu_el, 's'
                )/(T_el**4)
            )
        with open(files+"sea_u_v_2.pickle", "wb") as file:
            pickle.dump(sea_u_v_2, file)
        with open(files+"sea_d_v_2.pickle", "wb") as file:
            pickle.dump(sea_d_v_2, file)
        with open(files+"sea_s_v_2.pickle", "wb") as file:
            pickle.dump(sea_s_v_2, file)

        print("Perturbative pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            perturbative_u_v_2.append(lq_temp/(T_el**4))
            perturbative_d_v_2.append(lq_temp/(T_el**4))
            perturbative_s_v_2.append(
                pnjl.thermo.gcp_perturbative.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 's'
                )/(T_el**4)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                perturbative_gluon_v_2.append(
                    -pnjl.thermo.gcp_perturbative.gcp_boson_real(
                        T_el, mu_el, phi_re_el, phi_im_el
                    )/(T_el**4)
                )
            else:
                perturbative_gluon_v_2.append(0.0)
        with open(files+"perturbative_u_v_2.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_2, file)
        with open(files+"perturbative_d_v_2.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_2, file)
        with open(files+"perturbative_s_v_2.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_2, file)
        with open(files+"perturbative_gluon_v_2.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_2, file)

        print("PNJL pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_q(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.pressure_q(
                T_el, mu_el, phi_re_el, phi_im_el, 's'
            )
            pnjl_u_v_2.append(lq_temp/(T_el**4))
            pnjl_d_v_2.append(lq_temp/(T_el**4))
            pnjl_s_v_2.append(sq_temp/(T_el**4))
        with open(files+"pnjl_u_v_2.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_2, file)
        with open(files+"pnjl_d_v_2.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_2, file)
        with open(files+"pnjl_s_v_2.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_2, file)
    else:
        with open(files+"phi_re_v_2.pickle", "rb") as file:
            phi_re_v_2 = pickle.load(file)
        with open(files+"phi_im_v_2.pickle", "rb") as file:
            phi_im_v_2 = pickle.load(file)
        with open(files+"sigma_v_2.pickle", "rb") as file:
            sigma_v_2 = pickle.load(file)
        with open(files+"gluon_v_2.pickle", "rb") as file:
            gluon_v_2 = pickle.load(file)
        with open(files+"sea_u_v_2.pickle", "rb") as file:
            sea_u_v_2 = pickle.load(file)
        with open(files+"sea_d_v_2.pickle", "rb") as file:
            sea_d_v_2 = pickle.load(file)
        with open(files+"sea_s_v_2.pickle", "rb") as file:
            sea_s_v_2 = pickle.load(file)
        with open(files+"perturbative_u_v_2.pickle", "rb") as file:
            perturbative_u_v_2 = pickle.load(file)
        with open(files+"perturbative_d_v_2.pickle", "rb") as file:
            perturbative_d_v_2 = pickle.load(file)
        with open(files+"perturbative_s_v_2.pickle", "rb") as file:
            perturbative_s_v_2 = pickle.load(file)
        with open(files+"perturbative_gluon_v_2.pickle", "rb") as file:
            perturbative_gluon_v_2 = pickle.load(file)
        with open(files+"pnjl_u_v_2.pickle", "rb") as file:
            pnjl_u_v_2 = pickle.load(file)
        with open(files+"pnjl_d_v_2.pickle", "rb") as file:
            pnjl_d_v_2 = pickle.load(file)
        with open(files+"pnjl_s_v_2.pickle", "rb") as file:
            pnjl_s_v_2 = pickle.load(file)

    if calc_1 and False:

        print("Pion pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**4)
            )
        with open(files+"pi_v_1.pickle", "wb") as file:
            pickle.dump(pi_v_1, file)

        print("Kaon pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**4)
            )
        with open(files+"K_v_1.pickle", "wb") as file:
            pickle.dump(K_v_1, file)

        print("Rho pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**4)
            )
        with open(files+"rho_v_1.pickle", "wb") as file:
            pickle.dump(rho_v_1, file)

        print("Omega pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**4)
            )
        with open(files+"omega_v_1.pickle", "wb") as file:
            pickle.dump(omega_v_1, file)

        print("Diquark pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**4)
            )
        with open(files+"D_v_1.pickle", "wb") as file:
            pickle.dump(D_v_1, file)

        print("Nucleon pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**4)
            )
        with open(files+"N_v_1.pickle", "wb") as file:
            pickle.dump(N_v_1, file)

        print("Tetraquark pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**4)
            )
        with open(files+"T_v_1.pickle", "wb") as file:
            pickle.dump(T_v_1, file)

        print("F-quark pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**4)
            )
        with open(files+"F_v_1.pickle", "wb") as file:
            pickle.dump(F_v_1, file)

        print("Pentaquark pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**4)
            )
        with open(files+"P_v_1.pickle", "wb") as file:
            pickle.dump(P_v_1, file)

        print("Q-quark pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**4)
            )
        with open(files+"Q_v_1.pickle", "wb") as file:
            pickle.dump(Q_v_1, file)

        print("Hexaquark pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**4)
            )
        with open(files+"H_v_1.pickle", "wb") as file:
            pickle.dump(H_v_1, file)
    else:
        with open(files+"pi_v_1.pickle", "rb") as file:
            pi_v_1 = pickle.load(file)
        with open(files+"K_v_1.pickle", "rb") as file:
            K_v_1 = pickle.load(file)
        with open(files+"rho_v_1.pickle", "rb") as file:
            rho_v_1 = pickle.load(file)
        with open(files+"omega_v_1.pickle", "rb") as file:
            omega_v_1 = pickle.load(file)
        with open(files+"D_v_1.pickle", "rb") as file:
            D_v_1 = pickle.load(file)
        with open(files+"N_v_1.pickle", "rb") as file:
            N_v_1 = pickle.load(file)
        with open(files+"T_v_1.pickle", "rb") as file:
            T_v_1 = pickle.load(file)
        with open(files+"F_v_1.pickle", "rb") as file:
            F_v_1 = pickle.load(file)
        with open(files+"P_v_1.pickle", "rb") as file:
            P_v_1 = pickle.load(file)
        with open(files+"Q_v_1.pickle", "rb") as file:
            Q_v_1 = pickle.load(file)
        with open(files+"H_v_1.pickle", "rb") as file:
            H_v_1 = pickle.load(file)

    if calc_1:

        print("Pion pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**4)
            )
        with open(files+"pi_v_1s.pickle", "wb") as file:
            pickle.dump(pi_v_1s, file)

        print("Kaon pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**4)
            )
        with open(files+"K_v_1s.pickle", "wb") as file:
            pickle.dump(K_v_1s, file)

        print("Rho pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**4)
            )
        with open(files+"rho_v_1s.pickle", "wb") as file:
            pickle.dump(rho_v_1s, file)

        print("Omega pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**4)
            )
        with open(files+"omega_v_1s.pickle", "wb") as file:
            pickle.dump(omega_v_1s, file)

        print("Diquark pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**4)
            )
        with open(files+"D_v_1s.pickle", "wb") as file:
            pickle.dump(D_v_1s, file)

        print("Nucleon pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**4)
            )
        with open(files+"N_v_1s.pickle", "wb") as file:
            pickle.dump(N_v_1s, file)

        print("Tetraquark pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**4)
            )
        with open(files+"T_v_1s.pickle", "wb") as file:
            pickle.dump(T_v_1s, file)

        print("F-quark pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**4)
            )
        with open(files+"F_v_1s.pickle", "wb") as file:
            pickle.dump(F_v_1s, file)

        print("Pentaquark pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**4)
            )
        with open(files+"P_v_1s.pickle", "wb") as file:
            pickle.dump(P_v_1s, file)

        print("Q-quark pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**4)
            )
        with open(files+"Q_v_1s.pickle", "wb") as file:
            pickle.dump(Q_v_1s, file)

        print("Hexaquark pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1s.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**4)
            )
        with open(files+"H_v_1s.pickle", "wb") as file:
            pickle.dump(H_v_1s, file)
    else:
        with open(files+"pi_v_1s.pickle", "rb") as file:
            pi_v_1s = pickle.load(file)
        with open(files+"K_v_1s.pickle", "rb") as file:
            K_v_1s = pickle.load(file)
        with open(files+"rho_v_1s.pickle", "rb") as file:
            rho_v_1s = pickle.load(file)
        with open(files+"omega_v_1s.pickle", "rb") as file:
            omega_v_1s = pickle.load(file)
        with open(files+"D_v_1s.pickle", "rb") as file:
            D_v_1s = pickle.load(file)
        with open(files+"N_v_1s.pickle", "rb") as file:
            N_v_1s = pickle.load(file)
        with open(files+"T_v_1s.pickle", "rb") as file:
            T_v_1s = pickle.load(file)
        with open(files+"F_v_1s.pickle", "rb") as file:
            F_v_1s = pickle.load(file)
        with open(files+"P_v_1s.pickle", "rb") as file:
            P_v_1s = pickle.load(file)
        with open(files+"Q_v_1s.pickle", "rb") as file:
            Q_v_1s = pickle.load(file)
        with open(files+"H_v_1s.pickle", "rb") as file:
            H_v_1s = pickle.load(file)

    if calc_2 and False:

        print("Pion pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**4)
            )
        with open(files+"pi_v_2.pickle", "wb") as file:
            pickle.dump(pi_v_2, file)

        print("Kaon pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            K_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**4)
            )
        with open(files+"K_v_2.pickle", "wb") as file:
            pickle.dump(K_v_2, file)

        print("Rho pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            rho_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**4)
            )
        with open(files+"rho_v_2.pickle", "wb") as file:
            pickle.dump(rho_v_2, file)

        print("Omega pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            omega_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**4)
            )
        with open(files+"omega_v_2.pickle", "wb") as file:
            pickle.dump(omega_v_2, file)

        print("Diquark pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**4)
            )
        with open(files+"D_v_2.pickle", "wb") as file:
            pickle.dump(D_v_2, file)

        print("Nucleon pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**4)
            )
        with open(files+"N_v_2.pickle", "wb") as file:
            pickle.dump(N_v_2, file)

        print("Tetraquark pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            T_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**4)
            )
        with open(files+"T_v_2.pickle", "wb") as file:
            pickle.dump(T_v_2, file)

        print("F-quark pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**4)
            )
        with open(files+"F_v_2.pickle", "wb") as file:
            pickle.dump(F_v_2, file)

        print("Pentaquark pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**4)
            )
        with open(files+"P_v_2.pickle", "wb") as file:
            pickle.dump(P_v_2, file)

        print("Q-quark pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**4)
            )
        with open(files+"Q_v_2.pickle", "wb") as file:
            pickle.dump(Q_v_2, file)

        print("Hexaquark pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**4)
            )
        with open(files+"H_v_2.pickle", "wb") as file:
            pickle.dump(H_v_2, file)
    else:
        with open(files+"pi_v_2.pickle", "rb") as file:
            pi_v_2 = pickle.load(file)
        with open(files+"K_v_2.pickle", "rb") as file:
            K_v_2 = pickle.load(file)
        with open(files+"rho_v_2.pickle", "rb") as file:
            rho_v_2 = pickle.load(file)
        with open(files+"omega_v_2.pickle", "rb") as file:
            omega_v_2 = pickle.load(file)
        with open(files+"D_v_2.pickle", "rb") as file:
            D_v_2 = pickle.load(file)
        with open(files+"N_v_2.pickle", "rb") as file:
            N_v_2 = pickle.load(file)
        with open(files+"T_v_2.pickle", "rb") as file:
            T_v_2 = pickle.load(file)
        with open(files+"F_v_2.pickle", "rb") as file:
            F_v_2 = pickle.load(file)
        with open(files+"P_v_2.pickle", "rb") as file:
            P_v_2 = pickle.load(file)
        with open(files+"Q_v_2.pickle", "rb") as file:
            Q_v_2 = pickle.load(file)
        with open(files+"H_v_2.pickle", "rb") as file:
            H_v_2 = pickle.load(file)

    if calc_2:

        print("Pion pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**4)
            )
        with open(files+"pi_v_2s.pickle", "wb") as file:
            pickle.dump(pi_v_2s, file)

        print("Kaon pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            K_v_2s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**4)
            )
        with open(files+"K_v_2s.pickle", "wb") as file:
            pickle.dump(K_v_2s, file)

        print("Rho pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            rho_v_2s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**4)
            )
        with open(files+"rho_v_2s.pickle", "wb") as file:
            pickle.dump(rho_v_2s, file)

        print("Omega pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            omega_v_2s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**4)
            )
        with open(files+"omega_v_2s.pickle", "wb") as file:
            pickle.dump(omega_v_2s, file)

        print("Diquark pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**4)
            )
        with open(files+"D_v_2s.pickle", "wb") as file:
            pickle.dump(D_v_2s, file)

        print("Nucleon pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**4)
            )
        with open(files+"N_v_2s.pickle", "wb") as file:
            pickle.dump(N_v_2s, file)

        print("Tetraquark pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            T_v_2s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**4)
            )
        with open(files+"T_v_2s.pickle", "wb") as file:
            pickle.dump(T_v_2s, file)

        print("F-quark pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**4)
            )
        with open(files+"F_v_2s.pickle", "wb") as file:
            pickle.dump(F_v_2s, file)

        print("Pentaquark pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**4)
            )
        with open(files+"P_v_2s.pickle", "wb") as file:
            pickle.dump(P_v_2s, file)

        print("Q-quark pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**4)
            )
        with open(files+"Q_v_2s.pickle", "wb") as file:
            pickle.dump(Q_v_2s, file)

        print("Hexaquark pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2s.append(
                cluster_s.pressure_ib_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**4)
            )
        with open(files+"H_v_2s.pickle", "wb") as file:
            pickle.dump(H_v_2s, file)
    else:
        with open(files+"pi_v_2s.pickle", "rb") as file:
            pi_v_2s = pickle.load(file)
        with open(files+"K_v_2s.pickle", "rb") as file:
            K_v_2s = pickle.load(file)
        with open(files+"rho_v_2s.pickle", "rb") as file:
            rho_v_2s = pickle.load(file)
        with open(files+"omega_v_2s.pickle", "rb") as file:
            omega_v_2s = pickle.load(file)
        with open(files+"D_v_2s.pickle", "rb") as file:
            D_v_2s = pickle.load(file)
        with open(files+"N_v_2s.pickle", "rb") as file:
            N_v_2s = pickle.load(file)
        with open(files+"T_v_2s.pickle", "rb") as file:
            T_v_2s = pickle.load(file)
        with open(files+"F_v_2s.pickle", "rb") as file:
            F_v_2s = pickle.load(file)
        with open(files+"P_v_2s.pickle", "rb") as file:
            P_v_2s = pickle.load(file)
        with open(files+"Q_v_2s.pickle", "rb") as file:
            Q_v_2s = pickle.load(file)
        with open(files+"H_v_2s.pickle", "rb") as file:
            H_v_2s = pickle.load(file)

    with open(
        lattice + "bazavov_1407_6387_mu0.pickle", "rb"
    ) as file:
        bazavov_1407_6387_mu0 = pickle.load(file)
    with open(
        lattice + "borsanyi_1309_5258_mu0.pickle", "rb"
    ) as file:
        borsanyi_1309_5258_mu0 = pickle.load(file)
    with open(
        lattice + "bazavov_1710_05024_mu0.pickle", "rb"
    ) as file:
        bazavov_1710_05024_mu0 = pickle.load(file)
    with open(
        lattice + "borsanyi_1204_6710v2_mu0.pickle", "rb"
    ) as file:
        borsanyi_1204_6710v2_mu0 = pickle.load(file)

    with open(
        lattice + "borsanyi_1204_6710v2_mu200.pickle", "rb"
    ) as file:
        borsanyi_1204_6710v2_mu200 = pickle.load(file)

    total_qgp_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1,gluon_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_qgp_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2,gluon_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_cluster_1 = [
        sum(el) for el in 
            zip(
                pi_v_1, K_v_1, rho_v_1, omega_v_1, N_v_1,
                T_v_1, P_v_1, H_v_1
            )
    ]
    total_ccluster_1 = [
        sum(el) for el in 
            zip(
                D_v_1, F_v_1, Q_v_1
            )
    ]
    total_cluster_1s = [
        sum(el) for el in 
            zip(
                pi_v_1s, K_v_1s, rho_v_1s, omega_v_1s, N_v_1s,
                T_v_1s, P_v_1s, H_v_1s
            )
    ]
    total_ccluster_1s = [
        sum(el) for el in 
            zip(
                D_v_1s, F_v_1s, Q_v_1s
            )
    ]
    total_1 = [sum(el) for el in zip(total_qgp_1, total_cluster_1, total_ccluster_1)]
    total_1s = [sum(el) for el in zip(total_qgp_1, total_cluster_1s, total_ccluster_1s)]
    total_cluster_2 = [
        sum(el) for el in 
            zip(
                pi_v_2, K_v_2, rho_v_2, omega_v_2, N_v_2,
                T_v_2, P_v_2, H_v_2
            )
    ]
    total_cluster_2s = [
        sum(el) for el in 
            zip(
                pi_v_2s, K_v_2s, rho_v_2s, omega_v_2s, N_v_2s,
                T_v_2s, P_v_2s, H_v_2s
            )
    ]
    total_ccluster_2 = [
        sum(el) for el in 
            zip(
                D_v_2, F_v_2, Q_v_2
            )
    ]
    total_ccluster_2s = [
        sum(el) for el in 
            zip(
                D_v_2s, F_v_2s, Q_v_2s
            )
    ]
    total_2 = [sum(el) for el in zip(total_qgp_2, total_cluster_2, total_ccluster_2)]
    total_2s = [sum(el) for el in zip(total_qgp_2, total_cluster_2s, total_ccluster_2s)]

    fig1 = matplotlib.pyplot.figure(num=1, figsize=(12.0, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.axis([10., 400., 0., 4.2])

    ax1.add_patch(
        matplotlib.patches.Polygon(
            bazavov_1407_6387_mu0, closed = True, fill = True, color = 'red', alpha = 0.3
        )
    )
    ax1.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1309_5258_mu0, closed = True, fill = True, color = 'green', alpha = 0.3
        )
    )
    ax1.add_patch(
        matplotlib.patches.Polygon(
            bazavov_1710_05024_mu0, closed = True, fill = True, color = 'magenta', alpha = 0.3
        )
    )
    ax1.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1204_6710v2_mu0, closed = True, fill = True, color = 'blue', alpha = 0.3
        )
    )

    ax1.plot(T_1, total_cluster_1, '-', c = 'green')
    ax1.plot(T_1, total_cluster_1s, '--', c = 'green')
    ax1.plot(T_1, total_ccluster_1, '-', c = 'red')
    ax1.plot(T_1, total_ccluster_1s, '--', c = 'red')
    ax1.plot(T_1, total_1, '-', c = 'black')
    ax1.plot(T_1, total_1s, '--', c = 'black')
    ax1.plot(T_1, total_qgp_1, '-.', c = 'blue')

    ax1.text(
        180.0, 0.1, r'Color charged clusters', color='red', fontsize=14
    )
    ax1.text(
        165.0, 0.34, r'Color singlet clusters', color='green', fontsize=14
    )
    ax1.text(
        170.0, 0.58, r'PNJL', color='blue', fontsize=14
    )
    ax1.text(
        195.0, 1.16, r'total pressure', color='black', fontsize=14
    )
    ax1.text(
        21.0, 3.9, r'$\mathrm{\mu_B=0}$', color='black', fontsize=14
    )
    ax1.text(
        22.0, 2.2, r'Borsanyi et al. (2012)', color='blue',
        alpha=0.7, fontsize=14
    )
    ax1.text(
        175.0, 3.7, r'Borsanyi et al. (2014)', color='green',
        alpha=0.7, fontsize=14
    )
    ax1.text(
        75.0, 3.0, r'Bazavov et al. (2014)', color='red',
        alpha=0.7, fontsize=14
    )

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize=16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize=16)

    ax3 = fig1.add_subplot(1, 2, 2)
    ax3.axis([10., 400., 0., 4.2])

    ax3.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1204_6710v2_mu200, closed=True, fill=True, color='blue', alpha=0.3
        )
    )

    ax3.plot(T_2, total_cluster_2, '-', color='green')
    ax3.plot(T_2, total_cluster_2s, '--', color='green')
    ax3.plot(T_2, total_ccluster_2, '-', color='red')
    ax3.plot(T_2, total_ccluster_2s, '--', color='red')
    ax3.plot(T_2, total_2, '-', c='black')
    ax3.plot(T_2, total_2s, '--', c='black')
    ax3.plot(T_2, total_qgp_2, '-.', c='blue')

    ax3.text(
        180.0, 0.1, r'Color charged clusters', color='red', fontsize=14
    )
    ax3.text(
        165.0, 0.34, r'Color singlet clusters', color='green', fontsize=14
    )
    ax3.text(
        170.0, 0.58, r'PNJL', color='blue', fontsize=14
    )
    ax3.text(
        20.0, 1.5, r'total pressure', color='black', fontsize=14
    )
    ax3.text(
        21.0, 3.9, r'$\mathrm{\mu_B=200}$ MeV', color='black', fontsize=14
    )
    ax3.text(
        18.0, 2.2, r'Borsanyi et al. (2012)', color='blue',
        alpha=0.7, fontsize=14
    )

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize=16)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_ylabel(r'$\mathrm{p/T^4}$', fontsize=16)

    fig1.tight_layout(pad=0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_experimental_hrg_benchmark():

    import tqdm
    import math
    import numpy
    import pickle
    import warnings
    import platform

    import matplotlib.pyplot

    import utils
    import pnjl.thermo.gcp_pl.polynomial
    import pnjl.thermo.gcp_perturbative.const
    import pnjl.thermo.gcp_pnjl.lattice_cut_sea
    import pnjl.thermo.gcp_cluster.hrg \
        as cluster
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_mhrg
    import pnjl.thermo.gcp_cluster.bound_step_continuum_quad \
        as cluster_mhrg2

    import pnjl.thermo.solvers.\
        pnjl_lattice_cut_sea.\
        pl_polynomial.\
        pert_const.\
        no_clusters \
    as solver

    warnings.filterwarnings("ignore")
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")
    
    calc_1 = False
    calc_2 = False

    files = "D:/EoS/epja/experimental/hrg_benchmark/"
    lattice_files = "D:/EoS/epja/lattice_data_raw/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/experimental/hrg_benchmark/"
        lattice_files = "/home/mcierniak/Data/2023_epja/lattice_data_raw/"

    T_1 = numpy.linspace(1.0, 280.0, 200)
    T_2 = numpy.linspace(1.0, 280.0, 200)

    muB_1 = [0.0 for el in T_1]
    muB_2 = [2.5 * el for el in T_2]

    hadrons_reduced = (
        'pi0', 'pi', 'K', 'K0', 'eta', 'rho', 'omega', 'K*(892)', 'K*0(892)',
        'p', 'n', 'etaPrime', 'a0', 'f0', 'phi', 'Lambda', 'h1', 'Sigma+',
        'Sigma0', 'Sigma-', 'b1', 'a1', 'Delta'
    )

    phi_re_v_1, phi_im_v_1 = list(), list()
    phi_re_v_2, phi_im_v_2 = list(), list()

    qgp_sdensity_v_1, qgp_partial_sdensity_v_1 = list(), list()
    hrg_full_sdensity_v_1, hrg_full_partial_sdensity_v_1 = list(), list()
    mhrg_sdensity_v_1, mhrg_partial_sdensity_v_1 = list(), list()
    mhrg2_sdensity_v_1, mhrg2_partial_sdensity_v_1 = list(), list()
    hrg_reduced_sdensity_v_1, hrg_reduced_partial_sdensity_v_1 = list(), list()

    qgp_bdensity_v_1, qgp_partial_bdensity_v_1 = list(), list()
    hrg_full_bdensity_v_1, hrg_full_partial_bdensity_v_1 = list(), list()
    mhrg_bdensity_v_1, mhrg_partial_bdensity_v_1 = list(), list()
    mhrg2_bdensity_v_1, mhrg2_partial_bdensity_v_1 = list(), list()
    hrg_reduced_bdensity_v_1, hrg_reduced_partial_bdensity_v_1 = list(), list()

    qgp_sdensity_v_2, qgp_partial_sdensity_v_2 = list(), list()
    hrg_full_sdensity_v_2, hrg_full_partial_sdensity_v_2 = list(), list()
    mhrg_sdensity_v_2, mhrg_partial_sdensity_v_2 = list(), list()
    mhrg2_sdensity_v_2, mhrg2_partial_sdensity_v_2 = list(), list()
    hrg_reduced_sdensity_v_2, hrg_reduced_partial_sdensity_v_2 = list(), list()

    qgp_bdensity_v_2, qgp_partial_bdensity_v_2 = list(), list()
    hrg_full_bdensity_v_2, hrg_full_partial_bdensity_v_2 = list(), list()
    mhrg_bdensity_v_2, mhrg_partial_bdensity_v_2 = list(), list()
    mhrg2_bdensity_v_2, mhrg2_partial_bdensity_v_2 = list(), list()
    hrg_reduced_bdensity_v_2, hrg_reduced_partial_bdensity_v_2 = list(), list()

    def qgp_sdensity_all(T: float, muB: float, phi_re_0=1e-5, phi_im_0=2e-5, calc_phi=True):
        partial = list()
        phi_result = (phi_re_0, phi_im_0)
        if calc_phi:
            phi_result = solver.Polyakov_loop(T, muB/3.0, phi_re_0, phi_im_0)
        pars = (T, muB/3.0, phi_result[0], phi_result[1], solver.Polyakov_loop)
        #Sigma sdensity
        partial.append(
            pnjl.thermo.gcp_pnjl.lattice_cut_sea.sdensity_field(T, muB/3.0)/(T**3)
        )
        #Gluon sdensity
        partial.append(pnjl.thermo.gcp_pl.polynomial.sdensity(*pars)/(T**3))
        #Sea sdensity
        lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.sdensity_sea(T, muB/3.0, 'l')/(T**3)
        sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.sdensity_sea(T, muB/3.0, 's')/(T**3)
        partial.append(lq_temp)
        partial.append(lq_temp)
        partial.append(sq_temp)
        #Perturbative sdensity
        lq_temp = pnjl.thermo.gcp_perturbative.const.sdensity(*pars)/(T**3)
        partial.append(lq_temp)
        partial.append(lq_temp)
        partial.append(lq_temp)
        #PNJL sdensity
        lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.sdensity_q(*pars, 'l')/(T**3)
        sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.sdensity_q(*pars, 's')/(T**3)
        partial.append(lq_temp)
        partial.append(lq_temp)
        partial.append(sq_temp)
        return phi_result[0], phi_result[1], math.fsum(partial), (*partial,)
    
    def qgp_bdensity_all(T: float, muB: float, phi_re_0=1e-5, phi_im_0=2e-5, calc_phi=True):
        partial = list()
        phi_result = (phi_re_0, phi_im_0)
        if calc_phi:
            phi_result = solver.Polyakov_loop(T, muB/3.0, phi_re_0, phi_im_0)
        pars = (T, muB/3.0, phi_result[0], phi_result[1], solver.Polyakov_loop)
        #Sigma bdensity
        partial.append(
            pnjl.thermo.gcp_pnjl.lattice_cut_sea.bdensity_field(T, muB/3.0)/(T**3)
        )
        #Gluon bdensity
        partial.append(pnjl.thermo.gcp_pl.polynomial.bdensity(*pars)/(T**3))
        #Sea bdensity
        lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.bdensity_sea(T, muB/3.0, 'l')/(T**3)
        sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.bdensity_sea(T, muB/3.0, 's')/(T**3)
        partial.append(lq_temp)
        partial.append(lq_temp)
        partial.append(sq_temp)
        #Perturbative bdensity
        lq_temp = pnjl.thermo.gcp_perturbative.const.bdensity(*pars)/(T**3)
        partial.append(lq_temp)
        partial.append(lq_temp)
        partial.append(lq_temp)
        #PNJL bdensity
        lq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.bdensity_q(*pars, 'l')/(T**3)
        sq_temp = pnjl.thermo.gcp_pnjl.lattice_cut_sea.bdensity_q(*pars, 's')/(T**3)
        partial.append(lq_temp)
        partial.append(lq_temp)
        partial.append(sq_temp)
        return phi_result[0], phi_result[1], math.fsum(partial), (*partial,)

    lQCD_1_x, lQCD_1_y = \
        utils.data_load(
            lattice_files+"2212_09043_fig13_top_right_0p0_alt2.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_1 = [[x, y] for x, y in zip(lQCD_1_x, lQCD_1_y)]

    lQCD_2_x, lQCD_2_y = \
        utils.data_load(
            lattice_files+"2212_09043_fig13_top_right_2p5.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_2 = [[x, y] for x, y in zip(lQCD_2_x, lQCD_2_y)]

    lQCD_new_x, lQCD_new_y = \
        utils.data_load(
            lattice_files+"2202_09184v2_fig2_mub_T_2p5_nb.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_new = [[x, y] for x, y in zip(lQCD_new_x, lQCD_new_y)]

    if calc_1:
        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("QGP PL and sdensity #1")
        for T_el, muB_el in tqdm.tqdm(
            zip(T_1, muB_1), total=len(T_1), ncols=100
        ):
            phi_re_0, phi_im_0, temp_qgp, temp_qgp_partials = qgp_sdensity_all(
                T_el, muB_el, phi_re_0, phi_im_0
            )
            phi_re_v_1.append(phi_re_0)
            phi_im_v_1.append(phi_im_0)
            qgp_sdensity_v_1.append(temp_qgp)
            qgp_partial_sdensity_v_1.append(temp_qgp_partials)
        with open(files+"phi_re_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)
        with open(files+"qgp_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(qgp_sdensity_v_1, file)
        with open(files+"qgp_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(qgp_partial_sdensity_v_1, file)
        print("QGP bdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            _, _, temp_qgp, temp_qgp_partials = qgp_bdensity_all(
                T_el, muB_el, phi_re_el, phi_im_el, calc_phi=False
            )
            qgp_bdensity_v_1.append(temp_qgp)
            qgp_partial_bdensity_v_1.append(temp_qgp_partials)
        with open(files+"qgp_bdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(qgp_bdensity_v_1, file)
        with open(files+"qgp_partial_bdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(qgp_partial_bdensity_v_1, file)
    else:
        with open(files+"phi_re_v_0p0.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_0p0.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
        with open(files+"qgp_sdensity_v_0p0.pickle", "rb") as file:
            qgp_sdensity_v_1 = pickle.load(file)
        with open(files+"qgp_partial_sdensity_v_0p0.pickle", "rb") as file:
            qgp_partial_sdensity_v_1 = pickle.load(file)
        with open(files+"qgp_bdensity_v_0p0.pickle", "rb") as file:
            qgp_bdensity_v_1 = pickle.load(file)
        with open(files+"qgp_partial_bdensity_v_0p0.pickle", "rb") as file:
            qgp_partial_bdensity_v_1 = pickle.load(file)

    if calc_2:
        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("QGP PL and sdensity #2")
        for T_el, muB_el in tqdm.tqdm(
            zip(T_2, muB_2), total=len(T_2), ncols=100
        ):
            phi_re_0, phi_im_0, temp_qgp, temp_qgp_partials = qgp_sdensity_all(
                T_el, muB_el, phi_re_0, phi_im_0
            )
            phi_re_v_2.append(phi_re_0)
            phi_im_v_2.append(phi_im_0)
            qgp_sdensity_v_2.append(temp_qgp)
            qgp_partial_sdensity_v_2.append(temp_qgp_partials)
        with open(files+"phi_re_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)
        with open(files+"qgp_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_sdensity_v_2, file)
        with open(files+"qgp_partial_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_partial_sdensity_v_2, file)
        print("QGP bdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            _, _, temp_qgp, temp_qgp_partials = qgp_bdensity_all(
                T_el, muB_el, phi_re_el, phi_im_el, calc_phi=False
            )
            qgp_bdensity_v_2.append(temp_qgp)
            qgp_partial_bdensity_v_2.append(temp_qgp_partials)
        with open(files+"qgp_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_bdensity_v_2, file)
        with open(files+"qgp_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_partial_bdensity_v_2, file)
    else:
        with open(files+"phi_re_v_2p5.pickle", "rb") as file:
            phi_re_v_2 = pickle.load(file)
        with open(files+"phi_im_v_2p5.pickle", "rb") as file:
            phi_im_v_2 = pickle.load(file)
        with open(files+"qgp_sdensity_v_2p5.pickle", "rb") as file:
            qgp_sdensity_v_2 = pickle.load(file)
        with open(files+"qgp_partial_sdensity_v_2p5.pickle", "rb") as file:
            qgp_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"qgp_bdensity_v_2p5.pickle", "rb") as file:
            qgp_bdensity_v_2 = pickle.load(file)
        with open(files+"qgp_partial_bdensity_v_2p5.pickle", "rb") as file:
            qgp_partial_bdensity_v_2 = pickle.load(file)

    if False:
        print("Full HRG sdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.sdensity_multi(T_el, muB_el)
            hrg_full_sdensity_v_1.append(temp_hrg/(T_el**3))
            hrg_full_partial_sdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_full_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg_full_sdensity_v_1, file)
        with open(files+"hrg_full_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg_full_partial_sdensity_v_1, file)
        print("Full HRG bdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.bdensity_multi(T_el, muB_el)
            hrg_full_bdensity_v_1.append(temp_hrg/(T_el**3))
            hrg_full_partial_bdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_full_bdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg_full_bdensity_v_1, file)
        with open(files+"hrg_full_partial_bdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg_full_partial_bdensity_v_1, file)
    else:
        with open(files+"hrg_full_sdensity_v_0p0.pickle", "rb") as file:
            hrg_full_sdensity_v_1 = pickle.load(file)
        with open(files+"hrg_full_partial_sdensity_v_0p0.pickle", "rb") as file:
            hrg_full_partial_sdensity_v_1 = pickle.load(file)
        with open(files+"hrg_full_bdensity_v_0p0.pickle", "rb") as file:
            hrg_full_bdensity_v_1 = pickle.load(file)
        with open(files+"hrg_full_partial_bdensity_v_0p0.pickle", "rb") as file:
            hrg_full_partial_bdensity_v_1 = pickle.load(file)

    if False:
        print("MHRG sdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg.sdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg_sdensity_v_1.append(temp_hrg/(T_el**3))
            mhrg_partial_sdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg_sdensity_v_1, file)
        with open(files+"mhrg_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg_partial_sdensity_v_1, file)
        print("MHRG bdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg.bdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg_bdensity_v_1.append(temp_hrg/(T_el**3))
            mhrg_partial_bdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg_bdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg_bdensity_v_1, file)
        with open(files+"mhrg_partial_bdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg_partial_bdensity_v_1, file)
    else:
        with open(files+"mhrg_sdensity_v_0p0.pickle", "rb") as file:
            mhrg_sdensity_v_1 = pickle.load(file)
        with open(files+"mhrg_partial_sdensity_v_0p0.pickle", "rb") as file:
            mhrg_partial_sdensity_v_1 = pickle.load(file)
        with open(files+"mhrg_bdensity_v_0p0.pickle", "rb") as file:
            mhrg_bdensity_v_1 = pickle.load(file)
        with open(files+"mhrg_partial_bdensity_v_0p0.pickle", "rb") as file:
            mhrg_partial_bdensity_v_1 = pickle.load(file)

    if False:
        print("MHRG (continuum) sdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg2.sdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg2_sdensity_v_1.append(temp_hrg/(T_el**3))
            mhrg2_partial_sdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg2_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg2_sdensity_v_1, file)
        with open(files+"mhrg2_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg2_partial_sdensity_v_1, file)
        print("MHRG (continuum) bdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg2.bdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg2_bdensity_v_1.append(temp_hrg/(T_el**3))
            mhrg2_partial_bdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg2_bdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg2_bdensity_v_1, file)
        with open(files+"mhrg2_partial_bdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg2_partial_bdensity_v_1, file)
    else:
        with open(files+"mhrg2_sdensity_v_0p0.pickle", "rb") as file:
            mhrg2_sdensity_v_1 = pickle.load(file)
        with open(files+"mhrg2_partial_sdensity_v_0p0.pickle", "rb") as file:
            mhrg2_partial_sdensity_v_1 = pickle.load(file)
        with open(files+"mhrg2_bdensity_v_0p0.pickle", "rb") as file:
            mhrg2_bdensity_v_1 = pickle.load(file)
        with open(files+"mhrg2_partial_bdensity_v_0p0.pickle", "rb") as file:
            mhrg2_partial_bdensity_v_1 = pickle.load(file)

    if False:
        print("Reduced HRG sdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.sdensity_multi(T_el, muB_el, hadrons=hadrons_reduced)
            hrg_reduced_sdensity_v_1.append(temp_hrg/(T_el**3))
            hrg_reduced_partial_sdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_reduced_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg_reduced_sdensity_v_1, file)
        with open(files+"hrg_reduced_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg_reduced_partial_sdensity_v_1, file)
        print("Reduced HRG bdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.bdensity_multi(T_el, muB_el, hadrons=hadrons_reduced)
            hrg_reduced_bdensity_v_1.append(temp_hrg/(T_el**3))
            hrg_reduced_partial_bdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_reduced_bdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg_reduced_bdensity_v_1, file)
        with open(files+"hrg_reduced_partial_bdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg_reduced_partial_bdensity_v_1, file)
    else:
        with open(files+"hrg_reduced_sdensity_v_0p0.pickle", "rb") as file:
            hrg_reduced_sdensity_v_1 = pickle.load(file)
        with open(files+"hrg_reduced_partial_sdensity_v_0p0.pickle", "rb") as file:
            hrg_reduced_partial_sdensity_v_1 = pickle.load(file)
        with open(files+"hrg_reduced_bdensity_v_0p0.pickle", "rb") as file:
            hrg_reduced_bdensity_v_1 = pickle.load(file)
        with open(files+"hrg_reduced_partial_bdensity_v_0p0.pickle", "rb") as file:
            hrg_reduced_partial_bdensity_v_1 = pickle.load(file)

    if False:
        print("Full HRG sdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.sdensity_multi(T_el, muB_el)
            hrg_full_sdensity_v_2.append(temp_hrg/(T_el**3))
            hrg_full_partial_sdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_full_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_full_sdensity_v_2, file)
        with open(files+"hrg_full_partial_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_full_partial_sdensity_v_2, file)
        print("Full HRG bdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.bdensity_multi(T_el, muB_el)
            hrg_full_bdensity_v_2.append(temp_hrg/(T_el**3))
            hrg_full_partial_bdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_full_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_full_bdensity_v_2, file)
        with open(files+"hrg_full_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_full_partial_bdensity_v_2, file)
    else:
        with open(files+"hrg_full_sdensity_v_2p5.pickle", "rb") as file:
            hrg_full_sdensity_v_2 = pickle.load(file)
        with open(files+"hrg_full_partial_sdensity_v_2p5.pickle", "rb") as file:
            hrg_full_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"hrg_full_bdensity_v_2p5.pickle", "rb") as file:
            hrg_full_bdensity_v_2 = pickle.load(file)
        with open(files+"hrg_full_partial_bdensity_v_2p5.pickle", "rb") as file:
            hrg_full_partial_bdensity_v_2 = pickle.load(file)

    if False:
        print("MHRG sdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg.sdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg_sdensity_v_2.append(temp_hrg/(T_el**3))
            mhrg_partial_sdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg_sdensity_v_2, file)
        with open(files+"mhrg_partial_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg_partial_sdensity_v_2, file)
        print("MHRG bdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg.bdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg_bdensity_v_2.append(temp_hrg/(T_el**3))
            mhrg_partial_bdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg_bdensity_v_2, file)
        with open(files+"mhrg_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg_partial_bdensity_v_2, file)
    else:
        with open(files+"mhrg_sdensity_v_2p5.pickle", "rb") as file:
            mhrg_sdensity_v_2 = pickle.load(file)
        with open(files+"mhrg_partial_sdensity_v_2p5.pickle", "rb") as file:
            mhrg_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"mhrg_bdensity_v_2p5.pickle", "rb") as file:
            mhrg_bdensity_v_2 = pickle.load(file)
        with open(files+"mhrg_partial_bdensity_v_2p5.pickle", "rb") as file:
            mhrg_partial_bdensity_v_2 = pickle.load(file)

    if False:
        print("MHRG (continuum) sdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg2.sdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg2_sdensity_v_2.append(temp_hrg/(T_el**3))
            mhrg2_partial_sdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg2_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg2_sdensity_v_2, file)
        with open(files+"mhrg2_partial_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg2_partial_sdensity_v_2, file)
        print("MHRG (continuum) bdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg2.bdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg2_bdensity_v_2.append(temp_hrg/(T_el**3))
            mhrg2_partial_bdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg2_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg2_bdensity_v_2, file)
        with open(files+"mhrg2_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg2_partial_bdensity_v_2, file)
    else:
        with open(files+"mhrg2_sdensity_v_2p5.pickle", "rb") as file:
            mhrg2_sdensity_v_2 = pickle.load(file)
        with open(files+"mhrg2_partial_sdensity_v_2p5.pickle", "rb") as file:
            mhrg2_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"mhrg2_bdensity_v_2p5.pickle", "rb") as file:
            mhrg2_bdensity_v_2 = pickle.load(file)
        with open(files+"mhrg2_partial_bdensity_v_2p5.pickle", "rb") as file:
            mhrg2_partial_bdensity_v_2 = pickle.load(file)

    if False:
        print("Reduced HRG sdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.sdensity_multi(T_el, muB_el, hadrons=hadrons_reduced)
            hrg_reduced_sdensity_v_2.append(temp_hrg/(T_el**3))
            hrg_reduced_partial_sdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_reduced_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_reduced_sdensity_v_2, file)
        with open(files+"hrg_reduced_partial_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_reduced_partial_sdensity_v_2, file)
        print("Reduced HRG bdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.bdensity_multi(T_el, muB_el, hadrons=hadrons_reduced)
            hrg_reduced_bdensity_v_2.append(temp_hrg/(T_el**3))
            hrg_reduced_partial_bdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_reduced_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_reduced_bdensity_v_2, file)
        with open(files+"hrg_reduced_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_reduced_partial_bdensity_v_2, file)
    else:
        with open(files+"hrg_reduced_sdensity_v_2p5.pickle", "rb") as file:
            hrg_reduced_sdensity_v_2 = pickle.load(file)
        with open(files+"hrg_reduced_partial_sdensity_v_2p5.pickle", "rb") as file:
            hrg_reduced_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"hrg_reduced_bdensity_v_2p5.pickle", "rb") as file:
            hrg_reduced_bdensity_v_2 = pickle.load(file)
        with open(files+"hrg_reduced_partial_bdensity_v_2p5.pickle", "rb") as file:
            hrg_reduced_partial_bdensity_v_2 = pickle.load(file)

    mhrg_sdensity_color_clusters_1 = [
        math.fsum([
            el[23], el[24], el[25], el[26], el[27], el[28], el[31], el[32], el[33]
        ]) for el in mhrg_partial_sdensity_v_1
    ]
    mhrg_sdensity_singlet_clusters_1 = [
        el_total - el_color for el_total, el_color in zip(mhrg_sdensity_v_1, mhrg_sdensity_color_clusters_1)
    ]
    mhrg2_sdensity_color_clusters_1 = [
        math.fsum([
            el[23], el[24], el[25], el[26], el[27], el[28], el[31], el[32], el[33]
        ]) for el in mhrg2_partial_sdensity_v_1
    ]
    mhrg2_sdensity_singlet_clusters_1 = [
        el_total - el_color for el_total, el_color in zip(mhrg2_sdensity_v_1, mhrg2_sdensity_color_clusters_1)
    ]

    mhrg_sdensity_color_clusters_2 = [
        math.fsum([
            el[23], el[24], el[25], el[26], el[27], el[28], el[31], el[32], el[33]
        ]) for el in mhrg_partial_sdensity_v_2
    ]
    mhrg_sdensity_singlet_clusters_2 = [
        el_total - el_color for el_total, el_color in zip(mhrg_sdensity_v_2, mhrg_sdensity_color_clusters_2)
    ]
    mhrg2_sdensity_color_clusters_2 = [
        math.fsum([
            el[23], el[24], el[25], el[26], el[27], el[28], el[31], el[32], el[33]
        ]) for el in mhrg2_partial_sdensity_v_2
    ]
    mhrg2_sdensity_singlet_clusters_2 = [
        el_total - el_color for el_total, el_color in zip(mhrg2_sdensity_v_2, mhrg2_sdensity_color_clusters_2)
    ]

    mhrg_bdensity_color_clusters_2 = [
        math.fsum([
            el[23], el[24], el[25], el[26], el[27], el[28], el[31], el[32], el[33]
        ]) for el in mhrg_partial_bdensity_v_2
    ]
    mhrg_bdensity_singlet_clusters_2 = [
        el_total - el_color for el_total, el_color in zip(mhrg_bdensity_v_2, mhrg_bdensity_color_clusters_2)
    ]
    mhrg2_bdensity_color_clusters_2 = [
        math.fsum([
            el[23], el[24], el[25], el[26], el[27], el[28], el[31], el[32], el[33]
        ]) for el in mhrg2_partial_bdensity_v_2
    ]
    mhrg2_bdensity_singlet_clusters_2 = [
        el_total - el_color for el_total, el_color in zip(mhrg2_bdensity_v_2, mhrg2_bdensity_color_clusters_2)
    ]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (18.0, 5.0))
    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.axis([80., 280., 0.0, 20.0])

    ax1.add_patch(
        matplotlib.patches.Polygon(lQCD_1, 
            closed = True, fill = True, color = 'green', alpha = 0.3
        )
    )

    ax1.plot(T_1, hrg_full_sdensity_v_1, '-', c = 'black')
    ax1.plot(T_1, hrg_reduced_sdensity_v_1, '-.', c = 'black')
    ax1.plot(T_1, mhrg_sdensity_v_1, '-.', c = 'blue')
    ax1.plot(T_1, mhrg_sdensity_color_clusters_1, '-.', c = 'red')
    ax1.plot(T_1, mhrg_sdensity_singlet_clusters_1, '-.', c = 'green')
    ax1.plot(T_1, mhrg2_sdensity_v_1, '-', c = 'blue')
    ax1.plot(T_1, mhrg2_sdensity_color_clusters_1, '-', c = 'red')
    ax1.plot(T_1, mhrg2_sdensity_singlet_clusters_1, '-', c = 'green')

    ax1.text(85, 18.5, r"$\mathrm{\mu_B/T=0}$", color="black", fontsize=14)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.axis([80., 280., 0.0, 20.0])

    ax2.add_patch(
        matplotlib.patches.Polygon(lQCD_2, 
            closed = True, fill = True, color = 'green', alpha = 0.3
        )
    )

    ax2.plot(T_2, hrg_full_sdensity_v_2, '-', c = 'black')
    ax2.plot(T_2, hrg_reduced_sdensity_v_2, '-.', c = 'black')
    ax2.plot(T_2, mhrg_sdensity_v_2, '-.', c = 'blue')
    ax2.plot(T_2, mhrg_sdensity_color_clusters_2, '-.', c = 'red')
    ax2.plot(T_2, mhrg_sdensity_singlet_clusters_2, '-.', c = 'green')
    ax2.plot(T_2, mhrg2_sdensity_v_2, '-', c = 'blue')
    ax2.plot(T_2, mhrg2_sdensity_color_clusters_2, '-', c = 'red')
    ax2.plot(T_2, mhrg2_sdensity_singlet_clusters_2, '-', c = 'green')
    # ax2.plot(T_2, [el[0] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # pi0
    # ax2.plot(T_2, [el[1] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # pi
    # ax2.plot(T_2, [el[2] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # K
    # ax2.plot(T_2, [el[3] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # K0
    # ax2.plot(T_2, [el[4] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # eta
    # ax2.plot(T_2, [el[5] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # rho
    # ax2.plot(T_2, [el[6] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # omega
    # ax2.plot(T_2, [el[7] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # K*(892)
    # ax2.plot(T_2, [el[8] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # K*0(892)
    # ax2.plot(T_2, [el[9] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # p
    # ax2.plot(T_2, [el[10] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # n
    # ax2.plot(T_2, [el[11] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # etaPrime
    # ax2.plot(T_2, [el[12] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # a0
    # ax2.plot(T_2, [el[13] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # f0
    # ax2.plot(T_2, [el[14] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # phi
    # ax2.plot(T_2, [el[15] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # Lambda
    # ax2.plot(T_2, [el[16] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # h1
    # ax2.plot(T_2, [el[17] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # Sigma+
    # ax2.plot(T_2, [el[18] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # Sigma0
    # ax2.plot(T_2, [el[19] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # Sigma-
    # ax2.plot(T_2, [el[20] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # b1
    # ax2.plot(T_2, [el[21] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # a1
    # ax2.plot(T_2, [el[22] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # Delta
    # ax2.plot(T_2, [el[23] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # D1
    # ax2.plot(T_2, [el[24] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # D2
    # ax2.plot(T_2, [el[25] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # 4q1
    # ax2.plot(T_2, [el[26] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # 4q2
    # ax2.plot(T_2, [el[27] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # 4q3
    # ax2.plot(T_2, [el[28] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # 4q4
    # ax2.plot(T_2, [el[29] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # P1
    # ax2.plot(T_2, [el[30] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # P2
    # ax2.plot(T_2, [el[31] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # 5q1
    # ax2.plot(T_2, [el[32] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # 5q2
    # ax2.plot(T_2, [el[33] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # 5q3
    # ax2.plot(T_2, [el[34] for el in mhrg2_partial_sdensity_v_2], ':', c = 'green') # d
    
    ax2.text(85, 18.5, r"$\mathrm{\mu_B/T=2.5}$", color="black", fontsize=14)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    ax3 = fig1.add_subplot(1, 3, 3)
    ax3.axis([80., 280., 0.0, 1.0])

    ax3.add_patch(
        matplotlib.patches.Polygon(
            lQCD_new, closed = True, fill = True, color = "green", alpha = 0.3
        )
    )

    ax3.plot(T_2, hrg_full_bdensity_v_2, '-', c = 'black')
    ax3.plot(T_2, hrg_reduced_bdensity_v_2, '-.', c = 'black')
    ax3.plot(T_2, mhrg_bdensity_v_2, '-.', c = 'blue')
    ax3.plot(T_2, mhrg_bdensity_color_clusters_2, '-.', c = 'red')
    ax3.plot(T_2, mhrg_bdensity_singlet_clusters_2, '-.', c = 'green')
    ax3.plot(T_2, mhrg2_bdensity_v_2, '-', c = 'blue')
    ax3.plot(T_2, mhrg2_bdensity_color_clusters_2, '-', c = 'red')
    ax3.plot(T_2, mhrg2_bdensity_singlet_clusters_2, '-', c = 'green')
    
    ax3.text(85.0, 0.93, r"$\mathrm{\mu_B/T=2.5}$", color="black", fontsize=14)

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_experimental_qgp_pressure():

    import math
    import numpy
    import pickle
    import platform
    import warnings

    import matplotlib.pyplot
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    import pnjl.thermo.solvers.\
        pnjl_lattice_cut_sea.\
        pl_lo.\
        pert_l_const_s_mass.\
        no_clusters \
    as solver

    warnings.filterwarnings("ignore")

    calc_1 = True

    files = "D:/EoS/epja/experimental/qgp_pressure/"
    old_files = "D:/EoS/epja/legacy/figure5/"
    lattice = "D:/EoS/epja/lattice_data_pickled/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/experimental/qgp_pressure/"
        old_files = "/home/mcierniak/Data/2023_epja/legacy/figure5/"
        lattice = "/home/mcierniak/Data/2023_epja/lattice_data_pickled/"

    T = numpy.linspace(1.0, 2000.0, num=200)

    muB = [0.0 for el in T]

    phi_re_v_1, phi_im_v_1 = list(), list()
    qgp_pressure_v_1, qgp_partial_pressure_v_1 = list(), list()

    def old(files):
        total, partial = list(), list()
        sigma_v_1, gluon_v_1 = list(), list()
        sea_u_v_1, sea_d_v_1, sea_s_v_1 = list(), list(), list()
        perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
            list(), list(), list()
        pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = list(), list(), list()
        with open(files+"sigma_v_1.pickle", "rb") as file:
            sigma_v_1 = pickle.load(file)
        with open(files+"gluon_v_1.pickle", "rb") as file:
            gluon_v_1 = pickle.load(file)
        with open(files+"sea_u_v_1.pickle", "rb") as file:
            sea_u_v_1 = pickle.load(file)
        with open(files+"sea_d_v_1.pickle", "rb") as file:
            sea_d_v_1 = pickle.load(file)
        with open(files+"sea_s_v_1.pickle", "rb") as file:
            sea_s_v_1 = pickle.load(file)
        with open(files+"perturbative_u_v_1.pickle", "rb") as file:
            perturbative_u_v_1 = pickle.load(file)
        with open(files+"perturbative_d_v_1.pickle", "rb") as file:
            perturbative_d_v_1 = pickle.load(file)
        with open(files+"perturbative_s_v_1.pickle", "rb") as file:
            perturbative_s_v_1 = pickle.load(file)
        with open(files+"pnjl_u_v_1.pickle", "rb") as file:
            pnjl_u_v_1 = pickle.load(file)
        with open(files+"pnjl_d_v_1.pickle", "rb") as file:
            pnjl_d_v_1 = pickle.load(file)
        with open(files+"pnjl_s_v_1.pickle", "rb") as file:
            pnjl_s_v_1 = pickle.load(file)
        for parts in zip(sigma_v_1, gluon_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                         perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                         pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1):
            (sigma_el, gluon_el, sea_u_el, sea_d_el, sea_s_el,
             pert_u_el, pert_d_el, pert_s_el, pnjl_u_el, 
             pnjl_d_el, pnjl_s_el) = parts
            partial_temp = (sigma_el, sea_u_el, sea_d_el, sea_s_el,
                            gluon_el, pnjl_u_el, pnjl_d_el, pnjl_s_el,
                            pert_u_el, pert_d_el, pert_s_el)
            total.append(math.fsum(partial_temp))
            partial.append(partial_temp)
        return total, partial

    if calc_1:
        sol = solver.pressure_all(T, muB, label="Traced Polyakov loop and pressure")
        phi_re_v_1 = sol[0]
        phi_im_v_1 = sol[1]
        qgp_pressure_v_1 = sol[2]
        qgp_partial_pressure_v_1 = sol[3]
        with open(files+"phi_re_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)
        with open(files+"qgp_pressure_v_0p0.pickle", "wb") as file:
            pickle.dump(qgp_pressure_v_1, file)
        with open(files+"qgp_partial_pressure_v_0p0.pickle", "wb") as file:
            pickle.dump(qgp_partial_pressure_v_1, file)
    else:
        with open(files+"phi_re_v_0p0.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_0p0.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
        with open(files+"qgp_pressure_v_0p0.pickle", "rb") as file:
            qgp_pressure_v_1 = pickle.load(file)
        with open(files+"qgp_partial_pressure_v_0p0.pickle", "rb") as file:
            qgp_partial_pressure_v_1 = pickle.load(file)

    with open(lattice + "bazavov_1407_6387_mu0.pickle", "rb") as file:
        bazavov_1407_6387_mu0 = pickle.load(file)
    with open(lattice + "borsanyi_1309_5258_mu0.pickle", "rb") as file:
        borsanyi_1309_5258_mu0 = pickle.load(file)
    with open(lattice + "bazavov_1710_05024_mu0.pickle", "rb") as file:
        bazavov_1710_05024_mu0 = pickle.load(file)
    with open(lattice + "borsanyi_1204_6710v2_mu0.pickle","rb") as file:
        borsanyi_1204_6710v2_mu0 = pickle.load(file)

    old_qgp_pressure_v_1, old_qgp_partial_pressure_v_1 = old(old_files)

    pertrubative_total = [sum([el[8], el[9], el[10]]) for el in qgp_partial_pressure_v_1]
    pnjl_total = [sum([el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7]]) for el in qgp_partial_pressure_v_1]
    quark_total = [sum([el[0], el[1], el[2], el[3], el[5], el[6], el[7]]) for el in qgp_partial_pressure_v_1]
    gluon_total = [el[4] for el in qgp_partial_pressure_v_1]

    old_pertrubative_total = [sum([el[8], el[9], el[10]]) for el in old_qgp_partial_pressure_v_1]
    old_pnjl_total = [sum([el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7]]) for el in old_qgp_partial_pressure_v_1]
    old_quark_total = [sum([el[0], el[1], el[2], el[3], el[5], el[6], el[7]]) for el in old_qgp_partial_pressure_v_1]
    old_gluon_total = [el[4] for el in old_qgp_partial_pressure_v_1]

    fig1 = matplotlib.pyplot.figure(num=1, figsize=(5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([10., 2000., -1.5, 5.1])

    ax2 = ax1.inset_axes([0.58,0.02,0.4,0.4])
    ax2.axis([50., 250., -0.5, 2.5])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax1.add_patch(
        matplotlib.patches.Polygon(
            bazavov_1407_6387_mu0, closed=True, fill=True,
            color='red', alpha=0.3
        )
    )
    ax1.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1309_5258_mu0, closed=True, fill=True,
            color='green', alpha=0.3
        )
    )
    ax1.add_patch(
        matplotlib.patches.Polygon(
            bazavov_1710_05024_mu0, closed=True, fill=True,
            color='magenta', alpha=0.3
        )
    )
    ax1.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1204_6710v2_mu0, closed=True, fill=True,
            color='blue', alpha=0.3
        )
    )

    ax2.add_patch(
        matplotlib.patches.Polygon(
            bazavov_1407_6387_mu0, closed=True, fill=True,
            color='red', alpha=0.3
        )
    )
    ax2.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1309_5258_mu0, closed=True, fill=True,
            color='green', alpha=0.3
        )
    )
    ax2.add_patch(
        matplotlib.patches.Polygon(
            bazavov_1710_05024_mu0, closed=True, fill=True,
            color='magenta', alpha=0.3
        )
    )
    ax2.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1204_6710v2_mu0, closed=True, fill=True,
            color='blue', alpha=0.3
        )
    )

    ax1.plot(T, qgp_pressure_v_1, '-', c='black')
    ax1.plot(T, gluon_total, '-', c='red')
    ax1.plot(T, pertrubative_total, '-', c='pink')
    ax1.plot(T, pnjl_total, '-', c='blue')

    ax2.plot(T, qgp_pressure_v_1, '-', c='black')
    ax2.plot(T, gluon_total, '-', c='red')
    ax2.plot(T, pertrubative_total, '-', c='pink')
    ax2.plot(T, pnjl_total, '-', c='blue')
    
    ax1.plot(T, old_qgp_pressure_v_1, ':', c='black')
    ax1.plot(T, old_gluon_total, ':', c='red')
    ax1.plot(T, old_pertrubative_total, ':', c='pink')
    ax1.plot(T, old_pnjl_total, ':', c='blue')

    ax2.plot(T, old_qgp_pressure_v_1, ':', c='black')
    ax2.plot(T, old_gluon_total, ':', c='red')
    ax2.plot(T, old_pertrubative_total, ':', c='pink')
    ax2.plot(T, old_pnjl_total, ':', c='blue')

    ax1.text(
        400.0, 1.6,
        r'Polyakov--loop potential',
        color='red', fontsize=14
    )
    ax1.text(
        220.0, -0.6,
        r'Perturbative correction',
        color='red', alpha=0.6, fontsize=14)
    ax1.text(
        1230.0, 3.5,
        r'PNJL quarks',
        color='blue', fontsize=14
    )
    ax1.text(
        1230.0, 4.3,
        r'total pressure',
        color='black', fontsize=14
    )
    ax1.text(
        45.0, 4.8,
        r'$\mathrm{\mu_B=0}$',
        color='black', fontsize=14
    )
    ax1.text(
        430.0, 3.1,
        r'Borsanyi et al. (2012)',
        color='blue', alpha=0.7, fontsize=14
    )
    ax1.text(
        580.0, 3.88,
        r'Borsanyi et al. (2014)',
        color='green', alpha=0.7, fontsize=14
    )
    ax1.text(
        345.0, 2.5,
        r'Bazavov et al. (2014)',
        color='red', alpha=0.7, fontsize=14
    )
    ax1.text(
        1000.0, 4.8,
        r'Bazavov et al. (2018)',
        color='magenta', alpha=0.7, fontsize=14
    )

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_experimental_pert():
    
    import tqdm
    import math
    import numpy
    import pickle
    import warnings
    import platform

    import matplotlib.pyplot

    import utils
    import pnjl.thermo.gcp_perturbative.l_const_s_mass as pert

    import pnjl.thermo.solvers.\
        pnjl_lattice_cut_sea.\
        pl_lo.\
        pert_l_const_s_mass.\
        no_clusters \
    as solver

    warnings.filterwarnings("ignore")
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    calc_1 = True
    calc_2 = True

    files = "D:/EoS/epja/experimental/pert/"
    lattice_files = "D:/EoS/epja/lattice_data_raw/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/experimental/pert/"
        lattice_files = "/home/mcierniak/Data/2023_epja/lattice_data_raw/"

    T_1 = numpy.linspace(1.0, 280.0, 200)
    T_2 = numpy.linspace(1.0, 280.0, 200)

    muB_1 = [0.0 for el in T_1]
    muB_2 = [2.5*el for el in T_2]

    phi_re_v_1, phi_im_v_1 = list(), list()
    phi_re_v_2, phi_im_v_2 = list(), list()

    qgp_sdensity_v_1, qgp_partial_sdensity_v_1 = list(), list()
    qgp_sdensity_v_2, qgp_partial_sdensity_v_2 = list(), list()
    qgp_bdensity_v_2, qgp_partial_bdensity_v_2 = list(), list()
    qgp_old_sdensity_v_1, qgp_old_partial_sdensity_v_1 = list(), list()
    qgp_old_sdensity_v_2, qgp_old_partial_sdensity_v_2 = list(), list()
    qgp_old_bdensity_v_2, qgp_old_partial_bdensity_v_2 = list(), list()

    def qgp_bdensity_old(filez):
        sigma_v_2, gluon_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2 = \
            list(), list(), list(), list(), list()
        perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2 = \
            list(), list(), list()
        perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2 = \
            list(), list(), list(), list()
        with open(filez+"b_old_sigma_v_2p5.pickle", "rb") as file:
            sigma_v_2 = pickle.load(file)
        with open(filez+"b_old_gluon_v_2p5.pickle", "rb") as file:
            gluon_v_2 = pickle.load(file)
        with open(filez+"b_old_sea_u_v_2p5.pickle", "rb") as file:
            sea_u_v_2 = pickle.load(file)
        with open(filez+"b_old_sea_d_v_2p5.pickle", "rb") as file:
            sea_d_v_2 = pickle.load(file)
        with open(filez+"b_old_sea_s_v_2p5.pickle", "rb") as file:
            sea_s_v_2 = pickle.load(file)
        with open(filez+"b_old_perturbative_u_v_2p5.pickle", "rb") as file:
            perturbative_u_v_2 = pickle.load(file)
        with open(filez+"b_old_perturbative_d_v_2p5.pickle", "rb") as file:
            perturbative_d_v_2 = pickle.load(file)
        with open(filez+"b_old_perturbative_s_v_2p5.pickle", "rb") as file:
            perturbative_s_v_2 = pickle.load(file)
        with open(filez+"b_old_perturbative_gluon_v_2p5.pickle", "rb") as file:
            perturbative_gluon_v_2 = pickle.load(file)
        with open(filez+"b_old_pnjl_u_v_2p5.pickle", "rb") as file:
            pnjl_u_v_2 = pickle.load(file)
        with open(filez+"b_old_pnjl_d_v_2p5.pickle", "rb") as file:
            pnjl_d_v_2 = pickle.load(file)
        with open(filez+"b_old_pnjl_s_v_2p5.pickle", "rb") as file:
            pnjl_s_v_2 = pickle.load(file)
        partial = list()
        total = list()
        for el in zip(
            sigma_v_2, gluon_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
            perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
            perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
        ):
            partial.append(el)
            total.append(math.fsum(el))
        return total, partial
    
    def qgp_sdensity_old_1(filez):
        sigma_v_1, gluon_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1 = \
            list(), list(), list(), list(), list()
        perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
            list(), list(), list()
        perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = \
            list(), list(), list(), list()
        with open(filez+"s_old_sigma_v_0p0.pickle", "rb") as file:
            sigma_v_1 = pickle.load(file)
        with open(filez+"s_old_gluon_v_0p0.pickle", "rb") as file:
            gluon_v_1 = pickle.load(file)
        with open(filez+"s_old_sea_u_v_0p0.pickle", "rb") as file:
            sea_u_v_1 = pickle.load(file)
        with open(filez+"s_old_sea_d_v_0p0.pickle", "rb") as file:
            sea_d_v_1 = pickle.load(file)
        with open(filez+"s_old_sea_s_v_0p0.pickle", "rb") as file:
            sea_s_v_1 = pickle.load(file)
        with open(filez+"s_old_perturbative_u_v_0p0.pickle", "rb") as file:
            perturbative_u_v_1 = pickle.load(file)
        with open(filez+"s_old_perturbative_d_v_0p0.pickle", "rb") as file:
            perturbative_d_v_1 = pickle.load(file)
        with open(filez+"s_old_perturbative_s_v_0p0.pickle", "rb") as file:
            perturbative_s_v_1 = pickle.load(file)
        with open(filez+"s_old_perturbative_gluon_v_0p0.pickle", "rb") as file:
            perturbative_gluon_v_1 = pickle.load(file)
        with open(filez+"s_old_pnjl_u_v_0p0.pickle", "rb") as file:
            pnjl_u_v_1 = pickle.load(file)
        with open(filez+"s_old_pnjl_d_v_0p0.pickle", "rb") as file:
            pnjl_d_v_1 = pickle.load(file)
        with open(filez+"s_old_pnjl_s_v_0p0.pickle", "rb") as file:
            pnjl_s_v_1 = pickle.load(file)
        partial = list()
        total = list()
        for el in zip(
            sigma_v_1, gluon_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
            perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
            perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
        ):
            partial.append(el)
            total.append(math.fsum(el))
        return total, partial
    
    def qgp_sdensity_old_2(filez):
        sigma_v_2, gluon_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2 = \
            list(), list(), list(), list(), list()
        perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2 = \
            list(), list(), list()
        perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2 = \
            list(), list(), list(), list()
        with open(filez+"s_old_sigma_v_2p5.pickle", "rb") as file:
            sigma_v_2 = pickle.load(file)
        with open(filez+"s_old_gluon_v_2p5.pickle", "rb") as file:
            gluon_v_2 = pickle.load(file)
        with open(filez+"s_old_sea_u_v_2p5.pickle", "rb") as file:
            sea_u_v_2 = pickle.load(file)
        with open(filez+"s_old_sea_d_v_2p5.pickle", "rb") as file:
            sea_d_v_2 = pickle.load(file)
        with open(filez+"s_old_sea_s_v_2p5.pickle", "rb") as file:
            sea_s_v_2 = pickle.load(file)
        with open(filez+"s_old_perturbative_u_v_2p5.pickle", "rb") as file:
            perturbative_u_v_2 = pickle.load(file)
        with open(filez+"s_old_perturbative_d_v_2p5.pickle", "rb") as file:
            perturbative_d_v_2 = pickle.load(file)
        with open(filez+"s_old_perturbative_s_v_2p5.pickle", "rb") as file:
            perturbative_s_v_2 = pickle.load(file)
        with open(filez+"s_old_perturbative_gluon_v_2p5.pickle", "rb") as file:
            perturbative_gluon_v_2 = pickle.load(file)
        with open(filez+"s_old_pnjl_u_v_2p5.pickle", "rb") as file:
            pnjl_u_v_2 = pickle.load(file)
        with open(filez+"s_old_pnjl_d_v_2p5.pickle", "rb") as file:
            pnjl_d_v_2 = pickle.load(file)
        with open(filez+"s_old_pnjl_s_v_2p5.pickle", "rb") as file:
            pnjl_s_v_2 = pickle.load(file)
        partial = list()
        total = list()
        for el in zip(
            sigma_v_2, gluon_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
            perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
            perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
        ):
            partial.append(el)
            total.append(math.fsum(el))
        return total, partial

    if calc_1:
        sol = solver.sdensity_all(T_1, muB_1, label="QGP PL and sdensity #1")
        phi_re_v_1 = sol[0]
        phi_im_v_1 = sol[1]
        qgp_sdensity_v_1 = sol[2]
        qgp_partial_sdensity_v_1 = sol[3]
        with open(files+"phi_re_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)
        with open(files+"qgp_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(qgp_sdensity_v_1, file)
        with open(files+"qgp_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(qgp_partial_sdensity_v_1, file)
    else:
        with open(files+"phi_re_v_0p0.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_0p0.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
        with open(files+"qgp_sdensity_v_0p0.pickle", "rb") as file:
            qgp_sdensity_v_1 = pickle.load(file)
        with open(files+"qgp_partial_sdensity_v_0p0.pickle", "rb") as file:
            qgp_partial_sdensity_v_1 = pickle.load(file)

    if calc_2:
        sol = solver.sdensity_all(T_2, muB_2, label="QGP PL and sdensity #2")
        phi_re_v_2 = sol[0]
        phi_im_v_2 = sol[1]
        qgp_sdensity_v_2 = sol[2]
        qgp_partial_sdensity_v_2 = sol[3]
        sol = solver.bdensity_all(T_2, muB_2, phi_re=phi_re_v_2, phi_im=phi_im_v_2, label="QGP bdensity #2")
        qgp_bdensity_v_2 = sol[0]
        qgp_partial_bdensity_v_2 = sol[1]
        with open(files+"phi_re_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)
        with open(files+"qgp_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_sdensity_v_2, file)
        with open(files+"qgp_partial_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_partial_sdensity_v_2, file)
        with open(files+"qgp_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_bdensity_v_2, file)
        with open(files+"qgp_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_partial_bdensity_v_2, file)
    else:
        with open(files+"phi_re_v_2p5.pickle", "rb") as file:
            phi_re_v_2 = pickle.load(file)
        with open(files+"phi_im_v_2p5.pickle", "rb") as file:
            phi_im_v_2 = pickle.load(file)
        with open(files+"qgp_sdensity_v_2p5.pickle", "rb") as file:
            qgp_sdensity_v_2 = pickle.load(file)
        with open(files+"qgp_partial_sdensity_v_2p5.pickle", "rb") as file:
            qgp_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"qgp_bdensity_v_2p5.pickle", "rb") as file:
            qgp_bdensity_v_2 = pickle.load(file)
        with open(files+"qgp_partial_bdensity_v_2p5.pickle", "rb") as file:
            qgp_partial_bdensity_v_2 = pickle.load(file)

    qgp_old_bdensity_v_2, qgp_old_partial_bdensity_v_2 = qgp_bdensity_old(files)
    qgp_old_sdensity_v_1, qgp_old_partial_sdensity_v_1 = qgp_sdensity_old_1(files)
    qgp_old_sdensity_v_2, qgp_old_partial_sdensity_v_2 = qgp_sdensity_old_2(files)

    t_pmod_sdensity_1 = [
        (2.0*pert.sdensity(T_el, muB_el/3.0, phi_re_el, phi_im_el, 'l') + pert.sdensity(T_el, muB_el/3.0, phi_re_el, phi_im_el, 's'))/(T_el**3)
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(
            T_1, muB_1, phi_re_v_1, phi_im_v_1
        ), total=len(T_2), ncols=100)
    ]
    t_pmod_sdensity_2 = [
        (2.0*pert.sdensity(T_el, muB_el/3.0, phi_re_el, phi_im_el, 'l') + pert.sdensity(T_el, muB_el/3.0, phi_re_el, phi_im_el, 's'))/(T_el**3)
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(
            T_2, muB_2, phi_re_v_2, phi_im_v_2
        ), total=len(T_2), ncols=100)
    ]
    t_pmod_bdensity_2 = [
        (2.0*pert.bdensity(T_el, muB_el/3.0, phi_re_el, phi_im_el, 'l') + pert.bdensity(T_el, muB_el/3.0, phi_re_el, phi_im_el, 's'))/(T_el**3)
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(zip(
            T_2, muB_2, phi_re_v_2, phi_im_v_2
        ), total=len(T_2), ncols=100)
    ]

    t_pnjl_bdensity_2 = [
        math.fsum([el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7]]) 
        for el in qgp_partial_bdensity_v_2
    ]
    t_pert_bdensity_2 = [
        math.fsum([el[8], el[9], el[10]]) 
        for el in qgp_partial_bdensity_v_2
    ]
    t_pert_ud_bdensity_2 = [
        math.fsum([el[8], el[9]]) 
        for el in qgp_partial_bdensity_v_2
    ]
    t_gluon_bdensity_2 = [
        el[4]
        for el in qgp_partial_bdensity_v_2
    ]
    t_quark_sdensity_1 = [
        math.fsum([el[0], el[1], el[2], el[3], el[5], el[6], el[7]]) 
        for el in qgp_partial_sdensity_v_1
    ]
    t_pnjl_sdensity_1 = [
        math.fsum([el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7]]) 
        for el in qgp_partial_sdensity_v_1
    ]
    t_pert_sdensity_1 = [
        math.fsum([el[8], el[9], el[10]])
        for el in qgp_partial_sdensity_v_1
    ]
    t_pert_ud_sdensity_1 = [
        math.fsum([el[8], el[9]])
        for el in qgp_partial_sdensity_v_1
    ]
    t_gluon_sdensity_1 = [
        el[4]
        for el in qgp_partial_sdensity_v_1
    ]
    t_quark_sdensity_2 = [
        math.fsum([el[0], el[1], el[2], el[3], el[5], el[6], el[7]]) 
        for el in qgp_partial_sdensity_v_2
    ]
    t_pnjl_sdensity_2 = [
        math.fsum([el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7]]) 
        for el in qgp_partial_sdensity_v_2
    ]
    t_pert_sdensity_2 = [
        math.fsum([el[8], el[9], el[10]]) 
        for el in qgp_partial_sdensity_v_2
    ]
    t_pert_ud_sdensity_2 = [
        math.fsum([el[8], el[9]]) 
        for el in qgp_partial_sdensity_v_2
    ]
    t_gluon_sdensity_2 = [
        el[4]
        for el in qgp_partial_sdensity_v_2
    ]

    t_pnjl_bdensity_old_2 = [
        math.fsum([el[0], el[1], el[2], el[3], el[4], el[9], el[10], el[11]]) 
        for el in qgp_old_partial_bdensity_v_2
    ]
    t_pert_bdensity_old_2 = [
        math.fsum([el[5], el[6], el[7], el[8]]) 
        for el in qgp_old_partial_bdensity_v_2
    ]
    t_quark_sdensity_old_1 = [
        math.fsum([el[0], el[2], el[3], el[4], el[9], el[10], el[11]]) 
        for el in qgp_old_partial_sdensity_v_1
    ]
    t_pnjl_sdensity_old_1 = [
        math.fsum([el[0], el[1], el[2], el[3], el[4], el[9], el[10], el[11]]) 
        for el in qgp_old_partial_sdensity_v_1
    ]
    t_pert_sdensity_old_1 = [
        math.fsum([el[5], el[6], el[7], el[8]]) 
        for el in qgp_old_partial_sdensity_v_1
    ]
    t_gluon_sdensity_old_1 = [
        el[1]
        for el in qgp_old_partial_sdensity_v_1
    ]
    t_quark_sdensity_old_2 = [
        math.fsum([el[0], el[2], el[3], el[4], el[9], el[10], el[11]]) 
        for el in qgp_old_partial_sdensity_v_2
    ]
    t_pnjl_sdensity_old_2 = [
        math.fsum([el[0], el[1], el[2], el[3], el[4], el[9], el[10], el[11]]) 
        for el in qgp_old_partial_sdensity_v_2
    ]
    t_pert_sdensity_old_2 = [
        math.fsum([el[5], el[6], el[7], el[8]]) 
        for el in qgp_old_partial_sdensity_v_2
    ]
    t_gluon_sdensity_old_2 = [
        el[1]
        for el in qgp_old_partial_sdensity_v_2
    ]

    lQCD_sdensity_1_x, lQCD_sdensity_1_y = \
        utils.data_load(
            lattice_files+"2212_09043_fig13_top_right_0p0_alt2.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_sdensity_1 = [[x, y] for x, y in zip(lQCD_sdensity_1_x, lQCD_sdensity_1_y)]

    lQCD_sdensity_2_x, lQCD_sdensity_2_y = \
        utils.data_load(
            lattice_files+"2212_09043_fig13_top_right_2p5.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_sdensity_2 = [[x, y] for x, y in zip(lQCD_sdensity_2_x, lQCD_sdensity_2_y)]

    lQCD_bdensity_2_x, lQCD_bdensity_2_y = \
        utils.data_load(
            lattice_files+"2202_09184v2_fig2_mub_T_2p5_nb.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_bdensity_2 = [[x, y] for x, y in zip(lQCD_bdensity_2_x, lQCD_bdensity_2_y)]

    def g2_1loop(T: float, kT: float, LMS: float):
        beta0 = (11.0 - 2.0)/(4.0*math.pi)
        aux_log = (kT*math.pi*T)/LMS
        return 2*math.pi/(beta0*math.log(aux_log))

    def g2(T: float, kT: float, LMS: float):
        try:
            beta0 = (11.0 - 2.0)/(4.0*math.pi)
            beta1 = (102 - 38)/(16*(math.pi**2))
            b10 = beta1/beta0
            aux_log = (kT*math.pi*T)/LMS
            log2 = 2.0*math.log(aux_log)
            return g2_1loop(T, kT, LMS)*(1.0 - b10*(math.log(log2)/log2))
        except ValueError:
            return float('nan')

    def dg2dT(T: float, kT: float, LMS: float):
        try:
            aux_log = (kT*math.pi*T)/LMS
            log = math.log(aux_log)
            return -8.0*math.pi*(8.0+9.0*math.pi*log-16.0*math.log(2.0*log))/(81.0*(log**3))
        except ValueError:
            return float('nan')

    def p_id(T: float, mu: float):
        muT = mu/T
        zero = (23.75*math.pi**2)/45.0
        mu1 = 0.5*(muT**2)
        piterm = 1.0/(4.0*(math.pi**2))
        mu2 = muT**4
        return zero + 3.0*(mu1 + piterm*mu2)

    def n_id(T: float, mu: float):
        muT = mu/T
        return muT + (1.0/(math.pi**2))*(muT**3)

    def s_id(T: float, mu: float):
        muT = mu/T
        return -3.0*((muT**2) + (muT**4)/(math.pi**2))

    def p_2(T: float, mu: float):
        muT = mu/T
        zero = (1.0 + ((5.0*3.0)/12.0))/6.0
        piterm = 1.0/(4.0*(math.pi**2))
        mu1 = 0.5*(muT**2)
        mu2 = muT**4
        return zero + (2.0*3.0*piterm)*(mu1 + piterm*mu2)

    def n_2(T: float, mu: float):
        return n_id(T, mu)/(2.0*(math.pi**2))

    def s_2(T: float, mu: float):
        return s_id(T, mu)/(2.0*(math.pi**2))

    def n_pert_og2(T: float, mu: float, kT: float, LMS: float):
        return n_id(T, mu) - g2(T, kT, LMS)*n_2(T, mu)

    def s_pert_og2(T: float, mu: float, kT: float, LMS: float):
        return 4.0*p_id(T, mu) + s_id(T, mu) - 4.0*g2(T, kT, LMS) - p_2(T, mu)*dg2dT(T, kT, LMS) - g2(T, kT, LMS)*s_2(T, mu)

    n_pert_og2_v_2_low = [n_pert_og2(T_el, mu_el/3.0, 4.0, 351.0) for T_el, mu_el in zip(T_2, muB_2)]
    n_pert_og2_v_2_high = [n_pert_og2(T_el, mu_el/3.0, 8.0, 327.0) for T_el, mu_el in zip(T_2, muB_2)]

    s_pert_og2_v_1_low = [s_pert_og2(T_el, mu_el/3.0, 4.0, 351.0) for T_el, mu_el in zip(T_1, muB_1)]
    s_pert_og2_v_1_high = [s_pert_og2(T_el, mu_el/3.0, 8.0, 327.0) for T_el, mu_el in zip(T_1, muB_1)]
    s_pert_og2_v_2_low = [s_pert_og2(T_el, mu_el/3.0, 4.0, 351.0) for T_el, mu_el in zip(T_2, muB_2)]
    s_pert_og2_v_2_high = [s_pert_og2(T_el, mu_el/3.0, 8.0, 327.0) for T_el, mu_el in zip(T_2, muB_2)]

    pQCD_bdensity_2 = [[x, y] for x, y in zip(T_2, n_pert_og2_v_2_high)]
    for x, y in zip(T_2[::-1], n_pert_og2_v_2_low[::-1]):
        pQCD_bdensity_2.append([x, y])
    pQCD_sdensity_1 = [[x, y] for x, y in zip(T_1, s_pert_og2_v_1_high)]
    for x, y in zip(T_1[::-1], s_pert_og2_v_1_low[::-1]):
        pQCD_sdensity_1.append([x, y])
    pQCD_sdensity_2 = [[x, y] for x, y in zip(T_2, s_pert_og2_v_2_high)]
    for x, y in zip(T_2[::-1], s_pert_og2_v_2_low[::-1]):
        pQCD_sdensity_2.append([x, y])

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (18.0, 5.0))

    fig1.subplots_adjust(
        left=0.167, bottom=0.11, right=0.988, top=0.979, wspace=0.2, hspace=0.2
    )

    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.axis([80., 280., -6.0, 20.0])

    ax1.add_patch(
        matplotlib.patches.Polygon(
            lQCD_sdensity_1, 
            closed = True, fill = True, color = 'green', alpha = 0.3
        )
    )
    ax1.add_patch(
        matplotlib.patches.Polygon(
            pQCD_sdensity_1, 
            closed = True, fill = True, color = 'red', alpha = 0.3
        )
    )

    ax1.plot(T_1, qgp_sdensity_v_1, '-', c = 'black')
    ax1.plot(T_1, t_pnjl_sdensity_1, '-', c = 'blue')
    ax1.plot(T_1, t_pert_sdensity_1, '-', c = 'magenta')
    ax1.plot(T_1, t_pert_ud_sdensity_1, '--', c = 'magenta')
    ax1.plot(T_1, t_gluon_sdensity_1, '-', c = 'red')
    ax1.plot(T_1, t_quark_sdensity_1, '-', c = 'purple')
    ax1.plot(T_1, t_pmod_sdensity_1, ':', c = 'magenta')
    # ax1.plot(T_1, [el[0] for el in qgp_partial_sdensity_v_1], ':', c = 'cyan')
    # ax1.plot(T_1, [sum([el[1], el[2], el[3]]) for el in qgp_partial_sdensity_v_1], ':', c = 'orange')

    ax1.text(85, 18.5, r"$\mathrm{\mu_B/T=0}$", color="black", fontsize=14)
    ax1.text(250, 15.5, r"QGP", color="black", fontsize=14)
    ax1.text(190, 17.5, r"PNJL", color="blue", fontsize=14)
    ax1.text(188, -3.5, r"Perturbative correction", color="magenta", fontsize=14)
    ax1.text(220, 5.5, r"Polyakov-loop", color="red", fontsize=14)
    ax1.text(170, 12.5, r"Quarks", color="purple", fontsize=14)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.axis([80., 280., -6.0, 20.0])

    ax2.add_patch(
        matplotlib.patches.Polygon(
            lQCD_sdensity_2, 
            closed = True, fill = True, color = 'green', alpha = 0.3
        )
    )
    ax2.add_patch(
        matplotlib.patches.Polygon(
            pQCD_sdensity_2, 
            closed = True, fill = True, color = 'red', alpha = 0.3
        )
    )

    ax2.plot(T_2, qgp_sdensity_v_2, '-', c = 'black')
    ax2.plot(T_2, t_pnjl_sdensity_2, '-', c = 'blue')
    ax2.plot(T_2, t_pert_sdensity_2, '-', c = 'magenta')
    ax2.plot(T_2, t_pert_ud_sdensity_2, '--', c = 'magenta')
    ax2.plot(T_2, t_gluon_sdensity_2, '-', c = 'red')
    ax2.plot(T_2, t_quark_sdensity_2, '-', c = 'purple')
    ax2.plot(T_2, t_pmod_sdensity_2, ':', c = 'magenta')
    # ax2.plot(T_2, [el[0] for el in qgp_partial_sdensity_v_2], ':', c = 'cyan')
    # ax2.plot(T_2, [sum([el[1], el[2], el[3]]) for el in qgp_partial_sdensity_v_2], ':', c = 'orange')

    ax2.text(85, 18.5, r"$\mathrm{\mu_B/T=2.5}$", color="black", fontsize=14)
    ax2.text(250, 17.3, r"QGP", color="black", fontsize=14)
    ax2.text(165.5, 18.5, r"PNJL", color="blue", fontsize=14)
    ax2.text(188, -3.5, r"Perturbative correction", color="magenta", fontsize=14)
    ax2.text(220, 5.5, r"Polyakov-loop", color="red", fontsize=14)
    ax2.text(170, 14.5, r"Quarks", color="purple", fontsize=14)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    ax3 = fig1.add_subplot(1, 3, 3)
    ax3.axis([80., 280., -0.4, 1.2])

    ax3.add_patch(
        matplotlib.patches.Polygon(
            lQCD_bdensity_2, closed = True, fill = True, color = "green", alpha = 0.3
        )
    )
    ax3.add_patch(
        matplotlib.patches.Polygon(
            pQCD_bdensity_2, closed = True, fill = True, color = "red", alpha = 0.3
        )
    )

    ax3.plot(T_2, qgp_bdensity_v_2, '-', c = 'black')
    ax3.plot(T_2, t_pnjl_bdensity_2, '-', c = 'blue')
    ax3.plot(T_2, t_pert_bdensity_2, '-', c = 'magenta')
    ax3.plot(T_2, t_gluon_bdensity_2, '-', c = 'red')
    ax3.plot(T_2, t_pert_ud_bdensity_2, '--', c = 'magenta')
    ax3.plot(T_2, t_pmod_bdensity_2, ':', c = 'magenta')
    # ax3.plot(T_2, [el[0] for el in qgp_partial_bdensity_v_2], ':', c = 'cyan')
    # ax3.plot(T_2, [sum([el[1], el[2], el[3]]) for el in qgp_partial_bdensity_v_2], ':', c = 'orange')

    ax3.text(85, 1.1, r"$\mathrm{\mu_B/T=2.5}$", color="black", fontsize=14)
    ax3.text(188, -0.18, r"Perturbative correction", color="magenta", fontsize=14)
    ax3.text(250, 0.95, r"PNJL", color="blue", fontsize=14)
    ax3.text(200, 0.67, r"QGP", color="black", fontsize=14)

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_experimental_full():
    
    import tqdm
    import math
    import numpy
    import pickle
    import warnings
    import platform

    import matplotlib.pyplot

    import utils
    import pnjl.thermo.gcp_cluster.hrg \
        as cluster
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_mhrg
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_mhrg2
    import pnjl.thermo.solvers.\
        pnjl_lattice_cut_sea.\
        pl_lo.\
        pert_const.\
        no_clusters \
    as solver

    warnings.filterwarnings("ignore")
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    calc_1 = False
    calc_2 = False

    files = "D:/EoS/epja/experimental/full/"
    lattice_files = "D:/EoS/epja/lattice_data_raw/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/experimental/full/"
        lattice_files = "/home/mcierniak/Data/2023_epja/lattice_data_raw/"

    T_1 = numpy.linspace(1.0, 280.0, num=200)
    T_2 = numpy.linspace(1.0, 280.0, num=200)

    muB_1 = [0.0 for el in T_1]
    muB_2 = [2.5*el for el in T_2]

    reduced_spectrum = (
        'pi0', 'pi', 'K', 'K0', 'eta', 'rho', 'omega', 'K*(892)', 'K*0(892)',
        'p', 'n', 'etaPrime', 'a0', 'f0', 'phi', 'Lambda', 'h1', 'Sigma+',
        'Sigma0', 'Sigma-', 'b1', 'a1', 'Delta'
    )

    phi_re_v_1, phi_im_v_1 = list(), list()
    phi_re_v_2, phi_im_v_2 = list(), list()

    qgp_sdensity_v_1, qgp_partial_sdensity_v_1 = list(), list()
    hrg_full_sdensity_v_1, hrg_full_partial_sdensity_v_1 = list(), list()
    hrg2_full_sdensity_v_1, hrg2_full_partial_sdensity_v_1 = list(), list()
    mhrg_sdensity_v_1, mhrg_partial_sdensity_v_1 = list(), list()
    mhrg2_sdensity_v_1, mhrg2_partial_sdensity_v_1 = list(), list()

    qgp_sdensity_v_2, qgp_partial_sdensity_v_2 = list(), list()
    hrg_full_sdensity_v_2, hrg_full_partial_sdensity_v_2 = list(), list()
    hrg2_full_sdensity_v_2, hrg2_full_partial_sdensity_v_2 = list(), list()
    mhrg_sdensity_v_2, mhrg_partial_sdensity_v_2 = list(), list()
    mhrg2_sdensity_v_2, mhrg2_partial_sdensity_v_2 = list(), list()

    qgp_bdensity_v_2, qgp_partial_bdensity_v_2 = list(), list()
    hrg_full_bdensity_v_2, hrg_full_partial_bdensity_v_2 = list(), list()
    hrg2_full_bdensity_v_2, hrg2_full_partial_bdensity_v_2 = list(), list()
    mhrg_bdensity_v_2, mhrg_partial_bdensity_v_2 = list(), list()
    mhrg2_bdensity_v_2, mhrg2_partial_bdensity_v_2 = list(), list()

    #QGP #1
    if False:
        sol = solver.sdensity_all(T_1, muB_1, label="QGP PL and sdensity #1")
        phi_re_v_1 = sol[0]
        phi_im_v_1 = sol[1]
        qgp_sdensity_v_1 = sol[2]
        qgp_partial_sdensity_v_1 = sol[3]
        with open(files+"phi_re_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)
        with open(files+"qgp_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(qgp_sdensity_v_1, file)
        with open(files+"qgp_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(qgp_partial_sdensity_v_1, file)
    else:
        with open(files+"phi_re_v_0p0.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_0p0.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
        with open(files+"qgp_sdensity_v_0p0.pickle", "rb") as file:
            qgp_sdensity_v_1 = pickle.load(file)
        with open(files+"qgp_partial_sdensity_v_0p0.pickle", "rb") as file:
            qgp_partial_sdensity_v_1 = pickle.load(file)

    #HRG #1
    if calc_1:
        print("Full HRG sdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.sdensity_multi(T_el, muB_el)
            hrg_full_sdensity_v_1.append(temp_hrg/(T_el**3))
            hrg_full_partial_sdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_full_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg_full_sdensity_v_1, file)
        with open(files+"hrg_full_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg_full_partial_sdensity_v_1, file)
    else:
        with open(files+"hrg_full_sdensity_v_0p0.pickle", "rb") as file:
            hrg_full_sdensity_v_1 = pickle.load(file)
        with open(files+"hrg_full_partial_sdensity_v_0p0.pickle", "rb") as file:
            hrg_full_partial_sdensity_v_1 = pickle.load(file)

    #HRG reduced #1
    if calc_1:
        print("Reduced HRG sdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.sdensity_multi(T_el, muB_el, hadrons=reduced_spectrum)
            hrg2_full_sdensity_v_1.append(temp_hrg/(T_el**3))
            hrg2_full_partial_sdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg2_full_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg2_full_sdensity_v_1, file)
        with open(files+"hrg2_full_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(hrg2_full_partial_sdensity_v_1, file)
    else:
        with open(files+"hrg2_full_sdensity_v_0p0.pickle", "rb") as file:
            hrg2_full_sdensity_v_1 = pickle.load(file)
        with open(files+"hrg2_full_partial_sdensity_v_0p0.pickle", "rb") as file:
            hrg2_full_partial_sdensity_v_1 = pickle.load(file)

    #MHRG #1
    if False:
        print("MHRG sdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg.sdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg_sdensity_v_1.append(temp_hrg/(T_el**3))
            mhrg_partial_sdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg_sdensity_v_1, file)
        with open(files+"mhrg_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg_partial_sdensity_v_1, file)
    else:
        with open(files+"mhrg_sdensity_v_0p0.pickle", "rb") as file:
            mhrg_sdensity_v_1 = pickle.load(file)
        with open(files+"mhrg_partial_sdensity_v_0p0.pickle", "rb") as file:
            mhrg_partial_sdensity_v_1 = pickle.load(file)

    #MHRG (continuum) #1
    if calc_1:
        print("MHRG (continuum) sdensity #1")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, muB_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg2.sdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg2_sdensity_v_1.append(temp_hrg/(T_el**3))
            mhrg2_partial_sdensity_v_1.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg2_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg2_sdensity_v_1, file)
        with open(files+"mhrg2_partial_sdensity_v_0p0.pickle", "wb") as file:
            pickle.dump(mhrg2_partial_sdensity_v_1, file)
    else:
        with open(files+"mhrg2_sdensity_v_0p0.pickle", "rb") as file:
            mhrg2_sdensity_v_1 = pickle.load(file)
        with open(files+"mhrg2_partial_sdensity_v_0p0.pickle", "rb") as file:
            mhrg2_partial_sdensity_v_1 = pickle.load(file)

    #QGP #2
    if False:
        sol = solver.sdensity_all(T_2, muB_2, label="QGP PL and sdensity #2")
        phi_re_v_2 = sol[0]
        phi_im_v_2 = sol[1]
        qgp_sdensity_v_2 = sol[2]
        qgp_partial_sdensity_v_2 = sol[3]
        sol = solver.bdensity_all(T_2, muB_2, phi_re=phi_re_v_2, phi_im=phi_im_v_2, label="QGP bdensity #2")
        qgp_bdensity_v_2 = sol[0]
        qgp_partial_bdensity_v_2 = sol[1]
        with open(files+"phi_re_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)
        with open(files+"qgp_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_sdensity_v_2, file)
        with open(files+"qgp_partial_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_partial_sdensity_v_2, file)
        with open(files+"qgp_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_bdensity_v_2, file)
        with open(files+"qgp_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(qgp_partial_bdensity_v_2, file)
    else:
        with open(files+"phi_re_v_2p5.pickle", "rb") as file:
            phi_re_v_2 = pickle.load(file)
        with open(files+"phi_im_v_2p5.pickle", "rb") as file:
            phi_im_v_2 = pickle.load(file)
        with open(files+"qgp_sdensity_v_2p5.pickle", "rb") as file:
            qgp_sdensity_v_2 = pickle.load(file)
        with open(files+"qgp_partial_sdensity_v_2p5.pickle", "rb") as file:
            qgp_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"qgp_bdensity_v_2p5.pickle", "rb") as file:
            qgp_bdensity_v_2 = pickle.load(file)
        with open(files+"qgp_partial_bdensity_v_2p5.pickle", "rb") as file:
            qgp_partial_bdensity_v_2 = pickle.load(file)

    #HRG #2
    if calc_2:
        print("Full HRG sdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.sdensity_multi(T_el, muB_el)
            hrg_full_sdensity_v_2.append(temp_hrg/(T_el**3))
            hrg_full_partial_sdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_full_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_full_sdensity_v_2, file)
        with open(files+"hrg_full_partial_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_full_partial_sdensity_v_2, file)
        print("Full HRG bdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.bdensity_multi(T_el, muB_el)
            hrg_full_bdensity_v_2.append(temp_hrg/(T_el**3))
            hrg_full_partial_bdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg_full_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_full_bdensity_v_2, file)
        with open(files+"hrg_full_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg_full_partial_bdensity_v_2, file)
    else:
        with open(files+"hrg_full_sdensity_v_2p5.pickle", "rb") as file:
            hrg_full_sdensity_v_2 = pickle.load(file)
        with open(files+"hrg_full_partial_sdensity_v_2p5.pickle", "rb") as file:
            hrg_full_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"hrg_full_bdensity_v_2p5.pickle", "rb") as file:
            hrg_full_bdensity_v_2 = pickle.load(file)
        with open(files+"hrg_full_partial_bdensity_v_2p5.pickle", "rb") as file:
            hrg_full_partial_bdensity_v_2 = pickle.load(file)

    #HRG reduced #2
    if calc_2:
        print("Reduced HRG sdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.sdensity_multi(T_el, muB_el, hadrons=reduced_spectrum)
            hrg2_full_sdensity_v_2.append(temp_hrg/(T_el**3))
            hrg2_full_partial_sdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg2_full_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg2_full_sdensity_v_2, file)
        with open(files+"hrg2_full_partial_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg2_full_partial_sdensity_v_2, file)
        print("Reduced HRG bdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster.bdensity_multi(T_el, muB_el, hadrons=reduced_spectrum)
            hrg2_full_bdensity_v_2.append(temp_hrg/(T_el**3))
            hrg2_full_partial_bdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"hrg2_full_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg2_full_bdensity_v_2, file)
        with open(files+"hrg2_full_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(hrg2_full_partial_bdensity_v_2, file)
    else:
        with open(files+"hrg2_full_sdensity_v_2p5.pickle", "rb") as file:
            hrg2_full_sdensity_v_2 = pickle.load(file)
        with open(files+"hrg2_full_partial_sdensity_v_2p5.pickle", "rb") as file:
            hrg2_full_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"hrg2_full_bdensity_v_2p5.pickle", "rb") as file:
            hrg2_full_bdensity_v_2 = pickle.load(file)
        with open(files+"hrg2_full_partial_bdensity_v_2p5.pickle", "rb") as file:
            hrg2_full_partial_bdensity_v_2 = pickle.load(file)

    #MHRG #2
    if False:
        print("MHRG sdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg.sdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg_sdensity_v_2.append(temp_hrg/(T_el**3))
            mhrg_partial_sdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg_sdensity_v_2, file)
        with open(files+"mhrg_partial_sdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg_partial_sdensity_v_2, file)
        print("MHRG bdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            temp_hrg, temp_hrg_partials = cluster_mhrg.bdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg_bdensity_v_2.append(temp_hrg/(T_el**3))
            mhrg_partial_bdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg_bdensity_v_2, file)
        with open(files+"mhrg_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg_partial_bdensity_v_2, file)
    else:
        with open(files+"mhrg_sdensity_v_2p5.pickle", "rb") as file:
            mhrg_sdensity_v_2 = pickle.load(file)
        with open(files+"mhrg_partial_sdensity_v_2p5.pickle", "rb") as file:
            mhrg_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"mhrg_bdensity_v_2p5.pickle", "rb") as file:
            mhrg_bdensity_v_2 = pickle.load(file)
        with open(files+"mhrg_partial_bdensity_v_2p5.pickle", "rb") as file:
            mhrg_partial_bdensity_v_2 = pickle.load(file)

    #MHRG (continuum) #2
    if calc_2:
        # print("MHRG (continuum) sdensity #2")
        # for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
        #     zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        # ):
        #     temp_hrg, temp_hrg_partials = cluster_mhrg2.sdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
        #     mhrg2_sdensity_v_2.append(temp_hrg/(T_el**3))
        #     mhrg2_partial_sdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        # with open(files+"mhrg2_sdensity_v_2p5.pickle", "wb") as file:
        #     pickle.dump(mhrg2_sdensity_v_2, file)
        # with open(files+"mhrg2_partial_sdensity_v_2p5.pickle", "wb") as file:
        #     pickle.dump(mhrg2_partial_sdensity_v_2, file)
        with open(files+"mhrg2_sdensity_v_2p5.pickle", "rb") as file:
            mhrg2_sdensity_v_2 = pickle.load(file)
        with open(files+"mhrg2_partial_sdensity_v_2p5.pickle", "rb") as file:
            mhrg2_partial_sdensity_v_2 = pickle.load(file)
        print("MHRG (continuum) bdensity #2")
        for T_el, muB_el, phi_re_el, phi_im_el in zip(T_2, muB_2, phi_re_v_2, phi_im_v_2):
        # for T_el, muB_el, phi_re_el, phi_im_el in tqdm.tqdm(
        #     zip(T_2, muB_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        # ):
            temp_hrg, temp_hrg_partials = cluster_mhrg2.bdensity_multi(T_el, muB_el, phi_re_el, phi_im_el)
            mhrg2_bdensity_v_2.append(temp_hrg/(T_el**3))
            mhrg2_partial_bdensity_v_2.append(tuple([el/(T_el**3) for el in temp_hrg_partials]))
        with open(files+"mhrg2_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg2_bdensity_v_2, file)
        with open(files+"mhrg2_partial_bdensity_v_2p5.pickle", "wb") as file:
            pickle.dump(mhrg2_partial_bdensity_v_2, file)
    else:
        with open(files+"mhrg2_sdensity_v_2p5.pickle", "rb") as file:
            mhrg2_sdensity_v_2 = pickle.load(file)
        with open(files+"mhrg2_partial_sdensity_v_2p5.pickle", "rb") as file:
            mhrg2_partial_sdensity_v_2 = pickle.load(file)
        with open(files+"mhrg2_bdensity_v_2p5.pickle", "rb") as file:
            mhrg2_bdensity_v_2 = pickle.load(file)
        with open(files+"mhrg2_partial_bdensity_v_2p5.pickle", "rb") as file:
            mhrg2_partial_bdensity_v_2 = pickle.load(file)

    t_pnjl_bdensity_2 = [
        math.fsum([el[0], el[1], el[2], el[3], el[4], el[8], el[9], el[10]]) 
        for el in qgp_partial_bdensity_v_2
    ]
    t_pert_bdensity_2 = [
        math.fsum([el[5], el[6], el[7]]) 
        for el in qgp_partial_bdensity_v_2
    ]
    t_quark_sdensity_1 = [
        math.fsum([el[0], el[2], el[3], el[4], el[8], el[9], el[10]]) 
        for el in qgp_partial_sdensity_v_1
    ]
    t_pnjl_sdensity_1 = [
        math.fsum([el[0], el[1], el[2], el[3], el[4], el[8], el[9], el[10]]) 
        for el in qgp_partial_sdensity_v_1
    ]
    t_pert_sdensity_1 = [
        math.fsum([el[5], el[6], el[7]]) 
        for el in qgp_partial_sdensity_v_1
    ]
    t_gluon_sdensity_1 = [
        el[1]
        for el in qgp_partial_sdensity_v_1
    ]
    t_quark_sdensity_2 = [
        math.fsum([el[0], el[2], el[3], el[4], el[8], el[9], el[10]]) 
        for el in qgp_partial_sdensity_v_2
    ]
    t_pnjl_sdensity_2 = [
        math.fsum([el[0], el[1], el[2], el[3], el[4], el[8], el[9], el[10]]) 
        for el in qgp_partial_sdensity_v_2
    ]
    t_pert_sdensity_2 = [
        math.fsum([el[5], el[6], el[7]]) 
        for el in qgp_partial_sdensity_v_2
    ]
    t_gluon_sdensity_2 = [
        el[1]
        for el in qgp_partial_sdensity_v_2
    ]

    lQCD_sdensity_1_x, lQCD_sdensity_1_y = \
        utils.data_load(
            lattice_files+"2212_09043_fig13_top_right_0p0_alt2.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_sdensity_1 = [[x, y] for x, y in zip(lQCD_sdensity_1_x, lQCD_sdensity_1_y)]

    lQCD_sdensity_2_x, lQCD_sdensity_2_y = \
        utils.data_load(
            lattice_files+"2212_09043_fig13_top_right_2p5.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_sdensity_2 = [[x, y] for x, y in zip(lQCD_sdensity_2_x, lQCD_sdensity_2_y)]

    lQCD_bdensity_2_x, lQCD_bdensity_2_y = \
        utils.data_load(
            lattice_files+"2202_09184v2_fig2_mub_T_2p5_nb.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_bdensity_2 = [[x, y] for x, y in zip(lQCD_bdensity_2_x, lQCD_bdensity_2_y)]

    def g2_1loop(T: float, kT: float, LMS: float):
        beta0 = (11.0 - 2.0)/(4.0*math.pi)
        aux_log = (kT*math.pi*T)/LMS
        return 2*math.pi/(beta0*math.log(aux_log))

    def g2(T: float, kT: float, LMS: float):
        try:
            beta0 = (11.0 - 2.0)/(4.0*math.pi)
            beta1 = (102 - 38)/(16*(math.pi**2))
            b10 = beta1/beta0
            aux_log = (kT*math.pi*T)/LMS
            log2 = 2.0*math.log(aux_log)
            return g2_1loop(T, kT, LMS)*(1.0 - b10*(math.log(log2)/log2))
        except ValueError:
            return float('nan')

    def dg2dT(T: float, kT: float, LMS: float):
        try:
            aux_log = (kT*math.pi*T)/LMS
            log = math.log(aux_log)
            return -8.0*math.pi*(8.0+9.0*math.pi*log-16.0*math.log(2.0*log))/(81.0*(log**3))
        except ValueError:
            return float('nan')

    def p_id(T: float, mu: float):
        muT = mu/T
        zero = (23.75*math.pi**2)/45.0
        mu1 = 0.5*(muT**2)
        piterm = 1.0/(4.0*(math.pi**2))
        mu2 = muT**4
        return zero + 3.0*(mu1 + piterm*mu2)

    def n_id(T: float, mu: float):
        muT = mu/T
        return muT + (1.0/(math.pi**2))*(muT**3)

    def s_id(T: float, mu: float):
        muT = mu/T
        return -3.0*((muT**2) + (muT**4)/(math.pi**2))

    def p_2(T: float, mu: float):
        muT = mu/T
        zero = (1.0 + ((5.0*3.0)/12.0))/6.0
        piterm = 1.0/(4.0*(math.pi**2))
        mu1 = 0.5*(muT**2)
        mu2 = muT**4
        return zero + (2.0*3.0*piterm)*(mu1 + piterm*mu2)

    def n_2(T: float, mu: float):
        return n_id(T, mu)/(2.0*(math.pi**2))

    def s_2(T: float, mu: float):
        return s_id(T, mu)/(2.0*(math.pi**2))

    def n_pert_og2(T: float, mu: float, kT: float, LMS: float):
        return n_id(T, mu) - g2(T, kT, LMS)*n_2(T, mu)

    def s_pert_og2(T: float, mu: float, kT: float, LMS: float):
        return 4.0*p_id(T, mu) + s_id(T, mu) - 4.0*g2(T, kT, LMS) - p_2(T, mu)*dg2dT(T, kT, LMS) - g2(T, kT, LMS)*s_2(T, mu)

    n_pert_og2_v_2_low = [n_pert_og2(T_el, mu_el/3.0, 4.0, 351.0) for T_el, mu_el in zip(T_2, muB_2)]
    n_pert_og2_v_2_high = [n_pert_og2(T_el, mu_el/3.0, 8.0, 327.0) for T_el, mu_el in zip(T_2, muB_2)]

    s_pert_og2_v_1_low = [s_pert_og2(T_el, mu_el/3.0, 4.0, 351.0) for T_el, mu_el in zip(T_1, muB_1)]
    s_pert_og2_v_1_high = [s_pert_og2(T_el, mu_el/3.0, 8.0, 327.0) for T_el, mu_el in zip(T_1, muB_1)]
    s_pert_og2_v_2_low = [s_pert_og2(T_el, mu_el/3.0, 4.0, 351.0) for T_el, mu_el in zip(T_2, muB_2)]
    s_pert_og2_v_2_high = [s_pert_og2(T_el, mu_el/3.0, 8.0, 327.0) for T_el, mu_el in zip(T_2, muB_2)]

    pQCD_bdensity_2 = [[x, y] for x, y in zip(T_2, n_pert_og2_v_2_high)]
    for x, y in zip(T_2[::-1], n_pert_og2_v_2_low[::-1]):
        pQCD_bdensity_2.append([x, y])
    pQCD_sdensity_1 = [[x, y] for x, y in zip(T_1, s_pert_og2_v_1_high)]
    for x, y in zip(T_1[::-1], s_pert_og2_v_1_low[::-1]):
        pQCD_sdensity_1.append([x, y])
    pQCD_sdensity_2 = [[x, y] for x, y in zip(T_2, s_pert_og2_v_2_high)]
    for x, y in zip(T_2[::-1], s_pert_og2_v_2_low[::-1]):
        pQCD_sdensity_2.append([x, y])

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (18.0, 5.0))

    fig1.subplots_adjust(
        left=0.167, bottom=0.11, right=0.988, top=0.979, wspace=0.2, hspace=0.2
    )

    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.axis([80., 280., -6.0, 20.0])

    ax1.add_patch(
        matplotlib.patches.Polygon(
            lQCD_sdensity_1, 
            closed = True, fill = True, color = 'green', alpha = 0.3
        )
    )
    ax1.add_patch(
        matplotlib.patches.Polygon(
            pQCD_sdensity_1, 
            closed = True, fill = True, color = 'red', alpha = 0.3
        )
    )

    ax1.plot(T_1, qgp_sdensity_v_1, '-', c = 'blue')
    ax1.plot(T_1, hrg_full_sdensity_v_1, '-', c = 'purple')
    ax1.plot(T_1, hrg2_full_sdensity_v_1, '--', c = 'purple')
    ax1.plot(T_1, mhrg_sdensity_v_1, '-', c = 'green')
    ax1.plot(T_1, [sum(el) for el in zip(mhrg_sdensity_v_1, qgp_sdensity_v_1)], '-', c = 'black')
    ax1.plot(T_1, [el[4] for el in qgp_partial_sdensity_v_1], '-', c = 'red')
    ax1.plot(T_1, [sum([el[8], el[9], el[10]]) for el in qgp_partial_sdensity_v_1], '-', c = 'magenta')
    ax1.plot(T_1, [el[8] for el in qgp_partial_sdensity_v_1], '--', c = 'magenta')
    ax1.plot(T_1, [el[10] for el in qgp_partial_sdensity_v_1], ':', c = 'magenta')

    ax1.text(85, 18.5, r"$\mathrm{\mu_B/T=0}$", color="black", fontsize=14)
    ax1.text(250, 15.5, r"QGP", color="blue", fontsize=14)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.axis([80., 280., -6.0, 20.0])

    ax2.add_patch(
        matplotlib.patches.Polygon(
            lQCD_sdensity_2, 
            closed = True, fill = True, color = 'green', alpha = 0.3
        )
    )
    ax2.add_patch(
        matplotlib.patches.Polygon(
            pQCD_sdensity_2, 
            closed = True, fill = True, color = 'red', alpha = 0.3
        )
    )

    ax2.plot(T_2, qgp_sdensity_v_2, '-', c = 'blue')
    ax2.plot(T_2, hrg_full_sdensity_v_2, '-', c = 'purple')
    ax2.plot(T_2, hrg2_full_sdensity_v_2, '--', c = 'purple')
    ax2.plot(T_2, mhrg_sdensity_v_2, '-', c = 'green')
    ax2.plot(T_2, [sum(el) for el in zip(mhrg_sdensity_v_2, qgp_sdensity_v_2)], '-', c = 'black')
    ax2.plot(T_2, [el[4] for el in qgp_partial_sdensity_v_2], '-', c = 'red')
    ax2.plot(T_2, [sum([el[8], el[9], el[10]]) for el in qgp_partial_sdensity_v_2], '-', c = 'magenta')
    ax2.plot(T_2, [el[8] for el in qgp_partial_sdensity_v_2], '--', c = 'magenta')
    ax2.plot(T_2, [el[10] for el in qgp_partial_sdensity_v_2], ':', c = 'magenta')

    ax2.text(85, 18.5, r"$\mathrm{\mu_B/T=2.5}$", color="black", fontsize=14)
    ax2.text(250, 17.3, r"QGP", color="blue", fontsize=14)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    ax3 = fig1.add_subplot(1, 3, 3)
    ax3.axis([80., 280., -0.4, 1.2])

    ax3.add_patch(
        matplotlib.patches.Polygon(
            lQCD_bdensity_2, closed = True, fill = True, color = "green", alpha = 0.3
        )
    )
    ax3.add_patch(
        matplotlib.patches.Polygon(
            pQCD_bdensity_2, closed = True, fill = True, color = "red", alpha = 0.3
        )
    )

    ax3.plot(T_2, qgp_bdensity_v_2, '-', c = 'blue')
    ax3.plot(T_2, hrg_full_bdensity_v_2, '-', c = 'purple')
    ax3.plot(T_2, hrg2_full_bdensity_v_2, '--', c = 'purple')
    ax3.plot(T_2, mhrg_bdensity_v_2, '-', c = 'green')
    ax3.plot(T_2, [sum(el) for el in zip(mhrg_bdensity_v_2, qgp_bdensity_v_2)], '-', c = 'black')
    ax3.plot(T_2, [el[4] for el in qgp_partial_bdensity_v_2], '-', c = 'red')
    ax3.plot(T_2, [sum([el[8], el[9], el[10]]) for el in qgp_partial_bdensity_v_2], '-', c = 'magenta')
    ax3.plot(T_2, [el[8] for el in qgp_partial_bdensity_v_2], '--', c = 'magenta')
    ax3.plot(T_2, [el[10] for el in qgp_partial_bdensity_v_2], ':', c = 'magenta')

    ax3.text(85, 1.1, r"$\mathrm{\mu_B/T=2.5}$", color="black", fontsize=14)
    ax3.text(200, 0.67, r"QGP", color="blue", fontsize=14)

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


if __name__ == '__main__':

    epja_figure1()

    print("END")