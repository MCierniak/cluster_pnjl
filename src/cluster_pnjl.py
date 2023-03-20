

def epja_figure1():

    import numpy

    import matplotlib.pyplot
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    import pnjl.defaults

    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    def phase(M, T, mu):

        MI_N = pnjl.defaults.MI['N']
        Mth_N = cluster.M_th(T, mu, 'N')

        if Mth_N > MI_N:
            if M < MI_N:
                return 0.0
            if M >= MI_N and M < Mth_N:
                return 1.0
            else:
                return cluster.continuum_factor1(M, T, mu, 'N')
        else:
            return cluster.continuum_factor2(M, T, mu, 'N')

    T_list = numpy.linspace(135, 160, num=17)
    M_list = numpy.linspace(0.5, 2.5, num=1000)

    phase_list = [
        [phase(M_el*1000.0, T_el, 0.0) for M_el in M_list]
        for T_el in T_list
    ]

    M_I = pnjl.defaults.MI['N']
    nLambda = pnjl.defaults.NI['N']*pnjl.defaults.L

    Mthi_vec = [
        cluster.M_th(T_el, 0.0, 'N')/1000.0 for T_el in T_list
    ]

    Mthi0_vec = [
        (cluster.M_th(0.0, 0.0, 'N')+nLambda)/1000.0 for _ in T_list
    ]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    ax1 = fig1.add_subplot(111, projection='3d')
    fig1.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)

    ax1.set_ylim3d(max(T_list), min(T_list))
    ax1.set_xlim3d(min(M_list), max(M_list))
    ax1.set_zlim3d(0, 1)

    ax1.plot3D(
        M_list, [T_list[0] for el in M_list], phase_list[0], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[2] for el in M_list], phase_list[2], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[4] for el in M_list], phase_list[4], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[6] for el in M_list], phase_list[6], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[8] for el in M_list], phase_list[8], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[10] for el in M_list], phase_list[10], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[12] for el in M_list], phase_list[12], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[14] for el in M_list], phase_list[14], '-',
        c = 'black'
    )
    ax1.plot3D(
        M_list, [T_list[16] for el in M_list], phase_list[16], '-',
        c='black'
    )
    
    ax1.plot3D(Mthi0_vec, T_list, [0.0 for el in T_list], '--', c='red')
    ax1.plot3D(Mthi_vec, T_list, [0.0 for el in T_list], '--', c='green')
    ax1.plot3D(
        [M_I/1000.0 for _ in T_list], T_list, [0.0 for el in T_list], '--',
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
    ax1.set_yticks([135, 140, 145, 150, 155, 160])
    ax1.set_yticklabels([135, 140, 145, 150, 155, ''])
    
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

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure2():

    import numpy

    import matplotlib.pyplot
    import matplotlib.patches

    import pnjl.defaults
    import pnjl.thermo.gcp_sigma_lattice

    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
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
            0.0, 1.05 * max([cluster.M_th(el, 0.0, 'H') for el in T])
        ]
    )

    ax.plot(
        T, [pnjl.defaults.MI['N'] for el in T], '-', c=col_n,
        label = r'nucleon'
    )
    ax.add_patch(
        matplotlib.patches.FancyArrowPatch(
            (30., 1750.), (30., 1000.), arrowstyle='<->',
            mutation_scale=20, color=col_n
        )
    )
    ax.plot(T, [cluster.M_th(el, 0.0, 'N') for el in T], '--', c=col_n)
    ax.text(35., 1320., r'nucleon', fontsize = 14)

    ax.plot(
        T, [pnjl.defaults.MI['pi'] for el in T], '-', c=col_pi,
        label=r'$\mathrm{\pi}$'
    )
    ax.add_patch(
        matplotlib.patches.FancyArrowPatch(
            (50., 1150.), (50., 135.), arrowstyle='<->',
            mutation_scale=20, color=col_pi
        )
    )
    ax.plot(T, [cluster.M_th(el, 0.0, 'pi') for el in T], '--', c=col_pi)
    ax.text(55., 620., r'pion', fontsize = 14)

    ax.plot(
        T, [pnjl.defaults.MI['H'] for el in T], '-', c=col_h,
        label = r'hexaquark'
    )
    ax.add_patch(
        matplotlib.patches.FancyArrowPatch(
            (10., 3450.), (10., 1900.), arrowstyle='<->',
            mutation_scale=20, color=col_h
        )
    )
    ax.plot(T, [cluster.M_th(el, 0.0, 'H') for el in T], '--', c=col_h)
    ax.text(15., 2650., r'hexaquark', fontsize = 14)

    ax.plot(
        T, [pnjl.thermo.gcp_sigma_lattice.Ml(el, 0.0) for el in T],
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

    import numpy
    import pickle

    import matplotlib.pyplot

    import pnjl.thermo.gcp_sigma_lattice

    T = numpy.linspace(1.0, 250.0, num = 200)

    sigma = [pnjl.thermo.gcp_sigma_lattice.Delta_ls(el, 0.0) for el in T]

    with open(
        "D:/EoS/epja/lattice_data_pickled/1005_3508_table3_delta.pickle",
        "rb"
    ) as file:
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


def epja_figure4():

    import tqdm
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

    import pnjl.thermo.gcp_sigma_lattice

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_n

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        clusters_bound_step_continuum_step \
    as solver_s

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        clusters_bound_step_continuum_acos_cos \
    as solver_c

    warnings.filterwarnings("ignore")

    calc_n = False
    calc_s = False
    calc_c = False

    files = "D:/EoS/epja/figure4/"

    T0 = numpy.linspace(1.0, 250.0, num=200)
    T600 = numpy.linspace(1.0, 250.0, num=200)

    phi_re_0_n, phi_im_0_n = \
        list(), list()
    phi_re_0_s, phi_im_0_s = \
        list(), list()
    phi_re_0_c, phi_im_0_c = \
        list(), list()

    phi_re_600_n, phi_im_600_n = \
        list(), list()
    phi_re_600_s, phi_im_600_s = \
        list(), list()
    phi_re_600_c, phi_im_600_c = \
        list(), list()
    
    sigma_0 = [
        pnjl.thermo.gcp_sigma_lattice.Ml(T_el, 0.0) / \
        pnjl.thermo.gcp_sigma_lattice.Ml(0.0, 0.0)
        for T_el in T0]
    sigma_600 = [
        pnjl.thermo.gcp_sigma_lattice.Ml(T_el, 600.0) / \
        pnjl.thermo.gcp_sigma_lattice.Ml(0.0, 0.0)
        for T_el in T0]

    if calc_n:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print(
            "Traced Polyakov loop, no clusters, muB = 0 MeV"
        )
        for T_el in tqdm.tqdm(T0, total=len(T0), ncols=100):
            phi_result = solver_n.Polyakov_loop(
                T_el, 0.0, phi_re_0, phi_im_0
            )
            phi_re_0_n.append(phi_result[0])
            phi_im_0_n.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_0_n.pickle", "wb") as file:
            pickle.dump(phi_re_0_n, file)
        with open(files+"phi_im_0_n.pickle", "wb") as file:
            pickle.dump(phi_im_0_n, file)
            
        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print(
            "Traced Polyakov loop, no clusters, muB = 600 MeV"
        )
        for T_el in tqdm.tqdm(T600, total=len(T600), ncols=100):
            phi_result = solver_n.Polyakov_loop(
                T_el, 600.0/3.0, phi_re_0, phi_im_0
            )
            phi_re_600_n.append(phi_result[0])
            phi_im_600_n.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_600_n.pickle", "wb") as file:
            pickle.dump(phi_re_600_n, file)
        with open(files+"phi_im_600_n.pickle", "wb") as file:
            pickle.dump(phi_im_600_n, file)
    else:
        with open(files+"phi_re_0_n.pickle", "rb") as file:
            phi_re_0_n = pickle.load(file)
        with open(files+"phi_im_0_n.pickle", "rb") as file:
            phi_im_0_n = pickle.load(file)
        with open(files+"phi_re_600_n.pickle", "rb") as file:
            phi_re_600_n = pickle.load(file)
        with open(files+"phi_im_600_n.pickle", "rb") as file:
            phi_im_600_n = pickle.load(file)

    if calc_s:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print(
            "Traced Polyakov loop, clusters step-up-step-down, muB = 0 MeV"
        )
        for T_el in tqdm.tqdm(T0, total=len(T0), ncols=100):
            phi_result = solver_s.Polyakov_loop(
                T_el, 0.0, phi_re_0, phi_im_0
            )
            phi_re_0_s.append(phi_result[0])
            phi_im_0_s.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_0_s.pickle", "wb") as file:
            pickle.dump(phi_re_0_s, file)
        with open(files+"phi_im_0_s.pickle", "wb") as file:
            pickle.dump(phi_im_0_s, file)

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print(
            "Traced Polyakov loop, clusters step-up-step-down, muB = 600 MeV"
        )
        for T_el in tqdm.tqdm(T600, total=len(T600), ncols=100):
            phi_result = solver_s.Polyakov_loop(
                T_el, 600.0/3.0, phi_re_0, phi_im_0
            )
            phi_re_600_s.append(phi_result[0])
            phi_im_600_s.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_600_s.pickle", "wb") as file:
            pickle.dump(phi_re_600_s, file)
        with open(files+"phi_im_600_s.pickle", "wb") as file:
            pickle.dump(phi_im_600_s, file)
    else:
        with open(files+"phi_re_0_s.pickle", "rb") as file:
            phi_re_0_s = pickle.load(file)
        with open(files+"phi_im_0_s.pickle", "rb") as file:
            phi_im_0_s = pickle.load(file)
        with open(files+"phi_re_600_s.pickle", "rb") as file:
            phi_re_600_s = pickle.load(file)
        with open(files+"phi_im_600_s.pickle", "rb") as file:
            phi_im_600_s = pickle.load(file)

    if calc_c:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print(
            "Traced Polyakov loop, clusters step-up-acos-cos, muB = 0 MeV"
        )
        for T_el in tqdm.tqdm(T0, total=len(T0), ncols=100):
            phi_result = solver_c.Polyakov_loop(
                T_el, 0.0, phi_re_0, phi_im_0
            )
            phi_re_0_c.append(phi_result[0])
            phi_im_0_c.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_0_c.pickle", "wb") as file:
            pickle.dump(phi_re_0_c, file)
        with open(files+"phi_im_0_c.pickle", "wb") as file:
            pickle.dump(phi_im_0_c, file)

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print(
            "Traced Polyakov loop, clusters step-up-acos-cos, muB = 600 MeV"
        )
        for T_el in tqdm.tqdm(T600, total=len(T600), ncols=100):
            phi_result = solver_c.Polyakov_loop(
                T_el, 600.0/3.0, phi_re_0, phi_im_0
            )
            phi_re_600_c.append(phi_result[0])
            phi_im_600_c.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_600_c.pickle", "wb") as file:
            pickle.dump(phi_re_600_c, file)
        with open(files+"phi_im_600_c.pickle", "wb") as file:
            pickle.dump(phi_im_600_c, file)
    else:
        with open(files+"phi_re_0_c.pickle", "rb") as file:
            phi_re_0_c = pickle.load(file)
        with open(files+"phi_im_0_c.pickle", "rb") as file:
            phi_im_0_c = pickle.load(file)
        with open(files+"phi_re_600_c.pickle", "rb") as file:
            phi_re_600_c = pickle.load(file)
        with open(files+"phi_im_600_c.pickle", "rb") as file:
            phi_im_600_c = pickle.load(file)

    fig = matplotlib.pyplot.figure(num = 1, figsize = (11.0, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.axis([100.0, 180.0, 0.0, 1.1])

    ax.fill_between(T0, phi_re_0_c, y2=phi_re_0_s, color='blue', alpha=0.5)
    ax.plot(T0, phi_re_0_s, '-', c='blue')
    ax.plot(T0, phi_re_0_c, '-', c='blue')

    ax.plot(T0, phi_re_0_n, '--', c='red')

    ax.plot(T0, sigma_0, c = 'green')

    ax.text(
        103., 1.0, r'$\mathrm{M_q}$ / $\mathrm{M_{q,vac}}$', fontsize=14
    )
    ax.text(168., 1.03, r'$\mathrm{\mu_B=0}$', fontsize=14)
    ax.text(140., 0.25, r'$\mathrm{\Phi}$', fontsize=14, color='blue')
    ax.text(150., 0.25, r'$\mathrm{\Phi_0}$', fontsize=14, color='red')

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    ax.set_xlabel(r'T [MeV]', fontsize=16)
    ax.set_ylabel(r'$\mathrm{\Phi}$', fontsize=16)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis([100.0, 180.0, 0.0, 1.1])

    ax2.fill_between(
        T600, phi_re_600_c, y2=phi_re_600_s, color='blue', alpha=0.5
    )
    ax2.plot(T600, phi_re_600_s, '-', c='blue')
    ax2.plot(T600, phi_re_600_c, '-', c='blue')

    ax2.plot(T600, phi_re_600_n, '--', c='red')

    ax2.plot(T600, sigma_600, c = 'green')

    ax2.text(
        103., 0.88, r'$\mathrm{M_q}$ / $\mathrm{M_{q,vac}}$',
        fontsize=14
    )
    ax2.text(155., 1.03, r'$\mathrm{\mu_B=600}$ MeV', fontsize=14)
    ax2.text(123., 0.25, r'$\mathrm{\Phi}$', fontsize=14, color='blue')
    ax2.text(140., 0.25, r'$\mathrm{\Phi_0}$', fontsize=14, color='red')

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{\Phi}$', fontsize = 16)

    fig.tight_layout()

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure5():

    import tqdm
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")

    calc_1 = False

    files = "D:/EoS/epja/figure5/"

    T = numpy.linspace(1.0, 2000.0, num=200)

    mu = [0.0/3.0 for el in T]

    phi_re_v_1 = list()
    phi_im_v_1 = list()

    sigma_v_1 = list()
    gluon_v_1 = list()
    sea_u_v_1 = list()
    sea_d_v_1 = list()
    sea_s_v_1 = list()
    perturbative_u_v_1 = list()
    perturbative_d_v_1 = list()
    perturbative_s_v_1 = list()
    perturbative_gluon_v_1 = list()
    pnjl_u_v_1 = list()
    pnjl_d_v_1 = list()
    pnjl_s_v_1 = list()

    if calc_1:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop")
        for T_el, mu_el in tqdm.tqdm(
            zip(T, mu), total=len(T), ncols=100
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

        print("Sigma pressure")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T, mu, phi_re_v_1, phi_im_v_1), total=len(T), ncols=100
        ):
            sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.pressure(
                    T_el, mu_el
                )/(T_el**4)
            )
        with open(files+"sigma_v_1.pickle", "wb") as file:
            pickle.dump(sigma_v_1, file)

        print("Gluon pressure")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T, mu, phi_re_v_1, phi_im_v_1), total=len(T), ncols=100
        ):
            gluon_v_1.append(
                pnjl.thermo.gcp_pl_polynomial.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el
                )/(T_el**4)
            )
        with open(files+"gluon_v_1.pickle", "wb") as file:
            pickle.dump(gluon_v_1, file)

        print("Sea pressure")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T, mu, phi_re_v_1, phi_im_v_1), total=len(T), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el, 'l')
            sea_u_v_1.append(lq_temp/(T_el**4))
            sea_d_v_1.append(lq_temp/(T_el**4))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
                    T_el, mu_el, 's'
                )/(T_el**4)
            )
        with open(files+"sea_u_v_1.pickle", "wb") as file:
            pickle.dump(sea_u_v_1, file)
        with open(files+"sea_d_v_1.pickle", "wb") as file:
            pickle.dump(sea_d_v_1, file)
        with open(files+"sea_s_v_1.pickle", "wb") as file:
            pickle.dump(sea_s_v_1, file)

        print("Perturbative pressure")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T, mu, phi_re_v_1, phi_im_v_1), total=len(T), ncols=100
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
            perturbative_gluon_v_1.append(0.0/(T_el**4))
        with open(files+"perturbative_u_v_1.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_1, file)
        with open(files+"perturbative_d_v_1.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_1, file)
        with open(files+"perturbative_s_v_1.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_1, file)
        with open(files+"perturbative_gluon_v_1.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_1, file)

        print("PNJL pressure")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T, mu, phi_re_v_1, phi_im_v_1), total=len(T), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
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

    with open(
        "D:/EoS/epja/lattice_data_pickled/bazavov_1407_6387_mu0.pickle",
        "rb"
    ) as file:
        bazavov_1407_6387_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1309_5258_mu0.pickle",
        "rb"
    ) as file:
        borsanyi_1309_5258_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/bazavov_1710_05024_mu0.pickle",
        "rb"
    ) as file:
        bazavov_1710_05024_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu0.pickle",
        "rb"
    ) as file:
        borsanyi_1204_6710v2_mu0 = pickle.load(file)

    total_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1,gluon_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]

    pertrubative_total = [
        sum(el) for el in zip(
            perturbative_u_v_1, perturbative_d_v_1,
            perturbative_s_v_1, perturbative_gluon_v_1
        )
    ]

    pnjl_total = [
        sum(el) for el in zip(pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1)
    ]

    fig1 = matplotlib.pyplot.figure(num=1, figsize=(5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([10., 2000., -1.5, 5.1])
    #ax1.axis([50., 300., -1.5, 3.5])

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

    ax1.plot(T, total_1, '-', c='black')
    ax1.plot(T, gluon_v_1, '--', c='red')
    ax1.plot(T, pertrubative_total, '--', c='pink')
    ax1.plot(T, pnjl_total, '--', c='blue')

    ax2.plot(T, total_1, '-', c='black')
    ax2.plot(T, gluon_v_1, '--', c='red')
    ax2.plot(T, pertrubative_total, '--', c='pink')
    ax2.plot(T, pnjl_total, '--', c='blue')

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


def epja_figure6():

    import tqdm
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")

    calc_1 = False
    calc_2 = False
    calc_3 = False
    calc_4 = False

    files = "D:/EoS/epja/figure6/"

    T_1 = numpy.linspace(10.0, 540.0, num=200)
    T_2 = numpy.linspace(10.0, 2000.0, num=200)
    T_3 = numpy.linspace(10.0, 540.0, num=200)
    T_4 = numpy.linspace(10.0, 540.0, num=200)

    mu_1 = [0.0/3.0 for el in T_1]
    mu_2 = [0.0/3.0 for el in T_2]
    mu_3 = [200.0/3.0 for el in T_3]
    mu_4 = [400.0/3.0 for el in T_4]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()
    phi_re_v_2, phi_im_v_2 = \
        list(), list()
    phi_re_v_3, phi_im_v_3 = \
        list(), list()
    phi_re_v_4, phi_im_v_4 = \
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
    sigma_v_3, gluon_v_3, sea_u_v_3, sea_d_v_3, sea_s_v_3 = \
        list(), list(), list(), list(), list()
    perturbative_u_v_3, perturbative_d_v_3, perturbative_s_v_3 = \
        list(), list(), list()
    perturbative_gluon_v_3, pnjl_u_v_3, pnjl_d_v_3, pnjl_s_v_3 = \
        list(), list(), list(), list()
    sigma_v_4, gluon_v_4, sea_u_v_4, sea_d_v_4, sea_s_v_4 = \
        list(), list(), list(), list(), list()
    perturbative_u_v_4, perturbative_d_v_4, perturbative_s_v_4 = \
        list(), list(), list()
    perturbative_gluon_v_4, pnjl_u_v_4, pnjl_d_v_4, pnjl_s_v_4 = \
        list(), list(), list(), list()

    pi_v_1, K_v_1, rho_v_1, omega_v_1, D_v_1, N_v_1, T_v_1, F_v_1 = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_1, Q_v_1, H_v_1 = \
        list(), list(), list()
    pi_v_2, K_v_2, rho_v_2, omega_v_2, D_v_2, N_v_2, T_v_2, F_v_2 = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_2, Q_v_2, H_v_2 = \
        list(), list(), list()
    pi_v_3, K_v_3, rho_v_3, omega_v_3, D_v_3, N_v_3, T_v_3, F_v_3 = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_3, Q_v_3, H_v_3 = \
        list(), list(), list()
    pi_v_4, K_v_4, rho_v_4, omega_v_4, D_v_4, N_v_4, T_v_4, F_v_4 = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_4, Q_v_4, H_v_4 = \
        list(), list(), list()

    pi_v_1s, K_v_1s, rho_v_1s, omega_v_1s, D_v_1s, N_v_1s, T_v_1s, F_v_1s = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_1s, Q_v_1s, H_v_1s = \
        list(), list(), list()
    pi_v_2s, K_v_2s, rho_v_2s, omega_v_2s, D_v_2s, N_v_2s, T_v_2s, F_v_2s = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_2s, Q_v_2s, H_v_2s = \
        list(), list(), list()
    pi_v_3s, K_v_3s, rho_v_3s, omega_v_3s, D_v_3s, N_v_3s, T_v_3s, F_v_3s = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_3s, Q_v_3s, H_v_3s = \
        list(), list(), list()
    pi_v_4s, K_v_4s, rho_v_4s, omega_v_4s, D_v_4s, N_v_4s, T_v_4s, F_v_4s = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_4s, Q_v_4s, H_v_4s = \
        list(), list(), list()

    if calc_1:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #1")
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
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1),
            ncols=100
        ):
            sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.pressure(
                    T_el, mu_el
                )/(T_el**4)
            )
        with open(files+"sigma_v_1.pickle", "wb") as file:
            pickle.dump(sigma_v_1, file)

        print("Gluon pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1),
            ncols=100
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
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el, 'l')
            sea_u_v_1.append(lq_temp/(T_el**4))
            sea_d_v_1.append(lq_temp/(T_el**4))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
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
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1),
            ncols=100
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
            perturbative_gluon_v_1.append(0.0/(T_el**4))
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
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
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

    if calc_2:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #2")
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
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2),
            ncols=100
        ):
            sigma_v_2.append(
                pnjl.thermo.gcp_sigma_lattice.pressure(
                    T_el, mu_el
                )/(T_el**4)
            )
        with open(files+"sigma_v_2.pickle", "wb") as file:
            pickle.dump(sigma_v_2, file)

        print("Gluon pressure #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2),
            ncols=100
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
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el, 'l')
            sea_u_v_2.append(lq_temp/(T_el**4))
            sea_d_v_2.append(lq_temp/(T_el**4))
            sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
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
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2),
            ncols=100
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
            perturbative_gluon_v_2.append(0.0/(T_el**4))
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
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
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

    if calc_3:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #3")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_3, mu_3), total=len(T_3), ncols=100
        ):
            phi_result = solver_1.Polyakov_loop(
                T_el, mu_el, phi_re_0, phi_im_0
            )
            phi_re_v_3.append(phi_result[0])
            phi_im_v_3.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_3.pickle", "wb") as file:
            pickle.dump(phi_re_v_3, file)
        with open(files+"phi_im_v_3.pickle", "wb") as file:
            pickle.dump(phi_im_v_3, file)

        print("Sigma pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3),
            ncols=100
        ):
            sigma_v_3.append(
                pnjl.thermo.gcp_sigma_lattice.pressure(
                    T_el, mu_el
                )/(T_el**4)
            )
        with open(files+"sigma_v_3.pickle", "wb") as file:
            pickle.dump(sigma_v_3, file)

        print("Gluon pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3),
            ncols=100
        ):
            gluon_v_3.append(
                pnjl.thermo.gcp_pl_polynomial.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el
                )/(T_el**4)
            )
        with open(files+"gluon_v_3.pickle", "wb") as file:
            pickle.dump(gluon_v_3, file)

        print("Sea pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el, 'l')
            sea_u_v_3.append(lq_temp/(T_el**4))
            sea_d_v_3.append(lq_temp/(T_el**4))
            sea_s_v_3.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
                    T_el, mu_el, 's'
                )/(T_el**4)
            )
        with open(files+"sea_u_v_3.pickle", "wb") as file:
            pickle.dump(sea_u_v_3, file)
        with open(files+"sea_d_v_3.pickle", "wb") as file:
            pickle.dump(sea_d_v_3, file)
        with open(files+"sea_s_v_3.pickle", "wb") as file:
            pickle.dump(sea_s_v_3, file)

        print("Perturbative pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            perturbative_u_v_3.append(lq_temp/(T_el**4))
            perturbative_d_v_3.append(lq_temp/(T_el**4))
            perturbative_s_v_3.append(
                pnjl.thermo.gcp_perturbative.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 's'
                )/(T_el**4)
            )
            perturbative_gluon_v_3.append(0.0/(T_el**4))
        with open(files+"perturbative_u_v_3.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_3, file)
        with open(files+"perturbative_d_v_3.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_3, file)
        with open(files+"perturbative_s_v_3.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_3, file)
        with open(files+"perturbative_gluon_v_3.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_3, file)

        print("PNJL pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 's'
            )
            pnjl_u_v_3.append(lq_temp/(T_el**4))
            pnjl_d_v_3.append(lq_temp/(T_el**4))
            pnjl_s_v_3.append(sq_temp/(T_el**4))
        with open(files+"pnjl_u_v_3.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_3, file)
        with open(files+"pnjl_d_v_3.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_3, file)
        with open(files+"pnjl_s_v_3.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_3, file)
    else:
        with open(files+"phi_re_v_3.pickle", "rb") as file:
            phi_re_v_3 = pickle.load(file)
        with open(files+"phi_im_v_3.pickle", "rb") as file:
            phi_im_v_3 = pickle.load(file)
        with open(files+"sigma_v_3.pickle", "rb") as file:
            sigma_v_3 = pickle.load(file)
        with open(files+"gluon_v_3.pickle", "rb") as file:
            gluon_v_3 = pickle.load(file)
        with open(files+"sea_u_v_3.pickle", "rb") as file:
            sea_u_v_3 = pickle.load(file)
        with open(files+"sea_d_v_3.pickle", "rb") as file:
            sea_d_v_3 = pickle.load(file)
        with open(files+"sea_s_v_3.pickle", "rb") as file:
            sea_s_v_3 = pickle.load(file)
        with open(files+"perturbative_u_v_3.pickle", "rb") as file:
            perturbative_u_v_3 = pickle.load(file)
        with open(files+"perturbative_d_v_3.pickle", "rb") as file:
            perturbative_d_v_3 = pickle.load(file)
        with open(files+"perturbative_s_v_3.pickle", "rb") as file:
            perturbative_s_v_3 = pickle.load(file)
        with open(files+"perturbative_gluon_v_3.pickle", "rb") as file:
            perturbative_gluon_v_3 = pickle.load(file)
        with open(files+"pnjl_u_v_3.pickle", "rb") as file:
            pnjl_u_v_3 = pickle.load(file)
        with open(files+"pnjl_d_v_3.pickle", "rb") as file:
            pnjl_d_v_3 = pickle.load(file)
        with open(files+"pnjl_s_v_3.pickle", "rb") as file:
            pnjl_s_v_3 = pickle.load(file)

    if calc_4:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #4")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_4, mu_4), total=len(T_4), ncols=100
        ):
            phi_result = solver_1.Polyakov_loop(
                T_el, mu_el, phi_re_0, phi_im_0
            )
            phi_re_v_4.append(phi_result[0])
            phi_im_v_4.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_4.pickle", "wb") as file:
            pickle.dump(phi_re_v_4, file)
        with open(files+"phi_im_v_4.pickle", "wb") as file:
            pickle.dump(phi_im_v_4, file)

        print("Sigma pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4),
            ncols=100
        ):
            sigma_v_4.append(
                pnjl.thermo.gcp_sigma_lattice.pressure(
                    T_el, mu_el
                )/(T_el**4)
            )
        with open(files+"sigma_v_4.pickle", "wb") as file:
            pickle.dump(sigma_v_4, file)

        print("Gluon pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4),
            ncols=100
        ):
            gluon_v_4.append(
                pnjl.thermo.gcp_pl_polynomial.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el
                )/(T_el**4)
            )
        with open(files+"gluon_v_4.pickle", "wb") as file:
            pickle.dump(gluon_v_4, file)

        print("Sea pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el, 'l')
            sea_u_v_4.append(lq_temp/(T_el**4))
            sea_d_v_4.append(lq_temp/(T_el**4))
            sea_s_v_4.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
                    T_el, mu_el, 's'
                )/(T_el**4)
            )
        with open(files+"sea_u_v_4.pickle", "wb") as file:
            pickle.dump(sea_u_v_4, file)
        with open(files+"sea_d_v_4.pickle", "wb") as file:
            pickle.dump(sea_d_v_4, file)
        with open(files+"sea_s_v_4.pickle", "wb") as file:
            pickle.dump(sea_s_v_4, file)

        print("Perturbative pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            perturbative_u_v_4.append(lq_temp/(T_el**4))
            perturbative_d_v_4.append(lq_temp/(T_el**4))
            perturbative_s_v_4.append(
                pnjl.thermo.gcp_perturbative.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 's'
                )/(T_el**4)
            )
            perturbative_gluon_v_4.append(0.0/(T_el**4))
        with open(files+"perturbative_u_v_4.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_4, file)
        with open(files+"perturbative_d_v_4.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_4, file)
        with open(files+"perturbative_s_v_4.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_4, file)
        with open(files+"perturbative_gluon_v_4.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_4, file)

        print("PNJL pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 's'
            )
            pnjl_u_v_4.append(lq_temp/(T_el**4))
            pnjl_d_v_4.append(lq_temp/(T_el**4))
            pnjl_s_v_4.append(sq_temp/(T_el**4))
        with open(files+"pnjl_u_v_4.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_4, file)
        with open(files+"pnjl_d_v_4.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_4, file)
        with open(files+"pnjl_s_v_4.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_4, file)
    else:
        with open(files+"phi_re_v_4.pickle", "rb") as file:
            phi_re_v_4 = pickle.load(file)
        with open(files+"phi_im_v_4.pickle", "rb") as file:
            phi_im_v_4 = pickle.load(file)
        with open(files+"sigma_v_4.pickle", "rb") as file:
            sigma_v_4 = pickle.load(file)
        with open(files+"gluon_v_4.pickle", "rb") as file:
            gluon_v_4 = pickle.load(file)
        with open(files+"sea_u_v_4.pickle", "rb") as file:
            sea_u_v_4 = pickle.load(file)
        with open(files+"sea_d_v_4.pickle", "rb") as file:
            sea_d_v_4 = pickle.load(file)
        with open(files+"sea_s_v_4.pickle", "rb") as file:
            sea_s_v_4 = pickle.load(file)
        with open(files+"perturbative_u_v_4.pickle", "rb") as file:
            perturbative_u_v_4 = pickle.load(file)
        with open(files+"perturbative_d_v_4.pickle", "rb") as file:
            perturbative_d_v_4 = pickle.load(file)
        with open(files+"perturbative_s_v_4.pickle", "rb") as file:
            perturbative_s_v_4 = pickle.load(file)
        with open(files+"perturbative_gluon_v_4.pickle", "rb") as file:
            perturbative_gluon_v_4 = pickle.load(file)
        with open(files+"pnjl_u_v_4.pickle", "rb") as file:
            pnjl_u_v_4 = pickle.load(file)
        with open(files+"pnjl_d_v_4.pickle", "rb") as file:
            pnjl_d_v_4 = pickle.load(file)
        with open(files+"pnjl_s_v_4.pickle", "rb") as file:
            pnjl_s_v_4 = pickle.load(file)

    if calc_1:

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

    if calc_2:

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

    if calc_3:

        print("Pion pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            pi_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**4)
            )
        with open(files+"pi_v_3.pickle", "wb") as file:
            pickle.dump(pi_v_3, file)

        print("Kaon pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            K_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**4)
            )
        with open(files+"K_v_3.pickle", "wb") as file:
            pickle.dump(K_v_3, file)

        print("Rho pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            rho_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**4)
            )
        with open(files+"rho_v_3.pickle", "wb") as file:
            pickle.dump(rho_v_3, file)

        print("Omega pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            omega_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**4)
            )
        with open(files+"omega_v_3.pickle", "wb") as file:
            pickle.dump(omega_v_3, file)

        print("Diquark pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            D_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**4)
            )
        with open(files+"D_v_3.pickle", "wb") as file:
            pickle.dump(D_v_3, file)

        print("Nucleon pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            N_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**4)
            )
        with open(files+"N_v_3.pickle", "wb") as file:
            pickle.dump(N_v_3, file)

        print("Tetraquark pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            T_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**4)
            )
        with open(files+"T_v_3.pickle", "wb") as file:
            pickle.dump(T_v_3, file)

        print("F-quark pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            F_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**4)
            )
        with open(files+"F_v_3.pickle", "wb") as file:
            pickle.dump(F_v_3, file)

        print("Pentaquark pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            P_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**4)
            )
        with open(files+"P_v_3.pickle", "wb") as file:
            pickle.dump(P_v_3, file)

        print("Q-quark pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            Q_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**4)
            )
        with open(files+"Q_v_3.pickle", "wb") as file:
            pickle.dump(Q_v_3, file)

        print("Hexaquark pressure #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            H_v_3.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**4)
            )
        with open(files+"H_v_3.pickle", "wb") as file:
            pickle.dump(H_v_3, file)
    else:
        with open(files+"pi_v_3.pickle", "rb") as file:
            pi_v_3 = pickle.load(file)
        with open(files+"K_v_3.pickle", "rb") as file:
            K_v_3 = pickle.load(file)
        with open(files+"rho_v_3.pickle", "rb") as file:
            rho_v_3 = pickle.load(file)
        with open(files+"omega_v_3.pickle", "rb") as file:
            omega_v_3 = pickle.load(file)
        with open(files+"D_v_3.pickle", "rb") as file:
            D_v_3 = pickle.load(file)
        with open(files+"N_v_3.pickle", "rb") as file:
            N_v_3 = pickle.load(file)
        with open(files+"T_v_3.pickle", "rb") as file:
            T_v_3 = pickle.load(file)
        with open(files+"F_v_3.pickle", "rb") as file:
            F_v_3 = pickle.load(file)
        with open(files+"P_v_3.pickle", "rb") as file:
            P_v_3 = pickle.load(file)
        with open(files+"Q_v_3.pickle", "rb") as file:
            Q_v_3 = pickle.load(file)
        with open(files+"H_v_3.pickle", "rb") as file:
            H_v_3 = pickle.load(file)

    if calc_4:

        print("Pion pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            pi_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**4)
            )
        with open(files+"pi_v_4.pickle", "wb") as file:
            pickle.dump(pi_v_4, file)

        print("Kaon pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            K_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**4)
            )
        with open(files+"K_v_4.pickle", "wb") as file:
            pickle.dump(K_v_4, file)

        print("Rho pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            rho_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**4)
            )
        with open(files+"rho_v_4.pickle", "wb") as file:
            pickle.dump(rho_v_4, file)

        print("Omega pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            omega_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**4)
            )
        with open(files+"omega_v_4.pickle", "wb") as file:
            pickle.dump(omega_v_4, file)

        print("Diquark pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            D_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**4)
            )
        with open(files+"D_v_4.pickle", "wb") as file:
            pickle.dump(D_v_4, file)

        print("Nucleon pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            N_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**4)
            )
        with open(files+"N_v_4.pickle", "wb") as file:
            pickle.dump(N_v_4, file)

        print("Tetraquark pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            T_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**4)
            )
        with open(files+"T_v_4.pickle", "wb") as file:
            pickle.dump(T_v_4, file)

        print("F-quark pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            F_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**4)
            )
        with open(files+"F_v_4.pickle", "wb") as file:
            pickle.dump(F_v_4, file)

        print("Pentaquark pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            P_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**4)
            )
        with open(files+"P_v_4.pickle", "wb") as file:
            pickle.dump(P_v_4, file)

        print("Q-quark pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            Q_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**4)
            )
        with open(files+"Q_v_4.pickle", "wb") as file:
            pickle.dump(Q_v_4, file)

        print("Hexaquark pressure #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            H_v_4.append(
                cluster.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**4)
            )
        with open(files+"H_v_4.pickle", "wb") as file:
            pickle.dump(H_v_4, file)
    else:
        with open(files+"pi_v_4.pickle", "rb") as file:
            pi_v_4 = pickle.load(file)
        with open(files+"K_v_4.pickle", "rb") as file:
            K_v_4 = pickle.load(file)
        with open(files+"rho_v_4.pickle", "rb") as file:
            rho_v_4 = pickle.load(file)
        with open(files+"omega_v_4.pickle", "rb") as file:
            omega_v_4 = pickle.load(file)
        with open(files+"D_v_4.pickle", "rb") as file:
            D_v_4 = pickle.load(file)
        with open(files+"N_v_4.pickle", "rb") as file:
            N_v_4 = pickle.load(file)
        with open(files+"T_v_4.pickle", "rb") as file:
            T_v_4 = pickle.load(file)
        with open(files+"F_v_4.pickle", "rb") as file:
            F_v_4 = pickle.load(file)
        with open(files+"P_v_4.pickle", "rb") as file:
            P_v_4 = pickle.load(file)
        with open(files+"Q_v_4.pickle", "rb") as file:
            Q_v_4 = pickle.load(file)
        with open(files+"H_v_4.pickle", "rb") as file:
            H_v_4 = pickle.load(file)

    if calc_1:

        print("Pion pressure #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1s.append(
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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

    if calc_2:

        print("Pion pressure #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2s.append(
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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

    if calc_3:

        print("Pion pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            pi_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**4)
            )
        with open(files+"pi_v_3s.pickle", "wb") as file:
            pickle.dump(pi_v_3s, file)

        print("Kaon pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            K_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**4)
            )
        with open(files+"K_v_3s.pickle", "wb") as file:
            pickle.dump(K_v_3s, file)

        print("Rho pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            rho_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**4)
            )
        with open(files+"rho_v_3s.pickle", "wb") as file:
            pickle.dump(rho_v_3s, file)

        print("Omega pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            omega_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**4)
            )
        with open(files+"omega_v_3s.pickle", "wb") as file:
            pickle.dump(omega_v_3s, file)

        print("Diquark pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            D_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**4)
            )
        with open(files+"D_v_3s.pickle", "wb") as file:
            pickle.dump(D_v_3s, file)

        print("Nucleon pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            N_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**4)
            )
        with open(files+"N_v_3s.pickle", "wb") as file:
            pickle.dump(N_v_3s, file)

        print("Tetraquark pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            T_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**4)
            )
        with open(files+"T_v_3s.pickle", "wb") as file:
            pickle.dump(T_v_3s, file)

        print("F-quark pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            F_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**4)
            )
        with open(files+"F_v_3s.pickle", "wb") as file:
            pickle.dump(F_v_3s, file)

        print("Pentaquark pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            P_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**4)
            )
        with open(files+"P_v_3s.pickle", "wb") as file:
            pickle.dump(P_v_3s, file)

        print("Q-quark pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            Q_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**4)
            )
        with open(files+"Q_v_3s.pickle", "wb") as file:
            pickle.dump(Q_v_3s, file)

        print("Hexaquark pressure #3 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols=100
        ):
            H_v_3s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**4)
            )
        with open(files+"H_v_3s.pickle", "wb") as file:
            pickle.dump(H_v_3s, file)
    else:
        with open(files+"pi_v_3s.pickle", "rb") as file:
            pi_v_3s = pickle.load(file)
        with open(files+"K_v_3s.pickle", "rb") as file:
            K_v_3s = pickle.load(file)
        with open(files+"rho_v_3s.pickle", "rb") as file:
            rho_v_3s = pickle.load(file)
        with open(files+"omega_v_3s.pickle", "rb") as file:
            omega_v_3s = pickle.load(file)
        with open(files+"D_v_3s.pickle", "rb") as file:
            D_v_3s = pickle.load(file)
        with open(files+"N_v_3s.pickle", "rb") as file:
            N_v_3s = pickle.load(file)
        with open(files+"T_v_3s.pickle", "rb") as file:
            T_v_3s = pickle.load(file)
        with open(files+"F_v_3s.pickle", "rb") as file:
            F_v_3s = pickle.load(file)
        with open(files+"P_v_3s.pickle", "rb") as file:
            P_v_3s = pickle.load(file)
        with open(files+"Q_v_3s.pickle", "rb") as file:
            Q_v_3s = pickle.load(file)
        with open(files+"H_v_3s.pickle", "rb") as file:
            H_v_3s = pickle.load(file)

    if calc_4:

        print("Pion pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            pi_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**4)
            )
        with open(files+"pi_v_4s.pickle", "wb") as file:
            pickle.dump(pi_v_4s, file)

        print("Kaon pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            K_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**4)
            )
        with open(files+"K_v_4s.pickle", "wb") as file:
            pickle.dump(K_v_4s, file)

        print("Rho pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            rho_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**4)
            )
        with open(files+"rho_v_4s.pickle", "wb") as file:
            pickle.dump(rho_v_4s, file)

        print("Omega pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            omega_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**4)
            )
        with open(files+"omega_v_4s.pickle", "wb") as file:
            pickle.dump(omega_v_4s, file)

        print("Diquark pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            D_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**4)
            )
        with open(files+"D_v_4s.pickle", "wb") as file:
            pickle.dump(D_v_4s, file)

        print("Nucleon pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            N_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**4)
            )
        with open(files+"N_v_4s.pickle", "wb") as file:
            pickle.dump(N_v_4s, file)

        print("Tetraquark pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            T_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**4)
            )
        with open(files+"T_v_4s.pickle", "wb") as file:
            pickle.dump(T_v_4s, file)

        print("F-quark pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            F_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**4)
            )
        with open(files+"F_v_4s.pickle", "wb") as file:
            pickle.dump(F_v_4s, file)

        print("Pentaquark pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            P_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**4)
            )
        with open(files+"P_v_4s.pickle", "wb") as file:
            pickle.dump(P_v_4s, file)

        print("Q-quark pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            Q_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**4)
            )
        with open(files+"Q_v_4s.pickle", "wb") as file:
            pickle.dump(Q_v_4s, file)

        print("Hexaquark pressure #4 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols=100
        ):
            H_v_4s.append(
                cluster_s.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**4)
            )
        with open(files+"H_v_4s.pickle", "wb") as file:
            pickle.dump(H_v_4s, file)
    else:
        with open(files+"pi_v_4s.pickle", "rb") as file:
            pi_v_4s = pickle.load(file)
        with open(files+"K_v_4s.pickle", "rb") as file:
            K_v_4s = pickle.load(file)
        with open(files+"rho_v_4s.pickle", "rb") as file:
            rho_v_4s = pickle.load(file)
        with open(files+"omega_v_4s.pickle", "rb") as file:
            omega_v_4s = pickle.load(file)
        with open(files+"D_v_4s.pickle", "rb") as file:
            D_v_4s = pickle.load(file)
        with open(files+"N_v_4s.pickle", "rb") as file:
            N_v_4s = pickle.load(file)
        with open(files+"T_v_4s.pickle", "rb") as file:
            T_v_4s = pickle.load(file)
        with open(files+"F_v_4s.pickle", "rb") as file:
            F_v_4s = pickle.load(file)
        with open(files+"P_v_4s.pickle", "rb") as file:
            P_v_4s = pickle.load(file)
        with open(files+"Q_v_4s.pickle", "rb") as file:
            Q_v_4s = pickle.load(file)
        with open(files+"H_v_4s.pickle", "rb") as file:
            H_v_4s = pickle.load(file)

    qgp_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1,gluon_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]

    cluster_1 = [
        sum(el) for el in 
            zip(
                pi_v_1, K_v_1, rho_v_1, omega_v_1, D_v_1, N_v_1,
                T_v_1, F_v_1, P_v_1, Q_v_1, H_v_1
            )
    ]

    total_1 = [sum(el) for el in zip(qgp_1, cluster_1)]

    cluster_1s = [
        sum(el) for el in 
            zip(
                pi_v_1s, K_v_1s, rho_v_1s, omega_v_1s, D_v_1s, N_v_1s,
                T_v_1s, F_v_1s, P_v_1s, Q_v_1s, H_v_1s
            )
    ]

    total_1s = [sum(el) for el in zip(qgp_1, cluster_1s)]

    qgp_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2,gluon_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]

    cluster_2 = [
        sum(el) for el in 
            zip(
                pi_v_2, K_v_2, rho_v_2, omega_v_2, D_v_2, N_v_2,
                T_v_2, F_v_2, P_v_2, Q_v_2, H_v_2
            )
    ]

    cluster_2s = [
        sum(el) for el in 
            zip(
                pi_v_2s, K_v_2s, rho_v_2s, omega_v_2s, D_v_2s, N_v_2s,
                T_v_2s, F_v_2s, P_v_2s, Q_v_2s, H_v_2s
            )
    ]

    total_2 = [sum(el) for el in zip(qgp_2, cluster_2)]

    total_2s = [sum(el) for el in zip(qgp_2, cluster_2s)]

    qgp_3 = [
        sum(el) for el in 
            zip(
                sigma_v_3,gluon_v_3, sea_u_v_3, sea_d_v_3, sea_s_v_3,
                perturbative_u_v_3, perturbative_d_v_3, perturbative_s_v_3,
                perturbative_gluon_v_3, pnjl_u_v_3, pnjl_d_v_3, pnjl_s_v_3
            )
    ]

    cluster_3 = [
        sum(el) for el in 
            zip(
                pi_v_3, K_v_3, rho_v_3, omega_v_3, D_v_3, N_v_3,
                T_v_3, F_v_3, P_v_3, Q_v_3, H_v_3
            )
    ]

    cluster_3s = [
        sum(el) for el in 
            zip(
                pi_v_3s, K_v_3s, rho_v_3s, omega_v_3s, D_v_3s, N_v_3s,
                T_v_3s, F_v_3s, P_v_3s, Q_v_3s, H_v_3s
            )
    ]

    total_3 = [sum(el) for el in zip(qgp_3, cluster_3)]

    total_3s = [sum(el) for el in zip(qgp_3, cluster_3s)]

    qgp_4 = [
        sum(el) for el in 
            zip(
                sigma_v_4, gluon_v_4, sea_u_v_4, sea_d_v_4, sea_s_v_4,
                perturbative_u_v_4, perturbative_d_v_4, perturbative_s_v_4,
                perturbative_gluon_v_4, pnjl_u_v_4, pnjl_d_v_4, pnjl_s_v_4
            )
    ]

    cluster_4 = [
        sum(el) for el in 
            zip(
                pi_v_4, K_v_4, rho_v_4, omega_v_4, D_v_4, N_v_4,
                T_v_4, F_v_4, P_v_4, Q_v_4, H_v_4
            )
    ]

    cluster_4s = [
        sum(el) for el in 
            zip(
                pi_v_4s, K_v_4s, rho_v_4s, omega_v_4s, D_v_4s, N_v_4s,
                T_v_4s, F_v_4s, P_v_4s, Q_v_4s, H_v_4s
            )
    ]

    total_4 = [sum(el) for el in zip(qgp_4, cluster_4)]

    total_4s = [sum(el) for el in zip(qgp_4, cluster_4s)]

    with open(
        "D:/EoS/epja/lattice_data_pickled/bazavov_1407_6387_mu0.pickle",
        "rb"
    ) as file:
        bazavov_1407_6387_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1309_5258_mu0.pickle",
        "rb"
    ) as file:
        borsanyi_1309_5258_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/bazavov_1710_05024_mu0.pickle",
        "rb"
    ) as file:
        bazavov_1710_05024_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu0.pickle",
        "rb"
    ) as file:
        borsanyi_1204_6710v2_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu200.pickle",
        "rb") as file:
        borsanyi_1204_6710v2_mu200 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu400.pickle",
        "rb") as file:
        borsanyi_1204_6710v2_mu400 = pickle.load(file)

    fig1 = matplotlib.pyplot.figure(num=1, figsize=(10.5, 10))
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.axis([10., 540., 0., 4.2])

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

    ax1.plot(T_1, total_1, '-', c='black')
    ax1.plot(T_1, total_1s, '-', c='black')
    ax1.fill_between(T_1, total_1, y2=total_1s, color='black', alpha=0.7)
    ax1.plot(T_1, cluster_1, '--', c='red')
    ax1.plot(T_1, cluster_1s, '--', c='red')
    ax1.fill_between(T_1, cluster_1, y2=cluster_1s, color='red', alpha=0.7)
    
    ax1.plot(T_1, qgp_1, '--', c='blue')

    ax1.text(180.0, 0.1, r'Clusters', color='red', fontsize=14)
    ax1.text(170.0, 0.58, r'PNJL', color='blue', fontsize=14)
    ax1.text(195.0, 1.16, r'total pressure', color='black', fontsize=14)
    ax1.text(21.0, 3.9, r'$\mathrm{\mu_B=0}$', color='black', fontsize=14)
    ax1.text(
        228.0, 1.8, r'Borsanyi et al. (2012)', color='blue',
        alpha=0.7, fontsize=14
    )
    ax1.text(
        175.0, 3.95, r'Borsanyi et al. (2014)', color='green',
        alpha=0.7, fontsize=14
    )
    ax1.text(
        30.0, 3.4, r'Bazavov et al. (2014)', color='red',
        alpha=0.7, fontsize=14
    )

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.axis([10., 2000., 0., 5.0])

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

    ax2.plot(T_2, total_2, '-', c='black')
    ax2.plot(T_2, total_2s, '-', c='black')
    ax2.fill_between(T_2, total_2, y2=total_2s, color='black', alpha=0.7)
    ax2.plot(T_2, cluster_2, '--', c='red')
    ax2.plot(T_2, cluster_2s, '--', c='red')
    ax2.fill_between(T_2, cluster_2, y2=cluster_2s, color='red', alpha=0.7)
    ax2.plot(T_2, qgp_2, '--', c='blue')

    ax2.text(220.0, 0.1, r'Clusters', color='red', fontsize=14)
    ax2.text(210.0, 0.58, r'PNJL', color='blue', fontsize=14)
    ax2.text(220.0, 1.16, r'total pressure', color='black', fontsize=14)
    ax2.text(65.0, 4.65, r'$\mathrm{\mu_B=0}$', color='black', fontsize=14)
    ax2.text(
        260.0, 2.0, r'Borsanyi et al. (2012)', color='blue',
        alpha=0.7, fontsize=14
    )
    ax2.text(
        435.0, 3.28, r'Borsanyi et al. (2014)', color='green',
        alpha=0.7, fontsize=14
    )
    ax2.text(
        350.0, 2.66, r'Bazavov et al. (2014)', color='red',
        alpha=0.7, fontsize=14
    )
    ax2.text(
        850.0, 4.15, r'Bazavov et al. (2018)', color='magenta',
        alpha=0.7, fontsize=14
    )

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.axis([10., 540., 0., 4.2])

    ax3.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1204_6710v2_mu200, closed=True, fill=True,
            color='blue', alpha=0.3
        )
    )

    ax3.plot(T_3, total_3, '-', c='black')
    ax3.plot(T_3, total_3s, '-', c='black')
    ax3.fill_between(T_3, total_3, y2=total_3s, color='black', alpha=0.7)
    ax3.plot(T_3, cluster_3, '--', c='red')
    ax3.plot(T_3, cluster_3s, '--', c='red')
    ax3.fill_between(T_3, cluster_3, y2=cluster_3s, color='red', alpha=0.7)
    ax3.plot(T_3, qgp_3, '--', c='blue')

    ax3.text(180.0, 0.1, r'Clusters', color='red', fontsize=14)
    ax3.text(170.0, 0.58, r'PNJL', color='blue', fontsize=14)
    ax3.text(195.0, 1.16, r'total pressure', color='black', fontsize=14)
    ax3.text(
        21.0, 3.9, r'$\mathrm{\mu_B=200}$ MeV', color='black',
        fontsize=14
    )
    ax3.text(
        228.0, 1.8, r'Borsanyi et al. (2012)', color='blue',
        alpha=0.7, fontsize=14
    )

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    ax4 = fig1.add_subplot(2, 2, 4)
    ax4.axis([10., 540., 0., 4.2])

    ax4.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1204_6710v2_mu400, closed=True, fill=True,
            color='blue', alpha=0.3
        )
    )
    
    ax4.plot(T_4, total_4, '-', c='black')
    ax4.plot(T_4, total_4s, '-', c='black')
    ax4.fill_between(T_4, total_4, y2=total_4s, color='black', alpha=0.7)
    ax4.plot(T_4, cluster_4, '--', c='red')
    ax4.plot(T_4, cluster_4s, '--', c='red')
    ax4.fill_between(T_4, cluster_4, y2=cluster_4s, color='red', alpha=0.7)
    ax4.plot(T_4, qgp_4, '--', c='blue')

    ax4.text(180.0, 0.1, r'Clusters', color='red', fontsize=14)
    ax4.text(170.0, 0.58, r'PNJL', color='blue', fontsize=14)
    ax4.text(195.0, 1.16, r'total pressure', color='black', fontsize=14)
    ax4.text(
        21.0, 3.9, r'$\mathrm{\mu_B=400}$ MeV', color='black',
        fontsize=14
    )
    ax4.text(
        205.0, 1.8, r'Borsanyi et al. (2012)', color='blue',
        alpha=0.7, fontsize=14
    )

    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax4.set_xlabel(r'T [MeV]', fontsize = 16)
    ax4.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure7():

    import tqdm
    import numpy
    import pickle
    import warnings
    import platform

    import matplotlib.pyplot
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")
    
    calc_1 = False
    calc_2 = False

    files = "D:/EoS/epja/figure7/"
    lattice = "D:/EoS/epja/lattice_data_pickled/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/figure7/"
        lattice = "/home/mcierniak/Data/2023_epja/lattice_data_pickled/"

    T_1 = numpy.linspace(1.0, 400.0, 200)
    T_2 = numpy.linspace(1.0, 400.0, 200)

    mu_1 = [0.0 / 3.0 for el in T_1]
    mu_2 = [0.0 / 3.0 for el in T_2]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()
    phi_re_v_2 = [1.0 for _ in T_2]
    phi_im_v_2 = [0.0 for _ in T_2]

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

    if calc_1:

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
                pnjl.thermo.gcp_sigma_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(
                T_el, mu_el, 'l'
            )
            sea_u_v_1.append(lq_temp/(T_el**4))
            sea_d_v_1.append(lq_temp/(T_el**4))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
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

    if calc_2:

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
                pnjl.thermo.gcp_sigma_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(
                T_el, mu_el, 'l'
            )
            sea_u_v_2.append(lq_temp/(T_el**4))
            sea_d_v_2.append(lq_temp/(T_el**4))
            sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
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

    if calc_1:

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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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

    if calc_2:

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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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

    norm_QGP_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_1, total_1)
    ]
    norm_QGP_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_1, total_1s)
    ]
    norm_pi_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(pi_v_1, total_1)
    ]
    norm_pi_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(pi_v_1s, total_1s)
    ]
    norm_K_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(K_v_1, total_1)
    ]
    norm_K_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(K_v_1s, total_1s)
    ]
    norm_rho_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(rho_v_1, total_1)
    ]
    norm_rho_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(rho_v_1s, total_1s)
    ]
    norm_omega_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(omega_v_1, total_1)
    ]
    norm_omega_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(omega_v_1s, total_1s)
    ]
    norm_D_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_1, total_1)
    ]
    norm_D_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_1s, total_1s)
    ]
    norm_N_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_1, total_1)
    ]
    norm_N_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_1s, total_1s)
    ]
    norm_T_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(T_v_1, total_1)
    ]
    norm_T_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(T_v_1s, total_1s)
    ]
    norm_F_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_1, total_1)
    ]
    norm_F_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_1s, total_1s)
    ]
    norm_P_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_1, total_1)
    ]
    norm_P_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_1s, total_1s)
    ]
    norm_Q_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_1, total_1)
    ]
    norm_Q_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_1s, total_1s)
    ]
    norm_H_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_1, total_1)
    ]
    norm_H_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_1s, total_1s)
    ]

    norm_QGP_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_2, total_2)
    ]
    norm_QGP_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_2, total_2s)
    ]
    norm_pi_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(pi_v_2, total_2)
    ]
    norm_pi_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(pi_v_2s, total_2s)
    ]
    norm_K_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(K_v_2, total_2)
    ]
    norm_K_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(K_v_2s, total_2s)
    ]
    norm_rho_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(rho_v_2, total_2)
    ]
    norm_rho_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(rho_v_2s, total_2s)
    ]
    norm_omega_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(omega_v_2, total_2)
    ]
    norm_omega_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(omega_v_2s, total_2s)
    ]
    norm_D_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_2, total_2)
    ]
    norm_D_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_2s, total_2s)
    ]
    norm_N_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_2, total_2)
    ]
    norm_N_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_2s, total_2s)
    ]
    norm_T_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(T_v_2, total_2)
    ]
    norm_T_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(T_v_2s, total_2s)
    ]
    norm_F_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_2, total_2)
    ]
    norm_F_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_2s, total_2s)
    ]
    norm_P_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_2, total_2)
    ]
    norm_P_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_2s, total_2s)
    ]
    norm_Q_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_2, total_2)
    ]
    norm_Q_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_2s, total_2s)
    ]
    norm_H_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_2, total_2)
    ]
    norm_H_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_2s, total_2s)
    ]

    fig1 = matplotlib.pyplot.figure(num=1, figsize=(12.0, 10))
    ax1 = fig1.add_subplot(2, 2, 1)
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

    ax1.fill_between(
        T_1, total_cluster_1, y2=total_cluster_1s, color='green', alpha=0.7
    )
    ax1.plot(T_1, total_cluster_1, '--', c = 'green')
    ax1.plot(T_1, total_cluster_1s, '--', c = 'green')
    ax1.fill_between(
        T_1, total_ccluster_1, y2=total_ccluster_1s, color='red', alpha=0.7
    )
    ax1.plot(T_1, total_ccluster_1, '--', c = 'red')
    ax1.plot(T_1, total_ccluster_1s, '--', c = 'red')
    ax1.fill_between(T_1, total_1, y2=total_1s, color='black', alpha=0.7)
    ax1.plot(T_1, total_1, '-', c = 'black')
    ax1.plot(T_1, total_1s, '-', c = 'black')
    ax1.plot(T_1, total_qgp_1, '--', c = 'blue')

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

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.axis([20., 265., 1e-6, 1e1])

    ax2.plot(T_1, [1.0 for el in T_1], '-', c='black')
    ax2.fill_between(T_1, norm_QGP_1, y2=norm_QGP_1s, color='blue', alpha=0.7)
    ax2.plot(T_1, norm_QGP_1, '-', c='blue')
    ax2.plot(T_1, norm_QGP_1s, '-', c='blue')
    ax2.fill_between(T_1, norm_pi_1, y2=norm_pi_1s, color='#653239', alpha=0.7)
    ax2.plot(T_1, norm_pi_1, '-', c='#653239')
    ax2.plot(T_1, norm_pi_1s, '-', c = '#653239')
    ax2.fill_between(T_1, norm_rho_1, y2=norm_rho_1s, color='#858AE3', alpha=0.7)
    ax2.plot(T_1, norm_rho_1, '-', c = '#858AE3')
    ax2.plot(T_1, norm_rho_1s, '-', c = '#858AE3')
    ax2.fill_between(T_1, norm_omega_1, y2=norm_omega_1s, color='#FF37A6', alpha=0.7)
    ax2.plot(T_1, norm_omega_1, '-', c='#FF37A6')
    ax2.plot(T_1, norm_omega_1s, '-', c='#FF37A6')
    ax2.fill_between(T_1, norm_K_1, y2=norm_K_1s, color='red', alpha=0.7)
    ax2.plot(T_1, norm_K_1, '-', c='red')
    ax2.plot(T_1, norm_K_1s, '-', c='red')
    ax2.fill_between(T_1, norm_D_1, y2=norm_D_1s, color='#4CB944', alpha=0.7)
    ax2.plot(T_1, norm_D_1, '--', c='#4CB944')
    ax2.plot(T_1, norm_D_1s, '--', c='#4CB944')
    ax2.fill_between(T_1, norm_N_1, y2=norm_N_1s, color='#DEA54B', alpha=0.7)
    ax2.plot(T_1, norm_N_1, '-', c='#DEA54B')
    ax2.plot(T_1, norm_N_1s, '-', c='#DEA54B')
    ax2.fill_between(T_1, norm_T_1, y2=norm_T_1s, color='#23CE6B', alpha=0.7)
    ax2.plot(T_1, norm_T_1, '-', c='#23CE6B')
    ax2.plot(T_1, norm_T_1s, '-', c='#23CE6B')
    ax2.fill_between(T_1, norm_F_1, y2=norm_F_1s, color='#DB222A', alpha=0.7)
    ax2.plot(T_1, norm_F_1, '--', c='#DB222A')
    ax2.plot(T_1, norm_F_1s, '--', c='#DB222A')
    ax2.fill_between(T_1, norm_P_1, y2=norm_P_1s, color='#78BC61', alpha=0.7)
    ax2.plot(T_1, norm_P_1, '-', c = '#78BC61')
    ax2.plot(T_1, norm_P_1s, '-', c = '#78BC61')
    ax2.fill_between(T_1, norm_Q_1, y2=norm_Q_1s, color='#55DBCB', alpha=0.7)
    ax2.plot(T_1, norm_Q_1, '--', c = '#55DBCB')
    ax2.plot(T_1, norm_Q_1s, '--', c = '#55DBCB')
    ax2.fill_between(T_1, norm_H_1, y2=norm_H_1s, color='#A846A0', alpha=0.7)
    ax2.plot(T_1, norm_H_1, '-', c = '#A846A0')
    ax2.plot(T_1, norm_H_1s, '-', c = '#A846A0')

    ax2.text(
        198, 3e-6, r'$\mathrm{\pi}$', color='#653239', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        48.0, 0.008, r'K', color='red', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        135.0, 6.0e-5, r'H', color='#A846A0', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        55.0, 0.002, r'$\mathrm{\omega}$', color='#FF37A6', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        73.0, 1.4e-5, r'D', color='#4CB944', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.001', fc='white', ec='none')
    )
    ax2.text(
        104.0, 0.0035, r'N', color='#DEA54B', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')
    )
    ax2.text(
        81.0, 0.0002, r'T', color='#23CE6B', fontsize=16,
        bbox=dict(boxstyle='square,pad=-0.1', fc='white', ec='none')
    )
    ax2.text(
        108.0, 1.5e-5, r'F', color='#DB222A', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        113.0, 0.00015, r'P', color='#78BC61', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        108.0, 2.2e-6, r'Q', color='#55DBCB', fontsize=16,
        bbox=dict(boxstyle='square,pad=0', fc='white', ec='none')
    )
    ax2.text(
        77.0, 0.015, r'$\mathrm{\rho}$', color='#858AE3', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(22., 1.5, r'total pressure', color='black', fontsize=14)
    ax2.text(225., 1.5, r'PNJL', color = 'blue', fontsize=14)

    ax2.set_yscale('log')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize=16)
    ax2.set_ylabel(r'$\mathrm{\log~p}$', fontsize=16)

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.axis([10., 400., 0., 4.2])

    ax3.add_patch(
        matplotlib.patches.Polygon(
            bazavov_1407_6387_mu0, closed=True, fill=True, color='red', alpha=0.3
        )
    )
    ax3.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1309_5258_mu0, closed=True, fill=True, color='green', alpha=0.3
        )
    )
    ax3.add_patch(
        matplotlib.patches.Polygon(
            bazavov_1710_05024_mu0, closed=True, fill=True, color='magenta', alpha=0.3
        )
    )
    ax3.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1204_6710v2_mu0, closed=True, fill=True, color='blue', alpha=0.3
        )
    )

    ax3.fill_between(T_2, total_cluster_2, y2=total_cluster_2s, color='green', alpha=0.7)
    ax3.plot(T_2, total_cluster_2, '--', color='green')
    ax3.plot(T_2, total_cluster_2s, '--', color='green')
    ax3.fill_between(T_2, total_ccluster_2, y2=total_ccluster_2s, color='red', alpha=0.7)
    ax3.plot(T_2, total_ccluster_2, '--', color='red')
    ax3.plot(T_2, total_ccluster_2s, '--', color='red')
    ax3.fill_between(T_2, total_2, y2=total_2s, color='black', alpha=0.7)
    ax3.plot(T_2, total_2, '-', c='black')
    ax3.plot(T_2, total_2s, '-', c='black')
    ax3.plot(T_2, total_qgp_2, '--', c='blue')

    ax3.text(
        180.0, 0.1, r'Color charged clusters', color='red', fontsize=14
    )
    ax3.text(
        165.0, 0.34, r'Color singlet clusters', color='green', fontsize=14
    )
    ax3.text(
        170.0, 0.58, r'NJL', color='blue', fontsize=14
    )
    ax3.text(
        20.0, 1.5, r'total pressure', color='black', fontsize=14
    )
    ax3.text(
        21.0, 3.9, r'$\mathrm{\mu_B=0}$', color='black', fontsize=14
    )
    ax3.text(
        18.0, 2.2, r'Borsanyi et al. (2012)', color='blue',
        alpha=0.7, fontsize=14
    )
    ax3.text(
        175.0, 3.7, r'Borsanyi et al. (2014)', color='green',
        alpha=0.7, fontsize=14
    )
    ax3.text(
        75.0, 3.0, r'Bazavov et al. (2014)', color='red', alpha=0.7,
        fontsize=14
    )

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    #ax3.yaxis.set_ticklabels([])
    ax3.set_xlabel(r'T [MeV]', fontsize=16)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_ylabel(r'$\mathrm{p/T^4}$', fontsize=16)

    ax4 = fig1.add_subplot(2, 2, 4)
    ax4.axis([20., 265., 1e-6, 1e1])

    ax4.plot(T_2, [1.0 for el in T_2], '-', c = 'black')
    ax4.fill_between(T_2, norm_QGP_2, y2=norm_QGP_2s, color='blue', alpha=0.7)
    ax4.plot(T_2, norm_QGP_2, '-', c='blue')
    ax4.plot(T_2, norm_QGP_2s, '-', c='blue')
    ax4.fill_between(T_2, norm_pi_2, y2=norm_pi_2s, color='#653239', alpha=0.7)
    ax4.plot(T_2, norm_pi_2, '-', c='#653239')
    ax4.plot(T_2, norm_pi_2s, '-', c='#653239')
    ax4.fill_between(T_2, norm_rho_2, y2=norm_rho_2s, color='#858AE3', alpha=0.7)
    ax4.plot(T_2, norm_rho_2, '-', c='#858AE3')
    ax4.plot(T_2, norm_rho_2s, '-', c='#858AE3')
    ax4.fill_between(T_2, norm_omega_2, y2=norm_omega_2s, color='#FF37A6', alpha=0.7)
    ax4.plot(T_2, norm_omega_2, '-', c='#FF37A6')
    ax4.plot(T_2, norm_omega_2s, '-', c='#FF37A6')
    ax4.fill_between(T_2, norm_K_2, y2=norm_K_2s, color='red', alpha=0.7)
    ax4.plot(T_2, norm_K_2, '-', c='red')
    ax4.plot(T_2, norm_K_2s, '-', c='red')
    ax4.fill_between(T_2, norm_D_2, y2=norm_D_2s, color='#4CB944', alpha=0.7)
    ax4.plot(T_2, norm_D_2, '--', c='#4CB944')
    ax4.plot(T_2, norm_D_2s, '--', c='#4CB944')
    ax4.fill_between(T_2, norm_N_2, y2=norm_N_2s, color='#DEA54B', alpha=0.7)
    ax4.plot(T_2, norm_N_2, '-', c='#DEA54B')
    ax4.plot(T_2, norm_N_2s, '-', c='#DEA54B')
    ax4.fill_between(T_2, norm_T_2, y2=norm_T_2s, color='#23CE6B', alpha=0.7)
    ax4.plot(T_2, norm_T_2, '-', c='#23CE6B')
    ax4.plot(T_2, norm_T_2s, '-', c='#23CE6B')
    ax4.fill_between(T_2, norm_F_2, y2=norm_F_2s, color='#DB222A', alpha=0.7)
    ax4.plot(T_2, norm_F_2, '--', c='#DB222A')
    ax4.plot(T_2, norm_F_2s, '--', c='#DB222A')
    ax4.fill_between(T_2, norm_P_2, y2=norm_P_2s, color='#78BC61', alpha=0.7)
    ax4.plot(T_2, norm_P_2, '-', c='#78BC61')
    ax4.plot(T_2, norm_P_2s, '-', c='#78BC61')
    ax4.fill_between(T_2, norm_Q_2, y2=norm_Q_2s, color='#55DBCB', alpha=0.7)
    ax4.plot(T_2, norm_Q_2, '--', c='#55DBCB')
    ax4.plot(T_2, norm_Q_2s, '--', c='#55DBCB')
    ax4.fill_between(T_2, norm_H_2, y2=norm_H_2s, color='#A846A0', alpha=0.7)
    ax4.plot(T_2, norm_H_2, '-', c='#A846A0')
    ax4.plot(T_2, norm_H_2s, '-', c='#A846A0')

    ax4.text(
        198, 3e-6, r'$\mathrm{\pi}$', color='#653239', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        48.0, 0.005, r'K', color='red', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        135.0, 2.5e-5, r'H', color='#A846A0', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        55.0, 0.0015, r'$\mathrm{\omega}$', color='#FF37A6', fontsize=16,
        bbox=dict(boxstyle='square,pad=-0.05', fc='white', ec='none')
    )
    ax4.text(
        100., 0.017, r'D', color='#4CB944',fontsize=16,
        bbox=dict(boxstyle='square,pad=0.001', fc='white', ec='none')
    )
    ax4.text(
        74.0, 0.00014, r'N', color='#DEA54B', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        110.0, 0.0008, r'T', color='#23CE6B', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.01', fc='white', ec='none')
    )
    ax4.text(
        71.0, 2.3e-6, r'F', color='#DB222A', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        102.0, 1.9e-5, r'P', color='#78BC61', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        120.0, 9.5e-5, r'Q', color='#55DBCB', fontsize=16,
        bbox=dict(boxstyle='square,pad=0', fc='white', ec='none')
    )
    ax4.text(
        77.0, 0.008, r'$\mathrm{\rho}$', color='#858AE3', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        22., 1.5, r'total pressure', color='black', fontsize=14
    )
    ax4.text(
        225., 1.5, r'NJL', color='blue', fontsize=14
    )

    ax4.set_yscale('log')
    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    #ax4.yaxis.set_ticklabels([])
    ax4.set_xlabel(r'T [MeV]', fontsize=16)
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax4.set_ylabel(r'$\mathrm{\log~p}$', fontsize=16)

    fig1.tight_layout(pad=0.1)
    #fig1.subplots_adjust(left=0.093, right=0.98, top=0.995, bottom=0.05, wspace=0.3, hspace=0.2)

    #matplotlib.pyplot.savefig("C:/Users/Mateusz Cierniak/Desktop/figure_7_new.png")
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure7_alt():

    import tqdm
    import numpy
    import pickle
    import warnings
    import platform

    import matplotlib.pyplot
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")
    
    calc_1 = True
    calc_2 = True

    files = "D:/EoS/epja/figure7_alt/"
    lattice = "D:/EoS/epja/lattice_data_pickled/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/figure7_alt/"
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
                pnjl.thermo.gcp_sigma_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(
                T_el, mu_el, 'l'
            )
            sea_u_v_1.append(lq_temp/(T_el**4))
            sea_d_v_1.append(lq_temp/(T_el**4))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
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
                pnjl.thermo.gcp_sigma_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(
                T_el, mu_el, 'l'
            )
            sea_u_v_2.append(lq_temp/(T_el**4))
            sea_d_v_2.append(lq_temp/(T_el**4))
            sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
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


def epja_figure8():

    import tqdm
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")
    
    calc_1 = False
    calc_2 = False

    files = "D:/EoS/epja/figure8/"

    T_1 = numpy.linspace(1.0, 400.0, 200)
    T_2 = numpy.linspace(1.0, 400.0, 200)

    mu_1 = [200.0 / 3.0 for el in T_1]
    mu_2 = [200.0 / 3.0 for el in T_2]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()
    phi_re_v_2 = [1.0 for _ in T_2]
    phi_im_v_2 = [0.0 for _ in T_2]

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

    if calc_1:

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
                pnjl.thermo.gcp_sigma_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(
                T_el, mu_el, 'l'
            )
            sea_u_v_1.append(lq_temp/(T_el**4))
            sea_d_v_1.append(lq_temp/(T_el**4))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
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

    if calc_2:

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
                pnjl.thermo.gcp_sigma_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(
                T_el, mu_el, 'l'
            )
            sea_u_v_2.append(lq_temp/(T_el**4))
            sea_d_v_2.append(lq_temp/(T_el**4))
            sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
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
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
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

    if calc_1:

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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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

    if calc_2:

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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu200.pickle",
        "rb"
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

    norm_QGP_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_1, total_1)
    ]
    norm_QGP_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_1, total_1s)
    ]
    norm_pi_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(pi_v_1, total_1)
    ]
    norm_pi_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(pi_v_1s, total_1s)
    ]
    norm_K_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(K_v_1, total_1)
    ]
    norm_K_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(K_v_1s, total_1s)
    ]
    norm_rho_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(rho_v_1, total_1)
    ]
    norm_rho_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(rho_v_1s, total_1s)
    ]
    norm_omega_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(omega_v_1, total_1)
    ]
    norm_omega_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(omega_v_1s, total_1s)
    ]
    norm_D_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_1, total_1)
    ]
    norm_D_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_1s, total_1s)
    ]
    norm_N_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_1, total_1)
    ]
    norm_N_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_1s, total_1s)
    ]
    norm_T_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(T_v_1, total_1)
    ]
    norm_T_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(T_v_1s, total_1s)
    ]
    norm_F_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_1, total_1)
    ]
    norm_F_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_1s, total_1s)
    ]
    norm_P_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_1, total_1)
    ]
    norm_P_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_1s, total_1s)
    ]
    norm_Q_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_1, total_1)
    ]
    norm_Q_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_1s, total_1s)
    ]
    norm_H_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_1, total_1)
    ]
    norm_H_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_1s, total_1s)
    ]

    norm_QGP_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_2, total_2)
    ]
    norm_QGP_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_2, total_2s)
    ]
    norm_pi_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(pi_v_2, total_2)
    ]
    norm_pi_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(pi_v_2s, total_2s)
    ]
    norm_K_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(K_v_2, total_2)
    ]
    norm_K_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(K_v_2s, total_2s)
    ]
    norm_rho_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(rho_v_2, total_2)
    ]
    norm_rho_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(rho_v_2s, total_2s)
    ]
    norm_omega_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(omega_v_2, total_2)
    ]
    norm_omega_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(omega_v_2s, total_2s)
    ]
    norm_D_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_2, total_2)
    ]
    norm_D_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_2s, total_2s)
    ]
    norm_N_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_2, total_2)
    ]
    norm_N_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_2s, total_2s)
    ]
    norm_T_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(T_v_2, total_2)
    ]
    norm_T_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(T_v_2s, total_2s)
    ]
    norm_F_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_2, total_2)
    ]
    norm_F_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_2s, total_2s)
    ]
    norm_P_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_2, total_2)
    ]
    norm_P_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_2s, total_2s)
    ]
    norm_Q_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_2, total_2)
    ]
    norm_Q_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_2s, total_2s)
    ]
    norm_H_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_2, total_2)
    ]
    norm_H_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_2s, total_2s)
    ]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12.0, 10))
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.axis([10., 400., 0., 4.2])

    ax1.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'blue', alpha = 0.3
        )
    )

    ax1.fill_between(
        T_1, total_cluster_1, y2=total_cluster_1s, color='green', alpha=0.7
    )
    ax1.plot(T_1, total_cluster_1, '--', c = 'green')
    ax1.plot(T_1, total_cluster_1s, '--', c = 'green')
    ax1.fill_between(
        T_1, total_ccluster_1, y2=total_ccluster_1s, color='red', alpha=0.7
    )
    ax1.plot(T_1, total_ccluster_1, '--', c = 'red')
    ax1.plot(T_1, total_ccluster_1s, '--', c = 'red')
    ax1.fill_between(T_1, total_1, y2=total_1s, color='black', alpha=0.7)
    ax1.plot(T_1, total_1, '-', c = 'black')
    ax1.plot(T_1, total_1s, '-', c = 'black')
    ax1.plot(T_1, total_qgp_1, '--', c = 'blue')

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
        21.0, 3.9, r'$\mathrm{\mu_B=200}$ MeV', color='black', fontsize=14
    )
    ax1.text(
        22.0, 2.2, r'Borsanyi et al. (2012)', color='blue',
        alpha=0.7, fontsize=14
    )

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.axis([20., 265., 1e-6, 1e1])

    ax2.plot(T_1, [1.0 for el in T_1], '-', c='black')
    ax2.fill_between(T_1, norm_QGP_1, y2=norm_QGP_1s, color='blue', alpha=0.7)
    ax2.plot(T_1, norm_QGP_1, '-', c='blue')
    ax2.plot(T_1, norm_QGP_1s, '-', c='blue')
    ax2.fill_between(T_1, norm_pi_1, y2=norm_pi_1s, color='#653239', alpha=0.7)
    ax2.plot(T_1, norm_pi_1, '-', c='#653239')
    ax2.plot(T_1, norm_pi_1s, '-', c = '#653239')
    ax2.fill_between(T_1, norm_rho_1, y2=norm_rho_1s, color='#858AE3', alpha=0.7)
    ax2.plot(T_1, norm_rho_1, '-', c = '#858AE3')
    ax2.plot(T_1, norm_rho_1s, '-', c = '#858AE3')
    ax2.fill_between(T_1, norm_omega_1, y2=norm_omega_1s, color='#FF37A6', alpha=0.7)
    ax2.plot(T_1, norm_omega_1, '-', c='#FF37A6')
    ax2.plot(T_1, norm_omega_1s, '-', c='#FF37A6')
    ax2.fill_between(T_1, norm_K_1, y2=norm_K_1s, color='red', alpha=0.7)
    ax2.plot(T_1, norm_K_1, '-', c='red')
    ax2.plot(T_1, norm_K_1s, '-', c='red')
    ax2.fill_between(T_1, norm_D_1, y2=norm_D_1s, color='#4CB944', alpha=0.7)
    ax2.plot(T_1, norm_D_1, '--', c='#4CB944')
    ax2.plot(T_1, norm_D_1s, '--', c='#4CB944')
    ax2.fill_between(T_1, norm_N_1, y2=norm_N_1s, color='#DEA54B', alpha=0.7)
    ax2.plot(T_1, norm_N_1, '-', c='#DEA54B')
    ax2.plot(T_1, norm_N_1s, '-', c='#DEA54B')
    ax2.fill_between(T_1, norm_T_1, y2=norm_T_1s, color='#23CE6B', alpha=0.7)
    ax2.plot(T_1, norm_T_1, '-', c='#23CE6B')
    ax2.plot(T_1, norm_T_1s, '-', c='#23CE6B')
    ax2.fill_between(T_1, norm_F_1, y2=norm_F_1s, color='#DB222A', alpha=0.7)
    ax2.plot(T_1, norm_F_1, '--', c='#DB222A')
    ax2.plot(T_1, norm_F_1s, '--', c='#DB222A')
    ax2.fill_between(T_1, norm_P_1, y2=norm_P_1s, color='#78BC61', alpha=0.7)
    ax2.plot(T_1, norm_P_1, '-', c = '#78BC61')
    ax2.plot(T_1, norm_P_1s, '-', c = '#78BC61')
    ax2.fill_between(T_1, norm_Q_1, y2=norm_Q_1s, color='#55DBCB', alpha=0.7)
    ax2.plot(T_1, norm_Q_1, '--', c = '#55DBCB')
    ax2.plot(T_1, norm_Q_1s, '--', c = '#55DBCB')
    ax2.fill_between(T_1, norm_H_1, y2=norm_H_1s, color='#A846A0', alpha=0.7)
    ax2.plot(T_1, norm_H_1, '-', c = '#A846A0')
    ax2.plot(T_1, norm_H_1s, '-', c = '#A846A0')

    ax2.text(
        198, 3e-6, r'$\mathrm{\pi}$', color='#653239', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        48.0, 0.008, r'K', color='red', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        135.0, 0.00055, r'H', color='#A846A0', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        55.0, 0.002, r'$\mathrm{\omega}$', color='#FF37A6', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        62.0, 1.4e-5, r'D', color='#4CB944', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.001', fc='white', ec='none')
    )
    ax2.text(
        117.6, 0.015, r'N', color='#DEA54B', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')
    )
    ax2.text(
        125.5, 0.0038, r'T', color='#23CE6B', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        95.0, 1.5e-5, r'F', color='#DB222A', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        99.0, 0.00015, r'P', color='#78BC61', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        93.2, 2.2e-6, r'Q', color='#55DBCB', fontsize=16,
        bbox=dict(boxstyle='square,pad=0', fc='white', ec='none')
    )
    ax2.text(
        77.0, 0.015, r'$\mathrm{\rho}$', color='#858AE3', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(22., 1.5, r'total pressure', color='black', fontsize=14)
    ax2.text(225., 1.5, r'PNJL', color = 'blue', fontsize = 14)

    ax2.set_yscale('log')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize=16)
    ax2.set_ylabel(r'$\mathrm{\log~p}$', fontsize=16)

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.axis([10., 400., 0., 4.2])

    ax3.add_patch(
        matplotlib.patches.Polygon(
            borsanyi_1204_6710v2_mu200, closed = True, fill = True, color = 'blue', alpha = 0.3
        )
    )

    ax3.fill_between(T_2, total_cluster_2, y2=total_cluster_2s, color='green', alpha=0.7)
    ax3.plot(T_2, total_cluster_2, '--', color='green')
    ax3.plot(T_2, total_cluster_2s, '--', color='green')
    ax3.fill_between(T_2, total_ccluster_2, y2=total_ccluster_2s, color='red', alpha=0.7)
    ax3.plot(T_2, total_ccluster_2, '--', color='red')
    ax3.plot(T_2, total_ccluster_2s, '--', color='red')
    ax3.fill_between(T_2, total_2, y2=total_2s, color='black', alpha=0.7)
    ax3.plot(T_2, total_2, '-', c='black')
    ax3.plot(T_2, total_2s, '-', c='black')
    ax3.plot(T_2, total_qgp_2, '--', c='blue')

    ax3.text(
        180.0, 0.1, r'Color charged clusters', color='red', fontsize=14
    )
    ax3.text(
        165.0, 0.34, r'Color singlet clusters', color='green', fontsize=14
    )
    ax3.text(
        170.0, 0.58, r'NJL', color='blue', fontsize=14
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
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{p/T^4}$', fontsize = 16)

    ax4 = fig1.add_subplot(2, 2, 4)
    ax4.axis([20., 265., 1e-6, 1e1])

    ax4.plot(T_2, [1.0 for el in T_2], '-', c = 'black')
    ax4.fill_between(T_2, norm_QGP_2, y2=norm_QGP_2s, color='blue', alpha=0.7)
    ax4.plot(T_2, norm_QGP_2, '-', c='blue')
    ax4.plot(T_2, norm_QGP_2s, '-', c='blue')
    ax4.fill_between(T_2, norm_pi_2, y2=norm_pi_2s, color='#653239', alpha=0.7)
    ax4.plot(T_2, norm_pi_2, '-', c='#653239')
    ax4.plot(T_2, norm_pi_2s, '-', c='#653239')
    ax4.fill_between(T_2, norm_rho_2, y2=norm_rho_2s, color='#858AE3', alpha=0.7)
    ax4.plot(T_2, norm_rho_2, '-', c='#858AE3')
    ax4.plot(T_2, norm_rho_2s, '-', c='#858AE3')
    ax4.fill_between(T_2, norm_omega_2, y2=norm_omega_2s, color='#FF37A6', alpha=0.7)
    ax4.plot(T_2, norm_omega_2, '-', c='#FF37A6')
    ax4.plot(T_2, norm_omega_2s, '-', c='#FF37A6')
    ax4.fill_between(T_2, norm_K_2, y2=norm_K_2s, color='red', alpha=0.7)
    ax4.plot(T_2, norm_K_2, '-', c='red')
    ax4.plot(T_2, norm_K_2s, '-', c='red')
    ax4.fill_between(T_2, norm_D_2, y2=norm_D_2s, color='#4CB944', alpha=0.7)
    ax4.plot(T_2, norm_D_2, '--', c='#4CB944')
    ax4.plot(T_2, norm_D_2s, '--', c='#4CB944')
    ax4.fill_between(T_2, norm_N_2, y2=norm_N_2s, color='#DEA54B', alpha=0.7)
    ax4.plot(T_2, norm_N_2, '-', c='#DEA54B')
    ax4.plot(T_2, norm_N_2s, '-', c='#DEA54B')
    ax4.fill_between(T_2, norm_T_2, y2=norm_T_2s, color='#23CE6B', alpha=0.7)
    ax4.plot(T_2, norm_T_2, '-', c='#23CE6B')
    ax4.plot(T_2, norm_T_2s, '-', c='#23CE6B')
    ax4.fill_between(T_2, norm_F_2, y2=norm_F_2s, color='#DB222A', alpha=0.7)
    ax4.plot(T_2, norm_F_2, '--', c='#DB222A')
    ax4.plot(T_2, norm_F_2s, '--', c='#DB222A')
    ax4.fill_between(T_2, norm_P_2, y2=norm_P_2s, color='#78BC61', alpha=0.7)
    ax4.plot(T_2, norm_P_2, '-', c='#78BC61')
    ax4.plot(T_2, norm_P_2s, '-', c='#78BC61')
    ax4.fill_between(T_2, norm_Q_2, y2=norm_Q_2s, color='#55DBCB', alpha=0.7)
    ax4.plot(T_2, norm_Q_2, '--', c='#55DBCB')
    ax4.plot(T_2, norm_Q_2s, '--', c='#55DBCB')
    ax4.fill_between(T_2, norm_H_2, y2=norm_H_2s, color='#A846A0', alpha=0.7)
    ax4.plot(T_2, norm_H_2, '-', c='#A846A0')
    ax4.plot(T_2, norm_H_2s, '-', c='#A846A0')

    ax4.text(
        198, 3e-6, r'$\mathrm{\pi}$', color='#653239', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        45.0, 0.005, r'K', color='red', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        135.0, 0.00017, r'H', color='#A846A0', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        55.0, 0.0015, r'$\mathrm{\omega}$', color='#FF37A6', fontsize=16,
        bbox=dict(boxstyle='square,pad=-0.05', fc='white', ec='none')
    )
    ax4.text(
        100., 0.032, r'D', color='#4CB944',fontsize=16,
        bbox=dict(boxstyle='square,pad=0.001', fc='white', ec='none')
    )
    ax4.text(
        74.0, 0.0006, r'N', color='#DEA54B', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        110.0, 0.0006, r'T', color='#23CE6B', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.01', fc='white', ec='none')
    )
    ax4.text(
        54.6, 2.3e-6, r'F', color='#DB222A', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        102.0, 5e-5, r'P', color='#78BC61', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        130.3, 0.0011, r'Q', color='#55DBCB', fontsize=16,
        bbox=dict(boxstyle='square,pad=0', fc='white', ec='none')
    )
    ax4.text(
        77.0, 0.008, r'$\mathrm{\rho}$', color='#858AE3', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        22., 1.5, r'total pressure', color='black', fontsize=14
    )
    ax4.text(
        225., 1.5, r'NJL', color='blue', fontsize=14
    )

    ax4.set_yscale('log')
    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax4.set_xlabel(r'T [MeV]', fontsize = 16)
    ax4.set_ylabel(r'$\mathrm{\log~p}$', fontsize = 16)

    fig1.tight_layout(pad=0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure9():

    import tqdm
    import numpy
    import pickle
    import warnings
    import platform

    import matplotlib.pyplot
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")
    
    calc_1 = False
    calc_2 = False

    files = "D:/EoS/epja/figure9/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/figure9/"

    T_1 = numpy.linspace(1.0, 400.0, 200)
    T_2 = numpy.linspace(1.0, 400.0, 200)

    mu_1 = [200.0 / 3.0 for el in T_1]
    mu_2 = [200.0 / 3.0 for el in T_2]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()
    phi_re_v_2 = [1.0 for _ in T_2]
    phi_im_v_2 = [0.0 for _ in T_2]

    sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1 = \
        list(), list(), list(), list()
    perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
        list(), list(), list()
    perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = \
        list(), list(), list(), list()

    sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2 = \
        list(), list(), list(), list()
    perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2 = \
        list(), list(), list()
    perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2 = \
        list(), list(), list(), list()

    D_v_1, N_v_1, F_v_1, P_v_1, Q_v_1, H_v_1 = \
        list(), list(), list(), list(), list(), list()      

    D_v_1s, N_v_1s, F_v_1s, P_v_1s, Q_v_1s, H_v_1s = \
        list(), list(), list(), list(), list(), list()
    
    D_v_2, N_v_2, F_v_2, P_v_2, Q_v_2, H_v_2 = \
        list(), list(), list(), list(), list(), list()

    D_v_2s, N_v_2s, F_v_2s, P_v_2s, Q_v_2s, H_v_2s = \
        list(), list(), list(), list(), list(), list()

    if calc_1:

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

        print("Sigma bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"sigma_v_1.pickle", "wb") as file:
            pickle.dump(sigma_v_1, file)

        print("Sea bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_1.append(lq_temp/(T_el**3))
            sea_d_v_1.append(lq_temp/(T_el**3))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"sea_u_v_1.pickle", "wb") as file:
            pickle.dump(sea_u_v_1, file)
        with open(files+"sea_d_v_1.pickle", "wb") as file:
            pickle.dump(sea_d_v_1, file)
        with open(files+"sea_s_v_1.pickle", "wb") as file:
            pickle.dump(sea_s_v_1, file)

        print("Perturbative bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            perturbative_u_v_1.append(lq_temp/(T_el**3))
            perturbative_d_v_1.append(lq_temp/(T_el**3))
            perturbative_s_v_1.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
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

        print("PNJL bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            pnjl_u_v_1.append(lq_temp/(T_el**3))
            pnjl_d_v_1.append(lq_temp/(T_el**3))
            pnjl_s_v_1.append(sq_temp/(T_el**3))
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

    if calc_2:

        with open(files+"phi_re_v_2.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)

        print("Sigma bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            sigma_v_2.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"sigma_v_2.pickle", "wb") as file:
            pickle.dump(sigma_v_2, file)

        print("Sea bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_2.append(lq_temp/(T_el**3))
            sea_d_v_2.append(lq_temp/(T_el**3))
            sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"sea_u_v_2.pickle", "wb") as file:
            pickle.dump(sea_u_v_2, file)
        with open(files+"sea_d_v_2.pickle", "wb") as file:
            pickle.dump(sea_d_v_2, file)
        with open(files+"sea_s_v_2.pickle", "wb") as file:
            pickle.dump(sea_s_v_2, file)

        print("Perturbative bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            perturbative_u_v_2.append(lq_temp/(T_el**3))
            perturbative_d_v_2.append(lq_temp/(T_el**3))
            perturbative_s_v_2.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
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

        print("PNJL bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            pnjl_u_v_2.append(lq_temp/(T_el**3))
            pnjl_d_v_2.append(lq_temp/(T_el**3))
            pnjl_s_v_2.append(sq_temp/(T_el**3))
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

    if calc_1:

        print("Diquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"D_v_1.pickle", "wb") as file:
            pickle.dump(D_v_1, file)

        print("Nucleon bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"N_v_1.pickle", "wb") as file:
            pickle.dump(N_v_1, file)

        print("F-quark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"F_v_1.pickle", "wb") as file:
            pickle.dump(F_v_1, file)

        print("Pentaquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"P_v_1.pickle", "wb") as file:
            pickle.dump(P_v_1, file)

        print("Q-quark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"Q_v_1.pickle", "wb") as file:
            pickle.dump(Q_v_1, file)

        print("Hexaquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"H_v_1.pickle", "wb") as file:
            pickle.dump(H_v_1, file)
    else:
        with open(files+"D_v_1.pickle", "rb") as file:
            D_v_1 = pickle.load(file)
        with open(files+"N_v_1.pickle", "rb") as file:
            N_v_1 = pickle.load(file)
        with open(files+"F_v_1.pickle", "rb") as file:
            F_v_1 = pickle.load(file)
        with open(files+"P_v_1.pickle", "rb") as file:
            P_v_1 = pickle.load(file)
        with open(files+"Q_v_1.pickle", "rb") as file:
            Q_v_1 = pickle.load(file)
        with open(files+"H_v_1.pickle", "rb") as file:
            H_v_1 = pickle.load(file)

    if calc_1:

        print("Diquark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"D_v_1s.pickle", "wb") as file:
            pickle.dump(D_v_1s, file)

        print("Nucleon bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"N_v_1s.pickle", "wb") as file:
            pickle.dump(N_v_1s, file)

        print("F-quark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"F_v_1s.pickle", "wb") as file:
            pickle.dump(F_v_1s, file)

        print("Pentaquark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"P_v_1s.pickle", "wb") as file:
            pickle.dump(P_v_1s, file)

        print("Q-quark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"Q_v_1s.pickle", "wb") as file:
            pickle.dump(Q_v_1s, file)

        print("Hexaquark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1s.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"H_v_1s.pickle", "wb") as file:
            pickle.dump(H_v_1s, file)
    else:
        with open(files+"D_v_1s.pickle", "rb") as file:
            D_v_1s = pickle.load(file)
        with open(files+"N_v_1s.pickle", "rb") as file:
            N_v_1s = pickle.load(file)
        with open(files+"F_v_1s.pickle", "rb") as file:
            F_v_1s = pickle.load(file)
        with open(files+"P_v_1s.pickle", "rb") as file:
            P_v_1s = pickle.load(file)
        with open(files+"Q_v_1s.pickle", "rb") as file:
            Q_v_1s = pickle.load(file)
        with open(files+"H_v_1s.pickle", "rb") as file:
            H_v_1s = pickle.load(file)

    if calc_2:

        print("Diquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"D_v_2.pickle", "wb") as file:
            pickle.dump(D_v_2, file)

        print("Nucleon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"N_v_2.pickle", "wb") as file:
            pickle.dump(N_v_2, file)

        print("F-quark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"F_v_2.pickle", "wb") as file:
            pickle.dump(F_v_2, file)

        print("Pentaquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"P_v_2.pickle", "wb") as file:
            pickle.dump(P_v_2, file)

        print("Q-quark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"Q_v_2.pickle", "wb") as file:
            pickle.dump(Q_v_2, file)

        print("Hexaquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"H_v_2.pickle", "wb") as file:
            pickle.dump(H_v_2, file)
    else:
        with open(files+"D_v_2.pickle", "rb") as file:
            D_v_2 = pickle.load(file)
        with open(files+"N_v_2.pickle", "rb") as file:
            N_v_2 = pickle.load(file)
        with open(files+"F_v_2.pickle", "rb") as file:
            F_v_2 = pickle.load(file)
        with open(files+"P_v_2.pickle", "rb") as file:
            P_v_2 = pickle.load(file)
        with open(files+"Q_v_2.pickle", "rb") as file:
            Q_v_2 = pickle.load(file)
        with open(files+"H_v_2.pickle", "rb") as file:
            H_v_2 = pickle.load(file)

    if calc_2:

        print("Diquark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"D_v_2s.pickle", "wb") as file:
            pickle.dump(D_v_2s, file)

        print("Nucleon bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"N_v_2s.pickle", "wb") as file:
            pickle.dump(N_v_2s, file)

        print("F-quark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"F_v_2s.pickle", "wb") as file:
            pickle.dump(F_v_2s, file)

        print("Pentaquark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"P_v_2s.pickle", "wb") as file:
            pickle.dump(P_v_2s, file)

        print("Q-quark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"Q_v_2s.pickle", "wb") as file:
            pickle.dump(Q_v_2s, file)

        print("Hexaquark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"H_v_2s.pickle", "wb") as file:
            pickle.dump(H_v_2s, file)
    else:
        with open(files+"D_v_2s.pickle", "rb") as file:
            D_v_2s = pickle.load(file)
        with open(files+"N_v_2s.pickle", "rb") as file:
            N_v_2s = pickle.load(file)
        with open(files+"F_v_2s.pickle", "rb") as file:
            F_v_2s = pickle.load(file)
        with open(files+"P_v_2s.pickle", "rb") as file:
            P_v_2s = pickle.load(file)
        with open(files+"Q_v_2s.pickle", "rb") as file:
            Q_v_2s = pickle.load(file)
        with open(files+"H_v_2s.pickle", "rb") as file:
            H_v_2s = pickle.load(file)

    total_qgp_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_qgp_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_cluster_1 = [
        sum(el) for el in 
            zip(N_v_1, P_v_1, H_v_1)
    ]
    total_ccluster_1 = [
        sum(el) for el in 
            zip(D_v_1, F_v_1, Q_v_1)
    ]
    total_cluster_1s = [
        sum(el) for el in 
            zip(N_v_1s, P_v_1s, H_v_1s)
    ]
    total_ccluster_1s = [
        sum(el) for el in 
            zip(D_v_1s, F_v_1s, Q_v_1s)
    ]
    total_1 = [sum(el) for el in zip(total_qgp_1, total_cluster_1, total_ccluster_1)]
    total_1s = [sum(el) for el in zip(total_qgp_1, total_cluster_1s, total_ccluster_1s)]
    total_cluster_2 = [
        sum(el) for el in 
            zip(N_v_2, P_v_2, H_v_2)
    ]
    total_cluster_2s = [
        sum(el) for el in 
            zip(N_v_2s, P_v_2s, H_v_2s)
    ]
    total_ccluster_2 = [
        sum(el) for el in 
            zip(D_v_2, F_v_2, Q_v_2)
    ]
    total_ccluster_2s = [
        sum(el) for el in 
            zip(D_v_2s, F_v_2s, Q_v_2s)
    ]
    total_2 = [sum(el) for el in zip(total_qgp_2, total_cluster_2, total_ccluster_2)]
    total_2s = [sum(el) for el in zip(total_qgp_2, total_cluster_2s, total_ccluster_2s)]

    norm_QGP_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_1, total_1)
    ]
    norm_QGP_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_1, total_1s)
    ]
    norm_D_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_1, total_1)
    ]
    norm_D_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_1s, total_1s)
    ]
    norm_N_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_1, total_1)
    ]
    norm_N_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_1s, total_1s)
    ]
    norm_F_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_1, total_1)
    ]
    norm_F_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_1s, total_1s)
    ]
    norm_P_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_1, total_1)
    ]
    norm_P_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_1s, total_1s)
    ]
    norm_Q_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_1, total_1)
    ]
    norm_Q_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_1s, total_1s)
    ]
    norm_H_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_1, total_1)
    ]
    norm_H_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_1s, total_1s)
    ]

    norm_QGP_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_2, total_2)
    ]
    norm_QGP_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_2, total_2s)
    ]
    norm_D_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_2, total_2)
    ]
    norm_D_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_2s, total_2s)
    ]
    norm_N_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_2, total_2)
    ]
    norm_N_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_2s, total_2s)
    ]
    norm_F_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_2, total_2)
    ]
    norm_F_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_2s, total_2s)
    ]
    norm_P_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_2, total_2)
    ]
    norm_P_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_2s, total_2s)
    ]
    norm_Q_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_2, total_2)
    ]
    norm_Q_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_2s, total_2s)
    ]
    norm_H_2 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_2, total_2)
    ]
    norm_H_2s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_2s, total_2s)
    ]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12.0, 10))
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.axis([10., 400., 0., 0.3])

    ax1.fill_between(
        T_1, total_cluster_1, y2=total_cluster_1s, color='green', alpha=0.7
    )
    ax1.plot(T_1, total_cluster_1, '--', c = 'green')
    ax1.plot(T_1, total_cluster_1s, '--', c = 'green')
    ax1.fill_between(
        T_1, total_ccluster_1, y2=total_ccluster_1s, color='red', alpha=0.7
    )
    ax1.plot(T_1, total_ccluster_1, '--', c = 'red')
    ax1.plot(T_1, total_ccluster_1s, '--', c = 'red')
    ax1.fill_between(T_1, total_1, y2=total_1s, color='black', alpha=0.7)
    ax1.plot(T_1, total_1, '-', c = 'black')
    ax1.plot(T_1, total_1s, '-', c = 'black')
    ax1.plot(T_1, total_qgp_1, '--', c = 'blue')

    ax1.text(
        173.3, 0.005, r'Color charged clusters', color='red', fontsize=14
    )
    ax1.text(
        165.0, 0.03, r'Color singlet clusters', color='green', fontsize=14
    )
    ax1.text(
        170.0, 0.065, r'PNJL', color='blue', fontsize=14
    )
    ax1.text(
        20.0, 0.23, r'total baryon density', color='black', fontsize=14
    )
    ax1.text(
        19.0, 0.28, r'$\mathrm{\mu_B=200}$ MeV', color='black', fontsize=14
    )

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.axis([35., 200., 1e-6, 1e1])

    ax2.plot(T_1, [1.0 for el in T_1], '-', c='black')
    ax2.fill_between(T_1, norm_QGP_1, y2=norm_QGP_1s, color='blue', alpha=0.7)
    ax2.plot(T_1, norm_QGP_1, '-', c='blue')
    ax2.plot(T_1, norm_QGP_1s, '-', c='blue')
    ax2.fill_between(T_1, norm_D_1, y2=norm_D_1s, color='#4CB944', alpha=0.7)
    ax2.plot(T_1, norm_D_1, '--', c='#4CB944')
    ax2.plot(T_1, norm_D_1s, '--', c='#4CB944')
    ax2.fill_between(T_1, norm_N_1, y2=norm_N_1s, color='#DEA54B', alpha=0.7)
    ax2.plot(T_1, norm_N_1, '-', c='#DEA54B')
    ax2.plot(T_1, norm_N_1s, '-', c='#DEA54B')
    ax2.fill_between(T_1, norm_F_1, y2=norm_F_1s, color='#DB222A', alpha=0.7)
    ax2.plot(T_1, norm_F_1, '--', c='#DB222A')
    ax2.plot(T_1, norm_F_1s, '--', c='#DB222A')
    ax2.fill_between(T_1, norm_P_1, y2=norm_P_1s, color='#78BC61', alpha=0.7)
    ax2.plot(T_1, norm_P_1, '-', c = '#78BC61')
    ax2.plot(T_1, norm_P_1s, '-', c = '#78BC61')
    ax2.fill_between(T_1, norm_Q_1, y2=norm_Q_1s, color='#55DBCB', alpha=0.7)
    ax2.plot(T_1, norm_Q_1, '--', c = '#55DBCB')
    ax2.plot(T_1, norm_Q_1s, '--', c = '#55DBCB')
    ax2.fill_between(T_1, norm_H_1, y2=norm_H_1s, color='#A846A0', alpha=0.7)
    ax2.plot(T_1, norm_H_1, '-', c = '#A846A0')
    ax2.plot(T_1, norm_H_1s, '-', c = '#A846A0')

    ax2.text(
        59.0, 6.0e-5, r'H', color='#A846A0', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        73.0, 0.015, r'D', color='#4CB944', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.001', fc='white', ec='none')
    )
    ax2.text(
        104.0, 0.469, r'N', color='#DEA54B', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')
    )
    ax2.text(
        61.5, 1.5e-5, r'F', color='#DB222A', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        76.0, 0.0016, r'P', color='#78BC61', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax2.text(
        62.8, 2.2e-6, r'Q', color='#55DBCB', fontsize=16,
        bbox=dict(boxstyle='square,pad=0', fc='white', ec='none')
    )
    ax2.text(37., 1.5, r'total baryon density', color='black', fontsize=14)
    ax2.text(183., 1.5, r'PNJL', color = 'blue', fontsize = 14)

    ax2.set_yscale('log')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize=16)
    ax2.set_ylabel(r'$\mathrm{\log~n_B}$', fontsize=16)

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.axis([10., 400., 0., 0.3])

    ax3.fill_between(T_2, total_cluster_2, y2=total_cluster_2s, color='green', alpha=0.7)
    ax3.plot(T_2, total_cluster_2, '--', color='green')
    ax3.plot(T_2, total_cluster_2s, '--', color='green')
    ax3.fill_between(T_2, total_ccluster_2, y2=total_ccluster_2s, color='red', alpha=0.7)
    ax3.plot(T_2, total_ccluster_2, '--', color='red')
    ax3.plot(T_2, total_ccluster_2s, '--', color='red')
    ax3.fill_between(T_2, total_2, y2=total_2s, color='black', alpha=0.7)
    ax3.plot(T_2, total_2, '-', c='black')
    ax3.plot(T_2, total_2s, '-', c='black')
    ax3.plot(T_2, total_qgp_2, '--', c='blue')

    ax3.text(
        165.0, 0.03, r'Color charged clusters', color='red', fontsize=14
    )
    ax3.text(
        173.3, 0.005, r'Color singlet clusters', color='green', fontsize=14
    )
    ax3.text(
        125.0, 0.065, r'NJL', color='blue', fontsize=14
    )
    ax3.text(
        196.0, 0.26, r'total baryon density', color='black', fontsize=14
    )
    ax3.text(
        19.0, 0.28, r'$\mathrm{\mu_B=200}$ MeV', color='black', fontsize=14
    )

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    ax4 = fig1.add_subplot(2, 2, 4)
    ax4.axis([35., 200., 1e-6, 1e1])

    ax4.plot(T_2, [1.0 for el in T_2], '-', c = 'black')
    ax4.fill_between(T_2, norm_QGP_2, y2=norm_QGP_2s, color='blue', alpha=0.7)
    ax4.plot(T_2, norm_QGP_2, '-', c='blue')
    ax4.plot(T_2, norm_QGP_2s, '-', c='blue')
    ax4.fill_between(T_2, norm_D_2, y2=norm_D_2s, color='#4CB944', alpha=0.7)
    ax4.plot(T_2, norm_D_2, '--', c='#4CB944')
    ax4.plot(T_2, norm_D_2s, '--', c='#4CB944')
    ax4.fill_between(T_2, norm_N_2, y2=norm_N_2s, color='#DEA54B', alpha=0.7)
    ax4.plot(T_2, norm_N_2, '-', c='#DEA54B')
    ax4.plot(T_2, norm_N_2s, '-', c='#DEA54B')
    ax4.fill_between(T_2, norm_F_2, y2=norm_F_2s, color='#DB222A', alpha=0.7)
    ax4.plot(T_2, norm_F_2, '--', c='#DB222A')
    ax4.plot(T_2, norm_F_2s, '--', c='#DB222A')
    ax4.fill_between(T_2, norm_P_2, y2=norm_P_2s, color='#78BC61', alpha=0.7)
    ax4.plot(T_2, norm_P_2, '-', c='#78BC61')
    ax4.plot(T_2, norm_P_2s, '-', c='#78BC61')
    ax4.fill_between(T_2, norm_Q_2, y2=norm_Q_2s, color='#55DBCB', alpha=0.7)
    ax4.plot(T_2, norm_Q_2, '--', c='#55DBCB')
    ax4.plot(T_2, norm_Q_2s, '--', c='#55DBCB')
    ax4.fill_between(T_2, norm_H_2, y2=norm_H_2s, color='#A846A0', alpha=0.7)
    ax4.plot(T_2, norm_H_2, '-', c='#A846A0')
    ax4.plot(T_2, norm_H_2s, '-', c='#A846A0')

    ax4.text(
        94.0, 6.0e-5, r'H', color='#A846A0', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        73.0, 0.065, r'D', color='#4CB944', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.001', fc='white', ec='none')
    )
    ax4.text(
        104.0, 0.025, r'N', color='#DEA54B', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')
    )
    ax4.text(
        59.7, 6e-5, r'F', color='#DB222A', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        76.0, 2.2e-5, r'P', color='#78BC61', fontsize=16,
        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none')
    )
    ax4.text(
        56.0, 2.2e-6, r'Q', color='#55DBCB', fontsize=16,
        bbox=dict(boxstyle='square,pad=0', fc='white', ec='none')
    )
    ax4.text(37., 1.5, r'total baryon density', color='black', fontsize=14)
    ax4.text(183., 1.5, r'NJL', color='blue', fontsize=14)

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


def epja_figure9_alt():

    import tqdm
    import numpy
    import pickle
    import warnings
    import platform

    import matplotlib.pyplot
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")
    
    calc_1 = False
    calc_2 = False

    files = "D:/EoS/epja/figure9/"

    files = "D:/EoS/epja/figure9/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/figure9/"

    T_1 = numpy.linspace(1.0, 400.0, 200)
    T_2 = numpy.linspace(1.0, 400.0, 200)

    mu_1 = [200.0 / 3.0 for el in T_1]
    mu_2 = [200.0 / 3.0 for el in T_2]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()
    phi_re_v_2 = [1.0 for _ in T_2]
    phi_im_v_2 = [0.0 for _ in T_2]

    sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1 = \
        list(), list(), list(), list()
    perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
        list(), list(), list()
    perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = \
        list(), list(), list(), list()

    sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2 = \
        list(), list(), list(), list()
    perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2 = \
        list(), list(), list()
    perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2 = \
        list(), list(), list(), list()

    D_v_1, N_v_1, F_v_1, P_v_1, Q_v_1, H_v_1 = \
        list(), list(), list(), list(), list(), list()      

    D_v_1s, N_v_1s, F_v_1s, P_v_1s, Q_v_1s, H_v_1s = \
        list(), list(), list(), list(), list(), list()
    
    D_v_2, N_v_2, F_v_2, P_v_2, Q_v_2, H_v_2 = \
        list(), list(), list(), list(), list(), list()

    D_v_2s, N_v_2s, F_v_2s, P_v_2s, Q_v_2s, H_v_2s = \
        list(), list(), list(), list(), list(), list()

    if calc_1:

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

        print("Sigma bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"sigma_v_1.pickle", "wb") as file:
            pickle.dump(sigma_v_1, file)

        print("Sea bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_1.append(lq_temp/(T_el**3))
            sea_d_v_1.append(lq_temp/(T_el**3))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"sea_u_v_1.pickle", "wb") as file:
            pickle.dump(sea_u_v_1, file)
        with open(files+"sea_d_v_1.pickle", "wb") as file:
            pickle.dump(sea_d_v_1, file)
        with open(files+"sea_s_v_1.pickle", "wb") as file:
            pickle.dump(sea_s_v_1, file)

        print("Perturbative bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            perturbative_u_v_1.append(lq_temp/(T_el**3))
            perturbative_d_v_1.append(lq_temp/(T_el**3))
            perturbative_s_v_1.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
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

        print("PNJL bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            pnjl_u_v_1.append(lq_temp/(T_el**3))
            pnjl_d_v_1.append(lq_temp/(T_el**3))
            pnjl_s_v_1.append(sq_temp/(T_el**3))
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

    if calc_2:

        with open(files+"phi_re_v_2.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)

        print("Sigma bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            sigma_v_2.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"sigma_v_2.pickle", "wb") as file:
            pickle.dump(sigma_v_2, file)

        print("Sea bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_2.append(lq_temp/(T_el**3))
            sea_d_v_2.append(lq_temp/(T_el**3))
            sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"sea_u_v_2.pickle", "wb") as file:
            pickle.dump(sea_u_v_2, file)
        with open(files+"sea_d_v_2.pickle", "wb") as file:
            pickle.dump(sea_d_v_2, file)
        with open(files+"sea_s_v_2.pickle", "wb") as file:
            pickle.dump(sea_s_v_2, file)

        print("Perturbative bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            perturbative_u_v_2.append(lq_temp/(T_el**3))
            perturbative_d_v_2.append(lq_temp/(T_el**3))
            perturbative_s_v_2.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
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

        print("PNJL bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            pnjl_u_v_2.append(lq_temp/(T_el**3))
            pnjl_d_v_2.append(lq_temp/(T_el**3))
            pnjl_s_v_2.append(sq_temp/(T_el**3))
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

    if calc_1:

        print("Diquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"D_v_1.pickle", "wb") as file:
            pickle.dump(D_v_1, file)

        print("Nucleon bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"N_v_1.pickle", "wb") as file:
            pickle.dump(N_v_1, file)

        print("F-quark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"F_v_1.pickle", "wb") as file:
            pickle.dump(F_v_1, file)

        print("Pentaquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"P_v_1.pickle", "wb") as file:
            pickle.dump(P_v_1, file)

        print("Q-quark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"Q_v_1.pickle", "wb") as file:
            pickle.dump(Q_v_1, file)

        print("Hexaquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"H_v_1.pickle", "wb") as file:
            pickle.dump(H_v_1, file)
    else:
        with open(files+"D_v_1.pickle", "rb") as file:
            D_v_1 = pickle.load(file)
        with open(files+"N_v_1.pickle", "rb") as file:
            N_v_1 = pickle.load(file)
        with open(files+"F_v_1.pickle", "rb") as file:
            F_v_1 = pickle.load(file)
        with open(files+"P_v_1.pickle", "rb") as file:
            P_v_1 = pickle.load(file)
        with open(files+"Q_v_1.pickle", "rb") as file:
            Q_v_1 = pickle.load(file)
        with open(files+"H_v_1.pickle", "rb") as file:
            H_v_1 = pickle.load(file)

    if calc_1:

        print("Diquark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"D_v_1s.pickle", "wb") as file:
            pickle.dump(D_v_1s, file)

        print("Nucleon bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"N_v_1s.pickle", "wb") as file:
            pickle.dump(N_v_1s, file)

        print("F-quark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"F_v_1s.pickle", "wb") as file:
            pickle.dump(F_v_1s, file)

        print("Pentaquark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"P_v_1s.pickle", "wb") as file:
            pickle.dump(P_v_1s, file)

        print("Q-quark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"Q_v_1s.pickle", "wb") as file:
            pickle.dump(Q_v_1s, file)

        print("Hexaquark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1s.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"H_v_1s.pickle", "wb") as file:
            pickle.dump(H_v_1s, file)
    else:
        with open(files+"D_v_1s.pickle", "rb") as file:
            D_v_1s = pickle.load(file)
        with open(files+"N_v_1s.pickle", "rb") as file:
            N_v_1s = pickle.load(file)
        with open(files+"F_v_1s.pickle", "rb") as file:
            F_v_1s = pickle.load(file)
        with open(files+"P_v_1s.pickle", "rb") as file:
            P_v_1s = pickle.load(file)
        with open(files+"Q_v_1s.pickle", "rb") as file:
            Q_v_1s = pickle.load(file)
        with open(files+"H_v_1s.pickle", "rb") as file:
            H_v_1s = pickle.load(file)

    if calc_2:

        print("Diquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"D_v_2.pickle", "wb") as file:
            pickle.dump(D_v_2, file)

        print("Nucleon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"N_v_2.pickle", "wb") as file:
            pickle.dump(N_v_2, file)

        print("F-quark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"F_v_2.pickle", "wb") as file:
            pickle.dump(F_v_2, file)

        print("Pentaquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"P_v_2.pickle", "wb") as file:
            pickle.dump(P_v_2, file)

        print("Q-quark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"Q_v_2.pickle", "wb") as file:
            pickle.dump(Q_v_2, file)

        print("Hexaquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"H_v_2.pickle", "wb") as file:
            pickle.dump(H_v_2, file)
    else:
        with open(files+"D_v_2.pickle", "rb") as file:
            D_v_2 = pickle.load(file)
        with open(files+"N_v_2.pickle", "rb") as file:
            N_v_2 = pickle.load(file)
        with open(files+"F_v_2.pickle", "rb") as file:
            F_v_2 = pickle.load(file)
        with open(files+"P_v_2.pickle", "rb") as file:
            P_v_2 = pickle.load(file)
        with open(files+"Q_v_2.pickle", "rb") as file:
            Q_v_2 = pickle.load(file)
        with open(files+"H_v_2.pickle", "rb") as file:
            H_v_2 = pickle.load(file)

    if calc_2:

        print("Diquark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"D_v_2s.pickle", "wb") as file:
            pickle.dump(D_v_2s, file)

        print("Nucleon bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"N_v_2s.pickle", "wb") as file:
            pickle.dump(N_v_2s, file)

        print("F-quark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"F_v_2s.pickle", "wb") as file:
            pickle.dump(F_v_2s, file)

        print("Pentaquark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"P_v_2s.pickle", "wb") as file:
            pickle.dump(P_v_2s, file)

        print("Q-quark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"Q_v_2s.pickle", "wb") as file:
            pickle.dump(Q_v_2s, file)

        print("Hexaquark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"H_v_2s.pickle", "wb") as file:
            pickle.dump(H_v_2s, file)
    else:
        with open(files+"D_v_2s.pickle", "rb") as file:
            D_v_2s = pickle.load(file)
        with open(files+"N_v_2s.pickle", "rb") as file:
            N_v_2s = pickle.load(file)
        with open(files+"F_v_2s.pickle", "rb") as file:
            F_v_2s = pickle.load(file)
        with open(files+"P_v_2s.pickle", "rb") as file:
            P_v_2s = pickle.load(file)
        with open(files+"Q_v_2s.pickle", "rb") as file:
            Q_v_2s = pickle.load(file)
        with open(files+"H_v_2s.pickle", "rb") as file:
            H_v_2s = pickle.load(file)

    total_qgp_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_pert_1 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1
            )
    ]
    total_pnjl_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_qgp_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_pnjl_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
                pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_pert_2 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2
            )
    ]
    total_cluster_1 = [
        sum(el) for el in 
            zip(N_v_1, P_v_1, H_v_1)
    ]
    total_ccluster_1 = [
        sum(el) for el in 
            zip(D_v_1, F_v_1, Q_v_1)
    ]
    total_tcluster_1 = [
        sum(el) for el in zip(total_cluster_1, total_ccluster_1)
    ]
    total_cluster_1s = [
        sum(el) for el in 
            zip(N_v_1s, P_v_1s, H_v_1s)
    ]
    total_ccluster_1s = [
        sum(el) for el in 
            zip(D_v_1s, F_v_1s, Q_v_1s)
    ]
    total_tcluster_1s = [
        sum(el) for el in zip(total_cluster_1s, total_ccluster_1s)
    ]
    total_1 = [sum(el) for el in zip(total_qgp_1, total_cluster_1, total_ccluster_1)]
    total_1s = [sum(el) for el in zip(total_qgp_1, total_cluster_1s, total_ccluster_1s)]
    total_cluster_2 = [
        sum(el) for el in 
            zip(N_v_2, P_v_2, H_v_2)
    ]
    total_ccluster_2 = [
        sum(el) for el in 
            zip(D_v_2, F_v_2, Q_v_2)
    ]
    total_tcluster_2 = [
        sum(el) for el in zip(total_cluster_2, total_ccluster_2)
    ]
    total_cluster_2s = [
        sum(el) for el in 
            zip(N_v_2s, P_v_2s, H_v_2s)
    ]
    total_ccluster_2s = [
        sum(el) for el in 
            zip(D_v_2s, F_v_2s, Q_v_2s)
    ]
    total_tcluster_2s = [
        sum(el) for el in zip(total_cluster_2s, total_ccluster_2s)
    ]
    total_2 = [sum(el) for el in zip(total_qgp_2, total_cluster_2, total_ccluster_2)]
    total_2s = [sum(el) for el in zip(total_qgp_2, total_cluster_2s, total_ccluster_2s)]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12.0, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.axis([10., 400., -0.15, 0.4])

    ax1.plot(T_1, total_pert_1, '-.', c = 'magenta')
    ax1.plot(T_1, total_pnjl_1, '-.', c = 'blue')
    ax1.plot(T_1, total_tcluster_1, '--', c = 'green')
    ax1.plot(T_1, total_qgp_1, '--', c = 'blue')
    ax1.plot(T_1, total_1, '-', c = 'black')

    ax1.text(
        160.0, 0.01, r'MHRG', color='green', fontsize=14
    )
    ax1.text(
        157.0, 0.065, r'QGP', color='blue', fontsize=14
    )
    ax1.text(
        280.0, 0.25, r'PNJL', color='blue', fontsize=14
    )
    ax1.text(
        158.0, -0.05, r'Perturbative', color='magenta', fontsize=14
    )
    ax1.text(
        232.0, 0.165, r'Total', color='black', fontsize=14
    )
    ax1.text(
        15.0, 0.37, r'$\mathrm{\mu_B=200}$ MeV', color='black', fontsize=14
    )

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    ax3 = fig1.add_subplot(1, 2, 2)
    ax3.axis([10., 400., -0.15, 0.4])

    ax3.plot(T_2, total_pert_2, '-.', c = 'magenta')
    ax3.plot(T_2, total_pnjl_2, '-.', c = 'blue')
    ax3.plot(T_2, total_tcluster_2, '--', color='green')
    ax3.plot(T_2, total_qgp_2, '--', c='blue')
    ax3.plot(T_2, total_2, '-', c='black')

    ax3.text(
        160.0, 0.01, r'MHRG', color='green', fontsize=14
    )
    ax3.text(
        140.0, 0.123, r'QGP', color='blue', fontsize=14
    )
    ax3.text(
        280.0, 0.25, r'PNJL', color='blue', fontsize=14
    )
    ax3.text(
        130.0, -0.05, r'Perturbative', color='magenta', fontsize=14
    )
    ax3.text(
        232.0, 0.165, r'Total', color='black', fontsize=14
    )
    ax3.text(
        15.0, 0.37, r'$\mathrm{\mu_B=200}$ MeV', color='black', fontsize=14
    )

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax3.set_xlabel(r'T [MeV]', fontsize = 16)
    ax3.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    fig1.tight_layout(pad=0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure10():

    import tqdm
    import math
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

    import utils
    import pnjl.defaults
    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")
    
    calc_lines = True

    calc_0 = False
    calc_1 = False
    calc_2 = False

    def intersect(x_v, y_v, target):
        y_prev = y_v[-1]
        x_prev = x_v[-1]
        sol_x = list()
        for x_el, y_el in zip(x_v[::-1][1:], y_v[::-1][1:]):
            if (y_el-target)*(y_prev-target) < 0.0:
                temp_num = math.fsum([
                    x_el*target, -x_prev*target, x_prev*y_el, -x_el*y_prev
                ])
                temp_den = math.fsum([
                    y_el, -y_prev
                ])
                sol_x.append(temp_num/temp_den)
            y_prev = y_el
            x_prev = x_el
        return sol_x

    T_30_1, T_45_1, T_300_1 = list(), list(), list()
    T_30_2, T_45_2, T_300_2 = list(), list(), list()
    T_30_qgp, T_45_qgp, T_300_qgp = list(), list(), list()
    mu_30, mu_45, mu_300 = list(), list(), list()

    files = "D:/EoS/epja/figure10/"

    T = numpy.linspace(1.0, 500.0, num=200)

    for mu in numpy.linspace(1.0, 200.0, num=200):

        print(mu)

        mu_round = math.floor(mu*10.0)/10.0

        phi_re_v, phi_im_v = list(), list()

        b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v = \
            list(), list(), list(), list(), list()
        b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v = \
            list(), list(), list()
        b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v = \
            list(), list(), list(), list()

        s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v = \
            list(), list(), list(), list(), list()
        s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v = \
            list(), list(), list()
        s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v = \
            list(), list(), list(), list()

        b_pi_v_1, b_K_v_1, b_rho_v_1, b_omega_v_1, b_D_v_1, b_N_v_1 = \
            list(), list(), list(), list(), list(), list()
        b_T_v_1, b_F_v_1, b_P_v_1, b_Q_v_1, b_H_v_1 = \
            list(), list(), list(), list(), list()

        s_pi_v_1, s_K_v_1, s_rho_v_1, s_omega_v_1, s_D_v_1, s_N_v_1 = \
            list(), list(), list(), list(), list(), list()
        s_T_v_1, s_F_v_1, s_P_v_1, s_Q_v_1, s_H_v_1 = \
            list(), list(), list(), list(), list()
    
        b_pi_v_2, b_K_v_2, b_rho_v_2, b_omega_v_2, b_D_v_2, b_N_v_2 = \
            list(), list(), list(), list(), list(), list()
        b_T_v_2, b_F_v_2, b_P_v_2, b_Q_v_2, b_H_v_2 = \
            list(), list(), list(), list(), list()
    
        s_pi_v_2, s_K_v_2, s_rho_v_2, s_omega_v_2, s_D_v_2, s_N_v_2 = \
            list(), list(), list(), list(), list(), list()
        s_T_v_2, s_F_v_2, s_P_v_2, s_Q_v_2, s_H_v_2 = \
            list(), list(), list(), list(), list()

        if calc_0:

            phi_re_0 = 1e-5
            phi_im_0 = 2e-5
            print("Traced Polyakov loop, mu =", mu_round)
            for T_el in tqdm.tqdm(T, total=len(T), ncols=100):
                phi_result = solver_1.Polyakov_loop(
                    T_el, mu_round, phi_re_0, phi_im_0
                )
                phi_re_v.append(phi_result[0])
                phi_im_v.append(phi_result[1])
                phi_re_0 = phi_result[0]
                phi_im_0 = phi_result[1]
            with open(
                files+"phi_re_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "wb"
            ) as file:
                pickle.dump(phi_re_v, file)
            with open(
                files+"phi_im_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "wb"
            ) as file:
                pickle.dump(phi_im_v, file)

            print("Sigma thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_sigma_v.append(
                    pnjl.thermo.gcp_sigma_lattice.bdensity(T_el, mu_round)
                )
                s_sigma_v.append(
                    pnjl.thermo.gcp_sigma_lattice.sdensity(T_el, mu_round)
                )
            with open(
                files+"b_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sigma_v, file)
            with open(
                files+"s_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sigma_v, file)
            
            print("Gluon thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_gluon_v.append(
                    pnjl.thermo.gcp_pl_polynomial.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                    )
                )
                s_gluon_v.append(
                    pnjl.thermo.gcp_pl_polynomial.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                    )
                )
            with open(
                files+"b_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_gluon_v, file)
            with open(
                files+"s_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_gluon_v, file)

            print("Sea thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_round, 'l'
                )
                b_sea_u_v.append(b_lq_temp)
                b_sea_d_v.append(b_lq_temp)
                b_sea_s_v.append(
                    pnjl.thermo.gcp_sea_lattice.bdensity(T_el, mu_round, 's')
                )
                s_lq_temp = pnjl.thermo.gcp_sea_lattice.sdensity(
                    T_el, mu_round, 'l'
                )
                s_sea_u_v.append(s_lq_temp)
                s_sea_d_v.append(s_lq_temp)
                s_sea_s_v.append(
                    pnjl.thermo.gcp_sea_lattice.sdensity(T_el, mu_round, 's')
                )
            with open(
                files+"b_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sea_u_v, file)
            with open(
                files+"b_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sea_d_v, file)
            with open(
                files+"b_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sea_s_v, file)
            with open(
                files+"s_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sea_u_v, file)
            with open(
                files+"s_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sea_d_v, file)
            with open(
                files+"s_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sea_s_v, file)
            
            print("Perturbative thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                b_perturbative_u_v.append(b_lq_temp)
                b_perturbative_d_v.append(b_lq_temp)
                b_perturbative_s_v.append(
                    pnjl.thermo.gcp_perturbative.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                    )
                )
                s_lq_temp = pnjl.thermo.gcp_perturbative.sdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                s_perturbative_u_v.append(s_lq_temp)
                s_perturbative_d_v.append(s_lq_temp)
                s_perturbative_s_v.append(
                    pnjl.thermo.gcp_perturbative.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                    )
                )
                if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                    raise RuntimeError(
                        "Perturbative gluon bdensity/sdensity not implemented!"
                    )
                else:
                    b_perturbative_gluon_v.append(0.0)
                    s_perturbative_gluon_v.append(0.0)
            with open(
                files+"b_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_u_v, file)
            with open(
                files+"b_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_d_v, file)
            with open(
                files+"b_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_s_v, file)
            with open(
                files+"s_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_u_v, file)
            with open(
                files+"s_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_d_v, file)
            with open(
                files+"s_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_s_v, file)
            with open(
                files+"b_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_gluon_v, file)
            with open(
                files+"s_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_gluon_v, file)

            print("PNJL thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                b_sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )
                s_lq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                s_sq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )
                b_pnjl_u_v.append(b_lq_temp)
                b_pnjl_d_v.append(b_lq_temp)
                b_pnjl_s_v.append(b_sq_temp)
                s_pnjl_u_v.append(s_lq_temp)
                s_pnjl_d_v.append(s_lq_temp)
                s_pnjl_s_v.append(s_sq_temp)
            with open(
                files+"b_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pnjl_u_v, file)
            with open(
                files+"b_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pnjl_d_v, file)
            with open(
                files+"b_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pnjl_s_v, file)
            with open(
                files+"s_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pnjl_u_v, file)
            with open(
                files+"s_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pnjl_d_v, file)
            with open(
                files+"s_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pnjl_s_v, file)
        else:
            with open(
                files+"phi_re_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "rb"
            ) as file:
                phi_re_v = pickle.load(file)
            with open(
                files+"phi_im_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "rb"
            ) as file:
                phi_im_v = pickle.load(file)
            with open(
                files+"b_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sigma_v = pickle.load(file)
            with open(
                files+"s_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sigma_v = pickle.load(file)
            with open(
                files+"b_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_gluon_v = pickle.load(file)
            with open(
                files+"s_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_gluon_v = pickle.load(file)
            with open(
                files+"b_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sea_u_v = pickle.load(file)
            with open(
                files+"b_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sea_d_v = pickle.load(file)
            with open(
                files+"b_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sea_s_v = pickle.load(file)
            with open(
                files+"s_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sea_u_v = pickle.load(file)
            with open(
                files+"s_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sea_d_v = pickle.load(file)
            with open(
                files+"s_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sea_s_v = pickle.load(file)
            with open(
                files+"b_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_u_v = pickle.load(file)
            with open(
                files+"b_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_d_v = pickle.load(file)
            with open(
                files+"b_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_s_v = pickle.load(file)
            with open(
                files+"s_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_u_v = pickle.load(file)
            with open(
                files+"s_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_d_v = pickle.load(file)
            with open(
                files+"s_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_s_v = pickle.load(file)
            with open(
                files+"b_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_gluon_v = pickle.load(file)
            with open(
                files+"s_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_gluon_v = pickle.load(file)
            with open(
                files+"b_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pnjl_u_v = pickle.load(file)
            with open(
                files+"b_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pnjl_d_v = pickle.load(file)
            with open(
                files+"b_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pnjl_s_v = pickle.load(file)
            with open(
                files+"s_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pnjl_u_v = pickle.load(file)
            with open(
                files+"s_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pnjl_d_v = pickle.load(file)
            with open(
                files+"s_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pnjl_s_v = pickle.load(file)

        if calc_1:

            print("Pion thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_pi_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                    )
                )
                s_pi_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                    )
                )
            with open(
                files+"b_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pi_v_1, file)
            with open(
                files+"s_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pi_v_1, file)

            print("Kaon thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_K_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                    )
                )
                s_K_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                    )
                )
            with open(
                files+"b_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_K_v_1, file)
            with open(
                files+"s_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_K_v_1, file)

            print("Rho thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_rho_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                    )
                )
                s_rho_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                    )
                )
            with open(
                files+"b_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_rho_v_1, file)
            with open(
                files+"s_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_rho_v_1, file)

            print("Omega thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_omega_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                    )
                )
                s_omega_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                    )
                )
            with open(
                files+"b_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_omega_v_1, file)
            with open(
                files+"s_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_omega_v_1, file)

            print("Diquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_D_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                    )
                )
                s_D_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                    )
                )
            with open(
                files+"b_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_D_v_1, file)
            with open(
                files+"s_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_D_v_1, file)

            print("Nucleon thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_N_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                    )
                )
                s_N_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                    )
                )
            with open(
                files+"b_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_N_v_1, file)
            with open(
                files+"s_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_N_v_1, file)

            print("Tetraquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_T_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                    )
                )
                s_T_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                    )
                )
            with open(
                files+"b_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_T_v_1, file)
            with open(
                files+"s_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_T_v_1, file)

            print("F-quark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_F_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                    )
                )
                s_F_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                    )
                )
            with open(
                files+"b_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_F_v_1, file)
            with open(
                files+"s_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_F_v_1, file)

            print("Pentaquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_P_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                    )
                )
                s_P_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                    )
                )
            with open(
                files+"b_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_P_v_1, file)
            with open(
                files+"s_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_P_v_1, file)

            print("Q-quark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_Q_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                    )
                )
                s_Q_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                    )
                )
            with open(
                files+"b_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_Q_v_1, file)
            with open(
                files+"s_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_Q_v_1, file)

            print("Hexaquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_H_v_1.append(
                    cluster_s.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                    )
                )
                s_H_v_1.append(
                    cluster_s.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                    )
                )
            with open(
                files+"b_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_H_v_1, file)
            with open(
                files+"s_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_H_v_1, file)
        else:
            with open(
                files+"b_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pi_v_1 = pickle.load(file)
            with open(
                files+"s_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pi_v_1 = pickle.load(file)
            with open(
                files+"b_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_K_v_1 = pickle.load(file)
            with open(
                files+"s_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_K_v_1 = pickle.load(file)
            with open(
                files+"b_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_rho_v_1 = pickle.load(file)
            with open(
                files+"s_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_rho_v_1 = pickle.load(file)
            with open(
                files+"b_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_omega_v_1 = pickle.load(file)
            with open(
                files+"s_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_omega_v_1 = pickle.load(file)
            with open(
                files+"b_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_D_v_1 = pickle.load(file)
            with open(
                files+"s_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_D_v_1 = pickle.load(file)
            with open(
                files+"b_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_N_v_1 = pickle.load(file)
            with open(
                files+"s_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_N_v_1 = pickle.load(file)
            with open(
                files+"b_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_T_v_1 = pickle.load(file)
            with open(
                files+"s_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_T_v_1 = pickle.load(file)
            with open(
                files+"b_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_F_v_1 = pickle.load(file)
            with open(
                files+"s_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_F_v_1 = pickle.load(file)
            with open(
                files+"b_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_P_v_1 = pickle.load(file)
            with open(
                files+"s_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_P_v_1 = pickle.load(file)
            with open(
                files+"b_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_Q_v_1 = pickle.load(file)
            with open(
                files+"s_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_Q_v_1 = pickle.load(file)
            with open(
                files+"b_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_H_v_1 = pickle.load(file)
            with open(
                files+"s_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_H_v_1 = pickle.load(file)

        if calc_2:

            print("Pion thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_pi_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                    )
                )
                s_pi_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                    )
                )
            with open(
                files+"b_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pi_v_2, file)
            with open(
                files+"s_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pi_v_2, file)

            print("Kaon thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_K_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                    )
                )
                s_K_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                    )
                )
            with open(
                files+"b_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_K_v_2, file)
            with open(
                files+"s_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_K_v_2, file)

            print("Rho thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_rho_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                    )
                )
                s_rho_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                    )
                )
            with open(
                files+"b_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_rho_v_2, file)
            with open(
                files+"s_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_rho_v_2, file)

            print("Omega thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_omega_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                    )
                )
                s_omega_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                    )
                )
            with open(
                files+"b_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_omega_v_2, file)
            with open(
                files+"s_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_omega_v_2, file)

            print("Diquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_D_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                    )
                )
                s_D_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                    )
                )
            with open(
                files+"b_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_D_v_2, file)
            with open(
                files+"s_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_D_v_2, file)

            print("Nucleon thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_N_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                    )
                )
                s_N_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                    )
                )
            with open(
                files+"b_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_N_v_2, file)
            with open(
                files+"s_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_N_v_2, file)

            print("Tetraquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_T_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                    )
                )
                s_T_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                    )
                )
            with open(
                files+"b_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_T_v_2, file)
            with open(
                files+"s_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_T_v_2, file)

            print("F-quark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_F_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                    )
                )
                s_F_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                    )
                )
            with open(
                files+"b_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_F_v_2, file)
            with open(
                files+"s_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_F_v_2, file)

            print("Pentaquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_P_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                    )
                )
                s_P_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                    )
                )
            with open(
                files+"b_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_P_v_2, file)
            with open(
                files+"s_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_P_v_2, file)

            print("Q-quark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_Q_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                    )
                )
                s_Q_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                    )
                )
            with open(
                files+"b_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_Q_v_2, file)
            with open(
                files+"s_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_Q_v_2, file)

            print("Hexaquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_H_v_2.append(
                    cluster.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                    )
                )
                s_H_v_2.append(
                    cluster.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                    )
                )
            with open(
                files+"b_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_H_v_2, file)
            with open(
                files+"s_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_H_v_2, file)
        else:
            with open(
                files+"b_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pi_v_2 = pickle.load(file)
            with open(
                files+"s_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pi_v_2 = pickle.load(file)
            with open(
                files+"b_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_K_v_2 = pickle.load(file)
            with open(
                files+"s_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_K_v_2 = pickle.load(file)
            with open(
                files+"b_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_rho_v_2 = pickle.load(file)
            with open(
                files+"s_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_rho_v_2 = pickle.load(file)
            with open(
                files+"b_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_omega_v_2 = pickle.load(file)
            with open(
                files+"s_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_omega_v_2 = pickle.load(file)
            with open(
                files+"b_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_D_v_2 = pickle.load(file)
            with open(
                files+"s_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_D_v_2 = pickle.load(file)
            with open(
                files+"b_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_N_v_2 = pickle.load(file)
            with open(
                files+"s_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_N_v_2 = pickle.load(file)
            with open(
                files+"b_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_T_v_2 = pickle.load(file)
            with open(
                files+"s_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_T_v_2 = pickle.load(file)
            with open(
                files+"b_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_F_v_2 = pickle.load(file)
            with open(
                files+"s_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_F_v_2 = pickle.load(file)
            with open(
                files+"b_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_P_v_2 = pickle.load(file)
            with open(
                files+"s_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_P_v_2 = pickle.load(file)
            with open(
                files+"b_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_Q_v_2 = pickle.load(file)
            with open(
                files+"s_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_Q_v_2 = pickle.load(file)
            with open(
                files+"b_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_H_v_2 = pickle.load(file)
            with open(
                files+"s_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_H_v_2 = pickle.load(file)

        total_b_1 = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v,
                b_pi_v_1, b_K_v_1, b_rho_v_1, b_omega_v_1, b_D_v_1, b_N_v_1,
                b_T_v_1, b_F_v_1, b_P_v_1, b_Q_v_1, b_H_v_1
            )
        ]
        total_b_2 = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v,
                b_pi_v_2, b_K_v_2, b_rho_v_2, b_omega_v_2, b_D_v_2, b_N_v_2,
                b_T_v_2, b_F_v_2, b_P_v_2, b_Q_v_2, b_H_v_2
            )
        ]
        total_b_qgp = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v
            )
        ]
        total_s_1 = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v,
                s_pi_v_1, s_K_v_1, s_rho_v_1, s_omega_v_1, s_D_v_1, s_N_v_1,
                s_T_v_1, s_F_v_1, s_P_v_1, s_Q_v_1, s_H_v_1
            )
        ]
        total_s_2 = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v,
                s_pi_v_2, s_K_v_2, s_rho_v_2, s_omega_v_2, s_D_v_2, s_N_v_2,
                s_T_v_2, s_F_v_2, s_P_v_2, s_Q_v_2, s_H_v_2
            )
        ]
        total_s_qgp = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v
            )
        ]
        total_1 = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_1, total_b_1)]
        total_2 = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_2, total_b_2)]
        total_qgp = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_qgp, total_b_qgp)]

        if calc_lines:
            T_30_1.append(intersect(T, total_1, 30.0))
            T_45_1.append(intersect(T, total_1, 45.0))
            T_300_1.append(intersect(T, total_1, 300.0))
            T_30_2.append(intersect(T, total_2, 30.0))
            T_45_2.append(intersect(T, total_2, 45.0))
            T_300_2.append(intersect(T, total_2, 300.0))
            T_30_qgp.append(intersect(T, total_qgp, 30.0))
            T_45_qgp.append(intersect(T, total_qgp, 45.0))
            T_300_qgp.append(intersect(T, total_qgp, 300.0))
            mu_30.append(mu_round)
            mu_45.append(mu_round)
            mu_300.append(mu_round)

    if calc_lines:
         with open(files+"T_30_1.pickle","wb") as file:
            pickle.dump(T_30_1, file)
         with open(files+"T_45_1.pickle","wb") as file:
            pickle.dump(T_45_1, file)
         with open(files+"T_300_1.pickle","wb") as file:
            pickle.dump(T_300_1, file)
         with open(files+"T_30_2.pickle","wb") as file:
            pickle.dump(T_30_2, file)
         with open(files+"T_45_2.pickle","wb") as file:
            pickle.dump(T_45_2, file)
         with open(files+"T_300_2.pickle","wb") as file:
            pickle.dump(T_300_2, file)
         with open(files+"mu_30.pickle","wb") as file:
            pickle.dump(mu_30, file)
         with open(files+"mu_45.pickle","wb") as file:
            pickle.dump(mu_45, file)
         with open(files+"mu_300.pickle","wb") as file:
            pickle.dump(mu_300, file)
    else:
        with open(files+"T_30_1.pickle","rb") as file:
            T_30_1 = pickle.load(file)
        with open(files+"T_45_1.pickle","rb") as file:
            T_45_1 = pickle.load(file)
        with open(files+"T_300_1.pickle","rb") as file:
            T_300_1 = pickle.load(file)
        with open(files+"T_30_2.pickle","rb") as file:
            T_30_2 = pickle.load(file)
        with open(files+"T_45_2.pickle","rb") as file:
            T_45_2 = pickle.load(file)
        with open(files+"T_300_2.pickle","rb") as file:
            T_300_2 = pickle.load(file)
        with open(files+"mu_30.pickle","rb") as file:
            mu_30 = pickle.load(file)
        with open(files+"mu_45.pickle","rb") as file:
            mu_45 = pickle.load(file)
        with open(files+"mu_300.pickle","rb") as file:
            mu_300 = pickle.load(file)

    mu_300_2_0 = [mu_el*3.0 if len(T_el)>0 else float('nan') for mu_el, T_el in zip(mu_300, T_300_2)]
    mu_300_2_1 = [mu_el*3.0 if len(T_el)>1 else float('nan') for mu_el, T_el in zip(mu_300, T_300_2)]
    mu_300_2_2 = [mu_el*3.0 if len(T_el)>2 else float('nan') for mu_el, T_el in zip(mu_300, T_300_2)]
    mu_300_2_3 = [mu_el*3.0 if len(T_el)>3 else float('nan') for mu_el, T_el in zip(mu_300, T_300_2)]
    mu_300_2_4 = [mu_el*3.0 if len(T_el)>4 else float('nan') for mu_el, T_el in zip(mu_300, T_300_2)]

    T_300_2_0 = [el[0] if len(el)>0 else float('nan') for el in T_300_2]
    T_300_2_1 = [el[1] if len(el)>1 else float('nan') for el in T_300_2]
    T_300_2_2 = [el[2] if len(el)>2 else float('nan') for el in T_300_2]
    T_300_2_3 = [el[3] if len(el)>3 else float('nan') for el in T_300_2]
    T_300_2_4 = [el[4] if len(el)>4 else float('nan') for el in T_300_2]

    mu_300_1_0 = [mu_el*3.0 if len(T_el)>0 else float('nan') for mu_el, T_el in zip(mu_300, T_300_1)]
    mu_300_1_1 = [mu_el*3.0 if len(T_el)>1 else float('nan') for mu_el, T_el in zip(mu_300, T_300_1)]
    mu_300_1_2 = [mu_el*3.0 if len(T_el)>2 else float('nan') for mu_el, T_el in zip(mu_300, T_300_1)]
    mu_300_1_3 = [mu_el*3.0 if len(T_el)>3 else float('nan') for mu_el, T_el in zip(mu_300, T_300_1)]
    mu_300_1_4 = [mu_el*3.0 if len(T_el)>4 else float('nan') for mu_el, T_el in zip(mu_300, T_300_1)]
    mu_300_1_5 = [mu_el*3.0 if len(T_el)>5 else float('nan') for mu_el, T_el in zip(mu_300, T_300_1)]
    mu_300_1_6 = [mu_el*3.0 if len(T_el)>6 else float('nan') for mu_el, T_el in zip(mu_300, T_300_1)]

    T_300_1_0 = [el[0] if len(el)>0 else float('nan') for el in T_300_1]
    T_300_1_1 = [el[1] if len(el)>1 else float('nan') for el in T_300_1]
    T_300_1_2 = [el[2] if len(el)>2 else float('nan') for el in T_300_1]
    T_300_1_3 = [el[3] if len(el)>3 else float('nan') for el in T_300_1]
    T_300_1_4 = [el[4] if len(el)>4 else float('nan') for el in T_300_1]
    T_300_1_5 = [el[5] if len(el)>5 else float('nan') for el in T_300_1]
    T_300_1_6 = [el[6] if len(el)>6 else float('nan') for el in T_300_1]

    mu_300_qgp_0 = [mu_el*3.0 if len(T_el)>0 else float('nan') for mu_el, T_el in zip(mu_300, T_300_qgp)]
    mu_300_qgp_1 = [mu_el*3.0 if len(T_el)>1 else float('nan') for mu_el, T_el in zip(mu_300, T_300_qgp)]
    mu_300_qgp_2 = [mu_el*3.0 if len(T_el)>2 else float('nan') for mu_el, T_el in zip(mu_300, T_300_qgp)]
    mu_300_qgp_3 = [mu_el*3.0 if len(T_el)>3 else float('nan') for mu_el, T_el in zip(mu_300, T_300_qgp)]
    mu_300_qgp_4 = [mu_el*3.0 if len(T_el)>4 else float('nan') for mu_el, T_el in zip(mu_300, T_300_qgp)]

    T_300_qgp_0 = [el[0] if len(el)>0 else float('nan') for el in T_300_qgp]
    T_300_qgp_1 = [el[1] if len(el)>1 else float('nan') for el in T_300_qgp]
    T_300_qgp_2 = [el[2] if len(el)>2 else float('nan') for el in T_300_qgp]
    T_300_qgp_3 = [el[3] if len(el)>3 else float('nan') for el in T_300_qgp]
    T_300_qgp_4 = [el[4] if len(el)>4 else float('nan') for el in T_300_qgp]

    mu_45_1_0 = [mu_el*3.0 if len(T_el)>0 else float('nan') for mu_el, T_el in zip(mu_45, T_45_1)]
    mu_45_1_1 = [mu_el*3.0 if len(T_el)>1 else float('nan') for mu_el, T_el in zip(mu_45, T_45_1)]
    mu_45_1_2 = [mu_el*3.0 if len(T_el)>2 else float('nan') for mu_el, T_el in zip(mu_45, T_45_1)]
    mu_45_1_3 = [mu_el*3.0 if len(T_el)>3 else float('nan') for mu_el, T_el in zip(mu_45, T_45_1)]
    mu_45_1_4 = [mu_el*3.0 if len(T_el)>4 else float('nan') for mu_el, T_el in zip(mu_45, T_45_1)]
    mu_45_1_5 = [mu_el*3.0 if len(T_el)>5 else float('nan') for mu_el, T_el in zip(mu_45, T_45_1)]
    mu_45_1_6 = [mu_el*3.0 if len(T_el)>6 else float('nan') for mu_el, T_el in zip(mu_45, T_45_1)]

    T_45_1_0 = [el[0] if len(el)>0 else float('nan') for el in T_45_1]
    T_45_1_1 = [el[1] if len(el)>1 else float('nan') for el in T_45_1]
    T_45_1_2 = [el[2] if len(el)>2 else float('nan') for el in T_45_1]
    T_45_1_3 = [el[3] if len(el)>3 else float('nan') for el in T_45_1]
    T_45_1_4 = [el[4] if len(el)>4 else float('nan') for el in T_45_1]
    T_45_1_5 = [el[5] if len(el)>5 else float('nan') for el in T_45_1]
    T_45_1_6 = [el[6] if len(el)>6 else float('nan') for el in T_45_1]

    mu_45_2_0 = [mu_el*3.0 if len(T_el)>0 else float('nan') for mu_el, T_el in zip(mu_45, T_45_2)]
    mu_45_2_1 = [mu_el*3.0 if len(T_el)>1 else float('nan') for mu_el, T_el in zip(mu_45, T_45_2)]
    mu_45_2_2 = [mu_el*3.0 if len(T_el)>2 else float('nan') for mu_el, T_el in zip(mu_45, T_45_2)]
    mu_45_2_3 = [mu_el*3.0 if len(T_el)>3 else float('nan') for mu_el, T_el in zip(mu_45, T_45_2)]
    mu_45_2_4 = [mu_el*3.0 if len(T_el)>4 else float('nan') for mu_el, T_el in zip(mu_45, T_45_2)]
    mu_45_2_5 = [mu_el*3.0 if len(T_el)>5 else float('nan') for mu_el, T_el in zip(mu_45, T_45_2)]
    mu_45_2_6 = [mu_el*3.0 if len(T_el)>6 else float('nan') for mu_el, T_el in zip(mu_45, T_45_2)]

    T_45_2_0 = [el[0] if len(el)>0 else float('nan') for el in T_45_2]
    T_45_2_1 = [el[1] if len(el)>1 else float('nan') for el in T_45_2]
    T_45_2_2 = [el[2] if len(el)>2 else float('nan') for el in T_45_2]
    T_45_2_3 = [el[3] if len(el)>3 else float('nan') for el in T_45_2]
    T_45_2_4 = [el[4] if len(el)>4 else float('nan') for el in T_45_2]
    T_45_2_5 = [el[5] if len(el)>5 else float('nan') for el in T_45_2]
    T_45_2_6 = [el[6] if len(el)>6 else float('nan') for el in T_45_2]

    mu_45_qgp_0 = [mu_el*3.0 if len(T_el)>0 else float('nan') for mu_el, T_el in zip(mu_45, T_45_qgp)]
    mu_45_qgp_1 = [mu_el*3.0 if len(T_el)>1 else float('nan') for mu_el, T_el in zip(mu_45, T_45_qgp)]
    mu_45_qgp_2 = [mu_el*3.0 if len(T_el)>2 else float('nan') for mu_el, T_el in zip(mu_45, T_45_qgp)]
    mu_45_qgp_3 = [mu_el*3.0 if len(T_el)>3 else float('nan') for mu_el, T_el in zip(mu_45, T_45_qgp)]
    mu_45_qgp_4 = [mu_el*3.0 if len(T_el)>4 else float('nan') for mu_el, T_el in zip(mu_45, T_45_qgp)]
    mu_45_qgp_5 = [mu_el*3.0 if len(T_el)>5 else float('nan') for mu_el, T_el in zip(mu_45, T_45_qgp)]
    mu_45_qgp_6 = [mu_el*3.0 if len(T_el)>6 else float('nan') for mu_el, T_el in zip(mu_45, T_45_qgp)]

    T_45_qgp_0 = [el[0] if len(el)>0 else float('nan') for el in T_45_qgp]
    T_45_qgp_1 = [el[1] if len(el)>1 else float('nan') for el in T_45_qgp]
    T_45_qgp_2 = [el[2] if len(el)>2 else float('nan') for el in T_45_qgp]
    T_45_qgp_3 = [el[3] if len(el)>3 else float('nan') for el in T_45_qgp]
    T_45_qgp_4 = [el[4] if len(el)>4 else float('nan') for el in T_45_qgp]

    mu_30_1_0 = [mu_el*3.0 if len(T_el)>0 else float('nan') for mu_el, T_el in zip(mu_30, T_30_1)]
    mu_30_1_1 = [mu_el*3.0 if len(T_el)>1 else float('nan') for mu_el, T_el in zip(mu_30, T_30_1)]
    mu_30_1_2 = [mu_el*3.0 if len(T_el)>2 else float('nan') for mu_el, T_el in zip(mu_30, T_30_1)]
    mu_30_1_3 = [mu_el*3.0 if len(T_el)>3 else float('nan') for mu_el, T_el in zip(mu_30, T_30_1)]
    mu_30_1_4 = [mu_el*3.0 if len(T_el)>4 else float('nan') for mu_el, T_el in zip(mu_30, T_30_1)]
    mu_30_1_5 = [mu_el*3.0 if len(T_el)>5 else float('nan') for mu_el, T_el in zip(mu_30, T_30_1)]
    mu_30_1_6 = [mu_el*3.0 if len(T_el)>6 else float('nan') for mu_el, T_el in zip(mu_30, T_30_1)]
    mu_30_1_7 = [mu_el*3.0 if len(T_el)>7 else float('nan') for mu_el, T_el in zip(mu_30, T_30_1)]
    mu_30_1_8 = [mu_el*3.0 if len(T_el)>8 else float('nan') for mu_el, T_el in zip(mu_30, T_30_1)]

    T_30_1_0 = [el[0] if len(el)>0 else float('nan') for el in T_30_1]
    T_30_1_1 = [el[1] if len(el)>1 else float('nan') for el in T_30_1]
    T_30_1_2 = [el[2] if len(el)>2 else float('nan') for el in T_30_1]
    T_30_1_3 = [el[3] if len(el)>3 else float('nan') for el in T_30_1]
    T_30_1_4 = [el[4] if len(el)>4 else float('nan') for el in T_30_1]
    T_30_1_5 = [el[5] if len(el)>5 else float('nan') for el in T_30_1]
    T_30_1_6 = [el[6] if len(el)>6 else float('nan') for el in T_30_1]
    T_30_1_7 = [el[7] if len(el)>7 else float('nan') for el in T_30_1]
    T_30_1_8 = [el[8] if len(el)>8 else float('nan') for el in T_30_1]

    mu_30_2_0 = [mu_el*3.0 if len(T_el)>0 else float('nan') for mu_el, T_el in zip(mu_30, T_30_2)]
    mu_30_2_1 = [mu_el*3.0 if len(T_el)>1 else float('nan') for mu_el, T_el in zip(mu_30, T_30_2)]
    mu_30_2_2 = [mu_el*3.0 if len(T_el)>2 else float('nan') for mu_el, T_el in zip(mu_30, T_30_2)]
    mu_30_2_3 = [mu_el*3.0 if len(T_el)>3 else float('nan') for mu_el, T_el in zip(mu_30, T_30_2)]
    mu_30_2_4 = [mu_el*3.0 if len(T_el)>4 else float('nan') for mu_el, T_el in zip(mu_30, T_30_2)]

    T_30_2_0 = [el[0] if len(el)>0 else float('nan') for el in T_30_2]
    T_30_2_1 = [el[1] if len(el)>1 else float('nan') for el in T_30_2]
    T_30_2_2 = [el[2] if len(el)>2 else float('nan') for el in T_30_2]
    T_30_2_3 = [el[3] if len(el)>3 else float('nan') for el in T_30_2]
    T_30_2_4 = [el[4] if len(el)>4 else float('nan') for el in T_30_2]

    mu_30_qgp_0 = [mu_el*3.0 if len(T_el)>0 else float('nan') for mu_el, T_el in zip(mu_30, T_30_qgp)]
    mu_30_qgp_1 = [mu_el*3.0 if len(T_el)>1 else float('nan') for mu_el, T_el in zip(mu_30, T_30_qgp)]
    mu_30_qgp_2 = [mu_el*3.0 if len(T_el)>2 else float('nan') for mu_el, T_el in zip(mu_30, T_30_qgp)]
    mu_30_qgp_3 = [mu_el*3.0 if len(T_el)>3 else float('nan') for mu_el, T_el in zip(mu_30, T_30_qgp)]
    mu_30_qgp_4 = [mu_el*3.0 if len(T_el)>4 else float('nan') for mu_el, T_el in zip(mu_30, T_30_qgp)]
    mu_30_qgp_5 = [mu_el*3.0 if len(T_el)>5 else float('nan') for mu_el, T_el in zip(mu_30, T_30_qgp)]
    mu_30_qgp_6 = [mu_el*3.0 if len(T_el)>6 else float('nan') for mu_el, T_el in zip(mu_30, T_30_qgp)]

    T_30_qgp_0 = [el[0] if len(el)>0 else float('nan') for el in T_30_qgp]
    T_30_qgp_1 = [el[1] if len(el)>1 else float('nan') for el in T_30_qgp]
    T_30_qgp_2 = [el[2] if len(el)>2 else float('nan') for el in T_30_qgp]
    T_30_qgp_3 = [el[3] if len(el)>3 else float('nan') for el in T_30_qgp]
    T_30_qgp_4 = [el[4] if len(el)>4 else float('nan') for el in T_30_qgp]
    T_30_qgp_5 = [el[5] if len(el)>5 else float('nan') for el in T_30_qgp]
    T_30_qgp_6 = [el[6] if len(el)>6 else float('nan') for el in T_30_qgp]

    mu_300_2_full = mu_300_2_0[33:][::-1] \
        + mu_300_2_1[15:33][::-1] + mu_300_2_3[10:15][::-1] \
            + mu_300_2_2[10:15] + mu_300_2_1[10:15][::-1] + mu_300_2_0[10:33]
    T_300_2_full = T_300_2_0[33:][::-1] \
        + T_300_2_1[15:33][::-1] + T_300_2_3[10:15][::-1] \
            + T_300_2_2[10:15] + T_300_2_1[10:15][::-1] + T_300_2_0[10:33]

    mu_300_1_full = mu_300_1_0[33:][::-1] + mu_300_1_1[15:33][::-1] \
        + mu_300_1_3[11:15][::-1] + mu_300_1_5[10:11] \
            + mu_300_1_1[8:10][::-1] + mu_300_1_0[8:10] \
                + mu_300_1_4[10:11] + mu_300_1_2[11:15] \
                    + mu_300_1_1[11:15][::-1] + mu_300_1_3[10:11] \
                        + mu_300_1_2[10:11] + mu_300_1_1[10:11] \
                            + mu_300_1_0[10:33]
    T_300_1_full = T_300_1_0[33:][::-1] + T_300_1_1[15:33][::-1] \
        + T_300_1_3[11:15][::-1] + T_300_1_5[10:11] \
            + T_300_1_1[8:10][::-1] + T_300_1_0[8:10] \
                + T_300_1_4[10:11] + T_300_1_2[11:15] \
                    + T_300_1_1[11:15][::-1] + T_300_1_3[10:11] \
                        + T_300_1_2[10:11] + T_300_1_1[10:11] \
                            + T_300_1_0[10:33]

    mu_45_1_full = mu_45_1_1[104:][::-1] + mu_45_1_3[80:104][::-1] \
        + mu_45_1_5[77:80][::-1] + mu_45_1_3[74:77][::-1] \
            + mu_45_1_1[52:74][::-1] + mu_45_1_0[52:74] \
                + mu_45_1_2[74:77] + mu_45_1_4[77:80] \
                    + mu_45_1_2[80:104] + mu_45_1_1[80:104][::-1] \
                        + mu_45_1_3[77:80][::-1] + mu_45_1_2[77:80] \
                            + mu_45_1_1[74:80][::-1] + mu_45_1_0[74:]
    T_45_1_full = T_45_1_1[104:][::-1] + T_45_1_3[80:104][::-1] \
        + T_45_1_5[77:80][::-1] + T_45_1_3[74:77][::-1] \
            + T_45_1_1[52:74][::-1] + T_45_1_0[52:74] \
                + T_45_1_2[74:77] + T_45_1_4[77:80] \
                    + T_45_1_2[80:104] + T_45_1_1[80:104][::-1] \
                        + T_45_1_3[77:80][::-1] + T_45_1_2[77:80] \
                            + T_45_1_1[74:80][::-1] + T_45_1_0[74:]

    mu_45_2_full = mu_45_2_1[102:][::-1] + mu_45_2_3[72:102][::-1] \
        + mu_45_2_1[64:71][::-1] + mu_45_2_0[64:71] \
            + mu_45_2_2[72:102] + mu_45_2_1[72:102][::-1] \
                + mu_45_2_0[72:]
    T_45_2_full = T_45_2_1[102:][::-1] + T_45_2_3[72:102][::-1] \
        + T_45_2_1[64:71][::-1] + T_45_2_0[64:71] \
            + T_45_2_2[72:102] + T_45_2_1[72:102][::-1] \
                + T_45_2_0[72:]

    mu_30_1_full = mu_30_1_1[169:][::-1] + mu_30_1_3[166:169][::-1] \
        + mu_30_1_1[160:166][::-1] + mu_30_1_3[155:160][::-1] \
        + mu_30_1_5[154:155] + mu_30_1_3[124:154][::-1] \
        + mu_30_1_5[123:124] + mu_30_1_7[122:123] \
        + mu_30_1_5[120:122][::-1] + mu_30_1_3[113:120][::-1] \
        + mu_30_1_1[73:112][::-1] + mu_30_1_0[73:113] \
        + mu_30_1_2[113:120] + mu_30_1_4[121:122] \
        + mu_30_1_6[122:123] + mu_30_1_4[123:124] \
        + mu_30_1_2[124:154] + mu_30_1_4[154:155] \
        + mu_30_1_2[155:160] + mu_30_1_2[166:169] \
        + mu_30_1_1[166:169][::-1] + mu_30_1_1[155:160][::-1] \
        + mu_30_1_3[154:155] + mu_30_1_1[124:154][::-1] \
        + mu_30_1_3[123:124] + mu_30_1_5[122:123] \
        + mu_30_1_3[120:122][::-1] + mu_30_1_2[120:122] \
        + mu_30_1_4[122:123] + mu_30_1_2[123:124] \
        + mu_30_1_1[123:124] + mu_30_1_3[122:123] \
        + mu_30_1_1[113:121][::-1] + mu_30_1_0[113:122] \
        + mu_30_1_2[122:123] + mu_30_1_1[122:123] \
        + mu_30_1_0[122:154] + mu_30_1_2[154:155] \
        + mu_30_1_0[155:]
    T_30_1_full = T_30_1_1[169:][::-1] + T_30_1_3[166:169][::-1] \
        + T_30_1_1[160:166][::-1] + T_30_1_3[155:160][::-1] \
        + T_30_1_5[154:155] + T_30_1_3[124:154][::-1] \
        + T_30_1_5[123:124] + T_30_1_7[122:123] \
        + T_30_1_5[120:122][::-1] + T_30_1_3[113:120][::-1] \
        + T_30_1_1[73:112][::-1] + T_30_1_0[73:113] \
        + T_30_1_2[113:120] + T_30_1_4[121:122] \
        + T_30_1_6[122:123] + T_30_1_4[123:124] \
        + T_30_1_2[124:154] + T_30_1_4[154:155] \
        + T_30_1_2[155:160] + T_30_1_2[166:169] \
        + T_30_1_1[166:169][::-1] + T_30_1_1[155:160][::-1] \
        + T_30_1_3[154:155] + T_30_1_1[124:154][::-1] \
        + T_30_1_3[123:124] + T_30_1_5[122:123] \
        + T_30_1_3[120:122][::-1] + T_30_1_2[120:122] \
        + T_30_1_4[122:123] + T_30_1_2[123:124] \
        + T_30_1_1[123:124] + T_30_1_3[122:123] \
        + T_30_1_1[113:121][::-1] + T_30_1_0[113:122] \
        + T_30_1_2[122:123] + T_30_1_1[122:123] \
        + T_30_1_0[122:154] + T_30_1_2[154:155] \
        + T_30_1_0[155:]

    mu_30_2_full = mu_30_2_1[155:][::-1] + mu_30_2_3[110:155][::-1] \
        + mu_30_2_1[90:109][::-1] + mu_30_2_0[90:109] \
        + mu_30_2_2[110:154] + mu_30_2_1[110:154][::-1] \
        + mu_30_2_0[110:154] + mu_30_2_2[154:155] \
        + mu_30_2_0[155:]
    T_30_2_full = T_30_2_1[155:][::-1] + T_30_2_3[110:155][::-1] \
        + T_30_2_1[90:109][::-1] + T_30_2_0[90:109] \
        + T_30_2_2[110:154] + T_30_2_1[110:154][::-1] \
        + T_30_2_0[110:154] + T_30_2_2[154:155] \
        + T_30_2_0[155:]

    mu_300_qgp_full = mu_300_qgp_3[4:5] + mu_300_qgp_2[4:5] \
        + mu_300_qgp_1[4:5] + mu_300_qgp_0[4:12] \
        + mu_300_qgp_2[12:14] + mu_300_qgp_1[12:14][::-1] \
        + mu_300_qgp_0[12:33]

    T_300_qgp_full = T_300_qgp_3[4:5] + T_300_qgp_2[4:5] \
        + T_300_qgp_1[4:5] + T_300_qgp_0[4:12] \
        + T_300_qgp_2[12:14] + T_300_qgp_1[12:14][::-1] \
        + T_300_qgp_0[12:33]

    mu_45_qgp_full = mu_45_qgp_0[40:81] + mu_45_qgp_2[81:100] \
        + mu_45_qgp_1[81:100][::-1] + mu_45_qgp_0[81:]

    T_45_qgp_full = T_45_qgp_0[40:81] + T_45_qgp_2[81:100] \
        + T_45_qgp_1[81:100][::-1] + T_45_qgp_0[81:]

    mu_30_qgp_full = mu_30_qgp_0[70:122] + mu_30_qgp_2[122:154] \
        + mu_30_qgp_4[154:155] + mu_30_qgp_2[155:161] \
        + mu_30_qgp_1[155:161][::-1] + mu_30_qgp_3[154:155] \
        + mu_30_qgp_1[122:154][::-1] + mu_30_qgp_0[122:154] \
        + mu_30_qgp_2[154:155] + mu_30_qgp_0[155:]

    T_30_qgp_full = T_30_qgp_0[70:122] + T_30_qgp_2[122:154] \
        + T_30_qgp_4[154:155] + T_30_qgp_2[155:161] \
        + T_30_qgp_1[155:161][::-1] + T_30_qgp_3[154:155] \
        + T_30_qgp_1[122:154][::-1] + T_30_qgp_0[122:154] \
        + T_30_qgp_2[154:155] + T_30_qgp_0[155:]

    with open("D:/EoS/epja/lattice_data_pickled/Schmidt_EPJC_2009_figure6_sn30_N4.pickle", "rb") as file:
        Schmidt_EPJC_2009_figure6_sn30_N4 = pickle.load(file)
    with open("D:/EoS/epja/lattice_data_pickled/Schmidt_EPJC_2009_figure6_sn45_N4.pickle", "rb") as file:
        Schmidt_EPJC_2009_figure6_sn45_N4 = pickle.load(file)
    with open("D:/EoS/epja/lattice_data_pickled/Schmidt_EPJC_2009_figure6_sn300_N4.pickle", "rb") as file:
        Schmidt_EPJC_2009_figure6_sn300_N4 = pickle.load(file)

    Schmidt_new_mu_T_c_30, Schmidt_new_mu_T_err_30, Schmidt_new_T_30 = \
        utils.data_load(
            "D:\EoS\lattice_data\muB_T_LCP_snB\muB_T_LCP_snB30.d", 0, 1, 2,
            firstrow=1, delim='  '
        )
    Schmidt_new_mu_low_30 = [
        math.fsum([muT_el, -err_ell])*T_el for muT_el, err_ell, T_el in zip(
            Schmidt_new_mu_T_c_30, Schmidt_new_mu_T_err_30, Schmidt_new_T_30
        )
    ]
    Schmidt_new_mu_high_30 = [
        math.fsum([muT_el, err_ell])*T_el for muT_el, err_ell, T_el in zip(
            Schmidt_new_mu_T_c_30, Schmidt_new_mu_T_err_30, Schmidt_new_T_30
        )
    ]

    Schmidt_new_mu_T_c_50, Schmidt_new_mu_T_err_50, Schmidt_new_T_50 = \
        utils.data_load(
            "D:\EoS\lattice_data\muB_T_LCP_snB\muB_T_LCP_snB50.d", 0, 1, 2,
            firstrow=1, delim='  '
        )
    Schmidt_new_mu_low_50 = [
        math.fsum([muT_el, -err_ell])*T_el for muT_el, err_ell, T_el in zip(
            Schmidt_new_mu_T_c_50, Schmidt_new_mu_T_err_50, Schmidt_new_T_50
        )
    ]
    Schmidt_new_mu_high_50 = [
        math.fsum([muT_el, err_ell])*T_el for muT_el, err_ell, T_el in zip(
            Schmidt_new_mu_T_c_50, Schmidt_new_mu_T_err_50, Schmidt_new_T_50
        )
    ]

    Schmidt_new_mu_T_c_200, Schmidt_new_mu_T_err_200, Schmidt_new_T_200 = \
        utils.data_load(
            "D:\EoS\lattice_data\muB_T_LCP_snB\muB_T_LCP_snB200.d", 0, 1, 2,
            firstrow=1, delim='  '
        )
    Schmidt_new_mu_low_200 = [
        math.fsum([muT_el, -err_ell])*T_el for muT_el, err_ell, T_el in zip(
            Schmidt_new_mu_T_c_200, Schmidt_new_mu_T_err_200, Schmidt_new_T_200
        )
    ]
    Schmidt_new_mu_high_200 = [
        math.fsum([muT_el, err_ell])*T_el for muT_el, err_ell, T_el in zip(
            Schmidt_new_mu_T_c_200, Schmidt_new_mu_T_err_200, Schmidt_new_T_200
        )
    ]

    Schmidt_new_mu_T_c_400, Schmidt_new_mu_T_err_400, Schmidt_new_T_400 = \
        utils.data_load(
            "D:\EoS\lattice_data\muB_T_LCP_snB\muB_T_LCP_snB400.d", 0, 1, 2,
            firstrow=1, delim='  '
        )
    Schmidt_new_mu_low_400 = [
        math.fsum([muT_el, -err_ell])*T_el for muT_el, err_ell, T_el in zip(
            Schmidt_new_mu_T_c_400, Schmidt_new_mu_T_err_400, Schmidt_new_T_400
        )
    ]
    Schmidt_new_mu_high_400 = [
        math.fsum([muT_el, err_ell])*T_el for muT_el, err_ell, T_el in zip(
            Schmidt_new_mu_T_c_400, Schmidt_new_mu_T_err_400, Schmidt_new_T_400
        )
    ]

    fig1 = matplotlib.pyplot.figure(num=1, figsize=(11.0, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    ax1.axis([0., 600., 120., 220.])
    ax2.axis([0., 600., 120., 220.])

    #fig1 = matplotlib.pyplot.figure(num = 1, figsize = (5.9, 5))
    #ax1 = fig1.add_subplot(1, 1, 1)
    #ax1.axis([0., 600., 20., 500.])

    ax1.add_patch(matplotlib.patches.Polygon(Schmidt_EPJC_2009_figure6_sn30_N4, 
            closed = True, fill = True, color = 'red', alpha = 0.3))
    ax1.add_patch(matplotlib.patches.Polygon(Schmidt_EPJC_2009_figure6_sn45_N4, 
            closed = True, fill = True, color = 'red', alpha = 0.3))
    ax1.add_patch(matplotlib.patches.Polygon(Schmidt_EPJC_2009_figure6_sn300_N4, 
            closed = True, fill = True, color = 'red', alpha = 0.3))

    ax2.add_patch(matplotlib.patches.Polygon(Schmidt_EPJC_2009_figure6_sn30_N4, 
            closed = True, fill = True, color = 'red', alpha = 0.3))
    ax2.add_patch(matplotlib.patches.Polygon(Schmidt_EPJC_2009_figure6_sn45_N4, 
            closed = True, fill = True, color = 'red', alpha = 0.3))
    ax2.add_patch(matplotlib.patches.Polygon(Schmidt_EPJC_2009_figure6_sn300_N4, 
            closed = True, fill = True, color = 'red', alpha = 0.3))

    ax2.plot(mu_300_2_full, T_300_2_full, '-', c = 'black')
    ax2.plot(mu_300_qgp_full, T_300_qgp_full, '-.', c = 'black')
    ax2.plot(mu_45_2_full, T_45_2_full, '-', c = 'blue')
    ax2.plot(mu_45_qgp_full, T_45_qgp_full, '-.', c = 'blue')
    ax2.plot(mu_30_2_full, T_30_2_full, '-', c = 'green')
    ax2.plot(mu_30_qgp_full, T_30_qgp_full, '-.', c = 'green')

    ax1.plot(mu_300_1_full, T_300_1_full, '-', c = 'black')
    ax1.plot(mu_300_qgp_full, T_300_qgp_full, '-.', c = 'black')
    ax1.plot(mu_45_1_full, T_45_1_full, '-', c = 'blue')
    ax1.plot(mu_45_qgp_full, T_45_qgp_full, '-.', c = 'blue')
    ax1.plot(mu_30_1_full, T_30_1_full, '-', c = 'green')
    ax1.plot(mu_30_qgp_full, T_30_qgp_full, '-.', c = 'green')

    # ax1.plot(Schmidt_new_mu_low_30, Schmidt_new_T_30, '-', c = 'magenta')
    # ax1.plot(Schmidt_new_mu_high_30, Schmidt_new_T_30, '-', c = 'magenta')
    # ax2.plot(Schmidt_new_mu_low_30, Schmidt_new_T_30, '-', c = 'magenta')
    # ax2.plot(Schmidt_new_mu_high_30, Schmidt_new_T_30, '-', c = 'magenta')

    # ax1.plot(Schmidt_new_mu_low_50, Schmidt_new_T_50, '-', c = 'cyan')
    # ax1.plot(Schmidt_new_mu_high_50, Schmidt_new_T_50, '-', c = 'cyan')
    # ax2.plot(Schmidt_new_mu_low_50, Schmidt_new_T_50, '-', c = 'cyan')
    # ax2.plot(Schmidt_new_mu_high_50, Schmidt_new_T_50, '-', c = 'cyan')

    # ax1.plot(Schmidt_new_mu_low_200, Schmidt_new_T_200, '-', c = 'grey')
    # ax1.plot(Schmidt_new_mu_high_200, Schmidt_new_T_200, '-', c = 'grey')
    # ax2.plot(Schmidt_new_mu_low_200, Schmidt_new_T_200, '-', c = 'grey')
    # ax2.plot(Schmidt_new_mu_high_200, Schmidt_new_T_200, '-', c = 'grey')

    # ax1.plot(Schmidt_new_mu_low_400, Schmidt_new_T_400, '-', c = 'grey')
    # ax1.plot(Schmidt_new_mu_high_400, Schmidt_new_T_400, '-', c = 'grey')
    # ax2.plot(Schmidt_new_mu_low_400, Schmidt_new_T_400, '-', c = 'grey')
    # ax2.plot(Schmidt_new_mu_high_400, Schmidt_new_T_400, '-', c = 'grey')

    ax1.plot([el*3.0 for el in mu_300], [pnjl.thermo.gcp_sigma_lattice.Tc(el) for el in mu_300], ':', c = 'black')
    ax2.plot([el*3.0 for el in mu_300], [pnjl.thermo.gcp_sigma_lattice.Tc(el) for el in mu_300], ':', c = 'black')

    ax1.text(69.0, 214.0, r'T/${\rm \mu}$=5.3', c = 'black', fontsize = 14)
    ax1.text(286.0, 214.0, r'T/${\rm \mu}$=0.79', c = 'blue', fontsize = 14)
    ax1.text(440.0, 214.0, r'T/${\rm \mu}$=0.52', c = 'green', fontsize = 14)
    ax1.text(81.0, 132.0, r's/n=300', c = 'black', fontsize = 14)
    ax1.text(225.0, 123.0, r's/n=45', c = 'blue', fontsize = 14)
    ax1.text(450.0, 123.0, r's/n=30', c = 'green', fontsize = 14)
    ax1.text(55, 187, r'Schmidt et al. (2009)', c = 'red', alpha = 0.5, fontsize = 14)
    ax1.text(520, 146, r'$\mathrm{T_c(\mu)}$', c = 'black', fontsize = 14)

    ax2.text(69.0, 214.0, r'T/${\rm \mu}$=5.3', c = 'black', fontsize = 14)
    ax2.text(286.0, 214.0, r'T/${\rm \mu}$=0.79', c = 'blue', fontsize = 14)
    ax2.text(440.0, 214.0, r'T/${\rm \mu}$=0.52', c = 'green', fontsize = 14)
    ax2.text(81.0, 132.0, r's/n=300', c = 'black', fontsize = 14)
    ax2.text(225.0, 123.0, r's/n=45', c = 'blue', fontsize = 14)
    ax2.text(450.0, 123.0, r's/n=30', c = 'green', fontsize = 14)
    ax2.text(55, 187, r'Schmidt et al. (2009)', c = 'red', alpha = 0.5, fontsize = 14)
    ax2.text(520, 146, r'$\mathrm{T_c(\mu)}$', c = 'black', fontsize = 14)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'$\mathrm{\mu_B}$ [MeV]', fontsize = 16)
    ax1.set_ylabel(r'T [MeV]', fontsize = 16)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'$\mathrm{\mu_B}$ [MeV]', fontsize = 16)
    ax2.set_ylabel(r'T [MeV]', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure10_buns():

    import math
    import tqdm
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot
    import matplotlib.patches

    import utils
    import pnjl.defaults
    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")

    calc_0 = True
    calc_1 = True
    calc_2 = True

    calc_mesh_0   = True
    calc_mesh_1   = True
    calc_mesh_2   = True
    calc_mesh_qgp = True
    calc_mesh_no_pert = True

    files = "D:/EoS/epja/figure10_buns/"

    T = numpy.linspace(1.0, 500.0, num=200)

    T_linear = numpy.linspace(1.0, 500.0, num=200)
    mu_linear = numpy.linspace(1.0, 200.0, num=200)

    mu_meshgrid, T_meshgrid = numpy.meshgrid(mu_linear, T_linear)
    mu_T_meshgrid = numpy.zeros_like(mu_meshgrid)
    total_1_meshgrid = numpy.zeros_like(mu_meshgrid)
    total_2_meshgrid = numpy.zeros_like(mu_meshgrid)
    total_qgp_meshgrid = numpy.zeros_like(mu_meshgrid)
    total_no_pert_meshgrid = numpy.zeros_like(mu_meshgrid)

    for mu in numpy.linspace(1.0, 200.0, num=200):

        print(mu)

        mu_round = math.floor(mu*10.0)/10.0

        phi_re_v, phi_im_v = list(), list()

        b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v = \
            list(), list(), list(), list(), list()
        b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v = \
            list(), list(), list()
        b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v = \
            list(), list(), list(), list()

        s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v = \
            list(), list(), list(), list(), list()
        s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v = \
            list(), list(), list()
        s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v = \
            list(), list(), list(), list()

        b_pi_v_1, b_K_v_1, b_rho_v_1, b_omega_v_1, b_D_v_1, b_N_v_1 = \
            list(), list(), list(), list(), list(), list()
        b_T_v_1, b_F_v_1, b_P_v_1, b_Q_v_1, b_H_v_1 = \
            list(), list(), list(), list(), list()

        s_pi_v_1, s_K_v_1, s_rho_v_1, s_omega_v_1, s_D_v_1, s_N_v_1 = \
            list(), list(), list(), list(), list(), list()
        s_T_v_1, s_F_v_1, s_P_v_1, s_Q_v_1, s_H_v_1 = \
            list(), list(), list(), list(), list()
    
        b_pi_v_2, b_K_v_2, b_rho_v_2, b_omega_v_2, b_D_v_2, b_N_v_2 = \
            list(), list(), list(), list(), list(), list()
        b_T_v_2, b_F_v_2, b_P_v_2, b_Q_v_2, b_H_v_2 = \
            list(), list(), list(), list(), list()
    
        s_pi_v_2, s_K_v_2, s_rho_v_2, s_omega_v_2, s_D_v_2, s_N_v_2 = \
            list(), list(), list(), list(), list(), list()
        s_T_v_2, s_F_v_2, s_P_v_2, s_Q_v_2, s_H_v_2 = \
            list(), list(), list(), list(), list()

        if calc_0:

            phi_re_0 = 1e-5
            phi_im_0 = 2e-5
            print("Traced Polyakov loop, mu =", mu_round)
            for T_el in tqdm.tqdm(T, total=len(T), ncols=100):
                phi_result = solver_1.Polyakov_loop(
                    T_el, mu_round, phi_re_0, phi_im_0
                )
                phi_re_v.append(phi_result[0])
                phi_im_v.append(phi_result[1])
                phi_re_0 = phi_result[0]
                phi_im_0 = phi_result[1]
            with open(
                files+"phi_re_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "wb"
            ) as file:
                pickle.dump(phi_re_v, file)
            with open(
                files+"phi_im_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "wb"
            ) as file:
                pickle.dump(phi_im_v, file)

            print("Sigma thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_sigma_v.append(
                    pnjl.thermo.gcp_sigma_lattice.bdensity(T_el, mu_round)
                )
                s_sigma_v.append(
                    pnjl.thermo.gcp_sigma_lattice.sdensity(T_el, mu_round)
                )
            with open(
                files+"b_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sigma_v, file)
            with open(
                files+"s_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sigma_v, file)
            
            print("Gluon thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_gluon_v.append(
                    pnjl.thermo.gcp_pl_polynomial.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                    )
                )
                s_gluon_v.append(
                    pnjl.thermo.gcp_pl_polynomial.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                    )
                )
            with open(
                files+"b_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_gluon_v, file)
            with open(
                files+"s_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_gluon_v, file)

            print("Sea thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_round, 'l'
                )
                b_sea_u_v.append(b_lq_temp)
                b_sea_d_v.append(b_lq_temp)
                b_sea_s_v.append(
                    pnjl.thermo.gcp_sea_lattice.bdensity(T_el, mu_round, 's')
                )
                s_lq_temp = pnjl.thermo.gcp_sea_lattice.sdensity(
                    T_el, mu_round, 'l'
                )
                s_sea_u_v.append(s_lq_temp)
                s_sea_d_v.append(s_lq_temp)
                s_sea_s_v.append(
                    pnjl.thermo.gcp_sea_lattice.sdensity(T_el, mu_round, 's')
                )
            with open(
                files+"b_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sea_u_v, file)
            with open(
                files+"b_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sea_d_v, file)
            with open(
                files+"b_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sea_s_v, file)
            with open(
                files+"s_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sea_u_v, file)
            with open(
                files+"s_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sea_d_v, file)
            with open(
                files+"s_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sea_s_v, file)
            
            print("Perturbative thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                b_perturbative_u_v.append(b_lq_temp)
                b_perturbative_d_v.append(b_lq_temp)
                b_perturbative_s_v.append(
                    pnjl.thermo.gcp_perturbative.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                    )
                )
                s_lq_temp = pnjl.thermo.gcp_perturbative.sdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                s_perturbative_u_v.append(s_lq_temp)
                s_perturbative_d_v.append(s_lq_temp)
                s_perturbative_s_v.append(
                    pnjl.thermo.gcp_perturbative.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                    )
                )
                if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                    raise RuntimeError(
                        "Perturbative gluon bdensity/sdensity not implemented!"
                    )
                else:
                    b_perturbative_gluon_v.append(0.0)
                    s_perturbative_gluon_v.append(0.0)
            with open(
                files+"b_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_u_v, file)
            with open(
                files+"b_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_d_v, file)
            with open(
                files+"b_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_s_v, file)
            with open(
                files+"s_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_u_v, file)
            with open(
                files+"s_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_d_v, file)
            with open(
                files+"s_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_s_v, file)
            with open(
                files+"b_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_gluon_v, file)
            with open(
                files+"s_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_gluon_v, file)

            print("PNJL thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                b_sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )
                s_lq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                s_sq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )
                b_pnjl_u_v.append(b_lq_temp)
                b_pnjl_d_v.append(b_lq_temp)
                b_pnjl_s_v.append(b_sq_temp)
                s_pnjl_u_v.append(s_lq_temp)
                s_pnjl_d_v.append(s_lq_temp)
                s_pnjl_s_v.append(s_sq_temp)
            with open(
                files+"b_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pnjl_u_v, file)
            with open(
                files+"b_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pnjl_d_v, file)
            with open(
                files+"b_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pnjl_s_v, file)
            with open(
                files+"s_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pnjl_u_v, file)
            with open(
                files+"s_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pnjl_d_v, file)
            with open(
                files+"s_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pnjl_s_v, file)
        else:
            with open(
                files+"phi_re_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "rb"
            ) as file:
                phi_re_v = pickle.load(file)
            with open(
                files+"phi_im_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "rb"
            ) as file:
                phi_im_v = pickle.load(file)
            with open(
                files+"b_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sigma_v = pickle.load(file)
            with open(
                files+"s_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sigma_v = pickle.load(file)
            with open(
                files+"b_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_gluon_v = pickle.load(file)
            with open(
                files+"s_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_gluon_v = pickle.load(file)
            with open(
                files+"b_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sea_u_v = pickle.load(file)
            with open(
                files+"b_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sea_d_v = pickle.load(file)
            with open(
                files+"b_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sea_s_v = pickle.load(file)
            with open(
                files+"s_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sea_u_v = pickle.load(file)
            with open(
                files+"s_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sea_d_v = pickle.load(file)
            with open(
                files+"s_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sea_s_v = pickle.load(file)
            with open(
                files+"b_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_u_v = pickle.load(file)
            with open(
                files+"b_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_d_v = pickle.load(file)
            with open(
                files+"b_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_s_v = pickle.load(file)
            with open(
                files+"s_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_u_v = pickle.load(file)
            with open(
                files+"s_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_d_v = pickle.load(file)
            with open(
                files+"s_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_s_v = pickle.load(file)
            with open(
                files+"b_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_gluon_v = pickle.load(file)
            with open(
                files+"s_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_gluon_v = pickle.load(file)
            with open(
                files+"b_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pnjl_u_v = pickle.load(file)
            with open(
                files+"b_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pnjl_d_v = pickle.load(file)
            with open(
                files+"b_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pnjl_s_v = pickle.load(file)
            with open(
                files+"s_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pnjl_u_v = pickle.load(file)
            with open(
                files+"s_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pnjl_d_v = pickle.load(file)
            with open(
                files+"s_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pnjl_s_v = pickle.load(file)

        if calc_1:

            print("Pion thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_pi_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'pi'
                    )
                )
                s_pi_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'pi'
                    )
                )
            with open(
                files+"b_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pi_v_1, file)
            with open(
                files+"s_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pi_v_1, file)

            print("Kaon thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_K_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'K'
                    )
                )
                s_K_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'K'
                    )
                )
            with open(
                files+"b_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_K_v_1, file)
            with open(
                files+"s_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_K_v_1, file)

            print("Rho thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_rho_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'rho'
                    )
                )
                s_rho_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'rho'
                    )
                )
            with open(
                files+"b_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_rho_v_1, file)
            with open(
                files+"s_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_rho_v_1, file)

            print("Omega thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_omega_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'omega'
                    )
                )
                s_omega_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'omega'
                    )
                )
            with open(
                files+"b_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_omega_v_1, file)
            with open(
                files+"s_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_omega_v_1, file)

            print("Diquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_D_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'D'
                    )
                )
                s_D_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'D'
                    )
                )
            with open(
                files+"b_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_D_v_1, file)
            with open(
                files+"s_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_D_v_1, file)

            print("Nucleon thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_N_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'N'
                    )
                )
                s_N_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'N'
                    )
                )
            with open(
                files+"b_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_N_v_1, file)
            with open(
                files+"s_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_N_v_1, file)

            print("Tetraquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_T_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'T'
                    )
                )
                s_T_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'T'
                    )
                )
            with open(
                files+"b_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_T_v_1, file)
            with open(
                files+"s_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_T_v_1, file)

            print("F-quark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_F_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'F'
                    )
                )
                s_F_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'F'
                    )
                )
            with open(
                files+"b_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_F_v_1, file)
            with open(
                files+"s_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_F_v_1, file)

            print("Pentaquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_P_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'P'
                    )
                )
                s_P_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'P'
                    )
                )
            with open(
                files+"b_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_P_v_1, file)
            with open(
                files+"s_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_P_v_1, file)

            print("Q-quark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_Q_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'Q'
                    )
                )
                s_Q_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'Q'
                    )
                )
            with open(
                files+"b_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_Q_v_1, file)
            with open(
                files+"s_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_Q_v_1, file)

            print("Hexaquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_H_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'H'
                    )
                )
                s_H_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'H'
                    )
                )
            with open(
                files+"b_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_H_v_1, file)
            with open(
                files+"s_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_H_v_1, file)
        else:
            with open(
                files+"b_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pi_v_1 = pickle.load(file)
            with open(
                files+"s_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pi_v_1 = pickle.load(file)
            with open(
                files+"b_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_K_v_1 = pickle.load(file)
            with open(
                files+"s_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_K_v_1 = pickle.load(file)
            with open(
                files+"b_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_rho_v_1 = pickle.load(file)
            with open(
                files+"s_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_rho_v_1 = pickle.load(file)
            with open(
                files+"b_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_omega_v_1 = pickle.load(file)
            with open(
                files+"s_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_omega_v_1 = pickle.load(file)
            with open(
                files+"b_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_D_v_1 = pickle.load(file)
            with open(
                files+"s_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_D_v_1 = pickle.load(file)
            with open(
                files+"b_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_N_v_1 = pickle.load(file)
            with open(
                files+"s_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_N_v_1 = pickle.load(file)
            with open(
                files+"b_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_T_v_1 = pickle.load(file)
            with open(
                files+"s_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_T_v_1 = pickle.load(file)
            with open(
                files+"b_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_F_v_1 = pickle.load(file)
            with open(
                files+"s_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_F_v_1 = pickle.load(file)
            with open(
                files+"b_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_P_v_1 = pickle.load(file)
            with open(
                files+"s_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_P_v_1 = pickle.load(file)
            with open(
                files+"b_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_Q_v_1 = pickle.load(file)
            with open(
                files+"s_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_Q_v_1 = pickle.load(file)
            with open(
                files+"b_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_H_v_1 = pickle.load(file)
            with open(
                files+"s_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_H_v_1 = pickle.load(file)

        if calc_2:

            print("Pion thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_pi_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'pi'
                    )
                )
                s_pi_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'pi'
                    )
                )
            with open(
                files+"b_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pi_v_2, file)
            with open(
                files+"s_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pi_v_2, file)

            print("Kaon thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_K_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'K'
                    )
                )
                s_K_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'K'
                    )
                )
            with open(
                files+"b_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_K_v_2, file)
            with open(
                files+"s_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_K_v_2, file)

            print("Rho thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_rho_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'rho'
                    )
                )
                s_rho_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'rho'
                    )
                )
            with open(
                files+"b_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_rho_v_2, file)
            with open(
                files+"s_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_rho_v_2, file)

            print("Omega thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_omega_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'omega'
                    )
                )
                s_omega_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'omega'
                    )
                )
            with open(
                files+"b_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_omega_v_2, file)
            with open(
                files+"s_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_omega_v_2, file)

            print("Diquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_D_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'D'
                    )
                )
                s_D_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'D'
                    )
                )
            with open(
                files+"b_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_D_v_2, file)
            with open(
                files+"s_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_D_v_2, file)

            print("Nucleon thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_N_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'N'
                    )
                )
                s_N_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'N'
                    )
                )
            with open(
                files+"b_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_N_v_2, file)
            with open(
                files+"s_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_N_v_2, file)

            print("Tetraquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_T_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'T'
                    )
                )
                s_T_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'T'
                    )
                )
            with open(
                files+"b_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_T_v_2, file)
            with open(
                files+"s_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_T_v_2, file)

            print("F-quark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_F_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'F'
                    )
                )
                s_F_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'F'
                    )
                )
            with open(
                files+"b_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_F_v_2, file)
            with open(
                files+"s_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_F_v_2, file)

            print("Pentaquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_P_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'P'
                    )
                )
                s_P_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'P'
                    )
                )
            with open(
                files+"b_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_P_v_2, file)
            with open(
                files+"s_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_P_v_2, file)

            print("Q-quark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_Q_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'Q'
                    )
                )
                s_Q_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'Q'
                    )
                )
            with open(
                files+"b_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_Q_v_2, file)
            with open(
                files+"s_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_Q_v_2, file)

            print("Hexaquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_H_v_2.append(
                    cluster.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'H'
                    )
                )
                s_H_v_2.append(
                    cluster.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, 'H'
                    )
                )
            with open(
                files+"b_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_H_v_2, file)
            with open(
                files+"s_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_H_v_2, file)
        else:
            with open(
                files+"b_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pi_v_2 = pickle.load(file)
            with open(
                files+"s_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pi_v_2 = pickle.load(file)
            with open(
                files+"b_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_K_v_2 = pickle.load(file)
            with open(
                files+"s_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_K_v_2 = pickle.load(file)
            with open(
                files+"b_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_rho_v_2 = pickle.load(file)
            with open(
                files+"s_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_rho_v_2 = pickle.load(file)
            with open(
                files+"b_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_omega_v_2 = pickle.load(file)
            with open(
                files+"s_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_omega_v_2 = pickle.load(file)
            with open(
                files+"b_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_D_v_2 = pickle.load(file)
            with open(
                files+"s_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_D_v_2 = pickle.load(file)
            with open(
                files+"b_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_N_v_2 = pickle.load(file)
            with open(
                files+"s_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_N_v_2 = pickle.load(file)
            with open(
                files+"b_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_T_v_2 = pickle.load(file)
            with open(
                files+"s_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_T_v_2 = pickle.load(file)
            with open(
                files+"b_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_F_v_2 = pickle.load(file)
            with open(
                files+"s_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_F_v_2 = pickle.load(file)
            with open(
                files+"b_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_P_v_2 = pickle.load(file)
            with open(
                files+"s_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_P_v_2 = pickle.load(file)
            with open(
                files+"b_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_Q_v_2 = pickle.load(file)
            with open(
                files+"s_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_Q_v_2 = pickle.load(file)
            with open(
                files+"b_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_H_v_2 = pickle.load(file)
            with open(
                files+"s_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_H_v_2 = pickle.load(file)

        total_b_1 = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v,
                b_pi_v_1, b_K_v_1, b_rho_v_1, b_omega_v_1, b_D_v_1, b_N_v_1,
                b_T_v_1, b_F_v_1, b_P_v_1, b_Q_v_1, b_H_v_1
            )
        ]
        total_b_2 = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v,
                b_pi_v_2, b_K_v_2, b_rho_v_2, b_omega_v_2, b_D_v_2, b_N_v_2,
                b_T_v_2, b_F_v_2, b_P_v_2, b_Q_v_2, b_H_v_2
            )
        ]
        total_b_qgp = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v
            )
        ]
        total_b_no_pert = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v
            )
        ]
        total_s_1 = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v,
                s_pi_v_1, s_K_v_1, s_rho_v_1, s_omega_v_1, s_D_v_1, s_N_v_1,
                s_T_v_1, s_F_v_1, s_P_v_1, s_Q_v_1, s_H_v_1
            )
        ]
        total_s_2 = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v,
                s_pi_v_2, s_K_v_2, s_rho_v_2, s_omega_v_2, s_D_v_2, s_N_v_2,
                s_T_v_2, s_F_v_2, s_P_v_2, s_Q_v_2, s_H_v_2
            )
        ]
        total_s_qgp = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v
            )
        ]
        total_s_no_pert = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v
            )
        ]
        total_1 = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_1, total_b_1)]
        total_2 = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_2, total_b_2)]
        total_qgp = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_qgp, total_b_qgp)]
        total_no_pert = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_no_pert, total_b_no_pert)]

        if calc_mesh_0:
            for test_el_T in T_linear:
                test_mub = mu_round
                testing = list(zip(*numpy.where(mu_meshgrid == test_mub)))
                testing2 = list(zip(*numpy.where(T_meshgrid == test_el_T)))
                testing3 = list(set(testing).intersection(testing2))
                if len(testing3) != 1:
                    raise RuntimeError("Coś nie pykło!")
                mu_T_meshgrid[testing3[0][0],testing3[0][1]] = 3.0 * test_mub / test_el_T

        if calc_mesh_1:
            for test_el_T, el_t1 in zip(T_linear, total_1):
                test_mub = mu_round
                testing = list(zip(*numpy.where(mu_meshgrid == test_mub)))
                testing2 = list(zip(*numpy.where(T_meshgrid == test_el_T)))
                testing3 = list(set(testing).intersection(testing2))
                if len(testing3) != 1:
                    raise RuntimeError("Coś nie pykło!")
                total_1_meshgrid[testing3[0][0],testing3[0][1]] = el_t1

        if calc_mesh_2:
            for test_el_T, el_t2 in zip(T_linear, total_2):
                test_mub = mu_round
                testing = list(zip(*numpy.where(mu_meshgrid == test_mub)))
                testing2 = list(zip(*numpy.where(T_meshgrid == test_el_T)))
                testing3 = list(set(testing).intersection(testing2))
                if len(testing3) != 1:
                    raise RuntimeError("Coś nie pykło!")
                total_2_meshgrid[testing3[0][0],testing3[0][1]] = el_t2

        if calc_mesh_qgp:
            for test_el_T, el_tqgp in zip(T_linear, total_qgp):
                test_mub = mu_round
                testing = list(zip(*numpy.where(mu_meshgrid == test_mub)))
                testing2 = list(zip(*numpy.where(T_meshgrid == test_el_T)))
                testing3 = list(set(testing).intersection(testing2))
                if len(testing3) != 1:
                    raise RuntimeError("Coś nie pykło!")
                total_qgp_meshgrid[testing3[0][0],testing3[0][1]] = el_tqgp

        if calc_mesh_no_pert:
            for test_el_T, el_tpert in zip(T_linear, total_no_pert):
                test_mub = mu_round
                testing = list(zip(*numpy.where(mu_meshgrid == test_mub)))
                testing2 = list(zip(*numpy.where(T_meshgrid == test_el_T)))
                testing3 = list(set(testing).intersection(testing2))
                if len(testing3) != 1:
                    raise RuntimeError("Coś nie pykło!")
                total_no_pert_meshgrid[testing3[0][0],testing3[0][1]] = el_tpert

    if calc_mesh_0:
        with open(files+"mu_T_meshgrid.pickle", "wb") as file:
                pickle.dump(mu_T_meshgrid, file)
    else:
        with open(files+"mu_T_meshgrid.pickle", "rb") as file:
                mu_T_meshgrid = pickle.load(file)

    if calc_mesh_1:
        with open(files+"total_1_meshgrid.pickle", "wb") as file:
                pickle.dump(total_1_meshgrid, file)
    else:
        with open(files+"total_1_meshgrid.pickle", "rb") as file:
                total_1_meshgrid = pickle.load(file)

    if calc_mesh_2:
        with open(files+"total_2_meshgrid.pickle", "wb") as file:
                pickle.dump(total_2_meshgrid, file)
    else:
        with open(files+"total_2_meshgrid.pickle", "rb") as file:
                total_2_meshgrid = pickle.load(file)

    if calc_mesh_qgp:
        with open(files+"total_qgp_meshgrid.pickle", "wb") as file:
                pickle.dump(total_qgp_meshgrid, file)
    else:
        with open(files+"total_qgp_meshgrid.pickle", "rb") as file:
                total_qgp_meshgrid = pickle.load(file)

    if calc_mesh_no_pert:
        with open(files+"total_no_pert_meshgrid.pickle", "wb") as file:
                pickle.dump(total_no_pert_meshgrid, file)
    else:
        with open(files+"total_no_pert_meshgrid.pickle", "rb") as file:
                total_no_pert_meshgrid = pickle.load(file)

    lQCD_Tc_x, lQCD_Tc_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_Tc_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_Tc = [[x, y] for x, y in zip(lQCD_Tc_x, lQCD_Tc_y)]

    lQCD_25_x, lQCD_25_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_25_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_25 = [[x, y] for x, y in zip(lQCD_25_x, lQCD_25_y)]
    pQCD_25_x, pQCD_25_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_25_pQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    pQCD_25 = [[x, y] for x, y in zip(pQCD_25_x, pQCD_25_y)]
    lQCD_50_x, lQCD_50_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_50_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_50 = [[x, y] for x, y in zip(lQCD_50_x, lQCD_50_y)]
    pQCD_50_x, pQCD_50_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_50_pQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    pQCD_50 = [[x, y] for x, y in zip(pQCD_50_x, pQCD_50_y)]
    lQCD_100_x, lQCD_100_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_100_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_100 = [[x, y] for x, y in zip(lQCD_100_x, lQCD_100_y)]
    pQCD_100_x, pQCD_100_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_100_pQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    pQCD_100 = [[x, y] for x, y in zip(pQCD_100_x, pQCD_100_y)]
    lQCD_200_x, lQCD_200_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_200_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_200 = [[x, y] for x, y in zip(lQCD_200_x, lQCD_200_y)]
    pQCD_200_x, pQCD_200_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_200_pQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    pQCD_200 = [[x, y] for x, y in zip(pQCD_200_x, pQCD_200_y)]
    lQCD_400_x, lQCD_400_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_400_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_400 = [[x, y] for x, y in zip(lQCD_400_x, lQCD_400_y)]
    pQCD_400_x, pQCD_400_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_400_pQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    pQCD_400 = [[x, y] for x, y in zip(pQCD_400_x, pQCD_400_y)]

    mu_Tc_v = numpy.linspace(0.0, 300.0, 200)
    Tc_Tc_v = [pnjl.thermo.gcp_sigma_lattice.Tc(el) for el in mu_Tc_v]
    mu_over_Tc_v = [3.0 * el_mu / el_tc for el_mu, el_tc in zip(mu_Tc_v, Tc_Tc_v)]

    fig1 = matplotlib.pyplot.figure(num=1, figsize=(11.0, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    ax1.axis([0., 3.5, 120., 300.])
    #ax1.axis([0., 700.0, 50., 250.])
    ax2.axis([0., 3.5, 120., 300.])

    ax1.add_patch(matplotlib.patches.Polygon(lQCD_Tc, 
            closed = True, fill = True, color = 'yellow', alpha = 0.3))

    ax1.add_patch(matplotlib.patches.Polygon(lQCD_25, 
            closed = True, fill = True, color = 'green', alpha = 0.3))
    ax1.add_patch(matplotlib.patches.Polygon(pQCD_25, 
            closed = True, fill = True, color = 'green', alpha = 0.4))
    ax1.add_patch(matplotlib.patches.Polygon(lQCD_50, 
            closed = True, fill = True, color = 'blue', alpha = 0.3))
    ax1.add_patch(matplotlib.patches.Polygon(pQCD_50, 
            closed = True, fill = True, color = 'blue', alpha = 0.4))
    ax1.add_patch(matplotlib.patches.Polygon(lQCD_100, 
            closed = True, fill = True, color = 'purple', alpha = 0.3))
    ax1.add_patch(matplotlib.patches.Polygon(pQCD_100, 
            closed = True, fill = True, color = 'purple', alpha = 0.4))
    # ax1.add_patch(matplotlib.patches.Polygon(lQCD_200, 
    #         closed = True, fill = True, color = 'orange', alpha = 0.3))
    # ax1.add_patch(matplotlib.patches.Polygon(pQCD_200, 
    #         closed = True, fill = True, color = 'orange', alpha = 0.4))
    # ax1.add_patch(matplotlib.patches.Polygon(lQCD_400, 
    #         closed = True, fill = True, color = 'yellow', alpha = 0.3))
    # ax1.add_patch(matplotlib.patches.Polygon(pQCD_400, 
    #         closed = True, fill = True, color = 'yellow', alpha = 0.4))

    CS_1 = ax1.contour(mu_T_meshgrid, T_meshgrid, total_1_meshgrid, levels=[25], colors=["green"])
    CS_1_qgp = ax1.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[25], linestyles="dashed", colors=["green"])
    CS_1_pert = ax1.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[25], linestyles="dotted", colors=["green"])

    CS_3 = ax1.contour(mu_T_meshgrid, T_meshgrid, total_1_meshgrid, levels=[100], colors=["purple"])
    CS_3_qgp = ax1.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[100], linestyles="dashed", colors=["purple"])
    CS_3_pert = ax1.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[100], linestyles="dotted", colors=["purple"])

    CS_5 = ax1.contour(mu_T_meshgrid, T_meshgrid, total_1_meshgrid, levels=[50], colors=["blue"])
    CS_5_qgp = ax1.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[50], linestyles="dashed", colors=["blue"])
    CS_5_pert = ax1.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[50], linestyles="dotted", colors=["blue"])

    ax1.plot([2.3395, 2.38], [258.0, 310.0],':', color="green")
    ax1.plot([2.4051, 2.45], [249.4, 310.0],'-', color="green")

    utils.contour_remove_crap(fig1, CS_1)
    utils.contour_remove_crap(fig1, CS_3)
    utils.contour_remove_crap(fig1, CS_5)

    ax1.plot(mu_over_Tc_v, Tc_Tc_v, '--', c='black')

    ax1.text(3.286, 141.5, r"$\mathrm{T_c}$", color="black", fontsize=14)
    ax1.text(2.32, 285.5, r"25", color="green", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax1.text(1.09, 285.5, r"50", color="blue", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax1.text(0.45, 285.5, r"100", color="purple", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))

    ax2.add_patch(matplotlib.patches.Polygon(lQCD_Tc, 
            closed = True, fill = True, color = 'yellow', alpha = 0.3))

    ax2.add_patch(matplotlib.patches.Polygon(lQCD_25, 
            closed = True, fill = True, color = 'green', alpha = 0.3))
    ax2.add_patch(matplotlib.patches.Polygon(pQCD_25, 
            closed = True, fill = True, color = 'green', alpha = 0.4))
    ax2.add_patch(matplotlib.patches.Polygon(lQCD_50, 
            closed = True, fill = True, color = 'blue', alpha = 0.3))
    ax2.add_patch(matplotlib.patches.Polygon(pQCD_50, 
            closed = True, fill = True, color = 'blue', alpha = 0.4))
    ax2.add_patch(matplotlib.patches.Polygon(lQCD_100, 
            closed = True, fill = True, color = 'purple', alpha = 0.3))
    ax2.add_patch(matplotlib.patches.Polygon(pQCD_100, 
            closed = True, fill = True, color = 'purple', alpha = 0.4))
    # ax2.add_patch(matplotlib.patches.Polygon(lQCD_200, 
    #         closed = True, fill = True, color = 'orange', alpha = 0.3))
    # ax2.add_patch(matplotlib.patches.Polygon(pQCD_200, 
    #         closed = True, fill = True, color = 'orange', alpha = 0.4))
    # ax2.add_patch(matplotlib.patches.Polygon(lQCD_400, 
    #         closed = True, fill = True, color = 'yellow', alpha = 0.3))
    # ax2.add_patch(matplotlib.patches.Polygon(pQCD_400, 
    #         closed = True, fill = True, color = 'yellow', alpha = 0.4))

    CS_2 = ax2.contour(mu_T_meshgrid, T_meshgrid, total_2_meshgrid, levels=[25], colors=["green"])
    CS_2_qgp = ax2.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[25], linestyles="dashed", colors=["green"])
    CS_2_pert = ax2.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[25], linestyles="dotted", colors=["green"])

    CS_4 = ax2.contour(mu_T_meshgrid, T_meshgrid, total_2_meshgrid, levels=[100], colors=["purple"])
    CS_4_qgp = ax2.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[100], linestyles="dashed", colors=["purple"])
    CS_4_pert = ax2.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[100], linestyles="dotted", colors=["purple"])

    CS_6 = ax2.contour(mu_T_meshgrid, T_meshgrid, total_2_meshgrid, levels=[50], colors=["blue"])
    CS_6_qgp = ax2.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[50], linestyles="dashed", colors=["blue"])
    CS_6_pert = ax2.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[50], linestyles="dotted", colors=["blue"])

    ax2.plot([2.3395, 2.38], [258.0, 310.0],':', color="green")
    ax2.plot([2.4051, 2.45], [249.4, 310.0],'-', color="green")

    ax2.plot(mu_over_Tc_v, Tc_Tc_v, '--', c='black')

    ax2.text(3.286, 141.5, r"$\mathrm{T_c}$", color="black", fontsize=14)
    ax2.text(2.32, 285.5, r"25", color="green", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax2.text(1.09, 285.5, r"50", color="blue", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax2.text(0.45, 285.5, r"100", color="purple", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'$\mathrm{\mu_B/T}$', fontsize = 16)
    ax1.set_ylabel(r'T [MeV]', fontsize = 16)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'$\mathrm{\mu_B/T}$', fontsize = 16)
    ax2.set_ylabel(r'T [MeV]', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure10_bu():

    import math
    import tqdm
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot
    import matplotlib.patches

    import utils
    import pnjl.defaults
    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")

    calc_0 = False
    calc_1 = False
    calc_2 = True

    calc_mesh_0   = True
    calc_mesh_1   = True
    calc_mesh_2   = True
    calc_mesh_qgp = True
    calc_mesh_no_pert = True

    files = "D:/EoS/epja/figure10_bu/"

    T = numpy.linspace(1.0, 500.0, num=200)

    T_linear = numpy.linspace(1.0, 500.0, num=200)
    mu_linear = numpy.linspace(1.0, 200.0, num=200)

    mu_meshgrid, T_meshgrid = numpy.meshgrid(mu_linear, T_linear)
    mu_T_meshgrid = numpy.zeros_like(mu_meshgrid)
    total_1_meshgrid = numpy.zeros_like(mu_meshgrid)
    total_2_meshgrid = numpy.zeros_like(mu_meshgrid)
    total_qgp_meshgrid = numpy.zeros_like(mu_meshgrid)
    total_no_pert_meshgrid = numpy.zeros_like(mu_meshgrid)

    for mu in numpy.linspace(1.0, 200.0, num=200):

        print(mu)

        mu_round = math.floor(mu*10.0)/10.0

        phi_re_v, phi_im_v = list(), list()

        b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v = \
            list(), list(), list(), list(), list()
        b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v = \
            list(), list(), list()
        b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v = \
            list(), list(), list(), list()

        s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v = \
            list(), list(), list(), list(), list()
        s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v = \
            list(), list(), list()
        s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v = \
            list(), list(), list(), list()

        b_pi_v_1, b_K_v_1, b_rho_v_1, b_omega_v_1, b_D_v_1, b_N_v_1 = \
            list(), list(), list(), list(), list(), list()
        b_T_v_1, b_F_v_1, b_P_v_1, b_Q_v_1, b_H_v_1 = \
            list(), list(), list(), list(), list()

        s_pi_v_1, s_K_v_1, s_rho_v_1, s_omega_v_1, s_D_v_1, s_N_v_1 = \
            list(), list(), list(), list(), list(), list()
        s_T_v_1, s_F_v_1, s_P_v_1, s_Q_v_1, s_H_v_1 = \
            list(), list(), list(), list(), list()
    
        b_pi_v_2, b_K_v_2, b_rho_v_2, b_omega_v_2, b_D_v_2, b_N_v_2 = \
            list(), list(), list(), list(), list(), list()
        b_T_v_2, b_F_v_2, b_P_v_2, b_Q_v_2, b_H_v_2 = \
            list(), list(), list(), list(), list()
    
        s_pi_v_2, s_K_v_2, s_rho_v_2, s_omega_v_2, s_D_v_2, s_N_v_2 = \
            list(), list(), list(), list(), list(), list()
        s_T_v_2, s_F_v_2, s_P_v_2, s_Q_v_2, s_H_v_2 = \
            list(), list(), list(), list(), list()

        if calc_0:

            phi_re_0 = 1e-5
            phi_im_0 = 2e-5
            print("Traced Polyakov loop, mu =", mu_round)
            for T_el in tqdm.tqdm(T, total=len(T), ncols=100):
                phi_result = solver_1.Polyakov_loop(
                    T_el, mu_round, phi_re_0, phi_im_0
                )
                phi_re_v.append(phi_result[0])
                phi_im_v.append(phi_result[1])
                phi_re_0 = phi_result[0]
                phi_im_0 = phi_result[1]
            with open(
                files+"phi_re_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "wb"
            ) as file:
                pickle.dump(phi_re_v, file)
            with open(
                files+"phi_im_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "wb"
            ) as file:
                pickle.dump(phi_im_v, file)

            print("Sigma thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_sigma_v.append(
                    pnjl.thermo.gcp_sigma_lattice.bdensity(T_el, mu_round)
                )
                s_sigma_v.append(
                    pnjl.thermo.gcp_sigma_lattice.sdensity(T_el, mu_round)
                )
            with open(
                files+"b_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sigma_v, file)
            with open(
                files+"s_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sigma_v, file)
            
            print("Gluon thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_gluon_v.append(
                    pnjl.thermo.gcp_pl_polynomial.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                    )
                )
                s_gluon_v.append(
                    pnjl.thermo.gcp_pl_polynomial.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                    )
                )
            with open(
                files+"b_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_gluon_v, file)
            with open(
                files+"s_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_gluon_v, file)

            print("Sea thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_round, 'l'
                )
                b_sea_u_v.append(b_lq_temp)
                b_sea_d_v.append(b_lq_temp)
                b_sea_s_v.append(
                    pnjl.thermo.gcp_sea_lattice.bdensity(T_el, mu_round, 's')
                )
                s_lq_temp = pnjl.thermo.gcp_sea_lattice.sdensity(
                    T_el, mu_round, 'l'
                )
                s_sea_u_v.append(s_lq_temp)
                s_sea_d_v.append(s_lq_temp)
                s_sea_s_v.append(
                    pnjl.thermo.gcp_sea_lattice.sdensity(T_el, mu_round, 's')
                )
            with open(
                files+"b_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sea_u_v, file)
            with open(
                files+"b_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sea_d_v, file)
            with open(
                files+"b_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_sea_s_v, file)
            with open(
                files+"s_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sea_u_v, file)
            with open(
                files+"s_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sea_d_v, file)
            with open(
                files+"s_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_sea_s_v, file)
            
            print("Perturbative thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                b_perturbative_u_v.append(b_lq_temp)
                b_perturbative_d_v.append(b_lq_temp)
                b_perturbative_s_v.append(
                    pnjl.thermo.gcp_perturbative.bdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                    )
                )
                s_lq_temp = pnjl.thermo.gcp_perturbative.sdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                s_perturbative_u_v.append(s_lq_temp)
                s_perturbative_d_v.append(s_lq_temp)
                s_perturbative_s_v.append(
                    pnjl.thermo.gcp_perturbative.sdensity(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                    )
                )
                if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                    raise RuntimeError(
                        "Perturbative gluon bdensity/sdensity not implemented!"
                    )
                else:
                    b_perturbative_gluon_v.append(0.0)
                    s_perturbative_gluon_v.append(0.0)
            with open(
                files+"b_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_u_v, file)
            with open(
                files+"b_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_d_v, file)
            with open(
                files+"b_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_s_v, file)
            with open(
                files+"s_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_u_v, file)
            with open(
                files+"s_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_d_v, file)
            with open(
                files+"s_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_s_v, file)
            with open(
                files+"b_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_perturbative_gluon_v, file)
            with open(
                files+"s_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_perturbative_gluon_v, file)

            print("PNJL thermo, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                b_sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )
                s_lq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
                )
                s_sq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                    T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )
                b_pnjl_u_v.append(b_lq_temp)
                b_pnjl_d_v.append(b_lq_temp)
                b_pnjl_s_v.append(b_sq_temp)
                s_pnjl_u_v.append(s_lq_temp)
                s_pnjl_d_v.append(s_lq_temp)
                s_pnjl_s_v.append(s_sq_temp)
            with open(
                files+"b_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pnjl_u_v, file)
            with open(
                files+"b_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pnjl_d_v, file)
            with open(
                files+"b_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pnjl_s_v, file)
            with open(
                files+"s_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pnjl_u_v, file)
            with open(
                files+"s_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pnjl_d_v, file)
            with open(
                files+"s_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pnjl_s_v, file)
        else:
            with open(
                files+"phi_re_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "rb"
            ) as file:
                phi_re_v = pickle.load(file)
            with open(
                files+"phi_im_v_" + str(mu_round).replace('.', 'p')+ ".pickle",
                "rb"
            ) as file:
                phi_im_v = pickle.load(file)
            with open(
                files+"b_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sigma_v = pickle.load(file)
            with open(
                files+"s_sigma_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sigma_v = pickle.load(file)
            with open(
                files+"b_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_gluon_v = pickle.load(file)
            with open(
                files+"s_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_gluon_v = pickle.load(file)
            with open(
                files+"b_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sea_u_v = pickle.load(file)
            with open(
                files+"b_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sea_d_v = pickle.load(file)
            with open(
                files+"b_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_sea_s_v = pickle.load(file)
            with open(
                files+"s_sea_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sea_u_v = pickle.load(file)
            with open(
                files+"s_sea_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sea_d_v = pickle.load(file)
            with open(
                files+"s_sea_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_sea_s_v = pickle.load(file)
            with open(
                files+"b_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_u_v = pickle.load(file)
            with open(
                files+"b_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_d_v = pickle.load(file)
            with open(
                files+"b_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_s_v = pickle.load(file)
            with open(
                files+"s_perturbative_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_u_v = pickle.load(file)
            with open(
                files+"s_perturbative_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_d_v = pickle.load(file)
            with open(
                files+"s_perturbative_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_s_v = pickle.load(file)
            with open(
                files+"b_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_perturbative_gluon_v = pickle.load(file)
            with open(
                files+"s_perturbative_gluon_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_perturbative_gluon_v = pickle.load(file)
            with open(
                files+"b_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pnjl_u_v = pickle.load(file)
            with open(
                files+"b_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pnjl_d_v = pickle.load(file)
            with open(
                files+"b_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pnjl_s_v = pickle.load(file)
            with open(
                files+"s_pnjl_u_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pnjl_u_v = pickle.load(file)
            with open(
                files+"s_pnjl_d_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pnjl_d_v = pickle.load(file)
            with open(
                files+"s_pnjl_s_v_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pnjl_s_v = pickle.load(file)

        if calc_1:

            print("Pion thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_pi_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                    )
                )
                s_pi_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                    )
                )
            with open(
                files+"b_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pi_v_1, file)
            with open(
                files+"s_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pi_v_1, file)

            print("Kaon thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_K_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                    )
                )
                s_K_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                    )
                )
            with open(
                files+"b_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_K_v_1, file)
            with open(
                files+"s_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_K_v_1, file)

            print("Rho thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_rho_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                    )
                )
                s_rho_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                    )
                )
            with open(
                files+"b_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_rho_v_1, file)
            with open(
                files+"s_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_rho_v_1, file)

            print("Omega thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_omega_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                    )
                )
                s_omega_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                    )
                )
            with open(
                files+"b_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_omega_v_1, file)
            with open(
                files+"s_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_omega_v_1, file)

            print("Diquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_D_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                    )
                )
                s_D_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                    )
                )
            with open(
                files+"b_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_D_v_1, file)
            with open(
                files+"s_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_D_v_1, file)

            print("Nucleon thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_N_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                    )
                )
                s_N_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                    )
                )
            with open(
                files+"b_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_N_v_1, file)
            with open(
                files+"s_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_N_v_1, file)

            print("Tetraquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_T_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                    )
                )
                s_T_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                    )
                )
            with open(
                files+"b_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_T_v_1, file)
            with open(
                files+"s_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_T_v_1, file)

            print("F-quark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_F_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                    )
                )
                s_F_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                    )
                )
            with open(
                files+"b_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_F_v_1, file)
            with open(
                files+"s_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_F_v_1, file)

            print("Pentaquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_P_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                    )
                )
                s_P_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                    )
                )
            with open(
                files+"b_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_P_v_1, file)
            with open(
                files+"s_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_P_v_1, file)

            print("Q-quark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_Q_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                    )
                )
                s_Q_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                    )
                )
            with open(
                files+"b_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_Q_v_1, file)
            with open(
                files+"s_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_Q_v_1, file)

            print("Hexaquark thermo #1, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_H_v_1.append(
                    cluster_s.bdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                    )
                )
                s_H_v_1.append(
                    cluster_s.sdensity_buns(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                    )
                )
            with open(
                files+"b_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_H_v_1, file)
            with open(
                files+"s_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_H_v_1, file)
        else:
            with open(
                files+"b_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pi_v_1 = pickle.load(file)
            with open(
                files+"s_pi_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pi_v_1 = pickle.load(file)
            with open(
                files+"b_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_K_v_1 = pickle.load(file)
            with open(
                files+"s_K_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_K_v_1 = pickle.load(file)
            with open(
                files+"b_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_rho_v_1 = pickle.load(file)
            with open(
                files+"s_rho_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_rho_v_1 = pickle.load(file)
            with open(
                files+"b_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_omega_v_1 = pickle.load(file)
            with open(
                files+"s_omega_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_omega_v_1 = pickle.load(file)
            with open(
                files+"b_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_D_v_1 = pickle.load(file)
            with open(
                files+"s_D_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_D_v_1 = pickle.load(file)
            with open(
                files+"b_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_N_v_1 = pickle.load(file)
            with open(
                files+"s_N_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_N_v_1 = pickle.load(file)
            with open(
                files+"b_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_T_v_1 = pickle.load(file)
            with open(
                files+"s_T_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_T_v_1 = pickle.load(file)
            with open(
                files+"b_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_F_v_1 = pickle.load(file)
            with open(
                files+"s_F_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_F_v_1 = pickle.load(file)
            with open(
                files+"b_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_P_v_1 = pickle.load(file)
            with open(
                files+"s_P_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_P_v_1 = pickle.load(file)
            with open(
                files+"b_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_Q_v_1 = pickle.load(file)
            with open(
                files+"s_Q_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_Q_v_1 = pickle.load(file)
            with open(
                files+"b_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_H_v_1 = pickle.load(file)
            with open(
                files+"s_H_v_1_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_H_v_1 = pickle.load(file)

        if calc_2:

            print("Pion thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_pi_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                    )
                )
                s_pi_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                    )
                )
            with open(
                files+"b_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_pi_v_2, file)
            with open(
                files+"s_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_pi_v_2, file)

            print("Kaon thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_K_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                    )
                )
                s_K_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                    )
                )
            with open(
                files+"b_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_K_v_2, file)
            with open(
                files+"s_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_K_v_2, file)

            print("Rho thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_rho_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                    )
                )
                s_rho_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                    )
                )
            with open(
                files+"b_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_rho_v_2, file)
            with open(
                files+"s_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_rho_v_2, file)

            print("Omega thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_omega_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                    )
                )
                s_omega_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                    )
                )
            with open(
                files+"b_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_omega_v_2, file)
            with open(
                files+"s_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_omega_v_2, file)

            print("Diquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_D_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                    )
                )
                s_D_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                    )
                )
            with open(
                files+"b_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_D_v_2, file)
            with open(
                files+"s_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_D_v_2, file)

            print("Nucleon thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_N_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                    )
                )
                s_N_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                    )
                )
            with open(
                files+"b_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_N_v_2, file)
            with open(
                files+"s_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_N_v_2, file)

            print("Tetraquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_T_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                    )
                )
                s_T_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                    )
                )
            with open(
                files+"b_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_T_v_2, file)
            with open(
                files+"s_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_T_v_2, file)

            print("F-quark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_F_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                    )
                )
                s_F_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                    )
                )
            with open(
                files+"b_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_F_v_2, file)
            with open(
                files+"s_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_F_v_2, file)

            print("Pentaquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_P_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                    )
                )
                s_P_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                    )
                )
            with open(
                files+"b_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_P_v_2, file)
            with open(
                files+"s_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_P_v_2, file)

            print("Q-quark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_Q_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                    )
                )
                s_Q_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                    )
                )
            with open(
                files+"b_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_Q_v_2, file)
            with open(
                files+"s_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_Q_v_2, file)

            print("Hexaquark thermo #2, mu =", mu_round)
            for T_el, phi_re_el, phi_im_el in tqdm.tqdm(
                zip(T, phi_re_v, phi_im_v), total=len(T), ncols=100
            ):
                b_H_v_2.append(
                    cluster.bdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                    )
                )
                s_H_v_2.append(
                    cluster.sdensity_bu(
                        T_el, mu_round, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                    )
                )
            with open(
                files+"b_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(b_H_v_2, file)
            with open(
                files+"s_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "wb"
            ) as file:
                pickle.dump(s_H_v_2, file)
        else:
            with open(
                files+"b_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_pi_v_2 = pickle.load(file)
            with open(
                files+"s_pi_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_pi_v_2 = pickle.load(file)
            with open(
                files+"b_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_K_v_2 = pickle.load(file)
            with open(
                files+"s_K_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_K_v_2 = pickle.load(file)
            with open(
                files+"b_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_rho_v_2 = pickle.load(file)
            with open(
                files+"s_rho_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_rho_v_2 = pickle.load(file)
            with open(
                files+"b_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_omega_v_2 = pickle.load(file)
            with open(
                files+"s_omega_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_omega_v_2 = pickle.load(file)
            with open(
                files+"b_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_D_v_2 = pickle.load(file)
            with open(
                files+"s_D_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_D_v_2 = pickle.load(file)
            with open(
                files+"b_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_N_v_2 = pickle.load(file)
            with open(
                files+"s_N_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_N_v_2 = pickle.load(file)
            with open(
                files+"b_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_T_v_2 = pickle.load(file)
            with open(
                files+"s_T_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_T_v_2 = pickle.load(file)
            with open(
                files+"b_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_F_v_2 = pickle.load(file)
            with open(
                files+"s_F_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_F_v_2 = pickle.load(file)
            with open(
                files+"b_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_P_v_2 = pickle.load(file)
            with open(
                files+"s_P_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_P_v_2 = pickle.load(file)
            with open(
                files+"b_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_Q_v_2 = pickle.load(file)
            with open(
                files+"s_Q_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_Q_v_2 = pickle.load(file)
            with open(
                files+"b_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                b_H_v_2 = pickle.load(file)
            with open(
                files+"s_H_v_2_" + str(mu_round).replace('.', 'p') + ".pickle",
                "rb"
            ) as file:
                s_H_v_2 = pickle.load(file)

        total_b_1 = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v,
                b_pi_v_1, b_K_v_1, b_rho_v_1, b_omega_v_1, b_D_v_1, b_N_v_1,
                b_T_v_1, b_F_v_1, b_P_v_1, b_Q_v_1, b_H_v_1
            )
        ]
        total_b_2 = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v,
                b_pi_v_2, b_K_v_2, b_rho_v_2, b_omega_v_2, b_D_v_2, b_N_v_2,
                b_T_v_2, b_F_v_2, b_P_v_2, b_Q_v_2, b_H_v_2
            )
        ]
        total_b_qgp = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_gluon_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v
            )
        ]
        total_b_no_pert = [
            math.fsum(el) for el in zip(
                b_sigma_v, b_sea_u_v, b_sea_d_v, b_sea_s_v,
                b_perturbative_u_v, b_perturbative_d_v, b_perturbative_s_v,
                b_perturbative_gluon_v, b_pnjl_u_v, b_pnjl_d_v, b_pnjl_s_v
            )
        ]
        total_s_1 = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v,
                s_pi_v_1, s_K_v_1, s_rho_v_1, s_omega_v_1, s_D_v_1, s_N_v_1,
                s_T_v_1, s_F_v_1, s_P_v_1, s_Q_v_1, s_H_v_1
            )
        ]
        total_s_2 = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v,
                s_pi_v_2, s_K_v_2, s_rho_v_2, s_omega_v_2, s_D_v_2, s_N_v_2,
                s_T_v_2, s_F_v_2, s_P_v_2, s_Q_v_2, s_H_v_2
            )
        ]
        total_s_qgp = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_gluon_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v
            )
        ]
        total_s_no_pert = [
            math.fsum(el) for el in zip(
                s_sigma_v, s_sea_u_v, s_sea_d_v, s_sea_s_v,
                s_perturbative_u_v, s_perturbative_d_v, s_perturbative_s_v,
                s_perturbative_gluon_v, s_pnjl_u_v, s_pnjl_d_v, s_pnjl_s_v
            )
        ]
        total_1 = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_1, total_b_1)]
        total_2 = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_2, total_b_2)]
        total_qgp = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_qgp, total_b_qgp)]
        total_no_pert = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_no_pert, total_b_no_pert)]

        if calc_mesh_0:
            for test_el_T in T_linear:
                test_mub = mu_round
                testing = list(zip(*numpy.where(mu_meshgrid == test_mub)))
                testing2 = list(zip(*numpy.where(T_meshgrid == test_el_T)))
                testing3 = list(set(testing).intersection(testing2))
                if len(testing3) != 1:
                    raise RuntimeError("Coś nie pykło!")
                mu_T_meshgrid[testing3[0][0],testing3[0][1]] = 3.0 * test_mub / test_el_T

        if calc_mesh_1:
            for test_el_T, el_t1 in zip(T_linear, total_1):
                test_mub = mu_round
                testing = list(zip(*numpy.where(mu_meshgrid == test_mub)))
                testing2 = list(zip(*numpy.where(T_meshgrid == test_el_T)))
                testing3 = list(set(testing).intersection(testing2))
                if len(testing3) != 1:
                    raise RuntimeError("Coś nie pykło!")
                total_1_meshgrid[testing3[0][0],testing3[0][1]] = el_t1

        if calc_mesh_2:
            for test_el_T, el_t2 in zip(T_linear, total_2):
                test_mub = mu_round
                testing = list(zip(*numpy.where(mu_meshgrid == test_mub)))
                testing2 = list(zip(*numpy.where(T_meshgrid == test_el_T)))
                testing3 = list(set(testing).intersection(testing2))
                if len(testing3) != 1:
                    raise RuntimeError("Coś nie pykło!")
                total_2_meshgrid[testing3[0][0],testing3[0][1]] = el_t2

        if calc_mesh_qgp:
            for test_el_T, el_tqgp in zip(T_linear, total_qgp):
                test_mub = mu_round
                testing = list(zip(*numpy.where(mu_meshgrid == test_mub)))
                testing2 = list(zip(*numpy.where(T_meshgrid == test_el_T)))
                testing3 = list(set(testing).intersection(testing2))
                if len(testing3) != 1:
                    raise RuntimeError("Coś nie pykło!")
                total_qgp_meshgrid[testing3[0][0],testing3[0][1]] = el_tqgp

        if calc_mesh_no_pert:
            for test_el_T, el_tpert in zip(T_linear, total_no_pert):
                test_mub = mu_round
                testing = list(zip(*numpy.where(mu_meshgrid == test_mub)))
                testing2 = list(zip(*numpy.where(T_meshgrid == test_el_T)))
                testing3 = list(set(testing).intersection(testing2))
                if len(testing3) != 1:
                    raise RuntimeError("Coś nie pykło!")
                total_no_pert_meshgrid[testing3[0][0],testing3[0][1]] = el_tpert

    if calc_mesh_0:
        with open(files+"mu_T_meshgrid.pickle", "wb") as file:
                pickle.dump(mu_T_meshgrid, file)
    else:
        with open(files+"mu_T_meshgrid.pickle", "rb") as file:
                mu_T_meshgrid = pickle.load(file)

    if calc_mesh_1:
        with open(files+"total_1_meshgrid.pickle", "wb") as file:
                pickle.dump(total_1_meshgrid, file)
    else:
        with open(files+"total_1_meshgrid.pickle", "rb") as file:
                total_1_meshgrid = pickle.load(file)

    if calc_mesh_2:
        with open(files+"total_2_meshgrid.pickle", "wb") as file:
                pickle.dump(total_2_meshgrid, file)
    else:
        with open(files+"total_2_meshgrid.pickle", "rb") as file:
                total_2_meshgrid = pickle.load(file)

    if calc_mesh_qgp:
        with open(files+"total_qgp_meshgrid.pickle", "wb") as file:
                pickle.dump(total_qgp_meshgrid, file)
    else:
        with open(files+"total_qgp_meshgrid.pickle", "rb") as file:
                total_qgp_meshgrid = pickle.load(file)

    if calc_mesh_no_pert:
        with open(files+"total_no_pert_meshgrid.pickle", "wb") as file:
                pickle.dump(total_no_pert_meshgrid, file)
    else:
        with open(files+"total_no_pert_meshgrid.pickle", "rb") as file:
                total_no_pert_meshgrid = pickle.load(file)

    lQCD_Tc_x, lQCD_Tc_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_Tc_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_Tc = [[x, y] for x, y in zip(lQCD_Tc_x, lQCD_Tc_y)]

    lQCD_25_x, lQCD_25_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_25_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_25 = [[x, y] for x, y in zip(lQCD_25_x, lQCD_25_y)]
    pQCD_25_x, pQCD_25_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_25_pQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    pQCD_25 = [[x, y] for x, y in zip(pQCD_25_x, pQCD_25_y)]
    lQCD_50_x, lQCD_50_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_50_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_50 = [[x, y] for x, y in zip(lQCD_50_x, lQCD_50_y)]
    pQCD_50_x, pQCD_50_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_50_pQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    pQCD_50 = [[x, y] for x, y in zip(pQCD_50_x, pQCD_50_y)]
    lQCD_100_x, lQCD_100_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_100_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_100 = [[x, y] for x, y in zip(lQCD_100_x, lQCD_100_y)]
    pQCD_100_x, pQCD_100_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_100_pQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    pQCD_100 = [[x, y] for x, y in zip(pQCD_100_x, pQCD_100_y)]
    lQCD_200_x, lQCD_200_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_200_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_200 = [[x, y] for x, y in zip(lQCD_200_x, lQCD_200_y)]
    pQCD_200_x, pQCD_200_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_200_pQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    pQCD_200 = [[x, y] for x, y in zip(pQCD_200_x, pQCD_200_y)]
    lQCD_400_x, lQCD_400_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_400_lQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_400 = [[x, y] for x, y in zip(lQCD_400_x, lQCD_400_y)]
    pQCD_400_x, pQCD_400_y = \
        utils.data_load(
            "D://EoS//lattice_data//sn_muB_over_T//2212_09043_fig14_400_pQCD.dat", 0, 1,
            firstrow=0, delim=' '
        )
    pQCD_400 = [[x, y] for x, y in zip(pQCD_400_x, pQCD_400_y)]

    mu_Tc_v = numpy.linspace(0.0, 300.0, 200)
    Tc_Tc_v = [pnjl.thermo.gcp_sigma_lattice.Tc(el) for el in mu_Tc_v]
    mu_over_Tc_v = [3.0 * el_mu / el_tc for el_mu, el_tc in zip(mu_Tc_v, Tc_Tc_v)]

    fig1 = matplotlib.pyplot.figure(num=1, figsize=(11.0, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    ax1.axis([0., 3.5, 120., 300.])
    #ax1.axis([0., 700.0, 50., 250.])
    ax2.axis([0., 3.5, 120., 300.])

    ax1.add_patch(matplotlib.patches.Polygon(lQCD_Tc, 
            closed = True, fill = True, color = 'yellow', alpha = 0.3))

    ax1.add_patch(matplotlib.patches.Polygon(lQCD_25, 
            closed = True, fill = True, color = 'green', alpha = 0.3))
    ax1.add_patch(matplotlib.patches.Polygon(pQCD_25, 
            closed = True, fill = True, color = 'green', alpha = 0.4))
    ax1.add_patch(matplotlib.patches.Polygon(lQCD_50, 
            closed = True, fill = True, color = 'blue', alpha = 0.3))
    ax1.add_patch(matplotlib.patches.Polygon(pQCD_50, 
            closed = True, fill = True, color = 'blue', alpha = 0.4))
    ax1.add_patch(matplotlib.patches.Polygon(lQCD_100, 
            closed = True, fill = True, color = 'purple', alpha = 0.3))
    ax1.add_patch(matplotlib.patches.Polygon(pQCD_100, 
            closed = True, fill = True, color = 'purple', alpha = 0.4))
    # ax1.add_patch(matplotlib.patches.Polygon(lQCD_200, 
    #         closed = True, fill = True, color = 'orange', alpha = 0.3))
    # ax1.add_patch(matplotlib.patches.Polygon(pQCD_200, 
    #         closed = True, fill = True, color = 'orange', alpha = 0.4))
    # ax1.add_patch(matplotlib.patches.Polygon(lQCD_400, 
    #         closed = True, fill = True, color = 'yellow', alpha = 0.3))
    # ax1.add_patch(matplotlib.patches.Polygon(pQCD_400, 
    #         closed = True, fill = True, color = 'yellow', alpha = 0.4))

    CS_1 = ax1.contour(mu_T_meshgrid, T_meshgrid, total_1_meshgrid, levels=[25], colors=["green"])
    CS_1_qgp = ax1.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[25], linestyles="dashed", colors=["green"])
    CS_1_pert = ax1.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[25], linestyles="dotted", colors=["green"])

    CS_3 = ax1.contour(mu_T_meshgrid, T_meshgrid, total_1_meshgrid, levels=[100], colors=["purple"])
    CS_3_qgp = ax1.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[100], linestyles="dashed", colors=["purple"])
    CS_3_pert = ax1.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[100], linestyles="dotted", colors=["purple"])

    CS_5 = ax1.contour(mu_T_meshgrid, T_meshgrid, total_1_meshgrid, levels=[50], colors=["blue"])
    CS_5_qgp = ax1.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[50], linestyles="dashed", colors=["blue"])
    CS_5_pert = ax1.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[50], linestyles="dotted", colors=["blue"])

    ax1.plot([2.3395, 2.38], [258.0, 310.0],':', color="green")
    ax1.plot([2.4051, 2.45], [249.4, 310.0],'-', color="green")

    utils.contour_remove_crap(fig1, CS_1)
    utils.contour_remove_crap(fig1, CS_3)
    utils.contour_remove_crap(fig1, CS_5)

    ax1.plot(mu_over_Tc_v, Tc_Tc_v, '--', c='black')

    ax1.text(3.286, 141.5, r"$\mathrm{T_c}$", color="black", fontsize=14)
    ax1.text(2.32, 285.5, r"25", color="green", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax1.text(1.09, 285.5, r"50", color="blue", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax1.text(0.45, 285.5, r"100", color="purple", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))

    ax2.add_patch(matplotlib.patches.Polygon(lQCD_Tc, 
            closed = True, fill = True, color = 'yellow', alpha = 0.3))

    ax2.add_patch(matplotlib.patches.Polygon(lQCD_25, 
            closed = True, fill = True, color = 'green', alpha = 0.3))
    ax2.add_patch(matplotlib.patches.Polygon(pQCD_25, 
            closed = True, fill = True, color = 'green', alpha = 0.4))
    ax2.add_patch(matplotlib.patches.Polygon(lQCD_50, 
            closed = True, fill = True, color = 'blue', alpha = 0.3))
    ax2.add_patch(matplotlib.patches.Polygon(pQCD_50, 
            closed = True, fill = True, color = 'blue', alpha = 0.4))
    ax2.add_patch(matplotlib.patches.Polygon(lQCD_100, 
            closed = True, fill = True, color = 'purple', alpha = 0.3))
    ax2.add_patch(matplotlib.patches.Polygon(pQCD_100, 
            closed = True, fill = True, color = 'purple', alpha = 0.4))
    # ax2.add_patch(matplotlib.patches.Polygon(lQCD_200, 
    #         closed = True, fill = True, color = 'orange', alpha = 0.3))
    # ax2.add_patch(matplotlib.patches.Polygon(pQCD_200, 
    #         closed = True, fill = True, color = 'orange', alpha = 0.4))
    # ax2.add_patch(matplotlib.patches.Polygon(lQCD_400, 
    #         closed = True, fill = True, color = 'yellow', alpha = 0.3))
    # ax2.add_patch(matplotlib.patches.Polygon(pQCD_400, 
    #         closed = True, fill = True, color = 'yellow', alpha = 0.4))

    CS_2 = ax2.contour(mu_T_meshgrid, T_meshgrid, total_2_meshgrid, levels=[25], colors=["green"])
    CS_2_qgp = ax2.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[25], linestyles="dashed", colors=["green"])
    CS_2_pert = ax2.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[25], linestyles="dotted", colors=["green"])

    CS_4 = ax2.contour(mu_T_meshgrid, T_meshgrid, total_2_meshgrid, levels=[100], colors=["purple"])
    CS_4_qgp = ax2.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[100], linestyles="dashed", colors=["purple"])
    CS_4_pert = ax2.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[100], linestyles="dotted", colors=["purple"])

    CS_6 = ax2.contour(mu_T_meshgrid, T_meshgrid, total_2_meshgrid, levels=[50], colors=["blue"])
    CS_6_qgp = ax2.contour(mu_T_meshgrid, T_meshgrid, total_qgp_meshgrid, levels=[50], linestyles="dashed", colors=["blue"])
    CS_6_pert = ax2.contour(mu_T_meshgrid, T_meshgrid, total_no_pert_meshgrid, levels=[50], linestyles="dotted", colors=["blue"])

    ax2.plot([2.3395, 2.38], [258.0, 310.0],':', color="green")
    ax2.plot([2.4051, 2.45], [249.4, 310.0],'-', color="green")

    ax2.plot(mu_over_Tc_v, Tc_Tc_v, '--', c='black')

    ax2.text(3.286, 141.5, r"$\mathrm{T_c}$", color="black", fontsize=14)
    ax2.text(2.32, 285.5, r"25", color="green", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax2.text(1.09, 285.5, r"50", color="blue", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax2.text(0.45, 285.5, r"100", color="purple", fontsize=14, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'$\mathrm{\mu_B/T}$', fontsize = 16)
    ax1.set_ylabel(r'T [MeV]', fontsize = 16)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'$\mathrm{\mu_B/T}$', fontsize = 16)
    ax2.set_ylabel(r'T [MeV]', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure11():

    import tqdm
    import math
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

    import utils
    import pnjl.defaults
    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster
    import pnjl.thermo.gcp_cluster.hrg \
        as cluster_h

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")

    calc_1 = False
    calc_2 = False
    calc_3 = False
    calc_4 = False
    calc_5 = False
    calc_6 = False

    files = "D:/EoS/epja/figure11/"

    T_1 = numpy.linspace(1.0, 400.0, 200)
    T_2 = numpy.linspace(1.0, 400.0, 200)
    T_3 = numpy.linspace(1.0, 400.0, 200)
    T_4 = numpy.linspace(1.0, 400.0, 200)
    T_5 = numpy.linspace(1.0, 400.0, 200)
    T_6 = numpy.linspace(1.0, 400.0, 200)

    mu_1 = [0.4 * el for el in T_1]
    mu_2 = [0.4 * el for el in T_2]
    mu_3 = [0.8 * el for el in T_2]
    mu_4 = [0.8 * el for el in T_2]
    mu_5 = [0.4 * el for el in T_2]
    mu_6 = [0.8 * el for el in T_2]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()

    chi_sigma_v_1, chi_gluon_v_1, chi_sea_u_v_1, chi_sea_d_v_1 = \
        list(), list(), list(), list()
    chi_sea_s_v_1, chi_perturbative_u_v_1, chi_perturbative_d_v_1 = \
        list(), list(), list()
    chi_perturbative_s_v_1, chi_perturbative_gluon_v_1 = \
        list(), list()
    chi_pnjl_u_v_1, chi_pnjl_d_v_1, chi_pnjl_s_v_1 = \
        list(), list(), list()

    chi_pi_v_1, chi_K_v_1, chi_rho_v_1, chi_omega_v_1, chi_D_v_1 = \
        list(), list(), list(), list(), list()
    chi_N_v_1, chi_T_v_1, chi_F_v_1, chi_P_v_1, chi_Q_v_1, chi_H_v_1 = \
        list(), list(), list(), list(), list(), list()

    bden_sigma_v_1, bden_gluon_v_1, bden_sea_u_v_1, bden_sea_d_v_1 = \
        list(), list(), list(), list()
    bden_sea_s_v_1, bden_perturbative_u_v_1, bden_perturbative_d_v_1 = \
        list(), list(), list()
    bden_perturbative_s_v_1, bden_perturbative_gluon_v_1 = \
        list(), list()
    bden_pnjl_u_v_1, bden_pnjl_d_v_1, bden_pnjl_s_v_1 = \
        list(), list(), list()

    bden_pi_v_1, bden_K_v_1, bden_rho_v_1, bden_omega_v_1, bden_D_v_1 = \
        list(), list(), list(), list(), list()
    bden_N_v_1, bden_T_v_1, bden_F_v_1, bden_P_v_1, bden_Q_v_1 = \
        list(), list(), list(), list(), list()
    bden_H_v_1 = list()

    phi_re_v_2, phi_im_v_2 = \
        list(), list()

    chi_sigma_v_2, chi_gluon_v_2, chi_sea_u_v_2, chi_sea_d_v_2 = \
        list(), list(), list(), list()
    chi_sea_s_v_2, chi_perturbative_u_v_2, chi_perturbative_d_v_2 = \
        list(), list(), list()
    chi_perturbative_s_v_2, chi_perturbative_gluon_v_2 = \
        list(), list()
    chi_pnjl_u_v_2, chi_pnjl_d_v_2, chi_pnjl_s_v_2 = \
        list(), list(), list()

    chi_pi_v_2, chi_K_v_2, chi_rho_v_2, chi_omega_v_2, chi_D_v_2 = \
        list(), list(), list(), list(), list()
    chi_N_v_2, chi_T_v_2, chi_F_v_2, chi_P_v_2, chi_Q_v_2 = \
        list(), list(), list(), list(), list()
    chi_H_v_2 = list()

    bden_sigma_v_2, bden_gluon_v_2, bden_sea_u_v_2, bden_sea_d_v_2 = \
        list(), list(), list(), list()
    bden_sea_s_v_2, bden_perturbative_u_v_2, bden_perturbative_d_v_2 = \
        list(), list(), list()
    bden_perturbative_s_v_2, bden_perturbative_gluon_v_2 = \
        list(), list()
    bden_pnjl_u_v_2, bden_pnjl_d_v_2, bden_pnjl_s_v_2 = \
        list(), list(), list()

    bden_pi_v_2, bden_K_v_2, bden_rho_v_2, bden_omega_v_2 = \
        list(), list(), list(), list()
    bden_D_v_2, bden_N_v_2, bden_T_v_2, bden_F_v_2 = \
        list(), list(), list(), list()
    bden_P_v_2, bden_Q_v_2, bden_H_v_2 = \
        list(), list(), list()

    phi_re_v_3, phi_im_v_3 = \
        list(), list()

    chi_sigma_v_3, chi_gluon_v_3, chi_sea_u_v_3, chi_sea_d_v_3 = \
        list(), list(), list(), list()
    chi_sea_s_v_3, chi_perturbative_u_v_3, chi_perturbative_d_v_3 = \
        list(), list(), list()
    chi_perturbative_s_v_3, chi_perturbative_gluon_v_3 = \
        list(), list()
    chi_pnjl_u_v_3, chi_pnjl_d_v_3, chi_pnjl_s_v_3 = \
        list(), list(), list()

    chi_pi_v_3, chi_K_v_3, chi_rho_v_3, chi_omega_v_3, chi_D_v_3 = \
        list(), list(), list(), list(), list()
    chi_N_v_3, chi_T_v_3, chi_F_v_3, chi_P_v_3, chi_Q_v_3 = \
        list(), list(), list(), list(), list()
    chi_H_v_3 = list()

    bden_sigma_v_3, bden_gluon_v_3, bden_sea_u_v_3, bden_sea_d_v_3 = \
        list(), list(), list(), list()
    bden_sea_s_v_3, bden_perturbative_u_v_3, bden_perturbative_d_v_3 = \
        list(), list(), list()
    bden_perturbative_s_v_3, bden_perturbative_gluon_v_3 = \
        list(), list()
    bden_pnjl_u_v_3, bden_pnjl_d_v_3, bden_pnjl_s_v_3 = \
        list(), list(), list()

    bden_pi_v_3, bden_K_v_3, bden_rho_v_3, bden_omega_v_3 = \
        list(), list(), list(), list()
    bden_D_v_3, bden_N_v_3, bden_T_v_3, bden_F_v_3 = \
        list(), list(), list(), list()
    bden_P_v_3, bden_Q_v_3, bden_H_v_3 = \
        list(), list(), list()

    phi_re_v_4, phi_im_v_4 = \
        list(), list()

    chi_sigma_v_4, chi_gluon_v_4, chi_sea_u_v_4, chi_sea_d_v_4 = \
        list(), list(), list(), list()
    chi_sea_s_v_4, chi_perturbative_u_v_4, chi_perturbative_d_v_4 = \
        list(), list(), list()
    chi_perturbative_s_v_4, chi_perturbative_gluon_v_4 = \
        list(), list()
    chi_pnjl_u_v_4, chi_pnjl_d_v_4, chi_pnjl_s_v_4 = \
        list(), list(), list()

    chi_pi_v_4, chi_K_v_4, chi_rho_v_4, chi_omega_v_4, chi_D_v_4 = \
        list(), list(), list(), list(), list()
    chi_N_v_4, chi_T_v_4, chi_F_v_4, chi_P_v_4, chi_Q_v_4 = \
        list(), list(), list(), list(), list()
    chi_H_v_4 = list()

    bden_sigma_v_4, bden_gluon_v_4, bden_sea_u_v_4, bden_sea_d_v_4 = \
        list(), list(), list(), list()
    bden_sea_s_v_4, bden_perturbative_u_v_4, bden_perturbative_d_v_4 = \
        list(), list(), list()
    bden_perturbative_s_v_4, bden_perturbative_gluon_v_4 = \
        list(), list()
    bden_pnjl_u_v_4, bden_pnjl_d_v_4, bden_pnjl_s_v_4 = \
        list(), list(), list()

    bden_pi_v_4, bden_K_v_4, bden_rho_v_4, bden_omega_v_4 = \
        list(), list(), list(), list()
    bden_D_v_4, bden_N_v_4, bden_T_v_4, bden_F_v_4 = \
        list(), list(), list(), list()
    bden_P_v_4, bden_Q_v_4, bden_H_v_4 = \
        list(), list(), list()

    phi_re_v_5, phi_im_v_5 = \
        list(), list()

    chi_sigma_v_5, chi_gluon_v_5, chi_sea_u_v_5, chi_sea_d_v_5 = \
        list(), list(), list(), list()
    chi_sea_s_v_5, chi_perturbative_u_v_5, chi_perturbative_d_v_5 = \
        list(), list(), list()
    chi_perturbative_s_v_5, chi_perturbative_gluon_v_5 = \
        list(), list()
    chi_pnjl_u_v_5, chi_pnjl_d_v_5, chi_pnjl_s_v_5 = \
        list(), list(), list()

    chi_pi_v_5, chi_K_v_5, chi_rho_v_5, chi_omega_v_5, chi_D_v_5 = \
        list(), list(), list(), list(), list()
    chi_N_v_5, chi_T_v_5, chi_F_v_5, chi_P_v_5, chi_Q_v_5 = \
        list(), list(), list(), list(), list()
    chi_H_v_5 = list()

    bden_sigma_v_5, bden_gluon_v_5, bden_sea_u_v_5, bden_sea_d_v_5 = \
        list(), list(), list(), list()
    bden_sea_s_v_5, bden_perturbative_u_v_5, bden_perturbative_d_v_5 = \
        list(), list(), list()
    bden_perturbative_s_v_5, bden_perturbative_gluon_v_5 = \
        list(), list()
    bden_pnjl_u_v_5, bden_pnjl_d_v_5, bden_pnjl_s_v_5 = \
        list(), list(), list()

    bden_pi_v_5, bden_K_v_5, bden_rho_v_5, bden_omega_v_5 = \
        list(), list(), list(), list()
    bden_D_v_5, bden_N_v_5, bden_T_v_5, bden_F_v_5 = \
        list(), list(), list(), list()
    bden_P_v_5, bden_Q_v_5, bden_H_v_5 = \
        list(), list(), list()

    phi_re_v_6, phi_im_v_6 = \
        list(), list()

    chi_sigma_v_6, chi_gluon_v_6, chi_sea_u_v_6, chi_sea_d_v_6 = \
        list(), list(), list(), list()
    chi_sea_s_v_6, chi_perturbative_u_v_6, chi_perturbative_d_v_6 = \
        list(), list(), list()
    chi_perturbative_s_v_6, chi_perturbative_gluon_v_6 = \
        list(), list()
    chi_pnjl_u_v_6, chi_pnjl_d_v_6, chi_pnjl_s_v_6 = \
        list(), list(), list()

    chi_pi_v_6, chi_K_v_6, chi_rho_v_6, chi_omega_v_6, chi_D_v_6 = \
        list(), list(), list(), list(), list()
    chi_N_v_6, chi_T_v_6, chi_F_v_6, chi_P_v_6, chi_Q_v_6 = \
        list(), list(), list(), list(), list()
    chi_H_v_6 = list()

    bden_sigma_v_6, bden_gluon_v_6, bden_sea_u_v_6, bden_sea_d_v_6 = \
        list(), list(), list(), list()
    bden_sea_s_v_6, bden_perturbative_u_v_6, bden_perturbative_d_v_6 = \
        list(), list(), list()
    bden_perturbative_s_v_6, bden_perturbative_gluon_v_6 = \
        list(), list()
    bden_pnjl_u_v_6, bden_pnjl_d_v_6, bden_pnjl_s_v_6 = \
        list(), list(), list()

    bden_pi_v_6, bden_K_v_6, bden_rho_v_6, bden_omega_v_6 = \
        list(), list(), list(), list()
    bden_D_v_6, bden_N_v_6, bden_T_v_6, bden_F_v_6 = \
        list(), list(), list(), list()
    bden_P_v_6, bden_Q_v_6, bden_H_v_6 = \
        list(), list(), list()

    if calc_1:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #1")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_1, mu_1), total=len(T_1), ncols= 100
        ):
            phi_result = solver_1.Polyakov_loop(T_el, mu_el, phi_re_0, phi_im_0)
            phi_re_v_1.append(phi_result[0])
            phi_im_v_1.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_1.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_1.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)

        print("Sigma chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols= 100
        ):
            chi_sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.qnumber_cumulant(2, T_el, mu_el)
            )
        with open(files+"chi_sigma_v_1.pickle", "wb") as file:
            pickle.dump(chi_sigma_v_1, file)

        print("Gluon chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols= 100
        ):
            chi_gluon_v_1.append(
                pnjl.thermo.gcp_pl_polynomial.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"chi_gluon_v_1.pickle", "wb") as file:
            pickle.dump(chi_gluon_v_1, file)

        print("Sea chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                2, T_el, mu_el, 'l'
            )
            chi_sea_u_v_1.append(lq_temp)
            chi_sea_d_v_1.append(lq_temp)
            chi_sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                    2, T_el, mu_el, 's'
                )
            )
        with open(files+"chi_sea_u_v_1.pickle", "wb") as file:
            pickle.dump(chi_sea_u_v_1, file)
        with open(files+"chi_sea_d_v_1.pickle", "wb") as file:
            pickle.dump(chi_sea_d_v_1, file)
        with open(files+"chi_sea_s_v_1.pickle", "wb") as file:
            pickle.dump(chi_sea_s_v_1, file)

        print("Perturbative chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            chi_perturbative_u_v_1.append(lq_temp)
            chi_perturbative_d_v_1.append(lq_temp)
            chi_perturbative_s_v_1.append(
                pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon chi_q not implemented")
            else:
                chi_perturbative_gluon_v_1.append(0.0)
        with open(files+"chi_perturbative_u_v_1.pickle", "wb") as file:
            pickle.dump(chi_perturbative_u_v_1, file)
        with open(files+"chi_perturbative_d_v_1.pickle", "wb") as file:
            pickle.dump(chi_perturbative_d_v_1, file)
        with open(files+"chi_perturbative_s_v_1.pickle", "wb") as file:
            pickle.dump(chi_perturbative_s_v_1, file)
        with open(files+"chi_perturbative_gluon_v_1.pickle", "wb") as file:
            pickle.dump(chi_perturbative_gluon_v_1, file)

        print("PNJL chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            chi_pnjl_u_v_1.append(lq_temp)
            chi_pnjl_d_v_1.append(lq_temp)
            chi_pnjl_s_v_1.append(sq_temp)
        with open(files+"chi_pnjl_u_v_1.pickle", "wb") as file:
            pickle.dump(chi_pnjl_u_v_1, file)
        with open(files+"chi_pnjl_d_v_1.pickle", "wb") as file:
            pickle.dump(chi_pnjl_d_v_1, file)
        with open(files+"chi_pnjl_s_v_1.pickle", "wb") as file:
            pickle.dump(chi_pnjl_s_v_1, file)
    else:
        with open(files+"phi_re_v_1.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_1.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
        with open(files+"chi_sigma_v_1.pickle", "rb") as file:
            chi_sigma_v_1 = pickle.load(file)
        with open(files+"chi_gluon_v_1.pickle", "rb") as file:
            chi_gluon_v_1 = pickle.load(file)
        with open(files+"chi_sea_u_v_1.pickle", "rb") as file:
            chi_sea_u_v_1 = pickle.load(file)
        with open(files+"chi_sea_d_v_1.pickle", "rb") as file:
            chi_sea_d_v_1 = pickle.load(file)
        with open(files+"chi_sea_s_v_1.pickle", "rb") as file:
            chi_sea_s_v_1 = pickle.load(file)
        with open(files+"chi_perturbative_u_v_1.pickle", "rb") as file:
            chi_perturbative_u_v_1 = pickle.load(file)
        with open(files+"chi_perturbative_d_v_1.pickle", "rb") as file:
            chi_perturbative_d_v_1 = pickle.load(file)
        with open(files+"chi_perturbative_s_v_1.pickle", "rb") as file:
            chi_perturbative_s_v_1 = pickle.load(file)
        with open(files+"chi_perturbative_gluon_v_1.pickle", "rb") as file:
            chi_perturbative_gluon_v_1 = pickle.load(file)
        with open(files+"chi_pnjl_u_v_1.pickle", "rb") as file:
            chi_pnjl_u_v_1 = pickle.load(file)
        with open(files+"chi_pnjl_d_v_1.pickle", "rb") as file:
            chi_pnjl_d_v_1 = pickle.load(file)
        with open(files+"chi_pnjl_s_v_1.pickle", "rb") as file:
            chi_pnjl_s_v_1 = pickle.load(file)

    if calc_1:

        print("Pion chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_pi_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"chi_pi_v_1.pickle", "wb") as file:
            pickle.dump(chi_pi_v_1, file)

        print("Kaon chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_K_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"chi_K_v_1.pickle", "wb") as file:
            pickle.dump(chi_K_v_1, file)

        print("Rho chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_rho_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"chi_rho_v_1.pickle", "wb") as file:
            pickle.dump(chi_rho_v_1, file)

        print("Omega chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_omega_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"chi_omega_v_1.pickle", "wb") as file:
            pickle.dump(chi_omega_v_1, file)

        print("Diquark chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_D_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"chi_D_v_1.pickle", "wb") as file:
            pickle.dump(chi_D_v_1, file)

        print("Nucleon chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_N_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"chi_N_v_1.pickle", "wb") as file:
            pickle.dump(chi_N_v_1, file)

        print("Tetraquark chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_T_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"chi_T_v_1.pickle", "wb") as file:
            pickle.dump(chi_T_v_1, file)

        print("F-quark chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_F_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"chi_F_v_1.pickle", "wb") as file:
            pickle.dump(chi_F_v_1, file)

        print("Pentaquark chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_P_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"chi_P_v_1.pickle", "wb") as file:
            pickle.dump(chi_P_v_1, file)

        print("Q-quark chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_Q_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"chi_Q_v_1.pickle", "wb") as file:
            pickle.dump(chi_Q_v_1, file)

        print("Hexaquark chi_q #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            chi_H_v_1.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"chi_H_v_1.pickle", "wb") as file:
            pickle.dump(chi_H_v_1, file)
    else:
        with open(files+"chi_pi_v_1.pickle", "rb") as file:
            chi_pi_v_1 = pickle.load(file)
        with open(files+"chi_K_v_1.pickle", "rb") as file:
            chi_K_v_1 = pickle.load(file)
        with open(files+"chi_rho_v_1.pickle", "rb") as file:
            chi_rho_v_1 = pickle.load(file)
        with open(files+"chi_omega_v_1.pickle", "rb") as file:
            chi_omega_v_1 = pickle.load(file)
        with open(files+"chi_D_v_1.pickle", "rb") as file:
            chi_D_v_1 = pickle.load(file)
        with open(files+"chi_N_v_1.pickle", "rb") as file:
            chi_N_v_1 = pickle.load(file)
        with open(files+"chi_T_v_1.pickle", "rb") as file:
            chi_T_v_1 = pickle.load(file)
        with open(files+"chi_F_v_1.pickle", "rb") as file:
            chi_F_v_1 = pickle.load(file)
        with open(files+"chi_P_v_1.pickle", "rb") as file:
            chi_P_v_1 = pickle.load(file)
        with open(files+"chi_Q_v_1.pickle", "rb") as file:
            chi_Q_v_1 = pickle.load(file)
        with open(files+"chi_H_v_1.pickle", "rb") as file:
            chi_H_v_1 = pickle.load(file)

    if calc_1:

        print("Sigma bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols= 100
        ):
            bden_sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(T_el, mu_el)
            )
        with open(files+"bden_sigma_v_1.pickle", "wb") as file:
            pickle.dump(bden_sigma_v_1, file)

        print("Gluon bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols= 100
        ):
            bden_gluon_v_1.append(
                pnjl.thermo.gcp_pl_polynomial.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"bden_gluon_v_1.pickle", "wb") as file:
            pickle.dump(bden_gluon_v_1, file)

        print("Sea bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            bden_sea_u_v_1.append(lq_temp)
            bden_sea_d_v_1.append(lq_temp)
            bden_sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )
            )
        with open(files+"bden_sea_u_v_1.pickle", "wb") as file:
            pickle.dump(bden_sea_u_v_1, file)
        with open(files+"bden_sea_d_v_1.pickle", "wb") as file:
            pickle.dump(bden_sea_d_v_1, file)
        with open(files+"bden_sea_s_v_1.pickle", "wb") as file:
            pickle.dump(bden_sea_s_v_1, file)

        print("Perturbative bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            bden_perturbative_u_v_1.append(lq_temp)
            bden_perturbative_d_v_1.append(lq_temp)
            bden_perturbative_s_v_1.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon bdensity not implemented")
            else:
                bden_perturbative_gluon_v_1.append(0.0)
        with open(files+"bden_perturbative_u_v_1.pickle", "wb") as file:
            pickle.dump(bden_perturbative_u_v_1, file)
        with open(files+"bden_perturbative_d_v_1.pickle", "wb") as file:
            pickle.dump(bden_perturbative_d_v_1, file)
        with open(files+"bden_perturbative_s_v_1.pickle", "wb") as file:
            pickle.dump(bden_perturbative_s_v_1, file)
        with open(files+"bden_perturbative_gluon_v_1.pickle", "wb") as file:
            pickle.dump(bden_perturbative_gluon_v_1, file)

        print("PNJL bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            bden_pnjl_u_v_1.append(lq_temp)
            bden_pnjl_d_v_1.append(lq_temp)
            bden_pnjl_s_v_1.append(sq_temp)
        with open(files+"bden_pnjl_u_v_1.pickle", "wb") as file:
            pickle.dump(bden_pnjl_u_v_1, file)
        with open(files+"bden_pnjl_d_v_1.pickle", "wb") as file:
            pickle.dump(bden_pnjl_d_v_1, file)
        with open(files+"bden_pnjl_s_v_1.pickle", "wb") as file:
            pickle.dump(bden_pnjl_s_v_1, file)
    else:
        with open(files+"bden_sigma_v_1.pickle", "rb") as file:
            bden_sigma_v_1 = pickle.load(file)
        with open(files+"bden_gluon_v_1.pickle", "rb") as file:
            bden_gluon_v_1 = pickle.load(file)
        with open(files+"bden_sea_u_v_1.pickle", "rb") as file:
            bden_sea_u_v_1 = pickle.load(file)
        with open(files+"bden_sea_d_v_1.pickle", "rb") as file:
            bden_sea_d_v_1 = pickle.load(file)
        with open(files+"bden_sea_s_v_1.pickle", "rb") as file:
            bden_sea_s_v_1 = pickle.load(file)
        with open(files+"bden_perturbative_u_v_1.pickle", "rb") as file:
            bden_perturbative_u_v_1 = pickle.load(file)
        with open(files+"bden_perturbative_d_v_1.pickle", "rb") as file:
            bden_perturbative_d_v_1 = pickle.load(file)
        with open(files+"bden_perturbative_s_v_1.pickle", "rb") as file:
            bden_perturbative_s_v_1 = pickle.load(file)
        with open(files+"bden_perturbative_gluon_v_1.pickle", "rb") as file:
            bden_perturbative_gluon_v_1 = pickle.load(file)
        with open(files+"bden_pnjl_u_v_1.pickle", "rb") as file:
            bden_pnjl_u_v_1 = pickle.load(file)
        with open(files+"bden_pnjl_d_v_1.pickle", "rb") as file:
            bden_pnjl_d_v_1 = pickle.load(file)
        with open(files+"bden_pnjl_s_v_1.pickle", "rb") as file:
            bden_pnjl_s_v_1 = pickle.load(file)

    if calc_1:

        print("Pion bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_pi_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"bden_pi_v_1.pickle", "wb") as file:
            pickle.dump(bden_pi_v_1, file)

        print("Kaon bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_K_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"bden_K_v_1.pickle", "wb") as file:
            pickle.dump(bden_K_v_1, file)

        print("Rho bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_rho_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"bden_rho_v_1.pickle", "wb") as file:
            pickle.dump(bden_rho_v_1, file)

        print("Omega bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_omega_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"bden_omega_v_1.pickle", "wb") as file:
            pickle.dump(bden_omega_v_1, file)

        print("Diquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_D_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"bden_D_v_1.pickle", "wb") as file:
            pickle.dump(bden_D_v_1, file)

        print("Nucleon bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_N_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"bden_N_v_1.pickle", "wb") as file:
            pickle.dump(bden_N_v_1, file)

        print("Tetraquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_T_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"bden_T_v_1.pickle", "wb") as file:
            pickle.dump(bden_T_v_1, file)

        print("F-quark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_F_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"bden_F_v_1.pickle", "wb") as file:
            pickle.dump(bden_F_v_1, file)

        print("Pentaquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_P_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"bden_P_v_1.pickle", "wb") as file:
            pickle.dump(bden_P_v_1, file)

        print("Q-quark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_Q_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"bden_Q_v_1.pickle", "wb") as file:
            pickle.dump(bden_Q_v_1, file)

        print("Hexaquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1), ncols=100
        ):
            bden_H_v_1.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"bden_H_v_1.pickle", "wb") as file:
            pickle.dump(bden_H_v_1, file)
    else:
        with open(files+"bden_pi_v_1.pickle", "rb") as file:
            bden_pi_v_1 = pickle.load(file)
        with open(files+"bden_K_v_1.pickle", "rb") as file:
            bden_K_v_1 = pickle.load(file)
        with open(files+"bden_rho_v_1.pickle", "rb") as file:
            bden_rho_v_1 = pickle.load(file)
        with open(files+"bden_omega_v_1.pickle", "rb") as file:
            bden_omega_v_1 = pickle.load(file)
        with open(files+"bden_D_v_1.pickle", "rb") as file:
            bden_D_v_1 = pickle.load(file)
        with open(files+"bden_N_v_1.pickle", "rb") as file:
            bden_N_v_1 = pickle.load(file)
        with open(files+"bden_T_v_1.pickle", "rb") as file:
            bden_T_v_1 = pickle.load(file)
        with open(files+"bden_F_v_1.pickle", "rb") as file:
            bden_F_v_1 = pickle.load(file)
        with open(files+"bden_P_v_1.pickle", "rb") as file:
            bden_P_v_1 = pickle.load(file)
        with open(files+"bden_Q_v_1.pickle", "rb") as file:
            bden_Q_v_1 = pickle.load(file)
        with open(files+"bden_H_v_1.pickle", "rb") as file:
            bden_H_v_1 = pickle.load(file)
    
    if calc_2:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #2")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_2, mu_2), total=len(T_2), ncols= 100
        ):
            phi_result = solver_1.Polyakov_loop(T_el, mu_el, phi_re_0, phi_im_0)
            phi_re_v_2.append(phi_result[0])
            phi_im_v_2.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_2.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)

        print("Sigma chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols= 100
        ):
            chi_sigma_v_2.append(
                pnjl.thermo.gcp_sigma_lattice.qnumber_cumulant(2, T_el, mu_el)
            )
        with open(files+"chi_sigma_v_2.pickle", "wb") as file:
            pickle.dump(chi_sigma_v_2, file)

        print("Gluon chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols= 100
        ):
            chi_gluon_v_2.append(
                pnjl.thermo.gcp_pl_polynomial.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"chi_gluon_v_2.pickle", "wb") as file:
            pickle.dump(chi_gluon_v_2, file)

        print("Sea chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                2, T_el, mu_el, 'l'
            )
            chi_sea_u_v_2.append(lq_temp)
            chi_sea_d_v_2.append(lq_temp)
            chi_sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                    2, T_el, mu_el, 's'
                )
            )
        with open(files+"chi_sea_u_v_2.pickle", "wb") as file:
            pickle.dump(chi_sea_u_v_2, file)
        with open(files+"chi_sea_d_v_2.pickle", "wb") as file:
            pickle.dump(chi_sea_d_v_2, file)
        with open(files+"chi_sea_s_v_2.pickle", "wb") as file:
            pickle.dump(chi_sea_s_v_2, file)

        print("Perturbative chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            chi_perturbative_u_v_2.append(lq_temp)
            chi_perturbative_d_v_2.append(lq_temp)
            chi_perturbative_s_v_2.append(
                pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon chi_q not implemented")
            else:
                chi_perturbative_gluon_v_2.append(0.0)
        with open(files+"chi_perturbative_u_v_2.pickle", "wb") as file:
            pickle.dump(chi_perturbative_u_v_2, file)
        with open(files+"chi_perturbative_d_v_2.pickle", "wb") as file:
            pickle.dump(chi_perturbative_d_v_2, file)
        with open(files+"chi_perturbative_s_v_2.pickle", "wb") as file:
            pickle.dump(chi_perturbative_s_v_2, file)
        with open(files+"chi_perturbative_gluon_v_2.pickle", "wb") as file:
            pickle.dump(chi_perturbative_gluon_v_2, file)

        print("PNJL chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            chi_pnjl_u_v_2.append(lq_temp)
            chi_pnjl_d_v_2.append(lq_temp)
            chi_pnjl_s_v_2.append(sq_temp)
        with open(files+"chi_pnjl_u_v_2.pickle", "wb") as file:
            pickle.dump(chi_pnjl_u_v_2, file)
        with open(files+"chi_pnjl_d_v_2.pickle", "wb") as file:
            pickle.dump(chi_pnjl_d_v_2, file)
        with open(files+"chi_pnjl_s_v_2.pickle", "wb") as file:
            pickle.dump(chi_pnjl_s_v_2, file)
    else:
        with open(files+"phi_re_v_2.pickle", "rb") as file:
            phi_re_v_2 = pickle.load(file)
        with open(files+"phi_im_v_2.pickle", "rb") as file:
            phi_im_v_2 = pickle.load(file)
        with open(files+"chi_sigma_v_2.pickle", "rb") as file:
            chi_sigma_v_2 = pickle.load(file)
        with open(files+"chi_gluon_v_2.pickle", "rb") as file:
            chi_gluon_v_2 = pickle.load(file)
        with open(files+"chi_sea_u_v_2.pickle", "rb") as file:
            chi_sea_u_v_2 = pickle.load(file)
        with open(files+"chi_sea_d_v_2.pickle", "rb") as file:
            chi_sea_d_v_2 = pickle.load(file)
        with open(files+"chi_sea_s_v_2.pickle", "rb") as file:
            chi_sea_s_v_2 = pickle.load(file)
        with open(files+"chi_perturbative_u_v_2.pickle", "rb") as file:
            chi_perturbative_u_v_2 = pickle.load(file)
        with open(files+"chi_perturbative_d_v_2.pickle", "rb") as file:
            chi_perturbative_d_v_2 = pickle.load(file)
        with open(files+"chi_perturbative_s_v_2.pickle", "rb") as file:
            chi_perturbative_s_v_2 = pickle.load(file)
        with open(files+"chi_perturbative_gluon_v_2.pickle", "rb") as file:
            chi_perturbative_gluon_v_2 = pickle.load(file)
        with open(files+"chi_pnjl_u_v_2.pickle", "rb") as file:
            chi_pnjl_u_v_2 = pickle.load(file)
        with open(files+"chi_pnjl_d_v_2.pickle", "rb") as file:
            chi_pnjl_d_v_2 = pickle.load(file)
        with open(files+"chi_pnjl_s_v_2.pickle", "rb") as file:
            chi_pnjl_s_v_2 = pickle.load(file)

    if calc_2:

        print("Pion chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_pi_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"chi_pi_v_2.pickle", "wb") as file:
            pickle.dump(chi_pi_v_2, file)

        print("Kaon chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_K_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"chi_K_v_2.pickle", "wb") as file:
            pickle.dump(chi_K_v_2, file)

        print("Rho chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_rho_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"chi_rho_v_2.pickle", "wb") as file:
            pickle.dump(chi_rho_v_2, file)

        print("Omega chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_omega_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"chi_omega_v_2.pickle", "wb") as file:
            pickle.dump(chi_omega_v_2, file)

        print("Diquark chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_D_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"chi_D_v_2.pickle", "wb") as file:
            pickle.dump(chi_D_v_2, file)

        print("Nucleon chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_N_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"chi_N_v_2.pickle", "wb") as file:
            pickle.dump(chi_N_v_2, file)

        print("Tetraquark chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_T_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"chi_T_v_2.pickle", "wb") as file:
            pickle.dump(chi_T_v_2, file)

        print("F-quark chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_F_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"chi_F_v_2.pickle", "wb") as file:
            pickle.dump(chi_F_v_2, file)

        print("Pentaquark chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_P_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"chi_P_v_2.pickle", "wb") as file:
            pickle.dump(chi_P_v_2, file)

        print("Q-quark chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_Q_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"chi_Q_v_2.pickle", "wb") as file:
            pickle.dump(chi_Q_v_2, file)

        print("Hexaquark chi_q #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            chi_H_v_2.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"chi_H_v_2.pickle", "wb") as file:
            pickle.dump(chi_H_v_2, file)
    else:
        with open(files+"chi_pi_v_2.pickle", "rb") as file:
            chi_pi_v_2 = pickle.load(file)
        with open(files+"chi_K_v_2.pickle", "rb") as file:
            chi_K_v_2 = pickle.load(file)
        with open(files+"chi_rho_v_2.pickle", "rb") as file:
            chi_rho_v_2 = pickle.load(file)
        with open(files+"chi_omega_v_2.pickle", "rb") as file:
            chi_omega_v_2 = pickle.load(file)
        with open(files+"chi_D_v_2.pickle", "rb") as file:
            chi_D_v_2 = pickle.load(file)
        with open(files+"chi_N_v_2.pickle", "rb") as file:
            chi_N_v_2 = pickle.load(file)
        with open(files+"chi_T_v_2.pickle", "rb") as file:
            chi_T_v_2 = pickle.load(file)
        with open(files+"chi_F_v_2.pickle", "rb") as file:
            chi_F_v_2 = pickle.load(file)
        with open(files+"chi_P_v_2.pickle", "rb") as file:
            chi_P_v_2 = pickle.load(file)
        with open(files+"chi_Q_v_2.pickle", "rb") as file:
            chi_Q_v_2 = pickle.load(file)
        with open(files+"chi_H_v_2.pickle", "rb") as file:
            chi_H_v_2 = pickle.load(file)

    if calc_2:

        print("Sigma bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols= 100
        ):
            bden_sigma_v_2.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(T_el, mu_el)
            )
        with open(files+"bden_sigma_v_2.pickle", "wb") as file:
            pickle.dump(bden_sigma_v_2, file)

        print("Gluon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols= 100
        ):
            bden_gluon_v_2.append(
                pnjl.thermo.gcp_pl_polynomial.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"bden_gluon_v_2.pickle", "wb") as file:
            pickle.dump(bden_gluon_v_2, file)

        print("Sea bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            bden_sea_u_v_2.append(lq_temp)
            bden_sea_d_v_2.append(lq_temp)
            bden_sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )
            )
        with open(files+"bden_sea_u_v_2.pickle", "wb") as file:
            pickle.dump(bden_sea_u_v_2, file)
        with open(files+"bden_sea_d_v_2.pickle", "wb") as file:
            pickle.dump(bden_sea_d_v_2, file)
        with open(files+"bden_sea_s_v_2.pickle", "wb") as file:
            pickle.dump(bden_sea_s_v_2, file)

        print("Perturbative bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            bden_perturbative_u_v_2.append(lq_temp)
            bden_perturbative_d_v_2.append(lq_temp)
            bden_perturbative_s_v_2.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon bdensity not implemented")
            else:
                bden_perturbative_gluon_v_2.append(0.0)
        with open(files+"bden_perturbative_u_v_2.pickle", "wb") as file:
            pickle.dump(bden_perturbative_u_v_2, file)
        with open(files+"bden_perturbative_d_v_2.pickle", "wb") as file:
            pickle.dump(bden_perturbative_d_v_2, file)
        with open(files+"bden_perturbative_s_v_2.pickle", "wb") as file:
            pickle.dump(bden_perturbative_s_v_2, file)
        with open(files+"bden_perturbative_gluon_v_2.pickle", "wb") as file:
            pickle.dump(bden_perturbative_gluon_v_2, file)

        print("PNJL bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            bden_pnjl_u_v_2.append(lq_temp)
            bden_pnjl_d_v_2.append(lq_temp)
            bden_pnjl_s_v_2.append(sq_temp)
        with open(files+"bden_pnjl_u_v_2.pickle", "wb") as file:
            pickle.dump(bden_pnjl_u_v_2, file)
        with open(files+"bden_pnjl_d_v_2.pickle", "wb") as file:
            pickle.dump(bden_pnjl_d_v_2, file)
        with open(files+"bden_pnjl_s_v_2.pickle", "wb") as file:
            pickle.dump(bden_pnjl_s_v_2, file)
    else:
        with open(files+"bden_sigma_v_2.pickle", "rb") as file:
            bden_sigma_v_2 = pickle.load(file)
        with open(files+"bden_gluon_v_2.pickle", "rb") as file:
            bden_gluon_v_2 = pickle.load(file)
        with open(files+"bden_sea_u_v_2.pickle", "rb") as file:
            bden_sea_u_v_2 = pickle.load(file)
        with open(files+"bden_sea_d_v_2.pickle", "rb") as file:
            bden_sea_d_v_2 = pickle.load(file)
        with open(files+"bden_sea_s_v_2.pickle", "rb") as file:
            bden_sea_s_v_2 = pickle.load(file)
        with open(files+"bden_perturbative_u_v_2.pickle", "rb") as file:
            bden_perturbative_u_v_2 = pickle.load(file)
        with open(files+"bden_perturbative_d_v_2.pickle", "rb") as file:
            bden_perturbative_d_v_2 = pickle.load(file)
        with open(files+"bden_perturbative_s_v_2.pickle", "rb") as file:
            bden_perturbative_s_v_2 = pickle.load(file)
        with open(files+"bden_perturbative_gluon_v_2.pickle", "rb") as file:
            bden_perturbative_gluon_v_2 = pickle.load(file)
        with open(files+"bden_pnjl_u_v_2.pickle", "rb") as file:
            bden_pnjl_u_v_2 = pickle.load(file)
        with open(files+"bden_pnjl_d_v_2.pickle", "rb") as file:
            bden_pnjl_d_v_2 = pickle.load(file)
        with open(files+"bden_pnjl_s_v_2.pickle", "rb") as file:
            bden_pnjl_s_v_2 = pickle.load(file)

    if calc_2:

        print("Pion bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_pi_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"bden_pi_v_2.pickle", "wb") as file:
            pickle.dump(bden_pi_v_2, file)

        print("Kaon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_K_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"bden_K_v_2.pickle", "wb") as file:
            pickle.dump(bden_K_v_2, file)

        print("Rho bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_rho_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"bden_rho_v_2.pickle", "wb") as file:
            pickle.dump(bden_rho_v_2, file)

        print("Omega bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_omega_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"bden_omega_v_2.pickle", "wb") as file:
            pickle.dump(bden_omega_v_2, file)

        print("Diquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_D_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"bden_D_v_2.pickle", "wb") as file:
            pickle.dump(bden_D_v_2, file)

        print("Nucleon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_N_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"bden_N_v_2.pickle", "wb") as file:
            pickle.dump(bden_N_v_2, file)

        print("Tetraquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_T_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"bden_T_v_2.pickle", "wb") as file:
            pickle.dump(bden_T_v_2, file)

        print("F-quark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_F_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"bden_F_v_2.pickle", "wb") as file:
            pickle.dump(bden_F_v_2, file)

        print("Pentaquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_P_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"bden_P_v_2.pickle", "wb") as file:
            pickle.dump(bden_P_v_2, file)

        print("Q-quark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_Q_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"bden_Q_v_2.pickle", "wb") as file:
            pickle.dump(bden_Q_v_2, file)

        print("Hexaquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2), total=len(T_2), ncols=100
        ):
            bden_H_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"bden_H_v_2.pickle", "wb") as file:
            pickle.dump(bden_H_v_2, file)
    else:
        with open(files+"bden_pi_v_2.pickle", "rb") as file:
            bden_pi_v_2 = pickle.load(file)
        with open(files+"bden_K_v_2.pickle", "rb") as file:
            bden_K_v_2 = pickle.load(file)
        with open(files+"bden_rho_v_2.pickle", "rb") as file:
            bden_rho_v_2 = pickle.load(file)
        with open(files+"bden_omega_v_2.pickle", "rb") as file:
            bden_omega_v_2 = pickle.load(file)
        with open(files+"bden_D_v_2.pickle", "rb") as file:
            bden_D_v_2 = pickle.load(file)
        with open(files+"bden_N_v_2.pickle", "rb") as file:
            bden_N_v_2 = pickle.load(file)
        with open(files+"bden_T_v_2.pickle", "rb") as file:
            bden_T_v_2 = pickle.load(file)
        with open(files+"bden_F_v_2.pickle", "rb") as file:
            bden_F_v_2 = pickle.load(file)
        with open(files+"bden_P_v_2.pickle", "rb") as file:
            bden_P_v_2 = pickle.load(file)
        with open(files+"bden_Q_v_2.pickle", "rb") as file:
            bden_Q_v_2 = pickle.load(file)
        with open(files+"bden_H_v_2.pickle", "rb") as file:
            bden_H_v_2 = pickle.load(file)

    if calc_3:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #3")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_3, mu_3), total=len(T_3), ncols= 100
        ):
            phi_result = solver_1.Polyakov_loop(T_el, mu_el, phi_re_0, phi_im_0)
            phi_re_v_3.append(phi_result[0])
            phi_im_v_3.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_3.pickle", "wb") as file:
            pickle.dump(phi_re_v_3, file)
        with open(files+"phi_im_v_3.pickle", "wb") as file:
            pickle.dump(phi_im_v_3, file)

        print("Sigma chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols= 100
        ):
            chi_sigma_v_3.append(
                pnjl.thermo.gcp_sigma_lattice.qnumber_cumulant(2, T_el, mu_el)
            )
        with open(files+"chi_sigma_v_3.pickle", "wb") as file:
            pickle.dump(chi_sigma_v_3, file)

        print("Gluon chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols= 100
        ):
            chi_gluon_v_3.append(
                pnjl.thermo.gcp_pl_polynomial.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"chi_gluon_v_3.pickle", "wb") as file:
            pickle.dump(chi_gluon_v_3, file)

        print("Sea chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                2, T_el, mu_el, 'l'
            )
            chi_sea_u_v_3.append(lq_temp)
            chi_sea_d_v_3.append(lq_temp)
            chi_sea_s_v_3.append(
                pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                    2, T_el, mu_el, 's'
                )
            )
        with open(files+"chi_sea_u_v_3.pickle", "wb") as file:
            pickle.dump(chi_sea_u_v_3, file)
        with open(files+"chi_sea_d_v_3.pickle", "wb") as file:
            pickle.dump(chi_sea_d_v_3, file)
        with open(files+"chi_sea_s_v_3.pickle", "wb") as file:
            pickle.dump(chi_sea_s_v_3, file)

        print("Perturbative chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            chi_perturbative_u_v_3.append(lq_temp)
            chi_perturbative_d_v_3.append(lq_temp)
            chi_perturbative_s_v_3.append(
                pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon chi_q not implemented")
            else:
                chi_perturbative_gluon_v_3.append(0.0)
        with open(files+"chi_perturbative_u_v_3.pickle", "wb") as file:
            pickle.dump(chi_perturbative_u_v_3, file)
        with open(files+"chi_perturbative_d_v_3.pickle", "wb") as file:
            pickle.dump(chi_perturbative_d_v_3, file)
        with open(files+"chi_perturbative_s_v_3.pickle", "wb") as file:
            pickle.dump(chi_perturbative_s_v_3, file)
        with open(files+"chi_perturbative_gluon_v_3.pickle", "wb") as file:
            pickle.dump(chi_perturbative_gluon_v_3, file)

        print("PNJL chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3),
            total=len(T_3), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            chi_pnjl_u_v_3.append(lq_temp)
            chi_pnjl_d_v_3.append(lq_temp)
            chi_pnjl_s_v_3.append(sq_temp)
        with open(files+"chi_pnjl_u_v_3.pickle", "wb") as file:
            pickle.dump(chi_pnjl_u_v_3, file)
        with open(files+"chi_pnjl_d_v_3.pickle", "wb") as file:
            pickle.dump(chi_pnjl_d_v_3, file)
        with open(files+"chi_pnjl_s_v_3.pickle", "wb") as file:
            pickle.dump(chi_pnjl_s_v_3, file)
    else:
        with open(files+"phi_re_v_3.pickle", "rb") as file:
            phi_re_v_3 = pickle.load(file)
        with open(files+"phi_im_v_3.pickle", "rb") as file:
            phi_im_v_3 = pickle.load(file)
        with open(files+"chi_sigma_v_3.pickle", "rb") as file:
            chi_sigma_v_3 = pickle.load(file)
        with open(files+"chi_gluon_v_3.pickle", "rb") as file:
            chi_gluon_v_3 = pickle.load(file)
        with open(files+"chi_sea_u_v_3.pickle", "rb") as file:
            chi_sea_u_v_3 = pickle.load(file)
        with open(files+"chi_sea_d_v_3.pickle", "rb") as file:
            chi_sea_d_v_3 = pickle.load(file)
        with open(files+"chi_sea_s_v_3.pickle", "rb") as file:
            chi_sea_s_v_3 = pickle.load(file)
        with open(files+"chi_perturbative_u_v_3.pickle", "rb") as file:
            chi_perturbative_u_v_3 = pickle.load(file)
        with open(files+"chi_perturbative_d_v_3.pickle", "rb") as file:
            chi_perturbative_d_v_3 = pickle.load(file)
        with open(files+"chi_perturbative_s_v_3.pickle", "rb") as file:
            chi_perturbative_s_v_3 = pickle.load(file)
        with open(files+"chi_perturbative_gluon_v_3.pickle", "rb") as file:
            chi_perturbative_gluon_v_3 = pickle.load(file)
        with open(files+"chi_pnjl_u_v_3.pickle", "rb") as file:
            chi_pnjl_u_v_3 = pickle.load(file)
        with open(files+"chi_pnjl_d_v_3.pickle", "rb") as file:
            chi_pnjl_d_v_3 = pickle.load(file)
        with open(files+"chi_pnjl_s_v_3.pickle", "rb") as file:
            chi_pnjl_s_v_3 = pickle.load(file)

    if calc_3:

        print("Pion chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_pi_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"chi_pi_v_3.pickle", "wb") as file:
            pickle.dump(chi_pi_v_3, file)

        print("Kaon chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_K_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"chi_K_v_3.pickle", "wb") as file:
            pickle.dump(chi_K_v_3, file)

        print("Rho chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_rho_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"chi_rho_v_3.pickle", "wb") as file:
            pickle.dump(chi_rho_v_3, file)

        print("Omega chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_omega_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"chi_omega_v_3.pickle", "wb") as file:
            pickle.dump(chi_omega_v_3, file)

        print("Diquark chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_D_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"chi_D_v_3.pickle", "wb") as file:
            pickle.dump(chi_D_v_3, file)

        print("Nucleon chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_N_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"chi_N_v_3.pickle", "wb") as file:
            pickle.dump(chi_N_v_3, file)

        print("Tetraquark chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_T_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"chi_T_v_3.pickle", "wb") as file:
            pickle.dump(chi_T_v_3, file)

        print("F-quark chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_F_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"chi_F_v_3.pickle", "wb") as file:
            pickle.dump(chi_F_v_3, file)

        print("Pentaquark chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_P_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"chi_P_v_3.pickle", "wb") as file:
            pickle.dump(chi_P_v_3, file)

        print("Q-quark chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_Q_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"chi_Q_v_3.pickle", "wb") as file:
            pickle.dump(chi_Q_v_3, file)

        print("Hexaquark chi_q #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            chi_H_v_3.append(
                cluster_s.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"chi_H_v_3.pickle", "wb") as file:
            pickle.dump(chi_H_v_3, file)
    else:
        with open(files+"chi_pi_v_3.pickle", "rb") as file:
            chi_pi_v_3 = pickle.load(file)
        with open(files+"chi_K_v_3.pickle", "rb") as file:
            chi_K_v_3 = pickle.load(file)
        with open(files+"chi_rho_v_3.pickle", "rb") as file:
            chi_rho_v_3 = pickle.load(file)
        with open(files+"chi_omega_v_3.pickle", "rb") as file:
            chi_omega_v_3 = pickle.load(file)
        with open(files+"chi_D_v_3.pickle", "rb") as file:
            chi_D_v_3 = pickle.load(file)
        with open(files+"chi_N_v_3.pickle", "rb") as file:
            chi_N_v_3 = pickle.load(file)
        with open(files+"chi_T_v_3.pickle", "rb") as file:
            chi_T_v_3 = pickle.load(file)
        with open(files+"chi_F_v_3.pickle", "rb") as file:
            chi_F_v_3 = pickle.load(file)
        with open(files+"chi_P_v_3.pickle", "rb") as file:
            chi_P_v_3 = pickle.load(file)
        with open(files+"chi_Q_v_3.pickle", "rb") as file:
            chi_Q_v_3 = pickle.load(file)
        with open(files+"chi_H_v_3.pickle", "rb") as file:
            chi_H_v_3 = pickle.load(file)

    if calc_3:

        print("Sigma bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols= 100
        ):
            bden_sigma_v_3.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(T_el, mu_el)
            )
        with open(files+"bden_sigma_v_3.pickle", "wb") as file:
            pickle.dump(bden_sigma_v_3, file)

        print("Gluon bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols= 100
        ):
            bden_gluon_v_3.append(
                pnjl.thermo.gcp_pl_polynomial.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"bden_gluon_v_3.pickle", "wb") as file:
            pickle.dump(bden_gluon_v_3, file)

        print("Sea bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            bden_sea_u_v_3.append(lq_temp)
            bden_sea_d_v_3.append(lq_temp)
            bden_sea_s_v_3.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )
            )
        with open(files+"bden_sea_u_v_3.pickle", "wb") as file:
            pickle.dump(bden_sea_u_v_3, file)
        with open(files+"bden_sea_d_v_3.pickle", "wb") as file:
            pickle.dump(bden_sea_d_v_3, file)
        with open(files+"bden_sea_s_v_3.pickle", "wb") as file:
            pickle.dump(bden_sea_s_v_3, file)

        print("Perturbative bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            bden_perturbative_u_v_3.append(lq_temp)
            bden_perturbative_d_v_3.append(lq_temp)
            bden_perturbative_s_v_3.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon bdensity not implemented")
            else:
                bden_perturbative_gluon_v_3.append(0.0)
        with open(files+"bden_perturbative_u_v_3.pickle", "wb") as file:
            pickle.dump(bden_perturbative_u_v_3, file)
        with open(files+"bden_perturbative_d_v_3.pickle", "wb") as file:
            pickle.dump(bden_perturbative_d_v_3, file)
        with open(files+"bden_perturbative_s_v_3.pickle", "wb") as file:
            pickle.dump(bden_perturbative_s_v_3, file)
        with open(files+"bden_perturbative_gluon_v_3.pickle", "wb") as file:
            pickle.dump(bden_perturbative_gluon_v_3, file)

        print("PNJL bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            bden_pnjl_u_v_3.append(lq_temp)
            bden_pnjl_d_v_3.append(lq_temp)
            bden_pnjl_s_v_3.append(sq_temp)
        with open(files+"bden_pnjl_u_v_3.pickle", "wb") as file:
            pickle.dump(bden_pnjl_u_v_3, file)
        with open(files+"bden_pnjl_d_v_3.pickle", "wb") as file:
            pickle.dump(bden_pnjl_d_v_3, file)
        with open(files+"bden_pnjl_s_v_3.pickle", "wb") as file:
            pickle.dump(bden_pnjl_s_v_3, file)
    else:
        with open(files+"bden_sigma_v_3.pickle", "rb") as file:
            bden_sigma_v_3 = pickle.load(file)
        with open(files+"bden_gluon_v_3.pickle", "rb") as file:
            bden_gluon_v_3 = pickle.load(file)
        with open(files+"bden_sea_u_v_3.pickle", "rb") as file:
            bden_sea_u_v_3 = pickle.load(file)
        with open(files+"bden_sea_d_v_3.pickle", "rb") as file:
            bden_sea_d_v_3 = pickle.load(file)
        with open(files+"bden_sea_s_v_3.pickle", "rb") as file:
            bden_sea_s_v_3 = pickle.load(file)
        with open(files+"bden_perturbative_u_v_3.pickle", "rb") as file:
            bden_perturbative_u_v_3 = pickle.load(file)
        with open(files+"bden_perturbative_d_v_3.pickle", "rb") as file:
            bden_perturbative_d_v_3 = pickle.load(file)
        with open(files+"bden_perturbative_s_v_3.pickle", "rb") as file:
            bden_perturbative_s_v_3 = pickle.load(file)
        with open(files+"bden_perturbative_gluon_v_3.pickle", "rb") as file:
            bden_perturbative_gluon_v_3 = pickle.load(file)
        with open(files+"bden_pnjl_u_v_3.pickle", "rb") as file:
            bden_pnjl_u_v_3 = pickle.load(file)
        with open(files+"bden_pnjl_d_v_3.pickle", "rb") as file:
            bden_pnjl_d_v_3 = pickle.load(file)
        with open(files+"bden_pnjl_s_v_3.pickle", "rb") as file:
            bden_pnjl_s_v_3 = pickle.load(file)

    if calc_3:

        print("Pion bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_pi_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"bden_pi_v_3.pickle", "wb") as file:
            pickle.dump(bden_pi_v_3, file)

        print("Kaon bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_K_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"bden_K_v_3.pickle", "wb") as file:
            pickle.dump(bden_K_v_3, file)

        print("Rho bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_rho_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"bden_rho_v_3.pickle", "wb") as file:
            pickle.dump(bden_rho_v_3, file)

        print("Omega bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_omega_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"bden_omega_v_3.pickle", "wb") as file:
            pickle.dump(bden_omega_v_3, file)

        print("Diquark bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_D_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"bden_D_v_3.pickle", "wb") as file:
            pickle.dump(bden_D_v_3, file)

        print("Nucleon bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_N_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"bden_N_v_3.pickle", "wb") as file:
            pickle.dump(bden_N_v_3, file)

        print("Tetraquark bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_T_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"bden_T_v_3.pickle", "wb") as file:
            pickle.dump(bden_T_v_3, file)

        print("F-quark bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_F_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"bden_F_v_3.pickle", "wb") as file:
            pickle.dump(bden_F_v_3, file)

        print("Pentaquark bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_P_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"bden_P_v_3.pickle", "wb") as file:
            pickle.dump(bden_P_v_3, file)

        print("Q-quark bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_Q_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"bden_Q_v_3.pickle", "wb") as file:
            pickle.dump(bden_Q_v_3, file)

        print("Hexaquark bdensity #3")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_3, mu_3, phi_re_v_3, phi_im_v_3), total=len(T_3), ncols=100
        ):
            bden_H_v_3.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"bden_H_v_3.pickle", "wb") as file:
            pickle.dump(bden_H_v_3, file)
    else:
        with open(files+"bden_pi_v_3.pickle", "rb") as file:
            bden_pi_v_3 = pickle.load(file)
        with open(files+"bden_K_v_3.pickle", "rb") as file:
            bden_K_v_3 = pickle.load(file)
        with open(files+"bden_rho_v_3.pickle", "rb") as file:
            bden_rho_v_3 = pickle.load(file)
        with open(files+"bden_omega_v_3.pickle", "rb") as file:
            bden_omega_v_3 = pickle.load(file)
        with open(files+"bden_D_v_3.pickle", "rb") as file:
            bden_D_v_3 = pickle.load(file)
        with open(files+"bden_N_v_3.pickle", "rb") as file:
            bden_N_v_3 = pickle.load(file)
        with open(files+"bden_T_v_3.pickle", "rb") as file:
            bden_T_v_3 = pickle.load(file)
        with open(files+"bden_F_v_3.pickle", "rb") as file:
            bden_F_v_3 = pickle.load(file)
        with open(files+"bden_P_v_3.pickle", "rb") as file:
            bden_P_v_3 = pickle.load(file)
        with open(files+"bden_Q_v_3.pickle", "rb") as file:
            bden_Q_v_3 = pickle.load(file)
        with open(files+"bden_H_v_3.pickle", "rb") as file:
            bden_H_v_3 = pickle.load(file)
    
    if calc_4:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #4")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_4, mu_4), total=len(T_4), ncols= 100
        ):
            phi_result = solver_1.Polyakov_loop(T_el, mu_el, phi_re_0, phi_im_0)
            phi_re_v_4.append(phi_result[0])
            phi_im_v_4.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_4.pickle", "wb") as file:
            pickle.dump(phi_re_v_4, file)
        with open(files+"phi_im_v_4.pickle", "wb") as file:
            pickle.dump(phi_im_v_4, file)

        print("Sigma chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols= 100
        ):
            chi_sigma_v_4.append(
                pnjl.thermo.gcp_sigma_lattice.qnumber_cumulant(2, T_el, mu_el)
            )
        with open(files+"chi_sigma_v_4.pickle", "wb") as file:
            pickle.dump(chi_sigma_v_4, file)

        print("Gluon chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols= 100
        ):
            chi_gluon_v_4.append(
                pnjl.thermo.gcp_pl_polynomial.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"chi_gluon_v_4.pickle", "wb") as file:
            pickle.dump(chi_gluon_v_4, file)

        print("Sea chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                2, T_el, mu_el, 'l'
            )
            chi_sea_u_v_4.append(lq_temp)
            chi_sea_d_v_4.append(lq_temp)
            chi_sea_s_v_4.append(
                pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                    2, T_el, mu_el, 's'
                )
            )
        with open(files+"chi_sea_u_v_4.pickle", "wb") as file:
            pickle.dump(chi_sea_u_v_4, file)
        with open(files+"chi_sea_d_v_4.pickle", "wb") as file:
            pickle.dump(chi_sea_d_v_4, file)
        with open(files+"chi_sea_s_v_4.pickle", "wb") as file:
            pickle.dump(chi_sea_s_v_4, file)

        print("Perturbative chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            chi_perturbative_u_v_4.append(lq_temp)
            chi_perturbative_d_v_4.append(lq_temp)
            chi_perturbative_s_v_4.append(
                pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon chi_q not implemented")
            else:
                chi_perturbative_gluon_v_4.append(0.0)
        with open(files+"chi_perturbative_u_v_4.pickle", "wb") as file:
            pickle.dump(chi_perturbative_u_v_4, file)
        with open(files+"chi_perturbative_d_v_4.pickle", "wb") as file:
            pickle.dump(chi_perturbative_d_v_4, file)
        with open(files+"chi_perturbative_s_v_4.pickle", "wb") as file:
            pickle.dump(chi_perturbative_s_v_4, file)
        with open(files+"chi_perturbative_gluon_v_4.pickle", "wb") as file:
            pickle.dump(chi_perturbative_gluon_v_4, file)

        print("PNJL chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4),
            total=len(T_4), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            chi_pnjl_u_v_4.append(lq_temp)
            chi_pnjl_d_v_4.append(lq_temp)
            chi_pnjl_s_v_4.append(sq_temp)
        with open(files+"chi_pnjl_u_v_4.pickle", "wb") as file:
            pickle.dump(chi_pnjl_u_v_4, file)
        with open(files+"chi_pnjl_d_v_4.pickle", "wb") as file:
            pickle.dump(chi_pnjl_d_v_4, file)
        with open(files+"chi_pnjl_s_v_4.pickle", "wb") as file:
            pickle.dump(chi_pnjl_s_v_4, file)
    else:
        with open(files+"phi_re_v_4.pickle", "rb") as file:
            phi_re_v_4 = pickle.load(file)
        with open(files+"phi_im_v_4.pickle", "rb") as file:
            phi_im_v_4 = pickle.load(file)
        with open(files+"chi_sigma_v_4.pickle", "rb") as file:
            chi_sigma_v_4 = pickle.load(file)
        with open(files+"chi_gluon_v_4.pickle", "rb") as file:
            chi_gluon_v_4 = pickle.load(file)
        with open(files+"chi_sea_u_v_4.pickle", "rb") as file:
            chi_sea_u_v_4 = pickle.load(file)
        with open(files+"chi_sea_d_v_4.pickle", "rb") as file:
            chi_sea_d_v_4 = pickle.load(file)
        with open(files+"chi_sea_s_v_4.pickle", "rb") as file:
            chi_sea_s_v_4 = pickle.load(file)
        with open(files+"chi_perturbative_u_v_4.pickle", "rb") as file:
            chi_perturbative_u_v_4 = pickle.load(file)
        with open(files+"chi_perturbative_d_v_4.pickle", "rb") as file:
            chi_perturbative_d_v_4 = pickle.load(file)
        with open(files+"chi_perturbative_s_v_4.pickle", "rb") as file:
            chi_perturbative_s_v_4 = pickle.load(file)
        with open(files+"chi_perturbative_gluon_v_4.pickle", "rb") as file:
            chi_perturbative_gluon_v_4 = pickle.load(file)
        with open(files+"chi_pnjl_u_v_4.pickle", "rb") as file:
            chi_pnjl_u_v_4 = pickle.load(file)
        with open(files+"chi_pnjl_d_v_4.pickle", "rb") as file:
            chi_pnjl_d_v_4 = pickle.load(file)
        with open(files+"chi_pnjl_s_v_4.pickle", "rb") as file:
            chi_pnjl_s_v_4 = pickle.load(file)

    if calc_4:

        print("Pion chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_pi_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"chi_pi_v_4.pickle", "wb") as file:
            pickle.dump(chi_pi_v_4, file)

        print("Kaon chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_K_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"chi_K_v_4.pickle", "wb") as file:
            pickle.dump(chi_K_v_4, file)

        print("Rho chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_rho_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"chi_rho_v_4.pickle", "wb") as file:
            pickle.dump(chi_rho_v_4, file)

        print("Omega chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_omega_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"chi_omega_v_4.pickle", "wb") as file:
            pickle.dump(chi_omega_v_4, file)

        print("Diquark chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_D_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"chi_D_v_4.pickle", "wb") as file:
            pickle.dump(chi_D_v_4, file)

        print("Nucleon chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_N_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"chi_N_v_4.pickle", "wb") as file:
            pickle.dump(chi_N_v_4, file)

        print("Tetraquark chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_T_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"chi_T_v_4.pickle", "wb") as file:
            pickle.dump(chi_T_v_4, file)

        print("F-quark chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_F_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"chi_F_v_4.pickle", "wb") as file:
            pickle.dump(chi_F_v_4, file)

        print("Pentaquark chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_P_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"chi_P_v_4.pickle", "wb") as file:
            pickle.dump(chi_P_v_4, file)

        print("Q-quark chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_Q_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"chi_Q_v_4.pickle", "wb") as file:
            pickle.dump(chi_Q_v_4, file)

        print("Hexaquark chi_q #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            chi_H_v_4.append(
                cluster.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"chi_H_v_4.pickle", "wb") as file:
            pickle.dump(chi_H_v_4, file)
    else:
        with open(files+"chi_pi_v_4.pickle", "rb") as file:
            chi_pi_v_4 = pickle.load(file)
        with open(files+"chi_K_v_4.pickle", "rb") as file:
            chi_K_v_4 = pickle.load(file)
        with open(files+"chi_rho_v_4.pickle", "rb") as file:
            chi_rho_v_4 = pickle.load(file)
        with open(files+"chi_omega_v_4.pickle", "rb") as file:
            chi_omega_v_4 = pickle.load(file)
        with open(files+"chi_D_v_4.pickle", "rb") as file:
            chi_D_v_4 = pickle.load(file)
        with open(files+"chi_N_v_4.pickle", "rb") as file:
            chi_N_v_4 = pickle.load(file)
        with open(files+"chi_T_v_4.pickle", "rb") as file:
            chi_T_v_4 = pickle.load(file)
        with open(files+"chi_F_v_4.pickle", "rb") as file:
            chi_F_v_4 = pickle.load(file)
        with open(files+"chi_P_v_4.pickle", "rb") as file:
            chi_P_v_4 = pickle.load(file)
        with open(files+"chi_Q_v_4.pickle", "rb") as file:
            chi_Q_v_4 = pickle.load(file)
        with open(files+"chi_H_v_4.pickle", "rb") as file:
            chi_H_v_4 = pickle.load(file)

    if calc_4:

        print("Sigma bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols= 100
        ):
            bden_sigma_v_4.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(T_el, mu_el)
            )
        with open(files+"bden_sigma_v_4.pickle", "wb") as file:
            pickle.dump(bden_sigma_v_4, file)

        print("Gluon bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols= 100
        ):
            bden_gluon_v_4.append(
                pnjl.thermo.gcp_pl_polynomial.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"bden_gluon_v_4.pickle", "wb") as file:
            pickle.dump(bden_gluon_v_4, file)

        print("Sea bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            bden_sea_u_v_4.append(lq_temp)
            bden_sea_d_v_4.append(lq_temp)
            bden_sea_s_v_4.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )
            )
        with open(files+"bden_sea_u_v_4.pickle", "wb") as file:
            pickle.dump(bden_sea_u_v_4, file)
        with open(files+"bden_sea_d_v_4.pickle", "wb") as file:
            pickle.dump(bden_sea_d_v_4, file)
        with open(files+"bden_sea_s_v_4.pickle", "wb") as file:
            pickle.dump(bden_sea_s_v_4, file)

        print("Perturbative bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            bden_perturbative_u_v_4.append(lq_temp)
            bden_perturbative_d_v_4.append(lq_temp)
            bden_perturbative_s_v_4.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon bdensity not implemented")
            else:
                bden_perturbative_gluon_v_4.append(0.0)
        with open(files+"bden_perturbative_u_v_4.pickle", "wb") as file:
            pickle.dump(bden_perturbative_u_v_4, file)
        with open(files+"bden_perturbative_d_v_4.pickle", "wb") as file:
            pickle.dump(bden_perturbative_d_v_4, file)
        with open(files+"bden_perturbative_s_v_4.pickle", "wb") as file:
            pickle.dump(bden_perturbative_s_v_4, file)
        with open(files+"bden_perturbative_gluon_v_4.pickle", "wb") as file:
            pickle.dump(bden_perturbative_gluon_v_4, file)

        print("PNJL bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            bden_pnjl_u_v_4.append(lq_temp)
            bden_pnjl_d_v_4.append(lq_temp)
            bden_pnjl_s_v_4.append(sq_temp)
        with open(files+"bden_pnjl_u_v_4.pickle", "wb") as file:
            pickle.dump(bden_pnjl_u_v_4, file)
        with open(files+"bden_pnjl_d_v_4.pickle", "wb") as file:
            pickle.dump(bden_pnjl_d_v_4, file)
        with open(files+"bden_pnjl_s_v_4.pickle", "wb") as file:
            pickle.dump(bden_pnjl_s_v_4, file)
    else:
        with open(files+"bden_sigma_v_4.pickle", "rb") as file:
            bden_sigma_v_4 = pickle.load(file)
        with open(files+"bden_gluon_v_4.pickle", "rb") as file:
            bden_gluon_v_4 = pickle.load(file)
        with open(files+"bden_sea_u_v_4.pickle", "rb") as file:
            bden_sea_u_v_4 = pickle.load(file)
        with open(files+"bden_sea_d_v_4.pickle", "rb") as file:
            bden_sea_d_v_4 = pickle.load(file)
        with open(files+"bden_sea_s_v_4.pickle", "rb") as file:
            bden_sea_s_v_4 = pickle.load(file)
        with open(files+"bden_perturbative_u_v_4.pickle", "rb") as file:
            bden_perturbative_u_v_4 = pickle.load(file)
        with open(files+"bden_perturbative_d_v_4.pickle", "rb") as file:
            bden_perturbative_d_v_4 = pickle.load(file)
        with open(files+"bden_perturbative_s_v_4.pickle", "rb") as file:
            bden_perturbative_s_v_4 = pickle.load(file)
        with open(files+"bden_perturbative_gluon_v_4.pickle", "rb") as file:
            bden_perturbative_gluon_v_4 = pickle.load(file)
        with open(files+"bden_pnjl_u_v_4.pickle", "rb") as file:
            bden_pnjl_u_v_4 = pickle.load(file)
        with open(files+"bden_pnjl_d_v_4.pickle", "rb") as file:
            bden_pnjl_d_v_4 = pickle.load(file)
        with open(files+"bden_pnjl_s_v_4.pickle", "rb") as file:
            bden_pnjl_s_v_4 = pickle.load(file)

    if calc_4:

        print("Pion bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_pi_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"bden_pi_v_4.pickle", "wb") as file:
            pickle.dump(bden_pi_v_4, file)

        print("Kaon bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_K_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"bden_K_v_4.pickle", "wb") as file:
            pickle.dump(bden_K_v_4, file)

        print("Rho bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_rho_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"bden_rho_v_4.pickle", "wb") as file:
            pickle.dump(bden_rho_v_4, file)

        print("Omega bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_omega_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"bden_omega_v_4.pickle", "wb") as file:
            pickle.dump(bden_omega_v_4, file)

        print("Diquark bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_D_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"bden_D_v_4.pickle", "wb") as file:
            pickle.dump(bden_D_v_4, file)

        print("Nucleon bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_N_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"bden_N_v_4.pickle", "wb") as file:
            pickle.dump(bden_N_v_4, file)

        print("Tetraquark bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_T_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"bden_T_v_4.pickle", "wb") as file:
            pickle.dump(bden_T_v_4, file)

        print("F-quark bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_F_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"bden_F_v_4.pickle", "wb") as file:
            pickle.dump(bden_F_v_4, file)

        print("Pentaquark bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_P_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"bden_P_v_4.pickle", "wb") as file:
            pickle.dump(bden_P_v_4, file)

        print("Q-quark bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_Q_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"bden_Q_v_4.pickle", "wb") as file:
            pickle.dump(bden_Q_v_4, file)

        print("Hexaquark bdensity #4")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_4, mu_4, phi_re_v_4, phi_im_v_4), total=len(T_4), ncols=100
        ):
            bden_H_v_4.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"bden_H_v_4.pickle", "wb") as file:
            pickle.dump(bden_H_v_4, file)
    else:
        with open(files+"bden_pi_v_4.pickle", "rb") as file:
            bden_pi_v_4 = pickle.load(file)
        with open(files+"bden_K_v_4.pickle", "rb") as file:
            bden_K_v_4 = pickle.load(file)
        with open(files+"bden_rho_v_4.pickle", "rb") as file:
            bden_rho_v_4 = pickle.load(file)
        with open(files+"bden_omega_v_4.pickle", "rb") as file:
            bden_omega_v_4 = pickle.load(file)
        with open(files+"bden_D_v_4.pickle", "rb") as file:
            bden_D_v_4 = pickle.load(file)
        with open(files+"bden_N_v_4.pickle", "rb") as file:
            bden_N_v_4 = pickle.load(file)
        with open(files+"bden_T_v_4.pickle", "rb") as file:
            bden_T_v_4 = pickle.load(file)
        with open(files+"bden_F_v_4.pickle", "rb") as file:
            bden_F_v_4 = pickle.load(file)
        with open(files+"bden_P_v_4.pickle", "rb") as file:
            bden_P_v_4 = pickle.load(file)
        with open(files+"bden_Q_v_4.pickle", "rb") as file:
            bden_Q_v_4 = pickle.load(file)
        with open(files+"bden_H_v_4.pickle", "rb") as file:
            bden_H_v_4 = pickle.load(file)
 
    if calc_5:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #5")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_5, mu_5), total=len(T_5), ncols= 100
        ):
            phi_result = solver_1.Polyakov_loop(T_el, mu_el, phi_re_0, phi_im_0)
            phi_re_v_5.append(phi_result[0])
            phi_im_v_5.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_5.pickle", "wb") as file:
            pickle.dump(phi_re_v_5, file)
        with open(files+"phi_im_v_5.pickle", "wb") as file:
            pickle.dump(phi_im_v_5, file)

        print("Sigma chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols= 100
        ):
            chi_sigma_v_5.append(
                pnjl.thermo.gcp_sigma_lattice.qnumber_cumulant(2, T_el, mu_el)
            )
        with open(files+"chi_sigma_v_5.pickle", "wb") as file:
            pickle.dump(chi_sigma_v_5, file)

        print("Gluon chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5),
            total=len(T_5), ncols= 100
        ):
            chi_gluon_v_5.append(
                pnjl.thermo.gcp_pl_polynomial.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"chi_gluon_v_5.pickle", "wb") as file:
            pickle.dump(chi_gluon_v_5, file)

        print("Sea chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5),
            total=len(T_5), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                2, T_el, mu_el, 'l'
            )
            chi_sea_u_v_5.append(lq_temp)
            chi_sea_d_v_5.append(lq_temp)
            chi_sea_s_v_5.append(
                pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                    2, T_el, mu_el, 's'
                )
            )
        with open(files+"chi_sea_u_v_5.pickle", "wb") as file:
            pickle.dump(chi_sea_u_v_5, file)
        with open(files+"chi_sea_d_v_5.pickle", "wb") as file:
            pickle.dump(chi_sea_d_v_5, file)
        with open(files+"chi_sea_s_v_5.pickle", "wb") as file:
            pickle.dump(chi_sea_s_v_5, file)

        print("Perturbative chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5),
            total=len(T_5), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            chi_perturbative_u_v_5.append(lq_temp)
            chi_perturbative_d_v_5.append(lq_temp)
            chi_perturbative_s_v_5.append(
                pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon chi_q not implemented")
            else:
                chi_perturbative_gluon_v_5.append(0.0)
        with open(files+"chi_perturbative_u_v_5.pickle", "wb") as file:
            pickle.dump(chi_perturbative_u_v_5, file)
        with open(files+"chi_perturbative_d_v_5.pickle", "wb") as file:
            pickle.dump(chi_perturbative_d_v_5, file)
        with open(files+"chi_perturbative_s_v_5.pickle", "wb") as file:
            pickle.dump(chi_perturbative_s_v_5, file)
        with open(files+"chi_perturbative_gluon_v_5.pickle", "wb") as file:
            pickle.dump(chi_perturbative_gluon_v_5, file)

        print("PNJL chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5),
            total=len(T_5), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            chi_pnjl_u_v_5.append(lq_temp)
            chi_pnjl_d_v_5.append(lq_temp)
            chi_pnjl_s_v_5.append(sq_temp)
        with open(files+"chi_pnjl_u_v_5.pickle", "wb") as file:
            pickle.dump(chi_pnjl_u_v_5, file)
        with open(files+"chi_pnjl_d_v_5.pickle", "wb") as file:
            pickle.dump(chi_pnjl_d_v_5, file)
        with open(files+"chi_pnjl_s_v_5.pickle", "wb") as file:
            pickle.dump(chi_pnjl_s_v_5, file)
    else:
        with open(files+"phi_re_v_5.pickle", "rb") as file:
            phi_re_v_5 = pickle.load(file)
        with open(files+"phi_im_v_5.pickle", "rb") as file:
            phi_im_v_5 = pickle.load(file)
        with open(files+"chi_sigma_v_5.pickle", "rb") as file:
            chi_sigma_v_5 = pickle.load(file)
        with open(files+"chi_gluon_v_5.pickle", "rb") as file:
            chi_gluon_v_5 = pickle.load(file)
        with open(files+"chi_sea_u_v_5.pickle", "rb") as file:
            chi_sea_u_v_5 = pickle.load(file)
        with open(files+"chi_sea_d_v_5.pickle", "rb") as file:
            chi_sea_d_v_5 = pickle.load(file)
        with open(files+"chi_sea_s_v_5.pickle", "rb") as file:
            chi_sea_s_v_5 = pickle.load(file)
        with open(files+"chi_perturbative_u_v_5.pickle", "rb") as file:
            chi_perturbative_u_v_5 = pickle.load(file)
        with open(files+"chi_perturbative_d_v_5.pickle", "rb") as file:
            chi_perturbative_d_v_5 = pickle.load(file)
        with open(files+"chi_perturbative_s_v_5.pickle", "rb") as file:
            chi_perturbative_s_v_5 = pickle.load(file)
        with open(files+"chi_perturbative_gluon_v_5.pickle", "rb") as file:
            chi_perturbative_gluon_v_5 = pickle.load(file)
        with open(files+"chi_pnjl_u_v_5.pickle", "rb") as file:
            chi_pnjl_u_v_5 = pickle.load(file)
        with open(files+"chi_pnjl_d_v_5.pickle", "rb") as file:
            chi_pnjl_d_v_5 = pickle.load(file)
        with open(files+"chi_pnjl_s_v_5.pickle", "rb") as file:
            chi_pnjl_s_v_5 = pickle.load(file)

    if calc_5:

        print("Pion chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_pi_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"chi_pi_v_5.pickle", "wb") as file:
            pickle.dump(chi_pi_v_5, file)

        print("Kaon chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_K_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"chi_K_v_5.pickle", "wb") as file:
            pickle.dump(chi_K_v_5, file)

        print("Rho chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_rho_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"chi_rho_v_5.pickle", "wb") as file:
            pickle.dump(chi_rho_v_5, file)

        print("Omega chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_omega_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"chi_omega_v_5.pickle", "wb") as file:
            pickle.dump(chi_omega_v_5, file)

        print("Diquark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_D_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"chi_D_v_5.pickle", "wb") as file:
            pickle.dump(chi_D_v_5, file)

        print("Nucleon chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_N_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"chi_N_v_5.pickle", "wb") as file:
            pickle.dump(chi_N_v_5, file)

        print("Tetraquark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_T_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"chi_T_v_5.pickle", "wb") as file:
            pickle.dump(chi_T_v_5, file)

        print("F-quark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_F_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"chi_F_v_5.pickle", "wb") as file:
            pickle.dump(chi_F_v_5, file)

        print("Pentaquark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_P_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"chi_P_v_5.pickle", "wb") as file:
            pickle.dump(chi_P_v_5, file)

        print("Q-quark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_Q_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"chi_Q_v_5.pickle", "wb") as file:
            pickle.dump(chi_Q_v_5, file)

        print("Hexaquark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            chi_H_v_5.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"chi_H_v_5.pickle", "wb") as file:
            pickle.dump(chi_H_v_5, file)
    else:
        with open(files+"chi_pi_v_5.pickle", "rb") as file:
            chi_pi_v_5 = pickle.load(file)
        with open(files+"chi_K_v_5.pickle", "rb") as file:
            chi_K_v_5 = pickle.load(file)
        with open(files+"chi_rho_v_5.pickle", "rb") as file:
            chi_rho_v_5 = pickle.load(file)
        with open(files+"chi_omega_v_5.pickle", "rb") as file:
            chi_omega_v_5 = pickle.load(file)
        with open(files+"chi_D_v_5.pickle", "rb") as file:
            chi_D_v_5 = pickle.load(file)
        with open(files+"chi_N_v_5.pickle", "rb") as file:
            chi_N_v_5 = pickle.load(file)
        with open(files+"chi_T_v_5.pickle", "rb") as file:
            chi_T_v_5 = pickle.load(file)
        with open(files+"chi_F_v_5.pickle", "rb") as file:
            chi_F_v_5 = pickle.load(file)
        with open(files+"chi_P_v_5.pickle", "rb") as file:
            chi_P_v_5 = pickle.load(file)
        with open(files+"chi_Q_v_5.pickle", "rb") as file:
            chi_Q_v_5 = pickle.load(file)
        with open(files+"chi_H_v_5.pickle", "rb") as file:
            chi_H_v_5 = pickle.load(file)

    if calc_5:

        print("Sigma bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols= 100
        ):
            bden_sigma_v_5.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(T_el, mu_el)
            )
        with open(files+"bden_sigma_v_5.pickle", "wb") as file:
            pickle.dump(bden_sigma_v_5, file)

        print("Gluon bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols= 100
        ):
            bden_gluon_v_5.append(
                pnjl.thermo.gcp_pl_polynomial.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"bden_gluon_v_5.pickle", "wb") as file:
            pickle.dump(bden_gluon_v_5, file)

        print("Sea bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            bden_sea_u_v_5.append(lq_temp)
            bden_sea_d_v_5.append(lq_temp)
            bden_sea_s_v_5.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )
            )
        with open(files+"bden_sea_u_v_5.pickle", "wb") as file:
            pickle.dump(bden_sea_u_v_5, file)
        with open(files+"bden_sea_d_v_5.pickle", "wb") as file:
            pickle.dump(bden_sea_d_v_5, file)
        with open(files+"bden_sea_s_v_5.pickle", "wb") as file:
            pickle.dump(bden_sea_s_v_5, file)

        print("Perturbative bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            bden_perturbative_u_v_5.append(lq_temp)
            bden_perturbative_d_v_5.append(lq_temp)
            bden_perturbative_s_v_5.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon bdensity not implemented")
            else:
                bden_perturbative_gluon_v_5.append(0.0)
        with open(files+"bden_perturbative_u_v_5.pickle", "wb") as file:
            pickle.dump(bden_perturbative_u_v_5, file)
        with open(files+"bden_perturbative_d_v_5.pickle", "wb") as file:
            pickle.dump(bden_perturbative_d_v_5, file)
        with open(files+"bden_perturbative_s_v_5.pickle", "wb") as file:
            pickle.dump(bden_perturbative_s_v_5, file)
        with open(files+"bden_perturbative_gluon_v_5.pickle", "wb") as file:
            pickle.dump(bden_perturbative_gluon_v_5, file)

        print("PNJL bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            bden_pnjl_u_v_5.append(lq_temp)
            bden_pnjl_d_v_5.append(lq_temp)
            bden_pnjl_s_v_5.append(sq_temp)
        with open(files+"bden_pnjl_u_v_5.pickle", "wb") as file:
            pickle.dump(bden_pnjl_u_v_5, file)
        with open(files+"bden_pnjl_d_v_5.pickle", "wb") as file:
            pickle.dump(bden_pnjl_d_v_5, file)
        with open(files+"bden_pnjl_s_v_5.pickle", "wb") as file:
            pickle.dump(bden_pnjl_s_v_5, file)
    else:
        with open(files+"bden_sigma_v_5.pickle", "rb") as file:
            bden_sigma_v_5 = pickle.load(file)
        with open(files+"bden_gluon_v_5.pickle", "rb") as file:
            bden_gluon_v_5 = pickle.load(file)
        with open(files+"bden_sea_u_v_5.pickle", "rb") as file:
            bden_sea_u_v_5 = pickle.load(file)
        with open(files+"bden_sea_d_v_5.pickle", "rb") as file:
            bden_sea_d_v_5 = pickle.load(file)
        with open(files+"bden_sea_s_v_5.pickle", "rb") as file:
            bden_sea_s_v_5 = pickle.load(file)
        with open(files+"bden_perturbative_u_v_5.pickle", "rb") as file:
            bden_perturbative_u_v_5 = pickle.load(file)
        with open(files+"bden_perturbative_d_v_5.pickle", "rb") as file:
            bden_perturbative_d_v_5 = pickle.load(file)
        with open(files+"bden_perturbative_s_v_5.pickle", "rb") as file:
            bden_perturbative_s_v_5 = pickle.load(file)
        with open(files+"bden_perturbative_gluon_v_5.pickle", "rb") as file:
            bden_perturbative_gluon_v_5 = pickle.load(file)
        with open(files+"bden_pnjl_u_v_5.pickle", "rb") as file:
            bden_pnjl_u_v_5 = pickle.load(file)
        with open(files+"bden_pnjl_d_v_5.pickle", "rb") as file:
            bden_pnjl_d_v_5 = pickle.load(file)
        with open(files+"bden_pnjl_s_v_5.pickle", "rb") as file:
            bden_pnjl_s_v_5 = pickle.load(file)

    if calc_5:

        print("Pion bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_pi_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"bden_pi_v_5.pickle", "wb") as file:
            pickle.dump(bden_pi_v_5, file)

        print("Kaon bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_K_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"bden_K_v_5.pickle", "wb") as file:
            pickle.dump(bden_K_v_5, file)

        print("Rho bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_rho_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"bden_rho_v_5.pickle", "wb") as file:
            pickle.dump(bden_rho_v_5, file)

        print("Omega bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_omega_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"bden_omega_v_5.pickle", "wb") as file:
            pickle.dump(bden_omega_v_5, file)

        print("Diquark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_D_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"bden_D_v_5.pickle", "wb") as file:
            pickle.dump(bden_D_v_5, file)

        print("Nucleon bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_N_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"bden_N_v_5.pickle", "wb") as file:
            pickle.dump(bden_N_v_5, file)

        print("Tetraquark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_T_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"bden_T_v_5.pickle", "wb") as file:
            pickle.dump(bden_T_v_5, file)

        print("F-quark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_F_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"bden_F_v_5.pickle", "wb") as file:
            pickle.dump(bden_F_v_5, file)

        print("Pentaquark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_P_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"bden_P_v_5.pickle", "wb") as file:
            pickle.dump(bden_P_v_5, file)

        print("Q-quark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_Q_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"bden_Q_v_5.pickle", "wb") as file:
            pickle.dump(bden_Q_v_5, file)

        print("Hexaquark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_5, mu_5, phi_re_v_5, phi_im_v_5), total=len(T_5), ncols=100
        ):
            bden_H_v_5.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"bden_H_v_5.pickle", "wb") as file:
            pickle.dump(bden_H_v_5, file)
    else:
        with open(files+"bden_pi_v_5.pickle", "rb") as file:
            bden_pi_v_5 = pickle.load(file)
        with open(files+"bden_K_v_5.pickle", "rb") as file:
            bden_K_v_5 = pickle.load(file)
        with open(files+"bden_rho_v_5.pickle", "rb") as file:
            bden_rho_v_5 = pickle.load(file)
        with open(files+"bden_omega_v_5.pickle", "rb") as file:
            bden_omega_v_5 = pickle.load(file)
        with open(files+"bden_D_v_5.pickle", "rb") as file:
            bden_D_v_5 = pickle.load(file)
        with open(files+"bden_N_v_5.pickle", "rb") as file:
            bden_N_v_5 = pickle.load(file)
        with open(files+"bden_T_v_5.pickle", "rb") as file:
            bden_T_v_5 = pickle.load(file)
        with open(files+"bden_F_v_5.pickle", "rb") as file:
            bden_F_v_5 = pickle.load(file)
        with open(files+"bden_P_v_5.pickle", "rb") as file:
            bden_P_v_5 = pickle.load(file)
        with open(files+"bden_Q_v_5.pickle", "rb") as file:
            bden_Q_v_5 = pickle.load(file)
        with open(files+"bden_H_v_5.pickle", "rb") as file:
            bden_H_v_5 = pickle.load(file)
 
    if calc_6:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #5")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_6, mu_6), total=len(T_6), ncols= 100
        ):
            phi_result = solver_1.Polyakov_loop(T_el, mu_el, phi_re_0, phi_im_0)
            phi_re_v_6.append(phi_result[0])
            phi_im_v_6.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_6.pickle", "wb") as file:
            pickle.dump(phi_re_v_6, file)
        with open(files+"phi_im_v_6.pickle", "wb") as file:
            pickle.dump(phi_im_v_6, file)

        print("Sigma chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols= 100
        ):
            chi_sigma_v_6.append(
                pnjl.thermo.gcp_sigma_lattice.qnumber_cumulant(2, T_el, mu_el)
            )
        with open(files+"chi_sigma_v_6.pickle", "wb") as file:
            pickle.dump(chi_sigma_v_6, file)

        print("Gluon chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6),
            total=len(T_6), ncols= 100
        ):
            chi_gluon_v_6.append(
                pnjl.thermo.gcp_pl_polynomial.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"chi_gluon_v_6.pickle", "wb") as file:
            pickle.dump(chi_gluon_v_6, file)

        print("Sea chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6),
            total=len(T_6), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                2, T_el, mu_el, 'l'
            )
            chi_sea_u_v_6.append(lq_temp)
            chi_sea_d_v_6.append(lq_temp)
            chi_sea_s_v_6.append(
                pnjl.thermo.gcp_sea_lattice.qnumber_cumulant(
                    2, T_el, mu_el, 's'
                )
            )
        with open(files+"chi_sea_u_v_6.pickle", "wb") as file:
            pickle.dump(chi_sea_u_v_6, file)
        with open(files+"chi_sea_d_v_6.pickle", "wb") as file:
            pickle.dump(chi_sea_d_v_6, file)
        with open(files+"chi_sea_s_v_6.pickle", "wb") as file:
            pickle.dump(chi_sea_s_v_6, file)

        print("Perturbative chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6),
            total=len(T_6), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            chi_perturbative_u_v_6.append(lq_temp)
            chi_perturbative_d_v_6.append(lq_temp)
            chi_perturbative_s_v_6.append(
                pnjl.thermo.gcp_perturbative.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon chi_q not implemented")
            else:
                chi_perturbative_gluon_v_6.append(0.0)
        with open(files+"chi_perturbative_u_v_6.pickle", "wb") as file:
            pickle.dump(chi_perturbative_u_v_6, file)
        with open(files+"chi_perturbative_d_v_6.pickle", "wb") as file:
            pickle.dump(chi_perturbative_d_v_6, file)
        with open(files+"chi_perturbative_s_v_6.pickle", "wb") as file:
            pickle.dump(chi_perturbative_s_v_6, file)
        with open(files+"chi_perturbative_gluon_v_6.pickle", "wb") as file:
            pickle.dump(chi_perturbative_gluon_v_6, file)

        print("PNJL chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6),
            total=len(T_6), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.qnumber_cumulant(
                2, T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            chi_pnjl_u_v_6.append(lq_temp)
            chi_pnjl_d_v_6.append(lq_temp)
            chi_pnjl_s_v_6.append(sq_temp)
        with open(files+"chi_pnjl_u_v_6.pickle", "wb") as file:
            pickle.dump(chi_pnjl_u_v_6, file)
        with open(files+"chi_pnjl_d_v_6.pickle", "wb") as file:
            pickle.dump(chi_pnjl_d_v_6, file)
        with open(files+"chi_pnjl_s_v_6.pickle", "wb") as file:
            pickle.dump(chi_pnjl_s_v_6, file)
    else:
        with open(files+"phi_re_v_6.pickle", "rb") as file:
            phi_re_v_6 = pickle.load(file)
        with open(files+"phi_im_v_6.pickle", "rb") as file:
            phi_im_v_6 = pickle.load(file)
        with open(files+"chi_sigma_v_6.pickle", "rb") as file:
            chi_sigma_v_6 = pickle.load(file)
        with open(files+"chi_gluon_v_6.pickle", "rb") as file:
            chi_gluon_v_6 = pickle.load(file)
        with open(files+"chi_sea_u_v_6.pickle", "rb") as file:
            chi_sea_u_v_6 = pickle.load(file)
        with open(files+"chi_sea_d_v_6.pickle", "rb") as file:
            chi_sea_d_v_6 = pickle.load(file)
        with open(files+"chi_sea_s_v_6.pickle", "rb") as file:
            chi_sea_s_v_6 = pickle.load(file)
        with open(files+"chi_perturbative_u_v_6.pickle", "rb") as file:
            chi_perturbative_u_v_6 = pickle.load(file)
        with open(files+"chi_perturbative_d_v_6.pickle", "rb") as file:
            chi_perturbative_d_v_6 = pickle.load(file)
        with open(files+"chi_perturbative_s_v_6.pickle", "rb") as file:
            chi_perturbative_s_v_6 = pickle.load(file)
        with open(files+"chi_perturbative_gluon_v_6.pickle", "rb") as file:
            chi_perturbative_gluon_v_6 = pickle.load(file)
        with open(files+"chi_pnjl_u_v_6.pickle", "rb") as file:
            chi_pnjl_u_v_6 = pickle.load(file)
        with open(files+"chi_pnjl_d_v_6.pickle", "rb") as file:
            chi_pnjl_d_v_6 = pickle.load(file)
        with open(files+"chi_pnjl_s_v_6.pickle", "rb") as file:
            chi_pnjl_s_v_6 = pickle.load(file)

    if calc_6:

        print("Pion chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_pi_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"chi_pi_v_6.pickle", "wb") as file:
            pickle.dump(chi_pi_v_6, file)

        print("Kaon chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_K_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"chi_K_v_6.pickle", "wb") as file:
            pickle.dump(chi_K_v_6, file)

        print("Rho chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_rho_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"chi_rho_v_6.pickle", "wb") as file:
            pickle.dump(chi_rho_v_6, file)

        print("Omega chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_omega_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"chi_omega_v_6.pickle", "wb") as file:
            pickle.dump(chi_omega_v_6, file)

        print("Diquark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_D_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"chi_D_v_6.pickle", "wb") as file:
            pickle.dump(chi_D_v_6, file)

        print("Nucleon chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_N_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"chi_N_v_6.pickle", "wb") as file:
            pickle.dump(chi_N_v_6, file)

        print("Tetraquark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_T_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"chi_T_v_6.pickle", "wb") as file:
            pickle.dump(chi_T_v_6, file)

        print("F-quark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_F_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"chi_F_v_6.pickle", "wb") as file:
            pickle.dump(chi_F_v_6, file)

        print("Pentaquark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_P_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"chi_P_v_6.pickle", "wb") as file:
            pickle.dump(chi_P_v_6, file)

        print("Q-quark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_Q_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"chi_Q_v_6.pickle", "wb") as file:
            pickle.dump(chi_Q_v_6, file)

        print("Hexaquark chi_q #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            chi_H_v_6.append(
                cluster_h.qnumber_cumulant(
                    2, T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"chi_H_v_6.pickle", "wb") as file:
            pickle.dump(chi_H_v_6, file)
    else:
        with open(files+"chi_pi_v_6.pickle", "rb") as file:
            chi_pi_v_6 = pickle.load(file)
        with open(files+"chi_K_v_6.pickle", "rb") as file:
            chi_K_v_6 = pickle.load(file)
        with open(files+"chi_rho_v_6.pickle", "rb") as file:
            chi_rho_v_6 = pickle.load(file)
        with open(files+"chi_omega_v_6.pickle", "rb") as file:
            chi_omega_v_6 = pickle.load(file)
        with open(files+"chi_D_v_6.pickle", "rb") as file:
            chi_D_v_6 = pickle.load(file)
        with open(files+"chi_N_v_6.pickle", "rb") as file:
            chi_N_v_6 = pickle.load(file)
        with open(files+"chi_T_v_6.pickle", "rb") as file:
            chi_T_v_6 = pickle.load(file)
        with open(files+"chi_F_v_6.pickle", "rb") as file:
            chi_F_v_6 = pickle.load(file)
        with open(files+"chi_P_v_6.pickle", "rb") as file:
            chi_P_v_6 = pickle.load(file)
        with open(files+"chi_Q_v_6.pickle", "rb") as file:
            chi_Q_v_6 = pickle.load(file)
        with open(files+"chi_H_v_6.pickle", "rb") as file:
            chi_H_v_6 = pickle.load(file)

    if calc_6:

        print("Sigma bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols= 100
        ):
            bden_sigma_v_6.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(T_el, mu_el)
            )
        with open(files+"bden_sigma_v_6.pickle", "wb") as file:
            pickle.dump(bden_sigma_v_6, file)

        print("Gluon bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols= 100
        ):
            bden_gluon_v_6.append(
                pnjl.thermo.gcp_pl_polynomial.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )
            )
        with open(files+"bden_gluon_v_6.pickle", "wb") as file:
            pickle.dump(bden_gluon_v_6, file)

        print("Sea bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            bden_sea_u_v_6.append(lq_temp)
            bden_sea_d_v_6.append(lq_temp)
            bden_sea_s_v_6.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )
            )
        with open(files+"bden_sea_u_v_6.pickle", "wb") as file:
            pickle.dump(bden_sea_u_v_6, file)
        with open(files+"bden_sea_d_v_6.pickle", "wb") as file:
            pickle.dump(bden_sea_d_v_6, file)
        with open(files+"bden_sea_s_v_6.pickle", "wb") as file:
            pickle.dump(bden_sea_s_v_6, file)

        print("Perturbative bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            bden_perturbative_u_v_6.append(lq_temp)
            bden_perturbative_d_v_6.append(lq_temp)
            bden_perturbative_s_v_6.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 's'
                )
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Gluon bdensity not implemented")
            else:
                bden_perturbative_gluon_v_6.append(0.0)
        with open(files+"bden_perturbative_u_v_6.pickle", "wb") as file:
            pickle.dump(bden_perturbative_u_v_6, file)
        with open(files+"bden_perturbative_d_v_6.pickle", "wb") as file:
            pickle.dump(bden_perturbative_d_v_6, file)
        with open(files+"bden_perturbative_s_v_6.pickle", "wb") as file:
            pickle.dump(bden_perturbative_s_v_6, file)
        with open(files+"bden_perturbative_gluon_v_6.pickle", "wb") as file:
            pickle.dump(bden_perturbative_gluon_v_6, file)

        print("PNJL bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols= 100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            bden_pnjl_u_v_6.append(lq_temp)
            bden_pnjl_d_v_6.append(lq_temp)
            bden_pnjl_s_v_6.append(sq_temp)
        with open(files+"bden_pnjl_u_v_6.pickle", "wb") as file:
            pickle.dump(bden_pnjl_u_v_6, file)
        with open(files+"bden_pnjl_d_v_6.pickle", "wb") as file:
            pickle.dump(bden_pnjl_d_v_6, file)
        with open(files+"bden_pnjl_s_v_6.pickle", "wb") as file:
            pickle.dump(bden_pnjl_s_v_6, file)
    else:
        with open(files+"bden_sigma_v_6.pickle", "rb") as file:
            bden_sigma_v_6 = pickle.load(file)
        with open(files+"bden_gluon_v_6.pickle", "rb") as file:
            bden_gluon_v_6 = pickle.load(file)
        with open(files+"bden_sea_u_v_6.pickle", "rb") as file:
            bden_sea_u_v_6 = pickle.load(file)
        with open(files+"bden_sea_d_v_6.pickle", "rb") as file:
            bden_sea_d_v_6 = pickle.load(file)
        with open(files+"bden_sea_s_v_6.pickle", "rb") as file:
            bden_sea_s_v_6 = pickle.load(file)
        with open(files+"bden_perturbative_u_v_6.pickle", "rb") as file:
            bden_perturbative_u_v_6 = pickle.load(file)
        with open(files+"bden_perturbative_d_v_6.pickle", "rb") as file:
            bden_perturbative_d_v_6 = pickle.load(file)
        with open(files+"bden_perturbative_s_v_6.pickle", "rb") as file:
            bden_perturbative_s_v_6 = pickle.load(file)
        with open(files+"bden_perturbative_gluon_v_6.pickle", "rb") as file:
            bden_perturbative_gluon_v_6 = pickle.load(file)
        with open(files+"bden_pnjl_u_v_6.pickle", "rb") as file:
            bden_pnjl_u_v_6 = pickle.load(file)
        with open(files+"bden_pnjl_d_v_6.pickle", "rb") as file:
            bden_pnjl_d_v_6 = pickle.load(file)
        with open(files+"bden_pnjl_s_v_6.pickle", "rb") as file:
            bden_pnjl_s_v_6 = pickle.load(file)

    if calc_6:

        print("Pion bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_pi_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'pi'
                )
            )
        with open(files+"bden_pi_v_6.pickle", "wb") as file:
            pickle.dump(bden_pi_v_6, file)

        print("Kaon bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_K_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'K'
                )
            )
        with open(files+"bden_K_v_6.pickle", "wb") as file:
            pickle.dump(bden_K_v_6, file)

        print("Rho bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_rho_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'rho'
                )
            )
        with open(files+"bden_rho_v_6.pickle", "wb") as file:
            pickle.dump(bden_rho_v_6, file)

        print("Omega bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_omega_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'omega'
                )
            )
        with open(files+"bden_omega_v_6.pickle", "wb") as file:
            pickle.dump(bden_omega_v_6, file)

        print("Diquark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_D_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'D'
                )
            )
        with open(files+"bden_D_v_6.pickle", "wb") as file:
            pickle.dump(bden_D_v_6, file)

        print("Nucleon bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_N_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'N'
                )
            )
        with open(files+"bden_N_v_6.pickle", "wb") as file:
            pickle.dump(bden_N_v_6, file)

        print("Tetraquark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_T_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'T'
                )
            )
        with open(files+"bden_T_v_6.pickle", "wb") as file:
            pickle.dump(bden_T_v_6, file)

        print("F-quark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_F_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'F'
                )
            )
        with open(files+"bden_F_v_6.pickle", "wb") as file:
            pickle.dump(bden_F_v_6, file)

        print("Pentaquark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_P_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'P'
                )
            )
        with open(files+"bden_P_v_6.pickle", "wb") as file:
            pickle.dump(bden_P_v_6, file)

        print("Q-quark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_Q_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'Q'
                )
            )
        with open(files+"bden_Q_v_6.pickle", "wb") as file:
            pickle.dump(bden_Q_v_6, file)

        print("Hexaquark bdensity #5")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_6, mu_6, phi_re_v_6, phi_im_v_6), total=len(T_6), ncols=100
        ):
            bden_H_v_6.append(
                cluster_h.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el,
                    solver_1.Polyakov_loop, 'H'
                )
            )
        with open(files+"bden_H_v_6.pickle", "wb") as file:
            pickle.dump(bden_H_v_6, file)
    else:
        with open(files+"bden_pi_v_6.pickle", "rb") as file:
            bden_pi_v_6 = pickle.load(file)
        with open(files+"bden_K_v_6.pickle", "rb") as file:
            bden_K_v_6 = pickle.load(file)
        with open(files+"bden_rho_v_6.pickle", "rb") as file:
            bden_rho_v_6 = pickle.load(file)
        with open(files+"bden_omega_v_6.pickle", "rb") as file:
            bden_omega_v_6 = pickle.load(file)
        with open(files+"bden_D_v_6.pickle", "rb") as file:
            bden_D_v_6 = pickle.load(file)
        with open(files+"bden_N_v_6.pickle", "rb") as file:
            bden_N_v_6 = pickle.load(file)
        with open(files+"bden_T_v_6.pickle", "rb") as file:
            bden_T_v_6 = pickle.load(file)
        with open(files+"bden_F_v_6.pickle", "rb") as file:
            bden_F_v_6 = pickle.load(file)
        with open(files+"bden_P_v_6.pickle", "rb") as file:
            bden_P_v_6 = pickle.load(file)
        with open(files+"bden_Q_v_6.pickle", "rb") as file:
            bden_Q_v_6 = pickle.load(file)
        with open(files+"bden_H_v_6.pickle", "rb") as file:
            bden_H_v_6 = pickle.load(file)

    bden_qgp_total_1 = [
        sum(el) for el in zip(
            bden_sigma_v_1, bden_gluon_v_1, bden_perturbative_gluon_v_1,
            bden_sea_u_v_1, bden_sea_d_v_1, bden_sea_s_v_1,
            bden_perturbative_u_v_1, bden_perturbative_d_v_1,
            bden_perturbative_s_v_1, bden_pnjl_u_v_1, bden_pnjl_d_v_1,
            bden_pnjl_s_v_1
        )
    ]

    chi_qgp_total_1 = [
        sum(el) for el in zip(
            chi_sigma_v_1, chi_gluon_v_1, chi_perturbative_gluon_v_1,
            chi_sea_u_v_1, chi_sea_d_v_1, chi_sea_s_v_1,
            chi_perturbative_u_v_1, chi_perturbative_d_v_1,
            chi_perturbative_s_v_1, chi_pnjl_u_v_1, chi_pnjl_d_v_1,
            chi_pnjl_s_v_1
        )
    ]

    bden_cluster_total_1 = [
        sum(el) for el in zip(
            bden_pi_v_1, bden_K_v_1, bden_rho_v_1, bden_omega_v_1,
            bden_D_v_1, bden_N_v_1, bden_T_v_1, bden_F_v_1, bden_P_v_1,
            bden_Q_v_1, bden_H_v_1
        )
    ]

    chi_cluster_total_1 = [
        sum(el) for el in zip(
            chi_pi_v_1, chi_K_v_1, chi_rho_v_1, chi_omega_v_1,
            chi_D_v_1, chi_N_v_1, chi_T_v_1, chi_F_v_1, chi_P_v_1,
            chi_Q_v_1, chi_H_v_1
        )
    ]

    bden_total_1 = [sum(el) for el in zip(bden_qgp_total_1, bden_cluster_total_1)]
    
    chi_total_1 = [sum(el) for el in zip(chi_qgp_total_1, chi_cluster_total_1)]

    R12_calc_1 = [3.0*b_el/(mu_el*chi_el) for b_el, mu_el, chi_el in zip(bden_total_1, mu_1, chi_total_1)]

    bden_qgp_total_2 = [
        sum(el) for el in zip(
            bden_sigma_v_2, bden_gluon_v_2, bden_perturbative_gluon_v_2,
            bden_sea_u_v_2, bden_sea_d_v_2, bden_sea_s_v_2,
            bden_perturbative_u_v_2, bden_perturbative_d_v_2,
            bden_perturbative_s_v_2, bden_pnjl_u_v_2, bden_pnjl_d_v_2,
            bden_pnjl_s_v_2
        )
    ]

    chi_qgp_total_2 = [
        sum(el) for el in zip(
            chi_sigma_v_2, chi_gluon_v_2, chi_perturbative_gluon_v_2,
            chi_sea_u_v_2, chi_sea_d_v_2, chi_sea_s_v_2,
            chi_perturbative_u_v_2, chi_perturbative_d_v_2,
            chi_perturbative_s_v_2, chi_pnjl_u_v_2, chi_pnjl_d_v_2,
            chi_pnjl_s_v_2
        )
    ]

    bden_cluster_total_2 = [
        sum(el) for el in zip(
            bden_pi_v_2, bden_K_v_2, bden_rho_v_2, bden_omega_v_2,
            bden_D_v_2, bden_N_v_2, bden_T_v_2, bden_F_v_2, bden_P_v_2,
            bden_Q_v_2, bden_H_v_2
        )
    ]

    chi_cluster_total_2 = [
        sum(el) for el in zip(
            chi_pi_v_2, chi_K_v_2, chi_rho_v_2, chi_omega_v_2,
            chi_D_v_2, chi_N_v_2, chi_T_v_2, chi_F_v_2, chi_P_v_2,
            chi_Q_v_2, chi_H_v_2
        )
    ]

    bden_total_2 = [sum(el) for el in zip(bden_qgp_total_2, bden_cluster_total_2)]
    
    chi_total_2 = [sum(el) for el in zip(chi_qgp_total_2, chi_cluster_total_2)]

    R12_calc_2 = [3.0*b_el/(mu_el*chi_el) for b_el, mu_el, chi_el in zip(bden_total_2, mu_2, chi_total_2)]

    bden_qgp_total_3 = [
        sum(el) for el in zip(
            bden_sigma_v_3, bden_gluon_v_3, bden_perturbative_gluon_v_3,
            bden_sea_u_v_3, bden_sea_d_v_3, bden_sea_s_v_3,
            bden_perturbative_u_v_3, bden_perturbative_d_v_3,
            bden_perturbative_s_v_3, bden_pnjl_u_v_3, bden_pnjl_d_v_3,
            bden_pnjl_s_v_3
        )
    ]

    chi_qgp_total_3 = [
        sum(el) for el in zip(
            chi_sigma_v_3, chi_gluon_v_3, chi_perturbative_gluon_v_3,
            chi_sea_u_v_3, chi_sea_d_v_3, chi_sea_s_v_3,
            chi_perturbative_u_v_3, chi_perturbative_d_v_3,
            chi_perturbative_s_v_3, chi_pnjl_u_v_3, chi_pnjl_d_v_3,
            chi_pnjl_s_v_3
        )
    ]

    bden_cluster_total_3 = [
        sum(el) for el in zip(
            bden_pi_v_3, bden_K_v_3, bden_rho_v_3, bden_omega_v_3,
            bden_D_v_3, bden_N_v_3, bden_T_v_3, bden_F_v_3, bden_P_v_3,
            bden_Q_v_3, bden_H_v_3
        )
    ]

    chi_cluster_total_3 = [
        sum(el) for el in zip(
            chi_pi_v_3, chi_K_v_3, chi_rho_v_3, chi_omega_v_3,
            chi_D_v_3, chi_N_v_3, chi_T_v_3, chi_F_v_3, chi_P_v_3,
            chi_Q_v_3, chi_H_v_3
        )
    ]

    bden_total_3 = [sum(el) for el in zip(bden_qgp_total_3, bden_cluster_total_3)]
    
    chi_total_3 = [sum(el) for el in zip(chi_qgp_total_3, chi_cluster_total_3)]

    R12_calc_3 = [3.0*b_el/(mu_el*chi_el) for b_el, mu_el, chi_el in zip(bden_total_3, mu_3, chi_total_3)]

    bden_qgp_total_4 = [
        sum(el) for el in zip(
            bden_sigma_v_4, bden_gluon_v_4, bden_perturbative_gluon_v_4,
            bden_sea_u_v_4, bden_sea_d_v_4, bden_sea_s_v_4,
            bden_perturbative_u_v_4, bden_perturbative_d_v_4,
            bden_perturbative_s_v_4, bden_pnjl_u_v_4, bden_pnjl_d_v_4,
            bden_pnjl_s_v_4
        )
    ]

    chi_qgp_total_4 = [
        sum(el) for el in zip(
            chi_sigma_v_4, chi_gluon_v_4, chi_perturbative_gluon_v_4,
            chi_sea_u_v_4, chi_sea_d_v_4, chi_sea_s_v_4,
            chi_perturbative_u_v_4, chi_perturbative_d_v_4,
            chi_perturbative_s_v_4, chi_pnjl_u_v_4, chi_pnjl_d_v_4,
            chi_pnjl_s_v_4
        )
    ]

    bden_cluster_total_4 = [
        sum(el) for el in zip(
            bden_pi_v_4, bden_K_v_4, bden_rho_v_4, bden_omega_v_4,
            bden_D_v_4, bden_N_v_4, bden_T_v_4, bden_F_v_4, bden_P_v_4,
            bden_Q_v_4, bden_H_v_4
        )
    ]

    chi_cluster_total_4 = [
        sum(el) for el in zip(
            chi_pi_v_4, chi_K_v_4, chi_rho_v_4, chi_omega_v_4,
            chi_D_v_4, chi_N_v_4, chi_T_v_4, chi_F_v_4, chi_P_v_4,
            chi_Q_v_4, chi_H_v_4
        )
    ]

    bden_total_4 = [sum(el) for el in zip(bden_qgp_total_4, bden_cluster_total_4)]
    
    chi_total_4 = [sum(el) for el in zip(chi_qgp_total_4, chi_cluster_total_4)]

    R12_calc_4 = [3.0*b_el/(mu_el*chi_el) for b_el, mu_el, chi_el in zip(bden_total_4, mu_4, chi_total_4)]

    bden_qgp_total_5 = [
        sum(el) for el in zip(
            bden_sigma_v_5, bden_gluon_v_5, bden_perturbative_gluon_v_5,
            bden_sea_u_v_5, bden_sea_d_v_5, bden_sea_s_v_5,
            bden_perturbative_u_v_5, bden_perturbative_d_v_5,
            bden_perturbative_s_v_5, bden_pnjl_u_v_5, bden_pnjl_d_v_5,
            bden_pnjl_s_v_5
        )
    ]

    chi_qgp_total_5 = [
        sum(el) for el in zip(
            chi_sigma_v_5, chi_gluon_v_5, chi_perturbative_gluon_v_5,
            chi_sea_u_v_5, chi_sea_d_v_5, chi_sea_s_v_5,
            chi_perturbative_u_v_5, chi_perturbative_d_v_5,
            chi_perturbative_s_v_5, chi_pnjl_u_v_5, chi_pnjl_d_v_5,
            chi_pnjl_s_v_5
        )
    ]

    bden_pi_v_5 = [0.0 for _ in bden_pi_v_5]
    bden_K_v_5 = [0.0 for _ in bden_K_v_5]
    bden_rho_v_5 = [0.0 for _ in bden_rho_v_5]
    bden_omega_v_5 = [0.0 for _ in bden_omega_v_5]
    bden_D_v_5 = [0.0 for _ in bden_D_v_5]
    bden_N_v_5 = [0.0 for _ in bden_N_v_5]
    bden_T_v_5 = [0.0 for _ in bden_T_v_5]
    bden_F_v_5 = [0.0 for _ in bden_F_v_5]
    bden_P_v_5 = [0.0 for _ in bden_P_v_5]
    bden_Q_v_5 = [0.0 for _ in bden_Q_v_5]
    #bden_H_v_5 = [0.0 for _ in bden_H_v_5]
    bden_cluster_total_5 = [
        sum(el) for el in zip(
            bden_pi_v_5, bden_K_v_5, bden_rho_v_5, bden_omega_v_5,
            bden_D_v_5, bden_N_v_5, bden_T_v_5, bden_F_v_5, bden_P_v_5,
            bden_Q_v_5, bden_H_v_5
        )
    ]

    chi_pi_v_5 = [0.0 for _ in chi_pi_v_5]
    chi_K_v_5 = [0.0 for _ in chi_K_v_5]
    chi_rho_v_5 = [0.0 for _ in chi_rho_v_5]
    chi_omega_v_5 = [0.0 for _ in chi_omega_v_5]
    chi_D_v_5 = [0.0 for _ in chi_D_v_5]
    chi_N_v_5 = [0.0 for _ in chi_N_v_5]
    chi_T_v_5 = [0.0 for _ in chi_T_v_5]
    chi_F_v_5 = [0.0 for _ in chi_F_v_5]
    chi_P_v_5 = [0.0 for _ in chi_P_v_5]
    chi_Q_v_5 = [0.0 for _ in chi_Q_v_5]
    #chi_H_v_5 = [0.0 for _ in chi_H_v_5]
    chi_cluster_total_5 = [
        sum(el) for el in zip(
            chi_pi_v_5, chi_K_v_5, chi_rho_v_5, chi_omega_v_5,
            chi_D_v_5, chi_N_v_5, chi_T_v_5, chi_F_v_5, chi_P_v_5,
            chi_Q_v_5, chi_H_v_5
        )
    ]

    bden_total_5 = [sum(el) for el in zip(bden_qgp_total_5, bden_cluster_total_5)]
    
    chi_total_5 = [sum(el) for el in zip(chi_qgp_total_5, chi_cluster_total_5)]

    R12_calc_5 = [3.0*b_el/(mu_el*chi_el) for b_el, mu_el, chi_el in zip(bden_cluster_total_5, mu_5, chi_cluster_total_5)]

    bden_qgp_total_6 = [
        sum(el) for el in zip(
            bden_sigma_v_6, bden_gluon_v_6, bden_perturbative_gluon_v_6,
            bden_sea_u_v_6, bden_sea_d_v_6, bden_sea_s_v_6,
            bden_perturbative_u_v_6, bden_perturbative_d_v_6,
            bden_perturbative_s_v_6, bden_pnjl_u_v_6, bden_pnjl_d_v_6,
            bden_pnjl_s_v_6
        )
    ]

    chi_qgp_total_6 = [
        sum(el) for el in zip(
            chi_sigma_v_6, chi_gluon_v_6, chi_perturbative_gluon_v_6,
            chi_sea_u_v_6, chi_sea_d_v_6, chi_sea_s_v_6,
            chi_perturbative_u_v_6, chi_perturbative_d_v_6,
            chi_perturbative_s_v_6, chi_pnjl_u_v_6, chi_pnjl_d_v_6,
            chi_pnjl_s_v_6
        )
    ]

    bden_pi_v_6 = [0.0 for _ in bden_pi_v_6]
    bden_K_v_6 = [0.0 for _ in bden_K_v_6]
    bden_rho_v_6 = [0.0 for _ in bden_rho_v_6]
    bden_omega_v_6 = [0.0 for _ in bden_omega_v_6]
    bden_D_v_6 = [0.0 for _ in bden_D_v_6]
    bden_N_v_6 = [0.0 for _ in bden_N_v_6]
    bden_T_v_6 = [0.0 for _ in bden_T_v_6]
    bden_F_v_6 = [0.0 for _ in bden_F_v_6]
    bden_P_v_6 = [0.0 for _ in bden_P_v_6]
    bden_Q_v_6 = [0.0 for _ in bden_Q_v_6]
    #bden_H_v_6 = [0.0 for _ in bden_H_v_6]
    bden_cluster_total_6 = [
        sum(el) for el in zip(
            bden_pi_v_6, bden_K_v_6, bden_rho_v_6, bden_omega_v_6,
            bden_D_v_6, bden_N_v_6, bden_T_v_6, bden_F_v_6, bden_P_v_6,
            bden_Q_v_6, bden_H_v_6
        )
    ]

    chi_pi_v_6 = [0.0 for _ in chi_pi_v_6]
    chi_K_v_6 = [0.0 for _ in chi_K_v_6]
    chi_rho_v_6 = [0.0 for _ in chi_rho_v_6]
    chi_omega_v_6 = [0.0 for _ in chi_omega_v_6]
    chi_D_v_6 = [0.0 for _ in chi_D_v_6]
    chi_N_v_6 = [0.0 for _ in chi_N_v_6]
    chi_T_v_6 = [0.0 for _ in chi_T_v_6]
    chi_F_v_6 = [0.0 for _ in chi_F_v_6]
    chi_P_v_6 = [0.0 for _ in chi_P_v_6]
    chi_Q_v_6 = [0.0 for _ in chi_Q_v_6]
    #chi_H_v_6 = [0.0 for _ in chi_H_v_6]
    chi_cluster_total_6 = [
        sum(el) for el in zip(
            chi_pi_v_6, chi_K_v_6, chi_rho_v_6, chi_omega_v_6,
            chi_D_v_6, chi_N_v_6, chi_T_v_6, chi_F_v_6, chi_P_v_6,
            chi_Q_v_6, chi_H_v_6
        )
    ]

    bden_total_6 = [sum(el) for el in zip(bden_qgp_total_6, bden_cluster_total_6)]
    
    chi_total_6 = [sum(el) for el in zip(chi_qgp_total_6, chi_cluster_total_6)]

    R12_calc_6 = [3.0*b_el/(mu_el*chi_el) for b_el, mu_el, chi_el in zip(bden_cluster_total_6, mu_6, chi_cluster_total_6)]
    
    with open("D:/EoS/epja/lattice_data_pickled/2012_12894_R12_0p4.pickle", "rb") as file:
        R12_lQCD_0p4 = pickle.load(file)
    with open("D:/EoS/epja/lattice_data_pickled/2012_12894_R12_0p8.pickle", "rb") as file:
        R12_lQCD_0p8 = pickle.load(file)
    #T0 for lQCD - 170 MeV, https://arxiv.org/pdf/hep-lat/0501030.pdf

    R12_lQCD_0p4 = [[el[0]/170.0, el[1]] for el in R12_lQCD_0p4]
    R12_lQCD_0p8 = [[el[0]/170.0, el[1]] for el in R12_lQCD_0p8]

    T_Tc_1 = [T_el/pnjl.thermo.gcp_sigma_lattice.Tc(mu_el) for T_el, mu_el in zip(T_1, mu_1)]
    T_Tc_2 = [T_el/pnjl.thermo.gcp_sigma_lattice.Tc(mu_el) for T_el, mu_el in zip(T_2, mu_2)]
    T_Tc_3 = [T_el/pnjl.thermo.gcp_sigma_lattice.Tc(mu_el) for T_el, mu_el in zip(T_3, mu_3)]
    T_Tc_4 = [T_el/pnjl.thermo.gcp_sigma_lattice.Tc(mu_el) for T_el, mu_el in zip(T_4, mu_4)]
    T_Tc_5 = [T_el/pnjl.thermo.gcp_sigma_lattice.Tc(mu_el) for T_el, mu_el in zip(T_5, mu_5)]
    T_Tc_6 = [T_el/pnjl.thermo.gcp_sigma_lattice.Tc(mu_el) for T_el, mu_el in zip(T_6, mu_6)]

    T_HRG = numpy.linspace(0.7, 1.0, num=200)
    R12_HRG_1 = [(1.0/1.2)*math.tanh(1.2/1.0) for _ in T_HRG]
    R12_HRG_2 = [(1.0/2.4)*math.tanh(2.4/1.0) for _ in T_HRG]
    T_QGP = numpy.linspace(1.5, 1.8, num=200)
    R12_QGP_1 = [(1.0 + (1.0/math.pi**2)*0.16)/(1.0 + (3.0/math.pi**2)*0.16) for _ in T_QGP]
    R12_QGP_2 = [(1.0 + (1.0/math.pi**2)*0.64)/(1.0 + (3.0/math.pi**2)*0.64) for _ in T_QGP]

    T_lQCD1_158 = [
        158.0/155.517,
        158.0/153.524
    ]
    R12_lQCD1_158_high = [
        0.8732380845556268/1.2,
        0.8732380845556268/1.2
    ]
    R12_lQCD1_158_low = [
        0.8087681705155122/1.2,
        0.8087681705155122/1.2
    ]
    T_lQCD1_150 = [
        150.0/155.517,
        150.0/153.524
    ]
    R12_lQCD1_150_high = [
        0.8314825681903971/1.2,
        0.8314825681903971/1.2
    ]
    R12_lQCD1_150_low = [
        0.782597452237381/1.2,
        0.782597452237381/1.2
    ]
    T_lQCD1_Tc = [
        1.0
    ]
    R12_lQCD1_Tc_high = [
        0.840916537935716/1.2
    ]
    R12_lQCD1_Tc_low = [
        0.8057535597940728/1.2
    ]

    T_lQCD2_152 = [
        152.0/156.18,
        152.0/151.429
    ]
    R12_lQCD2_152_high = [
        0.8785793562708105/1.2,
        0.8785793562708105/1.2
    ]
    R12_lQCD2_152_low = [
        0.8179800221975586/1.2,
        0.8179800221975586/1.2
    ]

    T_lQCD2_155 = [
        155.0/156.18,
        155.0/151.429
    ]
    R12_lQCD2_155_high = [
        0.8859045504994454/1.2,
        0.8859045504994454/1.2
    ]
    R12_lQCD2_155_low = [
        0.8226415094339625/1.2,
        0.8226415094339625/1.2
    ]

    T_lQCD2_158 = [
        158.0/156.18,
        158.0/151.429
    ]
    R12_lQCD2_158_high = [
        0.889234184239734/1.2,
        0.889234184239734/1.2
    ]
    R12_lQCD2_158_low = [
        0.828634850166482/1.2,
        0.828634850166482/1.2
    ]

    T_lQCD2_161 = [
        161.0/156.18,
        161.0/151.429
    ]
    R12_lQCD2_161_high = [
        0.889234184239734/1.2,
        0.889234184239734/1.2
    ]
    R12_lQCD2_161_low = [
        0.8366259711431746/1.2,
        0.8366259711431746/1.2
    ]

    l_2202_09184_T200, l_2202_09184_BQS200, l_2202_09184_BQS200_err = \
        utils.data_load(
            "D:/EoS/lattice_data/datafiles_cumulants_2dII/fig1_exp_coefficients_B2_cont_extr_muQ_muS_zero.txt",
            0, 1, 3,
            firstrow=2, delim=None
        )

    l_2202_09184_T400, l_2202_09184_BQS400, l_2202_09184_BQS400_err = \
        utils.data_load(
            "D:/EoS/lattice_data/datafiles_cumulants_2dII/fig1_exp_coefficients_B4_cont_extr_muQ_muS_zero.txt",
            0, 1, 3,
            firstrow=1, delim=None
        )

    l_2202_09184_T600, l_2202_09184_BQS600, l_2202_09184_BQS600_err = \
        utils.data_load(
            "D:/EoS/lattice_data/datafiles_cumulants_2dII/fig1_exp_coefficients_B6_spline_muQ_muS_zero.txt",
            0, 1, 3,
            firstrow=21, delim=None
        )

    l_2202_09184_T800, l_2202_09184_BQS800, l_2202_09184_BQS800_err = \
        utils.data_load(
            "D:/EoS/lattice_data/datafiles_cumulants_2dII/fig1_exp_coefficients_B8_spline_muQ_muS_zero.txt",
            0, 1, 3,
            firstrow=21, delim=None
        )

    l_2202_09184_chi1_1p2 = [
        B2*1.2 + B4*(1.2**3)/6.0 + B6*(1.2**5)/120.0 + B8*(1.2**7)/5040.0
        for B2, B4, B6, B8 in zip(
            l_2202_09184_BQS200, l_2202_09184_BQS400,
            l_2202_09184_BQS600, l_2202_09184_BQS800
        )
    ]

    l_2202_09184_chi1_err_1p2 = [
        B2*1.2 + B4*(1.2**3)/6.0 + B6*(1.2**5)/120.0 + B8*(1.2**7)/5040.0
        for B2, B4, B6, B8 in zip(
            l_2202_09184_BQS200_err, l_2202_09184_BQS400_err,
            l_2202_09184_BQS600_err, l_2202_09184_BQS800_err
        )
    ]

    l_2202_09184_chi2_1p2 = [
        B2 + B4*(1.2**2)/2.0 + B6*(1.2**4)/24.0 + B8*(1.2**6)/720.0
        for B2, B4, B6, B8 in zip(
            l_2202_09184_BQS200, l_2202_09184_BQS400,
            l_2202_09184_BQS600, l_2202_09184_BQS800
        )
    ]

    l_2202_09184_chi2_err_1p2 = [
        B2 + B4*(1.2**2)/2.0 + B6*(1.2**4)/24.0 + B8*(1.2**6)/720.0
        for B2, B4, B6, B8 in zip(
            l_2202_09184_BQS200_err, l_2202_09184_BQS400_err,
            l_2202_09184_BQS600_err, l_2202_09184_BQS800_err
        )
    ]

    l_2202_09184_R12_1p2 = [
        chi1/chi2/1.2
        for chi1, chi2 in zip(
            l_2202_09184_chi1_1p2, l_2202_09184_chi2_1p2
        )
    ]

    l_2202_09184_R12_err_1p2 = [
        chi1_err/chi2/1.2+(chi1/(chi2**2)/1.2)*chi2_err
        for chi1_err, chi2_err, chi1, chi2 in zip(
            l_2202_09184_chi1_err_1p2, l_2202_09184_chi2_err_1p2,
            l_2202_09184_chi1_1p2, l_2202_09184_chi2_1p2
        )
    ]

    l_2202_09184_R12_high_1p2 = [
        R12+err
        for R12, err in zip(
            l_2202_09184_R12_1p2, l_2202_09184_R12_err_1p2
        )
    ]
    
    l_2202_09184_R12_low_1p2 = [
        R12-err
        for R12, err in zip(
            l_2202_09184_R12_1p2, l_2202_09184_R12_err_1p2
        )
    ]

    l_2202_09184_patch = [
        [x_el, y_el]
        for x_el, y_el in zip(
            [el/156.18 for el in l_2202_09184_T200], l_2202_09184_R12_high_1p2
        )
    ] +\
    [
        [x_el, y_el]
        for x_el, y_el in zip(
            [el/151.429 for el in l_2202_09184_T200], l_2202_09184_R12_low_1p2
        )
    ][::-1]

    fig1 = matplotlib.pyplot.figure(num=1, figsize=(11.0, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.axis([0.7, 1.8, 0.0, 1.2])
    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.axis([0.7, 1.8, 0.0, 1.2])

    ax1.add_patch(
        matplotlib.patches.Polygon(
            R12_lQCD_0p4, closed=True, fill=True, color='red', alpha=0.3
        )
    )
    ax2.add_patch(
        matplotlib.patches.Polygon(
            R12_lQCD_0p8, closed=True, fill=True, color='blue', alpha=0.3
        )
    )

    #ax1.fill_between(T_Tc_1, R12_calc_1, y2=R12_calc_2, color='red', alpha=0.7)
    ax1.plot(T_Tc_1, R12_calc_1, '-.', c='red')
    ax1.plot(T_Tc_2, R12_calc_2, '-', c='red')
    ax1.plot(T_HRG, R12_HRG_1, '--', c='red')
    ax1.plot(T_QGP, R12_QGP_1, '--', c='red')
    #ax1.plot(T_Tc_5, R12_calc_5, '-.', c='red')
    ax1.text(0.705, 0.63, r"HRG", color='black', fontsize=14)
    ax1.text(1.69, 1.0, r"QGP", color='black', fontsize=14)
    ax1.text(1.54, 0.04, r"$\mathrm{\mu_B/T=1.2}$", color='black', fontsize=14)

    ax1.text(
        0.8, 0.49, r'Allton et al. (2005)', color='red', fontsize=14
    )

    #ax1.plot([el/151.429 for el in l_2202_09184_T200], l_2202_09184_R12_low_1p2, '-', c='lightgreen')
    #ax1.plot([el/156.18 for el in l_2202_09184_T200], l_2202_09184_R12_high_1p2, '-', c='green')
    ax1.add_patch(
        matplotlib.patches.Polygon(
            l_2202_09184_patch, closed=True, fill=True, color='green', alpha=0.3
        )
    )

    ax1.text(
        1.08, 0.7, r'Bollweg et al. (2022)', color='green', fontsize=14
    )

    #ax2.fill_between(T_Tc_3, R12_calc_3, y2=R12_calc_4, color='blue', alpha=0.7)
    ax2.plot(T_Tc_3, R12_calc_3, '-.', c='blue')
    ax2.plot(T_Tc_4, R12_calc_4, '-', c='blue')
    ax2.plot(T_HRG, R12_HRG_2, '--', c='blue')
    ax2.plot(T_QGP, R12_QGP_2, '--', c='blue')
    #ax2.plot(T_Tc_6, R12_calc_6, '-.', c='blue')
    ax2.text(0.705, 0.35, r"HRG", color='black', fontsize=14)
    ax2.text(1.69, 0.91, r"QGP", color='black', fontsize=14)
    ax2.text(1.54, 0.04, r"$\mathrm{\mu_B/T=2.4}$", color='black', fontsize=14)

    ax2.text(
        0.86, 0.2, r'Allton et al. (2005)', color='blue', fontsize=14
    )

    #ax1.add_patch(
    #    matplotlib.patches.Polygon(
    #        [
    #            [el1, el2] for el1, el2 in zip(
    #                T_lQCD2_152[0:1]+T_lQCD2_155[0:1]+T_lQCD2_158+T_lQCD2_161 \
    #                +T_lQCD2_161[1:2]+T_lQCD2_158[1:2]+T_lQCD2_155[1:2] \
    #                +T_lQCD2_152[1:2]+T_lQCD2_152[0:1],
    #                R12_lQCD2_152_high[0:1]+R12_lQCD2_155_high[0:1] \
    #                +R12_lQCD2_158_high+R12_lQCD2_161_high \
    #                +R12_lQCD2_161_low[1:2]+R12_lQCD2_158_low[1:2] \
    #                +R12_lQCD2_155_low[1:2]+R12_lQCD2_152_low[1:2] \
    #                +R12_lQCD2_152_low[0:1]
    #            )
    #        ],
    #    closed=True, fill=True, color='cyan', alpha=0.6
    #    )
    #)
    #ax1.fill_between(
    #    T_lQCD1_150+T_lQCD1_Tc+T_lQCD1_158,
    #    R12_lQCD1_150_high+R12_lQCD1_Tc_high+R12_lQCD1_158_high,
    #    y2=R12_lQCD1_150_low+R12_lQCD1_Tc_low+R12_lQCD1_158_low,
    #    color="magenta", alpha=0.3
    #)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'$\mathrm{T/T_c}$', fontsize=16)
    ax1.set_ylabel(r'$\mathrm{R_{12}}$', fontsize=16)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'$\mathrm{T/T_c}$', fontsize=16)
    ax2.set_ylabel(r'$\mathrm{R_{12}}$', fontsize=16)

    fig1.tight_layout()

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure12_s():

    import tqdm
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

    import utils
    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_2

    warnings.filterwarnings("ignore")
    
    calc_1 = True
    calc_2 = True

    files = "D:/EoS/epja/figure12/"

    T_1 = numpy.linspace(1.0, 280.0, 200)
    T_2 = numpy.linspace(1.0, 280.0, 200)

    mu_1 = [0.0 / 3.0 for el in T_1]
    mu_2 = [(2.5 * el) / 3.0 for el in T_2]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()

    sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1 = \
        list(), list(), list(), list()
    gluon_v_1 = list()
    perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
        list(), list(), list()
    perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = \
        list(), list(), list(), list()

    pi_v_1, K_v_1, rho_v_1, D_v_1, N_v_1, F_v_1, P_v_1, Q_v_1, H_v_1 = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_1, T_v_1 = list(), list()

    pi_v_1s, K_v_1s, rho_v_1s, D_v_1s, N_v_1s, F_v_1s, P_v_1s, Q_v_1s, H_v_1s = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_1s, T_v_1s = list(), list()

    phi_re_v_2, phi_im_v_2 = \
        list(), list()

    sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2 = \
        list(), list(), list(), list()
    gluon_v_2 = list()
    perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2 = \
        list(), list(), list()
    perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2 = \
        list(), list(), list(), list()

    pi_v_2, K_v_2, rho_v_2, D_v_2, N_v_2, F_v_2, P_v_2, Q_v_2, H_v_2 = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_2, T_v_2 = list(), list()

    pi_v_2s, K_v_2s, rho_v_2s, D_v_2s, N_v_2s, F_v_2s, P_v_2s, Q_v_2s, H_v_2s = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_2s, T_v_2s = list(), list()

    if calc_1:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #1")
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
        with open(files+"phi_re_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)

        print("Sigma sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.sdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"s_sigma_v_0p0.pickle", "wb") as file:
            pickle.dump(sigma_v_1, file)

        print("Gluon sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            gluon_v_1.append(
                pnjl.thermo.gcp_pl_polynomial.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )/(T_el**3)
            )
        with open(files+"s_gluon_v_0p0.pickle", "wb") as file:
            pickle.dump(gluon_v_1, file)

        print("Sea sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.sdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_1.append(lq_temp/(T_el**3))
            sea_d_v_1.append(lq_temp/(T_el**3))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.sdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"s_sea_u_v_0p0.pickle", "wb") as file:
            pickle.dump(sea_u_v_1, file)
        with open(files+"s_sea_d_v_0p0.pickle", "wb") as file:
            pickle.dump(sea_d_v_1, file)
        with open(files+"s_sea_s_v_0p0.pickle", "wb") as file:
            pickle.dump(sea_s_v_1, file)

        print("Perturbative sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            perturbative_u_v_1.append(lq_temp/(T_el**3))
            perturbative_d_v_1.append(lq_temp/(T_el**3))
            perturbative_s_v_1.append(
                pnjl.thermo.gcp_perturbative.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
            else:
                perturbative_gluon_v_1.append(0.0)
        with open(files+"s_perturbative_u_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_1, file)
        with open(files+"s_perturbative_d_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_1, file)
        with open(files+"s_perturbative_s_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_1, file)
        with open(files+"s_perturbative_gluon_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_1, file)

        print("PNJL sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            pnjl_u_v_1.append(lq_temp/(T_el**3))
            pnjl_d_v_1.append(lq_temp/(T_el**3))
            pnjl_s_v_1.append(sq_temp/(T_el**3))
        with open(files+"s_pnjl_u_v_0p0.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_1, file)
        with open(files+"s_pnjl_d_v_0p0.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_1, file)
        with open(files+"s_pnjl_s_v_0p0.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_1, file)
    else:
        with open(files+"phi_re_v_0p0.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_0p0.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
        with open(files+"s_sigma_v_0p0.pickle", "rb") as file:
            sigma_v_1 = pickle.load(file)
        with open(files+"s_gluon_v_0p0.pickle", "rb") as file:
            gluon_v_1 = pickle.load(file)
        with open(files+"s_sea_u_v_0p0.pickle", "rb") as file:
            sea_u_v_1 = pickle.load(file)
        with open(files+"s_sea_d_v_0p0.pickle", "rb") as file:
            sea_d_v_1 = pickle.load(file)
        with open(files+"s_sea_s_v_0p0.pickle", "rb") as file:
            sea_s_v_1 = pickle.load(file)
        with open(files+"s_perturbative_u_v_0p0.pickle", "rb") as file:
            perturbative_u_v_1 = pickle.load(file)
        with open(files+"s_perturbative_d_v_0p0.pickle", "rb") as file:
            perturbative_d_v_1 = pickle.load(file)
        with open(files+"s_perturbative_s_v_0p0.pickle", "rb") as file:
            perturbative_s_v_1 = pickle.load(file)
        with open(files+"s_perturbative_gluon_v_0p0.pickle", "rb") as file:
            perturbative_gluon_v_1 = pickle.load(file)
        with open(files+"s_pnjl_u_v_0p0.pickle", "rb") as file:
            pnjl_u_v_1 = pickle.load(file)
        with open(files+"s_pnjl_d_v_0p0.pickle", "rb") as file:
            pnjl_d_v_1 = pickle.load(file)
        with open(files+"s_pnjl_s_v_0p0.pickle", "rb") as file:
            pnjl_s_v_1 = pickle.load(file)

    if calc_1:

        print("Pion sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"s_pi_v_0p0.pickle", "wb") as file:
            pickle.dump(pi_v_1, file)

        print("Kaon sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"s_K_v_0p0.pickle", "wb") as file:
            pickle.dump(K_v_1, file)

        print("Rho sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"s_rho_v_0p0.pickle", "wb") as file:
            pickle.dump(rho_v_1, file)

        print("Omega sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"s_omega_v_0p0.pickle", "wb") as file:
            pickle.dump(omega_v_1, file)

        print("Diquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"s_D_v_0p0.pickle", "wb") as file:
            pickle.dump(D_v_1, file)

        print("Nucleon sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"s_N_v_0p0.pickle", "wb") as file:
            pickle.dump(N_v_1, file)

        print("Tetraquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"s_T_v_0p0.pickle", "wb") as file:
            pickle.dump(T_v_1, file)

        print("F-quark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"s_F_v_0p0.pickle", "wb") as file:
            pickle.dump(F_v_1, file)

        print("Pentaquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"s_P_v_0p0.pickle", "wb") as file:
            pickle.dump(P_v_1, file)

        print("Q-quark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"s_Q_v_0p0.pickle", "wb") as file:
            pickle.dump(Q_v_1, file)

        print("Hexaquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"s_H_v_0p0.pickle", "wb") as file:
            pickle.dump(H_v_1, file)
    else:
        with open(files+"s_pi_v_0p0.pickle", "rb") as file:
            pi_v_1 = pickle.load(file)
        with open(files+"s_K_v_0p0.pickle", "rb") as file:
            K_v_1 = pickle.load(file)
        with open(files+"s_rho_v_0p0.pickle", "rb") as file:
            rho_v_1 = pickle.load(file)
        with open(files+"s_omega_v_0p0.pickle", "rb") as file:
            omega_v_1 = pickle.load(file)
        with open(files+"s_D_v_0p0.pickle", "rb") as file:
            D_v_1 = pickle.load(file)
        with open(files+"s_N_v_0p0.pickle", "rb") as file:
            N_v_1 = pickle.load(file)
        with open(files+"s_T_v_0p0.pickle", "rb") as file:
            T_v_1 = pickle.load(file)
        with open(files+"s_F_v_0p0.pickle", "rb") as file:
            F_v_1 = pickle.load(file)
        with open(files+"s_P_v_0p0.pickle", "rb") as file:
            P_v_1 = pickle.load(file)
        with open(files+"s_Q_v_0p0.pickle", "rb") as file:
            Q_v_1 = pickle.load(file)
        with open(files+"s_H_v_0p0.pickle", "rb") as file:
            H_v_1 = pickle.load(file)

    if calc_1:

        print("Pion sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"s_pi_v_0p0s.pickle", "wb") as file:
            pickle.dump(pi_v_1s, file)

        print("Kaon sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"s_K_v_0p0s.pickle", "wb") as file:
            pickle.dump(K_v_1s, file)

        print("Rho sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"s_rho_v_0p0s.pickle", "wb") as file:
            pickle.dump(rho_v_1s, file)

        print("Omega sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"s_omega_v_0p0s.pickle", "wb") as file:
            pickle.dump(omega_v_1s, file)

        print("Diquark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"s_D_v_0p0s.pickle", "wb") as file:
            pickle.dump(D_v_1s, file)

        print("Nucleon sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"s_N_v_0p0s.pickle", "wb") as file:
            pickle.dump(N_v_1s, file)

        print("Tetraquark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"s_T_v_0p0s.pickle", "wb") as file:
            pickle.dump(T_v_1s, file)

        print("F-quark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"s_F_v_0p0s.pickle", "wb") as file:
            pickle.dump(F_v_1s, file)

        print("Pentaquark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"s_P_v_0p0s.pickle", "wb") as file:
            pickle.dump(P_v_1s, file)

        print("Q-quark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"s_Q_v_0p0s.pickle", "wb") as file:
            pickle.dump(Q_v_1s, file)

        print("Hexaquark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"s_H_v_0p0s.pickle", "wb") as file:
            pickle.dump(H_v_1s, file)
    else:
        with open(files+"s_pi_v_0p0s.pickle", "rb") as file:
            pi_v_1s = pickle.load(file)
        with open(files+"s_K_v_0p0s.pickle", "rb") as file:
            K_v_1s = pickle.load(file)
        with open(files+"s_rho_v_0p0s.pickle", "rb") as file:
            rho_v_1s = pickle.load(file)
        with open(files+"s_omega_v_0p0s.pickle", "rb") as file:
            omega_v_1s = pickle.load(file)
        with open(files+"s_D_v_0p0s.pickle", "rb") as file:
            D_v_1s = pickle.load(file)
        with open(files+"s_N_v_0p0s.pickle", "rb") as file:
            N_v_1s = pickle.load(file)
        with open(files+"s_T_v_0p0s.pickle", "rb") as file:
            T_v_1s = pickle.load(file)
        with open(files+"s_F_v_0p0s.pickle", "rb") as file:
            F_v_1s = pickle.load(file)
        with open(files+"s_P_v_0p0s.pickle", "rb") as file:
            P_v_1s = pickle.load(file)
        with open(files+"s_Q_v_0p0s.pickle", "rb") as file:
            Q_v_1s = pickle.load(file)
        with open(files+"s_H_v_0p0s.pickle", "rb") as file:
            H_v_1s = pickle.load(file)

    if calc_2:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #2")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_2, mu_2), total=len(T_2), ncols=100
        ):
            phi_result = solver_2.Polyakov_loop(
                T_el, mu_el, phi_re_0, phi_im_0
            )
            phi_re_v_2.append(phi_result[0])
            phi_im_v_2.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)

        print("Sigma sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            sigma_v_2.append(
                pnjl.thermo.gcp_sigma_lattice.sdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"s_sigma_v_2p5.pickle", "wb") as file:
            pickle.dump(sigma_v_2, file)

        print("Gluon sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            gluon_v_2.append(
                pnjl.thermo.gcp_pl_polynomial.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop
                )/(T_el**3)
            )
        with open(files+"s_gluon_v_2p5.pickle", "wb") as file:
            pickle.dump(gluon_v_2, file)

        print("Sea sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.sdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_2.append(lq_temp/(T_el**3))
            sea_d_v_2.append(lq_temp/(T_el**3))
            sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.sdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"s_sea_u_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_u_v_2, file)
        with open(files+"s_sea_d_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_d_v_2, file)
        with open(files+"s_sea_s_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_s_v_2, file)

        print("Perturbative sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'l'
            )
            perturbative_u_v_2.append(lq_temp/(T_el**3))
            perturbative_d_v_2.append(lq_temp/(T_el**3))
            perturbative_s_v_2.append(
                pnjl.thermo.gcp_perturbative.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
            else:
                perturbative_gluon_v_2.append(0.0)
        with open(files+"s_perturbative_u_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_2, file)
        with open(files+"s_perturbative_d_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_2, file)
        with open(files+"s_perturbative_s_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_2, file)
        with open(files+"s_perturbative_gluon_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_2, file)

        print("PNJL sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 's'
            )
            pnjl_u_v_2.append(lq_temp/(T_el**3))
            pnjl_d_v_2.append(lq_temp/(T_el**3))
            pnjl_s_v_2.append(sq_temp/(T_el**3))
        with open(files+"s_pnjl_u_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_2, file)
        with open(files+"s_pnjl_d_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_2, file)
        with open(files+"s_pnjl_s_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_2, file)
    else:
        with open(files+"phi_re_v_2p5.pickle", "rb") as file:
            phi_re_v_2 = pickle.load(file)
        with open(files+"phi_im_v_2p5.pickle", "rb") as file:
            phi_im_v_2 = pickle.load(file)
        with open(files+"s_sigma_v_2p5.pickle", "rb") as file:
            sigma_v_2 = pickle.load(file)
        with open(files+"s_gluon_v_2p5.pickle", "rb") as file:
            gluon_v_2 = pickle.load(file)
        with open(files+"s_sea_u_v_2p5.pickle", "rb") as file:
            sea_u_v_2 = pickle.load(file)
        with open(files+"s_sea_d_v_2p5.pickle", "rb") as file:
            sea_d_v_2 = pickle.load(file)
        with open(files+"s_sea_s_v_2p5.pickle", "rb") as file:
            sea_s_v_2 = pickle.load(file)
        with open(files+"s_perturbative_u_v_2p5.pickle", "rb") as file:
            perturbative_u_v_2 = pickle.load(file)
        with open(files+"s_perturbative_d_v_2p5.pickle", "rb") as file:
            perturbative_d_v_2 = pickle.load(file)
        with open(files+"s_perturbative_s_v_2p5.pickle", "rb") as file:
            perturbative_s_v_2 = pickle.load(file)
        with open(files+"s_perturbative_gluon_v_2p5.pickle", "rb") as file:
            perturbative_gluon_v_2 = pickle.load(file)
        with open(files+"s_pnjl_u_v_2p5.pickle", "rb") as file:
            pnjl_u_v_2 = pickle.load(file)
        with open(files+"s_pnjl_d_v_2p5.pickle", "rb") as file:
            pnjl_d_v_2 = pickle.load(file)
        with open(files+"s_pnjl_s_v_2p5.pickle", "rb") as file:
            pnjl_s_v_2 = pickle.load(file)

    if calc_2:

        print("Pion sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"s_pi_v_2p5.pickle", "wb") as file:
            pickle.dump(pi_v_2, file)

        print("Kaon sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            K_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"s_K_v_2p5.pickle", "wb") as file:
            pickle.dump(K_v_2, file)

        print("Rho sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            rho_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"s_rho_v_2p5.pickle", "wb") as file:
            pickle.dump(rho_v_2, file)

        print("Omega sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            omega_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"s_omega_v_2p5.pickle", "wb") as file:
            pickle.dump(omega_v_2, file)

        print("Diquark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"s_D_v_2p5.pickle", "wb") as file:
            pickle.dump(D_v_2, file)

        print("Nucleon sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"s_N_v_2p5.pickle", "wb") as file:
            pickle.dump(N_v_2, file)

        print("Tetraquark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            T_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"s_T_v_2p5.pickle", "wb") as file:
            pickle.dump(T_v_2, file)

        print("F-quark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"s_F_v_2p5.pickle", "wb") as file:
            pickle.dump(F_v_2, file)

        print("Pentaquark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"s_P_v_2p5.pickle", "wb") as file:
            pickle.dump(P_v_2, file)

        print("Q-quark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"s_Q_v_2p5.pickle", "wb") as file:
            pickle.dump(Q_v_2, file)

        print("Hexaquark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"s_H_v_2p5.pickle", "wb") as file:
            pickle.dump(H_v_2, file)
    else:
        with open(files+"s_pi_v_2p5.pickle", "rb") as file:
            pi_v_2 = pickle.load(file)
        with open(files+"s_K_v_2p5.pickle", "rb") as file:
            K_v_2 = pickle.load(file)
        with open(files+"s_rho_v_2p5.pickle", "rb") as file:
            rho_v_2 = pickle.load(file)
        with open(files+"s_omega_v_2p5.pickle", "rb") as file:
            omega_v_2 = pickle.load(file)
        with open(files+"s_D_v_2p5.pickle", "rb") as file:
            D_v_2 = pickle.load(file)
        with open(files+"s_N_v_2p5.pickle", "rb") as file:
            N_v_2 = pickle.load(file)
        with open(files+"s_T_v_2p5.pickle", "rb") as file:
            T_v_2 = pickle.load(file)
        with open(files+"s_F_v_2p5.pickle", "rb") as file:
            F_v_2 = pickle.load(file)
        with open(files+"s_P_v_2p5.pickle", "rb") as file:
            P_v_2 = pickle.load(file)
        with open(files+"s_Q_v_2p5.pickle", "rb") as file:
            Q_v_2 = pickle.load(file)
        with open(files+"s_H_v_2p5.pickle", "rb") as file:
            H_v_2 = pickle.load(file)

    if calc_2:

        print("Pion sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"s_pi_v_2p5s.pickle", "wb") as file:
            pickle.dump(pi_v_2s, file)

        print("Kaon sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            K_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"s_K_v_2p5s.pickle", "wb") as file:
            pickle.dump(K_v_2s, file)

        print("Rho sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            rho_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"s_rho_v_2p5s.pickle", "wb") as file:
            pickle.dump(rho_v_2s, file)

        print("Omega sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            omega_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"s_omega_v_2p5s.pickle", "wb") as file:
            pickle.dump(omega_v_2s, file)

        print("Diquark sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"s_D_v_2p5s.pickle", "wb") as file:
            pickle.dump(D_v_2s, file)

        print("Nucleon sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"s_N_v_2p5s.pickle", "wb") as file:
            pickle.dump(N_v_2s, file)

        print("Tetraquark sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            T_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"s_T_v_2p5s.pickle", "wb") as file:
            pickle.dump(T_v_2s, file)

        print("F-quark sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"s_F_v_2p5s.pickle", "wb") as file:
            pickle.dump(F_v_2s, file)

        print("Pentaquark sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"s_P_v_2p5s.pickle", "wb") as file:
            pickle.dump(P_v_2s, file)

        print("Q-quark sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"s_Q_v_2p5s.pickle", "wb") as file:
            pickle.dump(Q_v_2s, file)

        print("Hexaquark sdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"s_H_v_2p5s.pickle", "wb") as file:
            pickle.dump(H_v_2s, file)
    else:
        with open(files+"s_pi_v_2p5s.pickle", "rb") as file:
            pi_v_2s = pickle.load(file)
        with open(files+"s_K_v_2p5s.pickle", "rb") as file:
            K_v_2s = pickle.load(file)
        with open(files+"s_rho_v_2p5s.pickle", "rb") as file:
            rho_v_2s = pickle.load(file)
        with open(files+"s_omega_v_2p5s.pickle", "rb") as file:
            omega_v_2s = pickle.load(file)
        with open(files+"s_D_v_2p5s.pickle", "rb") as file:
            D_v_2s = pickle.load(file)
        with open(files+"s_N_v_2p5s.pickle", "rb") as file:
            N_v_2s = pickle.load(file)
        with open(files+"s_T_v_2p5s.pickle", "rb") as file:
            T_v_2s = pickle.load(file)
        with open(files+"s_F_v_2p5s.pickle", "rb") as file:
            F_v_2s = pickle.load(file)
        with open(files+"s_P_v_2p5s.pickle", "rb") as file:
            P_v_2s = pickle.load(file)
        with open(files+"s_Q_v_2p5s.pickle", "rb") as file:
            Q_v_2s = pickle.load(file)
        with open(files+"s_H_v_2p5s.pickle", "rb") as file:
            H_v_2s = pickle.load(file)

    total_qgp_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1, gluon_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_qgp_nog_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_pert_1 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1
            )
    ]
    total_cluster_1 = [
        sum(el) for el in 
            zip(pi_v_1, K_v_1, rho_v_1, N_v_1, P_v_1, H_v_1, omega_v_1, T_v_1)
    ]
    total_ccluster_1 = [
        sum(el) for el in 
            zip(D_v_1, F_v_1, Q_v_1)
    ]
    total_cluster_1s = [
        sum(el) for el in 
            zip(pi_v_1s, K_v_1s, rho_v_1s, N_v_1s, P_v_1s, H_v_1s, omega_v_1s, T_v_1s)
    ]
    total_ccluster_1s = [
        sum(el) for el in 
            zip(D_v_1s, F_v_1s, Q_v_1s)
    ]
    total_1 = [sum(el) for el in zip(total_qgp_1, total_cluster_1, total_ccluster_1)]
    total_1s = [sum(el) for el in zip(total_qgp_1, total_cluster_1s, total_ccluster_1s)]

    total_qgp_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2, gluon_v_2,
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_qgp_nog_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_pert_2 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2
            )
    ]
    total_cluster_2 = [
        sum(el) for el in 
            zip(pi_v_2, K_v_2, rho_v_2, N_v_2, P_v_2, H_v_2, omega_v_2, T_v_2)
    ]
    total_ccluster_2 = [
        sum(el) for el in 
            zip(D_v_2, F_v_2, Q_v_2)
    ]
    total_cluster_2s = [
        sum(el) for el in 
            zip(pi_v_2s, K_v_2s, rho_v_2s, N_v_2s, P_v_2s, H_v_2s, omega_v_2s, T_v_2s)
    ]
    total_ccluster_2s = [
        sum(el) for el in 
            zip(D_v_2s, F_v_2s, Q_v_2s)
    ]
    total_2 = [sum(el) for el in zip(total_qgp_2, total_cluster_2, total_ccluster_2)]
    total_2s = [sum(el) for el in zip(total_qgp_2, total_cluster_2s, total_ccluster_2s)]

    lQCD_1_x, lQCD_1_y = \
        utils.data_load(
            "D://EoS//epja//lattice_data_raw//2212_09043_fig13_top_right_0p0_alt2.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_1 = [[x, y] for x, y in zip(lQCD_1_x, lQCD_1_y)]

    lQCD_2_x, lQCD_2_y = \
        utils.data_load(
            "D://EoS//epja//lattice_data_raw//2212_09043_fig13_top_right_2p5.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD_2 = [[x, y] for x, y in zip(lQCD_2_x, lQCD_2_y)]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12.0, 5.0))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.axis([80., 280., -3.0, 20.0])

    ax1.add_patch(matplotlib.patches.Polygon(lQCD_1, 
            closed = True, fill = True, color = 'green', alpha = 0.3))

    ax1.plot(T_1, total_cluster_1, '-', c = 'green')
    ax1.plot(T_1, total_cluster_1s, '-.', c = 'green')
    ax1.plot(T_1, total_ccluster_1, '-', c = 'red')
    ax1.plot(T_1, total_ccluster_1s, '-.', c = 'red')
    ax1.plot(T_1, total_1, '-', c = 'black')
    ax1.plot(T_1, total_1s, '-.', c = 'black')
    ax1.plot(T_1, total_qgp_nog_1, ':', c = 'blue')

    ax1.text(171, 5.5, r"Bollweg et al. (2022)", color="green", fontsize=14)
    ax1.text(85, 18.5, r"$\mathrm{\mu_B/T=0}$", color="black", fontsize=14)
    ax1.text(190, -1.5, r"Color singlet clusters", color="green", fontsize=14)
    ax1.text(190, 0.5, r"Color charged clusters", color="red", fontsize=14)
    ax1.text(250, 8.5, r"Quarks", color="blue", fontsize=14)
    ax1.text(125, 13, r"Total entropy density", color="black", fontsize=14)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.axis([80., 280., -3.0, 20.0])

    ax2.add_patch(matplotlib.patches.Polygon(lQCD_2, 
            closed = True, fill = True, color = 'green', alpha = 0.3))

    ax2.plot(T_2, total_cluster_2, '-', c = 'green')
    ax2.plot(T_2, total_cluster_2s, '-.', c = 'green')
    ax2.plot(T_2, total_ccluster_2, '-', c = 'red')
    ax2.plot(T_2, total_ccluster_2s, '-.', c = 'red')
    ax2.plot(T_2, total_2, '-', c = 'black')
    ax2.plot(T_2, total_2s, '-.', c = 'black')
    ax2.plot(T_2, total_qgp_nog_2, ':', c = 'blue')

    ax2.text(165, 6.5, r"Bollweg et al. (2022)", color="green", fontsize=14)
    ax2.text(85, 18.5, r"$\mathrm{\mu_B/T=2.5}$", color="black", fontsize=14)
    ax2.text(190, -1.5, r"Color singlet clusters", color="green", fontsize=14)
    ax2.text(190, 0.5, r"Color charged clusters", color="red", fontsize=14)
    ax2.text(250, 10.5, r"Quarks", color="blue", fontsize=14)
    ax2.text(100, 13, r"Total entropy density", color="black", fontsize=14)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_figure12_n():

    import tqdm
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

    import utils
    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_2

    warnings.filterwarnings("ignore")
    
    calc_1 = True
    calc_2 = True

    files = "D:/EoS/epja/figure12/"

    T_1 = numpy.linspace(1.0, 280.0, 200)
    T_2 = numpy.linspace(1.0, 280.0, 200)

    mu_1 = [0.0 / 3.0 for el in T_1]
    mu_2 = [(2.5 * el) / 3.0 for el in T_2]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()

    sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1 = \
        list(), list(), list(), list()
    gluon_v_1 = list()
    perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
        list(), list(), list()
    perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = \
        list(), list(), list(), list()

    pi_v_1, K_v_1, rho_v_1, D_v_1, N_v_1, F_v_1, P_v_1, Q_v_1, H_v_1 = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_1, T_v_1 = list(), list()

    pi_v_1s, K_v_1s, rho_v_1s, D_v_1s, N_v_1s, F_v_1s, P_v_1s, Q_v_1s, H_v_1s = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_1s, T_v_1s = list(), list()

    phi_re_v_2, phi_im_v_2 = \
        list(), list()

    sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2 = \
        list(), list(), list(), list()
    gluon_v_2 = list()
    perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2 = \
        list(), list(), list()
    perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2 = \
        list(), list(), list(), list()

    pi_v_2, K_v_2, rho_v_2, D_v_2, N_v_2, F_v_2, P_v_2, Q_v_2, H_v_2 = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_2, T_v_2 = list(), list()

    pi_v_2s, K_v_2s, rho_v_2s, D_v_2s, N_v_2s, F_v_2s, P_v_2s, Q_v_2s, H_v_2s = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_2s, T_v_2s = list(), list()

    if calc_1:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #1")
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
        with open(files+"phi_re_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)

        print("Sigma bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"b_sigma_v_0p0.pickle", "wb") as file:
            pickle.dump(sigma_v_1, file)

        print("Gluon bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            gluon_v_1.append(
                pnjl.thermo.gcp_pl_polynomial.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )/(T_el**3)
            )
        with open(files+"b_gluon_v_0p0.pickle", "wb") as file:
            pickle.dump(gluon_v_1, file)

        print("Sea bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_1.append(lq_temp/(T_el**3))
            sea_d_v_1.append(lq_temp/(T_el**3))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"b_sea_u_v_0p0.pickle", "wb") as file:
            pickle.dump(sea_u_v_1, file)
        with open(files+"b_sea_d_v_0p0.pickle", "wb") as file:
            pickle.dump(sea_d_v_1, file)
        with open(files+"b_sea_s_v_0p0.pickle", "wb") as file:
            pickle.dump(sea_s_v_1, file)

        print("Perturbative bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            perturbative_u_v_1.append(lq_temp/(T_el**3))
            perturbative_d_v_1.append(lq_temp/(T_el**3))
            perturbative_s_v_1.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
            else:
                perturbative_gluon_v_1.append(0.0)
        with open(files+"b_perturbative_u_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_1, file)
        with open(files+"b_perturbative_d_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_1, file)
        with open(files+"b_perturbative_s_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_1, file)
        with open(files+"b_perturbative_gluon_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_1, file)

        print("PNJL bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            pnjl_u_v_1.append(lq_temp/(T_el**3))
            pnjl_d_v_1.append(lq_temp/(T_el**3))
            pnjl_s_v_1.append(sq_temp/(T_el**3))
        with open(files+"b_pnjl_u_v_0p0.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_1, file)
        with open(files+"b_pnjl_d_v_0p0.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_1, file)
        with open(files+"b_pnjl_s_v_0p0.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_1, file)
    else:
        with open(files+"phi_re_v_0p0.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_0p0.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
        with open(files+"b_sigma_v_0p0.pickle", "rb") as file:
            sigma_v_1 = pickle.load(file)
        with open(files+"b_gluon_v_0p0.pickle", "rb") as file:
            gluon_v_1 = pickle.load(file)
        with open(files+"b_sea_u_v_0p0.pickle", "rb") as file:
            sea_u_v_1 = pickle.load(file)
        with open(files+"b_sea_d_v_0p0.pickle", "rb") as file:
            sea_d_v_1 = pickle.load(file)
        with open(files+"b_sea_s_v_0p0.pickle", "rb") as file:
            sea_s_v_1 = pickle.load(file)
        with open(files+"b_perturbative_u_v_0p0.pickle", "rb") as file:
            perturbative_u_v_1 = pickle.load(file)
        with open(files+"b_perturbative_d_v_0p0.pickle", "rb") as file:
            perturbative_d_v_1 = pickle.load(file)
        with open(files+"b_perturbative_s_v_0p0.pickle", "rb") as file:
            perturbative_s_v_1 = pickle.load(file)
        with open(files+"b_perturbative_gluon_v_0p0.pickle", "rb") as file:
            perturbative_gluon_v_1 = pickle.load(file)
        with open(files+"b_pnjl_u_v_0p0.pickle", "rb") as file:
            pnjl_u_v_1 = pickle.load(file)
        with open(files+"b_pnjl_d_v_0p0.pickle", "rb") as file:
            pnjl_d_v_1 = pickle.load(file)
        with open(files+"b_pnjl_s_v_0p0.pickle", "rb") as file:
            pnjl_s_v_1 = pickle.load(file)

    if calc_1:

        print("Pion bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"b_pi_v_0p0.pickle", "wb") as file:
            pickle.dump(pi_v_1, file)

        print("Kaon bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"b_K_v_0p0.pickle", "wb") as file:
            pickle.dump(K_v_1, file)

        print("Rho bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"b_rho_v_0p0.pickle", "wb") as file:
            pickle.dump(rho_v_1, file)

        print("Omega bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"b_omega_v_0p0.pickle", "wb") as file:
            pickle.dump(omega_v_1, file)

        print("Diquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"b_D_v_0p0.pickle", "wb") as file:
            pickle.dump(D_v_1, file)

        print("Nucleon bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"b_N_v_0p0.pickle", "wb") as file:
            pickle.dump(N_v_1, file)

        print("Tetraquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"b_T_v_0p0.pickle", "wb") as file:
            pickle.dump(T_v_1, file)

        print("F-quark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"b_F_v_0p0.pickle", "wb") as file:
            pickle.dump(F_v_1, file)

        print("Pentaquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"b_P_v_0p0.pickle", "wb") as file:
            pickle.dump(P_v_1, file)

        print("Q-quark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"b_Q_v_0p0.pickle", "wb") as file:
            pickle.dump(Q_v_1, file)

        print("Hexaquark bdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"b_H_v_0p0.pickle", "wb") as file:
            pickle.dump(H_v_1, file)
    else:
        with open(files+"b_pi_v_0p0.pickle", "rb") as file:
            pi_v_1 = pickle.load(file)
        with open(files+"b_K_v_0p0.pickle", "rb") as file:
            K_v_1 = pickle.load(file)
        with open(files+"b_rho_v_0p0.pickle", "rb") as file:
            rho_v_1 = pickle.load(file)
        with open(files+"b_omega_v_0p0.pickle", "rb") as file:
            omega_v_1 = pickle.load(file)
        with open(files+"b_D_v_0p0.pickle", "rb") as file:
            D_v_1 = pickle.load(file)
        with open(files+"b_N_v_0p0.pickle", "rb") as file:
            N_v_1 = pickle.load(file)
        with open(files+"b_T_v_0p0.pickle", "rb") as file:
            T_v_1 = pickle.load(file)
        with open(files+"b_F_v_0p0.pickle", "rb") as file:
            F_v_1 = pickle.load(file)
        with open(files+"b_P_v_0p0.pickle", "rb") as file:
            P_v_1 = pickle.load(file)
        with open(files+"b_Q_v_0p0.pickle", "rb") as file:
            Q_v_1 = pickle.load(file)
        with open(files+"b_H_v_0p0.pickle", "rb") as file:
            H_v_1 = pickle.load(file)

    if calc_1:

        print("Pion bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"b_pi_v_0p0s.pickle", "wb") as file:
            pickle.dump(pi_v_1s, file)

        print("Kaon bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"b_K_v_0p0s.pickle", "wb") as file:
            pickle.dump(K_v_1s, file)

        print("Rho bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"b_rho_v_0p0s.pickle", "wb") as file:
            pickle.dump(rho_v_1s, file)

        print("Omega bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"b_omega_v_0p0s.pickle", "wb") as file:
            pickle.dump(omega_v_1s, file)

        print("Diquark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"b_D_v_0p0s.pickle", "wb") as file:
            pickle.dump(D_v_1s, file)

        print("Nucleon bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"b_N_v_0p0s.pickle", "wb") as file:
            pickle.dump(N_v_1s, file)

        print("Tetraquark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"b_T_v_0p0s.pickle", "wb") as file:
            pickle.dump(T_v_1s, file)

        print("F-quark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"b_F_v_0p0s.pickle", "wb") as file:
            pickle.dump(F_v_1s, file)

        print("Pentaquark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"b_P_v_0p0s.pickle", "wb") as file:
            pickle.dump(P_v_1s, file)

        print("Q-quark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"b_Q_v_0p0s.pickle", "wb") as file:
            pickle.dump(Q_v_1s, file)

        print("Hexaquark bdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"b_H_v_0p0s.pickle", "wb") as file:
            pickle.dump(H_v_1s, file)
    else:
        with open(files+"b_pi_v_0p0s.pickle", "rb") as file:
            pi_v_1s = pickle.load(file)
        with open(files+"b_K_v_0p0s.pickle", "rb") as file:
            K_v_1s = pickle.load(file)
        with open(files+"b_rho_v_0p0s.pickle", "rb") as file:
            rho_v_1s = pickle.load(file)
        with open(files+"b_omega_v_0p0s.pickle", "rb") as file:
            omega_v_1s = pickle.load(file)
        with open(files+"b_D_v_0p0s.pickle", "rb") as file:
            D_v_1s = pickle.load(file)
        with open(files+"b_N_v_0p0s.pickle", "rb") as file:
            N_v_1s = pickle.load(file)
        with open(files+"b_T_v_0p0s.pickle", "rb") as file:
            T_v_1s = pickle.load(file)
        with open(files+"b_F_v_0p0s.pickle", "rb") as file:
            F_v_1s = pickle.load(file)
        with open(files+"b_P_v_0p0s.pickle", "rb") as file:
            P_v_1s = pickle.load(file)
        with open(files+"b_Q_v_0p0s.pickle", "rb") as file:
            Q_v_1s = pickle.load(file)
        with open(files+"b_H_v_0p0s.pickle", "rb") as file:
            H_v_1s = pickle.load(file)

    if calc_2:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #2")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_2, mu_2), total=len(T_2), ncols=100
        ):
            phi_result = solver_2.Polyakov_loop(
                T_el, mu_el, phi_re_0, phi_im_0
            )
            phi_re_v_2.append(phi_result[0])
            phi_im_v_2.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)

        print("Sigma bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            sigma_v_2.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"b_sigma_v_2p5.pickle", "wb") as file:
            pickle.dump(sigma_v_2, file)

        print("Gluon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            gluon_v_2.append(
                pnjl.thermo.gcp_pl_polynomial.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop
                )/(T_el**3)
            )
        with open(files+"b_gluon_v_2p5.pickle", "wb") as file:
            pickle.dump(gluon_v_2, file)

        print("Sea bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_2.append(lq_temp/(T_el**3))
            sea_d_v_2.append(lq_temp/(T_el**3))
            sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"b_sea_u_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_u_v_2, file)
        with open(files+"b_sea_d_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_d_v_2, file)
        with open(files+"b_sea_s_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_s_v_2, file)

        print("Perturbative bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'l'
            )
            perturbative_u_v_2.append(lq_temp/(T_el**3))
            perturbative_d_v_2.append(lq_temp/(T_el**3))
            perturbative_s_v_2.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
            else:
                perturbative_gluon_v_2.append(0.0)
        with open(files+"b_perturbative_u_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_2, file)
        with open(files+"b_perturbative_d_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_2, file)
        with open(files+"b_perturbative_s_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_2, file)
        with open(files+"b_perturbative_gluon_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_2, file)

        print("PNJL bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 's'
            )
            pnjl_u_v_2.append(lq_temp/(T_el**3))
            pnjl_d_v_2.append(lq_temp/(T_el**3))
            pnjl_s_v_2.append(sq_temp/(T_el**3))
        with open(files+"b_pnjl_u_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_2, file)
        with open(files+"b_pnjl_d_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_2, file)
        with open(files+"b_pnjl_s_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_2, file)
    else:
        with open(files+"phi_re_v_2p5.pickle", "rb") as file:
            phi_re_v_2 = pickle.load(file)
        with open(files+"phi_im_v_2p5.pickle", "rb") as file:
            phi_im_v_2 = pickle.load(file)
        with open(files+"b_sigma_v_2p5.pickle", "rb") as file:
            sigma_v_2 = pickle.load(file)
        with open(files+"b_gluon_v_2p5.pickle", "rb") as file:
            gluon_v_2 = pickle.load(file)
        with open(files+"b_sea_u_v_2p5.pickle", "rb") as file:
            sea_u_v_2 = pickle.load(file)
        with open(files+"b_sea_d_v_2p5.pickle", "rb") as file:
            sea_d_v_2 = pickle.load(file)
        with open(files+"b_sea_s_v_2p5.pickle", "rb") as file:
            sea_s_v_2 = pickle.load(file)
        with open(files+"b_perturbative_u_v_2p5.pickle", "rb") as file:
            perturbative_u_v_2 = pickle.load(file)
        with open(files+"b_perturbative_d_v_2p5.pickle", "rb") as file:
            perturbative_d_v_2 = pickle.load(file)
        with open(files+"b_perturbative_s_v_2p5.pickle", "rb") as file:
            perturbative_s_v_2 = pickle.load(file)
        with open(files+"b_perturbative_gluon_v_2p5.pickle", "rb") as file:
            perturbative_gluon_v_2 = pickle.load(file)
        with open(files+"b_pnjl_u_v_2p5.pickle", "rb") as file:
            pnjl_u_v_2 = pickle.load(file)
        with open(files+"b_pnjl_d_v_2p5.pickle", "rb") as file:
            pnjl_d_v_2 = pickle.load(file)
        with open(files+"b_pnjl_s_v_2p5.pickle", "rb") as file:
            pnjl_s_v_2 = pickle.load(file)

    if calc_2:

        print("Pion bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"b_pi_v_2p5.pickle", "wb") as file:
            pickle.dump(pi_v_2, file)

        print("Kaon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            K_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"b_K_v_2p5.pickle", "wb") as file:
            pickle.dump(K_v_2, file)

        print("Rho bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            rho_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"b_rho_v_2p5.pickle", "wb") as file:
            pickle.dump(rho_v_2, file)

        print("Omega bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            omega_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"b_omega_v_2p5.pickle", "wb") as file:
            pickle.dump(omega_v_2, file)

        print("Diquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"b_D_v_2p5.pickle", "wb") as file:
            pickle.dump(D_v_2, file)

        print("Nucleon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"b_N_v_2p5.pickle", "wb") as file:
            pickle.dump(N_v_2, file)

        print("Tetraquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            T_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"b_T_v_2p5.pickle", "wb") as file:
            pickle.dump(T_v_2, file)

        print("F-quark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"b_F_v_2p5.pickle", "wb") as file:
            pickle.dump(F_v_2, file)

        print("Pentaquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"b_P_v_2p5.pickle", "wb") as file:
            pickle.dump(P_v_2, file)

        print("Q-quark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"b_Q_v_2p5.pickle", "wb") as file:
            pickle.dump(Q_v_2, file)

        print("Hexaquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"b_H_v_2p5.pickle", "wb") as file:
            pickle.dump(H_v_2, file)
    else:
        with open(files+"b_pi_v_2p5.pickle", "rb") as file:
            pi_v_2 = pickle.load(file)
        with open(files+"b_K_v_2p5.pickle", "rb") as file:
            K_v_2 = pickle.load(file)
        with open(files+"b_rho_v_2p5.pickle", "rb") as file:
            rho_v_2 = pickle.load(file)
        with open(files+"b_omega_v_2p5.pickle", "rb") as file:
            omega_v_2 = pickle.load(file)
        with open(files+"b_D_v_2p5.pickle", "rb") as file:
            D_v_2 = pickle.load(file)
        with open(files+"b_N_v_2p5.pickle", "rb") as file:
            N_v_2 = pickle.load(file)
        with open(files+"b_T_v_2p5.pickle", "rb") as file:
            T_v_2 = pickle.load(file)
        with open(files+"b_F_v_2p5.pickle", "rb") as file:
            F_v_2 = pickle.load(file)
        with open(files+"b_P_v_2p5.pickle", "rb") as file:
            P_v_2 = pickle.load(file)
        with open(files+"b_Q_v_2p5.pickle", "rb") as file:
            Q_v_2 = pickle.load(file)
        with open(files+"b_H_v_2p5.pickle", "rb") as file:
            H_v_2 = pickle.load(file)

    if calc_2:

        print("Pion bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"b_pi_v_2p5s.pickle", "wb") as file:
            pickle.dump(pi_v_2s, file)

        print("Kaon bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            K_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"b_K_v_2p5s.pickle", "wb") as file:
            pickle.dump(K_v_2s, file)

        print("Rho bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            rho_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"b_rho_v_2p5s.pickle", "wb") as file:
            pickle.dump(rho_v_2s, file)

        print("Omega bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            omega_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"b_omega_v_2p5s.pickle", "wb") as file:
            pickle.dump(omega_v_2s, file)

        print("Diquark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"b_D_v_2p5s.pickle", "wb") as file:
            pickle.dump(D_v_2s, file)

        print("Nucleon bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"b_N_v_2p5s.pickle", "wb") as file:
            pickle.dump(N_v_2s, file)

        print("Tetraquark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            T_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"b_T_v_2p5s.pickle", "wb") as file:
            pickle.dump(T_v_2s, file)

        print("F-quark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"b_F_v_2p5s.pickle", "wb") as file:
            pickle.dump(F_v_2s, file)

        print("Pentaquark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"b_P_v_2p5s.pickle", "wb") as file:
            pickle.dump(P_v_2s, file)

        print("Q-quark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"b_Q_v_2p5s.pickle", "wb") as file:
            pickle.dump(Q_v_2s, file)

        print("Hexaquark bdensity #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2s.append(
                cluster_s.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"b_H_v_2p5s.pickle", "wb") as file:
            pickle.dump(H_v_2s, file)
    else:
        with open(files+"b_pi_v_2p5s.pickle", "rb") as file:
            pi_v_2s = pickle.load(file)
        with open(files+"b_K_v_2p5s.pickle", "rb") as file:
            K_v_2s = pickle.load(file)
        with open(files+"b_rho_v_2p5s.pickle", "rb") as file:
            rho_v_2s = pickle.load(file)
        with open(files+"b_omega_v_2p5s.pickle", "rb") as file:
            omega_v_2s = pickle.load(file)
        with open(files+"b_D_v_2p5s.pickle", "rb") as file:
            D_v_2s = pickle.load(file)
        with open(files+"b_N_v_2p5s.pickle", "rb") as file:
            N_v_2s = pickle.load(file)
        with open(files+"b_T_v_2p5s.pickle", "rb") as file:
            T_v_2s = pickle.load(file)
        with open(files+"b_F_v_2p5s.pickle", "rb") as file:
            F_v_2s = pickle.load(file)
        with open(files+"b_P_v_2p5s.pickle", "rb") as file:
            P_v_2s = pickle.load(file)
        with open(files+"b_Q_v_2p5s.pickle", "rb") as file:
            Q_v_2s = pickle.load(file)
        with open(files+"b_H_v_2p5s.pickle", "rb") as file:
            H_v_2s = pickle.load(file)

    total_qgp_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1, gluon_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_qgp_nog_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_pert_1 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1
            )
    ]
    total_cluster_1 = [
        sum(el) for el in 
            zip(pi_v_1, K_v_1, rho_v_1, N_v_1, P_v_1, H_v_1, omega_v_1, T_v_1)
    ]
    total_ccluster_1 = [
        sum(el) for el in 
            zip(D_v_1, F_v_1, Q_v_1)
    ]
    total_cluster_1s = [
        sum(el) for el in 
            zip(pi_v_1s, K_v_1s, rho_v_1s, N_v_1s, P_v_1s, H_v_1s, omega_v_1s, T_v_1s)
    ]
    total_ccluster_1s = [
        sum(el) for el in 
            zip(D_v_1s, F_v_1s, Q_v_1s)
    ]
    total_1 = [sum(el) for el in zip(total_qgp_1, total_cluster_1, total_ccluster_1)]
    total_1s = [sum(el) for el in zip(total_qgp_1, total_cluster_1s, total_ccluster_1s)]

    total_qgp_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2, gluon_v_2,
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_qgp_nog_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_pert_2 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2
            )
    ]
    total_cluster_2 = [
        sum(el) for el in 
            zip(pi_v_2, K_v_2, rho_v_2, N_v_2, P_v_2, H_v_2, omega_v_2, T_v_2)
    ]
    total_ccluster_2 = [
        sum(el) for el in 
            zip(D_v_2, F_v_2, Q_v_2)
    ]
    total_cluster_2s = [
        sum(el) for el in 
            zip(pi_v_2s, K_v_2s, rho_v_2s, N_v_2s, P_v_2s, H_v_2s, omega_v_2s, T_v_2s)
    ]
    total_ccluster_2s = [
        sum(el) for el in 
            zip(D_v_2s, F_v_2s, Q_v_2s)
    ]
    total_2 = [sum(el) for el in zip(total_qgp_2, total_cluster_2, total_ccluster_2)]
    total_2s = [sum(el) for el in zip(total_qgp_2, total_cluster_2s, total_ccluster_2s)]

    # lQCD_1_x, lQCD_1_y = \
    #     utils.data_load(
    #         "D://EoS//epja//lattice_data_raw//2212_09043_fig13_top_right_0p0_alt2.dat", 0, 1,
    #         firstrow=0, delim=' '
    #     )
    # lQCD_1 = [[x, y] for x, y in zip(lQCD_1_x, lQCD_1_y)]

    # lQCD_2_x, lQCD_2_y = \
    #     utils.data_load(
    #         "D://EoS//epja//lattice_data_raw//2212_09043_fig13_top_right_2p5.dat", 0, 1,
    #         firstrow=0, delim=' '
    #     )
    # lQCD_2 = [[x, y] for x, y in zip(lQCD_2_x, lQCD_2_y)]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (6.0, 5.0))
    ax2 = fig1.add_subplot(1, 1, 1)
    ax2.axis([80., 280., -0.2, 0.8])

    # ax2.add_patch(matplotlib.patches.Polygon(lQCD_2, 
    #         closed = True, fill = True, color = 'green', alpha = 0.3))

    ax2.plot(T_2, total_cluster_2, '-', c = 'green')
    ax2.plot(T_2, total_cluster_2s, '-.', c = 'green')
    ax2.plot(T_2, total_ccluster_2, '-', c = 'red')
    ax2.plot(T_2, total_ccluster_2s, '-.', c = 'red')
    ax2.plot(T_2, total_2, '-', c = 'black')
    ax2.plot(T_2, total_2s, '-.', c = 'black')
    ax2.plot(T_2, total_qgp_nog_2, ':', c = 'blue')

    # ax2.text(165, 6.5, r"Bollweg et al. (2022)", color="green", fontsize=14)
    ax2.text(85, 18.5, r"$\mathrm{\mu_B/T=2.5}$", color="black", fontsize=14)
    ax2.text(190, -1.5, r"Color singlet clusters", color="green", fontsize=14)
    ax2.text(190, 0.5, r"Color charged clusters", color="red", fontsize=14)
    ax2.text(250, 10.5, r"Quarks", color="blue", fontsize=14)
    ax2.text(100, 13, r"Total entropy density", color="black", fontsize=14)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_beth_uhlenbeck1():

    import tqdm
    import numpy
    import pickle
    import warnings
    import platform

    import matplotlib.pyplot
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    import utils
    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_2

    warnings.filterwarnings("ignore")
    
    calc_1 = False
    calc_2 = False

    files = "D:/EoS/epja/beth_uhlenbeck/"
    lattice_files = "D:/EoS/epja/lattice_data_raw/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/beth_uhlenbeck/"
        lattice_files = "/home/mcierniak/Data/2023_epja/lattice_data_raw/"

    T_1 = numpy.linspace(1.0, 280.0, 200)
    T_2 = numpy.linspace(1.0, 280.0, 200)

    mu_1 = [0.0 / 3.0 for el in T_1]
    mu_2 = [(2.5 * el) / 3.0 for el in T_2]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()

    sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1 = \
        list(), list(), list(), list()
    gluon_v_1 = list()
    perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
        list(), list(), list()
    perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = \
        list(), list(), list(), list()

    pi_v_1, K_v_1, rho_v_1, D_v_1, N_v_1, F_v_1, P_v_1, Q_v_1, H_v_1 = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_1, T_v_1 = list(), list()

    pi_v_1s, K_v_1s, rho_v_1s, D_v_1s, N_v_1s, F_v_1s, P_v_1s, Q_v_1s, H_v_1s = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_1s, T_v_1s = list(), list()

    phi_re_v_2, phi_im_v_2 = \
        list(), list()

    sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2 = \
        list(), list(), list(), list()
    gluon_v_2 = list()
    perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2 = \
        list(), list(), list()
    perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2 = \
        list(), list(), list(), list()

    pi_v_2, K_v_2, rho_v_2, D_v_2, N_v_2, F_v_2, P_v_2, Q_v_2, H_v_2 = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_2, T_v_2 = list(), list()

    pi_v_2s, K_v_2s, rho_v_2s, D_v_2s, N_v_2s, F_v_2s, P_v_2s, Q_v_2s, H_v_2s = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_2s, T_v_2s = list(), list()

    if False:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #1")
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
        with open(files+"phi_re_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_0p0.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)

        print("Sigma sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.sdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"s_sigma_v_0p0.pickle", "wb") as file:
            pickle.dump(sigma_v_1, file)

        print("Gluon sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            gluon_v_1.append(
                pnjl.thermo.gcp_pl_polynomial.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )/(T_el**3)
            )
        with open(files+"s_gluon_v_0p0.pickle", "wb") as file:
            pickle.dump(gluon_v_1, file)

        print("Sea sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.sdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_1.append(lq_temp/(T_el**3))
            sea_d_v_1.append(lq_temp/(T_el**3))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.sdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"s_sea_u_v_0p0.pickle", "wb") as file:
            pickle.dump(sea_u_v_1, file)
        with open(files+"s_sea_d_v_0p0.pickle", "wb") as file:
            pickle.dump(sea_d_v_1, file)
        with open(files+"s_sea_s_v_0p0.pickle", "wb") as file:
            pickle.dump(sea_s_v_1, file)

        print("Perturbative sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            perturbative_u_v_1.append(lq_temp/(T_el**3))
            perturbative_d_v_1.append(lq_temp/(T_el**3))
            perturbative_s_v_1.append(
                pnjl.thermo.gcp_perturbative.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
            else:
                perturbative_gluon_v_1.append(0.0)
        with open(files+"s_perturbative_u_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_1, file)
        with open(files+"s_perturbative_d_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_1, file)
        with open(files+"s_perturbative_s_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_1, file)
        with open(files+"s_perturbative_gluon_v_0p0.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_1, file)

        print("PNJL sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            pnjl_u_v_1.append(lq_temp/(T_el**3))
            pnjl_d_v_1.append(lq_temp/(T_el**3))
            pnjl_s_v_1.append(sq_temp/(T_el**3))
        with open(files+"s_pnjl_u_v_0p0.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_1, file)
        with open(files+"s_pnjl_d_v_0p0.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_1, file)
        with open(files+"s_pnjl_s_v_0p0.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_1, file)
    else:
        with open(files+"phi_re_v_0p0.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_0p0.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
        with open(files+"s_sigma_v_0p0.pickle", "rb") as file:
            sigma_v_1 = pickle.load(file)
        with open(files+"s_gluon_v_0p0.pickle", "rb") as file:
            gluon_v_1 = pickle.load(file)
        with open(files+"s_sea_u_v_0p0.pickle", "rb") as file:
            sea_u_v_1 = pickle.load(file)
        with open(files+"s_sea_d_v_0p0.pickle", "rb") as file:
            sea_d_v_1 = pickle.load(file)
        with open(files+"s_sea_s_v_0p0.pickle", "rb") as file:
            sea_s_v_1 = pickle.load(file)
        with open(files+"s_perturbative_u_v_0p0.pickle", "rb") as file:
            perturbative_u_v_1 = pickle.load(file)
        with open(files+"s_perturbative_d_v_0p0.pickle", "rb") as file:
            perturbative_d_v_1 = pickle.load(file)
        with open(files+"s_perturbative_s_v_0p0.pickle", "rb") as file:
            perturbative_s_v_1 = pickle.load(file)
        with open(files+"s_perturbative_gluon_v_0p0.pickle", "rb") as file:
            perturbative_gluon_v_1 = pickle.load(file)
        with open(files+"s_pnjl_u_v_0p0.pickle", "rb") as file:
            pnjl_u_v_1 = pickle.load(file)
        with open(files+"s_pnjl_d_v_0p0.pickle", "rb") as file:
            pnjl_d_v_1 = pickle.load(file)
        with open(files+"s_pnjl_s_v_0p0.pickle", "rb") as file:
            pnjl_s_v_1 = pickle.load(file)

    if False:

        print("Pion sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"s_pi_v_0p0.pickle", "wb") as file:
            pickle.dump(pi_v_1, file)

        print("Kaon sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"s_K_v_0p0.pickle", "wb") as file:
            pickle.dump(K_v_1, file)

        print("Rho sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"s_rho_v_0p0.pickle", "wb") as file:
            pickle.dump(rho_v_1, file)

        print("Omega sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"s_omega_v_0p0.pickle", "wb") as file:
            pickle.dump(omega_v_1, file)

        print("Diquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"s_D_v_0p0.pickle", "wb") as file:
            pickle.dump(D_v_1, file)

        print("Nucleon sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"s_N_v_0p0.pickle", "wb") as file:
            pickle.dump(N_v_1, file)

        print("Tetraquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"s_T_v_0p0.pickle", "wb") as file:
            pickle.dump(T_v_1, file)

        print("F-quark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"s_F_v_0p0.pickle", "wb") as file:
            pickle.dump(F_v_1, file)

        print("Pentaquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"s_P_v_0p0.pickle", "wb") as file:
            pickle.dump(P_v_1, file)

        print("Q-quark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"s_Q_v_0p0.pickle", "wb") as file:
            pickle.dump(Q_v_1, file)

        print("Hexaquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"s_H_v_0p0.pickle", "wb") as file:
            pickle.dump(H_v_1, file)
    else:
        with open(files+"s_pi_v_0p0.pickle", "rb") as file:
            pi_v_1 = pickle.load(file)
        with open(files+"s_K_v_0p0.pickle", "rb") as file:
            K_v_1 = pickle.load(file)
        with open(files+"s_rho_v_0p0.pickle", "rb") as file:
            rho_v_1 = pickle.load(file)
        with open(files+"s_omega_v_0p0.pickle", "rb") as file:
            omega_v_1 = pickle.load(file)
        with open(files+"s_D_v_0p0.pickle", "rb") as file:
            D_v_1 = pickle.load(file)
        with open(files+"s_N_v_0p0.pickle", "rb") as file:
            N_v_1 = pickle.load(file)
        with open(files+"s_T_v_0p0.pickle", "rb") as file:
            T_v_1 = pickle.load(file)
        with open(files+"s_F_v_0p0.pickle", "rb") as file:
            F_v_1 = pickle.load(file)
        with open(files+"s_P_v_0p0.pickle", "rb") as file:
            P_v_1 = pickle.load(file)
        with open(files+"s_Q_v_0p0.pickle", "rb") as file:
            Q_v_1 = pickle.load(file)
        with open(files+"s_H_v_0p0.pickle", "rb") as file:
            H_v_1 = pickle.load(file)

    if calc_1:

        print("Pion sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**3)
            )
        with open(files+"s_buns_pi_v_0p0s.pickle", "wb") as file:
            pickle.dump(pi_v_1s, file)

        print("Kaon sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**3)
            )
        with open(files+"s_buns_K_v_0p0s.pickle", "wb") as file:
            pickle.dump(K_v_1s, file)

        print("Rho sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**3)
            )
        with open(files+"s_buns_rho_v_0p0s.pickle", "wb") as file:
            pickle.dump(rho_v_1s, file)

        print("Omega sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**3)
            )
        with open(files+"s_buns_omega_v_0p0s.pickle", "wb") as file:
            pickle.dump(omega_v_1s, file)

        print("Diquark sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**3)
            )
        with open(files+"s_buns_D_v_0p0s.pickle", "wb") as file:
            pickle.dump(D_v_1s, file)

        print("Nucleon sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**3)
            )
        with open(files+"s_buns_N_v_0p0s.pickle", "wb") as file:
            pickle.dump(N_v_1s, file)

        print("Tetraquark sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**3)
            )
        with open(files+"s_buns_T_v_0p0s.pickle", "wb") as file:
            pickle.dump(T_v_1s, file)

        print("F-quark sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**3)
            )
        with open(files+"s_buns_F_v_0p0s.pickle", "wb") as file:
            pickle.dump(F_v_1s, file)

        print("Pentaquark sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**3)
            )
        with open(files+"s_buns_P_v_0p0s.pickle", "wb") as file:
            pickle.dump(P_v_1s, file)

        print("Q-quark sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**3)
            )
        with open(files+"s_buns_Q_v_0p0s.pickle", "wb") as file:
            pickle.dump(Q_v_1s, file)

        print("Hexaquark sdensity Beth-Uhlenbeck no sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**3)
            )
        with open(files+"s_buns_H_v_0p0s.pickle", "wb") as file:
            pickle.dump(H_v_1s, file)
    else:
        with open(files+"s_buns_pi_v_0p0s.pickle", "rb") as file:
            pi_v_1s = pickle.load(file)
        with open(files+"s_buns_K_v_0p0s.pickle", "rb") as file:
            K_v_1s = pickle.load(file)
        with open(files+"s_buns_rho_v_0p0s.pickle", "rb") as file:
            rho_v_1s = pickle.load(file)
        with open(files+"s_buns_omega_v_0p0s.pickle", "rb") as file:
            omega_v_1s = pickle.load(file)
        with open(files+"s_buns_D_v_0p0s.pickle", "rb") as file:
            D_v_1s = pickle.load(file)
        with open(files+"s_buns_N_v_0p0s.pickle", "rb") as file:
            N_v_1s = pickle.load(file)
        with open(files+"s_buns_T_v_0p0s.pickle", "rb") as file:
            T_v_1s = pickle.load(file)
        with open(files+"s_buns_F_v_0p0s.pickle", "rb") as file:
            F_v_1s = pickle.load(file)
        with open(files+"s_buns_P_v_0p0s.pickle", "rb") as file:
            P_v_1s = pickle.load(file)
        with open(files+"s_buns_Q_v_0p0s.pickle", "rb") as file:
            Q_v_1s = pickle.load(file)
        with open(files+"s_buns_H_v_0p0s.pickle", "rb") as file:
            H_v_1s = pickle.load(file)

    if False:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #2")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_2, mu_2), total=len(T_2), ncols=100
        ):
            phi_result = solver_2.Polyakov_loop(
                T_el, mu_el, phi_re_0, phi_im_0
            )
            phi_re_v_2.append(phi_result[0])
            phi_im_v_2.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)

        print("Sigma sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            sigma_v_2.append(
                pnjl.thermo.gcp_sigma_lattice.sdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"s_sigma_v_2p5.pickle", "wb") as file:
            pickle.dump(sigma_v_2, file)

        print("Gluon sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            gluon_v_2.append(
                pnjl.thermo.gcp_pl_polynomial.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop
                )/(T_el**3)
            )
        with open(files+"s_gluon_v_2p5.pickle", "wb") as file:
            pickle.dump(gluon_v_2, file)

        print("Sea sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.sdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_2.append(lq_temp/(T_el**3))
            sea_d_v_2.append(lq_temp/(T_el**3))
            sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.sdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"s_sea_u_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_u_v_2, file)
        with open(files+"s_sea_d_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_d_v_2, file)
        with open(files+"s_sea_s_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_s_v_2, file)

        print("Perturbative sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'l'
            )
            perturbative_u_v_2.append(lq_temp/(T_el**3))
            perturbative_d_v_2.append(lq_temp/(T_el**3))
            perturbative_s_v_2.append(
                pnjl.thermo.gcp_perturbative.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
            else:
                perturbative_gluon_v_2.append(0.0)
        with open(files+"s_perturbative_u_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_2, file)
        with open(files+"s_perturbative_d_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_2, file)
        with open(files+"s_perturbative_s_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_2, file)
        with open(files+"s_perturbative_gluon_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_2, file)

        print("PNJL sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 's'
            )
            pnjl_u_v_2.append(lq_temp/(T_el**3))
            pnjl_d_v_2.append(lq_temp/(T_el**3))
            pnjl_s_v_2.append(sq_temp/(T_el**3))
        with open(files+"s_pnjl_u_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_2, file)
        with open(files+"s_pnjl_d_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_2, file)
        with open(files+"s_pnjl_s_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_2, file)
    else:
        with open(files+"phi_re_v_2p5.pickle", "rb") as file:
            phi_re_v_2 = pickle.load(file)
        with open(files+"phi_im_v_2p5.pickle", "rb") as file:
            phi_im_v_2 = pickle.load(file)
        with open(files+"s_sigma_v_2p5.pickle", "rb") as file:
            sigma_v_2 = pickle.load(file)
        with open(files+"s_gluon_v_2p5.pickle", "rb") as file:
            gluon_v_2 = pickle.load(file)
        with open(files+"s_sea_u_v_2p5.pickle", "rb") as file:
            sea_u_v_2 = pickle.load(file)
        with open(files+"s_sea_d_v_2p5.pickle", "rb") as file:
            sea_d_v_2 = pickle.load(file)
        with open(files+"s_sea_s_v_2p5.pickle", "rb") as file:
            sea_s_v_2 = pickle.load(file)
        with open(files+"s_perturbative_u_v_2p5.pickle", "rb") as file:
            perturbative_u_v_2 = pickle.load(file)
        with open(files+"s_perturbative_d_v_2p5.pickle", "rb") as file:
            perturbative_d_v_2 = pickle.load(file)
        with open(files+"s_perturbative_s_v_2p5.pickle", "rb") as file:
            perturbative_s_v_2 = pickle.load(file)
        with open(files+"s_perturbative_gluon_v_2p5.pickle", "rb") as file:
            perturbative_gluon_v_2 = pickle.load(file)
        with open(files+"s_pnjl_u_v_2p5.pickle", "rb") as file:
            pnjl_u_v_2 = pickle.load(file)
        with open(files+"s_pnjl_d_v_2p5.pickle", "rb") as file:
            pnjl_d_v_2 = pickle.load(file)
        with open(files+"s_pnjl_s_v_2p5.pickle", "rb") as file:
            pnjl_s_v_2 = pickle.load(file)

    if False:

        print("Pion sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"s_pi_v_2p5.pickle", "wb") as file:
            pickle.dump(pi_v_2, file)

        print("Kaon sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            K_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"s_K_v_2p5.pickle", "wb") as file:
            pickle.dump(K_v_2, file)

        print("Rho sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            rho_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"s_rho_v_2p5.pickle", "wb") as file:
            pickle.dump(rho_v_2, file)

        print("Omega sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            omega_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"s_omega_v_2p5.pickle", "wb") as file:
            pickle.dump(omega_v_2, file)

        print("Diquark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"s_D_v_2p5.pickle", "wb") as file:
            pickle.dump(D_v_2, file)

        print("Nucleon sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"s_N_v_2p5.pickle", "wb") as file:
            pickle.dump(N_v_2, file)

        print("Tetraquark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            T_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"s_T_v_2p5.pickle", "wb") as file:
            pickle.dump(T_v_2, file)

        print("F-quark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"s_F_v_2p5.pickle", "wb") as file:
            pickle.dump(F_v_2, file)

        print("Pentaquark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"s_P_v_2p5.pickle", "wb") as file:
            pickle.dump(P_v_2, file)

        print("Q-quark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"s_Q_v_2p5.pickle", "wb") as file:
            pickle.dump(Q_v_2, file)

        print("Hexaquark sdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"s_H_v_2p5.pickle", "wb") as file:
            pickle.dump(H_v_2, file)
    else:
        with open(files+"s_pi_v_2p5.pickle", "rb") as file:
            pi_v_2 = pickle.load(file)
        with open(files+"s_K_v_2p5.pickle", "rb") as file:
            K_v_2 = pickle.load(file)
        with open(files+"s_rho_v_2p5.pickle", "rb") as file:
            rho_v_2 = pickle.load(file)
        with open(files+"s_omega_v_2p5.pickle", "rb") as file:
            omega_v_2 = pickle.load(file)
        with open(files+"s_D_v_2p5.pickle", "rb") as file:
            D_v_2 = pickle.load(file)
        with open(files+"s_N_v_2p5.pickle", "rb") as file:
            N_v_2 = pickle.load(file)
        with open(files+"s_T_v_2p5.pickle", "rb") as file:
            T_v_2 = pickle.load(file)
        with open(files+"s_F_v_2p5.pickle", "rb") as file:
            F_v_2 = pickle.load(file)
        with open(files+"s_P_v_2p5.pickle", "rb") as file:
            P_v_2 = pickle.load(file)
        with open(files+"s_Q_v_2p5.pickle", "rb") as file:
            Q_v_2 = pickle.load(file)
        with open(files+"s_H_v_2p5.pickle", "rb") as file:
            H_v_2 = pickle.load(file)

    if calc_2:

        print("Pion sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**3)
            )
        with open(files+"s_buns_pi_v_2p5s.pickle", "wb") as file:
            pickle.dump(pi_v_2s, file)

        print("Kaon sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            K_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**3)
            )
        with open(files+"s_buns_K_v_2p5s.pickle", "wb") as file:
            pickle.dump(K_v_2s, file)

        print("Rho sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            rho_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**3)
            )
        with open(files+"s_buns_rho_v_2p5s.pickle", "wb") as file:
            pickle.dump(rho_v_2s, file)

        print("Omega sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            omega_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**3)
            )
        with open(files+"s_buns_omega_v_2p5s.pickle", "wb") as file:
            pickle.dump(omega_v_2s, file)

        print("Diquark sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**3)
            )
        with open(files+"s_buns_D_v_2p5s.pickle", "wb") as file:
            pickle.dump(D_v_2s, file)

        print("Nucleon sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**3)
            )
        with open(files+"s_buns_N_v_2p5s.pickle", "wb") as file:
            pickle.dump(N_v_2s, file)

        print("Tetraquark sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            T_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**3)
            )
        with open(files+"s_buns_T_v_2p5s.pickle", "wb") as file:
            pickle.dump(T_v_2s, file)

        print("F-quark sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**3)
            )
        with open(files+"s_buns_F_v_2p5s.pickle", "wb") as file:
            pickle.dump(F_v_2s, file)

        print("Pentaquark sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**3)
            )
        with open(files+"s_buns_P_v_2p5s.pickle", "wb") as file:
            pickle.dump(P_v_2s, file)

        print("Q-quark sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**3)
            )
        with open(files+"s_buns_Q_v_2p5s.pickle", "wb") as file:
            pickle.dump(Q_v_2s, file)

        print("Hexaquark sdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2s.append(
                cluster.sdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**3)
            )
        with open(files+"s_buns_H_v_2p5s.pickle", "wb") as file:
            pickle.dump(H_v_2s, file)
    else:
        with open(files+"s_buns_pi_v_2p5s.pickle", "rb") as file:
            pi_v_2s = pickle.load(file)
        with open(files+"s_buns_K_v_2p5s.pickle", "rb") as file:
            K_v_2s = pickle.load(file)
        with open(files+"s_buns_rho_v_2p5s.pickle", "rb") as file:
            rho_v_2s = pickle.load(file)
        with open(files+"s_buns_omega_v_2p5s.pickle", "rb") as file:
            omega_v_2s = pickle.load(file)
        with open(files+"s_buns_D_v_2p5s.pickle", "rb") as file:
            D_v_2s = pickle.load(file)
        with open(files+"s_buns_N_v_2p5s.pickle", "rb") as file:
            N_v_2s = pickle.load(file)
        with open(files+"s_buns_T_v_2p5s.pickle", "rb") as file:
            T_v_2s = pickle.load(file)
        with open(files+"s_buns_F_v_2p5s.pickle", "rb") as file:
            F_v_2s = pickle.load(file)
        with open(files+"s_buns_P_v_2p5s.pickle", "rb") as file:
            P_v_2s = pickle.load(file)
        with open(files+"s_buns_Q_v_2p5s.pickle", "rb") as file:
            Q_v_2s = pickle.load(file)
        with open(files+"s_buns_H_v_2p5s.pickle", "rb") as file:
            H_v_2s = pickle.load(file)

    total_qgp_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1, gluon_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_q_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_pnjl_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1, gluon_v_1,
                pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_pert_1 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1
            )
    ]
    total_cluster_1 = [
        sum(el) for el in 
            zip(pi_v_1, K_v_1, rho_v_1, N_v_1, P_v_1, H_v_1, omega_v_1, T_v_1)
    ]
    total_ccluster_1 = [
        sum(el) for el in 
            zip(D_v_1, F_v_1, Q_v_1)
    ]
    total_cluster_1s = [
        sum(el) for el in 
            zip(pi_v_1s, K_v_1s, rho_v_1s, N_v_1s, P_v_1s, H_v_1s, omega_v_1s, T_v_1s)
    ]
    total_ccluster_1s = [
        sum(el) for el in 
            zip(D_v_1s, F_v_1s, Q_v_1s)
    ]
    total_1 = [sum(el) for el in zip(total_qgp_1, total_cluster_1, total_ccluster_1)]
    total_1s = [sum(el) for el in zip(total_qgp_1, total_cluster_1s, total_ccluster_1s)]
    total_MHRG_1s = [sum(el) for el in zip(total_cluster_1s, total_ccluster_1s)]

    total_qgp_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2, gluon_v_2,
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_q_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
                pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_pnjl_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2, gluon_v_2,
                pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_pert_2 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2
            )
    ]
    total_cluster_2 = [
        sum(el) for el in 
            zip(pi_v_2, K_v_2, rho_v_2, N_v_2, P_v_2, H_v_2, omega_v_2, T_v_2)
    ]
    total_ccluster_2 = [
        sum(el) for el in 
            zip(D_v_2, F_v_2, Q_v_2)
    ]
    total_cluster_2s = [
        sum(el) for el in 
            zip(pi_v_2s, K_v_2s, rho_v_2s, N_v_2s, P_v_2s, H_v_2s, omega_v_2s, T_v_2s)
    ]
    total_ccluster_2s = [
        sum(el) for el in 
            zip(D_v_2s, F_v_2s, Q_v_2s)
    ]
    total_2 = [sum(el) for el in zip(total_qgp_2, total_cluster_2, total_ccluster_2)]
    total_2s = [sum(el) for el in zip(total_qgp_2, total_cluster_2s, total_ccluster_2s)]
    total_MHRG_2s = [sum(el) for el in zip(total_cluster_2s, total_ccluster_2s)]    

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

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12.0, 5.0))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.axis([80., 280., -6.0, 20.0])

    ax1.add_patch(matplotlib.patches.Polygon(lQCD_1, 
            closed = True, fill = True, color = 'green', alpha = 0.3))

    ax1.plot(T_1, total_1s, '-', c = 'black')
    ax1.plot(T_1, total_MHRG_1s, '--', c = 'green')
    ax1.plot(T_1, total_qgp_1, '--', c = 'blue')
    ax1.plot(T_1, total_pnjl_1, '-.', c = 'darkblue')
    ax1.plot(T_1, total_pert_1, '-.', c = 'magenta')
    ax1.plot(T_1, gluon_v_1, ':', c = 'red')
    ax1.plot(T_1, total_q_1, ':', c = 'purple')

    ax1.text(185, 7.5, r"Bollweg et al. (2022)", color="green", fontsize=14)
    ax1.text(85, 18.5, r"$\mathrm{\mu_B/T=0}$", color="black", fontsize=14)
    ax1.text(105, 2.5, r"Total", color="black", fontsize=14)
    ax1.text(250, 15.5, r"QGP", color="blue", fontsize=14)
    ax1.text(190, 0.5, r"MHRG", color="green", fontsize=14)
    ax1.text(190, 17.5, r"PNJL", color="darkblue", fontsize=14)
    ax1.text(188, -3.5, r"Perturbative correction", color="magenta", fontsize=14)
    ax1.text(220, 5.5, r"Polyakov-loop", color="red", fontsize=14)
    ax1.text(170, 12.5, r"Quarks", color="purple", fontsize=14)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.axis([80., 280., -6.0, 20.0])

    ax2.add_patch(matplotlib.patches.Polygon(lQCD_2, 
            closed = True, fill = True, color = 'green', alpha = 0.3))

    ax2.plot(T_2, total_2s, '-', c = 'black')
    ax2.plot(T_1, total_MHRG_2s, '--', c = 'green')
    ax2.plot(T_1, total_qgp_2, '--', c = 'blue')
    ax2.plot(T_2, total_pnjl_2, '-.', c = 'darkblue')
    ax2.plot(T_2, total_pert_2, '-.', c = 'magenta')
    ax2.plot(T_2, gluon_v_2, ':', c = 'red')
    ax2.plot(T_2, total_q_2, ':', c = 'purple')

    ax2.text(185, 9.0, r"Bollweg et al. (2022)", color="green", fontsize=14)
    ax2.text(85, 18.5, r"$\mathrm{\mu_B/T=2.5}$", color="black", fontsize=14)
    ax2.text(102, 2.5, r"Total", color="black", fontsize=14)
    ax2.text(250, 17.5, r"QGP", color="blue", fontsize=14)
    ax2.text(190, 0.5, r"MHRG", color="green", fontsize=14)
    ax2.text(164, 17.5, r"PNJL", color="darkblue", fontsize=14)
    ax2.text(188, -3.8, r"Perturbative correction", color="magenta", fontsize=14)
    ax2.text(220, 5.5, r"Polyakov-loop", color="red", fontsize=14)
    ax2.text(160, 14.2, r"Quarks", color="purple", fontsize=14)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_beth_uhlenbeck2():

    import tqdm
    import numpy
    import pickle
    import warnings
    import platform

    import matplotlib.pyplot
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")

    import utils
    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_2

    warnings.filterwarnings("ignore")
    
    calc_1 = False
    calc_2 = False

    files = "D:/EoS/epja/beth_uhlenbeck/"
    lattice_files = "D:/EoS/epja/lattice_data_raw/"
    if platform.system() == "Linux":
        files = "/home/mcierniak/Data/2023_epja/beth_uhlenbeck/"
        lattice_files = "/home/mcierniak/Data/2023_epja/lattice_data_raw/"

    T_1 = numpy.linspace(1.0, 280.0, 200)
    T_2 = numpy.linspace(1.0, 280.0, 200)

    mu_1 = [(2.5 * el) / 3.0 for el in T_2]
    mu_2 = [(2.5 * el) / 3.0 for el in T_2]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()

    sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1 = \
        list(), list(), list(), list()
    gluon_v_1 = list()
    perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
        list(), list(), list()
    perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = \
        list(), list(), list(), list()

    pi_v_1, K_v_1, rho_v_1, D_v_1, N_v_1, F_v_1, P_v_1, Q_v_1, H_v_1 = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_1, T_v_1 = list(), list()

    pi_v_1s, K_v_1s, rho_v_1s, D_v_1s, N_v_1s, F_v_1s, P_v_1s, Q_v_1s, H_v_1s = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_1s, T_v_1s = list(), list()

    phi_re_v_2, phi_im_v_2 = \
        list(), list()

    sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2 = \
        list(), list(), list(), list()
    gluon_v_2 = list()
    perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2 = \
        list(), list(), list()
    perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2 = \
        list(), list(), list(), list()

    pi_v_2, K_v_2, rho_v_2, D_v_2, N_v_2, F_v_2, P_v_2, Q_v_2, H_v_2 = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_2, T_v_2 = list(), list()

    pi_v_2s, K_v_2s, rho_v_2s, D_v_2s, N_v_2s, F_v_2s, P_v_2s, Q_v_2s, H_v_2s = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_2s, T_v_2s = list(), list()

    if calc_2:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #2")
        for T_el, mu_el in tqdm.tqdm(
            zip(T_2, mu_2), total=len(T_2), ncols=100
        ):
            phi_result = solver_2.Polyakov_loop(
                T_el, mu_el, phi_re_0, phi_im_0
            )
            phi_re_v_2.append(phi_result[0])
            phi_im_v_2.append(phi_result[1])
            phi_re_0 = phi_result[0]
            phi_im_0 = phi_result[1]
        with open(files+"phi_re_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_re_v_2, file)
        with open(files+"phi_im_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_im_v_2, file)

        print("Sigma bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            sigma_v_2.append(
                pnjl.thermo.gcp_sigma_lattice.bdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"b_sigma_v_2p5.pickle", "wb") as file:
            pickle.dump(sigma_v_2, file)

        print("Gluon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            gluon_v_2.append(
                pnjl.thermo.gcp_pl_polynomial.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop
                )/(T_el**3)
            )
        with open(files+"b_gluon_v_2p5.pickle", "wb") as file:
            pickle.dump(gluon_v_2, file)

        print("Sea bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.bdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_2.append(lq_temp/(T_el**3))
            sea_d_v_2.append(lq_temp/(T_el**3))
            sea_s_v_2.append(
                pnjl.thermo.gcp_sea_lattice.bdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"b_sea_u_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_u_v_2, file)
        with open(files+"b_sea_d_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_d_v_2, file)
        with open(files+"b_sea_s_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_s_v_2, file)

        print("Perturbative bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'l'
            )
            perturbative_u_v_2.append(lq_temp/(T_el**3))
            perturbative_d_v_2.append(lq_temp/(T_el**3))
            perturbative_s_v_2.append(
                pnjl.thermo.gcp_perturbative.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
            else:
                perturbative_gluon_v_2.append(0.0)
        with open(files+"b_perturbative_u_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_2, file)
        with open(files+"b_perturbative_d_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_2, file)
        with open(files+"b_perturbative_s_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_2, file)
        with open(files+"b_perturbative_gluon_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_2, file)

        print("PNJL bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.bdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 's'
            )
            pnjl_u_v_2.append(lq_temp/(T_el**3))
            pnjl_d_v_2.append(lq_temp/(T_el**3))
            pnjl_s_v_2.append(sq_temp/(T_el**3))
        with open(files+"b_pnjl_u_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_2, file)
        with open(files+"b_pnjl_d_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_2, file)
        with open(files+"b_pnjl_s_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_2, file)
    else:
        with open(files+"phi_re_v_2p5.pickle", "rb") as file:
            phi_re_v_2 = pickle.load(file)
        with open(files+"phi_im_v_2p5.pickle", "rb") as file:
            phi_im_v_2 = pickle.load(file)
        with open(files+"b_sigma_v_2p5.pickle", "rb") as file:
            sigma_v_2 = pickle.load(file)
        with open(files+"b_gluon_v_2p5.pickle", "rb") as file:
            gluon_v_2 = pickle.load(file)
        with open(files+"b_sea_u_v_2p5.pickle", "rb") as file:
            sea_u_v_2 = pickle.load(file)
        with open(files+"b_sea_d_v_2p5.pickle", "rb") as file:
            sea_d_v_2 = pickle.load(file)
        with open(files+"b_sea_s_v_2p5.pickle", "rb") as file:
            sea_s_v_2 = pickle.load(file)
        with open(files+"b_perturbative_u_v_2p5.pickle", "rb") as file:
            perturbative_u_v_2 = pickle.load(file)
        with open(files+"b_perturbative_d_v_2p5.pickle", "rb") as file:
            perturbative_d_v_2 = pickle.load(file)
        with open(files+"b_perturbative_s_v_2p5.pickle", "rb") as file:
            perturbative_s_v_2 = pickle.load(file)
        with open(files+"b_perturbative_gluon_v_2p5.pickle", "rb") as file:
            perturbative_gluon_v_2 = pickle.load(file)
        with open(files+"b_pnjl_u_v_2p5.pickle", "rb") as file:
            pnjl_u_v_2 = pickle.load(file)
        with open(files+"b_pnjl_d_v_2p5.pickle", "rb") as file:
            pnjl_d_v_2 = pickle.load(file)
        with open(files+"b_pnjl_s_v_2p5.pickle", "rb") as file:
            pnjl_s_v_2 = pickle.load(file)

    if calc_2:

        print("Pion bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"b_pi_v_2p5.pickle", "wb") as file:
            pickle.dump(pi_v_2, file)

        print("Kaon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            K_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"b_K_v_2p5.pickle", "wb") as file:
            pickle.dump(K_v_2, file)

        print("Rho bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            rho_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"b_rho_v_2p5.pickle", "wb") as file:
            pickle.dump(rho_v_2, file)

        print("Omega bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            omega_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"b_omega_v_2p5.pickle", "wb") as file:
            pickle.dump(omega_v_2, file)

        print("Diquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"b_D_v_2p5.pickle", "wb") as file:
            pickle.dump(D_v_2, file)

        print("Nucleon bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"b_N_v_2p5.pickle", "wb") as file:
            pickle.dump(N_v_2, file)

        print("Tetraquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            T_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"b_T_v_2p5.pickle", "wb") as file:
            pickle.dump(T_v_2, file)

        print("F-quark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"b_F_v_2p5.pickle", "wb") as file:
            pickle.dump(F_v_2, file)

        print("Pentaquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"b_P_v_2p5.pickle", "wb") as file:
            pickle.dump(P_v_2, file)

        print("Q-quark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"b_Q_v_2p5.pickle", "wb") as file:
            pickle.dump(Q_v_2, file)

        print("Hexaquark bdensity #2")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2.append(
                cluster.bdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_2.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"b_H_v_2p5.pickle", "wb") as file:
            pickle.dump(H_v_2, file)
    else:
        with open(files+"b_pi_v_2p5.pickle", "rb") as file:
            pi_v_2 = pickle.load(file)
        with open(files+"b_K_v_2p5.pickle", "rb") as file:
            K_v_2 = pickle.load(file)
        with open(files+"b_rho_v_2p5.pickle", "rb") as file:
            rho_v_2 = pickle.load(file)
        with open(files+"b_omega_v_2p5.pickle", "rb") as file:
            omega_v_2 = pickle.load(file)
        with open(files+"b_D_v_2p5.pickle", "rb") as file:
            D_v_2 = pickle.load(file)
        with open(files+"b_N_v_2p5.pickle", "rb") as file:
            N_v_2 = pickle.load(file)
        with open(files+"b_T_v_2p5.pickle", "rb") as file:
            T_v_2 = pickle.load(file)
        with open(files+"b_F_v_2p5.pickle", "rb") as file:
            F_v_2 = pickle.load(file)
        with open(files+"b_P_v_2p5.pickle", "rb") as file:
            P_v_2 = pickle.load(file)
        with open(files+"b_Q_v_2p5.pickle", "rb") as file:
            Q_v_2 = pickle.load(file)
        with open(files+"b_H_v_2p5.pickle", "rb") as file:
            H_v_2 = pickle.load(file)

    if calc_2:

        print("Pion bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            pi_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**3)
            )
        with open(files+"b_buns_pi_v_2p5s.pickle", "wb") as file:
            pickle.dump(pi_v_2s, file)

        print("Kaon bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            K_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**3)
            )
        with open(files+"b_buns_K_v_2p5s.pickle", "wb") as file:
            pickle.dump(K_v_2s, file)

        print("Rho bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            rho_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**3)
            )
        with open(files+"b_buns_rho_v_2p5s.pickle", "wb") as file:
            pickle.dump(rho_v_2s, file)

        print("Omega bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            omega_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**3)
            )
        with open(files+"b_buns_omega_v_2p5s.pickle", "wb") as file:
            pickle.dump(omega_v_2s, file)

        print("Diquark bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            D_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**3)
            )
        with open(files+"b_buns_D_v_2p5s.pickle", "wb") as file:
            pickle.dump(D_v_2s, file)

        print("Nucleon bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            N_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**3)
            )
        with open(files+"b_buns_N_v_2p5s.pickle", "wb") as file:
            pickle.dump(N_v_2s, file)

        print("Tetraquark bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            T_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**3)
            )
        with open(files+"b_buns_T_v_2p5s.pickle", "wb") as file:
            pickle.dump(T_v_2s, file)

        print("F-quark bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            F_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**3)
            )
        with open(files+"b_buns_F_v_2p5s.pickle", "wb") as file:
            pickle.dump(F_v_2s, file)

        print("Pentaquark bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            P_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**3)
            )
        with open(files+"b_buns_P_v_2p5s.pickle", "wb") as file:
            pickle.dump(P_v_2s, file)

        print("Q-quark bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            Q_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**3)
            )
        with open(files+"b_buns_Q_v_2p5s.pickle", "wb") as file:
            pickle.dump(Q_v_2s, file)

        print("Hexaquark bdensity Beth-Uhlenbeck no sin #2 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_2, mu_2, phi_re_v_2, phi_im_v_2),
            total=len(T_2), ncols=100
        ):
            H_v_2s.append(
                cluster.bdensity_buns(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**3)
            )
        with open(files+"b_buns_H_v_2p5s.pickle", "wb") as file:
            pickle.dump(H_v_2s, file)
    else:
        with open(files+"b_buns_pi_v_2p5s.pickle", "rb") as file:
            pi_v_2s = pickle.load(file)
        with open(files+"b_buns_K_v_2p5s.pickle", "rb") as file:
            K_v_2s = pickle.load(file)
        with open(files+"b_buns_rho_v_2p5s.pickle", "rb") as file:
            rho_v_2s = pickle.load(file)
        with open(files+"b_buns_omega_v_2p5s.pickle", "rb") as file:
            omega_v_2s = pickle.load(file)
        with open(files+"b_buns_D_v_2p5s.pickle", "rb") as file:
            D_v_2s = pickle.load(file)
        with open(files+"b_buns_N_v_2p5s.pickle", "rb") as file:
            N_v_2s = pickle.load(file)
        with open(files+"b_buns_T_v_2p5s.pickle", "rb") as file:
            T_v_2s = pickle.load(file)
        with open(files+"b_buns_F_v_2p5s.pickle", "rb") as file:
            F_v_2s = pickle.load(file)
        with open(files+"b_buns_P_v_2p5s.pickle", "rb") as file:
            P_v_2s = pickle.load(file)
        with open(files+"b_buns_Q_v_2p5s.pickle", "rb") as file:
            Q_v_2s = pickle.load(file)
        with open(files+"b_buns_H_v_2p5s.pickle", "rb") as file:
            H_v_2s = pickle.load(file)

    phi_re_v_1 = phi_re_v_2
    phi_im_v_1 = phi_im_v_2

    sigma_v_1 = sigma_v_2
    sea_u_v_1 = sea_u_v_2
    sea_d_v_1 = sea_d_v_2
    sea_s_v_1 = sea_s_v_2
    gluon_v_1 = gluon_v_2
    perturbative_u_v_1 = perturbative_u_v_2
    perturbative_d_v_1 = perturbative_d_v_2
    perturbative_s_v_1 = perturbative_s_v_2
    perturbative_gluon_v_1 = perturbative_gluon_v_2
    pnjl_u_v_1 = pnjl_u_v_2
    pnjl_d_v_1 = pnjl_d_v_2
    pnjl_s_v_1 = pnjl_s_v_2

    pi_v_1 = pi_v_2s
    K_v_1 = K_v_2s
    rho_v_1 = rho_v_2s
    D_v_1 = D_v_2s
    N_v_1 = N_v_2s
    F_v_1 = F_v_2s
    P_v_1 = P_v_2s
    Q_v_1 = Q_v_2s
    H_v_1 = H_v_2s
    omega_v_1 = omega_v_2s
    T_v_1 = T_v_2s

    if calc_1:

        print("Pion bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**3)
            )
        with open(files+"b_buns_pi_v_0p0s.pickle", "wb") as file:
            pickle.dump(pi_v_1s, file)

        print("Kaon bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**3)
            )
        with open(files+"b_buns_K_v_0p0s.pickle", "wb") as file:
            pickle.dump(K_v_1s, file)

        print("Rho bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**3)
            )
        with open(files+"b_buns_rho_v_0p0s.pickle", "wb") as file:
            pickle.dump(rho_v_1s, file)

        print("Omega bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**3)
            )
        with open(files+"b_buns_omega_v_0p0s.pickle", "wb") as file:
            pickle.dump(omega_v_1s, file)

        print("Diquark bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**3)
            )
        with open(files+"b_buns_D_v_0p0s.pickle", "wb") as file:
            pickle.dump(D_v_1s, file)

        print("Nucleon bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**3)
            )
        with open(files+"b_buns_N_v_0p0s.pickle", "wb") as file:
            pickle.dump(N_v_1s, file)

        print("Tetraquark bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**3)
            )
        with open(files+"b_buns_T_v_0p0s.pickle", "wb") as file:
            pickle.dump(T_v_1s, file)

        print("F-quark bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**3)
            )
        with open(files+"b_buns_F_v_0p0s.pickle", "wb") as file:
            pickle.dump(F_v_1s, file)

        print("Pentaquark bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**3)
            )
        with open(files+"b_buns_P_v_0p0s.pickle", "wb") as file:
            pickle.dump(P_v_1s, file)

        print("Q-quark bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**3)
            )
        with open(files+"b_buns_Q_v_0p0s.pickle", "wb") as file:
            pickle.dump(Q_v_1s, file)

        print("Hexaquark bdensity Beth-Uhlenbeck sin #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1s.append(
                cluster.bdensity_bu(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**3)
            )
        with open(files+"b_buns_H_v_0p0s.pickle", "wb") as file:
            pickle.dump(H_v_1s, file)
    else:
        with open(files+"b_buns_pi_v_0p0s.pickle", "rb") as file:
            pi_v_1s = pickle.load(file)
        with open(files+"b_buns_K_v_0p0s.pickle", "rb") as file:
            K_v_1s = pickle.load(file)
        with open(files+"b_buns_rho_v_0p0s.pickle", "rb") as file:
            rho_v_1s = pickle.load(file)
        with open(files+"b_buns_omega_v_0p0s.pickle", "rb") as file:
            omega_v_1s = pickle.load(file)
        with open(files+"b_buns_D_v_0p0s.pickle", "rb") as file:
            D_v_1s = pickle.load(file)
        with open(files+"b_buns_N_v_0p0s.pickle", "rb") as file:
            N_v_1s = pickle.load(file)
        with open(files+"b_buns_T_v_0p0s.pickle", "rb") as file:
            T_v_1s = pickle.load(file)
        with open(files+"b_buns_F_v_0p0s.pickle", "rb") as file:
            F_v_1s = pickle.load(file)
        with open(files+"b_buns_P_v_0p0s.pickle", "rb") as file:
            P_v_1s = pickle.load(file)
        with open(files+"b_buns_Q_v_0p0s.pickle", "rb") as file:
            Q_v_1s = pickle.load(file)
        with open(files+"b_buns_H_v_0p0s.pickle", "rb") as file:
            H_v_1s = pickle.load(file)

    total_qgp_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1, gluon_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_q_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_pnjl_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1, gluon_v_1,
                pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_pert_1 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1
            )
    ]
    total_cluster_1 = [
        sum(el) for el in 
            zip(pi_v_1, K_v_1, rho_v_1, N_v_1, P_v_1, H_v_1, omega_v_1, T_v_1)
    ]
    total_ccluster_1 = [
        sum(el) for el in 
            zip(D_v_1, F_v_1, Q_v_1)
    ]
    total_cluster_1s = [
        sum(el) for el in 
            zip(pi_v_1s, K_v_1s, rho_v_1s, N_v_1s, P_v_1s, H_v_1s, omega_v_1s, T_v_1s)
    ]
    total_ccluster_1s = [
        sum(el) for el in 
            zip(D_v_1s, F_v_1s, Q_v_1s)
    ]
    total_1 = [sum(el) for el in zip(total_qgp_1, total_cluster_1, total_ccluster_1)]
    total_1s = [sum(el) for el in zip(total_qgp_1, total_cluster_1s, total_ccluster_1s)]
    total_MHRG_1s = [sum(el) for el in zip(total_cluster_1s, total_ccluster_1s)]

    total_qgp_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2, gluon_v_2,
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2, pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_q_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2,
                pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_pnjl_2 = [
        sum(el) for el in 
            zip(
                sigma_v_2, sea_u_v_2, sea_d_v_2, sea_s_v_2, gluon_v_2,
                pnjl_u_v_2, pnjl_d_v_2, pnjl_s_v_2
            )
    ]
    total_pert_2 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_2, perturbative_d_v_2, perturbative_s_v_2,
                perturbative_gluon_v_2
            )
    ]
    total_cluster_2 = [
        sum(el) for el in 
            zip(pi_v_2, K_v_2, rho_v_2, N_v_2, P_v_2, H_v_2, omega_v_2, T_v_2)
    ]
    total_ccluster_2 = [
        sum(el) for el in 
            zip(D_v_2, F_v_2, Q_v_2)
    ]
    total_cluster_2s = [
        sum(el) for el in 
            zip(pi_v_2s, K_v_2s, rho_v_2s, N_v_2s, P_v_2s, H_v_2s, omega_v_2s, T_v_2s)
    ]
    total_ccluster_2s = [
        sum(el) for el in 
            zip(D_v_2s, F_v_2s, Q_v_2s)
    ]
    total_2 = [sum(el) for el in zip(total_qgp_2, total_cluster_2, total_ccluster_2)]
    total_2s = [sum(el) for el in zip(total_qgp_2, total_cluster_2s, total_ccluster_2s)]
    total_MHRG_2s = [sum(el) for el in zip(total_cluster_2s, total_ccluster_2s)]

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

    with open(lattice_files+"2212_09043_mub_T_2p5_nT3.pickle", "rb") as file:
        n_poly = pickle.load(file)

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (6.0, 5.0))

    fig1.subplots_adjust(
        left=0.167, bottom=0.11, right=0.988, top=0.979, wspace=0.2, hspace=0.2
    )

    ax2 = fig1.add_subplot(1, 1, 1)
    ax2.axis([80., 280., -0.4, 1.2])

    ax2.add_patch(
        matplotlib.patches.Polygon(
            n_poly, closed = True, fill = True, color = "green", alpha = 0.3
        )
    )

    ax2.plot(T_2, total_MHRG_2s, '--', c = 'green')
    ax2.plot(T_2, total_qgp_2, '--', c = 'blue')
    ax2.plot(T_2, total_pnjl_2, '-.', c = 'darkblue')
    ax2.plot(T_2, total_pert_2, '-.', c = 'magenta')
    ax2.plot(T_2, total_2s, '-', c = 'black')

    ax2.text(196, 1.1, r"Bollweg et al. (2022)", color="green", fontsize=14)
    ax2.text(85, 1.1, r"$\mathrm{\mu_B/T=2.5}$", color="black", fontsize=14)
    ax2.text(150, 0.03, r"MHRG", color="green", fontsize=14)
    ax2.text(188, -0.18, r"Perturbative correction", color="magenta", fontsize=14)
    ax2.text(250, 0.8, r"PNJL", color="darkblue", fontsize=14)
    ax2.text(146, 0.15, r"QGP", color="blue", fontsize=14)
    ax2.text(250, 0.6, r"Total", color="black", fontsize=14)

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize = 16)
    ax2.set_ylabel(r'$\mathrm{n_B/T^3}$', fontsize = 16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_lattice_thermo2():

    import tqdm
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

    import utils
    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")
    
    calc_1 = False

    files = "D:/EoS/epja/lattice_thermo/"

    T_1 = numpy.linspace(1.0, 280.0, 200)

    mu_1 = [(2.5 * el) / 3.0 for el in T_1]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()

    sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1 = \
        list(), list(), list(), list()
    gluon_v_1 = list()
    perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
        list(), list(), list()
    perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = \
        list(), list(), list(), list()

    pi_v_1, K_v_1, rho_v_1, D_v_1, N_v_1, F_v_1, P_v_1, Q_v_1, H_v_1 = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_1, T_v_1 = list(), list()

    pi_v_1s, K_v_1s, rho_v_1s, D_v_1s, N_v_1s, F_v_1s, P_v_1s, Q_v_1s, H_v_1s = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    omega_v_1s, T_v_1s = list(), list()

    if calc_1:

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
        with open(files+"phi_re_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_re_v_1, file)
        with open(files+"phi_im_v_2p5.pickle", "wb") as file:
            pickle.dump(phi_im_v_1, file)

        print("Sigma sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.sdensity(
                    T_el, mu_el
                )/(T_el**3)
            )
        with open(files+"s_sigma_v_2p5.pickle", "wb") as file:
            pickle.dump(sigma_v_1, file)

        print("Gluon sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            gluon_v_1.append(
                pnjl.thermo.gcp_pl_polynomial.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop
                )/(T_el**3)
            )
        with open(files+"s_gluon_v_2p5.pickle", "wb") as file:
            pickle.dump(gluon_v_1, file)

        print("Sea sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.sdensity(
                T_el, mu_el, 'l'
            )
            sea_u_v_1.append(lq_temp/(T_el**3))
            sea_d_v_1.append(lq_temp/(T_el**3))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.sdensity(
                    T_el, mu_el, 's'
                )/(T_el**3)
            )
        with open(files+"s_sea_u_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_u_v_1, file)
        with open(files+"s_sea_d_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_d_v_1, file)
        with open(files+"s_sea_s_v_2p5.pickle", "wb") as file:
            pickle.dump(sea_s_v_1, file)

        print("Perturbative sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_perturbative.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            perturbative_u_v_1.append(lq_temp/(T_el**3))
            perturbative_d_v_1.append(lq_temp/(T_el**3))
            perturbative_s_v_1.append(
                pnjl.thermo.gcp_perturbative.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
                )/(T_el**3)
            )
            if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
                raise RuntimeError("Perturbative gluon bdensity not implemented!")
            else:
                perturbative_gluon_v_1.append(0.0)
        with open(files+"s_perturbative_u_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_u_v_1, file)
        with open(files+"s_perturbative_d_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_d_v_1, file)
        with open(files+"s_perturbative_s_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_s_v_1, file)
        with open(files+"s_perturbative_gluon_v_2p5.pickle", "wb") as file:
            pickle.dump(perturbative_gluon_v_1, file)

        print("PNJL sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.sdensity(
                T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 's'
            )
            pnjl_u_v_1.append(lq_temp/(T_el**3))
            pnjl_d_v_1.append(lq_temp/(T_el**3))
            pnjl_s_v_1.append(sq_temp/(T_el**3))
        with open(files+"s_pnjl_u_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_u_v_1, file)
        with open(files+"s_pnjl_d_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_d_v_1, file)
        with open(files+"s_pnjl_s_v_2p5.pickle", "wb") as file:
            pickle.dump(pnjl_s_v_1, file)
    else:
        with open(files+"phi_re_v_2p5.pickle", "rb") as file:
            phi_re_v_1 = pickle.load(file)
        with open(files+"phi_im_v_2p5.pickle", "rb") as file:
            phi_im_v_1 = pickle.load(file)
        with open(files+"s_sigma_v_2p5.pickle", "rb") as file:
            sigma_v_1 = pickle.load(file)
        with open(files+"s_gluon_v_2p5.pickle", "rb") as file:
            gluon_v_1 = pickle.load(file)
        with open(files+"s_sea_u_v_2p5.pickle", "rb") as file:
            sea_u_v_1 = pickle.load(file)
        with open(files+"s_sea_d_v_2p5.pickle", "rb") as file:
            sea_d_v_1 = pickle.load(file)
        with open(files+"s_sea_s_v_2p5.pickle", "rb") as file:
            sea_s_v_1 = pickle.load(file)
        with open(files+"s_perturbative_u_v_2p5.pickle", "rb") as file:
            perturbative_u_v_1 = pickle.load(file)
        with open(files+"s_perturbative_d_v_2p5.pickle", "rb") as file:
            perturbative_d_v_1 = pickle.load(file)
        with open(files+"s_perturbative_s_v_2p5.pickle", "rb") as file:
            perturbative_s_v_1 = pickle.load(file)
        with open(files+"s_perturbative_gluon_v_2p5.pickle", "rb") as file:
            perturbative_gluon_v_1 = pickle.load(file)
        with open(files+"s_pnjl_u_v_2p5.pickle", "rb") as file:
            pnjl_u_v_1 = pickle.load(file)
        with open(files+"s_pnjl_d_v_2p5.pickle", "rb") as file:
            pnjl_d_v_1 = pickle.load(file)
        with open(files+"s_pnjl_s_v_2p5.pickle", "rb") as file:
            pnjl_s_v_1 = pickle.load(file)

    if calc_1:

        print("Pion sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"s_pi_v_2p5.pickle", "wb") as file:
            pickle.dump(pi_v_1, file)

        print("Kaon sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"s_K_v_2p5.pickle", "wb") as file:
            pickle.dump(K_v_1, file)

        print("Rho sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"s_rho_v_2p5.pickle", "wb") as file:
            pickle.dump(rho_v_1, file)

        print("Omega sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"s_omega_v_2p5.pickle", "wb") as file:
            pickle.dump(omega_v_1, file)

        print("Diquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"s_D_v_2p5.pickle", "wb") as file:
            pickle.dump(D_v_1, file)

        print("Nucleon sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"s_N_v_2p5.pickle", "wb") as file:
            pickle.dump(N_v_1, file)

        print("Tetraquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"s_T_v_2p5.pickle", "wb") as file:
            pickle.dump(T_v_1, file)

        print("F-quark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"s_F_v_2p5.pickle", "wb") as file:
            pickle.dump(F_v_1, file)

        print("Pentaquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"s_P_v_2p5.pickle", "wb") as file:
            pickle.dump(P_v_1, file)

        print("Q-quark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"s_Q_v_2p5.pickle", "wb") as file:
            pickle.dump(Q_v_1, file)

        print("Hexaquark sdensity #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"s_H_v_2p5.pickle", "wb") as file:
            pickle.dump(H_v_1, file)
    else:
        with open(files+"s_pi_v_2p5.pickle", "rb") as file:
            pi_v_1 = pickle.load(file)
        with open(files+"s_K_v_2p5.pickle", "rb") as file:
            K_v_1 = pickle.load(file)
        with open(files+"s_rho_v_2p5.pickle", "rb") as file:
            rho_v_1 = pickle.load(file)
        with open(files+"s_omega_v_2p5.pickle", "rb") as file:
            omega_v_1 = pickle.load(file)
        with open(files+"s_D_v_2p5.pickle", "rb") as file:
            D_v_1 = pickle.load(file)
        with open(files+"s_N_v_2p5.pickle", "rb") as file:
            N_v_1 = pickle.load(file)
        with open(files+"s_T_v_2p5.pickle", "rb") as file:
            T_v_1 = pickle.load(file)
        with open(files+"s_F_v_2p5.pickle", "rb") as file:
            F_v_1 = pickle.load(file)
        with open(files+"s_P_v_2p5.pickle", "rb") as file:
            P_v_1 = pickle.load(file)
        with open(files+"s_Q_v_2p5.pickle", "rb") as file:
            Q_v_1 = pickle.load(file)
        with open(files+"s_H_v_2p5.pickle", "rb") as file:
            H_v_1 = pickle.load(file)

    if calc_1:

        print("Pion sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'pi'
                )/(T_el**3)
            )
        with open(files+"s_pi_v_2p5s.pickle", "wb") as file:
            pickle.dump(pi_v_1s, file)

        print("Kaon sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1s.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'K'
                )/(T_el**3)
            )
        with open(files+"s_K_v_2p5s.pickle", "wb") as file:
            pickle.dump(K_v_1s, file)

        print("Rho sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1s.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'rho'
                )/(T_el**3)
            )
        with open(files+"s_rho_v_2p5s.pickle", "wb") as file:
            pickle.dump(rho_v_1s, file)

        print("Omega sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1s.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'omega'
                )/(T_el**3)
            )
        with open(files+"s_omega_v_2p5s.pickle", "wb") as file:
            pickle.dump(omega_v_1s, file)

        print("Diquark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'D'
                )/(T_el**3)
            )
        with open(files+"s_D_v_2p5s.pickle", "wb") as file:
            pickle.dump(D_v_1s, file)

        print("Nucleon sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'N'
                )/(T_el**3)
            )
        with open(files+"s_N_v_2p5s.pickle", "wb") as file:
            pickle.dump(N_v_1s, file)

        print("Tetraquark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1s.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'T'
                )/(T_el**3)
            )
        with open(files+"s_T_v_2p5s.pickle", "wb") as file:
            pickle.dump(T_v_1s, file)

        print("F-quark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'F'
                )/(T_el**3)
            )
        with open(files+"s_F_v_2p5s.pickle", "wb") as file:
            pickle.dump(F_v_1s, file)

        print("Pentaquark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'P'
                )/(T_el**3)
            )
        with open(files+"s_P_v_2p5s.pickle", "wb") as file:
            pickle.dump(P_v_1s, file)

        print("Q-quark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1s.append(
                cluster_s.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'Q'
                )/(T_el**3)
            )
        with open(files+"s_Q_v_2p5s.pickle", "wb") as file:
            pickle.dump(Q_v_1s, file)

        print("Hexaquark sdensity #1 (step)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1s.append(
                cluster.sdensity(
                    T_el, mu_el, phi_re_el, phi_im_el, solver_1.Polyakov_loop, 'H'
                )/(T_el**3)
            )
        with open(files+"s_H_v_2p5s.pickle", "wb") as file:
            pickle.dump(H_v_1s, file)
    else:
        with open(files+"s_pi_v_2p5s.pickle", "rb") as file:
            pi_v_1s = pickle.load(file)
        with open(files+"s_K_v_2p5s.pickle", "rb") as file:
            K_v_1s = pickle.load(file)
        with open(files+"s_rho_v_2p5s.pickle", "rb") as file:
            rho_v_1s = pickle.load(file)
        with open(files+"s_omega_v_2p5s.pickle", "rb") as file:
            omega_v_1s = pickle.load(file)
        with open(files+"s_D_v_2p5s.pickle", "rb") as file:
            D_v_1s = pickle.load(file)
        with open(files+"s_N_v_2p5s.pickle", "rb") as file:
            N_v_1s = pickle.load(file)
        with open(files+"s_T_v_2p5s.pickle", "rb") as file:
            T_v_1s = pickle.load(file)
        with open(files+"s_F_v_2p5s.pickle", "rb") as file:
            F_v_1s = pickle.load(file)
        with open(files+"s_P_v_2p5s.pickle", "rb") as file:
            P_v_1s = pickle.load(file)
        with open(files+"s_Q_v_2p5s.pickle", "rb") as file:
            Q_v_1s = pickle.load(file)
        with open(files+"s_H_v_2p5s.pickle", "rb") as file:
            H_v_1s = pickle.load(file)

    total_qgp_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]
    total_pert_1 = [
        sum(el) for el in 
            zip(
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1
            )
    ]
    total_cluster_1 = [
        sum(el) for el in 
            zip(pi_v_1, K_v_1, rho_v_1, N_v_1, P_v_1, H_v_1, omega_v_1, T_v_1)
    ]
    total_ccluster_1 = [
        sum(el) for el in 
            zip(D_v_1, F_v_1, Q_v_1)
    ]
    total_cluster_1s = [
        sum(el) for el in 
            zip(pi_v_1s, K_v_1s, rho_v_1s, N_v_1s, P_v_1s, H_v_1s, omega_v_1s, T_v_1s)
    ]
    total_ccluster_1s = [
        sum(el) for el in 
            zip(D_v_1s, F_v_1s, Q_v_1s)
    ]
    total_1 = [sum(el) for el in zip(total_qgp_1, total_cluster_1, total_ccluster_1)]
    total_1s = [sum(el) for el in zip(total_qgp_1, total_cluster_1s, total_ccluster_1s)]

    norm_QGP_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_1, total_1)
    ]
    norm_QGP_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(total_qgp_1, total_1s)
    ]
    norm_pi_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(pi_v_1, total_1)
    ]
    norm_pi_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(pi_v_1s, total_1s)
    ]
    norm_K_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(K_v_1, total_1)
    ]
    norm_K_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(K_v_1s, total_1s)
    ]
    norm_rho_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(rho_v_1, total_1)
    ]
    norm_rho_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(rho_v_1s, total_1s)
    ]
    norm_D_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_1, total_1)
    ]
    norm_D_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(D_v_1s, total_1s)
    ]
    norm_N_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_1, total_1)
    ]
    norm_N_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(N_v_1s, total_1s)
    ]
    norm_F_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_1, total_1)
    ]
    norm_F_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(F_v_1s, total_1s)
    ]
    norm_P_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_1, total_1)
    ]
    norm_P_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(P_v_1s, total_1s)
    ]
    norm_Q_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_1, total_1)
    ]
    norm_Q_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(Q_v_1s, total_1s)
    ]
    norm_H_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_1, total_1)
    ]
    norm_H_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(H_v_1s, total_1s)
    ]
    norm_omega_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(omega_v_1, total_1)
    ]
    norm_omega_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(omega_v_1s, total_1s)
    ]
    norm_T_1 = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(T_v_1, total_1)
    ]
    norm_T_1s = [
        el_cluster / el_total 
        for el_cluster, el_total in zip(T_v_1s, total_1s)
    ]

    lQCD_x, lQCD_y = \
        utils.data_load(
            "D://EoS//epja//lattice_data_raw//2212_09043_fig13_top_right_2p5.dat", 0, 1,
            firstrow=0, delim=' '
        )
    lQCD = [[x, y] for x, y in zip(lQCD_x, lQCD_y)]

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12.0, 5.0))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.axis([80., 280., -3.0, 16.0])

    ax1.add_patch(matplotlib.patches.Polygon(lQCD, 
            closed = True, fill = True, color = 'green', alpha = 0.3))

    ax1.fill_between(
        T_1, total_cluster_1, y2=total_cluster_1s, color='green', alpha=0.7
    )
    ax1.plot(T_1, total_cluster_1, '--', c = 'green')
    ax1.plot(T_1, total_cluster_1s, '--', c = 'green')
    ax1.fill_between(
        T_1, total_ccluster_1, y2=total_ccluster_1s, color='red', alpha=0.7
    )
    ax1.plot(T_1, total_ccluster_1, '--', c = 'red')
    ax1.plot(T_1, total_ccluster_1s, '--', c = 'red')
    ax1.fill_between(T_1, total_1, y2=total_1s, color='black', alpha=0.7)
    ax1.plot(T_1, total_1, '-', c = 'black')
    ax1.plot(T_1, total_1s, '-', c = 'black')
    ax1.plot(T_1, total_qgp_1, '--', c = 'blue')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize = 16)
    ax1.set_ylabel(r'$\mathrm{s/T^3}$', fontsize = 16)

    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.axis([35., 200., 1e-6, 1e1])

    ax2.plot(T_1, [1.0 for el in T_1], '-', c = 'black')
    ax2.fill_between(T_1, norm_QGP_1, y2=norm_QGP_1s, color='blue', alpha=0.7)
    ax2.plot(T_1, norm_QGP_1, '-', c='blue')
    ax2.plot(T_1, norm_QGP_1s, '-', c='blue')
    ax2.fill_between(T_1, norm_pi_1, y2=norm_pi_1s, color='#653239', alpha=0.7)
    ax2.plot(T_1, norm_pi_1, '-', c='#653239')
    ax2.plot(T_1, norm_pi_1s, '-', c='#653239')
    ax2.fill_between(T_1, norm_rho_1, y2=norm_rho_1s, color='#858AE3', alpha=0.7)
    ax2.plot(T_1, norm_rho_1, '-', c='#858AE3')
    ax2.plot(T_1, norm_rho_1s, '-', c='#858AE3')
    ax2.fill_between(T_1, norm_omega_1, y2=norm_omega_1s, color='#FF37A6', alpha=0.7)
    ax2.plot(T_1, norm_omega_1, '-', c='#FF37A6')
    ax2.plot(T_1, norm_omega_1s, '-', c='#FF37A6')
    ax2.fill_between(T_1, norm_K_1, y2=norm_K_1s, color='red', alpha=0.7)
    ax2.plot(T_1, norm_K_1, '-', c='red')
    ax2.plot(T_1, norm_K_1s, '-', c='red')
    ax2.fill_between(T_1, norm_D_1, y2=norm_D_1s, color='#4CB944', alpha=0.7)
    ax2.plot(T_1, norm_D_1, '--', c='#4CB944')
    ax2.plot(T_1, norm_D_1s, '--', c='#4CB944')
    ax2.fill_between(T_1, norm_N_1, y2=norm_N_1s, color='#DEA54B', alpha=0.7)
    ax2.plot(T_1, norm_N_1, '-', c='#DEA54B')
    ax2.plot(T_1, norm_N_1s, '-', c='#DEA54B')
    ax2.fill_between(T_1, norm_T_1, y2=norm_T_1s, color='#23CE6B', alpha=0.7)
    ax2.plot(T_1, norm_T_1, '-', c='#23CE6B')
    ax2.plot(T_1, norm_T_1s, '-', c='#23CE6B')
    ax2.fill_between(T_1, norm_F_1, y2=norm_F_1s, color='#DB222A', alpha=0.7)
    ax2.plot(T_1, norm_F_1, '--', c='#DB222A')
    ax2.plot(T_1, norm_F_1s, '--', c='#DB222A')
    ax2.fill_between(T_1, norm_P_1, y2=norm_P_1s, color='#78BC61', alpha=0.7)
    ax2.plot(T_1, norm_P_1, '-', c='#78BC61')
    ax2.plot(T_1, norm_P_1s, '-', c='#78BC61')
    ax2.fill_between(T_1, norm_Q_1, y2=norm_Q_1s, color='#55DBCB', alpha=0.7)
    ax2.plot(T_1, norm_Q_1, '--', c='#55DBCB')
    ax2.plot(T_1, norm_Q_1s, '--', c='#55DBCB')
    ax2.fill_between(T_1, norm_H_1, y2=norm_H_1s, color='#A846A0', alpha=0.7)
    ax2.plot(T_1, norm_H_1, '-', c='#A846A0')
    ax2.plot(T_1, norm_H_1s, '-', c='#A846A0')

    ax2.set_yscale('log')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax2.set_xlabel(r'T [MeV]', fontsize=16)
    ax2.set_ylabel(r'$\mathrm{\log~s}$', fontsize=16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


def epja_mhrg_vs_hrg():

    import tqdm
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

    import pnjl.thermo.gcp_pnjl
    import pnjl.thermo.gcp_sea_lattice
    import pnjl.thermo.gcp_perturbative
    import pnjl.thermo.gcp_sigma_lattice
    import pnjl.thermo.gcp_pl_polynomial
    import pnjl.thermo.gcp_cluster.bound_step_continuum_acos_cos \
        as cluster
    import pnjl.thermo.gcp_cluster.bound_step_continuum_step \
        as cluster_s
    import pnjl.thermo.gcp_cluster.hrg \
        as cluster_h

    import pnjl.thermo.solvers.\
        sigma_lattice.\
        sea_lattice.\
        pl_polynomial.\
        pnjl.\
        perturbative.\
        no_clusters \
    as solver_1

    warnings.filterwarnings("ignore")

    calc_1 = False

    files = "D:/EoS/epja/mhrg_vs_hrg/"

    T_1 = numpy.linspace(10.0, 540.0, num=200)

    mu_1 = [0.0/3.0 for el in T_1]

    phi_re_v_1, phi_im_v_1 = \
        list(), list()

    sigma_v_1, gluon_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1 = \
        list(), list(), list(), list(), list()
    perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1 = \
        list(), list(), list()
    perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1 = \
        list(), list(), list(), list()

    pi_v_1, K_v_1, rho_v_1, omega_v_1, D_v_1, N_v_1, T_v_1, F_v_1 = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_1, Q_v_1, H_v_1 = \
        list(), list(), list()

    pi_v_1s, K_v_1s, rho_v_1s, omega_v_1s, D_v_1s, N_v_1s, T_v_1s, F_v_1s = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_1s, Q_v_1s, H_v_1s = \
        list(), list(), list()

    pi_v_1h, K_v_1h, rho_v_1h, omega_v_1h, D_v_1h, N_v_1h, T_v_1h, F_v_1h = \
        list(), list(), list(), list(), list(), list(), list(), list()
    P_v_1h, Q_v_1h, H_v_1h = \
        list(), list(), list()

    if calc_1:

        phi_re_0 = 1e-5
        phi_im_0 = 2e-5
        print("Traced Polyakov loop #1")
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
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1),
            ncols=100
        ):
            sigma_v_1.append(
                pnjl.thermo.gcp_sigma_lattice.pressure(
                    T_el, mu_el
                )/(T_el**4)
            )
        with open(files+"sigma_v_1.pickle", "wb") as file:
            pickle.dump(sigma_v_1, file)

        print("Gluon pressure #1")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1),
            ncols=100
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
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_sea_lattice.pressure(T_el, mu_el, 'l')
            sea_u_v_1.append(lq_temp/(T_el**4))
            sea_d_v_1.append(lq_temp/(T_el**4))
            sea_s_v_1.append(
                pnjl.thermo.gcp_sea_lattice.pressure(
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
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1),
            ncols=100
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
            perturbative_gluon_v_1.append(0.0/(T_el**4))
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
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1), total=len(T_1),
            ncols=100
        ):
            lq_temp = pnjl.thermo.gcp_pnjl.pressure(
                T_el, mu_el, phi_re_el, phi_im_el, 'l'
            )
            sq_temp = pnjl.thermo.gcp_pnjl.pressure(
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

    if calc_1:

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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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
                cluster_s.pressure(
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

    if calc_1:

        print("Pion pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            pi_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'pi'
                )/(T_el**4)
            )
        with open(files+"pi_v_1h.pickle", "wb") as file:
            pickle.dump(pi_v_1h, file)

        print("Kaon pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            K_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'K'
                )/(T_el**4)
            )
        with open(files+"K_v_1h.pickle", "wb") as file:
            pickle.dump(K_v_1h, file)

        print("Rho pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            rho_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'rho'
                )/(T_el**4)
            )
        with open(files+"rho_v_1h.pickle", "wb") as file:
            pickle.dump(rho_v_1h, file)

        print("Omega pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            omega_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'omega'
                )/(T_el**4)
            )
        with open(files+"omega_v_1h.pickle", "wb") as file:
            pickle.dump(omega_v_1h, file)

        print("Diquark pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            D_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'D'
                )/(T_el**4)
            )
        with open(files+"D_v_1h.pickle", "wb") as file:
            pickle.dump(D_v_1h, file)

        print("Nucleon pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            N_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'N'
                )/(T_el**4)
            )
        with open(files+"N_v_1h.pickle", "wb") as file:
            pickle.dump(N_v_1h, file)

        print("Tetraquark pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            T_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'T'
                )/(T_el**4)
            )
        with open(files+"T_v_1h.pickle", "wb") as file:
            pickle.dump(T_v_1h, file)

        print("F-quark pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            F_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'F'
                )/(T_el**4)
            )
        with open(files+"F_v_1h.pickle", "wb") as file:
            pickle.dump(F_v_1h, file)

        print("Pentaquark pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            P_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'P'
                )/(T_el**4)
            )
        with open(files+"P_v_1h.pickle", "wb") as file:
            pickle.dump(P_v_1h, file)

        print("Q-quark pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            Q_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'Q'
                )/(T_el**4)
            )
        with open(files+"Q_v_1h.pickle", "wb") as file:
            pickle.dump(Q_v_1h, file)

        print("Hexaquark pressure #1 (hrg)")
        for T_el, mu_el, phi_re_el, phi_im_el in tqdm.tqdm(
            zip(T_1, mu_1, phi_re_v_1, phi_im_v_1),
            total=len(T_1), ncols=100
        ):
            H_v_1h.append(
                cluster_h.pressure(
                    T_el, mu_el, phi_re_el, phi_im_el, 'H'
                )/(T_el**4)
            )
        with open(files+"H_v_1h.pickle", "wb") as file:
            pickle.dump(H_v_1h, file)
    else:
        with open(files+"pi_v_1h.pickle", "rb") as file:
            pi_v_1h = pickle.load(file)
        with open(files+"K_v_1h.pickle", "rb") as file:
            K_v_1h = pickle.load(file)
        with open(files+"rho_v_1h.pickle", "rb") as file:
            rho_v_1h = pickle.load(file)
        with open(files+"omega_v_1h.pickle", "rb") as file:
            omega_v_1h = pickle.load(file)
        with open(files+"D_v_1h.pickle", "rb") as file:
            D_v_1h = pickle.load(file)
        with open(files+"N_v_1h.pickle", "rb") as file:
            N_v_1h = pickle.load(file)
        with open(files+"T_v_1h.pickle", "rb") as file:
            T_v_1h = pickle.load(file)
        with open(files+"F_v_1h.pickle", "rb") as file:
            F_v_1h = pickle.load(file)
        with open(files+"P_v_1h.pickle", "rb") as file:
            P_v_1h = pickle.load(file)
        with open(files+"Q_v_1h.pickle", "rb") as file:
            Q_v_1h = pickle.load(file)
        with open(files+"H_v_1h.pickle", "rb") as file:
            H_v_1h = pickle.load(file)

    with open(
        "D:/EoS/epja/lattice_data_pickled/bazavov_1407_6387_mu0.pickle",
        "rb"
    ) as file:
        bazavov_1407_6387_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1309_5258_mu0.pickle",
        "rb"
    ) as file:
        borsanyi_1309_5258_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/bazavov_1710_05024_mu0.pickle",
        "rb"
    ) as file:
        bazavov_1710_05024_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu0.pickle",
        "rb"
    ) as file:
        borsanyi_1204_6710v2_mu0 = pickle.load(file)

    qgp_1 = [
        sum(el) for el in 
            zip(
                sigma_v_1,gluon_v_1, sea_u_v_1, sea_d_v_1, sea_s_v_1,
                perturbative_u_v_1, perturbative_d_v_1, perturbative_s_v_1,
                perturbative_gluon_v_1, pnjl_u_v_1, pnjl_d_v_1, pnjl_s_v_1
            )
    ]

    cluster_1 = [
        sum(el) for el in 
            zip(
                pi_v_1, K_v_1, rho_v_1, omega_v_1, D_v_1, N_v_1,
                T_v_1, F_v_1, P_v_1, Q_v_1, H_v_1
            )
    ]

    total_1 = [sum(el) for el in zip(qgp_1, cluster_1)]

    cluster_1s = [
        sum(el) for el in 
            zip(
                pi_v_1s, K_v_1s, rho_v_1s, omega_v_1s, D_v_1s, N_v_1s,
                T_v_1s, F_v_1s, P_v_1s, Q_v_1s, H_v_1s
            )
    ]

    total_1s = [sum(el) for el in zip(qgp_1, cluster_1s)]

    cluster_1h = [
        sum(el) for el in 
            zip(
                pi_v_1h, K_v_1h, rho_v_1h, omega_v_1h, D_v_1h, N_v_1h,
                T_v_1h, F_v_1h, P_v_1h, Q_v_1h, H_v_1h
            )
    ]

    total_1h = [sum(el) for el in zip(qgp_1, cluster_1h)]

    fig1 = matplotlib.pyplot.figure(num=1, figsize=(5.9, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([10., 540., 0., 4.2])

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

    ax1.plot(T_1, total_1, '-', c='black')
    ax1.plot(T_1, total_1s, '--', c='black')
    ax1.plot(T_1, total_1h, '-.', c='black')
    #ax1.fill_between(T_1, total_1, y2=total_1s, color='black', alpha=0.7)
    ax1.plot(T_1, cluster_1, '-', c='red')
    ax1.plot(T_1, cluster_1s, '--', c='red')
    ax1.plot(T_1, cluster_1h, '-.', c='red')
    #ax1.fill_between(T_1, cluster_1, y2=cluster_1s, color='red', alpha=0.7)
    
    ax1.plot(T_1, qgp_1, '--', c='blue')

    ax1.text(180.0, 0.1, r'Clusters', color='red', fontsize=14)
    ax1.text(170.0, 0.58, r'PNJL', color='blue', fontsize=14)
    ax1.text(195.0, 1.16, r'total pressure', color='black', fontsize=14)
    ax1.text(21.0, 3.9, r'$\mathrm{\mu_B=0}$', color='black', fontsize=14)
    ax1.text(
        228.0, 1.8, r'Borsanyi et al. (2012)', color='blue',
        alpha=0.7, fontsize=14
    )
    ax1.text(
        175.0, 3.95, r'Borsanyi et al. (2014)', color='green',
        alpha=0.7, fontsize=14
    )
    ax1.text(
        30.0, 3.4, r'Bazavov et al. (2014)', color='red',
        alpha=0.7, fontsize=14
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


def pnjl_pure():

    import math
    import cmath

    import scipy.optimize
    import scipy.integrate

    N_C = 3.0
    N_F = 2.0

    A0 = 6.75
    A1 = -1.95
    A2 = 2.625
    A3 = -7.44
    T0 = 270.0
    B3 = 0.75
    B4 = 7.5

    GS = 0.00001008
    GV = GS / 2.0
    G_SIGMA = GS
    G_PI = GS
    G_OMEGA = GV

    SIGMA_0 = 325.0
    OMEGA_0 = 0.0

    M_C = 5.0

    def find_Lambda_3DCD() -> float:
        def integrand(p, sigma):
            mass = M_C + sigma
            p2 = p ** 2
            return p2 * mass / math.sqrt(p2 + (mass ** 2))
        def minfunc(Lambda : float) -> float:
            integ, err = scipy.integrate.quad(integrand, 0, Lambda, args=(SIGMA_0,))
            return SIGMA_0 - ((6.0 * G_SIGMA * N_F) / (math.pi ** 2)) * integ
        res = scipy.optimize.root_scalar(minfunc, x0=600.0, x1=700.0, method="secant")
        if res.converged:
            return res.root
        else:
            raise RuntimeError("find_Lamda() failed!") 

    LAMBDA_3DCD = find_Lambda_3DCD()

    def En(p : float, M : float) -> float:
        #
        return math.sqrt((p ** 2) + (M ** 2))

    def b2(T : float) -> float:
        T_norm = T0 / T
        return A0 + A1 * T_norm + A2 * (T_norm ** 2) + A3 * (T_norm ** 3)

    def Traced_PL(phi3 : float, phi8 : float, T : float) -> complex:
        beta = 1.0 / T
        exp_red = cmath.exp(complex(0.0, beta * (phi3 + phi8)))
        exp_green = cmath.exp(complex(0.0, -beta * (phi3 - phi8)))
        exp_blue = cmath.exp(complex(0.0, -2.0 * beta * phi8))
        return (1.0 / N_C) * (exp_red + exp_green + exp_blue)

    def Traced_PL_bar(phi3 : float, phi8 : float, T : float) -> complex:
        beta = 1.0 / T
        exp_red = cmath.exp(complex(0.0, -beta * (phi3 + phi8)))
        exp_green = cmath.exp(complex(0.0, beta * (phi3 - phi8)))
        exp_blue = cmath.exp(complex(0.0, 2.0 * beta * phi8))
        return (1.0 / N_C) * (exp_red + exp_green + exp_blue)

    def PL_potential(phi3 : float, phi8 : float, T : float) -> float:
        Phi = Traced_PL(phi3, phi8, T)
        Phi_re = Phi.real
        Phi_im = Phi.imag
        PhibPhi = (Phi_re ** 2) + (Phi_im ** 2)
        Phi3pPhib3 = 2.0 * (Phi_re ** 3) - 6.0 * Phi_re * (Phi_im ** 2)
        return (T ** 4) * ( (B4 / 4.0) * (PhibPhi ** 2) \
                            - (b2(T) / 2.0) * PhibPhi \
                            - (B3 / 6.0) * Phi3pPhib3 )

    # def f_phi_plus(p : float, )

    def Omega_3dcd_integrand1(p : float, M : float, Mvac : float) -> float:
        #
        return En(p, M) - En(p, Mvac)

    def Omega_3dcd( sigma : float,  omega : float,
                    phi3 : float,   phi8 : float,
                    T : float,      mu : float) -> float:

        M = M_C + sigma
        Mvac = M_C + SIGMA_0
        mu_star = mu + omega

        sigma_term = ((sigma ** 2) - (SIGMA_0 ** 2)) / (4.0 * G_SIGMA)
        omega_term = ((omega ** 2) - (OMEGA_0 ** 2)) / (4.0 * G_OMEGA)
        phi_term = PL_potential(phi3, phi8, T)

        integ1, error1 = scipy.integrate.quad(
            Omega_3dcd_integrand1,
            0.0, LAMBDA_3DCD, args=(M, Mvac,)
        )
        # return sigma_term - omega_term + phi_term - (6.0 * N_F / (2.0 * (math.pi ** 2))) * (integ1 + integ2)


def lattice_thermo():
    
    import math
    import numpy
    import pickle
    import platform

    import scipy.interpolate
    import matplotlib.pyplot
    if platform.system() == "Linux":
        matplotlib.use("TkAgg")
    import matplotlib.patches

    lattice_files = "C:/Users/matci/Desktop/lattice_thermo/"
    if platform.system() == "Linux":
        lattice_files = "/home/mcierniak/Data/2023_epja/lattice_data_raw/"

    x_p_l, y_p_l = list(), list()
    x_p_h, y_p_h = list(), list()
    x_e_l, y_e_l = list(), list()
    x_e_h, y_e_h = list(), list()
    x_s_l, y_s_l = list(), list()
    x_s_h, y_s_h = list(), list()
    
    T = numpy.linspace(131.0, 279.0, num=200)
    mu = [el*2.5 for el in T]

    with open(lattice_files+"2212_09043_mub_T_2p5_p_low.dat", 'r') as file:
        for line in file:
            temp = [float(el) for el in line.split()]
            x_p_l.append(temp[0])
            y_p_l.append(temp[1])
    with open(lattice_files+"2212_09043_mub_T_2p5_p_high.dat", 'r') as file:
        for line in file:
            temp = [float(el) for el in line.split()]
            x_p_h.append(temp[0])
            y_p_h.append(temp[1])
    with open(lattice_files+"2212_09043_mub_T_2p5_e_low.dat", 'r') as file:
        for line in file:
            temp = [float(el) for el in line.split()]
            x_e_l.append(temp[0])
            y_e_l.append(temp[1])
    with open(lattice_files+"2212_09043_mub_T_2p5_e_high.dat", 'r') as file:
        for line in file:
            temp = [float(el) for el in line.split()]
            x_e_h.append(temp[0])
            y_e_h.append(temp[1])
    with open(lattice_files+"2212_09043_mub_T_2p5_s_low.dat", 'r') as file:
        for line in file:
            temp = [float(el) for el in line.split()]
            x_s_l.append(temp[0])
            y_s_l.append(temp[1])
    with open(lattice_files+"2212_09043_mub_T_2p5_s_high.dat", 'r') as file:
        for line in file:
            temp = [float(el) for el in line.split()]
            x_s_h.append(temp[0])
            y_s_h.append(temp[1])

    i_p_l = scipy.interpolate.interp1d(x_p_l, y_p_l)
    i_p_h = scipy.interpolate.interp1d(x_p_h, y_p_h)
    i_e_l = scipy.interpolate.interp1d(x_e_l, y_e_l)
    i_e_h = scipy.interpolate.interp1d(x_e_h, y_e_h)
    i_s_l = scipy.interpolate.interp1d(x_s_l, y_s_l)
    i_s_h = scipy.interpolate.interp1d(x_s_h, y_s_h)

    p_poly = [[x_el, y_el] for x_el, y_el in zip(T, i_p_l(T))]
    e_poly = [[x_el, y_el] for x_el, y_el in zip(T, i_e_l(T))]
    s_poly = [[x_el, y_el] for x_el, y_el in zip(T, i_s_l(T))]
    for x_el, y_el in zip(T[::-1], i_p_h(T)[::-1]):
        p_poly.append([x_el, y_el])
    for x_el, y_el in zip(T[::-1], i_e_h(T)[::-1]):
        e_poly.append([x_el, y_el])
    for x_el, y_el in zip(T[::-1], i_s_h(T)[::-1]):
        s_poly.append([x_el, y_el])
    p_avg = [(low+high)/2.0 for low, high in zip(i_p_l(T), i_p_h(T))]
    p_delta = [(high-low)/2.0 for low, high in zip(i_p_l(T), i_p_h(T))]
    e_avg = [(low+high)/2.0 for low, high in zip(i_e_l(T), i_e_h(T))]
    e_delta = [(high-low)/2.0 for low, high in zip(i_e_l(T), i_e_h(T))]
    s_avg = [(low+high)/2.0 for low, high in zip(i_s_l(T), i_s_h(T))]
    s_delta = [(high-low)/2.0 for low, high in zip(i_s_l(T), i_s_h(T))]
    n_avg = [(p_el+e_el-s_el)/2.5 for p_el, e_el, s_el in zip(p_avg, e_avg, s_avg)]
    n_delta = [(p_el+e_el+s_el)/2.5 for p_el, e_el, s_el in zip(p_delta, e_delta, s_delta)]

    n_h = [avg_el+delta_el for avg_el, delta_el in zip(n_avg, n_delta)]
    n_l = [avg_el-delta_el if avg_el-delta_el >= 0.0 else 0.0 for avg_el, delta_el in zip(n_avg, n_delta)]
    n_poly = [[x_el, y_el] for x_el, y_el in zip(T, n_l)]
    for x_el, y_el in zip(T[::-1], n_h[::-1]):
        n_poly.append([x_el, y_el])
    with open(lattice_files+"2212_09043_mub_T_2p5_nT3.pickle", "wb") as file:
        pickle.dump(n_poly, file)
    with open(lattice_files+"2212_09043_mub_T_2p5_sT3.pickle", "wb") as file:
        pickle.dump(s_poly, file)

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (6.0, 5.0))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.axis([130., 280., 0.0, 17.0])

    ax1.add_patch(
        matplotlib.patches.Polygon(
            p_poly, closed = True, fill = True, color = "blue", alpha = 0.3
        )
    )
    ax1.add_patch(
        matplotlib.patches.Polygon(
            e_poly, closed = True, fill = True, color = "green", alpha = 0.3
        )
    )
    ax1.add_patch(
        matplotlib.patches.Polygon(
            s_poly, closed = True, fill = True, color = "red", alpha = 0.3
        )
    )
    ax1.plot(T, p_avg, '--', c="blue")
    ax1.plot(T, e_avg, '--', c="green")
    ax1.plot(T, s_avg, '--', c="red")
    ax1.plot(T, n_avg, '--', c="cyan")
    ax1.add_patch(
        matplotlib.patches.Polygon(
            n_poly, closed = True, fill = True, color = "cyan", alpha = 0.3
        )
    )

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    ax1.set_xlabel(r'T [MeV]', fontsize=16)
    ax1.set_ylabel(r'p/$\mathrm{\epsilon}$/s', fontsize=16)

    fig1.tight_layout(pad = 0.1)

    matplotlib.pyplot.show()
    matplotlib.pyplot.close()


if __name__ == '__main__':

    epja_figure9_alt()

    print("END")