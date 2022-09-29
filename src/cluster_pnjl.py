

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
    #ax1.axis([10., 2000., -1.5, 5.1])
    ax1.axis([50., 300., -1.5, 3.5])

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

    ax1.plot(T, total_1, '-', c='black')
    ax1.plot(T, gluon_v_1, '--', c='red')
    ax1.plot(T, pertrubative_total, '--', c='pink')
    ax1.plot(T, pnjl_total, '--', c='blue')

    ax1.text(
        960.0, 1.15,
        r'Polyakov--loop potential',
        color='red', fontsize=14
    )
    ax1.text(
        960.0, -0.5,
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

    files = "D:/EoS/epja/figure7/"

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
        "D:/EoS/epja/lattice_data_pickled/bazavov_1407_6387_mu0.pickle", "rb"
    ) as file:
        bazavov_1407_6387_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1309_5258_mu0.pickle", "rb"
    ) as file:
        borsanyi_1309_5258_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/bazavov_1710_05024_mu0.pickle", "rb"
    ) as file:
        bazavov_1710_05024_mu0 = pickle.load(file)
    with open(
        "D:/EoS/epja/lattice_data_pickled/borsanyi_1204_6710v2_mu0.pickle", "rb"
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

    fig1 = matplotlib.pyplot.figure(num = 1, figsize = (12.0, 10))
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
        180.0, 0.1, r'Color triplet/antitriplet', color='red', fontsize=14
    )
    ax1.text(
        165.0, 0.34, r'Color singlet', color='green', fontsize=14
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
        180.0, 0.1, r'Color triplet/antitriplet', color='red', fontsize=14
    )
    ax3.text(
        165.0, 0.34, r'Color singlet', color='green', fontsize=14
    )
    ax3.text(
        170.0, 0.58, r'PNJL', color='blue', fontsize=14
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
        225., 1.5, r'PNJL', color='blue', fontsize=14
    )

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
        180.0, 0.1, r'Color triplet/antitriplet', color='red', fontsize=14
    )
    ax1.text(
        165.0, 0.34, r'Color singlet', color='green', fontsize=14
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
        180.0, 0.1, r'Color triplet/antitriplet', color='red', fontsize=14
    )
    ax3.text(
        165.0, 0.34, r'Color singlet', color='green', fontsize=14
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
        225., 1.5, r'PNJL', color='blue', fontsize=14
    )

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


def epja_figure9():

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

    files = "D:/EoS/epja/figure9/"

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
        173.3, 0.005, r'Color triplet/antitriplet', color='red', fontsize=14
    )
    ax1.text(
        165.0, 0.03, r'Color singlet', color='green', fontsize=14
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
        165.0, 0.03, r'Color triplet/antitriplet', color='red', fontsize=14
    )
    ax3.text(
        173.3, 0.005, r'Color singlet', color='green', fontsize=14
    )
    ax3.text(
        125.0, 0.065, r'PNJL', color='blue', fontsize=14
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
    ax4.text(183., 1.5, r'PNJL', color='blue', fontsize=14)

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


def epja_figure10():

    import tqdm
    import math
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

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
    calc_2 = False

    files = "D:/EoS/epja/figure10/"

    T = numpy.linspace(1.0, 500.0, num=200)

    for mu in numpy.linspace(1.0, 200.0, num=200):

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
        total_1 = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_1, total_b_1)]
        total_2 = [el1/el2 if el2 != 0.0 else 0.0 for el1, el2 in zip(total_s_2, total_b_2)]

        fig = matplotlib.pyplot.figure(num=1, figsize=(5.9, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis([1.0, 500.0, -1000.0, 10000.0])

        ax.plot(T, total_1, '-', c="blue")
        ax.plot(T, total_2, '-', c="red")

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        ax.set_xlabel(r'T [MeV]', fontsize = 16)
        ax.set_ylabel(r's/n', fontsize = 16)

        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        matplotlib.pyplot.close()


def epja_figure11():

    import tqdm
    import math
    import numpy
    import pickle
    import warnings

    import matplotlib.pyplot

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

    ax1.fill_between(T_Tc_1, R12_calc_1, y2=R12_calc_2, color='red', alpha=0.7)
    ax1.plot(T_Tc_1, R12_calc_1, '-', c='red')
    ax1.plot(T_Tc_2, R12_calc_2, '-', c='red')
    ax1.plot(T_HRG, R12_HRG_1, '--', c='red')
    ax1.plot(T_QGP, R12_QGP_1, '--', c='red')
    #ax1.plot(T_Tc_5, R12_calc_5, '-.', c='red')
    ax1.text(0.705, 0.63, r"HRG", color='black', fontsize=14)
    ax1.text(1.69, 1.0, r"QGP", color='black', fontsize=14)
    ax1.text(1.54, 0.04, r"$\mathrm{\mu_B/T=1.2}$", color='black', fontsize=14)

    ax1.text(
        0.8, 0.49, r'Alton et al. (2005)', color='red', fontsize=14
    )

    ax2.fill_between(T_Tc_3, R12_calc_3, y2=R12_calc_4, color='blue', alpha=0.7)
    ax2.plot(T_Tc_3, R12_calc_3, '-', c='blue')
    ax2.plot(T_Tc_4, R12_calc_4, '-', c='blue')
    ax2.plot(T_HRG, R12_HRG_2, '--', c='blue')
    ax2.plot(T_QGP, R12_QGP_2, '--', c='blue')
    #ax2.plot(T_Tc_6, R12_calc_6, '-.', c='blue')
    ax2.text(0.705, 0.35, r"HRG", color='black', fontsize=14)
    ax2.text(1.69, 0.91, r"QGP", color='black', fontsize=14)
    ax2.text(1.54, 0.04, r"$\mathrm{\mu_B/T=2.4}$", color='black', fontsize=14)

    ax2.text(
        0.86, 0.2, r'Alton et al. (2005)', color='blue', fontsize=14
    )

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


def epja_figure12():

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

    files = "D:/EoS/epja/figure12/"

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

if __name__ == '__main__':

    epja_figure10()

    print("END")