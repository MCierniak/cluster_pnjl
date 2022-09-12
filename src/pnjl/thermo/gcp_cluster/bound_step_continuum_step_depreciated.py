"""### Description
Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential and 
associated functions based on https://arxiv.org/pdf/2012.12894.pdf .

### Functions
gcp_real
    Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential 
    (real part).
gcp_imag
    Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential 
    (imaginary part).
pressure
    Generalized Beth-Uhlenbeck cluster pressure.
bdensity
    Generalized Beth-Uhlenbeck cluster baryon density.
qnumber_cumulant
    Generalized Beth-Uhlenbeck cluster quark number cumulant chi_q. Based on 
    Eq.29 of https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline 
    definition.
sdensity
    Generalized Beth-Uhlenbeck cluster entropy density.
Polyakov_loop
"""


import math
import typing
import functools

import scipy.optimize
import scipy.integrate

import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.gcp_sigma_lattice
import pnjl.thermo.gcp_pl_polynomial


@functools.lru_cache(maxsize=1024)
def gcp_boson_singlet_inner_integrand(
    M: float, p: float, T: float, mu: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, a, '+')
    fm = pnjl.thermo.distributions.f_boson_singlet(p, T, mu, M, a, '-')

    return (M/En)*math.fsum([fp, fm])


@functools.lru_cache(maxsize=1024)
def gcp_fermion_singlet_inner_integrand(
    M: float, p: float, T: float, mu: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, a, '+')
    fm = pnjl.thermo.distributions.f_fermion_singlet(p, T, mu, M, a, '-')

    return (M/En)*math.fsum([fp, fm])


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_inner_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, a, '+'
    ).real
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, a, '-'
    ).real

    return (M/En)*math.fsum([fp, fm])


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_inner_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, a, '+'
    ).real
    fm = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, a, '-'
    ).real

    return (M/En)*math.fsum([fp, fm])


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_inner_integrand_imag(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, a, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, a, '-'
    ).imag

    return (M/En)*math.fsum([fp, fm])


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_inner_integrand_imag(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_fermion_antitriplet(
        p, T, mu, phi_re, phi_im, M, a, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_fermion_triplet(
        p, T, mu, phi_re, phi_im, M, a, '-'
    ).imag

    return (M/En)*math.fsum([fp, fm])


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_inner_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, a, '+'
    ).real
    fm = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, a, '-'
    ).real

    return (M/En)*math.fsum([fp, fm])


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_inner_integrand_real(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, a, '+'
    ).real
    fm = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, a, '-'
    ).real

    return (M/En)*math.fsum([fp, fm])


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_inner_integrand_imag(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, a, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, a, '-'
    ).imag

    return (M/En)*math.fsum([fp, fm])


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_inner_integrand_imag(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int
) -> float:

    En = pnjl.thermo.distributions.En(p, M)
    fp = pnjl.thermo.distributions.f_boson_antitriplet(
        p, T, mu, phi_re, phi_im, M, a, '+'
    ).imag
    fm = pnjl.thermo.distributions.f_boson_triplet(
        p, T, mu, phi_re, phi_im, M, a, '-'
    ).imag

    return (M/En)*math.fsum([fp, fm])


@functools.lru_cache(maxsize=1024)
def gcp_boson_singlet_integrand(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_singlet_inner_integrand, M_I, M_TH_I, args = (p, T, mu, a)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_singlet_integrand(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_singlet_inner_integrand, M_I, M_TH_I, args = (p, T, mu, a)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_integrand_real(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_triplet_inner_integrand_real, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_integrand_real(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_antitriplet_inner_integrand_real, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_integrand_real(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_triplet_inner_integrand_real, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_integrand_real(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_antitriplet_inner_integrand_real, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_triplet_integrand_imag(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_triplet_inner_integrand_imag, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_boson_antitriplet_integrand_imag(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_boson_antitriplet_inner_integrand_imag, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_triplet_integrand_imag(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_triplet_inner_integrand_imag, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


@functools.lru_cache(maxsize=1024)
def gcp_fermion_antitriplet_integrand_imag(
    p: float, T: float, mu: float, phi_re: float, phi_im: float,
    M_I: float, M_TH_I: float, a: int
) -> float:

    integral, error = scipy.integrate.quad(
        gcp_fermion_antitriplet_inner_integrand_imag, M_I, M_TH_I,
        args = (p, T, mu, phi_re, phi_im, a)
    )

    return (p**2)*integral


gcp_real_hash = {
    'pi': gcp_boson_singlet_integrand,
    'K': gcp_boson_singlet_integrand,
    'rho': gcp_boson_singlet_integrand,
    'omega': gcp_boson_singlet_integrand,
    'D': gcp_boson_antitriplet_integrand_real,
    'N': gcp_fermion_singlet_integrand,
    'T': gcp_boson_singlet_integrand,
    'F': gcp_boson_triplet_integrand_real,
    'P': gcp_fermion_singlet_integrand,
    'Q': gcp_fermion_antitriplet_integrand_real,
    'H': gcp_boson_singlet_integrand
}


gcp_imag_hash = {
    'D': gcp_boson_antitriplet_integrand_imag,
    'F': gcp_boson_triplet_integrand_imag,
    'Q': gcp_fermion_antitriplet_integrand_imag,
}


@functools.lru_cache(maxsize=1024)
def gcp_real(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential 
    (real part).
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    gcp_real : float
        Value of the thermodynamic potential in MeV^4.
    """

    M_I = pnjl.defaults.MI[cluster]
    N_I = pnjl.defaults.NI[cluster]
    S_I = pnjl.defaults.S[cluster]
    M_th_i = math.fsum([
        math.fsum([N_I,-S_I])*pnjl.thermo.gcp_sigma_lattice.Ml(T, mu),
        S_I*pnjl.thermo.gcp_sigma_lattice.Ms(T, mu)
    ])
    D_I = pnjl.defaults.DI[cluster]
    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    integral = 0.0
    if M_th_i > M_I:
        integral, error = scipy.integrate.quad(
            gcp_real_hash[cluster], 0.0, math.inf,
            args = (T, mu, phi_re, phi_im, M_I, M_th_i, A_I)
        )

    return -((D_I*3.0)/(2.0*(math.pi**2)))*integral


@functools.lru_cache(maxsize=1024)
def gcp_imag(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster grandcanonical thermodynamic potential 
    (imaginary part).
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    gcp_imag : float
        Value of the thermodynamic potential in MeV^4.
    """

    M_I = pnjl.defaults.MI[cluster]
    N_I = pnjl.defaults.NI[cluster]
    S_I = pnjl.defaults.S[cluster]
    M_th_i = math.fsum([
        math.fsum([N_I,-S_I])*pnjl.thermo.gcp_sigma_lattice.Ml(T, mu),
        S_I*pnjl.thermo.gcp_sigma_lattice.Ms(T, mu)
    ])
    D_I = pnjl.defaults.DI[cluster]
    A_I = math.fsum([
        pnjl.defaults.NET_QL[cluster],
        pnjl.defaults.NET_QS[cluster]
    ])

    integral = 0.0
    if M_th_i > M_I and cluster in gcp_imag_hash:
        integral, error = scipy.integrate.quad(
            gcp_imag_hash[cluster], 0.0, math.inf,
            args = (T, mu, phi_re, phi_im, M_I, M_th_i, A_I)
        )

    return -((D_I*3.0)/(2.0*(math.pi**2)))*integral


@functools.lru_cache(maxsize=1024)
def pressure(
    T: float, mu: float, phi_re: float, phi_im: float, cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster pressure.
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    pressure : float
        Value of the thermodynamic pressure in MeV^4.
    """

    return -gcp_real(T, mu, phi_re, phi_im, cluster)


@functools.lru_cache(maxsize=1024)
def bdensity(
    T: float, mu: float, phi_re: float, phi_im: float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster baryon density.
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    phi_solver : Callable
        Function calculating the traced Polyakov-loop for given
        T and mu. Must be of the form
            (T: float, mu: float,
            phi_re0: float, phi_im0: float) -> Tuple[float, float],
            where
                T : temperature in MeV
                mu : quark chemical potential in MeV
                phi_re0 : initial guess for phi_re
                phi_im0 : initial guess for phi_im
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    bdensity : float
        Value of the thermodynamic baryon density in MeV^3.
    """

    h = 1e-2

    if math.fsum([mu, -2*h]) > 0.0:

        mu_vec = [
            math.fsum([mu, 2*h]), math.fsum([mu, h]),
            math.fsum([mu, -h]), math.fsum([mu, -2*h])
        ]
        deriv_coef = [
            -1.0/(12.0*h), 8.0/(12.0*h),
            -8.0/(12.0*h), 1.0/(12.0*h)
        ]
        phi_vec = []
        if pnjl.defaults.D_PHI_D_MU_0:
            phi_vec = [
                tuple([phi_re, phi_im])
                for _ in mu_vec
            ]
        else:
            phi_vec = [
                phi_solver(T, mu_el, phi_re, phi_im)
                for mu_el in mu_vec
            ]

        p_vec = [
            coef*pressure(T, mu_el, phi_el[0], phi_el[1], cluster)/3.0
            for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
        ]

        return math.fsum(p_vec)

    else:

        new_mu = math.fsum([mu, h])
        new_phi_re, new_phi_im = phi_re, phi_im
            
        if not pnjl.defaults.D_PHI_D_MU_0:
            new_phi_re, new_phi_im = phi_solver(T, new_mu, phi_re, phi_im)

        return bdensity(
            T, new_mu, new_phi_re, new_phi_im, 
            phi_solver, cluster
        )


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant(
    rank: int, T: float, mu: float, phi_re: float, phi_im: float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster quark number cumulant chi_q. Based on 
    Eq.29 of https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline 
    definition.
    
    ### Prameters
    rank : int
        Cumulant rank. Rank 1 equals to 3 times the baryon density.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    phi_solver : Callable
        Function calculating the traced Polyakov-loop for given
        T and mu. Must be of the form
            (T: float, mu: float,
            phi_re0: float, phi_im0: float) -> Tuple[float, float],
            where
                T : temperature in MeV
                mu : quark chemical potential in MeV
                phi_re0 : initial guess for phi_re
                phi_im0 : initial guess for phi_im
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    qnumber_cumulant : float
        Value of the thermodynamic quark number cumulant in MeV^3.
    """

    if rank == 1:

        return 3.0 * bdensity(T, mu, phi_re, phi_im,  phi_solver, cluster)

    else:

        h = 1e-2

        if math.fsum([mu, -2*h]) > 0.0:

            mu_vec = [
                math.fsum(mu, 2*h), math.fsum(mu, h),
                math.fsum(mu, -h), math.fsum(mu, -2*h)
            ]
            deriv_coef = [
                -1.0/(12.0*h), 8.0/(12.0*h),
                -8.0/(12.0*h), 1.0/(12.0*h)
            ]
            phi_vec = []
            if pnjl.defaults.D_PHI_D_MU_0:
                phi_vec = [
                    tuple([phi_re, phi_im])
                    for _ in mu_vec
                ]
            else:
                phi_vec = [
                    phi_solver(T, mu_el, phi_re, phi_im)
                    for mu_el in mu_vec
                ]

            out_vec = [
                coef*qnumber_cumulant(
                    rank-1, T, mu_el, phi_el[0], phi_el[1], 
                    phi_solver, cluster)
                for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
            ]

            return math.fsum(out_vec)

        else:

            new_mu = math.fsum([mu, h])
            new_phi_re, new_phi_im = phi_re, phi_im
            
            if not pnjl.defaults.D_PHI_D_MU_0:
                new_phi_re, new_phi_im = phi_solver(T, new_mu, phi_re, phi_im)

            return qnumber_cumulant(
                rank, T, new_mu, new_phi_re, new_phi_im, 
                phi_solver, cluster
            )


@functools.lru_cache(maxsize=1024)
def sdensity(
    T: float, mu: float, phi_re : float, phi_im : float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    cluster: str
) -> float:
    """### Description
    Generalized Beth-Uhlenbeck cluster entropy density.
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    phi_solver : Callable
        Function calculating the traced Polyakov-loop for given
        T and mu. Must be of the form
            (T: float, mu: float,
            phi_re0: float, phi_im0: float) -> Tuple[float, float],
            where
                T : temperature in MeV
                mu : quark chemical potential in MeV
                phi_re0 : initial guess for phi_re
                phi_im0 : initial guess for phi_im
    cluster : str
        Cluster type,
            'pi' : pion
            'K' : kaon
            'rho': rho meson
            'omega': omega meson
            'D': diquark
            'N': nucleon
            'T': tetraquark
            'F': four-quark
            'P': pentaquark
            'Q': five-quark
            'H': hexaquark
    
    ### Returns
    sdensity : float
        Value of the thermodynamic entropy density in MeV^3.
    """

    h = 1e-2

    if math.fsum([T, -2*h]) > 0.0:

        T_vec = [
            math.fsum(T, 2*h), math.fsum(T, h),
            math.fsum(T, -h), math.fsum(T, -2*h)
        ]
        deriv_coef = [
            -1.0/(12.0*h), 8.0/(12.0*h),
            -8.0/(12.0*h), 1.0/(12.0*h)
        ]
        phi_vec = []
        if pnjl.defaults.D_PHI_D_T_0:
            phi_vec = [
                tuple([phi_re, phi_im])
                for _ in T_vec
            ]
        else:
            phi_vec = [
                phi_solver(T_el, mu, phi_re, phi_im)
                for T_el in T_vec
            ]

        p_vec = [
            coef*pressure(T_el, mu, phi_el[0], phi_el[1], cluster)
            for T_el, coef, phi_el in zip(T_vec, deriv_coef, phi_vec)
        ]

        return math.fsum(p_vec)

    else:

        new_T = math.fsum([T, h])
        new_phi_re, new_phi_im = phi_re, phi_im
            
        if not pnjl.defaults.D_PHI_D_T_0:
            new_phi_re, new_phi_im = phi_solver(new_T, mu, phi_re, phi_im)

        return sdensity(
            new_T, mu, new_phi_re, new_phi_im, 
            phi_solver, cluster
        )