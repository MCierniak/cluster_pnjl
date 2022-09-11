"""### Description
Pertrubative correction of the PNJL grandcanonical potential.
Finite-mu extension of formulas in https://arxiv.org/pdf/2012.12894.pdf .
Based on https://inspirehep.net/files/901191eb2f4d03c023787042343325d2 .

### Functions
alpha_s
    QCD running coupling.
I_fermion
    Integral of the fermionic correction to the PNJL thermodynamic potential.
I_boson
    Integral of the bosonic correction to the PNJL thermodynamic potential.
gcp_fermion_l_real
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single light quark flavor (real part).
gcp_fermion_l_imag
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single light quark flavor (imaginary part).
gcp_fermion_s_real
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (real part).
gcp_fermion_s_imag
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (imaginary part).
gcp_boson_real
    Perturbative bosonic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (real part).
gcp_boson_imag
    Perturbative bosonic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (imag part).
pressure
    Pertrubative correction to the pressure of a single quark flavor.
bdensity
    Pertrubative correction to the baryon density of a single quark flavor.
qnumber_cumulant
    Pertrubative correction to the quark number cumulant chi_q of a single 
    quark flavor. Based on Eq.29 of https://arxiv.org/pdf/2012.12894.pdf and 
    the subsequent inline definition.
sdensity
    Pertrubative correction to the entropy density of a single quark flavor.
"""


import math
import typing
import functools

import scipy.integrate

import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sigma_lattice


@functools.lru_cache(maxsize=1024)
def alpha_s(T : float, mu : float) -> float:
    """### Description
    QCD running coupling.

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ### Returns
    alpha_s : float
        Value of the running coupling.
    """

    NF = 3.0
    NC = pnjl.defaults.NC
    C = pnjl.defaults.C
    D = pnjl.defaults.D

    den1 = math.fsum([11.0 * NC, -2.0*NF])
    den2 = math.fsum([
        2.0*math.log(D),
        2.0*math.log(T),
        -2.0*math.log(C)
    ])
    den3 = math.fsum([((D*T)**2), -(C**2)])

    par = math.fsum([1.0/den2, -(C**2)/den3])

    return ((12.0*math.pi)/den1)*par


fermion_hash = {
    '+' : pnjl.thermo.distributions.f_fermion_triplet,
    '-' : pnjl.thermo.distributions.f_fermion_antitriplet
}


boson_hash = {
    '+' : pnjl.thermo.distributions.f_boson_triplet,
    '-' : pnjl.thermo.distributions.f_boson_antitriplet
}


@functools.lru_cache(maxsize=1024)
def I_fermion_integrand_real(
    p: float, T: float, mu: float, mass: float,
    phi_re: float, phi_im: float, typ: str
) -> float:

    distr = fermion_hash[typ](
        p, T, mu, phi_re, phi_im, mass, 1, typ
    ).real

    return p * distr


@functools.lru_cache(maxsize=1024)
def I_fermion_integrand_imag(
    p: float, T: float, mu: float, mass: float,
    phi_re: float, phi_im: float, typ: str
) -> float:

    distr = fermion_hash[typ](
        p, T, mu, phi_re, phi_im, mass, 1, typ
    ).imag

    return p * distr


@functools.lru_cache(maxsize=1024)
def I_boson_integrand_real(
    p: float, T: float, mu: float, mass: float,
    phi_re: float, phi_im: float, typ: str
) -> float:

    distr = boson_hash[typ](
        p, T, mu, phi_re, phi_im, mass, 1, typ
    ).real

    return p * distr


@functools.lru_cache(maxsize=1024)
def I_boson_integrand_imag(
    p: float, T: float, mu: float, mass: float,
    phi_re: float, phi_im: float, typ: str
) -> float:

    distr = boson_hash[typ](
        p, T, mu, phi_re, phi_im, mass, 1, typ
    ).imag

    return p * distr


@functools.lru_cache(maxsize=1024)
def I_fermion(
    T: float, mu: float, mass: float,
    phi_re: float, phi_im: float, en_typ: str,
) -> complex:
    """### Description
    Integral of the fermionic correction to the PNJL thermodynamic potential.
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    en_typ : str
        Type of particle:
            '+' : positive energy particle
            '-' : negative energy antiparticle
    
    ### Returns
    I_fermion : complex
        Value of the integral.
    """

    integral_real, error_real = scipy.integrate.quad(
        I_fermion_integrand_real, 0.0, math.inf,
        args = (T, mu, mass, phi_re, phi_im, en_typ)
    )
    integral_imag, error_imag = scipy.integrate.quad(
        I_fermion_integrand_imag, 0.0, math.inf,
        args = (T, mu, mass, phi_re, phi_im, en_typ)
    )

    return complex(integral_real, integral_imag)/(T**2)


@functools.lru_cache(maxsize=1024)
def I_boson(
    T: float, mu: float, mass: float,
    phi_re: float, phi_im: float, en_typ: str,
) -> complex:
    """### Description
    Integral of the bosonic correction to the PNJL thermodynamic potential.
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    en_typ : str
        Type of particle:
            '+' : positive energy particle
            '-' : negative energy antiparticle
    
    ### Returns
    I_fermion : complex
        Value of the integral.
    """

    integral_real, error_real = scipy.integrate.quad(
        I_fermion_integrand_real, 0.0, math.inf,
        args = (T, mu, mass, phi_re, phi_im, en_typ)
    )
    integral_imag, error_imag = scipy.integrate.quad(
        I_fermion_integrand_imag, 0.0, math.inf,
        args = (T, mu, mass, phi_re, phi_im, en_typ)
    )

    return complex(integral_real, integral_imag)/(T**2)


@functools.lru_cache(maxsize=1024)
def gcp_fermion_l_real(
    T: float, mu: float, phi_re: float, phi_im: float
) -> float:
    """### Description
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single light quark flavor (real part).

    ### Parameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.

    ### Returns
    gcp_fermion_l_real, float
        Value of the thermodynamic potential in MeV^4.
    """

    mass = pnjl.thermo.gcp_sigma_lattice.Ml(T, mu)

    Ip = I_fermion(T, mu, mass, phi_re, phi_im, '+')
    Im = I_fermion(T, mu, mass, phi_re, phi_im, '-')
    alpha = alpha_s(T, mu)

    IpIm = complex(math.fsum([Ip.real, Im.real]), math.fsum([Ip.imag, Im.imag]))
    IpIm2 = IpIm**2

    par_real = math.fsum([IpIm.real, 1.0/(2.0*(math.pi**2))*IpIm2.real])

    return (4.0/(3.0*math.pi))*alpha*(T**4)*par_real


@functools.lru_cache(maxsize=1024)
def gcp_fermion_l_imag(
    T: float, mu: float, phi_re: float, phi_im: float
) -> float:
    """### Description
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single light quark flavor (imaginary part).
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    
    ### Returns
    gcp_fermion_l_imag : float
        Value of the thermodynamic potential in MeV^4.
    """

    mass = pnjl.thermo.gcp_sigma_lattice.Ml(T, mu)

    Ip = I_fermion(T, mu, mass, phi_re, phi_im, '+')
    Im = I_fermion(T, mu, mass, phi_re, phi_im, '-')
    alpha = alpha_s(T, mu)

    IpIm = complex(math.fsum([Ip.real, Im.real]), math.fsum([Ip.imag, Im.imag]))
    IpIm2 = IpIm**2

    par_imag = math.fsum([IpIm.imag, 1.0/(2.0*(math.pi**2))*IpIm2.imag])

    return (4.0/(3.0*math.pi))*alpha*(T**4)*par_imag


@functools.lru_cache(maxsize=1024)
def gcp_fermion_s_real(
    T: float, mu: float, phi_re: float, phi_im: float
) -> float:
    """### Description
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (real part).
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    
    ### Returns
    gcp_fermion_s_real : float
        Value of the thermodynamic potential in MeV^4.
    """

    mass = pnjl.thermo.gcp_sigma_lattice.Ms(T, mu)

    Ip = I_fermion(T, mu, mass, phi_re, phi_im, '+')
    Im = I_fermion(T, mu, mass, phi_re, phi_im, '-')
    alpha = alpha_s(T, mu)

    IpIm = complex(math.fsum([Ip.real, Im.real]), math.fsum([Ip.imag, Im.imag]))
    IpIm2 = IpIm**2

    par_real = math.fsum([IpIm.real, 1.0/(2.0*(math.pi**2))*IpIm2.real])

    return (4.0/(3.0*math.pi))*alpha*(T**4)*par_real


@functools.lru_cache(maxsize=1024)
def gcp_fermion_s_imag(
    T: float, mu: float, phi_re: float, phi_im: float
) -> float:
    """### Description
    Perturbative fermionic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (imaginary part).
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    
    ### Returns
    gcp_fermion_s_imag : float
        Value of the thermodynamic potential in MeV^4.
    """

    mass = pnjl.thermo.gcp_sigma_lattice.Ms(T, mu)

    Ip = I_fermion(T, mu, mass, phi_re, phi_im, '+')
    Im = I_fermion(T, mu, mass, phi_re, phi_im, '-')
    alpha = alpha_s(T, mu)

    IpIm = complex(math.fsum([Ip.real, Im.real]), math.fsum([Ip.imag, Im.imag]))
    IpIm2 = IpIm**2

    par_imag = math.fsum([IpIm.imag, 1.0/(2.0*(math.pi**2))*IpIm2.imag])

    return (4.0/(3.0*math.pi))*alpha*(T**4)*par_imag


@functools.lru_cache(maxsize=1024)
def gcp_boson_real(
    T: float, mu: float, phi_re: float, phi_im: float
) -> float:
    """### Description
    Perturbative bosonic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (real part).
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    
    ### Returns
    gcp_boson_real : float
        Value of the thermodynamic potential in MeV^4.
    """

    Ip = I_boson(T, 0.0, 0.0, phi_re, phi_im, '+')
    Im = I_boson(T, 0.0, 0.0, phi_re, phi_im, '-')
    alpha = alpha_s(T, mu)

    IpIm = complex(math.fsum([Ip.real, Im.real]), math.fsum([Ip.imag, Im.imag]))
    IpIm2 = IpIm**2

    return (9.0/(math.pi**3))*alpha*(T**4)*IpIm2.real


@functools.lru_cache(maxsize=1024)
def gcp_boson_imag(
    T: float, mu: float, phi_re: float, phi_im: float
) -> float:
    """### Description
    Perturbative bosonic correction to the grandcanonical thermodynamic 
    potential of a single strange quark flavor (imag part).
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    
    ### Returns
    gcp_boson_imag : float
        Value of the thermodynamic potential in MeV^4.
    """

    Ip = I_boson(T, 0.0, 0.0, phi_re, phi_im, '+')
    Im = I_boson(T, 0.0, 0.0, phi_re, phi_im, '-')
    alpha = alpha_s(T, mu)

    IpIm = complex(math.fsum([Ip.real, Im.real]), math.fsum([Ip.imag, Im.imag]))
    IpIm2 = IpIm**2

    return (9.0/(math.pi**3))*alpha*(T**4)*IpIm2.imag


gcp_hash = {
    'l' : gcp_fermion_l_real,
    's' : gcp_fermion_s_real
}


@functools.lru_cache(maxsize=1024)
def pressure(
    T: float, mu: float, phi_re: float, phi_im: float, typ: str
) -> float:
    """### Description
    Pertrubative correction to the pressure of a single quark flavor.
    
    ### Prameters
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.
    typ : str
        Type of quark
            'l' : up / down quark
            's' : strange quark
    
    ### Returns
    pressure : float
        Value of the thermodynamic pressure in MeV^4.
    """

    if pnjl.defaults.PERTURBATIVE_GLUON_CORRECTION:
        return math.fsum([
            -gcp_hash[typ](T, mu, phi_re, phi_im),
            -gcp_boson_real(T, mu, phi_re, phi_im)
        ])
    else:
        return -gcp_hash[typ](T, mu, phi_re, phi_im)


@functools.lru_cache(maxsize=1024)
def bdensity(
    T: float, mu: float, phi_re: float, phi_im: float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    typ: str
) -> float:
    """### Description
    Pertrubative correction to the baryon density of a single quark flavor.

    ### Parameters
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
    typ : str
        Type of quark
            'l' : up / down quark
            's' : strange quark

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
            coef*pressure(T, mu_el, phi_el[0], phi_el[1], typ)/3.0
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
            phi_solver, typ
        )


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant(
    rank: int, T: float, mu: float, phi_re: float, phi_im: float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    typ: str
) -> float:
    """### Description
    Pertrubative correction to the quark number cumulant chi_q of a single 
    quark flavor. Based on Eq.29 of https://arxiv.org/pdf/2012.12894.pdf and 
    the subsequent inline definition.

    ### Parameters
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
    typ : str
        Type of quark
            'l' : up / down quark
            's' : strange quark

    ### Returns
    qnumber_cumulant : float
        Value of the thermodynamic quark number cumulant in MeV^3.
    """

    if rank == 1:

        return 3.0 * bdensity(T, mu, phi_re, phi_im,  phi_solver, typ)

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
                    phi_solver, typ)
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
                phi_solver, typ
            )


@functools.lru_cache(maxsize=1024)
def sdensity(
    T: float, mu: float, phi_re : float, phi_im : float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    typ: str
) -> float:
    """### Description
    Pertrubative correction to the entropy density of a single quark flavor.

    ### Parameters
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
    typ : str
        Type of quark
            'l' : up / down quark
            's' : strange quark

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
            coef*pressure(T_el, mu, phi_el[0], phi_el[1], typ)
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
            phi_solver, typ
        )