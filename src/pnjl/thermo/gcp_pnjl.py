"""### Description
PNJL grandcanonical thermodynamic potential and associated functions.

### Functions
gcp_l_real
    PNJL grandcanonical thermodynamic potential of a single light quark flavor 
    (real part).
gcp_l_imag
    PNJL grandcanonical thermodynamic potential of a single light quark flavor 
    (imaginary part).
gcp_s_real
    PNJL grandcanonical thermodynamic potential of a single strange quark 
    flavor (real part).
gcp_s_imag
    PNJL grandcanonical thermodynamic potential of a single strange quark 
    flavor (imaginary part).
pressure
    PNJL pressure of a single quark flavor.
bdensity
    PNJL baryon density of a single quark flavor.
qnumber_cumulant
    PNJL quark number cumulant chi_q of a single quark flavor. Based on 
    Eq.29 of https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline 
    definition.
sdensity
    PNJL entropy density of a single quark flavor.
Polyakov_loop
"""


import math
import typing
import functools

import scipy.integrate

import pnjl.defaults
import pnjl.thermo.distributions
import pnjl.thermo.gcp_sigma_lattice


@functools.lru_cache(maxsize=1024)
def gcp_l_integrand_real(
    p: float, T: float, mu: float, phi_re: float, phi_im: float
) -> float:

        mass = pnjl.thermo.gcp_sigma_lattice.Ml(T, mu)
        En = pnjl.thermo.distributions.En(p, mass)

        fp = pnjl.thermo.distributions.f_fermion_triplet(
            p, T, mu, phi_re, phi_im, mass, 1, '+'
        )
        fm = pnjl.thermo.distributions.f_fermion_antitriplet(
            p, T, mu, phi_re, phi_im, mass, 1, '-'
        )

        return ((p**4)/En)*math.fsum([fp.real, fm.real])


@functools.lru_cache(maxsize=1024)
def gcp_l_integrand_imag(
    p: float, T: float, mu: float, phi_re: float, phi_im: float
) -> float:

        mass = pnjl.thermo.gcp_sigma_lattice.Ml(T, mu)
        En = pnjl.thermo.distributions.En(p, mass)
        
        fp = pnjl.thermo.distributions.f_fermion_triplet(
            p, T, mu, phi_re, phi_im, mass, 1, '+'
        )
        fm = pnjl.thermo.distributions.f_fermion_antitriplet(
            p, T, mu, phi_re, phi_im, mass, 1, '-'
        )

        return ((p**4)/En)*math.fsum([fp.imag, fm.imag])


@functools.lru_cache(maxsize=1024)
def gcp_s_integrand_real(
    p: float, T: float, mu: float, phi_re: float, phi_im: float
) -> float:

        mass = pnjl.thermo.gcp_sigma_lattice.Ms(T, mu)
        En = pnjl.thermo.distributions.En(p, mass)

        fp = pnjl.thermo.distributions.f_fermion_triplet(
            p, T, mu, phi_re, phi_im, mass, 1, '+'
        )
        fm = pnjl.thermo.distributions.f_fermion_antitriplet(
            p, T, mu, phi_re, phi_im, mass, 1, '-'
        )

        return ((p**4)/En)*math.fsum([fp.real, fm.real])


@functools.lru_cache(maxsize=1024)
def gcp_s_integrand_imag(
    p: float, T: float, mu: float, phi_re: float, phi_im: float
) -> float:

        mass = pnjl.thermo.gcp_sigma_lattice.Ms(T, mu)
        En = pnjl.thermo.distributions.En(p, mass)

        fp = pnjl.thermo.distributions.f_fermion_triplet(
            p, T, mu, phi_re, phi_im, mass, 1, '+'
        )
        fm = pnjl.thermo.distributions.f_fermion_antitriplet(
            p, T, mu, phi_re, phi_im, mass, 1, '-'
        )

        return ((p**4)/En)*math.fsum([fp.imag, fm.imag])


@functools.lru_cache(maxsize=1024)
def gcp_l_real(T : float, mu : float, phi_re : float, phi_im : float) -> float:
    """### Description
    PNJL grandcanonical thermodynamic potential of a single light quark flavor 
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
    
    ### Returns
    gcp_l_real : float
        Value of the thermodynamic potential in MeV^4.
    """

    NC = pnjl.defaults.NC

    integral, error = scipy.integrate.quad(
        gcp_l_integrand_real, 0.0, math.inf, args=(T, mu, phi_re, phi_im)
    )

    return -(NC/3.0)*(1.0/(math.pi**2))*integral


@functools.lru_cache(maxsize=1024)
def gcp_l_imag(T : float, mu : float, phi_re : float, phi_im : float) -> float:
    """### Description
    PNJL grandcanonical thermodynamic potential of a single light quark flavor 
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
    
    ### Returns
    gcp_l_imag : float
        Value of the thermodynamic potential in MeV^4.
    """

    NC = pnjl.defaults.NC

    integral, error = scipy.integrate.quad(
        gcp_l_integrand_imag, 0.0, math.inf, args=(T, mu, phi_re, phi_im)
    )

    return -(NC/3.0)*(1.0/(math.pi**2))*integral


@functools.lru_cache(maxsize=1024)
def gcp_s_real(T : float, mu : float, phi_re : float, phi_im : float) -> float:
    """### Description
    PNJL grandcanonical thermodynamic potential of a single strange quark 
    flavor (real part).
    
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
    gcp_s_real : float
        Value of the thermodynamic potential in MeV^4.
    """

    NC = pnjl.defaults.NC

    integral, error = scipy.integrate.quad(
        gcp_s_integrand_real, 0.0, math.inf, args=(T, mu, phi_re, phi_im)
    )

    return -(NC/3.0)*(1.0/(math.pi**2))*integral


@functools.lru_cache(maxsize=1024)
def gcp_s_imag(T : float, mu : float, phi_re : float, phi_im : float) -> float:
    """### Description
    PNJL grandcanonical thermodynamic potential of a single strange quark 
    flavor (imaginary part).
    
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
    gcp_s_imag : float
        Value of the thermodynamic potential in MeV^4.
    """

    NC = pnjl.defaults.NC

    integral, error = scipy.integrate.quad(
        gcp_s_integrand_imag, 0.0, math.inf, args=(T, mu, phi_re, phi_im)
    )

    return -(NC/3.0)*(1.0/(math.pi**2))*integral


gcp_hash = {
    'l': gcp_l_real,
    's': gcp_s_real
}


@functools.lru_cache(maxsize=1024)
def pressure(
    T: float, mu: float, phi_re: float, phi_im: float, typ: str
) -> float:
    """### Description
    PNJL pressure of a single quark flavor.
    
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

    return -gcp_hash[typ](T, mu, phi_re, phi_im)


@functools.lru_cache(maxsize=1024)
def bdensity(T: float, mu: float, phi_re: float, phi_im: float, typ: str) -> float:
    """### Description
    PNJL baryon density of a single quark flavor.
    
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
    typ : str
        Type of quark
            'l' : up / down quark
            's' : strange quark
    fast_calc : bool, optional
        Increase calculation speed by assuming phi(mu) ~= const. 
        Defaults to False.
    
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
        phi_vec = [
            tuple([phi_re, phi_im])
            for _ in mu_vec
        ]
        p_vec = [
            coef*pressure(T, mu_el, phi_el[0], phi_el[1], typ)/3.0
            for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
        ]
        return math.fsum(p_vec)
    else:
        new_mu = math.fsum([mu, h])
        new_phi_re, new_phi_im = phi_re, phi_im
        return bdensity(T, new_mu, new_phi_re, new_phi_im, typ)


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant(rank: int, T: float, mu: float, phi_re: float, phi_im: float, typ: str) -> float:
    """### Description
    PNJL quark number cumulant chi_q of a single quark flavor. Based on 
    Eq.29 of https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline definition.
    
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
    phi_solver : typing.Callable[ [float, float, float, float], typing.Tuple[float, float] ]
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
    fast_calc : bool, optional
        Increase calculation speed by assuming phi(mu) ~= const. Defaults to False.
    
    ### Returns
    qnumber_cumulant : float
        Value of the thermodynamic quark number cumulant in MeV^3.
    """

    if rank == 1:
        return 3.0 * bdensity(T, mu, phi_re, phi_im, typ)
    else:
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
            phi_vec = [
                tuple([phi_re, phi_im])
                for _ in mu_vec
            ]
            out_vec = [
                coef*qnumber_cumulant(rank-1, T, mu_el, phi_el[0], phi_el[1], typ)
                for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
            ]
            return math.fsum(out_vec)
        else:
            new_mu = math.fsum([mu, h])
            new_phi_re, new_phi_im = phi_re, phi_im
            return qnumber_cumulant(rank, T, new_mu, new_phi_re, new_phi_im, typ)


@functools.lru_cache(maxsize=1024)
def sdensity(
    T: float, mu: float, phi_re : float, phi_im : float, typ: str
) -> float:
    """### Description
    PNJL entropy density of a single quark flavor.
    
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
            math.fsum([T, 2*h]), math.fsum([T, h]),
            math.fsum([T, -h]), math.fsum([T, -2*h])
        ]
        deriv_coef = [
            -1.0/(12.0*h), 8.0/(12.0*h),
            -8.0/(12.0*h), 1.0/(12.0*h)
        ]
        phi_vec = [
            tuple([phi_re, phi_im])
            for _ in T_vec
        ]
        p_vec = [
            coef*pressure(T_el, mu, phi_el[0], phi_el[1], typ)
            for T_el, coef, phi_el in zip(T_vec, deriv_coef, phi_vec)
        ]
        return math.fsum(p_vec)
    else:
        new_T = math.fsum([T, h])
        new_phi_re, new_phi_im = phi_re, phi_im
        return sdensity(new_T, mu, new_phi_re, new_phi_im, typ)