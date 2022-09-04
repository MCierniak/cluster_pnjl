"""Particle and cluster distribution functions

### Functions
En
    Relativistic energy.
log_y
    Relativistic particle/antiparticle energy exponent logaritm.
f_fermion_singlet
    Color-singlet fermion distribution function.
f_boson_singlet
    Color-singlet boson distribution function.
f_fermion_triplet
    Color-triplet fermion distribution function.
f_fermion_antitriplet
    Color-antitriplet fermion distribution function.
f_boson_triplet
    Color-triplet boson distribution function.
f_boson_antitriplet
    Color-antitriplet boson distribution function.

### Globals
exp_limit:
    Limit for math.exp input before OverflowError is raised
log_y_hash:
    Dict for energy sign operator lookup.
"""


import operator
import math


exp_limit = 709.78271


def En(p: float, mass: float) -> float:
    """Relativistic energy.

    ### Parameters
    p : float
        Absoltue value of the 3-momentum vector in MeV.
    mass : float
        Relativistic mass in MeV.

    ### Returns
    En : float
        Relativistic energy value in MeV.
    """

    body = math.fsum([p**2, mass**2])
    return math.sqrt(body)


log_y_hash = {
    '+' : operator.neg,
    '-' : operator.pos
}


def log_y(
        p: float, T: float, mu: float, mass: float,
        mu_factor: int, en_factor: int, typ: str) -> float:
    """Relativistic particle/antiparticle energy exponent logaritm.

    ### Parameters
    p : float
        Absoltue value of the 3-momentum vector in MeV.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    mass : float
        Relativistic mass in MeV.
    mu_factor : int
        Multiplication factor for the chemical potential.
    en_factor : int
        Multiplication factor for the total energy exponent.
    typ : string
        Type of particle:
            '+' : positive energy particle
            '-' : negative energy antiparticle

    ### Returns
    log_y_plus : float
        Relativistic particle energy exponent logaritm.
    """

    ensum = math.fsum([En(p, mass), log_y_hash[typ](mu_factor*mu)])
    
    return en_factor*ensum/T


def f_fermion_singlet(
        p: float, T: float, mu: float,
        mass: float, mu_factor: int, typ: str) -> float:
    """Color-singlet fermion distribution function.

    ### Parameters
    p : float
        Absoltue value of the 3-momentum vector in MeV.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    mass : float
        Relativistic mass in MeV.
    mu_factor : int
        Multiplication factor for the chemical potential.
    typ : string
        Type of particle:
            '+' : positive energy particle
            '-' : negative energy antiparticle

    ### Returns
    f_fermion_singlet : float
        Value of the distribution function.
    """

    logy = log_y(p, T, mu, mass, mu_factor, 1, typ)

    if logy >= exp_limit:
        return 0.0
    else:
        return 1.0/math.fsum([math.exp(logy), 1.0])


def f_boson_singlet(
        p: float, T: float, mu: float, 
        mass: float, mu_factor: int, typ: str) -> float:
    """Color-singlet boson distribution function.

    ### Parameters
    p : float
        Absoltue value of the 3-momentum vector in MeV.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    mass : float
        Relativistic mass in MeV.
    mu_factor : int
        Multiplication factor for the chemical potential.
    typ : string
        Type of particle:
            '+' : positive energy particle
            '-' : negative energy antiparticle

    ### Returns
    f_boson_singlet : float
        Value of the distribution function.
    """

    logy = log_y(p, T, mu, mass, mu_factor, 1, typ)

    if logy >= exp_limit:
        return 0.0
    else:
        return 1.0/math.expm1(logy)


def f_fermion_triplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:
    """Color-triplet fermion distribution function.

    ### Parameters
    p : float
        Absoltue value of the 3-momentum vector in MeV.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re: float
        Real part of the traced Polyakov-loop in MeV.
    phi_im: float
        Imaginary part of the traced Polyakov-loop in MeV.
    mass : float
        Relativistic mass in MeV.
    mu_factor : int
        Multiplication factor for the chemical potential.
    typ : string
        Type of particle:
            '+' : positive energy particle
            '-' : negative energy antiparticle

    ### Returns
    f_fermion_triplet : complex
        Value of the distribution function.
    """

    logy_p1 = log_y(p, T, mu, mass, mu_factor, 1, typ)
    logy_p2 = log_y(p, T, mu, mass, mu_factor, 2, typ)
    logy_p3 = log_y(p, T, mu, mass, mu_factor, 3, typ)

    test = [el <= -exp_limit for el in [logy_p1, logy_p2, logy_p3]]

    match test:
        case [False, False, False]:

            num_real_el = [
                phi_re*math.exp(-logy_p1),
                2.0*phi_re*math.exp(-logy_p2),
                math.exp(-logy_p3)
            ]
            num_imag_el = [
                -phi_im*math.exp(-logy_p1),
                2.0*phi_im*math.exp(-logy_p2)
            ]
            den_real_el = [
                1.0,
                3.0*phi_re*math.exp(-logy_p1),
                3.0*phi_re*math.exp(-logy_p2),
                math.exp(-logy_p3)
            ]
            den_imag_el = [
                -3.0*phi_im*math.exp(-logy_p1),
                3.0*phi_im*math.exp(-logy_p2)
            ]

            num_real = math.fsum(num_real_el)
            num_imag = math.fsum(num_imag_el)
            den_real = math.fsum(den_real_el)
            den_imag = math.fsum(den_imag_el)

            return complex(num_real, num_imag)/complex(den_real, den_imag)

        case [False, False, True] | [False, True, True]:

            num_real_el = [
                phi_re*math.exp(logy_p1),
                2.0*phi_re,
                math.exp(-logy_p1)
            ]
            num_imag_el = [
                -phi_im*math.exp(logy_p1),
                2.0*phi_im
            ]
            den_real_el = [
                math.exp(logy_p2),
                3.0*phi_re*math.exp(logy_p1),
                3.0*phi_re,
                math.exp(-logy_p1)
            ]
            den_imag_el = [
                -3.0*phi_im*math.exp(logy_p1),
                3.0*phi_im
            ]

            num_real = math.fsum(num_real_el)
            num_imag = math.fsum(num_imag_el)
            den_real = math.fsum(den_real_el)
            den_imag = math.fsum(den_imag_el)

            return complex(num_real, num_imag)/complex(den_real, den_imag)

        case [True, True, True]:

            return complex(1.0, 0.0)

        case _:

            raise RuntimeError("Sanity test failed!")


def f_fermion_antitriplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:
    """Color-antitriplet fermion distribution function.

    ### Parameters
    p : float
        Absoltue value of the 3-momentum vector in MeV.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re: float
        Real part of the traced Polyakov-loop in MeV.
    phi_im: float
        Imaginary part of the traced Polyakov-loop in MeV.
    mass : float
        Relativistic mass in MeV.
    mu_factor : int
        Multiplication factor for the chemical potential.
    typ : string
        Type of particle:
            '+' : positive energy particle
            '-' : negative energy antiparticle

    ### Returns
    f_fermion_antitriplet : complex
        Value of the distribution function.
    """

    return f_fermion_triplet(p, T, mu, phi_re, -phi_im, mass, mu_factor, typ)


def f_boson_triplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:
    """Color-triplet boson distribution function.

    ### Parameters
    p : float
        Absoltue value of the 3-momentum vector in MeV.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re: float
        Real part of the traced Polyakov-loop in MeV.
    phi_im: float
        Imaginary part of the traced Polyakov-loop in MeV.
    mass : float
        Relativistic mass in MeV.
    mu_factor : int
        Multiplication factor for the chemical potential.
    typ : string
        Type of particle:
            '+' : positive energy particle
            '-' : negative energy antiparticle

    ### Returns
    f_boson_triplet : complex
        Value of the distribution function.
    """

    logy_p1 = log_y(p, T, mu, mass, mu_factor, 1, typ)
    logy_p2 = log_y(p, T, mu, mass, mu_factor, 2, typ)
    logy_p3 = log_y(p, T, mu, mass, mu_factor, 3, typ)

    test = [el <= -exp_limit for el in [logy_p1, logy_p2, logy_p3]]

    match test:
        case [False, False, False]:

            num_real_el = [
                phi_re*math.exp(-logy_p2),
                phi_re*math.exp(-logy_p1)*math.expm1(-logy_p1),
                -math.exp(-logy_p3)
            ]
            num_imag_el = [
                phi_im*math.exp(-logy_p1),
                2.0*phi_im*math.exp(-logy_p2)
            ]
            den_real_el = [
                3.0*phi_re*math.exp(-logy_p1)*math.expm1(-logy_p1),
                math.expm1(-logy_p3)
            ]
            den_imag_el = [
                3.0*phi_im*math.exp(-logy_p1),
                3.0*phi_im*math.exp(-logy_p2)
            ]

            num_real = math.fsum(num_real_el)
            num_imag = math.fsum(num_imag_el)
            den_real = math.fsum(den_real_el)
            den_imag = math.fsum(den_imag_el)

            return complex(num_real, num_imag)/complex(den_real, den_imag)

        case [False, False, True] | [False, True, True]:

            num_real_el = [
                phi_re,
                -phi_re*math.expm1(logy_p1),
                -math.exp(-logy_p1)
            ]
            num_imag_el = [
                2.0*phi_im,
                phi_im*math.exp(logy_p1)
            ]
            den_real_el = [
                math.exp(logy_p2),
                math.exp(-logy_p1),
                -3.0*phi_re*math.expm1(logy_p1)
            ]
            den_imag_el = [
                3.0*phi_im*math.exp(logy_p1),
                3.0*phi_im
            ]

            num_real = math.fsum(num_real_el)
            num_imag = math.fsum(num_imag_el)
            den_real = math.fsum(den_real_el)
            den_imag = math.fsum(den_imag_el)

            return complex(num_real, num_imag)/complex(den_real, den_imag)

        case [True, True, True]:

            return complex(-1.0, 0.0)

        case _:

            raise RuntimeError("Sanity test failed!")


def f_boson_antitriplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:
    """Color-antitriplet boson distribution function.

    ### Parameters
    p : float
        Absoltue value of the 3-momentum vector in MeV.
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    phi_re: float
        Real part of the traced Polyakov-loop in MeV.
    phi_im: float
        Imaginary part of the traced Polyakov-loop in MeV.
    mass : float
        Relativistic mass in MeV.
    mu_factor : int
        Multiplication factor for the chemical potential.
    typ : string
        Type of particle:
            '+' : positive energy particle
            '-' : negative energy antiparticle

    ### Returns
    f_boson_antitriplet : complex
        Value of the distribution function.
    """

    return f_boson_triplet(p, T, mu, phi_re, -phi_im, mass, mu_factor, typ)

