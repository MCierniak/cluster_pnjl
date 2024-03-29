"""### Description
Particle and cluster distribution functions

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
"""


import math
import cmath

import utils


log_y_hash = {
    '+' : -1.0,
    '-' : 1.0
}


def En(p: float, mass: float) -> float:
    """### Description
    Relativistic energy.

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


def log_y(
        p: float, T: float, mu: float, mass: float,
        mu_factor: int, en_factor: int, typ: str) -> float:
    """### Description
    Relativistic particle/antiparticle energy exponent logaritm.

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

    ensum = math.fsum([En(p, mass), log_y_hash[typ]*mu_factor*mu])
    
    return en_factor*ensum/T


def f_fermion_singlet(
        p: float, T: float, mu: float,
        mass: float, mu_factor: int, typ: str) -> float:
    """### Description
    Color-singlet fermion distribution function.

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

    if logy >= utils.EXP_LIMIT:
        return 0.0
    else:
        return 1.0/math.fsum([math.exp(logy), 1.0])


def f_boson_singlet(
        p: float, T: float, mu: float, 
        mass: float, mu_factor: int, typ: str) -> float:
    """### Description
    Color-singlet boson distribution function.

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

    if logy >= utils.EXP_LIMIT:
        return 0.0
    else:
        return 1.0/math.expm1(logy)


def z_fermion_triplet_case1(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:
    
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

    den_real = math.fsum(den_real_el)
    den_imag = math.fsum(den_imag_el)

    return cmath.log(complex(den_real, den_imag))


def z_fermion_triplet_case2(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:
    
    den_real_el = [
        math.exp(logy_p1),
        3.0*phi_re,
        3.0*phi_re*math.exp(-logy_p1),
        math.exp(-logy_p2)
    ]
    den_imag_el = [
        -3.0*phi_im,
        3.0*phi_im*math.exp(-logy_p1)
    ]

    den_real = math.fsum(den_real_el)
    den_imag = math.fsum(den_imag_el)

    return cmath.log(complex(den_real, den_imag)) - complex(logy_p1, 0.0)


def z_fermion_triplet_case3(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:
    
    den_real_el = [
        3.0*phi_re*math.exp(logy_p1),
        3.0*phi_re,
        math.exp(-logy_p1)
    ]
    den_imag_el = [
        -3.0*phi_im*math.exp(-logy_p1),
        3.0*phi_im*math.exp(-logy_p2)
    ]

    den_real = math.fsum(den_real_el)
    den_imag = math.fsum(den_imag_el)

    return cmath.log(complex(den_real, den_imag)) - complex(logy_p2, 0.0)


def z_fermion_triplet_case4(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:
    
    return complex(-logy_p3, 0.0)


z_fermion_triplet_hash = {
    (bool(False), bool(False), bool(False)) : z_fermion_triplet_case1,
    (bool(False), bool(False), bool(True)) : z_fermion_triplet_case2,
    (bool(False), bool(True), bool(True)) : z_fermion_triplet_case3,
    (bool(True), bool(True), bool(True)) : z_fermion_triplet_case4,
}


def z_fermion_triplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:

    logy_p1 = log_y(p, T, mu, mass, mu_factor, 1, typ)
    logy_p2 = log_y(p, T, mu, mass, mu_factor, 2, typ)
    logy_p3 = log_y(p, T, mu, mass, mu_factor, 3, typ)

    test = tuple([el <= -utils.EXP_LIMIT for el in [logy_p1, logy_p2, logy_p3]])

    return z_fermion_triplet_hash[test](logy_p1, logy_p2, logy_p3, phi_re, phi_im)


def z_fermion_antitriplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:

    logy_p1 = log_y(p, T, mu, mass, mu_factor, 1, typ)
    logy_p2 = log_y(p, T, mu, mass, mu_factor, 2, typ)
    logy_p3 = log_y(p, T, mu, mass, mu_factor, 3, typ)

    test = tuple([el <= -utils.EXP_LIMIT for el in [logy_p1, logy_p2, logy_p3]])

    return z_fermion_triplet_hash[test](logy_p1, logy_p2, logy_p3, phi_re, -phi_im)


def z_boson_triplet_case1(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:
    
    den_real_el = [
        1.0,
        -3.0*phi_re*math.exp(-logy_p1),
        3.0*phi_re*math.exp(-logy_p2),
        -math.exp(-logy_p3)
    ]
    den_imag_el = [
        3.0*phi_im*math.exp(-logy_p1),
        3.0*phi_im*math.exp(-logy_p2)
    ]

    den_real = math.fsum(den_real_el)
    den_imag = math.fsum(den_imag_el)

    return cmath.log(complex(den_real, den_imag))


def z_boson_triplet_case2(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:
    
    den_real_el = [
        math.exp(logy_p1),
        -3.0*phi_re,
        3.0*phi_re*math.exp(-logy_p1),
        -math.exp(-logy_p2)
    ]
    den_imag_el = [
        3.0*phi_im,
        3.0*phi_im*math.exp(-logy_p1)
    ]

    den_real = math.fsum(den_real_el)
    den_imag = math.fsum(den_imag_el)

    return cmath.log(complex(den_real, den_imag)) - complex(logy_p1, 0.0)


def z_boson_triplet_case3(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:
    
    den_real_el = [
        -3.0*phi_re*math.exp(logy_p1),
        3.0*phi_re,
        -math.exp(-logy_p1)
    ]
    den_imag_el = [
        3.0*phi_im*math.exp(logy_p1),
        3.0*phi_im
    ]

    den_real = math.fsum(den_real_el)
    den_imag = math.fsum(den_imag_el)

    return cmath.log(complex(den_real, den_imag)) - complex(logy_p2, 0.0)


def z_boson_triplet_case4(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:
    
    return complex(-logy_p3, 0.0)


z_boson_triplet_hash = {
    (bool(False), bool(False), bool(False)) : z_boson_triplet_case1,
    (bool(False), bool(False), bool(True)) : z_boson_triplet_case2,
    (bool(False), bool(True), bool(True)) : z_boson_triplet_case3,
    (bool(True), bool(True), bool(True)) : z_boson_triplet_case4,
}


def z_boson_triplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:

    logy_p1 = log_y(p, T, mu, mass, mu_factor, 1, typ)
    logy_p2 = log_y(p, T, mu, mass, mu_factor, 2, typ)
    logy_p3 = log_y(p, T, mu, mass, mu_factor, 3, typ)

    test = tuple([el <= -utils.EXP_LIMIT for el in [logy_p1, logy_p2, logy_p3]])

    return z_boson_triplet_hash[test](logy_p1, logy_p2, logy_p3, phi_re, phi_im)


def z_boson_antitriplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:

    logy_p1 = log_y(p, T, mu, mass, mu_factor, 1, typ)
    logy_p2 = log_y(p, T, mu, mass, mu_factor, 2, typ)
    logy_p3 = log_y(p, T, mu, mass, mu_factor, 3, typ)

    test = tuple([el <= -utils.EXP_LIMIT for el in [logy_p1, logy_p2, logy_p3]])

    return z_boson_triplet_hash[test](logy_p1, logy_p2, logy_p3, phi_re, -phi_im)


def f_fermion_triplet_case1(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:

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


def f_fermion_triplet_case2(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:

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


def f_fermion_triplet_case3(
    logy_p1: float, logy_p2: float, logy_p3: float,
    phi_re: float, phi_im: float
) -> complex:

    return complex(1.0, 0.0)


f_fermion_triplet_hash = {
    (bool(False), bool(False), bool(False)) : f_fermion_triplet_case1,
    (bool(False), bool(False), bool(True)) : f_fermion_triplet_case2,
    (bool(False), bool(True), bool(True)) : f_fermion_triplet_case2,
    (bool(True), bool(True), bool(True)) : f_fermion_triplet_case3,
}


def f_fermion_triplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:
    """### Description
    Color-triplet fermion distribution function.

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

    test = tuple([el <= -utils.EXP_LIMIT for el in [logy_p1, logy_p2, logy_p3]])

    return f_fermion_triplet_hash[test](logy_p1, logy_p2, logy_p3, phi_re, phi_im)


def f_fermion_antitriplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:
    """### Description
    Color-antitriplet fermion distribution function.

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


def f_boson_triplet_case1(
        logy_p1: float, logy_p2: float, logy_p3: float,
        phi_re: float, phi_im: float) -> complex:

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


def f_boson_triplet_case2(
        logy_p1: float, logy_p2: float, logy_p3: float,
        phi_re: float, phi_im: float) -> complex:

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


def f_boson_triplet_case3(
        logy_p1: float, logy_p2: float, logy_p3: float,
        phi_re: float, phi_im: float) -> complex:
    return complex(-1.0, 0.0)


f_boson_triplet_hash = {
    (bool(False), bool(False), bool(False)) : f_boson_triplet_case1,
    (bool(False), bool(False), bool(True)) : f_boson_triplet_case2,
    (bool(False), bool(True), bool(True)) : f_boson_triplet_case2,
    (bool(True), bool(True), bool(True)) : f_boson_triplet_case3,
}


def f_boson_triplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:
    """### Description
    Color-triplet boson distribution function.

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

    test = tuple([el <= -utils.EXP_LIMIT for el in [logy_p1, logy_p2, logy_p3]])

    return f_boson_triplet_hash[test](logy_p1, logy_p2, logy_p3, phi_re, phi_im)


def f_boson_antitriplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str) -> complex:
    """### Description
    Color-antitriplet boson distribution function.

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


def dfdM_boson_singlet(
        p: float, T: float, mu: float, 
        mass: float, mu_factor: int, typ: str
) -> float:
    logy = log_y(p, T, mu, mass, mu_factor, 1, typ)
    if logy >= utils.EXP_LIMIT:
        return 0.0
    else:
        energy = En(p, mass)
        exy = math.exp(logy)
        f2 = f_boson_singlet(p, T, mu, mass, mu_factor, typ)**2
        return exy*f2*(mass/(energy*T))
    

def dfdM_fermion_singlet(
        p: float, T: float, mu: float, 
        mass: float, mu_factor: int, typ: str
) -> float:
    logy = log_y(p, T, mu, mass, mu_factor, 1, typ)
    if logy >= utils.EXP_LIMIT:
        return 0.0
    else:
        energy = En(p, mass)
        exy = math.exp(logy)
        f2 = f_fermion_singlet(p, T, mu, mass, mu_factor, typ)**2
        return exy*f2*(mass/(energy*T))
    

def dfdM_aux_boson(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int, typ: str
) -> complex:
    log_1 = log_y(p, T, mu, M, a, 1, typ)
    log_2 = log_y(p, T, mu, M, a, 2, typ)
    if log_1 >= utils.EXP_LIMIT:
        return complex(0.0, 0.0)
    elif log_1 < utils.EXP_LIMIT and log_2 >= utils.EXP_LIMIT:
        num_1_re = 2.0*math.fsum([math.exp(-log_1)*phi_re, -phi_re])
        num_1_im = 2.0*math.fsum([math.exp(-log_1)*phi_im, phi_im])
        den_1_re = math.fsum([math.exp(log_1), math.exp(-log_1)*phi_re, -2.0*phi_re])
        den_1_im = math.fsum([math.exp(-log_1)*phi_im, 2.0*phi_im])
        num_2_re = 2.0*math.fsum([-math.exp(-log_1)*1.0, phi_re])
        num_2_im = 2.0*phi_im
        den_2_re = math.fsum([math.exp(-log_1), -2.0*phi_re, math.exp(log_1)*phi_re])
        den_2_im = math.fsum([-2.0*phi_im, -math.exp(log_1)*phi_im])
        return  -complex(1.0, 0.0) \
                +(complex(num_1_re, num_1_im)/complex(den_1_re, den_1_im)) \
                +(complex(num_2_re, num_2_im)/complex(den_2_re, den_2_im))
    else:
        num_1_re = 2.0*math.fsum([phi_re, -math.exp(log_1)*phi_re])
        num_1_im = 2.0*math.fsum([phi_im, math.exp(log_1)*phi_im])
        den_1_re = math.fsum([math.exp(log_2), phi_re, -2.0*phi_re*math.exp(log_1)])
        den_1_im = math.fsum([phi_im, 2.0*phi_im*math.exp(log_1)])
        num_2_re = 2.0*math.fsum([-1.0, math.exp(log_1)*phi_re])
        num_2_im = 2.0*math.exp(log_1)*phi_im
        den_2_re = math.fsum([1.0, -2.0*math.exp(log_1)*phi_re, math.exp(log_2)*phi_re])
        den_2_im = math.fsum([-2.0*math.exp(log_1)*phi_im, -math.exp(log_2)*phi_im])
        return  -complex(1.0, 0.0) \
                +(complex(num_1_re, num_1_im)/complex(den_1_re, den_1_im)) \
                +(complex(num_2_re, num_2_im)/complex(den_2_re, den_2_im))
    

def dfdM_aux_fermion(
    M: float, p: float, T: float, mu: float,
    phi_re: float, phi_im: float, a: int, typ: str
) -> complex:
    log_1 = log_y(p, T, mu, M, a, 1, typ)
    log_2 = log_y(p, T, mu, M, a, 2, typ)
    if log_1 >= utils.EXP_LIMIT:
        return complex(0.0, 0.0)
    elif log_1 < utils.EXP_LIMIT and log_2 >= utils.EXP_LIMIT:
        num_1_re = 2.0*math.fsum([math.exp(-log_1)*phi_re, phi_re])
        num_1_im = 2.0*math.fsum([math.exp(-log_1)*phi_im, -phi_im])
        den_1_re = math.fsum([math.exp(log_1), math.exp(-log_1)*phi_re, 2.0*phi_re])
        den_1_im = math.fsum([math.exp(-log_1)*phi_im, -2.0*phi_im])
        num_2_re = 2.0*math.fsum([math.exp(-log_1)*1.0, phi_re])
        num_2_im = 2.0*phi_im
        den_2_re = math.fsum([math.exp(-log_1), 2.0*phi_re, math.exp(log_1)*phi_re])
        den_2_im = math.fsum([2.0*phi_im, -math.exp(log_1)*phi_im])
        return  complex(1.0, 0.0) \
                -(complex(num_1_re, num_1_im)/complex(den_1_re, den_1_im)) \
                +(complex(num_2_re, num_2_im)/complex(den_2_re, den_2_im))
    else:
        num_1_re = 2.0*math.fsum([phi_re, math.exp(log_1)*phi_re])
        num_1_im = 2.0*math.fsum([phi_im, -math.exp(log_1)*phi_im])
        den_1_re = math.fsum([math.exp(log_2), phi_re, 2.0*phi_re*math.exp(log_1)])
        den_1_im = math.fsum([phi_im, -2.0*phi_im*math.exp(log_1)])
        num_2_re = 2.0*math.fsum([1.0, math.exp(log_1)*phi_re])
        num_2_im = 2.0*math.exp(log_1)*phi_im
        den_2_re = math.fsum([1.0, 2.0*math.exp(log_1)*phi_re, math.exp(log_2)*phi_re])
        den_2_im = math.fsum([2.0*math.exp(log_1)*phi_im, -math.exp(log_2)*phi_im])
        return  complex(1.0, 0.0) \
                -(complex(num_1_re, num_1_im)/complex(den_1_re, den_1_im)) \
                +(complex(num_2_re, num_2_im)/complex(den_2_re, den_2_im))
    

def dfdM_boson_triplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str
) -> complex:
    energy = En(p, mass)
    fp_i = f_boson_triplet(p, T, mu, phi_re, phi_im, mass, mu_factor, typ)
    fp2_i = fp_i**2
    aux_p = dfdM_aux_boson(mass, p, T, mu, phi_re, phi_im, mu_factor, typ)
    return (fp2_i+fp_i)*aux_p*(mass/(energy*T))
    

def dfdM_boson_antitriplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str
) -> complex:
    energy = En(p, mass)
    fp_i = f_boson_triplet(p, T, mu, phi_re, -phi_im, mass, mu_factor, typ)
    fp2_i = fp_i**2
    aux_p = dfdM_aux_boson(mass, p, T, mu, phi_re, -phi_im, mu_factor, typ)
    return (fp2_i+fp_i)*aux_p*(mass/(energy*T))


def dfdM_fermion_triplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str
) -> complex:
    energy = En(p, mass)
    fp_i = f_fermion_triplet(p, T, mu, phi_re, phi_im, mass, mu_factor, typ)
    fp2_i = fp_i**2
    aux_p = dfdM_aux_fermion(mass, p, T, mu, phi_re, phi_im, mu_factor, typ)
    return (fp2_i-fp_i)*aux_p*(mass/(energy*T))
    

def dfdM_fermion_antitriplet(
        p: float, T: float, mu: float, phi_re: float, phi_im: float,
        mass: float, mu_factor: int, typ: str
) -> complex:
    energy = En(p, mass)
    fp_i = f_fermion_triplet(p, T, mu, phi_re, -phi_im, mass, mu_factor, typ)
    fp2_i = fp_i**2
    aux_p = dfdM_aux_fermion(mass, p, T, mu, phi_re, -phi_im, mu_factor, typ)
    return (fp2_i-fp_i)*aux_p*(mass/(energy*T))