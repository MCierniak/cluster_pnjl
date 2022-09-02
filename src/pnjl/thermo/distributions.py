import math


def En(p: float, mass: float) -> float:
    """Relativistic energy.

    Parameters
    ----------
    p : float
        Absoltue value of the 3-momentum vector in MeV.
    mass : float
        Relativistic mass in MeV.

    Returns
    -------
    En : float
        Relativistic energy value in MeV.
    """

    body = math.fsum([p**2, mass**2])
    return math.sqrt(body)


def log_y(
        p: float, T: float, mu: float, mass: float,
        mu_factor: int, en_factor: int, typ: str) -> float:
    """Relativistic particle/antiparticle energy exponent logaritm.

    Parameters
    ----------
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

    Returns
    -------
    log_y_plus : float
        Relativistic particle energy exponent logaritm.
    """

    match typ:
        case '+':
            ensum = math.fsum([En(p, mass), -mu_factor*mu])
        case '-':
            ensum = math.fsum([En(p, mass), mu_factor*mu])
        case _:
            raise ValueError("log_y typ must be '+' or '-'!")        

    return en_factor*ensum/T


def f_fermion_singlet(p, T, mu, mass, mu_factor, en_factor, typ):
    if typ not in ['+', '-']:
        raise ValueError("Distribution type can only be + or -.")
    if y_status == 4:
        raise RuntimeError("Error in pnj.thermo.distributions.f_fermion_singlet, y value not passed...")
    else:
        return 1.0 / (y_val + 1.0)

def f_boson_singlet(y_val = 0.0, y_status = 4) -> float:
    if y_status == 4:
        raise RuntimeError("Error in pnj.thermo.distributions.f_baryon_singlet, y value not passed...")
    elif y_status == 2:
        return math.inf
    else:
        return 1.0 / (y_val - 1.0)

def f_fermion_triplet(
    Phi : complex, Phib : complex, 
    y_1_val = 0.0, y_1_status = 4,
    y_2_val = 0.0, y_2_status = 4,
    y_3_val = 0.0, y_3_status = 4,
    ) -> complex:
    if 4 in [y_1_status, y_2_status, y_3_status]:
        raise RuntimeError("Error in pnj.thermo.distributions.f_fermion_triplet, y value not passed...")

    y_m1_val = 1.0 / y_1_val if not y_1_status == 1 else math.inf
    y_m2_val = 1.0 / y_2_val if not y_2_status == 1 else math.inf

    Phi_m1_real = 3.0 * Phi.real * y_m1_val if not Phi.real == 0.0 and not y_m1_val == 0.0 else 0.0
    Phi_m1_imag = 3.0 * Phi.imag * y_m1_val if not Phi.imag == 0.0 and not y_m1_val == 0.0 else 0.0

    Phib_1_real = 3.0 * Phib.real * y_1_val if not Phib.real == 0.0 and not y_1_val == 0.0 else 0.0
    Phib_1_imag = 3.0 * Phib.imag * y_1_val if not Phib.imag == 0.0 and not y_1_val == 0.0 else 0.0

    Phib_2_real = 3.0 * Phib.real * y_2_val if not Phib.real == 0.0 and not y_2_val == 0.0 else 0.0
    Phib_2_imag = 3.0 * Phib.imag * y_2_val if not Phib.imag == 0.0 and not y_2_val == 0.0 else 0.0

    Phi_1_real = 3.0 * Phi.real * y_1_val if not Phi.real == 0.0 and not y_1_val == 0.0 else 0.0
    Phi_1_imag = 3.0 * Phi.imag * y_1_val if not Phi.imag == 0.0 and not y_1_val == 0.0 else 0.0

    den1 = y_1_val + 3.0 * Phib + complex(Phi_m1_real, Phi_m1_imag) + y_m2_val
    den2 = y_2_val + complex(Phib_1_real, Phib_1_imag) + 3.0 * Phi + y_m1_val
    den3 = y_3_val + complex(Phib_2_real, Phib_2_imag) + complex(Phi_1_real, Phi_1_imag) + 1.0

    part1 = 3.0 * Phib / den1 if not math.fabs(den1.real) == math.inf and not math.fabs(den1.imag) == math.inf else complex(0.0, 0.0)
    part2 = 6.0 * Phi / den2 if not math.fabs(den2.real) == math.inf and not math.fabs(den2.imag) == math.inf else complex(0.0, 0.0)
    part3 = 3.0 / den3 if not math.fabs(den3.real) == math.inf and not math.fabs(den3.imag) == math.inf else complex(0.0, 0.0)

    return part1 + part2 + part3

def f_fermion_antitriplet(
    Phi : complex, Phib : complex, 
    y_1_val = 0.0, y_1_status = 4,
    y_2_val = 0.0, y_2_status = 4,
    y_3_val = 0.0, y_3_status = 4,
    ) -> complex:
    #
    if 4 in [y_1_status, y_2_status, y_3_status]:
        raise RuntimeError("Error in pnj.thermo.distributions.f_fermion_antitriplet, y value not passed...")
    return f_fermion_triplet(Phib, Phi, y_1_val = y_1_val, y_1_status = y_1_status, y_2_val = y_2_val, y_2_status = y_2_status, y_3_val = y_3_val, y_3_status = y_3_status)

def f_boson_triplet(
    Phi : complex, Phib : complex, 
    y_1_val = 0.0, y_1_status = 4,
    y_2_val = 0.0, y_2_status = 4,
    y_3_val = 0.0, y_3_status = 4,
    ) -> complex:
    if 4 in [y_1_status, y_2_status, y_3_status]:
        raise RuntimeError("Error in pnj.thermo.distributions.f_baryon_triplet, y value not passed...")

    y_m1_val = 1.0 / y_1_val if not y_1_status == 1 else math.inf
    y_m2_val = 1.0 / y_2_val if not y_2_status == 1 else math.inf

    Phi_m1_real = 3.0 * Phi.real * y_m1_val if not Phi.real == 0.0 and not y_m1_val == 0.0 else 0.0
    Phi_m1_imag = 3.0 * Phi.imag * y_m1_val if not Phi.imag == 0.0 and not y_m1_val == 0.0 else 0.0

    Phi_1_real = 3.0 * Phi.real * y_1_val if not Phi.real == 0.0 and not y_1_val == 0.0 else 0.0
    Phi_1_imag = 3.0 * Phi.imag * y_1_val if not Phi.imag == 0.0 and not y_1_val == 0.0 else 0.0

    Phib_1_real = 3.0 * Phib.real * y_1_val if not Phib.real == 0.0 and not y_1_val == 0.0 else 0.0
    Phib_1_imag = 3.0 * Phib.imag * y_1_val if not Phib.imag == 0.0 and not y_1_val == 0.0 else 0.0

    Phib_2_real = 3.0 * Phib.real * y_2_val if not Phib.real == 0.0 and not y_2_val == 0.0 else 0.0
    Phib_2_imag = 3.0 * Phib.imag * y_2_val if not Phib.imag == 0.0 and not y_2_val == 0.0 else 0.0

    den1_sub2_real = Phi_m1_real - y_m2_val if not (math.fabs(Phi_m1_real) == math.inf and math.fabs(y_m2_val) == math.inf) else -math.inf
    den1_sub2_imag = Phi_m1_imag

    den2_sub1_real = y_2_val - Phib_1_real if not (math.fabs(y_2_val) == math.inf and math.fabs(Phib_1_real) == math.inf) else math.inf
    den2_sub1_imag = -Phib_1_imag

    den3_sub1_real = y_3_val - Phib_2_real if not (math.fabs(y_3_val) == math.inf and math.fabs(Phib_2_real) == math.inf) else math.inf
    den3_sub1_imag = -Phib_2_imag

    den1 = y_1_val - 3.0 * Phib + complex(den1_sub2_real, den1_sub2_imag)
    den2 = complex(den2_sub1_real, den2_sub1_imag) + 3.0 * Phi - y_m1_val
    den3 = complex(den3_sub1_real, den3_sub1_imag) + complex(Phi_1_real, Phi_1_imag) - 1.0

    part1 = 3.0 * Phib / den1 if not math.fabs(den1.real) == math.fabs(math.inf) and not math.fabs(den1.imag) == math.fabs(math.inf) else complex(0.0, 0.0)
    part2 = -6.0 * Phi / den2 if not math.fabs(den2.real) == math.fabs(math.inf) and not math.fabs(den2.imag) == math.fabs(math.inf) else complex(0.0, 0.0)
    part3 = 3.0 / den3 if not math.fabs(den3.real) == math.fabs(math.inf) and not math.fabs(den3.imag) == math.fabs(math.inf) else complex(0.0, 0.0)

    return part1 + part2 + part3

def f_boson_antitriplet(
    Phi : complex, Phib : complex, 
    y_1_val = 0.0, y_1_status = 4,
    y_2_val = 0.0, y_2_status = 4,
    y_3_val = 0.0, y_3_status = 4,
    ) -> complex:
    if 4 in [y_1_status, y_2_status, y_3_status]:
        raise RuntimeError("Error in pnj.thermo.distributions.f_baryon_antitriplet, y value not passed...")
    return f_boson_triplet(Phib, Phi, y_1_val, y_1_status, y_2_val, y_2_status, y_3_val, y_3_status)