"""### Description
Polyakov-loop grandcanonical thermodynamic potential and associated functions.
Polynomial approximation from Lo_5_PhysRevD_88_(2013) .

### Functions
...
U
    Polyakov-loop grandcanonical thermodynamic potential.
pressure
    Polyakov-loop pressure.
bdensity
    Polyakov-loop baryon density.
qnumber_cumulant
    Polyakov-loop quark number cumulant chi_q of a single quark flavor.
sdensity
    Polyakov-loop entropy density.
"""


import math
import functools


A1 = -44.14
A2 = 151.4
A3 = -90.0677
A4 = 2.77173
A5 = 3.56403

B1 = -0.32665
B2 = -82.9823
B3 = 3.0
B4 = 5.85559

C1 = -50.7961
C2 = 114.038
C3 = -89.4596
C4 = 3.08718
C5 = 6.72812

D1 = 27.0885
D2 = -56.0859
D3 = 71.2225
D4 = 2.9715
D5 = 6.61433

T0 = 240.0


@functools.lru_cache(maxsize=1024)
def a(T: float) -> float:
    T0T = T0/T
    T0T2 = T0T**2
    num = math.fsum([A1, A2*T0T, A3*T0T2])
    den = math.fsum([1.0, A4*T0T, A5*T0T2])    
    return num/den


@functools.lru_cache(maxsize=1024)
def b(T: float) -> float:
    T0T = T0/T
    return -B1*(T0T**B4)*math.expm1(B2*(T0T**B3))


@functools.lru_cache(maxsize=1024)
def c(T: float) -> float:
    T0T = T0/T
    T0T2 = T0T**2
    num = math.fsum([C1, C2*T0T, C3*T0T2])
    den = math.fsum([1.0, C4*T0T, C5*T0T2])    
    return num/den


@functools.lru_cache(maxsize=1024)
def d(T: float) -> float:
    T0T = T0/T
    T0T2 = T0T**2
    num = math.fsum([D1, D2*T0T, D3*T0T2])
    den = math.fsum([1.0, D4*T0T, D5*T0T2])    
    return num/den


@functools.lru_cache(maxsize=1024)
def M_H(phi_re: float, phi_im: float) -> float:
    phi_re2 = phi_re**2
    phi_re3 = phi_re**3
    phi_re4 = phi_re**4
    phi_im2 = phi_im**2
    phi_im4 = phi_im**4
    return math.fsum(
        [
            1.0, -6.0*phi_im2, -3.0*phi_im4, -24.0*phi_im2*phi_re,
            -6.0*phi_re2, -6.0*phi_im2*phi_re2, 8.0*phi_re3, -3.0*phi_re4
        ]
    )


@functools.lru_cache(maxsize=1024)
def U(T : float, phi_re : float, phi_im : float) -> float:
    """### Description
    Polyakov-loop grandcanonical thermodynamic potential.

    ### Parameters
    T : float
        Temperature in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.

    ### Returns
    U : float
        Potential value in MeV^4.
    """
    haar = M_H(phi_re, phi_im)
    phi_re2 = phi_re**2
    phi_re3 = phi_re**3
    phi_re4 = phi_re**4
    phi_im2 = phi_im**2
    phi_im4 = phi_im**4
    a_term = -(a(T)/2.0)*(phi_re2 + phi_im2)
    b_term = b(T)*math.log(haar)
    c_term = c(T)*(phi_re3 - 3.0*phi_im2*phi_re)
    d_term = d(T)*(phi_im4 + phi_re4 + 2.0*phi_im2*phi_re2)
    return (T**4)*math.fsum([a_term, b_term, c_term, d_term])


@functools.lru_cache(maxsize=1024)
def pressure(T: float, mu: float, phi_re: float, phi_im: float) -> float:
    """### Description
    Polyakov-loop pressure.

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
    pressure : float
        Value of the thermodynamic pressure in MeV^4.
    """
    return -U(T, phi_re, phi_im)


@functools.lru_cache(maxsize=1024)
def bdensity(T: float, mu: float, phi_re : float, phi_im : float) -> float:
    """### Description
    Polyakov-loop baryon density.

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
    bdensity : float
        Value of the thermodynamic baryon density in MeV^3.
    """
    return 0.0


@functools.lru_cache(maxsize=1024)
def qnumber_cumulant(rank: int, T: float, mu: float, phi_re : float, phi_im : float) -> float:
    """### Description
    Polyakov-loop quark number cumulant chi_q. Based on Eq.29 of
    https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline definition.

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

    ### Returns
    qnumber_cumulant : float
        Value of the thermodynamic quark number cumulant in MeV^3.
    """
    return 0.0


@functools.lru_cache(maxsize=1024)
def sdensity(T: float, mu: float, phi_re : float, phi_im : float) -> float:
    """### Description
    Polyakov-loop entropy density.

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
            coef*pressure(T_el, mu, phi_el[0], phi_el[1])
            for T_el, coef, phi_el in zip(T_vec, deriv_coef, phi_vec)
        ]
        return math.fsum(p_vec)
    else:
        new_T = math.fsum([T, h])
        new_phi_re, new_phi_im = phi_re, phi_im
        return sdensity(new_T, mu, new_phi_re, new_phi_im)