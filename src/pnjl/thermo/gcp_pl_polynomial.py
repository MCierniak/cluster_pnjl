"""Polyakov-loop grandcanonical thermodynamic potential and associated functions.
Polynomial approximation from https://arxiv.org/pdf/hep-ph/0506234.pdf .

### Functions
b2
    B2 coeficient of the Polyakov-loop potential.
U
    Polyakov-loop grandcanonical thermodynamic potential.
pressure
    Polyakov-loop pressure.
bdensity
    Polyakov-loop baryon density.
"""


import typing
import numpy
import math

import pnjl.defaults


def b2(T : float) -> float:
    """B2 coeficient of the Polyakov-loop potential.

    ### Parameters
    T : float
        Temperature in MeV.

    ### Returns
    b2 : float
        Value of the b2 coeficient.
    """

    T0 = pnjl.defaults.default_T0
    a0 = pnjl.defaults.default_a0
    a1 = pnjl.defaults.default_a1
    a2 = pnjl.defaults.default_a2
    a3 = pnjl.defaults.default_a3

    return math.fsum([a0, a1*(T0/T), a2*((T0/T)**2), a3*((T0/T)**3)])


def U(T : float, phi_re : float, phi_im : float) -> float:
    """Polyakov-loop grandcanonical thermodynamic potential.

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

    b3 = pnjl.defaults.default_b3
    b4 = pnjl.defaults.default_b4

    phi_sum_1 = math.fsum([phi_re**2, phi_im**2])
    phi_sum_2 = math.fsum([phi_re**2, -3.0*phi_im**2])

    phi_sum = [
        3.0*b4*(phi_sum_1**2), 
        -6.0*b2(T)*phi_sum_1,
        -4.0*b3*phi_re*phi_sum_2
    ]

    return ((T**4)/12.0)*math.fsum(phi_sum)


def pressure(T : float, phi_re : float, phi_im : float) -> float:
    """Polyakov-loop pressure.

    ### Parameters
    T : float
        Temperature in MeV.
    phi_re : float
        Real part of the traced Polyakov-loop in MeV.
    phi_im : float
        Imaginary part of the traced Polyakov-loop in MeV.

    ### Returns
    pressure : float
        Value of the thermodynamic pressure in MeV^4.
    """

    return -U(T, phi_re, phi_im)


def bdensity(
    T: float, mu: float, phi_re : float, phi_im : float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ]
) -> float:
    """Polyakov-loop baryon density.

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

    ### Returns
    bdensity : float
        Value of the thermodynamic baryon density in MeV^3.
    """

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
        phi_vec = [
            phi_solver(T, mu_el, phi_re, phi_im)
            for mu_el in mu_vec
        ]

        p_vec = [
            coef*pressure(T, phi_el[0], phi_el[1])/3.0
            for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
        ]

        return math.fsum(p_vec)

    else:
        return bdensity(T, math.fsum([mu, h]), phi_solver)


#correct form of the T vector would be T_vec = [T + 2 * h, T + h, T - h, T - 2 * h] with some interval h
#Phi/Phib vectors should correspond to Phi/Phib at the appropriate values of T!
def sdensity(T_vec : list, Phi_vec : list, Phib_vec : list, **kwargs):
    
    if len(T_vec) == len(Phi_vec) and len(T_vec) == len(Phib_vec):
        if len(T_vec) == 4 and numpy.all(T_vec[i] > T_vec[i + 1] for i, el in enumerate(T_vec[:-1])):
            h = T_vec[0] - T_vec[1]
            p_vec = [pressure(T_el, Phi_el, Phib_el, **kwargs) for T_el, Phi_el, Phib_el in zip(T_vec, Phi_vec, Phib_vec)]
            return (1.0 / 3.0) * (p_vec[3] - 8.0 * p_vec[2] + 8.0 * p_vec[1] - p_vec[0]) / (12.0 * h)
        elif len(T_vec) == 2 and T_vec[0] > T_vec[1]:
            h = T_vec[0]
            p_vec = [pressure(T_el, Phi_el, Phib_el, **kwargs) for T_el, Phi_el, Phib_el in zip(T_vec, Phi_vec, Phib_vec)]
            return (1.0 / 3.0) * (p_vec[0] - p_vec[1]) / h
        else:
            raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
    else:
        raise RuntimeError("Value vectors don't match!")