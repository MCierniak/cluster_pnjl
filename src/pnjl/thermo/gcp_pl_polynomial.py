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
qnumber_cumulant
    Polyakov-loop quark number cumulant chi_q of a single quark flavor.
sdensity
    Polyakov-loop entropy density.
"""


import typing
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

    T0 = pnjl.defaults.T0
    A0 = pnjl.defaults.A0
    A1 = pnjl.defaults.A1
    A2 = pnjl.defaults.A2
    A3 = pnjl.defaults.A3

    return math.fsum([A0, A1*(T0/T), A2*((T0/T)**2), A3*((T0/T)**3)])


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

    B3 = pnjl.defaults.B3
    B4 = pnjl.defaults.B4

    phi_sum_1 = math.fsum([phi_re**2, phi_im**2])
    phi_sum_2 = math.fsum([phi_re**2, -3.0*phi_im**2])

    phi_sum = [
        3.0*B4*(phi_sum_1**2), 
        -6.0*b2(T)*phi_sum_1,
        -4.0*B3*phi_re*phi_sum_2
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
                ],
    fast_calc : bool = False
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
            math.fsum(mu, 2*h), math.fsum(mu, h),
            math.fsum(mu, -h), math.fsum(mu, -2*h)
        ]
        deriv_coef = [
            -1.0/(12.0*h), 8.0/(12.0*h),
            -8.0/(12.0*h), 1.0/(12.0*h)
        ]
        phi_vec = []
        if fast_calc:
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
            coef*pressure(T, phi_el[0], phi_el[1])/3.0
            for coef, phi_el in zip(deriv_coef, phi_vec)
        ]

        return math.fsum(p_vec)

    else:

        new_mu = math.fsum([mu, h])
        new_phi_re, new_phi_im = phi_re, phi_im
            
        if not fast_calc:
            new_phi_re, new_phi_im = phi_solver(T, new_mu, phi_re, phi_im)

        return bdensity(
            T, new_mu, new_phi_re, new_phi_im, 
            phi_solver, fast_calc=fast_calc
        )


def qnumber_cumulant(
    rank: int, T: float, mu: float, phi_re : float, phi_im : float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    fast_calc : bool = False
) -> float:
    """Polyakov-loop quark number cumulant chi_q of a single quark flavor. Based on Eq.29 of
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
    fast_calc : bool, optional
        Increase calculation speed by assuming phi(mu) ~= const.
        Defaults to False.

    ### Returns
    qnumber_cumulant : float
        Value of the thermodynamic quark number cumulant in MeV^3.
    """

    if rank == 1:

        return 3.0 * bdensity(
            T, mu, phi_re, phi_im, 
            phi_solver, fast_calc=fast_calc
        )

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
            if fast_calc:
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
                    phi_solver, fast_calc=fast_calc)
                for mu_el, coef, phi_el in zip(mu_vec, deriv_coef, phi_vec)
            ]

            return math.fsum(out_vec)

        else:

            new_mu = math.fsum([mu, h])
            new_phi_re, new_phi_im = phi_re, phi_im
            
            if not fast_calc:
                new_phi_re, new_phi_im = phi_solver(T, new_mu, phi_re, phi_im)

            return qnumber_cumulant(
                rank, T, new_mu, new_phi_re, new_phi_im, 
                phi_solver, fast_calc=fast_calc
            )


def sdensity(
    T: float, mu: float, phi_re : float, phi_im : float,
    phi_solver: typing.Callable[
                    [float, float, float, float],
                    typing.Tuple[float, float]
                ],
    fast_calc : bool = False
) -> float:
    """Polyakov-loop entropy density.

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
    fast_calc : bool, optional
        Increase calculation speed by assuming phi(T) ~= const.
        Defaults to False.

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
        if fast_calc:
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
            coef*pressure(T_el, phi_el[0], phi_el[1])
            for T_el, coef, phi_el in zip(T_vec, deriv_coef, phi_vec)
        ]

        return math.fsum(p_vec)

    else:

        new_T = math.fsum([T, h])
        new_phi_re, new_phi_im = phi_re, phi_im
            
        if not fast_calc:
            new_phi_re, new_phi_im = phi_solver(new_T, mu, phi_re, phi_im)

        return sdensity(
            new_T, mu, new_phi_re, new_phi_im, 
            phi_solver, fast_calc=fast_calc
        )

