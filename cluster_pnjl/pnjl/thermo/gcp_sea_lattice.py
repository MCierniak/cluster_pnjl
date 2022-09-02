import math

import scipy.integrate

import pnjl.aux_functions
import pnjl.defaults


def Tc(mu):
    """Pseudo-critical temperature ansatz.

    ---- Parameters ----
    mu : float
        Quark chemical potential in MeV.

    ---- Returns ----
    Tc : float
        Value of the pseudocritical temperature in MeV for a given mu.
    """

    Tc0 = pnjl.defaults.default_Tc0
    kappa = pnjl.defaults.default_kappa

    return Tc0 * (1.0 - kappa * (( mu / Tc0) ** 2))


def Delta_ls(T, mu):
    """LQCD-fit ansatz for the 2+1 Nf normalized chiral condensate.

    Details in https://arxiv.org/pdf/2012.12894.pdf .

    ---- Parameters ----
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ---- Returns ----
    Delta_ls : float
        Value of the normalized chiral condensate for a given T and mu.
    """

    delta_T = pnjl.defaults.default_delta_T

    return 0.5 * (1.0 - math.tanh((T - Tc(mu)) / delta_T))


def Ml(T, mu):
    """Mass of up / down quarks (ansatz)

    ---- Parameters ----
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ---- Returns ----
    Ml : float
        Quark mass in MeV.
    """
    
    M0 = pnjl.defaults.default_M0
    ml = pnjl.defaults.default_ml
    
    return M0 * Delta_ls(T, mu) + ml


def Ms(T, mu):
    """Mass of the strange quarks (ansatz)

    ---- Parameters ----
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ---- Returns ----
    Ms : float
        Quark mass in MeV.
    """

    M0 = pnjl.defaults.default_M0
    ml = pnjl.defaults.default_ms
    
    return M0 * Delta_ls(T, mu) + ml


def V(T, mu):
    """Sigma mean-field grandcanonical thermodynamic potential.

    ---- Parameters ----
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.

    ---- Returns ----
    V : float
        Mean-field value in MeV^4.
    """

    M0 = pnjl.defaults.default_M0
    Gs = pnjl.defaults.default_Gs

    return ((Delta_ls(T, mu)**2) * (M0**2)) / (4.0*Gs)


def gcp_real(T, mu, type, **kwargs):
    """Fermi sea grandcanonical thermodynamic potential of a single quark flavor.

    ---- Parameters ----
    T : float
        Temperature in MeV.
    mu : float
        Quark chemical potential in MeV.
    type : string 
        Type of quark
            - 'l' : up / down quark
            - 's' : strange quark
    no_sea : bool, optional
        No-sea approximation flag.

    ---- Returns ----
    gcp : float
        Value of the thermodynamic potential in MeV^4.
    """
    options = {'no_sea' : True}
    options.update(kwargs)

    M0 = pnjl.defaults.default_M0
    Nc = pnjl.defaults.default_Nc
    Lambda = pnjl.defaults.default_Lambda
    nosea = options['no_sea']

    if nosea:
        return 0.0
    else:
        def integrand(p):
            mass = Ml(T, mu)
            mass0 = Ml(0.0, 0.0)
            energy = pnjl.aux_functions.En(p, mass)
           = lambda p, _T, _mu, key : (p ** 2) * (pnjl.aux_functions.En(p, M(_T, _mu, **key), **key) - pnjl.aux_functions.En(p, M(0.0, 0.0, **key), **key))

    #integrand = lambda p, _T, _mu, key : (p ** 2) * (pnjl.aux_functions.En(p, M(_T, _mu, **key), **key) - pnjl.aux_functions.En(p, M(0.0, 0.0, **key), **key))

    #sigma_contrib = V(T, mu, **kwargs) - V(0.0, 0.0, **kwargs)

    #print(T, mu, sigma_contrib)
    #input()

    #integral, error = scipy.integrate.quad(integrand, 0.0, Lambda, args = (T, mu, kwargs))

    return 0.0#sigma_contrib - (Nf / (math.pi ** 2)) * (Nc / 3.0) * integral

#Extensive thermodynamic properties

def pressure(T : float, mu : float, **kwargs):
    #
    return -gcp_real(T, mu, **kwargs)
def bdensity(T : float, mu : float, **kwargs):
    
    h = 1e-2

    if mu - 2 * h > 0.0:
        mu_vec = [mu + 2 * h, mu + h, mu - h, mu - 2 * h]
        p_vec = [pressure(T, el, **kwargs) for el in mu_vec]
        return (1.0 / 3.0) * (p_vec[3] - 8.0 * p_vec[2] + 8.0 * p_vec[1] - p_vec[0]) / (12.0 * h)
    else:
        mu_vec = [mu + h, mu]
        if numpy.any([el < 0.0 for el in mu_vec]):
            return bdensity(T, mu + h, **kwargs)
        p_vec = [pressure(T, el, **kwargs) for el in mu_vec]
        return (1.0 / 3.0) * (p_vec[0] - p_vec[1]) / h
def sdensity(T : float, mu : float, **kwargs):
    
    h = 1e-2

    if T > 0.0:
        T_vec = [T + 2 * h, T + h, T - h, T - 2 * h]
        p_vec = [pressure(el, mu, **kwargs) for el in T_vec]
        return (1.0 / 3.0) * (p_vec[3] - 8.0 * p_vec[2] + 8.0 * p_vec[1] - p_vec[0]) / (12.0 * h)
    else:
        T_vec = [h, 0.0]
        p_vec = [pressure(el, mu, **kwargs) for el in T_vec]
        return (1.0 / 3.0) * (p_vec[0] - p_vec[1]) / h

def bnumber_cumulant(rank : int, T : float, mu : float, **kwargs):

    if rank == 1:
        return bdensity(T, mu, **kwargs) / (T ** 3)
    elif rank < 1:
        raise RuntimeError("Cumulant rank lower than 1!")
    else:
        h = 1e-2
        mu_vec = []
        if mu - 2 * h > 0.0:
            mu_vec = [mu + 2 * h, mu + h, mu - h, mu - 2 * h]
        else:
            mu_vec = [mu + h, mu]
            if numpy.any([el < 0.0 for el in mu_vec]):
                return bnumber_cumulant(rank, T, mu + h, **kwargs)

        if len(mu_vec) == 4 and numpy.all(mu_vec[i] > mu_vec[i + 1] for i, el in enumerate(mu_vec[:-1])):
            out_vec = [bnumber_cumulant(rank - 1, T, mu_el, **kwargs) for mu_el in mu_vec]
            return T * (out_vec[3] - 8.0 * out_vec[2] + 8.0 * out_vec[1] - out_vec[0]) / (12.0 * h)
        elif len(mu_vec) == 2 and mu_vec[0] > mu_vec[1]:
            out_vec = [bnumber_cumulant(rank - 1, T, mu_el, **kwargs) for mu_el in mu_vec]
            return T * (out_vec[0] - out_vec[1]) / h
        else:
            raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
