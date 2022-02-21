import math

import pnjl.defaults

def En(p : float, mass : float, **kwargs) -> float:
    #
    return math.sqrt((p ** 2) + (mass ** 2))
def dEn_dmu(p : float, T : float, mu : float, mass : float, deriv_M_mu : float, **kwargs) -> float:
    #
    return (mass / En(p, mass, **kwargs)) * deriv_M_mu
def dEn_dT(p : float, T : float, mu : float, mass : float, deriv_M_T : float, **kwargs) -> float:
    #
    return (mass / En(p, mass, **kwargs)) * deriv_M_T
def y_plus(p : float, T : float, mu : float, mass : float, a : float, b : float, **kwargs) -> (float, int):
    body = b * (En(p, mass, **kwargs) - a * mu) / T
    if body > 709.7827:
        return (math.inf, 0)
    elif body < -709.7827:
        return (0.0, 1)
    elif body == 0.0:
        return (1.0, 2)
    else:
        return (math.exp(body), 3)
def y_minus(p : float, T : float, mu : float, mass : float, a : float, b : float, **kwargs) -> (float, int):
    body = b * (En(p, mass, **kwargs) + a * mu) / T
    if body > 709.7827:
        return (math.inf, 0)
    elif body < -709.7827:
        return (0.0, 1)
    elif body == 0.0:
        return (1.0, 2)
    else:
        return (math.exp(body), 3)
