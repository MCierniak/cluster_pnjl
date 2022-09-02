import pnjl.defaults
import numpy

#Polynomial approximation of the Polyakov-loop potential from https://arxiv.org/pdf/hep-ph/0506234.pdf

def b2(T : float, **kwargs) -> float:
    
    options = {'T0' : pnjl.defaults.default_T0, 'a0' : pnjl.defaults.default_a0, 'a1' : pnjl.defaults.default_a1, 'a2' : pnjl.defaults.default_a2, 'a3' : pnjl.defaults.default_a3}
    options.update(kwargs)

    T0 = options['T0']
    a0 = options['a0']
    a1 = options['a1']
    a2 = options['a2']
    a3 = options['a3']

    return a0 + a1 * (T0 / T) + a2 * ((T0 / T) ** 2) + a3 * ((T0 / T) ** 3)
def U(T : float, Phi : complex, Phib : complex, **kwargs) -> complex:
    
    options = {'b3' : pnjl.defaults.default_b3, 'b4' : pnjl.defaults.default_b4}
    options.update(kwargs)

    b3 = options['b3']
    b4 = options['b4']

    return -(T ** 4) * ((b2(T, **kwargs) / 2.0) * Phi * Phib + (b3 / 6.0) * ((Phi ** 3) + (Phib ** 3)) - (b4 / 4.0) * ((Phi * Phib) ** 2))

#Grandcanonical potential (Polyakov-loop part)

def gcp_real(T : float, Phi : complex, Phib : complex, **kwargs) -> float:
    #
    return U(T, Phi, Phib, **kwargs).real
def gcp_imag(T : float, Phi : complex, Phib : complex, **kwargs) -> float:
    #
    return U(T, Phi, Phib, **kwargs).imag

#Extensive thermodynamic properties

def pressure(T : float, Phi : complex, Phib : complex, **kwargs):
    #
    return -gcp_real(T, Phi, Phib, **kwargs)

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