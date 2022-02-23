import pnjl.defaults

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