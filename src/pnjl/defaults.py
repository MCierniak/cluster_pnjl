"""Model parameters

### Globals
default_Tc0 : float
default_delta_T : float
default_ml : float
    Parameters of the lattice-fit ansatz for the chiral condensate. 
    Data from https://arxiv.org/pdf/2012.12894.pdf (Tc0 = T0).
    Tc0 set to Nf=2+1. Alternative Tc0 for Nf=2 available at 
    https://arxiv.org/pdf/hep-lat/0501030.pdf .
default_c : float
default_d : float
    Parameters of the perturbative running coupling ansatz from
    https://arxiv.org/pdf/2012.12894.pdf (modified).
default_kappa : float
    Lattice expansion parameter for Tc=f(mu) from
    https://arxiv.org/pdf/1812.08235.pdf .
default_Gs : float
    Scalar coupling constant from
    https://arxiv.org/pdf/hep-ph/0506234.pdf.
default_T0 : float
    Polyakov-loop polynomial potential pseudocritical temperature from
    https://arxiv.org/pdf/hep-ph/0506234.pdf (modified). Alternative
    Nf=2+1 result available at https://arxiv.org/pdf/0704.3234.pdf .
default_a0 : float
default_a1 : float
default_a2 : float
default_a3 : float
default_b3 : float
default_b4 : float
    Polyakov-loop polynomial potential parameters from
    https://arxiv.org/pdf/hep-ph/0506234.pdf .
default_Nc : float
    Quark color degeneracy factor.
default_M0 : float
    Vacuum mass of light and strange quarks. 
default_ms : float
    Strange quark current mass.
default_Lambda : float
    gcp_sea_lattice momentum hard-cutoff parameter.
default_L : float
    Cluster continuum energy scale (different from
    gcp_sea_lattice cutoff).
default_B : float
    Cluster bound state mass deficit per Jakobi coordinate.
default_M : dict[str, float]
    Cluster bound state masses
        'pi' : pion
        'K' : kaon
        'rho': rho meson
        'omega': omega meson
        'D': diquark
        'N': nucleon
        'T': tetraquark
        'F': four-quark
        'P': pentaquark
        'Q': five-quark
        'H': hexaquark
default_d : dict[str, float]
    Cluster degeneracy factors
        'pi' : pion
        'K' : kaon
        'rho': rho meson
        'omega': omega meson
        'D': diquark
        'N': nucleon
        'T': tetraquark
        'F': four-quark
        'P': pentaquark
        'Q': five-quark
        'H': hexaquark
default_N : dict[str, float]
    Cluster total number of degrees of freedom.
        'pi' : pion
        'K' : kaon
        'rho': rho meson
        'omega': omega meson
        'D': diquark
        'N': nucleon
        'T': tetraquark
        'F': four-quark
        'P': pentaquark
        'Q': five-quark
        'H': hexaquark
default_a : dict[str, float]
    Cluster net valence light quarks.
        'pi' : pion
        'K' : kaon
        'rho': rho meson
        'omega': omega meson
        'D': diquark
        'N': nucleon
        'T': tetraquark
        'F': four-quark
        'P': pentaquark
        'Q': five-quark
        'H': hexaquark
default_b : dict[str, float]
    Cluster net valence strange quarks.
        'pi' : pion
        'K' : kaon
        'rho': rho meson
        'omega': omega meson
        'D': diquark
        'N': nucleon
        'T': tetraquark
        'F': four-quark
        'P': pentaquark
        'Q': five-quark
        'H': hexaquark
default_s : dict[str, float]
    Cluster total number of strange degrees of freedom.
        'pi' : pion
        'K' : kaon
        'rho': rho meson
        'omega': omega meson
        'D': diquark
        'N': nucleon
        'T': tetraquark
        'F': four-quark
        'P': pentaquark
        'Q': five-quark
        'H': hexaquark
"""


import math


default_Tc0 = 154.#170.
default_delta_T = 26.
default_ml = 5.5
default_c = 300.0
default_d = 3.2
default_kappa = 0.012
default_Gs = 10.08e-6
default_a0 = 6.75
default_a1 = -1.95
default_a2 = 2.625
default_a3 = -7.44
default_b3 = 0.75
default_b4 = 7.5
default_T0 = 175.0#187.
default_Nc = 3.0
default_M0 = 400.
default_ms = 100.0
default_Lambda = 900.0
default_L = 50.0
default_B = 100
default_M = {
    'pi': 200.0,
    'K': 500.0,
    'rho': math.fsum([2.0*default_M0, 2.0*default_ml, -default_B]),
    'omega': math.fsum([2.0*default_M0, 2.0*default_ml, -default_B]),
    'D': math.fsum([2.0*default_M0, 2.0*default_ml, -default_B]),
    'N': math.fsum([3.0*default_M0, 3.0*default_ml, -2.0*default_B]),
    'T': math.fsum([4.0*default_M0, 4.0*default_ml, -3.0*default_B]),
    'F': math.fsum([4.0*default_M0, 4.0*default_ml, -3.0*default_B]),
    'P': math.fsum([5.0*default_M0, 5.0*default_ml, -4.0*default_B]),
    'Q': math.fsum([5.0*default_M0, 5.0*default_ml, -4.0*default_B]),
    'H': math.fsum([6.0*default_M0, 6.0*default_ml, -5.0*default_B]),
}
default_d = {
    'pi': (1.0*3.0*1.0),
    'K': (1.0*6.0*1.0),
    'rho': (3.0*3.0*1.0),
    'omega': (3.0*1.0*1.0),
    'D': (1.0*1.0*3.0)/2.0,
    'N': (2.0*2.0*1.0)/2.0,
    'T': math.fsum([(1.0*5.0*1.0)/2.0, (5.0*1.0*1.0)/2.0, (3.0*3.0*1.0)/2.0]),
    'F': (1.0*1.0*3.0)/2.0,
    'P': math.fsum([(4.0*2.0*1.0)/2.0, (2.0*4.0 *1.0)/2.0]),
    'Q': (2.0*1.0*3.0)/2.0,
    'H': math.fsum([(1.0*3.0 *1.0)/2.0, (3.0*1.0*1.0)/2.0])
}
default_N = {
    'pi': 2.0,
    'K': 2.0,
    'rho': 2.0,
    'omega': 2.0,
    'D': 2.0,
    'N': 3.0,
    'T': 4.0,
    'F': 4.0,
    'P': 5.0,
    'Q': 5.0,
    'H': 6.0
}
default_a = {
    'pi': 0,
    'K': 1,
    'rho': 0,
    'omega': 0,
    'D': 2,
    'N': 3,
    'T': 0,
    'F': 4,
    'P': 3,
    'Q': 5,
    'H': 6
}
default_b = {
    'pi': 0,
    'K': -1,
    'rho': 0,
    'omega': 0,
    'D': 0,
    'N': 0,
    'T': 0,
    'F': 0,
    'P': 0,
    'Q': 0,
    'H': 0
}
default_s = {
    'pi': 0,
    'K': 1,
    'rho': 0,
    'omega': 0,
    'D': 0,
    'N': 0,
    'T': 0,
    'F': 0,
    'P': 0,
    'Q': 0,
    'H': 0
}