"""### Description
Model parameters

### Functions
get_all_defaults
    Function returning a dictionary of all global variables.

### Globals
TC0 : float
DELTA_T : float
ML : float
    Parameters of the lattice-fit ansatz for the chiral condensate. 
    Data from https://arxiv.org/pdf/2012.12894.pdf (Tc0 = T0).
    Tc0 set to Nf=2+1. Alternative Tc0 for Nf=2 available at 
    https://arxiv.org/pdf/hep-lat/0501030.pdf .
C : float
D : float
    Parameters of the perturbative running coupling ansatz from
    https://arxiv.org/pdf/2012.12894.pdf (modified).
KAPPA : float
    Lattice expansion parameter for Tc=f(mu) from
    https://arxiv.org/pdf/1812.08235.pdf .
GS : float
    Scalar coupling constant from
    https://arxiv.org/pdf/hep-ph/0506234.pdf.
T0 : float
    Polyakov-loop polynomial potential pseudocritical temperature from
    https://arxiv.org/pdf/hep-ph/0506234.pdf (modified). Alternative
    Nf=2+1 result available at https://arxiv.org/pdf/0704.3234.pdf .
A0 : float
A1 : float
A2 : float
A3 : float
B3 : float
B4 : float
    Polyakov-loop polynomial potential parameters from
    https://arxiv.org/pdf/hep-ph/0506234.pdf .
NC : float
    Quark color degeneracy factor.
M0 : float
    Vacuum mass of light and strange quarks. 
MS : float
    Strange quark current mass.
M_L_VAC : float
    Vacuum value of the light quark mass
M_S_VAC : float
    Vacuum value of the strange quark mass
LAMBDA : float
    gcp_sea_lattice momentum hard-cutoff parameter.
L : float
    Cluster continuum energy scale (different from
    gcp_sea_lattice cutoff).
B : float
    Cluster bound state mass deficit per Jakobi coordinate.
MI : dict[str, float]
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
DI : dict[str, float]
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
NI : dict[str, float]
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
NET_QL : dict[str, float]
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
NET_QS : dict[str, float]
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
S : dict[str, float]
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
import numpy
import typing


TC0 = 154.#170.
DELTA_T = 26.
ML = 5.5
C = 300.0
D = 3.2
KAPPA = 0.012
GS = 10.08e-6
A0 = 6.75
A1 = -1.95
A2 = 2.625
A3 = -7.44
B3 = 0.75
B4 = 7.5
T0 = 175.#187.
NC = 3.0
M0 = 400.
MS = 100.0
M_L_VAC = math.fsum([0.5*M0*math.fsum([1.0, math.tanh(TC0/DELTA_T)]), ML])
M_S_VAC = math.fsum([0.5*M0*math.fsum([1.0, math.tanh(TC0/DELTA_T)]), MS])
LAMBDA = 900.0
L = 50.0
B = 100.0
MI = {
    'pi': 200.0,
    'K': 500.0,
    'rho': math.fsum([2.0*M0, 2.0*ML, -B]),
    'omega': math.fsum([2.0*M0, 2.0*ML, -B]),
    'D': math.fsum([2.0*M0, 2.0*ML, -B]),
    'N': math.fsum([3.0*M0, 3.0*ML, -2.0*B]),
    'T': math.fsum([4.0*M0, 4.0*ML, -3.0*B]),
    'F': math.fsum([4.0*M0, 4.0*ML, -3.0*B]),
    'P': math.fsum([5.0*M0, 5.0*ML, -4.0*B]),
    'Q': math.fsum([5.0*M0, 5.0*ML, -4.0*B]),
    'H': math.fsum([6.0*M0, 6.0*ML, -5.0*B]),
}
DI = {
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
NI = {
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
NET_QL = {
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
NET_QS = {
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
S = {
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


def get_all_defaults(split_dict: bool = False) -> typing.Dict:
    """### Description
    Function returning a dictionary of all global variables.

    ### Parameters
    split_dict: bool, optional
        Split dict globals into "global_name:dict_key" : "dict_val"
        pair.

    ### Returns
    vars: dict
        Dictionary of all global variables.
    """
    output = {}
    for key, value in globals().items():
        if isinstance(value, (float, int)):
            output[str(key)] = value
        elif isinstance(value, dict):
            if numpy.all([
                isinstance(ival, (float, int)) for ival in value.values()
            ]):
                if split_dict:
                    for ikey, ival in value.items():
                        output[str(key)+':'+str(ikey)] = ival
                else:
                    output[str(key)] = value
    return output

