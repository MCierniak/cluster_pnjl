import scipy.integrate

import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.distributions
import pnjl.aux_functions
import pnjl.defaults

#Grandcanonical potential (PNJL quark part)

def gcp_real(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:

    options = {'Nf' : pnjl.defaults.default_Nf, 'Nc' : pnjl.defaults.default_Nc, 'gcp_quark_debug_flag' : False}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    debug_flag = options['gcp_quark_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _mass, key):
        yp1 = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 1.0, **key)
        yp2 = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 2.0, **key)
        yp3 = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 3.0, **key)
        ym1 = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 1.0, **key)
        ym2 = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 2.0, **key)
        ym3 = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 3.0, **key)
        fp = pnjl.thermo.distributions.f_fermion_triplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        fm = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi, _Phib, **ym1, **ym2, **ym3)
        return ((p ** 4) / pnjl.aux_functions.En(p, _mass)) * (fp.real + fm.real)

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, mass, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in pnjl.thermo.gcp_quark.qcp_real did not succeed!")

    return -(Nf / (math.pi ** 2)) * (Nc / 3.0) * integral
def gcp_imag(T : float, mu : float, Phi : complex, Phib : complex, **kwargs) -> float:
    
    options = {'Nf' : pnjl.defaults.default_Nf, 'Nc' : pnjl.defaults.default_Nc, 'gcp_quark_debug_flag' : False}
    options.update(kwargs)

    Nf = options['Nf']
    Nc = options['Nc']
    debug_flag = options['gcp_quark_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _mass, key):
        yp1 = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 1.0, **key)
        yp2 = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 2.0, **key)
        yp3 = pnjl.aux_functions.y_plus(p, _T, _mu, _mass, 1.0, 3.0, **key)
        ym1 = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 1.0, **key)
        ym2 = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 2.0, **key)
        ym3 = pnjl.aux_functions.y_minus(p, _T, _mu, _mass, 1.0, 3.0, **key)
        fp = pnjl.thermo.distributions.f_fermion_triplet(_Phi, _Phib, **yp1, **yp2, **yp3)
        fm = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi, _Phib, **ym1, **ym2, **ym3)
        return ((p ** 4) / pnjl.aux_functions.En(p, _mass)) * (fp.imag + fm.imag)

    mass = pnjl.thermo.gcp_sea_lattice.M(T, mu, **kwargs)
    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, mass, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in pnjl.thermo.gcp_quark.qcp_imag did not succeed!")

    return -(Nf / (math.pi ** 2)) * (Nc / 3.0) * integral