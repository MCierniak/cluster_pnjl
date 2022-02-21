import scipy.integrate

import pnjl.thermo.distributions
import pnjl.aux_functions

#Mott-hadron resonance gas contribution to the GCP from https://arxiv.org/pdf/2012.12894.pdf

def gcp_real_a0(T : float, bmass : float, thmass : float, d : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def integrand(p, _T, _M, _Mth, key):
        yp_bound = pnjl.aux_functions.y_plus(p, _T, 0.0, _M, 0.0, 1.0, **key)
        ym_bound = pnjl.aux_functions.y_minus(p, _T, 0.0, _M, 0.0, 1.0, **key)
        yp_cont = pnjl.aux_functions.y_plus(p, _T, 0.0, _Mth, 0.0, 1.0, **key)
        ym_cont = pnjl.aux_functions.y_minus(p, _T, 0.0, _Mth, 0.0, 1.0, **key)
        bound = pnjl.thermo.distributions.f_baryon_singlet(**yp_bound).real + pnjl.thermo.distributions.f_baryon_singlet(**ym_bound).real
        cont = pnjl.thermo.distributions.f_baryon_singlet(**yp_cont).real + pnjl.thermo.distributions.f_baryon_singlet(**ym_cont).real
        return ((p ** 4) / pnjl.aux_functions.En(p, _M)) * bound - ((p ** 4) / pnjl.aux_functions.En(p, _Mth)) * cont

    integral, error = scipy.integrate.quad(integrand, 0.0, np.inf, args = (T, bmass, thmass, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in pnjl.thermo.gcp_cluster.gcp_real_a0 did not succeed!")

    return -(d / (4.0 * (math.pi ** 2))) * integral

#Grandcanonical potential (MHRG Beth-Uhlenbeck part)

def gcp_real(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, **kwargs) -> float:
    
    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, _a, key):
        bound = z_plus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).real + z_minus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).real
        scattering = z_plus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).real + z_minus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).real
        return ((p ** 2) / 3.0) * (bound - scattering) * np.heaviside(_Mth - _M, 0.5)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, bmass, thmass, a, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in Omega_N_real did not succeed!")

    if a == 0:
        return ((-1.0) ** a) * (1.0 / (4.0 * (math.pi ** 2))) * integral
    else:
        return ((-1.0) ** a) * (1.0 / (2.0 * (math.pi ** 2))) * integral
def gcp_imag(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, **kwargs) -> float:
    
    options = {'Omega_cluster_imag_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['Omega_cluster_imag_debug_flag']

    def integrand(p, _T, _mu, _Phi, _Phib, _M, _Mth, _a, key):
        bound = z_plus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).imag + z_minus(p, _T, _mu, _Phi, _Phib, _M, _a, **key).imag
        scattering = z_plus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).imag + z_minus(p, _T, _mu, _Phi, _Phib, _Mth, _a, **key).imag
        return ((p ** 2) / 3.0) * (bound - scattering) * np.heaviside(_Mth - _M, 0.5)

    integral, error = quad(integrand, 0.0, np.inf, args = (T, mu, Phi, Phib, bmass, thmass, a, kwargs))

    if ((abs(integral) > 1e-5 and abs(error / integral) > 0.01) or (abs(integral) <= 1e-5 and abs(error) > 0.01)) and debug_flag :
        print("The integration in Omega_N_real did not succeed!")

    if a == 0:
        return ((-1.0) ** a) * (1.0 / (4.0 * (math.pi ** 2))) * integral
    else:
        return ((-1.0) ** a) * (1.0 / (2.0 * (math.pi ** 2))) * integral