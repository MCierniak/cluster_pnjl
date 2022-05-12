import scipy.integrate
import numpy
import math

import pnjl.thermo.gcp_cluster.bound_step_continuum_step
import pnjl.thermo.distributions
import pnjl.aux_functions

#Mott-hadron resonance gas contribution to the GCP from https://arxiv.org/pdf/2012.12894.pdf , extended with arccos-cos continuum

def gcp_real_a0(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _thmass, _Ni, _L, key2):
        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, 0.0, _m, 0.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside(_thmass + nlambda - _m, hz)

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        return (_m / pnjl.aux_functions.En(_p, _m)) * pnjl.thermo.distributions.f_baryon_singlet(**yp) * continuum_factor
    def integrand1(p, _T, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _Mth + _ni * _l, args = (p, _T, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _bmass, _thmass, _Ni, _L, key2):
        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, 0.0, _m, 0.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        heavi5 = numpy.heaviside(_thmass + nlambda - _bmass, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((_thmass + nlambda - _bmass) / nlambda)
        return (_m / pnjl.aux_functions.En(_p, _m)) * pnjl.thermo.distributions.f_baryon_singlet(**yp) * continuum_factor
    def integrand2(p, _T, _Mi, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mth + _ni * _l, args = (p, _T, _Mi, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int

    step = 0.0
    integral = 0.0
    if thmass >= bmass:
        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a0(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
        integral, _ = scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, thmass, Ni, L, kwargs))
    elif thmass + Ni * L >= bmass:
        integral, _ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, bmass, thmass, Ni, L, kwargs))

    return step - (d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a0(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):
    #
    return 0.0

def gcp_real_a1_bm1(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _thmass, _Ni, _L, key2):
        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 1.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside(_thmass + nlambda - _m, hz)

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        return (_m / pnjl.aux_functions.En(_p, _m)) * pnjl.thermo.distributions.f_baryon_singlet(**yp) * continuum_factor
    def integrand1(p, _T, _mu, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _bmass, _thmass, _Ni, _L, key2):
        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 1.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        heavi5 = numpy.heaviside(_thmass + nlambda - _bmass, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((_thmass + nlambda - _bmass) / nlambda)
        return (_m / pnjl.aux_functions.En(_p, _m)) * pnjl.thermo.distributions.f_baryon_singlet(**yp) * continuum_factor
    def integrand2(p, _T, _mu, _Mi, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Mi, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int

    step = 0.0
    integral = 0.0
    if thmass >= bmass:
        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a0_bm1(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
        integral, _ = scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, thmass, Ni, L, kwargs))
    elif thmass + Ni * L >= bmass:
        integral, _ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, bmass, thmass, Ni, L, kwargs))

    return step - (d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a1_bm1(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):
    #
    return 0.0

def gcp_real_a2(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside(_thmass + nlambda - _m, hz)

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        heavi5 = numpy.heaviside(_thmass + nlambda - _bmass, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((_thmass + nlambda - _bmass) / nlambda)
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    
    step = 0.0
    integral = 0.0
    if thmass >= bmass:
        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a2(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
        integral, _ = scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))
    elif thmass + Ni * L >= bmass:
        integral, _ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, kwargs))

    return step - (d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a2(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside(_thmass + nlambda - _m, hz)

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 2.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        heavi5 = numpy.heaviside(_thmass + nlambda - _bmass, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((_thmass + nlambda - _bmass) / nlambda)
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    
    step = 0.0
    integral = 0.0
    if thmass >= bmass:
        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a2(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
        integral, _ = scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))
    elif thmass + Ni * L >= bmass:
        integral, _ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, kwargs))

    return step - (d / (2.0 * (math.pi ** 2))) * integral
def gcp_real_a3(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _thmass, _Ni, _L, key2):
        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 3.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 3.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside(_thmass + nlambda - _m, hz)

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        fpe = pnjl.thermo.distributions.f_fermion_singlet(**yp)
        fme = pnjl.thermo.distributions.f_fermion_singlet(**ym)
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _bmass, _thmass, _Ni, _L, key2):
        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 3.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 3.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        heavi5 = numpy.heaviside(_thmass + nlambda - _bmass, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((_thmass + nlambda - _bmass) / nlambda)
        fpe = pnjl.thermo.distributions.f_fermion_singlet(**yp)
        fme = pnjl.thermo.distributions.f_fermion_singlet(**ym)
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand2(p, _T, _mu, _Mi, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Mi, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    if thmass >= bmass:
        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a3(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
        integral, _ = scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, thmass, Ni, L, kwargs))
    elif thmass + Ni * L >= bmass:
        integral, _ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, bmass, thmass, Ni, L, kwargs))

    return step - (d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a3(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):
    #
    return 0.0
def gcp_real_a4(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside(_thmass + nlambda - _m, hz)

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        heavi5 = numpy.heaviside(_thmass + nlambda - _bmass, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((_thmass + nlambda - _bmass) / nlambda)
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    if thmass >= bmass:
        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a4(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
        integral, _ = scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))
    elif thmass + Ni * L >= bmass:
        integral, _ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, kwargs))

    return step - (d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a4(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside(_thmass + nlambda - _m, hz)

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 4.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        heavi5 = numpy.heaviside(_thmass + nlambda - _bmass, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((_thmass + nlambda - _bmass) / nlambda)
        fpe = pnjl.thermo.distributions.f_baryon_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_baryon_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    if thmass >= bmass:
        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a4(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
        integral, _ = scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))
    elif thmass + Ni * L >= bmass:
        integral, _ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, kwargs))

    return step - (d / (2.0 * (math.pi ** 2))) * integral
def gcp_real_a5(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"]  = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside(_thmass + nlambda - _m, hz)

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"]  = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        heavi5 = numpy.heaviside(_thmass + nlambda - _bmass, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((_thmass + nlambda - _bmass) / nlambda)
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    if thmass >= bmass:
        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a5(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
        integral, _ = scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))
    elif thmass + Ni * L >= bmass:
        integral, _ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, kwargs))

    return step - (d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a5(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"]  = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside(_thmass + nlambda - _m, hz)

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, key2):
        yp1 = {}
        yp2 = {}
        yp3 = {}
        ym1 = {}
        ym2 = {}
        ym3 = {}
        yp1["y_1_val"], yp1["y_1_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        yp2["y_2_val"], yp2["y_2_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        yp3["y_3_val"], yp3["y_3_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)
        ym1["y_1_val"], ym1["y_1_status"]  = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 1.0, **key2)
        ym2["y_2_val"], ym2["y_2_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 2.0, **key2)
        ym3["y_3_val"], ym3["y_3_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 5.0, 3.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        heavi5 = numpy.heaviside(_thmass + nlambda - _bmass, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((_thmass + nlambda - _bmass) / nlambda)
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    if thmass >= bmass:
        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a5(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
        integral, _ = scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))
    elif thmass + Ni * L >= bmass:
        integral, _ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, kwargs))

    return step - (d / (2.0 * (math.pi ** 2))) * integral
def gcp_real_a6(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _thmass, _Ni, _L, key2):
        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside(_thmass + nlambda - _m, hz)

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        fpe = pnjl.thermo.distributions.f_baryon_singlet(**yp)
        fme = pnjl.thermo.distributions.f_baryon_singlet(**ym)
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _bmass, _thmass, _Ni, _L, key2):
        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        frac1 = _m / nlambda
        frac2 = _thmass / nlambda

        heavi5 = numpy.heaviside(_thmass + nlambda - _bmass, hz)
        heavi7 = numpy.heaviside(math.pi * (frac1 - frac2), hz)
        heavi8 = numpy.heaviside(math.pi * (frac2 + 1.0 - frac1), hz)
        
        arccos_in = 2.0 * frac1 - 2.0 * frac2 - 1.0
        cos_in = math.pi * arccos_in / 2.0

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = heavi5 * heavi7 * heavi8 * math.cos(cos_in) * (arccos_el2 / math.pi) * ((_thmass + nlambda - _bmass) / nlambda)
        fpe = pnjl.thermo.distributions.f_baryon_singlet(**yp)
        fme = pnjl.thermo.distributions.f_baryon_singlet(**ym)
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand2(p, _T, _mu, _Mi, _Mth, _ni, _l, key):
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mth + _ni * _l, args = (p, _T, _mu, _Mi, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    if thmass >= bmass:
        step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a6(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
        integral, _ = scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, thmass, Ni, L, kwargs))
    elif thmass + Ni * L >= bmass:
        integral, _ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, bmass, thmass, Ni, L, kwargs))

    return step - (d / (2.0 * (math.pi ** 2))) * integral
def gcp_imag_a6(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, **kwargs):
    #
    return 0.0

#Grandcanonical potential (MHRG Beth-Uhlenbeck part)

def gcp_real(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, Ni : float, L : float, **kwargs) -> float:
    #
    return {(0, 0) : gcp_real_a0, (2, 0) : gcp_real_a2, (3, 0) : gcp_real_a3, (4, 0) : gcp_real_a4, (5, 0) : gcp_real_a5, (6, 0) : gcp_real_a6, (1, -1) : gcp_real_a1_bm1}[a, b](T, mu, Phi, Phib, bmass, thmass, d, Ni, L)
def gcp_imag(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, Ni : float, L : float, **kwargs) -> float:
    #
    return {(0, 0) : gcp_imag_a0, (2, 0) : gcp_imag_a2, (3, 0) : gcp_imag_a3, (4, 0) : gcp_imag_a4, (5, 0) : gcp_imag_a5, (6, 0) : gcp_imag_a6, (1, -1) : gcp_imag_a1_bm1}[a](T, mu, Phi, Phib, bmass, thmass, d, Ni, L)

#Extensive thermodynamic properties

def pressure(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, Ni : float, L : float, **kwargs):
    #
    return -gcp_real(T, mu, Phi, Phib, bmass, thmass, a, b, d, Ni, L, **kwargs)

#correct form of the mu vector would be mu_vec = [mu + 2 * h, mu + h, mu - h, mu - 2 * h] with some interval h
#or if mu = 0 then mu_vec = [h, 0.0]
#Phi/Phib/thmass vectors should correspond to Phi/Phib at the appropriate values of mu!
def bdensity(T : float, mu_vec : list, Phi_vec : list, Phib_vec : list, bmass : float, thmass : list, a : int, b : int, d : float, Ni : float, L : float, **kwargs):
    
    if len(mu_vec) == len(Phi_vec) and len(mu_vec) == len(Phib_vec) and len(mu_vec) == len(thmass):
        if len(mu_vec) == 4 and numpy.all(mu_vec[i] > mu_vec[i + 1] for i, el in enumerate(mu_vec[:-1])):
            h = mu_vec[0] - mu_vec[1]
            p_vec = [pressure(T, mu_el, Phi_el, Phib_el, bmass, Mth_el, a, b, d, Ni, L, **kwargs) for mu_el, Phi_el, Phib_el, Mth_el in zip(mu_vec, Phi_vec, Phib_vec, thmass)]
            return (1.0 / 3.0) * (p_vec[3] - 8.0 * p_vec[2] + 8.0 * p_vec[1] - p_vec[0]) / (12.0 * h)
        elif len(mu_vec) == 2 and mu_vec[0] > mu_vec[1]:
            h = mu_vec[0]
            p_vec = [pressure(T, mu_el, Phi_el, Phib_el, bmass, Mth_el, a, b, d, Ni, L, **kwargs) for mu_el, Phi_el, Phib_el, Mth_el in zip(mu_vec, Phi_vec, Phib_vec, thmass)]
            return (1.0 / 3.0) * (p_vec[0] - p_vec[1]) / h
        else:
            raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
    else:
        raise RuntimeError("Value vectors don't match!")

#correct form of the T vector would be T_vec = [T + 2 * h, T + h, T - h, T - 2 * h] with some interval h
#Phi/Phib vectors should correspond to Phi/Phib at the appropriate values of T!
def sdensity(T_vec : list, mu : float, Phi_vec : list, Phib_vec : list, bmass : float, thmass : list, a : int, b : int, d : float, Ni : float, L : float, **kwargs):
    
    if len(T_vec) == len(Phi_vec) and len(T_vec) == len(Phib_vec) and len(T_vec) == len(thmass):
        if len(T_vec) == 4 and numpy.all(T_vec[i] > T_vec[i + 1] for i, el in enumerate(T_vec[:-1])):
            h = T_vec[0] - T_vec[1]
            p_vec = [pressure(T_el, mu, Phi_el, Phib_el, bmass, Mth_el, a, b, d, Ni, L, **kwargs) for T_el, Phi_el, Phib_el, Mth_el in zip(T_vec, Phi_vec, Phib_vec, thmass)]
            return (1.0 / 3.0) * (p_vec[3] - 8.0 * p_vec[2] + 8.0 * p_vec[1] - p_vec[0]) / (12.0 * h)
        elif len(T_vec) == 2 and T_vec[0] > T_vec[1]:
            h = T_vec[0]
            p_vec = [pressure(T_el, mu, Phi_el, Phib_el, bmass, Mth_el, a, b, d, Ni, L, **kwargs) for T_el, Phi_el, Phib_el, Mth_el in zip(T_vec, Phi_vec, Phib_vec, thmass)]
            return (1.0 / 3.0) * (p_vec[0] - p_vec[1]) / h
        else:
            raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
    else:
        raise RuntimeError("Value vectors don't match!")