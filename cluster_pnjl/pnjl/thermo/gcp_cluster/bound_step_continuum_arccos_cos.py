import numpy
import math

import scipy.integrate

import pnjl.thermo.gcp_cluster.bound_step_continuum_step
import pnjl.thermo.gcp_sea_lattice
import pnjl.thermo.distributions
import pnjl.aux_functions
import pnjl.defaults

#Mott-hadron resonance gas contribution to the GCP from https://arxiv.org/pdf/2012.12894.pdf ,
#extended with arccos-cos continuum

default_T_slope = 6.0
default_continuum_boost = 1.0
default_nlambda2_par = 2.5
default_factor2_exponent = 1.0

def gcp_real_a0(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _thmass, _Ni, _L, key2):

        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, 0.0, _m, 0.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside((_Ni * M0 + _Ni * ml) + nlambda - _m, hz)
        arccos_in = -2.0 * (_m / (_thmass - M0 * _Ni - ml * _Ni - nlambda)) + ((_thmass + M0 * _Ni + ml * _Ni + nlambda) / (_thmass - M0 * _Ni - ml * _Ni - nlambda))

        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        return (_m / pnjl.aux_functions.En(_p, _m)) * pnjl.thermo.distributions.f_boson_singlet(**yp) * continuum_factor
    def integrand1(p, _T, _Mth, _ni, _l, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _ni * (M0 + ml) + _ni * _l, args = (p, _T, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _bmass, _thmass, _Ni, _L, _T_crit, _reg, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, 0.0, _m, 0.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L
        nlambda2 = default_nlambda2_par * nlambda
        M_gap = (_Ni * M0 + _Ni * ml) - _bmass
        T_slope = default_T_slope

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (_T2 - _T_crit), hz)
        heavi7 = numpy.heaviside(math.pi * (_m - _thmass), hz)
        heavi8 = numpy.heaviside(math.pi * (((_bmass + T_slope * (_T2 - _T_crit) + M_gap) / nlambda) + 1.0 - (_m / nlambda)), hz)
        arccos_in = (2.0 * _m / (_bmass - _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap)) + ((_bmass + _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap) / (_thmass - _bmass - nlambda - T_slope * (_T2 - _T_crit) - M_gap))
        cos_in = math.pi * arccos_in / 2.0
        factor = default_continuum_boost * ((nlambda2 - T_slope * (_T2 - _T_crit)) / nlambda2)
        factor2 = default_continuum_boost * (((nlambda - T_slope * (_T2 - _T_crit)) ** default_factor2_exponent) / (nlambda ** default_factor2_exponent))

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = 0.0
        if factor2 > 0.0:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
        else:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        return (_m / pnjl.aux_functions.En(_p, _m)) * pnjl.thermo.distributions.f_boson_singlet(**yp) * continuum_factor * _reg
    def integrand2(p, _T, _Mi, _Mth, _ni, _l, _T_c, _regulator, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        T_slope = default_T_slope
        M_gap = (_ni * M0 + _ni * ml) - _Mi
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mi + T_slope * (_T - _T_c) + _ni * _l + M_gap, args = (p, _T, _Mi, _Mth, _ni, _l, _T_c, _regulator, key))
        return (p ** 2) * inner_int

    step = 0.0
    integral = 0.0
    T_slope = default_T_slope
    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a0(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
    if thmass >= bmass:
        integral += (d / (2.0 * (math.pi ** 2))) * scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, thmass, Ni, L, kwargs))[0]
    if thmass < bmass and default_nlambda2_par * Ni * L >= T_slope * (T - T_crit):
        integ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, bmass, thmass, Ni, L, T_crit, regulator, kwargs))[0]
        integral += (d / (2.0 * (math.pi ** 2))) * integ - ((regulator2 * T_slope * (T - T_crit)) / (default_nlambda2_par * Ni * L )) + regulator2

    return step - integral
def gcp_imag_a0(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, regulator : float, regulator2 : float, **kwargs):
    #
    return 0.0

def gcp_real_a1_bm1(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _thmass, _Ni, _L, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']
        ms = pnjl.defaults.default_ms

        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, 0.0, _m, 1.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L

        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside((_Ni - 1.0) * (M0 + ml) + (M0 + ms) + nlambda - _m, hz)
        arccos_in = -2.0 * (_m / (_thmass + ml - ms - M0 * _Ni - ml * _Ni - nlambda)) + ((_thmass + ms - ml + _Ni * (M0 + ml) + nlambda) / (_thmass - _Ni * (M0 + ml) - ms + ml - nlambda))

        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        return (_m / pnjl.aux_functions.En(_p, _m)) * pnjl.thermo.distributions.f_boson_singlet(**yp) * continuum_factor
    def integrand1(p, _T, _Mth, _ni, _l, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        ms = pnjl.defaults.default_ms
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, (_ni - 1.0) * (M0 + ml) + (M0 + ms) + _ni * _l, args = (p, _T, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _bmass, _thmass, _Ni, _L, _T_crit, _reg, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']
        ms = pnjl.defaults.default_ms

        yp = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, 0.0, _m, 1.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L
        nlambda2 = default_nlambda2_par * nlambda
        M_gap = (_Ni - 1.0) * (M0 + ml) + (M0 + ms) - _bmass
        T_slope = default_T_slope

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (_T2 - _T_crit), hz)
        heavi7 = numpy.heaviside(math.pi * (_m - _thmass), hz)
        heavi8 = numpy.heaviside(math.pi * (((_bmass + T_slope * (_T2 - _T_crit) + M_gap) / nlambda) + 1.0 - (_m / nlambda)), hz)
        arccos_in = (2.0 * _m / (_bmass - _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap)) + ((_bmass + _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap) / (_thmass - _bmass - nlambda - T_slope * (_T2 - _T_crit) - M_gap))
        cos_in = math.pi * arccos_in / 2.0
        factor = default_continuum_boost * ((nlambda2 - T_slope * (_T2 - _T_crit)) / nlambda2)
        factor2 = default_continuum_boost * (((nlambda - T_slope * (_T2 - _T_crit)) ** default_factor2_exponent) / (nlambda ** default_factor2_exponent))

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = 0.0
        if factor2 > 0.0:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
        else:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        return (_m / pnjl.aux_functions.En(_p, _m)) * pnjl.thermo.distributions.f_boson_singlet(**yp) * continuum_factor * _reg
    def integrand2(p, _T, _Mi, _Mth, _ni, _l, _T_c, _regulator, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        ms = pnjl.defaults.default_ms
        T_slope = default_T_slope
        M_gap = (_ni - 1.0) * (M0 + ml) + (M0 + ms) - _Mi
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mi + T_slope * (_T - _T_c) + _ni * _l + M_gap, args = (p, _T, _Mi, _Mth, _ni, _l, _T_c, _regulator, key))
        return (p ** 2) * inner_int

    step = 0.0
    integral = 0.0
    T_slope = default_T_slope
    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a1_bm1(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
    if thmass > bmass:
        integral += (d / (2.0 * (math.pi ** 2))) * scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, thmass, Ni, L, kwargs))[0]
    if thmass <= bmass and default_nlambda2_par * Ni * L >= T_slope * (T - T_crit):
        integ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, bmass, thmass, Ni, L, T_crit, regulator, kwargs))[0]
        integral += (d / (2.0 * (math.pi ** 2))) * integ - ((regulator2 * T_slope * (T - T_crit)) / (default_nlambda2_par * Ni * L )) + regulator2

    return step - integral
def gcp_imag_a1_bm1(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, regulator : float, regulator2 : float, **kwargs):
    #
    return 0.0

def gcp_real_a2(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

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
        heavi4 = numpy.heaviside((_Ni * M0 + _Ni * ml) + nlambda - _m, hz)
        arccos_in = -2.0 * (_m / (_thmass - M0 * _Ni - ml * _Ni - nlambda)) + ((_thmass + M0 * _Ni + ml * _Ni + nlambda) / (_thmass - M0 * _Ni - ml * _Ni - nlambda))

        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _ni * (M0 + ml) + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, _T_crit, _reg, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

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
        nlambda2 = default_nlambda2_par * nlambda
        M_gap = (_Ni * M0 + _Ni * ml) - _bmass
        T_slope = default_T_slope

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (_T2 - _T_crit), hz)
        heavi7 = numpy.heaviside(math.pi * (_m - _thmass), hz)
        heavi8 = numpy.heaviside(math.pi * (((_bmass + T_slope * (_T2 - _T_crit) + M_gap) / nlambda) + 1.0 - (_m / nlambda)), hz)
        arccos_in = (2.0 * _m / (_bmass - _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap)) + ((_bmass + _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap) / (_thmass - _bmass - nlambda - T_slope * (_T2 - _T_crit) - M_gap))
        cos_in = math.pi * arccos_in / 2.0
        factor = default_continuum_boost * ((nlambda2 - T_slope * (_T2 - _T_crit)) / nlambda2)
        factor2 = default_continuum_boost * (((nlambda - T_slope * (_T2 - _T_crit)) ** default_factor2_exponent) / (nlambda ** default_factor2_exponent))

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = 0.0
        if factor2 > 0.0:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
        else:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor * _reg
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        T_slope = default_T_slope
        M_gap = (_ni * M0 + _ni * ml) - _Mi
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mi + T_slope * (_T - _T_c) + _ni * _l + M_gap, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key))
        return (p ** 2) * inner_int
    
    step = 0.0
    integral = 0.0
    T_slope = default_T_slope
    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a2(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
    if thmass >= bmass:
        integral += (d / (2.0 * (math.pi ** 2))) * scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))[0]
    if thmass < bmass and default_nlambda2_par * Ni * L >= T_slope * (T - T_crit):
        integ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, T_crit, regulator, kwargs))[0]
        integral += (d / (2.0 * (math.pi ** 2))) * integ - ((regulator2 * T_slope * (T - T_crit)) / (default_nlambda2_par * Ni * L )) + regulator2

    return step - integral
def gcp_imag_a2(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):

        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

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
        heavi4 = numpy.heaviside((_Ni * M0 + _Ni * ml) + nlambda - _m, hz)
        arccos_in = -2.0 * (_m / (_thmass - M0 * _Ni - ml * _Ni - nlambda)) + ((_thmass + M0 * _Ni + ml * _Ni + nlambda) / (_thmass - M0 * _Ni - ml * _Ni - nlambda))

        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _ni * (M0 + ml) + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, _T_crit, _reg, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

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
        nlambda2 = default_nlambda2_par * nlambda
        M_gap = (_Ni * M0 + _Ni * ml) - _bmass
        T_slope = default_T_slope

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (_T2 - _T_crit), hz)
        heavi7 = numpy.heaviside(math.pi * (_m - _thmass), hz)
        heavi8 = numpy.heaviside(math.pi * (((_bmass + T_slope * (_T2 - _T_crit) + M_gap) / nlambda) + 1.0 - (_m / nlambda)), hz)
        arccos_in = (2.0 * _m / (_bmass - _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap)) + ((_bmass + _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap) / (_thmass - _bmass - nlambda - T_slope * (_T2 - _T_crit) - M_gap))
        cos_in = math.pi * arccos_in / 2.0
        factor = default_continuum_boost * ((nlambda2 - T_slope * (_T2 - _T_crit)) / nlambda2)
        factor2 = default_continuum_boost * (((nlambda - T_slope * (_T2 - _T_crit)) ** default_factor2_exponent) / (nlambda ** default_factor2_exponent))

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = 0.0
        if factor2 > 0.0:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
        else:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor * _reg
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        T_slope = default_T_slope
        M_gap = (_ni * M0 + _ni * ml) - _Mi
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mi + T_slope * (_T - _T_c) + _ni * _l + M_gap, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key))
        return (p ** 2) * inner_int
    
    step = 0.0
    integral = 0.0
    T_slope = default_T_slope
    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a2(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
    if thmass >= bmass:
        integral += (d / (2.0 * (math.pi ** 2))) * scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))[0]
    if thmass < bmass and default_nlambda2_par * Ni * L >= T_slope * (T - T_crit):
        integ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, T_crit, regulator, kwargs))[0]
        integral += (d / (2.0 * (math.pi ** 2))) * integ - ((regulator2 * T_slope * (T - T_crit)) / (default_nlambda2_par * Ni * L )) + regulator2

    return step - integral
def gcp_real_a3(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _thmass, _Ni, _L, key2):

        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 3.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 3.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L
        
        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside((_Ni * M0 + _Ni * ml) + nlambda - _m, hz)
        arccos_in = -2.0 * (_m / (_thmass - M0 * _Ni - ml * _Ni - nlambda)) + ((_thmass + M0 * _Ni + ml * _Ni + nlambda) / (_thmass - M0 * _Ni - ml * _Ni - nlambda))

        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_fermion_singlet(**yp)
        fme = pnjl.thermo.distributions.f_fermion_singlet(**ym)
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Mth, _ni, _l, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _ni * (M0 + ml) + _ni * _l, args = (p, _T, _mu, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _bmass, _thmass, _Ni, _L, _T_crit, _reg, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']
        
        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 3.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 3.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L
        nlambda2 = default_nlambda2_par * nlambda
        M_gap = (_Ni * M0 + _Ni * ml) - _bmass
        T_slope = default_T_slope

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (_T2 - _T_crit), hz)
        heavi7 = numpy.heaviside(math.pi * (_m - _thmass), hz)
        heavi8 = numpy.heaviside(math.pi * (((_bmass + T_slope * (_T2 - _T_crit) + M_gap) / nlambda) + 1.0 - (_m / nlambda)), hz)
        arccos_in = (2.0 * _m / (_bmass - _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap)) + ((_bmass + _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap) / (_thmass - _bmass - nlambda - T_slope * (_T2 - _T_crit) - M_gap))
        cos_in = math.pi * arccos_in / 2.0
        factor = default_continuum_boost * ((nlambda2 - T_slope * (_T2 - _T_crit)) / nlambda2)
        factor2 = default_continuum_boost * (((nlambda - T_slope * (_T2 - _T_crit)) ** default_factor2_exponent) / (nlambda ** default_factor2_exponent))

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = 0.0
        if factor2 > 0.0:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
        else:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_fermion_singlet(**yp)
        fme = pnjl.thermo.distributions.f_fermion_singlet(**ym)
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor * _reg
    def integrand2(p, _T, _mu, _Mi, _Mth, _ni, _l, _T_c, _regulator, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        T_slope = default_T_slope
        M_gap = (_ni * M0 + _ni * ml) - _Mi
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mi + T_slope * (_T - _T_c) + _ni * _l + M_gap, args = (p, _T, _mu, _Mi, _Mth, _ni, _l, _T_c, _regulator, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    T_slope = default_T_slope
    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a3(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
    if thmass >= bmass:
        integral += (d / (2.0 * (math.pi ** 2))) * scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, thmass, Ni, L, kwargs))[0]
    if thmass < bmass and default_nlambda2_par * Ni * L >= T_slope * (T - T_crit):
        integ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, bmass, thmass, Ni, L, T_crit, regulator, kwargs))[0]
        integral += (d / (2.0 * (math.pi ** 2))) * integ - ((regulator2 * T_slope * (T - T_crit)) / (default_nlambda2_par * Ni * L )) + regulator2

    return step - integral
def gcp_imag_a3(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):
    #
    return 0.0
def gcp_real_a4(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

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
        heavi4 = numpy.heaviside((_Ni * M0 + _Ni * ml) + nlambda - _m, hz)
        arccos_in = -2.0 * (_m / (_thmass - M0 * _Ni - ml * _Ni - nlambda)) + ((_thmass + M0 * _Ni + ml * _Ni + nlambda) / (_thmass - M0 * _Ni - ml * _Ni - nlambda))

        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _ni * (M0 + ml) + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, _T_crit, _reg, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

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
        nlambda2 = default_nlambda2_par * nlambda
        M_gap = (_Ni * M0 + _Ni * ml) - _bmass
        T_slope = default_T_slope

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (_T2 - _T_crit), hz)
        heavi7 = numpy.heaviside(math.pi * (_m - _thmass), hz)
        heavi8 = numpy.heaviside(math.pi * (((_bmass + T_slope * (_T2 - _T_crit) + M_gap) / nlambda) + 1.0 - (_m / nlambda)), hz)
        arccos_in = (2.0 * _m / (_bmass - _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap)) + ((_bmass + _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap) / (_thmass - _bmass - nlambda - T_slope * (_T2 - _T_crit) - M_gap))
        cos_in = math.pi * arccos_in / 2.0
        factor = default_continuum_boost * ((nlambda2 - T_slope * (_T2 - _T_crit)) / nlambda2)
        factor2 = default_continuum_boost * (((nlambda - T_slope * (_T2 - _T_crit)) ** default_factor2_exponent) / (nlambda ** default_factor2_exponent))

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = 0.0
        if factor2 > 0.0:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
        else:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor * _reg
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        T_slope = default_T_slope
        M_gap = (_ni * M0 + _ni * ml) - _Mi
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mi + T_slope * (_T - _T_c) + _ni * _l + M_gap, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    T_slope = default_T_slope
    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a4(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
    if thmass >= bmass:
        integral += (d / (2.0 * (math.pi ** 2))) * scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))[0]
    if thmass < bmass and default_nlambda2_par * Ni * L >= T_slope * (T - T_crit):
        integ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, T_crit, regulator, kwargs))[0]
        integral += (d / (2.0 * (math.pi ** 2))) * integ - ((regulator2 * T_slope * (T - T_crit)) / (default_nlambda2_par * Ni * L )) + regulator2

    return step - integral
def gcp_imag_a4(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

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
        heavi4 = numpy.heaviside((_Ni * M0 + _Ni * ml) + nlambda - _m, hz)
        arccos_in = -2.0 * (_m / (_thmass - M0 * _Ni - ml * _Ni - nlambda)) + ((_thmass + M0 * _Ni + ml * _Ni + nlambda) / (_thmass - M0 * _Ni - ml * _Ni - nlambda))

        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _ni * (M0 + ml) + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, _T_crit, _reg, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']
        
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
        nlambda2 = default_nlambda2_par * nlambda
        M_gap = (_Ni * M0 + _Ni * ml) - _bmass
        T_slope = default_T_slope

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (_T2 - _T_crit), hz)
        heavi7 = numpy.heaviside(math.pi * (_m - _thmass), hz)
        heavi8 = numpy.heaviside(math.pi * (((_bmass + T_slope * (_T2 - _T_crit) + M_gap) / nlambda) + 1.0 - (_m / nlambda)), hz)
        arccos_in = (2.0 * _m / (_bmass - _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap)) + ((_bmass + _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap) / (_thmass - _bmass - nlambda - T_slope * (_T2 - _T_crit) - M_gap))
        cos_in = math.pi * arccos_in / 2.0
        factor = default_continuum_boost * ((nlambda2 - T_slope * (_T2 - _T_crit)) / nlambda2)
        factor2 = default_continuum_boost * (((nlambda - T_slope * (_T2 - _T_crit)) ** default_factor2_exponent) / (nlambda ** default_factor2_exponent))

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = 0.0
        if factor2 > 0.0:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
        else:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_boson_antitriplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_boson_triplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor * _reg
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        T_slope = default_T_slope
        M_gap = (_ni * M0 + _ni * ml) - _Mi
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mi + T_slope * (_T - _T_c) + _ni * _l + M_gap, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    T_slope = default_T_slope
    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a4(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
    if thmass >= bmass:
        integral += (d / (2.0 * (math.pi ** 2))) * scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))[0]
    if thmass < bmass and default_nlambda2_par * Ni * L >= T_slope * (T - T_crit):
        integ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, T_crit, regulator, kwargs))[0]
        integral += (d / (2.0 * (math.pi ** 2))) * integ - ((regulator2 * T_slope * (T - T_crit)) / (default_nlambda2_par * Ni * L )) + regulator2

    return step - integral
def gcp_real_a5(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

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
        heavi4 = numpy.heaviside((_Ni * M0 + _Ni * ml) + nlambda - _m, hz)
        arccos_in = -2.0 * (_m / (_thmass - M0 * _Ni - ml * _Ni - nlambda)) + ((_thmass + M0 * _Ni + ml * _Ni + nlambda) / (_thmass - M0 * _Ni - ml * _Ni - nlambda))

        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _ni * (M0 + ml) + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, _T_crit, _reg, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']
        
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
        nlambda2 = default_nlambda2_par * nlambda
        M_gap = (_Ni * M0 + _Ni * ml) - _bmass
        T_slope = default_T_slope

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (_T2 - _T_crit), hz)
        heavi7 = numpy.heaviside(math.pi * (_m - _thmass), hz)
        heavi8 = numpy.heaviside(math.pi * (((_bmass + T_slope * (_T2 - _T_crit) + M_gap) / nlambda) + 1.0 - (_m / nlambda)), hz)
        arccos_in = (2.0 * _m / (_bmass - _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap)) + ((_bmass + _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap) / (_thmass - _bmass - nlambda - T_slope * (_T2 - _T_crit) - M_gap))
        cos_in = math.pi * arccos_in / 2.0
        factor = default_continuum_boost * ((nlambda2 - T_slope * (_T2 - _T_crit)) / nlambda2)
        factor2 = default_continuum_boost * (((nlambda - T_slope * (_T2 - _T_crit)) ** default_factor2_exponent) / (nlambda ** default_factor2_exponent))

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = 0.0
        if factor2 > 0.0:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
        else:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).real
        fme = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).real
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor * _reg
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        T_slope = default_T_slope
        M_gap = (_ni * M0 + _ni * ml) - _Mi
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mi + T_slope * (_T - _T_c) + _ni * _l + M_gap, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    T_slope = default_T_slope
    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a5(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
    if thmass >= bmass:
        integral += (d / (2.0 * (math.pi ** 2))) * scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))[0]
    if thmass < bmass and default_nlambda2_par * Ni * L >= T_slope * (T - T_crit):
        integ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, T_crit, regulator, kwargs))[0]
        integral += (d / (2.0 * (math.pi ** 2))) * integ - ((regulator2 * T_slope * (T - T_crit)) / (default_nlambda2_par * Ni * L )) + regulator2

    return step - integral
def gcp_imag_a5(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _Phi2, _Phib2, _thmass, _Ni, _L, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

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
        heavi4 = numpy.heaviside((_Ni * M0 + _Ni * ml) + nlambda - _m, hz)
        arccos_in = -2.0 * (_m / (_thmass - M0 * _Ni - ml * _Ni - nlambda)) + ((_thmass + M0 * _Ni + ml * _Ni + nlambda) / (_thmass - M0 * _Ni - ml * _Ni - nlambda))

        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor
    def integrand1(p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _ni * (M0 + ml) + _ni * _l, args = (p, _T, _mu, _Phi, _Phib, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _Phi2, _Phib2, _bmass, _thmass, _Ni, _L, _T_crit, _reg, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

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
        nlambda2 = default_nlambda2_par * nlambda
        M_gap = (_Ni * M0 + _Ni * ml) - _bmass
        T_slope = default_T_slope

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (_T2 - _T_crit), hz)
        heavi7 = numpy.heaviside(math.pi * (_m - _thmass), hz)
        heavi8 = numpy.heaviside(math.pi * (((_bmass + T_slope * (_T2 - _T_crit) + M_gap) / nlambda) + 1.0 - (_m / nlambda)), hz)
        arccos_in = (2.0 * _m / (_bmass - _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap)) + ((_bmass + _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap) / (_thmass - _bmass - nlambda - T_slope * (_T2 - _T_crit) - M_gap))
        cos_in = math.pi * arccos_in / 2.0
        factor = default_continuum_boost * ((nlambda2 - T_slope * (_T2 - _T_crit)) / nlambda2)
        factor2 = default_continuum_boost * (((nlambda - T_slope * (_T2 - _T_crit)) ** default_factor2_exponent) / (nlambda ** default_factor2_exponent))

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = 0.0
        if factor2 > 0.0:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
        else:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        if continuum_factor > 1.0:
            continuum_factor = 1.0
        fpe = pnjl.thermo.distributions.f_fermion_triplet(_Phi2, _Phib2, **yp1, **yp2, **yp3).imag
        fme = pnjl.thermo.distributions.f_fermion_antitriplet(_Phi2, _Phib2, **ym1, **ym2, **ym3).imag
        return (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor * _reg
    def integrand2(p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        T_slope = default_T_slope
        M_gap = (_ni * M0 + _ni * ml) - _Mi
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mi + T_slope * (_T - _T_c) + _ni * _l + M_gap, args = (p, _T, _mu, _Phi, _Phib, _Mi, _Mth, _ni, _l, _T_c, _regulator, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    T_slope = default_T_slope
    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a5(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
    if thmass >= bmass:
        integral += (d / (2.0 * (math.pi ** 2))) * scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, Phi, Phib, thmass, Ni, L, kwargs))[0]
    if thmass < bmass and default_nlambda2_par * Ni * L >= T_slope * (T - T_crit):
        integ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, Phi, Phib, bmass, thmass, Ni, L, T_crit, regulator, kwargs))[0]
        integral += (d / (2.0 * (math.pi ** 2))) * integ - ((regulator2 * T_slope * (T - T_crit)) / (default_nlambda2_par * Ni * L )) + regulator2

    return step - integral
def gcp_real_a6(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    options = {'gcp_cluster_debug_flag' : False}
    options.update(kwargs)

    debug_flag = options['gcp_cluster_debug_flag']

    def mass_integrand1(_m, _p, _T2, _mu2, _thmass, _Ni, _L, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L
        
        heavi3 = numpy.heaviside((_m ** 2) - (_thmass ** 2), hz)
        heavi4 = numpy.heaviside((_Ni * M0 + _Ni * ml) + nlambda - _m, hz)
        arccos_in = -2.0 * (_m / (_thmass - M0 * _Ni - ml * _Ni - nlambda)) + ((_thmass + M0 * _Ni + ml * _Ni + nlambda) / (_thmass - M0 * _Ni - ml * _Ni - nlambda))

        arccos_el = math.acos(arccos_in)

        continuum_factor = heavi3 * heavi4 *arccos_el / math.pi
        if continuum_factor > 1.0:
            continuum_factor = 1.0

        fpe = pnjl.thermo.distributions.f_boson_singlet(**yp)
        fme = pnjl.thermo.distributions.f_boson_singlet(**ym)
        out = (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor

        return out
    def integrand1(p, _T, _mu, _Mth, _ni, _l, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        inner_int, _ = scipy.integrate.quad(mass_integrand1, _Mth, _ni * (M0 + ml) + _ni * _l, args = (p, _T, _mu, _Mth, _ni, _l, key))
        return (p ** 2) * inner_int
    def mass_integrand2(_m, _p, _T2, _mu2, _bmass, _thmass, _Ni, _L, _T_crit, _reg, key2):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key2)
        M0 = options['M0']
        ml = options['ml']

        yp = {}
        ym = {}
        yp["y_val"], yp["y_status"] = pnjl.aux_functions.y_plus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)
        ym["y_val"], ym["y_status"] = pnjl.aux_functions.y_minus(_p, _T2, _mu2, _m, 6.0, 1.0, **key2)

        hz = 0.5
        nlambda = _Ni * _L
        nlambda2 = default_nlambda2_par * nlambda
        M_gap = (_Ni * M0 + _Ni * ml) - _bmass
        T_slope = default_T_slope

        heavi5 = numpy.heaviside(nlambda2 - T_slope * (_T2 - _T_crit), hz)
        heavi7 = numpy.heaviside(math.pi * (_m - _thmass), hz)
        heavi8 = numpy.heaviside(math.pi * (((_bmass + T_slope * (_T2 - _T_crit) + M_gap) / nlambda) + 1.0 - (_m / nlambda)), hz)
        arccos_in = (2.0 * _m / (_bmass - _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap)) + ((_bmass + _thmass + nlambda + T_slope * (_T2 - _T_crit) + M_gap) / (_thmass - _bmass - nlambda - T_slope * (_T2 - _T_crit) - M_gap))
        cos_in = math.pi * arccos_in / 2.0
        factor = default_continuum_boost * ((nlambda2 - T_slope * (_T2 - _T_crit)) / nlambda2)
        factor2 = default_continuum_boost * (((nlambda - T_slope * (_T2 - _T_crit)) ** default_factor2_exponent) / (nlambda ** default_factor2_exponent))

        arccos_el2 = math.acos(arccos_in)

        continuum_factor = 0.0
        if factor2 > 0.0:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * (factor2 * (arccos_el2 / math.pi) + (1.0 - factor2) * math.cos(cos_in) * (arccos_el2 / math.pi))
        else:
            continuum_factor = heavi5 * heavi7 * heavi8 * factor * math.cos(cos_in) * (arccos_el2 / math.pi)
        if continuum_factor > 1.0:
            continuum_factor = 1.0

        fpe = pnjl.thermo.distributions.f_boson_singlet(**yp)
        fme = pnjl.thermo.distributions.f_boson_singlet(**ym)
        out = (_m / pnjl.aux_functions.En(_p, _m)) * (fpe + fme) * continuum_factor * _reg

        return out
    def integrand2(p, _T, _mu, _Mi, _Mth, _ni, _l, _T_c, _regulator, key):
        options = {'M0' : pnjl.defaults.default_M0, 'ml' : pnjl.defaults.default_ml}
        options.update(key)
        M0 = options['M0']
        ml = options['ml']
        T_slope = default_T_slope
        M_gap = (_ni * M0 + _ni * ml) - _Mi
        inner_int, _ = scipy.integrate.quad(mass_integrand2, _Mth, _Mi + T_slope * (_T - _T_c) + _ni * _l + M_gap, args = (p, _T, _mu, _Mi, _Mth, _ni, _l, _T_c, _regulator, key))
        return (p ** 2) * inner_int

    integral = 0.0
    step = 0.0
    T_slope = default_T_slope
    step = pnjl.thermo.gcp_cluster.bound_step_continuum_step.gcp_real_a6(T, mu, Phi, Phib, bmass, thmass, d, **kwargs)
    if thmass >= bmass:
        integral += (d / (2.0 * (math.pi ** 2))) * scipy.integrate.quad(integrand1, 0.0, math.inf, args = (T, mu, thmass, Ni, L, kwargs))[0]
    if thmass < bmass and default_nlambda2_par * Ni * L >= T_slope * (T - T_crit):
        integ = scipy.integrate.quad(integrand2, 0.0, math.inf, args = (T, mu, bmass, thmass, Ni, L, T_crit, regulator, kwargs))[0]
        integral += (d / (2.0 * (math.pi ** 2))) * integ - ((regulator2 * T_slope * (T - T_crit)) / (default_nlambda2_par * Ni * L )) + regulator2

    return step - integral
def gcp_imag_a6(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):
    #
    return 0.0

#Grandcanonical potential (MHRG Beth-Uhlenbeck part)

def gcp_real(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs) -> float:
    #
    return {(0, 0) : gcp_real_a0, (2, 0) : gcp_real_a2, (3, 0) : gcp_real_a3, (4, 0) : gcp_real_a4, (5, 0) : gcp_real_a5, (6, 0) : gcp_real_a6, (1, -1) : gcp_real_a1_bm1}[a, b](T, mu, Phi, Phib, bmass, thmass, d, Ni, L, T_crit, regulator, regulator2, **kwargs)
def gcp_imag(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs) -> float:
    #
    return {(0, 0) : gcp_imag_a0, (2, 0) : gcp_imag_a2, (3, 0) : gcp_imag_a3, (4, 0) : gcp_imag_a4, (5, 0) : gcp_imag_a5, (6, 0) : gcp_imag_a6, (1, -1) : gcp_imag_a1_bm1}[a](T, mu, Phi, Phib, bmass, thmass, d, Ni, L, T_crit, regulator, regulator2, **kwargs)

#Thermodynamic solver
def calc_PL(
    T : float, mu : float, phi_re0 : float, phi_im0 : float,
    light_kwargs = None, strange_kwargs = None, gluon_kwargs = None, perturbative_kwargs = None,
    with_clusters : bool = True) -> (float, float):
    
    if light_kwargs is None:
        light_kwargs = {}
    if strange_kwargs is None:
        strange_kwargs = {'Nf' : 1.0, 'ml' : pnjl.defaults.default_ms}
    if gluon_kwargs is None:
        gluon_kwargs = {}
    if perturbative_kwargs is None:
        perturbative_kwargs = {'Nf' : 3.0}

    sd = 1234
    attempt = 1
    max_attempt = 100
    bnds = ((0.0, 3.0), (-3.0, 3.0),)

    def thermodynamic_potential_with_clusters(
        x, 
        _T, _mu, 
        _diquark_bmass, _diquark_thmass, _d_diquark, _ni_diquark, _l_diquark,
        _fquark_bmass, _fquark_thmass, _d_fquark, _ni_fquark, _l_fquark,
        _qquark_bmass, _qquark_thmass, _d_qquark, _ni_qquark, _l_qquark,
        _s_kwargs, _q_kwargs, _pert_kwargs, _g_kwargs):
        sea = pnjl.thermo.gcp_sea_lattice.gcp_real(_T, _mu)
        sq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_s_kwargs)
        lq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_q_kwargs)
        per = pnjl.thermo.gcp_perturbative.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_pert_kwargs)
        glue = pnjl.thermo.gcp_pl_polynomial.gcp_real(_T, complex(x[0], x[1]), complex(x[0], -x[1]), **_g_kwargs)

        diquark = gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _diquark_bmass, _diquark_thmass, 2, 0, _d_diquark, _ni_diquark, _l_diquark)
        fquark  = gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _fquark_bmass , _fquark_thmass , 4, 0, _d_fquark , _ni_fquark , _l_fquark)
        qquark  = gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), _qquark_bmass , _qquark_thmass , 5, 0, _d_qquark , _ni_qquark , _l_qquark)

        return sea + sq + lq + per + glue + diquark + fquark + qquark

    def thermodynamic_potential(x, _T, _mu, _s_kwargs, _q_kwargs, _pert_kwargs, _g_kwargs):
        sea = pnjl.thermo.gcp_sea_lattice.gcp_real(_T, _mu)
        sq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_s_kwargs)
        lq = pnjl.thermo.gcp_quark.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_q_kwargs)
        per = pnjl.thermo.gcp_perturbative.gcp_real(_T, _mu, complex(x[0], x[1]), complex(x[0], -x[1]), **_pert_kwargs)
        glue = pnjl.thermo.gcp_pl_polynomial.gcp_real(_T, complex(x[0], x[1]), complex(x[0], -x[1]), **_g_kwargs)

        return sea + sq + lq + per + glue

    omega_result = None
    if with_clusters:

        diquark_bmass = pnjl.defaults.default_MD
        diquark_thmass = 2.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, mu)
        d_diquark = 1.0 * 1.0 * 3.0
        ni_diquark = 2.0
        l_diquark = pnjl.defaults.default_L
        fquark_bmass = pnjl.defaults.default_MF
        fquark_thmass = 4.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, mu)
        d_fquark = 1.0 * 1.0 * 3.0
        ni_fquark = 4.0
        l_fquark = pnjl.defaults.default_L
        qquark_bmass = pnjl.defaults.default_MQ
        qquark_thmass = 5.0 * math.sqrt(2) * pnjl.thermo.gcp_sea_lattice.M(T, mu)
        d_qquark = 2.0 * 2.0 * 3.0
        ni_qquark = 5.0
        l_qquark = pnjl.defaults.default_L

        omega_result = scipy.optimize.dual_annealing(
                thermodynamic_potential_with_clusters,
                bounds = bnds,
                args = (T, mu, 
                        diquark_bmass, diquark_thmass, d_diquark, ni_diquark, l_diquark, 
                        fquark_bmass, fquark_thmass, d_fquark, ni_fquark, l_fquark,
                        qquark_bmass, qquark_thmass, d_qquark, ni_qquark, l_qquark,
                        strange_kwargs, light_kwargs, perturbative_kwargs, gluon_kwargs),
                x0 = [phi_re0, phi_im0],
                maxiter = 20,
                seed = sd
                )
    else:
        omega_result = scipy.optimize.dual_annealing(
                thermodynamic_potential,
                bounds = bnds,
                args = (T, mu, strange_kwargs, light_kwargs, perturbative_kwargs, gluon_kwargs),
                x0 = [phi_re0, phi_im0],
                maxiter = 20,
                seed = sd
                )

    return (omega_result.x[0], omega_result.x[1])

#cluster regulator functions
def cluster_regulator_calc(mu : float, nophi = False):

    regulator_N = 0.0
    regulator_P = 0.0
    regulator_H = 0.0
    regulator_pi = 0.0
    regulator_K = 0.0
    regulator_rho = 0.0
    regulator_omega = 0.0
    regulator_T = 0.0
    regulator_D = 0.0
    regulator_F = 0.0
    regulator_Q5 = 0.0

    regulator_N2 = 0.0
    regulator_P2 = 0.0
    regulator_H2 = 0.0
    regulator_pi2 = 0.0
    regulator_K2 = 0.0
    regulator_rho2 = 0.0
    regulator_omega2 = 0.0
    regulator_T2 = 0.0
    regulator_D2 = 0.0
    regulator_F2 = 0.0
    regulator_Q52 = 0.0

    temp_mu = []
    temp_regN = []
    temp_regP = []
    temp_regH = []
    temp_regpi = []
    temp_regK = []
    temp_regrho = []
    temp_regomega = []
    temp_regT = []
    temp_regD = []
    temp_regF = []
    temp_regQ5 = []

    temp_regN2 = []
    temp_regP2 = []
    temp_regH2 = []
    temp_regpi2 = []
    temp_regK2 = []
    temp_regrho2 = []
    temp_regomega2 = []
    temp_regT2 = []
    temp_regD2 = []
    temp_regF2 = []
    temp_regQ52 = []

    print("Retreiving regulator data...")

    if os.path.exists("D:/EoS/epja/regulators.dat"):
        temp_mu, temp_regN = utils.data_collect(0, 1, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regN2 = utils.data_collect(0, 2, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regP = utils.data_collect(0, 3, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regP2 = utils.data_collect(0, 4, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regH = utils.data_collect(0, 5, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regH2 = utils.data_collect(0, 6, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regpi = utils.data_collect(0, 7, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regpi2 = utils.data_collect(0, 8, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regK = utils.data_collect(0, 9, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regK2 = utils.data_collect(0, 10, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regrho = utils.data_collect(0, 11, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regrho2 = utils.data_collect(0, 12, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regomega = utils.data_collect(0, 13, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regomega2 = utils.data_collect(0, 14, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regT = utils.data_collect(0, 15, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regT2 = utils.data_collect(0, 16, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regD = utils.data_collect(0, 17, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regD2 = utils.data_collect(0, 18, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regF = utils.data_collect(0, 19, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regF2 = utils.data_collect(0, 20, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regQ5 = utils.data_collect(0, 21, "D:/EoS/epja/regulators.dat")
        temp_mu, temp_regQ52 = utils.data_collect(0, 22, "D:/EoS/epja/regulators.dat")

    if mu in temp_mu:
        regulator_N = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regN) if reg_mu == mu][0]
        regulator_P = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regP) if reg_mu == mu][0]
        regulator_H = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regH) if reg_mu == mu][0]
        regulator_pi = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regpi) if reg_mu == mu][0]
        regulator_K = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regK) if reg_mu == mu][0]
        regulator_rho = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regrho) if reg_mu == mu][0]
        regulator_omega = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regomega) if reg_mu == mu][0]
        regulator_T = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regT) if reg_mu == mu][0]
        regulator_D = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regD) if reg_mu == mu][0]
        regulator_F = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regF) if reg_mu == mu][0]
        regulator_Q5 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regQ5) if reg_mu == mu][0]

        regulator_N2 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regN2) if reg_mu == mu][0]
        regulator_P2 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regP2) if reg_mu == mu][0]
        regulator_H2 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regH2) if reg_mu == mu][0]
        regulator_pi2 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regpi2) if reg_mu == mu][0]
        regulator_K2 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regK2) if reg_mu == mu][0]
        regulator_rho2 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regrho2) if reg_mu == mu][0]
        regulator_omega2 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regomega2) if reg_mu == mu][0]
        regulator_T2 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regT2) if reg_mu == mu][0]
        regulator_D2 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regD2) if reg_mu == mu][0]
        regulator_F2 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regF2) if reg_mu == mu][0]
        regulator_Q52 = [reg_el for reg_mu, reg_el in zip(temp_mu, temp_regQ52) if reg_mu == mu][0]
    else:

        print("Regulator data missing. Calculating...")

        Tc0 = pnjl.defaults.default_Tc0
        kappa = pnjl.defaults.default_kappa
        delta_T = pnjl.defaults.default_delta_T
        M0 = pnjl.defaults.default_M0
        ml = pnjl.defaults.default_ml
        ms = pnjl.defaults.default_ms

        MN = pnjl.defaults.default_MN
        NN = 3.0
        sN = 0.0
        dN = (2.0 * 2.0 * 1.0) / 2.0
        MP = pnjl.defaults.default_MP
        NP = 5.0
        sP = 0.0
        dP = ((4.0 * 2.0 * 1.0) + (2.0 * 4.0 * 1.0)) / 2.0
        MH = pnjl.defaults.default_MH
        NH = 6.0
        sH = 0.0
        dH = ((1.0 * 3.0 * 1.0) + (3.0 * 1.0 * 1.0)) / 2.0
        Mpi = pnjl.defaults.default_Mpi
        Npi = 2.0
        spi = 0.0
        dpi = (1.0 * 3.0 * 1.0)
        MK = pnjl.defaults.default_MK
        NK = 2.0
        sK = 1.0
        dK = (1.0 * 6.0 * 1.0)
        Mrho = pnjl.defaults.default_MM
        Nrho = 2.0
        srho = 0.0
        drho = (3.0 * 3.0 * 1.0)
        Momega = pnjl.defaults.default_MM
        Nomega = 2.0
        somega = 0.0
        domega = (3.0 * 1.0 * 1.0)
        MT = pnjl.defaults.default_MT
        NT = 4.0
        sT = 0.0
        dT = ((1.0 * 5.0 * 1.0) + (5.0 * 1.0 * 1.0) + (3.0 * 3.0 * 1.0)) / 2.0
        MD = pnjl.defaults.default_MD
        ND = 2.0
        sD = 0.0
        dD = (1.0 * 1.0 * 3.0) / 2.0
        MF = pnjl.defaults.default_MF
        NF = 4.0
        sF = 0.0
        dF = (1.0 * 1.0 * 3.0) / 2.0
        MQ5 = pnjl.defaults.default_MQ
        NQ5 = 5.0
        sQ5 = 0.0
        dQ5 = (2.0 * 1.0 * 3.0) / 2.0

        L = pnjl.defaults.default_L

        print("T_Mott...")

        T_crit_N = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * NN + 2.0 * (ms - ml) * sN - 2.0 * MN) / (M0 * NN))) / Tc0
        T_crit_P = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * NP + 2.0 * (ms - ml) * sP - 2.0 * MP) / (M0 * NP))) / Tc0
        T_crit_H = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * NH + 2.0 * (ms - ml) * sH - 2.0 * MH) / (M0 * NH))) / Tc0
        T_crit_pi = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * Npi + 2.0 * (ms - ml) * spi - 2.0 * Mpi) / (M0 * Npi))) / Tc0
        T_crit_K = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * NK + 2.0 * (ms - ml) * sK - 2.0 * MK) / (M0 * NK))) / Tc0
        T_crit_rho = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * Nrho + 2.0 * (ms - ml) * srho - 2.0 * Mrho) / (M0 * Nrho))) / Tc0
        T_crit_omega = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * Nomega + 2.0 * (ms - ml) * somega - 2.0 * Momega) / (M0 * Nomega))) / Tc0
        T_crit_T = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * NT + 2.0 * (ms - ml) * sT - 2.0 * MT) / (M0 * NT))) / Tc0
        T_crit_D = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * ND + 2.0 * (ms - ml) * sD - 2.0 * MD) / (M0 * ND))) / Tc0
        T_crit_F = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * NF + 2.0 * (ms - ml) * sF - 2.0 * MF) / (M0 * NF))) / Tc0
        T_crit_Q5 = ((Tc0 ** 2) - kappa * (mu ** 2) + Tc0 * delta_T * math.atanh(((M0 + 2.0 * ml) * NQ5 + 2.0 * (ms - ml) * sQ5 - 2.0 * MQ5) / (M0 * NQ5))) / Tc0

        print("Phi...")

        phi_re_N_low, phi_im_N_low = (1.0, 0.0)
        phi_re_N_high, phi_im_N_high = (1.0, 0.0)

        phi_re_P_low, phi_im_P_low = (1.0, 0.0)
        phi_re_P_high, phi_im_P_high = (1.0, 0.0)

        phi_re_H_low, phi_im_H_low = (1.0, 0.0)
        phi_re_H_high, phi_im_H_high = (1.0, 0.0)

        phi_re_pi_low, phi_im_pi_low = (1.0, 0.0)
        phi_re_pi_high, phi_im_pi_high = (1.0, 0.0)

        phi_re_K_low, phi_im_K_low = (1.0, 0.0)
        phi_re_K_high, phi_im_K_high = (1.0, 0.0)

        phi_re_rho_low, phi_im_rho_low = (1.0, 0.0)
        phi_re_rho_high, phi_im_rho_high = (1.0, 0.0)

        phi_re_omega_low, phi_im_omega_low = (1.0, 0.0)
        phi_re_omega_high, phi_im_omega_high = (1.0, 0.0)

        phi_re_T_low, phi_im_T_low = (1.0, 0.0)
        phi_re_T_high, phi_im_T_high = (1.0, 0.0)

        phi_re_D_low, phi_im_D_low = (1.0, 0.0)
        phi_re_D_high, phi_im_D_high = (1.0, 0.0)

        phi_re_F_low, phi_im_F_low = (1.0, 0.0)
        phi_re_F_high, phi_im_F_high = (1.0, 0.0)

        phi_re_Q5_low, phi_im_Q5_low = (1.0, 0.0)
        phi_re_Q5_high, phi_im_Q5_high = (1.0, 0.0)

        if not nophi:
            phi_re_N_low, phi_im_N_low = calc_PL_c(T_crit_N - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_N_high, phi_im_N_high = calc_PL_c(T_crit_N + 3e-2, mu, phi_re_N_low, phi_im_N_low, with_clusters = False)

            phi_re_P_low, phi_im_P_low = calc_PL_c(T_crit_P - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_P_high, phi_im_P_high = calc_PL_c(T_crit_P + 3e-2, mu, phi_re_P_low, phi_im_P_low, with_clusters = False)

            phi_re_H_low, phi_im_H_low = calc_PL_c(T_crit_H - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_H_high, phi_im_H_high = calc_PL_c(T_crit_H + 3e-2, mu, phi_re_H_low, phi_im_H_low, with_clusters = False)

            phi_re_pi_low, phi_im_pi_low = calc_PL_c(T_crit_pi - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_pi_high, phi_im_pi_high = calc_PL_c(T_crit_pi + 3e-2, mu, phi_re_pi_low, phi_im_pi_low, with_clusters = False)

            phi_re_K_low, phi_im_K_low = calc_PL_c(T_crit_K - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_K_high, phi_im_K_high = calc_PL_c(T_crit_K + 3e-2, mu, phi_re_K_low, phi_im_K_low, with_clusters = False)

            phi_re_rho_low, phi_im_rho_low = calc_PL_c(T_crit_rho - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_rho_high, phi_im_rho_high = calc_PL_c(T_crit_rho + 3e-2, mu, phi_re_rho_low, phi_im_rho_low, with_clusters = False)

            phi_re_omega_low, phi_im_omega_low = calc_PL_c(T_crit_omega - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_omega_high, phi_im_omega_high = calc_PL_c(T_crit_omega + 3e-2, mu, phi_re_omega_low, phi_im_omega_low, with_clusters = False)

            phi_re_T_low, phi_im_T_low = calc_PL_c(T_crit_T - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_T_high, phi_im_T_high = calc_PL_c(T_crit_T + 3e-2, mu, phi_re_T_low, phi_im_T_low, with_clusters = False)

            phi_re_D_low, phi_im_D_low = calc_PL_c(T_crit_D - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_D_high, phi_im_D_high = calc_PL_c(T_crit_D + 3e-2, mu, phi_re_D_low, phi_im_D_low, with_clusters = False)

            phi_re_F_low, phi_im_F_low = calc_PL_c(T_crit_F - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_F_high, phi_im_F_high = calc_PL_c(T_crit_F + 3e-2, mu, phi_re_F_low, phi_im_F_low, with_clusters = False)

            phi_re_Q5_low, phi_im_Q5_low = calc_PL_c(T_crit_Q5 - 3e-2, mu, 1e-15, 2e-15, with_clusters = False)
            phi_re_Q5_high, phi_im_Q5_high = calc_PL_c(T_crit_Q5 + 3e-2, mu, phi_re_Q5_low, phi_im_Q5_low, with_clusters = False)

        print("M_th...")

        MthN_low = ((NN - sN) * pnjl.thermo.gcp_sea_lattice.M(T_crit_N - 3e-2, mu) + sN * pnjl.thermo.gcp_sea_lattice.M(T_crit_N - 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthN_high = ((NN - sN) * pnjl.thermo.gcp_sea_lattice.M(T_crit_N + 3e-2, mu) + sN * pnjl.thermo.gcp_sea_lattice.M(T_crit_N + 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthP_low = ((NP - sP) * pnjl.thermo.gcp_sea_lattice.M(T_crit_P - 3e-2, mu) + sP * pnjl.thermo.gcp_sea_lattice.M(T_crit_P - 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthP_high = ((NP - sP) * pnjl.thermo.gcp_sea_lattice.M(T_crit_P + 3e-2, mu) + sP * pnjl.thermo.gcp_sea_lattice.M(T_crit_P + 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthH_low = ((NH - sH) * pnjl.thermo.gcp_sea_lattice.M(T_crit_H - 3e-2, mu) + sH * pnjl.thermo.gcp_sea_lattice.M(T_crit_H - 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthH_high = ((NH - sH) * pnjl.thermo.gcp_sea_lattice.M(T_crit_H + 3e-2, mu) + sH * pnjl.thermo.gcp_sea_lattice.M(T_crit_H + 3e-2, mu, ml = pnjl.defaults.default_ms))
        Mthpi_low = ((Npi - spi) * pnjl.thermo.gcp_sea_lattice.M(T_crit_pi - 3e-2, mu) + spi * pnjl.thermo.gcp_sea_lattice.M(T_crit_pi - 3e-2, mu, ml = pnjl.defaults.default_ms))
        Mthpi_high = ((Npi - spi) * pnjl.thermo.gcp_sea_lattice.M(T_crit_pi + 3e-2, mu) + spi * pnjl.thermo.gcp_sea_lattice.M(T_crit_pi + 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthK_low = ((NK - sK) * pnjl.thermo.gcp_sea_lattice.M(T_crit_K - 3e-2, mu) + sK * pnjl.thermo.gcp_sea_lattice.M(T_crit_K - 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthK_high = ((NK - sK) * pnjl.thermo.gcp_sea_lattice.M(T_crit_K + 3e-2, mu) + sK * pnjl.thermo.gcp_sea_lattice.M(T_crit_K + 3e-2, mu, ml = pnjl.defaults.default_ms))
        Mthrho_low = ((Nrho - srho) * pnjl.thermo.gcp_sea_lattice.M(T_crit_rho - 3e-2, mu) + srho * pnjl.thermo.gcp_sea_lattice.M(T_crit_rho - 3e-2, mu, ml = pnjl.defaults.default_ms))
        Mthrho_high = ((Nrho - srho) * pnjl.thermo.gcp_sea_lattice.M(T_crit_rho + 3e-2, mu) + srho * pnjl.thermo.gcp_sea_lattice.M(T_crit_rho + 3e-2, mu, ml = pnjl.defaults.default_ms))
        Mthomega_low = ((Nomega - somega) * pnjl.thermo.gcp_sea_lattice.M(T_crit_omega - 3e-2, mu) + somega * pnjl.thermo.gcp_sea_lattice.M(T_crit_omega - 3e-2, mu, ml = pnjl.defaults.default_ms))
        Mthomega_high = ((Nomega - somega) * pnjl.thermo.gcp_sea_lattice.M(T_crit_omega + 3e-2, mu) + somega * pnjl.thermo.gcp_sea_lattice.M(T_crit_omega + 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthT_low = ((NT - sT) * pnjl.thermo.gcp_sea_lattice.M(T_crit_T - 3e-2, mu) + sT * pnjl.thermo.gcp_sea_lattice.M(T_crit_T - 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthT_high = ((NT - sT) * pnjl.thermo.gcp_sea_lattice.M(T_crit_T + 3e-2, mu) + sT * pnjl.thermo.gcp_sea_lattice.M(T_crit_T + 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthD_low = ((ND - sD) * pnjl.thermo.gcp_sea_lattice.M(T_crit_D - 3e-2, mu) + sD * pnjl.thermo.gcp_sea_lattice.M(T_crit_D - 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthD_high = ((ND - sD) * pnjl.thermo.gcp_sea_lattice.M(T_crit_D + 3e-2, mu) + sD * pnjl.thermo.gcp_sea_lattice.M(T_crit_D + 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthF_low = ((NF - sF) * pnjl.thermo.gcp_sea_lattice.M(T_crit_F - 3e-2, mu) + sF * pnjl.thermo.gcp_sea_lattice.M(T_crit_F - 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthF_high = ((NF - sF) * pnjl.thermo.gcp_sea_lattice.M(T_crit_F + 3e-2, mu) + sF * pnjl.thermo.gcp_sea_lattice.M(T_crit_F + 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthQ5_low = ((NQ5 - sQ5) * pnjl.thermo.gcp_sea_lattice.M(T_crit_Q5 - 3e-2, mu) + sQ5 * pnjl.thermo.gcp_sea_lattice.M(T_crit_Q5 - 3e-2, mu, ml = pnjl.defaults.default_ms))
        MthQ5_high = ((NQ5 - sQ5) * pnjl.thermo.gcp_sea_lattice.M(T_crit_Q5 + 3e-2, mu) + sQ5 * pnjl.thermo.gcp_sea_lattice.M(T_crit_Q5 + 3e-2, mu, ml = pnjl.defaults.default_ms))

        def slope_diff(reg, reg1, _T_crit, _mu, _phire_low, _phiim_low, _phire_high, _phiim_high, _M, _Mth_low, _Mth_high, _d, _N, _L, _a, _b, name):
            print("Cluster", name, "regulator value", reg[0])
            low_p = pressure(_T_crit - 3e-2, _mu, complex(_phire_low, _phiim_low), complex(_phire_low, -_phiim_low), _M, _Mth_low, _a, _b, _d, _N, _L, _T_crit, reg1, reg[0])
            high_p = pressure(_T_crit + 3e-2, _mu, complex(_phire_high, _phiim_high), complex(_phire_high, -_phiim_high), _M, _Mth_high, _a, _b, _d, _N, _L, _T_crit, reg1, reg[0])
            print("Cluster", name, "regulator error", math.fabs(low_p - high_p))
            return math.fabs(low_p - high_p)
        def slope_diff_alt(_T_crit, _mu, _phire_low, _phiim_low, _phire_high, _phiim_high, _M, _Mth_low, _Mth_high, _d, _N, _L, _a, _b, name):
            print("Cluster", name)
            low = sdensity(_T_crit - 3e-2, _mu, complex(_phire_low, _phiim_low), complex(_phire_low, -_phiim_low), _M, _Mth_low, _a, _b, _d, _N, _L, _T_crit, 1.0, 0.0)
            high = sdensity(_T_crit + 3e-2, _mu, complex(_phire_high, _phiim_high), complex(_phire_high, -_phiim_high), _M, _Mth_high, _a, _b, _d, _N, _L, _T_crit, 1.0, 0.0)
            reg1 = low / high
            low_p = pressure(_T_crit - 3e-2, _mu, complex(_phire_low, _phiim_low), complex(_phire_low, -_phiim_low), _M, _Mth_low, _a, _b, _d, _N, _L, _T_crit, reg1, 0.0)
            high_p = pressure(_T_crit + 3e-2, _mu, complex(_phire_high, _phiim_high), complex(_phire_high, -_phiim_high), _M, _Mth_high, _a, _b, _d, _N, _L, _T_crit, reg1, 0.0)
            return reg1, low_p - high_p

        args_N = (T_crit_N, mu, phi_re_N_low, phi_im_N_low, phi_re_N_high, phi_im_N_high, MN, MthN_low, MthN_high, dN, NN, L, 3, 0, "N")
        args_P = (T_crit_P, mu, phi_re_P_low, phi_im_P_low, phi_re_P_high, phi_im_P_high, MP, MthP_low, MthP_high, dP, NP, L, 3, 0, "P")
        args_H = (T_crit_H, mu, phi_re_H_low, phi_im_H_low, phi_re_H_high, phi_im_H_high, MH, MthH_low, MthH_high, dH, NH, L, 6, 0, "H")
        args_pi = (T_crit_pi, mu, phi_re_pi_low, phi_im_pi_low, phi_re_pi_high, phi_im_pi_high, Mpi, Mthpi_low, Mthpi_high, dpi, Npi, L, 0, 0, "pi")
        args_K = (T_crit_K, mu, phi_re_K_low, phi_im_K_low, phi_re_K_high, phi_im_K_high, MK, MthK_low, MthK_high, dK, NK, L, 1, -1, "K")
        args_rho = (T_crit_rho, mu, phi_re_rho_low, phi_im_rho_low, phi_re_rho_high, phi_im_rho_high, Mrho, Mthrho_low, Mthrho_high, drho, Nrho, L, 0, 0, "rho")
        args_omega = (T_crit_omega, mu, phi_re_omega_low, phi_im_omega_low, phi_re_omega_high, phi_im_omega_high, Momega, Mthomega_low, Mthomega_high, domega, Nomega, L, 0, 0, "omega")
        args_T = (T_crit_T, mu, phi_re_T_low, phi_im_T_low, phi_re_T_high, phi_im_T_high, MT, MthT_low, MthT_high, dT, NT, L, 0, 0, "T")
        args_D = (T_crit_D, mu, phi_re_D_low, phi_im_D_low, phi_re_D_high, phi_im_D_high, MD, MthD_low, MthD_high, dD, ND, L, 2, 0, "D")
        args_F = (T_crit_F, mu, phi_re_F_low, phi_im_F_low, phi_re_F_high, phi_im_F_high, MF, MthF_low, MthF_high, dF, NF, L, 4, 0, "F")
        args_Q5 = (T_crit_Q5, mu, phi_re_Q5_low, phi_im_Q5_low, phi_re_Q5_high, phi_im_Q5_high, MQ5, MthQ5_low, MthQ5_high, dQ5, NQ5, L, 5, 0, "Q5")
        
        print("Regulators...")

        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_N.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_N.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_N.dat")
        if mu in test_mu: 
            regulator_N = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_N2 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_N, regulator_N2 = slope_diff_alt(*args_N)
            with open("D:/EoS/epja/regulator_N.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_N, regulator_N2])
                file.flush()
        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_P.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_P.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_P.dat")
        if mu in test_mu: 
            regulator_P = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_P2 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_P, regulator_P2 = slope_diff_alt(*args_P)
            with open("D:/EoS/epja/regulator_P.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_P, regulator_P2])
                file.flush()
        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_H.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_H.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_H.dat")
        if mu in test_mu: 
            regulator_H = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_H2 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_H, regulator_H2 = slope_diff_alt(*args_H)
            with open("D:/EoS/epja/regulator_H.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_H, regulator_H2])
                file.flush()
        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_pi.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_pi.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_pi.dat")
        if mu in test_mu: 
            regulator_pi = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_pi2 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_pi, regulator_pi2 = slope_diff_alt(*args_pi)
            with open("D:/EoS/epja/regulator_pi.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_pi, regulator_pi2])
                file.flush()
        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_K.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_K.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_K.dat")
        if mu in test_mu: 
            regulator_K = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_K2 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_K, regulator_K2 = slope_diff_alt(*args_K)
            with open("D:/EoS/epja/regulator_K.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_K, regulator_K2])
                file.flush()
        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_rho.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_rho.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_rho.dat")
        if mu in test_mu: 
            regulator_rho = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_rho2 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_rho, regulator_rho2 = slope_diff_alt(*args_rho)
            with open("D:/EoS/epja/regulator_rho.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_rho, regulator_rho2])
                file.flush()
        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_omega.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_omega.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_omega.dat")
        if mu in test_mu: 
            regulator_omega = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_omega2 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_omega, regulator_omega2 = slope_diff_alt(*args_omega)
            with open("D:/EoS/epja/regulator_omega.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_omega, regulator_omega2])
                file.flush()
        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_T.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_T.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_T.dat")
        if mu in test_mu: 
            regulator_T = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_T2 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_T, regulator_T2 = slope_diff_alt(*args_T)
            with open("D:/EoS/epja/regulator_T.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_T, regulator_T2])
                file.flush()
        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_D.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_D.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_D.dat")
        if mu in test_mu: 
            regulator_D = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_D2 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_D, regulator_D2 = slope_diff_alt(*args_D)
            with open("D:/EoS/epja/regulator_D.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_D, regulator_D2])
                file.flush()
        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_F.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_F.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_F.dat")
        if mu in test_mu: 
            regulator_F = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_F2 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_F, regulator_F2 = slope_diff_alt(*args_F)
            with open("D:/EoS/epja/regulator_F.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_F, regulator_F2])
                file.flush()
        test_mu = []
        test_reg = []
        if os.path.exists("D:/EoS/epja/regulator_Q5.dat"): 
            test_mu, test_reg = utils.data_collect(0, 1, "D:/EoS/epja/regulator_Q5.dat")
            test_mu, test_reg2 = utils.data_collect(0, 2, "D:/EoS/epja/regulator_Q5.dat")
        if mu in test_mu: 
            regulator_Q5 = [reg_el for reg_el, reg_mu in zip(test_reg, test_mu) if reg_mu == mu][0]
            regulator_Q52 = [reg_el for reg_el, reg_mu in zip(test_reg2, test_mu) if reg_mu == mu][0]
        else:
            regulator_Q5, regulator_Q52 = slope_diff_alt(*args_Q5)
            with open("D:/EoS/epja/regulator_Q5.dat", 'w') as file:
                writer = csv.writer(file, delimiter = '\t')
                writer.writerow([mu, regulator_Q5, regulator_Q52])
                file.flush()

        with open("D:/EoS/epja/regulators.dat", 'a') as file:
            writer = csv.writer(file, delimiter = '\t')
            writer.writerow([mu, regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2, regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2, regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52])
            file.flush()

        for file in glob.glob("D:/EoS/epja/regulator_*.dat"): os.remove(file)

    print("Regulator data retreived.")

    return regulator_N, regulator_N2, regulator_P, regulator_P2, regulator_H, regulator_H2, regulator_pi, regulator_pi2, regulator_K, regulator_K2, regulator_rho, regulator_rho2, regulator_omega, regulator_omega2, regulator_T, regulator_T2, regulator_D, regulator_D2, regulator_F, regulator_F2, regulator_Q5, regulator_Q52

#Extensive thermodynamic properties

def pressure(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):
    #
    return -gcp_real(T, mu, Phi, Phib, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs)

#the function has a hardcoded mass formula and does not take into account s -- anti-s pairs!!!
def bdensity(T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):
    
    h = 1e-2
    mu_vec = []
    Phi_vec = []
    Phib_vec = []
    if mu - 2 * h > 0.0:
        mu_vec = [mu + 2 * h, mu + h, mu - h, mu - 2 * h]
    else:
        mu_vec = [mu + h, mu]
        if numpy.any([el < 0.0 for el in mu_vec]):
            return bdensity(T, mu + h, Phi, Phib, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs)
    Phi_vec = [Phi for el in mu_vec]
    Phib_vec = [Phib for el in mu_vec]
    Mth_vec = [((Ni - math.fabs(b)) * pnjl.thermo.gcp_sea_lattice.M(T, el) + math.fabs(b) * pnjl.thermo.gcp_sea_lattice.M(T, el, ml = pnjl.defaults.default_ms)) for el in mu_vec]

    if len(mu_vec) == len(Phi_vec) and len(mu_vec) == len(Phib_vec) and len(mu_vec) == len(Mth_vec):
        if len(mu_vec) == 4 and numpy.all(mu_vec[i] > mu_vec[i + 1] for i, el in enumerate(mu_vec[:-1])):
            p_vec = [pressure(T, mu_el, Phi_el, Phib_el, bmass, Mth_el, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs) for mu_el, Phi_el, Phib_el, Mth_el in zip(mu_vec, Phi_vec, Phib_vec, Mth_vec)]
            return (1.0 / 3.0) * (p_vec[3] - 8.0 * p_vec[2] + 8.0 * p_vec[1] - p_vec[0]) / (12.0 * h)
        elif len(mu_vec) == 2 and mu_vec[0] > mu_vec[1]:
            p_vec = [pressure(T, mu_el, Phi_el, Phib_el, bmass, Mth_el, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs) for mu_el, Phi_el, Phib_el, Mth_el in zip(mu_vec, Phi_vec, Phib_vec, Mth_vec)]
            return (1.0 / 3.0) * (p_vec[0] - p_vec[1]) / h
        else:
            raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
    else:
        raise RuntimeError("Value vectors don't match!")

#cumulant chi_q based on eq. 29 of https://arxiv.org/pdf/2012.12894.pdf and the subsequent inline definition.
def bnumber_cumulant_R(rank : int, T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    if rank == 1:
        return bdensity(T, mu, Phi, Phib, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs)
    elif rank < 1:
        raise RuntimeError("Cumulant rank lower than 1!")
    else:
        h = 1e-2
        mu_vec = []
        Phi_vec = []
        Phib_vec = []
        if mu - 2 * h > 0.0:
            mu_vec = [mu + 2 * h, mu + h, mu - h, mu - 2 * h]
        else:
            mu_vec = [mu + h, mu]
            if numpy.any([el < 0.0 for el in mu_vec]):
                return bnumber_cumulant(rank, T, mu + h, Phi, Phib, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs)
        Phi_vec = [Phi for el in mu_vec]
        Phib_vec = [Phib for el in mu_vec]

        if len(mu_vec) == len(Phi_vec) and len(mu_vec) == len(Phib_vec):
            if len(mu_vec) == 4 and numpy.all(mu_vec[i] > mu_vec[i + 1] for i, el in enumerate(mu_vec[:-1])):
                out_vec = [bnumber_cumulant(rank - 1, T, mu_el, Phi_el, Phib_el, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs) for mu_el, Phi_el, Phib_el in zip(mu_vec, Phi_vec, Phib_vec)]
                return (1.0 / 3.0) * (out_vec[3] - 8.0 * out_vec[2] + 8.0 * out_vec[1] - out_vec[0]) / (12.0 * h)
            elif len(mu_vec) == 2 and mu_vec[0] > mu_vec[1]:
                out_vec = [bnumber_cumulant(rank - 1, T, mu_el, Phi_el, Phib_el, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs) for mu_el, Phi_el, Phib_el in zip(mu_vec, Phi_vec, Phib_vec)]
                return (1.0 / 3.0) * (out_vec[0] - out_vec[1]) / h
            else:
                raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
        else:
            raise RuntimeError("Value vectors don't match!")

#cumulant chi_B based on https://arxiv.org/pdf/1701.04325.pdf and my calculations (incomplete)
def bnumber_cumulant_chiB(rank : int, T : float, mu : float, Phi : complex, Phib : complex, bmass : float, thmass : float, a : int, b : int, d : float, Ni : float, L : float, T_crit : float, regulator : float, regulator2 : float, **kwargs):

    if rank == 1:
        return bdensity(T, mu, Phi, Phib, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs) / (T ** 3)
    elif rank < 1:
        raise RuntimeError("Cumulant rank lower than 1!")
    else:
        h = 1e-2
        mu_vec = []
        Phi_vec = []
        Phib_vec = []
        if mu - 2 * h > 0.0:
            mu_vec = [mu + 2 * h, mu + h, mu - h, mu - 2 * h]
        else:
            mu_vec = [mu + h, mu]
            if numpy.any([el < 0.0 for el in mu_vec]):
                return bnumber_cumulant(rank, T, mu + h, Phi, Phib, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs)
        Phi_vec = [Phi for el in mu_vec]
        Phib_vec = [Phib for el in mu_vec]

        if len(mu_vec) == len(Phi_vec) and len(mu_vec) == len(Phib_vec):
            if len(mu_vec) == 4 and numpy.all(mu_vec[i] > mu_vec[i + 1] for i, el in enumerate(mu_vec[:-1])):
                out_vec = [bnumber_cumulant(rank - 1, T, mu_el, Phi_el, Phib_el, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs) for mu_el, Phi_el, Phib_el in zip(mu_vec, Phi_vec, Phib_vec)]
                return (T / 3.0) * (out_vec[3] - 8.0 * out_vec[2] + 8.0 * out_vec[1] - out_vec[0]) / (12.0 * h)
            elif len(mu_vec) == 2 and mu_vec[0] > mu_vec[1]:
                out_vec = [bnumber_cumulant(rank - 1, T, mu_el, Phi_el, Phib_el, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs) for mu_el, Phi_el, Phib_el in zip(mu_vec, Phi_vec, Phib_vec)]
                return (T / 3.0) * (out_vec[0] - out_vec[1]) / h
            else:
                raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
        else:
            raise RuntimeError("Value vectors don't match!")

#the function has a hardcoded mass formula and does not take into account s -- anti-s pairs!!!
def sdensity(
        T : float, mu : float, Phi : complex, Phib : complex, L : float,
        T_crit : float, regulator : float, regulator2 : float, type : str, **kwargs
) -> float:
    
    if type not in pnjl.defaults.default_M:
        raise RuntimeError("Unknown cluster type!")

    bmass : float, a : int, b : int, d : float, Ni : float, 

    h = 1e-2
    T_vec = []
    Phi_vec = []
    Phib_vec = []
    if T - 2 * h > 0.0:
        T_vec = [T + 2 * h, T + h, T - h, T - 2 * h]
    else:
        T_vec = [T + h, T]
        if numpy.any([el < 0.0 for el in T_vec]):
            return sdensity(T + h, mu, Phi, Phib, bmass, thmass, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs)
    Phi_vec = [Phi for el in T_vec]
    Phib_vec = [Phib for el in T_vec]
    Mth_vec = [((Ni - math.fabs(b)) * pnjl.thermo.gcp_sea_lattice.M(el, mu) + math.fabs(b) * pnjl.thermo.gcp_sea_lattice.M(el, mu, ml = pnjl.defaults.default_ms)) for el in T_vec]

    if len(T_vec) == len(Phi_vec) and len(T_vec) == len(Phib_vec) and len(T_vec) == len(Mth_vec):
        if len(T_vec) == 4 and numpy.all(T_vec[i] > T_vec[i + 1] for i, el in enumerate(T_vec[:-1])):
            p_vec = [pressure(T_el, mu, Phi_el, Phib_el, bmass, Mth_el, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs) for T_el, Phi_el, Phib_el, Mth_el in zip(T_vec, Phi_vec, Phib_vec, Mth_vec)]
            return (1.0 / 3.0) * (p_vec[3] - 8.0 * p_vec[2] + 8.0 * p_vec[1] - p_vec[0]) / (12.0 * h)
        elif len(T_vec) == 2 and T_vec[0] > T_vec[1]:
            p_vec = [pressure(T_el, mu, Phi_el, Phib_el, bmass, Mth_el, a, b, d, Ni, L, T_crit, regulator, regulator2, **kwargs) for T_el, Phi_el, Phib_el, Mth_el in zip(T_vec, Phi_vec, Phib_vec, Mth_vec)]
            return (1.0 / 3.0) * (p_vec[0] - p_vec[1]) / h
        else:
            raise RuntimeError("Vectors have wrong size or are not strictly decreasing!")
    else:
        raise RuntimeError("Value vectors don't match!")


def cluster_thermo(T, mu, phi_re, phi_im, L, regulator, regulator2, type, **kwargs):
    """Main thermodynamics output function for clusters with step-up - step-down + arccos-cos continuum.

    ---- Parameters ----
    T : list
        Temperature values in MeV (length n).
    mu : list
        Quark chemical potential values in MeV (length n).
    phi_re : list
        Real part of the Polyakov loop in ??? (length n).
    phi_im : list
        Imaginary part of the Polyakov loop in ??? (length n).
    L: float
        Continuum energy range in ???.
    regulator: float
        Phase shift correction factor for entropy.
    regulator: float
        Phase shift correction factor for pressure in MeV^4.
    type : string 
        Cluster predefined shorthand
            - 'pi' : pion
            - 'K' : kaon
            - 'rho' : rho meson
            - 'omega' : omega meson
            - 'D' : colored diquark
            - 'N' : nucleon
            - 'T' : tetraquark
            - 'F' : colored four-quark
            - 'P' : pentaquark
            - 'Q' : colored five-quark
            - 'H' : hexaquark.
    with_pressure : bool, optional
        Set to True to output pressure.
    with_bdensity : bool, optional
        Set to True to output baryon density.
    with_sdensity : bool, optional
        Set to True to output entropy density.
    with_bnumber_R2 : bool, optional
        Set to True to output baryon number 2nd rank susceptibility R2,
        definition in https://arxiv.org/pdf/2012.12894.pdf .

    ---- Returns ----
    out : dict
        Contains keys
            - 'pressure' if with_pressure set to True
            - 'bdensity' if with_bdensity set to True
            - 'sdensity' if with_sdensity set to True
            - 'bnumber_R2' if with_bnumber_R2 set to True.
    """

    if type not in pnjl.defaults.default_M:
        raise RuntimeError("Unknown cluster type!")

    M = pnjl.defaults.default_M[type]
    #a = pnjl.defaults.default_a[type]
    #b = pnjl.defaults.default_b[type]
    #dx = pnjl.defaults.default_d[type]
    Ni = pnjl.defaults.default_N[type]
    s = pnjl.defaults.default_s[type]

    options = {
        'with_pressure' : True,
        'with_bdensity' : False,
        'with_sdensity' : False,
        'with_bnumber_R2' : False
    }
    options.update(kwargs)

    Tc0 = pnjl.defaults.default_Tc0
    kappa = pnjl.defaults.default_kappa
    delta_T = pnjl.defaults.default_delta_T
    M0 = pnjl.defaults.default_M0
    ml = pnjl.defaults.default_ml
    ms =  pnjl.defaults.default_ms
    with_pressure = options['with_pressure']
    with_bdensity = options['with_bdensity']
    with_sdensity = options['with_sdensity']
    with_bnumber_R2 = options['with_bnumber_R2']

    out = dict()

    T_crit_v = [
        (
            (Tc0**2)
            - kappa * (el**2)
            + Tc0 * delta_T
            * math.atanh(((M0 + 2.0*ml)*Ni + 2.0*(ms-ml)*s - 2.0*M) / (M0*Ni))
        ) / Tc0 for el in mu
    ]
    Mth = [
        (Ni-s) * pnjl.thermo.gcp_sea_lattice.Ml(T_el, mu_el)
        + s * pnjl.thermo.gcp_sea_lattice.Ms(T_el, mu_el)
        for T_el, mu_el in zip(T, mu)
    ]

    if with_pressure:
        Pres = [
            pressure(
                T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el),
                Mth_el, L, T_crit_el, regulator, regulator2, type
            )
            for T_el, T_crit_el, mu_el, \
                phi_re_el, phi_im_el, Mth_el in tqdm.tqdm(
                                                    zip(T, T_crit_v, mu, phi_re, phi_im, Mth),
                                                    desc = "Pres", total = len(T), ascii = True
                                                )
        ]
        out['pressure'] = Pres

    if with_bdensity:
        BDen = [
            bdensity(
                T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el),
                L, T_crit_el, regulator, regulator2, type
            )
            for T_el, T_crit_el, mu_el, \
                phi_re_el, phi_im_el in tqdm.tqdm(
                                            zip(T, T_crit_v, mu, phi_re, phi_im),
                                            desc = "BDen", total = len(T), ascii = True
                                        )
        ]
        out['bdensity'] = BDen

    if with_sdensity:
        SDen = [
            sdensity(
                T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el),
                L, T_crit_el, regulator, regulator2, type
            )
            for T_el, T_crit_el, mu_el, \
                phi_re_el, phi_im_el in tqdm.tqdm(
                                            zip(T, T_crit_v, mu, phi_re, phi_im),
                                            desc = "SDen", total = len(T), ascii = True
                                        )
        ]
        out['sdensity'] = SDen

    if with_bnumber_R2:
        cumulant = [
            bnumber_cumulant_R(
                2, T_el, mu_el, complex(phi_re_el, phi_im_el), complex(phi_re_el, -phi_im_el),
                L, T_crit_el, regulator, regulator2
            )
            for T_el, T_crit_el, mu_el, \
                phi_re_el, phi_im_el in tqdm.tqdm(
                                            zip(T, T_crit_v, mu, phi_re, phi_im),
                                            desc = "bnumber R2", total = len(T), ascii = True
                                        )
        ]
        out['bnumber_R2'] = cumulant

    return out


