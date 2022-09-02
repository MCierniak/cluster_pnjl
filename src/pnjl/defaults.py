#Model parameter defaults.

#defaults from https://arxiv.org/pdf/2012.12894.pdf (Tc0 = T0, delta_T, ml)
default_Tc0         = 154. # 2+1 Nf Tc0
#default_Tc0        = 170. # 2 Nf Tc0 from https://arxiv.org/pdf/hep-lat/0501030.pdf
default_delta_T     = 26.
default_ml          = 5.5

#defaults from https://arxiv.org/pdf/2012.12894.pdf modified by me
default_c           = 300.0
default_d           = 3.2

#defaults from https://arxiv.org/pdf/1812.08235.pdf (kappa)
default_kappa       = 0.012

#defaults from https://arxiv.org/pdf/hep-ph/0506234.pdf (Gs, T0, a0, a1, a2, a3, b3, b4)
default_Gs          = 10.08e-6
#default_T0          = 187. #Nf = 2 + 1 result from https://arxiv.org/pdf/0704.3234.pdf
default_a0          = 6.75
default_a1          = -1.95
default_a2          = 2.625
default_a3          = -7.44
default_b3          = 0.75
default_b4          = 7.5

#defaults from https://arxiv.org/pdf/hep-ph/0506234.pdf modified by me
default_T0          = 175.0

#defaults from me (M0, Nf, Nc, ms)
default_M0          = 400.
default_Nc          = 3.0
default_ms          = 100.0

#defaults from me (Lambda - gcp_sea_lattice cutoff parameter)
default_Lambda      = 900.0

#default cluster continuum energy scale (different from the gcp_sea_lattice Lambda!)
default_L           = 50.0

#default cluster parameters

#cluster masses
default_B           = 100
default_M = {
        'pi' : 200.0,
        'K' : 500.0,
        'rho' : 2 * (default_M0 + default_ml) - default_B,
        'omega' : 2 * (default_M0 + default_ml) - default_B,
        'D' : 2 * (default_M0 + default_ml) - default_B,
        'N' : 3 * (default_M0 + default_ml) - 2 * default_B,
        'T' : 4 * (default_M0 + default_ml) - 3 * default_B,
        'F' : 4 * (default_M0 + default_ml) - 3 * default_B,
        'P' : 5 * (default_M0 + default_ml) - 4 * default_B,
        'Q' : 5 * (default_M0 + default_ml) - 4 * default_B,
        'H' : 6 * (default_M0 + default_ml) - 5 * default_B
    }

#cluster degeneracy
default_d = {
        'pi' : (1.0 * 3.0 * 1.0),
        'K' : (1.0 * 6.0 * 1.0),
        'rho' : (3.0 * 3.0 * 1.0),
        'omega' : (3.0 * 1.0 * 1.0),
        'D' : (1.0 * 1.0 * 3.0) / 2.0,
        'N' : (2.0 * 2.0 * 1.0) / 2.0,
        'T' : ((1.0 * 5.0 * 1.0) + (5.0 * 1.0 * 1.0) + (3.0 * 3.0 * 1.0)) / 2.0,
        'F' : (1.0 * 1.0 * 3.0) / 2.0,
        'P' : ((4.0 * 2.0 * 1.0) + (2.0 * 4.0 * 1.0)) / 2.0,
        'Q' : (2.0 * 1.0 * 3.0) / 2.0,
        'H' : ((1.0 * 3.0 * 1.0) + (3.0 * 1.0 * 1.0)) / 2.0
    }

#cluster d.o.f's
default_N = {
        'pi' : 2.0,
        'K' : 2.0,
        'rho' : 2.0,
        'omega' : 2.0,
        'D' : 2.0,
        'N' : 3.0,
        'T' : 4.0,
        'F' : 4.0,
        'P' : 5.0,
        'Q' : 5.0,
        'H' : 6.0
    }

#cluster net valence light quarks
default_a = {
        'pi' : 0,
        'K' : 1,
        'rho' : 0,
        'omega' : 0,
        'D' : 2,
        'N' : 3,
        'T' : 0,
        'F' : 4,
        'P' : 3,
        'Q' : 5,
        'H' : 6
    }

#cluster net valence strange quarks
default_b = {
        'pi' : 0,
        'K' : -1,
        'rho' : 0,
        'omega' : 0,
        'D' : 0,
        'N' : 0,
        'T' : 0,
        'F' : 0,
        'P' : 0,
        'Q' : 0,
        'H' : 0
    }

#cluster absolute strangeness
default_s = {
        'pi' : 0,
        'K' : 1,
        'rho' : 0,
        'omega' : 0,
        'D' : 0,
        'N' : 0,
        'T' : 0,
        'F' : 0,
        'P' : 0,
        'Q' : 0,
        'H' : 0
    }

