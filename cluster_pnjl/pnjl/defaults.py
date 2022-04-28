#defaults from https://arxiv.org/pdf/2012.12894.pdf (Tc0 = T0, delta_T, ml, c, d)

default_Tc0         = 154. # 2+1 Nf Tc0
#default_Tc0        = 170. # 2 Nf Tc0 from https://arxiv.org/pdf/hep-lat/0501030.pdf
default_delta_T     = 26.
default_ml          = 5.5

#fit to lattice (see below)
#default_c           = 350.0
#default_d           = 3.2

#defaults from https://arxiv.org/pdf/1812.08235.pdf (kappa)

default_kappa       = 0.012

#defaults from https://arxiv.org/pdf/hep-ph/0506234.pdf (Gs, T0, a0, a1, a2, a3, b3, b4)

default_Gs          = 10.08e-6

#fit to lattice (see below)
#default_T0          = 187. #Nf = 2 + 1 result from https://arxiv.org/pdf/0704.3234.pdf

default_a0          = 6.75
default_a1          = -1.95
default_a2          = 2.625
default_a3          = -7.44
default_b3          = 0.75
default_b4          = 7.5

#defaults from me (M0, Nf, Nc, ms)

default_M0          = 400.
default_Nf          = 2.0
default_Nc          = 3.0
default_ms          = 100.0

#defaults from me (Lambda, gcp_sea_lattice cutoff parameter)

#fit to lattice (see below)
#default_Lambda      = 900.0

#default cluster masses

default_B           = 100
default_Mpi         = 140.0 #to nie używa https://pdg.lbl.gov/2020/reviews/rpp2020-rev-pseudoscalar-meson-decay-cons.pdf i GMOR, trzeba zmienić!
default_MK          = 500.0 #to musi być zmienione zgodnie z GMOR
default_MM          = 2 * (default_M0 + default_ml) - default_B
default_MD          = 2 * (default_M0 + default_ml) - default_B
default_MN          = 3 * (default_M0 + default_ml) - 2 * default_B
default_MT          = 4 * (default_M0 + default_ml) - 3 * default_B
default_MF          = 4 * (default_M0 + default_ml) - 3 * default_B
default_MP          = 5 * (default_M0 + default_ml) - 4 * default_B
default_MQ          = 5 * (default_M0 + default_ml) - 4 * default_B
default_MH          = 6 * (default_M0 + default_ml) - 5 * default_B

#default cluster continuum energy scale

#fit to lattice (see below)
#default_L           = 100.0

#parameters fitted to lattice pressure 

#initial
#default_L           = 100.0 #bigger value - greater pressure for T < T_Mott
#default_T0          = 187.0 #bigger value - smaller pressure slope for T > T_Mott
#default_c           = 350.0 #bigger value - smaller total pressure
#default_d           = 3.2   #bigger value - ???

#current 1
#default_L           = 60.0
#default_T0          = 180.0
#default_c           = 550.0
#default_d           = 6.0
#default_Lambda      = 900.0

#current 2
default_L           = 60.0
default_T0          = 175.0
default_c           = 300.0
default_d           = 3.2
default_Lambda      = 798.13