#ifndef MODULE1_H
#define MODULE1_H

#include <Python.h>

#include <functional>
#include <iostream>
#include <complex>

#include <gsl/gsl_integration.h>

//Free parameters gcp_pnjl - lattice_cut_sea

#define G_SQRT2 sqrt(2.0)

#define G_T_MOTT_0 160.0

#define G_M_L_VAC 400.0
#define G_M_S_VAC 550.0

#define G_M_C_S 155.0
#define G_M_C_L 5.0
#define G_M_0 395.0

//Constants gcp_pnjl - lattice_cut_sea

#define G_DELTA_T 26.0              //source: https://arxiv.org/pdf/2012.12894.pdf
#define G_T_C_0 154.0               //source: https://arxiv.org/pdf/2012.12894.pdf
#define G_KAPPA 0.012               //source: https://arxiv.org/pdf/1812.08235.pdf
#define G_NC 3.0                    //source: -

//Constants gcp_cluster - bound_step_continuum_step

#define G_MI_PI0 135.0              //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_PI0 (1.0/2.0)          //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_PI0 0.0                //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_PI0 2.0                //source: (EPJA draft)
#define G_SI_PI0 0.0                //source: (EPJA draft)

#define G_MI_PI 140.0               //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_PI (2.0/2.0)           //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_PI 0.0                 //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_PI 2.0                 //source: (EPJA draft)
#define G_SI_PI 0.0                 //source: (EPJA draft)

#define G_MI_K 494.0                //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_K (2.0/2.0)            //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_K 0.0                  //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_K 2.0                  //source: (EPJA draft)
#define G_SI_K 1.0                  //source: (EPJA draft)

#define G_MI_K0 498.0               //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_K0 (2.0/2.0)           //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_K0 0.0                 //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_K0 2.0                 //source: (EPJA draft)
#define G_SI_K0 1.0                 //source: (EPJA draft)

#define G_MI_ETA 548.0              //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_ETA (1.0/2.0)          //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_ETA 0.0                //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_ETA 2.0                //source: (EPJA draft)
#define G_SI_ETA 1.0                //source: (EPJA draft)

#define G_MI_RHO 775.0              //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_RHO (9.0/2.0)          //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_RHO 0.0                //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_RHO 2.0                //source: (EPJA draft)
#define G_SI_RHO 0.0                //source: (EPJA draft)

#define G_MI_OMEGA 783.0            //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_OMEGA (3.0/2.0)        //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_OMEGA 0.0              //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_OMEGA 2.0              //source: (EPJA draft)
#define G_SI_OMEGA 0.0              //source: (EPJA draft)

#define G_MI_KSTAR_892 892.0        //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_KSTAR_892 (6.0/2.0)    //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_KSTAR_892 0.0          //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_KSTAR_892 2.0          //source: (EPJA draft)
#define G_SI_KSTAR_892 1.0          //source: (EPJA draft)

#define G_MI_KSTAR0_892 896.0       //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_KSTAR0_892 (6.0/2.0)   //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_KSTAR0_892 0.0         //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_KSTAR0_892 2.0         //source: (EPJA draft)
#define G_SI_KSTAR0_892 1.0         //source: (EPJA draft)

#define G_MI_P 938.0                //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_P 2.0                  //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_P 1.0                  //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_P 3.0                  //source: (EPJA draft)
#define G_SI_P 0.0                  //source: (EPJA draft)

#define G_MI_N 940.0                //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_N 2.0                  //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_N 1.0                  //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_N 3.0                  //source: (EPJA draft)
#define G_SI_N 0.0                  //source: (EPJA draft)

#define G_MI_ETAPRIME 958.0         //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_ETAPRIME (1.0/2.0)     //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_ETAPRIME 0.0           //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_ETAPRIME 2.0           //source: (EPJA draft)
#define G_SI_ETAPRIME 1.0           //source: (EPJA draft)

#define G_MI_A0 980.0               //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_A0 (3.0/2.0)           //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_A0 0.0                 //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_A0 4.0                 //source: (EPJA draft)
#define G_SI_A0 1.0                 //source: (EPJA draft)

#define G_MI_F0 990.0               //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_F0 (1.0/2.0)           //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_F0 0.0                 //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_F0 4.0                 //source: (EPJA draft)
#define G_SI_F0 0.0                 //source: (EPJA draft)

#define G_MI_PHI 1019.0             //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_PHI (3.0/2.0)          //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_PHI 0.0                //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_PHI 2.0                //source: (EPJA draft)
#define G_SI_PHI 2.0                //source: (EPJA draft)

#define G_MI_LAMBDA 1116.0          //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_LAMBDA 2.0             //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_LAMBDA 1.0             //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_LAMBDA 3.0             //source: (EPJA draft)
#define G_SI_LAMBDA 1.0             //source: (EPJA draft)

#define G_MI_H1 1170.0              //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_H1 (3.0/2.0)           //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_H1 0.0                 //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_H1 2.0                 //source: (EPJA draft)
#define G_SI_H1 2.0                 //source: (EPJA draft)

#define G_MI_SIGMAPLUS 1189.0       //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_SIGMAPLUS 2.0          //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_SIGMAPLUS 1.0          //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_SIGMAPLUS 3.0          //source: (EPJA draft)
#define G_SI_SIGMAPLUS 1.0          //source: (EPJA draft)

#define G_MI_SIGMA0 1193.0          //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_SIGMA0 2.0             //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_SIGMA0 1.0             //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_SIGMA0 3.0             //source: (EPJA draft)
#define G_SI_SIGMA0 1.0             //source: (EPJA draft)

#define G_MI_SIGMAMINUS 1197.0      //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_SIGMAMINUS 2.0         //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_SIGMAMINUS 1.0         //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_SIGMAMINUS 3.0         //source: (EPJA draft)
#define G_SI_SIGMAMINUS 1.0         //source: (EPJA draft)

#define G_MI_B1 1230.0              //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_B1 (9.0/2.0)           //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_B1 0.0                 //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_B1 4.0                 //source: (EPJA draft)
#define G_SI_B1 0.0                 //source: (EPJA draft)

#define G_MI_A1 1230.0              //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_A1 (9.0/2.0)           //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_A1 0.0                 //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_A1 4.0                 //source: (EPJA draft)
#define G_SI_A1 0.0                 //source: (EPJA draft)

#define G_MI_DELTA 1232.0           //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_DI_DELTA 16.0             //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_BI_DELTA 1.0              //source: https://arxiv.org/pdf/1404.7540.pdf
#define G_NI_DELTA 3.0              //source: (EPJA draft)
#define G_SI_DELTA 0.0              //source: (EPJA draft)

#define G_MI_D1 700.0               //source: (EPJA draft)
#define G_DI_D1 3.0                 //source: (EPJA draft)
#define G_BI_D1 (2.0/3.0)           //source: (EPJA draft)
#define G_NI_D1 2.0                 //source: (EPJA draft)
#define G_SI_D1 0.0                 //source: (EPJA draft)

#define G_MI_D2 850.0               //source: (EPJA draft)
#define G_DI_D2 6.0                 //source: (EPJA draft)
#define G_BI_D2 (2.0/3.0)           //source: (EPJA draft)
#define G_NI_D2 2.0                 //source: (EPJA draft)
#define G_SI_D2 1.0                 //source: (EPJA draft)

#define G_MI_4Q1 1300.0             //source: (EPJA draft)
#define G_DI_4Q1 3.0                //source: (EPJA draft)
#define G_BI_4Q1 (4.0/3.0)          //source: (EPJA draft)
#define G_NI_4Q1 4.0                //source: (EPJA draft)
#define G_SI_4Q1 0.0                //source: (EPJA draft)

#define G_MI_4Q2 1450.0             //source: (EPJA draft)
#define G_DI_4Q2 6.0                //source: (EPJA draft)
#define G_BI_4Q2 (4.0/3.0)          //source: (EPJA draft)
#define G_NI_4Q2 4.0                //source: (EPJA draft)
#define G_SI_4Q2 1.0                //source: (EPJA draft)

#define G_MI_4Q3 1600.0             //source: (EPJA draft)
#define G_DI_4Q3 3.0                //source: (EPJA draft)
#define G_BI_4Q3 (4.0/3.0)          //source: (EPJA draft)
#define G_NI_4Q3 4.0                //source: (EPJA draft)
#define G_SI_4Q3 2.0                //source: (EPJA draft)

#define G_MI_4Q4 1750.0             //source: (EPJA draft)
#define G_DI_4Q4 6.0                //source: (EPJA draft)
#define G_BI_4Q4 (4.0/3.0)          //source: (EPJA draft)
#define G_NI_4Q4 4.0                //source: (EPJA draft)
#define G_SI_4Q4 3.0                //source: (EPJA draft)

#define G_MI_P1 1540.0              //source: (EPJA draft)
#define G_DI_P1 16.0                //source: (EPJA draft)
#define G_BI_P1 1.0                 //source: (EPJA draft)
#define G_NI_P1 5.0                 //source: (EPJA draft)
#define G_SI_P1 1.0                 //source: (EPJA draft)

#define G_MI_P2 1860.0              //source: (EPJA draft)
#define G_DI_P2 16.0                //source: (EPJA draft)
#define G_BI_P2 1.0                 //source: (EPJA draft)
#define G_NI_P2 5.0                 //source: (EPJA draft)
#define G_SI_P2 2.0                 //source: (EPJA draft)

#define G_MI_5Q1 1600.0             //source: (EPJA draft)
#define G_DI_5Q1 12.0               //source: (EPJA draft)
#define G_BI_5Q1 (5.0/3.0)          //source: (EPJA draft)
#define G_NI_5Q1 5.0                //source: (EPJA draft)
#define G_SI_5Q1 0.0                //source: (EPJA draft)

#define G_MI_5Q2 1750.0             //source: (EPJA draft)
#define G_DI_5Q2 24.0               //source: (EPJA draft)
#define G_BI_5Q2 (5.0/3.0)          //source: (EPJA draft)
#define G_NI_5Q2 5.0                //source: (EPJA draft)
#define G_SI_5Q2 1.0                //source: (EPJA draft)

#define G_MI_5Q3 1900.0             //source: (EPJA draft)
#define G_DI_5Q3 36.0               //source: (EPJA draft)
#define G_BI_5Q3 (5.0/3.0)          //source: (EPJA draft)
#define G_NI_5Q3 5.0                //source: (EPJA draft)
#define G_SI_5Q3 2.0                //source: (EPJA draft)

#define G_MI_D 1880.0               //source: (EPJA draft)
#define G_DI_D 3.0                  //source: (EPJA draft)
#define G_BI_D 2.0                  //source: (EPJA draft)
#define G_NI_D 6.0                  //source: (EPJA draft)
#define G_SI_D 0.0                  //source: (EPJA draft)

// Utility macros

#define G_EXP_LIMIT log(std::numeric_limits<double>::max())
#define cdouble std::complex<double>

//Utility objects

struct thermo
{
    double T;
    double muB;
    double phiRe;
    double phiIm;
};

struct particle
{
    double mass;
    double chargeB;
};

class gsl_function_wrapper
{
private:
    gsl_integration_workspace *w;
    gsl_function F;
    size_t limit;
public:
    gsl_function_wrapper(double (*func)(double, void*), size_t i);
    gsl_function_wrapper(double (*func)(double, void*));
    ~gsl_function_wrapper();

    double eval(double x, void *pars);

    double qag(void *pars, double a, double b, double epsabs = 0.0, double epsrel = 1e-7, int key = 6);
    double qagiu(void *pars, double a, double epsabs = 0.0, double epsrel = 1e-7);
};

//Utility methods

void pbar(double x, double x_min, double x_max, double x_step);
void pbar(int i, int i_min, int i_max);

#endif