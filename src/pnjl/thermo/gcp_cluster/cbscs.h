#ifndef BOUND_STEP_CONTINUUM_STEP_H
#define BOUND_STEP_CONTINUUM_STEP_H

#include "../gcp_pnjl/c_lattice_cut_sea.h"

namespace bscs
{
    struct cluster_thermo
    {
        double T;
        double muB;
        double phiRe;
        double phiIm;
        double mi;
        double mth;
        double chargeB;
    };
    
    double cBDensityBosonSingletIntegrand(double p, void *pars);
    gsl_function_wrapper cwBDensityBosonSingletIntegrand(cBDensityBosonSingletIntegrand);
    double cBDensityFermionSingletIntegrand(double p, void *pars);
    gsl_function_wrapper cwBDensityFermionSingletIntegrand(cBDensityFermionSingletIntegrand);
    double cBDensityBosonTripletIntegrandReal(double p, void *pars);
    gsl_function_wrapper cwBDensityBosonTripletIntegrandReal(cBDensityBosonTripletIntegrandReal);
    double cBDensityBosonAntitripletIntegrandReal(double p, void *pars);
    gsl_function_wrapper cwBDensityBosonAntitripletIntegrandReal(cBDensityBosonAntitripletIntegrandReal);
    double cBDensityFermionTripletIntegrandReal(double p, void *pars);
    gsl_function_wrapper cwBDensityFermionTripletIntegrandReal(cBDensityFermionTripletIntegrandReal);
    double cBDensityFermionAntitripletIntegrandReal(double p, void *pars);
    gsl_function_wrapper cwBDensityFermionAntitripletIntegrandReal(cBDensityFermionAntitripletIntegrandReal);
    
    double cSDensityBosonSingletIntegrand(double p, void *pars);
    gsl_function_wrapper cwSDensityBosonSingletIntegrand(cSDensityBosonSingletIntegrand);
    double cSDensityFermionSingletIntegrand(double p, void *pars);
    gsl_function_wrapper cwSDensityFermionSingletIntegrand(cSDensityFermionSingletIntegrand);
    double cSDensityBosonTripletIntegrandReal(double p, void *pars);
    gsl_function_wrapper cwSDensityBosonTripletIntegrandReal(cSDensityBosonTripletIntegrandReal);
    double cSDensityBosonAntitripletIntegrandReal(double p, void *pars);
    gsl_function_wrapper cwSDensityBosonAntitripletIntegrandReal(cSDensityBosonAntitripletIntegrandReal);
    double cSDensityFermionTripletIntegrandReal(double p, void *pars);
    gsl_function_wrapper cwSDensityFermionTripletIntegrandReal(cSDensityFermionTripletIntegrandReal);
    double cSDensityFermionAntitripletIntegrandReal(double p, void *pars);
    gsl_function_wrapper cwSDensityFermionAntitripletIntegrandReal(cSDensityFermionAntitripletIntegrandReal);

    class cluster {
    protected:
        double MI;
        double DI;
        double BI;
        double NI;
        double SI;
    public:
        cluster(double mi, double di, double bi, double ni, double si);
        virtual ~cluster() = 0;

        friend double cMTh(const cluster &hadron, const thermo &thArgs);
    };

    class pi0 : cluster {
    public:
        pi0();
        ~pi0();

        friend double cBDensityCluster(const pi0 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const pi0 &hadron, const thermo &thArgs);
    };

    class pi : cluster {
    public:
        pi();
        ~pi();

        friend double cBDensityCluster(const pi &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const pi &hadron, const thermo &thArgs);
    };

    class K : cluster {
    public:
        K();
        ~K();

        friend double cBDensityCluster(const K &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const K &hadron, const thermo &thArgs);
    };

    class K0 : cluster {
    public:
        K0();
        ~K0();

        friend double cBDensityCluster(const K0 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const K0 &hadron, const thermo &thArgs);
    };

    class eta : cluster {
    public:
        eta();
        ~eta();

        friend double cBDensityCluster(const eta &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const eta &hadron, const thermo &thArgs);
    };

    class rho : cluster {
    public:
        rho();
        ~rho();

        friend double cBDensityCluster(const rho &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const rho &hadron, const thermo &thArgs);
    };

    class omega : cluster {
    public:
        omega();
        ~omega();

        friend double cBDensityCluster(const omega &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const omega &hadron, const thermo &thArgs);
    };

    class Kstar_892 : cluster {
    public:
        Kstar_892();
        ~Kstar_892();

        friend double cBDensityCluster(const Kstar_892 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const Kstar_892 &hadron, const thermo &thArgs);
    };

    class Kstar0_892 : cluster {
    public:
        Kstar0_892();
        ~Kstar0_892();

        friend double cBDensityCluster(const Kstar0_892 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const Kstar0_892 &hadron, const thermo &thArgs);
    };

    class p : cluster {
    public:
        p();
        ~p();

        friend double cBDensityCluster(const p &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const p &hadron, const thermo &thArgs);
    };

    class n : cluster {
    public:
        n();
        ~n();

        friend double cBDensityCluster(const n &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const n &hadron, const thermo &thArgs);
    };

    class etaPrime : cluster {
    public:
        etaPrime();
        ~etaPrime();

        friend double cBDensityCluster(const etaPrime &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const etaPrime &hadron, const thermo &thArgs);
    };

    class a0 : cluster {
    public:
        a0();
        ~a0();

        friend double cBDensityCluster(const a0 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const a0 &hadron, const thermo &thArgs);
    };

    class f0 : cluster {
    public:
        f0();
        ~f0();

        friend double cBDensityCluster(const f0 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const f0 &hadron, const thermo &thArgs);
    };

    class phi : cluster {
    public:
        phi();
        ~phi();

        friend double cBDensityCluster(const phi &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const phi &hadron, const thermo &thArgs);
    };

    class Lambda : cluster {
    public:
        Lambda();
        ~Lambda();

        friend double cBDensityCluster(const Lambda &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const Lambda &hadron, const thermo &thArgs);
    };

    class h1 : cluster {
    public:
        h1();
        ~h1();

        friend double cBDensityCluster(const h1 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const h1 &hadron, const thermo &thArgs);
    };

    class SigmaPlus : cluster {
    public:
        SigmaPlus();
        ~SigmaPlus();

        friend double cBDensityCluster(const SigmaPlus &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const SigmaPlus &hadron, const thermo &thArgs);
    };

    class Sigma0 : cluster {
    public:
        Sigma0();
        ~Sigma0();

        friend double cBDensityCluster(const Sigma0 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const Sigma0 &hadron, const thermo &thArgs);
    };

    class SigmaMinus : cluster {
    public:
        SigmaMinus();
        ~SigmaMinus();

        friend double cBDensityCluster(const SigmaMinus &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const SigmaMinus &hadron, const thermo &thArgs);
    };

    class b1 : cluster {
    public:
        b1();
        ~b1();

        friend double cBDensityCluster(const b1 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const b1 &hadron, const thermo &thArgs);
    };

    class a1 : cluster {
    public:
        a1();
        ~a1();

        friend double cBDensityCluster(const a1 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const a1 &hadron, const thermo &thArgs);
    };

    class Delta : cluster {
    public:
        Delta();
        ~Delta();

        friend double cBDensityCluster(const Delta &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const Delta &hadron, const thermo &thArgs);
    };

    class D1 : cluster {
    public:
        D1();
        ~D1();

        friend double cBDensityCluster(const D1 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const D1 &hadron, const thermo &thArgs);
    };

    class D2 : cluster {
    public:
        D2();
        ~D2();

        friend double cBDensityCluster(const D2 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const D2 &hadron, const thermo &thArgs);
    };

    class FourQ1 : cluster {
    public:
        FourQ1();
        ~FourQ1();

        friend double cBDensityCluster(const FourQ1 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const FourQ1 &hadron, const thermo &thArgs);
    };

    class FourQ2 : cluster {
    public:
        FourQ2();
        ~FourQ2();

        friend double cBDensityCluster(const FourQ2 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const FourQ2 &hadron, const thermo &thArgs);
    };

    class FourQ3 : cluster {
    public:
        FourQ3();
        ~FourQ3();

        friend double cBDensityCluster(const FourQ3 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const FourQ3 &hadron, const thermo &thArgs);
    };

    class FourQ4 : cluster {
    public:
        FourQ4();
        ~FourQ4();

        friend double cBDensityCluster(const FourQ4 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const FourQ4 &hadron, const thermo &thArgs);
    };

    class P1 : cluster {
    public:
        P1();
        ~P1();

        friend double cBDensityCluster(const P1 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const P1 &hadron, const thermo &thArgs);
    };

    class P2 : cluster {
    public:
        P2();
        ~P2();

        friend double cBDensityCluster(const P2 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const P2 &hadron, const thermo &thArgs);
    };

    class FiveQ1 : cluster {
    public:
        FiveQ1();
        ~FiveQ1();

        friend double cBDensityCluster(const FiveQ1 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const FiveQ1 &hadron, const thermo &thArgs);
    };

    class FiveQ2 : cluster {
    public:
        FiveQ2();
        ~FiveQ2();

        friend double cBDensityCluster(const FiveQ2 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const FiveQ2 &hadron, const thermo &thArgs);
    };

    class FiveQ3 : cluster {
    public:
        FiveQ3();
        ~FiveQ3();

        friend double cBDensityCluster(const FiveQ3 &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const FiveQ3 &hadron, const thermo &thArgs);
    };

    class d : cluster {
    public:
        d();
        ~d();

        friend double cBDensityCluster(const d &hadron, const thermo &thArgs);
        friend double cSDensityCluster(const d &hadron, const thermo &thArgs);
    };

    static PyObject* cSDensityClusterAll(PyObject *self, PyObject *args);
    static PyObject* cBDensityClusterAll(PyObject *self, PyObject *args);

    static PyMethodDef methods[] = {
        {"cSDensityClusterAll", cSDensityClusterAll, METH_VARARGS, "Collective sdensity of bscs clusters. Input must contain only four lists: T, muB, phiRe, phiIm respectively."},
        {"cBDensityClusterAll", cBDensityClusterAll, METH_VARARGS, "Collective bdensity of bscs clusters. Input must contain only four lists: T, muB, phiRe, phiIm respectively."},
        {NULL, NULL, 0, NULL}
    };

    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT,
        "cbscs",
        "Thermodynamics of bscs clusters",
        -1,
        methods
    };
}

PyMODINIT_FUNC PyInit_cbscs(void);

#endif